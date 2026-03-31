# eval_foundation_v6.py
#
# Post-training evaluation for JAISP Foundation v6.
#
# Three checks:
#
#   1. Reconstruction quality across ALL validation tiles and ALL band pairs
#      - Pearson r, MAE, PSNR at bright pixels (info-weighted)
#      - Compared against a zero-prediction baseline (predicting sky background)
#      - Results printed as a band×band table and saved as eval_results.npz
#
#   2. Spatial precision sanity check
#      - For a few tiles, show that features at pixel (x,y) predict flux at (x,y)
#        better than features at a random offset (x+dx, y+dy)
#      - This is the key property JEPA lacked
#
#   3. W&B summary (optional)
#      - Logs all metrics to the same run if WANDB_RUN_ID is provided
#
# Usage:
#   python eval_foundation_v6.py \
#       --checkpoint ./checkpoints/jaisp_v6/checkpoint_best.pt \
#       --rubin_dir ../data/rubin_tiles_ecdfs \
#       --euclid_dir ../data/euclid_tiles_ecdfs
#
#   # With W&B logging (resuming the training run):
#   python eval_foundation_v6.py \
#       --checkpoint ./checkpoints/jaisp_v6/checkpoint_best.pt \
#       --rubin_dir ../data/rubin_tiles_ecdfs \
#       --euclid_dir ../data/euclid_tiles_ecdfs \
#       --wandb_run_id nyg3gf3u   # from the training run URL

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from jaisp_foundation_v6 import JAISPFoundationV6, ALL_BANDS, RUBIN_BANDS
from jaisp_dataset_v6 import JAISPDatasetV6


# ============================================================
# Helpers
# ============================================================

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float('nan')
    c = np.corrcoef(a.ravel(), b.ravel())
    return float(c[0, 1])


def psnr(truth: np.ndarray, pred: np.ndarray, max_val: float = None) -> float:
    """PSNR in noise-normalised units."""
    mse = np.mean((truth - pred) ** 2)
    if mse < 1e-12:
        return float('inf')
    if max_val is None:
        max_val = np.nanpercentile(np.abs(truth), 99)
    return float(10 * np.log10(max_val ** 2 / mse))


def info_weighted_mask(info: np.ndarray, top_frac: float = 0.10) -> np.ndarray:
    """Boolean mask: top top_frac of pixels by information weight."""
    thresh = np.nanpercentile(info, (1 - top_frac) * 100)
    return info >= thresh


# ============================================================
# Load model
# ============================================================

def load_model(checkpoint_path: str, device: torch.device) -> JAISPFoundationV6:
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Reconstruct model with same config — read from checkpoint if saved, else use defaults
    cfg = ckpt.get('config', {})
    model = JAISPFoundationV6(
        band_names=cfg.get('band_names', ALL_BANDS),
        stem_ch=cfg.get('stem_ch', 64),
        encoder_dims=tuple(cfg.get('encoder_dims', [128, 256, 512])),
        blocks_per_stage=cfg.get('blocks_per_stage', 2),
        transformer_depth=cfg.get('transformer_depth', 4),
        transformer_heads=cfg.get('transformer_heads', 8),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    epoch = ckpt.get('epoch', '?')
    print(f'Loaded checkpoint from epoch {epoch}  ({checkpoint_path})')
    return model


# ============================================================
# Check 1: Reconstruction quality across all band pairs
# ============================================================

@torch.no_grad()
def eval_reconstruction(
    model: JAISPFoundationV6,
    dataset: JAISPDatasetV6,
    tile_indices: list,
    device: torch.device,
    n_tiles: int = None,
) -> dict:
    """
    For every tile, for every Rubin band:
      - use the other N-1 Rubin bands as context
      - predict the held-out band
      - compute Pearson r and MAE at bright pixels
      - also compute baseline: predict zeros (= background sky)

    Returns a dict:
      per_band:  {band: {'r': [...], 'mae': [...], 'psnr': [...],
                          'r_baseline': [...], 'mae_baseline': [...]}}
      summary:   {band: {'r_mean', 'r_std', 'mae_mean', 'psnr_mean',
                          'r_baseline_mean', 'improvement_r'}}
    """
    if n_tiles is not None:
        tile_indices = tile_indices[:n_tiles]

    per_band = defaultdict(lambda: defaultdict(list))

    for tile_idx in tqdm(tile_indices, desc='Reconstruction eval'):
        sample = dataset[tile_idx]
        avail = list(sample['rubin'].keys())
        if len(avail) < 2:
            continue

        for target_band in avail:
            context_bands = [b for b in avail if b != target_band]

            ctx_img = {b: sample['rubin'][b]['image'].unsqueeze(0).to(device) for b in context_bands}
            ctx_rms = {b: sample['rubin'][b]['rms'].unsqueeze(0).to(device)   for b in context_bands}
            tgt_img = sample['rubin'][target_band]['image'].unsqueeze(0).to(device)
            tgt_rms = sample['rubin'][target_band]['rms'].unsqueeze(0).to(device)

            out = model(ctx_img, ctx_rms, target_band, tgt_img, tgt_rms)

            truth = out['target_norm'].squeeze().cpu().numpy()   # noise-normalised
            pred  = out['pred'].squeeze().cpu().numpy()
            info  = out['info_weights'].squeeze().cpu().numpy()

            mask = info_weighted_mask(info, top_frac=0.10)
            if mask.sum() < 10:
                continue

            t_bright = truth[mask]
            p_bright = pred[mask]

            r   = pearson_r(t_bright, p_bright)
            mae = float(np.mean(np.abs(p_bright - t_bright)))
            ps  = psnr(truth, pred)

            # Baseline: predict zeros (sky background = 0 in noise-normalised units)
            r_base   = pearson_r(t_bright, np.zeros_like(t_bright))
            mae_base = float(np.mean(np.abs(t_bright)))   # |truth - 0|

            per_band[target_band]['r'].append(r)
            per_band[target_band]['mae'].append(mae)
            per_band[target_band]['psnr'].append(ps)
            per_band[target_band]['r_baseline'].append(r_base)
            per_band[target_band]['mae_baseline'].append(mae_base)

    # Summarise
    summary = {}
    for band in RUBIN_BANDS:
        if band not in per_band:
            continue
        d = per_band[band]
        r_arr = np.array(d['r'])
        summary[band] = {
            'r_mean':          float(np.nanmean(r_arr)),
            'r_std':           float(np.nanstd(r_arr)),
            'mae_mean':        float(np.nanmean(d['mae'])),
            'psnr_mean':       float(np.nanmean(d['psnr'])),
            'r_baseline_mean': float(np.nanmean(d['r_baseline'])),
            'mae_baseline':    float(np.nanmean(d['mae_baseline'])),
            'improvement_r':   float(np.nanmean(r_arr) - np.nanmean(d['r_baseline'])),
            'n_tiles':         len(r_arr),
        }

    return {'per_band': dict(per_band), 'summary': summary}


def print_reconstruction_table(summary: dict) -> None:
    print('\n' + '=' * 75)
    print('RECONSTRUCTION QUALITY  (bright pixels, info-weighted top-10%)')
    print('=' * 75)
    print(f'{"Band":<14} {"r (model)":>10} {"r (baseline)":>13} {"Δr":>7} '
          f'{"MAE model":>10} {"MAE base":>10} {"PSNR":>8} {"N":>5}')
    print('-' * 75)
    for band in RUBIN_BANDS:
        if band not in summary:
            continue
        s = summary[band]
        print(f'{band:<14} {s["r_mean"]:>10.3f} {s["r_baseline_mean"]:>13.3f} '
              f'{s["improvement_r"]:>+7.3f} {s["mae_mean"]:>10.2f} '
              f'{s["mae_baseline"]:>10.2f} {s["psnr_mean"]:>8.1f} {s["n_tiles"]:>5}')
    print('=' * 75)
    print('Δr > 0.1 is good. Δr > 0.3 is excellent.')
    print('MAE model < MAE base means the model beats predicting sky background.\n')


# ============================================================
# Check 2: Spatial precision
# ============================================================

@torch.no_grad()
def eval_spatial_precision(
    model: JAISPFoundationV6,
    dataset: JAISPDatasetV6,
    tile_indices: list,
    device: torch.device,
    n_tiles: int = 20,
    offsets_px: list = None,
) -> dict:
    """
    Test whether encoder features preserve spatial layout.

    For each tile, encode the context bands and extract the decoder's full-res
    output at matched vs shifted positions. Then compute:

      r_at_correct_position:  Pearson r between pred[x,y] and truth[x,y]
      r_at_offset_position:   Pearson r between pred[x+dx, y+dy] and truth[x,y]

    If spatial info is preserved: r_correct >> r_offset.
    If the model just learns the global average: r_correct ≈ r_offset.

    offsets_px: list of (dx, dy) pixel offsets to test. Default: 8, 16, 32, 64 px.
    """
    if offsets_px is None:
        offsets_px = [(8, 0), (16, 0), (32, 0), (64, 0), (0, 16), (16, 16)]

    tile_indices = tile_indices[:n_tiles]
    results = defaultdict(list)

    for tile_idx in tqdm(tile_indices, desc='Spatial precision eval'):
        sample = dataset[tile_idx]
        avail = list(sample['rubin'].keys())
        if len(avail) < 2:
            continue

        target_band = avail[0]
        context_bands = avail[1:]

        ctx_img = {b: sample['rubin'][b]['image'].unsqueeze(0).to(device) for b in context_bands}
        ctx_rms = {b: sample['rubin'][b]['rms'].unsqueeze(0).to(device)   for b in context_bands}
        tgt_img = sample['rubin'][target_band]['image'].unsqueeze(0).to(device)
        tgt_rms = sample['rubin'][target_band]['rms'].unsqueeze(0).to(device)

        out = model(ctx_img, ctx_rms, target_band, tgt_img, tgt_rms)

        truth = out['target_norm'].squeeze().cpu().numpy()
        pred  = out['pred'].squeeze().cpu().numpy()
        info  = out['info_weights'].squeeze().cpu().numpy()
        mask  = info_weighted_mask(info, top_frac=0.10)

        # Correlation at correct position
        r_correct = pearson_r(truth[mask], pred[mask])
        results['r_correct'].append(r_correct)

        # Correlation at offset positions
        H, W = truth.shape
        for dx, dy in offsets_px:
            # Roll prediction by (dy rows, dx cols)
            pred_shifted = np.roll(np.roll(pred, dy, axis=0), dx, axis=1)
            # Only use pixels where the shift doesn't wrap (centre crop)
            cy0, cy1 = max(0, dy), min(H, H - dy) if dy >= 0 else (0, H)
            cx0, cx1 = max(0, dx), min(W, W - dx) if dx >= 0 else (0, W)
            t_crop = truth[cy0:cy1, cx0:cx1]
            p_crop = pred_shifted[cy0:cy1, cx0:cx1]
            m_crop = mask[cy0:cy1, cx0:cx1]
            if m_crop.sum() < 10:
                continue
            r_off = pearson_r(t_crop[m_crop], p_crop[m_crop])
            results[f'r_offset_{dx}dx_{dy}dy'].append(r_off)

    summary = {k: float(np.nanmean(v)) for k, v in results.items()}
    return summary


def print_spatial_table(spatial: dict) -> None:
    print('\n' + '=' * 55)
    print('SPATIAL PRECISION CHECK')
    print('=' * 55)
    print(f'{"Condition":<35} {"Pearson r":>10}')
    print('-' * 55)
    r_correct = spatial.get('r_correct', float('nan'))
    print(f'{"At correct position (0 offset)":<35} {r_correct:>10.3f}')
    for k, v in spatial.items():
        if k == 'r_correct':
            continue
        label = k.replace('r_offset_', 'Shifted by ').replace('dx_', 'px x, ').replace('dy', 'px y')
        print(f'  {label:<33} {v:>10.3f}')
    print('=' * 55)
    print('r_correct should be >> any shifted r.')
    print('If shifted r ≈ r_correct, the model ignores spatial position.\n')


# ============================================================
# Visualisation: per-band reconstruction grid
# ============================================================

@torch.no_grad()
def plot_band_grid(
    model: JAISPFoundationV6,
    dataset: JAISPDatasetV6,
    tile_idx: int,
    device: torch.device,
    save_path: str = None,
) -> None:
    """
    For one tile, show all 6 Rubin bands side-by-side: truth vs prediction.
    Each column is one band. Row 0 = truth, Row 1 = pred, Row 2 = residual.
    """
    sample = dataset[tile_idx]
    avail = list(sample['rubin'].keys())
    n = len(avail)
    if n < 2:
        print('Not enough bands for this tile.')
        return

    fig = plt.figure(figsize=(3.5 * n, 9))
    gs = gridspec.GridSpec(3, n, figure=fig, hspace=0.05, wspace=0.04)
    row_labels = ['Truth (noise units)', 'Prediction', 'Residual']

    for col, target_band in enumerate(avail):
        context_bands = [b for b in avail if b != target_band]
        ctx_img = {b: sample['rubin'][b]['image'].unsqueeze(0).to(device) for b in context_bands}
        ctx_rms = {b: sample['rubin'][b]['rms'].unsqueeze(0).to(device)   for b in context_bands}
        tgt_img = sample['rubin'][target_band]['image'].unsqueeze(0).to(device)
        tgt_rms = sample['rubin'][target_band]['rms'].unsqueeze(0).to(device)

        out = model(ctx_img, ctx_rms, target_band, tgt_img, tgt_rms)
        truth = out['target_norm'].squeeze().cpu().numpy()
        pred  = out['pred'].squeeze().cpu().numpy()
        resid = pred - truth

        vlo, vhi = np.nanpercentile(truth, [1, 99])
        rmax = float(np.nanpercentile(np.abs(resid), 99)) or 1.0

        arrays = [truth, pred, resid]
        cmaps  = ['gray', 'gray', 'RdBu_r']
        vranges = [(vlo, vhi), (vlo, vhi), (-rmax, rmax)]

        for row, (arr, cmap, vr) in enumerate(zip(arrays, cmaps, vranges)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(arr, cmap=cmap, vmin=vr[0], vmax=vr[1], origin='lower')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
            if row == 0:
                ax.set_title(target_band.split('_')[1], fontsize=10, pad=3)

    fig.suptitle(f'All-band reconstruction — Tile {sample["tile_id"]}', fontsize=11, y=1.01)
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate JAISP Foundation v6')
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--rubin_dir',   default='../data/rubin_tiles_ecdfs')
    parser.add_argument('--euclid_dir',  default='../data/euclid_tiles_ecdfs')
    parser.add_argument('--n_eval_tiles',type=int, default=None, help='Cap number of eval tiles (default: all)')
    parser.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir',    default=None, help='Directory to save figures and results')
    parser.add_argument('--wandb_run_id',default=None, help='W&B run ID to log summary metrics to')
    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir) if args.save_dir else Path(args.checkpoint).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, device)

    # Load dataset (no augmentation for eval)
    print('Loading dataset...')
    dataset = JAISPDatasetV6(
        rubin_dir=args.rubin_dir,
        euclid_dir=args.euclid_dir,
        augment=False,
        load_euclid=False,
    )
    all_indices = list(range(len(dataset)))

    # ---- Check 1: Reconstruction quality -----------------------------------
    print('\n[1/3] Reconstruction quality...')
    recon = eval_reconstruction(model, dataset, all_indices, device, n_tiles=args.n_eval_tiles)
    print_reconstruction_table(recon['summary'])

    # Save raw results
    np.savez(save_dir / 'eval_reconstruction.npz',
             summary=recon['summary'],
             per_band={b: dict(v) for b, v in recon['per_band'].items()})

    # ---- Check 2: Spatial precision ----------------------------------------
    print('\n[2/3] Spatial precision...')
    spatial = eval_spatial_precision(model, dataset, all_indices, device)
    print_spatial_table(spatial)

    # ---- Check 3: Per-band grid for 2 tiles --------------------------------
    print('\n[3/3] Band reconstruction grids...')
    for i, tile_idx in enumerate(all_indices[:2]):
        plot_band_grid(model, dataset, tile_idx, device,
                       save_path=str(save_dir / f'band_grid_tile{i}.png'))

    # ---- Optional: log summary to W&B -------------------------------------
    if args.wandb_run_id:
        import wandb
        wandb.init(id=args.wandb_run_id, resume='must')
        log = {}
        for band, s in recon['summary'].items():
            log[f'eval/{band}/r_mean']       = s['r_mean']
            log[f'eval/{band}/mae_mean']      = s['mae_mean']
            log[f'eval/{band}/improvement_r'] = s['improvement_r']
        log['eval/spatial/r_correct']    = spatial.get('r_correct', float('nan'))
        log['eval/spatial/r_offset_16px'] = spatial.get('r_offset_16dx_0dy', float('nan'))
        wandb.log(log)
        wandb.finish()
        print('Metrics logged to W&B.')

    # ---- Summary judgement -------------------------------------------------
    print('\n=== OVERALL JUDGEMENT ===')
    avg_r = np.mean([s['r_mean'] for s in recon['summary'].values()])
    avg_dr = np.mean([s['improvement_r'] for s in recon['summary'].values()])
    r_correct = spatial.get('r_correct', 0)
    r_offset  = spatial.get('r_offset_16dx_0dy', 0)
    spatial_gap = r_correct - r_offset

    print(f'  Average r across bands:          {avg_r:.3f}   (target: >0.75)')
    print(f'  Average improvement over zero:   {avg_dr:+.3f}  (target: >0.30)')
    print(f'  Spatial precision gap (16px):    {spatial_gap:+.3f}  (target: >0.20)')

    if avg_r > 0.75 and avg_dr > 0.30 and spatial_gap > 0.20:
        print('\n  ✓ PASS — Ready to test on astrometry downstream task.')
    elif avg_r > 0.60 and avg_dr > 0.15:
        print('\n  ~ PARTIAL — Reconstruction is learning but spatial precision needs checking.')
        print('    Consider training longer or reducing transformer_depth to reduce over-smoothing.')
    else:
        print('\n  ✗ FAIL — Model is not learning useful representations.')
        print('    Check: loss curve, LR warmup, whether tiles have valid data.')


if __name__ == '__main__':
    main()
