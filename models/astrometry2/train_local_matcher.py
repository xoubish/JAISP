"""Train the standalone native-resolution local astrometry matcher."""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence

import numpy as np
import torch

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import (
    MatchedPatchDataset,
    build_patch_samples,
    discover_tile_pairs,
    make_loader,
    normalize_rubin_band,
    normalize_rubin_bands,
    split_tile_pairs,
)
from infer_concordance import predict_tile
from matcher import LocalAstrometryMatcher
from viz import make_tile_diagnostic_figure

try:
    import wandb
except ImportError:
    wandb = None


def _target_pixel_shift(target_offset_arcsec: torch.Tensor, pixel_to_sky: torch.Tensor) -> torch.Tensor:
    inv = torch.linalg.pinv(pixel_to_sky)
    return torch.bmm(inv, target_offset_arcsec.unsqueeze(-1)).squeeze(-1)


def compute_loss(
    out: Dict[str, torch.Tensor],
    target_offset_arcsec: torch.Tensor,
    pixel_to_sky: torch.Tensor,
    pixel_loss_weight: float,
) -> Dict[str, torch.Tensor]:
    pred = out['pred_offset_arcsec']
    err = pred - target_offset_arcsec
    radial = torch.sqrt((err ** 2).sum(dim=1) + 1e-10)
    sigma = torch.exp(out['log_sigma']).clamp_min(1e-4)
    loss_main = (radial / sigma + out['log_sigma']).mean()
    target_px = _target_pixel_shift(target_offset_arcsec, pixel_to_sky)
    loss_px = (
        torch.nn.functional.smooth_l1_loss(out['dx_px'], target_px[:, 0])
        + torch.nn.functional.smooth_l1_loss(out['dy_px'], target_px[:, 1])
    )
    loss_reg = 0.01 * ((out['dx_px'] ** 2 + out['dy_px'] ** 2).mean())
    loss_total = loss_main + float(max(0.0, pixel_loss_weight)) * loss_px + loss_reg
    return {
        'loss_total': loss_total,
        'loss_main': loss_main,
        'loss_px': loss_px,
        'loss_reg': loss_reg,
    }


@torch.no_grad()
def compute_metrics(
    out: Dict[str, torch.Tensor],
    target_offset_arcsec: torch.Tensor,
    pixel_to_sky: torch.Tensor,
) -> Dict[str, float]:
    pred = out['pred_offset_arcsec']
    err = pred - target_offset_arcsec
    err_ra = err[:, 0].abs()
    err_dec = err[:, 1].abs()
    err_total = torch.sqrt(err_ra ** 2 + err_dec ** 2 + 1e-10)
    target_px = _target_pixel_shift(target_offset_arcsec, pixel_to_sky)
    err_px = torch.sqrt(
        (out['dx_px'] - target_px[:, 0]) ** 2 + (out['dy_px'] - target_px[:, 1]) ** 2 + 1e-10
    )
    return {
        'mae_ra': float(err_ra.mean().item()),
        'mae_dec': float(err_dec.mean().item()),
        'mae_total': float(err_total.mean().item()),
        'p68_total': float(torch.quantile(err_total, 0.68).item()),
        'mae_px': float(err_px.mean().item()),
        'frac_01arcsec': float((err_total < 0.1).float().mean().item()),
        'frac_02arcsec': float((err_total < 0.2).float().mean().item()),
        'sigma_median': float(torch.exp(out['log_sigma']).median().item()),
        'confidence_mean': float(out['confidence'].mean().item()),
    }


def make_preview(model, sample: Dict, device: torch.device, epoch: int, split_name: str):
    if wandb is None:
        return None
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        rubin = sample['rubin_patch'].unsqueeze(0).to(device)
        vis = sample['vis_patch'].unsqueeze(0).to(device)
        pix2sky = sample['pixel_to_sky'].unsqueeze(0).to(device)
        out = model(rubin, vis, pix2sky)
    pred = out['pred_offset_arcsec'][0].cpu().numpy() * 1000.0
    gt = sample['target_offset_arcsec'].numpy() * 1000.0
    sigma = float(torch.exp(out['log_sigma'][0]).cpu().item() * 1000.0)

    rubin_img = sample['rubin_patch'].numpy().mean(axis=0)
    vis_img = sample['vis_patch'].numpy()[0]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rubin_img, origin='lower', cmap='magma')
    ax.set_title('Rubin reprojection patch')

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(vis_img, origin='lower', cmap='gray')
    ax.set_title('VIS patch')

    ax = fig.add_subplot(2, 2, 3)
    ax.scatter([gt[0]], [pred[0]], c='tab:blue', s=40, label='DRA*')
    ax.scatter([gt[1]], [pred[1]], c='tab:orange', s=40, label='DDec')
    lim = max(20.0, abs(gt).max(), abs(pred).max())
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('GT (mas)')
    ax.set_ylabel('Pred (mas)')
    ax.legend(loc='upper left')
    ax.set_title('Offset prediction')

    ax = fig.add_subplot(2, 2, 4)
    ax.axis('off')
    txt = (
        f'{split_name} preview\n'
        f"tile: {sample['tile_id']}\n"
        f"target: {sample['target_band']}\n"
        f"input: {', '.join(sample['input_bands'])}\n"
        f'GT DRA*: {gt[0]:+.1f} mas\n'
        f'GT DDec: {gt[1]:+.1f} mas\n'
        f'Pred DRA*: {pred[0]:+.1f} mas\n'
        f'Pred DDec: {pred[1]:+.1f} mas\n'
        f'Pred sigma: {sigma:.1f} mas\n'
        f"Confidence: {float(out['confidence'][0].cpu().item()):.3f}"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top', fontfamily='monospace')

    fig.suptitle(f'Epoch {epoch} | standalone local matcher')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def make_field_preview(
    model,
    device: torch.device,
    preview_pairs: Sequence[tuple[str, str, str]],
    target_band: str,
    input_bands: list[str],
    detect_bands: list[str],
    args,
    epoch: int,
    split_name: str,
):
    if wandb is None:
        return None

    preview_args = SimpleNamespace(
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        min_matches=args.min_matches,
        max_matches=args.max_matches,
        max_sep_arcsec=args.max_sep_arcsec,
        clip_sigma=args.clip_sigma,
        rubin_nsig=args.rubin_nsig,
        vis_nsig=args.vis_nsig,
        rubin_smooth=args.rubin_smooth,
        vis_smooth=args.vis_smooth,
        rubin_min_dist=args.rubin_min_dist,
        vis_min_dist=args.vis_min_dist,
        max_sources_rubin=args.max_sources_rubin,
        max_sources_vis=args.max_sources_vis,
        detect_clip_sigma=args.detect_clip_sigma,
        refine_radius=args.refine_radius,
        refine_flux_floor_sigma=args.refine_flux_floor_sigma,
        grid_h=args.preview_grid_h,
        grid_w=args.preview_grid_w,
        smooth_lambda=args.preview_smooth_lambda,
        dstep=args.preview_dstep,
    )
    for tile_id, rubin_path, euclid_path in preview_pairs:
        item = predict_tile(model, device, rubin_path, euclid_path, target_band, input_bands, detect_bands, preview_args)
        if item is None:
            continue
        fig = make_tile_diagnostic_figure(item, tile_id, target_band, input_bands)
        fig.suptitle(f"Epoch {epoch} | {split_name} fixed tile | standalone local matcher", y=0.99)
        out = wandb.Image(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
        return out
    return None


def run_epoch(split_name, loader, model, optimizer, device, grad_clip: float, pixel_loss_weight: float) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    agg = defaultdict(float)
    n_batches = 0
    n_samples = 0

    for batch in loader:
        rubin = batch['rubin_patch'].float().to(device)
        vis = batch['vis_patch'].float().to(device)
        target = batch['target_offset_arcsec'].float().to(device)
        pix2sky = batch['pixel_to_sky'].float().to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            out = model(rubin, vis, pix2sky)
            losses = compute_loss(out, target, pix2sky, pixel_loss_weight=pixel_loss_weight)
            loss = losses['loss_total']
            if is_train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        metrics = compute_metrics(out, target, pix2sky)
        for k, v in losses.items():
            agg[k] += float(v.detach().item())
        for k, v in metrics.items():
            agg[k] += float(v)
        n_batches += 1
        n_samples += int(target.shape[0])

    denom = max(1, n_batches)
    out = {k: v / denom for k, v in agg.items()}
    out['batches'] = n_batches
    out['samples'] = n_samples
    if hasattr(model, 'temperature'):
        out['temp'] = float(model.temperature.detach().item())
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Train the standalone native-resolution local astrometry matcher.')
    p.add_argument('--rubin-dir', type=str, required=True)
    p.add_argument('--euclid-dir', type=str, required=True)
    p.add_argument('--rubin-band', type=str, default='r')
    p.add_argument('--context-bands', type=str, nargs='+', default=[], help='Optional extra Rubin bands for the local patch matcher.')
    p.add_argument('--detect-bands', type=str, nargs='+', default=['g', 'r', 'i', 'z'])
    p.add_argument('--output-dir', type=str, default='models/checkpoints/astrometry2_matcher')

    p.add_argument('--patch-size', type=int, default=33)
    p.add_argument('--max-patches-per-tile', type=int, default=64)
    p.add_argument('--min-matches', type=int, default=20)
    p.add_argument('--max-matches', type=int, default=256)
    p.add_argument('--max-sep-arcsec', type=float, default=0.12)
    p.add_argument('--clip-sigma', type=float, default=3.5)
    p.add_argument('--rubin-nsig', type=float, default=4.5)
    p.add_argument('--vis-nsig', type=float, default=4.0)
    p.add_argument('--rubin-smooth', type=float, default=1.0)
    p.add_argument('--vis-smooth', type=float, default=1.2)
    p.add_argument('--rubin-min-dist', type=int, default=7)
    p.add_argument('--vis-min-dist', type=int, default=9)
    p.add_argument('--max-sources-rubin', type=int, default=600)
    p.add_argument('--max-sources-vis', type=int, default=800)
    p.add_argument('--detect-clip-sigma', type=float, default=8.0)
    p.add_argument('--refine-radius', type=int, default=3)
    p.add_argument('--refine-flux-floor-sigma', type=float, default=1.5)

    p.add_argument('--hidden-channels', type=int, default=32)
    p.add_argument('--encoder-depth', type=int, default=4)
    p.add_argument('--search-radius', type=int, default=3)
    p.add_argument('--softmax-temp', type=float, default=0.05)
    p.add_argument('--mlp-hidden', type=int, default=128)

    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='')

    p.add_argument('--wandb-project', type=str, default='JAISP-Astrometry2')
    p.add_argument('--wandb-run-name', type=str, default='')
    p.add_argument('--wandb-mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    return p


def train(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_band = normalize_rubin_band(args.rubin_band)
    context_bands = normalize_rubin_bands(args.context_bands)
    detect_bands = normalize_rubin_bands(args.detect_bands) or [f'rubin_{b}' for b in ('g', 'r', 'i', 'z')]

    pairs = discover_tile_pairs(args.rubin_dir, args.euclid_dir)
    train_pairs, val_pairs = split_tile_pairs(pairs, args.val_frac, args.seed)

    common_kwargs = dict(
        target_band=target_band,
        context_bands=context_bands,
        detect_bands=detect_bands,
        patch_size=args.patch_size,
        max_patches_per_tile=args.max_patches_per_tile,
        min_matches=args.min_matches,
        max_matches=args.max_matches,
        max_sep_arcsec=args.max_sep_arcsec,
        clip_sigma=args.clip_sigma,
        rubin_nsig=args.rubin_nsig,
        vis_nsig=args.vis_nsig,
        rubin_smooth=args.rubin_smooth,
        vis_smooth=args.vis_smooth,
        rubin_min_dist=args.rubin_min_dist,
        vis_min_dist=args.vis_min_dist,
        max_sources_rubin=args.max_sources_rubin,
        max_sources_vis=args.max_sources_vis,
        detect_clip_sigma=args.detect_clip_sigma,
        refine_radius=args.refine_radius,
        refine_flux_floor_sigma=args.refine_flux_floor_sigma,
        seed=args.seed,
    )

    train_samples = build_patch_samples(train_pairs, split_name='train', **common_kwargs)
    val_samples = build_patch_samples(val_pairs, split_name='val', **common_kwargs) if val_pairs else []

    train_dataset = MatchedPatchDataset(train_samples)
    val_dataset = MatchedPatchDataset(val_samples) if val_samples else None
    train_loader = make_loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = make_loader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) if val_dataset else None

    rubin_channels = int(train_samples[0]['rubin_patch'].shape[0])
    model = LocalAstrometryMatcher(
        rubin_channels=rubin_channels,
        hidden_channels=args.hidden_channels,
        encoder_depth=args.encoder_depth,
        search_radius=args.search_radius,
        softmax_temp=args.softmax_temp,
        mlp_hidden=args.mlp_hidden,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    preview_sample = val_dataset[0] if val_dataset is not None and len(val_dataset) > 0 else train_dataset[0]
    preview_split = 'val' if val_dataset is not None and len(val_dataset) > 0 else 'train'

    wandb_run = None
    if args.wandb_mode != 'disabled' and wandb is not None:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or None,
                config=vars(args),
                mode=args.wandb_mode,
                dir=str(out_dir),
            )
        except Exception as exc:
            print(f'W&B init failed: {exc}')

    best_score = float('inf')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch('train', train_loader, model, optimizer, device, args.grad_clip)
        scheduler.step()
        train_metrics['lr'] = optimizer.param_groups[0]['lr']

        val_metrics = {}
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch('val', val_loader, model, None, device, args.grad_clip)

        score = val_metrics.get('mae_total', train_metrics['mae_total'])
        elapsed = time.time() - t0
        val_str = ''
        if val_metrics:
            val_str = (
                f" | val_MAE={val_metrics.get('mae_total', 0.0) * 1000:.1f}mas"
                f" val_p68={val_metrics.get('p68_total', 0.0) * 1000:.1f}mas"
            )
        temp_str = f" temp={train_metrics.get('temp', 0.0):.4f}" if 'temp' in train_metrics else ''
        print(
            f"Epoch {epoch:03d} | "
            f"train_MAE={train_metrics.get('mae_total', 0.0) * 1000:.1f}mas "
            f"train_p68={train_metrics.get('p68_total', 0.0) * 1000:.1f}mas "
            f"loss={train_metrics.get('loss_total', 0.0):.5f}"
            f"{val_str}{temp_str} t={elapsed:.1f}s"
        )

        log_data = {f'train/{k}': v for k, v in train_metrics.items()}
        log_data.update({f'val/{k}': v for k, v in val_metrics.items()})
        log_data['epoch'] = epoch
        log_data['time_sec'] = elapsed
        if wandb_run is not None:
            try:
                preview = make_preview(model, preview_sample, device, epoch, preview_split)
                if preview is not None:
                    log_data['preview/fixed_patch'] = preview
            except Exception as exc:
                print(f'Preview rendering failed at epoch {epoch}: {exc}')
            wandb_run.log(log_data, step=epoch)

        save_args = dict(vars(args))
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'args': save_args,
            'rubin_channels': rubin_channels,
            'input_bands': train_samples[0]['input_bands'],
            'target_band': target_band,
        }
        torch.save(ckpt, out_dir / 'last_matcher.pt')
        if score < best_score:
            best_score = score
            torch.save(ckpt, out_dir / 'best_matcher.pt')
            if wandb_run is not None:
                wandb_run.summary['best_val_mae_total_mas'] = score * 1000.0
                wandb_run.summary['best_epoch'] = epoch

        with open(out_dir / 'latest_metrics.json', 'w') as f:
            json.dump({'epoch': epoch, 'time_sec': elapsed, 'train': train_metrics, 'val': val_metrics, 'best_score_arcsec': best_score}, f, indent=2)

    if wandb_run is not None:
        wandb_run.finish()
    print(f'\nDone. Best score: {best_score * 1000.0:.2f} mas')
    print(f'Checkpoints in: {out_dir}')


if __name__ == '__main__':
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault('PYTHONHASHSEED', str(args.seed))
    train(args)
