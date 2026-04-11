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
MODELS_DIR = SCRIPT_DIR.parent
for p in (MODELS_DIR, SCRIPT_DIR):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

from astrometry2.dataset import (
    ALL_BAND_ORDER,
    MatchedPatchDataset,
    build_patch_samples,
    build_patch_samples_multiband,
    discover_tile_pairs,
    make_loader,
    normalize_rubin_band,
    normalize_rubin_bands,
    split_tile_pairs,
)
from astrometry2.infer_concordance import predict_tile
from older_architectures.matcher import LocalAstrometryMatcher
from astrometry2.viz import make_tile_diagnostic_figure, make_band_montage

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
    label_sigma: torch.Tensor = None,
    label_noise_floor: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Rayleigh NLL loss with label-noise-aware effective sigma.

    Motivated by SITCOMTN-159 (Wilson & Naylor 2025): Rubin single-visit
    positions have a ~5 mas systematic floor not reflected in pipeline
    uncertainties.  The training target (VIS-Rubin centroid offset) inherits
    this noise, so the model should not try to fit below it.

    Parameters
    ----------
    label_sigma : [B] per-sample label noise sigma in arcsec, from
        expected_centroid_sigma_arcsec().  If None, uses label_noise_floor
        as a constant floor.
    label_noise_floor : scalar fallback floor (arcsec) when label_sigma
        is not available.  Default 0 preserves backward compatibility;
        set to 0.005 (5 mas) for SITCOMTN-159 motivated training.
    """
    pred = out['pred_offset_arcsec']
    err = pred - target_offset_arcsec
    radial = torch.sqrt((err ** 2).sum(dim=1) + 1e-10)
    sigma_pred = torch.exp(out['log_sigma']).clamp_min(1e-4)

    # Effective sigma includes both model uncertainty and label noise.
    # This prevents the model from overfitting to noisy bright-end targets.
    if label_sigma is not None:
        label_var = label_sigma ** 2
    elif label_noise_floor > 0:
        label_var = torch.full_like(sigma_pred, label_noise_floor ** 2)
    else:
        label_var = torch.zeros_like(sigma_pred)

    sigma_eff = torch.sqrt(sigma_pred ** 2 + label_var)
    log_sigma_eff = torch.log(sigma_eff)

    loss_main = (radial / sigma_eff + log_sigma_eff).mean()

    target_px = _target_pixel_shift(target_offset_arcsec, pixel_to_sky)
    # Weight pixel-space loss by predicted uncertainty so noisy sources are
    # down-weighted consistently with the sky-space NLL.  Detach sigma so
    # the gradient through loss_px doesn't drive sigma to grow artificially.
    sigma_detached = sigma_eff.detach()
    loss_px = (
        (
            torch.nn.functional.smooth_l1_loss(out['dx_px'], target_px[:, 0], reduction='none')
            + torch.nn.functional.smooth_l1_loss(out['dy_px'], target_px[:, 1], reduction='none')
        ) / sigma_detached
    ).mean()
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
    """
    2×3 fixed-patch diagnostic figure logged to W&B each vis_every epochs.

    Row 1: Rubin patch (normalized) | VIS patch (normalized) | Cost volume heatmap
    Row 2: GT vs Pred offset arrows with σ circle (wide) | Stats text
    """
    if wandb is None:
        return None
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Unwrap DataParallel if needed for attribute access.
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()
    with torch.no_grad():
        rubin = sample['rubin_patch'].unsqueeze(0).to(device)
        vis = sample['vis_patch'].unsqueeze(0).to(device)
        pix2sky = sample['pixel_to_sky'].unsqueeze(0).to(device)
        out = raw_model(rubin, vis, pix2sky)

    pred_mas = out['pred_offset_arcsec'][0].cpu().numpy() * 1000.0
    gt_mas = sample['target_offset_arcsec'].numpy() * 1000.0
    sigma_mas = float(torch.exp(out['log_sigma'][0]).cpu().item()) * 1000.0
    conf = float(out['confidence'][0].cpu().item())
    coarse_dx = float(out['coarse_dx_px'][0].cpu().item())
    coarse_dy = float(out['coarse_dy_px'][0].cpu().item())
    mlp_dx = float((out['dx_px'] - out['coarse_dx_px'])[0].cpu().item())
    mlp_dy = float((out['dy_px'] - out['coarse_dy_px'])[0].cpu().item())
    err_mas = float(np.hypot(pred_mas[0] - gt_mas[0], pred_mas[1] - gt_mas[1]))

    # Patches are already background-subtracted + noise-normalized (from __getitem__).
    rubin_np = sample['rubin_patch'].numpy()[0]   # first Rubin channel
    vis_np = sample['vis_patch'].numpy()[0]
    r = raw_model.search_radius

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    def _show_patch(ax, img, title, cmap):
        p1, p99 = np.percentile(img, [1, 99])
        if p1 >= p99:
            p1, p99 = float(img.min()), float(img.max()) + 1e-6
        ax.imshow(img, origin='lower', cmap=cmap, vmin=p1, vmax=p99)
        h, w = img.shape
        ax.axhline(h / 2, color='cyan', lw=0.7, alpha=0.7)
        ax.axvline(w / 2, color='cyan', lw=0.7, alpha=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('x (VIS px)', fontsize=8)
        ax.tick_params(labelsize=7)

    band_name = str(sample['target_band'])
    if band_name.startswith('nisp_'):
        patch_title = f"NISP {band_name.split('_', 1)[1]}-band\n(bkg-sub, noise-norm)"
    else:
        patch_title = f"Rubin {band_name.split('_', 1)[1]}-band\n(bkg-sub, noise-norm)"

    # (0,0) First input patch channel
    _show_patch(fig.add_subplot(gs[0, 0]), rubin_np,
                patch_title,
                'magma')

    # (0,1) VIS patch — the reference frame
    _show_patch(fig.add_subplot(gs[0, 1]), vis_np,
                'Euclid VIS\n(bkg-sub, noise-norm)', 'gray')

    # (0,2) Cost volume — the (2r+1)² correlation scores between Rubin and VIS.
    # The bright peak shows where the coarse cross-correlation found the best match.
    # A sharp, isolated peak means high confidence. A flat/noisy map means poor match.
    ax = fig.add_subplot(gs[0, 2])
    if r > 0 and 'logits' in out:
        logits_np = out['logits'][0].cpu().numpy()
        cost_vol = logits_np.reshape(2 * r + 1, 2 * r + 1)
        im = ax.imshow(cost_vol, origin='lower', cmap='hot',
                       extent=[-r - 0.5, r + 0.5, -r - 0.5, r + 0.5])
        plt.colorbar(im, ax=ax, fraction=0.046, label='cosine sim', pad=0.02)
        ax.plot(coarse_dx, coarse_dy, 'c+', ms=12, mew=2.0,
                label=f'coarse peak\n({coarse_dx:+.2f}, {coarse_dy:+.2f}) px')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlabel('Δx (pixels)', fontsize=8)
        ax.set_ylabel('Δy (pixels)', fontsize=8)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'search_radius=0', ha='center', va='center',
                transform=ax.transAxes, fontsize=9)
    ax.set_title('Cost volume  (bright peak = coarse match)\nsharp isolated peak → high confidence', fontsize=9)
    ax.tick_params(labelsize=7)

    # (1, 0:2) Sky-plane offset diagram — GT and Pred arrows from origin.
    # The orange circle shows the model's predicted 1σ uncertainty radius.
    # The closer the red arrow (Pred) is to the blue arrow (GT), the better.
    ax = fig.add_subplot(gs[1, :2])
    lim = max(30.0, float(np.abs(np.concatenate([gt_mas, pred_mas])).max()) * 1.6, sigma_mas * 2.5)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill(sigma_mas * np.cos(theta), sigma_mas * np.sin(theta),
            alpha=0.15, color='orange', zorder=1)
    ax.plot(sigma_mas * np.cos(theta), sigma_mas * np.sin(theta),
            color='darkorange', lw=1.0, ls='--', zorder=2,
            label=f'1σ = {sigma_mas:.1f} mas  (predicted uncertainty radius)')
    ax.annotate('', xy=(gt_mas[0], gt_mas[1]), xytext=(0, 0), zorder=3,
                arrowprops=dict(arrowstyle='->', color='tab:blue', lw=2.2))
    ax.annotate('', xy=(pred_mas[0], pred_mas[1]), xytext=(0, 0), zorder=3,
                arrowprops=dict(arrowstyle='->', color='tab:red', lw=2.2))
    ax.scatter([gt_mas[0]], [gt_mas[1]], c='tab:blue', s=70, zorder=5,
               label=f'Ground truth   DRA*={gt_mas[0]:+.1f}  DDec={gt_mas[1]:+.1f} mas')
    ax.scatter([pred_mas[0]], [pred_mas[1]], c='tab:red', s=70, zorder=5,
               label=f'NN prediction  DRA*={pred_mas[0]:+.1f}  DDec={pred_mas[1]:+.1f} mas')
    ax.axhline(0, color='lightgray', lw=0.8)
    ax.axvline(0, color='lightgray', lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('ΔRA* (mas)  →  East', fontsize=9)
    ax.set_ylabel('ΔDec (mas)  →  North', fontsize=9)
    ax.set_title('Predicted vs ground-truth sky offset  (arrows from origin = sky displacement from Rubin to VIS)', fontsize=9)
    ax.legend(loc='upper left', fontsize=8)
    ax.tick_params(labelsize=7)

    # (1,2) Stats text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    calib = err_mas / max(sigma_mas, 1e-3)
    calib_note = ('good' if 0.3 < calib < 3.0 else
                  'overconfident — σ too small' if calib > 3.0 else
                  'underconfident — σ too large')
    txt = (
        f'{split_name}  —  epoch {epoch}\n\n'
        f"Tile:   {sample['tile_id']}\n"
        f"Band:   {sample['target_band']}\n\n"
        f'GT:      ({gt_mas[0]:+.1f}, {gt_mas[1]:+.1f}) mas\n'
        f'Pred:    ({pred_mas[0]:+.1f}, {pred_mas[1]:+.1f}) mas\n'
        f'|error|:  {err_mas:.1f} mas\n\n'
        f'σ (predicted): {sigma_mas:.1f} mas\n'
        f'|err|/σ: {calib:.2f}  [{calib_note}]\n\n'
        f'Confidence:  {conf:.3f}\n'
        f'Temperature: {float(raw_model.temperature.detach().item()):.4f}\n\n'
        f'Coarse (cost vol): ({coarse_dx:+.2f}, {coarse_dy:+.2f}) px\n'
        f'MLP residual:      ({mlp_dx:+.2f}, {mlp_dy:+.2f}) px'
    )
    ax.text(0.05, 0.97, txt, transform=ax.transAxes, va='top',
            fontfamily='monospace', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', alpha=0.95))

    fig.suptitle(f'Epoch {epoch}  |  fixed-patch preview  |  standalone local matcher',
                 y=1.0, fontsize=11)
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def make_field_preview(
    model,
    device: torch.device,
    preview_pairs: Sequence[tuple[str, str, str]],
    target_bands: Sequence[str],
    input_bands: list[str],
    detect_bands: list[str],
    args,
    epoch: int,
    split_name: str,
):
    if wandb is None:
        return {}

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
        anchor_lambda=getattr(args, 'anchor_lambda', 1e-3),
        anchor_radius_px=getattr(args, 'anchor_radius_px', 0.0),
        auto_grid=True,  # always use adaptive grid + anchor for previews
        dstep=args.preview_dstep,
    )
    if hasattr(args, 'detector_conf_threshold'):
        preview_args.detector_conf_threshold = getattr(args, 'detector_conf_threshold')
    if hasattr(args, '_detr_detector'):
        preview_args._detr_detector = getattr(args, '_detr_detector')
    out: Dict[str, object] = {}
    max_preview_tiles = int(getattr(args, 'preview_max_tiles', 8))
    preview_cols = 5
    if len(target_bands) > 1:
        if not target_bands:
            return {}
        ref_band = 'rubin_r' if 'rubin_r' in target_bands else target_bands[0]
        best = None
        best_matches = -1
        for tile_id, rubin_path, euclid_path in preview_pairs[:max_preview_tiles]:
            try:
                item = predict_tile(model, device, rubin_path, euclid_path, ref_band, input_bands, detect_bands, preview_args)
            except Exception as exc:
                print(f'[preview] skip {tile_id} {ref_band}: {exc}')
                continue
            if item is None:
                continue
            n_matches = int(item.get('summary', {}).get('matches', 0))
            if n_matches > best_matches:
                best = (tile_id, rubin_path, euclid_path)
                best_matches = n_matches
        if best is None:
            return {}
        tile_id, rubin_path, euclid_path = best

        items = {}
        for target_band in target_bands:
            try:
                item = predict_tile(model, device, rubin_path, euclid_path, target_band, input_bands, detect_bands, preview_args)
            except Exception as exc:
                print(f'[preview] skip {tile_id} {target_band}: {exc}')
                continue
            if item is None:
                continue
            items[target_band] = item

        if not items:
            return {}

        montage_title = (
            f"Epoch {epoch} | {split_name} representative tile | {tile_id} "
            f"(bands={len(items)})"
        )
        montages = [
            ("anchors", "NN predicted corrections at source anchors"),
            ("field", "Solved field magnitude + direction"),
            ("residual_hist", "Residual offset components"),
            ("raw_vs_pred", "Raw WCS vs NN corrections"),
        ]
        for key, label in montages:
            fig = make_band_montage(items, key, ncols=preview_cols, title=f"{montage_title} | {label}")
            if fig is None:
                continue
            out[f'preview/montage/{key}'] = wandb.Image(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
        return out

    for target_band in target_bands:
        best = None
        best_matches = -1
        for tile_id, rubin_path, euclid_path in preview_pairs[:max_preview_tiles]:
            try:
                item = predict_tile(model, device, rubin_path, euclid_path, target_band, input_bands, detect_bands, preview_args)
            except Exception as exc:
                print(f'[preview] skip {tile_id} {target_band}: {exc}')
                continue
            if item is None:
                continue
            n_matches = int(item.get('summary', {}).get('matches', 0))
            if n_matches > best_matches:
                best = (tile_id, item)
                best_matches = n_matches
        if best is None:
            continue
        tile_id, item = best
        fig = make_tile_diagnostic_figure(item, tile_id, target_band, input_bands)
        fig.suptitle(
            f"Epoch {epoch} | {split_name} representative tile | {target_band} "
            f"(best of {min(len(preview_pairs), max_preview_tiles)})",
            y=0.99,
        )
        out[f'preview/fixed_tile/{target_band}'] = wandb.Image(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
    return out


def run_epoch(
    split_name, loader, model, optimizer, device,
    grad_clip: float, pixel_loss_weight: float,
    label_noise_floor: float = 0.0,
) -> Dict[str, float]:
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

        band_idx = batch.get('band_idx')
        if band_idx is not None:
            band_idx = band_idx.to(device)

        # Per-pixel RMS (available when dataset extracted variance maps).
        rms_kwargs = {}
        if 'rubin_rms_patch' in batch:
            rms_kwargs['rubin_rms'] = batch['rubin_rms_patch'].float().to(device)
            rms_kwargs['vis_rms'] = batch['vis_rms_patch'].float().to(device)

        # Per-source label noise from SITCOMTN-159 motivated centroid uncertainty.
        label_sigma = None
        if 'label_sigma_arcsec' in batch:
            label_sigma = batch['label_sigma_arcsec'].float().to(device)

        with torch.set_grad_enabled(is_train):
            out = model(rubin, vis, pix2sky, band_idx=band_idx, **rms_kwargs)
            losses = compute_loss(
                out, target, pix2sky,
                pixel_loss_weight=pixel_loss_weight,
                label_sigma=label_sigma,
                label_noise_floor=label_noise_floor,
            )
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
    raw = model.module if hasattr(model, 'module') else model
    if hasattr(raw, 'temperature'):
        out['temp'] = float(raw.temperature.detach().item())
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Train the standalone native-resolution local astrometry matcher.')
    p.add_argument('--rubin-dir', type=str, required=True)
    p.add_argument('--euclid-dir', type=str, required=True)
    p.add_argument('--rubin-band', type=str, default='r')
    p.add_argument('--context-bands', type=str, nargs='+', default=[], help='Optional extra Rubin bands for the local patch matcher.')
    p.add_argument('--detect-bands', type=str, nargs='+', default=['g', 'r', 'i', 'z'])
    # Multi-band mode
    p.add_argument('--multiband', action='store_true',
                   help='Train in multi-band mode: all 6 Rubin bands as input, per-band MLP head.')
    p.add_argument('--target-bands', type=str, nargs='+', default=['all', 'all_nisp'],
                   help='Target bands for multiband mode. "all" = all 6 Rubin bands, '
                        '"all_nisp" adds NISP Y/J/H. Default: all + all_nisp.')
    p.add_argument('--bands', type=str, nargs='+', default=None,
                   help='Alias for --target-bands. Accepts Rubin letters (u g r i z y), '
                        'and NISP bands as "nisp_Y/J/H" or uppercase "Y/J/H".')
    p.add_argument('--include-nisp', action='store_true',
                   help='Include NISP Y/J/H as input channels (multiband mode only).')
    p.add_argument('--band-embed-dim', type=int, default=16,
                   help='Band embedding dimension for the multiband MLP head.')
    p.add_argument('--output-dir', type=str, default='models/checkpoints/astrometry2_matcher')

    p.add_argument('--patch-size', type=int, default=33)
    p.add_argument('--max-patches-per-tile', type=int, default=64)
    p.add_argument('--offset-bias', type=float, default=0.5,
                   help='Bias sampling toward larger raw offsets when subsampling patches. '
                        '0 = uniform, 1 = fully magnitude-weighted (default: 0.5).')
    p.add_argument('--offset-bias-power', type=float, default=1.0,
                   help='Power applied to |offset| when building sampling weights (default: 1.0).')
    p.add_argument('--offset-bias-floor-mas', type=float, default=5.0,
                   help='Additive floor (mas) before weighting to avoid zero-probability (default: 5).')
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
    p.add_argument('--pixel-loss-weight', type=float, default=1.0)
    p.add_argument('--label-noise-floor', type=float, default=0.005,
                   help='Label noise floor in arcsec (SITCOMTN-159: ~5 mas for Rubin single-visit). '
                        'Added in quadrature to predicted sigma in the Rayleigh NLL loss to prevent '
                        'overfitting to noisy bright-source labels. Set to 0 to disable. (default: 0.005)')
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='')
    p.add_argument('--vis-every', type=int, default=1,
                   help='Render the fixed patch/tile previews every N epochs.')
    p.add_argument('--jitter-arcsec', type=float, default=0.0,
                   help='Gaussian jitter (arcsec) applied to Rubin patches during training '
                        'to force learning tiny offsets. 0 disables (default: 0).')
    p.add_argument('--jitter-max-arcsec', type=float, default=0.0,
                   help='Clip jitter to +/- this arcsec (default: 0 = no clip).')
    p.add_argument('--jitter-prob', type=float, default=1.0,
                   help='Probability of applying jitter per sample (default: 1).')
    p.add_argument('--preview-grid-h', type=int, default=12)
    p.add_argument('--preview-grid-w', type=int, default=12)
    p.add_argument('--preview-smooth-lambda', type=float, default=1e-2)
    p.add_argument('--preview-dstep', type=int, default=4)
    p.add_argument('--preview-max-tiles', type=int, default=8,
                   help='Preview chooser: search this many val tiles and show the one with the most matches.')

    p.add_argument('--wandb-project', type=str, default='JAISP-Astrometry2')
    p.add_argument('--wandb-run-name', type=str, default='')
    p.add_argument('--wandb-mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    return p


def apply_band_overrides(args):
    """Normalize band-related args for multiband training/inference."""
    # Alias: --bands overrides --target-bands if provided.
    if getattr(args, 'bands', None):
        expanded = []
        for item in args.bands:
            expanded.extend([p.strip() for p in str(item).split(',') if str(p).strip()])
        args.target_bands = expanded
    elif getattr(args, 'target_bands', None):
        # Expand comma-separated entries in target_bands too.
        expanded = []
        for item in args.target_bands:
            expanded.extend([p.strip() for p in str(item).split(',') if str(p).strip()])
        args.target_bands = expanded

    # Default to Rubin + NISP if target_bands missing/empty.
    if not getattr(args, 'target_bands', None):
        args.target_bands = ['all', 'all_nisp']

    # Auto-enable NISP inputs if any NISP targets are requested.
    wants_nisp = False
    for b in args.target_bands:
        s = str(b).strip()
        if not s:
            continue
        if s.lower().startswith('nisp_') or s.lower() == 'all_nisp' or s in ('Y', 'J', 'H'):
            wants_nisp = True
            break
    if wants_nisp:
        args.include_nisp = True
    return args


def train(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detect_bands = normalize_rubin_bands(args.detect_bands) or [f'rubin_{b}' for b in ('g', 'r', 'i', 'z')]

    pairs = discover_tile_pairs(args.rubin_dir, args.euclid_dir)
    train_pairs, val_pairs = split_tile_pairs(pairs, args.val_frac, args.seed)

    detection_kwargs = dict(
        detect_bands=detect_bands,
        patch_size=args.patch_size,
        max_patches_per_tile=args.max_patches_per_tile,
        offset_bias=args.offset_bias,
        offset_bias_power=args.offset_bias_power,
        offset_bias_floor_mas=args.offset_bias_floor_mas,
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

    if getattr(args, 'multiband', False):
        multiband_kwargs = dict(
            target_bands=args.target_bands,
            include_nisp=args.include_nisp,
            **detection_kwargs,
        )
        train_samples = build_patch_samples_multiband(train_pairs, split_name='train', **multiband_kwargs)
        val_samples = build_patch_samples_multiband(val_pairs, split_name='val', **multiband_kwargs) if val_pairs else []
        # Samples carry global band indices (dataset.ALL_BAND_ORDER).  Size the
        # embedding table by max index + 1 so sparse subsets remain valid.
        all_band_idxs = sorted(set(s['band_idx'] for s in train_samples))
        n_target_bands = max(all_band_idxs) + 1
        if n_target_bands > len(ALL_BAND_ORDER):
            raise RuntimeError(
                f'Invalid band_idx in training samples: max={max(all_band_idxs)} '
                f'but supported range is [0, {len(ALL_BAND_ORDER) - 1}]'
            )
        target_band = 'multiband'
        preview_target_bands = sorted(
            set(s['target_band'] for s in train_samples),
            key=lambda b: ALL_BAND_ORDER.index(b) if b in ALL_BAND_ORDER else len(ALL_BAND_ORDER),
        )
    else:
        target_band = normalize_rubin_band(args.rubin_band)
        context_bands = normalize_rubin_bands(args.context_bands)
        common_kwargs = dict(
            target_band=target_band,
            context_bands=context_bands,
            **detection_kwargs,
        )
        train_samples = build_patch_samples(train_pairs, split_name='train', **common_kwargs)
        val_samples = build_patch_samples(val_pairs, split_name='val', **common_kwargs) if val_pairs else []
        n_target_bands = 1
        preview_target_bands = [target_band]

    train_dataset = MatchedPatchDataset(
        train_samples,
        augment=True,
        jitter_arcsec=args.jitter_arcsec,
        jitter_max_arcsec=args.jitter_max_arcsec,
        jitter_prob=args.jitter_prob,
    )
    val_dataset = MatchedPatchDataset(val_samples, augment=False) if val_samples else None
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
        n_target_bands=n_target_bands,
        band_embed_dim=getattr(args, 'band_embed_dim', 16),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    preview_sample = val_dataset[0] if val_dataset is not None and len(val_dataset) > 0 else train_dataset[0]
    preview_split = 'val' if val_dataset is not None and len(val_dataset) > 0 else 'train'
    preview_pairs = val_pairs if val_pairs else train_pairs

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
        train_metrics = run_epoch(
            'train',
            train_loader,
            model,
            optimizer,
            device,
            args.grad_clip,
            args.pixel_loss_weight,
        )
        scheduler.step()
        train_metrics['lr'] = optimizer.param_groups[0]['lr']

        val_metrics = {}
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    'val',
                    val_loader,
                    model,
                    None,
                    device,
                    args.grad_clip,
                    args.pixel_loss_weight,
                )

        score = val_metrics.get('mae_total', train_metrics['mae_total'])
        elapsed = time.time() - t0
        val_str = ''
        if val_metrics:
            val_str = (
                f" | val_MAE={val_metrics.get('mae_total', 0.0) * 1000:.1f}mas"
                f" val_p68={val_metrics.get('p68_total', 0.0) * 1000:.1f}mas"
                f" val_px={val_metrics.get('mae_px', 0.0):.3f}px"
            )
        temp_str = f" temp={train_metrics.get('temp', 0.0):.4f}" if 'temp' in train_metrics else ''
        print(
            f"Epoch {epoch:03d} | "
            f"train_MAE={train_metrics.get('mae_total', 0.0) * 1000:.1f}mas "
            f"train_p68={train_metrics.get('p68_total', 0.0) * 1000:.1f}mas "
            f"train_px={train_metrics.get('mae_px', 0.0):.3f}px "
            f"loss={train_metrics.get('loss_total', 0.0):.5f}"
            f"{val_str}{temp_str} t={elapsed:.1f}s"
        )

        log_data = {f'train/{k}': v for k, v in train_metrics.items()}
        log_data.update({f'val/{k}': v for k, v in val_metrics.items()})
        log_data['epoch'] = epoch
        log_data['time_sec'] = elapsed
        if wandb_run is not None:
            try:
                if epoch % max(1, int(args.vis_every)) == 0:
                    preview = make_preview(model, preview_sample, device, epoch, preview_split)
                    if preview is not None:
                        log_data['preview/fixed_patch'] = preview
                    field_previews = make_field_preview(
                        model,
                        device,
                        preview_pairs,
                        preview_target_bands,
                        train_samples[0]['input_bands'],
                        detect_bands,
                        args,
                        epoch,
                        preview_split,
                    )
                    if field_previews:
                        log_data.update(field_previews)
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
            'n_target_bands': n_target_bands,
            'input_bands': train_samples[0]['input_bands'],
            'target_band': target_band,
            'target_bands': list(set(s['target_band'] for s in train_samples)),
            'include_nisp': getattr(args, 'include_nisp', False),
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
    args = apply_band_overrides(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault('PYTHONHASHSEED', str(args.seed))
    train(args)
