"""Train the latent-space canonical position head.

Tile-level training loop:
  1. Load a full tile (all 10 bands + RMS).
  2. Run the frozen V7 encoder once → bottleneck + VIS stem features.
  3. Detect sources in VIS, PSF-refine centroids → ground-truth positions.
  4. Jitter the centroids to create approximate "input" positions.
  5. Run the trainable LatentPositionHead on the encoder features.
  6. Loss: Rayleigh NLL on the offset from jittered → true position.

The jitter training strategy teaches the head to recover canonical source
positions from the full multi-band latent representation.  At inference,
any rough initial position (from detection, catalog, or cross-match) can
be refined to the chromatically-informed canonical position.

Usage:
    python train_latent_position.py \
        --rubin-dir  /path/to/rubin_tiles \
        --euclid-dir /path/to/euclid_tiles \
        --v7-checkpoint /path/to/jaisp_v7_baseline.pt \
        --epochs 30 --lr 3e-4
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    build_full_context_detector_inputs,
    detect_sources,
    detect_sources_multiband,
    discover_tile_pairs,
    local_vis_pixel_to_sky_matrix,
    signal_mask_in_band,
    split_tile_pairs,
    _to_float32,
)
from astrometry2.latent_position_head import (
    FrozenEncoder,
    LatentPositionHead,
    load_latent_position_head,
)
from source_matching import (
    refine_centroids_psf_fit,
    safe_header_from_card_string,
)
from astropy.wcs import WCS

try:
    import wandb
except ImportError:
    wandb = None


# ============================================================
# Tile-level data processing
# ============================================================

# Re-exported from models/foundation_utils.py — see that module for the
# canonical definition. Existing imports of ``load_tile_data`` from this
# module continue to work via this alias.
from foundation_utils import load_tile_data  # noqa: F401, E402


def load_cached_bottleneck(
    cache_dir: Optional[Path],
    tile_id: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Load a cached foundation bottleneck for ``tile_id`` if present."""
    if cache_dir is None:
        return None
    for name in (f'{tile_id}_aug0.pt', f'{tile_id}.pt'):
        path = cache_dir / name
        if not path.exists():
            continue
        payload = torch.load(str(path), map_location=device, weights_only=False)
        feat = payload['features'] if isinstance(payload, dict) and 'features' in payload else payload
        feat = feat.float().to(device)
        if feat.ndim == 3:
            feat = feat.unsqueeze(0)
        return feat
    return None


@torch.no_grad()
def encode_tile_features(
    frozen_encoder: FrozenEncoder,
    tile_id: str,
    img_t: Dict[str, torch.Tensor],
    rms_t: Dict[str, torch.Tensor],
    device: torch.device,
    features_cache_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Return bottleneck + VIS stem, using cached bottleneck when available."""
    cached = load_cached_bottleneck(features_cache_dir, tile_id, device)
    if cached is None:
        return frozen_encoder.encode_tile(img_t, rms_t)

    vis_img = img_t['euclid_VIS']
    vis_rms = rms_t['euclid_VIS']
    vis_stem = frozen_encoder.vis_stem(vis_img, vis_rms)
    return {
        'bottleneck': cached,
        'vis_stem': vis_stem,
        'fused_hw': (cached.shape[-2], cached.shape[-1]),
        'vis_hw': (vis_img.shape[-2], vis_img.shape[-1]),
    }


def detect_and_refine_vis(
    vis_img: np.ndarray,
    vis_wcs: WCS,
    *,
    nsig: float = 4.0,
    smooth: float = 1.2,
    min_dist: int = 9,
    max_sources: int = 800,
    refine_radius: int = 3,
    flux_floor_sigma: float = 1.5,
    psf_field=None,
    epsf_head=None,
    vis_rms: np.ndarray = None,
    epsf_features: Dict[str, object] = None,
    epsf_stamp_size: int = 21,
    epsf_search_radius_px: float = 0.5,
    epsf_search_steps: int = 7,
    psf_device=None,
    detections_vis_px: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect sources in VIS and refine centroids.

    Uses FoundationEPSFHead or archived PSFField template centroiding when
    provided; otherwise falls back to the classical 2D-Gaussian
    ``refine_centroids_psf_fit``.

    If ``detections_vis_px`` is given, skip internal detection and use those
    positions as candidates (e.g. CenterNet pseudo-labels).

    Returns
    -------
    vis_xy : [N, 2]  refined centroids (x, y) in VIS pixels
    vis_snr : [N]  peak SNR per source
    """
    if detections_vis_px is not None and detections_vis_px.shape[0] > 0:
        seed_xy = detections_vis_px.astype(np.float32)
    else:
        vx, vy = detect_sources(
            vis_img, nsig=nsig, smooth_sigma=smooth,
            min_dist=min_dist, max_sources=max_sources,
        )
        if vx.size == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)
        seed_xy = np.stack([vx, vy], axis=1).astype(np.float32)

    # Filter by signal presence.
    keep = signal_mask_in_band(
        vis_img, seed_xy, radius=refine_radius, flux_floor_sigma=flux_floor_sigma,
    )
    if not keep.any():
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)
    seed_xy = seed_xy[keep]

    # Foundation ePSF centroiding, archived PSFField centroiding, or Gaussian fallback.
    if epsf_head is not None:
        from psf.epsf_centroid_refinement import refine_centroids_foundation_epsf
        epsf_features = epsf_features or {}
        vis_xy, vis_snr, _ = refine_centroids_foundation_epsf(
            vis_img,
            seed_xy,
            epsf_head,
            band_name='euclid_VIS',
            tile_hw=vis_img.shape,
            rms=vis_rms,
            bottleneck=epsf_features.get('bottleneck'),
            vis_stem_features=epsf_features.get('vis_stem'),
            source_positions_vis=seed_xy,
            fused_hw=epsf_features.get('fused_hw'),
            vis_hw=epsf_features.get('vis_hw', vis_img.shape),
            stamp_size=epsf_stamp_size,
            search_radius_px=epsf_search_radius_px,
            search_steps=epsf_search_steps,
            flux_floor_sigma=flux_floor_sigma,
        )
    elif psf_field is not None:
        from psf.centroid_refinement import refine_centroids_psf_field
        vis_xy, vis_snr, _ = refine_centroids_psf_field(
            vis_img, seed_xy, psf_field,
            band_name='euclid_VIS',
            tile_hw=vis_img.shape,
            device=psf_device,
            radius=refine_radius,
            flux_floor_sigma=flux_floor_sigma,
        )
    else:
        vis_xy, vis_snr, _ = refine_centroids_psf_fit(
            vis_img, seed_xy, radius=refine_radius,
            flux_floor_sigma=flux_floor_sigma, fwhm_guess=2.5,
        )
    return vis_xy.astype(np.float32), vis_snr.astype(np.float32)


def build_tile_batch(
    vis_xy: np.ndarray,
    vis_wcs: WCS,
    jitter_arcsec: float,
    jitter_max_arcsec: float,
    max_sources: int,
    rng: np.random.RandomState,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Build per-source tensors for one tile.

    Jitters the VIS PSF-fit centroids to create approximate input positions.
    Target offset = (true_centroid - jittered_position) converted to arcsec
    via the local pixel→sky Jacobian.

    Returns None if the tile has too few sources.
    """
    N = vis_xy.shape[0]
    if N < 5:
        return None

    # Subsample if too many sources.
    if N > max_sources:
        idx = rng.choice(N, max_sources, replace=False)
        vis_xy = vis_xy[idx]
        N = max_sources

    # Jitter: add Gaussian noise in VIS pixel space.
    jitter_px = jitter_arcsec / 0.1  # Convert arcsec → VIS pixels (0.1"/px)
    jitter = rng.normal(scale=max(jitter_px, 0.01), size=(N, 2)).astype(np.float32)
    if jitter_max_arcsec > 0:
        max_px = jitter_max_arcsec / 0.1
        jitter = np.clip(jitter, -max_px, max_px)
    vis_xy_jittered = vis_xy + jitter

    # Compute per-source Jacobian and target offset (in arcsec).
    target_offsets = np.zeros((N, 2), dtype=np.float32)
    pix2sky_arr = np.zeros((N, 2, 2), dtype=np.float32)

    for i in range(N):
        p2s = local_vis_pixel_to_sky_matrix(vis_wcs, vis_xy_jittered[i])
        dx = vis_xy[i, 0] - vis_xy_jittered[i, 0]
        dy = vis_xy[i, 1] - vis_xy_jittered[i, 1]
        target_offsets[i] = p2s @ np.array([dx, dy], dtype=np.float32)
        pix2sky_arr[i] = p2s

    return {
        'positions': torch.from_numpy(vis_xy_jittered).to(device),
        'target_offset_arcsec': torch.from_numpy(target_offsets).to(device),
        'pixel_to_sky': torch.from_numpy(pix2sky_arr).to(device),
    }


# ============================================================
# Loss and metrics
# ============================================================

def compute_loss(
    out: Dict[str, torch.Tensor],
    target_offset_arcsec: torch.Tensor,
    pixel_to_sky: torch.Tensor,
    label_noise_floor: float = 0.005,
) -> Dict[str, torch.Tensor]:
    """Rayleigh NLL loss with label-noise-aware effective sigma."""
    pred = out['pred_offset_arcsec']
    err = pred - target_offset_arcsec
    radial = torch.sqrt((err ** 2).sum(dim=1) + 1e-10)
    sigma_pred = torch.exp(out['log_sigma']).clamp_min(1e-4)

    label_var = torch.full_like(sigma_pred, label_noise_floor ** 2)
    sigma_eff = torch.sqrt(sigma_pred ** 2 + label_var)

    loss_main = (radial / sigma_eff + torch.log(sigma_eff)).mean()

    # Pixel-space smooth L1 loss for direct supervision.
    inv = torch.linalg.pinv(pixel_to_sky)
    target_px = torch.bmm(inv, target_offset_arcsec.unsqueeze(-1)).squeeze(-1)
    sigma_det = sigma_eff.detach()
    loss_px = (
        (
            torch.nn.functional.smooth_l1_loss(out['dx_px'], target_px[:, 0], reduction='none')
            + torch.nn.functional.smooth_l1_loss(out['dy_px'], target_px[:, 1], reduction='none')
        ) / sigma_det
    ).mean()

    loss_total = loss_main + loss_px
    return {
        'loss_total': loss_total,
        'loss_main': loss_main,
        'loss_px': loss_px,
    }


@torch.no_grad()
def compute_metrics(
    out: Dict[str, torch.Tensor],
    target_offset_arcsec: torch.Tensor,
) -> Dict[str, float]:
    pred = out['pred_offset_arcsec']
    err = pred - target_offset_arcsec
    err_ra = err[:, 0].abs()
    err_dec = err[:, 1].abs()
    err_total = torch.sqrt(err_ra ** 2 + err_dec ** 2 + 1e-10)
    sigma = torch.exp(out['log_sigma'])
    return {
        'mae_ra': float(err_ra.mean()),
        'mae_dec': float(err_dec.mean()),
        'mae_total': float(err_total.mean()),
        'p68_total': float(torch.quantile(err_total, 0.68)),
        'frac_01arcsec': float((err_total < 0.1).float().mean()),
        'sigma_median': float(sigma.median()),
    }


# ============================================================
# Visualization (logged to W&B)
# ============================================================

def make_preview_figure(
    head: LatentPositionHead,
    frozen_encoder: FrozenEncoder,
    tile_pair: Tuple[str, str, str],
    device: torch.device,
    args,
    epoch: int,
    rng: np.random.RandomState,
):
    """Generate a 2x2 diagnostic figure for one val tile.

    Panel layout:
      (0,0) Predicted vs GT offset scatter (RA* and Dec)
      (0,1) Radial error histogram + predicted sigma distribution
      (1,0) VIS image with quiver arrows (GT blue, Pred red)
      (1,1) Radial error vs confidence (1/sigma)
    """
    if wandb is None:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tile_id, rubin_path, euclid_path = tile_pair
    head.eval()

    try:
        img_t, rms_t, vis_hw, vis_wcs = load_tile_data(rubin_path, euclid_path, device)
        enc_out = encode_tile_features(
            frozen_encoder,
            tile_id,
            img_t,
            rms_t,
            device,
            features_cache_dir=getattr(args, '_features_cache_dir', None),
        )
        del img_t, rms_t

        edata = np.load(euclid_path, allow_pickle=True)
        vis_img_np = np.nan_to_num(_to_float32(edata['img_VIS']), nan=0.0)
        vis_var_np = _to_float32(edata['var_VIS']) if 'var_VIS' in edata.files else None
        vis_rms_np = (
            np.maximum(np.nan_to_num(np.sqrt(np.clip(vis_var_np, 0, None)), nan=1.0), 1e-10)
            if vis_var_np is not None else None
        )

        vis_xy, vis_snr = detect_and_refine_vis(
            vis_img_np, vis_wcs,
            nsig=args.vis_nsig, smooth=args.vis_smooth,
            min_dist=args.vis_min_dist, max_sources=args.max_sources_vis,
            refine_radius=args.refine_radius,
            flux_floor_sigma=args.refine_flux_floor_sigma,
            epsf_head=getattr(args, '_epsf_head', None),
            vis_rms=vis_rms_np,
            epsf_features={
                'bottleneck': enc_out['bottleneck'],
                'vis_stem': enc_out['vis_stem'],
                'fused_hw': enc_out['fused_hw'],
                'vis_hw': vis_hw,
            },
            epsf_stamp_size=args.epsf_centroid_stamp_size,
            epsf_search_radius_px=args.epsf_centroid_search_px,
            epsf_search_steps=args.epsf_centroid_search_steps,
        )
        if vis_xy.shape[0] < 5:
            return None

        batch = build_tile_batch(
            vis_xy, vis_wcs,
            jitter_arcsec=args.jitter_arcsec,
            jitter_max_arcsec=args.jitter_max_arcsec,
            max_sources=args.max_sources_per_tile,
            rng=rng, device=device,
        )
        if batch is None:
            return None

        with torch.no_grad():
            out = head(
                enc_out['bottleneck'], enc_out['vis_stem'],
                batch['positions'], batch['pixel_to_sky'],
                enc_out['fused_hw'], vis_hw,
            )
    except Exception as exc:
        print(f'[preview] skip {tile_id}: {exc}')
        return None

    # Extract numpy arrays in mas.
    pred = out['pred_offset_arcsec'].cpu().numpy() * 1000.0
    gt = batch['target_offset_arcsec'].cpu().numpy() * 1000.0
    sigma = torch.exp(out['log_sigma']).cpu().numpy() * 1000.0
    positions = batch['positions'].cpu().numpy()
    err_ra = pred[:, 0] - gt[:, 0]
    err_dec = pred[:, 1] - gt[:, 1]
    err_radial = np.sqrt(err_ra**2 + err_dec**2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # --- (0,0) Predicted vs GT scatter ---
    ax = axes[0, 0]
    lim = max(30, np.percentile(np.abs(np.concatenate([pred.ravel(), gt.ravel()])), 98) * 1.3)
    ax.scatter(gt[:, 0], pred[:, 0], s=8, alpha=0.5, c='tab:blue', label='RA*')
    ax.scatter(gt[:, 1], pred[:, 1], s=8, alpha=0.5, c='tab:red', label='Dec')
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('GT offset (mas)')
    ax.set_ylabel('Predicted offset (mas)')
    ax.set_title(f'Pred vs GT  |  N={len(pred)}')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

    # --- (0,1) Error histogram + sigma ---
    ax = axes[0, 1]
    bins = np.linspace(0, min(100, np.percentile(err_radial, 99) * 1.5), 40)
    ax.hist(err_radial, bins=bins, alpha=0.7, color='steelblue', edgecolor='white',
            label=f'radial error\nMAE={np.mean(err_radial):.1f} mas\np68={np.percentile(err_radial, 68):.1f} mas')
    ax.axvline(np.median(sigma), color='darkorange', ls='--', lw=2,
               label=f'median σ = {np.median(sigma):.1f} mas')
    ax.set_xlabel('Radial error (mas)')
    ax.set_ylabel('Count')
    ax.set_title('Error distribution')
    ax.legend(fontsize=8)

    # --- (1,0) VIS image + quiver ---
    ax = axes[1, 0]
    p1, p99 = np.percentile(vis_img_np, [1, 99])
    ax.imshow(vis_img_np, origin='lower', cmap='gray', vmin=p1, vmax=p99)
    # Convert offsets to VIS pixel units for display, then magnify for visibility.
    # This avoids the "giant arrows" confusion from plotting mas directly.
    pix2sky = batch['pixel_to_sky'].cpu().numpy()            # [N, 2, 2] arcsec / px
    target_arcsec = batch['target_offset_arcsec'].cpu().numpy()
    inv = np.linalg.pinv(pix2sky)
    gt_px = np.einsum('nij,nj->ni', inv, target_arcsec)      # [N, 2] px
    pred_px = np.stack([
        out['dx_px'].detach().cpu().numpy(),
        out['dy_px'].detach().cpu().numpy(),
    ], axis=1)
    magnify = float(getattr(args, 'quiver_magnify', 10.0))
    ax.quiver(positions[:, 0], positions[:, 1],
              gt_px[:, 0] * magnify, gt_px[:, 1] * magnify,
              color='tab:blue', alpha=0.6, scale=1, scale_units='xy',
              width=0.002, headwidth=3, label='GT offset (px)')
    ax.quiver(positions[:, 0], positions[:, 1],
              pred_px[:, 0] * magnify, pred_px[:, 1] * magnify,
              color='tab:red', alpha=0.6, scale=1, scale_units='xy',
              width=0.002, headwidth=3, label='Predicted (px)')
    ax.set_title(f'Offset quiver on VIS  |  {tile_id}  (arrows {magnify:.0f}× magnified)')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0, vis_img_np.shape[1])
    ax.set_ylim(0, vis_img_np.shape[0])

    # --- (1,1) Error vs confidence ---
    ax = axes[1, 1]
    confidence = 1.0 / np.clip(sigma, 1, None)
    ax.scatter(confidence, err_radial, s=10, alpha=0.4, c='tab:green')
    ax.set_xlabel('Confidence (1/σ, mas⁻¹)')
    ax.set_ylabel('Radial error (mas)')
    ax.set_title('Error vs Confidence  (lower-right = well-calibrated)')
    # Add calibration line: error should roughly equal sigma
    conf_range = np.linspace(confidence.min(), confidence.max(), 50)
    ax.plot(conf_range, 1.0 / conf_range, 'k--', lw=0.8, alpha=0.5, label='error = σ')
    ax.legend(fontsize=8)

    fig.suptitle(f'Epoch {epoch}  |  Latent Position Head  |  {tile_id}', fontsize=12, y=1.01)
    fig.tight_layout()

    img = wandb.Image(fig)
    plt.close(fig)
    return img


# ============================================================
# Training loop
# ============================================================

def _prefetch_tile(
    tile_id, rubin_path, euclid_path,
    frozen_encoder, enc_device, train_device, args, vis_hw_out,
):
    """Encode one tile on `enc_device`, transfer features to `train_device`.

    Returns None on failure, or a dict with ready-to-train tensors.
    Used by the dual-GPU prefetch thread (encoder on GPU 0, head on GPU 1).
    """
    try:
        img_t, rms_t, vis_hw, vis_wcs = load_tile_data(
            rubin_path, euclid_path, enc_device)
    except Exception:
        return None
    vis_hw_out.append(vis_hw)

    enc_out = encode_tile_features(
        frozen_encoder,
        tile_id,
        img_t,
        rms_t,
        enc_device,
        features_cache_dir=getattr(args, '_features_cache_dir', None),
    )
    del img_t, rms_t

    # Transfer features to the training device
    bottleneck = enc_out['bottleneck'].to(train_device)
    vis_stem = enc_out['vis_stem'].to(train_device)
    fused_hw = enc_out['fused_hw']

    # Detection + centroiding (CPU-bound, runs on any thread)
    try:
        edata = np.load(euclid_path, allow_pickle=True)
        vis_img_np_arr = np.nan_to_num(_to_float32(edata['img_VIS']), nan=0.0)
        vis_var_np = _to_float32(edata['var_VIS']) if 'var_VIS' in edata.files else None
        vis_rms_np_arr = (
            np.maximum(np.nan_to_num(np.sqrt(np.clip(vis_var_np, 0, None)), nan=1.0), 1e-10)
            if vis_var_np is not None else None
        )
    except Exception:
        return None

    det_px = None
    _cn = getattr(args, '_centernet_labels_dict', None)
    if _cn is not None and tile_id in _cn:
        entry = _cn[tile_id]
        xy_norm = entry[0] if isinstance(entry, tuple) else entry
        H_vis, W_vis = vis_hw
        det_px = np.stack([
            xy_norm[:, 0] * max(W_vis - 1, 1),
            xy_norm[:, 1] * max(H_vis - 1, 1),
        ], axis=1).astype(np.float32)

    vis_xy, vis_snr = detect_and_refine_vis(
        vis_img_np_arr, vis_wcs,
        nsig=args.vis_nsig, smooth=args.vis_smooth,
        min_dist=args.vis_min_dist, max_sources=args.max_sources_vis,
        refine_radius=args.refine_radius,
        flux_floor_sigma=args.refine_flux_floor_sigma,
        psf_field=getattr(args, '_psf_field', None),
        epsf_head=getattr(args, '_epsf_head', None),
        vis_rms=vis_rms_np_arr,
        epsf_features={
            'bottleneck': bottleneck,
            'vis_stem': vis_stem,
            'fused_hw': fused_hw,
            'vis_hw': vis_hw,
        },
        epsf_stamp_size=args.epsf_centroid_stamp_size,
        epsf_search_radius_px=args.epsf_centroid_search_px,
        epsf_search_steps=args.epsf_centroid_search_steps,
        psf_device=enc_device,
        detections_vis_px=det_px,
    )
    if vis_xy.shape[0] < 5:
        return None

    return {
        'bottleneck': bottleneck,
        'vis_stem': vis_stem,
        'fused_hw': fused_hw,
        'vis_hw': vis_hw,
        'vis_xy': vis_xy,
        'vis_wcs': vis_wcs,
        'tile_id': tile_id,
        'euclid_path': euclid_path,
    }


def run_epoch(
    split_name: str,
    pairs: Sequence[Tuple[str, str, str]],
    frozen_encoder: FrozenEncoder,
    head: LatentPositionHead,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    args,
    rng: np.random.RandomState,
) -> Dict[str, float]:
    """One epoch: iterate over tiles, encode, predict, update."""
    is_train = optimizer is not None
    head.train(mode=is_train)

    agg = defaultdict(float)
    n_tiles = 0
    n_sources = 0

    order = rng.permutation(len(pairs)) if is_train else np.arange(len(pairs))

    # --- Dual-GPU prefetch mode ---
    # GPU 0 encodes + detects tiles in a background thread;
    # GPU 1 trains the head on the pre-encoded features.
    dual_gpu = getattr(args, 'dual_gpu', False)
    if dual_gpu:
        from concurrent.futures import ThreadPoolExecutor
        enc_device = torch.device('cuda:0')
        frozen_encoder.to(enc_device)
        if getattr(args, '_psf_field', None) is not None:
            args._psf_field.to(enc_device)

    def _get_tile_data(idx):
        """Produce encoded tile data (dual or single GPU)."""
        tile_id, rubin_path, euclid_path = pairs[idx]
        if dual_gpu:
            vis_hw_box = []
            result = _prefetch_tile(
                tile_id, rubin_path, euclid_path,
                frozen_encoder, enc_device, device, args, vis_hw_box,
            )
            return result
        else:
            # --- Original single-GPU path ---
            try:
                img_t, rms_t, vis_hw, vis_wcs = load_tile_data(
                    rubin_path, euclid_path, device)
            except Exception:
                return None
            enc_out = encode_tile_features(
                frozen_encoder,
                tile_id,
                img_t,
                rms_t,
                device,
                features_cache_dir=getattr(args, '_features_cache_dir', None),
            )
            bottleneck = enc_out['bottleneck']
            vis_stem = enc_out['vis_stem']
            fused_hw = enc_out['fused_hw']
            del img_t, rms_t
            try:
                edata = np.load(euclid_path, allow_pickle=True)
                vis_img_np_arr = np.nan_to_num(
                    _to_float32(edata['img_VIS']), nan=0.0)
                vis_var_np = _to_float32(edata['var_VIS']) if 'var_VIS' in edata.files else None
                vis_rms_np_arr = (
                    np.maximum(np.nan_to_num(np.sqrt(np.clip(vis_var_np, 0, None)), nan=1.0), 1e-10)
                    if vis_var_np is not None else None
                )
            except Exception:
                return None
            det_px = None
            _cn = getattr(args, '_centernet_labels_dict', None)
            if _cn is not None and tile_id in _cn:
                entry = _cn[tile_id]
                xy_norm = entry[0] if isinstance(entry, tuple) else entry
                H_vis, W_vis = vis_hw
                det_px = np.stack([
                    xy_norm[:, 0] * max(W_vis - 1, 1),
                    xy_norm[:, 1] * max(H_vis - 1, 1),
                ], axis=1).astype(np.float32)
            vis_xy, vis_snr = detect_and_refine_vis(
                vis_img_np_arr, vis_wcs,
                nsig=args.vis_nsig, smooth=args.vis_smooth,
                min_dist=args.vis_min_dist,
                max_sources=args.max_sources_vis,
                refine_radius=args.refine_radius,
                flux_floor_sigma=args.refine_flux_floor_sigma,
                psf_field=getattr(args, '_psf_field', None),
                epsf_head=getattr(args, '_epsf_head', None),
                vis_rms=vis_rms_np_arr,
                epsf_features={
                    'bottleneck': bottleneck,
                    'vis_stem': vis_stem,
                    'fused_hw': fused_hw,
                    'vis_hw': vis_hw,
                },
                epsf_stamp_size=args.epsf_centroid_stamp_size,
                epsf_search_radius_px=args.epsf_centroid_search_px,
                epsf_search_steps=args.epsf_centroid_search_steps,
                psf_device=device,
                detections_vis_px=det_px,
            )
            if vis_xy.shape[0] < 5:
                return None
            return {
                'bottleneck': bottleneck, 'vis_stem': vis_stem,
                'fused_hw': fused_hw, 'vis_hw': vis_hw,
                'vis_xy': vis_xy, 'vis_wcs': vis_wcs,
                'tile_id': tile_id, 'euclid_path': euclid_path,
            }

    # Submit first tile ahead of time in dual-GPU mode
    prefetch_pool = ThreadPoolExecutor(max_workers=1) if dual_gpu else None
    pending_future = None
    if prefetch_pool and len(order) > 0:
        pending_future = prefetch_pool.submit(_get_tile_data, order[0])

    for i, idx in enumerate(order):
        # Get current tile's data (from prefetch or inline)
        if pending_future is not None:
            tile_data = pending_future.result()
            # Submit NEXT tile for prefetch
            if i + 1 < len(order):
                pending_future = prefetch_pool.submit(
                    _get_tile_data, order[i + 1])
            else:
                pending_future = None
        else:
            tile_data = _get_tile_data(idx)

        if tile_data is None:
            continue

        bottleneck = tile_data['bottleneck']
        vis_stem   = tile_data['vis_stem']
        fused_hw   = tile_data['fused_hw']
        vis_hw     = tile_data['vis_hw']
        vis_xy     = tile_data['vis_xy']
        vis_wcs    = tile_data['vis_wcs']

        batch = build_tile_batch(
            vis_xy, vis_wcs,
            jitter_arcsec=args.jitter_arcsec,
            jitter_max_arcsec=args.jitter_max_arcsec,
            max_sources=args.max_sources_per_tile,
            rng=rng,
            device=device,
        )
        if batch is None:
            continue

        # --- Forward through head ---
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            out = head(
                bottleneck, vis_stem,
                batch['positions'],
                batch['pixel_to_sky'],
                fused_hw, vis_hw,
            )
            losses = compute_loss(
                out,
                batch['target_offset_arcsec'],
                batch['pixel_to_sky'],
                label_noise_floor=args.label_noise_floor,
            )
            if is_train:
                losses['loss_total'].backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
                optimizer.step()

        metrics = compute_metrics(out, batch['target_offset_arcsec'])
        for k, v in losses.items():
            agg[k] += float(v.detach())
        for k, v in metrics.items():
            agg[k] += float(v)
        n_tiles += 1
        n_sources += batch['positions'].shape[0]

        # Free large tensors between tiles.
        del bottleneck, vis_stem, batch, out

    if prefetch_pool is not None:
        prefetch_pool.shutdown(wait=False)

    denom = max(1, n_tiles)
    result = {k: v / denom for k, v in agg.items()}
    result['tiles'] = n_tiles
    result['sources'] = n_sources
    return result


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Train the latent-space canonical position head.',
    )
    p.add_argument('--rubin-dir', type=str, required=True)
    p.add_argument('--euclid-dir', type=str, required=True)
    p.add_argument('--foundation-checkpoint', type=str, required=True,
                   help='Path to the frozen foundation model (v7/v8/v9/v10, auto-detected).')
    p.add_argument('--psf-checkpoint', type=str, default=None,
                   help='Path to a trained PSF checkpoint. Supports the current '
                        'FoundationEPSFHead checkpoints and archived PSFField '
                        'checkpoints. If provided, VIS centroid labels use PSF '
                        'template fitting instead of classical 2D Gaussian.')
    p.add_argument('--epsf-centroid-stamp-size', type=int, default=21,
                   help='Native-pixel stamp size for FoundationEPSF centroiding.')
    p.add_argument('--epsf-centroid-search-px', type=float, default=0.5,
                   help='Half-width of FoundationEPSF centroid search grid in native pixels.')
    p.add_argument('--epsf-centroid-search-steps', type=int, default=7,
                   help='Grid samples per axis for FoundationEPSF centroid search.')
    p.add_argument('--centernet-labels', type=str, default=None,
                   help='Path to CenterNet pseudo_labels .pt (e.g. '
                        'data/detection_labels/centernet_v10_790_thresh03.pt). '
                        'If provided, VIS-normalised detections replace classical '
                        'peak-finding for source candidates.')
    p.add_argument('--features-cache-dir', type=str, default=None,
                   help='Optional precomputed foundation bottleneck cache '
                        '(e.g. data/cached_features_v10_warmstart). The VIS '
                        'stem is still computed live.')
    p.add_argument('--output-dir', type=str,
                   default='models/checkpoints/latent_position_head')

    # Source detection.
    p.add_argument('--vis-nsig', type=float, default=4.0)
    p.add_argument('--vis-smooth', type=float, default=1.2)
    p.add_argument('--vis-min-dist', type=int, default=9)
    p.add_argument('--max-sources-vis', type=int, default=800)
    p.add_argument('--max-sources-per-tile', type=int, default=200)
    p.add_argument('--refine-radius', type=int, default=3)
    p.add_argument('--refine-flux-floor-sigma', type=float, default=1.5)

    # Jitter augmentation.
    p.add_argument('--jitter-arcsec', type=float, default=0.03,
                   help='Gaussian jitter σ in arcsec applied to source positions '
                        'during training. 30 mas ≈ 0.3 VIS pixels (default: 0.03).')
    p.add_argument('--jitter-max-arcsec', type=float, default=0.1,
                   help='Clip jitter to ±this arcsec (default: 0.1).')

    # Head architecture.
    p.add_argument('--bottleneck-out', type=int, default=128)
    p.add_argument('--stem-out', type=int, default=64)
    p.add_argument('--mlp-hidden', type=int, default=128)
    p.add_argument('--bottleneck-window', type=int, default=5)
    p.add_argument('--stem-window', type=int, default=17)

    # Training.
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--label-noise-floor', type=float, default=0.005,
                   help='Label noise floor in arcsec (default: 5 mas).')
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--val-patches', type=str, default=None,
                   help='Comma-separated patch ids (e.g. "25") to hold out as a '
                        'spatially disjoint val set. Overrides the --val-frac '
                        'random tile split, which leaks through the 50%% tile '
                        'overlap (overlapping tiles are crops of the same '
                        'mosaic, so duplicated sources carry identical labels).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='')
    p.add_argument('--dual-gpu', action='store_true',
                   help='Use GPU 0 for frozen encoder + detection, '
                        'GPU 1 for head training. Pre-encodes tiles in a '
                        'background thread to overlap encoder with head.')
    p.add_argument('--vis-every', type=int, default=3,
                   help='Log diagnostic preview figure every N epochs (default: 3).')
    p.add_argument('--quiver-magnify', type=float, default=10.0,
                   help='Visual magnification for quiver arrows in the preview plot '
                        '(in pixel units; default: 10×).')

    # Logging.
    p.add_argument('--wandb-project', type=str, default='JAISP-LatentPosition')
    p.add_argument('--wandb-run-name', type=str, default='')
    p.add_argument('--wandb-mode', type=str, default='online',
                   choices=['online', 'offline', 'disabled'])
    return p


def serializable_args(args) -> Dict[str, object]:
    """Return CLI/config args without runtime model/cache objects."""
    return {
        k: v for k, v in vars(args).items()
        if not k.startswith('_')
    }


def train(args):
    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    pairs = discover_tile_pairs(args.rubin_dir, args.euclid_dir)
    if args.val_patches:
        val_patch_ids = {s.strip() for s in args.val_patches.split(',') if s.strip()}
        def _patch_id(pair):
            return pair[0].rsplit('_patch_', 1)[-1]
        train_pairs = [p for p in pairs if _patch_id(p) not in val_patch_ids]
        val_pairs = [p for p in pairs if _patch_id(p) in val_patch_ids]
        if not val_pairs:
            raise SystemExit(f'--val-patches {sorted(val_patch_ids)} matched no tiles')
        print(f'Patch-disjoint split: val patches {sorted(val_patch_ids)}')
    else:
        train_pairs, val_pairs = split_tile_pairs(pairs, args.val_frac, args.seed)
    print(f'Tiles: {len(train_pairs)} train, {len(val_pairs)} val')
    args._features_cache_dir = Path(args.features_cache_dir) if args.features_cache_dir else None
    if args._features_cache_dir is not None:
        print(f'Using cached foundation bottlenecks from {args._features_cache_dir}')

    # --- Model ---
    frozen_encoder, head = load_latent_position_head(
        args.foundation_checkpoint,
        device=device,
        bottleneck_out=args.bottleneck_out,
        stem_out=args.stem_out,
        mlp_hidden=args.mlp_hidden,
        bottleneck_window=args.bottleneck_window,
        stem_window=args.stem_window,
    )

    # --- PSF-template centroid labels (optional) ---
    args._epsf_head = None
    if args.psf_checkpoint:
        import torch as _t
        _ckpt = _t.load(args.psf_checkpoint, map_location='cpu', weights_only=False)
        if 'psf_field_state' in _ckpt:
            from psf.psf_field import PSFField as _PSFField
            _cfg = _ckpt['config']
            _psf = _PSFField(
                sed_embed_dim=_cfg['sed_embed_dim'],
                band_embed_dim=_cfg['band_embed_dim'],
                tile_freqs=_cfg['tile_freqs'],
                siren_hidden=_cfg['siren_hidden'],
                siren_depth=_cfg['siren_depth'],
                w0_first=_cfg['w0_first'],
                envelope_r_rubin=_cfg.get('envelope_r_rubin', 0.0),
                envelope_r_euclid=_cfg.get('envelope_r_euclid', 0.0),
                envelope_power=_cfg.get('envelope_power', 4.0),
            ).to(device).eval()
            _psf.load_state_dict(_ckpt['psf_field_state'])
            args._psf_field = _psf
            print(f'Loaded archived PSFField from {args.psf_checkpoint}')
        else:
            from psf.foundation_epsf_head import load_foundation_epsf_head
            args._epsf_head = load_foundation_epsf_head(args.psf_checkpoint, device=device)
            args._psf_field = None
            print(f'Loaded FoundationEPSFHead from {args.psf_checkpoint}')
    else:
        args._psf_field = None

    # --- CenterNet labels (optional) ---
    if args.centernet_labels:
        import torch as _t
        _pl = _t.load(args.centernet_labels, map_location='cpu', weights_only=False)
        args._centernet_labels_dict = _pl['labels'] if 'labels' in _pl else _pl
        print(f'Loaded CenterNet labels for '
              f'{len(args._centernet_labels_dict)} tiles')
    else:
        args._centernet_labels_dict = None

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # --- W&B ---
    wandb_run = None
    if args.wandb_mode != 'disabled' and wandb is not None:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or None,
                config=serializable_args(args),
                mode=args.wandb_mode,
                dir=str(out_dir),
            )
        except Exception as exc:
            print(f'W&B init failed: {exc}')

    # --- Train ---
    rng = np.random.RandomState(args.seed)
    best_mae = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(
            'train', train_pairs, frozen_encoder, head,
            optimizer, device, args, rng,
        )
        scheduler.step()

        val_metrics = {}
        if val_pairs:
            val_metrics = run_epoch(
                'val', val_pairs, frozen_encoder, head,
                None, device, args, rng,
            )

        dt = time.time() - t0

        # Log.
        train_mae_mas = train_metrics.get('mae_total', 0) * 1000
        val_mae_mas = val_metrics.get('mae_total', 0) * 1000
        train_p68_mas = train_metrics.get('p68_total', 0) * 1000
        val_p68_mas = val_metrics.get('p68_total', 0) * 1000
        lr = optimizer.param_groups[0]['lr']

        line = (
            f'E{epoch:3d} | '
            f'train MAE={train_mae_mas:.1f} p68={train_p68_mas:.1f} σ={train_metrics.get("sigma_median", 0)*1000:.1f} | '
        )
        if val_metrics:
            line += (
                f'val MAE={val_mae_mas:.1f} p68={val_p68_mas:.1f} σ={val_metrics.get("sigma_median", 0)*1000:.1f} | '
            )
        line += f'lr={lr:.1e} | {dt:.0f}s | {train_metrics.get("tiles", 0)}+{val_metrics.get("tiles", 0)} tiles'
        print(line)

        if wandb_run:
            log_d = {f'train/{k}': v for k, v in train_metrics.items()}
            log_d.update({f'val/{k}': v for k, v in val_metrics.items()})
            log_d['lr'] = lr
            log_d['epoch'] = epoch

            # Diagnostic preview figure.
            preview_pairs = val_pairs if val_pairs else train_pairs
            if epoch % args.vis_every == 0 and preview_pairs:
                preview_rng = np.random.RandomState(args.seed + epoch)
                preview_img = make_preview_figure(
                    head, frozen_encoder, preview_pairs[0],
                    device, args, epoch, preview_rng,
                )
                if preview_img is not None:
                    log_d['preview/diagnostic'] = preview_img

            wandb_run.log(log_d, step=epoch)

        # Checkpoint.
        score = val_metrics.get('mae_total', train_metrics.get('mae_total', float('inf')))
        ckpt = {
            'epoch': epoch,
            'head_state_dict': head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': serializable_args(args),
        }
        torch.save(ckpt, out_dir / 'latest.pt')
        if score < best_mae:
            best_mae = score
            torch.save(ckpt, out_dir / 'best.pt')
            print(f'  → new best: MAE={score*1000:.1f} mas')

    print(f'\nDone. Best val MAE: {best_mae*1000:.1f} mas')
    print(f'Checkpoints: {out_dir}')
    if wandb_run:
        wandb_run.finish()


def main():
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
