"""Self-training loop for CenterNet source detection.

Round 1: Train on VIS pseudo-labels (classical peak-finding at 0.1"/px).
         The model learns what sources look like in 10-band feature space.

Round 2: Run the trained detector on all tiles. High-confidence predictions
         that don't match any VIS pseudo-label are promoted as new labels.
         Retrain on VIS labels + promoted labels.

The promoted detections are sources the 10-band encoder can see but VIS
alone cannot (e.g. very red high-z galaxies visible in NISP Y/J/H,
or UV sources in Rubin u/g).

Usage
-----
    # Step 1: Precompute features (run once)
    python models/precompute_features.py \
        --rubin_dir ../data/rubin_tiles_all \
        --euclid_dir ../data/euclid_tiles_all \
        --encoder_ckpt ../checkpoints/jaisp_v7_tiles_all_ddp_online/checkpoint_best.pt \
        --out_dir ../data/cached_features_v7_tiles_all

    # Step 2: Start with round 1 only for the first comparison
    python detection/self_train.py \
        --feature_dir  ../data/cached_features_v7_tiles_all \
        --rubin_dir    ../data/rubin_tiles_all \
        --euclid_dir   ../data/euclid_tiles_all \
        --out_dir      ../checkpoints/centernet_v7_tiles_all_round1 \
        --rounds 1 \
        --batch_size 4 \
        --wandb_project jaisp-detection
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from detection.centernet_detector import CenterNetDetector
from detection.dataset import (
    PSEUDO_LABEL_CACHE_VERSION,
    _vis_bright_core_and_spike_mask,
    _vis_bright_extended_rescue_labels,
)


def _dedupe_points(points: np.ndarray, radius: float) -> np.ndarray:
    """Keep the first point in each small normalized-radius cluster."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] <= 1:
        return pts
    keep = []
    r = float(max(radius, 0.0))
    for pt in pts:
        if not np.isfinite(pt).all():
            continue
        if not keep:
            keep.append(pt)
            continue
        old = np.stack(keep, axis=0)
        if np.hypot(old[:, 0] - pt[0], old[:, 1] - pt[1]).min() > r:
            keep.append(pt)
    if not keep:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(keep, axis=0).astype(np.float32)


def _run_training_round(
    feature_dir: str,
    rubin_dir: str,
    euclid_dir: str,
    out_path: str,
    extra_labels: str = None,
    init_checkpoint: str = None,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    sigma: float = 2.0,
    nsig: float = 3.0,
    head_ch: int = 256,
    labels_mode: str = 'vis_peak',
    val_patches: str = None,
    mer_fits: str = None,
    uncertain_ignore: bool = False,
    uncertain_nsig: float = 1.8,
    uncertain_radius_px: float = 5.0,
    synthetic_sources_per_tile: int = 0,
    synthetic_prob: float = 1.0,
    synthetic_min_snr: float = 5.0,
    synthetic_max_snr: float = 20.0,
    synthetic_min_sigma_px: float = 1.1,
    synthetic_max_sigma_px: float = 3.5,
    synthetic_weight: float = 1.5,
    viz_spike_veto_radius: int = 0,
    viz_spike_veto_width: float = 0.0,
    device: torch.device = None,
    wandb_project: str = None,
    wandb_name: str = None,
) -> None:
    """Run one round of CenterNet training on cached features."""
    import subprocess
    cmd = [
        sys.executable, str(_HERE / 'train_centernet.py'),
        '--feature_dir', feature_dir,
        '--rubin_dir', rubin_dir,
        '--out', out_path,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
        '--sigma', str(sigma),
        '--nsig', str(nsig),
        '--head_ch', str(head_ch),
        '--labels_mode', labels_mode,
        '--uncertain_nsig', str(uncertain_nsig),
        '--uncertain_radius_px', str(uncertain_radius_px),
        '--synthetic_sources_per_tile', str(synthetic_sources_per_tile),
        '--synthetic_prob', str(synthetic_prob),
        '--synthetic_min_snr', str(synthetic_min_snr),
        '--synthetic_max_snr', str(synthetic_max_snr),
        '--synthetic_min_sigma_px', str(synthetic_min_sigma_px),
        '--synthetic_max_sigma_px', str(synthetic_max_sigma_px),
        '--synthetic_weight', str(synthetic_weight),
        '--viz_spike_veto_radius', str(viz_spike_veto_radius),
        '--viz_spike_veto_width', str(viz_spike_veto_width),
    ]
    if euclid_dir:
        cmd += ['--euclid_dir', euclid_dir]
    if extra_labels:
        cmd += ['--extra_labels', extra_labels]
    if init_checkpoint:
        cmd += ['--init_checkpoint', init_checkpoint]
    if val_patches:
        cmd += ['--val_patches', val_patches]
    if mer_fits:
        cmd += ['--mer_fits', mer_fits]
    if uncertain_ignore:
        cmd += ['--uncertain_ignore']
    if wandb_project:
        cmd += ['--wandb_project', wandb_project]
    if wandb_name:
        cmd += ['--wandb_run', wandb_name]
    if device is not None:
        cmd += ['--device', str(device)]

    print(f'\n{"="*60}')
    print(f'Running: {" ".join(cmd)}')
    print(f'{"="*60}\n')
    subprocess.run(cmd, check=True)


def _refine_labels(
    feature_dir: str,
    checkpoint: str,
    rubin_dir: str,
    euclid_dir: str,
    nsig: float,
    promote_conf: float = 0.8,
    demote_conf: float = 0.3,
    match_radius: float = 0.01,
    promotion_spike_radius: int = 20,
    promotion_spike_width: float = 3.0,
    bright_rescue: bool = False,
    bright_rescue_nsig: float = 2.5,
    bright_rescue_min_area: int = 45,
    bright_rescue_min_radius: float = 5.0,
    bright_rescue_min_peak_snr: float = 8.0,
    bright_rescue_match_scale: float = 1.5,
    bright_rescue_match_min: float = 8.0,
    bright_rescue_match_max: float = 35.0,
    bright_rescue_max_per_tile: int = 32,
    labels_mode: str = 'vis_peak',
    device: torch.device = None,
) -> tuple:
    """Run trained detector on all tiles. Promote novel high-confidence
    detections and demote existing pseudo-labels the model rejects.

    Returns (promoted, demoted):
        promoted: dict[tile_id] -> np.ndarray[N, 2] new labels to ADD
        demoted:  dict[tile_id] -> np.ndarray[N, 2] old labels to REMOVE
    """
    import torch.nn.functional as F

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained model (head only). Checkpoints saved from the live-encoder
    # training path contain encoder.* weights; strip them — refinement runs on
    # cached features so the encoder is irrelevant here.
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = CenterNetDetector(
        encoder=None,
        encoder_dim=ckpt['encoder_dim'],
        head_ch=ckpt.get('head_ch', 256),
    )
    head_state = {k: v for k, v in ckpt['state_dict'].items()
                  if not k.startswith('encoder.')}
    missing, unexpected = model.load_state_dict(head_state, strict=False)
    missing_non_encoder = [k for k in missing if not k.startswith('encoder.')]
    if missing_non_encoder:
        raise RuntimeError(
            f'Missing detector keys in {checkpoint}: {missing_non_encoder}')
    if unexpected:
        print(f'  [warn] Unexpected keys in {checkpoint}: {unexpected}')
    model = model.to(device).eval()

    feature_dir = Path(feature_dir)
    feat_files = sorted(feature_dir.glob('tile_*_aug0.pt'))

    # Load cached pseudo-labels (written by CachedFeatureDataset._compute_labels)
    label_cache_path = feature_dir / ('pseudo_labels.pt' if labels_mode == 'vis_peak' else f'pseudo_labels_{labels_mode}.pt')
    if label_cache_path.exists():
        saved_labels = torch.load(label_cache_path, map_location='cpu',
                                  weights_only=False)
        if isinstance(saved_labels, dict) and 'labels' in saved_labels:
            cached_nsig = saved_labels.get('nsig')
            cached_version = saved_labels.get('label_version')
            if (
                cached_version == PSEUDO_LABEL_CACHE_VERSION
                and (cached_nsig is None or abs(cached_nsig - nsig) <= 1e-6)
            ):
                label_cache = saved_labels['labels']
                print(f'  Loaded cached pseudo-labels from {label_cache_path}')
            else:
                label_cache = None
                print(
                    f'  [warn] Ignoring stale pseudo-label cache '
                    f'(version={cached_version}, nsig={cached_nsig}); '
                    f'computing labels on the fly for refinement'
                )
        else:
            label_cache = None
            print(
                f'  [warn] Ignoring legacy pseudo-label cache without metadata; '
                f'computing labels on the fly for refinement'
            )
    else:
        # Fallback: compute on the fly (slow path, handled by _load_labels_fallback)
        label_cache = None
        print('  [warn] No cached pseudo-labels found, computing on the fly')

    promoted = {}
    demoted = {}
    total_promoted = 0
    total_demoted = 0
    total_tiles = 0
    total_artifact_vetoed = 0
    total_bright_rescue = 0

    for feat_path in feat_files:
        tile_id = feat_path.stem.rsplit('_aug', 1)[0]
        cached = torch.load(feat_path, map_location='cpu', weights_only=True)
        feats = cached['features'].unsqueeze(0).to(device)

        promotion_mask = None
        vis_img = None
        if euclid_dir and (promotion_spike_radius > 0 or bright_rescue):
            vis_path = Path(euclid_dir) / f'{tile_id}_euclid.npz'
            if vis_path.exists():
                try:
                    edata = np.load(str(vis_path), allow_pickle=True, mmap_mode='r')
                    vis_img = np.nan_to_num(
                        np.asarray(edata['img_VIS'], dtype=np.float32), nan=0.0)
                    if promotion_spike_radius > 0:
                        _, _, promotion_mask = _vis_bright_core_and_spike_mask(
                            vis_img,
                            spike_radius=promotion_spike_radius,
                            spike_width=promotion_spike_width,
                            include_core=False,
                        )
                except Exception:
                    vis_img = None
                    promotion_mask = None

        # Run detector
        with torch.no_grad():
            neck_out = model.neck(feats)
            hm = model.hm_head(neck_out).sigmoid()  # [1, 1, H, W]
            off = model.off_head(neck_out)

        _, _, fH, fW = hm.shape

        # Get original pseudo-labels for this tile
        if label_cache is not None and tile_id in label_cache:
            orig_labels = label_cache[tile_id][0]  # (centroids, classes) tuple
        else:
            # Slow fallback
            orig_labels = _load_labels_fallback(
                tile_id, euclid_dir, rubin_dir, nsig)

        rescue_xy = np.zeros((0, 2), dtype=np.float32)
        if bright_rescue and vis_img is not None:
            try:
                rescue_xy, _ = _vis_bright_extended_rescue_labels(
                    vis_img,
                    existing_norm=orig_labels,
                    thresh_nsig=bright_rescue_nsig,
                    min_area=bright_rescue_min_area,
                    min_radius=bright_rescue_min_radius,
                    min_peak_snr=bright_rescue_min_peak_snr,
                    match_radius_scale=bright_rescue_match_scale,
                    match_radius_min=bright_rescue_match_min,
                    match_radius_max=bright_rescue_match_max,
                    spike_radius=promotion_spike_radius,
                    spike_width=promotion_spike_width,
                    max_rescue_per_tile=bright_rescue_max_per_tile,
                )
            except Exception as exc:
                print(f'  [warn] bright rescue failed for {tile_id}: {exc}')
                rescue_xy = np.zeros((0, 2), dtype=np.float32)
        total_bright_rescue += int(rescue_xy.shape[0])

        hm_np = hm[0, 0].cpu().numpy()

        # --- DEMOTE: vectorized confidence check at label positions ---
        if orig_labels.shape[0] > 0:
            fx = np.clip(np.round(orig_labels[:, 0] * (fW - 1)).astype(int),
                         0, fW - 1)
            fy = np.clip(np.round(orig_labels[:, 1] * (fH - 1)).astype(int),
                         0, fH - 1)
            confs = hm_np[fy, fx]
            demote_mask = confs < demote_conf
            if demote_mask.any():
                demoted[tile_id] = orig_labels[demote_mask]
                total_demoted += int(demote_mask.sum())

        # --- PROMOTE: find high-confidence predictions with no matching label ---
        hm_max = F.max_pool2d(hm, 3, stride=1, padding=1)
        peaks = (hm == hm_max) & (hm > promote_conf)
        novel = np.zeros((0, 2), dtype=np.float32)

        if peaks.any():
            yi, xi = torch.where(peaks[0, 0])
            dx = off[0, 0, yi, xi]
            dy = off[0, 1, yi, xi]
            pred_x = (xi.float() + dx) / max(fW - 1, 1)
            pred_y = (yi.float() + dy) / max(fH - 1, 1)
            pred_xy = torch.stack([pred_x, pred_y], dim=1).cpu().numpy()

            if promotion_mask is not None and len(pred_xy) > 0:
                if promotion_mask.shape != (fH, fW):
                    pm = torch.from_numpy(promotion_mask.astype(np.float32))
                    pm = pm.unsqueeze(0).unsqueeze(0)
                    pm = F.interpolate(pm, size=(fH, fW), mode='nearest')
                    promotion_mask = pm[0, 0].cpu().numpy() > 0
                px = np.clip(np.round(pred_xy[:, 0] * (fW - 1)).astype(int), 0, fW - 1)
                py = np.clip(np.round(pred_xy[:, 1] * (fH - 1)).astype(int), 0, fH - 1)
                keep = ~promotion_mask[py, px]
                total_artifact_vetoed += int((~keep).sum())
                pred_xy = pred_xy[keep]

            if orig_labels.shape[0] == 0:
                novel = pred_xy
            else:
                # Vectorized: cdist between predictions and existing labels
                from scipy.spatial.distance import cdist
                dists = cdist(pred_xy, orig_labels)  # [n_pred, n_labels]
                novel = pred_xy[dists.min(axis=1) > match_radius]

        if rescue_xy.shape[0] > 0:
            novel = np.concatenate([novel, rescue_xy], axis=0)
            novel = _dedupe_points(novel, radius=min(match_radius, 0.005))

        if len(novel) > 0:
            promoted[tile_id] = novel.astype(np.float32)
            total_promoted += len(novel)

        total_tiles += 1

    print(f'\nLabel refinement across {total_tiles} tiles:')
    print(f'  Promoted: {total_promoted} novel high-confidence detections')
    print(f'  Demoted:  {total_demoted} low-confidence pseudo-labels (likely artifacts)')
    if promotion_spike_radius > 0:
        print(
            f'  Artifact-vetoed: {total_artifact_vetoed} high-confidence peaks '
            f'on thin bright-star spike masks '
            f'(radius={promotion_spike_radius}px, width={promotion_spike_width:.1f}px)'
        )
    if bright_rescue:
        print(
            f'  Bright/extended rescue labels: {total_bright_rescue} '
            f'(not already covered by VIS pseudo-labels)'
        )
    return promoted, demoted


def _load_labels_fallback(tile_id, euclid_dir, rubin_dir, nsig):
    """Slow path: recompute pseudo-labels from raw images."""
    from detection.dataset import _pseudo_labels_vis, _pseudo_labels
    euclid_dir_p = Path(euclid_dir) if euclid_dir else None
    rubin_dir_p = Path(rubin_dir)

    if euclid_dir_p:
        ep = euclid_dir_p / f'{tile_id}_euclid.npz'
        if ep.exists():
            try:
                edata = np.load(str(ep), allow_pickle=True, mmap_mode='r')
                vis_img = np.nan_to_num(
                    np.asarray(edata['img_VIS'], dtype=np.float32), nan=0.0)
                labels, _, _, _ = _pseudo_labels_vis(vis_img, nsig, 1000)
                return labels
            except Exception:
                pass

    rp = rubin_dir_p / f'{tile_id}.npz'
    candidates = list(rubin_dir_p.glob(f'{tile_id}*.npz'))
    rp = candidates[0] if candidates else rp
    try:
        rdata = np.load(str(rp), allow_pickle=True, mmap_mode='r')
        raw_img = np.nan_to_num(
            np.asarray(rdata['img'], dtype=np.float32), nan=0.0)
        labels, _, _, _ = _pseudo_labels(raw_img, nsig, 1000)
        return labels
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description='Self-training loop for CenterNet detection.')
    p.add_argument('--feature_dir', required=True, help='Precomputed features directory')
    p.add_argument('--rubin_dir',   required=True)
    p.add_argument('--euclid_dir',  default=None)
    p.add_argument('--out_dir',     required=True, help='Output directory for checkpoints + labels')
    p.add_argument('--rounds',      type=int, default=2, help='Number of self-training rounds')
    p.add_argument('--start_round', type=int, default=1,
                   help='Round number to start from (default 1; use 2 to continue from an existing round-1 checkpoint)')
    p.add_argument('--epochs',      type=int, default=100, help='Epochs per round')
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--nsig',        type=float, default=3.0)
    p.add_argument('--sigma',       type=float, default=2.0)
    p.add_argument('--head_ch',     type=int, default=256,
                   help='CenterNet decoder width. Lower is faster/lighter; default 256.')
    p.add_argument('--labels_mode', default='vis_peak', choices=['vis_peak', 'multiband', 'vis_sep', 'mer'],
                   help='Pseudo-label source: improved VIS classical labels or multi-band SEP labels')
    p.add_argument('--val_patches', default=None,
                   help='Comma-separated patch ids held out for a PATCH-DISJOINT split (e.g. 25)')
    p.add_argument('--mer_fits', default=None, help='MER Q1 catalogue FITS (for --labels_mode mer)')
    p.add_argument('--uncertain_ignore', action='store_true',
                   help='Ignore negative heatmap loss around low-threshold uncertain source proposals')
    p.add_argument('--uncertain_nsig', type=float, default=1.8,
                   help='Low-threshold proposal significance for uncertainty ignore masks')
    p.add_argument('--uncertain_radius_px', type=float, default=5.0,
                   help='VIS-pixel radius ignored around uncertain proposals')
    p.add_argument('--synthetic_sources_per_tile', type=int, default=0,
                   help='Number of synthetic perfect-label sources injected per augmented training tile')
    p.add_argument('--synthetic_prob', type=float, default=1.0,
                   help='Probability of injecting synthetic sources into a training tile')
    p.add_argument('--synthetic_min_snr', type=float, default=5.0)
    p.add_argument('--synthetic_max_snr', type=float, default=20.0)
    p.add_argument('--synthetic_min_sigma_px', type=float, default=1.1)
    p.add_argument('--synthetic_max_sigma_px', type=float, default=3.5)
    p.add_argument('--synthetic_weight', type=float, default=1.5,
                   help='Positive loss weight for synthetic perfect-label sources')
    p.add_argument('--promote_conf', type=float, default=0.8,
                   help='Confidence threshold for promoting novel detections')
    p.add_argument('--demote_conf', type=float, default=0.3,
                   help='Confidence threshold below which existing labels are demoted '
                        '(model says "this is not a real source in 10-band space")')
    p.add_argument('--match_radius', type=float, default=0.01,
                   help='Normalized distance below which a prediction matches an existing label')
    p.add_argument('--promotion_spike_radius', type=int, default=20,
                   help='Bright-star spike search radius for promoting novel detections; set 0 to disable')
    p.add_argument('--promotion_spike_width', type=float, default=3.0,
                   help='Thin spike veto half-width in VIS pixels for self-training promotion')
    p.add_argument('--bright_rescue', action='store_true',
                   help='Append conservative bright/extended VIS rescue labels to round refinement')
    p.add_argument('--bright_rescue_nsig', type=float, default=2.5,
                   help='Robust S/N threshold for bright/extended rescue components')
    p.add_argument('--bright_rescue_min_area', type=int, default=45,
                   help='Minimum connected VIS area in pixels for bright/extended rescue')
    p.add_argument('--bright_rescue_min_radius', type=float, default=5.0,
                   help='Minimum footprint/moment radius in VIS pixels for bright/extended rescue')
    p.add_argument('--bright_rescue_min_peak_snr', type=float, default=8.0,
                   help='Minimum peak robust S/N for bright/extended rescue')
    p.add_argument('--bright_rescue_match_scale', type=float, default=1.5,
                   help='Adaptive existing-label match radius = scale * footprint radius')
    p.add_argument('--bright_rescue_match_min', type=float, default=8.0,
                   help='Minimum existing-label match radius in VIS pixels for rescue coverage')
    p.add_argument('--bright_rescue_match_max', type=float, default=35.0,
                   help='Maximum existing-label match radius in VIS pixels for rescue coverage')
    p.add_argument('--bright_rescue_max_per_tile', type=int, default=32,
                   help='Maximum bright/extended rescue labels to add per tile')
    p.add_argument('--init_checkpoint', default=None,
                   help='Existing detector checkpoint to bootstrap from when start_round > 1')
    p.add_argument('--refine_only', action='store_true',
                   help='Run label refinement against --init_checkpoint and exit. '
                        'Writes refined_labels_round1.pt to --out_dir. Use when you '
                        'want to drive round-2 training manually via train_centernet.py '
                        '(e.g. to preserve uncertain/synthetic flags that the cached '
                        'training path does not support).')
    p.add_argument('--wandb_project', default=None)
    p.add_argument('--wandb_run', default=None,
                   help='W&B run name. If set, used verbatim (collides across rounds); '
                        'if unset, defaults to "round{N}" per round.')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_labels_path = None
    init_checkpoint = args.init_checkpoint

    if args.refine_only:
        if not init_checkpoint:
            raise ValueError('--refine_only requires --init_checkpoint')
        print(f'\n--- Refine-only mode: refining labels against {init_checkpoint} ---')
        promoted, demoted_labels = _refine_labels(
            feature_dir=args.feature_dir,
            checkpoint=init_checkpoint,
            labels_mode=args.labels_mode,
            rubin_dir=args.rubin_dir,
            euclid_dir=args.euclid_dir,
            nsig=args.nsig,
            promote_conf=args.promote_conf,
            demote_conf=args.demote_conf,
            match_radius=args.match_radius,
            promotion_spike_radius=args.promotion_spike_radius,
            promotion_spike_width=args.promotion_spike_width,
            bright_rescue=args.bright_rescue,
            bright_rescue_nsig=args.bright_rescue_nsig,
            bright_rescue_min_area=args.bright_rescue_min_area,
            bright_rescue_min_radius=args.bright_rescue_min_radius,
            bright_rescue_min_peak_snr=args.bright_rescue_min_peak_snr,
            bright_rescue_match_scale=args.bright_rescue_match_scale,
            bright_rescue_match_min=args.bright_rescue_match_min,
            bright_rescue_match_max=args.bright_rescue_match_max,
            bright_rescue_max_per_tile=args.bright_rescue_max_per_tile,
            device=device,
        )
        out_path = str(out_dir / 'refined_labels_round1.pt')
        torch.save({'promoted': promoted, 'demoted': demoted_labels}, out_path)
        print(f'Saved refined labels to {out_path}')
        return

    if args.start_round < 1 or args.start_round > args.rounds:
        raise ValueError(f'--start_round must be between 1 and --rounds ({args.rounds})')

    if args.start_round > 1 and not init_checkpoint:
        raise ValueError('--init_checkpoint is required when --start_round > 1')

    if args.start_round > 1:
        prior_round = args.start_round - 1
        print(f'\n--- Bootstrapping round {args.start_round} from {init_checkpoint} ---')
        promoted, demoted_labels = _refine_labels(
            feature_dir=args.feature_dir,
            checkpoint=init_checkpoint,
            labels_mode=args.labels_mode,
            rubin_dir=args.rubin_dir,
            euclid_dir=args.euclid_dir,
            nsig=args.nsig,
            promote_conf=args.promote_conf,
            demote_conf=args.demote_conf,
            match_radius=args.match_radius,
            promotion_spike_radius=args.promotion_spike_radius,
            promotion_spike_width=args.promotion_spike_width,
            bright_rescue=args.bright_rescue,
            bright_rescue_nsig=args.bright_rescue_nsig,
            bright_rescue_min_area=args.bright_rescue_min_area,
            bright_rescue_min_radius=args.bright_rescue_min_radius,
            bright_rescue_min_peak_snr=args.bright_rescue_min_peak_snr,
            bright_rescue_match_scale=args.bright_rescue_match_scale,
            bright_rescue_match_min=args.bright_rescue_match_min,
            bright_rescue_match_max=args.bright_rescue_match_max,
            bright_rescue_max_per_tile=args.bright_rescue_max_per_tile,
            device=device,
        )
        extra_labels_path = str(out_dir / f'refined_labels_round{prior_round}.pt')
        torch.save({'promoted': promoted, 'demoted': demoted_labels}, extra_labels_path)
        print(f'Saved refined labels to {extra_labels_path}')

    for round_num in range(args.start_round, args.rounds + 1):
        print(f'\n{"#"*60}')
        print(f'# SELF-TRAINING ROUND {round_num}/{args.rounds}')
        print(f'{"#"*60}')

        ckpt_path = str(out_dir / f'centernet_round{round_num}.pt')

        _run_training_round(
            feature_dir=args.feature_dir,
            rubin_dir=args.rubin_dir,
            euclid_dir=args.euclid_dir,
            out_path=ckpt_path,
            extra_labels=extra_labels_path,
            init_checkpoint=init_checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            sigma=args.sigma,
            nsig=args.nsig,
            head_ch=args.head_ch,
            labels_mode=args.labels_mode,
            val_patches=args.val_patches,
            mer_fits=args.mer_fits,
            uncertain_ignore=args.uncertain_ignore,
            uncertain_nsig=args.uncertain_nsig,
            uncertain_radius_px=args.uncertain_radius_px,
            synthetic_sources_per_tile=args.synthetic_sources_per_tile,
            synthetic_prob=args.synthetic_prob,
            synthetic_min_snr=args.synthetic_min_snr,
            synthetic_max_snr=args.synthetic_max_snr,
            synthetic_min_sigma_px=args.synthetic_min_sigma_px,
            synthetic_max_sigma_px=args.synthetic_max_sigma_px,
            synthetic_weight=args.synthetic_weight,
            viz_spike_veto_radius=args.promotion_spike_radius,
            viz_spike_veto_width=args.promotion_spike_width,
            device=device,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_run if args.wandb_run else f'round{round_num}',
        )
        init_checkpoint = None

        # Refine labels for next round: promote novel + demote artifacts
        if round_num < args.rounds:
            print(f'\n--- Refining labels (promote > {args.promote_conf}, '
                  f'demote < {args.demote_conf}) ---')
            promoted, demoted_labels = _refine_labels(
                feature_dir=args.feature_dir,
                checkpoint=ckpt_path,
                labels_mode=args.labels_mode,
                rubin_dir=args.rubin_dir,
                euclid_dir=args.euclid_dir,
                nsig=args.nsig,
                promote_conf=args.promote_conf,
                demote_conf=args.demote_conf,
                match_radius=args.match_radius,
                promotion_spike_radius=args.promotion_spike_radius,
                promotion_spike_width=args.promotion_spike_width,
                bright_rescue=args.bright_rescue,
                bright_rescue_nsig=args.bright_rescue_nsig,
                bright_rescue_min_area=args.bright_rescue_min_area,
                bright_rescue_min_radius=args.bright_rescue_min_radius,
                bright_rescue_min_peak_snr=args.bright_rescue_min_peak_snr,
                bright_rescue_match_scale=args.bright_rescue_match_scale,
                bright_rescue_match_min=args.bright_rescue_match_min,
                bright_rescue_match_max=args.bright_rescue_match_max,
                bright_rescue_max_per_tile=args.bright_rescue_max_per_tile,
                device=device,
            )
            extra_labels_path = str(out_dir / f'refined_labels_round{round_num}.pt')
            torch.save({'promoted': promoted, 'demoted': demoted_labels},
                       extra_labels_path)
            print(f'Saved refined labels to {extra_labels_path}')

    # Copy final checkpoint
    final = out_dir / 'centernet_best.pt'
    import shutil
    shutil.copy2(ckpt_path, str(final))
    print(f'\nDone. Final checkpoint: {final}')


if __name__ == '__main__':
    main()
