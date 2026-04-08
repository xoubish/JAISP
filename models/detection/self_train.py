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
    python detection/precompute_features.py \
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
    ]
    if euclid_dir:
        cmd += ['--euclid_dir', euclid_dir]
    if extra_labels:
        cmd += ['--extra_labels', extra_labels]
    if init_checkpoint:
        cmd += ['--init_checkpoint', init_checkpoint]
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

    # Load trained model (head only)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = CenterNetDetector(
        encoder=None,
        encoder_dim=ckpt['encoder_dim'],
        head_ch=ckpt.get('head_ch', 256),
    )
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device).eval()

    feature_dir = Path(feature_dir)
    feat_files = sorted(feature_dir.glob('tile_*_aug0.pt'))

    # Load cached pseudo-labels (written by CachedFeatureDataset._compute_labels)
    label_cache_path = feature_dir / 'pseudo_labels.pt'
    if label_cache_path.exists():
        label_cache = torch.load(label_cache_path, map_location='cpu',
                                 weights_only=False)
        print(f'  Loaded cached pseudo-labels from {label_cache_path}')
    else:
        # Fallback: compute on the fly (slow path, handled by _load_labels_fallback)
        label_cache = None
        print('  [warn] No cached pseudo-labels found, computing on the fly')

    promoted = {}
    demoted = {}
    total_promoted = 0
    total_demoted = 0
    total_tiles = 0

    for feat_path in feat_files:
        tile_id = feat_path.stem.rsplit('_aug', 1)[0]
        cached = torch.load(feat_path, map_location='cpu', weights_only=True)
        feats = cached['features'].unsqueeze(0).to(device)

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

        if peaks.any():
            yi, xi = torch.where(peaks[0, 0])
            dx = off[0, 0, yi, xi]
            dy = off[0, 1, yi, xi]
            pred_x = (xi.float() + dx) / max(fW - 1, 1)
            pred_y = (yi.float() + dy) / max(fH - 1, 1)
            pred_xy = torch.stack([pred_x, pred_y], dim=1).cpu().numpy()

            if orig_labels.shape[0] == 0:
                novel = pred_xy
            else:
                # Vectorized: cdist between predictions and existing labels
                from scipy.spatial.distance import cdist
                dists = cdist(pred_xy, orig_labels)  # [n_pred, n_labels]
                novel = pred_xy[dists.min(axis=1) > match_radius]

            if len(novel) > 0:
                promoted[tile_id] = novel.astype(np.float32)
                total_promoted += len(novel)

        total_tiles += 1

    print(f'\nLabel refinement across {total_tiles} tiles:')
    print(f'  Promoted: {total_promoted} novel high-confidence detections')
    print(f'  Demoted:  {total_demoted} low-confidence pseudo-labels (likely artifacts)')
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
    p.add_argument('--promote_conf', type=float, default=0.8,
                   help='Confidence threshold for promoting novel detections')
    p.add_argument('--demote_conf', type=float, default=0.3,
                   help='Confidence threshold below which existing labels are demoted '
                        '(model says "this is not a real source in 10-band space")')
    p.add_argument('--match_radius', type=float, default=0.01,
                   help='Normalized distance below which a prediction matches an existing label')
    p.add_argument('--init_checkpoint', default=None,
                   help='Existing detector checkpoint to bootstrap from when start_round > 1')
    p.add_argument('--wandb_project', default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_labels_path = None
    init_checkpoint = args.init_checkpoint

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
            rubin_dir=args.rubin_dir,
            euclid_dir=args.euclid_dir,
            nsig=args.nsig,
            promote_conf=args.promote_conf,
            demote_conf=args.demote_conf,
            match_radius=args.match_radius,
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
            device=device,
            wandb_project=args.wandb_project,
            wandb_name=f'round{round_num}',
        )
        init_checkpoint = None

        # Refine labels for next round: promote novel + demote artifacts
        if round_num < args.rounds:
            print(f'\n--- Refining labels (promote > {args.promote_conf}, '
                  f'demote < {args.demote_conf}) ---')
            promoted, demoted_labels = _refine_labels(
                feature_dir=args.feature_dir,
                checkpoint=ckpt_path,
                rubin_dir=args.rubin_dir,
                euclid_dir=args.euclid_dir,
                nsig=args.nsig,
                promote_conf=args.promote_conf,
                demote_conf=args.demote_conf,
                match_radius=args.match_radius,
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
