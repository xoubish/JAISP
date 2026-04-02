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
        --rubin_dir ../data/rubin_tiles_ecdfs \
        --euclid_dir ../data/euclid_tiles_ecdfs \
        --encoder_ckpt ../checkpoints/jaisp_v7_baseline/checkpoint_best.pt \
        --out_dir ../data/cached_features_v7

    # Step 2: Self-training (runs round 1 + round 2 automatically)
    python detection/self_train.py \
        --feature_dir  ../data/cached_features_v7 \
        --rubin_dir    ../data/rubin_tiles_ecdfs \
        --euclid_dir   ../data/euclid_tiles_ecdfs \
        --out_dir      ../checkpoints/centernet_v7_selftrain \
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

from detection.cached_dataset import CachedFeatureDataset, collate_cached
from detection.centernet_detector import CenterNetDetector
from detection.centernet_loss import CenterNetLoss


def _run_training_round(
    feature_dir: str,
    rubin_dir: str,
    euclid_dir: str,
    out_path: str,
    extra_labels: str = None,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    sigma: float = 2.0,
    nsig: float = 3.0,
    device: torch.device = None,
    wandb_project: str = None,
    wandb_name: str = None,
):
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
    ]
    if euclid_dir:
        cmd += ['--euclid_dir', euclid_dir]
    if extra_labels:
        cmd += ['--extra_labels', extra_labels]
    if wandb_project:
        cmd += ['--wandb_project', wandb_project]
    if wandb_name:
        cmd += ['--wandb_run', wandb_name]

    print(f'\n{"="*60}')
    print(f'Running: {" ".join(cmd)}')
    print(f'{"="*60}\n')
    subprocess.run(cmd, check=True)


def _promote_detections(
    feature_dir: str,
    checkpoint: str,
    rubin_dir: str,
    euclid_dir: str,
    nsig: float,
    conf_threshold: float = 0.8,
    match_radius: float = 0.01,
    device: torch.device = None,
) -> dict:
    """Run trained detector on all tiles, find novel high-confidence detections.

    Returns dict[tile_id] -> np.ndarray[N, 2] of promoted centroid coords
    (normalized [0,1], in the label coordinate frame).
    """
    from detection.cached_dataset import CachedFeatureDataset
    from detection.dataset import _pseudo_labels_vis, _pseudo_labels

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained model (head only, no encoder needed)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = CenterNetDetector(encoder=None, encoder_dim=ckpt['encoder_dim'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device).eval()

    # Load features (no augmentation -- use aug0 only for inference)
    feature_dir = Path(feature_dir)
    feat_files = sorted(feature_dir.glob('tile_*_aug0.pt'))

    # Load original VIS pseudo-labels for comparison
    euclid_dir_p = Path(euclid_dir) if euclid_dir else None
    rubin_dir_p = Path(rubin_dir)

    promoted = {}
    total_promoted = 0
    total_tiles = 0

    for feat_path in feat_files:
        tile_id = feat_path.stem.rsplit('_aug', 1)[0]
        cached = torch.load(feat_path, map_location='cpu', weights_only=True)
        feats = cached['features'].unsqueeze(0).to(device)  # [1, 256, H, W]

        # Run detector
        with torch.no_grad():
            neck_out = model.neck(feats)
            hm = model.hm_head(neck_out).sigmoid()
            off = model.off_head(neck_out)

        # Find peaks
        import torch.nn.functional as F
        hm_max = F.max_pool2d(hm, 3, stride=1, padding=1)
        peaks = (hm == hm_max) & (hm > conf_threshold)

        if not peaks.any():
            total_tiles += 1
            continue

        yi, xi = torch.where(peaks[0, 0])
        scores = hm[0, 0, yi, xi]
        dx = off[0, 0, yi, xi]
        dy = off[0, 1, yi, xi]
        _, H, W = cached['features'].shape
        pred_x = (xi.float() + dx) / max(W - 1, 1)
        pred_y = (yi.float() + dy) / max(H - 1, 1)
        pred_xy = torch.stack([pred_x, pred_y], dim=1).cpu().numpy()

        # Load original VIS labels for this tile
        orig_labels = None
        if euclid_dir_p:
            ep = euclid_dir_p / f'{tile_id}_euclid.npz'
            if ep.exists():
                try:
                    edata = np.load(str(ep), allow_pickle=True, mmap_mode='r')
                    vis_img = np.nan_to_num(np.asarray(edata['img_VIS'], dtype=np.float32), nan=0.0)
                    orig_labels, _, _, _ = _pseudo_labels_vis(vis_img, nsig, 1000)
                except Exception:
                    pass

        if orig_labels is None:
            rp = rubin_dir_p / f'{tile_id}.npz'
            candidates = list(rubin_dir_p.glob(f'{tile_id}*.npz'))
            rp = candidates[0] if candidates else rp
            try:
                rdata = np.load(str(rp), allow_pickle=True, mmap_mode='r')
                raw_img = np.nan_to_num(np.asarray(rdata['img'], dtype=np.float32), nan=0.0)
                orig_labels, _, _, _ = _pseudo_labels(raw_img, nsig, 1000)
            except Exception:
                orig_labels = np.zeros((0, 2), dtype=np.float32)

        # Find novel detections: predictions with no nearby original label
        novel = []
        for i in range(len(pred_xy)):
            if orig_labels.shape[0] == 0:
                novel.append(pred_xy[i])
                continue
            dists = np.sqrt(((orig_labels - pred_xy[i]) ** 2).sum(axis=1))
            if dists.min() > match_radius:
                novel.append(pred_xy[i])

        if novel:
            promoted[tile_id] = np.array(novel, dtype=np.float32)
            total_promoted += len(novel)
        total_tiles += 1

    print(f'\nPromotion: {total_promoted} novel detections across {total_tiles} tiles')
    return promoted


def main():
    p = argparse.ArgumentParser(description='Self-training loop for CenterNet detection.')
    p.add_argument('--feature_dir', required=True, help='Precomputed features directory')
    p.add_argument('--rubin_dir',   required=True)
    p.add_argument('--euclid_dir',  default=None)
    p.add_argument('--out_dir',     required=True, help='Output directory for checkpoints + labels')
    p.add_argument('--rounds',      type=int, default=2, help='Number of self-training rounds')
    p.add_argument('--epochs',      type=int, default=100, help='Epochs per round')
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--nsig',        type=float, default=3.0)
    p.add_argument('--sigma',       type=float, default=2.0)
    p.add_argument('--promote_conf', type=float, default=0.8,
                   help='Confidence threshold for promoting novel detections')
    p.add_argument('--match_radius', type=float, default=0.01,
                   help='Normalized distance below which a prediction matches an existing label')
    p.add_argument('--wandb_project', default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_labels_path = None

    for round_num in range(1, args.rounds + 1):
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
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            sigma=args.sigma,
            nsig=args.nsig,
            device=device,
            wandb_project=args.wandb_project,
            wandb_name=f'round{round_num}',
        )

        # Promote novel detections for next round
        if round_num < args.rounds:
            print(f'\n--- Promoting novel detections (conf > {args.promote_conf}) ---')
            promoted = _promote_detections(
                feature_dir=args.feature_dir,
                checkpoint=ckpt_path,
                rubin_dir=args.rubin_dir,
                euclid_dir=args.euclid_dir,
                nsig=args.nsig,
                conf_threshold=args.promote_conf,
                match_radius=args.match_radius,
                device=device,
            )
            extra_labels_path = str(out_dir / f'promoted_round{round_num}.pt')
            torch.save(promoted, extra_labels_path)
            print(f'Saved promoted labels to {extra_labels_path}')

    # Copy final checkpoint
    final = out_dir / 'centernet_best.pt'
    import shutil
    shutil.copy2(ckpt_path, str(final))
    print(f'\nDone. Final checkpoint: {final}')


if __name__ == '__main__':
    main()
