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

    The model trained on 10-band features knows more than the VIS pseudo-labels:
    - Sources only visible in VIS (like diffraction spikes) will have LOW
      confidence because the other 9 bands show nothing there.
    - Sources invisible in VIS but present in other bands will have HIGH
      confidence as novel detections.

    Returns (promoted, demoted):
        promoted: dict[tile_id] -> np.ndarray[N, 2] new labels to ADD
        demoted:  dict[tile_id] -> np.ndarray[N, 2] old labels to REMOVE
    """
    from detection.dataset import _pseudo_labels_vis, _pseudo_labels

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained model (head only)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = CenterNetDetector(encoder=None, encoder_dim=ckpt['encoder_dim'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device).eval()

    feature_dir = Path(feature_dir)
    feat_files = sorted(feature_dir.glob('tile_*_aug0.pt'))
    euclid_dir_p = Path(euclid_dir) if euclid_dir else None
    rubin_dir_p = Path(rubin_dir)

    promoted = {}
    demoted = {}
    total_promoted = 0
    total_demoted = 0
    total_tiles = 0

    import torch.nn.functional as F

    for feat_path in feat_files:
        tile_id = feat_path.stem.rsplit('_aug', 1)[0]
        cached = torch.load(feat_path, map_location='cpu', weights_only=True)
        feats = cached['features'].unsqueeze(0).to(device)

        # Run detector
        with torch.no_grad():
            neck_out = model.neck(feats)
            hm = model.hm_head(neck_out).sigmoid()  # [1, 1, H, W]
            off = model.off_head(neck_out)

        # hm dimensions (may be upsampled vs bottleneck, e.g. 1040x1040 for VIS)
        _, _, fH, fW = hm.shape

        # Load original pseudo-labels for this tile
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

        hm_np = hm[0, 0].cpu().numpy()

        # --- DEMOTE: check model confidence at each existing pseudo-label ---
        tile_demoted = []
        if orig_labels.shape[0] > 0:
            for i in range(orig_labels.shape[0]):
                # Map normalized coords to feature-map pixel
                fx = int(round(orig_labels[i, 0] * (fW - 1)))
                fy = int(round(orig_labels[i, 1] * (fH - 1)))
                fx = max(0, min(fx, fW - 1))
                fy = max(0, min(fy, fH - 1))
                conf = float(hm_np[fy, fx])
                if conf < demote_conf:
                    tile_demoted.append(orig_labels[i])

        if tile_demoted:
            demoted[tile_id] = np.array(tile_demoted, dtype=np.float32)
            total_demoted += len(tile_demoted)

        # --- PROMOTE: find high-confidence predictions with no matching label ---
        hm_max = F.max_pool2d(hm, 3, stride=1, padding=1)
        peaks = (hm == hm_max) & (hm > promote_conf)

        tile_promoted = []
        if peaks.any():
            yi, xi = torch.where(peaks[0, 0])
            dx = off[0, 0, yi, xi]
            dy = off[0, 1, yi, xi]
            pred_x = (xi.float() + dx) / max(fW - 1, 1)
            pred_y = (yi.float() + dy) / max(fH - 1, 1)
            pred_xy = torch.stack([pred_x, pred_y], dim=1).cpu().numpy()

            for j in range(len(pred_xy)):
                if orig_labels.shape[0] == 0:
                    tile_promoted.append(pred_xy[j])
                    continue
                dists = np.sqrt(((orig_labels - pred_xy[j]) ** 2).sum(axis=1))
                if dists.min() > match_radius:
                    tile_promoted.append(pred_xy[j])

        if tile_promoted:
            promoted[tile_id] = np.array(tile_promoted, dtype=np.float32)
            total_promoted += len(tile_promoted)

        total_tiles += 1

    print(f'\nLabel refinement across {total_tiles} tiles:')
    print(f'  Promoted: {total_promoted} novel high-confidence detections')
    print(f'  Demoted:  {total_demoted} low-confidence pseudo-labels (likely artifacts)')
    return promoted, demoted


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
    p.add_argument('--demote_conf', type=float, default=0.3,
                   help='Confidence threshold below which existing labels are demoted '
                        '(model says "this is not a real source in 10-band space")')
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
