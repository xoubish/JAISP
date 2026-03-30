"""
Train JaispDetector on top of the frozen JAISPEncoderV6.

Phase 1 (default): freeze encoder, train decoder + heads only (~10M params).
Phase 2 (--finetune_encoder): unfreeze encoder for end-to-end fine-tuning.

Usage
-----
    # With MAE v6 checkpoint (recommended)
    python detection/train_detection.py \
        --rubin_dir    ../data/rubin_tiles_ecdfs \
        --encoder_ckpt ../checkpoints/jaisp_v6_phaseB2/checkpoint_best.pt \
        --out          ../checkpoints/detector_v1.pt \
        --epochs 50 --wandb_project jaisp-detection

    # Without MAE checkpoint (stub CNN encoder, for quick testing)
    python detection/train_detection.py \
        --rubin_dir ../data/rubin_tiles_ecdfs \
        --out       ../checkpoints/detector_v1.pt \
        --epochs 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from typing import Optional

_HERE   = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jaisp_foundation_v6 import JAISPFoundationV6, ALL_BANDS, RUBIN_BANDS
from detection.detector import JaispDetector, JAISPEncoderWrapper, _StubEncoder
from detection.matcher  import DetectionLoss
from detection.dataset  import TileDetectionDataset, collate_fn


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def _load_encoder(
    encoder_ckpt: Optional[str],
    device: torch.device,
    freeze: bool = True,
) -> torch.nn.Module:
    """
    Load JAISPEncoderV6 from a JAISPFoundationV6 checkpoint and wrap it.
    Falls back to stub CNN if no checkpoint is provided.
    """
    if encoder_ckpt is None:
        print('  [warn] No --encoder_ckpt given — using stub CNN encoder.')
        stub = _StubEncoder(in_channels=len(RUBIN_BANDS)).to(device)
        return stub

    print(f'  Loading MAE v6 encoder from {encoder_ckpt}')
    ckpt = torch.load(encoder_ckpt, map_location='cpu', weights_only=False)

    # Reconstruct the full foundation model to get the right encoder config,
    # then extract only the encoder — we don't need the decoder at all.
    model = JAISPFoundationV6(band_names=ALL_BANDS)
    model.load_state_dict(ckpt['model'])

    wrapper = JAISPEncoderWrapper(model.encoder, freeze=freeze).to(device)
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f'  Encoder loaded ({n_enc/1e6:.1f}M params, frozen={freeze})')
    return wrapper


# ---------------------------------------------------------------------------
# W&B visualisation
# ---------------------------------------------------------------------------

def _log_tile(batch: dict, out: dict, wandb, step: int, conf_thr: float = 0.5):
    """Overlay GT and predicted source positions on the r-band image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # r-band from the images dict
    r_band = batch['images']['rubin_r'][0, 0].cpu().numpy()
    lo, hi = np.percentile(r_band, [1, 99])
    rgb = np.clip((r_band - lo) / max(hi - lo, 1e-6), 0, 1)

    H, W = rgb.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb, origin='lower', cmap='gray')

    # GT
    gt_xy  = batch['centroids'][0].cpu().numpy()
    gt_cls = batch['classes'][0].cpu().numpy()
    if len(gt_xy):
        stars = gt_xy[gt_cls == 0]
        gals  = gt_xy[gt_cls == 1]
        if len(stars):
            ax.scatter(stars[:, 0] * (W-1), stars[:, 1] * (H-1),
                       s=20, marker='+', c='cyan',  lw=0.8, label='GT star')
        if len(gals):
            ax.scatter(gals[:, 0]  * (W-1), gals[:, 1]  * (H-1),
                       s=20, marker='o', c='yellow', lw=0.8,
                       facecolors='none', label='GT galaxy')

    # Predictions
    scores = out['conf'][0].sigmoid().detach().cpu().numpy()
    keep   = scores > conf_thr
    if keep.any():
        pxy = out['centroids'][0][keep].detach().cpu().numpy()
        ax.scatter(pxy[:, 0] * (W-1), pxy[:, 1] * (H-1),
                   s=15, marker='x', c='red', lw=0.8, label='pred')

    ax.legend(fontsize=6)
    ax.set_title(f'step {step}  GT={len(gt_xy)}  pred={keep.sum()}')
    ax.axis('off')
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training JaispDetector on {device}')

    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    # Dataset
    full_ds = TileDetectionDataset(
        rubin_dir=args.rubin_dir,
        euclid_dir=args.euclid_dir,
        nsig=5.0,
        max_sources=args.num_queries,
        use_all_bands=args.euclid_dir is not None,
        augment=True,
    )
    n_val = max(1, int(0.1 * len(full_ds)))
    n_tr  = len(full_ds) - n_val
    tr_ds, val_ds = random_split(
        full_ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    tr_loader  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
    print(f'Train: {n_tr} tiles   Val: {n_val} tiles')

    # Encoder + detector
    encoder = _load_encoder(args.encoder_ckpt, device,
                            freeze=not args.finetune_encoder)
    encoder_dim = 512   # JAISPEncoderV6 bottleneck channels

    model = JaispDetector(
        encoder=encoder,
        num_queries=args.num_queries,
        d_model=args.d_model,
        encoder_dim=encoder_dim,
    ).to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_train/1e6:.2f}M')

    criterion = DetectionLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float('inf')
    step = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        tr_losses = []
        for batch in tr_loader:
            images = {b: v.to(device) for b, v in batch['images'].items()}
            rms    = {b: v.to(device) for b, v in batch['rms'].items()}
            out    = model(images, rms)

            losses = criterion(
                out['centroids'], out['logits'], out['conf'],
                [c.to(device) for c in batch['centroids']],
                [y.to(device) for y in batch['classes']],
            )
            optimizer.zero_grad()
            losses['loss_total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            tr_losses.append(float(losses['loss_total']))
            step += 1
            if use_wandb and step % 10 == 0:
                wandb.log({
                    'train/loss':          float(losses['loss_total']),
                    'train/loss_pos':      float(losses['loss_pos']),
                    'train/loss_cls':      float(losses['loss_cls']),
                    'train/loss_conf_obj': float(losses['loss_conf_obj']),
                    'train/n_matched':     float(losses['n_matched']),
                    'step': step,
                })

        scheduler.step()
        mean_tr = float(np.mean(tr_losses)) if tr_losses else float('nan')

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                images = {b: v.to(device) for b, v in batch['images'].items()}
                rms    = {b: v.to(device) for b, v in batch['rms'].items()}
                out    = model(images, rms)
                losses = criterion(
                    out['centroids'], out['logits'], out['conf'],
                    [c.to(device) for c in batch['centroids']],
                    [y.to(device) for y in batch['classes']],
                )
                val_losses.append(float(losses['loss_total']))
        mean_val = float(np.mean(val_losses)) if val_losses else float('nan')

        lr_now = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1:3d}/{args.epochs}  '
              f'tr={mean_tr:.4f}  val={mean_val:.4f}  lr={lr_now:.2e}')

        if use_wandb:
            log = {
                'train/loss_epoch': mean_tr,
                'val/loss': mean_val,
                'train/lr': lr_now,
                'epoch': epoch + 1,
            }
            if (epoch + 1) % 5 == 0:
                sample = next(iter(val_loader))
                with torch.no_grad():
                    sim = {b: v.to(device) for b, v in sample['images'].items()}
                    srm = {b: v.to(device) for b, v in sample['rms'].items()}
                    sout = model(sim, srm)
                log['viz/tile'] = _log_tile(sample, sout, wandb, step)
            wandb.log(log)

        if mean_val < best_val:
            best_val = mean_val
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            model.save(args.out)
            print(f'  ✓ saved → {args.out}')
            if use_wandb:
                wandb.run.summary['best_val_loss'] = best_val

    if use_wandb:
        wandb.finish()
    print(f'Done. Best val loss: {best_val:.4f}')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rubin_dir',        required=True)
    p.add_argument('--euclid_dir',       default=None)
    p.add_argument('--encoder_ckpt',     default=None,
                   help='JAISPFoundationV6 checkpoint; omit to use stub encoder')
    p.add_argument('--out',              default='../checkpoints/detector_v1.pt')
    p.add_argument('--epochs',           type=int,   default=50)
    p.add_argument('--batch_size',       type=int,   default=4)
    p.add_argument('--num_queries',      type=int,   default=300)
    p.add_argument('--d_model',          type=int,   default=256)
    p.add_argument('--lr',               type=float, default=1e-4)
    p.add_argument('--finetune_encoder', action='store_true')
    p.add_argument('--wandb_project',    default=None)
    p.add_argument('--wandb_run',        default=None)
    args = p.parse_args()
    train(args)
