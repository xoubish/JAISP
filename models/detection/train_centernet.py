"""Train CenterNet source detector on frozen JAISPFoundationV7 encoder.

Usage
-----
    python detection/train_centernet.py \
        --rubin_dir    ../data/rubin_tiles_ecdfs \
        --euclid_dir   ../data/euclid_tiles_ecdfs \
        --encoder_ckpt ../checkpoints/jaisp_v7_baseline/checkpoint_best.pt \
        --out          ../checkpoints/centernet_v7.pt \
        --epochs 100 --wandb_project jaisp-detection
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jaisp_foundation_v7 import JAISPFoundationV7, ALL_BANDS, RUBIN_BANDS
from detection.centernet_detector import CenterNetDetector, JAISPEncoderWrapper, _StubEncoder
from detection.centernet_loss import CenterNetLoss
from detection.dataset import TileDetectionDataset, collate_fn


# ---------------------------------------------------------------------------
# Encoder loading (reused from train_detection.py)
# ---------------------------------------------------------------------------

def _load_encoder(encoder_ckpt, device, freeze=True):
    if encoder_ckpt is None:
        print('  [warn] No --encoder_ckpt — using stub CNN encoder.')
        stub = _StubEncoder(in_channels=len(RUBIN_BANDS)).to(device)
        return stub, 512

    print(f'  Loading v7 encoder from {encoder_ckpt}')
    ckpt = torch.load(encoder_ckpt, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})

    model = JAISPFoundationV7(
        band_names               = cfg.get('band_names', ALL_BANDS),
        stem_ch                  = cfg.get('stem_ch', 64),
        hidden_ch                = cfg.get('hidden_ch', 256),
        blocks_per_stage         = cfg.get('blocks_per_stage', 2),
        transformer_depth        = cfg.get('transformer_depth', 4),
        transformer_heads        = cfg.get('transformer_heads', 8),
        fused_pixel_scale_arcsec = cfg.get('fused_pixel_scale_arcsec', 0.8),
    )
    missing, _ = model.load_state_dict(ckpt['model'], strict=False)
    enc_missing = [k for k in missing if not k.startswith(('encoder.skip_projs', 'target_decoders'))]
    if enc_missing:
        print(f'  [warn] Missing encoder keys: {enc_missing}')

    encoder_dim = cfg.get('hidden_ch', 256)
    wrapper = JAISPEncoderWrapper(model, freeze=freeze).to(device)
    n_enc = sum(p.numel() for p in wrapper.encoder.parameters())
    print(f'  v7 encoder loaded ({n_enc/1e6:.1f}M params, frozen={freeze}, encoder_dim={encoder_dim})')
    return wrapper, encoder_dim


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _log_tile(batch, out, wandb, step, conf_thr=0.3, nms_kernel=3):
    """Overlay GT centroids and CenterNet heatmap + detected peaks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    r_band = batch['images']['rubin_r'][0, 0].cpu().numpy()
    lo, hi = np.percentile(r_band, [1, 99])
    rgb = np.clip((r_band - lo) / max(hi - lo, 1e-6), 0, 1)
    H, W = rgb.shape

    hm = out['heatmap'][0, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: r-band + GT + predictions
    ax = axes[0]
    ax.imshow(rgb, origin='lower', cmap='gray')

    gt_xy = batch['centroids'][0].cpu().numpy()
    if len(gt_xy):
        ax.scatter(gt_xy[:, 0] * (W - 1), gt_xy[:, 1] * (H - 1),
                   s=60, marker='o', edgecolors='lime', facecolors='none',
                   lw=1.2, label=f'GT ({len(gt_xy)})')

    # Detect peaks from heatmap (all on CPU for viz)
    hm_t = out['heatmap'][0:1].detach().cpu()
    pad = nms_kernel // 2
    hm_max = F.max_pool2d(hm_t, nms_kernel, stride=1, padding=pad)
    peaks = (hm_t == hm_max) & (hm_t > conf_thr)
    n_pred = peaks.sum().item()

    if n_pred > 0:
        off = out['offset'][0].detach().cpu()
        yi, xi = torch.where(peaks[0, 0])
        hm_h, hm_w = hm.shape
        px = (xi.float() + off[0, yi, xi]) / max(hm_w - 1, 1) * (W - 1)
        py = (yi.float() + off[1, yi, xi]) / max(hm_h - 1, 1) * (H - 1)
        ax.scatter(px.numpy(), py.numpy(),
                   s=15, marker='x', c='red', lw=0.8, label=f'pred ({n_pred})')

    ax.legend(fontsize=6)
    ax.set_title(f'step {step}  GT={len(gt_xy)}  pred={n_pred}')
    ax.axis('off')

    # Right: heatmap
    ax2 = axes[1]
    ax2.imshow(hm, origin='lower', cmap='hot', vmin=0, vmax=1)
    ax2.set_title(f'heatmap (max={hm.max():.3f})')
    ax2.axis('off')

    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training CenterNet detector on {device}')

    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    # Dataset
    full_ds = TileDetectionDataset(
        rubin_dir=args.rubin_dir,
        euclid_dir=args.euclid_dir,
        nsig=args.nsig,
        max_sources=1000,  # no fixed query budget -- detect as many as classical finds
        use_all_bands=args.euclid_dir is not None,
        augment=True,
    )
    n_val = max(1, int(0.1 * len(full_ds)))
    n_tr = len(full_ds) - n_val
    tr_ds, val_ds = random_split(
        full_ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
    print(f'Train: {n_tr} tiles   Val: {n_val} tiles')

    # Encoder + detector
    encoder, encoder_dim = _load_encoder(args.encoder_ckpt, device)

    model = CenterNetDetector(
        encoder=encoder,
        encoder_dim=encoder_dim,
        predict_profile=args.predict_profile,
    ).to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_train / 1e6:.2f}M')

    criterion = CenterNetLoss(sigma=args.sigma)
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
            rms = {b: v.to(device) for b, v in batch['rms'].items()}
            out = model(images, rms)

            losses = criterion(
                out,
                [c.to(device) for c in batch['centroids']],
            )
            optimizer.zero_grad()
            losses['loss_total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_losses.append(float(losses['loss_total']))
            step += 1
            if use_wandb and step % 10 == 0:
                wandb.log({
                    'train/loss':     float(losses['loss_total']),
                    'train/loss_hm':  float(losses['loss_hm']),
                    'train/loss_off': float(losses['loss_off']),
                    'train/n_sources': float(losses['n_sources']),
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
                rms = {b: v.to(device) for b, v in batch['rms'].items()}
                out = model(images, rms)
                losses = criterion(
                    out,
                    [c.to(device) for c in batch['centroids']],
                )
                val_losses.append(float(losses['loss_total']))
        mean_val = float(np.mean(val_losses)) if val_losses else float('nan')

        lr_now = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch + 1:3d}/{args.epochs}  '
              f'tr={mean_tr:.4f}  val={mean_val:.4f}  lr={lr_now:.2e}')

        if use_wandb:
            log = {
                'train/loss_epoch': mean_tr,
                'val/loss': mean_val,
                'train/lr': lr_now,
                'epoch': epoch + 1,
            }
            if (epoch + 1) % 5 == 0 or epoch == 0:
                try:
                    sample = next(iter(val_loader))
                    with torch.no_grad():
                        sim = {b: v.to(device) for b, v in sample['images'].items()}
                        srm = {b: v.to(device) for b, v in sample['rms'].items()}
                        sout = model(sim, srm)
                    log['viz/tile'] = _log_tile(sample, sout, wandb, step)
                except Exception as exc:
                    print(f'  [warn] viz failed: {exc}')
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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rubin_dir',        required=True)
    p.add_argument('--euclid_dir',       default=None)
    p.add_argument('--encoder_ckpt',     default=None)
    p.add_argument('--out',              default='../checkpoints/centernet_v7.pt')
    p.add_argument('--epochs',           type=int,   default=100)
    p.add_argument('--batch_size',       type=int,   default=2,
                   help='Batch size (default 2; Euclid tiles are large)')
    p.add_argument('--lr',               type=float, default=1e-4)
    p.add_argument('--nsig',             type=float, default=3.0,
                   help='Detection significance for pseudo-labels')
    p.add_argument('--sigma',            type=float, default=2.0,
                   help='Gaussian sigma for heatmap targets (feature-map pixels)')
    p.add_argument('--predict_profile',  action='store_true',
                   help='Add profile head (e1, e2, r_half, sersic_n)')
    p.add_argument('--wandb_project',    default=None)
    p.add_argument('--wandb_run',        default=None)
    args = p.parse_args()
    train(args)
