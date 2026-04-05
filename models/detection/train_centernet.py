"""Train CenterNet source detector on frozen JAISPFoundationV7 encoder.

Usage
-----
    python detection/train_centernet.py \
        --rubin_dir    ../data/rubin_tiles_all \
        --euclid_dir   ../data/euclid_tiles_all \
        --encoder_ckpt ../checkpoints/jaisp_v7_tiles_all_ddp_online/checkpoint_best.pt \
        --out          ../checkpoints/centernet_v7_tiles_all_live.pt \
        --epochs 60 --wandb_project jaisp-detection
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split

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

def _log_tile(batch, out, wandb, step, conf_thr=0.3, nms_kernel=7, euclid_dir=None):
    """Overlay GT centroids and CenterNet heatmap + detected peaks."""
    import sys
    import matplotlib
    # matplotlib.use() must be called before pyplot is first imported; after
    # that it is silently ignored.  Guard so repeated calls don't warn.
    if 'matplotlib.pyplot' not in sys.modules:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    hm = out['heatmap'][0, 0].detach().cpu().numpy()
    hm_h, hm_w = hm.shape

    # Background image: r-band (live mode), VIS (cached mode), or black fallback
    rgb = None
    if 'images' in batch and 'rubin_r' in batch['images']:
        r_band = batch['images']['rubin_r'][0, 0].cpu().numpy()
        lo, hi = np.percentile(r_band, [1, 99])
        rgb = np.clip((r_band - lo) / max(hi - lo, 1e-6), 0, 1)
    elif euclid_dir and 'tile_id' in batch:
        # Cached mode: load VIS image and apply same augmentation as features
        tile_id = batch['tile_id'][0] if isinstance(batch['tile_id'], list) else batch['tile_id']
        vis_path = Path(euclid_dir) / f'{tile_id}_euclid.npz'
        if vis_path.exists():
            try:
                edata = np.load(str(vis_path), allow_pickle=True, mmap_mode='r')
                vis = np.nan_to_num(np.asarray(edata['img_VIS'], dtype=np.float32), nan=0.0)
                # Apply the same augmentation that was applied to the cached features
                aug_idx = batch.get('aug_idx', [0])
                aug_idx = aug_idx[0] if isinstance(aug_idx, list) else aug_idx
                all_augs = [(r, fu, fl) for r in range(4)
                            for fu in (False, True) for fl in (False, True)]
                if aug_idx < len(all_augs):
                    n_rot, flip_ud, flip_lr = all_augs[aug_idx]
                    vis = np.rot90(vis, n_rot).copy()
                    if flip_ud:
                        vis = np.flipud(vis).copy()
                    if flip_lr:
                        vis = np.fliplr(vis).copy()
                lo, hi = np.percentile(vis, [1, 99])
                rgb = np.clip((vis - lo) / max(hi - lo, 1e-6), 0, 1)
            except Exception:
                pass

    if rgb is None:
        rgb = np.zeros((hm_h, hm_w))

    H, W = rgb.shape

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
        # Normalize heatmap coords → [0,1], then scale to background image pixels.
        # Clamp so out-of-range offsets don't scatter outside the axes.
        px = ((xi.float() + off[0, yi, xi]) / max(hm_w - 1, 1) * (W - 1)).clamp(0, W - 1)
        py = ((yi.float() + off[1, yi, xi]) / max(hm_h - 1, 1) * (H - 1)).clamp(0, H - 1)
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

def _cached_forward(model, feats: torch.Tensor) -> dict:
    """Run the CenterNet head on pre-computed encoder features.

    Used in cached-feature training mode where the encoder is not called.
    Centralizes the neck→head logic so profile_head is never silently skipped.
    """
    neck_out = model.neck(feats)
    out = {
        'heatmap':  model.hm_head(neck_out).sigmoid(),
        'offset':   model.off_head(neck_out),
        'log_flux': model.flux_head(neck_out).squeeze(1),
    }
    if model.profile_head is not None:
        out['profile'] = model.profile_head(neck_out)
    return out


def train(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    use_cached = args.feature_dir is not None
    mode_str = 'cached features' if use_cached else 'live encoder'
    print(f'Training CenterNet detector on {device} ({mode_str})')

    if args.predict_profile:
        raise ValueError(
            '--predict_profile is not ready yet: the current CenterNet loss '
            'has no profile supervision.'
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    if use_cached:
        from detection.cached_dataset import CachedFeatureDataset, collate_cached

        full_ds = CachedFeatureDataset(
            feature_dir=args.feature_dir,
            rubin_dir=args.rubin_dir,
            euclid_dir=args.euclid_dir,
            nsig=args.nsig,
            max_sources=1000,
            extra_labels=args.extra_labels,
        )
        # Split at tile level so all augmentations of a tile stay together.
        # A sample-level split leaks: different augments of the same tile end up
        # in both train and val, making val loss misleadingly low then rising.
        rng = np.random.default_rng(args.seed)
        tile_ids = full_ds._tile_ids  # list[str]
        shuffled = rng.permutation(len(tile_ids))
        n_val_tiles = max(1, int(0.1 * len(tile_ids)))
        val_tile_set = {tile_ids[int(i)] for i in shuffled[:n_val_tiles]}
        tr_indices  = [i for i, (tid, _) in enumerate(full_ds._samples) if tid not in val_tile_set]
        val_indices = [i for i, (tid, _) in enumerate(full_ds._samples) if tid in val_tile_set]
        tr_ds  = Subset(full_ds, tr_indices)
        val_ds = Subset(full_ds, val_indices)
        n_tr, n_val = len(tr_indices), len(val_indices)
        val_workers = max(1, args.num_workers // 2) if args.num_workers > 0 else 0
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_cached, num_workers=args.num_workers,
                               pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_cached, num_workers=val_workers)
        encoder_dim = 256  # V7 default
    else:
        full_ds = TileDetectionDataset(
            rubin_dir=args.rubin_dir,
            euclid_dir=args.euclid_dir,
            nsig=args.nsig,
            max_sources=1000,
            use_all_bands=args.use_euclid_bands,
            augment=True,
        )
        n_val = max(1, int(0.1 * len(full_ds)))
        n_tr  = len(full_ds) - n_val
        tr_ds, val_ds = random_split(
            full_ds, [n_tr, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        val_workers = max(1, args.num_workers // 2) if args.num_workers > 0 else 0
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=args.num_workers,
                               pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=val_workers)
        encoder_dim = None  # determined by _load_encoder

    print(f'Train: {n_tr} samples   Val: {n_val} samples')

    # Model
    if use_cached:
        # Head-only model (no encoder needed)
        model = CenterNetDetector(
            encoder=None,
            encoder_dim=encoder_dim,
            predict_profile=args.predict_profile,
        ).to(device)
    else:
        encoder, encoder_dim = _load_encoder(args.encoder_ckpt, device)
        model = CenterNetDetector(
            encoder=encoder,
            encoder_dim=encoder_dim,
            predict_profile=args.predict_profile,
        ).to(device)

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        missing_non_encoder = [k for k in missing if not k.startswith('encoder.')]
        if missing_non_encoder:
            print(f'  [warn] Missing init-checkpoint keys: {missing_non_encoder}')
        if unexpected:
            print(f'  [warn] Unexpected init-checkpoint keys: {unexpected}')
        print(f'  Initialized detector weights from {args.init_checkpoint}')

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

    # Pre-build a single fixed viz sample (middle of val set) so the same tile
    # is shown every epoch, enabling direct cross-epoch comparison.
    _collate = collate_cached if use_cached else collate_fn
    if use_cached:
        viz_sample = _collate([val_ds[len(val_ds) // 2]])
    else:
        prev_augment = full_ds._base.augment
        full_ds._base.augment = False
        try:
            viz_sample = _collate([val_ds[len(val_ds) // 2]])
        finally:
            full_ds._base.augment = prev_augment

    for epoch in range(args.epochs):
        # Train
        model.train()
        tr_losses = []
        for batch in tr_loader:
            if use_cached:
                out = _cached_forward(model, batch['features'].to(device))
            else:
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
                    'train/loss':      float(losses['loss_total']),
                    'train/loss_hm':   float(losses['loss_hm']),
                    'train/loss_off':  float(losses['loss_off']),
                    'train/loss_flux': float(losses['loss_flux']),
                    'train/n_sources': float(losses['n_sources']),
                }, step=step)

        scheduler.step()
        mean_tr = float(np.mean(tr_losses)) if tr_losses else float('nan')

        # Validate
        model.eval()
        val_losses = []
        prev_augment = None
        if not use_cached:
            prev_augment = full_ds._base.augment
            full_ds._base.augment = False
        with torch.no_grad():
            try:
                for batch in val_loader:
                    if use_cached:
                        out = _cached_forward(model, batch['features'].to(device))
                    else:
                        images = {b: v.to(device) for b, v in batch['images'].items()}
                        rms = {b: v.to(device) for b, v in batch['rms'].items()}
                        out = model(images, rms)
                    losses = criterion(
                        out,
                        [c.to(device) for c in batch['centroids']],
                    )
                    val_losses.append(float(losses['loss_total']))
            finally:
                if prev_augment is not None:
                    full_ds._base.augment = prev_augment
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
                    with torch.no_grad():
                        if use_cached:
                            sout = _cached_forward(model, viz_sample['features'].to(device))
                        else:
                            sim = {b: v.to(device) for b, v in viz_sample['images'].items()}
                            srm = {b: v.to(device) for b, v in viz_sample['rms'].items()}
                            sout = model(sim, srm)
                    log['viz/tile'] = _log_tile(
                        viz_sample, sout, wandb, step, euclid_dir=args.euclid_dir)
                except Exception as exc:
                    print(f'  [warn] viz failed: {exc}')
            wandb.log(log, step=step)

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
    p.add_argument('--use_euclid_bands', action='store_true',
                   help='Load Euclid bands into encoder (slow: VIS is 1050x1050). '
                        'Default: Rubin-only for speed.')
    p.add_argument('--encoder_ckpt',     default=None)
    p.add_argument('--feature_dir',     default=None,
                   help='Precomputed feature directory (from precompute_features.py). '
                        'If provided, skips encoder and trains on cached features (fast).')
    p.add_argument('--extra_labels',    default=None,
                   help='Extra pseudo-labels .pt from self-training round')
    p.add_argument('--init_checkpoint', default=None,
                   help='Optional detector checkpoint to initialize from before training')
    p.add_argument('--out',              default='../checkpoints/centernet_v7.pt')
    p.add_argument('--epochs',           type=int,   default=100)
    p.add_argument('--batch_size',       type=int,   default=8,
                   help='Batch size (default 8 for cached features, reduce for live encoder)')
    p.add_argument('--lr',               type=float, default=1e-4)
    p.add_argument('--nsig',             type=float, default=3.0,
                   help='Detection significance for pseudo-labels')
    p.add_argument('--sigma',            type=float, default=2.0,
                   help='Gaussian sigma for heatmap targets (feature-map pixels)')
    p.add_argument('--predict_profile',  action='store_true',
                   help='Add profile head (e1, e2, r_half, sersic_n)')
    p.add_argument('--wandb_project',    default=None)
    p.add_argument('--wandb_run',        default=None)
    p.add_argument('--device',           default='',
                   help='Device to use: "cuda", "cuda:1", "cpu" (default: auto)')
    p.add_argument('--num_workers',      type=int, default=4,
                   help='DataLoader workers for training (val uses about half)')
    p.add_argument('--seed',             type=int, default=42,
                   help='Random seed for train/val split and training setup')
    args = p.parse_args()
    train(args)
