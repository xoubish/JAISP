"""Train the astrometry matcher with the v6 foundation model backbone.

This script is a thin wrapper around train_local_matcher.py.
It reuses ALL training infrastructure (loss, metrics, epoch loop, W&B,
field solver previews) unchanged. The only difference is model instantiation:
LocalAstrometryMatcher → V6AstrometryMatcher (frozen v6 stems + adapter).

Extra CLI args vs train_local_matcher.py:
  --v6-checkpoint     path to the v6 foundation checkpoint_best.pt  [REQUIRED]
  --adapter-blocks    number of trainable ConvNeXt adapter blocks on top of
                      the frozen stems (default: 2)
  --unfreeze-stems    if set, unfreeze the v6 stems and fine-tune them too
                      (default: stems are frozen)

Usage:
    python train_astro_v6.py \
        --v6-checkpoint ../checkpoints/jaisp_v6/checkpoint_best.pt \
        --rubin-dir     ../../data/rubin_tiles_ecdfs \
        --euclid-dir    ../../data/euclid_tiles_ecdfs \
        --multiband \
        --output-dir    ../checkpoints/astro_v6 \
        --wandb-project JAISP-Astrometry-v6

Comparison run (identical settings, no v6 backbone):
    python train_local_matcher.py \
        --rubin-dir     ../../data/rubin_tiles_ecdfs \
        --euclid-dir    ../../data/euclid_tiles_ecdfs \
        --multiband \
        --output-dir    ../checkpoints/astro_baseline \
        --wandb-project JAISP-Astrometry-v6
"""

import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch

# Ensure this directory is on sys.path for local imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR  = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _MODELS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---- Import ALL infrastructure from the existing training script -----------
from train_local_matcher import (
    build_parser,
    compute_loss,
    compute_metrics,
    make_preview,
    make_field_preview,
    run_epoch,
)
from dataset import (
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
from matcher_v6 import load_v6_matcher

try:
    import wandb
except ImportError:
    wandb = None


# ============================================================
# Build arg parser (extends the baseline parser)
# ============================================================

def build_v6_parser() -> argparse.ArgumentParser:
    p = build_parser()   # inherit all baseline arguments
    p.description = 'Train the astrometry matcher with the v6 foundation backbone.'

    # v6-specific arguments
    g = p.add_argument_group('v6 backbone')
    g.add_argument('--v6-checkpoint', type=str, required=True,
                   help='Path to jaisp_foundation_v6 checkpoint_best.pt')
    g.add_argument('--adapter-blocks', type=int, default=2,
                   help='Trainable ConvNeXt adapter blocks on top of frozen stems (default: 2)')
    g.add_argument('--unfreeze-stems', action='store_true',
                   help='Unfreeze the v6 BandStem weights (fine-tune end-to-end)')

    # Sensible defaults that differ from the baseline
    p.set_defaults(
        hidden_channels=64,        # v6 stems output 64ch; match for cost volume
        wandb_project='JAISP-Astrometry-v6',
        output_dir='../checkpoints/astro_v6',
    )
    return p


# ============================================================
# Main training function
# ============================================================

def train(args):
    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset (identical to baseline) -----------------------------------
    detect_bands = normalize_rubin_bands(args.detect_bands) or [
        f'rubin_{b}' for b in ('g', 'r', 'i', 'z')
    ]
    pairs = discover_tile_pairs(args.rubin_dir, args.euclid_dir)
    train_pairs, val_pairs = split_tile_pairs(pairs, args.val_frac, args.seed)

    detection_kwargs = dict(
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

    if getattr(args, 'multiband', False):
        multiband_kwargs = dict(
            target_bands=args.target_bands,
            include_nisp=getattr(args, 'include_nisp', False),
            **detection_kwargs,
        )
        train_samples = build_patch_samples_multiband(train_pairs, split_name='train', **multiband_kwargs)
        val_samples   = build_patch_samples_multiband(val_pairs,   split_name='val',   **multiband_kwargs) if val_pairs else []
        all_band_idxs = sorted(set(s['band_idx'] for s in train_samples))
        n_target_bands = max(all_band_idxs) + 1
        preview_target_bands = sorted(
            set(s['target_band'] for s in train_samples),
            key=lambda b: ALL_BAND_ORDER.index(b) if b in ALL_BAND_ORDER else len(ALL_BAND_ORDER),
        )
    else:
        target_band   = normalize_rubin_band(args.rubin_band)
        context_bands = normalize_rubin_bands(args.context_bands)
        common_kwargs = dict(target_band=target_band, context_bands=context_bands, **detection_kwargs)
        train_samples = build_patch_samples(train_pairs, split_name='train', **common_kwargs)
        val_samples   = build_patch_samples(val_pairs,   split_name='val',   **common_kwargs) if val_pairs else []
        n_target_bands = 1
        preview_target_bands = [target_band]

    train_dataset = MatchedPatchDataset(train_samples, augment=True)
    val_dataset   = MatchedPatchDataset(val_samples,   augment=False) if val_samples else None
    train_loader  = make_loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader    = make_loader(val_dataset,   batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) if val_dataset else None

    n_rubin_bands = int(train_samples[0]['rubin_patch'].shape[0])
    print(f'Dataset: {len(train_samples)} train / {len(val_samples)} val patches')
    print(f'Rubin input channels: {n_rubin_bands}  |  target bands: {n_target_bands}')

    # ---- Model (V6 backbone instead of LocalAstrometryMatcher) -------------
    print(f'\nLoading v6 backbone from: {args.v6_checkpoint}')
    model = load_v6_matcher(
        v6_checkpoint   = args.v6_checkpoint,
        device          = device,
        n_rubin_bands   = n_rubin_bands,
        hidden_channels = args.hidden_channels,
        n_adapter_blocks= args.adapter_blocks,
        freeze_stems    = not args.unfreeze_stems,
        search_radius   = args.search_radius,
        n_target_bands  = n_target_bands,
        band_embed_dim  = getattr(args, 'band_embed_dim', 16),
        mlp_hidden      = args.mlp_hidden,
    )

    # Separate LRs: adapter/VIS/MLP get full LR; stems (if unfrozen) get 0.1×
    stem_params     = list(model.rubin_encoder.band_stems.parameters())
    stem_param_ids  = {id(p) for p in stem_params}
    other_params    = [p for p in model.parameters() if id(p) not in stem_param_ids and p.requires_grad]
    stem_trainable  = [p for p in stem_params if p.requires_grad]

    param_groups = [{'params': other_params, 'lr': args.lr}]
    if stem_trainable:
        param_groups.append({'params': stem_trainable, 'lr': args.lr * 0.1,
                             'name': 'v6_stems'})
        print(f'Stem LR: {args.lr * 0.1:.2e}  (0.1× of base LR)')

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- W&B ---------------------------------------------------------------
    preview_sample = val_dataset[0]  if (val_dataset  and len(val_dataset)  > 0) else train_dataset[0]
    preview_split  = 'val'           if (val_dataset  and len(val_dataset)  > 0) else 'train'
    preview_pairs  = val_pairs       if val_pairs else train_pairs

    wandb_run = None
    if getattr(args, 'wandb_mode', 'online') != 'disabled' and wandb is not None:
        try:
            wandb_run = wandb.init(
                project = args.wandb_project,
                name    = getattr(args, 'wandb_run_name', None) or f'v6_adapt{args.adapter_blocks}',
                config  = vars(args),
                mode    = getattr(args, 'wandb_mode', 'online'),
                dir     = str(out_dir),
            )
        except Exception as exc:
            print(f'W&B init failed: {exc}')

    # ---- Training loop (identical to baseline) -----------------------------
    best_score = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(
            'train', train_loader, model, optimizer, device,
            args.grad_clip, args.pixel_loss_weight,
        )
        scheduler.step()
        train_metrics['lr'] = optimizer.param_groups[0]['lr']

        val_metrics = {}
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    'val', val_loader, model, None, device,
                    args.grad_clip, args.pixel_loss_weight,
                )

        score   = val_metrics.get('mae_total', train_metrics['mae_total'])
        elapsed = time.time() - t0

        # Console log
        val_str = (f"  val p68={val_metrics['p68_total']*1000:.1f}mas "
                   f"mae={val_metrics['mae_total']*1000:.1f}mas"
                   if val_metrics else '')
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train p68={train_metrics['p68_total']*1000:.1f}mas "
            f"mae={train_metrics['mae_total']*1000:.1f}mas"
            f"{val_str} | {elapsed:.1f}s"
        )

        # Save best
        if score < best_score:
            best_score = score
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'val_score': score},
                       out_dir / 'checkpoint_best.pt')

        # Save latest
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   out_dir / 'checkpoint_latest.pt')

        # W&B logging
        if wandb_run is not None:
            log = {f'train/{k}': v for k, v in train_metrics.items()}
            log.update({f'val/{k}': v for k, v in val_metrics.items()})
            log['epoch'] = epoch

            vis_every = getattr(args, 'vis_every', 1)
            if epoch % vis_every == 0 or epoch == 1:
                img = make_preview(model, preview_sample, device, epoch, preview_split)
                if img is not None:
                    log['preview/patch'] = img

                field_imgs = make_field_preview(
                    model, device, preview_pairs[:1],
                    preview_target_bands[:1],
                    [f'rubin_{b}' for b in ('u', 'g', 'r', 'i', 'z', 'y')][:n_rubin_bands],
                    detect_bands, args, epoch, preview_split,
                )
                log.update(field_imgs)

            wandb_run.log(log)

    if wandb_run is not None:
        wandb_run.finish()

    print(f'\nDone. Best val score: {best_score*1000:.2f} mas')
    print(f'Checkpoint: {out_dir / "checkpoint_best.pt"}')


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    args = build_v6_parser().parse_args()
    train(args)
