"""Train the astrometry matcher with the V7 mixed-resolution MAE backbone.

Thin wrapper around train_local_matcher.py (identical to train_astro_v6.py
but loads V7 stems instead of V6).

Extra CLI args vs train_local_matcher.py:
  --v7-checkpoint     path to the V7 foundation checkpoint_best.pt  [REQUIRED]
  --adapter-blocks    trainable ConvNeXt adapter blocks (default: 2)
  --unfreeze-stems    fine-tune V7 stems (default: frozen)
  --detector-checkpoint  path to centernet_best.pt for neural source detection
                         (omit to fall back to classical peak-finding)

Usage:
    python train_astro_v7.py \
        --v7-checkpoint ../../models/checkpoints/jaisp_v7_concat/checkpoint_best.pt \
        --rubin-dir     ../../data/rubin_tiles_ecdfs \
        --euclid-dir    ../../data/euclid_tiles_ecdfs \
        --multiband \
        --output-dir    ../checkpoints/astro_v7 \
        --wandb-project JAISP-Astrometry-v7

    # With CenterNet source detection:
    python train_astro_v7.py \
        --v7-checkpoint       ../../models/checkpoints/jaisp_v7_concat/checkpoint_best.pt \
        --detector-checkpoint ../checkpoints/centernet_v7_rms_aware/centernet_best.pt \
        --rubin-dir      ../../data/rubin_tiles_ecdfs \
        --euclid-dir     ../../data/euclid_tiles_ecdfs \
        --multiband \
        --output-dir     ../checkpoints/astro_v7_centernet \
        --wandb-project  JAISP-Astrometry-v7
"""

import sys
import time
import argparse
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _SCRIPT_DIR.parent
for _p in (_MODELS_DIR, _SCRIPT_DIR):
    _sp = str(_p)
    if _sp in sys.path:
        sys.path.remove(_sp)
    sys.path.insert(0, _sp)

from astrometry2.train_local_matcher import (
    apply_band_overrides,
    build_parser,
    compute_loss,
    compute_metrics,
    make_preview,
    make_field_preview,
    run_epoch,
)
from astrometry2.dataset import (
    ALL_BAND_ORDER,
    MatchedPatchDataset,
    build_patch_samples,
    build_patch_samples_multiband,
    discover_tile_pairs,
    load_v7_stems,
    make_loader,
    normalize_rubin_band,
    normalize_rubin_bands,
    split_tile_pairs,
)
from astrometry2.matcher_v7 import load_v7_matcher

try:
    import wandb
except ImportError:
    wandb = None


# ============================================================
# DETR source detection loader
# ============================================================

def _load_detector(detector_checkpoint: str, v7_checkpoint: str, device: torch.device):
    """Load the CenterNet detector for source finding (optional)."""
    from detection.centernet_detector import CenterNetDetector
    from detection.detector import JAISPEncoderWrapper
    from load_foundation import load_foundation

    model = load_foundation(v7_checkpoint, device=torch.device('cpu'))
    encoder = JAISPEncoderWrapper(model, freeze=True)
    detector = CenterNetDetector.load(detector_checkpoint, encoder, device=device)
    detector.eval()
    print(f'  CenterNet detector loaded from {detector_checkpoint}')
    return detector


# ============================================================
# Arg parser
# ============================================================

def build_v7_parser() -> argparse.ArgumentParser:
    p = build_parser()
    p.description = 'Train the astrometry matcher with the V7 foundation backbone.'

    g = p.add_argument_group('v7 backbone')
    g.add_argument('--v7-checkpoint', type=str, required=True,
                   help='Path to V7 foundation checkpoint_best.pt (RMS-aware recommended).')
    g.add_argument('--adapter-blocks', type=int, default=2,
                   help='Trainable ConvNeXt adapter blocks (default: 2)')
    g.add_argument('--unfreeze-stems', action='store_true',
                   help='Unfreeze V7 BandStem weights')
    g.add_argument('--detector-checkpoint', type=str, default=None,
                   help='Path to centernet_best.pt for neural source detection '
                        '(omit for classical peak-finding)')
    g.add_argument('--detector-conf-threshold', type=float, default=0.3,
                   help='CenterNet confidence threshold for source detection (default: 0.3)')
    g.add_argument('--stream-stages', type=int, default=0,
                   help='Number of frozen V7 stream encoder stages to use after the stem. '
                        '0 = stem only (V6-equivalent). 1 = adds one ConvNeXt downsample '
                        'stage per stream, giving richer features at half spatial resolution. '
                        '(default: 0)')
    g.add_argument('--resume', action='store_true',
                   help='Resume training from checkpoint_latest.pt in the output directory.')
    g.add_argument('--encoder-centroids', action='store_true',
                   help='Enable encoder-based centroiding using V7 BandStem feature peaks. '
                        'Experimental — off by default because the BandStem was trained for '
                        'reconstruction, not localization. PSF-fit centroiding is more reliable '
                        'for label generation. The proper data-driven path is self-training.')
    g.add_argument('--no-dp', action='store_true',
                   help='Disable DataParallel even when multiple GPUs are available.')

    p.set_defaults(
        hidden_channels=64,
        wandb_project='JAISP-Astrometry-v7',
        output_dir='../checkpoints/astro_v7',
    )
    return p


# ============================================================
# Main
# ============================================================

def train(args):
    args = apply_band_overrides(args)
    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Optional DETR detector ------------------------------------------
    detr_detector = None
    if args.detector_checkpoint:
        detr_detector = _load_detector(
            args.detector_checkpoint, args.v7_checkpoint, device)
    # Keep the preview path on the exact same detector as dataset building.
    args._detr_detector = detr_detector

    # ---- V7 BandStems for encoder-based centroiding -----------------------
    # NOTE: encoder-based centroiding via BandStem feature-energy peaks is
    # available but OFF by default.  The BandStem was trained for reconstruction,
    # not localization, so its energy peaks don't reliably coincide with source
    # centroids.  PSF-fit centroiding gives better labels for now.
    # The proper data-driven path is self-training: train the matcher on PSF-fit
    # labels, then use the matcher's own predictions (via cost volume) as refined
    # labels for retraining.  Use --encoder-centroids to enable.
    if getattr(args, 'encoder_centroids', False):
        print(f'Loading V7 BandStems for encoder-based centroiding...')
        v7_stems = load_v7_stems(args.v7_checkpoint, device=device)
        stems_device = device
        print(f'  Loaded {len(v7_stems)} band stems: {sorted(v7_stems.keys())}')
    else:
        v7_stems = None
        stems_device = None

    # ---- Dataset ---------------------------------------------------------
    detect_bands = normalize_rubin_bands(args.detect_bands) or [
        f'rubin_{b}' for b in ('g', 'r', 'i', 'z')
    ]
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

    # Inject DETR detector into dataset builder if available
    if detr_detector is not None:
        detection_kwargs['detr_detector'] = detr_detector
        detection_kwargs['detr_device'] = device
        detection_kwargs['detr_conf_threshold'] = args.detector_conf_threshold

    # Inject V7 BandStems for encoder-based centroiding
    if v7_stems is not None:
        detection_kwargs['v7_stems'] = v7_stems
        detection_kwargs['stems_device'] = stems_device

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
        target_band = 'multiband'
        preview_target_bands = sorted(
            set(s['target_band'] for s in train_samples),
            key=lambda b: ALL_BAND_ORDER.index(b) if b in ALL_BAND_ORDER else len(ALL_BAND_ORDER),
        )
        if 'rubin_r' in preview_target_bands:
            preview_target_bands = ['rubin_r'] + [b for b in preview_target_bands if b != 'rubin_r']
    else:
        target_band   = normalize_rubin_band(args.rubin_band)
        context_bands = normalize_rubin_bands(args.context_bands)
        common_kwargs = dict(target_band=target_band, context_bands=context_bands, **detection_kwargs)
        train_samples = build_patch_samples(train_pairs, split_name='train', **common_kwargs)
        val_samples   = build_patch_samples(val_pairs,   split_name='val',   **common_kwargs) if val_pairs else []
        n_target_bands = 1
        preview_target_bands = [target_band]

    train_dataset = MatchedPatchDataset(
        train_samples,
        augment=True,
        jitter_arcsec=args.jitter_arcsec,
        jitter_max_arcsec=args.jitter_max_arcsec,
        jitter_prob=args.jitter_prob,
    )
    val_dataset   = MatchedPatchDataset(val_samples,   augment=False) if val_samples else None
    train_loader  = make_loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader    = make_loader(val_dataset,   batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) if val_dataset else None

    n_rubin_bands = int(train_samples[0]['rubin_patch'].shape[0])
    print(f'Dataset: {len(train_samples)} train / {len(val_samples)} val patches')
    print(f'Rubin input channels: {n_rubin_bands}  |  target bands: {n_target_bands}')

    # ---- Model (V7 backbone) ---------------------------------------------
    print(f'\nLoading V7 backbone from: {args.v7_checkpoint}')
    model = load_v7_matcher(
        v7_checkpoint    = args.v7_checkpoint,
        device           = device,
        n_rubin_bands    = n_rubin_bands,
        hidden_channels  = args.hidden_channels,
        n_adapter_blocks = args.adapter_blocks,
        freeze_stems     = not args.unfreeze_stems,
        search_radius    = args.search_radius,
        n_target_bands   = n_target_bands,
        band_embed_dim   = getattr(args, 'band_embed_dim', 16),
        mlp_hidden       = args.mlp_hidden,
        n_stream_stages  = getattr(args, 'stream_stages', 0),
    )

    # Multi-GPU via DataParallel when device is 'cuda' (use all GPUs) or
    # when explicitly requested.  The underlying model is always accessible
    # via raw_model for checkpoint saving and parameter grouping.
    raw_model = model
    if torch.cuda.device_count() > 1 and str(device).startswith('cuda') and not getattr(args, 'no_dp', False):
        gpu_ids = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(raw_model, device_ids=gpu_ids)
        print(f'DataParallel on {len(gpu_ids)} GPUs: {gpu_ids}')

    # Separate LRs: adapter/heads get full LR; frozen-origin params (stems +
    # stream stages) get 0.1× if unfrozen to avoid destabilizing pretrained weights.
    frozen_origin_params  = list(raw_model.rubin_encoder.band_stems.parameters())
    frozen_origin_params += list(raw_model.vis_encoder.vis_stem.parameters())
    frozen_origin_params += list(raw_model.rubin_encoder.stream_stages.parameters())
    frozen_origin_params += list(raw_model.vis_encoder.stream_stages.parameters())
    frozen_origin_ids = {id(p) for p in frozen_origin_params}
    other_params      = [p for p in raw_model.parameters() if id(p) not in frozen_origin_ids and p.requires_grad]
    stem_trainable    = [p for p in frozen_origin_params if p.requires_grad]

    param_groups = [{'params': other_params, 'lr': args.lr}]
    if stem_trainable:
        param_groups.append({'params': stem_trainable, 'lr': args.lr * 0.1,
                             'name': 'v7_stems'})
        print(f'Stem LR: {args.lr * 0.1:.2e}  (0.1× of base LR)')

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- Resume from checkpoint ------------------------------------------
    start_epoch = 1
    best_score = float('inf')
    resume_id = None
    resume_ckpt = out_dir / 'checkpoint_latest.pt'
    if getattr(args, 'resume', False) and resume_ckpt.exists():
        print(f'Resuming from {resume_ckpt}')
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_score = ckpt.get('val_score', float('inf'))
        resume_id = ckpt.get('wandb_id', None)
        # Advance scheduler to the correct epoch
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f'  Resumed at epoch {start_epoch}, best_score={best_score*1000:.2f} mas')

    # ---- W&B -------------------------------------------------------------
    preview_sample = val_dataset[0] if (val_dataset and len(val_dataset) > 0) else train_dataset[0]
    preview_split  = 'val'          if (val_dataset and len(val_dataset) > 0) else 'train'
    preview_pairs  = val_pairs      if val_pairs else train_pairs

    wandb_run = None
    if getattr(args, 'wandb_mode', 'online') != 'disabled' and wandb is not None:
        try:
            wandb_kwargs = dict(
                project = args.wandb_project,
                name    = getattr(args, 'wandb_run_name', None) or f'v7_adapt{args.adapter_blocks}',
                config  = vars(args),
                mode    = getattr(args, 'wandb_mode', 'online'),
                dir     = str(out_dir),
            )
            if resume_id:
                wandb_kwargs['id'] = resume_id
                wandb_kwargs['resume'] = 'must'
            wandb_run = wandb.init(**wandb_kwargs)
        except Exception as exc:
            print(f'W&B init failed: {exc}')

    # ---- Training loop ---------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        label_noise_floor = getattr(args, 'label_noise_floor', 0.005)
        train_metrics = run_epoch(
            'train', train_loader, model, optimizer, device,
            args.grad_clip, args.pixel_loss_weight,
            label_noise_floor=label_noise_floor,
        )
        scheduler.step()
        train_metrics['lr'] = optimizer.param_groups[0]['lr']

        val_metrics = {}
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    'val', val_loader, model, None, device,
                    args.grad_clip, args.pixel_loss_weight,
                    label_noise_floor=label_noise_floor,
                )

        score   = val_metrics.get('mae_total', train_metrics['mae_total'])
        elapsed = time.time() - t0

        val_str = (f"  val p68={val_metrics['p68_total']*1000:.1f}mas "
                   f"mae={val_metrics['mae_total']*1000:.1f}mas"
                   if val_metrics else '')
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train p68={train_metrics['p68_total']*1000:.1f}mas "
            f"mae={train_metrics['mae_total']*1000:.1f}mas"
            f"{val_str} | {elapsed:.1f}s"
        )

        save_meta = {
            'epoch': epoch,
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': dict(vars(args)),
            'rubin_channels': n_rubin_bands,
            'n_target_bands': n_target_bands,
            'input_bands': list(train_samples[0]['input_bands']),
            'target_band': target_band,
            'target_bands': sorted(set(s['target_band'] for s in train_samples)),
            'include_nisp': getattr(args, 'include_nisp', False),
            'wandb_id': wandb_run.id if wandb_run is not None else None,
        }
        if score < best_score:
            best_score = score
            torch.save({**save_meta, 'val_score': score},
                       out_dir / 'checkpoint_best.pt')

        torch.save(save_meta, out_dir / 'checkpoint_latest.pt')

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
                    model, device, preview_pairs,
                    preview_target_bands,
                    [f'rubin_{b}' for b in ('u', 'g', 'r', 'i', 'z', 'y')][:n_rubin_bands],
                    detect_bands, args, epoch, preview_split,
                )
                log.update(field_imgs)

            wandb_run.log(log)

    if wandb_run is not None:
        wandb_run.finish()

    print(f'\nDone. Best val score: {best_score*1000:.2f} mas')
    print(f'Checkpoint: {out_dir / "checkpoint_best.pt"}')


if __name__ == '__main__':
    args = build_v7_parser().parse_args()
    train(args)
