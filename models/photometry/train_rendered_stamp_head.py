#!/usr/bin/env python
"""
Train the end-to-end RenderedStampHead.

MVP goal: given detected sources, produce per-band per-source rendered stamps
such that the scene residual chi-square vanishes on normal (non-blended)
galaxies. No explicit PSF, no morphology template, no convolution step.

Per tile:
  1. Load frozen V8 bottleneck (from cache) + recompute VIS stem.
  2. Detect VIS-frame source positions via CenterNet, refine with astrometry.
  3. Project the single VIS position to all bands via WCS.
  4. RenderedStampHead produces a unit-sum [N, B, S, S] stamp.
  5. Fluxes solved analytically per scene; chi-square is the only loss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
_ROOT = _MODELS.parent
for _p in (_ROOT, _MODELS, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from astrometry2.dataset import discover_tile_pairs, split_tile_pairs
from astrometry2.train_latent_position import load_tile_data
from models.photometry.rendered_stamp_head import (
    RenderedStampHead,
    rendered_head_loss,
)
from models.photometry.scarlet_like import build_neighbor_groups
from models.photometry.train_foundation_photometry_head import (
    EUCLID_BANDS,
    aggregate_metrics,
    apply_astrometry_correction,
    detect_vis_positions,
    encode_tile_with_cache,
    filter_positions_in_all_bands,
    load_cached_bottleneck,
    load_centernet_detector,
    load_encoder_and_astrometry,
    load_euclid_native_arrays,
    make_residual_gallery_figure,
    project_vis_positions_to_euclid_bands,
)


def make_head_from_config(cfg: Dict[str, object], stamp_size: int) -> RenderedStampHead:
    fused = float(cfg.get("fused_pixel_scale_arcsec", 0.4))
    raw_bn = round(4.0 / fused)
    bn_window = raw_bn if raw_bn % 2 == 1 else raw_bn + 1
    bn_window = max(5, bn_window)
    return RenderedStampHead(
        n_bands=len(EUCLID_BANDS),
        hidden_ch=int(cfg.get("hidden_ch", 256)),
        stem_ch=int(cfg.get("stem_ch", 64)),
        stamp_size=stamp_size,
        stem_window=stamp_size,
        bottleneck_window=int(cfg.get("bottleneck_window", bn_window)),
        fused_pixel_scale=fused,
    )


def run_one_tile(
    split: str,
    tile: Tuple[str, str, str],
    frozen_encoder,
    astro_head,
    head: RenderedStampHead,
    detector,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    args,
    return_visuals: bool = False,
) -> Optional[Dict[str, object]]:
    tile_id, rubin_path, euclid_path = tile
    train = optimizer is not None
    head.train(mode=train)

    try:
        context_images, context_rms, vis_hw, vis_wcs = load_tile_data(rubin_path, euclid_path, device)
        cached_bn = load_cached_bottleneck(args.features_cache_dir, tile_id, device)
        enc_out = encode_tile_with_cache(frozen_encoder, context_images, context_rms, cached_bn)
        euclid_tile_cpu, euclid_rms_cpu, band_wcs = load_euclid_native_arrays(euclid_path)
    except Exception as exc:
        print(f"{split} skip {tile_id}: load/encode failed: {exc}")
        return None

    det_xy = detect_vis_positions(
        tile_id, rubin_path, euclid_path, detector, device,
        conf_threshold=args.detector_conf,
        max_sources=args.max_sources,
        margin=args.margin,
        use_classical=args.classical_detection,
    )
    if det_xy.shape[0] < args.min_sources:
        return None

    corrected_vis = apply_astrometry_correction(astro_head, enc_out, det_xy, vis_wcs, device)
    positions_by_band = project_vis_positions_to_euclid_bands(
        corrected_vis, vis_wcs, band_wcs, device,
    )
    keep = filter_positions_in_all_bands(
        corrected_vis,
        positions_by_band,
        tile_hw=tuple(euclid_tile_cpu.shape[-2:]),
        margin=args.margin,
    )
    if int(keep.sum()) < args.min_sources:
        return None
    corrected_vis = corrected_vis[keep]
    positions_by_band = positions_by_band[keep]

    if corrected_vis.shape[0] > args.max_sources_per_step:
        corrected_vis = corrected_vis[: args.max_sources_per_step]
        positions_by_band = positions_by_band[: args.max_sources_per_step]

    euclid_tile = euclid_tile_cpu.to(device)
    euclid_rms = euclid_rms_cpu.to(device)
    groups = build_neighbor_groups(
        corrected_vis.detach().cpu(),
        radius_px=args.group_radius,
    )

    if train:
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(train):
        out = rendered_head_loss(
            head,
            bottleneck=enc_out["bottleneck"],
            vis_stem=enc_out["vis_stem"],
            source_positions_vis=corrected_vis,
            positions_per_band=positions_by_band,
            tile=euclid_tile,
            rms=euclid_rms,
            fused_hw=enc_out["fused_hw"],
            vis_hw=enc_out["vis_hw"],
            groups=groups,
            min_scene_size=args.min_scene_size,
            max_scene_size=args.max_scene_size,
            return_scenes=return_visuals,
        )
        if train:
            out["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()

    chi2 = out["chi2_dof"].detach()
    result = {
        "loss": float(out["loss"].detach().cpu()),
        "chi2_vis": float(torch.nanmedian(chi2[:, 0]).detach().cpu()),
        "chi2_y": float(torch.nanmedian(chi2[:, 1]).detach().cpu()),
        "chi2_j": float(torch.nanmedian(chi2[:, 2]).detach().cpu()),
        "chi2_h": float(torch.nanmedian(chi2[:, 3]).detach().cpu()),
        "sources": int(corrected_vis.shape[0]),
        "groups": int(len(groups)),
    }
    if return_visuals:
        result["_visuals"] = {
            "tile_id": tile_id,
            "scene_results": out.get("scene_results", []),
        }
    return result


def log_wandb_epoch(
    wandb_run,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    visual_rows: Sequence[Dict[str, object]],
    args,
    out_dir: Path,
) -> None:
    if wandb_run is None:
        return
    import wandb
    import matplotlib.pyplot as plt

    payload = {f"train/{k}": v for k, v in train_metrics.items()}
    payload.update({f"val/{k}": v for k, v in val_metrics.items()})
    payload["epoch"] = epoch

    if visual_rows and args.wandb_log_images and epoch % max(1, args.wandb_image_every) == 0:
        visual_dir = out_dir / "wandb_visuals"
        visual_dir.mkdir(parents=True, exist_ok=True)
        band_idx = max(0, min(int(args.wandb_image_band), len(EUCLID_BANDS) - 1))
        for row_i, row in enumerate(visual_rows[: args.wandb_image_tiles]):
            visuals = row.get("_visuals", {})
            scenes = visuals.get("scene_results")
            tile_id = visuals.get("tile_id", f"tile{row_i}")
            for mode in ("good", "worst"):
                fig = make_residual_gallery_figure(
                    scenes,
                    band_idx=band_idx,
                    band_name=EUCLID_BANDS[band_idx],
                    title=f"epoch {epoch} {tile_id}",
                    mode=mode,
                    max_scenes=args.wandb_max_scenes,
                )
                if fig is None:
                    continue
                image_path = visual_dir / f"epoch{epoch:03d}_{tile_id}_{mode}.png"
                fig.savefig(image_path, dpi=160, bbox_inches="tight")
                payload[f"val/residual_gallery_{mode}_{row_i}"] = wandb.Image(
                    str(image_path),
                    caption=f"epoch {epoch} {tile_id} {EUCLID_BANDS[band_idx]} {mode}",
                )
                plt.close(fig)

    wandb_run.log(payload, step=epoch)


def save_checkpoint(
    path: Path,
    head: RenderedStampHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Dict[str, object],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "band_names": EUCLID_BANDS,
        },
        path,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubin-dir", default="data/rubin_tiles_all")
    parser.add_argument("--euclid-dir", default="data/euclid_tiles_all")
    parser.add_argument("--foundation-checkpoint", default="models/checkpoints/jaisp_v8_fine/checkpoint_best.pt")
    parser.add_argument("--detector-checkpoint", default="checkpoints/centernet_v8_fine/centernet_best.pt")
    parser.add_argument("--astrometry-checkpoint", default="models/checkpoints/latent_position_v8_no_psf/best.pt")
    parser.add_argument("--features-cache-dir", default="data/cached_features_v8_fine")
    parser.add_argument("--output-dir", default="models/checkpoints/rendered_stamp_head_v1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-sources", type=int, default=48)
    parser.add_argument("--max-sources-per-step", type=int, default=24)
    parser.add_argument("--min-sources", type=int, default=3)
    parser.add_argument("--detector-conf", type=float, default=0.30)
    parser.add_argument("--classical-detection", action="store_true")
    parser.add_argument("--stamp-size", type=int, default=31)
    parser.add_argument("--group-radius", type=float, default=15.0)
    parser.add_argument("--min-scene-size", type=int, default=49)
    parser.add_argument("--max-scene-size", type=int, default=91)
    parser.add_argument("--margin", type=int, default=48)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-images", action="store_true")
    parser.add_argument("--wandb-image-every", type=int, default=1)
    parser.add_argument("--wandb-image-tiles", type=int, default=1)
    parser.add_argument("--wandb-image-band", type=int, default=0)
    parser.add_argument("--wandb-max-scenes", type=int, default=4)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    pairs = discover_tile_pairs(args.rubin_dir, args.euclid_dir)
    if args.max_tiles and args.max_tiles > 0:
        pairs = pairs[: args.max_tiles]
    train_pairs, val_pairs = split_tile_pairs(pairs, val_frac=args.val_frac, seed=args.seed)
    print(f"Train tiles={len(train_pairs)} val tiles={len(val_pairs)} device={device}")

    frozen_encoder, astro_head = load_encoder_and_astrometry(
        args.foundation_checkpoint, args.astrometry_checkpoint, device,
    )
    detector = load_centernet_detector(
        args.foundation_checkpoint, args.detector_checkpoint, device,
    )

    fckpt = torch.load(args.foundation_checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(fckpt.get("config", {}))
    cfg["band_names"] = EUCLID_BANDS
    cfg["foundation_checkpoint"] = args.foundation_checkpoint
    cfg["astrometry_checkpoint"] = args.astrometry_checkpoint
    cfg["detector_checkpoint"] = args.detector_checkpoint
    cfg["stamp_size"] = args.stamp_size

    head = make_head_from_config(cfg, stamp_size=args.stamp_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config.update(cfg)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled":
        try:
            import wandb
            wandb_dir = out_dir / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("WANDB_DIR", str(wandb_dir))
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or out_dir.name,
                config=config,
                mode=args.wandb_mode,
                dir=str(wandb_dir),
            )
            print(f"W&B logging enabled: {wandb_run.name}")
        except Exception as exc:
            print(f"W&B init failed ({exc}); continuing without logging.")
            wandb_run = None

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        train_order = list(rng.permutation(len(train_pairs)))
        train_rows = []
        for idx in train_order:
            row = run_one_tile(
                "train", train_pairs[int(idx)],
                frozen_encoder, astro_head, head, detector, optimizer, device, args,
            )
            if row is not None:
                train_rows.append(row)

        val_rows = []
        with torch.no_grad():
            for val_i, tile in enumerate(val_pairs):
                row = run_one_tile(
                    "val", tile,
                    frozen_encoder, astro_head, head, detector, None, device, args,
                    return_visuals=(
                        wandb_run is not None
                        and args.wandb_log_images
                        and epoch % max(1, args.wandb_image_every) == 0
                        and val_i < args.wandb_image_tiles
                    ),
                )
                if row is not None:
                    val_rows.append(row)

        train_metrics = aggregate_metrics(train_rows)
        val_metrics = aggregate_metrics(val_rows)
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("  train:", train_metrics)
        print("  val:  ", val_metrics)

        log_wandb_epoch(
            wandb_run, epoch, train_metrics, val_metrics,
            [r for r in val_rows if "_visuals" in r], args, out_dir,
        )

        save_checkpoint(
            out_dir / "checkpoint_latest.pt",
            head, optimizer, epoch, config, train_metrics, val_metrics,
        )
        val_loss = val_metrics.get("loss", train_metrics.get("loss", math.inf))
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                out_dir / "checkpoint_best.pt",
                head, optimizer, epoch, config, train_metrics, val_metrics,
            )
            print(f"  saved best: {out_dir / 'checkpoint_best.pt'}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
