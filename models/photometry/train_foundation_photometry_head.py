#!/usr/bin/env python
"""
Train the V8 foundation photometry head with residual chi-square.

Pipeline per tile:

1. Run CenterNet on the 10-band tile to get VIS-frame detections.
2. Encode the tile once with frozen JAISP v8 foundation features.
3. Optionally apply the trained latent astrometry head to correct positions.
4. Build VIS positive morphology initializers.
5. The trainable photometry head predicts morphology refinements from V8
   bottleneck + VIS stem features.
6. PSFField renders per-band PSFs, fluxes are solved analytically, and the
   training loss is the local scene residual chi-square.

The initial supported data grid is Euclid-native VIS/Y/J/H at 0.1"/px, so VIS
morphology templates and NISP images live in the same pixel units.
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
from astropy.wcs import WCS

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
_ROOT = _MODELS.parent
for _p in (_ROOT, _MODELS, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from astrometry2.dataset import (
    detect_sources_multiband,
    discover_tile_pairs,
    local_vis_pixel_to_sky_matrix,
    safe_header_from_card_string,
    split_tile_pairs,
)
from astrometry2.source_matching import detect_sources
from astrometry2.train_latent_position import load_tile_data
from astrometry2.latent_position_head import FrozenEncoder, load_latent_position_head
from detection.centernet_detector import CenterNetDetector
from detection.detector import JAISPEncoderWrapper
from load_foundation import load_foundation
from models.photometry import PSFFieldPhotometryPipeline
from models.photometry.foundation_head import (
    FoundationScarletPhotometryHead,
    photometry_head_loss,
)
from models.photometry.scarlet_like import (
    build_neighbor_groups,
    make_positive_morphology_templates,
)


EUCLID_BANDS = ["euclid_VIS", "euclid_Y", "euclid_J", "euclid_H"]


def _to_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def load_euclid_native_arrays(euclid_path: str) -> Tuple[torch.Tensor, torch.Tensor, List[WCS]]:
    data = np.load(euclid_path, allow_pickle=True)
    imgs, rms, wcs = [], [], []
    for short in ["VIS", "Y", "J", "H"]:
        img = np.nan_to_num(_to_float32(data[f"img_{short}"]), nan=0.0)
        if f"var_{short}" in data:
            var = _to_float32(data[f"var_{short}"])
            # NaN/inf variance pixels are masked by giving them a huge finite
            # value so the flux solver and chi2 treat them as ignored weight=0
            # without propagating NaN into the model/residual/chi2.
            bad = ~np.isfinite(var)
            var = np.where(bad, 1e20, var).astype(np.float32)
        else:
            var = np.ones_like(img)
        imgs.append(img)
        rms.append(np.sqrt(np.clip(var, 1e-8, 1e20)).astype(np.float32))
        wcs.append(WCS(safe_header_from_card_string(data[f"wcs_{short}"].item())))
    return (
        torch.from_numpy(np.stack(imgs).astype(np.float32)),
        torch.from_numpy(np.stack(rms).astype(np.float32)),
        wcs,
    )


class EmpiricalPSFOverride:
    """Substitute per-band stacked empirical PSFs for specific bands.

    Delegates to the underlying PSFFieldPhotometryPipeline for any band not in
    ``band_stacks`` and replaces overridden bands with a static, unit-sum stamp
    broadcasted over all sources. Used as a PSFField-width sanity test.
    """

    def __init__(self, pipe, band_stacks: Dict[str, object]) -> None:
        self.pipe = pipe
        self.band_names = pipe.band_names
        self.stamp_size = pipe.stamp_size
        self.device = pipe.device
        self.overrides: Dict[int, torch.Tensor] = {}
        for name, stamp in band_stacks.items():
            if name not in self.band_names:
                continue
            band_idx = self.band_names.index(name)
            st = torch.as_tensor(stamp, dtype=torch.float32, device=self.device)
            if st.ndim != 2 or st.shape[0] != st.shape[1]:
                raise ValueError(
                    f"empirical PSF for {name} must be a square 2D tensor, got {tuple(st.shape)}"
                )
            if st.shape[0] != self.stamp_size:
                raise ValueError(
                    f"empirical PSF size {st.shape[0]} != pipeline stamp_size {self.stamp_size}"
                )
            st = st.clamp_min(0.0)
            total = float(st.sum())
            if total <= 0 or not np.isfinite(total):
                raise ValueError(f"empirical PSF for {name} has non-positive sum")
            self.overrides[band_idx] = st / total
        if not self.overrides:
            raise ValueError(
                f"none of band_stacks={list(band_stacks)} matched pipeline bands={self.band_names}"
            )
        replaced = [self.band_names[i] for i in self.overrides]
        print(f"Empirical PSF override active for bands: {replaced}")

    @torch.no_grad()
    def render_psfs(self, positions_px, tile_hw, sed_vec=None):
        psfs = self.pipe.render_psfs(positions_px, tile_hw=tile_hw, sed_vec=sed_vec)
        for band_idx, stamp in self.overrides.items():
            psfs[:, band_idx] = stamp.unsqueeze(0).expand(psfs.shape[0], -1, -1)
        return psfs


def load_cached_bottleneck(
    cache_dir: str,
    tile_id: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return the precomputed V8 bottleneck [1, C, H, W] or None if missing.

    Consumes the no-augment variant produced by
    ``detection/precompute_features.py`` (``{tile_id}_aug0.pt``).
    """
    if not cache_dir:
        return None
    path = Path(cache_dir) / f"{tile_id}_aug0.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    feat = payload["features"]
    if feat.dim() == 3:
        feat = feat.unsqueeze(0)
    return feat.to(device=device, dtype=torch.float32)


@torch.no_grad()
def encode_tile_with_cache(
    frozen_encoder,
    context_images: Dict[str, torch.Tensor],
    context_rms: Dict[str, torch.Tensor],
    cached_bottleneck: Optional[torch.Tensor],
) -> Dict[str, object]:
    """Build encoder outputs, reusing a cached bottleneck when provided.

    Cache hit skips the full ConvNeXt trunk — only the VIS stem is rerun,
    which is the minimum the morphology head needs at native resolution.
    """
    vis_img = context_images["euclid_VIS"]
    vis_rms = context_rms["euclid_VIS"]
    vis_stem = frozen_encoder.vis_stem(vis_img, vis_rms)
    vis_hw = (vis_img.shape[-2], vis_img.shape[-1])
    if cached_bottleneck is not None:
        bottleneck = cached_bottleneck
    else:
        bottleneck = frozen_encoder.encoder(context_images, context_rms)["bottleneck"]
    fused_hw = (bottleneck.shape[-2], bottleneck.shape[-1])
    return {
        "bottleneck": bottleneck,
        "vis_stem": vis_stem,
        "fused_hw": fused_hw,
        "vis_hw": vis_hw,
    }


def load_centernet_detector(
    foundation_checkpoint: str,
    detector_checkpoint: str,
    device: torch.device,
) -> Optional[CenterNetDetector]:
    if not detector_checkpoint:
        return None
    if not Path(detector_checkpoint).exists():
        raise FileNotFoundError(detector_checkpoint)
    foundation = load_foundation(
        foundation_checkpoint,
        device=torch.device("cpu"),
        freeze=True,
    )
    encoder = JAISPEncoderWrapper(foundation, freeze=True)
    detector = CenterNetDetector.load(detector_checkpoint, encoder, device=device).eval()
    print(f"Loaded CenterNet detector: {detector_checkpoint}")
    return detector


def load_encoder_and_astrometry(
    foundation_checkpoint: str,
    astrometry_checkpoint: str,
    device: torch.device,
):
    if astrometry_checkpoint:
        frozen_encoder, astro_head = load_latent_position_head(
            foundation_checkpoint,
            device=device,
        )
        ckpt = torch.load(astrometry_checkpoint, map_location="cpu", weights_only=False)
        astro_head.load_state_dict(ckpt["head_state_dict"])
        astro_head.eval()
        for p in astro_head.parameters():
            p.requires_grad = False
        print(f"Loaded astrometry head: {astrometry_checkpoint}")
        return frozen_encoder, astro_head

    foundation = load_foundation(
        foundation_checkpoint,
        device=torch.device("cpu"),
        freeze=True,
    )
    return FrozenEncoder(foundation).to(device), None


def detect_vis_positions(
    tile_id: str,
    rubin_path: str,
    euclid_path: str,
    detector,
    device: torch.device,
    conf_threshold: float,
    max_sources: int,
    margin: int,
    use_classical: bool = False,
) -> np.ndarray:
    rdata = np.load(rubin_path, allow_pickle=True)
    edata = np.load(euclid_path, allow_pickle=True)
    vis = np.nan_to_num(_to_float32(edata["img_VIS"]), nan=0.0)

    if detector is not None and not use_classical:
        x, y = detect_sources_multiband(
            edata,
            rdata["img"],
            rdata["var"] if "var" in rdata else None,
            detector,
            device,
            conf_threshold=conf_threshold,
        )
    else:
        x, y = detect_sources(
            vis,
            nsig=4.0,
            smooth_sigma=1.2,
            min_dist=9,
            max_sources=max_sources * 4,
        )

    xy = np.stack([x, y], axis=1).astype(np.float32) if len(x) else np.zeros((0, 2), dtype=np.float32)
    if len(xy) == 0:
        return xy

    h, w = vis.shape
    keep = (
        np.isfinite(xy[:, 0])
        & np.isfinite(xy[:, 1])
        & (xy[:, 0] >= margin)
        & (xy[:, 0] < w - margin)
        & (xy[:, 1] >= margin)
        & (xy[:, 1] < h - margin)
    )
    xy = xy[keep]
    if len(xy) == 0:
        return xy
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    order = np.argsort(vis[yi, xi])[::-1][:max_sources]
    xy = xy[order].astype(np.float32)
    print(f"{tile_id}: detections kept={len(xy)}")
    return xy


def apply_astrometry_correction(
    astro_head,
    enc_out: Dict[str, torch.Tensor],
    positions_vis: np.ndarray,
    vis_wcs: WCS,
    device: torch.device,
) -> torch.Tensor:
    pos = torch.from_numpy(positions_vis.astype(np.float32)).to(device)
    if astro_head is None or pos.numel() == 0:
        return pos
    pix2sky = np.zeros((pos.shape[0], 2, 2), dtype=np.float32)
    for i, xy in enumerate(positions_vis):
        pix2sky[i] = local_vis_pixel_to_sky_matrix(vis_wcs, xy)
    with torch.no_grad():
        out = astro_head(
            enc_out["bottleneck"],
            enc_out["vis_stem"],
            pos,
            torch.from_numpy(pix2sky).to(device),
            enc_out["fused_hw"],
            enc_out["vis_hw"],
        )
    corrected = pos + torch.stack([out["dx_px"], out["dy_px"]], dim=1)
    return corrected.detach()


def project_vis_positions_to_euclid_bands(
    corrected_vis: torch.Tensor,
    vis_wcs: WCS,
    band_wcs: Sequence[WCS],
    device: torch.device,
) -> torch.Tensor:
    xy = corrected_vis.detach().cpu().numpy().astype(np.float32)
    ra, dec = vis_wcs.wcs_pix2world(xy[:, 0], xy[:, 1], 0)
    per_band = []
    for wcs in band_wcs:
        bx, by = wcs.wcs_world2pix(ra, dec, 0)
        per_band.append(np.stack([bx, by], axis=1).astype(np.float32))
    return torch.from_numpy(np.stack(per_band, axis=1)).to(device)


def filter_positions_in_all_bands(
    corrected_vis: torch.Tensor,
    positions_by_band: torch.Tensor,
    tile_hw: Tuple[int, int],
    margin: int,
) -> torch.Tensor:
    h, w = tile_hw
    pos = positions_by_band
    keep = (
        torch.isfinite(pos).all(dim=(1, 2))
        & (pos[..., 0] >= margin).all(dim=1)
        & (pos[..., 0] < w - margin).all(dim=1)
        & (pos[..., 1] >= margin).all(dim=1)
        & (pos[..., 1] < h - margin).all(dim=1)
        & torch.isfinite(corrected_vis).all(dim=1)
    )
    return keep


def make_head_from_config(config: Dict[str, object]) -> FoundationScarletPhotometryHead:
    fused = float(config.get("fused_pixel_scale_arcsec", 0.4))
    raw = round(4.0 / fused)
    bottleneck_window = raw if raw % 2 == 1 else raw + 1
    bottleneck_window = max(5, bottleneck_window)
    return FoundationScarletPhotometryHead(
        hidden_ch=int(config.get("hidden_ch", 256)),
        stem_ch=int(config.get("stem_ch", 64)),
        morph_size=int(config.get("morph_size", 31)),
        bottleneck_window=int(config.get("bottleneck_window", bottleneck_window)),
        stem_window=int(config.get("stem_window", 31)),
        fused_pixel_scale=fused,
        delta_scale=float(config.get("delta_scale", 1.5)),
    )


def make_residual_gallery_figure(
    scene_results,
    band_idx: int = 0,
    band_name: str = "euclid_VIS",
    title: str = "",
    mode: str = "good",
    max_scenes: int = 4,
):
    """Create a data/model/residual gallery for local files or W&B."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    scenes = list(scene_results or [])
    if not scenes:
        return None
    scenes = sorted(
        scenes,
        key=lambda s: float(s.chi2_dof[band_idx]),
        reverse=(mode == "worst"),
    )[:max_scenes]

    fig, axes = plt.subplots(len(scenes), 3, figsize=(8.0, 2.45 * len(scenes)))
    if len(scenes) == 1:
        axes = axes[None, :]

    def _pct(arr: np.ndarray, q: float, fallback: float = 1.0) -> float:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return fallback
        v = float(np.nanpercentile(finite, q))
        return v if np.isfinite(v) and v > 0 else fallback

    for row, scene in enumerate(scenes):
        data = scene.data[band_idx].numpy()
        bg = float(scene.background[band_idx])
        model = np.nan_to_num(scene.model[band_idx].numpy(), nan=bg)
        resid = np.nan_to_num(scene.resid[band_idx].numpy(), nan=0.0)
        data_sub = data - bg
        model_sub = model - bg
        vmax = _pct(np.abs(data_sub), 99)
        mmax = _pct(model_sub, 99, fallback=vmax)
        rv = _pct(np.abs(resid), 98, fallback=vmax)
        panels = [
            (data_sub, "data-bg", "magma", -vmax, vmax),
            (model_sub, "learned model", "magma", 0.0, mmax),
            (resid, "residual", "coolwarm", -rv, rv),
        ]
        for col, (arr, label, cmap, vmin, vmax_i) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax_i)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label)
        axes[row, 0].set_ylabel(
            f"g{scene.group_id} n={len(scene.indices)}\n"
            f"chi2={float(scene.chi2_dof[band_idx]):.2f}",
            fontsize=8,
        )

    fig.suptitle(f"{title} - {band_name} ({mode})", y=1.01)
    fig.tight_layout()
    return fig


def log_wandb_epoch(
    wandb_run,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    visual_rows: Sequence[Dict[str, object]],
    args,
    out_dir: Path,
) -> None:
    """Log scalar curves and residual galleries to an active W&B run."""
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


def run_one_tile(
    split: str,
    tile: Tuple[str, str, str],
    frozen_encoder,
    astro_head,
    phot_head: FoundationScarletPhotometryHead,
    detector,
    psf_pipe: PSFFieldPhotometryPipeline,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    args,
    return_visuals: bool = False,
) -> Optional[Dict[str, float]]:
    tile_id, rubin_path, euclid_path = tile
    train = optimizer is not None
    phot_head.train(mode=train)

    try:
        context_images, context_rms, vis_hw, vis_wcs = load_tile_data(rubin_path, euclid_path, device)
        cached_bn = load_cached_bottleneck(args.features_cache_dir, tile_id, device)
        enc_out = encode_tile_with_cache(frozen_encoder, context_images, context_rms, cached_bn)
        euclid_tile_cpu, euclid_rms_cpu, band_wcs = load_euclid_native_arrays(euclid_path)
    except Exception as exc:
        print(f"{split} skip {tile_id}: load/encode failed: {exc}")
        return None

    det_xy = detect_vis_positions(
        tile_id,
        rubin_path,
        euclid_path,
        detector,
        device,
        conf_threshold=args.detector_conf,
        max_sources=args.max_sources,
        margin=args.margin,
        use_classical=args.classical_detection,
    )
    if det_xy.shape[0] < args.min_sources:
        return None

    corrected_vis = apply_astrometry_correction(
        astro_head,
        enc_out,
        det_xy,
        vis_wcs,
        device,
    )
    positions_by_band = project_vis_positions_to_euclid_bands(
        corrected_vis,
        vis_wcs,
        band_wcs,
        device,
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
    morph_pack = make_positive_morphology_templates(
        euclid_tile[0],
        corrected_vis,
        stamp_size=args.morph_size,
        bg_inner_radius=11.0,
        bg_outer_radius=15.0,
        smooth_sigma=args.morph_smooth,
    )
    init_morph = morph_pack["templates"]
    groups = build_neighbor_groups(
        corrected_vis.detach().cpu(),
        radius_px=args.group_radius,
    )

    with torch.no_grad():
        psfs = psf_pipe.render_psfs(
            positions_by_band,
            tile_hw=euclid_tile.shape[-2:],
        )

    if train:
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(train):
        out = photometry_head_loss(
            phot_head,
            bottleneck=enc_out["bottleneck"],
            vis_stem=enc_out["vis_stem"],
            source_positions_vis=corrected_vis,
            init_morphology=init_morph,
            tile=euclid_tile,
            rms=euclid_rms,
            positions_px=positions_by_band,
            psfs=psfs,
            fused_hw=enc_out["fused_hw"],
            vis_hw=enc_out["vis_hw"],
            groups=groups,
            min_scene_size=args.min_scene_size,
            max_scene_size=args.max_scene_size,
            tv_weight=args.tv_weight,
            anchor_weight=args.anchor_weight,
            return_scenes=return_visuals,
        )
        if train:
            out["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(phot_head.parameters(), args.grad_clip)
            optimizer.step()

    chi2 = out["chi2_dof"].detach()
    result = {
        "loss": float(out["loss"].detach().cpu()),
        "loss_data": float(out["loss_data"].detach().cpu()),
        "loss_reg": float(out["loss_reg"].detach().cpu()),
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


def aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"tiles": 0.0}
    keys = [
        k for k in rows[0].keys()
        if not k.startswith("_") and isinstance(rows[0][k], (int, float, np.floating))
    ]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["tiles"] = float(len(rows))
    return out


def save_checkpoint(
    path: Path,
    head: FoundationScarletPhotometryHead,
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
    parser.add_argument("--psf-checkpoint", default="models/checkpoints/psf_field_v3.pt")
    parser.add_argument(
        "--features-cache-dir",
        default="",
        help="Directory of precomputed V8 bottleneck features "
             "(e.g. data/cached_features_v8_fine). When set, the ConvNeXt trunk "
             "is skipped; only the VIS stem is rerun per tile.",
    )
    parser.add_argument(
        "--empirical-psf-path",
        default="",
        help="Path to a .pt file containing {'band_stacks': {band_name: HxW tensor}} "
             "to override PSFField renders for those bands. Used as a sanity test "
             "when PSFField width is suspected of dominating residuals.",
    )
    parser.add_argument("--output-dir", default="models/checkpoints/photometry_foundation_v1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-tiles", type=int, default=0, help="Limit total train+val tiles for quick experiments.")
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
    parser.add_argument("--morph-size", type=int, default=31)
    parser.add_argument("--morph-smooth", type=float, default=0.0)
    parser.add_argument("--group-radius", type=float, default=15.0)
    parser.add_argument("--min-scene-size", type=int, default=49)
    parser.add_argument("--max-scene-size", type=int, default=91)
    parser.add_argument("--margin", type=int, default=48)
    parser.add_argument("--tv-weight", type=float, default=5e-5)
    parser.add_argument("--anchor-weight", type=float, default=1e-2)
    parser.add_argument("--sub-grid", type=int, default=2)
    parser.add_argument("--wandb-project", default="", help="Enable W&B logging with this project name.")
    parser.add_argument("--wandb-name", default="", help="Optional W&B run name. Defaults to output-dir name.")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-images", action="store_true", help="Log validation residual galleries to W&B.")
    parser.add_argument("--wandb-image-every", type=int, default=1, help="Log galleries every N epochs.")
    parser.add_argument("--wandb-image-tiles", type=int, default=1, help="Number of validation tiles to visualize.")
    parser.add_argument("--wandb-image-band", type=int, default=0, help="Band index for galleries: 0=VIS, 1=Y, 2=J, 3=H.")
    parser.add_argument("--wandb-max-scenes", type=int, default=4, help="Scenes per good/worst gallery.")
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
        args.foundation_checkpoint,
        args.astrometry_checkpoint,
        device,
    )
    detector = load_centernet_detector(
        args.foundation_checkpoint,
        args.detector_checkpoint,
        device,
    )

    fckpt = torch.load(args.foundation_checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(fckpt.get("config", {}))
    cfg.update(
        {
            "morph_size": args.morph_size,
            "stem_window": args.morph_size,
            "foundation_checkpoint": args.foundation_checkpoint,
            "astrometry_checkpoint": args.astrometry_checkpoint,
            "detector_checkpoint": args.detector_checkpoint,
            "psf_checkpoint": args.psf_checkpoint,
            "band_names": EUCLID_BANDS,
        }
    )
    phot_head = make_head_from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(
        phot_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    psf_pipe = PSFFieldPhotometryPipeline.from_checkpoint(
        args.psf_checkpoint,
        band_names=EUCLID_BANDS,
        stamp_size=args.morph_size,
        sub_grid=args.sub_grid,
        bg_inner_radius=10.0,
        bg_outer_radius=14.0,
        device=device,
    )
    if args.empirical_psf_path:
        payload = torch.load(args.empirical_psf_path, map_location="cpu", weights_only=False)
        psf_pipe = EmpiricalPSFOverride(psf_pipe, payload["band_stacks"])

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
            os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_dir / "cache"))
            os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_dir / "config"))
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or out_dir.name,
                config=config,
                mode=args.wandb_mode,
                dir=str(wandb_dir),
            )
            print(f"W&B logging enabled: project={args.wandb_project} name={wandb_run.name}")
        except ImportError:
            print("W&B requested but wandb is not installed; continuing without W&B logging.")
        except Exception as exc:
            print(f"W&B requested but failed to initialize ({exc}); continuing without W&B logging.")
            wandb_run = None

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        train_order = list(rng.permutation(len(train_pairs)))
        train_rows = []
        for idx in train_order:
            row = run_one_tile(
                "train",
                train_pairs[int(idx)],
                frozen_encoder,
                astro_head,
                phot_head,
                detector,
                psf_pipe,
                optimizer,
                device,
                args,
            )
            if row is not None:
                train_rows.append(row)

        val_rows = []
        with torch.no_grad():
            for val_i, tile in enumerate(val_pairs):
                row = run_one_tile(
                    "val",
                    tile,
                    frozen_encoder,
                    astro_head,
                    phot_head,
                    detector,
                    psf_pipe,
                    None,
                    device,
                    args,
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
            wandb_run,
            epoch,
            train_metrics,
            val_metrics,
            [r for r in val_rows if "_visuals" in r],
            args,
            out_dir,
        )

        save_checkpoint(
            out_dir / "checkpoint_latest.pt",
            phot_head,
            optimizer,
            epoch,
            config,
            train_metrics,
            val_metrics,
        )
        val_loss = val_metrics.get("loss_data", train_metrics.get("loss_data", math.inf))
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                out_dir / "checkpoint_best.pt",
                phot_head,
                optimizer,
                epoch,
                config,
                train_metrics,
                val_metrics,
            )
            print(f"  saved best: {out_dir / 'checkpoint_best.pt'}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
