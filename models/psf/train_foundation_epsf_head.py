"""Train the foundation-conditioned low-rank ePSF head.

The trainer consumes the star stamps produced by ``build_psf_v4_training_set.py``
and the original Rubin/Euclid tile files. For each tile it:

  1. Runs the frozen 10-band JAISP foundation encoder once.
  2. Samples the tile's held star stamps across all bands.
  3. Predicts a constrained oversampled ePSF per star from frozen features,
     tile position, and band id.
  4. Renders to native pixels at the recorded sub-pixel phase.
  5. Solves flux and a local constant background analytically, then optimizes
     residual chi/Charbonnier loss plus small calibration regularizers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from load_foundation import load_foundation
    from foundation_utils import FrozenEncoder, discover_tile_pairs, load_tile_data
    from astrometry2.dataset import split_tile_pairs
    from psf.foundation_epsf_head import (
        ALL_BANDS,
        DEFAULT_CORE_SIGMA_MAS,
        RUBIN_BANDS,
        FoundationEPSFHead,
        analytic_epsf_bank,
        load_base_epsf_bank,
    )
except ImportError:
    from models.load_foundation import load_foundation
    from models.foundation_utils import FrozenEncoder, discover_tile_pairs, load_tile_data
    from models.astrometry2.dataset import split_tile_pairs
    from models.psf.foundation_epsf_head import (
        ALL_BANDS,
        DEFAULT_CORE_SIGMA_MAS,
        RUBIN_BANDS,
        FoundationEPSFHead,
        analytic_epsf_bank,
        load_base_epsf_bank,
    )

try:
    import wandb
except ImportError:
    wandb = None


Record = Mapping[str, object]


def parse_band_float_overrides(spec: Optional[str], *, name: str) -> Dict[str, float]:
    """Parse ``band=value,band=value`` CLI overrides."""
    if spec is None or not str(spec).strip():
        return {}
    out: Dict[str, float] = {}
    for raw_part in str(spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"{name} override {part!r} must look like band=value")
        band, value = [x.strip() for x in part.split("=", 1)]
        if band not in ALL_BANDS:
            raise ValueError(f"{name} override has unknown band {band!r}; valid bands: {ALL_BANDS}")
        out[band] = float(value)
    return out


def load_psf_records_by_tile(
    train_dir: Path,
    band_names: Sequence[str] = ALL_BANDS,
    min_snr: float = 0.0,
    max_snr: float = 0.0,
) -> Dict[str, List[Record]]:
    """Load v4 per-band NPZ stamps and group records by tile id."""
    by_tile: Dict[str, List[Record]] = {}
    counts: Dict[str, int] = {}
    for band_idx, band in enumerate(band_names):
        path = train_dir / f"{band}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        required = ("stamps", "rms", "frac_xy", "pos_norm", "pos_pix", "snr", "tile_id")
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"{path} is missing keys: {missing}")

        stamps = np.asarray(data["stamps"], dtype=np.float32)
        rms = np.asarray(data["rms"], dtype=np.float32)
        frac_xy = np.asarray(data["frac_xy"], dtype=np.float32)
        pos_norm = np.asarray(data["pos_norm"], dtype=np.float32)
        pos_pix = np.asarray(data["pos_pix"], dtype=np.float32)
        pos_vis_pix = (
            np.asarray(data["pos_vis_pix"], dtype=np.float32)
            if "pos_vis_pix" in data.files else None
        )
        centroid_resid_px = (
            np.asarray(data["centroid_resid_px"], dtype=np.float32)
            if "centroid_resid_px" in data.files else None
        )
        tile_ids = np.asarray(data["tile_id"])
        snr = np.asarray(data["snr"], dtype=np.float32)
        keep_mask = snr >= float(min_snr)
        if float(max_snr) > 0:
            keep_mask &= snr <= float(max_snr)
        keep = np.where(keep_mask)[0]
        counts[band] = int(len(keep))
        for k in keep:
            tile_id = str(tile_ids[k])
            by_tile.setdefault(tile_id, []).append(
                {
                    "band": band,
                    "band_idx": int(band_idx),
                    "stamp": stamps[k],
                    "rms": rms[k],
                    "frac_xy": frac_xy[k],
                    "pos_norm": pos_norm[k],
                    "pos_pix": pos_pix[k],
                    "source_positions_vis": (
                        pos_vis_pix[k] if pos_vis_pix is not None else None
                    ),
                    "centroid_resid_px": (
                        float(centroid_resid_px[k])
                        if centroid_resid_px is not None else float("nan")
                    ),
                    "snr": float(snr[k]),
                }
            )
    if not by_tile:
        raise RuntimeError(f"No PSF records found under {train_dir}")
    print(
        f"Loaded {sum(counts.values())} PSF stamps from {len(by_tile)} tiles: "
        + "  ".join(f"{b}={n}" for b, n in counts.items())
    )
    if float(max_snr) > 0:
        print(f"  SNR hard filter: {float(min_snr):g} <= snr <= {float(max_snr):g}")
    else:
        print(f"  SNR hard filter: snr >= {float(min_snr):g}")
    return by_tile


def make_record_batch(
    records: Sequence[Record],
    device: torch.device,
    rng: np.random.RandomState,
    max_stars: int = 256,
) -> Dict[str, torch.Tensor]:
    """Stack a tile's records into tensors, optionally subsampling stars."""
    if len(records) == 0:
        raise ValueError("records must not be empty")
    if max_stars > 0 and len(records) > max_stars:
        idx = rng.choice(len(records), size=max_stars, replace=False)
        records = [records[int(i)] for i in idx]

    stamp = torch.from_numpy(np.stack([r["stamp"] for r in records]).astype(np.float32)).to(device)
    rms = torch.from_numpy(np.stack([r["rms"] for r in records]).astype(np.float32)).to(device)
    frac_xy = torch.from_numpy(np.stack([r["frac_xy"] for r in records]).astype(np.float32)).to(device)
    pos_norm = torch.from_numpy(np.stack([r["pos_norm"] for r in records]).astype(np.float32)).to(device)
    pos_pix = np.stack([r["pos_pix"] for r in records]).astype(np.float32)
    band_idx_np = np.array([r["band_idx"] for r in records], dtype=np.int64)

    source_positions_vis = pos_pix.copy()
    rubin_mask = band_idx_np < len(RUBIN_BANDS)
    source_positions_vis[rubin_mask] *= 2.0
    for i, record in enumerate(records):
        if record.get("source_positions_vis") is not None:
            source_positions_vis[i] = np.asarray(
                record["source_positions_vis"],
                dtype=np.float32,
            )

    return {
        "stamp": stamp,
        "rms": rms,
        "frac_xy": frac_xy,
        "pos_norm": pos_norm,
        "pos_pix": torch.from_numpy(pos_pix).to(device),
        "pos_vis_pix": torch.from_numpy(source_positions_vis).to(device),
        "source_positions_vis": torch.from_numpy(source_positions_vis).to(device),
        "band_idx": torch.from_numpy(band_idx_np).long().to(device),
        "centroid_resid_px": torch.tensor(
            [float(r.get("centroid_resid_px", float("nan"))) for r in records],
            dtype=torch.float32,
            device=device,
        ),
        "snr": torch.tensor([float(r["snr"]) for r in records], dtype=torch.float32, device=device),
    }


def load_cached_bottleneck(
    cache_dir: Optional[Path],
    tile_id: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if cache_dir is None:
        return None
    for name in (f"{tile_id}_aug0.pt", f"{tile_id}.pt"):
        path = cache_dir / name
        if not path.exists():
            continue
        payload = torch.load(str(path), map_location=device, weights_only=False)
        feat = payload["features"] if isinstance(payload, dict) and "features" in payload else payload
        feat = feat.float().to(device)
        if feat.ndim == 3:
            feat = feat.unsqueeze(0)
        return feat
    return None


@torch.no_grad()
def encode_tile(
    frozen_encoder: FrozenEncoder,
    tile_id: str,
    rubin_path: str,
    euclid_path: str,
    device: torch.device,
    features_cache_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Load image arrays and return frozen encoder features for one tile."""
    context_images, context_rms, _vis_hw, _vis_wcs = load_tile_data(rubin_path, euclid_path, device)
    vis_img = context_images["euclid_VIS"]
    vis_rms = context_rms["euclid_VIS"]
    vis_stem = frozen_encoder.vis_stem(vis_img, vis_rms)
    cached = load_cached_bottleneck(features_cache_dir, tile_id, device)
    if cached is None:
        bottleneck = frozen_encoder.encoder(context_images, context_rms)["bottleneck"]
    else:
        bottleneck = cached
    return {
        "bottleneck": bottleneck,
        "vis_stem": vis_stem,
        "fused_hw": (bottleneck.shape[-2], bottleneck.shape[-1]),
        "vis_hw": (vis_img.shape[-2], vis_img.shape[-1]),
    }


def effective_loss_rms(
    data: torch.Tensor,
    rms: torch.Tensor,
    loss_snr_cap: float = 0.0,
) -> torch.Tensor:
    """Floor RMS in bright pixels so extreme-SNR stars cannot dominate loss.

    The floor is based on a robust background-subtracted signal estimate per
    stamp. A cap of 1000 means no individual pixel is trusted at >~1000 sigma,
    while low and moderate SNR pixels keep their measured RMS.
    """
    rms_eff = rms.clamp(min=1e-6)
    cap = float(loss_snr_cap)
    if cap <= 0:
        return rms_eff
    bg = data.flatten(1).median(dim=1).values.view(-1, 1, 1)
    signal = (data - bg).abs()
    return torch.maximum(rms_eff, signal / max(cap, 1e-6))


def effective_loss_rms_stats(rms: torch.Tensor, rms_eff: torch.Tensor) -> Dict[str, torch.Tensor]:
    ratio = rms_eff / rms.clamp(min=1e-6)
    active = ratio > 1.0001
    return {
        "loss_rms_cap_frac": active.float().mean(),
        "loss_rms_eff_ratio_median": ratio.median(),
        "loss_rms_eff_ratio_p95": torch.quantile(ratio.flatten(), 0.95),
    }


def radial_loss_weight(
    data: torch.Tensor,
    frac_xy: torch.Tensor,
    loss_radius_px: float = 0.0,
    loss_taper_px: float = 0.0,
) -> Optional[torch.Tensor]:
    """Circular loss window centred on each star, with optional cosine taper."""
    radius = float(loss_radius_px)
    if radius <= 0:
        return None
    taper = max(float(loss_taper_px), 0.0)
    n, h, w = data.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=data.device, dtype=data.dtype),
        torch.arange(w, device=data.device, dtype=data.dtype),
        indexing="ij",
    )
    cx = (w // 2) + frac_xy[:, 0].view(n, 1, 1).to(dtype=data.dtype)
    cy = (h // 2) + frac_xy[:, 1].view(n, 1, 1).to(dtype=data.dtype)
    rr = torch.hypot(xx.view(1, h, w) - cx, yy.view(1, h, w) - cy)
    if taper <= 0:
        return (rr <= radius).to(dtype=data.dtype)
    inner = max(radius - taper, 0.0)
    weight = torch.ones_like(rr)
    ramp = (radius - rr) / max(taper, 1e-6)
    weight = torch.where(rr > inner, ramp.clamp(0.0, 1.0), weight)
    return torch.where(rr <= radius, weight, torch.zeros_like(weight))


def solve_flux_background(
    model_unit: torch.Tensor,
    data: torch.Tensor,
    rms: torch.Tensor,
    pixel_weight: Optional[torch.Tensor] = None,
    fit_background: bool = True,
    nonnegative_flux: bool = True,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted least-squares solve for flux and optional constant background."""
    pred = model_unit.squeeze(1)
    weight = 1.0 / rms.clamp(min=1e-6).pow(2)
    if pixel_weight is not None:
        weight = weight * pixel_weight.to(dtype=weight.dtype)

    if not fit_background:
        num = (weight * pred * data).sum(dim=(-2, -1))
        den = (weight * pred * pred).sum(dim=(-2, -1)).clamp(min=eps)
        flux = num / den
        if nonnegative_flux:
            flux = flux.clamp_min(0.0)
        bg = torch.zeros_like(flux)
        return flux, bg

    a = (weight * pred * pred).sum(dim=(-2, -1))
    b = (weight * pred).sum(dim=(-2, -1))
    c = weight.sum(dim=(-2, -1))
    d = (weight * pred * data).sum(dim=(-2, -1))
    e = (weight * data).sum(dim=(-2, -1))
    det = (a * c - b * b).clamp(min=eps)
    flux = (d * c - b * e) / det
    bg = (a * e - b * d) / det
    if nonnegative_flux:
        flux = flux.clamp_min(0.0)
    return flux, bg


def _psf_fit_per_star(
    model_unit: torch.Tensor,
    data: torch.Tensor,
    rms: torch.Tensor,
    *,
    pixel_weight: Optional[torch.Tensor] = None,
    charbonnier_eps: float = 1e-3,
    fit_background: bool = True,
    nonnegative_flux: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flux, bg = solve_flux_background(
        model_unit,
        data,
        rms,
        pixel_weight=pixel_weight,
        fit_background=fit_background,
        nonnegative_flux=nonnegative_flux,
    )
    pred = model_unit.squeeze(1)
    model = flux.view(-1, 1, 1) * pred + bg.view(-1, 1, 1)
    z = (model - data) / rms.clamp(min=1e-6)
    loss_pix = torch.sqrt(z * z + float(charbonnier_eps) ** 2) - float(charbonnier_eps)
    if pixel_weight is not None:
        w = pixel_weight.to(dtype=loss_pix.dtype)
        loss_per_star = (loss_pix * w).sum(dim=(-2, -1)) / w.sum(dim=(-2, -1)).clamp(min=1e-6)
    else:
        loss_per_star = loss_pix.mean(dim=(-2, -1))
    return loss_per_star, flux, bg, model, z


def psf_residual_loss(
    model_unit: torch.Tensor,
    data: torch.Tensor,
    rms: torch.Tensor,
    *,
    loss_snr_cap: float = 0.0,
    pixel_weight: Optional[torch.Tensor] = None,
    charbonnier_eps: float = 1e-3,
    fit_background: bool = True,
    nonnegative_flux: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    rms_loss = effective_loss_rms(data, rms, loss_snr_cap=loss_snr_cap)
    loss_per_star, flux, bg, model, z = _psf_fit_per_star(
        model_unit,
        data,
        rms_loss,
        pixel_weight=pixel_weight,
        charbonnier_eps=charbonnier_eps,
        fit_background=fit_background,
        nonnegative_flux=nonnegative_flux,
    )
    stats = {
        "flux": flux,
        "background": bg,
        "chi_abs_median": z.abs().median(),
        "chi_abs_median_raw": ((model - data) / rms.clamp(min=1e-6)).abs().median(),
        "flux_median": flux.median(),
        "background_median": bg.median(),
    }
    if pixel_weight is not None:
        stats["loss_pixel_weight_frac"] = (pixel_weight > 0).float().mean()
    stats.update(effective_loss_rms_stats(rms, rms_loss))
    return loss_per_star.mean(), stats


def centroid_shift_grid(
    max_px: float,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Small Cartesian grid of native-pixel centroid offsets, including zero."""
    max_px = float(max_px)
    steps = int(steps)
    if max_px <= 0 or steps <= 1:
        return torch.zeros(1, 2, device=device, dtype=dtype)
    if steps % 2 == 0:
        steps += 1
    vals = torch.linspace(-max_px, max_px, steps, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(vals, vals, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def psf_residual_loss_with_centroid_nuisance(
    head: FoundationEPSFHead,
    epsf: torch.Tensor,
    frac_xy: torch.Tensor,
    data: torch.Tensor,
    rms: torch.Tensor,
    *,
    stamp_size: int,
    centroid_fit_max_px: float = 0.0,
    centroid_fit_steps: int = 1,
    loss_snr_cap: float = 0.0,
    loss_radius_px: float = 0.0,
    loss_taper_px: float = 0.0,
    charbonnier_eps: float = 1e-3,
    fit_background: bool = True,
    nonnegative_flux: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """Fit flux/background after marginalizing over a small centroid-offset grid.

    The grid choice is a hard nuisance assignment: first choose the best shift
    with gradients disabled, then rerender only that selected shift so normal
    gradients still flow to the ePSF shape.
    """
    shifts = centroid_shift_grid(
        centroid_fit_max_px,
        centroid_fit_steps,
        device=frac_xy.device,
        dtype=frac_xy.dtype,
    )
    rms_loss = effective_loss_rms(data, rms, loss_snr_cap=loss_snr_cap)
    if shifts.shape[0] == 1:
        native = head.render_at_native(epsf, frac_xy, stamp_size=stamp_size)
        pixel_weight = radial_loss_weight(
            data,
            frac_xy,
            loss_radius_px=loss_radius_px,
            loss_taper_px=loss_taper_px,
        )
        loss, stats = psf_residual_loss(
            native,
            data,
            rms,
            loss_snr_cap=loss_snr_cap,
            pixel_weight=pixel_weight,
            charbonnier_eps=charbonnier_eps,
            fit_background=fit_background,
            nonnegative_flux=nonnegative_flux,
        )
        zero = torch.zeros(frac_xy.shape[0], device=frac_xy.device, dtype=frac_xy.dtype)
        stats["centroid_shift_dx_median"] = zero.median()
        stats["centroid_shift_dy_median"] = zero.median()
        stats["centroid_shift_r_median"] = zero.median()
        stats["centroid_shift_r_p90"] = zero.median()
        stats["centroid_shift_edge_frac"] = zero.mean()
        return loss, stats, native

    n = frac_xy.shape[0]
    best_loss = torch.full((n,), float("inf"), device=frac_xy.device, dtype=frac_xy.dtype)
    best_shift = torch.zeros_like(frac_xy)
    max_r = shifts.norm(dim=1).max().clamp_min(1e-12)

    with torch.no_grad():
        epsf_detached = epsf.detach()
        for shift in shifts:
            frac_trial = frac_xy + shift.view(1, 2)
            native_trial = head.render_at_native(
                epsf_detached,
                frac_trial,
                stamp_size=stamp_size,
            )
            pixel_weight_trial = radial_loss_weight(
                data,
                frac_trial,
                loss_radius_px=loss_radius_px,
                loss_taper_px=loss_taper_px,
            )
            loss_star, _flux, _bg, _model, _z = _psf_fit_per_star(
                native_trial,
                data,
                rms_loss,
                pixel_weight=pixel_weight_trial,
                charbonnier_eps=charbonnier_eps,
                fit_background=fit_background,
                nonnegative_flux=nonnegative_flux,
            )
            take = loss_star < best_loss
            best_loss = torch.where(take, loss_star, best_loss)
            best_shift = torch.where(take[:, None], shift.view(1, 2), best_shift)

    native = head.render_at_native(epsf, frac_xy + best_shift, stamp_size=stamp_size)
    pixel_weight = radial_loss_weight(
        data,
        frac_xy + best_shift,
        loss_radius_px=loss_radius_px,
        loss_taper_px=loss_taper_px,
    )
    loss, stats = psf_residual_loss(
        native,
        data,
        rms,
        loss_snr_cap=loss_snr_cap,
        pixel_weight=pixel_weight,
        charbonnier_eps=charbonnier_eps,
        fit_background=fit_background,
        nonnegative_flux=nonnegative_flux,
    )
    shift_r = best_shift.norm(dim=1)
    stats["centroid_shift_dx_median"] = best_shift[:, 0].median()
    stats["centroid_shift_dy_median"] = best_shift[:, 1].median()
    stats["centroid_shift_r_median"] = shift_r.median()
    stats["centroid_shift_r_p90"] = torch.quantile(shift_r, 0.90)
    stats["centroid_shift_edge_frac"] = (shift_r >= 0.95 * max_r).float().mean()
    stats["centroid_shift"] = best_shift
    return loss, stats, native


def _float_dict(row: Mapping[str, object]) -> Dict[str, float]:
    out = {}
    for key, value in row.items():
        if isinstance(value, torch.Tensor):
            out[key] = float(value.detach().cpu())
        elif isinstance(value, (int, float, np.number)):
            out[key] = float(value)
    return out


def aggregate(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row.keys() if isinstance(row[k], (int, float, np.number))})
    return {k: float(np.mean([float(row[k]) for row in rows if k in row])) for k in keys}


def _shape_moments(
    img: torch.Tensor,
    frac_xy: Optional[torch.Tensor] = None,
    aperture_pix: float = 6.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aperture-weighted second moments per stamp.

    Returns ``(T, e1, e2)`` of shape ``[N]`` where:
        T  = Q11 + Q22  (size, in native pixels^2)
        e1 = (Q11 - Q22) / T
        e2 = 2 * Q12 / T
    The aperture is a top-hat of radius ``aperture_pix`` native pixels around
    ``(H/2 + frac_x, W/2 + frac_y)``. Background should already be subtracted.
    """
    if img.ndim == 4:
        img = img.squeeze(1)
    if img.ndim != 3:
        raise ValueError("img must be [N, H, W] or [N, 1, H, W]")
    N, H, W = img.shape
    device = img.device
    dtype = img.dtype
    yy = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    xx = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)
    if frac_xy is None:
        cx = float(W // 2)
        cy = float(H // 2)
        dx = xx - cx
        dy = yy - cy
    else:
        cx = (W // 2) + frac_xy[:, 0].view(N, 1, 1)
        cy = (H // 2) + frac_xy[:, 1].view(N, 1, 1)
        dx = xx - cx
        dy = yy - cy
    r2 = dx * dx + dy * dy
    mask = (r2 <= float(aperture_pix) ** 2).to(dtype)
    w = mask * img.clamp_min(0.0)
    norm = w.flatten(1).sum(dim=1).clamp_min(eps)
    Qxx = (w * dx * dx).flatten(1).sum(dim=1) / norm
    Qyy = (w * dy * dy).flatten(1).sum(dim=1) / norm
    Qxy = (w * dx * dy).flatten(1).sum(dim=1) / norm
    T = (Qxx + Qyy).clamp_min(eps)
    e1 = (Qxx - Qyy) / T
    e2 = 2.0 * Qxy / T
    return T, e1, e2


def _robust_limits(x: np.ndarray, q: float = 99.0) -> Tuple[float, float]:
    finite = np.asarray(x)[np.isfinite(x)]
    if finite.size == 0:
        return 0.0, 1.0
    vmax = float(np.percentile(finite, q))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(finite)))
    vmax = max(vmax, 1e-8)
    return 0.0, vmax


def _pearson_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nan_to_num(a.flatten(1), nan=0.0, posinf=0.0, neginf=0.0)
    b = torch.nan_to_num(b.flatten(1), nan=0.0, posinf=0.0, neginf=0.0)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    den = (a.pow(2).sum(dim=1).sqrt() * b.pow(2).sum(dim=1).sqrt()).clamp(min=1e-12)
    return (a * b).sum(dim=1) / den


def _make_visual_batch(
    records: Sequence[Record],
    device: torch.device,
    max_examples: int,
) -> Dict[str, torch.Tensor]:
    """Deterministic visual batch with guaranteed per-band coverage.

    Picks the highest-SNR record from every band present in ``records`` first
    (so faint NISP stars are not crowded out by bright Rubin/VIS detections),
    then pads up to ``max_examples`` with the next-highest-SNR records overall.
    """
    finite_records = [r for r in records if np.isfinite(float(r["snr"]))]
    pool = finite_records if finite_records else list(records)
    if not pool:
        rng = np.random.RandomState(0)
        return make_record_batch(list(records), device, rng, max_stars=0)

    best_per_band: Dict[int, Record] = {}
    for r in pool:
        bi = int(r["band_idx"])
        if bi not in best_per_band or float(r["snr"]) > float(best_per_band[bi]["snr"]):
            best_per_band[bi] = r
    band_reps = sorted(best_per_band.values(), key=lambda r: float(r["snr"]), reverse=True)
    rep_ids = {id(r) for r in band_reps}
    extras = sorted(
        (r for r in pool if id(r) not in rep_ids),
        key=lambda r: float(r["snr"]),
        reverse=True,
    )
    selected = list(band_reps)
    if max_examples > 0 and len(selected) < max_examples:
        selected.extend(extras[: max_examples - len(selected)])
    rng = np.random.RandomState(0)
    return make_record_batch(selected, device, rng, max_stars=0)


@torch.no_grad()
def _run_visual_tile(
    pair: Tuple[str, str, str],
    records: Sequence[Record],
    frozen_encoder: FrozenEncoder,
    head: FoundationEPSFHead,
    args: argparse.Namespace,
    device: torch.device,
    max_examples: int,
) -> Optional[Dict[str, torch.Tensor]]:
    if not records:
        return None
    tile_id, rubin_path, euclid_path = pair
    enc = None
    if not args.no_foundation_features:
        enc = encode_tile(
            frozen_encoder,
            tile_id,
            rubin_path,
            euclid_path,
            device,
            features_cache_dir=args.features_cache_dir,
        )
    batch = _make_visual_batch(records, device, max_examples=max_examples)
    feature_kwargs = {}
    if enc is not None:
        feature_kwargs = {
            "bottleneck": enc["bottleneck"],
            "vis_stem_features": enc["vis_stem"],
            "source_positions_vis": batch["source_positions_vis"],
            "fused_hw": enc["fused_hw"],
            "vis_hw": enc["vis_hw"],
        }
    pred = head(
        batch["pos_norm"],
        batch["band_idx"],
        return_dict=True,
        **feature_kwargs,
    )
    _loss, stats, native = psf_residual_loss_with_centroid_nuisance(
        head,
        pred["epsf"],
        batch["frac_xy"],
        batch["stamp"],
        batch["rms"],
        stamp_size=batch["stamp"].shape[-1],
        centroid_fit_max_px=args.centroid_fit_max_px,
        centroid_fit_steps=args.centroid_fit_steps,
        loss_snr_cap=args.loss_snr_cap,
        loss_radius_px=args.loss_radius_px,
        loss_taper_px=args.loss_taper_px,
        charbonnier_eps=args.charbonnier_eps,
        fit_background=args.fit_background,
        nonnegative_flux=args.nonnegative_flux,
    )
    flux = stats["flux"]
    bg = stats["background"]
    model = flux.view(-1, 1, 1) * native.squeeze(1) + bg.view(-1, 1, 1)
    resid = batch["stamp"] - model
    chi = resid / batch["rms"].clamp(min=1e-6)
    corr = _pearson_torch(model, batch["stamp"])
    return {
        "tile_id": tile_id,
        "batch": batch,
        "pred": pred,
        "native": native,
        "model": model,
        "resid": resid,
        "chi": chi,
        "corr": corr,
        "flux": flux,
        "background": bg,
        "centroid_shift": stats.get("centroid_shift"),
    }


def _plot_fit_examples(vis: Mapping[str, object], epoch: int, max_examples: int):
    batch = vis["batch"]
    model = vis["model"].detach().cpu().numpy()
    resid = vis["resid"].detach().cpu().numpy()
    epsf = vis["pred"]["epsf"].detach().cpu().numpy()[:, 0]
    stamps = batch["stamp"].detach().cpu().numpy()
    rms = batch["rms"].detach().cpu().numpy()
    band_idx = batch["band_idx"].detach().cpu().numpy()
    snr = batch["snr"].detach().cpu().numpy()
    corr = vis["corr"].detach().cpu().numpy()
    centroid_shift = vis.get("centroid_shift")
    shift_r = None
    if centroid_shift is not None:
        shift_r = centroid_shift.detach().cpu().norm(dim=1).numpy()
    n = min(max_examples, stamps.shape[0])
    fig, axes = plt.subplots(4, n, figsize=(2.2 * n, 8.2), squeeze=False)
    for col in range(n):
        obs = stamps[col]
        mod = model[col]
        res = resid[col]
        eps = epsf[col]
        _, vmax = _robust_limits(obs)
        rlim = max(float(np.nanpercentile(np.abs(res[np.isfinite(res)]), 99)) if np.isfinite(res).any() else 1e-8, 1e-8)
        chi_abs = np.nanmedian(np.abs(res / np.maximum(rms[col], 1e-6)))
        axes[0, col].imshow(obs, origin="lower", cmap="gray", vmin=0.0, vmax=vmax)
        axes[1, col].imshow(mod, origin="lower", cmap="gray", vmin=0.0, vmax=vmax)
        axes[2, col].imshow(res, origin="lower", cmap="RdBu_r", vmin=-rlim, vmax=rlim)
        axes[3, col].imshow(eps, origin="lower", cmap="inferno")
        shift_txt = "" if shift_r is None else f" dxy={shift_r[col]:.2f}"
        axes[0, col].set_title(
            f"{ALL_BANDS[int(band_idx[col])]}\nSNR={snr[col]:.0f} r={corr[col]:.2f} |chi|={chi_abs:.1f}{shift_txt}",
            fontsize=7,
        )
        for row in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    for row, label in enumerate(["Observed", "Model", "Residual", "ePSF"]):
        axes[row, 0].set_ylabel(label, fontsize=9)
    fig.suptitle(f"Held-out Gaia star fits - epoch {epoch} - tile {vis['tile_id']}", fontsize=10)
    fig.tight_layout()
    return fig


def _collect_best_band_examples(
    vis_list: Sequence[Mapping[str, object]],
) -> Dict[int, Dict[str, object]]:
    """Pick a representative validation example per band.

    Picks the star whose per-stamp median |chi| is closest to the band's
    overall median |chi| -- i.e. a typical-quality fit, not the brightest
    star (which is often saturated/contaminated and dominates with a huge
    chi that misrepresents the model's actual quality).
    """
    pool: Dict[int, list] = {}
    for vis in vis_list:
        batch = vis["batch"]
        stamps = batch["stamp"].detach().cpu().numpy()
        rms = batch["rms"].detach().cpu().numpy()
        band_idx = batch["band_idx"].detach().cpu().numpy()
        snr = batch["snr"].detach().cpu().numpy()
        model = vis["model"].detach().cpu().numpy()
        resid = vis["resid"].detach().cpu().numpy()
        chi = vis["chi"].detach().cpu().numpy()
        epsf = vis["pred"]["epsf"].detach().cpu().numpy()[:, 0]
        corr = vis["corr"].detach().cpu().numpy()
        centroid_shift = vis.get("centroid_shift")
        shift_r = None
        if centroid_shift is not None:
            shift_r = centroid_shift.detach().cpu().norm(dim=1).numpy()
        for idx in range(len(band_idx)):
            bi = int(band_idx[idx])
            star_med_abs_chi = float(np.nanmedian(np.abs(chi[idx])))
            pool.setdefault(bi, []).append({
                "med_abs_chi": star_med_abs_chi,
                "snr": float(snr[idx]),
                "obs": stamps[idx],
                "model": model[idx],
                "resid": resid[idx],
                "chi": chi[idx],
                "rms": rms[idx],
                "epsf": epsf[idx],
                "corr": float(corr[idx]),
                "tile_id": vis["tile_id"],
                "shift_r": float(shift_r[idx]) if shift_r is not None else 0.0,
            })

    best: Dict[int, Dict[str, object]] = {}
    for bi, items in pool.items():
        chis = np.array([it["med_abs_chi"] for it in items])
        snrs = np.array([it["snr"] for it in items])
        # Floor the SNR so we don't show a low-SNR star where everything looks
        # fine just because of noise; pick the typical chi within that subset.
        snr_thr = max(20.0, float(np.percentile(snrs, 25))) if len(snrs) else 0.0
        keep_mask = snrs >= snr_thr
        if not keep_mask.any():
            keep_mask = np.ones_like(snrs, dtype=bool)
        keep_idx = np.where(keep_mask)[0]
        target = float(np.median(chis[keep_idx]))
        pick = keep_idx[int(np.argmin(np.abs(chis[keep_idx] - target)))]
        chosen = items[pick]
        chosen.pop("med_abs_chi", None)
        best[bi] = chosen
    return best


def _plot_band_fit_grid(vis_list: Sequence[Mapping[str, object]], epoch: int):
    best = _collect_best_band_examples(vis_list)
    if not best:
        return None
    fig, axes = plt.subplots(4, len(ALL_BANDS), figsize=(1.7 * len(ALL_BANDS), 6.8), squeeze=False)
    for bi, band in enumerate(ALL_BANDS):
        item = best.get(bi)
        if item is None:
            for row in range(4):
                axes[row, bi].set_facecolor("#222")
                axes[row, bi].set_xticks([])
                axes[row, bi].set_yticks([])
            axes[0, bi].set_title(f"{band}\nno val example", fontsize=7)
            continue
        obs = item["obs"]
        mod = item["model"]
        res = item["resid"]
        eps = item["epsf"]
        rms = item["rms"]
        shift_r = float(item.get("shift_r", 0.0))
        _, vmax = _robust_limits(obs)
        rlim = max(float(np.nanpercentile(np.abs(res[np.isfinite(res)]), 99)) if np.isfinite(res).any() else 1e-8, 1e-8)
        chi_abs = np.nanmedian(np.abs(res / np.maximum(rms, 1e-6)))
        axes[0, bi].imshow(obs, origin="lower", cmap="gray", vmin=0.0, vmax=vmax)
        axes[1, bi].imshow(mod, origin="lower", cmap="gray", vmin=0.0, vmax=vmax)
        axes[2, bi].imshow(res, origin="lower", cmap="RdBu_r", vmin=-rlim, vmax=rlim)
        axes[3, bi].imshow(eps, origin="lower", cmap="inferno")
        axes[0, bi].set_title(
            f"{band}\nSNR={item['snr']:.0f} r={item['corr']:.2f} |chi|={chi_abs:.1f} dxy={shift_r:.2f}",
            fontsize=7,
        )
        for row in range(4):
            axes[row, bi].set_xticks([])
            axes[row, bi].set_yticks([])
    for row, label in enumerate(["Observed", "Model", "Residual", "ePSF"]):
        axes[row, 0].set_ylabel(label, fontsize=9)
    fig.suptitle(f"Per-band held-out Gaia star fits - epoch {epoch}", fontsize=10)
    fig.tight_layout()
    return fig


def _plot_epsf_gallery(vis_list: Sequence[Mapping[str, object]], head: FoundationEPSFHead, epoch: int):
    by_band: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}
    base = head.base_epsf().detach().cpu().numpy()
    for vis in vis_list:
        batch = vis["batch"]
        epsf = vis["pred"]["epsf"].detach().cpu().numpy()[:, 0]
        coeff = vis["pred"]["coeff"].detach().cpu().numpy()
        snr = batch["snr"].detach().cpu().numpy()
        band_idx = batch["band_idx"].detach().cpu().numpy()
        order = np.argsort(snr)[::-1]
        for idx in order:
            bi = int(band_idx[idx])
            if bi not in by_band:
                by_band[bi] = (epsf[idx], coeff[idx], float(snr[idx]))
    if not by_band:
        return None
    fig, axes = plt.subplots(2, len(ALL_BANDS), figsize=(1.6 * len(ALL_BANDS), 3.4), squeeze=False)
    for bi, band in enumerate(ALL_BANDS):
        axes[0, bi].imshow(base[bi], origin="lower", cmap="inferno")
        axes[0, bi].set_title(band, fontsize=7)
        if bi in by_band:
            eps, coeff, snr = by_band[bi]
            axes[1, bi].imshow(eps, origin="lower", cmap="inferno")
            axes[1, bi].set_title(f"SNR={snr:.0f} |c|={np.mean(np.abs(coeff)):.2f}", fontsize=7)
        else:
            axes[1, bi].set_facecolor("#222")
        for row in range(2):
            axes[row, bi].set_xticks([])
            axes[row, bi].set_yticks([])
    axes[0, 0].set_ylabel("Base", fontsize=9)
    axes[1, 0].set_ylabel("Head", fontsize=9)
    fig.suptitle(f"Per-band ePSF base vs foundation head - epoch {epoch}", fontsize=10)
    fig.tight_layout()
    return fig


def _plot_coeff_hist(vis_list: Sequence[Mapping[str, object]], epoch: int):
    coeffs = []
    band_ids = []
    for vis in vis_list:
        coeffs.append(vis["pred"]["coeff"].detach().cpu().numpy())
        band_ids.append(vis["batch"]["band_idx"].detach().cpu().numpy())
    if not coeffs:
        return None
    coeff = np.concatenate(coeffs, axis=0)
    band_idx = np.concatenate(band_ids, axis=0)
    rank = coeff.shape[1]
    fig, axes = plt.subplots(1, rank, figsize=(2.0 * rank, 2.4), squeeze=False)
    for k in range(rank):
        ax = axes[0, k]
        ax.hist(coeff[:, k], bins=24, color="steelblue", alpha=0.8)
        ax.axvline(0.0, color="black", lw=0.8)
        ax.set_title(f"c{k}", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Residual-basis coefficient histograms - epoch {epoch} - N={len(band_idx)}", fontsize=10)
    fig.tight_layout()
    return fig


@torch.no_grad()
def build_wandb_visuals(
    head: FoundationEPSFHead,
    val_pairs: Sequence[Tuple[str, str, str]],
    records_by_tile: Mapping[str, Sequence[Record]],
    frozen_encoder: FrozenEncoder,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> Dict[str, object]:
    if wandb is None or not val_pairs or args.wandb_image_every <= 0:
        return {}
    head.eval()
    vis_list = []
    want_tiles = max(1, args.wandb_max_visual_tiles)
    search_tiles = max(want_tiles, int(getattr(args, "wandb_band_search_tiles", want_tiles)))
    found_bands = set()
    for pair in val_pairs[:search_tiles]:
        try:
            vis = _run_visual_tile(
                pair,
                records_by_tile.get(pair[0], []),
                frozen_encoder,
                head,
                args,
                device,
                max_examples=args.wandb_n_examples,
            )
        except Exception as exc:
            print(f"  [warn] W&B visual skip {pair[0]}: {type(exc).__name__}: {exc}")
            continue
        if vis is not None:
            vis_list.append(vis)
            band_idx = vis["batch"]["band_idx"].detach().cpu().numpy()
            found_bands.update(int(x) for x in np.unique(band_idx))
            if len(vis_list) >= want_tiles and len(found_bands) == len(ALL_BANDS):
                break
    if not vis_list:
        return {}

    payload: Dict[str, object] = {}
    fig = _plot_fit_examples(vis_list[0], epoch, max_examples=args.wandb_n_examples)
    payload["vis/heldout_fits"] = wandb.Image(fig)
    plt.close(fig)

    fig = _plot_epsf_gallery(vis_list, head, epoch)
    if fig is not None:
        payload["vis/epsf_base_vs_head"] = wandb.Image(fig)
        plt.close(fig)

    fig = _plot_band_fit_grid(vis_list, epoch)
    if fig is not None:
        payload["vis/per_band_fits"] = wandb.Image(fig)
        plt.close(fig)

    fig = _plot_coeff_hist(vis_list, epoch)
    if fig is not None:
        payload["vis/coeff_hist"] = wandb.Image(fig)
        plt.close(fig)

    corrs = torch.cat([v["corr"].detach().cpu() for v in vis_list]).numpy()
    chi_abs = torch.cat([v["chi"].abs().flatten(1).median(dim=1).values.detach().cpu() for v in vis_list]).numpy()
    payload["vis/median_pearson"] = float(np.nanmedian(corrs))
    payload["vis/median_abs_chi"] = float(np.nanmedian(chi_abs))

    # Weak-lensing-relevant shape diagnostics: per-band median residuals in
    # T (size), e1, e2. Aperture-weighted unweighted moments after background
    # subtraction. dT/T should be < 1% (Rubin) or < 0.5% (Euclid) for WL use;
    # |de1|, |de2| should be < ~1e-3.
    band_idx_all = []
    dT_rel_all = []
    de1_all = []
    de2_all = []
    for v in vis_list:
        batch = v["batch"]
        bg = v["background"].detach().to(batch["stamp"].dtype)
        flux = v["flux"].detach().to(batch["stamp"].dtype)
        data_bs = batch["stamp"] - bg.view(-1, 1, 1)
        model_bs = flux.view(-1, 1, 1) * v["native"].squeeze(1)
        T_d, e1_d, e2_d = _shape_moments(data_bs, batch["frac_xy"])
        T_m, e1_m, e2_m = _shape_moments(model_bs, batch["frac_xy"])
        dT_rel_all.append(((T_d - T_m) / T_d.clamp_min(1e-12)).detach().cpu())
        de1_all.append((e1_d - e1_m).detach().cpu())
        de2_all.append((e2_d - e2_m).detach().cpu())
        band_idx_all.append(batch["band_idx"].detach().cpu())
    band_idx_all = torch.cat(band_idx_all).numpy()
    dT_rel_all = torch.cat(dT_rel_all).numpy()
    de1_all = torch.cat(de1_all).numpy()
    de2_all = torch.cat(de2_all).numpy()
    payload["vis/median_dT_rel"] = float(np.nanmedian(np.abs(dT_rel_all)))
    payload["vis/median_abs_de1"] = float(np.nanmedian(np.abs(de1_all)))
    payload["vis/median_abs_de2"] = float(np.nanmedian(np.abs(de2_all)))
    for bi, band in enumerate(ALL_BANDS):
        m = band_idx_all == bi
        if not m.any():
            continue
        payload[f"vis/median_dT_rel/{band}"] = float(np.nanmedian(np.abs(dT_rel_all[m])))
        payload[f"vis/median_abs_de1/{band}"] = float(np.nanmedian(np.abs(de1_all[m])))
        payload[f"vis/median_abs_de2/{band}"] = float(np.nanmedian(np.abs(de2_all[m])))
        payload[f"vis/median_abs_chi/{band}"] = float(np.nanmedian(np.abs(chi_abs[m]) if chi_abs.shape[0] == band_idx_all.shape[0] else chi_abs))

    return payload


def run_one_tile(
    pair: Tuple[str, str, str],
    records: Sequence[Record],
    frozen_encoder: FrozenEncoder,
    head: FoundationEPSFHead,
    optimizer: Optional[torch.optim.Optimizer],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.RandomState,
) -> Optional[Dict[str, float]]:
    tile_id, rubin_path, euclid_path = pair
    if not records:
        return None

    train = optimizer is not None
    head.train(mode=train)
    enc = None
    if not args.no_foundation_features:
        enc = encode_tile(
            frozen_encoder,
            tile_id,
            rubin_path,
            euclid_path,
            device,
            features_cache_dir=args.features_cache_dir,
        )
    batch = make_record_batch(records, device, rng, max_stars=args.max_stars_per_tile)
    stamp_size = int(batch["stamp"].shape[-1])

    with torch.set_grad_enabled(train):
        feature_kwargs = {}
        if enc is not None:
            feature_kwargs = {
                "bottleneck": enc["bottleneck"],
                "vis_stem_features": enc["vis_stem"],
                "source_positions_vis": batch["source_positions_vis"],
                "fused_hw": enc["fused_hw"],
                "vis_hw": enc["vis_hw"],
            }
        pred = head(
            batch["pos_norm"],
            batch["band_idx"],
            return_dict=True,
            **feature_kwargs,
        )
        data_loss, stats, native = psf_residual_loss_with_centroid_nuisance(
            head,
            pred["epsf"],
            batch["frac_xy"],
            batch["stamp"],
            batch["rms"],
            stamp_size=stamp_size,
            centroid_fit_max_px=args.centroid_fit_max_px,
            centroid_fit_steps=args.centroid_fit_steps,
            loss_snr_cap=args.loss_snr_cap,
            loss_radius_px=args.loss_radius_px,
            loss_taper_px=args.loss_taper_px,
            charbonnier_eps=args.charbonnier_eps,
            fit_background=args.fit_background,
            nonnegative_flux=args.nonnegative_flux,
        )
        reg = head.regularization(
            pred,
            coeff_l2_weight=args.coeff_l2_weight,
            residual_l2_weight=args.residual_l2_weight,
            basis_l2_weight=args.basis_l2_weight,
            basis_tv_weight=args.basis_tv_weight,
            base_tv_weight=args.base_tv_weight,
            epsf_tv_weight=args.epsf_tv_weight,
        )
        loss = data_loss + reg["loss"]

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()

    row = {
        "loss": loss,
        "loss_data": data_loss,
        "loss_reg": reg["loss"],
        "chi_abs_median": stats["chi_abs_median"],
        "chi_abs_median_raw": stats["chi_abs_median_raw"],
        "flux_median": stats["flux_median"],
        "background_median": stats["background_median"],
        "n_star": int(batch["stamp"].shape[0]),
    }
    for key in (
        "centroid_shift_dx_median",
        "centroid_shift_dy_median",
        "centroid_shift_r_median",
        "centroid_shift_r_p90",
        "centroid_shift_edge_frac",
        "loss_rms_cap_frac",
        "loss_rms_eff_ratio_median",
        "loss_rms_eff_ratio_p95",
        "loss_pixel_weight_frac",
    ):
        if key in stats:
            row[key] = stats[key]
    if "coeff" in pred:
        coeff_abs = pred["coeff"].detach().abs().flatten()
        coeff_sat_threshold = 0.98 * float(getattr(head, "coeff_scale", 1.0))
        row["coeff_abs_mean"] = coeff_abs.mean()
        row["coeff_abs_p95"] = torch.quantile(coeff_abs, 0.95)
        row["coeff_sat_frac"] = (coeff_abs > coeff_sat_threshold).float().mean()
    if "residual_logits" in pred:
        row["residual_logits_rms"] = pred["residual_logits"].detach().pow(2).mean().sqrt()
    for key, value in reg.items():
        if key != "loss":
            row[f"reg_{key}"] = value
    return _float_dict(row)


def save_checkpoint(
    path: Path,
    head: FoundationEPSFHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Mapping[str, object],
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "head_state": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "config": dict(config),
        "train_metrics": dict(train_metrics),
        "val_metrics": dict(val_metrics),
        "base_epsf": head.base_epsf().detach().cpu(),
    }
    try:
        torch.save(payload, path)
    except (OSError, RuntimeError) as exc:
        print(f"  WARN: local save to {path} failed ({exc}); training continues")
        return
    if wandb is not None and wandb.run is not None:
        try:
            wandb.save(str(path), base_path=str(path.parent), policy="now")
        except Exception as exc:
            print(f"  WARN: wandb.save({path.name}) failed: {exc}")


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    records_by_tile = load_psf_records_by_tile(
        args.train_dir,
        min_snr=args.min_snr,
        max_snr=args.max_snr,
    )
    pairs = discover_tile_pairs(str(args.rubin_dir), str(args.euclid_dir))
    pairs = [p for p in pairs if p[0] in records_by_tile]
    if not pairs:
        raise RuntimeError("No discovered tile pairs have PSF training records.")
    train_pairs, val_pairs = split_tile_pairs(pairs, val_frac=args.val_frac, seed=args.seed)
    if args.max_train_tiles > 0:
        train_pairs = train_pairs[:args.max_train_tiles]
    if args.max_val_tiles > 0:
        val_pairs = val_pairs[:args.max_val_tiles]
    print(f"Train tiles={len(train_pairs)} val tiles={len(val_pairs)}")

    foundation_ckpt = torch.load(args.foundation_checkpoint, map_location="cpu", weights_only=False)
    fcfg = foundation_ckpt.get("config", {})
    fused_scale = float(fcfg.get("fused_pixel_scale_arcsec", args.fused_pixel_scale))
    hidden_ch = int(fcfg.get("hidden_ch", args.hidden_ch))
    stem_ch = int(fcfg.get("stem_ch", args.stem_ch))

    foundation = load_foundation(args.foundation_checkpoint, device=torch.device("cpu"), freeze=True)
    frozen_encoder = FrozenEncoder(foundation).to(device).eval()

    base_mode = str(args.base_epsf_mode).lower()
    if base_mode == "auto":
        base_mode = "checkpoint" if args.base_epsf_checkpoint else "gaussian"

    base_sigmas_mas = parse_band_float_overrides(args.base_sigma_mas, name="base-sigma-mas")
    base_epsf = None
    if base_mode == "checkpoint":
        if not args.base_epsf_checkpoint:
            raise ValueError("--base-epsf-mode checkpoint requires --base-epsf-checkpoint")
        base_epsf = load_base_epsf_bank(
            args.base_epsf_checkpoint,
            band_names=ALL_BANDS,
            psf_size=args.psf_size,
            oversampling=args.oversampling,
        )
        print(f"Loaded base ePSF bank: {args.base_epsf_checkpoint}")
    elif base_mode in ("gaussian", "moffat"):
        base_epsf = analytic_epsf_bank(
            band_names=ALL_BANDS,
            psf_size=args.psf_size,
            oversampling=args.oversampling,
            kind=base_mode,
            sigmas_mas=base_sigmas_mas or None,
            sigma_scale=args.base_sigma_scale,
            moffat_beta=args.base_moffat_beta,
        )
        used_sigmas = {
            band: float(base_sigmas_mas.get(band, DEFAULT_CORE_SIGMA_MAS[band])) * float(args.base_sigma_scale)
            for band in ALL_BANDS
        }
        print(
            f"Initialized analytic {base_mode} base ePSF "
            f"(sigma_scale={args.base_sigma_scale:g}, sigmas_mas={used_sigmas})"
        )
    else:
        raise ValueError(f"unknown --base-epsf-mode {args.base_epsf_mode!r}")

    head = FoundationEPSFHead(
        psf_size=args.psf_size,
        oversampling=args.oversampling,
        band_names=ALL_BANDS,
        basis_rank=args.basis_rank,
        hidden_ch=hidden_ch,
        stem_ch=stem_ch,
        bottleneck_out=args.bottleneck_out,
        stem_out=args.stem_out,
        bottleneck_window=args.bottleneck_window,
        stem_window=args.stem_window,
        fused_pixel_scale=fused_scale,
        vis_pixel_scale=args.vis_pixel_scale,
        band_embed_dim=args.band_embed_dim,
        mlp_hidden=args.mlp_hidden,
        pos_freqs=args.pos_freqs,
        coeff_scale=args.coeff_scale,
        delta_scale=args.delta_scale,
        basis_init=args.basis_init,
        base_epsf=base_epsf,
        train_base=args.train_base,
        use_foundation_features=not args.no_foundation_features,
    ).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    config = {
        "foundation_checkpoint": str(args.foundation_checkpoint),
        "base_epsf_mode": base_mode,
        "base_epsf_checkpoint": str(args.base_epsf_checkpoint) if args.base_epsf_checkpoint else None,
        "base_sigma_mas": {
            band: float(base_sigmas_mas.get(band, DEFAULT_CORE_SIGMA_MAS[band]))
            for band in ALL_BANDS
        },
        "base_sigma_scale": args.base_sigma_scale,
        "base_moffat_beta": args.base_moffat_beta,
        "psf_size": args.psf_size,
        "oversampling": args.oversampling,
        "band_names": ALL_BANDS,
        "basis_rank": args.basis_rank,
        "hidden_ch": hidden_ch,
        "stem_ch": stem_ch,
        "bottleneck_out": args.bottleneck_out,
        "stem_out": args.stem_out,
        "bottleneck_window": args.bottleneck_window,
        "stem_window": args.stem_window,
        "fused_pixel_scale": fused_scale,
        "vis_pixel_scale": args.vis_pixel_scale,
        "band_embed_dim": args.band_embed_dim,
        "mlp_hidden": args.mlp_hidden,
        "pos_freqs": args.pos_freqs,
        "coeff_scale": args.coeff_scale,
        "delta_scale": args.delta_scale,
        "basis_init": args.basis_init,
        "train_base": args.train_base,
        "use_foundation_features": not args.no_foundation_features,
        "min_snr": args.min_snr,
        "max_snr": args.max_snr,
        "loss_snr_cap": args.loss_snr_cap,
        "loss_radius_px": args.loss_radius_px,
        "loss_taper_px": args.loss_taper_px,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = None
    if args.wandb_project:
        if wandb is None:
            print("[warn] wandb not installed; continuing without W&B logging.")
        else:
            run = wandb.init(project=args.wandb_project, name=args.wandb_name, config={**vars(args), **config})

    best_val = float("inf")
    rng = np.random.RandomState(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = list(rng.permutation(len(train_pairs)))
        train_rows = []
        for idx in order:
            pair = train_pairs[int(idx)]
            try:
                row = run_one_tile(
                    pair,
                    records_by_tile.get(pair[0], []),
                    frozen_encoder,
                    head,
                    optimizer,
                    args,
                    device,
                    rng,
                )
            except Exception as exc:
                print(f"train skip {pair[0]}: {type(exc).__name__}: {exc}")
                continue
            if row is not None:
                train_rows.append(row)
        if not train_rows:
            raise RuntimeError(
                "No training tiles completed successfully in this epoch. "
                "Check the skipped-tile messages above."
            )
        scheduler.step()

        val_rows = []
        with torch.no_grad():
            for pair in val_pairs:
                try:
                    row = run_one_tile(
                        pair,
                        records_by_tile.get(pair[0], []),
                        frozen_encoder,
                        head,
                        None,
                        args,
                        device,
                        rng,
                    )
                except Exception as exc:
                    print(f"val skip {pair[0]}: {type(exc).__name__}: {exc}")
                    continue
                if row is not None:
                    val_rows.append(row)

        train_metrics = aggregate(train_rows)
        val_metrics = aggregate(val_rows)
        train_loss = train_metrics.get("loss", float("nan"))
        val_loss = val_metrics.get("loss", train_loss)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train={train_loss:.5f} val={val_loss:.5f} "
            f"val_coeff_sat={val_metrics.get('coeff_sat_frac', float('nan')):.3f} "
            f"val_dxy={val_metrics.get('centroid_shift_r_median', float('nan')):.3f}px "
            f"val_cap={val_metrics.get('loss_rms_cap_frac', float('nan')):.3f} "
            f"val_win={val_metrics.get('loss_pixel_weight_frac', 1.0):.3f} "
            f"val_rawchi={val_metrics.get('chi_abs_median_raw', float('nan')):.3f} "
            f"n_train_tiles={len(train_rows)} n_val_tiles={len(val_rows)}"
        )

        save_checkpoint(
            out_dir / "checkpoint_latest.pt",
            head,
            optimizer,
            epoch,
            config,
            train_metrics,
            val_metrics,
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                out_dir / "checkpoint_best.pt",
                head,
                optimizer,
                epoch,
                config,
                train_metrics,
                val_metrics,
            )
            print(f"  saved best: {out_dir / 'checkpoint_best.pt'}")

        if run is not None:
            payload = {f"train/{k}": v for k, v in train_metrics.items()}
            payload.update({f"val/{k}": v for k, v in val_metrics.items()})
            payload["epoch"] = epoch
            payload["lr"] = optimizer.param_groups[0]["lr"]
            if (
                args.wandb_image_every > 0
                and (epoch == 1 or epoch == args.epochs or epoch % args.wandb_image_every == 0)
            ):
                payload.update(
                    build_wandb_visuals(
                        head,
                        val_pairs,
                        records_by_tile,
                        frozen_encoder,
                        args,
                        device,
                        epoch,
                    )
                )
            wandb.log(payload)

    summary = {
        "best_val_loss": best_val,
        "epochs_run": args.epochs,
        "n_train_tiles": len(train_pairs),
        "n_val_tiles": len(val_pairs),
        "final_val_coeff_sat_frac": val_metrics.get("coeff_sat_frac", float("nan")),
        "final_val_coeff_abs_p95": val_metrics.get("coeff_abs_p95", float("nan")),
        "final_val_centroid_shift_r_median": val_metrics.get("centroid_shift_r_median", float("nan")),
        "final_val_centroid_shift_r_p90": val_metrics.get("centroid_shift_r_p90", float("nan")),
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Best val loss: {best_val:.5f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", required=True, type=Path)
    p.add_argument("--rubin-dir", required=True, type=Path)
    p.add_argument("--euclid-dir", required=True, type=Path)
    p.add_argument("--foundation-checkpoint", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--base-epsf-mode",
        choices=("auto", "gaussian", "moffat", "checkpoint"),
        default="auto",
        help="Base ePSF prior. auto uses checkpoint when provided, otherwise analytic Gaussian.",
    )
    p.add_argument("--base-epsf-checkpoint", type=Path, default=None)
    p.add_argument(
        "--base-sigma-mas",
        type=str,
        default="",
        help=(
            "Optional comma-separated analytic core sigma overrides in mas, "
            "e.g. rubin_r=390,euclid_VIS=120."
        ),
    )
    p.add_argument(
        "--base-sigma-scale",
        type=float,
        default=1.0,
        help="Multiplicative scale applied to analytic base widths.",
    )
    p.add_argument(
        "--base-moffat-beta",
        type=float,
        default=3.5,
        help="Moffat beta for --base-epsf-mode moffat.",
    )
    p.add_argument("--features-cache-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--min-snr", type=float, default=10.0)
    p.add_argument(
        "--max-snr",
        type=float,
        default=0.0,
        help=(
            "Optional hard upper SNR cut for PSF stamps. 0 disables; prefer "
            "--loss-snr-cap for keeping bright stars without letting them dominate."
        ),
    )
    p.add_argument("--max-stars-per-tile", type=int, default=256)
    p.add_argument(
        "--max-train-tiles",
        type=int,
        default=0,
        help="Optional cap on train tiles after the deterministic split. 0 = all.",
    )
    p.add_argument(
        "--max-val-tiles",
        type=int,
        default=0,
        help="Optional cap on validation tiles after the deterministic split. 0 = all.",
    )

    p.add_argument("--psf-size", type=int, default=99)
    p.add_argument("--oversampling", type=int, default=5)
    p.add_argument("--basis-rank", type=int, default=8)
    p.add_argument("--hidden-ch", type=int, default=256)
    p.add_argument("--stem-ch", type=int, default=64)
    p.add_argument("--bottleneck-out", type=int, default=128)
    p.add_argument("--stem-out", type=int, default=64)
    p.add_argument("--bottleneck-window", type=int, default=11)
    p.add_argument("--stem-window", type=int, default=31)
    p.add_argument("--fused-pixel-scale", type=float, default=0.4)
    p.add_argument("--vis-pixel-scale", type=float, default=0.1)
    p.add_argument("--band-embed-dim", type=int, default=16)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--pos-freqs", type=int, default=6)
    p.add_argument("--coeff-scale", type=float, default=0.5)
    p.add_argument("--delta-scale", type=float, default=0.20)
    p.add_argument("--basis-init", type=float, default=1e-3)
    p.add_argument("--train-base", action="store_true")
    p.add_argument("--no-foundation-features", action="store_true")

    p.add_argument("--charbonnier-eps", type=float, default=1e-3)
    p.add_argument(
        "--loss-snr-cap",
        type=float,
        default=1000.0,
        help=(
            "Cap per-pixel S/N in the PSF loss by flooring effective RMS at "
            "|stamp - median(stamp)| / cap. Set <=0 to disable."
        ),
    )
    p.add_argument(
        "--loss-radius-px",
        type=float,
        default=13.0,
        help=(
            "Circular native-pixel radius for the PSF loss window. This keeps "
            "square stamp edges/corners from teaching non-PSF structure. Set <=0 to disable."
        ),
    )
    p.add_argument(
        "--loss-taper-px",
        type=float,
        default=2.0,
        help="Cosine-like taper width at --loss-radius-px. 0 gives a hard aperture.",
    )
    p.add_argument(
        "--centroid-fit-max-px",
        type=float,
        default=0.20,
        help=(
            "Marginalize each star over a small native-pixel centroid-offset "
            "grid with this half-width. Set 0 to disable."
        ),
    )
    p.add_argument(
        "--centroid-fit-steps",
        type=int,
        default=5,
        help="Number of grid samples per centroid axis; even values are rounded up.",
    )
    p.add_argument("--no-fit-background", action="store_false", dest="fit_background")
    p.set_defaults(fit_background=True)
    p.add_argument("--allow-negative-flux", action="store_false", dest="nonnegative_flux")
    p.set_defaults(nonnegative_flux=True)
    p.add_argument("--coeff-l2-weight", type=float, default=1e-3)
    p.add_argument("--residual-l2-weight", type=float, default=5e-5)
    p.add_argument("--basis-l2-weight", type=float, default=1e-6)
    p.add_argument("--basis-tv-weight", type=float, default=1e-4)
    p.add_argument("--base-tv-weight", type=float, default=1e-3,
                   help="TV smoothness on base_logits when --train-base is set "
                        "(0 disables; 1e-3 is a reasonable WL-quality starting point).")
    p.add_argument("--epsf-tv-weight", type=float, default=1e-5)

    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument(
        "--wandb-image-every",
        type=int,
        default=5,
        help="Log W&B diagnostic images every N epochs, plus first/last. 0 disables images.",
    )
    p.add_argument(
        "--wandb-n-examples",
        type=int,
        default=6,
        help="Number of held-out star examples shown in the W&B fit gallery.",
    )
    p.add_argument(
        "--wandb-max-visual-tiles",
        type=int,
        default=2,
        help="Maximum validation tiles to encode for W&B images.",
    )
    p.add_argument(
        "--wandb-band-search-tiles",
        type=int,
        default=24,
        help="Maximum validation tiles to search for representative per-band W&B examples.",
    )
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
