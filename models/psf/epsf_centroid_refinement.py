"""Centroid refinement using the v10 foundation-conditioned ePSF head.

This is the bridge between the current ``FoundationEPSFHead`` PSF product and
the astrometry pipeline.  It mirrors the simple Gaussian centroider's public
shape while replacing the circular Gaussian with a local ePSF rendered from the
frozen foundation features.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_MODELS, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from astrometry2.source_matching import robust_sigma
from psf.foundation_epsf_head import FoundationEPSFHead


def _constant_rms_like(image: np.ndarray) -> np.ndarray:
    sig = max(float(robust_sigma(np.asarray(image, dtype=np.float32))), 1e-6)
    return np.full_like(np.asarray(image, dtype=np.float32), sig, dtype=np.float32)


def _cut_stamp(
    image: np.ndarray,
    rms: np.ndarray,
    x_pix: float,
    y_pix: float,
    stamp_size: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float, bool]:
    h, w = image.shape
    half = int(stamp_size) // 2
    ix = int(round(float(x_pix)))
    iy = int(round(float(y_pix)))
    x0 = ix - half
    y0 = iy - half
    x1 = x0 + int(stamp_size)
    y1 = y0 + int(stamp_size)
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None, None, 0.0, 0.0, False
    return (
        image[y0:y1, x0:x1].astype(np.float32, copy=True),
        rms[y0:y1, x0:x1].astype(np.float32, copy=True),
        float(x_pix) - ix,
        float(y_pix) - iy,
        True,
    )


def _fit_flux_bg_chi(
    psf_native: torch.Tensor,
    data: torch.Tensor,
    rms: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted least-squares flux/background fit for each source stamp."""
    p = psf_native.clamp_min(0.0)
    d = data
    w = 1.0 / rms.clamp_min(1e-6).square()
    reduce = (-2, -1)

    s_pp = (w * p * p).sum(dim=reduce)
    s_p = (w * p).sum(dim=reduce)
    s_1 = w.sum(dim=reduce).clamp_min(1e-12)
    s_pd = (w * p * d).sum(dim=reduce)
    s_d = (w * d).sum(dim=reduce)
    det = (s_pp * s_1 - s_p * s_p).clamp_min(1e-12)

    flux = (s_pd * s_1 - s_d * s_p) / det
    bg = (s_pp * s_d - s_p * s_pd) / det

    bg_zero_flux = s_d / s_1
    bad_flux = flux <= 0
    flux = torch.where(bad_flux, torch.zeros_like(flux), flux)
    bg = torch.where(bad_flux, bg_zero_flux, bg)

    resid = (flux[:, None, None] * p + bg[:, None, None] - d) / rms.clamp_min(1e-6)
    chi = resid.square().mean(dim=reduce)
    return chi, flux, bg


def _shift_grid(max_px: float, steps: int, device: torch.device) -> torch.Tensor:
    max_px = float(max_px)
    steps = int(steps)
    if max_px <= 0 or steps <= 1:
        return torch.zeros(1, 2, device=device, dtype=torch.float32)
    if steps % 2 == 0:
        steps += 1
    vals = torch.linspace(-max_px, max_px, steps, device=device)
    yy, xx = torch.meshgrid(vals, vals, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


@torch.no_grad()
def refine_centroids_foundation_epsf(
    image: np.ndarray,
    seed_xy: np.ndarray,
    epsf_head: FoundationEPSFHead,
    *,
    band_name: str,
    tile_hw: Tuple[int, int],
    rms: Optional[np.ndarray] = None,
    bottleneck: Optional[torch.Tensor] = None,
    vis_stem_features: Optional[torch.Tensor] = None,
    source_positions_vis: Optional[np.ndarray] = None,
    fused_hw: Optional[Tuple[int, int]] = None,
    vis_hw: Optional[Tuple[int, int]] = None,
    stamp_size: int = 21,
    search_radius_px: float = 0.5,
    search_steps: int = 7,
    flux_floor_sigma: float = 1.5,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine centroids by grid-searching the rendered ePSF sub-pixel phase.

    Parameters are intentionally close to ``refine_centroids_psf_fit``.  The
    returned ``centroid_sigma_px`` is an approximate matched-filter uncertainty
    from the fitted ePSF flux; use it as a ranking/diagnostic quantity, not a
    calibrated covariance.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("image must be a 2D array")
    rms_img = _constant_rms_like(img) if rms is None else np.asarray(rms, dtype=np.float32)
    if rms_img.shape != img.shape:
        raise ValueError(f"rms shape {rms_img.shape} does not match image shape {img.shape}")

    seeds = np.asarray(seed_xy, dtype=np.float32)
    n_src = seeds.shape[0]
    out = seeds.copy()
    snr = np.ones(n_src, dtype=np.float32)
    centroid_sigma_px = np.full(n_src, np.inf, dtype=np.float32)
    if n_src == 0:
        return out, snr, centroid_sigma_px

    if band_name not in epsf_head.band_to_idx:
        raise ValueError(f"{band_name!r} is not in ePSF head band list")
    band_idx_value = int(epsf_head.band_to_idx[band_name])
    device = next(epsf_head.parameters()).device
    h, w = img.shape
    tile_h, tile_w = tile_hw
    global_sig = max(float(robust_sigma(img)), 1e-6)
    stamp_size = int(stamp_size)
    if stamp_size % 2 == 0:
        stamp_size += 1

    records = []
    for i, (xf, yf) in enumerate(seeds):
        stamp, stamp_rms, frac_x, frac_y, ok = _cut_stamp(img, rms_img, xf, yf, stamp_size)
        if not ok or stamp is None or stamp_rms is None:
            continue
        bg_quick = float(np.percentile(stamp, 30))
        signal = float(np.clip(stamp - bg_quick, 0.0, None).sum())
        if signal <= float(flux_floor_sigma) * global_sig:
            continue
        records.append((i, stamp, stamp_rms, frac_x, frac_y))

    if not records:
        return out, snr, centroid_sigma_px

    src_vis = (
        np.asarray(source_positions_vis, dtype=np.float32)
        if source_positions_vis is not None else seeds
    )
    shifts = _shift_grid(search_radius_px, search_steps, device=device)

    for start in range(0, len(records), int(batch_size)):
        chunk = records[start:start + int(batch_size)]
        idx_np = np.array([r[0] for r in chunk], dtype=np.int64)
        data = torch.from_numpy(np.stack([r[1] for r in chunk])).to(device)
        rms_t = torch.from_numpy(np.stack([r[2] for r in chunk])).to(device)
        frac0 = torch.tensor([[r[3], r[4]] for r in chunk], dtype=torch.float32, device=device)

        xy = seeds[idx_np]
        pos_norm = torch.from_numpy(np.stack([
            2.0 * xy[:, 0] / max(tile_w - 1, 1) - 1.0,
            2.0 * xy[:, 1] / max(tile_h - 1, 1) - 1.0,
        ], axis=1).astype(np.float32)).to(device)
        band_idx = torch.full((len(chunk),), band_idx_value, dtype=torch.long, device=device)

        kwargs = {}
        if epsf_head.use_foundation_features:
            missing = [
                name for name, value in (
                    ("bottleneck", bottleneck),
                    ("vis_stem_features", vis_stem_features),
                    ("source_positions_vis", src_vis),
                    ("fused_hw", fused_hw),
                    ("vis_hw", vis_hw),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "Foundation ePSF centroiding needs " + ", ".join(missing)
                )
            kwargs = {
                "bottleneck": bottleneck,
                "vis_stem_features": vis_stem_features,
                "source_positions_vis": torch.from_numpy(src_vis[idx_np]).float().to(device),
                "fused_hw": fused_hw,
                "vis_hw": vis_hw,
            }

        pred = epsf_head(pos_norm, band_idx, return_dict=True, **kwargs)
        epsf = pred["epsf"]

        best_chi = torch.full((len(chunk),), float("inf"), device=device)
        best_shift = torch.zeros((len(chunk), 2), dtype=torch.float32, device=device)
        best_flux = torch.zeros((len(chunk),), dtype=torch.float32, device=device)
        best_flux_var = torch.full((len(chunk),), float("inf"), dtype=torch.float32, device=device)

        for shift in shifts:
            native = epsf_head.render_at_native(epsf, frac0 + shift.view(1, 2), stamp_size)
            p = native.squeeze(1)
            chi, flux, _bg = _fit_flux_bg_chi(p, data, rms_t)
            w = 1.0 / rms_t.clamp_min(1e-6).square()
            fisher = (w * p * p).sum(dim=(-2, -1)).clamp_min(1e-12)
            flux_var = 1.0 / fisher
            take = chi < best_chi
            best_chi = torch.where(take, chi, best_chi)
            best_shift = torch.where(take[:, None], shift.view(1, 2), best_shift)
            best_flux = torch.where(take, flux, best_flux)
            best_flux_var = torch.where(take, flux_var, best_flux_var)

        out[idx_np] = seeds[idx_np] + best_shift.detach().cpu().numpy().astype(np.float32)
        snr_val = (best_flux / best_flux_var.sqrt().clamp_min(1e-6)).detach().cpu().numpy()
        snr[idx_np] = np.maximum(snr_val.astype(np.float32), 1.0)
        centroid_sigma_px[idx_np] = (
            max(float(search_radius_px), 0.05) / np.maximum(snr[idx_np], 1.0)
        ).astype(np.float32)

    return out.astype(np.float32), snr.astype(np.float32), centroid_sigma_px
