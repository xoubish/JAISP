"""Joint multi-band canonical centroiding (fast profile-likelihood version).

Fits ONE sky position per source across all available bands simultaneously.
For a trial position offset (du, dv) arcsec, each band's amplitude and
background are LINEAR parameters with a closed-form weighted least-squares
solution, so the outer optimization is only 2-dimensional. Band pixel grids
are linked through per-band tile-level affine WCS Jacobians (distortion
variation across a 102" tile is negligible for a <1" fit region).

Validated against injected truth on 2026-06-11: joint fitting beats the
VIS-only Gaussian canonical target 2.4-2.5x at all SNR levels
(io/_nb09_outputs/joint_centroid_truth_results.json).

Two-pass strategy for real (SED-diverse) sources: fit all bands, drop bands
with insignificant fitted amplitude, refit on the survivors. Sources where
the joint fit fails keep their classical VIS position (caller's fallback).

Intended use: precompute canonical labels once per tile
(precompute_joint_canonical_labels.py) and feed train_latent_position.py
via --canonical-labels.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.wcs import WCS
from scipy.optimize import minimize

SIGMA_PX_VIS = 2.5 / 2.355
SIGMA_PX_NISP = 2.5 / 2.355
SIGMA_PX_RUBIN = 3.0 / 2.355

RUBIN_BAND_ORDER = ["u", "g", "r", "i", "z", "y"]
NISP_BAND_ORDER = ["Y", "J", "H"]


def build_band_stack(rdata, edata) -> List[Dict]:
    """Assemble per-band image/rms/WCS/sigma dicts from raw tile npz dicts."""
    from astrometry2.source_matching import safe_header_from_card_string

    bands = []
    vwcs = WCS(safe_header_from_card_string(edata["wcs_VIS"].item()))
    for short in ["VIS"] + NISP_BAND_ORDER:
        key = f"img_{short}"
        if key not in edata:
            continue
        img = np.nan_to_num(np.asarray(edata[key], dtype=np.float32), nan=0.0)
        var = np.asarray(edata.get(f"var_{short}"), dtype=np.float32)
        rms = np.sqrt(np.clip(var, 1e-12, None))
        w = vwcs if short == "VIS" else WCS(safe_header_from_card_string(edata[f"wcs_{short}"].item()))
        bands.append(dict(name=f"euclid_{short}", img=img, rms=rms, wcs=w,
                          sigma=SIGMA_PX_VIS if short == "VIS" else SIGMA_PX_NISP))
    rwcs = WCS(rdata["wcs_hdr"].item())
    rcube = np.nan_to_num(np.asarray(rdata["img"], dtype=np.float32), nan=0.0)
    rrms = np.sqrt(np.clip(np.asarray(rdata["var"], dtype=np.float32), 1e-12, None))
    for bi, short in enumerate(RUBIN_BAND_ORDER):
        if bi >= rcube.shape[0]:
            break
        bands.append(dict(name=f"rubin_{short}", img=rcube[bi], rms=rrms[bi], wcs=rwcs,
                          sigma=SIGMA_PX_RUBIN))
    return bands


def _cutout(img, rms, x, y, r):
    H, W = img.shape
    xi, yi = int(round(x)), int(round(y))
    if xi - r < 0 or yi - r < 0 or xi + r + 1 > W or yi + r + 1 > H:
        return None
    sl = (slice(yi - r, yi + r + 1), slice(xi - r, xi + r + 1))
    yy, xx = np.mgrid[sl]
    return img[sl].astype(np.float64), rms[sl].astype(np.float64), xx.astype(np.float64), yy.astype(np.float64)


def _band_chi2_and_amp(s, du, dv):
    """Profile chi2 of one band stamp at sky offset (du, dv); linear (amp, bg)."""
    cx = s["cx"][0] + s["J"][0, 0] * du + s["J"][0, 1] * dv
    cy = s["cx"][1] + s["J"][1, 0] * du + s["J"][1, 1] * dv
    g = np.exp(-0.5 * (((s["xx"] - cx) / s["sigma"]) ** 2 + ((s["yy"] - cy) / s["sigma"]) ** 2))
    w = s["w"]  # 1/rms^2
    # weighted LSQ for d = amp*g + bg*1
    Sgg = float(np.sum(w * g * g)); Sg1 = float(np.sum(w * g)); S11 = s["S11"]
    Sgd = float(np.sum(w * g * s["img"])); S1d = s["S1d"]
    det = Sgg * S11 - Sg1 * Sg1
    if det <= 1e-12:
        return s["chi2_0"], 0.0
    amp = (Sgd * S11 - Sg1 * S1d) / det
    if amp < 0.0:  # non-negativity: clamp to bg-only model
        return s["chi2_0"], 0.0
    bg = (S1d - Sg1 * amp) / S11
    chi2 = s["Sdd"] - amp * Sgd - bg * S1d  # = sum w (d - model)^2 for linear LSQ
    return float(chi2), float(amp)


def _fit_duv(stamps) -> Optional[Tuple[float, float, np.ndarray]]:
    """Minimize total profile chi2 over (du, dv) arcsec. Returns (du, dv, amps)."""

    def total(p):
        return sum(_band_chi2_and_amp(s, p[0], p[1])[0] for s in stamps)

    res = minimize(total, np.zeros(2), method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-3, maxiter=200))
    du, dv = float(res.x[0]), float(res.x[1])
    if not (np.isfinite(du) and np.isfinite(dv)) or np.hypot(du, dv) > 0.55:
        return None
    amps = np.array([_band_chi2_and_amp(s, du, dv)[1] for s in stamps])
    return du, dv, amps


def _position_sigma(stamps, du, dv, step=0.003):
    """Per-axis 1-sigma of (du, dv) in arcsec from the chi2 curvature.

    For chi2(theta), cov = 2 * H^{-1} with H the Hessian; estimated by
    central finite differences at the optimum.
    """
    def c(a, b):
        return sum(_band_chi2_and_amp(s, a, b)[0] for s in stamps)

    c0 = c(du, dv)
    h = np.zeros((2, 2))
    h[0, 0] = (c(du + step, dv) - 2 * c0 + c(du - step, dv)) / step ** 2
    h[1, 1] = (c(du, dv + step) - 2 * c0 + c(du, dv - step)) / step ** 2
    h[0, 1] = h[1, 0] = (
        c(du + step, dv + step) - c(du + step, dv - step)
        - c(du - step, dv + step) + c(du - step, dv - step)
    ) / (4 * step ** 2)
    try:
        cov = 2.0 * np.linalg.inv(h)
        var = np.diag(cov)
        if np.any(var <= 0):
            return np.nan, np.nan
        return float(np.sqrt(var[0])), float(np.sqrt(var[1]))
    except np.linalg.LinAlgError:
        return np.nan, np.nan


def joint_refine_positions(
    bands: List[Dict],
    vis_seed_xy: np.ndarray,
    vis_wcs: WCS,
    stamp_r: int = 5,
    min_bands: int = 3,
    amp_sig_floor: float = 1.5,
    return_sigma: bool = False,
):
    """Joint-refine canonical positions for VIS-frame seeds.

    Returns (xy [N,2] VIS px, ok [N] bool, n_bands_used [N]) and, when
    return_sigma=True, additionally sigma [N,2] per-axis 1-sigma in arcsec
    (NaN where unavailable).
    """
    N = vis_seed_xy.shape[0]
    out_xy = vis_seed_xy.astype(np.float64).copy()
    ok = np.zeros(N, dtype=bool)
    nbu = np.zeros(N, dtype=np.int16)
    pos_sig = np.full((N, 2), np.nan, dtype=np.float32)
    if N == 0:
        if return_sigma:
            return out_xy.astype(np.float32), ok, nbu, pos_sig
        return out_xy.astype(np.float32), ok, nbu

    # vectorized: sky positions of all seeds, per-band pixel positions,
    # ONE tile-level affine Jacobian per band (evaluated at the tile center).
    ras, decs = vis_wcs.pixel_to_world_values(vis_seed_xy[:, 0], vis_seed_xy[:, 1])
    ras = np.atleast_1d(ras); decs = np.atleast_1d(decs)
    cosd = np.cos(np.deg2rad(np.median(decs)))
    ra_c, dec_c = float(np.median(ras)), float(np.median(decs))
    e = 0.1 / 3600.0

    per_band = []
    for bd in bands:
        bx, by = bd["wcs"].world_to_pixel_values(ras, decs)
        px0, py0 = bd["wcs"].world_to_pixel_values(ra_c, dec_c)
        pxa, pya = bd["wcs"].world_to_pixel_values(ra_c + e / cosd, dec_c)
        pxb, pyb = bd["wcs"].world_to_pixel_values(ra_c, dec_c + e)
        J = np.array([[(pxa - px0) / 0.1, (pxb - px0) / 0.1],
                      [(pya - py0) / 0.1, (pyb - py0) / 0.1]], dtype=np.float64)
        per_band.append(dict(bd, bx=np.atleast_1d(bx), by=np.atleast_1d(by), J=J))

    for k in range(N):
        stamps = []
        for bd in per_band:
            c = _cutout(bd["img"], bd["rms"], float(bd["bx"][k]), float(bd["by"][k]), stamp_r)
            if c is None:
                continue
            img_s, rms_s, xx, yy = c
            if not np.all(np.isfinite(rms_s)):
                continue
            w = 1.0 / np.clip(rms_s, 1e-9, None) ** 2
            S11 = float(np.sum(w)); S1d = float(np.sum(w * img_s)); Sdd = float(np.sum(w * img_s * img_s))
            chi2_0 = Sdd - S1d * S1d / max(S11, 1e-12)  # bg-only chi2
            stamps.append(dict(img=img_s, w=w, xx=xx, yy=yy,
                               cx=(float(bd["bx"][k]), float(bd["by"][k])), J=bd["J"],
                               sigma=bd["sigma"], S11=S11, S1d=S1d, Sdd=Sdd, chi2_0=chi2_0,
                               noise=float(np.median(rms_s))))
        if len(stamps) < min_bands:
            continue

        fit = _fit_duv(stamps)
        if fit is None:
            continue
        du, dv, amps = fit

        sig = np.array([amps[i] / max(stamps[i]["noise"], 1e-9) for i in range(len(stamps))])
        keep = sig >= amp_sig_floor
        if keep.sum() >= min_bands and keep.sum() < len(stamps):
            fit2 = _fit_duv([s for s, kp in zip(stamps, keep) if kp])
            if fit2 is not None:
                du, dv, _ = fit2
                nbu[k] = int(keep.sum())
            else:
                nbu[k] = len(stamps)
        else:
            nbu[k] = len(stamps)

        # back to VIS px via the VIS-band affine (first entry is VIS)
        Jv = per_band[0]["J"]
        fx = vis_seed_xy[k, 0] + Jv[0, 0] * du + Jv[0, 1] * dv
        fy = vis_seed_xy[k, 1] + Jv[1, 0] * du + Jv[1, 1] * dv
        if np.hypot(fx - vis_seed_xy[k, 0], fy - vis_seed_xy[k, 1]) * 0.1 > 0.6:
            continue
        out_xy[k] = (float(fx), float(fy))
        ok[k] = True
        if return_sigma:
            final_stamps = [s for s, kp in zip(stamps, keep) if kp] if (
                keep.sum() >= min_bands and keep.sum() < len(stamps)) else stamps
            pos_sig[k] = _position_sigma(final_stamps, du, dv)

    if return_sigma:
        return out_xy.astype(np.float32), ok, nbu, pos_sig
    return out_xy.astype(np.float32), ok, nbu
