"""Joint multi-band canonical centroiding.

Fits ONE sky position per source across all available bands simultaneously
(shared (dRA, dDec); per-band amplitude >= 0 and background; per-pixel
inverse-variance weights; band pixel grids linked through local WCS affines).
Validated against injected truth on 2026-06-11: 2.4-2.5x better than the
VIS-only Gaussian canonical target at all SNR levels
(io/_nb09_outputs/joint_centroid_truth_results.json).

Two-pass strategy for real (SED-diverse) sources: fit all bands, drop bands
whose fitted amplitude is insignificant, refit on the survivors. Falls back to
the VIS-only solution when the joint fit fails or wanders.

Intended use: precompute canonical labels once per tile
(scripts: precompute_joint_canonical_labels.py) and feed them to
train_latent_position.py via --canonical-labels.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.wcs import WCS
from scipy.optimize import least_squares

# Gaussian sigma (px) assumed per band family, matched to the classical
# centroid fitter's FWHM guesses used across train/eval.
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


def _fit(stamps) -> Optional[Tuple[float, float, np.ndarray]]:
    """Shared (du, dv) in arcsec + per-band (amp, bg). Returns (du, dv, amps)."""
    p0 = [0.0, 0.0]
    lo = [-0.6, -0.6]
    hi = [0.6, 0.6]
    for s in stamps:
        bg0 = float(np.median(s["img"]))
        amp0 = max(float(s["img"].max() - bg0), 1e-3)
        p0 += [amp0, bg0]
        lo += [0.0, -np.inf]
        hi += [np.inf, np.inf]

    def resid(p):
        du, dv = p[0], p[1]
        out = []
        for i, s in enumerate(stamps):
            amp, bg = p[2 + 2 * i], p[3 + 2 * i]
            cx = s["cx"][0] + s["J"][0, 0] * du + s["J"][0, 1] * dv
            cy = s["cx"][1] + s["J"][1, 0] * du + s["J"][1, 1] * dv
            model = amp * np.exp(
                -0.5 * (((s["xx"] - cx) / s["sigma"]) ** 2 + ((s["yy"] - cy) / s["sigma"]) ** 2)
            ) + bg
            out.append(((s["img"] - model) / s["rms"]).ravel())
        return np.concatenate(out)

    try:
        res = least_squares(resid, np.asarray(p0), bounds=(lo, hi), method="trf", max_nfev=300)
    except Exception:
        return None
    du, dv = float(res.x[0]), float(res.x[1])
    if not (np.isfinite(du) and np.isfinite(dv)) or np.hypot(du, dv) > 0.55:
        return None
    amps = res.x[2::2]
    return du, dv, amps


def joint_refine_positions(
    bands: List[Dict],
    vis_seed_xy: np.ndarray,
    vis_wcs: WCS,
    stamp_r: int = 5,
    min_bands: int = 3,
    amp_sig_floor: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint-refine canonical positions for VIS-frame seeds.

    Returns (xy [N,2] VIS px, ok [N] bool, n_bands_used [N]).
    Sources where the joint fit fails keep their input seed and ok=False
    (caller decides the fallback, e.g. classical VIS refine).
    """
    N = vis_seed_xy.shape[0]
    out_xy = vis_seed_xy.astype(np.float64).copy()
    ok = np.zeros(N, dtype=bool)
    nbu = np.zeros(N, dtype=np.int16)

    ra0s, dec0s = vis_wcs.pixel_to_world_values(vis_seed_xy[:, 0], vis_seed_xy[:, 1])
    cosd = np.cos(np.deg2rad(np.median(dec0s)))
    e = 0.1 / 3600.0

    for k in range(N):
        ra0, dec0 = float(ra0s[k]), float(dec0s[k])
        stamps = []
        for bd in bands:
            px0, py0 = bd["wcs"].world_to_pixel_values(ra0, dec0)
            c = _cutout(bd["img"], bd["rms"], float(px0), float(py0), stamp_r)
            if c is None:
                continue
            img_s, rms_s, xx, yy = c
            if not np.all(np.isfinite(rms_s)) or float(np.nanmax(rms_s)) <= 0:
                continue
            pxa, pya = bd["wcs"].world_to_pixel_values(ra0 + e / cosd, dec0)
            pxb, pyb = bd["wcs"].world_to_pixel_values(ra0, dec0 + e)
            J = np.array([[(pxa - px0) / 0.1, (pxb - px0) / 0.1],
                          [(pya - py0) / 0.1, (pyb - py0) / 0.1]], dtype=np.float64)
            stamps.append(dict(img=img_s, rms=rms_s, xx=xx, yy=yy,
                               cx=(float(px0), float(py0)), J=J, sigma=bd["sigma"],
                               noise=float(np.median(rms_s))))
        if len(stamps) < min_bands:
            continue

        fit = _fit(stamps)
        if fit is None:
            continue
        du, dv, amps = fit

        # pass 2: keep bands with significant amplitude, refit
        sig = np.array([amps[i] / max(stamps[i]["noise"], 1e-9) for i in range(len(stamps))])
        keep = sig >= amp_sig_floor
        if keep.sum() >= min_bands and keep.sum() < len(stamps):
            fit2 = _fit([s for s, kp in zip(stamps, keep) if kp])
            if fit2 is not None:
                du, dv, _ = fit2
                nbu[k] = int(keep.sum())
            else:
                nbu[k] = len(stamps)
        else:
            nbu[k] = len(stamps)

        ra_f = ra0 + du / 3600.0 / cosd
        dec_f = dec0 + dv / 3600.0
        fx, fy = vis_wcs.world_to_pixel_values(ra_f, dec_f)
        # sanity: stay within 0.6" of the seed
        if np.hypot(fx - vis_seed_xy[k, 0], fy - vis_seed_xy[k, 1]) * 0.1 > 0.6:
            continue
        out_xy[k] = (float(fx), float(fy))
        ok[k] = True

    return out_xy.astype(np.float32), ok, nbu
