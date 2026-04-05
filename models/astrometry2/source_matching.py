"""
Shared source detection and Rubin<->VIS matching utilities for astrometry.
"""

from typing import Dict, List

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, maximum_filter

import astropy.units as u

RUBIN_BAND_ORDER = ['u', 'g', 'r', 'i', 'z', 'y']


def _to_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return max(1e-8, 1.4826 * mad)


def safe_header_from_card_string(raw: str) -> fits.Header:
    """
    Parse a FITS-card string while skipping malformed CONTINUE cards.
    """
    hdr = fits.Header()
    text = str(raw)
    n = (len(text) // 80) * 80
    for i in range(0, n, 80):
        chunk = text[i:i + 80]
        if not chunk.strip():
            continue
        try:
            card = fits.Card.fromstring(chunk)
        except Exception:
            continue
        key = card.keyword
        if not key or key in {"END", "CONTINUE"}:
            continue
        try:
            hdr[key] = card.value
        except Exception:
            continue
    return hdr


def detect_sources(
    image: np.ndarray,
    nsig: float = 4.0,
    smooth_sigma: float = 1.0,
    min_dist: int = 7,
    max_sources: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Local-peak detection with simple subpixel centroiding.
    """
    x = np.nan_to_num(image, nan=0.0).astype(np.float32, copy=False)
    med = float(np.median(x))
    sigma = robust_sigma(x)

    y = gaussian_filter(x, float(max(0.0, smooth_sigma)))
    thresh = med + float(nsig) * sigma

    local_max = maximum_filter(y, size=max(3, int(min_dist)), mode="nearest")
    mask = (y == local_max) & (y > thresh)
    yy, xx = np.where(mask)
    if yy.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    score = y[yy, xx]
    order = np.argsort(score)[::-1]
    yy = yy[order]
    xx = xx[order]
    if yy.size > max_sources:
        yy = yy[:max_sources]
        xx = xx[:max_sources]

    H, W = x.shape
    x_sub = np.zeros((yy.size,), dtype=np.float32)
    y_sub = np.zeros((yy.size,), dtype=np.float32)
    for i, (yi, xi) in enumerate(zip(yy, xx)):
        y0 = max(0, int(yi) - 2)
        y1 = min(H, int(yi) + 3)
        x0 = max(0, int(xi) - 2)
        x1 = min(W, int(xi) + 3)
        patch = x[y0:y1, x0:x1]
        patch = np.clip(patch - np.percentile(patch, 30), 0.0, None)
        s = float(patch.sum())
        if s <= 0.0:
            x_sub[i] = float(xi)
            y_sub[i] = float(yi)
            continue
        gy, gx = np.mgrid[y0:y1, x0:x1]
        x_sub[i] = float((patch * gx).sum() / s)
        y_sub[i] = float((patch * gy).sum() / s)
    return x_sub, y_sub


def match_sources_wcs(
    rubin_x: np.ndarray,
    rubin_y: np.ndarray,
    vis_x: np.ndarray,
    vis_y: np.ndarray,
    rubin_wcs: WCS,
    vis_wcs: WCS,
    max_sep_arcsec: float,
    clip_sigma: float,
    max_matches: int,
    dedup_vis_px: float = 50.0,
    dedup_rubin_px: float = 25.0,
) -> Dict[str, np.ndarray]:
    """
    Mutual-nearest WCS matching with sigma clipping in radial offset space.
    Nearby matched peaks on the same bright object are merged so anchors are
    closer to object-level matches than knot-level detections.
    """
    empty = {
        "rubin_xy": np.zeros((0, 2), dtype=np.float32),
        "vis_xy": np.zeros((0, 2), dtype=np.float32),
        "offsets": np.zeros((0, 2), dtype=np.float32),
        "sep_arcsec": np.zeros((0,), dtype=np.float32),
    }
    if rubin_x.size == 0 or vis_x.size == 0:
        return empty

    r_ra, r_dec = rubin_wcs.wcs_pix2world(rubin_x, rubin_y, 0)
    v_ra, v_dec = vis_wcs.wcs_pix2world(vis_x, vis_y, 0)
    c_r = SkyCoord(r_ra * u.deg, r_dec * u.deg)
    c_v = SkyCoord(v_ra * u.deg, v_dec * u.deg)

    idx_v2r, sep_v2r, _ = c_v.match_to_catalog_sky(c_r)
    idx_r2v, _, _ = c_r.match_to_catalog_sky(c_v)
    v_idx = np.arange(c_v.size, dtype=np.int64)
    keep = (idx_r2v[idx_v2r] == v_idx) & (sep_v2r.arcsec < float(max_sep_arcsec))
    if keep.sum() == 0:
        return empty

    vis_sel = v_idx[keep]
    rub_sel = idx_v2r[keep]
    vv = c_v[vis_sel]
    rr = c_r[rub_sel]

    dra = (vv.ra.deg - rr.ra.deg) * np.cos(np.deg2rad(vv.dec.deg)) * 3600.0
    ddec = (vv.dec.deg - rr.dec.deg) * 3600.0
    sep = sep_v2r.arcsec[keep]

    radial = np.hypot(dra, ddec)
    med = float(np.median(radial))
    sig = robust_sigma(radial)
    clip = radial <= (med + float(clip_sigma) * sig)
    if clip.sum() == 0:
        return empty

    vis_sel = vis_sel[clip]
    rub_sel = rub_sel[clip]
    dra = dra[clip]
    ddec = ddec[clip]
    sep = sep[clip]

    if vis_sel.size > int(max_matches):
        order = np.argsort(sep)[:int(max_matches)]
        vis_sel = vis_sel[order]
        rub_sel = rub_sel[order]
        dra = dra[order]
        ddec = ddec[order]
        sep = sep[order]

    # Object-level deduplication: local-peak detection can place several
    # matched anchors on different knots of the same bright extended source.
    # For astrometry anchors we want one representative match per object, so
    # greedily keep the smallest-separation pair and suppress nearby matches
    # in both VIS and Rubin frames.
    if vis_sel.size > 1 and (float(dedup_vis_px) > 0.0 or float(dedup_rubin_px) > 0.0):
        vis_xy = np.stack([vis_x[vis_sel], vis_y[vis_sel]], axis=1).astype(np.float32)
        rub_xy = np.stack([rubin_x[rub_sel], rubin_y[rub_sel]], axis=1).astype(np.float32)
        keep = np.zeros((vis_sel.size,), dtype=bool)
        kept_idx: list[int] = []
        order = np.argsort(sep)  # prefer the tightest WCS agreement
        vis_thr = float(max(0.0, dedup_vis_px))
        rub_thr = float(max(0.0, dedup_rubin_px))
        for idx in order:
            if not kept_idx:
                keep[idx] = True
                kept_idx.append(int(idx))
                continue
            prev = np.asarray(kept_idx, dtype=np.int64)
            vis_close = np.zeros((prev.size,), dtype=bool)
            rub_close = np.zeros((prev.size,), dtype=bool)
            if vis_thr > 0.0:
                dvis = np.hypot(
                    vis_xy[idx, 0] - vis_xy[prev, 0],
                    vis_xy[idx, 1] - vis_xy[prev, 1],
                )
                vis_close = dvis < vis_thr
            if rub_thr > 0.0:
                drub = np.hypot(
                    rub_xy[idx, 0] - rub_xy[prev, 0],
                    rub_xy[idx, 1] - rub_xy[prev, 1],
                )
                rub_close = drub < rub_thr
            if np.any(vis_close) or np.any(rub_close):
                continue
            keep[idx] = True
            kept_idx.append(int(idx))
        vis_sel = vis_sel[keep]
        rub_sel = rub_sel[keep]
        dra = dra[keep]
        ddec = ddec[keep]
        sep = sep[keep]

    return {
        "rubin_xy": np.stack([rubin_x[rub_sel], rubin_y[rub_sel]], axis=1).astype(np.float32),
        "vis_xy": np.stack([vis_x[vis_sel], vis_y[vis_sel]], axis=1).astype(np.float32),
        "offsets": np.stack([dra, ddec], axis=1).astype(np.float32),
        "sep_arcsec": sep.astype(np.float32),
    }


def build_detection_image(
    rubin_cube: np.ndarray,
    detect_bands: List[str],
    clip_sigma: float = 8.0,
) -> np.ndarray:
    """
    Build a shared Rubin detection image from multiple bands.
    """
    parts = []
    for band in detect_bands:
        idx = RUBIN_BAND_ORDER.index(band.split("_", 1)[1])
        if idx >= rubin_cube.shape[0]:
            continue
        img = np.nan_to_num(_to_float32(rubin_cube[idx]), nan=0.0)
        med = float(np.median(img))
        sig = robust_sigma(img)
        z = np.clip((img - med) / sig, 0.0, float(max(0.5, clip_sigma)))
        parts.append(z.astype(np.float32, copy=False))
    if not parts:
        raise ValueError("No valid Rubin bands available for detection image.")
    return np.mean(np.stack(parts, axis=0), axis=0).astype(np.float32, copy=False)


def refine_centroids_in_band(
    image: np.ndarray,
    seed_xy: np.ndarray,
    radius: int = 3,
    flux_floor_sigma: float = 1.5,
) -> np.ndarray:
    """
    Re-center source positions within a small local window in the target band.
    """
    img = np.asarray(image, dtype=np.float32)
    H, W = img.shape
    out = np.asarray(seed_xy, dtype=np.float32).copy()
    global_sig = robust_sigma(img)
    r = max(1, int(radius))
    for i, (x0f, y0f) in enumerate(seed_xy):
        x0 = int(round(float(x0f)))
        y0 = int(round(float(y0f)))
        xa = max(0, x0 - r)
        xb = min(W, x0 + r + 1)
        ya = max(0, y0 - r)
        yb = min(H, y0 + r + 1)
        if xa >= xb or ya >= yb:
            continue
        patch = img[ya:yb, xa:xb]
        if patch.size == 0:
            continue
        bg = float(np.percentile(patch, 30))
        w = np.clip(patch - bg, 0.0, None)
        if float(w.sum()) <= float(flux_floor_sigma) * global_sig:
            continue
        gy, gx = np.mgrid[ya:yb, xa:xb]
        out[i, 0] = float((w * gx).sum() / w.sum())
        out[i, 1] = float((w * gy).sum() / w.sum())
    return out
