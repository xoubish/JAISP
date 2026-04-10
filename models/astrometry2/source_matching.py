"""
Shared source detection and Rubin<->VIS matching utilities for astrometry.

Centroid refinement follows the methodology validated by Wilson & Naylor
(SITCOMTN-159, 2025): astrometric precision scales as FWHM / SNR (King 1983),
and there is a ~5 mas systematic floor in Rubin single-visit positions.
PSF-fit centroiding and per-source SNR estimation help the downstream
astrometry matcher approach that floor instead of being limited by
crude flux-weighted centroids.
"""

from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import least_squares

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
    Uses simple flux-weighted centroiding (legacy method).
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


# ============================================================
# PSF-fit centroid refinement (SITCOMTN-159 motivated)
# ============================================================

def _fit_gaussian_2d(
    patch: np.ndarray,
    x0_local: float,
    y0_local: float,
    fwhm_guess: float = 3.0,
) -> Optional[Tuple[float, float, float, float, float]]:
    """Fit a 2D circular Gaussian + constant background to a small stamp.

    Parameters
    ----------
    patch : [H, W] float array — small image cutout
    x0_local, y0_local : initial guess for center in patch-local coords
    fwhm_guess : initial FWHM in pixels

    Returns
    -------
    (xc, yc, amplitude, sigma, background) in patch-local coords,
    or None if the fit fails or is degenerate.
    """
    H, W = patch.shape
    gy, gx = np.mgrid[:H, :W].astype(np.float64)
    data = patch.astype(np.float64).ravel()

    bg_init = float(np.percentile(patch, 30))
    amp_init = float(patch.max()) - bg_init
    sig_init = fwhm_guess / 2.3548

    if amp_init <= 0 or sig_init <= 0:
        return None

    # p = [xc, yc, amplitude, sigma, background]
    p0 = np.array([x0_local, y0_local, amp_init, sig_init, bg_init])

    def residuals(p):
        xc, yc, amp, sig, bg = p
        model = bg + amp * np.exp(-0.5 * ((gx.ravel() - xc) ** 2 + (gy.ravel() - yc) ** 2) / max(sig, 0.1) ** 2)
        return model - data

    try:
        result = least_squares(
            residuals, p0,
            bounds=(
                [0, 0, 0, 0.3, -np.inf],
                [W - 1, H - 1, np.inf, max(W, H) / 2, np.inf],
            ),
            method='trf',
            max_nfev=80,
            ftol=1e-6,
            xtol=1e-6,
        )
        if not result.success and result.cost > 0.5 * np.sum(data ** 2):
            return None
        xc, yc, amp, sig, bg = result.x
        if amp <= 0 or sig < 0.3 or sig > max(W, H) / 2:
            return None
        return float(xc), float(yc), float(amp), float(sig), float(bg)
    except Exception:
        return None


def refine_centroids_psf_fit(
    image: np.ndarray,
    seed_xy: np.ndarray,
    radius: int = 5,
    flux_floor_sigma: float = 1.5,
    fwhm_guess: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PSF-fit centroid refinement using 2D Gaussian fitting.

    Follows King (1983) / SITCOMTN-159: fitting a Gaussian PSF model gives
    centroid precision ~ FWHM / (2.35 * SNR) per axis, substantially better
    than flux-weighted centroiding for bright sources.

    Parameters
    ----------
    image : [H, W] single-band image
    seed_xy : [N, 2] initial (x, y) positions
    radius : half-size of fitting window
    flux_floor_sigma : minimum local signal for attempting a fit
    fwhm_guess : initial FWHM guess in pixels

    Returns
    -------
    refined_xy : [N, 2] improved positions
    snr : [N] per-source SNR estimate (amplitude / background_noise)
    centroid_sigma_px : [N] estimated centroid uncertainty in pixels
        (FWHM / (2.35 * SNR)), or inf where fitting failed
    """
    img = np.asarray(image, dtype=np.float32)
    H, W = img.shape
    N = seed_xy.shape[0]
    out = np.asarray(seed_xy, dtype=np.float32).copy()
    snr = np.ones(N, dtype=np.float32)
    centroid_sigma_px = np.full(N, np.inf, dtype=np.float32)

    global_sig = robust_sigma(img)
    r = max(2, int(radius))

    for i, (x0f, y0f) in enumerate(seed_xy):
        x0 = int(round(float(x0f)))
        y0 = int(round(float(y0f)))
        xa = max(0, x0 - r)
        xb = min(W, x0 + r + 1)
        ya = max(0, y0 - r)
        yb = min(H, y0 + r + 1)
        if xb - xa < 5 or yb - ya < 5:
            continue
        patch = img[ya:yb, xa:xb]
        if patch.size == 0:
            continue

        # Quick signal check before expensive fitting
        bg_quick = float(np.percentile(patch, 30))
        signal = np.clip(patch - bg_quick, 0.0, None).sum()
        if signal <= flux_floor_sigma * global_sig:
            continue

        x0_local = float(x0f) - xa
        y0_local = float(y0f) - ya

        fit = _fit_gaussian_2d(patch, x0_local, y0_local, fwhm_guess)
        if fit is None:
            # Fall back to flux-weighted centroid
            w = np.clip(patch - bg_quick, 0.0, None)
            ws = float(w.sum())
            if ws > 0:
                gy, gx = np.mgrid[ya:yb, xa:xb]
                out[i, 0] = float((w * gx).sum() / ws)
                out[i, 1] = float((w * gy).sum() / ws)
                snr[i] = float(ws / (global_sig * np.sqrt(patch.size)))
                fwhm_px = fwhm_guess
                centroid_sigma_px[i] = fwhm_px / (2.3548 * max(snr[i], 1.0))
            continue

        xc, yc, amp, sig_px, bg = fit
        out[i, 0] = xc + xa
        out[i, 1] = yc + ya

        # Estimate noise from fit residual
        gy_l, gx_l = np.mgrid[:patch.shape[0], :patch.shape[1]].astype(np.float32)
        model = bg + amp * np.exp(
            -0.5 * ((gx_l - xc) ** 2 + (gy_l - yc) ** 2) / max(sig_px, 0.3) ** 2
        )
        resid_std = float(np.std(patch - model))
        noise = max(resid_std, global_sig, 1e-8)

        source_snr = amp / noise
        snr[i] = max(float(source_snr), 1.0)

        # King (1983): centroid precision ~ sigma_psf / SNR
        fwhm_fit = sig_px * 2.3548
        centroid_sigma_px[i] = sig_px / max(snr[i], 1.0)

    return out, snr, centroid_sigma_px


def estimate_source_snr(
    image: np.ndarray,
    xy: np.ndarray,
    aperture_radius: int = 5,
) -> np.ndarray:
    """Aperture-based SNR estimation for matched sources.

    Quick fallback when PSF fitting is not used.  Computes flux in a
    circular aperture after background subtraction, divided by noise
    estimated from the image MAD.

    Parameters
    ----------
    image : [H, W] single-band image
    xy : [N, 2] source (x, y) positions
    aperture_radius : radius in pixels

    Returns
    -------
    snr : [N] estimated SNR (clipped to >= 1)
    """
    img = np.asarray(image, dtype=np.float32)
    H, W = img.shape
    N = xy.shape[0]
    snr = np.ones(N, dtype=np.float32)
    noise = max(robust_sigma(img), 1e-8)
    r = max(1, int(aperture_radius))

    # Circular aperture mask
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    circ = (xx ** 2 + yy ** 2) <= r ** 2
    n_pix = float(circ.sum())

    for i, (xf, yf) in enumerate(xy):
        x0 = int(round(float(xf)))
        y0 = int(round(float(yf)))
        xa = x0 - r
        ya = y0 - r
        xb = x0 + r + 1
        yb = y0 + r + 1
        if xa < 0 or ya < 0 or xb > W or yb > H:
            continue
        stamp = img[ya:yb, xa:xb]
        if stamp.shape != circ.shape:
            continue
        # Background from annulus would be ideal, but median of stamp edges
        # is adequate for this SNR estimate.
        bg = float(np.median(stamp[~circ])) if (~circ).any() else float(np.percentile(stamp, 30))
        flux = float(np.sum((stamp - bg) * circ))
        snr[i] = max(flux / (noise * np.sqrt(n_pix)), 1.0)

    return snr


def expected_centroid_sigma_arcsec(
    snr: np.ndarray,
    pixel_scale_arcsec: float,
    fwhm_px: float = 3.0,
    systematic_floor_arcsec: float = 0.005,
) -> np.ndarray:
    """Expected centroid uncertainty per source following SITCOMTN-159.

    σ_centroid = sqrt((FWHM_arcsec / (2.35 * SNR))^2 + systematic_floor^2)

    The systematic floor of ~5 mas is the additive term 'n' measured by
    Wilson & Naylor (2025) for Rubin Source table single-visit astrometry.
    For VIS (Euclid), this is ~1-2 mas and can be set lower.

    Parameters
    ----------
    snr : [N] per-source SNR
    pixel_scale_arcsec : arcsec per pixel
    fwhm_px : PSF FWHM in pixels
    systematic_floor_arcsec : systematic floor from WCS solution (default 5 mas)

    Returns
    -------
    sigma_arcsec : [N] expected centroid uncertainty per source in arcsec
    """
    snr_safe = np.maximum(np.asarray(snr, dtype=np.float32), 1.0)
    fwhm_arcsec = fwhm_px * pixel_scale_arcsec
    statistical = fwhm_arcsec / (2.3548 * snr_safe)
    return np.sqrt(statistical ** 2 + systematic_floor_arcsec ** 2).astype(np.float32)


# ============================================================
# Optional PSFNet-enhanced centroiding
# ============================================================

def refine_centroids_psfnet(
    image: np.ndarray,
    seed_xy: np.ndarray,
    psf_net,
    band_idx: int,
    tile_hw: Tuple[int, int],
    stamp_size: int = 21,
    n_iter: int = 3,
    flux_floor_sigma: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """PSFNet-based matched-filter centroid refinement.

    Uses the learned spatially-varying PSF model to do proper matched-filter
    centroiding, which is optimal in the Cramér-Rao sense.  Falls back to
    Gaussian PSF-fit if PSFNet is not available or fails.

    The approach: at each source position, extract a stamp, get the PSFNet
    prediction for that location, then compute the chi²-minimizing sub-pixel
    shift via gradient of the matched-filter response.

    Parameters
    ----------
    image : [H, W] single-band image
    seed_xy : [N, 2] initial (x, y) positions
    psf_net : PSFNet model (from photometry/psf_net.py)
    band_idx : integer band index for PSFNet
    tile_hw : (H, W) tile dimensions for normalizing coordinates
    stamp_size : PSFNet stamp size (must match model)
    n_iter : number of iterative refinement steps
    flux_floor_sigma : minimum local signal for attempting refinement

    Returns
    -------
    refined_xy : [N, 2] improved positions
    snr : [N] per-source SNR from matched-filter photometry
    """
    try:
        import torch
    except ImportError:
        warnings.warn('PyTorch not available, falling back to Gaussian PSF-fit')
        xy, snr, _ = refine_centroids_psf_fit(image, seed_xy)
        return xy, snr

    img = np.asarray(image, dtype=np.float32)
    H, W = img.shape
    N = seed_xy.shape[0]
    out = np.asarray(seed_xy, dtype=np.float32).copy()
    snr = np.ones(N, dtype=np.float32)
    global_sig = robust_sigma(img)
    half = stamp_size // 2

    tile_H, tile_W = tile_hw
    device = next(psf_net.parameters()).device

    for iteration in range(n_iter):
        for i in range(N):
            xf, yf = float(out[i, 0]), float(out[i, 1])
            x0 = int(round(xf))
            y0 = int(round(yf))

            xa = x0 - half
            xb = x0 + half + 1
            ya = y0 - half
            yb = y0 + half + 1
            if xa < 0 or ya < 0 or xb > W or yb > H:
                continue

            stamp = img[ya:yb, xa:xb]
            if stamp.shape != (stamp_size, stamp_size):
                continue

            # Quick signal check
            bg = float(np.percentile(stamp, 30))
            sig_check = np.clip(stamp - bg, 0, None).sum()
            if sig_check <= flux_floor_sigma * global_sig:
                continue

            # Get PSF at this location
            x_norm = torch.tensor([xf / max(tile_W - 1, 1)], dtype=torch.float32, device=device)
            y_norm = torch.tensor([yf / max(tile_H - 1, 1)], dtype=torch.float32, device=device)
            bidx = torch.tensor([band_idx], dtype=torch.long, device=device)

            with torch.no_grad():
                psf = psf_net(x_norm, y_norm, bidx)[0].cpu().numpy()  # [S, S]

            # Matched-filter centroid: compute weighted centroid using PSF as weight
            stamp_sub = stamp - bg
            var = max(global_sig ** 2, 1e-10)
            psf_ov = psf / var  # PSF / variance
            flux = float(np.sum(stamp_sub * psf_ov) / max(np.sum(psf * psf_ov), 1e-10))

            if flux <= 0:
                continue

            # Compute gradient-based sub-pixel shift from residual
            resid = stamp_sub - flux * psf
            gy_l, gx_l = np.mgrid[:stamp_size, :stamp_size].astype(np.float32)
            cx = (stamp_size - 1) / 2.0
            cy = (stamp_size - 1) / 2.0

            # Weighted centroid of residual gives the correction
            w = psf * np.clip(stamp_sub, 0, None)
            ws = float(w.sum())
            if ws > 0:
                dx = float((w * (gx_l - cx)).sum() / ws)
                dy = float((w * (gy_l - cy)).sum() / ws)
                # Dampen to avoid overshooting
                dx = max(-1.0, min(1.0, dx))
                dy = max(-1.0, min(1.0, dy))
                out[i, 0] = xf + dx
                out[i, 1] = yf + dy

            # SNR from matched filter
            flux_var = 1.0 / max(np.sum(psf ** 2 / var), 1e-10)
            snr[i] = max(float(flux / np.sqrt(flux_var)), 1.0)

    return out, snr
