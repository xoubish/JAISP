"""
PSFField-based sub-pixel centroid refinement.

Replaces the Gaussian-fit ``refine_centroids_psf_fit()`` from
``astrometry2.source_matching`` with a template-matching centroider that uses
the learned, spatially-varying, chromatic PSFField.

Usage
-----
    from psf.centroid_refinement import refine_centroids_psf_field

    refined_xy, snr, fwhm = refine_centroids_psf_field(
        image, seed_xy, psf_field,
        band_name='euclid_VIS', tile_hw=(1084, 1084),
    )
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import shift as ndi_shift

from .psf_field import (
    PSFField, BAND_TO_IDX, BAND_PX_SCALE, N_BANDS,
    normalise_psf,
)


def refine_centroids_psf_field(
    image: np.ndarray,
    seed_xy: np.ndarray,
    psf_field: PSFField,
    band_name: str,
    tile_hw: Tuple[int, int],
    device: torch.device = None,
    radius: int = 5,
    n_iter: int = 3,
    flux_floor_sigma: float = 1.5,
    sed_vec: torch.Tensor = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterative sub-pixel centroid refinement using PSFField-rendered templates.

    For each source, fits ``(dx, dy, flux)`` against the data cutout by
    shifting the PSFField-rendered template via bilinear interpolation and
    computing a Gauss–Newton step on the chi² surface.

    Parameters
    ----------
    image       : [H, W] single-band image (background-subtracted or raw)
    seed_xy     : [N, 2] initial (x, y) pixel positions
    psf_field   : trained PSFField in eval mode
    band_name   : e.g. 'euclid_VIS', 'rubin_r'
    tile_hw     : (H, W) of the full tile (for normalizing tile positions)
    device      : torch device for PSFField
    radius      : half-size of fitting box in pixels
    n_iter      : sub-pixel refinement iterations per source
    flux_floor_sigma : skip sources fainter than this × global noise
    sed_vec     : [N, 10] per-source SED, or None for mean-SED

    Returns
    -------
    refined_xy : [N, 2]  sub-pixel-refined (x, y)
    snr        : [N]     peak signal-to-noise
    psf_fwhm   : [N]     effective FWHM from PSF template (arcsec)
    """
    if device is None:
        device = next(psf_field.parameters()).device

    H_tile, W_tile = tile_hw
    img = np.asarray(image, dtype=np.float32)
    H, W = img.shape
    seed_xy = np.asarray(seed_xy, dtype=np.float32).copy()
    N = seed_xy.shape[0]
    bi = BAND_TO_IDX.get(band_name, 0)
    px_scale = float(BAND_PX_SCALE[bi])

    # Global noise estimate
    med = float(np.median(img))
    mad = float(np.median(np.abs(img - med)))
    global_sig = max(1.4826 * mad, 1e-10)

    refined = seed_xy.copy()
    snr_out = np.ones(N, dtype=np.float32)
    fwhm_out = np.full(N, 3.0 * px_scale, dtype=np.float32)

    # Render PSF stamps for all sources at once
    psf_field.eval()
    r = int(max(2, radius))
    box = 2 * r + 1

    with torch.no_grad():
        x_norm = torch.from_numpy(seed_xy[:, 0] / max(W_tile - 1, 1)).float().to(device)
        y_norm = torch.from_numpy(seed_xy[:, 1] / max(H_tile - 1, 1)).float().to(device)
        tile_pos = torch.stack([x_norm, y_norm], dim=-1)
        band_idx = torch.full((N,), bi, dtype=torch.long, device=device)

        if sed_vec is None:
            sed = torch.zeros(N, N_BANDS, device=device)
        else:
            sed = sed_vec.to(device)

        # Render at zero centroid offset → gives the PSF centered on stamp
        psf_stamps = psf_field.render_stamps(
            centroids_arcsec=torch.zeros(N, 2, device=device),
            tile_pos=tile_pos,
            band_idx=band_idx,
            sed_vec=sed,
            stamp_size=box,
            px_scale=px_scale,
            sub_grid=4,
            apply_dcr=False,
        ).cpu().numpy()

    # Estimate FWHM per source from the PSF stamp
    for i in range(N):
        row = psf_stamps[i, r, :]
        peak = row.max()
        if peak > 0:
            above = row >= 0.5 * peak
            fwhm_out[i] = float(above.sum()) * px_scale

    # Per-source iterative centroid fitting
    for i in range(N):
        x0, y0 = float(refined[i, 0]), float(refined[i, 1])
        xi, yi = int(round(x0)), int(round(y0))

        if xi < r or xi >= W - r or yi < r or yi >= H - r:
            continue

        cutout = img[yi - r:yi + r + 1, xi - r:xi + r + 1].copy()
        bg = float(np.median(cutout))
        cutout_sub = cutout - bg

        peak_val = cutout_sub[r, r]
        if peak_val < flux_floor_sigma * global_sig:
            continue
        snr_out[i] = peak_val / global_sig

        # PSF template for this source
        psf_crop = psf_stamps[i].copy()
        psf_sum = psf_crop.sum()
        if psf_sum > 0:
            psf_crop /= psf_sum

        dx_accum, dy_accum = 0.0, 0.0

        for _ in range(n_iter):
            shifted = ndi_shift(psf_crop, [dy_accum, dx_accum],
                                order=1, mode='constant', cval=0.0)
            s_sum = shifted.sum()
            if s_sum <= 0:
                break
            shifted /= s_sum

            flux = float((cutout_sub * shifted).sum() /
                         max((shifted * shifted).sum(), 1e-12))
            if flux <= 0:
                break

            resid = cutout_sub - flux * shifted
            eps = 0.1
            dpsf_dx = (ndi_shift(psf_crop, [dy_accum, dx_accum + eps],
                                 order=1, mode='constant', cval=0.0)
                       - ndi_shift(psf_crop, [dy_accum, dx_accum - eps],
                                   order=1, mode='constant', cval=0.0)) / (2 * eps)
            dpsf_dy = (ndi_shift(psf_crop, [dy_accum + eps, dx_accum],
                                 order=1, mode='constant', cval=0.0)
                       - ndi_shift(psf_crop, [dy_accum - eps, dx_accum],
                                   order=1, mode='constant', cval=0.0)) / (2 * eps)

            JtJ_xx = float((flux * dpsf_dx * flux * dpsf_dx).sum())
            JtJ_yy = float((flux * dpsf_dy * flux * dpsf_dy).sum())
            JtJ_xy = float((flux * dpsf_dx * flux * dpsf_dy).sum())
            Jtr_x = float((resid * flux * dpsf_dx).sum())
            Jtr_y = float((resid * flux * dpsf_dy).sum())

            det = JtJ_xx * JtJ_yy - JtJ_xy * JtJ_xy
            if abs(det) < 1e-20:
                break

            step_x = max(-0.5, min(0.5, (JtJ_yy * Jtr_x - JtJ_xy * Jtr_y) / det))
            step_y = max(-0.5, min(0.5, (JtJ_xx * Jtr_y - JtJ_xy * Jtr_x) / det))

            dx_accum += step_x
            dy_accum += step_y

            if abs(dx_accum) > r or abs(dy_accum) > r:
                dx_accum, dy_accum = 0.0, 0.0
                break

        refined[i, 0] = x0 + dx_accum
        refined[i, 1] = y0 + dy_accum

    return refined, snr_out, fwhm_out
