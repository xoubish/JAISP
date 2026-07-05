"""Bright-star + saturation mask for detection.

Replaces the old geometric ``spike_veto`` (which suppressed real bright/large
sources) with a physically-motivated mask: disks around Gaia bright stars whose
radius scales with brightness, plus a high-pixel saturation catch for very
bright cores (incl. non-Gaia sources). Used to (a) drop spurious spike/halo
detections at inference/eval and (b) avoid penalizing the head near bright
stars during training / injection.

Gaia cache: ``data/gaia_ecdfs_astrometry_cache.npz`` (source_id, ra, dec,
ref_epoch, pmra, pmdec, phot_g_mean_mag). Proper motion (2016->survey epoch,
~tens of mas over 8 yr) is negligible vs the tens-to-hundreds-of-pixel mask
radius, so it is not applied here.
"""
from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def load_gaia_cache(path):
    d = np.load(path, allow_pickle=True)
    return {"ra": np.asarray(d["ra"], np.float64),
            "dec": np.asarray(d["dec"], np.float64),
            "g": np.asarray(d["phot_g_mean_mag"], np.float32)}


def bright_star_saturation_mask(vis_img, wcs_header, gaia,
                                g_bright=18.0, r0=15.0, slope=16.0, rmax=150.0,
                                sat_thresh=50.0, margin_deg=0.03):
    """Return a boolean [H,W] mask of bright-star / saturated regions (True = masked).

    r_px(G) = clip(r0 + slope*(g_bright - G), r0, rmax): brighter stars -> larger disks.
    sat_thresh: also mask pixels with VIS value above this (saturated cores);
                None disables the saturation catch.
    """
    vis_img = np.asarray(vis_img, np.float32)
    H, W = vis_img.shape
    wcs = WCS(fits.Header.fromstring(wcs_header))
    mask = np.zeros((H, W), dtype=bool)

    ra, dec, g = gaia["ra"], gaia["dec"], gaia["g"]
    ra0, dec0 = wcs.all_pix2world((W - 1) / 2.0, (H - 1) / 2.0, 0)
    ra0, dec0 = float(ra0), float(dec0)
    cosd = np.cos(np.deg2rad(dec0))
    sel = (g < g_bright) & (np.abs(dec - dec0) < margin_deg) & \
        (np.abs(((ra - ra0 + 180) % 360) - 180) * cosd < margin_deg)
    if sel.any():
        x, y = wcs.all_world2pix(ra[sel], dec[sel], 0)
        gg = g[sel]
        yy, xx = np.ogrid[:H, :W]
        for xi, yi, gi in zip(x, y, gg):
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            r = float(np.clip(r0 + slope * (g_bright - gi), r0, rmax))
            mask |= ((xx - xi) ** 2 + (yy - yi) ** 2) <= r * r

    if sat_thresh is not None:
        mask |= np.nan_to_num(vis_img, nan=0.0) > sat_thresh

    return mask
