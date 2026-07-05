"""MER Q1 round-1 detection labels.

Turns the public Euclid **Q1** MER catalogue into per-tile CenterNet training
labels, in the same normalized VIS-frame convention the classical label
functions use (``_pseudo_labels_vis``): ``centroids[M,2] = [x/(W-1), y/(H-1)]``,
classes all 0.

Why this is the clean choice now: the imaging is Euclid Q1 (see
``io/refetch_euclid_q1.py``), so the MER **Q1** catalogue is the *same release*
as the pixels — no DR1-imaging-vs-Q1-catalogue mismatch. Each tile stores its
own ``wcs_VIS``, so labels are placed by projecting catalogue (ra,dec) onto that
tile's VIS grid.

Cleaning / placement:
  - keep only reliable rows: ``vis_det == 1 & spurious_flag == 0``;
  - project (ra,dec) -> VIS pixel via the tile's WCS, keep in-frame sources;
  - **snap point-like sources to the local VIS flux max** (±``snap_half`` px) so
    the head learns peak-centered localization (MER centroids sit slightly off
    the peak; badly so for extended galaxies) — extended sources keep the
    catalogue centroid.

Catalogue columns used: ra, dec, vis_det, spurious_flag, point_like_flag,
mag_vis, flux_detection_total.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def load_mer_catalogue(fits_path):
    """Load the compact MER Q1 catalogue into plain arrays (cheap to pass to workers)."""
    d = fits.open(fits_path)[1].data
    return {
        "ra": np.asarray(d["ra"], np.float64),
        "dec": np.asarray(d["dec"], np.float64),
        "vis_det": np.asarray(d["vis_det"], np.int32),
        "spurious_flag": np.asarray(d["spurious_flag"], np.int32),
        "point_like_flag": np.asarray(d["point_like_flag"], np.int32),
        "mag_vis": np.asarray(d["mag_vis"], np.float32),
    }


def _snap_to_peak(vis_img, xs, ys, half):
    """Move each (x,y) to the brightest pixel in a (2*half+1) window; returns new x,y."""
    H, W = vis_img.shape
    nx = xs.copy().astype(np.float32)
    ny = ys.copy().astype(np.float32)
    for i in range(len(xs)):
        xi = int(np.clip(round(float(xs[i])), 0, W - 1))
        yi = int(np.clip(round(float(ys[i])), 0, H - 1))
        x0, x1 = max(0, xi - half), min(W, xi + half + 1)
        y0, y1 = max(0, yi - half), min(H, yi + half + 1)
        patch = vis_img[y0:y1, x0:x1]
        if patch.size == 0 or not np.isfinite(patch).any():
            continue
        dy, dx = np.unravel_index(int(np.nanargmax(patch)), patch.shape)
        nx[i] = x0 + dx
        ny[i] = y0 + dy
    return nx, ny


def mer_labels_for_tile(vis_img, wcs_header, cat, mag_cap=None,
                        snap_half=2, snap_point_like=True, margin_deg=0.03):
    """Return (centroids_norm [M,2], classes [M], H, W) for one VIS tile.

    vis_img     : (H,W) VIS image (NaNs ok)
    wcs_header  : the tile's ``wcs_VIS`` header string
    cat         : dict from load_mer_catalogue (global catalogue)
    mag_cap     : optional faint VIS-mag cut (keep mag_vis <= mag_cap)
    """
    vis_img = np.nan_to_num(np.asarray(vis_img, np.float32), nan=0.0)
    H, W = vis_img.shape
    wcs = WCS(fits.Header.fromstring(wcs_header))

    keep = (cat["vis_det"] == 1) & (cat["spurious_flag"] == 0)
    if mag_cap is not None:
        keep &= np.isfinite(cat["mag_vis"]) & (cat["mag_vis"] <= mag_cap)

    # cheap RA/Dec pre-filter around the tile centre before the WCS projection.
    # Use the *centre pixel* projected through the WCS (CRVAL is the mosaic
    # reference point, not this cutout's centre).
    ra, dec = cat["ra"], cat["dec"]
    ra0, dec0 = wcs.all_pix2world((W - 1) / 2.0, (H - 1) / 2.0, 0)
    ra0, dec0 = float(ra0), float(dec0)
    cosd = np.cos(np.deg2rad(dec0))
    box = keep & (np.abs(dec - dec0) < margin_deg) & \
        (np.abs(((ra - ra0 + 180) % 360) - 180) * cosd < margin_deg)
    idx = np.where(box)[0]
    if idx.size == 0:
        return (np.zeros((0, 2), np.float32), np.zeros(0, np.int64), H, W)

    x, y = wcs.all_world2pix(ra[idx], dec[idx], 0)  # 0-based pixel coords
    inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    idx, x, y = idx[inb], x[inb], y[inb]
    if idx.size == 0:
        return (np.zeros((0, 2), np.float32), np.zeros(0, np.int64), H, W)

    if snap_point_like:
        pl = cat["point_like_flag"][idx] == 1
        if pl.any():
            sx, sy = _snap_to_peak(vis_img, x[pl], y[pl], snap_half)
            x = x.astype(np.float32); y = y.astype(np.float32)
            x[pl], y[pl] = sx, sy

    centroids = np.stack([x / max(W - 1, 1), y / max(H - 1, 1)], axis=1).astype(np.float32)
    np.clip(centroids, 0.0, 1.0, out=centroids)   # guard sub-pixel edge sources
    classes = np.zeros(len(centroids), dtype=np.int64)
    return centroids, classes, H, W
