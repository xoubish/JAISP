"""WCS parsing and postage-stamp cutouts on the native VIS grid."""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def parse_wcs(hdr):
    """Parse the WCS stored in a tile NPZ: a FITS header card string (Euclid tiles,
    e.g. 'wcs_VIS') or a dict of keywords (Rubin tiles, 'wcs_hdr', stored as a 0-d
    object array)."""
    if isinstance(hdr, np.ndarray) and hdr.dtype == object:
        hdr = hdr.item()
    if isinstance(hdr, dict):
        return WCS(fits.Header(hdr))
    return WCS(fits.Header.fromstring(str(hdr)))


def pixel_scale_arcsec(wcs):
    return float(np.mean(np.linalg.norm(wcs.pixel_scale_matrix, axis=0)) * 3600.0)


def world_to_pixel(wcs, ra, dec):
    x, y = wcs.world_to_pixel_values(ra, dec)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def cut_stamp(image, xc, yc, size):
    """Integer-centred size x size cutout. Returns None if it would fall off-edge.

    Integer centring keeps the pixels exact (no interpolation), which pairs with the
    exact dihedral augmentation used in training. Sub-pixel centring is unnecessary
    for a total-flux regression baseline.
    """
    half = size // 2
    ix, iy = int(round(xc)), int(round(yc))
    x0, x1 = ix - half, ix - half + size
    y0, y1 = iy - half, iy - half + size
    h, w = image.shape
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return image[y0:y1, x0:x1]


def border_distance(xc, yc, shape, size):
    """Distance (px) from the stamp edge to the tile edge; larger = more centred."""
    half = size // 2
    h, w = shape
    return min(xc - half, yc - half, (w - 1 - xc) - half, (h - 1 - yc) - half)
