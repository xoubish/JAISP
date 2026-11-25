"""Access to Euclid MER PSF products (Q1 format).

Two products per tile and band:
  GRID-PSF    : a mosaic-sized image with PSF stamps (STMPSIZE px) placed on a regular
                grid; table 'Stamps information' (x, y, FWHM) gives the stamp centres in
                mosaic pixels.  Use for interpolation at any position.
  CATALOG-PSF : a packed image of one stamp per source; table (x_center, y_center, x, y,
                RA, Dec, FWHM): (x_center, y_center) is the stamp centre in the packed
                image, (x, y) the source position in the mosaic, with RA/Dec.
The stamps are on the 0.1"/px MER grid (mosaic WCS in the image header).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


class MerPsf:
    def __init__(self, psf_root, tile: int, band: str, kind: str = "GRID"):
        band_tag = "VIS" if band.upper() == "VIS" else f"NIR-{band.upper()}"
        pat = str(Path(psf_root) / str(tile) / f"EUC_MER_{kind}-PSF-{band_tag}_TILE{tile}-*.fits.gz")
        fn = sorted(glob.glob(pat))
        if not fn:
            raise FileNotFoundError(pat)
        self.file = fn[-1]
        h = fits.open(self.file, memmap=True)
        self.hdr = h["PSF image"].header
        self.img = h["PSF image"].data           # memmap (gz: loads lazily but fully on first access)
        self.tab = h["Stamps information"].data
        self.size = int(self.hdr["STMPSIZE"])
        self.kind = kind
        self.band = band
        self.wcs = WCS(self.hdr)                 # mosaic WCS (grid product) – same CRVAL/CRPIX as the tile mosaic
        self.fwhm_global = float(self.hdr.get("FWHM", np.nan))
        if kind == "GRID":
            self.xy = np.column_stack([self.tab["x"], self.tab["y"]])
        else:
            self.xy = np.column_stack([self.tab["x"], self.tab["y"]])
            self.xy_center = np.column_stack([self.tab["x_center"], self.tab["y_center"]])
            self.radec = np.column_stack([self.tab["RA"], self.tab["Dec"]])

    def _cut(self, cx, cy):
        r = self.size // 2
        x0, y0 = int(round(cx)) - r, int(round(cy)) - r
        st = np.array(self.img[y0:y0 + self.size, x0:x0 + self.size], float)
        return st / st.sum()

    def stamp_at_mosaic_xy(self, x, y):
        """Nearest stamp (GRID: nearest grid node; CATALOG: nearest source) at mosaic pixel (x, y)."""
        i = int(np.argmin((self.xy[:, 0] - x) ** 2 + (self.xy[:, 1] - y) ** 2))
        if self.kind == "GRID":
            return self._cut(self.xy[i, 0], self.xy[i, 1]), float(self.tab["FWHM"][i])
        return self._cut(self.xy_center[i, 0], self.xy_center[i, 1]), float(self.tab["FWHM"][i])

    def stamp_at_sky(self, ra, dec):
        x, y = self.wcs.world_to_pixel_values(ra, dec)
        return self.stamp_at_mosaic_xy(float(x), float(y))

    def stamp_for_source(self, ra, dec, max_sep_arcsec=0.5):
        """CATALOG product only: the stamp of the catalogued source at (ra, dec)."""
        assert self.kind == "CATALOG"
        d2 = ((self.radec[:, 0] - ra) * np.cos(np.deg2rad(dec))) ** 2 + (self.radec[:, 1] - dec) ** 2
        i = int(np.argmin(d2))
        if np.sqrt(d2[i]) * 3600 > max_sep_arcsec:
            return None, None
        return self._cut(self.xy_center[i, 0], self.xy_center[i, 1]), float(self.tab["FWHM"][i])


def tile_for(psf_root, ra, dec):
    """Pick the tile whose grid-PSF mosaic WCS contains (ra, dec), preferring the one nearest its centre."""
    best = None
    for d in sorted(glob.glob(str(Path(psf_root) / "*"))):
        fn = sorted(glob.glob(d + "/EUC_MER_GRID-PSF-VIS_*.fits.gz"))
        if not fn:
            continue
        hdr = fits.getheader(fn[-1], "PSF image")
        w = WCS(hdr); x, y = w.world_to_pixel_values(ra, dec)
        if 0 <= x < hdr["NAXIS1"] and 0 <= y < hdr["NAXIS2"]:
            dist = (x - hdr["NAXIS1"] / 2) ** 2 + (y - hdr["NAXIS2"] / 2) ** 2
            if best is None or dist < best[0]:
                best = (dist, int(Path(d).name))
    return None if best is None else best[1]
