"""VIS/NIR cutouts from MER BGSUB mosaics (0.1"/px VIS, 0.3"/px NIR), same interface as TileStamps.

Used where the JAISP tiles do not reach (e.g. the overlap strips between neighbouring NISP
observations).  Mosaics are loaded lazily, one tile per band, and kept in memory (1.5 GB per VIS tile).
"""
import glob
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

BAND_TAG = {"VIS": "VIS", "Y": "NIR-Y", "J": "NIR-J", "H": "NIR-H"}


class MerMosaicStamps:
    def __init__(self, mer_root):
        self.root = Path(mer_root)
        self.tiles = {}            # tile id -> {band: path}
        for d in sorted(glob.glob(str(self.root / "*"))):
            tid = Path(d).name
            if not tid.isdigit():
                continue
            self.tiles[int(tid)] = {b: sorted(glob.glob(f"{d}/EUC_MER_BGSUB-MOSAIC-{t}_TILE{tid}-*.fits.gz")) for b, t in BAND_TAG.items()}
        self._hdr = {}; self._img = {}

    def _header(self, tile, band):
        k = (tile, band)
        if k not in self._hdr:
            fn = self.tiles[tile][band]
            self._hdr[k] = (fits.getheader(fn[-1]), WCS(fits.getheader(fn[-1]))) if fn else (None, None)
        return self._hdr[k]

    def _image(self, tile, band):
        k = (tile, band)
        if k not in self._img:
            fn = self.tiles[tile][band]
            self._img[k] = np.asarray(fits.getdata(fn[-1]), dtype=np.float32) if fn else None
        return self._img[k]

    def tile_of(self, ra, dec, band="VIS", margin=64):
        best = None
        for tile in self.tiles:
            hdr, w = self._header(tile, band)
            if hdr is None:
                continue
            x, y = w.world_to_pixel_values(ra, dec); x, y = float(x), float(y)
            if margin <= x < hdr["NAXIS1"] - margin and margin <= y < hdr["NAXIS2"] - margin:
                dist = (x - hdr["NAXIS1"] / 2) ** 2 + (y - hdr["NAXIS2"] / 2) ** 2
                if best is None or dist < best[0]:
                    best = (dist, tile, x, y)
        return best

    def stamp(self, ra, dec, band="VIS", half_arcsec=3.0):
        hit = self.tile_of(ra, dec, band)
        if hit is None:
            return None, None
        _, tile, x, y = hit
        hdr, w = self._header(tile, band); img = self._image(tile, band)
        scale = abs(float(hdr.get("CD1_1", hdr.get("CDELT1", 2.7777e-5)))) * 3600.0
        h = int(round(half_arcsec / scale))
        x0, y0 = int(round(x)) - h, int(round(y)) - h
        if x0 < 0 or y0 < 0 or x0 + 2 * h + 1 > img.shape[1] or y0 + 2 * h + 1 > img.shape[0]:
            return None, None
        cut = np.array(img[y0:y0 + 2 * h + 1, x0:x0 + 2 * h + 1], float)
        wc = w.deepcopy(); wc.wcs.crpix = wc.wcs.crpix - np.array([x0, y0])
        return cut, wc


class StampUnion:
    """Try several stamp sources in order (e.g. JAISP tiles first, MER mosaics where they do not reach)."""

    def __init__(self, *providers):
        self.providers = providers
        self.counts = {}

    def stamp(self, ra, dec, band="VIS", half_arcsec=3.0):
        for i, p in enumerate(self.providers):
            cut, wcs = p.stamp(ra, dec, band=band, half_arcsec=half_arcsec)
            if cut is not None:
                self.counts[i] = self.counts.get(i, 0) + 1
                return cut, wcs
        self.counts[-1] = self.counts.get(-1, 0) + 1
        return None, None


def mer_sky_catalog(mer_root, hmax=23.5, exclude=None, match_arcsec=0.6, cache=True):
    """Sky catalogue entries (same dict format as `sky_catalog`) from the MER FINAL catalogues of all
    tiles under `mer_root`: H < hmax (SIR's selection), magnitudes from the TEMPLFIT fluxes (uJy),
    star/galaxy from POINT_LIKE_FLAG.  Sources within `match_arcsec` of an entry of `exclude` are
    dropped, so the result supplements a SIR location-table catalogue outside its footprint."""
    import json
    from astropy.io import fits
    out_path = Path(mer_root) / f"mer_sky_catalog_h{hmax:.1f}.json"
    if cache and out_path.exists():
        cat = json.load(open(out_path))
    else:
        cat = []
        for d in sorted(glob.glob(str(Path(mer_root) / "*"))):
            fn = sorted(glob.glob(f"{d}/EUC_MER_FINAL-CAT_*.fits.gz"))
            if not fn:
                continue
            t = fits.getdata(fn[-1])
            fh = np.asarray(t["FLUX_H_TEMPLFIT"], float); ok = fh > 0
            mh = np.where(ok, 23.9 - 2.5 * np.log10(np.where(ok, fh, 1)), 99.0)
            sel = np.where(mh < hmax)[0]
            for i in sel:
                mags = {}
                for b in ("Y", "J", "H"):
                    f = float(t[f"FLUX_{b}_TEMPLFIT"][i])
                    if f > 0:
                        mags[b] = 23.9 - 2.5 * np.log10(f)
                a = float(t["SEMIMAJOR_AXIS"][i]) * 0.1; e = float(t["ELLIPTICITY"][i])
                pl = int(t["POINT_LIKE_FLAG"][i])
                cat.append(dict(id=f"mer{int(t['OBJECT_ID'][i])}", ra=float(t["RIGHT_ASCENSION"][i]), dec=float(t["DECLINATION"][i]),
                                type=1 if pl == 1 else 2, amaj=a, amin=a * (1 - e) if np.isfinite(e) else a, mags=mags,
                                tile=int(Path(d).name)))
        if cache:
            json.dump(cat, open(out_path, "w"))
    if exclude:
        era = np.deg2rad([s["ra"] for s in exclude]); edec = np.deg2rad([s["dec"] for s in exclude])
        from scipy.spatial import cKDTree
        def xyz(ra, dec):
            return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)
        tree = cKDTree(xyz(era, edec))
        cra = np.deg2rad([s["ra"] for s in cat]); cdec = np.deg2rad([s["dec"] for s in cat])
        dist, _ = tree.query(xyz(cra, cdec))
        keep = dist > np.deg2rad(match_arcsec / 3600.0)
        cat = [s for s, k in zip(cat, keep) if k]
    return cat
