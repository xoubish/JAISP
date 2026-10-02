"""Cutout discovery and fetch for Euclid Q1 MER mosaics.

Two cache layers: a small cutout-FITS cache at
``data_dir/<band>_<ptype>_<ra>_<dec>_<size>_native-v1.fits`` (ptype is science, rms,
or flag; flag planes are gzipped as ``.fits.gz``), read directly when
present; and a fallback to S3 lazy partial reads
(``fits.open(..., use_fsspec=True)`` plus ``hdu.section``) whose result is
written back to the cutout cache. Full mosaic tiles are never persisted.

Public: ``Cutout``, ``discover_mer_mosaics``, ``fetch_cutout``,
``trim_catalog_to_cutout``.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata.utils import overlap_slices, PartialOverlapError, NoOverlapError
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa
from astropy.wcs.utils import proj_plane_pixel_scales

from .config import (
    DEFAULT_CUTOUT_DIR,
    MER_COLLECTION,
    MER_TILE_HALF_DEG,
    SIA_SEARCH_PAD_DEG,
)


@dataclass
class Cutout:
    """One band's cutout: science array, RMS (when paired), WCS, header.

    ``flag`` is the optional MER FLG (quality) plane, an integer bitmask
    aligned with ``data`` (non-zero where a pixel is saturated, a cosmic ray,
    a bad pixel, or otherwise invalid). It is only populated when
    ``fetch_cutout(..., with_flag=True)`` is used; ``build_tractor_image``
    consumes it to drop flagged pixels from the fit.
    """
    band: str
    data: np.ndarray
    rms: np.ndarray | None
    wcs: WCS
    header: fits.Header
    flag: np.ndarray | None = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def pixel_scale_arcsec(self) -> float:
        return float(abs(self.wcs.pixel_scale_matrix[0, 0]) * 3600.0)


# ---------------------------------------------------------------------------
# Tile discovery
# ---------------------------------------------------------------------------

def _classify_ptype(fname: str) -> str | None:
    if "BGSUB-MOSAIC" in fname: return "science"
    if "RMS"          in fname: return "rms"
    if "GRID-PSF"     in fname: return "psf_grid"
    if "CATALOG-PSF"  in fname: return "psf_catalog"
    if "FLAG"         in fname: return "flag"
    if "BGMOD"        in fname: return "bgmodel"
    return None


def _separation_arcsec(t, target):
    if t["s_ra"] is None or t["s_dec"] is None:
        return float("inf")
    return (SkyCoord(t["s_ra"], t["s_dec"], unit="deg")
            .separation(target).arcsec)


def _closest_tile(tiles, target):
    with_coords = [t for t in tiles if t["s_ra"] is not None and t["s_dec"] is not None]
    if not with_coords:
        return tiles[0]
    return min(with_coords, key=lambda t: _separation_arcsec(t, target))


def discover_mer_mosaics(ra: float, dec: float, half_size_deg: float,
                         *, bands=("VIS", "Y", "J", "H"), verbose=False):
    """SIA query returning dict[band][ptype] with closest tile and ``tiles`` list."""
    from .netutils import retry
    target = SkyCoord(ra=ra, dec=dec, unit="deg")
    search_radius_deg = MER_TILE_HALF_DEG + half_size_deg + SIA_SEARCH_PAD_DEG
    sia_results = retry(
        lambda: Irsa.query_sia(
            pos=(target, search_radius_deg * u.deg),
            collection=MER_COLLECTION,
        ),
        what="IRSA SIA discovery")

    raw = []
    for row in sia_results:
        band = row["energy_bandpassname"]
        if band not in bands:
            continue
        url = str(row["access_url"])
        fname = url.split("/")[-1]
        ptype = _classify_ptype(fname)
        if ptype is None:
            continue
        cloud_meta = json.loads(row["cloud_access"])
        s3_path = f"{cloud_meta['aws']['bucket_name']}/{cloud_meta['aws']['key']}"
        m = re.search(r"TILE(\d+)", fname)
        tile_id = m.group(1) if m else fname
        try:
            s_ra = float(row["s_ra"]); s_dec = float(row["s_dec"])
        except Exception:
            s_ra = s_dec = None
        raw.append(dict(band=band, ptype=ptype, tile_id=tile_id,
                        s3=s3_path, url=url, s_ra=s_ra, s_dec=s_dec,
                        fname=fname))

    cosd = float(np.cos(np.radians(dec)))
    thresh_deg = MER_TILE_HALF_DEG + half_size_deg + 0.005
    keep = []
    for r in raw:
        if r["s_ra"] is None or r["s_dec"] is None:
            keep.append(r); continue
        # Wrap the RA difference into [-180, 180].
        d_ra = ((r["s_ra"] - ra + 540.0) % 360.0) - 180.0
        if (abs(d_ra) * cosd <= thresh_deg
                and abs(r["s_dec"] - dec) <= thresh_deg):
            keep.append(r)
    raw = keep

    by_bp = defaultdict(list)
    seen = set()
    for r in raw:
        key = (r["band"], r["ptype"], r["tile_id"])
        if key in seen:
            continue
        seen.add(key)
        by_bp[(r["band"], r["ptype"])].append(r)

    products = {}
    for (band, ptype), tiles in by_bp.items():
        products.setdefault(band, {})
        closest = _closest_tile(tiles, target)
        products[band][ptype] = {**closest, "tiles": tiles}

    if verbose:
        print(f"SIA: {len(raw)} matching products; "
              f"bands available: {', '.join(sorted(products))}")
    return products


# ---------------------------------------------------------------------------
# Cutout fetch: local FITS cache, else lazy partial read from S3
# ---------------------------------------------------------------------------

def _open_mosaic(s3_path: str, mosaic_cache_dir: Path | None):
    """Open a MER mosaic, preferring the local mosaic cache; with
    ``mosaic_cache_dir=None``, read straight from S3 and persist nothing."""
    fname = s3_path.split("/")[-1]
    if mosaic_cache_dir is not None:
        local_path = Path(mosaic_cache_dir) / fname
        if local_path.exists():
            try:
                return fits.open(str(local_path), memmap=True), f"local ({fname[:40]})"
            except OSError:
                pass
    from .netutils import S3_FSSPEC_KWARGS, retry
    return (
        retry(lambda: fits.open(f"s3://{s3_path}", use_fsspec=True,
                                fsspec_kwargs=S3_FSSPEC_KWARGS),
              what=f"S3 open {fname[:40]}"),
        f"s3-lazy ({fname[:40]})",
    )


class _TileCoverageError(ValueError):
    """The requested rectangle is not contained by this tile."""


def _single_tile_cutout(tile, ra, dec, size_arcsec, mosaic_cache_dir):
    """Read an exact native pixel rectangle, preserving the delivered PSF."""
    hdul_ctx, _ = _open_mosaic(tile["s3"], mosaic_cache_dir)
    with hdul_ctx as hdul:
        for hdu in hdul:
            if getattr(hdu, "shape", None) is None or len(hdu.shape) != 2:
                continue
            wcs = WCS(hdu.header)
            cx, cy = wcs.world_to_pixel_values(ra, dec)
            scale = proj_plane_pixel_scales(wcs) * 3600.0
            shape = tuple(max(1, int(round(size_arcsec / p))) for p in scale[::-1])
            try:
                slices, _ = overlap_slices(hdu.shape, shape,
                                           (float(cy), float(cx)), mode="strict")
            except (PartialOverlapError, NoOverlapError) as exc:
                raise _TileCoverageError(
                    f"Requested cutout extends beyond MER tile {tile.get('tile_id')}. "
                    "Use a smaller field or separate tile fits. Resampling multiple "
                    "tiles also requires transforming their PSFs and noise model."
                ) from exc
            data = np.asarray(hdu.section[slices]).copy()
            header = hdu.header.copy()
            header["MERTILE"] = str(tile["tile_id"])
            header["NATIVE"] = (True, "Native MER pixels; no extra resampling")
            return data, wcs.slice(slices), header
    raise ValueError(f"No 2D image in MER tile {tile.get('tile_id')}")


def _matching_tile(product, tile_id):
    for tile in product.get("tiles") or [product]:
        if tile.get("tile_id") == tile_id:
            return tile
    raise ValueError(f"No matching product for science tile {tile_id}")


def _check_alignment(data, wcs, other, other_wcs):
    if data.shape != other.shape or not wcs.wcs.compare(other_wcs.wcs, tolerance=1e-10):
        raise ValueError("Science, RMS and flag planes must share the native pixel grid")


def _cutout_cache_path(band, ra, dec, size_arcsec, data_dir, ptype="science"):
    # The sparse FLG bitmask shrinks ~5x under gzip; science/RMS images
    # do not compress.
    ext = "fits.gz" if ptype == "flag" else "fits"
    fname = (f"{band.lower()}_{ptype}_{ra:.8f}_{dec:.8f}_{size_arcsec:.4f}"
             f"_native-v1.{ext}")
    return Path(data_dir) / fname


def _write_cutout_fits(path: Path, data: np.ndarray, wcs: WCS, header: fits.Header):
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = wcs.to_header()
    for key in ("MAGZERO", "MAGZP", "TELESCOP", "INSTRUME", "FILTER",
                "BUNIT", "EXPTIME", "TIMESYS", "DATE-OBS", "MERTILE", "NATIVE"):
        if key in header:
            hdr[key] = header[key]
    # Write to a temp file first, then rename, so an interrupted run never
    # leaves a half-written cutout.
    tmp = path.with_suffix(path.suffix + ".tmp")
    fits.PrimaryHDU(data=data.astype(np.float32), header=hdr).writeto(
        tmp, overwrite=True)
    tmp.replace(path)


def _read_cutout_fits(path: Path) -> tuple[np.ndarray, WCS, fits.Header]:
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
        wcs = WCS(hdul[0].header)
        header = hdul[0].header.copy()
    return data, wcs, header


def _write_mask_fits(path: Path, mask: np.ndarray, wcs: WCS, header: fits.Header):
    """Cache an integer flag plane as int32 (a float32 cast would corrupt
    the high bits). astropy gzips only when the filename ends in ``.gz``,
    so the temp name preserves the suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = wcs.to_header()
    for key in ("MERTILE", "NATIVE"):
        if key in header:
            hdr[key] = header[key]
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    fits.PrimaryHDU(data=mask.astype(np.int32), header=hdr).writeto(tmp, overwrite=True)
    tmp.replace(path)


def fetch_cutout(band: str, ra: float, dec: float, size_arcsec: float,
                 *,
                 products: dict | None = None,
                 data_dir: str | Path = DEFAULT_CUTOUT_DIR,
                 mosaic_cache_dir: str | Path | None = None,
                 force_download: bool = False,
                 with_rms: bool = True,
                 with_flag: bool = False) -> Cutout:
    """Fetch a single-band cutout, reading the local FITS cache when present
    and otherwise downloading native pixels from S3 (and caching the result).

    Try the primary MER tile, then other candidate tiles if it cannot
    contain the rectangle. Science, RMS and flags come from that one tile.
    Pass the returned header's MERTILE to the PSF extractor as tile_id.
    No additional resampling is applied. Fields that cannot fit in any
    candidate tile raise an error; fit them as separate tiles.
    Versioned cache names prevent reuse of earlier resampled cutouts.

    Parameters
    ----------
    band : {'VIS','Y','J','H'}
    ra, dec : float
    size_arcsec : float
    products : optional, output of ``discover_mer_mosaics``. If None and a
        live download is required, this calls ``discover_mer_mosaics`` itself.
    data_dir : Path
        Where to look for / write the small post-cutout FITS cache.
    mosaic_cache_dir : Path or None
        If set, also consult this directory for full-mosaic files (read-only).
    force_download : bool
        If True, ignore the local cache and re-fetch from S3.
    with_rms : bool
        Whether to also fetch the matching RMS map.
    with_flag : bool
        Whether to also fetch the MER FLG quality plane; if the band has no
        FLG tile a warning is issued and the returned Cutout has flag=None.

    Returns
    -------
    Cutout
    """
    data_dir = Path(data_dir)
    sci_cache = _cutout_cache_path(band, ra, dec, size_arcsec, data_dir, "science")
    rms_cache = _cutout_cache_path(band, ra, dec, size_arcsec, data_dir, "rms")
    flag_cache = _cutout_cache_path(band, ra, dec, size_arcsec, data_dir, "flag")

    have_sci = sci_cache.exists()
    have_rms = rms_cache.exists() if with_rms else True
    have_flag = flag_cache.exists() if with_flag else True

    if have_sci and have_rms and have_flag and not force_download:
        data, wcs, header = _read_cutout_fits(sci_cache)
        rms = None
        if with_rms:
            rms, rms_wcs, rms_header = _read_cutout_fits(rms_cache)
            _check_alignment(data, wcs, rms, rms_wcs)
            if header.get("MERTILE") != rms_header.get("MERTILE"):
                raise ValueError("Cached science and RMS come from different tiles")
        flag = None
        if with_flag:
            flag_data, flag_wcs, flag_header = _read_cutout_fits(flag_cache)
            _check_alignment(data, wcs, flag_data, flag_wcs)
            if header.get("MERTILE") != flag_header.get("MERTILE"):
                raise ValueError("Cached science and flags come from different tiles")
            flag = flag_data.astype(np.int32)
        return Cutout(band=band, data=data, rms=rms, wcs=wcs, header=header,
                      flag=flag)

    if products is None:
        half_size_deg = size_arcsec / 2.0 / 3600.0
        products = discover_mer_mosaics(ra, dec, half_size_deg)

    if band not in products or "science" not in products[band]:
        raise ValueError(f"no science tile available for band {band!r}")

    science_tile = products[band]["science"]
    candidates = [science_tile] + [
        t for t in science_tile.get("tiles", [])
        if t["tile_id"] != science_tile["tile_id"]]
    for candidate in candidates:
        try:
            data, wcs, header = _single_tile_cutout(
                candidate, ra, dec, size_arcsec, mosaic_cache_dir)
        except _TileCoverageError:
            continue
        science_tile = candidate
        break
    else:
        raise ValueError(
            "No candidate MER tile contains this cutout. Use a smaller field "
            "or separate tile fits; automatic resampling is not performed.")
    tile_id = science_tile["tile_id"]

    rms = None
    if with_rms:
        if "rms" not in products[band]:
            raise ValueError(f"No RMS product available for band {band!r}")
        tile = _matching_tile(products[band]["rms"], tile_id)
        rms, rms_wcs, _ = _single_tile_cutout(
            tile, ra, dec, size_arcsec, mosaic_cache_dir)
        _check_alignment(data, wcs, rms, rms_wcs)

    flag = None
    if with_flag:
        if "flag" in products[band]:
            tile = _matching_tile(products[band]["flag"], tile_id)
            fdata, flag_wcs, _ = _single_tile_cutout(
                tile, ra, dec, size_arcsec, mosaic_cache_dir)
            _check_alignment(data, wcs, fdata, flag_wcs)
            flag = fdata.astype(np.int32)
        else:
            import warnings
            warnings.warn(
                f"with_flag=True but no FLG tile is available for band {band!r}; "
                "returning a cutout with flag=None (no pixel masking).",
                stacklevel=2)

    _write_cutout_fits(sci_cache, data, wcs, header)
    if rms is not None:
        _write_cutout_fits(rms_cache, rms, wcs, header)
    if flag is not None:
        _write_mask_fits(flag_cache, flag, wcs, header)

    return Cutout(band=band, data=data, rms=rms, wcs=wcs, header=header, flag=flag)


# ---------------------------------------------------------------------------
# Catalog trimming
# ---------------------------------------------------------------------------

def trim_catalog_to_cutout(mer_cat, wcs: WCS, shape, *, edge_margin_pix: int = 1):
    """Drop MER rows whose pixel positions fall off the cutout.

    Boundary rows stay in the source list (their light is in the pixels);
    their own photometry is flagged downstream
    (:func:`euclid_phot.flags.flag_sources`, column ``edge``).

    Returns
    -------
    astropy.table.Table  (possibly shorter than the input)
    """
    H, W = shape
    keep = np.ones(len(mer_cat), dtype=bool)
    for i, row in enumerate(mer_cat):
        px, py = wcs.world_to_pixel_values(row["ra"], row["dec"])
        if not (edge_margin_pix <= px < W - edge_margin_pix
                and edge_margin_pix <= py < H - edge_margin_pix):
            keep[i] = False
    return mer_cat[keep]
