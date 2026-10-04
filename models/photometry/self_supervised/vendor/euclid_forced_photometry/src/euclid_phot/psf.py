"""PSF-stamp extraction from MER PSF data products.

Each MER tile ships two PSF products, both holding stamps of a band-dependent
size packed into one large FITS image:

* CATALOG-PSF: one stamp per MER catalog source, at the source
  positions. For measurement positions that are MER sources (the default
  ``objects="mer"`` path).
* GRID-PSF: the PSF model sampled on a regular ~12 arcsec grid across
  the whole tile. For positions that are not MER sources: user-supplied
  coordinates, injected sources, empty-position error calibration.

Only the stamps near the cutout center are pulled (each file is 200-500 MB;
the stamps subset is ~MB) via S3 byte-range reads.

Public: ``extract_catalog_psf`` and ``extract_grid_psf`` (same return
schema, ``stamps``, ``ra``, ``dec``, ``x``, ``y``, ``fwhm``, ``stmpsize``,
so every consumer works with either product; cached as ``.npz`` under
``data_dir``), ``get_psf_for_source``, ``psf_summary``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from .config import DEFAULT_PSF_DIR


def _resolve_catalog_psf_s3(band: str, products: dict | None, tile_id=None) -> str:
    """Find the CATALOG-PSF S3 path for ``band``.

    Either takes it from a ``products`` dict (output of
    ``discover_mer_mosaics``) or raises with a helpful message.
    """
    if products is not None and band in products and "psf_catalog" in products[band]:
        product = products[band]["psf_catalog"]
        if tile_id is None and "science" in products[band]:
            tile_id = products[band]["science"]["tile_id"]
        if tile_id is not None:
            from .cutouts import _matching_tile
            product = _matching_tile(product, tile_id)
        return product["s3"]
    raise ValueError(
        f"CATALOG-PSF for band {band!r} not in the products dict. "
        "Pass the output of discover_mer_mosaics(..., bands=(...,'PSF...'))."
    )


_PSF_CACHE_VERSION = "archive-origin-v1"
_CATALOG_PSF_CACHE_VERSION = "fits-origin-v2"


def _tile_from_path(path):
    import re
    match = re.search(r"TILE([0-9]+)", path or "")
    return match.group(1) if match else None


def _requested_cache(cache, tile_id, s3_path):
    """Isolate explicit tile/product requests from unverified older caches."""
    import hashlib
    path_tile = _tile_from_path(s3_path)
    if tile_id is not None and path_tile is not None and str(tile_id) != path_tile:
        raise ValueError("Explicit PSF path does not match the science tile")
    suffix = ""
    if tile_id is not None:
        suffix += "_tile" + hashlib.sha256(str(tile_id).encode()).hexdigest()[:12]
    if s3_path is not None:
        suffix += "_src" + hashlib.sha256(s3_path.encode()).hexdigest()[:12]
    return cache.with_name(cache.stem + suffix + cache.suffix)


def extract_catalog_psf(band: str, ra: float, dec: float,
                        *,
                        s3_path: str | None = None,
                        products: dict | None = None,
                        tile_id: str | None = None,
                        radius_arcsec: float = 60.0,
                        data_dir: str | Path = DEFAULT_PSF_DIR,
                        force_download: bool = False) -> dict:
    """Download or load a region's worth of CATALOG-PSF stamps.

    Parameters
    ----------
    band : {'VIS','Y','J','H'}
    ra, dec : float
    s3_path : str, optional
        Explicit S3 path to the CATALOG-PSF FITS. Overrides ``products``.
    products : dict, optional
        Output of ``discover_mer_mosaics``. When neither ``s3_path`` nor
        ``products`` is given and the cache is empty, discovery runs
        automatically (one SIA query).
    tile_id : str, optional
        Science cutout MERTILE. Select its PSF product and use a tile-specific
        cache, including when discovery prefers another tile.
    radius_arcsec : float
        Stamp acceptance radius around (ra, dec), capped at ~60-200 arcsec
        for typical cutout sizes.
    data_dir : Path
        Where to cache the extracted stamps as a small ``.npz``.
    force_download : bool
        If True, ignore the .npz cache and re-download from S3.

    Returns
    -------
    dict with keys ``stamps``, ``ra``, ``dec``, ``x``, ``y``, ``fwhm``, ``stmpsize``.
    """
    data_dir = Path(data_dir)
    cache = data_dir / (
        f"psf_stamps_{band.lower()}"
        f"_{ra:.4f}_{dec:.4f}"
        f"_r{int(round(radius_arcsec))}_{_CATALOG_PSF_CACHE_VERSION}.npz"
    )
    if tile_id is None and products is not None:
        tile_id = products.get(band, {}).get("science", {}).get("tile_id")
    cache = _requested_cache(cache, tile_id, s3_path)
    if cache.exists() and not force_download:
        with np.load(cache) as d:
            return {key: (d[key].item() if d[key].ndim == 0 else d[key])
                    for key in d.files}

    if s3_path is None:
        if products is None:
            from .cutouts import discover_mer_mosaics
            products = discover_mer_mosaics(
                ra, dec, radius_arcsec / 3600.0, bands=(band,))
        s3_path = _resolve_catalog_psf_s3(band, products, tile_id)

    # Byte-range reads: HDU[2] (small stamp catalog) in full, then one
    # hdu.section read of the nearby-stamp bounding box from HDU[1]
    # (~1-2 MB instead of the 200-500 MB file).
    from .netutils import S3_FSSPEC_KWARGS, retry
    with retry(lambda: fits.open(f"s3://{s3_path}", use_fsspec=True,
                                 fsspec_kwargs=S3_FSSPEC_KWARGS),
               what=f"S3 open CATALOG-PSF {band}") as hdul:
        table = hdul[2].data
        stmpsize = int(hdul[1].header["STMPSIZE"])
        half = stmpsize // 2
        img_h = int(hdul[1].header["NAXIS2"])
        img_w = int(hdul[1].header["NAXIS1"])

        cos_dec = np.cos(np.radians(dec))
        # Wrap the RA difference into [-180, 180].
        d_ra = ((table["RA"] - ra + 540.0) % 360.0) - 180.0
        dist = np.sqrt((d_ra * cos_dec) ** 2
                       + (table["Dec"] - dec) ** 2) * 3600.0
        nearby = table[dist < radius_arcsec]
        if len(nearby) == 0:
            # Empty result instead of raising; consumers raise clearly.
            import warnings
            closest = float(dist.min()) if len(dist) else float("inf")
            warnings.warn(
                f"no CATALOG-PSF stamps within {radius_arcsec} arcsec "
                f"of ({ra}, {dec}) on band {band!r}; closest stamp is "
                f"{closest:.1f} arcsec away. Consider widening "
                f"radius_arcsec.",
                stacklevel=2)
            return {
                "stamps":   np.empty((0, 0, 0), dtype=np.float32),
                "ra":       np.array([]),
                "dec":      np.array([]),
                "x":        np.array([]),
                "y":        np.array([]),
                "fwhm":     np.array([]),
                "stmpsize": stmpsize,
            }

        # Bounding box of the nearby stamps in HDU[1] pixel coordinates.
        xc = np.asarray(nearby["x_center"], dtype=int) - 1
        yc = np.asarray(nearby["y_center"], dtype=int) - 1
        x0 = max(0, int(xc.min()) - half)
        x1 = min(img_w, int(xc.max()) + half + 1)
        y0 = max(0, int(yc.min()) - half)
        y1 = min(img_h, int(yc.max()) + half + 1)
        img_chunk = np.asarray(hdul[1].section[y0:y1, x0:x1])

        stamps, keep_idx = [], []
        for i in range(len(nearby)):
            xi = int(xc[i]); yi = int(yc[i])
            if (xi - half < 0 or xi + half + 1 > img_w
                    or yi - half < 0 or yi + half + 1 > img_h):
                continue
            sx = xi - half - x0
            sy = yi - half - y0
            s = img_chunk[sy:sy + stmpsize, sx:sx + stmpsize]
            if s.shape != (stmpsize, stmpsize):
                continue
            stamps.append(s)
            keep_idx.append(i)
        nearby = nearby[keep_idx]

        stamps = np.array(stamps)
        result = {
            "stamps":   stamps,
            "ra":       np.asarray(nearby["RA"]),
            "dec":      np.asarray(nearby["Dec"]),
            "x":        np.asarray(nearby["x"]),
            "y":        np.asarray(nearby["y"]),
            "fwhm":     np.asarray(nearby["FWHM"]),
            "stmpsize": stmpsize,
        }

    result["s3_path"] = s3_path
    result["tile_id"] = str(tile_id or _tile_from_path(s3_path) or "")
    result["product"] = "catalog" if "psf_stamps_" in cache.name else "grid"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first, then rename, so an interrupted run never
    # leaves a half-written cache. np.savez appends ".npz" to names that lack
    # it, so the temp name keeps the suffix (foo.tmp.npz), not a trailing ".tmp".
    tmp = cache.with_name(cache.stem + ".tmp" + cache.suffix)
    np.savez(tmp, **result)
    tmp.replace(cache)
    return result


def _resolve_grid_psf_s3(band: str, products: dict | None, tile_id=None) -> str:
    """Find the GRID-PSF S3 path for ``band`` (see _resolve_catalog_psf_s3)."""
    if products is not None and band in products and "psf_grid" in products[band]:
        product = products[band]["psf_grid"]
        if tile_id is None and "science" in products[band]:
            tile_id = products[band]["science"]["tile_id"]
        if tile_id is not None:
            from .cutouts import _matching_tile
            product = _matching_tile(product, tile_id)
        return product["s3"]
    raise ValueError(
        f"GRID-PSF for band {band!r} not in the products dict. "
        "Pass the output of discover_mer_mosaics(...)."
    )


def extract_grid_psf(band: str, ra: float, dec: float,
                     *,
                     s3_path: str | None = None,
                     products: dict | None = None,
                     tile_id: str | None = None,
                     radius_arcsec: float = 60.0,
                     data_dir: str | Path = DEFAULT_PSF_DIR,
                     force_download: bool = False) -> dict:
    """Download or load a region's worth of GRID-PSF stamps.

    GRID-PSF samples the tile's PSF model on a regular ~12 arcsec grid
    (Q1: 120-pixel spacing on the 19200x19200 tile), covering every sky
    position.

    File layout, as found on Q1 tile 102160059: HDU[1] ``PSF image`` is a
    tile-sized float32 image with the stamps packed at their grid positions
    and a full TAN WCS in its header (plus ``STMPSIZE``); HDU[2] ``Stamps
    information`` has columns ``x``, ``y`` (stamp centers, FITS 1-based
    pixels of HDU[1]) and ``FWHM`` (arcsec).

    Parameters and return schema are identical to
    :func:`extract_catalog_psf`. Cached to
    ``data_dir/psf_grid_stamps_<band>_<ra>_<dec>_r<radius>_<version>.npz``,
    a prefix distinct from the CATALOG-PSF cache.
    """
    data_dir = Path(data_dir)
    cache = data_dir / (
        f"psf_grid_stamps_{band.lower()}"
        f"_{ra:.4f}_{dec:.4f}"
        f"_r{int(round(radius_arcsec))}_{_PSF_CACHE_VERSION}.npz"
    )
    if tile_id is None and products is not None:
        tile_id = products.get(band, {}).get("science", {}).get("tile_id")
    cache = _requested_cache(cache, tile_id, s3_path)
    if cache.exists() and not force_download:
        with np.load(cache) as d:
            return {key: (d[key].item() if d[key].ndim == 0 else d[key])
                    for key in d.files}

    if s3_path is None:
        if products is None:
            from .cutouts import discover_mer_mosaics
            products = discover_mer_mosaics(
                ra, dec, radius_arcsec / 3600.0, bands=(band,))
        s3_path = _resolve_grid_psf_s3(band, products, tile_id)

    # Same byte-range strategy as extract_catalog_psf.
    from .netutils import S3_FSSPEC_KWARGS, retry
    with retry(lambda: fits.open(f"s3://{s3_path}", use_fsspec=True,
                                 fsspec_kwargs=S3_FSSPEC_KWARGS),
               what=f"S3 open GRID-PSF {band}") as hdul:
        from astropy.wcs import WCS
        hdr = hdul[1].header
        wcs = WCS(hdr)
        stmpsize = int(hdr["STMPSIZE"])
        half = stmpsize // 2
        img_h = int(hdr["NAXIS2"])
        img_w = int(hdr["NAXIS1"])

        tbl = hdul[2].data
        x = np.asarray(tbl["x"], dtype=float)
        y = np.asarray(tbl["y"], dtype=float)
        fwhm = np.asarray(tbl["FWHM"], dtype=float)
        # Stamp centers are FITS 1-based pixels of HDU[1]; astropy is 0-based.
        ra_s, dec_s = wcs.pixel_to_world_values(x - 1.0, y - 1.0)
        ra_s = np.atleast_1d(np.asarray(ra_s, dtype=float))
        dec_s = np.atleast_1d(np.asarray(dec_s, dtype=float))

        cos_dec = np.cos(np.radians(dec))
        d_ra = ((ra_s - ra + 540.0) % 360.0) - 180.0
        dist = np.sqrt((d_ra * cos_dec) ** 2
                       + (dec_s - dec) ** 2) * 3600.0
        sel = dist < radius_arcsec
        if not sel.any():
            import warnings
            closest = float(dist.min()) if dist.size else float("inf")
            warnings.warn(
                f"no GRID-PSF stamps within {radius_arcsec} arcsec of "
                f"({ra}, {dec}) on band {band!r}; closest stamp is "
                f"{closest:.1f} arcsec away. Consider widening "
                f"radius_arcsec.",
                stacklevel=2)
            return {
                "stamps":   np.empty((0, 0, 0), dtype=np.float32),
                "ra":       np.array([]),
                "dec":      np.array([]),
                "x":        np.array([]),
                "y":        np.array([]),
                "fwhm":     np.array([]),
                "stmpsize": stmpsize,
            }

        xc = np.asarray(np.round(x[sel] - 1.0), dtype=int)
        yc = np.asarray(np.round(y[sel] - 1.0), dtype=int)
        x0 = max(0, int(xc.min()) - half)
        x1 = min(img_w, int(xc.max()) + half + 1)
        y0 = max(0, int(yc.min()) - half)
        y1 = min(img_h, int(yc.max()) + half + 1)
        img_chunk = np.asarray(hdul[1].section[y0:y1, x0:x1])

        stamps, keep = [], []
        for i in range(len(xc)):
            xi, yi = int(xc[i]), int(yc[i])
            if (xi - half < 0 or xi + half + 1 > img_w
                    or yi - half < 0 or yi + half + 1 > img_h):
                continue
            s = img_chunk[yi - half - y0:yi + half + 1 - y0,
                          xi - half - x0:xi + half + 1 - x0]
            if s.shape != (stmpsize, stmpsize):
                continue
            ssum = float(np.sum(s))
            # A grid cell outside the tile's PSF-model coverage holds a
            # blank stamp; it cannot be normalized, so drop it.
            if not np.isfinite(ssum) or ssum <= 0:
                continue
            stamps.append(s)
            keep.append(i)

        if not stamps:
            import warnings
            warnings.warn(
                f"every GRID-PSF stamp within {radius_arcsec} arcsec of "
                f"({ra}, {dec}) on band {band!r} is blank (outside the "
                "tile's PSF-model coverage).", stacklevel=2)
            return {
                "stamps":   np.empty((0, 0, 0), dtype=np.float32),
                "ra":       np.array([]),
                "dec":      np.array([]),
                "x":        np.array([]),
                "y":        np.array([]),
                "fwhm":     np.array([]),
                "stmpsize": stmpsize,
            }
        keep = np.asarray(keep, dtype=int)
        stamps = np.array(stamps, dtype=np.float32)
        idx = np.where(sel)[0][keep]
        result = {
            "stamps":   stamps,
            "ra":       ra_s[idx],
            "dec":      dec_s[idx],
            "x":        x[idx],
            "y":        y[idx],
            "fwhm":     fwhm[idx],
            "stmpsize": stmpsize,
        }

    result["s3_path"] = s3_path
    result["tile_id"] = str(tile_id or _tile_from_path(s3_path) or "")
    result["product"] = "catalog" if "psf_stamps_" in cache.name else "grid"
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.stem + ".tmp" + cache.suffix)
    np.savez(tmp, **result)
    tmp.replace(cache)
    return result


def _require_stamps(psf_data):
    stamps = psf_data.get("stamps")
    if stamps is None or len(stamps) == 0:
        raise ValueError(
            "psf_data has no usable PSF stamps near the target. "
            "The selected product has no samples within radius_arcsec; widen "
            "radius_arcsec or confirm the field has MER PSF coverage.")
    return stamps


def get_psf_for_source(psf_data, source_ra: float, source_dec: float):
    """Return the nearest PSF sample to a source position.

    Returns
    -------
    (stamp, fwhm_arcsec, separation_arcsec)
        ``stamp`` is float32 and sums to 1.
    """
    _require_stamps(psf_data)
    cos_dec = np.cos(np.radians(source_dec))
    d_ra = ((np.asarray(psf_data["ra"]) - source_ra + 540.0) % 360.0) - 180.0
    dist = np.sqrt((d_ra * cos_dec) ** 2
                   + (psf_data["dec"] - source_dec) ** 2) * 3600.0
    idx = int(np.argmin(dist))
    stamp = psf_data["stamps"][idx].astype(np.float32).copy()
    stamp /= stamp.sum()
    return stamp, float(psf_data["fwhm"][idx]), float(dist[idx])


def psf_summary(psf_data) -> tuple[np.ndarray, float]:
    """Field-average normalized stamp + median FWHM (arcsec)."""
    stamps = _require_stamps(psf_data)
    avg = stamps.mean(axis=0).astype(np.float32)
    avg /= avg.sum()
    return avg, float(np.median(psf_data["fwhm"]))
