#!/usr/bin/env python3
"""Re-fetch Euclid tiles from the public Q1 MER mosaics (IRSA), preserving the
exact JAISP tiling and NPZ schema.

The production Euclid tiles (VIS + NISP Y/J/H) currently in
`data/euclid_tiles_all/` and `data/edf_s_ood/euclid_tiles_edfs/` were built from
bulk ESA-archive DR1 mosaics (notebook 03). This script regenerates the same
tiles from Euclid **Q1** MER mosaics served by IRSA (collection
`euclid_DpdMerBksMosaic`, `.../euclid/q1/MER/...`), so the imaging matches a
public release.

Tiling is defined entirely by the Rubin side: each Euclid tile is a fixed
1084x1084 cutout centred on the existing tile's (ra_center, dec_center). We read
those centres straight from the existing DR1 NPZ files, so the tile grid,
filenames, and array shapes are reproduced exactly. Only the pixel values, WCS,
and per-pixel rms change (they come from Q1 instead of DR1).

Output schema per tile (identical to DR1 product):
    ra_center, dec_center (float64), tile_id (bytes), euclid_tile_id (str = Q1
    MER tile used for VIS), and per band b in {VIS,Y,J,H}:
        img_b   (1084,1084) float32   -- science mosaic cutout
        var_b   (1084,1084) float32   -- rms^2 (rms sanitized: <=0 / huge -> nan)
        wcs_b   str                   -- FITS header string of the cutout WCS
        pixel_scale_b (float64)       -- arcsec/pix from the cutout WCS

Usage:
    # smoke test: one tile per field, into *_q1_smoke dirs
    python io/refetch_euclid_q1.py --limit 1 --smoke

    # full ECDFS/EDF-F set (790 tiles) -> data/euclid_tiles_all_q1/
    python io/refetch_euclid_q1.py --field ecdfs

    # full EDF-S OOD set (72 tiles) -> data/edf_s_ood/euclid_tiles_edfs_q1/
    python io/refetch_euclid_q1.py --field edfs
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
import time
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u
import fsspec

from astroquery.ipac.irsa import Irsa

BANDS_EUCLID = ("VIS", "Y", "J", "H")
COLLECTION = "euclid_DpdMerBksMosaic"
CUTOUT_PX = 1084          # match the DR1 product array shape exactly
SEARCH_RADIUS_ARCSEC = 60

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / ".." / "data").resolve()

FIELDS = {
    "ecdfs": {
        "src": DATA_DIR / "euclid_tiles_all",
        "out": DATA_DIR / "euclid_tiles_all_q1",
    },
    "edfs": {
        "src": DATA_DIR / "edf_s_ood" / "euclid_tiles_edfs",
        "out": DATA_DIR / "edf_s_ood" / "euclid_tiles_edfs_q1",
    },
}


def sanitize_rms(rms, huge=1e10):
    rms = rms.astype(np.float32, copy=False)
    bad = (~np.isfinite(rms)) | (rms <= 0) | (rms > huge)
    rms = rms.copy()
    rms[bad] = np.nan
    return rms


def _tile_id_from_name(fn: str) -> str:
    """`tile_x00000_y00000_..._euclid.npz` -> `tile_x00000_y00000_...` (drop _euclid)."""
    stem = Path(fn).name
    stem = re.sub(r"\.npz$", "", stem)
    stem = re.sub(r"_euclid$", "", stem)
    return stem


def _interior_margin(row, ra, dec):
    """Approximate on-sky margin (deg) from (ra,dec) to the nearest edge of a
    MER-mosaic footprint, using the SIA centre (s_ra,s_dec) and field-of-view
    (s_fov). Positive => inside; larger => more centrally contained. Returns
    None if the needed columns are missing."""
    cols = row.colnames
    if not ({"s_ra", "s_dec"} <= set(cols)):
        return None
    try:
        cra, cdec, fov = float(row["s_ra"]), float(row["s_dec"]), float(row["s_fov"])
    except Exception:
        return None
    half = fov / 2.0
    cosd = np.cos(np.deg2rad(dec))
    d_dec = abs(dec - cdec)
    d_ra = abs((ra - cra) * cosd)
    return min(half - d_dec, half - d_ra)


def _best_row(rows, ra, dec):
    """Among candidate MER-mosaic rows for one band, pick the mosaic that most
    fully contains the cutout box -- the one that maximises the interior margin
    to the footprint edges. Candidates arise because the tile centre lies in the
    overlap of adjacent MER tiles; the nearest-*centre* mosaic is NOT always the
    one that contains the box (a neighbour can be offset such that the point sits
    just inside its edge and the box spills into NaN)."""
    if len(rows) == 0:
        return None
    if len(rows) == 1:
        return rows[0]
    best, best_m = None, None
    for r in rows:
        m = _interior_margin(r, ra, dec)
        if m is None:
            return rows[0]  # no footprint columns; keep first
        if best_m is None or m > best_m:
            best, best_m = r, m
    return best


def _retry(fn, attempts=5, base_sleep=3.0):
    """Call fn() with retries + linear backoff. IRSA IBE/SIA intermittently
    returns HTTP 400 / malformed responses under sustained load; a short wait
    and retry recovers almost all of them."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 -- transient network/server errors
            last = e
            if i < attempts - 1:
                time.sleep(base_sleep * (i + 1))
    raise last


def _mer_tile_id(access_url: str) -> str | None:
    m = re.search(r"TILE(\d+)", str(access_url))
    return f"TILE{m.group(1)}" if m else None


def _coerce(data, w):
    """Force a cutout to exactly (CUTOUT_PX, CUTOUT_PX) with NaN padding.

    IBE returns CUTOUT_PX+1 for `size=<N>pix`; near a mosaic edge it can also
    return fewer rows/cols. Slicing/padding from the (0,0) origin leaves the WCS
    reference pixel (CRPIX) valid, so we keep the returned header WCS as-is."""
    data = np.asarray(data, dtype=np.float32)
    out = np.full((CUTOUT_PX, CUTOUT_PX), np.nan, dtype=np.float32)
    ny, nx = min(data.shape[0], CUTOUT_PX), min(data.shape[1], CUTOUT_PX)
    out[:ny, :nx] = data[:ny, :nx]
    return out, w


def _cut(access_url, ra, dec):
    """Return (data float32, WCS) for a fixed CUTOUT_PX box centred on (ra,dec).

    Uses the IRSA IBE server-side cutout API: the `access_url` already points at
    an IBE data path (`.../ibe/data/euclid/q1/MER/...fits`); appending
    `?center=&size=&gzip=false` makes IRSA cut the box server-side and return
    only the small cutout (one request, ~a few MB), instead of streaming the
    full ~1.5 GB mosaic."""
    url = f"{access_url}?center={ra},{dec}&size={CUTOUT_PX}pix&gzip=false"

    def _do():
        with fsspec.open(url, "rb") as fh:
            with fits.open(fh, memmap=False) as hdul:
                hdu = hdul[0] if hdul[0].header.get("NAXIS", 0) >= 2 else hdul[1]
                w = WCS(hdu.header)
                data = np.array(hdu.data, dtype=np.float32)
        return _coerce(data, w)

    return _retry(_do)


def fetch_tile(ra, dec):
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    tab = _retry(lambda: Irsa.query_sia(pos=(coord, SEARCH_RADIUS_ARCSEC * u.arcsec),
                                        collection=COLLECTION))
    try:
        tab = tab.to_table()
    except Exception:
        pass
    out = {}
    mer_tile_id = None
    for b in BANDS_EUCLID:
        sci_rows = tab[(tab["energy_bandpassname"] == b) & (tab["dataproduct_subtype"] == "science")]
        sci = _best_row(sci_rows, ra, dec)
        if sci is None:
            continue
        img, wcs_cut = _cut(sci["access_url"], ra, dec)
        out[f"img_{b}"] = img
        out[f"wcs_{b}"] = wcs_cut.to_header_string()
        try:
            out[f"pixel_scale_{b}"] = np.float64(np.mean(proj_plane_pixel_scales(wcs_cut)) * 3600.0)
        except Exception:
            out[f"pixel_scale_{b}"] = np.float64(np.nan)
        if b == "VIS":
            mer_tile_id = _mer_tile_id(sci["access_url"])

        rms_rows = tab[(tab["energy_bandpassname"] == b) & (tab["dataproduct_subtype"] == "noise")]
        rms_row = _best_row(rms_rows, ra, dec)
        if rms_row is not None:
            rms, _ = _cut(rms_row["access_url"], ra, dec)
            out[f"var_{b}"] = sanitize_rms(rms) ** 2
    return out, mer_tile_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", choices=list(FIELDS) + ["both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="max tiles per field (smoke test)")
    ap.add_argument("--smoke", action="store_true", help="write to *_q1_smoke dirs, don't touch real output")
    ap.add_argument("--overwrite", action="store_true", help="re-fetch tiles even if output NPZ exists")
    args = ap.parse_args()

    fields = list(FIELDS) if args.field == "both" else [args.field]
    for field in fields:
        src = FIELDS[field]["src"]
        out_dir = FIELDS[field]["out"]
        if args.smoke:
            out_dir = out_dir.parent / (out_dir.name + "_smoke")
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(glob.glob(str(src / "*.npz")))
        if args.limit:
            files = files[: args.limit]
        print(f"[{field}] {len(files)} tiles  src={src}  out={out_dir}", flush=True)

        n_ok = n_skip = n_fail = 0
        t0 = time.time()
        for i, f in enumerate(files):
            tile_id = _tile_id_from_name(f)
            out_fn = out_dir / f"{tile_id}_euclid.npz"
            if out_fn.exists() and not args.overwrite:
                n_skip += 1
                continue
            d = np.load(f, allow_pickle=True)
            ra, dec = float(d["ra_center"]), float(d["dec_center"])
            try:
                bands, mer_tile_id = fetch_tile(ra, dec)
            except Exception as e:
                n_fail += 1
                print(f"  [{i+1}/{len(files)}] FAIL {tile_id}: {e}", flush=True)
                continue
            if not any(k.startswith("img_") for k in bands):
                n_fail += 1
                print(f"  [{i+1}/{len(files)}] NO-DATA {tile_id}", flush=True)
                continue
            save = {
                "ra_center": np.float64(ra),
                "dec_center": np.float64(dec),
                "tile_id": np.bytes_(tile_id),
                "euclid_tile_id": np.str_(mer_tile_id or ""),
                **bands,
            }
            np.savez_compressed(out_fn, **save)
            n_ok += 1
            if (i + 1) % 10 == 0 or i == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"  [{i+1}/{len(files)}] ok={n_ok} skip={n_skip} fail={n_fail} "
                      f"({rate:.2f} tiles/s)", flush=True)
        print(f"[{field}] DONE ok={n_ok} skip={n_skip} fail={n_fail} "
              f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
