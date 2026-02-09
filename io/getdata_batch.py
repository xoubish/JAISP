#!/usr/bin/env python3
"""Batch Rubin + Euclid download with small on-disk footprint.

Workflow:
- Tile Rubin patches and write small batches (e.g. 5 tiles).
- Fetch Euclid cutouts for those tiles.
- Archive the batch to a tar.gz, then delete the individual NPZs.
"""

from __future__ import annotations

import os
import glob
import tarfile
from pathlib import Path

import numpy as np
from lsst.daf.butler import Butler
import lsst.geom as geom

from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import fsspec

# ---- Config ----
TRACT = 5063
SKYMAP = "lsst_cells_v1"
REPO = "dp1"
COLLECTION = "LSSTComCam/DP1"
DATASETTYPE = "deep_coadd"

BANDS_RUBIN = ("u", "g", "r", "i", "z", "y")
BANDS_EUCLID = ("VIS", "Y", "J", "H")

TILE_SIZE = 512
STRIDE = 256
BATCH_SIZE = 5
EUCLID_SIZE_ARCSEC = 105.0

DELETE_AFTER_ARCHIVE = True

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / ".." / "data").resolve()
OUT_RUBIN_ROOT = DATA_DIR / "rubin_tiles_tract5063"
OUT_EUCLID_DIR = DATA_DIR / "euclid_tiles_tract5063"
ARCHIVE_DIR = DATA_DIR / "batch_archives_tract5063"

OUT_RUBIN_ROOT.mkdir(parents=True, exist_ok=True)
OUT_EUCLID_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


butler = Butler(REPO, collections=COLLECTION)


def get_patches_in_tract(butler, tract, band="r", datasetType=DATASETTYPE, skymap=SKYMAP):
    refs = butler.query_datasets(
        datasetType,
        where="tract = tract AND band = band AND skymap = skymap",
        bind={"tract": tract, "band": band, "skymap": skymap},
        with_dimension_records=True,
    )
    return sorted({ref.dataId["patch"] for ref in refs})


def load_patch_exposures_by_id(butler, tract, patch, bands=BANDS_RUBIN, datasetType=DATASETTYPE, skymap=SKYMAP):
    exps = {}
    available = []
    for b in bands:
        dataId = {"tract": tract, "patch": patch, "band": b, "skymap": skymap}
        try:
            exps[b] = butler.get(datasetType, dataId=dataId)
            available.append(b)
        except Exception as e:
            print(f"  skipping band {b} for patch {patch}: {e}")
    if not available:
        raise RuntimeError(f"No bands found for patch {patch}")
    wcs_full = exps[available[0]].getWcs()
    return exps, wcs_full, available


def wcs_to_hdr_dict_lsst(wcs_lsst):
    md = wcs_lsst.getFitsMetadata()
    return {k: md.getScalar(k) for k in md.names()}


def sanitize_rms(rms, huge=1e10):
    rms = rms.astype(np.float32, copy=False)
    bad = (~np.isfinite(rms)) | (rms <= 0) | (rms > huge)
    rms = rms.copy(); rms[bad] = np.nan
    return rms


def load_euclid_cutouts(ra, dec, size_arcsec, bands=BANDS_EUCLID, collection="euclid_DpdMerBksMosaic", radius_arcsec=60):
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
    tab = Irsa.query_sia(pos=(coord, radius_arcsec*u.arcsec), collection=collection)
    if not isinstance(tab, Table):
        tab = tab.to_table()
    out_img = {b: None for b in bands}; out_var = {b: None for b in bands}; wcs_out = {}

    def get_row(band, subtype):
        m = (tab["energy_bandpassname"] == band) & (tab["dataproduct_subtype"] == subtype)
        rows = tab[m]; return rows[0] if len(rows) else None

    for b in bands:
        row_sci = get_row(b, "science")
        if row_sci is None:
            continue
        with fsspec.open(row_sci["access_url"], "rb") as f:
            with fits.open(f, memmap=False) as hdul:
                wcs0 = WCS(hdul[0].header)
                cut = Cutout2D(hdul[0].data, coord, size_arcsec * u.arcsec, wcs=wcs0)
                out_img[b] = np.array(cut.data, dtype=np.float32)
                wcs_out[b] = cut.wcs
        row_rms = get_row(b, "noise")
        if row_rms is None:
            continue
        with fsspec.open(row_rms["access_url"], "rb") as f:
            with fits.open(f, memmap=False) as hdul:
                wcsn = WCS(hdul[0].header)
                cutn = Cutout2D(hdul[0].data, coord, size_arcsec * u.arcsec, wcs=wcsn)
                rms = np.array(cutn.data, dtype=np.float32)
        rms = sanitize_rms(rms, huge=1e10)
        out_var[b] = rms * rms
    return out_img, out_var, wcs_out


def iter_tile_positions(h, w, tile_size, stride):
    for y0 in range(0, h - tile_size + 1, stride):
        for x0 in range(0, w - tile_size + 1, stride):
            yield x0, y0


def save_rubin_tile(exps, wcs_full, bands, out_dir, x0, y0, tile_size=TILE_SIZE, stride=STRIDE):
    patch_origin = exps[bands[0]].getXY0()
    x0_patch = patch_origin.getX()
    y0_patch = patch_origin.getY()

    global_cx = x0_patch + x0 + (tile_size - 1) / 2.0
    global_cy = y0_patch + y0 + (tile_size - 1) / 2.0
    sp_global = wcs_full.pixelToSky(global_cx, global_cy)
    ra_c  = sp_global.getRa().asDegrees()
    dec_c = sp_global.getDec().asDegrees()

    wcs_local = wcs_full.copyAtShiftedPixelOrigin(geom.Extent2D(-(x0_patch + x0), -(y0_patch + y0)))

    imgs, vars_, masks = [], [], []
    for b in bands:
        exp = exps[b]
        img = exp.image.array[y0:y0+tile_size, x0:x0+tile_size].astype(np.float32)
        var = exp.variance.array[y0:y0+tile_size, x0:x0+tile_size].astype(np.float32)
        msk = exp.mask.array[y0:y0+tile_size, x0:x0+tile_size].astype(np.int32)
        imgs.append(img); vars_.append(var); masks.append(msk)

    imgs_stacked  = np.stack(imgs,  axis=0)
    vars_stacked  = np.stack(vars_, axis=0)
    masks_stacked = np.stack(masks, axis=0)

    tile_id = f"tile_x{x0:05d}_y{y0:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"{tile_id}.npz"
    np.savez_compressed(
        fn,
        img=imgs_stacked,
        var=vars_stacked,
        mask=masks_stacked,
        wcs_hdr=wcs_to_hdr_dict_lsst(wcs_local),
        x0=np.int32(x0), y0=np.int32(y0),
        tile_id=np.bytes_(tile_id),
        ra_center=np.float64(ra_c),
        dec_center=np.float64(dec_c),
        tile_size=np.int32(tile_size),
        stride=np.int32(stride),
        bands=np.array(list(bands)),
    )
    return fn, tile_id, ra_c, dec_c


def save_euclid_tile(tile_id, ra_c, dec_c, out_dir):
    out_fn = out_dir / f"{tile_id}_euclid.npz"
    if out_fn.exists():
        return out_fn
    eu_imgs, eu_var, eu_wcss = load_euclid_cutouts(ra_c, dec_c, size_arcsec=EUCLID_SIZE_ARCSEC, bands=BANDS_EUCLID)
    save_dict = {"ra_center": ra_c, "dec_center": dec_c, "tile_id": tile_id}
    for b in BANDS_EUCLID:
        if eu_imgs[b] is not None:
            save_dict[f"img_{b}"] = eu_imgs[b]
            save_dict[f"wcs_{b}"] = eu_wcss[b].to_header_string()
        if eu_var[b] is not None:
            save_dict[f"var_{b}"] = eu_var[b]
    np.savez_compressed(out_fn, **save_dict)
    return out_fn


def archive_batch(batch_id, rubin_files, euclid_files):
    archive_path = ARCHIVE_DIR / f"batch_{batch_id:05d}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for p in rubin_files + euclid_files:
            if not p.exists():
                continue
            arcname = p.relative_to(DATA_DIR)
            tar.add(p, arcname=str(arcname))
    print(f"Archived batch {batch_id}: {archive_path}")
    return archive_path


def delete_files(paths):
    for p in paths:
        try:
            p.unlink()
        except Exception as e:
            print(f"  failed to delete {p}: {e}")


def main():
    patch_ids = get_patches_in_tract(butler, TRACT, band="r")
    print(f"Tract {TRACT} has {len(patch_ids)} patches with r-band: {patch_ids}")

    batch = []
    batch_id = 0

    for patch in patch_ids:
        try:
            exps, wcs_full, bands_present = load_patch_exposures_by_id(butler, tract=TRACT, patch=patch)
        except Exception as e:
            print(f"Skipping patch {patch}: {e}")
            continue

        h, w = exps[bands_present[0]].image.array.shape
        out_dir = OUT_RUBIN_ROOT / f"patch{int(patch):02d}"

        for x0, y0 in iter_tile_positions(h, w, TILE_SIZE, STRIDE):
            r_fn, tile_id, ra_c, dec_c = save_rubin_tile(
                exps, wcs_full, bands_present, out_dir, x0, y0,
                tile_size=TILE_SIZE, stride=STRIDE,
            )
            batch.append((r_fn, tile_id, ra_c, dec_c))

            if len(batch) >= BATCH_SIZE:
                batch_id += 1
                rubin_files = [b[0] for b in batch]
                euclid_files = []
                for _, tid, ra_c, dec_c in batch:
                    try:
                        euclid_files.append(save_euclid_tile(tid, ra_c, dec_c, OUT_EUCLID_DIR))
                        print(f"Saved Euclid match for {tid}")
                    except Exception as e:
                        print(f"Failed to fetch Euclid for {tid}: {e}")

                archive_batch(batch_id, rubin_files, euclid_files)
                if DELETE_AFTER_ARCHIVE:
                    delete_files(rubin_files + euclid_files)
                batch = []

    if batch:
        batch_id += 1
        rubin_files = [b[0] for b in batch]
        euclid_files = []
        for _, tid, ra_c, dec_c in batch:
            try:
                euclid_files.append(save_euclid_tile(tid, ra_c, dec_c, OUT_EUCLID_DIR))
                print(f"Saved Euclid match for {tid}")
            except Exception as e:
                print(f"Failed to fetch Euclid for {tid}: {e}")

        archive_batch(batch_id, rubin_files, euclid_files)
        if DELETE_AFTER_ARCHIVE:
            delete_files(rubin_files + euclid_files)

    print("Done.")


if __name__ == "__main__":
    main()
