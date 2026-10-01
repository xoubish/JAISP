"""Pre-flight check for new data: run this before prepare() when porting to other tiles
or another catalogue version. Reads only; no stamps are cut and nothing is trained.

    python -m simple_regressor.check_inputs --config configs/local.json [--n-check 5]

Checks: tiles found with cfg.tile_glob; npz keys; WCS parses; pixel scale vs
cfg.native_pixscale; catalogue HDU and every configured column; selection count;
how many selected sources fall inside the checked tiles; the patch split that
prepare() will use.
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

from .config import load_config
from . import geometry as geo
from .prepare import _find_tiles


def check(cfg, n_check=5):
    ok = True
    def bad(msg):
        nonlocal ok; ok = False; print("  [FAIL]", msg)

    print(f"== tiles: {cfg.tiles_root}  (pattern {cfg.tile_glob})")
    tiles = _find_tiles(cfg.tiles_root, cfg.tile_glob)
    print(f"  found {len(tiles)} tiles")
    if not tiles:
        bad("no tiles found; check tiles_root / tile_glob"); return False
    patches = sorted({Path(t).parent.name for t in tiles})
    print(f"  patch folders ({len(patches)}): {patches[:12]}{' ...' if len(patches) > 12 else ''}")

    print(f"== catalogue: {cfg.catalog}  (HDU {cfg.catalog_hdu})")
    try:
        with fits.open(cfg.catalog) as h:
            cols = {c.lower() for c in h[cfg.catalog_hdu].columns.names}
            nrow = len(h[cfg.catalog_hdu].data)
    except Exception as e:
        bad(f"cannot read catalogue: {e!r}"); return False
    print(f"  {nrow} rows")
    need = dict(id_column=cfg.id_column, ra_column=cfg.ra_column, dec_column=cfg.dec_column,
                mag_column=cfg.mag_column, flux_column=cfg.flux_column,
                fluxerr_column=cfg.fluxerr_column, sel_flux_column=cfg.sel_flux_column,
                sel_fluxerr_column=cfg.sel_fluxerr_column)
    for k, v in need.items():
        if v.lower() not in cols:
            bad(f"{k}='{v}' not in catalogue")
    for flag, on in [("vis_det", cfg.require_vis_det), ("spurious_flag", cfg.require_spurious_zero),
                     ("det_quality_flag", bool(cfg.det_quality_mask))]:
        if on and flag not in cols:
            print(f"  [warn] '{flag}' not in catalogue -> that cut is silently skipped")
    if not ok:
        return False

    from .catalog import load_sources
    src = load_sources(cfg)
    if len(src) == 0:
        bad("no sources pass catalog selection"); return False
    print(f"  selected {len(src)} sources; label mag range "
          f"{np.nanmin(src.mag):.2f}-{np.nanmax(src.mag):.2f}; "
          f"median label flux {np.median(src.flux):.3g} (expected unit: microJy if ZP={cfg.flux_zeropoint})")
    print(f"  catalogue RA {src.ra.min():.4f}..{src.ra.max():.4f}, Dec {src.dec.min():.4f}..{src.dec.max():.4f}")

    print(f"== checking {min(n_check, len(tiles))} tiles")
    idx = np.linspace(0, len(tiles) - 1, min(n_check, len(tiles))).astype(int)
    n_in_tot = 0
    for i in idx:
        tf = tiles[i]; name = f"{Path(tf).parent.name}/{Path(tf).name}"
        try:
            z = np.load(tf, allow_pickle=True)
            miss = [k for k in (cfg.img_key, cfg.wcs_key) if k not in z.files]
            if miss:
                bad(f"{name}: missing keys {miss}; has {z.files}"); continue
            if cfg.var_key not in z.files:
                bad(f"{name}: missing required variance map {cfg.var_key}"); continue
            img = np.asarray(z[cfg.img_key])
            if cfg.img_plane >= 0:
                img = img[cfg.img_plane]
            if img.ndim != 2:
                bad(f"{name}: image shape {img.shape}; set img_plane for a band cube"); continue
            wcs = geo.parse_wcs(z[cfg.wcs_key])
        except Exception as e:
            print(f"  [warn] {name}: unreadable ({e!r:.80}); prepare() will skip it"); continue
        ps = geo.pixel_scale_arcsec(wcs)
        x, y = geo.world_to_pixel(wcs, src.ra, src.dec)
        h, w = img.shape; half = cfg.stamp // 2
        n_in = int(((x >= half) & (y >= half) & (x < w - half) & (y < h - half)).sum())
        n_in_tot += n_in
        fin = np.isfinite(img).mean()
        print(f"  {name}: shape {img.shape}, {ps:.4f}\"/px, finite {fin:.0%}, "
              f"{n_in} selected sources fully inside a {cfg.stamp}px stamp")
        if abs(ps - cfg.native_pixscale) > 0.01 * cfg.native_pixscale:
            bad(f"pixel scale {ps:.4f} != cfg.native_pixscale {cfg.native_pixscale}")
    if n_in_tot == 0:
        bad("no selected source inside the checked tiles: catalogue and tiles do not overlap?")

    print("== split (split_by=%s)" % cfg.split_by)
    if cfg.split_by == "patch":
        val = list(cfg.val_patches) or ([patches[-2]] if len(patches) >= 3 else [])
        test = list(cfg.test_patches) or ([patches[-1]] if len(patches) >= 2 else [])
        print(f"  val={val}  test={test}  (auto = last two in *string* order; set "
              f"val_patches/test_patches explicitly for new data)")
        if set(val) & set(test):
            bad("validation and test patches overlap")
        if not val or not test or not (set(patches) - set(val + test)):
            bad("patch split needs nonempty train/validation/test; use RA mode for flat tile folders")
        for p in val + test:
            if p not in patches:
                bad(f"split patch '{p}' is not a tile folder name")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-check", type=int, default=5)
    a = ap.parse_args()
    raise SystemExit(0 if check(load_config(a.config), a.n_check) else 1)


if __name__ == "__main__":
    main()
