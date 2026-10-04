"""Select bright sources, cut VIS stamps from tiles_product, and build a cache.

Output cache (a single .npz + metadata.json) holds, per unique source:
  stamps  [N, C, S, S] float32   C=2: VIS image, VIS RMS  (channel use set at train time)
  flux    [N] float32  microJy   (MER FLUX_DETECTION_TOTAL)
  fluxerr [N] float32  microJy
  mag, ra, dec, object_id, split ("train"/"val"/"test")
Each source is assigned to the tile where it sits most centred; duplicates from
overlapping tiles are dropped.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from .config import load_config
from .catalog import load_sources
from . import geometry as geo


def _find_tiles(root, pattern="*_euclid.npz"):
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))


def _spatial_split(ra, dec, patch, cfg):
    """Assign train/val/test. 'patch' holds out whole patches; 'ra' uses RA stripes.

    "" marks a dropped source (guard band, RA mode only).
    """
    if cfg.split_by == "patch":
        uniq = sorted(set(patch.tolist()))
        val_p = list(cfg.val_patches) or ([uniq[-2]] if len(uniq) >= 3 else [])
        test_p = list(cfg.test_patches) or ([uniq[-1]] if len(uniq) >= 2 else [])
        if set(val_p) & set(test_p):
            raise ValueError("Validation and test patches must be disjoint")
        if not set(val_p + test_p).issubset(uniq):
            raise ValueError("Requested holdout patches have no usable sources")
        split = np.full(len(ra), "train", dtype=object)
        split[np.isin(patch, val_p)] = "val"
        split[np.isin(patch, test_p)] = "test"
        return split, dict(mode="patch", patches=uniq, val_patches=val_p,
                           test_patches=test_p, n_guard_dropped=0)
    if cfg.split_by == "ra":
        b1 = np.quantile(ra, cfg.train_frac)
        b2 = np.quantile(ra, cfg.train_frac + cfg.val_frac)
        buf_deg = (cfg.buffer_arcsec / 3600.0) / max(np.cos(np.deg2rad(np.median(dec))), 1e-3)
        split = np.full(len(ra), "", dtype=object)
        split[ra <= b1] = "train"
        split[(ra > b1) & (ra <= b2)] = "val"
        split[ra > b2] = "test"
        guard = (np.abs(ra - b1) < buf_deg) | (np.abs(ra - b2) < buf_deg)
        split[guard] = ""
        return split, dict(mode="ra", ra_boundary_1=float(b1), ra_boundary_2=float(b2),
                           buffer_deg=float(buf_deg), n_guard_dropped=int(guard.sum()))
    raise ValueError(f"Unknown split_by={cfg.split_by}")


def centre_snr(stamps, pixscale, r_aper):
    """S/N of the background-subtracted flux inside r_aper (arcsec) at the stamp centre.
    stamps: [N,2,S,S] (image, rms; rms==0 marks masked pixels). Sky = median of an annulus
    near the stamp edge (86-98% of the half-size)."""
    S = stamps.shape[-1]; c = (S - 1) / 2.0
    y, x = np.mgrid[0:S, 0:S]; r = np.hypot(x - c, y - c) * pixscale
    half = S / 2 * pixscale
    ap, sky = r <= r_aper, (r > 0.86 * half) & (r < 0.98 * half)
    out = np.zeros(len(stamps))
    for i in range(len(stamps)):
        im = stamps[i, 0].astype(np.float64); rm = stamps[i, 1].astype(np.float64); v = rm > 0
        b = np.median(im[sky & v]) if (sky & v).any() else 0.0
        m = ap & v
        var = (rm[m] ** 2).sum()
        out[i] = (im[m] - b).sum() / np.sqrt(var) if var > 0 else 0.0
    return out


def prepare(cfg):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = load_sources(cfg)
    print(f"[prepare] selected {len(src)} catalogue sources "
          f"(mag {cfg.mag_min} < {cfg.mag_column} < {cfg.mag_max})")

    tiles = _find_tiles(cfg.tiles_root, cfg.tile_glob)
    if not tiles:
        raise FileNotFoundError(f"No {cfg.tile_glob} under {cfg.tiles_root}")
    if cfg.n_tiles and len(tiles) > cfg.n_tiles:
        # Even stride over the path-sorted list spreads the subset across patches.
        stride = len(tiles) / cfg.n_tiles
        tiles = [tiles[int(i * stride)] for i in range(cfg.n_tiles)]
    print(f"[prepare] scanning {len(tiles)} tiles")

    S = cfg.stamp
    yy, xx = np.mgrid[:S, :S]
    rr = np.hypot(xx - (S - 1) / 2, yy - (S - 1) / 2)
    sky = (rr > 0.86 * S / 2) & (rr < 0.98 * S / 2)
    # best stamp per object_id: (border_distance, stamp[2,S,S])
    best = {}
    meta_flux = {}
    meta_patch = {}
    skipped = []
    for ti, tf in enumerate(tiles):
        patch = Path(tf).parent.name
        # Some tiles in the dataset are corrupt/unreadable; skip them instead of crashing.
        try:
            z = np.load(tf, allow_pickle=True)
            if cfg.img_key not in z.files:
                skipped.append([tf, f"no {cfg.img_key}"]); continue
            img = np.asarray(z[cfg.img_key], dtype=np.float32)
            var = np.asarray(z[cfg.var_key], dtype=np.float32) if cfg.var_key in z.files else None
            if cfg.img_plane >= 0:                  # band cube, e.g. Rubin [6,H,W]
                img = img[cfg.img_plane]
                var = var[cfg.img_plane] if var is not None else None
            if img.ndim != 2:
                raise ValueError(f"image is {img.shape}; set cfg.img_plane for a band cube")
            if var is None or var.shape != img.shape:
                raise ValueError("A matching variance map is required")
            wcs = geo.parse_wcs(z[cfg.wcs_key])
        except Exception as e:
            skipped.append([tf, repr(e)[:150]])
            print(f"[prepare]  SKIP unreadable tile {patch}/{Path(tf).name}: {repr(e)[:90]}")
            continue
        x, y = geo.world_to_pixel(wcs, src.ra, src.dec)
        half = S // 2
        h, w = img.shape
        inside = (x >= half + cfg.edge_margin) & (y >= half + cfg.edge_margin) & \
                 (x < w - half - cfg.edge_margin) & (y < h - half - cfg.edge_margin)
        idx = np.where(inside)[0]
        n_new = 0
        for i in idx:
            oid = int(src.object_id[i])
            bd = geo.border_distance(x[i], y[i], img.shape, S)
            if oid in best and best[oid][0] >= bd:
                continue
            stamp_img = geo.cut_stamp(img, x[i], y[i], S)
            if stamp_img is None:
                continue
            stamp_var = geo.cut_stamp(var, x[i], y[i], S)
            valid = np.isfinite(stamp_img) & np.isfinite(stamp_var) & (stamp_var > 0)
            if valid.mean() < cfg.valid_frac_min:
                continue
            if not (sky & valid).any():
                continue
            stamp_img = np.where(valid, stamp_img, 0.0)
            stamp_rms = np.sqrt(np.where(valid, stamp_var, 0.0))
            stack = np.stack([stamp_img, stamp_rms]).astype(np.float32)
            best[oid] = (bd, stack)
            meta_flux[oid] = (float(src.flux[i]), float(src.fluxerr[i]), float(src.mag[i]),
                              float(src.ra[i]), float(src.dec[i]))
            meta_patch[oid] = patch
            n_new += 1
        print(f"[prepare]  tile {ti+1}/{len(tiles)} {Path(tf).parent.name}/{Path(tf).name}: "
              f"{len(idx)} inside, {n_new} new/improved (total {len(best)})")

    if not best:
        raise RuntimeError("No usable stamps; check overlap, masks and variance maps")
    oids = np.array(sorted(best.keys()), dtype=np.int64)
    if cfg.max_sources and len(oids) > cfg.max_sources:
        rng0 = np.random.default_rng(cfg.seed)
        oids = np.sort(rng0.choice(oids, cfg.max_sources, replace=False))
    stamps = np.stack([best[o][1] for o in oids])                      # [N,2,S,S] f32
    flux = np.array([meta_flux[o][0] for o in oids], dtype=np.float32)
    fluxerr = np.array([meta_flux[o][1] for o in oids], dtype=np.float32)
    mag = np.array([meta_flux[o][2] for o in oids], dtype=np.float32)
    ra = np.array([meta_flux[o][3] for o in oids], dtype=np.float64)
    dec = np.array([meta_flux[o][4] for o in oids], dtype=np.float64)
    patch = np.array([meta_patch[o] for o in oids])

    split, split_info = _spatial_split(ra, dec, patch, cfg)
    kept = split != ""
    counts = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    print(f"[prepare] unique sources {len(oids)}; split {counts}; "
          f"dropped guard {split_info['n_guard_dropped']}")

    # ---- empty-centre cut (image-only) ----
    csnr = centre_snr(stamps, cfg.native_pixscale, cfg.centre_aper_arcsec)
    centre_drop = {}
    if cfg.centre_snr_min > 0:
        if cfg.centre_filter not in ("train", "all"):
            raise ValueError("centre_filter must be 'train' or 'all'")
        empty = csnr < cfg.centre_snr_min
        target = (split == "train") if cfg.centre_filter == "train" else (split != "")
        drop = empty & target & kept
        centre_drop = {s: int((drop & (split == s)).sum()) for s in ("train", "val", "test")}
        bymag = {f"{lo}-{lo+1}": int((drop & (mag >= lo) & (mag < lo + 1)).sum()) for lo in range(16, 27)}
        bymag = {k: v for k, v in bymag.items() if v}
        print(f"[prepare] empty-centre cut (S/N<{cfg.centre_snr_min} in r={cfg.centre_aper_arcsec}\", "
              f"{cfg.centre_filter}): dropped {int(drop.sum())} {centre_drop}; by label mag {bymag}")
        kept &= ~drop
        counts = {s: int(((split == s) & kept).sum()) for s in ("train", "val", "test")}
        print(f"[prepare] after cut: {counts}")

    cache = out / "stamps_cache.npz"
    np.savez(cache,
             stamps=stamps[kept], flux=flux[kept], fluxerr=fluxerr[kept],
             mag=mag[kept], ra=ra[kept], dec=dec[kept], patch=patch[kept].astype("U16"),
             object_id=oids[kept], split=split[kept].astype("U8"), centre_snr=csnr[kept])
    if skipped:
        print(f"[prepare] skipped {len(skipped)} unreadable/invalid tile(s)")
    meta = dict(cache_version=2, kind="simple_vis_flux_stamps", stamp=S, n_sources=int(kept.sum()),
                counts=counts, split_info=split_info, tiles=len(tiles),
                skipped_tiles=skipped, centre_cut_dropped=centre_drop,
                mag_max=cfg.mag_max, mag_min=cfg.mag_min,
                config=cfg.to_json())
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[prepare] wrote {cache} ({stamps[kept].nbytes/1e6:.1f} MB) and metadata.json")
    return str(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--tiles-root", default=None)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--mag-max", type=float, default=None)
    ap.add_argument("--max-sources", type=int, default=None)
    ap.add_argument("--n-tiles", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, tiles_root=a.tiles_root, catalog=a.catalog,
                      output_dir=a.output_dir, mag_max=a.mag_max,
                      max_sources=a.max_sources, n_tiles=a.n_tiles)
    prepare(cfg)


if __name__ == "__main__":
    main()
