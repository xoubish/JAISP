"""Build PSF training set using GAIA stars instead of CenterNet detections.

GAIA gives sub-mas RA/Dec — no per-stamp centroiding needed, no VIS-to-Rubin
half-pixel conversion bugs. Each band's WCS projects (RA, Dec) → pixel coords
directly, eliminating the entire class of failures that bit the previous build
(detected positions off by 8-15 pixels in low-SNR Rubin stamps).

Pipeline:
  1. Walk all tile pairs, get the field footprint from tile centres.
  2. Single GAIA DR3 cone-search over the union footprint with quality cuts:
     - phot_g_mean_mag in [mag_min, mag_max] (default 16-21)
     - astrometric_excess_noise < 1.5     (well-fit single source)
     - ipd_frac_multi_peak = 0            (not a blend)
     - duplicated_source = false
  3. For each tile, find GAIA stars whose VIS pixel falls inside.
  4. For each surviving star, project (RA, Dec) → pixel coords for every
     band using that band's own WCS, then cut a stamp_size × stamp_size
     stamp at sub-pixel precision.
  5. Output: per-band .npz files with the same schema as
     build_psf_v4_training_set.py so psf_field_pca.py drops in unchanged.

Run::

    PYTHONPATH=models python models/psf/build_psf_gaia_training_set.py \\
        --rubin-dir   data/rubin_tiles_all \\
        --euclid-dir  data/euclid_tiles_all \\
        --out-dir     data/psf_training_gaia \\
        --stamp-size  32 \\
        --max-centroid-resid-px 1.0
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io.fits import Header
from astropy.time import Time
from astropy.wcs import WCS

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from astrometry2.source_matching import safe_header_from_card_string  # noqa: E402
from psf.build_psf_v4_training_set import (  # noqa: E402
    _cut_stamp, _estimate_snr_flux, _isolation_mask, _refine_centroid_in_stamp,
)


ALL_BANDS = ["rubin_u", "rubin_g", "rubin_r", "rubin_i", "rubin_z", "rubin_y",
             "euclid_VIS", "euclid_Y", "euclid_J", "euclid_H"]
RUBIN_BANDS = ["rubin_u", "rubin_g", "rubin_r", "rubin_i", "rubin_z", "rubin_y"]


def _centroid_residual_in_stamp(
    stamp_img: np.ndarray,
    frac_x: float,
    frac_y: float,
    ap_radius: float = 5.0,
) -> Tuple[float, float, float]:
    """Measure aperture centroid residual relative to the recorded sub-pixel centre."""
    img = stamp_img[0] if stamp_img.ndim == 3 else stamp_img
    H, W = img.shape[-2:]
    yy, xx = np.indices((H, W), dtype=np.float64)
    tx = W // 2 + float(frac_x)
    ty = H // 2 + float(frac_y)
    r = np.hypot(xx - tx, yy - ty)
    bg_mask = r > min(14.0, H * 0.38)
    bg = float(np.nanmedian(img[bg_mask])) if bg_mask.any() else float(np.nanmedian(img))
    weight = np.clip(img - bg, 0.0, None) * (r < float(ap_radius))
    total = float(np.nansum(weight))
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    cx = float(np.nansum(xx * weight) / total)
    cy = float(np.nansum(yy * weight) / total)
    dx = cx - tx
    dy = cy - ty
    return dx, dy, float(np.hypot(dx, dy))


# ============================================================
# GAIA query
# ============================================================

def query_gaia_field(ra_centre: float, dec_centre: float, radius_deg: float,
                     mag_min: float, mag_max: float) -> Dict[str, np.ndarray]:
    from astroquery.gaia import Gaia
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    Gaia.ROW_LIMIT = -1

    adql = f"""
    SELECT source_id, ra, dec, ref_epoch, pmra, pmdec, phot_g_mean_mag,
           parallax, parallax_over_error,
           astrometric_excess_noise, ipd_frac_multi_peak,
           duplicated_source
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                       CIRCLE('ICRS', {ra_centre}, {dec_centre}, {radius_deg}))
      AND phot_g_mean_mag BETWEEN {mag_min} AND {mag_max}
      AND astrometric_excess_noise < 1.5
      AND ipd_frac_multi_peak = 0
      AND duplicated_source = 'false'
    """
    job = Gaia.launch_job_async(adql)
    t = job.get_results()
    return {
        "source_id": np.array(t["source_id"], dtype=np.int64),
        "ra": np.array(t["ra"], dtype=np.float64),
        "dec": np.array(t["dec"], dtype=np.float64),
        "ref_epoch": np.array(t["ref_epoch"], dtype=np.float64),
        "pmra": np.array(t["pmra"], dtype=np.float64),
        "pmdec": np.array(t["pmdec"], dtype=np.float64),
        "g_mag": np.array(t["phot_g_mean_mag"], dtype=np.float32),
    }


def _resolve_target_time(args: argparse.Namespace) -> Optional[Time]:
    if args.obs_epoch_mjd is not None:
        return Time(float(args.obs_epoch_mjd), format="mjd", scale="tcb")
    if args.obs_epoch_year is not None:
        return Time(float(args.obs_epoch_year), format="jyear", scale="tcb")
    return None


def _apply_gaia_proper_motion(
    gaia: Dict[str, np.ndarray],
    target_time: Optional[Time],
) -> Dict[str, np.ndarray]:
    if target_time is None:
        return gaia

    out = dict(gaia)
    ra = np.asarray(gaia["ra"], dtype=np.float64)
    dec = np.asarray(gaia["dec"], dtype=np.float64)
    ref_epoch = np.asarray(gaia.get("ref_epoch", np.full_like(ra, 2016.0)), dtype=np.float64)
    pmra = np.asarray(gaia.get("pmra", np.zeros_like(ra)), dtype=np.float64)
    pmdec = np.asarray(gaia.get("pmdec", np.zeros_like(ra)), dtype=np.float64)
    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(ref_epoch) & np.isfinite(pmra) & np.isfinite(pmdec)
    if not valid.any():
        print("No finite Gaia proper motions found; using catalog RA/Dec without propagation.")
        return out

    coord = SkyCoord(
        ra=ra[valid] * u.deg,
        dec=dec[valid] * u.deg,
        pm_ra_cosdec=pmra[valid] * u.mas / u.yr,
        pm_dec=pmdec[valid] * u.mas / u.yr,
        frame="icrs",
        obstime=Time(ref_epoch[valid], format="jyear", scale="tcb"),
    )
    moved = coord.apply_space_motion(new_obstime=target_time)
    ra_new = ra.copy()
    dec_new = dec.copy()
    ra_new[valid] = moved.ra.deg
    dec_new[valid] = moved.dec.deg
    out["ra"] = ra_new
    out["dec"] = dec_new
    out["propagated_epoch_jyear"] = np.full(ra.shape, float(target_time.jyear), dtype=np.float64)
    print(
        f"Propagated {int(valid.sum())}/{len(valid)} Gaia sources to epoch "
        f"{float(target_time.jyear):.3f}."
    )
    return out


# ============================================================
# Tile I/O
# ============================================================

def _read_rubin_tile(path: Path):
    d = np.load(path, allow_pickle=True, mmap_mode="r")
    img = np.asarray(d["img"], dtype=np.float32)
    var = np.asarray(d["var"], dtype=np.float32)
    rms = np.sqrt(np.clip(var, 0.0, None))
    hdr = Header(d["wcs_hdr"].item())
    wcs = WCS(hdr)
    bands = [str(b) for b in d["bands"]] if "bands" in d.files else RUBIN_BANDS
    ra_c = float(d["ra_center"]); dec_c = float(d["dec_center"])
    return img, rms, wcs, bands, ra_c, dec_c


def _read_euclid_tile(path: Path):
    d = np.load(path, allow_pickle=True, mmap_mode="r")
    imgs, rmss, wcss = {}, {}, {}
    for short in ["VIS", "Y", "J", "H"]:
        img_key, var_key, wcs_key = f"img_{short}", f"var_{short}", f"wcs_{short}"
        if img_key not in d.files:
            continue
        imgs[short] = np.asarray(d[img_key], dtype=np.float32)
        rmss[short] = np.sqrt(np.clip(np.asarray(d[var_key], dtype=np.float32),
                                      0.0, None))
        hdr = safe_header_from_card_string(str(d[wcs_key]))
        wcss[short] = WCS(hdr)
    return imgs, rmss, wcss


def _tile_pairs(rubin_dir: Path, euclid_dir: Path) -> List[Tuple[Path, Path, str]]:
    pairs = []
    for r in sorted(rubin_dir.glob("tile_*.npz")):
        e = euclid_dir / f"{r.stem}_euclid.npz"
        if e.exists():
            pairs.append((r, e, r.stem))
    return pairs


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rubin-dir", required=True, type=Path)
    p.add_argument("--euclid-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--stamp-size", type=int, default=32)
    p.add_argument("--isolation-arcsec", type=float, default=1.5)
    p.add_argument("--snr-thr", type=float, default=10.0)
    p.add_argument(
        "--max-centroid-resid-px",
        type=float,
        default=1.0,
        help=(
            "Reject stamps whose post-cut aperture centroid differs from the "
            "recorded sub-pixel centre by more than this many native pixels. "
            "Default: 1.0. Set <=0 to disable."
        ),
    )
    p.add_argument("--centroid-ap-radius", type=float, default=5.0)
    p.add_argument("--mag-min", type=float, default=16.0,
                   help="Min GAIA G mag (brighter than this saturates Rubin/Euclid).")
    p.add_argument("--mag-max", type=float, default=21.0,
                   help="Max GAIA G mag (fainter is too noisy).")
    p.add_argument("--max-tiles", type=int, default=None,
                   help="Process only the first N tiles (for testing).")
    p.add_argument("--cache", type=Path,
                   default=Path("/tmp/gaia_psf_cache.npz"),
                   help="Cache GAIA query result so reruns skip the network call.")
    p.add_argument(
        "--obs-epoch-year",
        type=float,
        default=None,
        help="Propagate Gaia coordinates to this Julian year before cutting stamps.",
    )
    p.add_argument(
        "--obs-epoch-mjd",
        type=float,
        default=None,
        help="Propagate Gaia coordinates to this MJD before cutting stamps.",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_time = _resolve_target_time(args)

    # ---- Tile pairs + field footprint ---------------------------------------
    pairs = _tile_pairs(args.rubin_dir, args.euclid_dir)
    if args.max_tiles:
        pairs = pairs[:args.max_tiles]
    print(f"Tile pairs: {len(pairs)}")
    if not pairs:
        raise RuntimeError("No tile pairs found.")

    # Field centre + radius from tile centres (Rubin half-tile ≈ 51″ → 0.014°).
    ras, decs = [], []
    for r_path, _, _ in pairs:
        d = np.load(r_path, allow_pickle=True, mmap_mode="r")
        ras.append(float(d["ra_center"])); decs.append(float(d["dec_center"]))
    ras, decs = np.array(ras), np.array(decs)
    ra_c = float(ras.mean()); dec_c = float(decs.mean())
    field_radius = float(max(np.abs(ras - ra_c).max(),
                             np.abs(decs - dec_c).max())) + 0.05
    print(f"Field centre: ({ra_c:.4f}, {dec_c:.4f})  cone radius: {field_radius:.3f} deg")

    # ---- GAIA ---------------------------------------------------------------
    required_cache_keys = {"source_id", "ra", "dec", "ref_epoch", "pmra", "pmdec", "g_mag"}
    if args.cache.exists():
        gaia = dict(np.load(args.cache))
        if not required_cache_keys.issubset(gaia):
            print(f"Cache {args.cache} is missing Gaia PM/source_id columns; refreshing query.")
            gaia = query_gaia_field(ra_c, dec_c, field_radius,
                                    mag_min=args.mag_min, mag_max=args.mag_max)
            np.savez(args.cache, **gaia)
        print(f"Loaded {len(gaia['ra'])} GAIA sources from cache "
              f"({args.cache}).")
    else:
        print("Querying GAIA DR3 (~30-60 s)...")
        gaia = query_gaia_field(ra_c, dec_c, field_radius,
                                mag_min=args.mag_min, mag_max=args.mag_max)
        np.savez(args.cache, **gaia)
        print(f"Got {len(gaia['ra'])} GAIA sources, cached.")
    if target_time is None:
        print("No observation epoch provided; using Gaia catalog coordinates without PM propagation.")
    gaia = _apply_gaia_proper_motion(gaia, target_time)

    # ---- Per-tile stamp extraction ------------------------------------------
    out: Dict[str, Dict[str, list]] = {b: {
        "stamps": [], "rms": [], "frac_xy": [],
        "pos_norm": [], "pos_pix": [], "pos_vis_pix": [],
        "centroid_resid_px": [],
        "snr": [], "flux": [], "tile_id": [], "g_mag": [], "source_id": [],
    } for b in ALL_BANDS}

    n_used_stars = 0
    for ti, (r_path, e_path, tile_id) in enumerate(pairs):
        r_img, r_rms, r_wcs, r_bands, _, _ = _read_rubin_tile(r_path)
        e_imgs, e_rmss, e_wcss = _read_euclid_tile(e_path)
        if "VIS" not in e_imgs:
            continue
        H_v, W_v = e_imgs["VIS"].shape[-2:]

        # Project all GAIA → VIS pixel and select those inside the tile.
        x_v, y_v = e_wcss["VIS"].wcs_world2pix(gaia["ra"], gaia["dec"], 0)
        margin = args.stamp_size // 2 + 2
        in_v = ((x_v >= margin) & (x_v < W_v - margin) &
                (y_v >= margin) & (y_v < H_v - margin))
        if not in_v.any():
            continue

        # Isolation in VIS pixel space (1 VIS px = 0.1″).
        xy_v_in = np.stack([x_v[in_v], y_v[in_v]], axis=-1)
        keep = _isolation_mask(xy_v_in, args.isolation_arcsec / 0.1)
        if not keep.any():
            continue

        ras_in = gaia["ra"][in_v][keep]
        decs_in = gaia["dec"][in_v][keep]
        x_vis_in = x_v[in_v][keep]
        y_vis_in = y_v[in_v][keep]
        mags_in = gaia["g_mag"][in_v][keep]
        source_ids_in = gaia["source_id"][in_v][keep]
        n_used_stars += int(len(ras_in))

        for band in ALL_BANDS:
            if band in RUBIN_BANDS:
                # Look up band index in r_bands (defaults to RUBIN_BANDS order).
                if band in r_bands:
                    bi = r_bands.index(band)
                else:
                    bi = RUBIN_BANDS.index(band)
                if bi >= r_img.shape[0]:
                    continue
                img = r_img[bi]; rms = r_rms[bi]; wcs = r_wcs
            else:
                short = band.split("_")[1]
                if short not in e_imgs:
                    continue
                img = e_imgs[short]; rms = e_rmss[short]; wcs = e_wcss[short]
            Hb, Wb = img.shape[-2:]

            x_b, y_b = wcs.wcs_world2pix(ras_in, decs_in, 0)
            half = args.stamp_size // 2
            for k in range(len(ras_in)):
                xs0, ys0 = float(x_b[k]), float(y_b[k])
                # First cut at the WCS-projected Gaia position so we have a
                # stamp to centroid on. Even after PM correction, coadd WCS
                # residuals (~tens of mas) shift the source by sub-pixel
                # amounts, leaving a clean dipole in residuals at high SNR.
                stamp_img0, _, _, _, ok = _cut_stamp(
                    img, rms, xs0, ys0, args.stamp_size,
                )
                if not ok:
                    continue
                refined = _refine_centroid_in_stamp(stamp_img0)
                if refined is None:
                    continue
                cy_stamp, cx_stamp = refined
                ix0, iy0 = int(round(xs0)), int(round(ys0))
                xs = (ix0 - half) + cx_stamp
                ys = (iy0 - half) + cy_stamp
                stamp_img, stamp_rms, fx, fy, ok = _cut_stamp(
                    img, rms, xs, ys, args.stamp_size,
                )
                if not ok:
                    continue
                # Reject stamps where the refined peak landed far from centre
                # (refinement locked onto a neighbour or noise spike).
                from scipy.ndimage import gaussian_filter
                _sm = gaussian_filter(stamp_img.astype(np.float32), 1.0,
                                      mode="constant")
                if _sm.max() <= 0:
                    continue
                _iy, _ix = np.unravel_index(int(_sm.argmax()), _sm.shape)
                if abs(_iy - half) > 3 or abs(_ix - half) > 3:
                    continue
                _, _, centroid_resid = _centroid_residual_in_stamp(
                    stamp_img,
                    fx,
                    fy,
                    ap_radius=args.centroid_ap_radius,
                )
                if not np.isfinite(centroid_resid):
                    continue
                if args.max_centroid_resid_px > 0 and centroid_resid > args.max_centroid_resid_px:
                    continue
                snr, flux = _estimate_snr_flux(stamp_img, stamp_rms, fx, fy)
                if snr < args.snr_thr:
                    continue
                pos_norm = np.array([
                    2.0 * xs / max(Wb - 1, 1) - 1.0,
                    2.0 * ys / max(Hb - 1, 1) - 1.0,
                ], dtype=np.float32)
                out[band]["stamps"].append(stamp_img.astype(np.float32))
                out[band]["rms"].append(stamp_rms.astype(np.float32))
                out[band]["frac_xy"].append([fx, fy])
                out[band]["pos_norm"].append(pos_norm)
                out[band]["pos_pix"].append([xs, ys])
                out[band]["pos_vis_pix"].append([float(x_vis_in[k]), float(y_vis_in[k])])
                out[band]["centroid_resid_px"].append(float(centroid_resid))
                out[band]["snr"].append(snr)
                out[band]["flux"].append(flux)
                out[band]["tile_id"].append(tile_id)
                out[band]["g_mag"].append(float(mags_in[k]))
                out[band]["source_id"].append(int(source_ids_in[k]))

        if (ti + 1) % 50 == 0:
            kept = sum(len(out[b]["stamps"]) for b in ALL_BANDS)
            print(f"  ... tile {ti+1:4d}/{len(pairs)}  "
                  f"GAIA stars used: {n_used_stars}  band-stamp total: {kept}")

    # ---- Save ---------------------------------------------------------------
    print()
    print(f"Total GAIA stars across tiles: {n_used_stars}")
    for band in ALL_BANDS:
        rec = out[band]
        n = len(rec["stamps"])
        if n == 0:
            print(f"  {band:14s}  0 stamps — skipping.")
            continue
        path = args.out_dir / f"{band}.npz"
        np.savez_compressed(
            path,
            stamps=np.stack(rec["stamps"]).astype(np.float32),
            rms=np.stack(rec["rms"]).astype(np.float32),
            frac_xy=np.array(rec["frac_xy"], dtype=np.float32),
            pos_norm=np.array(rec["pos_norm"], dtype=np.float32),
            pos_pix=np.array(rec["pos_pix"], dtype=np.float32),
            pos_vis_pix=np.array(rec["pos_vis_pix"], dtype=np.float32),
            centroid_resid_px=np.array(rec["centroid_resid_px"], dtype=np.float32),
            snr=np.array(rec["snr"], dtype=np.float32),
            flux=np.array(rec["flux"], dtype=np.float32),
            tile_id=np.array(rec["tile_id"], dtype=np.str_),
            g_mag=np.array(rec["g_mag"], dtype=np.float32),
            source_id=np.array(rec["source_id"], dtype=np.int64),
        )
        print(f"  {band:14s}  {n:6d} stamps  →  {path}")


if __name__ == "__main__":
    main()
