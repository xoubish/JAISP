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
        --stamp-size  32
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from astropy.io.fits import Header
from astropy.wcs import WCS

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from astrometry2.source_matching import safe_header_from_card_string  # noqa: E402
from psf.build_psf_v4_training_set import (  # noqa: E402
    _cut_stamp, _estimate_snr_flux, _isolation_mask,
)


ALL_BANDS = ["rubin_u", "rubin_g", "rubin_r", "rubin_i", "rubin_z", "rubin_y",
             "euclid_VIS", "euclid_Y", "euclid_J", "euclid_H"]
RUBIN_BANDS = ["rubin_u", "rubin_g", "rubin_r", "rubin_i", "rubin_z", "rubin_y"]


# ============================================================
# GAIA query
# ============================================================

def query_gaia_field(ra_centre: float, dec_centre: float, radius_deg: float,
                     mag_min: float, mag_max: float) -> Dict[str, np.ndarray]:
    from astroquery.gaia import Gaia
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    Gaia.ROW_LIMIT = -1

    adql = f"""
    SELECT ra, dec, phot_g_mean_mag,
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
        "ra": np.array(t["ra"], dtype=np.float64),
        "dec": np.array(t["dec"], dtype=np.float64),
        "g_mag": np.array(t["phot_g_mean_mag"], dtype=np.float32),
    }


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
    p.add_argument("--mag-min", type=float, default=16.0,
                   help="Min GAIA G mag (brighter than this saturates Rubin/Euclid).")
    p.add_argument("--mag-max", type=float, default=21.0,
                   help="Max GAIA G mag (fainter is too noisy).")
    p.add_argument("--max-tiles", type=int, default=None,
                   help="Process only the first N tiles (for testing).")
    p.add_argument("--cache", type=Path,
                   default=Path("/tmp/gaia_psf_cache.npz"),
                   help="Cache GAIA query result so reruns skip the network call.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

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
    if args.cache.exists():
        gaia = dict(np.load(args.cache))
        print(f"Loaded {len(gaia['ra'])} GAIA sources from cache "
              f"({args.cache}).")
    else:
        print("Querying GAIA DR3 (~30-60 s)...")
        gaia = query_gaia_field(ra_c, dec_c, field_radius,
                                mag_min=args.mag_min, mag_max=args.mag_max)
        np.savez(args.cache, **gaia)
        print(f"Got {len(gaia['ra'])} GAIA sources, cached.")

    # ---- Per-tile stamp extraction ------------------------------------------
    out: Dict[str, Dict[str, list]] = {b: {
        "stamps": [], "rms": [], "frac_xy": [],
        "pos_norm": [], "pos_pix": [],
        "snr": [], "flux": [], "tile_id": [], "g_mag": [],
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
        mags_in = gaia["g_mag"][in_v][keep]
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
            for k in range(len(ras_in)):
                xs, ys = float(x_b[k]), float(y_b[k])
                stamp_img, stamp_rms, fx, fy, ok = _cut_stamp(
                    img, rms, xs, ys, args.stamp_size,
                )
                if not ok:
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
                out[band]["snr"].append(snr)
                out[band]["flux"].append(flux)
                out[band]["tile_id"].append(tile_id)
                out[band]["g_mag"].append(float(mags_in[k]))

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
            snr=np.array(rec["snr"], dtype=np.float32),
            flux=np.array(rec["flux"], dtype=np.float32),
            tile_id=np.array(rec["tile_id"], dtype=np.str_),
            g_mag=np.array(rec["g_mag"], dtype=np.float32),
        )
        print(f"  {band:14s}  {n:6d} stamps  →  {path}")


if __name__ == "__main__":
    main()
