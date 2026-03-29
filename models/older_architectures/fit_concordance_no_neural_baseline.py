"""
Fit a smooth concordance field directly from Rubin<->VIS source matches.

This is the explicit non-neural baseline:
  1) detect sources from a shared multiband Rubin image and Euclid VIS
  2) match them in sky coordinates using tile WCS
  3) fit a smooth field = affine trend + RBF residual
  4) export DRA/DDE mesh extensions as a FITS concordance product
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from astropy.io import fits
    from astropy.wcs import WCS
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from source_matching import safe_header_from_card_string
from teacher_fields import build_teacher_field, normalize_band_list, normalize_band_name


def make_concordance_hdu(
    data: np.ndarray,
    extname: str,
    dstep: int,
    rubin_band: str,
    tile_id: str,
    vis_wcs_header: Optional["fits.Header"],
    method: str,
    nmatch: int,
) -> "fits.ImageHDU":
    hdu = fits.ImageHDU(data=data.astype(np.float32), name=extname)
    hdu.header["DSTEP"] = (int(dstep), "Mesh sampling step in VIS pixels")
    hdu.header["DUNIT"] = ("arcsec", "Unit of offset values")
    hdu.header["INTERP"] = ("rbf", "Recommended interpolation method")
    hdu.header["CONCRDNC"] = (True, "This is a concordance offset extension")
    hdu.header["RBNBAND"] = (rubin_band, "Rubin band for this concordance")
    hdu.header["REFFRAME"] = ("euclid_VIS", "Reference frame")
    hdu.header["TILEID"] = (tile_id, "Source tile identifier")
    hdu.header["FITMETH"] = (method[:68], "Field fit method")
    hdu.header["NMATCH"] = (int(nmatch), "Matched sources used in fit")

    if vis_wcs_header is not None:
        for key in [
            "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
            "CD1_1", "CD1_2", "CD2_1", "CD2_2",
            "PC1_1", "PC1_2", "PC2_1", "PC2_2",
            "CDELT1", "CDELT2",
            "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
        ]:
            if key in vis_wcs_header:
                val = vis_wcs_header[key]
                if key in ("CRPIX1", "CRPIX2"):
                    val = val / float(max(1, dstep))
                if key.startswith("CD") or key.startswith("CDELT"):
                    val = val * float(max(1, dstep))
                hdu.header[key] = val
    return hdu


def load_tile_pair(rubin_path: str, euclid_path: str):
    rdata = np.load(rubin_path, allow_pickle=True)
    edata = np.load(euclid_path, allow_pickle=True)
    vis_img = np.nan_to_num(edata["img_VIS"].astype(np.float32), nan=0.0)
    rwcs = WCS(rdata["wcs_hdr"].item())
    vhdr = safe_header_from_card_string(edata["wcs_VIS"].item())
    vwcs = WCS(vhdr)
    return rdata, vis_img, rwcs, vwcs, vhdr


def fit_tile(
    rubin_path: str,
    euclid_path: str,
    tile_id: str,
    rubin_band: str,
    args,
) -> Optional[Dict]:
    try:
        rdata, vis_img, rwcs, vwcs, vhdr = load_tile_pair(rubin_path, euclid_path)
    except Exception as exc:
        print(f"[skip] {tile_id}:{rubin_band} load/wcs failed ({exc})")
        return None

    try:
        built = build_teacher_field(
            rubin_cube=rdata["img"],
            vis_img=vis_img,
            rubin_wcs=rwcs,
            vis_wcs=vwcs,
            rubin_band=rubin_band,
            detect_bands=args.detect_bands,
            dstep=args.dstep,
            min_matches=args.min_matches,
            max_matches=args.max_matches,
            max_sep_arcsec=args.max_sep_arcsec,
            clip_sigma=args.clip_sigma,
            rubin_nsig=args.rubin_nsig,
            vis_nsig=args.vis_nsig,
            rubin_smooth=args.rubin_smooth,
            vis_smooth=args.vis_smooth,
            rubin_min_dist=args.rubin_min_dist,
            vis_min_dist=args.vis_min_dist,
            max_sources_rubin=args.max_sources_rubin,
            max_sources_vis=args.max_sources_vis,
            detect_clip_sigma=args.detect_clip_sigma,
            refine_band_centroids=args.refine_band_centroids,
            refine_radius=args.refine_radius,
            refine_flux_floor_sigma=args.refine_flux_floor_sigma,
            rbf_smoothing=args.rbf_smoothing,
            rbf_neighbors=args.rbf_neighbors,
            rbf_kernel=args.rbf_kernel,
        )
    except Exception as exc:
        print(f"[skip] {tile_id}:{rubin_band} fit failed ({exc})")
        return None

    if built is None:
        print(f"[skip] {tile_id}:{rubin_band} insufficient usable matches")
        return None

    raw_off_mas = np.hypot(built["matched"]["offsets"][:, 0], built["matched"]["offsets"][:, 1]) * 1000.0
    fit_resid_mas = built["fit"]["point_resid_mas"]
    nmatch = int(built["matched"]["vis_xy"].shape[0])
    print(
        f"[fit] {tile_id}:{rubin_band} matches={nmatch} "
        f"raw_med={np.median(raw_off_mas):.1f}mas raw_p68={np.percentile(raw_off_mas, 68):.1f}mas "
        f"fit_resid_med={np.median(fit_resid_mas):.1f}mas fit_resid_p68={np.percentile(fit_resid_mas, 68):.1f}mas"
    )
    return {
        "tile_id": tile_id,
        "rubin_band": normalize_band_name(rubin_band),
        "matched": built["matched"],
        "fit": built["fit"],
        "vis_wcs_header": vhdr,
        "vis_shape": built["vis_shape"],
    }


def export_results(results: List[Dict], output_path: str, dstep: int):
    method = "affine+RBF"
    hdus = [fits.PrimaryHDU()]
    hdus[0].header["CONCRDNC"] = (True, "JAISP astrometry concordance product")
    hdus[0].header["DSTEP"] = (int(dstep), "Mesh sampling step in VIS pixels")
    hdus[0].header["DUNIT"] = ("arcsec", "Offset unit")
    hdus[0].header["REFFRAME"] = ("euclid_VIS", "Reference astrometric frame")
    hdus[0].header["INTERP"] = ("rbf", "Recommended interpolation")
    hdus[0].header["FITMETH"] = (method, "Field fit method")

    for item in results:
        tile_id = item["tile_id"]
        rubin_band = item["rubin_band"]
        band_key = rubin_band.split("_", 1)[1]
        ext_prefix = f"{tile_id}.{band_key}" if tile_id else band_key
        nmatch = int(item["matched"]["vis_xy"].shape[0])
        hdus.append(
            make_concordance_hdu(
                item["fit"]["dra"],
                f"{ext_prefix}.DRA",
                dstep,
                rubin_band,
                tile_id,
                item["vis_wcs_header"],
                method,
                nmatch,
            )
        )
        hdus.append(
            make_concordance_hdu(
                item["fit"]["ddec"],
                f"{ext_prefix}.DDE",
                dstep,
                rubin_band,
                tile_id,
                item["vis_wcs_header"],
                method,
                nmatch,
            )
        )

    fits.HDUList(hdus).writeto(output_path, overwrite=True)


def write_summary(results: List[Dict], path: str):
    rows = []
    for item in results:
        raw_off_mas = np.hypot(item["matched"]["offsets"][:, 0], item["matched"]["offsets"][:, 1]) * 1000.0
        fit_resid_mas = item["fit"]["point_resid_mas"]
        rows.append(
            {
                "tile_id": item["tile_id"],
                "rubin_band": item["rubin_band"],
                "matches": int(item["matched"]["vis_xy"].shape[0]),
                "raw_median_mas": float(np.median(raw_off_mas)),
                "raw_p68_mas": float(np.percentile(raw_off_mas, 68)),
                "fit_resid_median_mas": float(np.median(fit_resid_mas)),
                "fit_resid_p68_mas": float(np.percentile(fit_resid_mas, 68)),
            }
        )
    with open(path, "w") as handle:
        json.dump(rows, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit direct smooth concordance fields from matched sources.")
    parser.add_argument("--rubin-dir", type=str, required=True)
    parser.add_argument("--euclid-dir", type=str, required=True)
    parser.add_argument("--rubin-band", type=str, default="r")
    parser.add_argument(
        "--rubin-bands",
        type=str,
        nargs="+",
        default=[],
        help="Optional Rubin bands to fit in one run (u g r i z y or rubin_*). Use 'all' for all six.",
    )
    parser.add_argument(
        "--detect-bands",
        type=str,
        nargs="+",
        default=["g", "r", "i", "z"],
        help="Rubin bands used to build the shared Rubin detection image. Use 'all' for all six.",
    )
    parser.add_argument("--tile-id", type=str, default="")
    parser.add_argument("--output", type=str, default="concordance_matched_rbf.fits")
    parser.add_argument("--summary-json", type=str, default="")

    parser.add_argument("--dstep", type=int, default=8)
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--max-matches", type=int, default=256)
    parser.add_argument("--max-sep-arcsec", type=float, default=0.12)
    parser.add_argument("--clip-sigma", type=float, default=3.5)

    parser.add_argument("--rubin-nsig", type=float, default=4.5)
    parser.add_argument("--vis-nsig", type=float, default=4.0)
    parser.add_argument("--rubin-smooth", type=float, default=1.0)
    parser.add_argument("--vis-smooth", type=float, default=1.2)
    parser.add_argument("--rubin-min-dist", type=int, default=7)
    parser.add_argument("--vis-min-dist", type=int, default=9)
    parser.add_argument("--max-sources-rubin", type=int, default=600)
    parser.add_argument("--max-sources-vis", type=int, default=800)
    parser.add_argument("--detect-clip-sigma", type=float, default=8.0)
    parser.add_argument(
        "--refine-band-centroids",
        action="store_true",
        default=True,
        help="After shared multiband detection, re-center Rubin source positions in the target band.",
    )
    parser.add_argument(
        "--no-refine-band-centroids",
        action="store_false",
        dest="refine_band_centroids",
        help="Disable per-band centroid refinement.",
    )
    parser.add_argument("--refine-radius", type=int, default=3)
    parser.add_argument("--refine-flux-floor-sigma", type=float, default=1.5)

    parser.add_argument(
        "--rbf-kernel",
        type=str,
        default="thin_plate_spline",
        choices=["thin_plate_spline", "cubic", "linear", "quintic"],
    )
    parser.add_argument("--rbf-smoothing", type=float, default=5e-4)
    parser.add_argument("--rbf-neighbors", type=int, default=32)
    return parser


def main():
    args = build_parser().parse_args()

    if not HAS_ASTROPY:
        raise RuntimeError("astropy is required for FITS export.")

    rubin_bands = normalize_band_list(args.rubin_bands)
    if not rubin_bands:
        rubin_bands = [normalize_band_name(args.rubin_band)]
    detect_bands = normalize_band_list(args.detect_bands)
    if not detect_bands:
        detect_bands = [f"rubin_{b}" for b in ("g", "r", "i", "z")]
    args.detect_bands = detect_bands

    import glob
    rubin_files = sorted(glob.glob(os.path.join(args.rubin_dir, "tile_x*_y*.npz")))
    results = []
    n_seen = 0
    for rubin_path in rubin_files:
        tile_id = os.path.splitext(os.path.basename(rubin_path))[0]
        if args.tile_id and tile_id != args.tile_id:
            continue
        euclid_path = os.path.join(args.euclid_dir, f"{tile_id}_euclid.npz")
        if not os.path.exists(euclid_path):
            continue
        n_seen += 1
        for rubin_band in rubin_bands:
            item = fit_tile(rubin_path, euclid_path, tile_id, rubin_band, args)
            if item is not None:
                results.append(item)

    if not results:
        raise RuntimeError("No tiles were successfully fit.")

    export_results(results, args.output, args.dstep)
    print(f"\nWrote {args.output}: {len(results)} fitted tile-band fields from {n_seen} candidate tiles")

    if args.summary_json:
        write_summary(results, args.summary_json)
        print(f"Wrote summary: {args.summary_json}")


if __name__ == "__main__":
    main()
