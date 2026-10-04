"""Command-line interface: ``euclid-phot run``.

    euclid-phot run --ra 269.48 --dec 67.30 --size 50 --out catalog.ecsv

Measures a single cutout with the default MER-prior workflow.
"""
from __future__ import annotations

import argparse
import sys

from .config import DEFAULT_DATA_DIR


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                   help="cache directory (default: %(default)s)")
    p.add_argument("--prior-band", default="VIS",
                   choices=("VIS", "Y", "J", "H"))
    p.add_argument("--objects", default="mer", choices=("mer", "free"),
                   help="source-list mode (user coords are API-only)")
    p.add_argument("--bands", default="Y,J,H",
                   help="comma-separated Euclid target bands "
                        "(default: %(default)s; '' for prior band only)")
    p.add_argument("--wise", default="",
                   help="comma-separated WISE bands (W1,W2); default none")
    p.add_argument("--no-calibrate-errors", action="store_true",
                   help="skip the empirical error calibration")
    p.add_argument("--psf-product", default="auto",
                   choices=("auto", "catalog", "grid"))
    p.add_argument("--workers", type=int, default=1,
                   help="threads inside one cutout fit (default 1)")


def _parse_bands(arg: str) -> tuple:
    return tuple(b.strip() for b in arg.split(",") if b.strip())


def _build_kwargs(args) -> dict:
    return dict(
        prior={"band": args.prior_band, "objects": args.objects},
        target_bands={"euclid": _parse_bands(args.bands),
                      "wise": _parse_bands(args.wise)},
        data_dir=args.data_dir,
        calibrate_errors=not args.no_calibrate_errors,
        psf_product=args.psf_product,
        n_workers=args.workers,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="euclid-phot",
        description="Forced photometry on Euclid Q1 (+unWISE) with Tractor.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="measure a single cutout")
    p_run.add_argument("--ra", type=float, required=True)
    p_run.add_argument("--dec", type=float, required=True)
    p_run.add_argument("--size", type=float, default=50.0,
                       help="cutout side, arcsec (default %(default)s)")
    p_run.add_argument("--out", default="catalog.ecsv",
                       help="output table path; format from the extension "
                            "(.ecsv/.fits/.parquet)")
    _add_common(p_run)

    args = parser.parse_args(argv)

    if args.command == "run":
        from .pipeline import run_forced_photometry
        result = run_forced_photometry(args.ra, args.dec, args.size,
                                       **_build_kwargs(args))
        tab = result.to_table()
        if args.out.endswith(".fits"):
            # FITS headers cannot carry the nested metadata; sanitize.
            tab = tab.copy()
            tab.meta = {}
        tab.write(args.out, overwrite=True)
        print(f"wrote {len(tab)} sources to {args.out}")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
