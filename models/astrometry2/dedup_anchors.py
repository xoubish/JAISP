"""Deduplicate per-band anchor caches from overlap-region tile duplicates.

CenterNet anchors are written per-tile by ``eval_latent_position.py
--save-anchors``; the same physical source detected in two overlapping tiles
appears once per tile in the saved npz. For density / number-count work we
need a deduplicated catalogue. For PINN/HGP residual fits, dedup is also the
preferred QA default when overlap duplicates would otherwise overweight the
same sky source; `fit_hierarchical_gp_concordance.py --dedup-radius-arcsec`
can now apply the same operation in memory at solve time.

Per-band greedy clustering:
  1. Sort entries by SNR descending.
  2. Walk through; for each not-yet-killed entry, kill every neighbour within
     ``--radius-arcsec`` and keep the highest-SNR rep.

Default ``--radius-arcsec 0.05`` (50 mas) is empirically calibrated:
overlap-region duplicates land at ~0 mas separation (adjacent tiles share
patch-boundary WCS), real same-tile neighbours within the catalogue are
>100 mas apart, so 50 mas comfortably separates the two regimes.

Usage::

    python models/astrometry2/dedup_anchors.py \\
        --anchors models/checkpoints/latent_position_v8_no_psf/anchors_centernet.npz \\
        --output  models/checkpoints/latent_position_v8_no_psf/anchors_centernet_dedup.npz \\
        --radius-arcsec 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree


PER_BAND_FIELDS = ("ra", "dec", "raw", "head_resid", "snr", "tiles")


def discover_bands(npz: Dict[str, np.ndarray]) -> List[str]:
    """Bands present in the npz (any key matching ``{band}_ra``)."""
    bands = []
    for key in npz.keys():
        if not key.endswith("_ra"):
            continue
        band = key[: -len("_ra")]
        if all(f"{band}_{field}" in npz for field in PER_BAND_FIELDS):
            bands.append(band)
    return bands


def dedup_one_band(
    ra: np.ndarray,
    dec: np.ndarray,
    snr: np.ndarray,
    radius_arcsec: float,
) -> np.ndarray:
    """Return boolean keep mask after greedy SNR-ordered clustering."""
    n = ra.size
    if n == 0:
        return np.zeros(0, dtype=bool)

    # Project to local tangent plane (arcsec) for cKDTree.
    ra0 = float(np.median(ra))
    dec0 = float(np.median(dec))
    cosd = float(np.cos(np.deg2rad(dec0)))
    x = (ra - ra0) * cosd * 3600.0
    y = (dec - dec0) * 3600.0
    pts = np.stack([x, y], axis=1).astype(np.float64)

    tree = cKDTree(pts)
    keep = np.ones(n, dtype=bool)

    # Visit in descending SNR so the kept rep of each cluster is the brightest.
    finite_snr = np.where(np.isfinite(snr), snr, -np.inf)
    order = np.argsort(finite_snr)[::-1]
    for idx in order:
        if not keep[idx]:
            continue
        neighbours = tree.query_ball_point(pts[idx], r=float(radius_arcsec))
        for j in neighbours:
            if j == idx:
                continue
            keep[j] = False
    return keep


def dedup_anchors(
    in_path: Path,
    out_path: Path,
    radius_arcsec: float,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Run dedup on every band in ``in_path`` and write a new npz to ``out_path``."""
    npz = dict(np.load(in_path, allow_pickle=False))
    bands = discover_bands(npz)
    if not bands:
        raise ValueError(f"No per-band fields found in {in_path}")

    out: Dict[str, np.ndarray] = {}
    before: Dict[str, int] = {}
    after: Dict[str, int] = {}
    for band in bands:
        ra = np.asarray(npz[f"{band}_ra"])
        dec = np.asarray(npz[f"{band}_dec"])
        snr = np.asarray(npz[f"{band}_snr"]).astype(np.float64)
        keep = dedup_one_band(ra, dec, snr, radius_arcsec)
        before[band] = int(ra.size)
        after[band] = int(keep.sum())
        for field in PER_BAND_FIELDS:
            key = f"{band}_{field}"
            arr = np.asarray(npz[key])
            out[key] = arr[keep]

    # Carry through any extra non-band keys (e.g. metadata) unchanged.
    for key, value in npz.items():
        if key not in out:
            out[key] = value

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    return before, after


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anchors", required=True, type=Path,
                   help="Input anchors .npz (from eval_latent_position.py --save-anchors).")
    p.add_argument("--output", required=True, type=Path,
                   help="Output deduplicated anchors .npz.")
    p.add_argument("--radius-arcsec", type=float, default=0.05,
                   help="Cluster radius in arcsec; entries within this radius "
                        "are merged, highest-SNR kept. Default 0.05.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    before, after = dedup_anchors(args.anchors, args.output, args.radius_arcsec)

    print(f"Deduplicated anchors written to {args.output}")
    print(f"Cluster radius: {args.radius_arcsec * 1000:.0f} mas\n")
    print(f"{'band':10s} {'before':>10s} {'after':>10s} {'kept':>8s}")
    total_before = 0
    total_after = 0
    for band in sorted(before.keys()):
        b = before[band]
        a = after[band]
        total_before += b
        total_after += a
        pct = 100.0 * a / max(b, 1)
        print(f"{band:10s} {b:>10d} {a:>10d} {pct:>7.1f}%")
    pct = 100.0 * total_after / max(total_before, 1)
    print(f"{'TOTAL':10s} {total_before:>10d} {total_after:>10d} {pct:>7.1f}%")


if __name__ == "__main__":
    main()
