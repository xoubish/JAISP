#!/usr/bin/env python
"""
Diagnose PSFField v3 VIS width against empirical bright-star stacks.

Pipeline
--------
1. Pick a handful of Euclid native-resolution tiles.
2. On each VIS image, detect bright isolated sources (no neighbour within
   ``--isolation`` arcsec, flux above ``--nsig`` sigma).
3. Reject extended sources by a rough second-moment FWHM cut.
4. Subpixel-recenter each stamp by centroid-of-light, subtract an annulus
   background, and normalise by flux.
5. Stack the surviving star stamps → empirical VIS PSF.
6. Render PSFField v3 stamps at the same (tile-position) coordinates and stack
   them the same way.
7. Print per-source and stacked FWHM_x / FWHM_y from second moments.
8. Write a PNG with empirical stack, PSFField stack, residual, and per-source
   FWHM scatter.

If the empirical FWHM is consistently narrower than the PSFField FWHM the
photometry head cannot clean up the residuals, and PSFField v3 needs retraining
or replacement.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
_ROOT = _MODELS.parent
for _p in (_ROOT, _MODELS, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from astropy.wcs import WCS
from scipy.ndimage import shift as nd_shift

from astrometry2.dataset import safe_header_from_card_string
from astrometry2.source_matching import detect_sources
from models.photometry import PSFFieldPhotometryPipeline


VIS_PX_SCALE_ARCSEC = 0.1


def _to_f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def load_vis(euclid_path: str) -> Tuple[np.ndarray, WCS]:
    data = np.load(euclid_path, allow_pickle=True)
    img = np.nan_to_num(_to_f32(data["img_VIS"]), nan=0.0)
    wcs = WCS(safe_header_from_card_string(data["wcs_VIS"].item()))
    return img, wcs


def _annulus_bg(stamp: np.ndarray, r_in: float, r_out: float) -> float:
    h, w = stamp.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.indices(stamp.shape)
    r = np.hypot(xx - cx, yy - cy)
    mask = (r >= r_in) & (r <= r_out) & np.isfinite(stamp)
    if mask.sum() < 8:
        return 0.0
    return float(np.median(stamp[mask]))


def _moments(
    stamp: np.ndarray,
    mask_radius: float | None = None,
) -> Tuple[float, float, float, float]:
    """Return (x0, y0, sigma_x, sigma_y) from flux-weighted moments.

    When ``mask_radius`` is set, only pixels within that radius of the stamp
    centre contribute. Essential for single-star FWHM on noisy stamps —
    otherwise far-field positive-noise pixels blow up the 2nd moments.
    """
    h, w = stamp.shape
    yy, xx = np.indices(stamp.shape)
    if mask_radius is not None:
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        mask = np.hypot(xx - cx, yy - cy) <= float(mask_radius)
        s_arr = np.where(mask, stamp, 0.0)
    else:
        s_arr = stamp
    s_arr = np.clip(s_arr, 0.0, None)
    s = float(s_arr.sum())
    if s <= 0 or not np.isfinite(s):
        return np.nan, np.nan, np.nan, np.nan
    x0 = float((xx * s_arr).sum() / s)
    y0 = float((yy * s_arr).sum() / s)
    mxx = float(((xx - x0) ** 2 * s_arr).sum() / s)
    myy = float(((yy - y0) ** 2 * s_arr).sum() / s)
    return x0, y0, np.sqrt(max(mxx, 0.0)), np.sqrt(max(myy, 0.0))


def extract_stamp(
    img: np.ndarray,
    x: float,
    y: float,
    stamp_size: int,
    bg_inner: float,
    bg_outer: float,
) -> np.ndarray | None:
    h, w = img.shape
    r = stamp_size // 2
    xi, yi = int(round(x)), int(round(y))
    if xi - r < 0 or xi + r >= w or yi - r < 0 or yi + r >= h:
        return None
    stamp = img[yi - r : yi + r + 1, xi - r : xi + r + 1].astype(np.float32, copy=True)
    bg = _annulus_bg(stamp, bg_inner, bg_outer)
    stamp = stamp - bg

    # Subpixel re-center via centroid-of-light on a small central window
    # so far-field noise doesn't bias the centroid.
    x0, y0, _, _ = _moments(stamp, mask_radius=4.0)
    if not (np.isfinite(x0) and np.isfinite(y0)):
        return None
    cx, cy = (stamp_size - 1) / 2.0, (stamp_size - 1) / 2.0
    stamp = nd_shift(stamp, shift=(cy - y0, cx - x0), order=3, mode="nearest")
    total = float(stamp.sum())
    if total <= 0 or not np.isfinite(total):
        return None
    return (stamp / total).astype(np.float32)


def find_bright_isolated_stars(
    vis: np.ndarray,
    nsig: float,
    isolation_px: float,
    margin: int,
    max_per_tile: int,
    fwhm_max_px: float,
    fwhm_min_px: float,
    tile_name: str = "",
) -> np.ndarray:
    stage_counts: List[Tuple[str, int]] = []
    x, y = detect_sources(
        vis,
        nsig=float(nsig),
        smooth_sigma=1.0,
        min_dist=5,
        max_sources=4096,
    )
    stage_counts.append(("detect", int(len(x))))
    if len(x) == 0:
        print(f"  {tile_name}: detect=0")
        return np.zeros((0, 2), dtype=np.float32)
    xy = np.stack([x, y], axis=1).astype(np.float32)

    h, w = vis.shape
    keep_edge = (
        (xy[:, 0] >= margin)
        & (xy[:, 0] < w - margin)
        & (xy[:, 1] >= margin)
        & (xy[:, 1] < h - margin)
    )
    xy = xy[keep_edge]
    stage_counts.append(("edge", int(xy.shape[0])))
    if xy.shape[0] == 0:
        print(f"  {tile_name}: " + " ".join(f"{k}={v}" for k, v in stage_counts))
        return xy

    if xy.shape[0] > 1:
        dist = np.hypot(
            xy[:, 0:1] - xy[None, :, 0], xy[:, 1:2] - xy[None, :, 1]
        )
        np.fill_diagonal(dist, np.inf)
        keep_iso = dist.min(axis=1) >= isolation_px
        xy = xy[keep_iso]
    stage_counts.append(("isolated", int(xy.shape[0])))
    if xy.shape[0] == 0:
        print(f"  {tile_name}: " + " ".join(f"{k}={v}" for k, v in stage_counts))
        return xy

    keep_star: List[int] = []
    fwhm_seen: List[float] = []
    for i, (xx, yy) in enumerate(xy):
        r = 15
        xi, yi = int(round(xx)), int(round(yy))
        if xi - r < 0 or xi + r >= w or yi - r < 0 or yi + r >= h:
            continue
        s = vis[yi - r : yi + r + 1, xi - r : xi + r + 1].astype(np.float32)
        s = s - _annulus_bg(s, 9, 13)
        _, _, sx, sy = _moments(s, mask_radius=5.0)
        if not (np.isfinite(sx) and np.isfinite(sy)):
            continue
        fwhm = 2.355 * 0.5 * (sx + sy)
        fwhm_seen.append(float(fwhm))
        if fwhm_min_px <= fwhm <= fwhm_max_px:
            keep_star.append(i)
    xy = xy[keep_star]
    stage_counts.append(("compact", int(xy.shape[0])))

    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    order = np.argsort(vis[yi, xi])[::-1][:max_per_tile]
    xy = xy[order].astype(np.float32)
    stage_counts.append(("kept", int(xy.shape[0])))

    fwhm_summary = ""
    if fwhm_seen:
        fa = np.asarray(fwhm_seen)
        fwhm_summary = (
            f" | fwhm seen (px): min={fa.min():.2f} med={np.median(fa):.2f} max={fa.max():.2f}"
        )
    print(f"  {tile_name}: " + " ".join(f"{k}={v}" for k, v in stage_counts) + fwhm_summary)
    return xy


def render_psf_field_at(
    pipe: PSFFieldPhotometryPipeline,
    positions_px: np.ndarray,
    tile_hw: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Render VIS-only PSF stamps at the given VIS-pixel positions."""
    pos = torch.from_numpy(positions_px.astype(np.float32)).to(device)
    psfs = pipe.render_psfs(pos, tile_hw=tile_hw)
    # psfs: [N, B, S, S]; we only care about VIS which is index 0 in our band list
    return psfs[:, 0]


def fwhm_from_stack(stack: np.ndarray, mask_radius: float = 8.0) -> Tuple[float, float]:
    _, _, sx, sy = _moments(stack, mask_radius=mask_radius)
    return 2.355 * sx, 2.355 * sy


def make_diagnostic_figure(
    emp: np.ndarray,
    psf: np.ndarray,
    per_star_fwhm_emp: np.ndarray,
    per_star_fwhm_psf: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    resid = emp - psf
    vmax = float(np.nanpercentile(emp, 99.5))
    rmax = float(np.nanpercentile(np.abs(resid), 99))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    axes[0].imshow(emp, cmap="magma", vmin=0, vmax=vmax)
    axes[0].set_title("empirical stack")
    axes[1].imshow(psf, cmap="magma", vmin=0, vmax=vmax)
    axes[1].set_title("PSFField stack")
    axes[2].imshow(resid, cmap="coolwarm", vmin=-rmax, vmax=rmax)
    axes[2].set_title("emp - psf")
    for a in axes[:3]:
        a.set_xticks([])
        a.set_yticks([])

    axes[3].scatter(
        per_star_fwhm_emp,
        per_star_fwhm_psf,
        s=12,
        alpha=0.7,
    )
    lo = float(
        np.nanmin([per_star_fwhm_emp.min(), per_star_fwhm_psf.min()])
    )
    hi = float(
        np.nanmax([per_star_fwhm_emp.max(), per_star_fwhm_psf.max()])
    )
    axes[3].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[3].set_xlabel("empirical FWHM [px]")
    axes[3].set_ylabel("PSFField FWHM [px]")
    axes[3].set_title("per-star FWHM")
    axes[3].set_aspect("equal")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--euclid-dir", default="data/euclid_tiles_all")
    ap.add_argument("--tiles-glob", default="tile_*.npz")
    ap.add_argument("--psf-checkpoint", default="models/checkpoints/psf_field_v3.pt")
    ap.add_argument("--out-dir", default="models/checkpoints/psf_field_v3/vis_psf_diagnostic")
    ap.add_argument("--stamp-size", type=int, default=31)
    ap.add_argument("--nsig", type=float, default=10.0)
    ap.add_argument("--isolation-arcsec", type=float, default=3.0)
    ap.add_argument("--margin", type=int, default=20)
    ap.add_argument("--fwhm-min-px", type=float, default=1.0)
    ap.add_argument("--fwhm-max-px", type=float, default=3.0)
    ap.add_argument("--max-per-tile", type=int, default=20)
    ap.add_argument("--max-tiles", type=int, default=10)
    ap.add_argument("--bg-inner", type=float, default=11.0)
    ap.add_argument("--bg-outer", type=float, default=15.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = PSFFieldPhotometryPipeline.from_checkpoint(
        args.psf_checkpoint,
        band_names=["euclid_VIS", "euclid_Y", "euclid_J", "euclid_H"],
        stamp_size=args.stamp_size,
        sub_grid=2,
        device=device,
    )

    paths = sorted(glob.glob(str(Path(args.euclid_dir) / args.tiles_glob)))
    if args.max_tiles > 0:
        paths = paths[: args.max_tiles]
    if not paths:
        print(f"No tiles matched: {Path(args.euclid_dir) / args.tiles_glob}")
        return
    print(f"Diagnosing {len(paths)} tiles ...")

    isolation_px = float(args.isolation_arcsec) / VIS_PX_SCALE_ARCSEC

    emp_stamps: List[np.ndarray] = []
    psf_stamps: List[np.ndarray] = []
    per_star_emp: List[Tuple[float, float]] = []
    per_star_psf: List[Tuple[float, float]] = []

    for path in paths:
        vis, _wcs = load_vis(path)
        stars = find_bright_isolated_stars(
            vis,
            nsig=args.nsig,
            isolation_px=isolation_px,
            margin=args.margin,
            max_per_tile=args.max_per_tile,
            fwhm_max_px=args.fwhm_max_px,
            fwhm_min_px=args.fwhm_min_px,
            tile_name=Path(path).name,
        )
        if stars.shape[0] == 0:
            continue

        psf_batch = render_psf_field_at(pipe, stars, tile_hw=vis.shape, device=device).cpu().numpy()

        kept = 0
        for i in range(stars.shape[0]):
            stamp = extract_stamp(
                vis,
                float(stars[i, 0]),
                float(stars[i, 1]),
                stamp_size=args.stamp_size,
                bg_inner=args.bg_inner,
                bg_outer=args.bg_outer,
            )
            if stamp is None:
                continue
            psf_stamp = psf_batch[i]
            psf_sum = float(psf_stamp.sum())
            if psf_sum <= 0:
                continue
            psf_stamp = psf_stamp / psf_sum

            emp_stamps.append(stamp)
            psf_stamps.append(psf_stamp)

            fwhm_e = fwhm_from_stack(stamp)
            fwhm_p = fwhm_from_stack(psf_stamp)
            per_star_emp.append(fwhm_e)
            per_star_psf.append(fwhm_p)
            kept += 1
        print(f"{Path(path).name}: kept {kept}/{stars.shape[0]} stars")

    if not emp_stamps:
        print("No valid star stamps — relax --nsig or --fwhm-max-px.")
        return

    emp = np.mean(np.stack(emp_stamps), axis=0)
    psf = np.mean(np.stack(psf_stamps), axis=0)

    per_star_emp_arr = np.array(per_star_emp)
    per_star_psf_arr = np.array(per_star_psf)

    fwhm_emp_stack_x, fwhm_emp_stack_y = fwhm_from_stack(emp)
    fwhm_psf_stack_x, fwhm_psf_stack_y = fwhm_from_stack(psf)

    fwhm_emp_mean = per_star_emp_arr.mean(axis=0)
    fwhm_psf_mean = per_star_psf_arr.mean(axis=0)

    summary = {
        "n_stars": int(len(emp_stamps)),
        "n_tiles": int(len(paths)),
        "stamp_size": int(args.stamp_size),
        "vis_px_scale_arcsec": VIS_PX_SCALE_ARCSEC,
        "fwhm_empirical_stack_px": [fwhm_emp_stack_x, fwhm_emp_stack_y],
        "fwhm_psf_field_stack_px": [fwhm_psf_stack_x, fwhm_psf_stack_y],
        "fwhm_empirical_stack_arcsec": [
            fwhm_emp_stack_x * VIS_PX_SCALE_ARCSEC,
            fwhm_emp_stack_y * VIS_PX_SCALE_ARCSEC,
        ],
        "fwhm_psf_field_stack_arcsec": [
            fwhm_psf_stack_x * VIS_PX_SCALE_ARCSEC,
            fwhm_psf_stack_y * VIS_PX_SCALE_ARCSEC,
        ],
        "per_star_mean_fwhm_emp_px": fwhm_emp_mean.tolist(),
        "per_star_mean_fwhm_psf_px": fwhm_psf_mean.tolist(),
        "psf_over_emp_ratio_stack": [
            fwhm_psf_stack_x / max(fwhm_emp_stack_x, 1e-6),
            fwhm_psf_stack_y / max(fwhm_emp_stack_y, 1e-6),
        ],
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Persist the empirical stack so the photometry trainer can use it in
    # place of PSFField for the overridden bands (sanity test).
    empirical_out = out_dir / "empirical_psf.pt"
    torch.save(
        {
            "band_stacks": {
                "euclid_VIS": torch.from_numpy(emp.astype(np.float32)),
            },
            "stamp_size": int(args.stamp_size),
            "vis_px_scale_arcsec": VIS_PX_SCALE_ARCSEC,
            "n_stars": int(len(emp_stamps)),
            "fwhm_stack_px": [fwhm_emp_stack_x, fwhm_emp_stack_y],
            "source_psf_checkpoint": str(args.psf_checkpoint),
        },
        empirical_out,
    )

    make_diagnostic_figure(
        emp,
        psf,
        per_star_emp_arr.mean(axis=1),
        per_star_psf_arr.mean(axis=1),
        out_path=out_dir / "vis_psf_comparison.png",
        title=(
            f"VIS PSF: empirical vs PSFField  "
            f"n={len(emp_stamps)}  "
            f"emp FWHM {fwhm_emp_stack_x:.2f}/{fwhm_emp_stack_y:.2f} px  "
            f"psf FWHM {fwhm_psf_stack_x:.2f}/{fwhm_psf_stack_y:.2f} px"
        ),
    )

    print("\nSummary:")
    print(f"  stars stacked          : {len(emp_stamps)}")
    print(
        f"  empirical FWHM (x, y)  : "
        f"{fwhm_emp_stack_x:.3f}, {fwhm_emp_stack_y:.3f} px  "
        f"({fwhm_emp_stack_x*VIS_PX_SCALE_ARCSEC:.3f}, "
        f"{fwhm_emp_stack_y*VIS_PX_SCALE_ARCSEC:.3f} arcsec)"
    )
    print(
        f"  PSFField  FWHM (x, y)  : "
        f"{fwhm_psf_stack_x:.3f}, {fwhm_psf_stack_y:.3f} px  "
        f"({fwhm_psf_stack_x*VIS_PX_SCALE_ARCSEC:.3f}, "
        f"{fwhm_psf_stack_y*VIS_PX_SCALE_ARCSEC:.3f} arcsec)"
    )
    print(
        f"  PSFField/empirical     : "
        f"{summary['psf_over_emp_ratio_stack'][0]:.3f} (x)  "
        f"{summary['psf_over_emp_ratio_stack'][1]:.3f} (y)"
    )
    print(f"\nFigure: {out_dir / 'vis_psf_comparison.png'}")
    print(f"JSON:   {out_dir / 'summary.json'}")
    print(f"PSF:    {empirical_out}")


if __name__ == "__main__":
    main()
