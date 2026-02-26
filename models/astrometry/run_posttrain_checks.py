"""
Run a compact post-train astrometry sanity check.

This script avoids long command chains by doing three steps in one call:
1) export a single-tile concordance FITS from a trained head checkpoint
2) run catalog astrometry evaluation with apply-sign = +1
3) run catalog astrometry evaluation with apply-sign = -1

It writes a summary JSON that compares both signs and reports which looks better.

Example:
    python3 models/astrometry/run_posttrain_checks.py \
        --head-ckpt models/checkpoints/jaisp_astrometry_50ep/best_astrometry.pt
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import torch

from astropy.io import fits

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_fits import HAS_ASTROPY, export_tile, load_backbone, load_head  # noqa: E402


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def _auto_head_ckpt() -> Path:
    cands = sorted(
        REPO_ROOT.glob("models/checkpoints/jaisp_astrometry*/best_astrometry.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(
            "Could not auto-find a head checkpoint under "
            "models/checkpoints/jaisp_astrometry*/best_astrometry.pt"
        )
    return cands[0]


def _make_export_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        backbone_ckpt=str(args.backbone_ckpt),
        head_ckpt=str(args.head_ckpt),
        embed_dim=args.embed_dim,
        proj_dim=args.proj_dim,
        depth=args.depth,
        patch_size=args.patch_size,
        search_radius=args.search_radius,
        softmax_temp=args.softmax_temp,
        refine_hidden=args.refine_hidden,
        refine_depth=args.refine_depth,
        use_stem_refine=args.use_stem_refine,
        stem_channels=args.stem_channels,
        stem_hidden=args.stem_hidden,
        stem_depth=args.stem_depth,
        stem_stride=args.stem_stride,
    )


def _export_single_tile_fits(args: argparse.Namespace, fits_out: Path) -> Tuple[Path, Path]:
    if not HAS_ASTROPY:
        raise RuntimeError("astropy is required for FITS export.")

    rubin_tile = args.rubin_dir / f"{args.tile_id}.npz"
    euclid_tile = args.euclid_dir / f"{args.tile_id}_euclid.npz"
    if not rubin_tile.exists():
        raise FileNotFoundError(f"Rubin tile not found: {rubin_tile}")
    if not euclid_tile.exists():
        raise FileNotFoundError(f"Euclid tile not found: {euclid_tile}")

    dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    exp_args = _make_export_args(args)

    backbone = load_backbone(dev, exp_args)
    head = load_head(dev, exp_args)

    hdus = export_tile(
        backbone=backbone,
        head=head,
        rubin_path=str(rubin_tile),
        euclid_path=str(euclid_tile),
        rubin_bands=[f"rubin_{args.rubin_band.lower()}"],
        device=dev,
        dstep=args.dstep,
        tile_id=args.tile_id,
    )
    if not hdus:
        raise RuntimeError(
            f"export_tile produced no HDUs for tile={args.tile_id}, band={args.rubin_band}."
        )

    primary = fits.PrimaryHDU()
    primary.header["CONCRDNC"] = (True, "JAISP astrometry concordance product")
    primary.header["DSTEP"] = (args.dstep, "Mesh sampling step in VIS pixels")
    primary.header["DUNIT"] = ("arcsec", "Offset unit")
    primary.header["REFFRAME"] = ("euclid_VIS", "Reference astrometric frame")
    primary.header["INTERP"] = ("bilinear", "Recommended interpolation")

    fits.HDUList([primary] + hdus).writeto(fits_out, overwrite=True)
    return rubin_tile, euclid_tile


def _run_eval(
    args: argparse.Namespace,
    rubin_tile: Path,
    euclid_tile: Path,
    fits_path: Path,
    sign: int,
    out_json: Path,
) -> Dict:
    eval_py = SCRIPT_DIR / "evaluate_catalog_astrometry.py"
    cmd = [
        sys.executable,
        str(eval_py),
        "--auto-from-tiles",
        "--rubin-tile",
        str(rubin_tile),
        "--euclid-tile",
        str(euclid_tile),
        "--rubin-band",
        args.rubin_band,
        "--euclid-band",
        args.euclid_band,
        "--concordance-fits",
        str(fits_path),
        "--max-sep-arcsec",
        str(args.max_sep_arcsec),
        "--clip-sigma",
        str(args.clip_sigma),
        "--apply-sign",
        str(sign),
        "--output-json",
        str(out_json),
    ]
    _run(cmd)
    return json.loads(out_json.read_text())


def _extract_metrics(res: Dict) -> Dict[str, float]:
    before = res["before_all_candidates"]
    after = res["after_correction"]
    b_all = before["all_matches"]
    a_all = after["all_matches"]
    fixed = res.get("fixed_pairs", {})
    fb_all = fixed.get("before_all", {})
    fa_all = fixed.get("after_all", {})
    fb_clip = fixed.get("before_clipped", {})
    fa_clip = fixed.get("after_clipped", {})

    def _f(d: Dict, key: str) -> float:
        return float(d.get(key, float("nan")))

    return {
        "before_matches": int(before["n_matches"]),
        "after_matches": int(after["n_matches"]),
        "before_median_offset_mas": float(b_all["median_offset_mas"]),
        "after_median_offset_mas": float(a_all["median_offset_mas"]),
        "before_centered_p68_mas": float(b_all["centered_p68_offset_mas"]),
        "after_centered_p68_mas": float(a_all["centered_p68_offset_mas"]),
        "fixed_pairs": int(fixed.get("n_pairs", 0)),
        "fixed_before_median_offset_mas": _f(fb_all, "median_offset_mas"),
        "fixed_after_median_offset_mas": _f(fa_all, "median_offset_mas"),
        "fixed_before_centered_p68_mas": _f(fb_all, "centered_p68_offset_mas"),
        "fixed_after_centered_p68_mas": _f(fa_all, "centered_p68_offset_mas"),
        "fixed_before_clipped_median_offset_mas": _f(fb_clip, "median_offset_mas"),
        "fixed_after_clipped_median_offset_mas": _f(fa_clip, "median_offset_mas"),
        "fixed_before_clipped_centered_p68_mas": _f(fb_clip, "centered_p68_offset_mas"),
        "fixed_after_clipped_centered_p68_mas": _f(fa_clip, "centered_p68_offset_mas"),
        "corr_median_mas": float(
            res.get("correction_sampling", {})
            .get("sampled_correction_abs_arcsec", {})
            .get("median_arcsec", float("nan"))
            * 1000.0
        ),
    }


def _pick_better_sign(plus: Dict, minus: Dict) -> int:
    m_plus = _extract_metrics(plus)
    m_minus = _extract_metrics(minus)
    use_fixed = (
        np.isfinite(m_plus["fixed_after_clipped_centered_p68_mas"])
        and np.isfinite(m_minus["fixed_after_clipped_centered_p68_mas"])
    )
    if use_fixed:
        score_plus = (
            m_plus["fixed_after_clipped_centered_p68_mas"],
            m_plus["fixed_after_clipped_median_offset_mas"],
            -m_plus["fixed_pairs"],
            m_plus["after_centered_p68_mas"],
            m_plus["after_median_offset_mas"],
            -m_plus["after_matches"],
        )
        score_minus = (
            m_minus["fixed_after_clipped_centered_p68_mas"],
            m_minus["fixed_after_clipped_median_offset_mas"],
            -m_minus["fixed_pairs"],
            m_minus["after_centered_p68_mas"],
            m_minus["after_median_offset_mas"],
            -m_minus["after_matches"],
        )
    else:
        score_plus = (-m_plus["after_matches"], m_plus["after_centered_p68_mas"], m_plus["after_median_offset_mas"])
        score_minus = (-m_minus["after_matches"], m_minus["after_centered_p68_mas"], m_minus["after_median_offset_mas"])
    return +1 if score_plus <= score_minus else -1


def _print_line(tag: str, m: Dict[str, float]) -> None:
    fixed_txt = ""
    if np.isfinite(m["fixed_after_clipped_centered_p68_mas"]):
        fixed_txt = (
            f" | fixed(n={m['fixed_pairs']}) p68c "
            f"{m['fixed_before_clipped_centered_p68_mas']:.2f}->{m['fixed_after_clipped_centered_p68_mas']:.2f} mas"
        )
    print(
        f"{tag:7s} "
        f"matches {m['before_matches']} -> {m['after_matches']} | "
        f"median {m['before_median_offset_mas']:.2f} -> {m['after_median_offset_mas']:.2f} mas | "
        f"p68c {m['before_centered_p68_mas']:.2f} -> {m['after_centered_p68_mas']:.2f} mas | "
        f"|delta|~{m['corr_median_mas']:.1f} mas"
        f"{fixed_txt}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run export + eval(+1/-1) for one astrometry tile.")
    p.add_argument("--head-ckpt", type=Path, default=None, help="Head checkpoint; default auto-detect latest.")
    p.add_argument("--backbone-ckpt", type=Path, default=REPO_ROOT / "models/checkpoints/jaisp_v5/best.pt")

    p.add_argument("--rubin-dir", type=Path, default=REPO_ROOT / "data/rubin_tiles_ecdfs")
    p.add_argument("--euclid-dir", type=Path, default=REPO_ROOT / "data/euclid_tiles_ecdfs")
    p.add_argument("--tile-id", type=str, default="tile_x00000_y00000")
    p.add_argument("--rubin-band", type=str, default="r")
    p.add_argument("--euclid-band", type=str, default="VIS")

    p.add_argument("--dstep", type=int, default=8)
    p.add_argument("--max-sep-arcsec", type=float, default=0.1)
    p.add_argument("--clip-sigma", type=float, default=3.5)
    p.add_argument("--device", type=str, default="")

    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "models/checkpoints/astrometry_postcheck")
    p.add_argument("--label", type=str, default="", help="Optional label for output filenames.")

    # Backbone/head shape defaults (used when ckpt args are missing).
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--search-radius", type=int, default=3)
    p.add_argument("--softmax-temp", type=float, default=0.1)
    p.add_argument("--refine-hidden", type=int, default=32)
    p.add_argument("--refine-depth", type=int, default=4)
    p.add_argument("--use-stem-refine", action="store_true")
    p.add_argument("--stem-channels", type=int, default=64)
    p.add_argument("--stem-hidden", type=int, default=32)
    p.add_argument("--stem-depth", type=int, default=4)
    p.add_argument("--stem-stride", type=int, default=4)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.head_ckpt is None:
        args.head_ckpt = _auto_head_ckpt()
    args.head_ckpt = args.head_ckpt.resolve()
    args.backbone_ckpt = args.backbone_ckpt.resolve()
    args.rubin_dir = args.rubin_dir.resolve()
    args.euclid_dir = args.euclid_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.head_ckpt.exists():
        raise FileNotFoundError(f"Head checkpoint not found: {args.head_ckpt}")
    if not args.backbone_ckpt.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {args.backbone_ckpt}")

    label = args.label.strip()
    if not label:
        label = f"{args.head_ckpt.parent.name}_{args.tile_id}_{args.rubin_band}"
    label = label.replace("/", "_")

    fits_path = args.output_dir / f"concordance_{label}.fits"
    plus_json = args.output_dir / f"eval_{label}_signp1.json"
    minus_json = args.output_dir / f"eval_{label}_signm1.json"
    summary_json = args.output_dir / f"summary_{label}.json"

    print(f"Head ckpt : {args.head_ckpt}")
    print(f"Backbone  : {args.backbone_ckpt}")
    print(f"Tile/band : {args.tile_id} / {args.rubin_band}")

    rubin_tile, euclid_tile = _export_single_tile_fits(args, fits_path)
    print(f"\nWrote tile FITS: {fits_path}")

    res_plus = _run_eval(args, rubin_tile, euclid_tile, fits_path, +1, plus_json)
    res_minus = _run_eval(args, rubin_tile, euclid_tile, fits_path, -1, minus_json)
    m_plus = _extract_metrics(res_plus)
    m_minus = _extract_metrics(res_minus)
    best_sign = _pick_better_sign(res_plus, res_minus)

    print("\nSummary")
    _print_line("sign +1", m_plus)
    _print_line("sign -1", m_minus)
    print(f"best_sign (fixed-pair clipped p68 first, then fallback rematch): {best_sign:+d}")

    summary = {
        "paths": {
            "fits": str(fits_path),
            "plus_json": str(plus_json),
            "minus_json": str(minus_json),
            "head_ckpt": str(args.head_ckpt),
            "backbone_ckpt": str(args.backbone_ckpt),
        },
        "config": {
            "tile_id": args.tile_id,
            "rubin_band": args.rubin_band,
            "euclid_band": args.euclid_band,
            "max_sep_arcsec": args.max_sep_arcsec,
            "clip_sigma": args.clip_sigma,
            "dstep": args.dstep,
        },
        "metrics": {
            "sign_plus": m_plus,
            "sign_minus": m_minus,
            "best_sign": best_sign,
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
