"""Overlap-aware evaluation of a finished scene run on its held-out frame.

Splits trace pixels into 'overlap' (the second-brightest contributing source supplies >= 20% of
the model) and 'isolated' (one source supplies >= 95%).  In overlap pixels compares:
  full model  : all sources
  single      : the dominant source only (every overlapping pixel attributed to the brightest source)
  no model    : data - background
Usage: python experiments/grism/eval_overlap.py --tag _fnd [--foundation-stamps] [--star-psf] [--kernel ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.grism.scene import GrismScene, TileStamps, load_frame, catalog_from_loctab  # noqa: E402

BASE = REPO / "data" / "sir_edff"
CALIB = BASE / "calib" / "dr1_3025"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--det", default="23")
    ap.add_argument("--sky-targets", default="0,1080,1170", help="'ref,y0,y1' or 'common' (multi-observation strip)")
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--dithers", default="0,1,2,3", help="frame keys, optionally key:DET (as in scene_test.py)")
    ap.add_argument("--mer-stamps", action="store_true")
    ap.add_argument("--mer-cat", action="store_true")
    ap.add_argument("--mer-root", default="/tmp/shemmati_grism/products/mer")
    ap.add_argument("--kernel", default=str(REPO / "experiments/grism/nisps_kernel.npz"))
    ap.add_argument("--star-psf", action="store_true")
    ap.add_argument("--foundation-stamps", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dithers = [(int(d.split(":")[0]), d.split(":")[1] if ":" in d else args.det) for d in args.dithers.split(",")]
    frames = [load_frame(BASE, d, det, CALIB) for d, det in dithers]
    from models.grism.scene import sky_catalog, catalog_for_frame
    skycat = sky_catalog(BASE)
    if args.mer_cat:
        from models.grism.mer_mosaic import mer_sky_catalog
        skycat = skycat + mer_sky_catalog(args.mer_root, exclude=skycat)
    for fr in frames:
        fr["cat"] = catalog_for_frame(fr, skycat)
    if args.sky_targets == "common":
        from collections import Counter
        def partial(s, frac=0.5):
            if s["bb1"] is None:
                return False
            lo, hi = s["bb1"][0], s["bb1"][1]
            return (min(hi[0], 2040) - max(lo[0], 0)) >= frac * max(hi[0] - lo[0], 1) and lo[1] >= 0 and hi[1] <= 2040
        cnt = {}
        for fr in frames:
            cnt.setdefault(fr["obs"], Counter()).update({s["id"] for s in fr["cat"] if partial(s)})
        common = set.intersection(*[{i for i, n in c.items() if n >= 2} for c in cnt.values()])
        allcat = {}
        for fr in frames:
            for s in fr["cat"]:
                allcat.setdefault(s["id"], s)
        targets = {i: allcat[i] for i in common}
    else:
        rd, ty0, ty1 = [int(v) for v in args.sky_targets.split(",")]
        ref = [fr for fr in frames if fr["dither"] == rd][0]
        targets = {s["id"]: s for s in ref["cat"]
                   if s["bb1"] is not None and s["bb1"][0, 1] >= ty0 and s["bb1"][1, 1] <= ty1 and s["bb1"][0, 0] >= 0 and s["bb1"][1, 0] <= 2040}
    tiles = TileStamps(str(REPO / "data/euclid_tiles_all/tile_*_euclid.npz"))
    if args.mer_stamps:
        from models.grism.mer_mosaic import MerMosaicStamps, StampUnion
        tiles = StampUnion(tiles, MerMosaicStamps(args.mer_root))
    provider = None; star_psf = None
    if args.foundation_stamps:
        from models.grism.foundation_stamps import FoundationStamps
        provider = FoundationStamps(device=args.device)
    if args.star_psf:
        from models.grism.mer_psf import MerPsf, tile_for
        root = BASE / "products/mer_psf"
        tc = (np.mean([s["ra"] for s in targets.values()]), np.mean([s["dec"] for s in targets.values()]))
        star_psf = MerPsf(root, tile_for(root, *tc), "VIS", "GRID")
    scene = GrismScene(frames, args.det, (0, 0), (2040, 2040), tiles, kernel_npz=args.kernel, device=args.device,
                       lam_step=13.4, targets=targets, stamp_provider=provider, star_psf=star_psf)
    sp = torch.load(REPO / f"experiments/grism/scene_holdout_det{args.det}_d{args.holdout}{args.tag}_spectra.pt")
    scene.init_params(spec_step=1)
    kt = REPO / f"experiments/grism/nisps_kernel_fit{args.tag}.npz"
    if kt.exists():
        scene.kernels = scene.kernels0 = None
        from models.grism.renderer import kernel_bank_from_table
        scene.kernels = kernel_bank_from_table(scene.lam, kt, device=args.device); scene.kernels0 = kernel_bank_from_table(scene.lam0, kt, device=args.device)
        print("using fitted kernel", kt.name)
    for sid, s in scene.sources.items():
        if sid in sp:
            with torch.no_grad():
                s.logf1.copy_(sp[sid]["logf1"]); s.logf0.copy_(sp[sid]["logf0"]); s.shift.copy_(sp[sid]["shift"])
    fr = [f for f in frames if f["dither"] == args.holdout][0]
    d = fr["dat"].numpy(); w = fr["w"].numpy(); good = w > 0
    noise = np.sqrt(1 / np.maximum(w, 1e-12))
    # per-source renders (no background) -> top-1 and top-2 contributions per pixel
    top1 = np.zeros_like(d); top2 = np.zeros_like(d); total = np.zeros_like(d)
    with torch.no_grad():
        for sid in scene.sources:
            img = scene.render_frame(fr, with_bkg=False, only={sid}).numpy()
            total += img
            new_top1 = np.maximum(top1, img)
            top2 = np.maximum(top2, np.minimum(top1, img))   # second largest
            top1 = new_top1
    bkg = np.median((d - total)[good & (total < noise)]) if (good & (total < noise)).sum() > 100 else 0.0
    trace = (total > 3 * noise) & good
    overlap = trace & (top2 >= 0.2 * total)
    isolated = trace & (top1 >= 0.95 * total)
    chi_full = (d - total - bkg) * np.sqrt(w)
    chi_single = (d - top1 - bkg) * np.sqrt(w)      # all flux attributed to the dominant source
    chi_none = (d - bkg) * np.sqrt(w)
    def r(a, m):
        return np.median(np.abs(a[m])) if m.sum() else np.nan
    def f5(a, m):
        return np.mean(np.abs(a[m]) > 5) if m.sum() else np.nan
    print(f"run {args.tag}: held-out dither {args.holdout}; trace px {trace.sum()}, overlap px {overlap.sum()} ({overlap.sum() / max(trace.sum(), 1):.1%}), isolated px {isolated.sum()}")
    print(f"  OVERLAP pixels : median|chi| full model {r(chi_full, overlap):.2f} | dominant-source-only {r(chi_single, overlap):.2f} | no model {r(chi_none, overlap):.2f};"
          f"  |chi|>5: {f5(chi_full, overlap):.3f} | {f5(chi_single, overlap):.3f} | {f5(chi_none, overlap):.3f}")
    print(f"  ISOLATED pixels: median|chi| full model {r(chi_full, isolated):.2f} | no model {r(chi_none, isolated):.2f};  |chi|>5: {f5(chi_full, isolated):.3f} | {f5(chi_none, isolated):.3f}")
    # contamination fraction actually explained: in overlap pixels, how much of the secondary flux is there
    sec = (total - top1)
    print(f"  secondary (contaminant) flux in overlap px: median fraction of model {np.median((sec / np.maximum(total, 1e-6))[overlap]):.2f}; "
          f"median |data - model| / contaminant flux {np.median(np.abs(d - total - bkg)[overlap] / np.maximum(sec[overlap], 1e-6)):.2f}")


if __name__ == "__main__":
    main()
