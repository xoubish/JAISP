"""Production decontamination of one detector of one (or several) observation(s): every source with
at least half its 1st-order trace on the detector in >= 2 frames is a target; all frames are fitted
(no held-out).  Writes experiments/grism/products/<obs>/det<DET>_spectra.h5 with, per target, our
fitted 1st-order spectrum (detector counts per wavelength sample, i.e. not yet flux-calibrated), the
0th-order flux, the fitted (dx, dy) shift, H magnitude, stamp kind and frame coverage, plus the
per-frame sensitivity/background parameters and the lambda grid.

Usage: run_detector.py --obs 3025 --det 23 [--dithers 0,1,2,3] [--device cuda] [--iters 100]
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.grism.scene import GrismScene, TileStamps, load_frame, sky_catalog, catalog_for_frame, frame_key  # noqa: E402

BASE = REPO / "data" / "sir_edff"
CALIB = BASE / "calib" / "dr1_3025"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", type=int, default=3025)
    ap.add_argument("--det", default="23")
    ap.add_argument("--dithers", default="0,1,2,3")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--kernel", default=str(REPO / "experiments/grism/nisps_kernel_fit_kfit2.npz"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lam-step", type=float, default=13.4)
    ap.add_argument("--smooth", type=float, default=1.0)
    ap.add_argument("--shape-prior", type=float, default=1.0)
    ap.add_argument("--spec-step", type=int, default=3)
    ap.add_argument("--min-frac", type=float, default=0.5, help="min fraction of the 1st-order trace on the detector")
    ap.add_argument("--min-frames", type=int, default=2)
    ap.add_argument("--mer-root", default="/tmp/shemmati_grism/products/mer")
    ap.add_argument("--no-mer", action="store_true", help="JAISP tiles only, no MER mosaics/catalogue supplement")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    keys = [frame_key(args.obs, int(d)) for d in args.dithers.split(",")]
    frames = [load_frame(BASE, k, args.det, CALIB) for k in keys]
    skycat = sky_catalog(BASE)
    if not args.no_mer:
        from models.grism.mer_mosaic import mer_sky_catalog
        skycat = skycat + mer_sky_catalog(args.mer_root, exclude=skycat)
    for fr in frames:
        fr["cat"] = catalog_for_frame(fr, skycat)
    tiles = TileStamps(str(REPO / "data/euclid_tiles_all/tile_*_euclid.npz"))
    if not args.no_mer:
        from models.grism.mer_mosaic import MerMosaicStamps, StampUnion
        tiles = StampUnion(tiles, MerMosaicStamps(args.mer_root))

    def frac_on(s):
        if s["bb1"] is None:
            return 0.0
        lo, hi = s["bb1"][0], s["bb1"][1]
        if lo[1] < 0 or hi[1] > 2040:
            return 0.0
        return max(min(hi[0], 2040) - max(lo[0], 0), 0) / max(hi[0] - lo[0], 1)
    cnt = Counter()
    allcat = {}
    for fr in frames:
        for s in fr["cat"]:
            allcat.setdefault(s["id"], s)
            if frac_on(s) >= args.min_frac:
                cnt[s["id"]] += 1
    targets = {i: allcat[i] for i, n in cnt.items() if n >= args.min_frames}
    print(f"obs {args.obs} det {args.det}: frames {keys}; catalogue {len(skycat)}; on detector per frame {[len(fr['cat']) for fr in frames]}; "
          f"targets {len(targets)} (>= {args.min_frac:.0%} of the trace in >= {args.min_frames} frames)", flush=True)
    from models.grism.mer_psf import MerPsf, tile_for
    root = BASE / "products/mer_psf"
    tc = (np.mean([s["ra"] for s in targets.values()]), np.mean([s["dec"] for s in targets.values()]))
    star_psf = MerPsf(root, tile_for(root, *tc), "VIS", "GRID")
    scene = GrismScene(frames, args.det, (0, 0), (2040, 2040), tiles, kernel_npz=args.kernel, device=args.device,
                       lam_step=args.lam_step, targets=targets, star_psf=star_psf)
    kinds = Counter(getattr(s, "stamp_kind", "raw") for s in scene.sources.values())
    print(f"scene: {len(scene.sources)} sources, stamp kinds {dict(kinds)}, setup {time.time() - t0:.0f}s", flush=True)
    scene.init_params(spec_step=args.spec_step)
    if args.shape_prior > 0:
        scene.build_templates()
    t1 = time.time()
    scene.fit(frames, iters=args.iters, lr=0.1, clip_sigma=8.0, smooth=args.smooth, shape_prior_weight=args.shape_prior)
    print(f"fit in {time.time() - t1:.0f}s; final loss {scene.loss_history[-1]:.4f}", flush=True)
    n1, n0 = scene.calibrate_uncovered(keys)
    # training-frame residual summary (sanity)
    for fr in frames:
        chi, m = scene.chi_map(fr)
        good = fr["w"].cpu().numpy() > 1e-6
        noise = np.sqrt(1 / np.maximum(fr["w"].cpu().numpy(), 1e-12)); trace = (m > 3 * noise) & good
        chi0 = (fr["dat"].cpu().numpy() - np.median(fr["dat"].cpu().numpy()[good])) * np.sqrt(fr["w"].cpu().numpy())
        print(f"  frame {fr['dither']} {fr['grism']}_{fr['tilt']}: trace px {trace.sum()}, median|chi| in traces {np.median(np.abs(chi[trace])):.2f} "
              f"(no-model {np.median(np.abs(chi0[trace])):.2f}), |chi|>5 {np.mean(np.abs(chi[trace]) > 5):.3f}", flush=True)
    out_dir = REPO / "experiments/grism/products" / str(args.obs); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"det{args.det}_spectra{args.tag}.h5"
    with h5py.File(out, "w") as h:
        h.attrs.update(dict(obs=args.obs, det=args.det, frames=np.array(keys), iters=args.iters, kernel=Path(args.kernel).name,
                            spec_step=args.spec_step, smooth=args.smooth, shape_prior=args.shape_prior, units="detector counts per wavelength sample (not flux calibrated)"))
        h.create_dataset("lambda", data=scene.lam); h.create_dataset("lambda0", data=scene.lam0)
        gf = h.create_group("frames")
        for fr in frames:
            g = gf.create_group(str(fr["dither"]))
            for k in ("sens", "bkg", "dxy"):
                if k in fr and torch.is_tensor(fr[k]):
                    g.create_dataset(k, data=fr[k].detach().cpu().numpy())
            g.attrs.update(dict(grism=fr["grism"], tilt=fr["tilt"], path=str(fr["path"])))
        gs = h.create_group("sources")
        for sid, s in scene.sources.items():
            g = gs.create_group(sid)
            g.create_dataset("spec1", data=np.exp(scene.lf1(s).detach().cpu().numpy()).astype("f4"))
            g.create_dataset("spec0", data=np.exp(s.logf0.detach().cpu().numpy()).astype("f4"))
            g.attrs.update(dict(ra=s.ra, dec=s.dec, hmag=s.hmag, target=sid in targets, stamp_kind=getattr(s, "stamp_kind", "raw"),
                                shift=s.shift.detach().cpu().numpy(), coverage=np.array(sorted(s.coverage)),
                                fallback1=sid in getattr(scene, "fallback1", set()), mags=str(getattr(s, "mags", {}))))
    print(f"wrote {out} ({len(scene.sources)} sources, {len(targets)} targets; fallbacks {n1}/{n0}); total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
