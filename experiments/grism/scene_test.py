"""Scene test on a DET23 sub-region: (1) render all sources with free spectra fitted to the
training frames; (2) held-out orientation: fit on 3 dithers, predict the 4th, compare chi
in the held-out frame with a 'no model' baseline and with a 'single source only' baseline.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.grism.scene import GrismScene, TileStamps, load_frame, catalog_from_loctab, sky_catalog, catalog_for_frame  # noqa: E402

BASE = REPO / "data" / "sir_edff"
CALIB = BASE / "calib" / "dr1_3025"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="23")
    ap.add_argument("--box", default="900,1000,1400,1300", help="x0,y0,x1,y1 detector pixels")
    ap.add_argument("--dithers", default="0,1,2,3", help="frame keys: 0-3 = obs 3025 dithers; obs*10+dither otherwise (e.g. 30260); "
                    "append ':DET' to use another detector for that frame (e.g. 30300:31)")
    ap.add_argument("--mer-stamps", action="store_true", help="VIS cutouts from the MER mosaics where the JAISP tiles do not reach")
    ap.add_argument("--mer-root", default="/tmp/shemmati_grism/products/mer")
    ap.add_argument("--mer-cat", action="store_true", help="add MER-catalogue sources (H<23.5) outside the SIR location-table footprint")
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--kernel", default=str(REPO / "experiments/grism/nisps_kernel.npz"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lam-step", type=float, default=13.4)
    ap.add_argument("--smooth", type=float, default=1.0, help="weight of the log-spectrum smoothness prior")
    ap.add_argument("--fit-kernel", action="store_true", help="make the NISP-S kernel learnable (3 lambda groups)")
    ap.add_argument("--foundation-stamps", action="store_true", help="use A6_v4 foundation templates where available")
    ap.add_argument("--star-psf", action="store_true", help="use the MER VIS PSF as the morphology of SIR Type-1 stars")
    ap.add_argument("--align-holdout", action="store_true", help="fit a global (dx,dy)+background of the held-out frame before evaluating")
    ap.add_argument("--prior-reduce", default="mean", choices=["mean", "sum"], help="per-source priors summed (sum) or averaged (mean)")
    ap.add_argument("--shape-prior", type=float, default=1.0, help="weight of the Y/J/H photometric shape prior (0 = off)")
    ap.add_argument("--spec-step", type=int, default=3, help="parametrise spectra every N wavelength samples (3 ~ one resolution element)")
    ap.add_argument("--sky-targets", default="", help="'ref_dither,y0,y1': targets = sources whose 1st-order trace in "
                    "ref_dither lies in rows y0..y1; the scene then spans the whole detector in every frame")
    args = ap.parse_args()
    x0, y0, x1, y1 = [int(v) for v in args.box.split(",")]
    dithers = [(int(d.split(":")[0]), d.split(":")[1] if ":" in d else args.det) for d in args.dithers.split(",")]
    t0 = time.time()
    frames = [load_frame(BASE, d, det, CALIB) for d, det in dithers]
    skycat = sky_catalog(BASE)
    if args.mer_cat:
        from models.grism.mer_mosaic import mer_sky_catalog
        extra = mer_sky_catalog(args.mer_root, exclude=skycat)
        print(f"MER catalogue supplement: {len(extra)} sources without a SIR counterpart")
        skycat = skycat + extra
    for fr in frames:
        fr["cat"] = catalog_for_frame(fr, skycat)
    print(f"sky catalogue: {len(skycat)} sources; on this detector per frame: {[len(fr['cat']) for fr in frames]}")
    tiles = TileStamps(str(REPO / "data/euclid_tiles_all/tile_*_euclid.npz"))
    if args.mer_stamps:
        from models.grism.mer_mosaic import MerMosaicStamps, StampUnion
        tiles = StampUnion(tiles, MerMosaicStamps(args.mer_root))
    targets = None
    if args.sky_targets == "common":
        # multi-observation targets: 1st-order trace fully on the detector in EVERY frame
        def inside(s):
            return s["bb1"] is not None and s["bb1"][0, 0] >= 0 and s["bb1"][1, 0] <= 2040 and s["bb1"][0, 1] >= 0 and s["bb1"][1, 1] <= 2040
        def partial(s, frac=0.5):
            # at least `frac` of the 1st-order trace length on the detector
            if s["bb1"] is None:
                return False
            lo, hi = s["bb1"][0], s["bb1"][1]
            ov = min(hi[0], 2040) - max(lo[0], 0)
            return ov >= frac * max(hi[0] - lo[0], 1) and lo[1] >= 0 and hi[1] <= 2040
        # traces run off the detector edge in some dithers, so: at least half the trace on the detector in
        # >= 2 frames of EVERY observation present (the strip between neighbouring pointings)
        from collections import Counter
        part = [{s["id"] for s in fr["cat"] if partial(s)} for fr in frames]
        obs_of = [fr["obs"] for fr in frames]
        cnt = {o: Counter() for o in set(obs_of)}
        for o, ids in zip(obs_of, part):
            cnt[o].update(ids)
        common = set.intersection(*[{i for i, n in c.items() if n >= 2} for c in cnt.values()])
        allcat = {}
        for fr in frames:
            for s in fr["cat"]:
                allcat.setdefault(s["id"], s)
        targets = {i: allcat[i] for i in common}
        x0, y0, x1, y1 = 0, 0, 2040, 2040
        print(f"sky targets: {len(targets)} sources with >=half the 1st-order trace on the detector in >=2 frames of each of "
              f"{len(cnt)} observations (>=half per frame {[len(x) for x in part]})")
    elif args.sky_targets:
        rd, ty0, ty1 = [int(v) for v in args.sky_targets.split(",")]
        ref = [fr for fr in frames if fr["dither"] == rd][0]
        targets = {s["id"]: s for s in ref["cat"]
                   if s["bb1"] is not None and s["bb1"][0, 1] >= ty0 and s["bb1"][1, 1] <= ty1 and s["bb1"][0, 0] >= 0 and s["bb1"][1, 0] <= 2040}
        x0, y0, x1, y1 = 0, 0, 2040, 2040
        print(f"sky targets: {len(targets)} sources from dither {rd} rows {ty0}-{ty1}")
    provider = None
    if args.foundation_stamps:
        from models.grism.foundation_stamps import FoundationStamps
        provider = FoundationStamps(device=args.device)
    star_psf = None
    if args.star_psf:
        from models.grism.mer_psf import MerPsf, tile_for
        root = BASE / "products/mer_psf"
        tc = (np.mean([s["ra"] for s in targets.values()]), np.mean([s["dec"] for s in targets.values()])) if targets else (53.09, -28.1)
        tile = tile_for(root, *tc)
        star_psf = MerPsf(root, tile, "VIS", "GRID")
    scene = GrismScene(frames, args.det, (x0, y0), (x1, y1), tiles, kernel_npz=args.kernel, device=args.device,
                       lam_step=args.lam_step, targets=targets, stamp_provider=provider, star_psf=star_psf)
    kinds = [getattr(s, "stamp_kind", "raw") for s in scene.sources.values()]
    print(f"stamp kinds: foundation {kinds.count('foundation')}, psf {kinds.count('psf')}, raw {kinds.count('raw')} of {len(kinds)} sources"
          + (f"; provider stats {provider.stats}" if provider is not None else "")
          + (f"; stamp sources (0=JAISP tiles, 1=MER mosaics, -1=none) {tiles.counts}" if args.mer_stamps else ""))
    if args.fit_kernel:
        scene.enable_kernel_fit(args.kernel, ngroups=3)
    if targets is not None:
        for fr in frames:
            print(f"  dither {fr['dither']}: {int(scene.pixmask[fr['dither']].sum())} masked pixels; "
                  f"targets with 1st-order trace on this detector: {sum((fr['dither'], 1) in scene.sources[t].traces for t in targets if t in scene.sources)}")
    train = [fr for fr in frames if fr["dither"] != args.holdout]
    test = [fr for fr in frames if fr["dither"] == args.holdout]
    train_keys = [fr["dither"] for fr in train]
    ncov = np.array([sum(k in src.coverage for k in train_keys) for src in scene.sources.values()])
    print(f"scene: {len(scene.sources)} sources in box {args.box}, L={scene.L} ({time.time() - t0:.0f}s setup); "
          f"sources with 1st-order coverage in >=2 training frames: {(ncov >= 2).sum()}, in 0: {(ncov == 0).sum()}")
    scene.init_params(spec_step=args.spec_step)
    scene.prior_reduce = args.prior_reduce
    if args.shape_prior > 0:
        scene.build_templates()
    print("cosmic rays masked per frame:", {fr["dither"]: fr.get("n_cr", 0) for fr in frames})
    t0 = time.time()
    scene.fit(train, iters=args.iters, lr=0.1, clip_sigma=8.0, smooth=args.smooth, shape_prior_weight=args.shape_prior)
    print(f"fit on dithers {train_keys} in {time.time() - t0:.0f}s")
    np.save(REPO / f"experiments/grism/scene_loss_det{args.det}_d{args.holdout}{args.tag}.npy", np.array(scene.loss_history))
    if args.fit_kernel:
        kt = scene.kernel_table()
        np.savez(REPO / f"experiments/grism/nisps_kernel_fit{args.tag}.npz", **kt, unit="MER px (0.1 arcsec)")
        for g in range(len(kt["lam"])):
            print(f"fitted kernel group {g} ({kt['lam'][g]:.0f} A): core FWHM {0.2355 * kt['sigma1'][g]:.3f}\", "
                  f"wing FWHM {0.2355 * kt['sigma2'][g]:.2f}\", wing frac {kt['f2'][g]:.3f}")
    n1, n0 = scene.calibrate_uncovered(train_keys)
    print(f"fallbacks: {n1} sources given H-scaled flat 1st-order spectra, {n0} given 0th-order flux from the 1st order (ratio {scene.ratio0:.3f})")
    # held-out frame: copy the sensitivity ratio of a training frame with the same grism (else reference)
    with torch.no_grad():
        for fr in test:
            # sensitivity from a trained frame of the same grism: prefer the same observation, then the frame
            # with the most target pixels (a frame with no target pixels never updates its sensitivity)
            def npix(t):
                return float(scene.pixmask[t["dither"]].sum()) if t["dither"] in getattr(scene, "pixmask", {}) else 1.0
            same = sorted([t for t in train if t["grism"] == fr["grism"] and npix(t) > 0],
                          key=lambda t: (t["obs"] == fr["obs"], npix(t)), reverse=True)
            src = same[0] if same else train[0]
            print(f"held-out frame {fr['dither']}: sensitivity copied from frame {src['dither']} ({src['grism']}_{src['tilt']}, {npix(src):.0f} target px)")
            fr["sens"].copy_(src["sens"])
            fr["bkg"].copy_(torch.stack([t["bkg"] for t in train if npix(t) > 0]).mean(0))
    well = {src.id for src in scene.sources.values() if sum(k in src.coverage for k in train_keys) >= 2}
    if args.align_holdout:
        for fr in test:
            # freeze everything except the held-out frame's global offset and background
            for p in scene.params(frames=False):
                p.requires_grad_(False)
            dxy = scene.align_frame(fr)
            print(f"held-out frame alignment: global offset (dx, dy) = ({dxy[0]:+.3f}, {dxy[1]:+.3f}) px")

    # evaluation
    def stats(fr, label):
        chi, m = scene.chi_map(fr)
        good = fr["w"].cpu().numpy() > 1e-6
        src_all = scene.render_frame(fr, with_bkg=False).detach().cpu().numpy()
        src_well = scene.render_frame(fr, with_bkg=False, only=well).detach().cpu().numpy()
        noise = np.sqrt(1 / np.maximum(fr["w"].cpu().numpy(), 1e-12))
        trace = (src_all > 3 * noise) & good
        constrained = trace & (src_well > 0.9 * src_all)          # pixels dominated by well-constrained sources
        d = fr["dat"].cpu().numpy(); w = fr["w"].cpu().numpy()
        chi0 = (d - np.median(d[good])) * np.sqrt(w)
        def r(a, m_):
            return np.median(np.abs(a[m_])) if m_.sum() else np.nan
        print(f"{label} dither {fr['dither']} {fr['grism']}_{fr['tilt']}: pixels {good.sum()}, trace px {trace.sum()}, constrained trace px {constrained.sum()}; "
              f"median|chi| all {r(chi, good):.2f} (no-model {r(chi0, good):.2f}); in traces {r(chi, trace):.2f} (no-model {r(chi0, trace):.2f}); "
              f"constrained traces {r(chi, constrained):.2f} (no-model {r(chi0, constrained):.2f}); "
              f"|chi|>5 frac all {np.mean(np.abs(chi[good]) > 5):.4f} (no-model {np.mean(np.abs(chi0[good]) > 5):.4f}), "
              f"in traces {np.mean(np.abs(chi[trace]) > 5):.3f} (no-model {np.mean(np.abs(chi0[trace]) > 5):.3f})")
        return chi, m
    def crop(fr):
        """rows/cols with any weight (for plotting the masked region only)"""
        w = fr["w"].cpu().numpy() > 0
        ys = np.where(w.any(1))[0]; xs = np.where(w.any(0))[0]
        return (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)) if len(ys) else (slice(None), slice(None))
    for fr in train:
        chi_tr, m_tr = stats(fr, "TRAIN")
        if fr is train[0]:
            sl = crop(fr); d = fr["dat"].cpu().numpy()[sl]; m_tr = m_tr[sl]; chi_tr = chi_tr[sl]; v = np.percentile(d, [1, 99.5])
            fig, axes = plt.subplots(3, 1, figsize=(18, 14))
            axes[0].imshow(d, origin="lower", vmin=v[0], vmax=v[1], cmap="gray", aspect="auto"); axes[0].set_title(f"train dither {fr['dither']} data")
            axes[1].imshow(m_tr, origin="lower", vmin=v[0], vmax=v[1], cmap="gray", aspect="auto"); axes[1].set_title("model")
            axes[2].imshow(chi_tr, origin="lower", vmin=-5, vmax=5, cmap="RdBu_r", aspect="auto"); axes[2].set_title("chi")
            fig.savefig(REPO / f"experiments/grism/scene_train_det{args.det}_d{fr['dither']}{args.tag}.png", dpi=90, bbox_inches="tight")
    chi_t, m_t = stats(test[0], "HELD-OUT")
    # figure for held-out frame
    fr = test[0]; sl = crop(fr); d = fr["dat"].cpu().numpy()[sl]; m_t = m_t[sl]; chi_t = chi_t[sl]
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    v = np.percentile(d, [1, 99.5])
    axes[0].imshow(d, origin="lower", vmin=v[0], vmax=v[1], cmap="gray", aspect="auto"); axes[0].set_title(f"held-out dither {fr['dither']} {fr['grism']}_{fr['tilt']} data")
    axes[1].imshow(m_t, origin="lower", vmin=v[0], vmax=v[1], cmap="gray", aspect="auto"); axes[1].set_title("predicted from the other 3 orientations")
    axes[2].imshow(chi_t, origin="lower", vmin=-5, vmax=5, cmap="RdBu_r", aspect="auto"); axes[2].set_title("chi")
    out = REPO / f"experiments/grism/scene_holdout_det{args.det}_d{args.holdout}{args.tag}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight"); print("wrote", out)
    saved = {sid: dict(logf1=scene.lf1(s).detach().cpu(), logf0=s.logf0.detach().cpu(), shift=s.shift.detach().cpu(), hmag=s.hmag,
                       stamp_kind=getattr(s, "stamp_kind", "raw"), coverage=list(s.coverage))
             for sid, s in scene.sources.items()}
    saved["_frames"] = {fr["dither"]: {k: fr[k].detach().cpu() for k in ("sens", "bkg", "dxy") if k in fr and torch.is_tensor(fr[k])} for fr in frames}
    torch.save(saved, str(out).replace(".png", "_spectra.pt"))
    # per-stamp-kind held-out metric: pixels dominated by foundation-stamp vs raw-stamp sources
    if provider is not None:
        fr = test[0]
        with torch.no_grad():
            fnd = {s.id for s in scene.sources.values() if getattr(s, "stamp_kind", "raw") == "foundation"}
            m_f = scene.render_frame(fr, with_bkg=False, only=fnd).cpu().numpy()
            m_all = scene.render_frame(fr, with_bkg=False).cpu().numpy()
            chi, _ = scene.chi_map(fr)
        good = fr["w"].cpu().numpy() > 1e-6
        noise = np.sqrt(1 / np.maximum(fr["w"].cpu().numpy(), 1e-12))
        tr_ = (m_all > 3 * noise) & good
        dom_f = tr_ & (m_f > 0.9 * m_all); dom_r = tr_ & (m_f < 0.1 * m_all)
        print(f"HELD-OUT by stamp kind: foundation-dominated trace px {dom_f.sum()} median|chi| {np.median(np.abs(chi[dom_f])):.2f}, "
              f"|chi|>5 {np.mean(np.abs(chi[dom_f]) > 5):.3f};  raw-dominated trace px {dom_r.sum()} median|chi| {np.median(np.abs(chi[dom_r])):.2f}, "
              f"|chi|>5 {np.mean(np.abs(chi[dom_r]) > 5):.3f}")


if __name__ == "__main__":
    main()
