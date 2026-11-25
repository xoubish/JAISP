"""Score a GELSA held-out prediction (rendered by GELSA's own spectral imager into dither 3, DET23)
against the data, side by side with our run, with the same trace-pixel chi metric.
Usage: eval_gelsa.py --gelsa <model_d3_det23.npy> [--tag _ss] [--star-psf]"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.grism.scene import GrismScene, TileStamps, load_frame, catalog_from_loctab  # noqa: E402

BASE = REPO / "data" / "sir_edff"; CALIB = BASE / "calib" / "dr1_3025"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gelsa", required=True)
    ap.add_argument("--tag", default="_ss")
    ap.add_argument("--star-psf", action="store_true")
    ap.add_argument("--kernel", default=str(REPO / "experiments/grism/nisps_kernel_fit_kfit2.npz"))
    ap.add_argument("--sky-targets", default="0,1080,1170")
    args = ap.parse_args()
    frames = [load_frame(BASE, d, "23", CALIB) for d in (0, 1, 2, 3)]
    fr = frames[3]
    g = np.load(args.gelsa).astype(float)
    # our model
    rd, ty0, ty1 = [int(v) for v in args.sky_targets.split(",")]
    ref = frames[rd]
    targets = {s["id"]: s for s in catalog_from_loctab(ref["lt"])
               if s["bb1"] is not None and s["bb1"][0, 1] >= ty0 and s["bb1"][1, 1] <= ty1 and s["bb1"][0, 0] >= 0 and s["bb1"][1, 0] <= 2040}
    tiles = TileStamps(str(REPO / "data/euclid_tiles_all/tile_*_euclid.npz"))
    star_psf = None
    if args.star_psf:
        from models.grism.mer_psf import MerPsf, tile_for
        root = BASE / "products/mer_psf"; star_psf = MerPsf(root, tile_for(root, 53.09, -28.1), "VIS", "GRID")
    scene = GrismScene(frames, "23", (0, 0), (2040, 2040), tiles, kernel_npz=args.kernel, device="cpu", lam_step=13.4, targets=targets, star_psf=star_psf)
    sp = torch.load(REPO / f"experiments/grism/scene_holdout_det23_d3{args.tag}_spectra.pt")
    scene.init_params(spec_step=1)
    for sid, s in scene.sources.items():
        if sid in sp:
            with torch.no_grad():
                s.logf1.copy_(sp[sid]["logf1"]); s.logf0.copy_(sp[sid]["logf0"]); s.shift.copy_(sp[sid]["shift"])
    d = fr["dat"].numpy(); w = fr["w"].numpy(); good = w > 0; noise = np.sqrt(1 / np.maximum(w, 1e-12))
    assert g.shape == d.shape, (g.shape, d.shape)
    with torch.no_grad():
        ours = scene.render_frame(fr, with_bkg=False).numpy()
        tmask = np.asarray(scene.pixmask[fr["dither"]]) > 0 if fr["dither"] in getattr(scene, "pixmask", {}) else np.ones_like(good)
    def bkg_of(model):
        m = good & (model < noise)
        return np.median((d - model)[m]) if m.sum() > 100 else 0.0
    def report(name, model, mask):
        b = bkg_of(model); chi = (d - model - b) * np.sqrt(w); chi0 = (d - b) * np.sqrt(w)
        m = mask & good
        print(f"  {name:28s} px {m.sum():7d}: median|chi| {np.median(np.abs(chi[m])):.2f} (no-model {np.median(np.abs(chi0[m])):.2f}); "
              f"|chi|>5 {np.mean(np.abs(chi[m]) > 5):.3f} (no-model {np.mean(np.abs(chi0[m]) > 5):.3f}); bkg {b:.3f}")
    # diagnostic: GELSA model scale vs data in its own trace pixels
    tg = (g > 3 * noise) & good; to = (ours > 3 * noise) & good
    if tg.sum():
        sc = np.sum(w[tg] * d[tg] * g[tg]) / np.sum(w[tg] * g[tg] ** 2)
        print(f"GELSA model: trace px {tg.sum()}, best-fit data/model scale {sc:.3f} (1 = same units)")
    print("Held-out dither 3, DET23 (targets region rows 1080-1170 in dither 0):")
    print(" on GELSA's trace pixels (model > 3 sigma):")
    report("GELSA", g, tg); report(f"ours{args.tag}", ours, tg)
    print(" on our trace pixels:")
    report("GELSA", g, to); report(f"ours{args.tag}", ours, to)
    print(" on the target region (our per-frame pixel mask):")
    report("GELSA", g, tmask & (to | tg)); report(f"ours{args.tag}", ours, tmask & (to | tg))
    np.save(str(Path(args.gelsa).with_suffix("")) + "_ours.npy", ours.astype("f4"))


if __name__ == "__main__":
    main()
