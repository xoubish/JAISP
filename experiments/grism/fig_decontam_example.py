"""Reader-facing decontamination figure for one target in the held-out frame.

Rows: (1) data along the target's 1st-order trace; (2) model of everything except the target
(contaminants, 0th orders, background); (3) data minus contaminants = cleaned 2D spectrogram;
(4) our target-only model; (5) 1D: raw boxcar extraction of the data, boxcar extraction of the
cleaned data, our fitted spectrum, SIR's 1D (normalised), with SIR NOT_USE pixels and the
contaminant-overlap ranges marked.
"""
import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.grism.scene import GrismScene, TileStamps, load_frame, catalog_from_loctab  # noqa: E402

BASE = REPO / "data" / "sir_edff"
CALIB = BASE / "calib" / "dr1_3025"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_kfit2")
    ap.add_argument("--kernel", default=str(REPO / "experiments/grism/nisps_kernel_fit_kfit2.npz"))
    ap.add_argument("--star-psf", action="store_true", default=True)
    ap.add_argument("--foundation-stamps", action="store_true")
    ap.add_argument("--frame", type=int, default=3, help="dither to show (3 = held-out)")
    ap.add_argument("--n", type=int, default=3, help="number of targets to show (most contaminated, H<20.5)")
    ap.add_argument("--half", type=int, default=7, help="half-height of the trace cutout [px]")
    args = ap.parse_args()
    frames = [load_frame(BASE, d, "23", CALIB) for d in (0, 1, 2, 3)]
    ref = frames[0]
    targets = {s["id"]: s for s in catalog_from_loctab(ref["lt"])
               if s["bb1"] is not None and 1080 <= s["bb1"][0, 1] and s["bb1"][1, 1] <= 1170 and s["bb1"][0, 0] >= 0 and s["bb1"][1, 0] <= 2040}
    tiles = TileStamps(str(REPO / "data/euclid_tiles_all/tile_*_euclid.npz"))
    star_psf = None
    if args.star_psf:
        from models.grism.mer_psf import MerPsf, tile_for
        root = BASE / "products/mer_psf"; star_psf = MerPsf(root, tile_for(root, 53.09, -28.1), "VIS", "GRID")
    provider = None
    if args.foundation_stamps:
        from models.grism.foundation_stamps import FoundationStamps
        provider = FoundationStamps(device="cpu")
    scene = GrismScene(frames, "23", (0, 0), (2040, 2040), tiles, kernel_npz=args.kernel, device="cpu", lam_step=13.4,
                       targets=targets, star_psf=star_psf, stamp_provider=provider)
    sp = torch.load(REPO / f"experiments/grism/scene_holdout_det23_d3{args.tag}_spectra.pt")
    scene.init_params()
    for sid, s in scene.sources.items():
        if sid in sp:
            with torch.no_grad():
                s.logf1.copy_(sp[sid]["logf1"]); s.logf0.copy_(sp[sid]["logf0"]); s.shift.copy_(sp[sid]["shift"])
    fr = [f for f in frames if f["dither"] == args.frame][0]
    d = fr["dat"].numpy(); w = fr["w"].numpy(); noise = np.sqrt(1 / np.maximum(w, 1e-12)); good = w > 0
    xt = h5py.File(sorted(glob.glob(str(BASE / f"products/3025/xtract/EUC_SIR_W-XTRACT_3025-{args.frame}-DET_23_*.h5")))[-1], "r")["Extracted Spectra"]
    lam = scene.lam
    # candidate targets: contamination fraction along their trace in this frame
    with torch.no_grad():
        full = scene.render_frame(fr, with_bkg=False).numpy()
        bkg = np.median((d - full)[good & (full < noise)])
    cands = []
    for sid in targets:
        s = scene.sources.get(sid)
        if s is None or (args.frame, 1) not in s.traces or not (16.5 < s.hmag < 19.5):
            continue
        trc = s.traces[(args.frame, 1)]; xx_ = trc.centre_px[0] + trc.shift_px[:, 0]; yy_ = trc.centre_px[1] + trc.shift_px[:, 1]
        if xx_.min() < 10 or xx_.max() > 2030 or yy_.min() < 15 or yy_.max() > 2025:
            continue   # trace must be fully on the detector in this frame
        with torch.no_grad():
            own = scene.render_frame(fr, with_bkg=False, only={sid}, orders=(1,)).numpy()
        m = own > 0.2 * own.max()
        if m.sum() < 200:
            continue
        cont = float(((full - own)[m]).sum() / max(own[m].sum(), 1))
        if 0.3 <= cont <= 3.0:
            cands.append((cont, sid))
    cands.sort(reverse=True)
    picks = [sid for c, sid in cands[: args.n]]
    print("targets (contamination fraction along trace):", [(round(c, 2), sid) for c, sid in cands[: args.n]])
    fig = plt.figure(figsize=(16, 7.0 * len(picks)))
    gs = fig.add_gridspec(len(picks) * 5, 1, height_ratios=[1, 1, 1, 1, 2.2] * len(picks), hspace=0.12)
    for k, sid in enumerate(picks):
        s = scene.sources[sid]; tr = s.traces[(args.frame, 1)]
        with torch.no_grad():
            own = scene.render_frame(fr, with_bkg=False, only={sid}, orders=(1,)).numpy()
        others = full - own
        # cutout along the trace: columns spanned by the trace, rows around its centre
        xs = tr.centre_px[0] + tr.shift_px[:, 0]; ys = tr.centre_px[1] + tr.shift_px[:, 1]
        x0, x1 = int(np.floor(xs.min())) - 5, int(np.ceil(xs.max())) + 5; yc = int(round(ys.mean()))
        y0, y1 = yc - args.half, yc + args.half + 1
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, 2040), min(y1, 2040)
        sl = (slice(y0, y1), slice(x0, x1))
        D = d[sl] - bkg; C = others[sl]; O = own[sl]; clean = D - C
        sig = np.median(noise[sl]); st = lambda im: np.arcsinh(im / (3 * sig)); vmax = st(np.percentile(D, 99.7))
        panels = [(D, "data (held-out orientation)"), (C, "model of all OTHER sources (contaminants, 0th orders)"),
                  (clean, "data - contaminants  (cleaned 2D spectrogram)"), (O, "our model of the target")]
        for j, (im, title) in enumerate(panels):
            ax = fig.add_subplot(gs[5 * k + j, 0])
            ax.imshow(st(im), origin="lower", cmap="gray", vmin=-1, vmax=vmax, aspect="auto", extent=[x0, x1, y0, y1])
            ax.plot(xs, ys, "c:", lw=0.6, alpha=0.5)
            ax.set_yticks([]); ax.set_xticks([])
            for lm in (13000, 15000, 17000, 19000):
                i = int(np.argmin(np.abs(lam - lm))); ax.axvline(xs[i], color="y", lw=0.4, alpha=0.6); ax.text(xs[i], y1 - 1.5, f"{lm}", color="y", fontsize=6, ha="center")
            ax.set_ylabel(title, fontsize=8, rotation=0, ha="right", va="center")
            if j == 0:
                ax.set_title(f"{sid}  H={s.hmag:.2f}  ({'star' if sid in targets and targets[sid]['type'] == 1 else 'galaxy'}), contamination/target flux along trace = {dict(cands)[sid] if False else [c for c, i in cands if i == sid][0]:.2f}", fontsize=10)
        # 1D: boxcar (+-3 px about the predicted trace) extraction per wavelength sample
        def boxcar(img):
            out = np.full(len(lam), np.nan)
            for i in range(len(lam)):
                x = int(round(xs[i])); y = int(round(ys[i]))
                if 0 <= x < 2040 and 3 <= y < 2037:
                    col = img[y - 3:y + 4, x]; ww = w[y - 3:y + 4, x]
                    out[i] = col[ww > 0].sum() if (ww > 0).sum() >= 5 else np.nan
            return out
        raw1d = boxcar(d - bkg); clean1d = boxcar(d - bkg - others); own1d = boxcar(own)
        sm = lambda a: np.convolve(np.nan_to_num(a), np.ones(5) / 5, mode="same")
        raw1d, clean1d = sm(raw1d), sm(clean1d)
        f_us = np.exp(s.logf1.detach().numpy()) * scene.sens_factor(fr, 1).detach().numpy()
        ax = fig.add_subplot(gs[5 * k + 4, 0])
        sel = (lam > 13000) & (lam < 18000)
        norm = np.nanmedian(own1d[sel]) if np.isfinite(np.nanmedian(own1d[sel])) and np.nanmedian(own1d[sel]) > 0 else 1.0
        ax.plot(lam, raw1d / norm, color="0.6", lw=0.8, label="raw boxcar extraction of the data (5-bin smoothed)")
        ax.plot(lam, clean1d / norm, color="k", lw=0.9, label="boxcar extraction after subtracting our contaminant model (5-bin smoothed)")
        ax.plot(lam, own1d / norm, color="tab:blue", lw=1.2, label="our fitted target spectrum (rendered, same aperture)")
        if sid in xt and "DATA/1D/Science" in xt[sid]["DET_23"]:
            g = xt[sid]["DET_23"]; s1 = np.array(g["DATA/1D/Science"])[0]; m1 = np.array(g["DATA/1D/Mask"])[0]
            l1 = float(g["DATA"].attrs["Wlen start"]) + float(g["DATA"].attrs["Wlen inc"]) * np.arange(len(s1))
            sel1 = (l1 > 13000) & (l1 < 18000) & ((m1 & 1) == 0)
            n1 = np.nanmedian(s1[sel1]) if sel1.sum() else 1.0
            ax.plot(l1, s1 / n1, color="tab:red", lw=0.8, alpha=0.8, label="SIR 1D (this dither, SIR decontamination)")
            bad = (m1 & 1) > 0
            ax.plot(l1[bad], (s1 / n1)[bad], "rx", ms=3, label="SIR NOT_USE")
        contam = C.sum(0) if False else None
        ax.set_ylim(-0.5, 3); ax.set_xlim(lam[0], lam[-1]); ax.set_ylabel("flux / continuum", fontsize=9)
        ax.legend(fontsize=7, loc="upper right", ncol=2); ax.grid(alpha=0.3)
        if k == len(picks) - 1:
            ax.set_xlabel("wavelength [A]")
    out = REPO / f"experiments/grism/fig_decontam_example{args.tag}_d{args.frame}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
