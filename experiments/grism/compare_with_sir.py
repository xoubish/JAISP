"""Compare our joint-fit decontamination with SIR, object by object, on one detector strip.

Inputs: a finished scene fit (scene_test.py writes *_spectra.pt with per-source log spectra),
the SIR XTRACT product (1D spectra, masks, CONTAMINANTS ids) for the reference dither.

Plots:
  (1) for the N most-contaminated stars (SIR CONTAMINANTS non-empty, Type 1): SIR 1D (normalised)
      vs our spectrum (normalised), with the wavelength ranges where SIR's contaminant traces
      overlap the target trace (from our geometry) shaded, and SIR's NOT_USE pixels marked.
  (2) summary: for all stars in the strip, rms of (spectrum / smooth fit - 1) inside vs outside
      overlap ranges, SIR vs ours.
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
from scipy.signal import savgol_filter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.grism.scene import GrismScene, TileStamps, load_frame  # noqa: E402

BASE = REPO / "data" / "sir_edff"
CALIB = BASE / "calib" / "dr1_3025"
NOT_USE = 1


def smooth_norm(lam, s, mask=None, win=61):
    s = np.array(s, float)
    if mask is not None:
        s = np.where(mask, np.nan, s)
    ok = np.isfinite(s)
    if ok.sum() < win + 5:
        return np.full_like(s, np.nan)
    si = np.interp(lam, lam[ok], s[ok])
    sm = savgol_filter(si, win, 2)
    return s / np.where(sm > 0, sm, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="23")
    ap.add_argument("--box", default="0,1000,2040,1250")
    ap.add_argument("--dither", type=int, default=0, help="reference dither for the SIR comparison")
    ap.add_argument("--spectra", required=True, help="*_spectra.pt from scene_test.py")
    ap.add_argument("--lam-step", type=float, default=26.8)
    ap.add_argument("--nshow", type=int, default=6)
    ap.add_argument("--hmax", type=float, default=20.5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--sky-targets", default="", help="'ref_dither,y0,y1' as in scene_test.py")
    ap.add_argument("--galaxies", action="store_true", help="include galaxies (default: stars only)")
    args = ap.parse_args()
    x0, y0, x1, y1 = [int(v) for v in args.box.split(",")]
    fr = load_frame(BASE, args.dither, args.det, CALIB)
    tiles = TileStamps(str(REPO / "data/euclid_tiles_all/tile_*_euclid.npz"))
    targets = None
    if args.sky_targets:
        from models.grism.scene import catalog_from_loctab
        rd, ty0, ty1 = [int(v) for v in args.sky_targets.split(",")]
        ref = fr if rd == args.dither else load_frame(BASE, rd, args.det, CALIB)
        targets = {s["id"]: s for s in catalog_from_loctab(ref["lt"])
                   if s["bb1"] is not None and s["bb1"][0, 1] >= ty0 and s["bb1"][1, 1] <= ty1 and s["bb1"][0, 0] >= 0 and s["bb1"][1, 0] <= 2040}
        x0, y0, x1, y1 = 0, 0, 2040, 2040
    scene = GrismScene([fr], args.det, (x0, y0), (x1, y1), tiles, kernel_npz=None, device="cpu", lam_step=args.lam_step,
                       targets=targets)
    spec = torch.load(args.spectra)
    xt = h5py.File(sorted(glob.glob(str(BASE / f"products/3025/xtract/EUC_SIR_W-XTRACT_3025-{args.dither}-DET_{args.det}_*.h5")))[-1], "r")
    es = xt["Extracted Spectra"]
    lam_us = scene.lam
    rows = []
    for sid, src in scene.sources.items():
        if sid not in spec or sid not in es or (args.dither, 1) not in src.traces:
            continue
        g = es[sid][f"DET_{args.det}"]
        if "DATA/1D/Science" not in g or src.hmag > args.hmax:
            continue
        if not args.galaxies and int(g["Astronomical Object"].attrs["Type"]) != 1:
            continue
        cont = [str(c) for c in np.array(g["CONTAMINANTS"]).tolist()] if "CONTAMINANTS" in g else []
        s_sir = np.array(g["DATA/1D/Science"])[0]; m_sir = np.array(g["DATA/1D/Mask"])[0]
        lam_sir = float(g["DATA"].attrs["Wlen start"]) + float(g["DATA"].attrs["Wlen inc"]) * np.arange(len(s_sir))
        f_us = np.exp(spec[sid]["logf1"].numpy())
        # overlap ranges on the target's wavelength axis: where any contaminant's rendered 1st/0th-order
        # box overlaps the target trace pixels (using our geometry in the reference dither)
        tr = src.traces[(args.dither, 1)]
        tgt_x = tr.centre_px[0] + tr.shift_px[:, 0]; tgt_y = tr.centre_px[1] + tr.shift_px[:, 1]
        overlap = np.zeros(len(lam_us), bool)
        for cid in cont:
            c = scene.sources.get(cid)
            if c is None:
                continue
            for order in (1, 0):
                ctr = c.traces.get((args.dither, order))
                if ctr is None:
                    continue
                lo, hi = ctr.bbox(c.stamp.shape, pad=3)
                overlap |= (tgt_x >= lo[0]) & (tgt_x <= hi[0]) & (tgt_y >= lo[1] - 2) & (tgt_y <= hi[1] + 2)
        rows.append(dict(id=sid, h=src.hmag, ncont=len(cont), overlap=overlap, lam_sir=lam_sir, s_sir=s_sir, m_sir=m_sir, f_us=f_us))
    print(f"{len(rows)} {'objects' if args.galaxies else 'stars'} with SIR spectra in the strip; with contaminants: {sum(r['ncont'] > 0 for r in rows)}; "
          f"with overlap on the trace (our geometry): {sum(r['overlap'].any() for r in rows)}")
    # ---- (1) individual examples: most overlap, brightest
    ex = sorted([r for r in rows if r["overlap"].sum() > 5], key=lambda r: (r["h"]))[: args.nshow]
    fig, axes = plt.subplots(len(ex), 1, figsize=(16, 3.0 * max(len(ex), 1)))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, ex):
        ns = smooth_norm(r["lam_sir"], r["s_sir"], mask=(r["m_sir"] & NOT_USE) > 0)
        nu = smooth_norm(lam_us, r["f_us"])
        sel = (r["lam_sir"] > 12500) & (r["lam_sir"] < 18500)
        ax.plot(r["lam_sir"], r["s_sir"] / np.nanmedian(r["s_sir"][sel]), "k-", lw=0.8, label="SIR 1D (after SIR decontamination)")
        bad = (r["m_sir"] & NOT_USE) > 0
        ax.plot(r["lam_sir"][bad], (r["s_sir"] / np.nanmedian(r["s_sir"][sel]))[bad], "rx", ms=3, label="SIR NOT_USE")
        ax.plot(lam_us, r["f_us"] / np.nanmedian(r["f_us"][(lam_us > 12500) & (lam_us < 18500)]), "b-", lw=0.9, label="ours (joint fit, instrumental units)")
        yl = ax.get_ylim()
        ov = r["overlap"]
        if ov.any():
            ax.fill_between(lam_us, yl[0], yl[1], where=ov, color="orange", alpha=0.25, label="contaminant overlap (our geometry)")
        ax.set_ylim(0, 2.5); ax.set_title(f"{r['id']} H={r['h']:.2f} SIR contaminants={r['ncont']}", fontsize=9); ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("lambda [A]")
    out1 = REPO / f"experiments/grism/compare_sir_examples_det{args.det}_d{args.dither}{args.tag}.png"
    fig.savefig(out1, dpi=100, bbox_inches="tight"); print("wrote", out1)
    # ---- (2) summary
    res = []
    for r in rows:
        ns = smooth_norm(r["lam_sir"], r["s_sir"], mask=(r["m_sir"] & NOT_USE) > 0)
        nu = smooth_norm(lam_us, r["f_us"])
        ov_sir = np.interp(r["lam_sir"], lam_us, r["overlap"].astype(float)) > 0.5
        def rms(a, m):
            a = a[m]; a = a[np.isfinite(a)]
            return np.sqrt(np.mean((a - 1) ** 2)) if a.size > 10 else np.nan
        res.append((r["h"], r["ncont"], r["overlap"].mean(),
                    rms(ns, ov_sir), rms(ns, ~ov_sir), rms(nu, r["overlap"]), rms(nu, ~r["overlap"]),
                    np.mean((r["m_sir"] & NOT_USE) > 0)))
    res = np.array(res)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ok = np.isfinite(res[:, 3]) & np.isfinite(res[:, 5])
    axes[0].scatter(res[ok, 3], res[ok, 5], c=res[ok, 0], s=18, cmap="viridis"); axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_xlabel("SIR: rms scatter about smooth continuum, in contaminated ranges"); axes[0].set_ylabel("ours: same"); axes[0].set_xlim(0, 0.8); axes[0].set_ylim(0, 0.8)
    axes[0].set_title(f"stars with contaminant overlap (n={ok.sum()}), colour = H")
    ok2 = np.isfinite(res[:, 4]) & np.isfinite(res[:, 6])
    axes[1].scatter(res[ok2, 4], res[ok2, 6], c=res[ok2, 0], s=18, cmap="viridis"); axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_xlabel("SIR: rms in clean ranges"); axes[1].set_ylabel("ours: rms in clean ranges"); axes[1].set_xlim(0, 0.8); axes[1].set_ylim(0, 0.8)
    axes[1].set_title("clean ranges (noise floor comparison)")
    axes[2].scatter(res[:, 0], res[:, 7], s=12); axes[2].set_xlabel("H"); axes[2].set_ylabel("fraction of SIR pixels flagged NOT_USE"); axes[2].set_title("what SIR discards")
    out2 = REPO / f"experiments/grism/compare_sir_summary_det{args.det}_d{args.dither}{args.tag}.png"
    fig.savefig(out2, dpi=100, bbox_inches="tight"); print("wrote", out2)
    print("median rms in contaminated ranges: SIR %.3f ours %.3f ; clean ranges: SIR %.3f ours %.3f" %
          (np.nanmedian(res[:, 3]), np.nanmedian(res[:, 5]), np.nanmedian(res[:, 4]), np.nanmedian(res[:, 6])))


if __name__ == "__main__":
    main()
