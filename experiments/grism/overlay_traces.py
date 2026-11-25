"""Overlay predicted 0th/1st-order traces on a real SIR BKGSUB frame detector.

Picks the brightest LOCTAB objects on the detector, draws the predicted trace
(from sky coordinates through our geometry) on the science image, and saves a
PNG.  Visual check of the pixel orientation conventions on the data itself.
"""
import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.grism.sir_geometry import FrameGeometry  # noqa: E402

BASE = REPO / "data" / "sir_edff"
CALIB = BASE / "calib" / "dr1_3025"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dither", type=int, default=0)
    ap.add_argument("--det", default="23")
    ap.add_argument("--nbright", type=int, default=12)
    ap.add_argument("--out", default=str(REPO / "experiments/grism/overlay_d{dither}_det{det}.png"))
    args = ap.parse_args()
    ptg = 13387 + args.dither
    frame = glob.glob(str(BASE / f"products/3025/frames/EUC_SIR_W-SCIFRM_BKGSUB_3025_{ptg}_{args.dither}_*.fits.gz"))[0]
    lt = h5py.File(sorted(glob.glob(str(BASE / f"products/3025/loctabs/EUC_SIR_W-LOCTAB_3025-{args.dither}-DET_{args.det}_*.h5")))[-1], "r")
    geo = FrameGeometry.from_frame(frame, CALIB).set_yshifts(dict(lt["Yshifts"].attrs))
    hd = fits.open(frame)
    sci = hd[f"DET{args.det}.SCI"].data
    lo = lt["Location Objects"]
    rows = []
    for i in lo.keys():
        o = lo[i]
        if "Order_1" not in o:
            continue
        ao = o["Astronomical Object"]
        mags = {r["Band"].decode() if isinstance(r["Band"], bytes) else str(r["Band"]): r["Value"] for r in np.array(ao["Magnitudes"])}
        rows.append((mags.get("H", 99.0), ao.attrs["RA"], ao.attrs["Dec"], int(ao.attrs["Type"]), i))
    rows.sort()
    sel = rows[: args.nbright]
    lam = np.linspace(11900, 19000, 40)
    fig, ax = plt.subplots(figsize=(14, 14))
    v = np.nanpercentile(sci, [5, 99.5])
    ax.imshow(sci, origin="lower", cmap="gray", vmin=v[0], vmax=v[1], interpolation="nearest")
    for h, ra, dec, typ, i in sel:
        for order, col in ((1, "cyan"), (0, "orange")):
            _, pix, sig = geo.trace_pix(ra, dec, lam if order == 1 else np.linspace(11900, 19000, 10), order, det_id=args.det)
            p = pix[0]
            ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=0.8)
            if order == 1:
                ax.text(p[0, 0], p[0, 1] + 15, f"H={h:.1f}", color="yellow", fontsize=7)
    ax.set_xlim(-10, sci.shape[1] + 10); ax.set_ylim(-10, sci.shape[0] + 10)
    ax.set_title(f"dither {args.dither} {geo.grism} tilt {geo.tilt} DET{args.det}: cyan=1st order, orange=0th order")
    out = args.out.format(dither=args.dither, det=args.det)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)
    # zoomed cutout around the brightest object's 1st order
    h, ra, dec, typ, i = sel[0]
    _, pix, sig = geo.trace_pix(ra, dec, lam, 1, det_id=args.det)
    p = pix[0]
    x0, x1 = int(np.nanmin(p[:, 0])) - 40, int(np.nanmax(p[:, 0])) + 40
    y0, y1 = int(np.nanmin(p[:, 1])) - 40, int(np.nanmax(p[:, 1])) + 40
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, sci.shape[1]), min(y1, sci.shape[0])
    cut = sci[y0:y1, x0:x1]
    fig, ax = plt.subplots(figsize=(16, 4))
    v = np.nanpercentile(cut, [5, 99.8])
    ax.imshow(cut, origin="lower", cmap="gray", vmin=v[0], vmax=v[1], extent=[x0, x1, y0, y1], aspect="auto")
    ax.plot(p[:, 0], p[:, 1], "c-", lw=0.8)
    ax.plot(p[:, 0], p[:, 1] + 3 * sig[0], "c:", lw=0.6); ax.plot(p[:, 0], p[:, 1] - 3 * sig[0], "c:", lw=0.6)
    for lam_mark in (12500, 14000, 15040.34, 16500, 18000):
        _, pm, _ = geo.trace_pix(ra, dec, [lam_mark], 1, det_id=args.det)
        ax.axvline(pm[0, 0, 0], color="m", lw=0.5); ax.text(pm[0, 0, 0], y1 - 8, f"{lam_mark:.0f}", color="m", fontsize=7)
    ax.set_title(f"brightest object H={h:.2f}, type {typ}: predicted 1st-order trace, dotted = +/-3 sigma(PSF model)")
    out2 = out.replace(".png", "_zoom.png")
    fig.savefig(out2, dpi=110, bbox_inches="tight")
    print("wrote", out2)


if __name__ == "__main__":
    main()
