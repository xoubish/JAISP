"""Empirical NISP-S cross-dispersion PSF versus wavelength from stars.

For each frame/detector: select stars (SIR Type 1) with hmin < H < hmax whose 1st-order
trace lies inside the detector.  For each star and wavelength bin, straighten the trace
(shift every column by its predicted fractional y), co-add the columns in the bin,
estimate the background from |y| > 5 px, fit a Gaussian centroid, then stack the
background-subtracted, centroid-registered, core-normalised profiles over stars with
S/N weights.  Outputs lambda, FWHM, wing fraction and two-Gaussian fit per bin, the
oversampled stacked profiles (npz) and a figure.
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
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.grism.sir_geometry import FrameGeometry  # noqa: E402

BASE = REPO / "data" / "sir_edff"
CALIB = BASE / "calib" / "dr1_3025"


def gauss1(y, a, s, y0, c):
    return a * np.exp(-0.5 * ((y - y0) / s) ** 2) + c


def gauss2(y, a1, s1, a2, s2):
    return a1 * np.exp(-0.5 * (y / s1) ** 2) + a2 * np.exp(-0.5 * (y / s2) ** 2)


def fwhm_of(ygrid, prof):
    pk = prof.max(); above = ygrid[prof > pk / 2]
    return above.max() - above.min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dithers", default="0,1,2,3")
    ap.add_argument("--dets", default="23,22,33,13")
    ap.add_argument("--hmin", type=float, default=16.8)
    ap.add_argument("--hmax", type=float, default=19.5)
    ap.add_argument("--nlam", type=int, default=10)
    ap.add_argument("--out", default=str(REPO / "experiments/grism/nisps_psf_empirical"))
    args = ap.parse_args()
    lam_edges = np.linspace(12100, 18700, args.nlam + 1)
    lam_c = 0.5 * (lam_edges[1:] + lam_edges[:-1])
    half = 9; ov = 4
    ygrid = np.arange(-7 * ov, 7 * ov + 1) / ov
    stack = np.zeros((args.nlam, len(ygrid))); wsum = np.zeros(args.nlam); nst = np.zeros(args.nlam, int)
    per_star = []
    yrel_fine = np.arange(-half, half + 1, 1.0)
    for dither in [int(x) for x in args.dithers.split(",")]:
        ptg = 13387 + dither
        frame = glob.glob(str(BASE / f"products/3025/frames/EUC_SIR_W-SCIFRM_BKGSUB_3025_{ptg}_{dither}_*.fits.gz"))[0]
        hd = fits.open(frame)
        for det in args.dets.split(","):
            lt = h5py.File(sorted(glob.glob(str(BASE / f"products/3025/loctabs/EUC_SIR_W-LOCTAB_3025-{dither}-DET_{det}_*.h5")))[-1], "r")
            geo = FrameGeometry.from_frame(frame, CALIB).set_yshifts(dict(lt["Yshifts"].attrs))
            sci = np.array(hd[f"DET{det}.SCI"].data, float); dq = np.array(hd[f"DET{det}.DQ"].data)
            sci = np.where((dq != 0) | ~np.isfinite(sci), np.nan, sci)
            lo_ = lt["Location Objects"]
            stars = []
            for i in lo_.keys():
                o = lo_[i]
                if "Order_1" not in o:
                    continue
                ao = o["Astronomical Object"]
                if int(ao.attrs["Type"]) != 1:
                    continue
                mags = {(r["Band"].decode() if isinstance(r["Band"], bytes) else str(r["Band"])): r["Value"] for r in np.array(ao["Magnitudes"])}
                h = mags.get("H", 99.0)
                bb = np.array(o["Order_1/Bounding Rectangle"])
                if not (args.hmin <= h < args.hmax) or bb[0, 0] < 20 or bb[1, 0] > 2020 or bb[0, 1] < 20 or bb[1, 1] > 2020:
                    continue
                stars.append((h, float(ao.attrs["RA"]), float(ao.attrs["Dec"]), i))
            nused = 0
            for h, ra, dec, oid in stars:
                lam = np.arange(12100, 18700, 13.4)
                _, pix, _ = geo.trace_pix(ra, dec, lam, 1, det_id=det)
                p = pix[0]
                star_rows = []
                for b in range(args.nlam):
                    sel = np.where((lam >= lam_edges[b]) & (lam < lam_edges[b + 1]))[0]
                    cols = []
                    for k in sel:
                        x = int(round(p[k, 0])); yc = p[k, 1]; y0 = int(np.floor(yc)) - half - 1
                        if x < 1 or x > 2038 or y0 < 0 or y0 + 2 * half + 3 > 2040:
                            continue
                        col = sci[y0:y0 + 2 * half + 3, x]
                        if np.isnan(col).sum() > 2:
                            continue
                        yy = np.arange(len(col)) + y0 - yc          # pixel centre relative to predicted trace
                        good = np.isfinite(col)
                        cols.append(np.interp(yrel_fine, yy[good], col[good]))
                    if len(cols) < 0.6 * len(sel):
                        continue
                    prof = np.median(np.array(cols), axis=0)
                    bkg = np.median(prof[np.abs(yrel_fine) > 5]); prof = prof - bkg
                    noise = 1.4826 * np.median(np.abs(prof[np.abs(yrel_fine) > 5] - np.median(prof[np.abs(yrel_fine) > 5]))) + 1e-6
                    try:
                        popt, _ = curve_fit(gauss1, yrel_fine, prof, p0=[prof.max(), 0.7, 0.0, 0.0], maxfev=5000)
                    except Exception:
                        continue
                    amp, sig, yc0 = popt[0], abs(popt[1]), popt[2]
                    snr = amp / noise
                    if snr < 8 or abs(yc0) > 1.5 or sig > 2.5:
                        continue
                    # register and resample to the oversampled grid
                    reg = np.interp(ygrid, yrel_fine - yc0, prof, left=np.nan, right=np.nan)
                    core = np.nansum(reg[np.abs(ygrid) <= 2]) / ov
                    if core <= 0:
                        continue
                    reg = reg / core
                    ok = np.isfinite(reg)
                    w = min(snr, 60.0) ** 2
                    stack[b, ok] += w * reg[ok]; wsum[b] += w; nst[b] += 1
                    star_rows.append((b, yc0, sig, snr))
                if star_rows:
                    nused += 1
                    per_star.append((dither, det, oid, h, float(np.median([r[1] for r in star_rows])), float(np.median([r[2] for r in star_rows]))))
            print(f"dither {dither} DET{det}: {len(stars)} candidate stars, {nused} used")
    stack /= np.maximum(wsum, 1e-9)[:, None]
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    for b in range(args.nlam):
        prof = stack[b]
        if nst[b] == 0:
            continue
        fw = fwhm_of(ygrid, prof)
        core = prof[np.abs(ygrid) <= 1.5].sum() / ov; wing = prof[(np.abs(ygrid) > 1.5) & (np.abs(ygrid) <= 5)].sum() / ov
        try:
            popt, _ = curve_fit(gauss2, ygrid, prof, p0=[prof.max(), 0.6, 0.05 * prof.max(), 2.0],
                                bounds=([0, 0.2, 0, 0.8], [np.inf, 2.0, np.inf, 6.0]), maxfev=20000)
            a1, s1, a2, s2 = popt; frac2 = a2 * s2 / (a1 * s1 + a2 * s2)
        except Exception:
            s1 = s2 = frac2 = np.nan
        rows.append((lam_c[b], nst[b], fw, wing / core, s1, s2, frac2))
        axes[0].plot(ygrid, prof, label=f"{lam_c[b]:.0f} A (n={nst[b]})")
        axes[1].semilogy(ygrid, np.clip(prof, 1e-4, None), label=f"{lam_c[b]:.0f}")
    axes[0].set_xlim(-5, 5); axes[0].set_title("stacked cross-dispersion profile (det px, core-normalised)"); axes[0].legend(fontsize=7)
    axes[1].set_xlim(-7, 7); axes[1].set_ylim(1e-3, 1); axes[1].set_title("log scale"); axes[1].grid(alpha=0.3)
    rows = np.array(rows)
    axes[2].plot(rows[:, 0], rows[:, 2], "o-", label="FWHM [px]"); axes[2].plot(rows[:, 0], rows[:, 4] * 2.355, "s--", label="core Gaussian FWHM")
    axes[2].plot(rows[:, 0], rows[:, 5] * 2.355 / 5, "d--", label="wing Gaussian FWHM / 5")
    axes[2].plot(rows[:, 0], rows[:, 3], "^-", label="wing/core flux (1.5<|y|<5 / |y|<1.5)"); axes[2].plot(rows[:, 0], rows[:, 6], "v-", label="2-Gauss wing flux fraction")
    axes[2].legend(fontsize=8); axes[2].set_xlabel("lambda [A]"); axes[2].grid(alpha=0.3)
    fig.savefig(args.out + ".png", dpi=100, bbox_inches="tight")
    np.savez(args.out + ".npz", lam=lam_c, ygrid=ygrid, profile=stack, n=nst, table=rows, per_star=np.array(per_star, dtype=object))
    print("\n lambda    n   FWHM  wing/core  sig_core  sig_wing  frac_wing")
    for r in rows:
        print(f" {r[0]:7.0f} {int(r[1]):4d}  {r[2]:5.2f}   {r[3]:6.3f}    {r[4]:5.2f}    {r[5]:5.2f}    {r[6]:5.3f}")
    ds = np.array([r[4] for r in per_star]); ss = np.array([r[5] for r in per_star])
    print(f"\nper-star y offset vs predicted trace: median {np.median(ds):+.3f} px, MAD {np.median(np.abs(ds - np.median(ds))):.3f} px; "
          f"per-star Gaussian sigma median {np.median(ss):.3f} px (FWHM {2.355 * np.median(ss):.2f}); n={len(ds)}")
    print("wrote", args.out + ".png / .npz")


if __name__ == "__main__":
    main()
