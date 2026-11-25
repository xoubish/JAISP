"""Fit the NISP-S PSF kernel by forward modelling many stars jointly.

Shared parameters: two-Gaussian kernel (sigma1, sigma2, f2) in MER px, optionally per
wavelength group.  Per star: free spectrum (log), background plane, (dx, dy) offset.
VIS stamps as morphology.  Writes experiments/grism/nisps_kernel_forward.npz in the same
format as nisps_kernel.npz (usable with render_star_test.py --kernel).
"""
import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/grism"))
from models.grism.sir_geometry import FrameGeometry  # noqa: E402
from models.grism.renderer import SourceTrace, render_source, lambda_grid, two_gauss_kernel  # noqa: E402
from render_star_test import TileIndex, BASE, CALIB  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dithers", default="0,3")
    ap.add_argument("--det", default="23")
    ap.add_argument("--nstars", type=int, default=12, help="per dither")
    ap.add_argument("--hmin", type=float, default=17.0)
    ap.add_argument("--hmax", type=float, default=18.5)
    ap.add_argument("--ngroups", type=int, default=3, help="wavelength groups for the kernel")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    lam = lambda_grid(); L = len(lam)
    edges = np.linspace(lam[0], lam[-1] + 1, args.ngroups + 1)
    gidx = torch.as_tensor(np.clip(np.searchsorted(edges, lam) - 1, 0, args.ngroups - 1), device=dev)
    tiles = TileIndex()
    stars = []
    for dither in [int(x) for x in args.dithers.split(",")]:
        ptg = 13387 + dither
        frame = glob.glob(str(BASE / f"products/3025/frames/EUC_SIR_W-SCIFRM_BKGSUB_3025_{ptg}_{dither}_*.fits.gz"))[0]
        lt = h5py.File(sorted(glob.glob(str(BASE / f"products/3025/loctabs/EUC_SIR_W-LOCTAB_3025-{dither}-DET_{args.det}_*.h5")))[-1], "r")
        geo = FrameGeometry.from_frame(frame, CALIB).set_yshifts(dict(lt["Yshifts"].attrs))
        hd = fits.open(frame)
        sci = np.array(hd[f"DET{args.det}.SCI"].data, float); var = np.array(hd[f"DET{args.det}.VAR"].data, float)
        dq = np.array(hd[f"DET{args.det}.DQ"].data)
        bad = (dq != 0) | ~np.isfinite(var) | ~np.isfinite(sci) | (var <= 0)
        var = np.where(bad, 1e12, var); sci = np.nan_to_num(sci)
        rows = []
        for i in lt["Location Objects"].keys():
            o = lt["Location Objects"][i]
            if "Order_1" not in o:
                continue
            ao = o["Astronomical Object"]
            if int(ao.attrs["Type"]) != 1 or float(ao.attrs["Major axis"]) > 0.6:
                continue
            mags = {(r["Band"].decode() if isinstance(r["Band"], bytes) else str(r["Band"])): r["Value"] for r in np.array(ao["Magnitudes"])}
            h = mags.get("H", 99.0); bb = np.array(o["Order_1/Bounding Rectangle"])
            if not (args.hmin <= h < args.hmax) or bb[0, 0] < 30 or bb[1, 0] > 2010 or bb[0, 1] < 30 or bb[1, 1] > 2010:
                continue
            rows.append((h, float(ao.attrs["RA"]), float(ao.attrs["Dec"]), i))
        rows.sort()
        for h, ra, dec, oid in rows[: args.nstars]:
            stamp, wcs = tiles.stamp(ra, dec, band="VIS", half_arcsec=3.0)
            if stamp is None:
                continue
            stamp = np.clip(np.nan_to_num(stamp) - np.median(stamp), 0, None); stamp /= stamp.sum()
            tr = SourceTrace(geo, args.det, ra, dec, wcs, stamp.shape, lam, order=1)
            lo, hi = tr.bbox(stamp.shape, pad=6); lo = np.maximum(lo, 0); hi = np.minimum(hi, [2040, 2040])
            d = sci[lo[1]:hi[1], lo[0]:hi[0]]; v = var[lo[1]:hi[1], lo[0]:hi[0]]
            stars.append(dict(oid=oid, h=h, tr=tr, lo=lo, hi=hi,
                              st=torch.tensor(stamp, dtype=torch.float32, device=dev),
                              dat=torch.tensor(d, dtype=torch.float32, device=dev),
                              w=torch.tensor(1 / np.clip(v, 1e-3, None), dtype=torch.float32, device=dev)))
    print(f"{len(stars)} stars")
    # parameters
    log_s1 = torch.full((args.ngroups,), float(np.log(2.0)), device=dev, requires_grad=True)
    log_s2 = torch.full((args.ngroups,), float(np.log(7.0)), device=dev, requires_grad=True)
    logit_f2 = torch.full((args.ngroups,), float(np.log(0.2 / 0.8)), device=dev, requires_grad=True)
    per = []
    for s in stars:
        H, W = s["dat"].shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
        s["xx"], s["yy"] = xx, yy
        s["logf"] = torch.full((L,), float(np.log(max(s["dat"].sum().item(), 1.0) / L)), device=dev, requires_grad=True)
        s["bkg"] = torch.zeros(3, device=dev, requires_grad=True)
        s["shift"] = torch.zeros(2, device=dev, requires_grad=True)
        per += [s["logf"], s["bkg"], s["shift"]]
    opt = torch.optim.Adam([{"params": [log_s1, log_s2, logit_f2], "lr": 0.03}, {"params": per, "lr": 0.1}])

    def bank():
        ks = [two_gauss_kernel(torch.exp(log_s1[g]), torch.exp(log_s2[g]), torch.sigmoid(logit_f2[g]), radius=24, device=dev)
              for g in range(args.ngroups)]
        return torch.stack(ks)[:, None]

    for it in range(args.iters):
        opt.zero_grad()
        kb = bank(); total = 0.0
        for s in stars:
            f = torch.exp(s["logf"])
            m = render_source(s["st"], f, s["tr"], s["lo"], s["hi"], extra_shift=s["shift"], kernels=(kb, gidx))
            m = m + s["bkg"][0] + s["bkg"][1] * s["xx"] + s["bkg"][2] * s["yy"]
            chi2 = (s["w"] * (m - s["dat"]) ** 2).mean()
            reg = 1e-2 * ((s["logf"][1:] - s["logf"][:-1]) ** 2).mean()
            total = total + chi2 + reg
        total.backward(); opt.step()
        if it % 100 == 0 or it == args.iters - 1:
            with torch.no_grad():
                s1 = torch.exp(log_s1).cpu().numpy(); s2 = torch.exp(log_s2).cpu().numpy(); f2 = torch.sigmoid(logit_f2).cpu().numpy()
            print(f"it {it}: loss {total.item() / len(stars):.3f}  sigma1 {np.round(s1, 2)}  sigma2 {np.round(s2, 2)}  f2 {np.round(f2, 3)}")
    # per-star chi in trace
    with torch.no_grad():
        kb = bank(); chis = []
        for s in stars:
            m = render_source(s["st"], torch.exp(s["logf"]), s["tr"], s["lo"], s["hi"], extra_shift=s["shift"], kernels=(kb, gidx))
            core = m > 0.05 * m.max()
            m = m + s["bkg"][0] + s["bkg"][1] * s["xx"] + s["bkg"][2] * s["yy"]
            chi = (s["dat"] - m) * torch.sqrt(s["w"])
            chis.append(float(torch.sqrt((chi[core] ** 2).mean())))
        print("per-star chi rms in trace:", np.round(chis, 2), " median", np.median(chis))
        s1 = torch.exp(log_s1).cpu().numpy(); s2 = torch.exp(log_s2).cpu().numpy(); f2 = torch.sigmoid(logit_f2).cpu().numpy()
    lam_g = 0.5 * (edges[1:] + edges[:-1])
    np.savez(REPO / "experiments/grism/nisps_kernel_forward.npz", lam=lam_g, sigma1=s1, sigma2=s2, f2=f2, unit="MER px (0.1 arcsec)")
    for g in range(args.ngroups):
        print(f"group {g} ({edges[g]:.0f}-{edges[g + 1]:.0f} A): core FWHM {0.1 * 2.355 * s1[g]:.3f}\", wing FWHM {0.1 * 2.355 * s2[g]:.2f}\", wing frac {f2[g]:.3f}")
    print("wrote experiments/grism/nisps_kernel_forward.npz")


if __name__ == "__main__":
    main()
