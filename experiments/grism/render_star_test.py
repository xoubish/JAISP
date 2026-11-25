"""First renderer test: forward-model bright isolated objects on DET23 and compare with the frame.

For each selected object: take its MER J-band stamp (0.1"/px) from the JAISP Euclid tiles,
fit a free 1D spectrum (non-negative least squares on the lambda grid, regularised) with the
renderer, and compare the rendered trace with the data: residual image, cross-dispersion
profile, and the extracted spectrum shape.
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
from astropy.io import fits
from astropy.wcs import WCS

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.grism.sir_geometry import FrameGeometry  # noqa: E402
from models.grism.renderer import SourceTrace, render_source, lambda_grid, kernel_bank_from_table  # noqa: E402

BASE = REPO / "data" / "sir_edff"
CALIB = BASE / "calib" / "dr1_3025"
TILES = sorted(glob.glob(str(REPO / "data/euclid_tiles_all/tile_*_euclid.npz")))


class TileIndex:
    def __init__(self):
        self.files = TILES
        self.ra = np.zeros(len(TILES)); self.dec = np.zeros(len(TILES))
        for i, f in enumerate(TILES):
            d = np.load(f, allow_pickle=True)
            self.ra[i] = float(d["ra_center"]); self.dec[i] = float(d["dec_center"])

    def stamp(self, ra, dec, band="J", half_arcsec=4.0):
        """Return (stamp, wcs) of half-size `half_arcsec` around (ra, dec) from the best tile."""
        d2 = ((self.ra - ra) * np.cos(np.deg2rad(dec))) ** 2 + (self.dec - dec) ** 2
        for i in np.argsort(d2)[:4]:
            d = np.load(self.files[i], allow_pickle=True)
            w = WCS(fits.Header.fromstring(str(d[f"wcs_{band}"].item()))) if hasattr(d[f"wcs_{band}"], "item") else None
            img = d[f"img_{band}"]
            x, y = w.world_to_pixel_values(ra, dec)
            x, y = float(x), float(y)
            h = int(round(half_arcsec / float(d[f"pixel_scale_{band}"])))
            x0, y0 = int(round(x)) - h, int(round(y)) - h
            if x0 < 0 or y0 < 0 or x0 + 2 * h + 1 > img.shape[1] or y0 + 2 * h + 1 > img.shape[0]:
                continue
            cut = np.array(img[y0:y0 + 2 * h + 1, x0:x0 + 2 * h + 1], float)
            wc = w.deepcopy(); wc.wcs.crpix = wc.wcs.crpix - np.array([x0, y0])
            return cut, wc
        return None, None


def fit_spectrum(data, var, stamp, trace, lo, hi, nlam, iters=400, lr=0.1, smooth=1e-2, device="cpu",
                 fit_shift=True, fit_blur=False, blur_init_px=1.0, kernels=None):
    """Fit a free spectrum, optionally a global (dx, dy) trace offset [det px] and a Gaussian
    blur sigma [stamp px] applied to the morphology stamp."""
    st = torch.as_tensor(stamp / stamp.sum(), dtype=torch.float32, device=device)
    dat = torch.as_tensor(data, dtype=torch.float32, device=device)
    w = torch.as_tensor(1.0 / np.clip(var, 1e-3, None), dtype=torch.float32, device=device)
    logf = torch.full((nlam,), float(np.log(max(dat.sum().item(), 1.0) / nlam)), device=device, requires_grad=True)
    shift = torch.zeros(2, device=device, requires_grad=fit_shift)
    log_blur = torch.tensor(float(np.log(blur_init_px)), device=device, requires_grad=fit_blur)
    # background: constant + linear gradient across the box (BKGSUB frames keep a residual level)
    bkg = torch.zeros(3, device=device, requires_grad=True)
    H, W = dat.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device), indexing="ij")
    params = [logf, bkg] + ([shift] if fit_shift else []) + ([log_blur] if fit_blur else [])
    opt = torch.optim.Adam(params, lr=lr)
    blur = (lambda: torch.exp(log_blur)) if (fit_blur or blur_init_px > 0.05) else (lambda: None)
    clip = torch.ones_like(dat)
    for it in range(iters):
        opt.zero_grad()
        f = torch.exp(logf)
        model = render_source(st, f, trace, lo, hi, extra_shift=shift, blur_sigma_px=blur(), kernels=kernels) + bkg[0] + bkg[1] * xx + bkg[2] * yy
        chi2 = (clip * w * (model - dat) ** 2).mean()
        reg = smooth * ((logf[1:] - logf[:-1]) ** 2).mean()
        (chi2 + reg).backward(); opt.step()
        if it in (iters // 3, 2 * iters // 3):   # clip unflagged hot pixels / cosmic rays away from the trace core
            with torch.no_grad():
                chi = (dat - model) * torch.sqrt(w)
                src = model - (bkg[0] + bkg[1] * xx + bkg[2] * yy)
                offcore = src < 0.1 * src.max()
                clip = (~((chi > 8) & offcore)).to(dat.dtype)
    with torch.no_grad():
        f = torch.exp(logf)
        model = render_source(st, f, trace, lo, hi, extra_shift=shift, blur_sigma_px=blur(), kernels=kernels) + bkg[0] + bkg[1] * xx + bkg[2] * yy
    fit_spectrum.last_blur = float(torch.exp(log_blur)) if fit_blur else None
    fit_spectrum.last_bkg = bkg.detach().cpu().numpy()
    fit_spectrum.last_clip_frac = float(1 - clip.mean())
    return f.cpu().numpy(), model.cpu().numpy(), shift.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dither", type=int, default=0)
    ap.add_argument("--det", default="23")
    ap.add_argument("--nobj", type=int, default=4)
    ap.add_argument("--hmax", type=float, default=19.0)
    ap.add_argument("--hmin", type=float, default=17.0)
    ap.add_argument("--amax", type=float, default=0.8, help="max MER major axis [arcsec] (point-like)")
    ap.add_argument("--band", default="J")
    ap.add_argument("--fit-blur", action="store_true", help="fit a Gaussian blur of the stamp (stamp px)")
    ap.add_argument("--blur", type=float, default=0.0, help="fixed Gaussian blur sigma of the stamp [stamp px]")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--kernel", default="", help="npz table of per-lambda two-Gaussian NISP-S kernel (stamp px)")
    args = ap.parse_args()
    ptg = 13387 + args.dither
    frame = glob.glob(str(BASE / f"products/3025/frames/EUC_SIR_W-SCIFRM_BKGSUB_3025_{ptg}_{args.dither}_*.fits.gz"))[0]
    lt = h5py.File(sorted(glob.glob(str(BASE / f"products/3025/loctabs/EUC_SIR_W-LOCTAB_3025-{args.dither}-DET_{args.det}_*.h5")))[-1], "r")
    geo = FrameGeometry.from_frame(frame, CALIB).set_yshifts(dict(lt["Yshifts"].attrs))
    hd = fits.open(frame)
    sci = np.array(hd[f"DET{args.det}.SCI"].data, float); var = np.array(hd[f"DET{args.det}.VAR"].data, float)
    dq = np.array(hd[f"DET{args.det}.DQ"].data)
    print("DQ nonzero fraction", np.mean(dq != 0), "VAR median", np.nanmedian(var), "SCI nan fraction", np.mean(~np.isfinite(sci)))
    bad = (dq != 0) | ~np.isfinite(var) | ~np.isfinite(sci) | (var <= 0)
    var = np.where(bad, 1e12, var); sci = np.where(np.isfinite(sci), sci, 0.0)
    tiles = TileIndex()
    lo_ = lt["Location Objects"]
    rows = []
    for i in lo_.keys():
        o = lo_[i]
        if "Order_1" not in o:
            continue
        ao = o["Astronomical Object"]
        mags = {(r["Band"].decode() if isinstance(r["Band"], bytes) else str(r["Band"])): r["Value"] for r in np.array(ao["Magnitudes"])}
        bb = np.array(o["Order_1/Bounding Rectangle"])
        if bb[0, 0] < 30 or bb[1, 0] > 2010 or bb[0, 1] < 30 or bb[1, 1] > 2010:
            continue   # keep traces fully inside the detector
        rows.append((mags.get("H", 99.0), ao.attrs["RA"], ao.attrs["Dec"], int(ao.attrs["Type"]), float(ao.attrs["Major axis"]), i))
    rows.sort()
    rows = [r for r in rows if args.hmin <= r[0] < args.hmax and r[3] == 1 and r[4] < args.amax][: args.nobj]
    lam = lambda_grid()
    kernels = kernel_bank_from_table(lam, args.kernel, device=args.device) if args.kernel else None
    fig, axes = plt.subplots(len(rows), 3, figsize=(20, 3.2 * len(rows)))
    for k, (h, ra, dec, typ, amaj, oid) in enumerate(rows):
        stamp, wcs = tiles.stamp(ra, dec, band=args.band, half_arcsec=3.0)
        if stamp is None:
            print("no stamp for", oid); continue
        stamp = np.clip(stamp - np.median(stamp), 0, None)
        tr = SourceTrace(geo, args.det, ra, dec, wcs, stamp.shape, lam, order=1)
        lo, hi = tr.bbox(stamp.shape, pad=6)
        lo = np.maximum(lo, 0); hi = np.minimum(hi, [sci.shape[1], sci.shape[0]])
        data = sci[lo[1]:hi[1], lo[0]:hi[0]]; v = var[lo[1]:hi[1], lo[0]:hi[0]]
        f, model, shift = fit_spectrum(data, v, stamp, tr, lo, hi, len(lam), fit_blur=args.fit_blur,
                                       blur_init_px=max(args.blur, 1e-3), device=args.device, kernels=kernels)
        resid = data - model
        chi = resid / np.sqrt(np.clip(v, 1e-3, None))
        # cross-dispersion profiles at the trace centre
        xc = int(round(tr.centre_px[0] + tr.shift_px[len(lam) // 2, 0])) - lo[0]
        prof_d = data[:, xc - 3:xc + 4].mean(1); prof_m = model[:, xc - 3:xc + 4].mean(1)
        blur_txt = f" blur sigma={fit_spectrum.last_blur:.2f} stamp px," if args.fit_blur else ""
        blur_txt += f" bkg={fit_spectrum.last_bkg[0]:.1f} e-, clipped {100 * fit_spectrum.last_clip_frac:.2f}%,"
        print(f"obj {oid} H={h:.2f} A={amaj:.2f}\" box {lo}-{hi}: fitted shift (dx,dy)={shift.round(2)} px,{blur_txt} "
              f"chi rms(in trace) {np.sqrt(np.mean(chi[model > 0.05 * model.max()] ** 2)):.2f}, "
              f"data/model flux ratio {data[model > 0.05 * model.max()].sum() / model[model > 0.05 * model.max()].sum():.3f}, "
              f"profile centroid data {np.sum(prof_d * np.arange(len(prof_d))) / prof_d.sum():.2f} model {np.sum(prof_m * np.arange(len(prof_m))) / prof_m.sum():.2f}")
        vmax = np.percentile(data, 99.5)
        ax = axes[k, 0]; ax.imshow(data, origin="lower", vmin=-vmax * 0.05, vmax=vmax, cmap="gray", aspect="auto"); ax.set_title(f"data {oid} H={h:.2f}")
        ax = axes[k, 1]; ax.imshow(model, origin="lower", vmin=-vmax * 0.05, vmax=vmax, cmap="gray", aspect="auto"); ax.set_title("model (free spectrum x J stamp)")
        ax = axes[k, 2]; ax.imshow(chi, origin="lower", vmin=-5, vmax=5, cmap="RdBu_r", aspect="auto"); ax.set_title("chi (data-model)/sigma")
    out = REPO / f"experiments/grism/render_test_d{args.dither}_det{args.det}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight"); print("wrote", out)
    # profiles + spectrum for the first object
    h, ra, dec, typ, amaj, oid = rows[0]
    stamp, wcs = tiles.stamp(ra, dec, band=args.band, half_arcsec=3.0); stamp = np.clip(stamp - np.median(stamp), 0, None)
    tr = SourceTrace(geo, args.det, ra, dec, wcs, stamp.shape, lam, order=1)
    lo, hi = tr.bbox(stamp.shape, pad=6); lo = np.maximum(lo, 0); hi = np.minimum(hi, [sci.shape[1], sci.shape[0]])
    data = sci[lo[1]:hi[1], lo[0]:hi[0]]; v = var[lo[1]:hi[1], lo[0]:hi[0]]
    f, model, shift = fit_spectrum(data, v, stamp, tr, lo, hi, len(lam), blur_init_px=max(args.blur, 1e-3), device=args.device, kernels=kernels)
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))
    xc = int(round(tr.centre_px[0] + tr.shift_px[len(lam) // 2, 0])) - lo[0]
    axes[0].plot(data[:, xc - 5:xc + 6].mean(1), "k-", label="data"); axes[0].plot(model[:, xc - 5:xc + 6].mean(1), "r--", label="model"); axes[0].legend(); axes[0].set_title("cross-dispersion profile at trace centre")
    axes[1].plot(data.sum(0), "k-", label="data column sums"); axes[1].plot(model.sum(0), "r--", label="model"); axes[1].legend(); axes[1].set_title("along dispersion")
    axes[2].plot(lam, f, "b-"); axes[2].set_title("fitted free spectrum (e- per stamp-flux unit)"); axes[2].set_xlabel("lambda [A]")
    out2 = REPO / f"experiments/grism/render_test_d{args.dither}_det{args.det}_profiles.png"
    fig.savefig(out2, dpi=100, bbox_inches="tight"); print("wrote", out2)


if __name__ == "__main__":
    main()
