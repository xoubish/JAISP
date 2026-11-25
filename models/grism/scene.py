"""Multi-source grism scene: render all catalogued sources into a detector region, for one or
several frames (dithers / grism settings), with per-source spectra shared across frames.

This is the "SIR-equivalent baseline inside our operator": morphology from the VIS mosaic
(convolved with the NISP-S kernel), one free spectrum per source (0th and 1st order), a
smooth per-frame sensitivity ratio, and a background plane per frame region.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.utils.checkpoint
from astropy.io import fits
from astropy.wcs import WCS

from .sir_geometry import FrameGeometry
from .renderer import SourceTrace, render_source, render_source_windowed, lambda_grid, kernel_bank_from_table, two_gauss_kernel


# --------------------------------------------------------------------------- #
# data access
# --------------------------------------------------------------------------- #
class TileStamps:
    """Cutouts from the JAISP Euclid tiles (0.1"/px VIS/Y/J/H, 105" tiles with 50% overlap)."""

    def __init__(self, tile_glob):
        self.files = sorted(glob.glob(tile_glob))
        self.ra = np.zeros(len(self.files)); self.dec = np.zeros(len(self.files))
        for i, f in enumerate(self.files):
            d = np.load(f, allow_pickle=True)
            self.ra[i] = float(d["ra_center"]); self.dec[i] = float(d["dec_center"])
        self._cache = {}

    def _load(self, i):
        if i not in self._cache:
            d = np.load(self.files[i], allow_pickle=True)
            bands = [b for b in ("VIS", "Y", "J", "H") if f"img_{b}" in d.files and f"wcs_{b}" in d.files]
            self._cache[i] = {f"img_{b}": d[f"img_{b}"] for b in bands}
            self._cache[i]["wcs"] = {b: WCS(fits.Header.fromstring(str(d[f"wcs_{b}"].item()))) for b in bands}
            self._cache[i]["scale"] = {b: float(d[f"pixel_scale_{b}"]) for b in bands}
        return self._cache[i]

    def stamp(self, ra, dec, band="VIS", half_arcsec=3.0):
        d2 = ((self.ra - ra) * np.cos(np.deg2rad(dec))) ** 2 + (self.dec - dec) ** 2
        for i in np.argsort(d2)[:4]:
            t = self._load(i)
            if band not in t["wcs"]:
                continue
            w = t["wcs"][band]; img = t[f"img_{band}"]
            x, y = w.world_to_pixel_values(ra, dec); x, y = float(x), float(y)
            h = int(round(half_arcsec / t["scale"][band]))
            x0, y0 = int(round(x)) - h, int(round(y)) - h
            if x0 < 0 or y0 < 0 or x0 + 2 * h + 1 > img.shape[1] or y0 + 2 * h + 1 > img.shape[0]:
                continue
            cut = np.array(img[y0:y0 + 2 * h + 1, x0:x0 + 2 * h + 1], float)
            wc = w.deepcopy(); wc.wcs.crpix = wc.wcs.crpix - np.array([x0, y0])
            return cut, wc
        return None, None


def load_frame(base, dither, det, calib):
    ptg = 13387 + dither
    frame = glob.glob(str(Path(base) / f"products/3025/frames/EUC_SIR_W-SCIFRM_BKGSUB_3025_{ptg}_{dither}_*.fits.gz"))[0]
    lt = h5py.File(sorted(glob.glob(str(Path(base) / f"products/3025/loctabs/EUC_SIR_W-LOCTAB_3025-{dither}-DET_{det}_*.h5")))[-1], "r")
    geo = FrameGeometry.from_frame(frame, calib).set_yshifts(dict(lt["Yshifts"].attrs))
    hd = fits.open(frame)
    sci = np.array(hd[f"DET{det}.SCI"].data, float); var = np.array(hd[f"DET{det}.VAR"].data, float)
    dq = np.array(hd[f"DET{det}.DQ"].data)
    bad = (dq != 0) | ~np.isfinite(var) | ~np.isfinite(sci) | (var <= 0)
    return dict(dither=dither, det=det, geo=geo, lt=lt, sci=np.nan_to_num(sci), var=np.where(bad, 1e12, var), bad=bad,
                grism=geo.grism, tilt=geo.tilt)


def catalog_from_loctab(lt):
    """All SIR location objects (MER sources passing SIR's selection) with their properties."""
    out = []
    lo = lt["Location Objects"]
    for i in lo.keys():
        o = lo[i]; ao = o["Astronomical Object"]
        mags = {(r["Band"].decode() if isinstance(r["Band"], bytes) else str(r["Band"])): float(r["Value"]) for r in np.array(ao["Magnitudes"])}
        bb1 = np.array(o["Order_1/Bounding Rectangle"]) if "Order_1" in o else None
        bb0 = np.array(o["Order_0/Bounding Rectangle"]) if "Order_0" in o else None
        out.append(dict(id=i, ra=float(ao.attrs["RA"]), dec=float(ao.attrs["Dec"]), type=int(ao.attrs["Type"]),
                        amaj=float(ao.attrs["Major axis"]), amin=float(ao.attrs["Minor axis"]), mags=mags, bb1=bb1, bb0=bb0))
    return out


def sources_touching(cat, box_lo, box_hi, pad=4):
    """Catalogue entries whose SIR 0th or 1st order bounding box intersects the pixel box."""
    sel = []
    for s in cat:
        hit = False
        for bb in (s["bb1"], s["bb0"]):
            if bb is None:
                continue
            if bb[1, 0] >= box_lo[0] - pad and bb[0, 0] <= box_hi[0] + pad and bb[1, 1] >= box_lo[1] - pad and bb[0, 1] <= box_hi[1] + pad:
                hit = True
        if hit:
            sel.append(s)
    return sel


# --------------------------------------------------------------------------- #
# scene model
# --------------------------------------------------------------------------- #
@dataclass
class SceneSource:
    id: str
    ra: float
    dec: float
    hmag: float
    stamp: torch.Tensor
    wcs: WCS
    traces: dict = field(default_factory=dict)   # (frame_key, order) -> SourceTrace
    logf1: torch.Tensor | None = None             # 1st-order spectrum (log)
    logf0: torch.Tensor | None = None             # 0th-order spectrum (log), coarse
    shift: torch.Tensor | None = None             # (dx, dy) per source, shared across frames (det px)


class GrismScene:
    """Scene over a detector box (box_lo, box_hi in detector px, same for all frames) or, with
    `targets` given (a sky selection: dict id -> catalogue row), over the whole detector with the
    loss restricted to pixels covered by the targets' traces in each frame; contaminants are all
    catalogued sources whose traces intersect those pixels."""

    def __init__(self, frames: list, det: str, box_lo, box_hi, tiles: TileStamps, kernel_npz=None,
                 device="cpu", lam=None, nlam0=8, stamp_half_arcsec=3.0, band="VIS", lam_step=13.4,
                 targets: dict | None = None, mask_pad=4, stamp_provider=None, star_psf=None):
        """stamp_provider: optional object with .stamp(ra, dec) -> (template, wcs) (e.g. FoundationStamps);
        used for the morphology when it returns a stamp, raw VIS cutouts otherwise.
        star_psf: optional MerPsf-like object with .stamp_at_sky(ra, dec) -> (psf, fwhm): stars (SIR Type 1)
        get the MER VIS PSF as morphology instead of their (possibly saturated) cutout."""
        self.device = device
        self.det = det
        self.box_lo = np.asarray(box_lo); self.box_hi = np.asarray(box_hi)
        self.frames = frames
        self.lam = lambda_grid(step=lam_step) if lam is None else np.asarray(lam)
        self.L = len(self.lam)
        self.lam0 = np.linspace(11900, 19000, nlam0)              # coarse grid for the 0th order
        self.kernels = kernel_bank_from_table(self.lam, kernel_npz, device=device) if kernel_npz else None
        self.kernels0 = kernel_bank_from_table(self.lam0, kernel_npz, device=device) if kernel_npz else None
        self.targets = targets
        # union of sources over frames
        H = int(self.box_hi[1] - self.box_lo[1]); W = int(self.box_hi[0] - self.box_lo[0])
        self.H, self.W = H, W
        self.sources = {}
        self.pixmask = {}
        for fr in frames:
            cat = catalog_from_loctab(fr["lt"])
            if targets is None:
                selected = sources_touching(cat, self.box_lo, self.box_hi)
            else:
                # pixel mask of this frame = union of the targets' SIR bounding boxes (both orders), padded
                m = np.zeros((H, W), bool)
                for s in cat:
                    if s["id"] not in targets:
                        continue
                    for bb in (s["bb1"], s["bb0"]):
                        if bb is None:
                            continue
                        x0 = int(max(bb[0, 0] - mask_pad - self.box_lo[0], 0)); x1 = int(min(bb[1, 0] + mask_pad - self.box_lo[0], W))
                        y0 = int(max(bb[0, 1] - mask_pad - self.box_lo[1], 0)); y1 = int(min(bb[1, 1] + mask_pad - self.box_lo[1], H))
                        if x1 > x0 and y1 > y0:
                            m[y0:y1 + 1, x0:x1 + 1] = True
                self.pixmask[fr["dither"]] = m
                selected = []
                for s in cat:   # contaminants: anything whose bbox intersects the mask
                    for bb in (s["bb1"], s["bb0"]):
                        if bb is None:
                            continue
                        x0 = int(max(bb[0, 0] - self.box_lo[0], 0)); x1 = int(min(bb[1, 0] - self.box_lo[0], W - 1))
                        y0 = int(max(bb[0, 1] - self.box_lo[1], 0)); y1 = int(min(bb[1, 1] - self.box_lo[1], H - 1))
                        if x1 >= x0 and y1 >= y0 and m[y0:y1 + 1, x0:x1 + 1].any():
                            selected.append(s); break
            for s in selected:
                if s["id"] in self.sources:
                    continue
                kind = "raw"
                stamp, wcs = (None, None)
                if star_psf is not None and s["type"] == 1:
                    psf, _ = star_psf.stamp_at_sky(s["ra"], s["dec"])
                    if psf is not None:
                        # PSF stamp centred on the source: build a WCS so that (ra, dec) is at the stamp centre
                        raw, wr = tiles.stamp(s["ra"], s["dec"], band=band, half_arcsec=stamp_half_arcsec)
                        if raw is not None:
                            h = (raw.shape[0] - 1) // 2
                            big = np.zeros_like(raw); r = psf.shape[0] // 2
                            x, y = wr.world_to_pixel_values(s["ra"], s["dec"]); cx, cy = int(round(float(x))), int(round(float(y)))
                            y0, x0 = cy - r, cx - r
                            if y0 >= 0 and x0 >= 0 and y0 + psf.shape[0] <= big.shape[0] and x0 + psf.shape[1] <= big.shape[1]:
                                big[y0:y0 + psf.shape[0], x0:x0 + psf.shape[1]] = psf
                                stamp, wcs, kind = big, wr, "psf"
                if stamp is None and stamp_provider is not None:
                    stamp, wcs = stamp_provider.stamp(s["ra"], s["dec"])
                    if stamp is not None:
                        kind = "foundation"
                        # crop the (12") template to the usual stamp size around the source
                        x, y = wcs.world_to_pixel_values(s["ra"], s["dec"]); x, y = float(x), float(y)
                        h = int(round(stamp_half_arcsec / 0.1))
                        x0, y0 = int(round(x)) - h, int(round(y)) - h
                        if x0 >= 0 and y0 >= 0 and x0 + 2 * h + 1 <= stamp.shape[1] and y0 + 2 * h + 1 <= stamp.shape[0]:
                            stamp = stamp[y0:y0 + 2 * h + 1, x0:x0 + 2 * h + 1]
                            wcs = wcs.deepcopy(); wcs.wcs.crpix = wcs.wcs.crpix - np.array([x0, y0])
                if stamp is None:
                    stamp, wcs = tiles.stamp(s["ra"], s["dec"], band=band, half_arcsec=stamp_half_arcsec)
                    if stamp is None:
                        continue
                    stamp = np.nan_to_num(stamp)
                    stamp = np.clip(stamp - np.median(stamp), 0, None)
                if not np.isfinite(stamp).all() or stamp.sum() <= 0:
                    continue
                stamp = stamp / stamp.sum()
                src = SceneSource(id=s["id"], ra=s["ra"], dec=s["dec"], hmag=s["mags"].get("H", 99.0),
                                  stamp=torch.tensor(np.ascontiguousarray(stamp), dtype=torch.float32, device=device), wcs=wcs)
                src.stamp_kind = kind
                self.sources[s["id"]] = src
        # traces per frame, and per-source coverage: in which frames does the 1st-order trace fall in the box
        for fr in frames:
            key = fr["dither"]
            for src in self.sources.values():
                for order, lam in ((1, self.lam), (0, self.lam0)):
                    try:
                        tr = SourceTrace(fr["geo"], det, src.ra, src.dec, src.wcs, src.stamp.shape, lam, order=order)
                    except Exception:
                        continue
                    lo, hi = tr.bbox(src.stamp.shape, pad=0)
                    lo = np.maximum(lo, self.box_lo); hi = np.minimum(hi, self.box_hi)
                    if (hi <= lo).any():
                        continue       # trace entirely outside the box in this frame
                    src.traces[(key, order)] = tr
        for src in self.sources.values():
            src.coverage = sorted({k for (k, o) in src.traces if o == 1})
            src.coverage0 = sorted({k for (k, o) in src.traces if o == 0})
        # data tensors
        for fr in frames:
            d = fr["sci"][self.box_lo[1]:self.box_hi[1], self.box_lo[0]:self.box_hi[0]]
            v = fr["var"][self.box_lo[1]:self.box_hi[1], self.box_lo[0]:self.box_hi[0]]
            fr["dat"] = torch.tensor(d, dtype=torch.float32, device=device)
            wv = 1 / np.clip(v, 1e-3, None)
            if fr["dither"] in self.pixmask:
                wv = np.where(self.pixmask[fr["dither"]], wv, 0.0)
            fr["w"] = torch.tensor(wv, dtype=torch.float32, device=device)
            fr["bkg"] = torch.zeros(3, device=device, requires_grad=True)
            fr["sens"] = torch.zeros(3, device=device, requires_grad=True)   # log sensitivity ratio, Chebyshev in lambda
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device), indexing="ij")
            fr["xx"], fr["yy"] = xx, yy
        self.lam_n = torch.tensor((2 * self.lam - self.lam[0] - self.lam[-1]) / (self.lam[-1] - self.lam[0]), dtype=torch.float32, device=device)
        self.lam0_n = torch.tensor((2 * self.lam0 - self.lam[0] - self.lam[-1]) / (self.lam[-1] - self.lam[0]), dtype=torch.float32, device=device)

    # ---- parameters ------------------------------------------------------ #
    def init_params(self, init_counts=None):
        """init_counts: per-source total e- guess for the 1st order (default from the data)."""
        tot = float(max(self.frames[0]["dat"].sum().item(), 1.0))
        for src in self.sources.values():
            flux_guess = 10 ** (-0.4 * (src.hmag - 20.0)) * 2e4 if np.isfinite(src.hmag) else 1e3
            src.logf1 = torch.full((self.L,), float(np.log(max(flux_guess / self.L, 1e-3))), device=self.device, requires_grad=True)
            src.logf0 = torch.full((len(self.lam0),), float(np.log(max(0.05 * flux_guess / len(self.lam0), 1e-3))), device=self.device, requires_grad=True)
            src.shift = torch.zeros(2, device=self.device, requires_grad=True)
        # photometric prior: log10(sum f1) ~ zp - 0.4 H, zp learned (one scalar), width mag_prior_sigma dex
        self.zp = torch.tensor(float(np.log10(2e4) + 0.4 * 20.0), device=self.device, requires_grad=True)
        if not hasattr(self, "kernel_params"):
            self.kernel_params = None   # set by enable_kernel_fit()
        self.mag_prior_sigma = 0.3
        self.shift_prior_px = 0.3   # prior width on per-source (dx, dy) trace offsets [det px]
        hm = np.array([s.hmag for s in self.sources.values()])
        self._h = torch.tensor(np.where(np.isfinite(hm) & (hm < 90), hm, np.nan), dtype=torch.float32, device=self.device)

    def enable_kernel_fit(self, table_npz, ngroups=3):
        """Make the two-Gaussian NISP-S kernel learnable (per wavelength group), initialised from a table."""
        t = np.load(table_npz)
        lam_t = np.asarray(t["lam"], float)
        edges = np.linspace(self.lam[0], self.lam[-1] + 1, ngroups + 1)
        gl = 0.5 * (edges[1:] + edges[:-1])
        s1 = np.interp(gl, lam_t, t["sigma1"]); s2 = np.interp(gl, lam_t, t["sigma2"]); f2 = np.interp(gl, lam_t, t["f2"])
        dev = self.device
        self.kernel_params = dict(
            log_s1=torch.tensor(np.log(s1), dtype=torch.float32, device=dev, requires_grad=True),
            log_s2=torch.tensor(np.log(s2), dtype=torch.float32, device=dev, requires_grad=True),
            logit_f2=torch.tensor(np.log(f2 / (1 - f2)), dtype=torch.float32, device=dev, requires_grad=True),
            edges=edges, lam_groups=gl)
        self.kidx = torch.as_tensor(np.clip(np.searchsorted(edges, self.lam) - 1, 0, ngroups - 1), device=dev)
        self.kidx0 = torch.as_tensor(np.clip(np.searchsorted(edges, self.lam0) - 1, 0, ngroups - 1), device=dev)
        return self

    def kernel_banks(self):
        """(bank, idx) for order 1 and order 0, from the learnable parameters (or the fixed tables)."""
        if self.kernel_params is None:
            return self.kernels, self.kernels0
        kp = self.kernel_params
        ks = [two_gauss_kernel(torch.exp(kp["log_s1"][g]), torch.exp(kp["log_s2"][g]), torch.sigmoid(kp["logit_f2"][g]),
                               radius=24, device=self.device) for g in range(len(kp["lam_groups"]))]
        bank = torch.stack(ks)[:, None]
        return (bank, self.kidx), (bank, self.kidx0)

    def kernel_table(self):
        kp = self.kernel_params
        return dict(lam=kp["lam_groups"], sigma1=torch.exp(kp["log_s1"]).detach().cpu().numpy(),
                    sigma2=torch.exp(kp["log_s2"]).detach().cpu().numpy(), f2=torch.sigmoid(kp["logit_f2"]).detach().cpu().numpy())

    def _reduce(self, x):
        # 'mean' (default, historical runs) dilutes a single source's violation by 1/N; 'sum' makes each
        # source's prior count on its own (per-source prior, the proper form)
        return x.sum() if getattr(self, "prior_reduce", "mean") == "sum" else x.mean()

    def mag_prior(self):
        logs = torch.stack([torch.log10(torch.exp(s.logf1).sum() + 1e-6) for s in self.sources.values()])
        ok = torch.isfinite(self._h)
        d = (logs - (self.zp - 0.4 * self._h))[ok] / self.mag_prior_sigma
        return self._reduce(d ** 2)

    def shift_prior(self):
        sh = torch.stack([s.shift for s in self.sources.values()])
        # saturated bright stars have MER positions off by up to ~1 px: looser prior for them
        width = torch.tensor([1.0 if (np.isfinite(s.hmag) and s.hmag < 17) else self.shift_prior_px for s in self.sources.values()],
                             device=sh.device, dtype=sh.dtype)
        return self._reduce(((sh / width[:, None]) ** 2).sum(1))

    def params(self, spectra=True, shifts=True, frames=True):
        p = [self.zp] if hasattr(self, "zp") else []
        if getattr(self, "kernel_params", None) is not None:
            p += [self.kernel_params["log_s1"], self.kernel_params["log_s2"], self.kernel_params["logit_f2"]]
        for src in self.sources.values():
            if spectra:
                p += [src.logf1, src.logf0]
            if shifts:
                p += [src.shift]
        if frames:
            for fr in self.frames:
                p += [fr["bkg"], fr["sens"]]
        return p

    # ---- rendering ------------------------------------------------------- #
    def sens_factor(self, fr, order=1):
        c = fr["sens"]; x = self.lam_n if order == 1 else self.lam0_n
        return torch.exp(c[0] + c[1] * x + c[2] * (2 * x * x - 1))

    def render_frame(self, fr, exclude=None, orders=(1, 0), with_bkg=True, only=None):
        key = fr["dither"]
        img = torch.zeros(self.H, self.W, device=self.device)
        s1 = self.sens_factor(fr, 1); s0 = self.sens_factor(fr, 0)
        kern1, kern0 = self.kernel_banks()
        for src in self.sources.values():
            if exclude is not None and src.id in exclude:
                continue
            if only is not None and src.id not in only:
                continue
            for order in orders:
                tr = src.traces.get((key, order))
                if tr is None:
                    continue
                lo, hi = tr.bbox(src.stamp.shape, pad=3)
                lo = np.maximum(lo, self.box_lo); hi = np.minimum(hi, self.box_hi)
                if (hi <= lo).any():
                    continue
                f = torch.exp(src.logf1) * s1 if order == 1 else torch.exp(src.logf0) * s0
                kern = kern1 if order == 1 else kern0
                sh = src.shift + fr["dxy"] if "dxy" in fr else src.shift     # optional per-frame global offset
                if torch.is_grad_enabled():
                    # recompute in backward instead of storing the (L,H,W) sampling tensors of every source
                    sub = torch.utils.checkpoint.checkpoint(
                        lambda st_, f_, sh_, kb_, tr=tr, lo=lo, hi=hi, ki=kern[1]: render_source_windowed(st_, f_, tr, lo, hi, extra_shift=sh_, kernels=(kb_, ki)),
                        src.stamp, f, sh, kern[0], use_reentrant=False)
                else:
                    sub = render_source_windowed(src.stamp, f, tr, lo, hi, extra_shift=sh, kernels=kern)
                y0, x0 = lo[1] - self.box_lo[1], lo[0] - self.box_lo[0]
                img[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = img[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] + sub
        if with_bkg:
            img = img + fr["bkg"][0] + fr["bkg"][1] * fr["xx"] + fr["bkg"][2] * fr["yy"]
        return img

    def loss(self, frames=None, smooth=1e-2, clip=None):
        frames = self.frames if frames is None else frames
        total = 0.0
        for fr in frames:
            m = self.render_frame(fr)
            r2 = fr["w"] * (m - fr["dat"]) ** 2
            if clip is not None:
                r2 = r2 * clip[fr["dither"]]
            npix = float((fr["w"] > 0).sum().item()) if fr["dither"] in self.pixmask else r2.numel()
            total = total + r2.sum() / max(npix, 1.0)
        reg = 0.0
        for src in self.sources.values():
            reg = reg + smooth * ((src.logf1[1:] - src.logf1[:-1]) ** 2).mean() + smooth * ((src.logf0[1:] - src.logf0[:-1]) ** 2).mean()
        return total + reg / max(len(self.sources), 1) + self.mag_prior_weight * (self.mag_prior() + self.shift_prior())

    def fit(self, frames=None, iters=100, lr=0.1, smooth=1e-2, log_every=10, clip_sigma=None, mag_prior_weight=1.0):
        frames = self.frames if frames is None else frames
        self.mag_prior_weight = mag_prior_weight
        opt = torch.optim.Adam([{"params": self.params(frames=False), "lr": lr},
                                {"params": [p for fr in frames for p in (fr["bkg"], fr["sens"])], "lr": 0.05}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.05)
        clip = None
        self.loss_history = []
        for it in range(iters):
            opt.zero_grad()
            loss = self.loss(frames, smooth=smooth, clip=clip)
            loss.backward(); opt.step(); sched.step()
            self.loss_history.append(float(loss.item()))
            with torch.no_grad():
                for fr in frames:
                    if fr is frames[0]:
                        fr["sens"].zero_()                      # reference frame fixed
                    else:
                        fr["sens"][0].clamp_(-1.0, 1.0); fr["sens"][1:].clamp_(-0.5, 0.5)
            if clip_sigma is not None and it in (iters // 3, 2 * iters // 3):
                with torch.no_grad():
                    clip = {}
                    for fr in frames:
                        m = self.render_frame(fr); chi = (fr["dat"] - m) * torch.sqrt(fr["w"])
                        src_only = self.render_frame(fr, with_bkg=False)
                        clip[fr["dither"]] = (~((chi > clip_sigma) & (src_only < 0.1 * src_only.max()))).float()
            if log_every and (it % log_every == 0 or it == iters - 1):
                print(f"  it {it}: loss {loss.item():.4f}  mag-prior {self.mag_prior().item():.3f}  zp {self.zp.item():.3f}", flush=True)
        return self

    def align_frame(self, fr, iters=40, lr=0.05):
        """Held-out-frame alignment: fit only a global (dx, dy) offset and the background plane of `fr`
        to its data, with every spectrum, per-source shift and kernel frozen (3 + 2 free parameters)."""
        fr["dxy"] = torch.zeros(2, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([fr["dxy"], fr["bkg"]], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            m = self.render_frame(fr)
            npix = float((fr["w"] > 0).sum().item())
            loss = (fr["w"] * (m - fr["dat"]) ** 2).sum() / max(npix, 1.0)
            loss.backward(); opt.step()
        return fr["dxy"].detach().cpu().numpy()

    @torch.no_grad()
    def calibrate_uncovered(self, train_keys):
        """SIR-like fallback for sources whose 1st order was never seen in a training frame:
        flat spectrum scaled from H via the counts-per-magnitude relation of the fitted stars."""
        ratios = []; shapes = []
        for src in self.sources.values():
            n = sum(k in src.coverage for k in train_keys)
            if n >= 2 and np.isfinite(src.hmag) and src.hmag < 21:
                f = torch.exp(src.logf1); tot = float(f.sum())
                ratios.append(np.log10(tot) + 0.4 * src.hmag); shapes.append((f / f.mean()).cpu().numpy())
        if not ratios:
            return 0
        zp = float(np.median(ratios)); shape = np.median(np.array(shapes), axis=0); shape /= shape.mean()
        # 0th/1st order flux ratio from sources constrained in both orders
        r0 = [float(torch.exp(src.logf0).sum() / torch.exp(src.logf1).sum()) for src in self.sources.values()
              if sum(k in src.coverage for k in train_keys) >= 2 and sum(k in src.coverage0 for k in train_keys) >= 1]
        ratio0 = float(np.median(r0)) if r0 else 0.03
        n_set = 0; n_set0 = 0
        for src in self.sources.values():
            n = sum(k in src.coverage for k in train_keys)
            if n == 0:
                tot = 10 ** (zp - 0.4 * src.hmag) if np.isfinite(src.hmag) else 0.0
                src.logf1.copy_(torch.log(torch.as_tensor(np.clip(tot / self.L * shape, 1e-3, None), dtype=torch.float32, device=self.device)))
                n_set += 1
            if sum(k in src.coverage0 for k in train_keys) == 0:
                tot1 = float(torch.exp(src.logf1).sum())
                src.logf0.fill_(float(np.log(max(ratio0 * tot1 / len(self.lam0), 1e-3))))
                n_set0 += 1
        self.ratio0 = ratio0
        return n_set, n_set0

    @torch.no_grad()
    def chi_map(self, fr):
        m = self.render_frame(fr)
        return ((fr["dat"] - m) * torch.sqrt(fr["w"])).cpu().numpy(), m.cpu().numpy()
