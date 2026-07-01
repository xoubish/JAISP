"""Truth-validate joint multi-band centroiding against the VIS-only canonical target.

Same injection protocol as injection_truth_test.py (same tiles, same RNG seed, same
positions). For each injected source, compares against truth:
  err_vis    : classical VIS-only Gaussian centroid (current canonical target)
  err_joint  : ONE sky position fitted jointly across all 10 bands
               (shared (dRA,dDec); per-band amplitude + background; per-pixel
                inverse-variance weights; band grids linked through local WCS affines)

If err_joint beats err_vis at the faint end, joint centroids become the new
training targets for the latent head (which transfers canonical localization).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ROOT = Path("/home/shemmati/Work/Projects/JAISP")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

from astropy.wcs import WCS
import torch

from astrometry2.source_matching import safe_header_from_card_string
from astrometry2.dataset import refine_centroids_psf_fit

RUBIN_BANDS = ["u", "g", "r", "i", "z", "y"]
NISP_BANDS = ["Y", "J", "H"]
SIGMA_PX = {"VIS": 2.5 / 2.355, "NISP": 2.5 / 2.355, "RUBIN": 3.0 / 2.355}
N_TILES = 6
N_INJECT = 20
SNR_LEVELS = [5, 7, 10, 15, 30]
RNG = np.random.default_rng(20260611)  # same seed => same positions as injection_truth_test
STAMP_R = 5  # px, per band

LABELS = torch.load(ROOT / "data/detection_labels/centernet_v10_790_vispeak_thresh03.pt",
                    map_location="cpu", weights_only=False)
LABELS = LABELS["labels"] if "labels" in LABELS else LABELS


def inject_gauss(img, x, y, sigma, amp):
    H, W = img.shape
    r = int(max(6, np.ceil(4 * sigma)))
    x0, x1 = max(0, int(x) - r), min(W, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(H, int(y) + r + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1] += (amp * np.exp(-0.5 * (((xx - x) / sigma) ** 2 + ((yy - y) / sigma) ** 2))).astype(img.dtype)


def local_rms(rms_img, x, y):
    xi = int(np.clip(round(x), 0, rms_img.shape[1] - 1))
    yi = int(np.clip(round(y), 0, rms_img.shape[0] - 1))
    v = float(rms_img[yi, xi])
    return v if np.isfinite(v) and v > 0 else float(np.nanmedian(rms_img))


def cutout(img, rms, x, y, r):
    H, W = img.shape
    xi, yi = int(round(x)), int(round(y))
    if xi - r < 0 or yi - r < 0 or xi + r + 1 > W or yi + r + 1 > H:
        return None
    sl = (slice(yi - r, yi + r + 1), slice(xi - r, xi + r + 1))
    yy, xx = np.mgrid[sl]
    return img[sl].astype(np.float64), rms[sl].astype(np.float64), xx.astype(np.float64), yy.astype(np.float64)


def joint_fit(stamps, p0_sky):
    """stamps: list of dicts {img, rms, xx, yy, cx(ra0,dec0 px), J (2x2 d(px)/d(arcsec)), sigma}
    Fit shared sky offset (du, dv arcsec) + per-band (amp, bg). Returns (du, dv) or None."""
    nb = len(stamps)
    # initial amplitudes from peak
    p0 = [0.0, 0.0]
    for s in stamps:
        p0 += [max(float(s["img"].max() - np.median(s["img"])), 1e-3), float(np.median(s["img"]))]
    p0 = np.asarray(p0, dtype=np.float64)

    def resid(p):
        du, dv = p[0], p[1]
        out = []
        for i, s in enumerate(stamps):
            amp, bg = p[2 + 2 * i], p[3 + 2 * i]
            cx = s["cx"][0] + s["J"][0, 0] * du + s["J"][0, 1] * dv
            cy = s["cx"][1] + s["J"][1, 0] * du + s["J"][1, 1] * dv
            model = amp * np.exp(-0.5 * (((s["xx"] - cx) / s["sigma"]) ** 2 + ((s["yy"] - cy) / s["sigma"]) ** 2)) + bg
            out.append(((s["img"] - model) / s["rms"]).ravel())
        return np.concatenate(out)

    try:
        res = least_squares(resid, p0, method="lm", max_nfev=400)
    except Exception:
        return None
    if not np.all(np.isfinite(res.x[:2])) or np.hypot(res.x[0], res.x[1]) > 1.0:
        return None
    return res.x[0], res.x[1]


rdir, edir = ROOT / "data/rubin_tiles_patch25", ROOT / "data/euclid_tiles_patch25"
stems = sorted(p.name[:-4] for p in rdir.glob("tile_*.npz"))[:N_TILES]
records = []

for stem in stems:
    rdata = dict(np.load(rdir / f"{stem}.npz", allow_pickle=True))
    edata = dict(np.load(edir / f"{stem}_euclid.npz", allow_pickle=True))
    vwcs = WCS(safe_header_from_card_string(edata["wcs_VIS"].item()))
    rwcs = WCS(rdata["wcs_hdr"].item())
    vis_img0 = np.nan_to_num(np.asarray(edata["img_VIS"], dtype=np.float32), nan=0.0)
    H, W = vis_img0.shape

    if stem in LABELS:
        ent = LABELS[stem]
        xy_norm = np.asarray(ent[0] if isinstance(ent, tuple) else ent, dtype=np.float32)
        existing = np.stack([xy_norm[:, 0] * (W - 1), xy_norm[:, 1] * (H - 1)], axis=1)
    else:
        existing = np.zeros((0, 2), dtype=np.float32)
    tree = cKDTree(existing) if len(existing) else None

    pts = []
    while len(pts) < N_INJECT:
        x = RNG.uniform(80, W - 80)
        y = RNG.uniform(80, H - 80)
        if tree is not None and np.isfinite(tree.query([[x, y]], distance_upper_bound=25.0)[0][0]):
            continue
        if pts and np.min(np.hypot(*(np.asarray(pts) - [x, y]).T)) < 60:
            continue
        pts.append([x, y])
    pts = np.asarray(pts, dtype=np.float32)
    ra_true, dec_true = vwcs.pixel_to_world_values(pts[:, 0], pts[:, 1])
    cosd = np.cos(np.deg2rad(np.median(dec_true)))

    # per-band WCS / images / rms
    band_defs = []
    for short in ["VIS"] + NISP_BANDS:
        w = vwcs if short == "VIS" else WCS(safe_header_from_card_string(edata[f"wcs_{short}"].item()))
        band_defs.append(dict(tag=short, wcs=w,
                              img0=np.nan_to_num(np.asarray(edata[f"img_{short}"], dtype=np.float32), nan=0.0),
                              rms=np.sqrt(np.clip(np.asarray(edata[f"var_{short}"], dtype=np.float32), 1e-12, None)),
                              sigma=SIGMA_PX["VIS" if short == "VIS" else "NISP"]))
    rcube0 = np.nan_to_num(np.asarray(rdata["img"], dtype=np.float32), nan=0.0)
    rrms = np.sqrt(np.clip(np.asarray(rdata["var"], dtype=np.float32), 1e-12, None))
    for bi, short in enumerate(RUBIN_BANDS):
        band_defs.append(dict(tag=f"rubin_{short}", wcs=rwcs, img0=rcube0[bi], rms=rrms[bi],
                              sigma=SIGMA_PX["RUBIN"]))

    for snr in SNR_LEVELS:
        injected = []
        for bd in band_defs:
            img = bd["img0"].copy()
            bx, by = bd["wcs"].world_to_pixel_values(ra_true, dec_true)
            for k in range(N_INJECT):
                inject_gauss(img, float(bx[k]), float(by[k]), bd["sigma"],
                             snr * local_rms(bd["rms"], bx[k], by[k]))
            injected.append(dict(bd, img=img, bx=bx, by=by))

        vis_inj = injected[0]["img"]
        seeds = pts + RNG.normal(scale=0.5, size=pts.shape).astype(np.float32)
        vis_xy, _, _ = refine_centroids_psf_fit(vis_inj, seeds, radius=3, fwhm_guess=2.5)
        err_vis = np.hypot(*(vis_xy - pts).T) * 0.1

        for k in range(N_INJECT):
            ra0 = ra_true[k] + (seeds[k, 0] - pts[k, 0]) * 0.1 / 3600.0 / cosd * -1  # seed offset in sky (approx)
            # simpler: seed sky position via wcs
            ra0, dec0 = vwcs.pixel_to_world_values(seeds[k, 0], seeds[k, 1])
            stamps = []
            for bd in injected:
                px0, py0 = bd["wcs"].world_to_pixel_values(ra0, dec0)
                c = cutout(bd["img"], bd["rms"], float(px0), float(py0), STAMP_R)
                if c is None:
                    continue
                img_s, rms_s, xx, yy = c
                # local affine d(px)/d(arcsec) via finite differences
                e = 0.1 / 3600.0
                pxa, pya = bd["wcs"].world_to_pixel_values(ra0 + e / cosd, dec0)
                pxb, pyb = bd["wcs"].world_to_pixel_values(ra0, dec0 + e)
                J = np.array([[(pxa - px0) / 0.1, (pxb - px0) / 0.1],
                              [(pya - py0) / 0.1, (pyb - py0) / 0.1]], dtype=np.float64)
                stamps.append(dict(img=img_s, rms=rms_s, xx=xx, yy=yy,
                                   cx=(float(px0), float(py0)), J=J, sigma=bd["sigma"]))
            if len(stamps) < 5:
                continue
            fit = joint_fit(stamps, None)
            if fit is None:
                continue
            du, dv = fit  # arcsec offsets from seed sky position
            ra_fit = ra0 + du / 3600.0 / cosd
            dec_fit = dec0 + dv / 3600.0
            fx, fy = vwcs.world_to_pixel_values(ra_fit, dec_fit)
            err_joint = float(np.hypot(fx - pts[k, 0], fy - pts[k, 1]) * 0.1)
            records.append(dict(tile=stem, snr=snr, err_vis=float(err_vis[k]), err_joint=err_joint))
        print(f"  {stem} snr={snr}: {len(records)} records", flush=True)

import json
print("\n==== canonical-position error vs TRUTH (mas, median) ====")
print(f"{'SNR':>4s} {'N':>5s} | {'VIS-only':>9s} {'joint-10band':>12s} {'gain':>6s}")
for snr in SNR_LEVELS:
    rs = [r for r in records if r["snr"] == snr]
    if not rs:
        continue
    mv = 1000 * float(np.median([r["err_vis"] for r in rs]))
    mj = 1000 * float(np.median([r["err_joint"] for r in rs]))
    print(f"{snr:4d} {len(rs):5d} | {mv:9.1f} {mj:12.1f} {mv/max(mj,1e-9):5.2f}x")
json.dump(records, open(ROOT / "io/_nb09_outputs/joint_centroid_truth_results.json", "w"))
print("saved io/_nb09_outputs/joint_centroid_truth_results.json")
