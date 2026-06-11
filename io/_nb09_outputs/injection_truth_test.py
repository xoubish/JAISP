"""Synthetic-injection truth test for the latent astrometry pipeline.

Injects Gaussian point sources with KNOWN sub-pixel sky positions into all 10
bands of held-out patch-25 tiles, then runs the exact eval pipeline (classical
VIS refine -> per-band centroid + project -> latent head) and measures every
stage against the injected truth. This is immune to the centroid-emulation
degeneracy of label-based residuals: truth is known by construction.

Measured per injected source, per band, per SNR level:
  err_vis      : classical VIS centroid vs truth          (the VIS anchor floor)
  err_band     : band centroid projected to VIS vs truth  (classical band measurement)
  err_head_*   : head-corrected canonical position vs truth (production / patchval head)

Truth convention: one sky position per source; injected into each band via that
band's own WCS, so WCS calibration is treated as truth and the test isolates
measurement + model error.
"""
import os
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/shemmati/Work/Projects/JAISP")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

from astropy.wcs import WCS
from scipy.spatial import cKDTree

from foundation_utils import load_tile_data
from astrometry2.latent_position_head import load_latent_position_head
from astrometry2.source_matching import safe_header_from_card_string
from astrometry2.dataset import refine_centroids_psf_fit, local_vis_pixel_to_sky_matrix
from train_latent_position import encode_tile_features  # noqa: E402  (models/astrometry2 on path)
sys.path.insert(0, str(ROOT / "models" / "astrometry2"))
from eval_latent_position import centroid_in_band_and_project  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOUNDATION = ROOT / "models/checkpoints/jaisp_v10_warmstart/checkpoint_best.pt"
HEADS = {
    "prod": ROOT / "models/checkpoints/latent_position_v10_no_psf/best.pt",
    "patchval": ROOT / "models/checkpoints/latent_position_v10_patchval25/best.pt",
}
LABELS = torch.load(ROOT / "data/detection_labels/centernet_v10_790_thresh03.pt",
                    map_location="cpu", weights_only=False)
LABELS = LABELS["labels"] if "labels" in LABELS else LABELS

N_TILES = 6
N_INJECT = 20
SNR_LEVELS = [5, 7, 10, 15, 30]
RNG = np.random.default_rng(20260611)

# injection PSF sigmas (px) matched to what the eval's centroid fitter assumes
SIGMA_PX = {"VIS": 2.5 / 2.355, "NISP": 2.5 / 2.355, "RUBIN": 3.0 / 2.355}
RUBIN_BANDS = ["u", "g", "r", "i", "z", "y"]
NISP_BANDS = ["Y", "J", "H"]


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


# ---- load heads (shared frozen encoder: same foundation checkpoint) ----
heads = {}
frozen_encoder = None
for tag, p in HEADS.items():
    ck = torch.load(p, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    enc, head = load_latent_position_head(
        str(FOUNDATION), device=DEVICE,
        bottleneck_window=cfg.get("bottleneck_window", 5) or 5,
        stem_window=cfg.get("stem_window", 17) or 17,
    )
    head.load_state_dict(ck["head_state_dict"])
    head.eval()
    heads[tag] = head
    if frozen_encoder is None:
        frozen_encoder = enc

# ---- tiles: first N sorted patch-25 stems ----
rdir, edir = ROOT / "data/rubin_tiles_patch25", ROOT / "data/euclid_tiles_patch25"
stems = sorted(p.name[:-4] for p in rdir.glob("tile_*.npz"))[:N_TILES]
print(f"tiles: {stems}")

records = []  # dicts: tile, band, snr_level, err_vis, err_band, err_head_prod, err_head_patchval

for stem in stems:
    rpath, epath = rdir / f"{stem}.npz", edir / f"{stem}_euclid.npz"
    rdata = dict(np.load(rpath, allow_pickle=True))
    edata = dict(np.load(epath, allow_pickle=True))
    vwcs = WCS(safe_header_from_card_string(edata["wcs_VIS"].item()))
    rwcs = WCS(rdata["wcs_hdr"].item())
    vis_img0 = np.nan_to_num(np.asarray(edata["img_VIS"], dtype=np.float32), nan=0.0)
    H, W = vis_img0.shape
    vis_rms = np.sqrt(np.clip(np.asarray(edata["var_VIS"], dtype=np.float32), 1e-12, None))

    # existing sources to avoid
    if stem in LABELS:
        ent = LABELS[stem]
        xy_norm = np.asarray(ent[0] if isinstance(ent, tuple) else ent, dtype=np.float32)
        existing = np.stack([xy_norm[:, 0] * (W - 1), xy_norm[:, 1] * (H - 1)], axis=1)
    else:
        existing = np.zeros((0, 2), dtype=np.float32)
    tree = cKDTree(existing) if len(existing) else None

    # injection positions: random sub-pixel, >=25 px from real sources and each other
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

    for snr in SNR_LEVELS:
        # ---- build injected copies of every band ----
        einj = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in edata.items()}
        rinj = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rdata.items()}
        # Euclid bands (VIS + NISP), native 0.1"/px grids with their own WCS
        for short in ["VIS"] + NISP_BANDS:
            key = f"img_{short}"
            if key not in einj:
                continue
            img = np.nan_to_num(np.asarray(einj[key], dtype=np.float32), nan=0.0)
            rms = np.sqrt(np.clip(np.asarray(einj[f"var_{short}"], dtype=np.float32), 1e-12, None))
            w = vwcs if short == "VIS" else WCS(safe_header_from_card_string(einj[f"wcs_{short}"].item()))
            bx, by = w.world_to_pixel_values(ra_true, dec_true)
            sig = SIGMA_PX["VIS" if short == "VIS" else "NISP"]
            for k in range(N_INJECT):
                inject_gauss(img, float(bx[k]), float(by[k]), sig, snr * local_rms(rms, bx[k], by[k]))
            einj[key] = img
        # Rubin cube, 0.2"/px, shared WCS
        rcube = np.nan_to_num(np.asarray(rinj["img"], dtype=np.float32), nan=0.0).copy()
        rrms_cube = np.sqrt(np.clip(np.asarray(rinj["var"], dtype=np.float32), 1e-12, None))
        rx, ry = rwcs.world_to_pixel_values(ra_true, dec_true)
        for bi in range(rcube.shape[0]):
            for k in range(N_INJECT):
                inject_gauss(rcube[bi], float(rx[k]), float(ry[k]), SIGMA_PX["RUBIN"],
                             snr * local_rms(rrms_cube[bi], rx[k], ry[k]))
        rinj["img"] = rcube

        # ---- temp npz so the standard loader builds encoder inputs ----
        with tempfile.TemporaryDirectory() as td:
            rp, ep = Path(td) / "r.npz", Path(td) / "e.npz"
            np.savez(rp, **rinj)
            np.savez(ep, **einj)
            img_t, rms_t, vis_hw, _ = load_tile_data(str(rp), str(ep), DEVICE)

        with torch.no_grad():
            enc_out = encode_tile_features(frozen_encoder, stem + "_INJ", img_t, rms_t, DEVICE,
                                           features_cache_dir=None)
        del img_t, rms_t

        vis_inj = np.asarray(einj["img_VIS"], dtype=np.float32)

        # ---- stage 1: classical VIS refine, seeded near truth (0.5 px jitter) ----
        seeds = pts + RNG.normal(scale=0.5, size=pts.shape).astype(np.float32)
        vis_xy, vis_snr_meas, _ = refine_centroids_psf_fit(vis_inj, seeds, radius=3, fwhm_guess=2.5)
        err_vis = np.hypot(*(vis_xy - pts).T) * 0.1  # arcsec

        # ---- stage 2+3: per-band centroid + head, exact eval path ----
        for band_name in [f"rubin_{b}" for b in RUBIN_BANDS] + [f"nisp_{b}" for b in NISP_BANDS]:
            if band_name.startswith("rubin_"):
                short = band_name.split("_", 1)[1]
                band_img = np.asarray(rinj["img"][RUBIN_BANDS.index(short)], dtype=np.float32)
                band_rms = rrms_cube[RUBIN_BANDS.index(short)]
                band_wcs, fwhm = rwcs, 3.0
            else:
                short = band_name.split("_", 1)[1]
                band_img = np.asarray(einj[f"img_{short}"], dtype=np.float32)
                band_rms = np.sqrt(np.clip(np.asarray(einj[f"var_{short}"], dtype=np.float32), 1e-12, None))
                band_wcs = WCS(safe_header_from_card_string(einj[f"wcs_{short}"].item()))
                fwhm = 2.5

            band_xy_vis, offset_arcsec, valid, band_snr = centroid_in_band_and_project(
                band_img, band_wcs, vis_xy, vwcs,
                refine_radius=3, fwhm_guess=fwhm, band_rms=band_rms,
            )
            if valid.sum() == 0:
                continue
            vi = np.where(valid)[0]
            band_pos = band_xy_vis[valid].astype(np.float32)
            err_band = np.hypot(*(band_pos - pts[vi]).T) * 0.1

            pix2sky = np.zeros((len(vi), 2, 2), dtype=np.float32)
            for i in range(len(vi)):
                pix2sky[i] = local_vis_pixel_to_sky_matrix(vwcs, band_pos[i])

            errs_head = {}
            for tag, head in heads.items():
                with torch.no_grad():
                    out = head(
                        enc_out["bottleneck"], enc_out["vis_stem"],
                        torch.from_numpy(band_pos).to(DEVICE),
                        torch.from_numpy(pix2sky).to(DEVICE),
                        enc_out["fused_hw"], vis_hw,
                    )
                pred = out["pred_offset_arcsec"].cpu().numpy()
                pred_px = np.einsum("nij,nj->ni", np.linalg.inv(pix2sky), pred)
                head_xy = band_pos + pred_px
                errs_head[tag] = np.hypot(*(head_xy - pts[vi]).T) * 0.1

            for i, k in enumerate(vi):
                records.append(dict(
                    tile=stem, band=band_name, snr=snr,
                    err_vis=float(err_vis[k]), err_band=float(err_band[i]),
                    err_head_prod=float(errs_head["prod"][i]),
                    err_head_patchval=float(errs_head["patchval"][i]),
                    band_snr_meas=float(band_snr[vi[i]] if hasattr(band_snr, "__len__") else band_snr),
                ))
        print(f"  {stem} snr={snr}: {len(records)} records total", flush=True)

# ---- aggregate ----
import collections
out = {"records": records}
print("\n==== medians vs TRUTH (mas) ====")
print(f"{'SNR':>4s} {'group':6s} {'N':>5s} | {'VIS cls':>8s} {'band cls':>8s} {'head prod':>9s} {'head pval':>9s}")
for snr in SNR_LEVELS:
    for grp, sel in [("rubin", lambda b: b.startswith("rubin")), ("nisp", lambda b: b.startswith("nisp"))]:
        rs = [r for r in records if r["snr"] == snr and sel(r["band"])]
        if not rs:
            continue
        med = lambda k: 1000 * float(np.median([r[k] for r in rs]))
        print(f"{snr:4d} {grp:6s} {len(rs):5d} | {med('err_vis'):8.1f} {med('err_band'):8.1f} "
              f"{med('err_head_prod'):9.1f} {med('err_head_patchval'):9.1f}")
json.dump(out, open(ROOT / "io/_nb09_outputs/injection_truth_results.json", "w"))
print("\nsaved io/_nb09_outputs/injection_truth_results.json")
