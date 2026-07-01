"""Gate test for the NISP field head (see field_head_DESIGN.md).

On held-out patch-25, for each NISP band, compare band-VIS offsets from
(a) the classical band Gaussian centroid and (b) the field head, and answer:

  COLLAPSE check  : does the head PRESERVE the field, or flatten it toward VIS?
     - systematic: median(band-VIS) head vs classical  (NISP ~ +7 mas; head->0 = collapsed)
     - spatial   : arcmin-binned field map, corr(head, classical)  (high = preserved)
  EMULATION check : does the head reduce per-source noise below classical, or just
     reproduce it? per-source residual scatter (after removing the median field),
     head vs classical, stratified by S/N (esp. faint S/N<10).
  BEATS-CLASSICAL : faint-only (S/N<10) split-half field reproducibility, head vs classical.

PASS = field preserved (not collapsed) AND faint per-source noise below classical.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=models python models/astrometry2/eval_field_head.py \
    --head-checkpoint models/checkpoints/field_head_nisp_v0/best.pt \
    --foundation-checkpoint models/checkpoints/jaisp_v10_warmstart/checkpoint_best.pt \
    --centernet-labels data/detection_labels/centernet_v10_790_vispeak_thresh03.pt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "models"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from foundation_utils import load_tile_data
from astrometry2.latent_position_head import load_latent_position_head
from astrometry2.dataset import local_vis_pixel_to_sky_matrix, refine_centroids_psf_fit
from astrometry2.source_matching import safe_header_from_card_string
from astrometry2.train_field_head import NISP_BANDS, IMGKEY, band_stem_features, discover_pairs, patch_of
from astropy.wcs import WCS


def madxy(dxy):
    m = np.median(dxy, axis=0)
    return 1.4826 * np.median(np.abs(dxy - m))


def split_half_field(ra, dec, off, bin_arcmin=1.0, rng=None):
    """Bin band-VIS offset on a sky grid; split sources in half; corr of the two maps."""
    rng = rng or np.random.default_rng(0)
    order = rng.permutation(len(ra)); h1, h2 = order[::2], order[1::2]
    cosd = np.cos(np.deg2rad(np.median(dec)))
    x = (ra - ra.min()) * cosd * 60.0; y = (dec - dec.min()) * 60.0  # arcmin
    nb = max(2, int(max(x.max(), y.max()) / bin_arcmin) + 1)

    def gridmap(idx):
        gx = np.clip((x[idx] / bin_arcmin).astype(int), 0, nb - 1)
        gy = np.clip((y[idx] / bin_arcmin).astype(int), 0, nb - 1)
        acc = np.zeros((nb, nb, 2)); cnt = np.zeros((nb, nb))
        for k in range(len(idx)):
            acc[gy[k], gx[k]] += off[idx[k]]; cnt[gy[k], gx[k]] += 1
        m = cnt >= 3
        return np.where(m[..., None], acc / np.maximum(cnt[..., None], 1), np.nan), m

    m1, k1 = gridmap(h1); m2, k2 = gridmap(h2)
    both = k1 & k2
    if both.sum() < 5:
        return np.nan
    a = m1[both].ravel(); b = m2[both].ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 5 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-checkpoint", required=True)
    ap.add_argument("--foundation-checkpoint", required=True)
    ap.add_argument("--rubin-dir", default="data/rubin_tiles_all")
    ap.add_argument("--euclid-dir", default="data/euclid_tiles_all")
    ap.add_argument("--centernet-labels", required=True)
    ap.add_argument("--val-patches", default="25")
    ap.add_argument("--bottleneck-window", type=int, default=5)
    ap.add_argument("--max-tiles", type=int, default=0)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    fe, head = load_latent_position_head(args.foundation_checkpoint, device=device,
                                         bottleneck_window=args.bottleneck_window, stem_window=17)
    ck = torch.load(args.head_checkpoint, map_location="cpu", weights_only=False)
    head.load_state_dict(ck["head_state_dict"]); head = head.to(device).eval()
    print(f"loaded field head from {args.head_checkpoint} (epoch {ck.get('epoch')})")

    L = torch.load(args.centernet_labels, map_location="cpu", weights_only=False)
    labels = L["labels"] if "labels" in L else L
    val = {s.strip() for s in args.val_patches.split(",") if s.strip()}
    pairs = [p for p in discover_pairs(args.rubin_dir, args.euclid_dir) if patch_of(p[0]) in val]
    if args.max_tiles:
        pairs = pairs[:args.max_tiles]
    print(f"held-out tiles: {len(pairs)} (patch {sorted(val)})", flush=True)

    agg = {b: {"ra": [], "dec": [], "snr": [], "cls": [], "hd": []} for b in NISP_BANDS}
    for ti,(stem, rp, ep) in enumerate(pairs):
        if ti%10==0: print(f"  tile {ti}/{len(pairs)}",flush=True)
        if stem not in labels:
            continue
        edata = dict(np.load(ep, allow_pickle=True))
        vwcs = WCS(safe_header_from_card_string(edata["wcs_VIS"].item()))
        H, W = np.asarray(edata["img_VIS"]).shape
        ent = labels[stem]; xyn = np.asarray(ent[0] if isinstance(ent, tuple) else ent, dtype=np.float32)
        seeds = np.stack([xyn[:, 0] * (W - 1), xyn[:, 1] * (H - 1)], 1).astype(np.float32)
        if len(seeds) < 5:
            continue
        vis_img = np.nan_to_num(np.asarray(edata["img_VIS"], dtype=np.float32))
        cen_vis, _, _ = refine_centroids_psf_fit(vis_img, seeds, radius=3, fwhm_guess=2.5)
        img_t, rms_t, vis_hw, _ = load_tile_data(str(rp), str(ep), device)
        with torch.no_grad():
            enc = fe.encode_tile(img_t, rms_t); enc["vis_hw"] = vis_hw
            stem_cache = {b: band_stem_features(fe, b, img_t, rms_t) for b in NISP_BANDS}
        for band in NISP_BANDS:
            bimg = np.nan_to_num(np.asarray(edata[IMGKEY[band]], dtype=np.float32))
            cen_b, snr, _ = refine_centroids_psf_fit(bimg, seeds, radius=3, fwhm_guess=2.5)
            good = np.isfinite(cen_b).all(1) & np.isfinite(cen_vis).all(1) & (snr > 3)
            if good.sum() < 5:
                continue
            gi = np.where(good)[0]
            q = cen_b[gi].astype(np.float32)
            pix2sky = np.stack([local_vis_pixel_to_sky_matrix(vwcs, p) for p in q]).astype(np.float32)
            with torch.no_grad():
                out = head(enc["bottleneck"], stem_cache[band], torch.from_numpy(q).to(device),
                           torch.from_numpy(pix2sky).to(device), enc["fused_hw"], vis_hw)
            pred_sky = out["pred_offset_arcsec"].cpu().numpy()
            off_px = np.einsum("nij,nj->ni", np.linalg.inv(pix2sky), pred_sky)
            cen_head = q + off_px
            # band - VIS in mas via shared VIS WCS
            def bmv(cxy):
                r, d = vwcs.all_pix2world(cxy[:, 0], cxy[:, 1], 0)
                rv, dv = vwcs.all_pix2world(cen_vis[gi, 0], cen_vis[gi, 1], 0)
                cosd = np.cos(np.deg2rad(dv))
                return np.stack([(r - rv) * cosd * 3.6e6, (d - dv) * 3.6e6], 1)  # mas
            agg[band]["cls"].append(bmv(q))
            agg[band]["hd"].append(bmv(cen_head))
            r, d = vwcs.all_pix2world(q[:, 0], q[:, 1], 0)
            agg[band]["ra"].append(r); agg[band]["dec"].append(d); agg[band]["snr"].append(snr[gi])
        del img_t, rms_t, enc, stem_cache

    print(f"\n{'band':7} {'N':>6} | {'sys_cls(dRA,dDec)':>18} {'sys_hd':>18} | "
          f"{'MADxy cls/hd all':>16} {'faint<10 cls/hd':>16} | {'split-half r cls/hd':>18}")
    print("-" * 110)
    for b in NISP_BANDS:
        A = {k: (np.concatenate(v) if v else np.array([])) for k, v in agg[b].items()}
        if len(A["snr"]) < 20:
            print(f"{b:7} too few"); continue
        cls, hd, snr = A["cls"], A["hd"], A["snr"]
        scls, shd = np.median(cls, 0), np.median(hd, 0)
        madc, madh = madxy(cls), madxy(hd)
        f = snr < 10
        madcf = madxy(cls[f]) if f.sum() > 10 else np.nan
        madhf = madxy(hd[f]) if f.sum() > 10 else np.nan
        rc = split_half_field(A["ra"][f], A["dec"][f], cls[f] - scls) if f.sum() > 30 else np.nan
        rh = split_half_field(A["ra"][f], A["dec"][f], hd[f] - shd) if f.sum() > 30 else np.nan
        print(f"{b:7} {len(snr):6d} | ({scls[0]:+5.1f},{scls[1]:+5.1f})     ({shd[0]:+5.1f},{shd[1]:+5.1f})     | "
              f"{madc:5.1f}/{madh:5.1f}       {madcf:5.1f}/{madhf:5.1f}      | {rc:5.2f}/{rh:5.2f}")
    print("\nPASS if: sys_hd preserves sys_cls (NOT ->0), and faint MADxy hd < cls, and split-half r hd >= cls.")


if __name__ == "__main__":
    main()
