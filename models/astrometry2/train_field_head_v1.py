"""Astrometry position head v1 — truth-trained, all-band native stems, heteroscedastic.

Goal (see field_head_DESIGN.md): beat the classical joint fit where it is MIS-SPECIFIED
(extended sources, faint, chromatic) by training a learned joint position estimator against
INJECTION TRUTH instead of the classical centroid, reading ALL 10 bands at NATIVE resolution.

Decisive differences vs v0 (train_field_head.py):
  1. TARGET = injected truth position (not the band's classical Gaussian centroid) -> breaks the
     emulation cap; the head can exceed classical.
  2. INPUT = frozen per-band stems at NATIVE resolution, ALL 10 bands (Euclid 0.1"/px via VIS WCS,
     Rubin 0.2"/px via Rubin WCS) + fused bottleneck as context -> bypasses the 0.4"/px ceiling and
     uses the same photons the classical joint fit uses. Frozen stems = transferable denoising (OOD).
  3. OUTPUT = (dx, dy, log_sigma) heteroscedastic -> uncertainty gating (never worse than classical).

Training regime: inject isotropic Gaussians of VARYING SIZE (point -> extended) with a random
per-band SED (color) at KNOWN sub-pixel positions, across an SNR range. The wide/faint/colored
injections are exactly where the classical fixed-FWHM point-fit is biased.

v1-alpha scope: isotropic sigma (not full 2D covariance), Gaussian (not ePSF/Sersic) injection.
These are the natural next refinements once the concept is shown.

Run:
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=models python -u \
    models/astrometry2/train_field_head_v1.py \
    --rubin-dir data/rubin_tiles_all --euclid-dir data/euclid_tiles_all \
    --foundation-checkpoint models/checkpoints/jaisp_v10_warmstart/checkpoint_best.pt \
    --val-patches 25 --output-dir models/checkpoints/field_head_v1 --epochs 30
"""
from __future__ import annotations
import argparse, sys, math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "models"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from astropy.wcs import WCS
from foundation_utils import load_tile_data
from astrometry2.latent_position_head import load_latent_position_head, extract_local_windows, vis_px_to_bottleneck_px
from astrometry2.dataset import local_vis_pixel_to_sky_matrix
from astrometry2.source_matching import safe_header_from_card_string
from jaisp_foundation_v10 import ConvNeXtBlock

EUCLID = ["euclid_VIS", "euclid_Y", "euclid_J", "euclid_H"]   # 0.1"/px, VIS WCS
RUBIN  = ["rubin_u", "rubin_g", "rubin_r", "rubin_i", "rubin_z", "rubin_y"]  # 0.2"/px, Rubin WCS
ALL_BANDS = RUBIN + EUCLID
EUC_STAMP, RUB_STAMP = 15, 9      # ~1.5"/1.8" windows
SNR_LEVELS = (5., 7., 10., 15., 30.)
SIZE_PX = (0.8, 3.5)             # injected Gaussian sigma range (VIS px); wide = "extended"
JIT_MAS, LABEL_NOISE_MAS, PIX_MAS = 30.0, 3.0, 100.0


def patch_of(stem): return stem.rsplit("_patch_", 1)[-1]

def discover(rubin_dir, euclid_dir):
    r = {p.name[:-4]: p for p in Path(rubin_dir).glob("tile_*.npz")}
    out = []
    for stem, rp in sorted(r.items()):
        ep = Path(euclid_dir) / f"{stem}_euclid.npz"
        if ep.exists(): out.append((stem, rp, ep))
    return out


def inject_gauss(img, x, y, sx, sy, amp):
    H, W = img.shape
    r = int(max(6, np.ceil(4 * max(sx, sy))))
    x0, x1 = max(0, int(x) - r), min(W, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(H, int(y) + r + 1)
    if x1 <= x0 or y1 <= y0: return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1] += (amp * np.exp(-0.5 * (((xx - x) / sx) ** 2 + ((yy - y) / sy) ** 2))).astype(img.dtype)


def local_rms(rms, x, y):
    xi = int(np.clip(round(x), 0, rms.shape[1] - 1)); yi = int(np.clip(round(y), 0, rms.shape[0] - 1))
    v = float(rms[yi, xi]); return v if np.isfinite(v) and v > 0 else float(np.nanmedian(rms))


class MultiBandPositionHead(nn.Module):
    """Per-band native-stem windows (all 10) + bottleneck context -> (dx,dy,log_sigma) in VIS px."""
    def __init__(self, stem_ch=64, hidden_ch=256, per_band_out=32, bn_out=64, mlp=256):
        super().__init__()
        self.bn_conv = nn.Sequential(ConvNeXtBlock(hidden_ch), nn.Conv2d(hidden_ch, bn_out, 1), nn.GELU())
        # shared small conv applied to each band's stem window
        self.stem_conv = nn.Sequential(nn.Conv2d(stem_ch, per_band_out, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(per_band_out, per_band_out, 3, padding=1), nn.GELU())
        feat = bn_out + per_band_out * len(ALL_BANDS)
        self.head = nn.Sequential(nn.Linear(feat, mlp), nn.GELU(), nn.Linear(mlp, mlp), nn.GELU(), nn.Linear(mlp, 3))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad(): self.head[-1].bias[2] = math.log(0.02)   # 20 mas init sigma
        self.bn_out, self.per_band_out = bn_out, per_band_out

    def _pool(self, w):  # global average pool [N,C,k,k]->[N,C]
        return w.mean(dim=(-2, -1))

    def forward(self, bottleneck, stems, pos_vis, pos_rub, pix2sky, fused_hw, vis_hw):
        # bottleneck context (VIS-frame position -> bottleneck px)
        pbn = vis_px_to_bottleneck_px(pos_vis, 0.1, 0.4, fused_hw, vis_hw)
        bn = self._pool(self.bn_conv(extract_local_windows(bottleneck, pbn, 5)))
        vecs = [bn]
        for b in ALL_BANDS:
            pos = pos_vis if b in EUCLID else pos_rub
            ws = EUC_STAMP if b in EUCLID else RUB_STAMP
            vecs.append(self._pool(self.stem_conv(extract_local_windows(stems[b], pos, ws))))
        out = self.head(torch.cat(vecs, 1))
        dxy = out[:, :2]; log_sigma = out[:, 2].clamp(-6, 3)
        pred_sky = torch.bmm(pix2sky, dxy.unsqueeze(-1)).squeeze(-1)   # arcsec
        return pred_sky, log_sigma


def gaussian_nll(pred, tgt, log_sigma):
    r2 = ((pred - tgt) ** 2).sum(-1)
    sig2 = torch.exp(log_sigma) ** 2 + (LABEL_NOISE_MAS / 1000.) ** 2
    return (0.5 * r2 / sig2 + torch.log(sig2)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubin-dir", default="data/rubin_tiles_all")
    ap.add_argument("--euclid-dir", default="data/euclid_tiles_all")
    ap.add_argument("--foundation-checkpoint", required=True)
    ap.add_argument("--val-patches", default="25")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--inject-per-tile", type=int, default=40)
    ap.add_argument("--device", default="")
    ap.add_argument("--max-tiles", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    frozen, _ = load_latent_position_head(args.foundation_checkpoint, device=dev, bottleneck_window=5)
    head = MultiBandPositionHead().to(dev)
    print(f"MultiBandPositionHead v1: {sum(p.numel() for p in head.parameters())/1e6:.2f}M trainable", flush=True)

    pairs = discover(args.rubin_dir, args.euclid_dir)
    val = {s.strip() for s in args.val_patches.split(",") if s.strip()}
    tr = [p for p in pairs if patch_of(p[0]) not in val]
    va = [p for p in pairs if patch_of(p[0]) in val]
    if args.max_tiles: tr, va = tr[:args.max_tiles], va[:max(2, args.max_tiles // 4)]
    print(f"tiles: {len(tr)} train, {len(va)} val", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    def run_tile(stem, rp, ep, train):
        rdata = dict(np.load(rp, allow_pickle=True)); edata = dict(np.load(ep, allow_pickle=True))
        vwcs = WCS(safe_header_from_card_string(edata["wcs_VIS"].item()))
        _rh = rdata["wcs_hdr"].item()
        rwcs = WCS(safe_header_from_card_string(_rh)) if isinstance(_rh, str) else WCS(_rh)
        H, W = np.asarray(edata["img_VIS"]).shape
        snr = float(rng.choice(SNR_LEVELS))
        # true positions (VIS px), sub-pixel, away from borders
        n = args.inject_per_tile
        pts = np.column_stack([rng.uniform(60, W - 60, n), rng.uniform(60, H - 60, n)]).astype(np.float32)
        ra_t, dec_t = vwcs.all_pix2world(pts[:, 0], pts[:, 1], 0)
        sed = rng.uniform(0.5, 1.5, (n, len(ALL_BANDS)))            # per-source per-band color
        sx = rng.uniform(*SIZE_PX, n); sy = sx * rng.uniform(0.7, 1.0, n)  # size (some extended)
        # inject into copies of all bands
        ei = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in edata.items()}
        ri = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rdata.items()}
        for bi, b in enumerate(ALL_BANDS):
            if b in EUCLID:
                short = b.split("_")[1]; key = f"img_{short}"; vkey = f"var_{short}"
                if key not in ei: continue
                img = np.nan_to_num(np.asarray(ei[key], np.float32)); rms = np.sqrt(np.clip(np.asarray(ei[vkey], np.float32), 1e-12, None))
                w = vwcs if short == "VIS" else WCS(safe_header_from_card_string(ei[f"wcs_{short}"].item()))
                bx, by = w.all_world2pix(ra_t, dec_t, 0)
                for k in range(n): inject_gauss(img, bx[k], by[k], sx[k], sy[k], snr * sed[k, bi] * local_rms(rms, bx[k], by[k]))
                ei[key] = img
            else:
                ci = RUBIN.index(b)
                cube = np.nan_to_num(np.asarray(ri["img"], np.float32)); rmsc = np.sqrt(np.clip(np.asarray(ri["var"], np.float32), 1e-12, None))
                rx, ry = rwcs.all_world2pix(ra_t, dec_t, 0)
                for k in range(n): inject_gauss(cube[ci], rx[k], ry[k], sx[k]*0.5, sy[k]*0.5, snr * sed[k, bi] * local_rms(rmsc[ci], rx[k], ry[k]))
                ri["img"] = cube
        # encode injected tile
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            rp2, ep2 = Path(td) / "r.npz", Path(td) / "e.npz"
            np.savez(rp2, **ri); np.savez(ep2, **ei)
            img_t, rms_t, vis_hw, _ = load_tile_data(str(rp2), str(ep2), dev)
        with torch.no_grad():
            enc = frozen.encode_tile(img_t, rms_t)
            stems = {b: frozen.encoder.stems[b](img_t[b], rms_t[b]) for b in ALL_BANDS}
        # query = jittered truth (VIS px); rubin query via rubin WCS
        q_vis = (pts + rng.normal(scale=JIT_MAS / PIX_MAS, size=pts.shape)).astype(np.float32)
        qra, qdec = vwcs.all_pix2world(q_vis[:, 0], q_vis[:, 1], 0)
        q_rub = np.column_stack(rwcs.all_world2pix(qra, qdec, 0)).astype(np.float32)
        pix2sky = np.stack([local_vis_pixel_to_sky_matrix(vwcs, p) for p in q_vis]).astype(np.float32)
        tgt_sky = np.einsum("nij,nj->ni", pix2sky, (pts - q_vis)).astype(np.float32)  # arcsec, query->truth
        pred, ls = head(enc["bottleneck"], stems, torch.from_numpy(q_vis).to(dev), torch.from_numpy(q_rub).to(dev),
                        torch.from_numpy(pix2sky).to(dev), enc["fused_hw"], vis_hw)
        loss = gaussian_nll(pred, torch.from_numpy(tgt_sky).to(dev), ls)
        # truth error in mas (for monitoring)
        with torch.no_grad():
            err = float(torch.median(torch.sqrt(((pred - torch.from_numpy(tgt_sky).to(dev)) ** 2).sum(-1))) * 1000)
        del img_t, rms_t, enc, stems
        return loss, err

    best = np.inf
    for epc in range(1, args.epochs + 1):
        head.train(); tl = te = nt = 0.0
        for stem, rp, ep in tr:
            loss, err = run_tile(stem, rp, ep, True)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
            tl += float(loss); te += err; nt += 1
        head.eval(); vl = ve = nv = 0.0
        with torch.no_grad():
            for stem, rp, ep in va:
                loss, err = run_tile(stem, rp, ep, False); vl += float(loss); ve += err; nv += 1
        sch.step()
        print(f"E {epc:2d} | train nll {tl/max(nt,1):.4f} err {te/max(nt,1):.1f}mas | "
              f"val nll {vl/max(nv,1):.4f} err {ve/max(nv,1):.1f}mas | lr {opt.param_groups[0]['lr']:.1e}", flush=True)
        torch.save({"head_state_dict": head.state_dict(), "config": vars(args), "epoch": epc}, out / "latest.pt")
        if vl / max(nv, 1) < best:
            best = vl / max(nv, 1)
            torch.save({"head_state_dict": head.state_dict(), "config": vars(args), "epoch": epc}, out / "best.pt")
            print(f"  -> new best {best:.4f}", flush=True)
    print("done. best", best, flush=True)


if __name__ == "__main__":
    main()
