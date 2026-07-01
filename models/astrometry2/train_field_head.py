"""Foundation FIELD-measurement head (v0: NISP Y/J/H).

See field_head_DESIGN.md. Unlike the production latent-position head (which regresses
every source to the VIS centroid and therefore ERASES the concordance field), this head
denoises each band's OWN centroid using that band's own stem + the fused bottleneck for
shape context. Target = the band's own high-S/N centroid, so the band-vs-VIS field F_X is
preserved by construction; the multi-band context only sharpens the centroid.

v0 scope: the three NISP bands (euclid_Y/J/H), which share VIS's 0.1"/px MER grid, so the
existing VIS-frame head machinery applies unchanged (query/target/output all in VIS pixels).
The ONLY differences from the production head:
  - stem features come from encoder.stems[band]  (not euclid_VIS)
  - training target is the band's own Gaussian centroid  (not the VIS centroid)

Rubin bands (0.2"/px) need per-band frame/Jacobian handling -> v1 (TODO).

Run (example):
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=models python -u \
    models/astrometry2/train_field_head.py \
    --rubin-dir data/rubin_tiles_all --euclid-dir data/euclid_tiles_all \
    --foundation-checkpoint models/checkpoints/jaisp_v10_warmstart/checkpoint_best.pt \
    --features-cache-dir data/cached_features_v10_warmstart \
    --centernet-labels data/detection_labels/centernet_v10_790_vispeak_thresh03.pt \
    --val-patches 25 \
    --output-dir models/checkpoints/field_head_nisp_v0 --epochs 30
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "models"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from foundation_utils import load_tile_data
from astrometry2.latent_position_head import (
    load_latent_position_head, extract_local_windows,
)
from astrometry2.dataset import local_vis_pixel_to_sky_matrix, refine_centroids_psf_fit
from astrometry2.source_matching import safe_header_from_card_string
from astropy.wcs import WCS

NISP_BANDS = ["euclid_Y", "euclid_J", "euclid_H"]   # v0: shared 0.1"/px VIS grid
IMGKEY = {"euclid_Y": "img_Y", "euclid_J": "img_J", "euclid_H": "img_H"}
VARKEY = {"euclid_Y": "var_Y", "euclid_J": "var_J", "euclid_H": "var_H"}
JIT_MAS = 30.0            # jitter for self-supervision (matches production)
LABEL_NOISE_MAS = 5.0     # SITCOMTN-159 floor
PIX_MAS = 100.0           # 0.1"/px -> 100 mas/px


def discover_pairs(rubin_dir, euclid_dir):
    r = {p.name[:-4]: p for p in Path(rubin_dir).glob("tile_*.npz")}
    out = []
    for stem, rp in sorted(r.items()):
        ep = Path(euclid_dir) / f"{stem}_euclid.npz"
        if ep.exists():
            out.append((stem, rp, ep))
    return out


def patch_of(stem): return stem.rsplit("_patch_", 1)[-1]


def band_stem_features(frozen_encoder, band, img_t, rms_t):
    """Raw per-band stem feature map [1, C, H, W] at the band's native (0.1") grid."""
    return frozen_encoder.encoder.stems[band](img_t[band], rms_t[band])


def rayleigh_nll(pred_off, target_off, log_sigma):
    # pred/target offsets in arcsec [N,2]; Rayleigh NLL with label-noise floor
    r2 = ((pred_off - target_off) ** 2).sum(-1)
    sig = torch.exp(log_sigma)
    sig_eff2 = sig ** 2 + (LABEL_NOISE_MAS / 1000.0) ** 2
    return (0.5 * r2 / sig_eff2 + torch.log(sig_eff2)).mean()


def gaussian_centroids(img, seeds_xy):
    """Band's own Gaussian centroids + measured S/N at seed positions (VIS-pixel frame)."""
    xy, snr, _ = refine_centroids_psf_fit(img, seeds_xy, radius=3, fwhm_guess=2.5)
    return xy, snr


def build_tile_batch(frozen_encoder, head, stem_cache, enc_out, band, seeds_xy, band_img,
                     vwcs, device, rng):
    """One band, one tile -> (pred_off, target_off, log_sigma) in arcsec, on device."""
    # target = the band's OWN Gaussian centroid; query = jittered target
    cen_xy, snr = gaussian_centroids(band_img, seeds_xy)
    good = np.isfinite(cen_xy).all(1) & (snr > 5)
    if good.sum() < 5:
        return None
    cen_xy = cen_xy[good].astype(np.float32); snr = snr[good]
    jit_px = JIT_MAS / PIX_MAS
    query_xy = (cen_xy + rng.normal(scale=jit_px, size=cen_xy.shape)).astype(np.float32)

    # per-source pixel->sky Jacobian at the query position
    pix2sky = np.stack([local_vis_pixel_to_sky_matrix(vwcs, q) for q in query_xy]).astype(np.float32)
    # target offset (query -> true centroid) in sky arcsec
    dpix = (cen_xy - query_xy)
    tgt_sky = np.einsum("nij,nj->ni", pix2sky, dpix).astype(np.float32)  # arcsec

    q = torch.from_numpy(query_xy).to(device)
    j = torch.from_numpy(pix2sky).to(device)
    out = head(enc_out["bottleneck"], stem_cache[band], q, j,
               enc_out["fused_hw"], enc_out["vis_hw"])
    pred = out["pred_offset_arcsec"]
    return pred, torch.from_numpy(tgt_sky).to(device), out["log_sigma"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubin-dir", default="data/rubin_tiles_all")
    ap.add_argument("--euclid-dir", default="data/euclid_tiles_all")
    ap.add_argument("--foundation-checkpoint", required=True)
    ap.add_argument("--features-cache-dir", default=None)
    ap.add_argument("--centernet-labels", default=None,
                    help="detector labels (normalized xy) used to seed sources")
    ap.add_argument("--val-patches", default="25")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bottleneck-window", type=int, default=5)
    ap.add_argument("--max-sources-per-tile", type=int, default=200)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    frozen_encoder, head = load_latent_position_head(
        args.foundation_checkpoint, device=device,
        bottleneck_window=args.bottleneck_window, stem_window=17)
    head = head.to(device)
    print(f"FieldHead (NISP v0): {sum(p.numel() for p in head.parameters() if p.requires_grad)/1e6:.2f}M trainable")

    labels = None
    if args.centernet_labels:
        L = torch.load(args.centernet_labels, map_location="cpu", weights_only=False)
        labels = L["labels"] if "labels" in L else L
        print(f"Seeding from detector labels for {len(labels)} tiles")

    pairs = discover_pairs(args.rubin_dir, args.euclid_dir)
    val_patches = {s.strip() for s in args.val_patches.split(",") if s.strip()}
    train_pairs = [p for p in pairs if patch_of(p[0]) not in val_patches]
    val_pairs = [p for p in pairs if patch_of(p[0]) in val_patches]
    print(f"Tiles: {len(train_pairs)} train, {len(val_pairs)} val (val patches {sorted(val_patches)})")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    def seeds_for(stem, W, H):
        if labels is not None and stem in labels:
            ent = labels[stem]
            xy = np.asarray(ent[0] if isinstance(ent, tuple) else ent, dtype=np.float32)
            return np.stack([xy[:, 0] * (W - 1), xy[:, 1] * (H - 1)], 1)
        return np.zeros((0, 2), np.float32)

    def run_epoch(prs, train: bool):
        head.train(train)
        tot, n = 0.0, 0
        for stem, rp, ep in prs:
            edata = dict(np.load(ep, allow_pickle=True))
            vwcs = WCS(safe_header_from_card_string(edata["wcs_VIS"].item()))
            H, W = np.asarray(edata["img_VIS"]).shape
            seeds = seeds_for(stem, W, H)
            if len(seeds) < 5:
                continue
            if len(seeds) > args.max_sources_per_tile:
                seeds = seeds[rng.choice(len(seeds), args.max_sources_per_tile, replace=False)]
            img_t, rms_t, vis_hw, _ = load_tile_data(str(rp), str(ep), device)
            with torch.no_grad():
                enc_out = frozen_encoder.encode_tile(img_t, rms_t)
                enc_out["vis_hw"] = vis_hw
                stem_cache = {b: band_stem_features(frozen_encoder, b, img_t, rms_t) for b in NISP_BANDS}
            for band in NISP_BANDS:
                band_img = np.nan_to_num(np.asarray(edata[IMGKEY[band]], dtype=np.float32))
                batch = build_tile_batch(frozen_encoder, head, stem_cache, enc_out, band,
                                         seeds, band_img, vwcs, device, rng)
                if batch is None:
                    continue
                pred, tgt, logsig = batch
                loss = rayleigh_nll(pred, tgt, logsig)
                if train:
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
                tot += float(loss); n += 1
            del img_t, rms_t, enc_out, stem_cache
        return tot / max(n, 1)

    best = np.inf
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(train_pairs, True)
        with torch.no_grad():
            va = run_epoch(val_pairs, False)
        sched.step()
        print(f"E {ep:2d} | train {tr:.4f} | val {va:.4f} | lr {opt.param_groups[0]['lr']:.1e}", flush=True)
        torch.save({"head_state_dict": head.state_dict(), "config": vars(args), "epoch": ep},
                   out_dir / "latest.pt")
        if va < best:
            best = va
            torch.save({"head_state_dict": head.state_dict(), "config": vars(args), "epoch": ep},
                       out_dir / "best.pt")
            print(f"  -> new best {va:.4f}")
    print("done. best val", best)


if __name__ == "__main__":
    main()
