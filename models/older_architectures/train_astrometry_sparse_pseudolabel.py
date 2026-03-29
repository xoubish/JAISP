"""
Train JAISP astrometry concordance head with pseudo-labels from source matching.

Unlike synthetic self-supervision, this uses real Rubin↔VIS source correspondences:
  - detect bright sources in Rubin and VIS images
  - match in sky coordinates using WCS
  - supervise predicted (dra, ddec) at matched VIS positions

The loss is sparse (only at matched sources), with optional smoothness regularization.
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, maximum_filter

import astropy.units as u

import sys
SCRIPT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = SCRIPT_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from jaisp_dataset_v4 import ALL_BANDS, RUBIN_BAND_ORDER, _safe_sqrt_var, _to_float32
from jaisp_foundation_v5 import JAISPFoundationV5
from head import AstrometryConcordanceHead

try:
    import wandb
except ImportError:
    wandb = None


def _robust_sigma(x: np.ndarray) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return max(1e-8, 1.4826 * mad)


def _safe_header_from_card_string(raw: str) -> fits.Header:
    """
    Build a FITS header by parsing 80-char cards and skipping malformed CONTINUE cards.
    """
    hdr = fits.Header()
    text = str(raw)
    n = (len(text) // 80) * 80
    for i in range(0, n, 80):
        chunk = text[i:i + 80]
        if not chunk.strip():
            continue
        try:
            card = fits.Card.fromstring(chunk)
        except Exception:
            continue
        key = card.keyword
        if not key or key in {"END", "CONTINUE"}:
            continue
        try:
            hdr[key] = card.value
        except Exception:
            continue
    return hdr


def _detect_sources(
    image: np.ndarray,
    nsig: float = 4.0,
    smooth_sigma: float = 1.0,
    min_dist: int = 7,
    max_sources: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basic local-peak detector with subpixel centroiding.
    """
    x = np.nan_to_num(image, nan=0.0).astype(np.float32, copy=False)
    med = float(np.median(x))
    sigma = _robust_sigma(x)

    y = gaussian_filter(x, float(max(0.0, smooth_sigma)))
    thresh = med + float(nsig) * sigma

    local_max = maximum_filter(y, size=max(3, int(min_dist)), mode="nearest")
    mask = (y == local_max) & (y > thresh)
    yy, xx = np.where(mask)
    if yy.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    score = y[yy, xx]
    order = np.argsort(score)[::-1]
    yy = yy[order]
    xx = xx[order]
    if yy.size > max_sources:
        yy = yy[:max_sources]
        xx = xx[:max_sources]

    # Subpixel centroid within 5x5 patch.
    H, W = x.shape
    x_sub = np.zeros((yy.size,), dtype=np.float32)
    y_sub = np.zeros((yy.size,), dtype=np.float32)
    for i, (yi, xi) in enumerate(zip(yy, xx)):
        y0 = max(0, int(yi) - 2)
        y1 = min(H, int(yi) + 3)
        x0 = max(0, int(xi) - 2)
        x1 = min(W, int(xi) + 3)
        patch = x[y0:y1, x0:x1]
        patch = np.clip(patch - np.percentile(patch, 30), 0.0, None)
        s = float(patch.sum())
        if s <= 0.0:
            x_sub[i] = float(xi)
            y_sub[i] = float(yi)
            continue
        gy, gx = np.mgrid[y0:y1, x0:x1]
        x_sub[i] = float((patch * gx).sum() / s)
        y_sub[i] = float((patch * gy).sum() / s)
    return x_sub, y_sub


def _match_sources_wcs(
    rubin_x: np.ndarray,
    rubin_y: np.ndarray,
    vis_x: np.ndarray,
    vis_y: np.ndarray,
    rubin_wcs: WCS,
    vis_wcs: WCS,
    max_sep_arcsec: float,
    clip_sigma: float,
    max_matches: int,
) -> Dict[str, np.ndarray]:
    if rubin_x.size == 0 or vis_x.size == 0:
        return {"rubin_xy": np.zeros((0, 2), dtype=np.float32),
                "vis_xy": np.zeros((0, 2), dtype=np.float32),
                "offsets": np.zeros((0, 2), dtype=np.float32),
                "sep_arcsec": np.zeros((0,), dtype=np.float32)}

    r_ra, r_dec = rubin_wcs.wcs_pix2world(rubin_x, rubin_y, 0)
    v_ra, v_dec = vis_wcs.wcs_pix2world(vis_x, vis_y, 0)

    c_r = SkyCoord(r_ra * u.deg, r_dec * u.deg)
    c_v = SkyCoord(v_ra * u.deg, v_dec * u.deg)

    # Mutual nearest-neighbor to reduce catastrophic mismatches.
    idx_v2r, sep_v2r, _ = c_v.match_to_catalog_sky(c_r)
    idx_r2v, _, _ = c_r.match_to_catalog_sky(c_v)

    v_idx = np.arange(c_v.size, dtype=np.int64)
    mutual = idx_r2v[idx_v2r] == v_idx
    sep_ok = sep_v2r.arcsec < float(max_sep_arcsec)
    keep = mutual & sep_ok
    if keep.sum() == 0:
        return {"rubin_xy": np.zeros((0, 2), dtype=np.float32),
                "vis_xy": np.zeros((0, 2), dtype=np.float32),
                "offsets": np.zeros((0, 2), dtype=np.float32),
                "sep_arcsec": np.zeros((0,), dtype=np.float32)}

    vis_sel = v_idx[keep]
    rub_sel = idx_v2r[keep]

    vv = c_v[vis_sel]
    rr = c_r[rub_sel]

    # Offset needed to bring Rubin into VIS frame.
    dra = (vv.ra.deg - rr.ra.deg) * np.cos(np.deg2rad(vv.dec.deg)) * 3600.0
    ddec = (vv.dec.deg - rr.dec.deg) * 3600.0
    sep = sep_v2r.arcsec[keep]

    # Sigma clipping in radial offset space.
    radial = np.hypot(dra, ddec)
    med = float(np.median(radial))
    sig = _robust_sigma(radial)
    clip = radial <= (med + float(clip_sigma) * sig)
    if clip.sum() == 0:
        return {"rubin_xy": np.zeros((0, 2), dtype=np.float32),
                "vis_xy": np.zeros((0, 2), dtype=np.float32),
                "offsets": np.zeros((0, 2), dtype=np.float32),
                "sep_arcsec": np.zeros((0,), dtype=np.float32)}

    vis_sel = vis_sel[clip]
    dra = dra[clip]
    ddec = ddec[clip]
    sep = sep[clip]

    # Keep best matches (smallest separation).
    if vis_sel.size > int(max_matches):
        order = np.argsort(sep)
        order = order[:int(max_matches)]
        vis_sel = vis_sel[order]
        dra = dra[order]
        ddec = ddec[order]
        sep = sep[order]

    rubin_xy = np.stack([rubin_x[rub_sel], rubin_y[rub_sel]], axis=1).astype(np.float32)
    vis_xy = np.stack([vis_x[vis_sel], vis_y[vis_sel]], axis=1).astype(np.float32)
    offsets = np.stack([dra, ddec], axis=1).astype(np.float32)
    sep = sep.astype(np.float32)
    return {"rubin_xy": rubin_xy, "vis_xy": vis_xy, "offsets": offsets, "sep_arcsec": sep}


class PseudoLabelAstrometryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rubin_dir: str,
        euclid_dir: str,
        rubin_band: str = "r",
        min_matches: int = 20,
        max_matches: int = 256,
        max_sep_arcsec: float = 0.12,
        clip_sigma: float = 3.5,
        rubin_nsig: float = 4.5,
        vis_nsig: float = 4.0,
        rubin_smooth: float = 1.0,
        vis_smooth: float = 1.2,
        rubin_min_dist: int = 7,
        vis_min_dist: int = 9,
        max_sources_rubin: int = 600,
        max_sources_vis: int = 800,
        finite_frac_thresh: float = 0.30,
        mmap: bool = True,
    ):
        self.rubin_dir = Path(rubin_dir)
        self.euclid_dir = Path(euclid_dir)
        self.mmap = bool(mmap)
        self.finite_frac_thresh = float(finite_frac_thresh)
        self.min_matches = int(min_matches)
        self.max_matches = int(max_matches)

        rb = str(rubin_band).strip().lower()
        if rb in RUBIN_BAND_ORDER:
            rb = f"rubin_{rb}"
        if not rb.startswith("rubin_"):
            rb = f"rubin_{rb}"
        if rb.replace("rubin_", "") not in RUBIN_BAND_ORDER:
            raise ValueError(f"Invalid rubin_band: {rubin_band}")
        self.rubin_band = rb
        self.band_idx = RUBIN_BAND_ORDER.index(rb.split("_")[1])

        rubin_files = sorted(self.rubin_dir.glob("tile_x*_y*.npz"))
        pairs: List[Tuple[str, str, str]] = []
        for rp in rubin_files:
            tid = rp.stem
            ep = self.euclid_dir / f"{tid}_euclid.npz"
            if ep.exists():
                pairs.append((tid, str(rp), str(ep)))
        if not pairs:
            raise FileNotFoundError("No tile pairs found for pseudo-label training.")

        self.samples: List[Dict] = []
        n_detect_fail = 0
        n_wcs_fail = 0
        n_low_matches = 0
        total_matches = 0

        print(f"PseudoLabelAstrometryDataset: building labels for {len(pairs)} tiles...")
        for i, (tid, rp, ep) in enumerate(pairs, start=1):
            try:
                rdata = np.load(rp, mmap_mode="r" if self.mmap else None, allow_pickle=True)
                edata = np.load(ep, mmap_mode="r" if self.mmap else None, allow_pickle=True)
            except Exception:
                n_detect_fail += 1
                continue

            try:
                rimg = _to_float32(rdata["img"][self.band_idx])
                vimg = _to_float32(edata["img_VIS"])
            except Exception:
                n_detect_fail += 1
                continue

            if np.isfinite(rimg).sum() / float(rimg.size) <= self.finite_frac_thresh:
                n_low_matches += 1
                continue
            if np.isfinite(vimg).sum() / float(vimg.size) <= self.finite_frac_thresh:
                n_low_matches += 1
                continue

            try:
                rwcs = WCS(rdata["wcs_hdr"].item())
                vhdr = _safe_header_from_card_string(edata["wcs_VIS"].item())
                vwcs = WCS(vhdr)
            except Exception:
                n_wcs_fail += 1
                continue

            try:
                rx, ry = _detect_sources(
                    rimg,
                    nsig=rubin_nsig,
                    smooth_sigma=rubin_smooth,
                    min_dist=rubin_min_dist,
                    max_sources=max_sources_rubin,
                )
                vx, vy = _detect_sources(
                    vimg,
                    nsig=vis_nsig,
                    smooth_sigma=vis_smooth,
                    min_dist=vis_min_dist,
                    max_sources=max_sources_vis,
                )
            except Exception:
                n_detect_fail += 1
                continue

            matched = _match_sources_wcs(
                rx,
                ry,
                vx,
                vy,
                rwcs,
                vwcs,
                max_sep_arcsec=max_sep_arcsec,
                clip_sigma=clip_sigma,
                max_matches=max_matches,
            )

            n = matched["vis_xy"].shape[0]
            if n < self.min_matches:
                n_low_matches += 1
                continue

            total_matches += n
            self.samples.append(
                {
                    "tile_id": tid,
                    "rubin_path": rp,
                    "euclid_path": ep,
                    "vis_xy": matched["vis_xy"],
                    "offsets": matched["offsets"],
                    "sep_arcsec": matched["sep_arcsec"],
                }
            )

            if i % 20 == 0 or i == len(pairs):
                print(f"  processed {i}/{len(pairs)} tiles, kept={len(self.samples)}")

        if not self.samples:
            raise RuntimeError("No pseudo-label samples were created; relax matching thresholds.")

        mean_matches = total_matches / float(len(self.samples))
        print(f"PseudoLabelAstrometryDataset: kept {len(self.samples)} / {len(pairs)} tiles")
        print(f"  rubin_band={self.rubin_band} min_matches={self.min_matches} max_matches={self.max_matches}")
        print(f"  mean_matches_per_tile={mean_matches:.1f}")
        print(f"  dropped: wcs_fail={n_wcs_fail} detect_fail={n_detect_fail} low_matches={n_low_matches}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        rdata = np.load(s["rubin_path"], mmap_mode="r" if self.mmap else None, allow_pickle=True)
        edata = np.load(s["euclid_path"], mmap_mode="r" if self.mmap else None, allow_pickle=True)

        rimg = _to_float32(rdata["img"][self.band_idx])
        rrms = _safe_sqrt_var(rdata["var"][self.band_idx]) if "var" in rdata else np.ones_like(rimg) * np.nanstd(rimg)
        vimg = _to_float32(edata["img_VIS"])
        vrms = _safe_sqrt_var(edata["var_VIS"]) if "var_VIS" in edata else np.ones_like(vimg) * np.nanstd(vimg)

        rimg = np.nan_to_num(rimg, nan=0.0)
        rrms = np.maximum(np.nan_to_num(rrms, nan=1.0), 1e-10)
        vimg = np.nan_to_num(vimg, nan=0.0)
        vrms = np.maximum(np.nan_to_num(vrms, nan=1.0), 1e-10)

        return {
            "tile_id": s["tile_id"],
            "rubin_band": self.rubin_band,
            "rubin_image": torch.from_numpy(rimg[None].copy()),   # [1,H,W]
            "rubin_rms": torch.from_numpy(rrms[None].copy()),
            "vis_image": torch.from_numpy(vimg[None].copy()),
            "vis_rms": torch.from_numpy(vrms[None].copy()),
            "vis_xy": torch.from_numpy(s["vis_xy"].copy()),       # [N,2] x,y in VIS pixels
            "offsets": torch.from_numpy(s["offsets"].copy()),     # [N,2] dra,ddec in arcsec
            "sep_arcsec": torch.from_numpy(s["sep_arcsec"].copy()),
        }


def collate_pseudolabel(batch: List[Dict]) -> Dict:
    return {
        "tile_id": [b["tile_id"] for b in batch],
        "rubin_band": [b["rubin_band"] for b in batch],
        "rubin_image": [b["rubin_image"] for b in batch],
        "rubin_rms": [b["rubin_rms"] for b in batch],
        "vis_image": [b["vis_image"] for b in batch],
        "vis_rms": [b["vis_rms"] for b in batch],
        "vis_xy": [b["vis_xy"] for b in batch],
        "offsets": [b["offsets"] for b in batch],
        "sep_arcsec": [b["sep_arcsec"] for b in batch],
    }


def make_loader(dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int, shuffle: bool):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_pseudolabel,
    )


def load_backbone(device, checkpoint_path, embed_dim, proj_dim, depth, patch_size):
    model = JAISPFoundationV5(
        band_names=ALL_BANDS,
        stem_ch=64,
        embed_dim=embed_dim,
        proj_dim=proj_dim,
        depth=depth,
        patch_size=patch_size,
        shift_temp=0.07,
    ).to(device)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded backbone: {checkpoint_path}")
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("No backbone checkpoint — random init.")
    return model


def encode_band(backbone, image, rms, band, device, freeze=True):
    image = image.unsqueeze(0).to(device)
    rms = rms.unsqueeze(0).to(device)
    if freeze:
        with torch.no_grad():
            feat = backbone.stems[band](image, rms)
            tokens, grid_size = backbone.encoder(feat)
    else:
        feat = backbone.stems[band](image, rms)
        tokens, grid_size = backbone.encoder(feat)
    return feat, tokens, grid_size


def _gradient_penalty(field: torch.Tensor) -> torch.Tensor:
    gx = (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean()
    gy = (field[:, :, 1:, :] - field[:, :, :-1, :]).abs().mean()
    return gx + gy


def _sample_points_bilinear(field: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """
    field: [1,1,H,W]
    xy: [N,2] in pixel coordinates (x,y) on same H,W grid.
    returns: [N]
    """
    H = field.shape[-2]
    W = field.shape[-1]
    x = xy[:, 0].clamp(0.0, float(W - 1))
    y = xy[:, 1].clamp(0.0, float(H - 1))
    x_norm = 2.0 * (x / max(1.0, float(W - 1))) - 1.0
    y_norm = 2.0 * (y / max(1.0, float(H - 1))) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)  # [1,N,1,2]
    vals = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return vals.view(-1)


def make_preview(
    vis_image: np.ndarray,
    pred_dra_map: np.ndarray,
    pred_ddec_map: np.ndarray,
    vis_xy: np.ndarray,
    gt_offsets: np.ndarray,
    pred_offsets: np.ndarray,
    sep_arcsec: np.ndarray,
    tile_id: str,
    rubin_band: str,
    epoch: int,
):
    import matplotlib.pyplot as plt

    if wandb is None:
        return None

    # Basic per-point errors in mas.
    err = pred_offsets - gt_offsets
    err_mag_mas = np.hypot(err[:, 0], err[:, 1]) * 1000.0
    gt_mag_mas = np.hypot(gt_offsets[:, 0], gt_offsets[:, 1]) * 1000.0
    pd_mag_mas = np.hypot(pred_offsets[:, 0], pred_offsets[:, 1]) * 1000.0

    # Robust image display limits.
    p1, p99 = np.percentile(vis_image, [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or p1 >= p99:
        p1, p99 = float(np.min(vis_image)), float(np.max(vis_image))
        if p1 >= p99:
            p1, p99 = 0.0, 1.0

    pred_mag_map = np.hypot(pred_dra_map, pred_ddec_map) * 1000.0
    vmax_pred = float(np.percentile(pred_mag_map, 99))
    if vmax_pred <= 0:
        vmax_pred = 1.0

    fig = plt.figure(figsize=(14, 10))

    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(vis_image, origin="lower", cmap="gray", vmin=p1, vmax=p99)
    sc = ax.scatter(vis_xy[:, 0], vis_xy[:, 1], c=err_mag_mas, s=9, cmap="magma", alpha=0.85)
    ax.set_title("VIS + matched sources (color: |residual| mas)")
    plt.colorbar(sc, ax=ax, fraction=0.046, label="mas")

    ax = fig.add_subplot(2, 2, 2)
    im = ax.imshow(pred_mag_map, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax_pred)
    # Show predicted vector field sparsely.
    H, W = pred_dra_map.shape
    step = max(1, min(H, W) // 25)
    yy = np.arange(0, H, step)
    xx = np.arange(0, W, step)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    ax.quiver(
        X,
        Y,
        pred_dra_map[::step, ::step] * 1000.0,
        pred_ddec_map[::step, ::step] * 1000.0,
        color="white",
        alpha=0.7,
        width=0.0025,
        scale=2500.0,
    )
    ax.set_title("Predicted |offset| map + field quiver")
    plt.colorbar(im, ax=ax, fraction=0.046, label="mas")

    ax = fig.add_subplot(2, 2, 3)
    n = vis_xy.shape[0]
    keep = np.arange(n)
    if n > 200:
        keep = np.random.choice(n, 200, replace=False)
    ax.quiver(
        vis_xy[keep, 0],
        vis_xy[keep, 1],
        err[keep, 0] * 1000.0,
        err[keep, 1] * 1000.0,
        angles="xy",
        scale_units="xy",
        scale=1200.0,
        color="crimson",
        alpha=0.7,
        width=0.003,
    )
    ax.set_xlim(0, vis_image.shape[1])
    ax.set_ylim(0, vis_image.shape[0])
    ax.set_title("Residual vectors (Pred - GT) at matched sources")

    ax = fig.add_subplot(2, 2, 4)
    n2 = vis_xy.shape[0]
    keep2 = np.arange(n2)
    if n2 > 1500:
        keep2 = np.random.choice(n2, 1500, replace=False)
    ax.scatter(gt_offsets[keep2, 0] * 1000.0, pred_offsets[keep2, 0] * 1000.0, s=4, alpha=0.4, label="DRA*")
    ax.scatter(gt_offsets[keep2, 1] * 1000.0, pred_offsets[keep2, 1] * 1000.0, s=4, alpha=0.4, label="DDec")
    lim = max(
        20.0,
        float(np.percentile(np.abs(gt_offsets[:, 0] * 1000.0), 99)),
        float(np.percentile(np.abs(gt_offsets[:, 1] * 1000.0), 99)),
        float(np.percentile(np.abs(pred_offsets[:, 0] * 1000.0), 99)),
        float(np.percentile(np.abs(pred_offsets[:, 1] * 1000.0), 99)),
    )
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("GT (mas)")
    ax.set_ylabel("Pred (mas)")
    ax.legend(loc="upper left")
    ax.set_title("Pointwise offsets: GT vs Pred")

    med_err = float(np.median(err_mag_mas))
    p68_err = float(np.percentile(err_mag_mas, 68))
    med_sep = float(np.median(sep_arcsec) * 1000.0) if sep_arcsec.size else float("nan")
    fig.suptitle(
        f"Epoch {epoch} | {tile_id} | {rubin_band} -> VIS | "
        f"residual median={med_err:.1f} mas p68={p68_err:.1f} mas | "
        f"match sep median={med_sep:.1f} mas | "
        f"GT|offset| median={float(np.median(gt_mag_mas)):.1f} mas Pred median={float(np.median(pd_mag_mas)):.1f} mas"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = wandb.Image(fig)
    plt.close(fig)
    return out


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PseudoLabelAstrometryDataset(
        rubin_dir=args.rubin_dir,
        euclid_dir=args.euclid_dir,
        rubin_band=args.rubin_band,
        min_matches=args.min_matches,
        max_matches=args.max_matches,
        max_sep_arcsec=args.max_sep_arcsec,
        clip_sigma=args.clip_sigma,
        rubin_nsig=args.rubin_nsig,
        vis_nsig=args.vis_nsig,
        rubin_smooth=args.rubin_smooth,
        vis_smooth=args.vis_smooth,
        rubin_min_dist=args.rubin_min_dist,
        vis_min_dist=args.vis_min_dist,
        max_sources_rubin=args.max_sources_rubin,
        max_sources_vis=args.max_sources_vis,
        finite_frac_thresh=args.finite_frac_thresh,
        mmap=not args.no_mmap,
    )
    loader = make_loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    backbone = load_backbone(
        device, args.backbone_ckpt, args.embed_dim, args.proj_dim, args.depth, args.patch_size
    )
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    print("Backbone frozen.")

    head = AstrometryConcordanceHead(
        embed_dim=args.embed_dim,
        search_radius=args.search_radius,
        softmax_temp=args.softmax_temp,
        patch_size=args.patch_size,
        global_hidden=args.global_hidden,
        local_hidden=args.local_hidden,
        local_depth=args.local_depth,
        use_stem_refine=args.use_stem_refine,
        match_dim=args.match_dim,
        residual_gain_init=args.residual_gain_init,
        stem_channels=args.stem_channels,
        stem_hidden=args.stem_hidden,
        stem_depth=args.stem_depth,
        stem_stride=args.stem_stride,
    ).to(device)

    n_params = sum(p.numel() for p in head.parameters())
    n_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"Head parameters: {n_params:,} (trainable={n_trainable:,})")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    wandb_run = None
    if args.wandb_mode != "disabled" and wandb is not None:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or None,
                config=vars(args),
                mode=args.wandb_mode,
                dir=str(out_dir),
            )
        except Exception as e:
            print(f"W&B init failed: {e}")

    best_loss = float("inf")
    global_step = 0
    vis_pixel_scale = 0.1

    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            head.train()
            agg = defaultdict(float)
            n_samples = 0
            n_points = 0
            preview_data = None

            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                sample_losses = []

                for i in range(len(batch["tile_id"])):
                    rubin_band = batch["rubin_band"][i]
                    if rubin_band not in backbone.band_names:
                        continue

                    rubin_img = batch["rubin_image"][i].float().to(device)
                    rubin_rms = batch["rubin_rms"][i].float().to(device)
                    vis_img = batch["vis_image"][i].float().to(device)
                    vis_rms = batch["vis_rms"][i].float().to(device)
                    vis_xy = batch["vis_xy"][i].float().to(device)       # [N,2]
                    gt_off = batch["offsets"][i].float().to(device)      # [N,2]
                    sep = batch["sep_arcsec"][i].float().to(device)      # [N]

                    if vis_xy.shape[0] < args.min_matches:
                        continue

                    rubin_feat, rubin_tokens, rubin_grid = encode_band(
                        backbone, rubin_img, rubin_rms, rubin_band, device, freeze=True
                    )
                    vis_feat, vis_tokens, vis_grid = encode_band(
                        backbone, vis_img, vis_rms, "euclid_VIS", device, freeze=True
                    )

                    H_vis, W_vis = vis_img.shape[-2], vis_img.shape[-1]
                    out = head(
                        rubin_tokens=rubin_tokens,
                        vis_tokens=vis_tokens,
                        rubin_grid=rubin_grid,
                        vis_grid=vis_grid,
                        vis_image_hw=(H_vis, W_vis),
                        vis_pixel_scale=vis_pixel_scale,
                        rubin_stem=rubin_feat if args.use_stem_refine else None,
                        vis_stem=vis_feat if args.use_stem_refine else None,
                    )

                    # Convert predictions to full VIS grid before sparse sampling.
                    pred_dra = F.interpolate(out["dra"], size=(H_vis, W_vis), mode="bilinear", align_corners=False)
                    pred_ddec = F.interpolate(out["ddec"], size=(H_vis, W_vis), mode="bilinear", align_corners=False)
                    pred_dra_pts = _sample_points_bilinear(pred_dra, vis_xy)
                    pred_ddec_pts = _sample_points_bilinear(pred_ddec, vis_xy)

                    # Separation-based weights: closer matches get slightly higher trust.
                    sep_scale = max(1e-4, args.max_sep_arcsec)
                    w = torch.exp(-0.5 * (sep / sep_scale) ** 2)
                    w = w / w.mean().clamp_min(1e-6)

                    l_ra = F.smooth_l1_loss(pred_dra_pts, gt_off[:, 0], reduction="none")
                    l_dec = F.smooth_l1_loss(pred_ddec_pts, gt_off[:, 1], reduction="none")
                    loss_sparse = (w * (l_ra + l_dec)).mean()

                    loss_smooth = _gradient_penalty(pred_dra) + _gradient_penalty(pred_ddec)
                    loss_total = loss_sparse + args.smooth_weight * loss_smooth

                    sample_losses.append(loss_total)

                    with torch.no_grad():
                        e_ra = (pred_dra_pts - gt_off[:, 0]).abs()
                        e_dec = (pred_ddec_pts - gt_off[:, 1]).abs()
                        e_tot = torch.sqrt(e_ra ** 2 + e_dec ** 2 + 1e-10)
                        agg["loss_total"] += float(loss_total.item())
                        agg["loss_sparse"] += float(loss_sparse.item())
                        agg["loss_smooth"] += float(loss_smooth.item())
                        agg["mae_ra"] += float(e_ra.mean().item())
                        agg["mae_dec"] += float(e_dec.mean().item())
                        agg["mae_total"] += float(e_tot.mean().item())
                        agg["p68_total"] += float(torch.quantile(e_tot, 0.68).item())
                        agg["frac_01arcsec"] += float((e_tot < 0.1).float().mean().item())
                        agg["frac_02arcsec"] += float((e_tot < 0.2).float().mean().item())
                        n_samples += 1
                        n_points += int(vis_xy.shape[0])
                        if (
                            preview_data is None
                            and wandb_run is not None
                            and args.vis_every > 0
                            and (epoch % args.vis_every == 0)
                        ):
                            preview_data = {
                                "vis_image": vis_img[0].detach().cpu().numpy(),
                                "pred_dra_map": pred_dra[0, 0].detach().cpu().numpy(),
                                "pred_ddec_map": pred_ddec[0, 0].detach().cpu().numpy(),
                                "vis_xy": vis_xy.detach().cpu().numpy(),
                                "gt_offsets": gt_off.detach().cpu().numpy(),
                                "pred_offsets": torch.stack([pred_dra_pts, pred_ddec_pts], dim=1).detach().cpu().numpy(),
                                "sep_arcsec": sep.detach().cpu().numpy(),
                                "tile_id": batch["tile_id"][i],
                                "rubin_band": rubin_band,
                            }

                if not sample_losses:
                    continue
                loss = torch.stack(sample_losses).mean()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
                optimizer.step()
                global_step += 1

            scheduler.step()

            n = max(1, n_samples)
            epoch_metrics = {k: v / n for k, v in agg.items()}
            epoch_metrics["epoch"] = epoch
            epoch_metrics["samples"] = n_samples
            epoch_metrics["points"] = n_points
            epoch_metrics["avg_points"] = n_points / float(max(1, n_samples))
            epoch_metrics["lr"] = optimizer.param_groups[0]["lr"]
            if hasattr(head, "temperature"):
                try:
                    epoch_metrics["temp"] = float(head.temperature.detach().item())
                except Exception:
                    pass
            if hasattr(head, "residual_gain"):
                try:
                    epoch_metrics["residual_gain"] = float(head.residual_gain.detach().item())
                except Exception:
                    pass
            epoch_metrics["time_sec"] = time.time() - t0

            temp_str = f" temp={epoch_metrics['temp']:.4f}" if "temp" in epoch_metrics else ""
            rg_str = f" rg={epoch_metrics['residual_gain']:.3f}" if "residual_gain" in epoch_metrics else ""
            print(
                f"Epoch {epoch:03d} | "
                f"loss={epoch_metrics.get('loss_total', 0.0):.5f} "
                f"MAE_total={epoch_metrics.get('mae_total', 0.0)*1000:.1f}mas "
                f"MAE_ra={epoch_metrics.get('mae_ra', 0.0)*1000:.1f}mas "
                f"MAE_dec={epoch_metrics.get('mae_dec', 0.0)*1000:.1f}mas "
                f"p68={epoch_metrics.get('p68_total', 0.0)*1000:.1f}mas "
                f"<0.1\"={epoch_metrics.get('frac_01arcsec', 0.0):.1%} "
                f"<0.2\"={epoch_metrics.get('frac_02arcsec', 0.0):.1%} "
                f"pts={epoch_metrics['points']} avg_pts={epoch_metrics['avg_points']:.1f} "
                f"{temp_str}{rg_str} "
                f"t={epoch_metrics['time_sec']:.1f}s"
            )

            if wandb_run is not None:
                wb = {f"train/{k}": v for k, v in epoch_metrics.items()}
                if preview_data is not None:
                    try:
                        img = make_preview(
                            vis_image=preview_data["vis_image"],
                            pred_dra_map=preview_data["pred_dra_map"],
                            pred_ddec_map=preview_data["pred_ddec_map"],
                            vis_xy=preview_data["vis_xy"],
                            gt_offsets=preview_data["gt_offsets"],
                            pred_offsets=preview_data["pred_offsets"],
                            sep_arcsec=preview_data["sep_arcsec"],
                            tile_id=preview_data["tile_id"],
                            rubin_band=preview_data["rubin_band"],
                            epoch=epoch,
                        )
                        if img is not None:
                            wb["train/preview"] = img
                    except Exception as e:
                        print(f"Preview rendering failed at epoch {epoch}: {e}")
                wandb_run.log(wb, step=epoch)

            ckpt = {
                "epoch": epoch,
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": epoch_metrics,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "last_astrometry.pt")

            score = epoch_metrics.get("loss_total", float("inf"))
            if score < best_loss:
                best_loss = score
                torch.save(ckpt, out_dir / "best_astrometry.pt")
                if wandb_run is not None:
                    wandb_run.summary["best_mae_total_mas"] = epoch_metrics.get("mae_total", 0.0) * 1000.0
                    wandb_run.summary["best_epoch"] = epoch

            if args.save_every > 0 and epoch % args.save_every == 0:
                torch.save(ckpt, out_dir / f"astrometry_epoch_{epoch:03d}.pt")

            with open(out_dir / "latest_metrics.json", "w") as f:
                json.dump(epoch_metrics, f, indent=2)

    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print(f"\nDone. Best loss: {best_loss:.5f}")
    print(f"Checkpoints in: {out_dir}")


def build_parser():
    p = argparse.ArgumentParser(description="Train astrometry head with pseudo-labels from source matching.")
    p.add_argument("--rubin-dir", type=str, required=True)
    p.add_argument("--euclid-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="models/checkpoints/jaisp_astrometry_pseudolabel")
    p.add_argument("--backbone-ckpt", type=str, default="models/checkpoints/jaisp_v5/best.pt")

    p.add_argument("--rubin-band", type=str, default="r",
                   help="Rubin band used for pseudo-label extraction (u/g/r/i/z/y or rubin_u..rubin_y).")
    p.add_argument("--min-matches", type=int, default=20)
    p.add_argument("--max-matches", type=int, default=256)
    p.add_argument("--max-sep-arcsec", type=float, default=0.12)
    p.add_argument("--clip-sigma", type=float, default=3.5)

    p.add_argument("--rubin-nsig", type=float, default=4.5)
    p.add_argument("--vis-nsig", type=float, default=4.0)
    p.add_argument("--rubin-smooth", type=float, default=1.0)
    p.add_argument("--vis-smooth", type=float, default=1.2)
    p.add_argument("--rubin-min-dist", type=int, default=7)
    p.add_argument("--vis-min-dist", type=int, default=9)
    p.add_argument("--max-sources-rubin", type=int, default=600)
    p.add_argument("--max-sources-vis", type=int, default=800)
    p.add_argument("--finite-frac-thresh", type=float, default=0.30)
    p.add_argument("--no-mmap", action="store_true")

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--smooth-weight", type=float, default=0.02)

    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--patch-size", type=int, default=16)

    p.add_argument("--search-radius", type=int, default=3)
    p.add_argument("--softmax-temp", type=float, default=0.1)
    p.add_argument("--global-hidden", type=int, default=128)
    p.add_argument("--local-hidden", type=int, default=64)
    p.add_argument("--local-depth", type=int, default=5)
    p.add_argument("--match-dim", type=int, default=64)
    p.add_argument("--residual-gain-init", type=float, default=1.2)
    p.add_argument("--use-stem-refine", action="store_true")
    p.add_argument("--stem-channels", type=int, default=64)
    p.add_argument("--stem-hidden", type=int, default=32)
    p.add_argument("--stem-depth", type=int, default=4)
    p.add_argument("--stem-stride", type=int, default=2)

    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--vis-every", type=int, default=1,
                   help="Log W&B preview every N epochs (<=0 disables).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")

    p.add_argument("--wandb-project", type=str, default="JAISP-Astrometry")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    train(args)
