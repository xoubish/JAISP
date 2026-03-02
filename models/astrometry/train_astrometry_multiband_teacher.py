"""
Train the astrometry concordance head against dense smooth teacher fields.

This is the intended neural path:
  - frozen JAISP foundation backbone
  - AstrometryConcordanceHead on top
  - dense teacher field built from matched Rubin<->VIS sources
  - multiband Rubin context fused in latent space, with the target band emphasized

Compared with the older trainers, this avoids synthetic tiny-warp supervision and
avoids asking the head to interpolate directly from sparse matched points.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from astropy.wcs import WCS

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from jaisp_dataset_v4 import ALL_BANDS, RUBIN_BAND_ORDER, _safe_sqrt_var, _to_float32
from jaisp_foundation_v5 import JAISPFoundationV5
from head import AstrometryConcordanceHead, PIXEL_SCALES, interpolate_tokens
from source_matching import safe_header_from_card_string
from teacher_fields import build_teacher_field, normalize_band_list, normalize_band_name

try:
    import wandb
except ImportError:
    wandb = None


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
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _finite_fraction(x: np.ndarray) -> float:
    return float(np.isfinite(x).sum()) / float(max(1, x.size))


def discover_tile_pairs(rubin_dir: str, euclid_dir: str) -> List[Tuple[str, str, str]]:
    rubin_root = Path(rubin_dir)
    euclid_root = Path(euclid_dir)
    pairs = []
    for rubin_path in sorted(rubin_root.glob("tile_x*_y*.npz")):
        tile_id = rubin_path.stem
        euclid_path = euclid_root / f"{tile_id}_euclid.npz"
        if euclid_path.exists():
            pairs.append((tile_id, str(rubin_path), str(euclid_path)))
    if not pairs:
        raise FileNotFoundError("No tile pairs found.")
    return pairs


def split_tile_pairs(
    pairs: Sequence[Tuple[str, str, str]],
    val_frac: float,
    seed: int,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    pairs = list(pairs)
    if val_frac <= 0.0 or len(pairs) <= 1:
        return pairs, []
    rng = np.random.RandomState(int(seed))
    order = np.arange(len(pairs))
    rng.shuffle(order)
    n_val = max(1, int(round(len(pairs) * float(val_frac))))
    n_val = min(n_val, len(pairs) - 1)
    val_idx = set(order[:n_val].tolist())
    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in val_idx]
    val_pairs = [pairs[i] for i in range(len(pairs)) if i in val_idx]
    return train_pairs, val_pairs


def build_teacher_samples(
    pairs: Sequence[Tuple[str, str, str]],
    target_bands: List[str],
    args,
    split_name: str,
) -> List[Dict]:
    samples: List[Dict] = []
    n_seen = 0
    n_kept = 0
    for tile_id, rubin_path, euclid_path in pairs:
        n_seen += 1
        try:
            rdata = np.load(rubin_path, allow_pickle=True)
            edata = np.load(euclid_path, allow_pickle=True)
            rubin_cube = rdata["img"]
            vis_img = _to_float32(edata["img_VIS"])
            rwcs = WCS(rdata["wcs_hdr"].item())
            vhdr = safe_header_from_card_string(edata["wcs_VIS"].item())
            vwcs = WCS(vhdr)
        except Exception as exc:
            print(f"[{split_name}] skip {tile_id}: load/wcs failed ({exc})")
            continue

        for target_band in target_bands:
            try:
                built = build_teacher_field(
                    rubin_cube=rubin_cube,
                    vis_img=vis_img,
                    rubin_wcs=rwcs,
                    vis_wcs=vwcs,
                    rubin_band=target_band,
                    detect_bands=args.detect_bands,
                    dstep=args.teacher_dstep,
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
                    detect_clip_sigma=args.detect_clip_sigma,
                    refine_band_centroids=not args.no_refine_band_centroids,
                    refine_radius=args.refine_radius,
                    refine_flux_floor_sigma=args.refine_flux_floor_sigma,
                    rbf_smoothing=args.rbf_smoothing,
                    rbf_neighbors=args.rbf_neighbors,
                    rbf_kernel=args.rbf_kernel,
                )
            except Exception as exc:
                print(f"[{split_name}] skip {tile_id}:{target_band} teacher build failed ({exc})")
                continue

            if built is None:
                continue

            matched = built["matched"]
            fit = built["fit"]
            raw_off_mas = np.hypot(matched["offsets"][:, 0], matched["offsets"][:, 1]) * 1000.0
            fit_resid_mas = fit["point_resid_mas"]
            samples.append(
                {
                    "tile_id": tile_id,
                    "rubin_path": rubin_path,
                    "euclid_path": euclid_path,
                    "target_band": normalize_band_name(target_band),
                    "teacher_dra": fit["dra"].astype(np.float32, copy=False),
                    "teacher_ddec": fit["ddec"].astype(np.float32, copy=False),
                    "teacher_x_mesh": fit["x_mesh"].astype(np.float32, copy=False),
                    "teacher_y_mesh": fit["y_mesh"].astype(np.float32, copy=False),
                    "matched_vis_xy": matched["vis_xy"].astype(np.float32, copy=False),
                    "matched_offsets": matched["offsets"].astype(np.float32, copy=False),
                    "match_count": int(matched["vis_xy"].shape[0]),
                    "raw_median_mas": float(np.median(raw_off_mas)),
                    "fit_resid_median_mas": float(np.median(fit_resid_mas)),
                }
            )
            n_kept += 1

        if n_seen % 20 == 0 or n_seen == len(pairs):
            print(f"[{split_name}] processed {n_seen}/{len(pairs)} tiles, teacher samples={n_kept}")

    if not samples:
        raise RuntimeError(f"No teacher samples created for split '{split_name}'.")

    match_counts = np.array([s["match_count"] for s in samples], dtype=np.float32)
    raw_medians = np.array([s["raw_median_mas"] for s in samples], dtype=np.float32)
    fit_resid_medians = np.array([s["fit_resid_median_mas"] for s in samples], dtype=np.float32)
    print(
        f"[{split_name}] kept {len(samples)} teacher samples from {len(pairs)} tiles | "
        f"matches median={np.median(match_counts):.1f} | "
        f"raw median={np.median(raw_medians):.1f} mas | "
        f"fit residual median={np.median(fit_resid_medians):.1f} mas"
    )
    return samples


class TeacherFieldDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: List[Dict],
        context_bands: List[str],
        finite_frac_thresh: float = 0.30,
        mmap: bool = True,
    ):
        self.samples = list(samples)
        self.context_bands = list(context_bands)
        self.finite_frac_thresh = float(finite_frac_thresh)
        self.mmap = bool(mmap)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        rdata = np.load(sample["rubin_path"], mmap_mode="r" if self.mmap else None, allow_pickle=True)
        edata = np.load(sample["euclid_path"], mmap_mode="r" if self.mmap else None, allow_pickle=True)

        requested = []
        seen = set()
        target_band = sample["target_band"]
        for band in [target_band] + list(self.context_bands):
            band = normalize_band_name(band)
            if band in seen:
                continue
            seen.add(band)
            requested.append(band)

        rubin_images = []
        rubin_rms = []
        input_bands = []
        for band in requested:
            band_idx = RUBIN_BAND_ORDER.index(band.split("_", 1)[1])
            if band_idx >= rdata["img"].shape[0]:
                continue
            img = _to_float32(rdata["img"][band_idx])
            if _finite_fraction(img) <= self.finite_frac_thresh:
                continue
            rms = (
                _safe_sqrt_var(rdata["var"][band_idx])
                if "var" in rdata
                else np.ones_like(img, dtype=np.float32) * max(1e-6, float(np.nanstd(img)))
            )
            img = np.nan_to_num(img, nan=0.0)
            rms = np.maximum(np.nan_to_num(rms, nan=1.0), 1e-10)
            rubin_images.append(torch.from_numpy(img[None].copy()))
            rubin_rms.append(torch.from_numpy(rms[None].copy()))
            input_bands.append(band)

        if target_band not in input_bands:
            band_idx = RUBIN_BAND_ORDER.index(target_band.split("_", 1)[1])
            img = _to_float32(rdata["img"][band_idx])
            rms = (
                _safe_sqrt_var(rdata["var"][band_idx])
                if "var" in rdata
                else np.ones_like(img, dtype=np.float32) * max(1e-6, float(np.nanstd(img)))
            )
            img = np.nan_to_num(img, nan=0.0)
            rms = np.maximum(np.nan_to_num(rms, nan=1.0), 1e-10)
            rubin_images.insert(0, torch.from_numpy(img[None].copy()))
            rubin_rms.insert(0, torch.from_numpy(rms[None].copy()))
            input_bands.insert(0, target_band)

        vis_img = _to_float32(edata["img_VIS"])
        vis_rms = (
            _safe_sqrt_var(edata["var_VIS"])
            if "var_VIS" in edata
            else np.ones_like(vis_img, dtype=np.float32) * max(1e-6, float(np.nanstd(vis_img)))
        )
        vis_img = np.nan_to_num(vis_img, nan=0.0)
        vis_rms = np.maximum(np.nan_to_num(vis_rms, nan=1.0), 1e-10)

        return {
            "tile_id": sample["tile_id"],
            "target_band": target_band,
            "input_bands": input_bands,
            "rubin_images": rubin_images,
            "rubin_rms": rubin_rms,
            "vis_image": torch.from_numpy(vis_img[None].copy()),
            "vis_rms": torch.from_numpy(vis_rms[None].copy()),
            "teacher_dra": torch.from_numpy(sample["teacher_dra"].copy()),
            "teacher_ddec": torch.from_numpy(sample["teacher_ddec"].copy()),
            "teacher_x_mesh": torch.from_numpy(sample["teacher_x_mesh"].copy()),
            "teacher_y_mesh": torch.from_numpy(sample["teacher_y_mesh"].copy()),
            "matched_vis_xy": torch.from_numpy(sample["matched_vis_xy"].copy()),
            "matched_offsets": torch.from_numpy(sample["matched_offsets"].copy()),
            "match_count": sample["match_count"],
        }


def collate_teacher(batch: List[Dict]) -> Dict:
    return {
        "tile_id": [b["tile_id"] for b in batch],
        "target_band": [b["target_band"] for b in batch],
        "input_bands": [b["input_bands"] for b in batch],
        "rubin_images": [b["rubin_images"] for b in batch],
        "rubin_rms": [b["rubin_rms"] for b in batch],
        "vis_image": [b["vis_image"] for b in batch],
        "vis_rms": [b["vis_rms"] for b in batch],
        "teacher_dra": [b["teacher_dra"] for b in batch],
        "teacher_ddec": [b["teacher_ddec"] for b in batch],
        "teacher_x_mesh": [b["teacher_x_mesh"] for b in batch],
        "teacher_y_mesh": [b["teacher_y_mesh"] for b in batch],
        "matched_vis_xy": [b["matched_vis_xy"] for b in batch],
        "matched_offsets": [b["matched_offsets"] for b in batch],
        "match_count": [b["match_count"] for b in batch],
    }


def make_loader(dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int, shuffle: bool):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_teacher,
    )


def encode_single_band(backbone, image, rms, band, device, freeze: bool = True):
    image = image.unsqueeze(0).to(device)
    rms = rms.unsqueeze(0).to(device)
    if freeze:
        with torch.no_grad():
            feat = backbone.stems[band](image, rms)
            tokens, grid = backbone.encoder(feat)
    else:
        feat = backbone.stems[band](image, rms)
        tokens, grid = backbone.encoder(feat)
    return feat, tokens, grid


def encode_multiband_rubin(
    backbone,
    images: List[torch.Tensor],
    rms_list: List[torch.Tensor],
    bands: List[str],
    target_band: str,
    device: torch.device,
    target_band_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], List[str]]:
    feats = []
    tokens = []
    grids = []
    used_bands = []
    for img, rms, band in zip(images, rms_list, bands):
        feat, tok, grid = encode_single_band(backbone, img.float(), rms.float(), band, device, freeze=True)
        feats.append(feat)
        tokens.append(tok)
        grids.append(grid)
        used_bands.append(band)

    if not tokens:
        raise RuntimeError("No Rubin bands available for multiband encoding.")

    common_grid = (
        max(int(g[0]) for g in grids),
        max(int(g[1]) for g in grids),
    )
    token_stack = [interpolate_tokens(tok, grid, common_grid) for tok, grid in zip(tokens, grids)]
    target_idx = used_bands.index(target_band) if target_band in used_bands else 0
    target_tokens = token_stack[target_idx]
    other_tokens = [tok for i, tok in enumerate(token_stack) if i != target_idx]
    context_tokens = target_tokens if not other_tokens else torch.stack(other_tokens, dim=0).mean(dim=0)
    alpha = float(np.clip(target_band_weight, 0.0, 1.0))
    fused_tokens = alpha * target_tokens + (1.0 - alpha) * context_tokens

    common_stem_hw = (
        max(int(x.shape[-2]) for x in feats),
        max(int(x.shape[-1]) for x in feats),
    )
    feat_stack = [
        x if x.shape[-2:] == common_stem_hw else F.interpolate(x, size=common_stem_hw, mode="bilinear", align_corners=False)
        for x in feats
    ]
    target_feat = feat_stack[target_idx]
    other_feats = [x for i, x in enumerate(feat_stack) if i != target_idx]
    context_feat = target_feat if not other_feats else torch.stack(other_feats, dim=0).mean(dim=0)
    fused_feat = alpha * target_feat + (1.0 - alpha) * context_feat
    return fused_feat, fused_tokens, common_grid, used_bands


def _gradient_penalty(field: torch.Tensor) -> torch.Tensor:
    gx = (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean()
    gy = (field[:, :, 1:, :] - field[:, :, :-1, :]).abs().mean()
    return gx + gy


def _compute_metrics(pred_dra, pred_ddec, gt_dra, gt_ddec) -> Dict[str, float]:
    with torch.no_grad():
        err_ra = (pred_dra - gt_dra).abs()
        err_dec = (pred_ddec - gt_ddec).abs()
        err_tot = torch.sqrt(err_ra ** 2 + err_dec ** 2 + 1e-10)
        return {
            "mae_ra": float(err_ra.mean().item()),
            "mae_dec": float(err_dec.mean().item()),
            "mae_total": float(err_tot.mean().item()),
            "p68_total": float(torch.quantile(err_tot.view(-1), 0.68).item()),
            "frac_01arcsec": float((err_tot < 0.1).float().mean().item()),
            "frac_02arcsec": float((err_tot < 0.2).float().mean().item()),
        }


def make_preview(sample: Dict, pred_dra_map: np.ndarray, pred_ddec_map: np.ndarray, epoch: int, split_name: str):
    if wandb is None:
        return None

    import matplotlib.pyplot as plt

    vis = sample["vis_image"][0].numpy()
    gt_dra = sample["teacher_dra"].numpy()
    gt_ddec = sample["teacher_ddec"].numpy()
    x_mesh = sample["teacher_x_mesh"].numpy()
    y_mesh = sample["teacher_y_mesh"].numpy()
    matched_xy = sample["matched_vis_xy"].numpy()
    matched_offsets = sample["matched_offsets"].numpy()

    gt_mag = np.hypot(gt_dra, gt_ddec) * 1000.0
    pred_mag = np.hypot(pred_dra_map, pred_ddec_map) * 1000.0
    resid_dra = pred_dra_map - gt_dra
    resid_ddec = pred_ddec_map - gt_ddec
    resid_mag = np.hypot(resid_dra, resid_ddec) * 1000.0
    raw_mag = np.hypot(matched_offsets[:, 0], matched_offsets[:, 1]) * 1000.0

    p1, p99 = np.percentile(vis, [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or p1 >= p99:
        p1 = float(np.min(vis))
        p99 = float(np.max(vis))
        if p1 >= p99:
            p1, p99 = 0.0, 1.0

    yy, xx = np.meshgrid(y_mesh, x_mesh, indexing="ij")
    step = max(1, min(gt_dra.shape[0], gt_dra.shape[1]) // 24)
    xx_s = xx[::step, ::step]
    yy_s = yy[::step, ::step]

    gt_u = gt_dra[::step, ::step] / PIXEL_SCALES["euclid_VIS"]
    gt_v = gt_ddec[::step, ::step] / PIXEL_SCALES["euclid_VIS"]
    pd_u = pred_dra_map[::step, ::step] / PIXEL_SCALES["euclid_VIS"]
    pd_v = pred_ddec_map[::step, ::step] / PIXEL_SCALES["euclid_VIS"]
    rs_u = resid_dra[::step, ::step] / PIXEL_SCALES["euclid_VIS"]
    rs_v = resid_ddec[::step, ::step] / PIXEL_SCALES["euclid_VIS"]
    quiver_norm = max(
        1.0,
        float(np.percentile(np.hypot(gt_u, gt_v), 95)),
        float(np.percentile(np.hypot(pd_u, pd_v), 95)),
    )
    scale_div = max(1.0, quiver_norm / 14.0)

    fig = plt.figure(figsize=(16, 10))

    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(vis, origin="lower", cmap="gray", vmin=p1, vmax=p99)
    sc = ax.scatter(matched_xy[:, 0], matched_xy[:, 1], c=raw_mag, s=12, cmap="magma", alpha=0.85)
    ax.set_title("VIS + matched sources (raw |offset|)")
    plt.colorbar(sc, ax=ax, fraction=0.046, label="mas")

    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(vis, origin="lower", cmap="gray", vmin=p1, vmax=p99)
    ax.quiver(
        xx_s,
        yy_s,
        gt_u / scale_div,
        gt_v / scale_div,
        color="deepskyblue",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0025,
        alpha=0.9,
    )
    ax.set_title("Teacher field (quiver on VIS)")

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(vis, origin="lower", cmap="gray", vmin=p1, vmax=p99)
    ax.quiver(
        xx_s,
        yy_s,
        pd_u / scale_div,
        pd_v / scale_div,
        color="white",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0025,
        alpha=0.9,
    )
    ax.set_title("Predicted field (quiver on VIS)")

    ax = fig.add_subplot(2, 3, 4)
    im = ax.imshow(resid_mag, origin="lower", cmap="magma")
    mesh_x_idx, mesh_y_idx = np.meshgrid(
        np.arange(0, resid_mag.shape[1], step),
        np.arange(0, resid_mag.shape[0], step),
        indexing="xy",
    )
    ax.quiver(
        mesh_x_idx,
        mesh_y_idx,
        rs_u / scale_div,
        rs_v / scale_div,
        color="white",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0025,
        alpha=0.8,
    )
    ax.set_title("Residual |Pred - Teacher| on mesh")
    plt.colorbar(im, ax=ax, fraction=0.046, label="mas")

    ax = fig.add_subplot(2, 3, 5)
    n_pts = min(3000, gt_dra.size)
    idx = np.random.choice(gt_dra.size, n_pts, replace=False)
    gt_ra_flat = gt_dra.ravel()[idx] * 1000.0
    gt_dec_flat = gt_ddec.ravel()[idx] * 1000.0
    pd_ra_flat = pred_dra_map.ravel()[idx] * 1000.0
    pd_dec_flat = pred_ddec_map.ravel()[idx] * 1000.0
    ax.scatter(gt_ra_flat, pd_ra_flat, s=3, alpha=0.35, label="DRA*")
    ax.scatter(gt_dec_flat, pd_dec_flat, s=3, alpha=0.35, label="DDec")
    lim = max(20.0, float(np.percentile(np.abs(np.concatenate([gt_ra_flat, gt_dec_flat, pd_ra_flat, pd_dec_flat])), 99)))
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Teacher (mas)")
    ax.set_ylabel("Pred (mas)")
    ax.legend(loc="upper left")
    ax.set_title("Mesh values: Teacher vs Pred")

    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    summary = (
        f"{split_name} preview\n"
        f"tile: {sample['tile_id']}\n"
        f"target: {sample['target_band']} -> euclid_VIS\n"
        f"input: {', '.join(sample['input_bands'])}\n"
        f"matches: {sample['match_count']}\n"
        f"teacher |offset| median: {float(np.median(gt_mag)):.1f} mas\n"
        f"pred |offset| median: {float(np.median(pred_mag)):.1f} mas\n"
        f"residual median: {float(np.median(resid_mag)):.1f} mas\n"
        f"residual p68: {float(np.percentile(resid_mag, 68)):.1f} mas\n"
        f"quiver scale divisor: {scale_div:.2f}"
    )
    ax.text(
        0.05,
        0.95,
        summary,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.95),
    )

    fig.suptitle(f"Epoch {epoch} | {split_name} | teacher-supervised multiband concordance")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = wandb.Image(fig)
    plt.close(fig)
    return out


def run_model_on_sample(
    backbone,
    head,
    sample: Dict,
    device: torch.device,
    target_band_weight: float,
    use_stem_refine: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], List[str]]:
    rubin_feat, rubin_tokens, rubin_grid, used_bands = encode_multiband_rubin(
        backbone,
        sample["rubin_images"],
        sample["rubin_rms"],
        sample["input_bands"],
        sample["target_band"],
        device=device,
        target_band_weight=target_band_weight,
    )
    vis_feat, vis_tokens, vis_grid = encode_single_band(
        backbone,
        sample["vis_image"].float(),
        sample["vis_rms"].float(),
        "euclid_VIS",
        device,
        freeze=True,
    )

    vis_h = int(sample["vis_image"].shape[-2])
    vis_w = int(sample["vis_image"].shape[-1])
    out = head(
        rubin_tokens=rubin_tokens,
        vis_tokens=vis_tokens,
        rubin_grid=rubin_grid,
        vis_grid=vis_grid,
        vis_image_hw=(vis_h, vis_w),
        vis_pixel_scale=PIXEL_SCALES["euclid_VIS"],
        rubin_stem=rubin_feat if use_stem_refine else None,
        vis_stem=vis_feat if use_stem_refine else None,
    )

    target_hw = tuple(int(x) for x in sample["teacher_dra"].shape[-2:])
    pred_dra = F.interpolate(out["dra"], size=target_hw, mode="bilinear", align_corners=False)
    pred_ddec = F.interpolate(out["ddec"], size=target_hw, mode="bilinear", align_corners=False)
    return pred_dra, pred_ddec, out, used_bands


def evaluate_preview_sample(backbone, head, dataset, device, args, split_name: str):
    if len(dataset) == 0:
        return None
    sample = dataset[0]
    with torch.no_grad():
        pred_dra, pred_ddec, _, used_bands = run_model_on_sample(
            backbone,
            head,
            sample,
            device=device,
            target_band_weight=args.target_band_weight,
            use_stem_refine=args.use_stem_refine,
        )
    sample_for_plot = dict(sample)
    sample_for_plot["input_bands"] = used_bands
    return make_preview(
        sample_for_plot,
        pred_dra[0, 0].detach().cpu().numpy(),
        pred_ddec[0, 0].detach().cpu().numpy(),
        epoch=args._epoch_for_preview,
        split_name=split_name,
    )


def run_epoch(
    split_name: str,
    loader,
    backbone,
    head,
    optimizer,
    device: torch.device,
    args,
) -> Dict[str, float]:
    is_train = optimizer is not None
    head.train(mode=is_train)
    agg = defaultdict(float)
    n_samples = 0
    n_fields = 0

    for batch in loader:
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        sample_losses = []

        for i in range(len(batch["tile_id"])):
            sample = {
                "tile_id": batch["tile_id"][i],
                "target_band": batch["target_band"][i],
                "input_bands": batch["input_bands"][i],
                "rubin_images": [x.float() for x in batch["rubin_images"][i]],
                "rubin_rms": [x.float() for x in batch["rubin_rms"][i]],
                "vis_image": batch["vis_image"][i].float(),
                "vis_rms": batch["vis_rms"][i].float(),
                "teacher_dra": batch["teacher_dra"][i].float(),
                "teacher_ddec": batch["teacher_ddec"][i].float(),
                "teacher_x_mesh": batch["teacher_x_mesh"][i].float(),
                "teacher_y_mesh": batch["teacher_y_mesh"][i].float(),
                "matched_vis_xy": batch["matched_vis_xy"][i].float(),
                "matched_offsets": batch["matched_offsets"][i].float(),
                "match_count": batch["match_count"][i],
            }

            with torch.set_grad_enabled(is_train):
                pred_dra, pred_ddec, out, used_bands = run_model_on_sample(
                    backbone,
                    head,
                    sample,
                    device=device,
                    target_band_weight=args.target_band_weight,
                    use_stem_refine=args.use_stem_refine,
                )
                gt_dra = sample["teacher_dra"].unsqueeze(0).unsqueeze(0).to(device)
                gt_ddec = sample["teacher_ddec"].unsqueeze(0).unsqueeze(0).to(device)

                loss_dense = F.smooth_l1_loss(pred_dra, gt_dra) + F.smooth_l1_loss(pred_ddec, gt_ddec)
                loss_smooth = _gradient_penalty(pred_dra) + _gradient_penalty(pred_ddec)
                loss_total = loss_dense + args.smooth_weight * loss_smooth
                sample_losses.append(loss_total)

            metrics = _compute_metrics(pred_dra.detach(), pred_ddec.detach(), gt_dra, gt_ddec)
            agg["loss_total"] += float(loss_total.detach().item())
            agg["loss_dense"] += float(loss_dense.detach().item())
            agg["loss_smooth"] += float(loss_smooth.detach().item())
            agg["mae_ra"] += metrics["mae_ra"]
            agg["mae_dec"] += metrics["mae_dec"]
            agg["mae_total"] += metrics["mae_total"]
            agg["p68_total"] += metrics["p68_total"]
            agg["frac_01arcsec"] += metrics["frac_01arcsec"]
            agg["frac_02arcsec"] += metrics["frac_02arcsec"]
            agg["match_count"] += float(sample["match_count"])
            agg["input_band_count"] += float(len(used_bands))
            n_samples += 1
            n_fields += 1

        if is_train and sample_losses:
            loss = torch.stack(sample_losses).mean()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()

    denom = max(1, n_samples)
    out = {k: v / denom for k, v in agg.items()}
    out["samples"] = n_samples
    out["fields"] = n_fields
    if hasattr(head, "temperature"):
        try:
            out["temp"] = float(head.temperature.detach().item())
        except Exception:
            pass
    if hasattr(head, "residual_gain"):
        try:
            out["residual_gain"] = float(head.residual_gain.detach().item())
        except Exception:
            pass
    return out


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_tile_pairs(args.rubin_dir, args.euclid_dir)
    train_pairs, val_pairs = split_tile_pairs(pairs, args.val_frac, args.seed)

    target_bands = normalize_band_list(args.target_bands)
    if not target_bands:
        target_bands = [normalize_band_name(args.target_band)]
    context_bands = normalize_band_list(args.context_bands)
    if not context_bands:
        context_bands = [f"rubin_{b}" for b in RUBIN_BAND_ORDER]
    detect_bands = normalize_band_list(args.detect_bands)
    if not detect_bands:
        detect_bands = [f"rubin_{b}" for b in ("g", "r", "i", "z")]
    args.detect_bands = detect_bands

    print(f"Target bands: {', '.join(target_bands)}")
    print(f"Context bands: {', '.join(context_bands)}")
    print(f"Detection bands: {', '.join(detect_bands)}")
    print(f"Tile split: train={len(train_pairs)} val={len(val_pairs)}")

    train_samples = build_teacher_samples(train_pairs, target_bands, args, "train")
    val_samples = []
    if val_pairs:
        try:
            val_samples = build_teacher_samples(val_pairs, target_bands, args, "val")
        except RuntimeError:
            print("[val] no usable teacher samples; continuing without validation split.")

    train_dataset = TeacherFieldDataset(
        train_samples,
        context_bands=context_bands,
        finite_frac_thresh=args.finite_frac_thresh,
        mmap=not args.no_mmap,
    )
    val_dataset = TeacherFieldDataset(
        val_samples,
        context_bands=context_bands,
        finite_frac_thresh=args.finite_frac_thresh,
        mmap=not args.no_mmap,
    ) if val_samples else None

    train_loader = make_loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = make_loader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) if val_dataset else None

    backbone = load_backbone(device, args.backbone_ckpt, args.embed_dim, args.proj_dim, args.depth, args.patch_size)
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
    print(f"Head parameters: {n_params:,}")

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
        except Exception as exc:
            print(f"W&B init failed: {exc}")

    best_score = float("inf")
    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            args._epoch_for_preview = epoch

            train_metrics = run_epoch("train", train_loader, backbone, head, optimizer, device, args)
            scheduler.step()
            train_metrics["lr"] = optimizer.param_groups[0]["lr"]

            val_metrics = {}
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = run_epoch("val", val_loader, backbone, head, None, device, args)

            elapsed = time.time() - t0
            score = val_metrics.get("mae_total", train_metrics.get("mae_total", float("inf")))

            temp_str = f" temp={train_metrics['temp']:.4f}" if "temp" in train_metrics else ""
            rg_str = f" rg={train_metrics['residual_gain']:.3f}" if "residual_gain" in train_metrics else ""
            val_str = ""
            if val_metrics:
                val_str = (
                    f" | val_MAE={val_metrics.get('mae_total', 0.0) * 1000:.1f}mas"
                    f" val_p68={val_metrics.get('p68_total', 0.0) * 1000:.1f}mas"
                )
            print(
                f"Epoch {epoch:03d} | "
                f"train_MAE={train_metrics.get('mae_total', 0.0) * 1000:.1f}mas "
                f"train_p68={train_metrics.get('p68_total', 0.0) * 1000:.1f}mas "
                f"loss={train_metrics.get('loss_total', 0.0):.5f} "
                f"bands={train_metrics.get('input_band_count', 0.0):.1f} "
                f"matches={train_metrics.get('match_count', 0.0):.1f}"
                f"{val_str}{temp_str}{rg_str} t={elapsed:.1f}s"
            )

            preview_img = None
            if wandb_run is not None and args.vis_every > 0 and (epoch % args.vis_every == 0):
                try:
                    preview_dataset = val_dataset if val_dataset is not None and len(val_dataset) > 0 else train_dataset
                    preview_split = "val" if val_dataset is not None and len(val_dataset) > 0 else "train"
                    preview_img = evaluate_preview_sample(backbone, head, preview_dataset, device, args, preview_split)
                except Exception as exc:
                    print(f"Preview rendering failed at epoch {epoch}: {exc}")

            flat_metrics = {}
            for key, value in train_metrics.items():
                flat_metrics[f"train/{key}"] = value
            for key, value in val_metrics.items():
                flat_metrics[f"val/{key}"] = value
            flat_metrics["epoch"] = epoch
            flat_metrics["time_sec"] = elapsed
            if preview_img is not None:
                flat_metrics["preview/fixed_tile"] = preview_img
            if wandb_run is not None:
                wandb_run.log(flat_metrics, step=epoch)

            save_args = {k: v for k, v in vars(args).items() if not str(k).startswith("_")}
            ckpt = {
                "epoch": epoch,
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "args": save_args,
            }
            torch.save(ckpt, out_dir / "last_astrometry.pt")
            if score < best_score:
                best_score = score
                torch.save(ckpt, out_dir / "best_astrometry.pt")
                if wandb_run is not None:
                    wandb_run.summary["best_val_mae_total_mas"] = score * 1000.0
                    wandb_run.summary["best_epoch"] = epoch

            metrics_out = {
                "epoch": epoch,
                "time_sec": elapsed,
                "train": train_metrics,
                "val": val_metrics,
                "best_score_arcsec": best_score,
            }
            with open(out_dir / "latest_metrics.json", "w") as handle:
                json.dump(metrics_out, handle, indent=2)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if hasattr(args, "_epoch_for_preview"):
            delattr(args, "_epoch_for_preview")

    print(f"\nDone. Best score: {best_score * 1000.0:.2f} mas")
    print(f"Checkpoints in: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the astrometry head against smooth multiband teacher fields.")
    p.add_argument("--rubin-dir", type=str, required=True)
    p.add_argument("--euclid-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="models/checkpoints/jaisp_astrometry_multiband_teacher")
    p.add_argument("--backbone-ckpt", type=str, default="models/checkpoints/jaisp_v5/best.pt")

    p.add_argument("--target-band", type=str, default="r")
    p.add_argument(
        "--target-bands",
        type=str,
        nargs="+",
        default=[],
        help="Optional list of Rubin target bands to supervise in one run. Use 'all' for all six.",
    )
    p.add_argument(
        "--context-bands",
        type=str,
        nargs="+",
        default=["all"],
        help="Rubin bands fed into the foundation for multiband context.",
    )
    p.add_argument(
        "--detect-bands",
        type=str,
        nargs="+",
        default=["g", "r", "i", "z"],
        help="Bands used to build the shared Rubin detection image for teacher labels.",
    )
    p.add_argument("--target-band-weight", type=float, default=0.7,
                   help="Blend weight for the target band when fusing target-band and context latent features.")

    p.add_argument("--teacher-dstep", type=int, default=8)
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
    p.add_argument("--detect-clip-sigma", type=float, default=8.0)
    p.add_argument("--no-refine-band-centroids", action="store_true")
    p.add_argument("--refine-radius", type=int, default=3)
    p.add_argument("--refine-flux-floor-sigma", type=float, default=1.5)
    p.add_argument(
        "--rbf-kernel",
        type=str,
        default="thin_plate_spline",
        choices=["thin_plate_spline", "cubic", "linear", "quintic"],
    )
    p.add_argument("--rbf-smoothing", type=float, default=5e-4)
    p.add_argument("--rbf-neighbors", type=int, default=32)

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--smooth-weight", type=float, default=0.02)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--finite-frac-thresh", type=float, default=0.30)
    p.add_argument("--no-mmap", action="store_true")

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

    p.add_argument("--vis-every", type=int, default=1)
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
