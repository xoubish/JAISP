import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Make parent models/ importable when running this script directly.
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from jaisp_dataset_v4 import ALL_BANDS
from jaisp_foundation_v5 import JAISPFoundationV5

from dataset import make_reconstruction_loader
from head import ResolutionAwareReconstructionHead, interpolate_tokens
from encoding import encode_target_and_context_with_stems
from masking import build_mask

try:
    import wandb
except ImportError:
    wandb = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_backbone(
    device: torch.device,
    checkpoint_path: str,
    stem_clamp_min: float,
    stem_clamp_max: float,
    embed_dim: int,
    proj_dim: int,
    depth: int,
    patch_size: int,
) -> JAISPFoundationV5:
    model = JAISPFoundationV5(
        band_names=ALL_BANDS,
        stem_ch=64,
        stem_clamp_min=stem_clamp_min,
        stem_clamp_max=stem_clamp_max,
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
        print(f"Loaded backbone checkpoint: {checkpoint_path}")
        print(f"  missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    else:
        print("No backbone checkpoint provided; starting from random initialization.")

    # Ensure stem clamp range is controlled by the reconstruction run arguments.
    model.set_stem_clamp_range(stem_clamp_min, stem_clamp_max, include_teacher=False)
    print(f"Backbone stem clamp (noise units): [{stem_clamp_min:.2f}, {stem_clamp_max:.2f}]")

    return model


def to_token_mask(pixel_mask: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
    # pixel_mask: [1,H,W] -> token mask [1,N]
    tm = F.interpolate(
        pixel_mask.unsqueeze(0), size=(int(grid_size[0]), int(grid_size[1])), mode="nearest"
    )
    return tm.view(1, -1)


def _normalize_mask_probs(p_random: float, p_object: float, p_hard: float) -> Dict[str, float]:
    pr = float(p_random)
    po = float(p_object)
    ph = float(p_hard)
    total = pr + po + ph
    if total <= 0:
        return {"random": 1.0, "object": 0.0, "hard": 0.0}
    return {"random": pr / total, "object": po / total, "hard": ph / total}


def get_epoch_mask_probs(args: argparse.Namespace, epoch: int) -> Dict[str, float]:
    target = _normalize_mask_probs(args.mask_random, args.mask_object, args.mask_hard)
    if args.no_mask_curriculum:
        return target

    n_curr = max(1, int(args.curriculum_epochs))
    alpha = min(1.0, max(0.0, float(epoch - 1) / float(n_curr)))
    hard_now = target["hard"] * alpha
    remaining = max(0.0, 1.0 - hard_now)

    base_non_hard = target["random"] + target["object"]
    if base_non_hard <= 1e-10:
        return {"random": remaining, "object": 0.0, "hard": hard_now}

    scale = remaining / base_non_hard
    return {
        "random": target["random"] * scale,
        "object": target["object"] * scale,
        "hard": hard_now,
    }


def _is_rubin_band(band: str) -> bool:
    return str(band).startswith("rubin_")


def _is_euclid_band(band: str) -> bool:
    return str(band).startswith("euclid_")


def _context_allowed(target_band: str, context_band: str, policy: str) -> bool:
    if policy == "all":
        return True
    if policy == "rubin_target_rubin_only":
        if _is_rubin_band(target_band):
            return _is_rubin_band(context_band)
        return True
    if policy == "same_survey":
        if _is_rubin_band(target_band):
            return _is_rubin_band(context_band)
        if _is_euclid_band(target_band):
            return _is_euclid_band(context_band)
        return True
    raise ValueError(f"Unknown context policy: {policy}")


def _filter_context_by_policy(
    target_band: str,
    context_images: List[torch.Tensor],
    context_rms: List[torch.Tensor],
    context_bands: List[str],
    policy: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    if policy == "all":
        return context_images, context_rms, context_bands

    keep = [j for j, b in enumerate(context_bands) if _context_allowed(target_band, str(b), policy)]
    return (
        [context_images[j] for j in keep],
        [context_rms[j] for j in keep],
        [str(context_bands[j]) for j in keep],
    )


def build_mask_with_seed(
    image: torch.Tensor,
    probs: Dict[str, float],
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, str]:
    if seed is None:
        return build_mask(image, probs)

    devices = [image.device] if image.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if image.is_cuda:
            torch.cuda.manual_seed_all(int(seed))
        return build_mask(image, probs)


# encode_target_and_context is now imported from encoding.py
# (encode_target_and_context_with_stems — returns stem features too)


def _masked_ssim_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Local SSIM over 3x3 neighborhoods.
    c1 = 0.01 * 0.01
    c2 = 0.03 * 0.03
    mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(pred * pred, kernel_size=3, stride=1, padding=1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size=3, stride=1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = (num / (den + 1e-8)).clamp(-1.0, 1.0)
    loss_map = 0.5 * (1.0 - ssim_map)

    den_mask = mask.sum().clamp(min=1.0)
    return (loss_map * mask).sum() / den_mask


def _masked_gradient_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Horizontal gradients.
    gx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]
    mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    den_x = mx.sum().clamp(min=1.0)
    lx = (gx_pred - gx_tgt).abs().mul(mx).sum() / den_x

    # Vertical gradients.
    gy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]
    my = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    den_y = my.sum().clamp(min=1.0)
    ly = (gy_pred - gy_tgt).abs().mul(my).sum() / den_y

    return 0.5 * (lx + ly)


def _masked_mean_and_std(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample masked mean/std for [B,1,H,W] tensors."""
    den = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    mean = (x * mask).sum(dim=(1, 2, 3)) / den
    var = ((x - mean.view(-1, 1, 1, 1)) ** 2 * mask).sum(dim=(1, 2, 3)) / den
    std = torch.sqrt(var.clamp(min=1e-8))
    return mean, std


def _masked_flux_conservation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    source_mask: torch.Tensor,
    min_pixels: float,
) -> torch.Tensor:
    """Relative source-flux mismatch over masked source pixels."""
    pix = source_mask.sum(dim=(1, 2, 3))
    flux_pred = (pred * source_mask).sum(dim=(1, 2, 3))
    flux_tgt = (target * source_mask).sum(dim=(1, 2, 3))

    rel_err = (flux_pred - flux_tgt).abs() / (flux_tgt.abs() + 1.0)
    valid = (pix >= float(min_pixels)).float()
    den = valid.sum().clamp(min=1.0)
    return (rel_err * valid).sum() / den


def _masked_background_stats_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    bg_mask: torch.Tensor,
    min_pixels: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match masked-background mean and std between prediction and target."""
    pix = bg_mask.sum(dim=(1, 2, 3))
    pred_mean, pred_std = _masked_mean_and_std(pred, bg_mask)
    tgt_mean, tgt_std = _masked_mean_and_std(target, bg_mask)

    mean_err = (pred_mean - tgt_mean).abs()
    std_err = (pred_std - tgt_std).abs()
    valid = (pix >= float(min_pixels)).float()
    den = valid.sum().clamp(min=1.0)

    loss_bg_mean = (mean_err * valid).sum() / den
    loss_bg_std = (std_err * valid).sum() / den
    return loss_bg_mean + loss_bg_std, loss_bg_mean, loss_bg_std


def compute_losses(
    pred: torch.Tensor,
    target_image: torch.Tensor,
    target_rms: torch.Tensor,
    pixel_mask: torch.Tensor,
    unmasked_weight: float,
    predict_noise_units: bool,
    target_clamp_min: float,
    target_clamp_max: float,
    source_loss_weight: float,
    source_snr_threshold: float,
    ssim_weight: float,
    grad_weight: float,
    flux_loss_weight: float,
    bg_stat_loss_weight: float,
    bg_snr_threshold: float,
    flux_min_pixels: float,
    bg_min_pixels: float,
) -> Dict[str, torch.Tensor]:
    """Compute masked and optional unmasked L1 losses on predicted image."""
    h_pred, w_pred = pred.shape[-2], pred.shape[-1]

    target_crop = target_image[:, :h_pred, :w_pred].unsqueeze(0).to(pred.device)  # [1,1,H,W]
    rms_crop = target_rms[:, :h_pred, :w_pred].unsqueeze(0).to(pred.device)
    if predict_noise_units:
        target_crop = target_crop / (rms_crop + 1e-10)
    target_crop = target_crop.clamp(min=float(target_clamp_min), max=float(target_clamp_max))
    mask_crop = pixel_mask[:, :h_pred, :w_pred].unsqueeze(0).to(pred.device)

    abs_err = (pred - target_crop).abs()

    masked_den = mask_crop.sum().clamp(min=1.0)
    unmasked_den = (1.0 - mask_crop).sum().clamp(min=1.0)

    # Source-weighted masked loss: emphasize high-SNR/source pixels in masked region.
    src_weight = 1.0 + float(source_loss_weight) * torch.sigmoid(
        (target_crop.abs() - float(source_snr_threshold)) * 2.0
    )
    weighted_mask = mask_crop * src_weight
    weighted_den = weighted_mask.sum().clamp(min=1.0)

    loss_masked = (abs_err * weighted_mask).sum() / weighted_den
    loss_unmasked = (abs_err * (1.0 - mask_crop)).sum() / unmasked_den
    loss_ssim = _masked_ssim_loss(pred, target_crop, mask_crop)
    loss_grad = _masked_gradient_loss(pred, target_crop, mask_crop)

    # Explicit source-flux conservation over masked high-SNR target pixels.
    src_flux_mask = mask_crop * (target_crop > float(source_snr_threshold)).float()
    loss_flux = _masked_flux_conservation_loss(
        pred=pred,
        target=target_crop,
        source_mask=src_flux_mask,
        min_pixels=flux_min_pixels,
    )

    # Explicit masked-background statistics matching (low-SNR region).
    bg_mask = mask_crop * (target_crop.abs() <= float(bg_snr_threshold)).float()
    loss_bg_stat, loss_bg_mean, loss_bg_std = _masked_background_stats_loss(
        pred=pred,
        target=target_crop,
        bg_mask=bg_mask,
        min_pixels=bg_min_pixels,
    )

    loss_total = (
        loss_masked
        + float(unmasked_weight) * loss_unmasked
        + float(ssim_weight) * loss_ssim
        + float(grad_weight) * loss_grad
        + float(flux_loss_weight) * loss_flux
        + float(bg_stat_loss_weight) * loss_bg_stat
    )

    sq_err = ((pred - target_crop) ** 2) * mask_crop
    mse_masked = sq_err.sum() / masked_den
    vals = target_crop[mask_crop > 0.5]
    if vals.numel() > 16:
        p01 = torch.quantile(vals, 0.01)
        p99 = torch.quantile(vals, 0.99)
        dyn = (p99 - p01).clamp(min=1e-3)
    else:
        dyn = torch.tensor(1.0, device=pred.device)
    psnr_masked = 20.0 * torch.log10(dyn) - 10.0 * torch.log10(mse_masked + 1e-8)

    return {
        "loss_total": loss_total,
        "loss_masked": loss_masked,
        "loss_unmasked": loss_unmasked,
        "loss_ssim": loss_ssim,
        "loss_grad": loss_grad,
        "loss_flux": loss_flux,
        "loss_bg_stat": loss_bg_stat,
        "loss_bg_mean": loss_bg_mean,
        "loss_bg_std": loss_bg_std,
        "psnr_masked": psnr_masked,
        "mask_frac": mask_crop.mean(),
    }


def build_masked_inputs(
    target_image: torch.Tensor,
    target_rms: torch.Tensor,
    pixel_mask: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Raw-space masked image for backbone encoding.
    target_masked_raw = target_image * (1.0 - pixel_mask) + args.mask_value * pixel_mask

    # Model-space masked image for residual/inpaint decoding.
    if args.predict_noise_units:
        target_model = target_image / (target_rms + 1e-10)
    else:
        target_model = target_image
    target_model = target_model.clamp(min=float(args.target_clamp_min), max=float(args.target_clamp_max))
    target_masked_model = target_model * (1.0 - pixel_mask) + args.mask_value * pixel_mask

    return target_masked_raw, target_masked_model


def _compute_display_limits(
    arrays: List[np.ndarray],
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> Tuple[float, float]:
    chunks = []
    for arr in arrays:
        x = np.asarray(arr, dtype=np.float32)
        finite = x[np.isfinite(x)]
        if finite.size > 0:
            chunks.append(finite)

    if not chunks:
        return 0.0, 1.0

    vals = np.concatenate(chunks, axis=0)
    lo, hi = np.percentile(vals, [float(low_pct), float(high_pct)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max()) if vals.max() > vals.min() else float(vals.min() + 1.0)
    return float(lo), float(hi)


def _norm_for_display(
    img_2d: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    x = np.asarray(img_2d, dtype=np.float32)
    if vmin is None or vmax is None:
        vmin, vmax = _compute_display_limits([x], low_pct=1.0, high_pct=99.0)
    x = np.nan_to_num(x, nan=float(vmin), posinf=float(vmax), neginf=float(vmin))
    x = np.clip((x - float(vmin)) / (float(vmax) - float(vmin) + 1e-8), 0.0, 1.0)
    return x


def _mask_bbox(mask_2d: np.ndarray, margin: int = 12) -> Tuple[int, int, int, int]:
    h, w = mask_2d.shape
    ys, xs = np.where(mask_2d > 0.5)
    if ys.size == 0:
        return 0, h, 0, w
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(h, int(ys.max()) + 1 + margin)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(w, int(xs.max()) + 1 + margin)
    return y0, y1, x0, x1


def _make_preview_image(preview: Dict, epoch: int):
    import matplotlib.pyplot as plt

    target_raw = np.asarray(preview["target"], dtype=np.float32)
    masked_raw = np.asarray(preview["masked"], dtype=np.float32)
    token_raw = np.asarray(preview["token_inpaint"], dtype=np.float32)
    pred_raw = np.asarray(preview["pred"], dtype=np.float32)
    mask = np.asarray(preview["mask"], dtype=np.float32)

    # Predicted maps are already inpainted in masked region (outside mask is unchanged by design).
    err_raw = np.abs(pred_raw - target_raw) * mask

    y0, y1, x0, x1 = _mask_bbox(mask, margin=12)

    # Shared intensity normalization across image-like panels to preserve photometric comparability.
    full_lo, full_hi = _compute_display_limits([target_raw], low_pct=0.5, high_pct=99.5)
    zoom_ref = target_raw[y0:y1, x0:x1]
    zoom_lo, zoom_hi = _compute_display_limits([zoom_ref], low_pct=0.5, high_pct=99.5)

    err_vals = err_raw[mask > 0.5]
    if err_vals.size > 0:
        err_scale = float(np.percentile(err_vals, 99.0))
        if not np.isfinite(err_scale) or err_scale <= 1e-8:
            err_scale = float(np.max(err_vals))
    else:
        err_scale = 1.0
    if not np.isfinite(err_scale) or err_scale <= 1e-8:
        err_scale = 1.0

    # Source-only flux ratio map inside the masked area.
    mask_bool = mask > 0.5
    src_mask = np.zeros_like(mask_bool, dtype=bool)
    if np.any(mask_bool):
        vals = target_raw[mask_bool]
        vals = vals[np.isfinite(vals)]
        if vals.size > 32:
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            sigma = max(1e-6, 1.4826 * mad)
            thr = med + 3.0 * sigma
            src_mask = mask_bool & np.isfinite(target_raw) & np.isfinite(pred_raw) & (target_raw > thr)
        if src_mask.sum() < 8 and vals.size > 0:
            thr = float(np.percentile(vals, 90.0))
            src_mask = mask_bool & np.isfinite(target_raw) & np.isfinite(pred_raw) & (target_raw > thr)

    ratio_map = np.full_like(target_raw, np.nan, dtype=np.float32)
    ratio_map[src_mask] = pred_raw[src_mask] / (target_raw[src_mask] + 1e-8)
    ratio_vals = ratio_map[src_mask]
    if ratio_vals.size > 0:
        ratio_median = float(np.nanmedian(ratio_vals))
        ratio_title = f"Flux Ratio (src, med={ratio_median:.2f})"
    else:
        ratio_title = "Flux Ratio (src, n/a)"

    panels = [
        ("Target", target_raw, "gray", "shared"),
        ("Masked Input", masked_raw, "gray", "shared"),
        ("Token Inpaint", token_raw, "gray", "shared"),
        ("Refined Inpaint", pred_raw, "gray", "shared"),
        ("|Error| in Mask", err_raw, "magma", "error"),
        (ratio_title, ratio_map, "coolwarm", "ratio"),
    ]

    fig, axes = plt.subplots(2, len(panels), figsize=(3.3 * len(panels), 7))
    for c, (title, arr, cmap, mode) in enumerate(panels):
        if mode == "shared":
            arr_full = _norm_for_display(arr, vmin=full_lo, vmax=full_hi)
            arr_zoom = _norm_for_display(arr[y0:y1, x0:x1], vmin=zoom_lo, vmax=zoom_hi)
            axes[0, c].imshow(arr_full, cmap=cmap, vmin=0.0, vmax=1.0)
            axes[1, c].imshow(arr_zoom, cmap=cmap, vmin=0.0, vmax=1.0)
        elif mode == "error":
            arr_full = np.clip(arr / (err_scale + 1e-8), 0.0, 1.0)
            arr_zoom = np.clip(arr[y0:y1, x0:x1] / (err_scale + 1e-8), 0.0, 1.0)
            axes[0, c].imshow(arr_full, cmap=cmap, vmin=0.0, vmax=1.0)
            axes[1, c].imshow(arr_zoom, cmap=cmap, vmin=0.0, vmax=1.0)
        elif mode == "ratio":
            ratio_cmap = plt.get_cmap(cmap).copy()
            ratio_cmap.set_bad(color="black")
            axes[0, c].imshow(arr, cmap=ratio_cmap, vmin=0.5, vmax=1.5)
            axes[1, c].imshow(arr[y0:y1, x0:x1], cmap=ratio_cmap, vmin=0.5, vmax=1.5)
        else:
            raise ValueError(f"Unknown preview mode: {mode}")

        axes[0, c].set_title(title, fontsize=10)
        axes[0, c].axis("off")
        axes[1, c].set_title(f"{title} (masked zoom)", fontsize=10)
        axes[1, c].axis("off")

    title = (
        f"Epoch {epoch} | target={preview['target_band']} | "
        f"context={','.join(preview['context_bands'])} | mask={preview['mask_type']}"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def evaluate_fixed_validation(
    backbone: JAISPFoundationV5,
    head: ResolutionAwareReconstructionHead,
    val_loader,
    args: argparse.Namespace,
    device: torch.device,
    val_mask_probs: Dict[str, float],
) -> Tuple[Optional[Dict[str, float]], Optional[Dict]]:
    # Reset validation dataset RNG so band/context sampling stays fixed across epochs.
    ds = val_loader.dataset
    if hasattr(ds, "dataset"):  # handle Subset if used later
        ds = ds.dataset
    if hasattr(ds, "rng"):
        ds.rng = np.random.RandomState(int(args.val_seed))

    agg = {
        "loss_total": 0.0,
        "loss_masked": 0.0,
        "loss_unmasked": 0.0,
        "loss_ssim": 0.0,
        "loss_grad": 0.0,
        "loss_flux": 0.0,
        "loss_bg_stat": 0.0,
        "loss_bg_mean": 0.0,
        "loss_bg_std": 0.0,
        "psnr_masked": 0.0,
        "mask_frac": 0.0,
        "context_count_sum": 0.0,
        "num_samples": 0,
    }
    preview = None

    head.eval()
    backbone.eval()
    with torch.no_grad():
        for bidx, batch in enumerate(val_loader):
            if args.val_max_batches > 0 and bidx >= args.val_max_batches:
                break

            bsz = len(batch["target_band"])
            for i in range(bsz):
                target_band = str(batch["target_band"][i])
                if target_band not in backbone.band_names:
                    continue

                target_image = batch["target_image"][i].float().to(device)
                target_rms = batch["target_rms"][i].float().to(device)

                context_images = [x.float().to(device) for x in batch["context_images"][i]]
                context_rms = [x.float().to(device) for x in batch["context_rms"][i]]
                context_bands = [str(x) for x in batch["context_bands"][i]]
                context_images, context_rms, context_bands = _filter_context_by_policy(
                    target_band=target_band,
                    context_images=context_images,
                    context_rms=context_rms,
                    context_bands=context_bands,
                    policy=args.context_policy,
                )
                valid = [j for j, b in enumerate(context_bands) if b in backbone.band_names]
                if not valid:
                    continue
                context_images = [context_images[j] for j in valid]
                context_rms = [context_rms[j] for j in valid]
                context_bands = [context_bands[j] for j in valid]

                sample_seed = int(args.val_seed + bidx * 1009 + i * 37)
                pixel_mask, mask_type = build_mask_with_seed(target_image, val_mask_probs, seed=sample_seed)
                target_masked_raw, target_masked_model = build_masked_inputs(
                    target_image=target_image,
                    target_rms=target_rms,
                    pixel_mask=pixel_mask,
                    args=args,
                )

                target_tokens, context_tokens, target_grid, target_stem_feat, context_stem_feats = \
                    encode_target_and_context_with_stems(
                    backbone=backbone,
                    target_masked=target_masked_raw,
                    target_rms=target_rms,
                    target_band=target_band,
                    context_images=context_images,
                    context_rms=context_rms,
                    context_bands=context_bands,
                    device=device,
                    freeze_backbone=args.freeze_backbone,
                    use_projector_tokens=args.use_projector_tokens,
                )

                target_hw = (int(target_image.shape[-2]), int(target_image.shape[-1]))
                token_mask = to_token_mask(pixel_mask, target_grid)
                head_out = head(
                    target_tokens=target_tokens,
                    context_tokens=context_tokens,
                    token_mask=token_mask,
                    grid_size=target_grid,
                    target_hw=target_hw,
                    target_stem_feat=target_stem_feat,
                    context_stem_feats=context_stem_feats,
                    masked_input=target_masked_model.unsqueeze(0),
                    pixel_mask=pixel_mask.unsqueeze(0),
                )
                pred_model = head_out["pred"]

                metrics = compute_losses(
                    pred=pred_model,
                    target_image=target_image,
                    target_rms=target_rms,
                    pixel_mask=pixel_mask,
                    unmasked_weight=args.unmasked_weight,
                    predict_noise_units=args.predict_noise_units,
                    target_clamp_min=args.target_clamp_min,
                    target_clamp_max=args.target_clamp_max,
                    source_loss_weight=args.source_loss_weight,
                    source_snr_threshold=args.source_snr_threshold,
                    ssim_weight=args.ssim_weight,
                    grad_weight=args.grad_weight,
                    flux_loss_weight=args.flux_loss_weight,
                    bg_stat_loss_weight=args.bg_stat_loss_weight,
                    bg_snr_threshold=args.bg_snr_threshold,
                    flux_min_pixels=args.flux_min_pixels,
                    bg_min_pixels=args.bg_min_pixels,
                )

                agg["loss_total"] += float(metrics["loss_total"].detach().item())
                agg["loss_masked"] += float(metrics["loss_masked"].detach().item())
                agg["loss_unmasked"] += float(metrics["loss_unmasked"].detach().item())
                agg["loss_ssim"] += float(metrics["loss_ssim"].detach().item())
                agg["loss_grad"] += float(metrics["loss_grad"].detach().item())
                agg["loss_flux"] += float(metrics["loss_flux"].detach().item())
                agg["loss_bg_stat"] += float(metrics["loss_bg_stat"].detach().item())
                agg["loss_bg_mean"] += float(metrics["loss_bg_mean"].detach().item())
                agg["loss_bg_std"] += float(metrics["loss_bg_std"].detach().item())
                agg["psnr_masked"] += float(metrics["psnr_masked"].detach().item())
                agg["mask_frac"] += float(metrics["mask_frac"].detach().item())
                agg["context_count_sum"] += len(context_bands)
                agg["num_samples"] += 1

                if preview is None:
                    h_pred, w_pred = pred_model.shape[-2], pred_model.shape[-1]
                    token_pred = head_out["token_inpaint"].detach().cpu().numpy()[0, 0]
                    refined_pred = pred_model.detach().cpu().numpy()[0, 0]
                    if args.predict_noise_units:
                        rms_np = target_rms[:, :h_pred, :w_pred].detach().cpu().numpy()[0]
                        token_pred = token_pred * rms_np
                        refined_pred = refined_pred * rms_np
                    preview = {
                        "target": target_image[:, :h_pred, :w_pred].detach().cpu().numpy()[0],
                        "masked": target_masked_raw[:, :h_pred, :w_pred].detach().cpu().numpy()[0],
                        "token_inpaint": token_pred,
                        "pred": refined_pred,
                        "mask": pixel_mask[:, :h_pred, :w_pred].detach().cpu().numpy()[0],
                        "target_band": target_band,
                        "context_bands": context_bands,
                        "mask_type": mask_type,
                    }

    n = agg["num_samples"]
    if n <= 0:
        return None, preview
    return (
        {
            "loss_total": agg["loss_total"] / n,
            "loss_masked": agg["loss_masked"] / n,
            "loss_unmasked": agg["loss_unmasked"] / n,
            "loss_ssim": agg["loss_ssim"] / n,
            "loss_grad": agg["loss_grad"] / n,
            "loss_flux": agg["loss_flux"] / n,
            "loss_bg_stat": agg["loss_bg_stat"] / n,
            "loss_bg_mean": agg["loss_bg_mean"] / n,
            "loss_bg_std": agg["loss_bg_std"] / n,
            "psnr_masked": agg["psnr_masked"] / n,
            "mask_frac": agg["mask_frac"] / n,
            "context_count_mean": agg["context_count_sum"] / n,
            "samples": n,
        },
        preview,
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, loader = make_reconstruction_loader(
        rubin_dir=args.rubin_dir,
        euclid_dir=args.euclid_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        augment=not args.no_augment,
        seed=args.seed,
        min_context_bands=args.min_context,
        max_context_bands=args.max_context,
        context_policy=args.context_policy,
    )
    _, val_loader = make_reconstruction_loader(
        rubin_dir=args.rubin_dir,
        euclid_dir=args.euclid_dir,
        batch_size=args.val_batch_size,
        num_workers=args.val_num_workers,
        shuffle=False,
        drop_last=False,
        augment=False,
        seed=args.val_seed,
        min_context_bands=args.min_context,
        max_context_bands=args.max_context,
        context_policy=args.context_policy,
    )

    backbone = load_backbone(
        device=device,
        checkpoint_path=args.backbone_ckpt,
        stem_clamp_min=args.stem_clamp_min,
        stem_clamp_max=args.stem_clamp_max,
        embed_dim=args.embed_dim,
        proj_dim=args.proj_dim,
        depth=args.depth,
        patch_size=args.patch_size,
    )

    head = ResolutionAwareReconstructionHead(
        embed_dim=(args.proj_dim if args.use_projector_tokens else args.embed_dim),
        stem_ch=args.stem_ch,
        patch_size=args.patch_size,
        depth=args.head_depth,
        num_heads=args.head_heads,
        mlp_ratio=args.head_mlp_ratio,
        skip_proj=args.skip_proj,
    ).to(device)

    if args.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        print("Backbone is frozen (default).")
    else:
        backbone.eval()
        print("Backbone is trainable (eval mode kept to stabilize BN with small batch sizes).")

    param_groups = [{"params": head.parameters(), "lr": args.lr}]
    if not args.freeze_backbone:
        trainable_backbone = [p for p in backbone.parameters() if p.requires_grad]
        if trainable_backbone:
            param_groups.append({"params": trainable_backbone, "lr": args.backbone_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-6)

    target_mask_probs = _normalize_mask_probs(args.mask_random, args.mask_object, args.mask_hard)
    accum_steps = max(1, int(args.grad_accum_steps))
    use_amp = bool(args.amp) and (device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    print(f"Device: {device}")
    print(f"Tiles: {len(dataset)}, steps/epoch: {len(loader)}")
    print(
        f"Context bands per sample: [{args.min_context}, {args.max_context}] "
        f"(policy={args.context_policy}) | "
        "target mask mix random/object/hard = "
        f"{target_mask_probs['random']:.2f}/{target_mask_probs['object']:.2f}/{target_mask_probs['hard']:.2f}"
    )
    print(
        f"Reconstruction tokens: {'projector(z)' if args.use_projector_tokens else 'encoder(pre-projector)'} | "
        f"target space: {'noise units (image/rms)' if args.predict_noise_units else 'raw image units'}"
    )
    print(
        "Source weighting: "
        f"weight={args.source_loss_weight:.2f}, snr_threshold={args.source_snr_threshold:.2f} | "
        "extra losses "
        f"flux={args.flux_loss_weight:.3f}, bg_stat={args.bg_stat_loss_weight:.3f} "
        f"(bg_snr<={args.bg_snr_threshold:.2f}) | "
        f"aux losses ssim={args.ssim_weight:.3f}, grad={args.grad_weight:.3f} | "
        f"mask curriculum={'off' if args.no_mask_curriculum else f'on ({args.curriculum_epochs} epochs)'}"
    )
    print(
        f"Train precision: {'AMP' if use_amp else 'fp32'}"
        + (f" ({args.amp_dtype})" if use_amp else "")
        + f" | grad_accum_steps={accum_steps}"
    )
    print(f"Best-checkpoint metric: {args.model_selection}")

    wandb_run = None
    if args.wandb_mode != "disabled":
        if wandb is None:
            print("W&B logging disabled: wandb is not installed.")
        else:
            try:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity or None,
                    name=args.wandb_run_name or None,
                    config=vars(args),
                    mode=args.wandb_mode,
                    dir=str(out_dir),
                )
                print(f"W&B initialized: project={args.wandb_project}, mode={args.wandb_mode}")
            except Exception as e:
                print(f"W&B init failed, continuing without logging: {e}")
                wandb_run = None

    global_step = 0
    best_loss = float("inf")

    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            head.train()
            if not args.freeze_backbone:
                backbone.eval()
            epoch_mask_probs = get_epoch_mask_probs(args, epoch)

            agg = {
                "loss_total": 0.0,
                "loss_masked": 0.0,
                "loss_unmasked": 0.0,
                "loss_ssim": 0.0,
                "loss_grad": 0.0,
                "loss_flux": 0.0,
                "loss_bg_stat": 0.0,
                "loss_bg_mean": 0.0,
                "loss_bg_std": 0.0,
                "psnr_masked": 0.0,
                "mask_frac": 0.0,
                "num_samples": 0,
                "context_count_sum": 0.0,
            }
            mask_type_counts = {"random": 0, "object": 0, "hard": 0}
            preview = None
            accum_counter = 0
            optimizer.zero_grad(set_to_none=True)

            for batch in loader:
                sample_losses = []

                bsz = len(batch["target_band"])
                for i in range(bsz):
                    target_band = str(batch["target_band"][i])
                    if target_band not in backbone.band_names:
                        continue

                    target_image = batch["target_image"][i].float().to(device)  # [1,H,W]
                    target_rms = batch["target_rms"][i].float().to(device)

                    context_images = [x.float().to(device) for x in batch["context_images"][i]]
                    context_rms = [x.float().to(device) for x in batch["context_rms"][i]]
                    context_bands = [str(x) for x in batch["context_bands"][i]]
                    context_images, context_rms, context_bands = _filter_context_by_policy(
                        target_band=target_band,
                        context_images=context_images,
                        context_rms=context_rms,
                        context_bands=context_bands,
                        policy=args.context_policy,
                    )

                    valid = [j for j, b in enumerate(context_bands) if b in backbone.band_names]
                    if not valid:
                        continue
                    context_images = [context_images[j] for j in valid]
                    context_rms = [context_rms[j] for j in valid]
                    context_bands = [context_bands[j] for j in valid]

                    pixel_mask, mask_type = build_mask_with_seed(target_image, epoch_mask_probs, seed=None)
                    mask_type_counts[mask_type] = mask_type_counts.get(mask_type, 0) + 1
                    target_masked_raw, target_masked_model = build_masked_inputs(
                        target_image=target_image,
                        target_rms=target_rms,
                        pixel_mask=pixel_mask,
                        args=args,
                    )

                    target_tokens, context_tokens, target_grid, target_stem_feat, context_stem_feats = \
                        encode_target_and_context_with_stems(
                        backbone=backbone,
                        target_masked=target_masked_raw,
                        target_rms=target_rms,
                        target_band=target_band,
                        context_images=context_images,
                        context_rms=context_rms,
                        context_bands=context_bands,
                        device=device,
                        freeze_backbone=args.freeze_backbone,
                        use_projector_tokens=args.use_projector_tokens,
                    )

                    target_hw = (int(target_image.shape[-2]), int(target_image.shape[-1]))
                    token_mask = to_token_mask(pixel_mask, target_grid)
                    ac = (
                        torch.autocast(device_type="cuda", dtype=amp_dtype)
                        if use_amp
                        else nullcontext()
                    )
                    with ac:
                        head_out = head(
                            target_tokens=target_tokens,
                            context_tokens=context_tokens,
                            token_mask=token_mask,
                            grid_size=target_grid,
                            target_hw=target_hw,
                            target_stem_feat=target_stem_feat,
                            context_stem_feats=context_stem_feats,
                            masked_input=target_masked_model.unsqueeze(0),
                            pixel_mask=pixel_mask.unsqueeze(0),
                        )
                        pred_model = head_out["pred"]

                        metrics = compute_losses(
                            pred=pred_model,
                            target_image=target_image,
                            target_rms=target_rms,
                            pixel_mask=pixel_mask,
                            unmasked_weight=args.unmasked_weight,
                            predict_noise_units=args.predict_noise_units,
                            target_clamp_min=args.target_clamp_min,
                            target_clamp_max=args.target_clamp_max,
                            source_loss_weight=args.source_loss_weight,
                            source_snr_threshold=args.source_snr_threshold,
                            ssim_weight=args.ssim_weight,
                            grad_weight=args.grad_weight,
                            flux_loss_weight=args.flux_loss_weight,
                            bg_stat_loss_weight=args.bg_stat_loss_weight,
                            bg_snr_threshold=args.bg_snr_threshold,
                            flux_min_pixels=args.flux_min_pixels,
                            bg_min_pixels=args.bg_min_pixels,
                        )

                    sample_losses.append(metrics["loss_total"])
                    agg["loss_total"] += float(metrics["loss_total"].detach().item())
                    agg["loss_masked"] += float(metrics["loss_masked"].detach().item())
                    agg["loss_unmasked"] += float(metrics["loss_unmasked"].detach().item())
                    agg["loss_ssim"] += float(metrics["loss_ssim"].detach().item())
                    agg["loss_grad"] += float(metrics["loss_grad"].detach().item())
                    agg["loss_flux"] += float(metrics["loss_flux"].detach().item())
                    agg["loss_bg_stat"] += float(metrics["loss_bg_stat"].detach().item())
                    agg["loss_bg_mean"] += float(metrics["loss_bg_mean"].detach().item())
                    agg["loss_bg_std"] += float(metrics["loss_bg_std"].detach().item())
                    agg["psnr_masked"] += float(metrics["psnr_masked"].detach().item())
                    agg["mask_frac"] += float(metrics["mask_frac"].detach().item())
                    agg["num_samples"] += 1
                    agg["context_count_sum"] += len(context_bands)

                    if (
                        wandb_run is not None
                        and preview is None
                        and args.wandb_log_images_every > 0
                        and epoch % args.wandb_log_images_every == 0
                    ):
                        h_pred, w_pred = pred_model.shape[-2], pred_model.shape[-1]
                        token_pred = head_out["token_inpaint"].detach().cpu().numpy()[0, 0]
                        refined_pred = pred_model.detach().cpu().numpy()[0, 0]
                        if args.predict_noise_units:
                            rms_np = target_rms[:, :h_pred, :w_pred].detach().cpu().numpy()[0]
                            token_pred = token_pred * rms_np
                            refined_pred = refined_pred * rms_np
                        preview = {
                            "target": target_image[:, :h_pred, :w_pred].detach().cpu().numpy()[0],
                            "masked": target_masked_raw[:, :h_pred, :w_pred].detach().cpu().numpy()[0],
                            "token_inpaint": token_pred,
                            "pred": refined_pred,
                            "mask": pixel_mask[:, :h_pred, :w_pred].detach().cpu().numpy()[0],
                            "target_band": target_band,
                            "context_bands": context_bands,
                            "mask_type": mask_type,
                        }

                if not sample_losses:
                    continue

                loss = torch.stack(sample_losses).mean()
                scaled_loss = loss / float(accum_steps)
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                accum_counter += 1
                if accum_counter % accum_steps == 0:
                    if args.grad_clip > 0:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=args.grad_clip)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

            if accum_counter % accum_steps != 0:
                if args.grad_clip > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=args.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            scheduler.step()

            n = max(1, agg["num_samples"])
            epoch_metrics = {
                "loss_total": agg["loss_total"] / n,
                "loss_masked": agg["loss_masked"] / n,
                "loss_unmasked": agg["loss_unmasked"] / n,
                "loss_ssim": agg["loss_ssim"] / n,
                "loss_grad": agg["loss_grad"] / n,
                "loss_flux": agg["loss_flux"] / n,
                "loss_bg_stat": agg["loss_bg_stat"] / n,
                "loss_bg_mean": agg["loss_bg_mean"] / n,
                "loss_bg_std": agg["loss_bg_std"] / n,
                "psnr_masked": agg["psnr_masked"] / n,
                "mask_frac": agg["mask_frac"] / n,
                "context_count_mean": agg["context_count_sum"] / n,
                "samples": agg["num_samples"],
                "lr_head": optimizer.param_groups[0]["lr"],
                "epoch_time_sec": time.time() - t0,
                "mask_counts": mask_type_counts,
            }
            if len(optimizer.param_groups) > 1:
                epoch_metrics["lr_backbone"] = optimizer.param_groups[1]["lr"]

            val_metrics = None
            val_preview = None
            if args.val_every > 0 and (epoch % args.val_every == 0):
                val_metrics, val_preview = evaluate_fixed_validation(
                    backbone=backbone,
                    head=head,
                    val_loader=val_loader,
                    args=args,
                    device=device,
                    val_mask_probs=target_mask_probs,
                )
                if val_metrics is not None:
                    epoch_metrics["val_loss_total"] = val_metrics["loss_total"]
                    epoch_metrics["val_loss_masked"] = val_metrics["loss_masked"]
                    epoch_metrics["val_loss_ssim"] = val_metrics["loss_ssim"]
                    epoch_metrics["val_loss_grad"] = val_metrics["loss_grad"]
                    epoch_metrics["val_loss_flux"] = val_metrics["loss_flux"]
                    epoch_metrics["val_loss_bg_stat"] = val_metrics["loss_bg_stat"]
                    epoch_metrics["val_loss_bg_mean"] = val_metrics["loss_bg_mean"]
                    epoch_metrics["val_loss_bg_std"] = val_metrics["loss_bg_std"]
                    epoch_metrics["val_psnr_masked"] = val_metrics["psnr_masked"]
                    epoch_metrics["val_mask_frac"] = val_metrics["mask_frac"]
                    epoch_metrics["val_context_count_mean"] = val_metrics["context_count_mean"]
                    epoch_metrics["val_samples"] = val_metrics["samples"]

            print(
                f"Epoch {epoch:03d} | "
                f"loss={epoch_metrics['loss_total']:.4f} "
                f"masked={epoch_metrics['loss_masked']:.4f} "
                f"flux={epoch_metrics['loss_flux']:.4f} "
                f"bg={epoch_metrics['loss_bg_stat']:.4f} "
                f"psnr_masked={epoch_metrics['psnr_masked']:.2f} "
                f"mask_frac={epoch_metrics['mask_frac']:.3f} "
                f"context_mean={epoch_metrics['context_count_mean']:.2f} "
                f"samples={epoch_metrics['samples']} "
                f"t={epoch_metrics['epoch_time_sec']:.1f}s "
                f"mask_mix={epoch_mask_probs['random']:.2f}/{epoch_mask_probs['object']:.2f}/{epoch_mask_probs['hard']:.2f}"
            )
            if val_metrics is not None:
                print(
                    f"           val_total={val_metrics['loss_total']:.4f} "
                    f"val_masked={val_metrics['loss_masked']:.4f} "
                    f"val_flux={val_metrics['loss_flux']:.4f} "
                    f"val_bg={val_metrics['loss_bg_stat']:.4f} "
                    f"val_psnr_masked={val_metrics['psnr_masked']:.2f} "
                    f"val_samples={val_metrics['samples']}"
                )

            if wandb_run is not None:
                wb = {
                    "train/loss_masked": epoch_metrics["loss_masked"],
                    "train/loss_flux": epoch_metrics["loss_flux"],
                    "train/loss_bg_stat": epoch_metrics["loss_bg_stat"],
                    "train/psnr_masked": epoch_metrics["psnr_masked"],
                    "train/mask_frac": epoch_metrics["mask_frac"],
                    "train/context_count_mean": epoch_metrics["context_count_mean"],
                    "train/lr_head": epoch_metrics["lr_head"],
                    "train/mask_frac_random_curriculum": epoch_mask_probs["random"],
                    "train/mask_frac_object_curriculum": epoch_mask_probs["object"],
                    "train/mask_frac_hard_curriculum": epoch_mask_probs["hard"],
                }
                if "lr_backbone" in epoch_metrics:
                    wb["train/lr_backbone"] = epoch_metrics["lr_backbone"]
                if val_metrics is not None:
                    wb["val/loss_total_fixed"] = val_metrics["loss_total"]
                    wb["val/loss_masked_fixed"] = val_metrics["loss_masked"]
                    wb["val/loss_flux_fixed"] = val_metrics["loss_flux"]
                    wb["val/loss_bg_stat_fixed"] = val_metrics["loss_bg_stat"]
                    wb["val/psnr_masked_fixed"] = val_metrics["psnr_masked"]
                    wb["val/mask_frac_fixed"] = val_metrics["mask_frac"]
                    wb["val/context_count_mean_fixed"] = val_metrics["context_count_mean"]
                    if args.wandb_log_detailed:
                        wb["val/loss_ssim_fixed"] = val_metrics["loss_ssim"]
                        wb["val/loss_grad_fixed"] = val_metrics["loss_grad"]
                        wb["val/loss_bg_mean_fixed"] = val_metrics["loss_bg_mean"]
                        wb["val/loss_bg_std_fixed"] = val_metrics["loss_bg_std"]
                if args.wandb_log_detailed:
                    mask_total = max(1, sum(mask_type_counts.values()))
                    wb.update(
                        {
                            "train/loss_total": epoch_metrics["loss_total"],
                            "train/loss_unmasked": epoch_metrics["loss_unmasked"],
                            "train/loss_ssim": epoch_metrics["loss_ssim"],
                            "train/loss_grad": epoch_metrics["loss_grad"],
                            "train/loss_bg_mean": epoch_metrics["loss_bg_mean"],
                            "train/loss_bg_std": epoch_metrics["loss_bg_std"],
                            "train/samples": epoch_metrics["samples"],
                            "train/epoch_time_sec": epoch_metrics["epoch_time_sec"],
                            "train/mask_count_random": mask_type_counts["random"],
                            "train/mask_count_object": mask_type_counts["object"],
                            "train/mask_count_hard": mask_type_counts["hard"],
                            "train/mask_frac_random": mask_type_counts["random"] / mask_total,
                            "train/mask_frac_object": mask_type_counts["object"] / mask_total,
                            "train/mask_frac_hard": mask_type_counts["hard"] / mask_total,
                        }
                    )
                if preview is not None:
                    wb["train/preview"] = _make_preview_image(preview, epoch)
                if val_preview is not None:
                    wb["val/preview"] = _make_preview_image(val_preview, epoch)
                wandb_run.log(wb, step=epoch)

            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "args": vars(args),
                "metrics": epoch_metrics,
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "backbone_state": backbone.state_dict() if not args.freeze_backbone else None,
                "backbone_checkpoint": args.backbone_ckpt,
            }

            last_path = out_dir / "last_reconstruction.pt"
            torch.save(ckpt, last_path)

            if val_metrics is not None:
                score = (
                    val_metrics["loss_total"]
                    if args.model_selection == "total"
                    else val_metrics["loss_masked"]
                )
            else:
                score = (
                    epoch_metrics["loss_total"]
                    if args.model_selection == "total"
                    else epoch_metrics["loss_masked"]
                )
            if score < best_loss:
                best_loss = score
                torch.save(ckpt, out_dir / "best_reconstruction.pt")
                if wandb_run is not None:
                    wandb_run.summary["best_score"] = float(best_loss)
                    wandb_run.summary["best_score_name"] = (
                        (
                            "val/loss_total_fixed"
                            if (val_metrics is not None and args.model_selection == "total")
                            else "val/loss_masked_fixed"
                        )
                        if val_metrics is not None
                        else (
                            "train/loss_total" if args.model_selection == "total" else "train/loss_masked"
                        )
                    )
                    wandb_run.summary["best_epoch"] = int(epoch)

            if args.save_every > 0 and epoch % args.save_every == 0:
                torch.save(ckpt, out_dir / f"epoch_{epoch:03d}.pt")

            with open(out_dir / "latest_metrics.json", "w", encoding="utf-8") as f:
                json.dump(epoch_metrics, f, indent=2)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print("Training complete.")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Saved checkpoints in: {out_dir}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train JAISP masked reconstruction head (multi-band k->1).")

    parser.add_argument("--rubin-dir", type=str, default="../data/rubin_tiles_ecdfs")
    parser.add_argument("--euclid-dir", type=str, default="../data/euclid_tiles_ecdfs")
    parser.add_argument("--output-dir", type=str, default="checkpoints/jaisp_reconstruction")
    parser.add_argument("--backbone-ckpt", type=str, default="checkpoints/jaisp_v5/best.pt")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"])

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--use-projector-tokens", action="store_true", default=False)

    parser.add_argument("--head-depth", type=int, default=2)
    parser.add_argument("--head-heads", type=int, default=8)
    parser.add_argument("--head-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--stem-ch", type=int, default=64)
    parser.add_argument("--skip-proj", type=int, default=16)

    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--train-backbone", dest="freeze_backbone", action="store_false")

    parser.add_argument("--min-context", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=9)
    parser.add_argument(
        "--context-policy",
        type=str,
        default="all",
        choices=["all", "same_survey", "rubin_target_rubin_only"],
    )

    parser.add_argument("--mask-random", type=float, default=0.50)
    parser.add_argument("--mask-object", type=float, default=0.40)
    parser.add_argument("--mask-hard", type=float, default=0.10)
    parser.add_argument("--mask-value", type=float, default=0.0)

    parser.add_argument("--unmasked-weight", type=float, default=0.0)
    parser.add_argument("--source-loss-weight", type=float, default=1.5)
    parser.add_argument("--source-snr-threshold", type=float, default=2.0)
    parser.add_argument("--ssim-weight", type=float, default=0.10)
    parser.add_argument("--grad-weight", type=float, default=0.05)
    parser.add_argument("--flux-loss-weight", type=float, default=0.0)
    parser.add_argument("--bg-stat-loss-weight", type=float, default=0.0)
    parser.add_argument("--bg-snr-threshold", type=float, default=1.5)
    parser.add_argument("--flux-min-pixels", type=float, default=16.0)
    parser.add_argument("--bg-min-pixels", type=float, default=128.0)
    parser.add_argument("--predict-noise-units", dest="predict_noise_units", action="store_true")
    parser.add_argument("--predict-raw-target", dest="predict_noise_units", action="store_false")
    parser.set_defaults(predict_noise_units=True)
    parser.add_argument("--stem-clamp-min", type=float, default=-10.0)
    parser.add_argument("--stem-clamp-max", type=float, default=100.0)
    parser.add_argument("--target-clamp-min", type=float, default=-10.0)
    parser.add_argument("--target-clamp-max", type=float, default=100.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--no-mask-curriculum", action="store_true")
    parser.add_argument("--curriculum-epochs", type=int, default=8)

    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--val-max-batches", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--val-num-workers", type=int, default=0)
    parser.add_argument("--val-seed", type=int, default=12345)
    parser.add_argument(
        "--model-selection",
        type=str,
        default="masked",
        choices=["masked", "total"],
        help="Metric used to choose best checkpoint (masked=masked L1, total=full weighted objective).",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="JAISP-Reconstruction")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--wandb-log-images-every", type=int, default=1)
    parser.add_argument("--wandb-log-detailed", action="store_true")

    return parser


if __name__ == "__main__":
    parser = build_argparser()
    train(parser.parse_args())
