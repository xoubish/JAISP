import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
from head import MaskedReconstructionHead, interpolate_tokens
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
    embed_dim: int,
    proj_dim: int,
    depth: int,
    patch_size: int,
) -> JAISPFoundationV5:
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
        print(f"Loaded backbone checkpoint: {checkpoint_path}")
        print(f"  missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    else:
        print("No backbone checkpoint provided; starting from random initialization.")

    return model


def to_token_mask(pixel_mask: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
    # pixel_mask: [1,H,W] -> token mask [1,N]
    tm = F.interpolate(
        pixel_mask.unsqueeze(0), size=(int(grid_size[0]), int(grid_size[1])), mode="nearest"
    )
    return tm.view(1, -1)


def encode_target_and_context(
    backbone: JAISPFoundationV5,
    target_masked: torch.Tensor,
    target_rms: torch.Tensor,
    target_band: str,
    context_images: List[torch.Tensor],
    context_rms: List[torch.Tensor],
    context_bands: List[str],
    device: torch.device,
    freeze_backbone: bool,
    use_projector_tokens: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Returns:
      target_tokens: [1,N,D]
      context_tokens: [1,N,D] (aggregated and aligned to target grid)
      target_grid: (Ht, Wt)
    """

    def _encode(image: torch.Tensor, rms: torch.Tensor, band: str):
        image = image.unsqueeze(0).to(device)  # [1,1,H,W]
        rms = rms.unsqueeze(0).to(device)
        if use_projector_tokens:
            if freeze_backbone:
                with torch.no_grad():
                    out = backbone.encode(image, rms, band)
            else:
                out = backbone.encode(image, rms, band)
            return out["z"], out["grid_size"]

        if freeze_backbone:
            with torch.no_grad():
                feat = backbone.stems[band](image, rms)
                tokens, grid_size = backbone.encoder(feat)
        else:
            feat = backbone.stems[band](image, rms)
            tokens, grid_size = backbone.encoder(feat)
        return tokens, grid_size

    target_tokens, target_grid = _encode(target_masked, target_rms, target_band)

    aligned_context = []
    for img, rms, band in zip(context_images, context_rms, context_bands):
        ctx_tokens, ctx_grid = _encode(img, rms, str(band))
        ctx = interpolate_tokens(ctx_tokens, ctx_grid, target_grid)
        aligned_context.append(ctx)

    if aligned_context:
        context_tokens = torch.stack(aligned_context, dim=0).mean(dim=0)
    else:
        context_tokens = torch.zeros_like(target_tokens)

    return target_tokens, context_tokens, target_grid


def compute_losses(
    pred: torch.Tensor,
    target_image: torch.Tensor,
    target_rms: torch.Tensor,
    pixel_mask: torch.Tensor,
    unmasked_weight: float,
    predict_noise_units: bool,
    target_clamp_min: float,
    target_clamp_max: float,
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

    loss_masked = (abs_err * mask_crop).sum() / masked_den
    loss_unmasked = (abs_err * (1.0 - mask_crop)).sum() / unmasked_den
    loss_total = loss_masked + float(unmasked_weight) * loss_unmasked

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
        "psnr_masked": psnr_masked,
        "mask_frac": mask_crop.mean(),
    }


def _norm_for_display(img_2d: np.ndarray) -> np.ndarray:
    x = np.asarray(img_2d, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    p1, p99 = np.percentile(finite, [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        p1 = float(finite.min())
        p99 = float(finite.max()) if finite.max() > finite.min() else float(finite.min() + 1.0)
    x = np.nan_to_num(x, nan=p1, posinf=p99, neginf=p1)
    x = np.clip((x - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
    return x


def _make_preview_image(preview: Dict, epoch: int):
    import matplotlib.pyplot as plt

    target = _norm_for_display(preview["target"])
    masked = _norm_for_display(preview["masked"])
    pred = _norm_for_display(preview["pred"])
    err = _norm_for_display(np.abs(preview["pred"] - preview["target"]))
    mask = np.asarray(preview["mask"], dtype=np.float32)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
    axes[0].imshow(target, cmap="gray")
    axes[0].set_title("Target")
    axes[1].imshow(masked, cmap="gray")
    axes[1].set_title("Masked Input")
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Prediction")
    axes[3].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Mask")
    axes[4].imshow(err, cmap="magma")
    axes[4].set_title("|Pred-Target|")

    for ax in axes:
        ax.axis("off")

    title = (
        f"Epoch {epoch} | target={preview['target_band']} | "
        f"context={','.join(preview['context_bands'])} | mask={preview['mask_type']}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


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
    )

    backbone = load_backbone(
        device=device,
        checkpoint_path=args.backbone_ckpt,
        embed_dim=args.embed_dim,
        proj_dim=args.proj_dim,
        depth=args.depth,
        patch_size=args.patch_size,
    )

    head = MaskedReconstructionHead(
        embed_dim=(args.proj_dim if args.use_projector_tokens else args.embed_dim),
        patch_size=args.patch_size,
        depth=args.head_depth,
        num_heads=args.head_heads,
        mlp_ratio=args.head_mlp_ratio,
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

    mask_probs = {
        "random": args.mask_random,
        "object": args.mask_object,
        "hard": args.mask_hard,
    }

    print(f"Device: {device}")
    print(f"Tiles: {len(dataset)}, steps/epoch: {len(loader)}")
    print(
        f"Context bands per sample: [{args.min_context}, {args.max_context}] | "
        f"mask mix random/object/hard = {args.mask_random:.2f}/{args.mask_object:.2f}/{args.mask_hard:.2f}"
    )
    print(
        f"Reconstruction tokens: {'projector(z)' if args.use_projector_tokens else 'encoder(pre-projector)'} | "
        f"target space: {'noise units (image/rms)' if args.predict_noise_units else 'raw image units'}"
    )

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

            agg = {
                "loss_total": 0.0,
                "loss_masked": 0.0,
                "loss_unmasked": 0.0,
                "psnr_masked": 0.0,
                "mask_frac": 0.0,
                "num_samples": 0,
                "context_count_sum": 0.0,
            }
            mask_type_counts = {"random": 0, "object": 0, "hard": 0}
            preview = None

            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
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

                    valid = [j for j, b in enumerate(context_bands) if b in backbone.band_names]
                    if not valid:
                        continue
                    context_images = [context_images[j] for j in valid]
                    context_rms = [context_rms[j] for j in valid]
                    context_bands = [context_bands[j] for j in valid]

                    pixel_mask, mask_type = build_mask(target_image, mask_probs)
                    mask_type_counts[mask_type] = mask_type_counts.get(mask_type, 0) + 1
                    target_masked = target_image * (1.0 - pixel_mask) + args.mask_value * pixel_mask

                    target_tokens, context_tokens, target_grid = encode_target_and_context(
                        backbone=backbone,
                        target_masked=target_masked,
                        target_rms=target_rms,
                        target_band=target_band,
                        context_images=context_images,
                        context_rms=context_rms,
                        context_bands=context_bands,
                        device=device,
                        freeze_backbone=args.freeze_backbone,
                        use_projector_tokens=args.use_projector_tokens,
                    )

                    token_mask = to_token_mask(pixel_mask, target_grid)
                    pred = head(
                        target_tokens=target_tokens,
                        context_tokens=context_tokens,
                        token_mask=token_mask,
                        grid_size=target_grid,
                    )

                    metrics = compute_losses(
                        pred=pred,
                        target_image=target_image,
                        target_rms=target_rms,
                        pixel_mask=pixel_mask,
                        unmasked_weight=args.unmasked_weight,
                        predict_noise_units=args.predict_noise_units,
                        target_clamp_min=args.target_clamp_min,
                        target_clamp_max=args.target_clamp_max,
                    )

                    sample_losses.append(metrics["loss_total"])
                    agg["loss_total"] += float(metrics["loss_total"].detach().item())
                    agg["loss_masked"] += float(metrics["loss_masked"].detach().item())
                    agg["loss_unmasked"] += float(metrics["loss_unmasked"].detach().item())
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
                        h_pred, w_pred = pred.shape[-2], pred.shape[-1]
                        preview = {
                            "target": target_image[:, :h_pred, :w_pred].detach().cpu().numpy()[0],
                            "masked": target_masked[:, :h_pred, :w_pred].detach().cpu().numpy()[0],
                            "pred": (
                                pred.detach().cpu().numpy()[0, 0]
                                if not args.predict_noise_units
                                else (
                                    pred.detach().cpu().numpy()[0, 0]
                                    * target_rms[:, :h_pred, :w_pred].detach().cpu().numpy()[0]
                                )
                            ),
                            "mask": pixel_mask[:, :h_pred, :w_pred].detach().cpu().numpy()[0],
                            "target_band": target_band,
                            "context_bands": context_bands,
                            "mask_type": mask_type,
                        }

                if not sample_losses:
                    continue

                loss = torch.stack(sample_losses).mean()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                global_step += 1

            scheduler.step()

            n = max(1, agg["num_samples"])
            epoch_metrics = {
                "loss_total": agg["loss_total"] / n,
                "loss_masked": agg["loss_masked"] / n,
                "loss_unmasked": agg["loss_unmasked"] / n,
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

            print(
                f"Epoch {epoch:03d} | "
                f"loss={epoch_metrics['loss_total']:.4f} "
                f"masked={epoch_metrics['loss_masked']:.4f} "
                f"psnr_masked={epoch_metrics['psnr_masked']:.2f} "
                f"mask_frac={epoch_metrics['mask_frac']:.3f} "
                f"context_mean={epoch_metrics['context_count_mean']:.2f} "
                f"samples={epoch_metrics['samples']} "
                f"t={epoch_metrics['epoch_time_sec']:.1f}s"
            )

            if wandb_run is not None:
                mask_total = max(1, sum(mask_type_counts.values()))
                wb = {
                    "train/loss_total": epoch_metrics["loss_total"],
                    "train/loss_masked": epoch_metrics["loss_masked"],
                    "train/loss_unmasked": epoch_metrics["loss_unmasked"],
                    "train/psnr_masked": epoch_metrics["psnr_masked"],
                    "train/mask_frac": epoch_metrics["mask_frac"],
                    "train/context_count_mean": epoch_metrics["context_count_mean"],
                    "train/samples": epoch_metrics["samples"],
                    "train/lr_head": epoch_metrics["lr_head"],
                    "train/epoch_time_sec": epoch_metrics["epoch_time_sec"],
                    "train/mask_count_random": mask_type_counts["random"],
                    "train/mask_count_object": mask_type_counts["object"],
                    "train/mask_count_hard": mask_type_counts["hard"],
                    "train/mask_frac_random": mask_type_counts["random"] / mask_total,
                    "train/mask_frac_object": mask_type_counts["object"] / mask_total,
                    "train/mask_frac_hard": mask_type_counts["hard"] / mask_total,
                }
                if "lr_backbone" in epoch_metrics:
                    wb["train/lr_backbone"] = epoch_metrics["lr_backbone"]
                if preview is not None:
                    wb["train/preview"] = _make_preview_image(preview, epoch)
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

            if epoch_metrics["loss_total"] < best_loss:
                best_loss = epoch_metrics["loss_total"]
                torch.save(ckpt, out_dir / "best_reconstruction.pt")
                if wandb_run is not None:
                    wandb_run.summary["best_loss_total"] = float(best_loss)
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

    parser.add_argument("--rubin-dir", type=str, required=True)
    parser.add_argument("--euclid-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/jaisp_reconstruction")
    parser.add_argument("--backbone-ckpt", type=str, default="checkpoints/jaisp_v5/best.pt")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--use-projector-tokens", action="store_true", default=False)

    parser.add_argument("--head-depth", type=int, default=2)
    parser.add_argument("--head-heads", type=int, default=8)
    parser.add_argument("--head-mlp-ratio", type=float, default=4.0)

    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--train-backbone", dest="freeze_backbone", action="store_false")

    parser.add_argument("--min-context", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=9)

    parser.add_argument("--mask-random", type=float, default=0.50)
    parser.add_argument("--mask-object", type=float, default=0.40)
    parser.add_argument("--mask-hard", type=float, default=0.10)
    parser.add_argument("--mask-value", type=float, default=0.0)

    parser.add_argument("--unmasked-weight", type=float, default=0.10)
    parser.add_argument("--predict-noise-units", dest="predict_noise_units", action="store_true")
    parser.add_argument("--predict-raw-target", dest="predict_noise_units", action="store_false")
    parser.set_defaults(predict_noise_units=True)
    parser.add_argument("--target-clamp-min", type=float, default=-10.0)
    parser.add_argument("--target-clamp-max", type=float, default=100.0)
    parser.add_argument("--save-every", type=int, default=5)

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

    return parser


if __name__ == "__main__":
    parser = build_argparser()
    train(parser.parse_args())
