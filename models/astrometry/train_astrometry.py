"""
Train JAISP astrometry concordance head.

Self-supervised: applies synthetic offset fields to Rubin images,
trains the head to recover them using cross-correlation at the token level.

Usage:
    python train_astrometry.py \
        --rubin-dir ../data/rubin_tiles_ecdfs \
        --euclid-dir ../data/euclid_tiles_ecdfs \
        --backbone-ckpt ../checkpoints/jaisp_v5/best.pt
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from jaisp_dataset_v4 import ALL_BANDS
from jaisp_foundation_v5 import JAISPFoundationV5

from head import AstrometryConcordanceHead, PIXEL_SCALES, interpolate_tokens
from dataset import make_astrometry_loader
from offsets import apply_offset_to_image, resample_offset_to_grid

try:
    import wandb
except ImportError:
    wandb = None


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_band(backbone, image, rms, band, device, freeze):
    """Encode a single band, return tokens + grid size."""
    image = image.unsqueeze(0).to(device)  # [1, 1, H, W]
    rms = rms.unsqueeze(0).to(device)
    if freeze:
        with torch.no_grad():
            feat = backbone.stems[band](image, rms)
            tokens, grid_size = backbone.encoder(feat)
    else:
        feat = backbone.stems[band](image, rms)
        tokens, grid_size = backbone.encoder(feat)
    return tokens, grid_size


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def compute_loss(
    pred_dra: torch.Tensor,
    pred_ddec: torch.Tensor,
    gt_dra: torch.Tensor,
    gt_ddec: torch.Tensor,
    smooth_weight: float = 0.1,
) -> dict:
    """
    Args:
        pred_dra, pred_ddec: [B, 1, H, W] predicted offsets in arcseconds
        gt_dra, gt_ddec: [B, 1, H, W] ground truth in arcseconds
        smooth_weight: penalty on spatial gradients of the predicted field

    Returns dict with loss components.
    """
    # Primary: smooth L1 (less sensitive to outliers than L1).
    loss_ra = F.smooth_l1_loss(pred_dra, gt_dra)
    loss_dec = F.smooth_l1_loss(pred_ddec, gt_ddec)
    loss_offset = loss_ra + loss_dec

    # Smoothness: penalize spatial gradients of the predicted field.
    # Astrometric distortions should be smooth, so large gradients are suspicious.
    def _gradient_penalty(field):
        gx = (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean()
        gy = (field[:, :, 1:, :] - field[:, :, :-1, :]).abs().mean()
        return gx + gy

    loss_smooth = _gradient_penalty(pred_dra) + _gradient_penalty(pred_ddec)

    loss_total = loss_offset + smooth_weight * loss_smooth

    # Metrics.
    with torch.no_grad():
        err_ra = (pred_dra - gt_dra).abs()
        err_dec = (pred_ddec - gt_ddec).abs()
        mae_ra = err_ra.mean()
        mae_dec = err_dec.mean()
        # Total positional error in arcseconds.
        pos_err = torch.sqrt(err_ra ** 2 + err_dec ** 2 + 1e-10)
        mae_total = pos_err.mean()
        # Fraction of positions with error < thresholds.
        frac_01 = (pos_err < 0.1).float().mean()  # < 100 mas
        frac_02 = (pos_err < 0.2).float().mean()  # < 200 mas

    return {
        "loss_total": loss_total,
        "loss_offset": loss_offset,
        "loss_smooth": loss_smooth,
        "mae_ra": mae_ra,
        "mae_dec": mae_dec,
        "mae_total": mae_total,
        "frac_01arcsec": frac_01,
        "frac_02arcsec": frac_02,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_preview(pred_dra, pred_ddec, gt_dra, gt_ddec, rubin_band, mode, epoch):
    """Create a wandb image showing predicted vs ground truth offset fields."""
    import matplotlib.pyplot as plt

    pd_ra = pred_dra[0, 0].detach().cpu().numpy()
    pd_de = pred_ddec[0, 0].detach().cpu().numpy()
    gt_ra = gt_dra[0, 0].detach().cpu().numpy()
    gt_de = gt_ddec[0, 0].detach().cpu().numpy()
    err_ra = np.abs(pd_ra - gt_ra)
    err_de = np.abs(pd_de - gt_de)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    vmax_ra = max(abs(gt_ra.min()), abs(gt_ra.max()), 0.01)
    vmax_de = max(abs(gt_de.min()), abs(gt_de.max()), 0.01)

    im = axes[0, 0].imshow(gt_ra, cmap="RdBu_r", vmin=-vmax_ra, vmax=vmax_ra, origin="lower")
    axes[0, 0].set_title("GT ΔRA* (arcsec)")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im = axes[0, 1].imshow(pd_ra, cmap="RdBu_r", vmin=-vmax_ra, vmax=vmax_ra, origin="lower")
    axes[0, 1].set_title("Pred ΔRA*")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = axes[0, 2].imshow(err_ra, cmap="magma", vmin=0, origin="lower")
    axes[0, 2].set_title("|Error| ΔRA*")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = axes[1, 0].imshow(gt_de, cmap="RdBu_r", vmin=-vmax_de, vmax=vmax_de, origin="lower")
    axes[1, 0].set_title("GT ΔDec (arcsec)")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(pd_de, cmap="RdBu_r", vmin=-vmax_de, vmax=vmax_de, origin="lower")
    axes[1, 1].set_title("Pred ΔDec")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    im = axes[1, 2].imshow(err_de, cmap="magma", vmin=0, origin="lower")
    axes[1, 2].set_title("|Error| ΔDec")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    fig.suptitle(f"Epoch {epoch} | {rubin_band} → VIS | mode={mode}", fontsize=12)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data.
    dataset, loader = make_astrometry_loader(
        rubin_dir=args.rubin_dir,
        euclid_dir=args.euclid_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_offset_arcsec=args.max_offset,
        curriculum_epochs=args.curriculum_epochs,
        seed=args.seed,
        augment=not args.no_augment,
    )

    # Backbone.
    backbone = load_backbone(
        device, args.backbone_ckpt,
        args.embed_dim, args.proj_dim, args.depth, args.patch_size,
    )
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    print("Backbone frozen.")

    # Head.
    head = AstrometryConcordanceHead(
        embed_dim=args.embed_dim,
        search_radius=args.search_radius,
        softmax_temp=args.softmax_temp,
        refine_hidden=args.refine_hidden,
        refine_depth=args.refine_depth,
        patch_size=args.patch_size,
    ).to(device)

    n_params = sum(p.numel() for p in head.parameters())
    print(f"Head parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # W&B.
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
    rubin_pixel_scale = 0.2
    vis_pixel_scale = 0.1

    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            head.train()
            dataset.epoch = epoch  # update curriculum

            agg = defaultdict(float)
            mode_counts = defaultdict(int)
            n_samples = 0
            preview_data = None

            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                sample_losses = []

                bsz = len(batch["rubin_band"])
                for i in range(bsz):
                    if not batch["has_vis"][i]:
                        continue

                    rubin_band = batch["rubin_band"][i]
                    if rubin_band not in backbone.band_names:
                        continue

                    rubin_img = batch["rubin_image"][i].float().to(device)   # [1, H_r, W_r]
                    rubin_rms = batch["rubin_rms"][i].float().to(device)
                    vis_img = batch["vis_image"][i].float().to(device)       # [1, H_v, W_v]
                    vis_rms = batch["vis_rms"][i].float().to(device)
                    gt_dra = batch["gt_dra"][i].to(device)                   # [H_r, W_r]
                    gt_ddec = batch["gt_ddec"][i].to(device)
                    offset_mode = batch["offset_mode"][i]
                    mode_counts[offset_mode] += 1

                    # Apply synthetic offset to Rubin image.
                    rubin_shifted = apply_offset_to_image(
                        rubin_img.unsqueeze(0), gt_dra, gt_ddec, rubin_pixel_scale,
                    )  # [1, 1, H_r, W_r]

                    # Encode both bands.
                    rubin_tokens, rubin_grid = encode_band(
                        backbone, rubin_shifted.squeeze(0), rubin_rms,
                        rubin_band, device, freeze=True,
                    )
                    vis_tokens, vis_grid = encode_band(
                        backbone, vis_img, vis_rms,
                        "euclid_VIS", device, freeze=True,
                    )

                    # Head forward.
                    H_vis, W_vis = vis_img.shape[-2], vis_img.shape[-1]
                    out = head(
                        rubin_tokens=rubin_tokens,
                        vis_tokens=vis_tokens,
                        rubin_grid=rubin_grid,
                        vis_grid=vis_grid,
                        vis_image_hw=(H_vis, W_vis),
                        vis_pixel_scale=vis_pixel_scale,
                    )

                    # Ground truth on the same grid as predictions.
                    pred_h, pred_w = out["dra"].shape[-2], out["dra"].shape[-1]
                    gt_dra_grid, gt_ddec_grid = resample_offset_to_grid(
                        gt_dra.cpu().numpy(), gt_ddec.cpu().numpy(), pred_h, pred_w,
                    )
                    gt_dra_grid = gt_dra_grid.unsqueeze(0).to(device)     # [1, 1, H, W]
                    gt_ddec_grid = gt_ddec_grid.unsqueeze(0).to(device)

                    metrics = compute_loss(
                        out["dra"], out["ddec"],
                        gt_dra_grid, gt_ddec_grid,
                        smooth_weight=args.smooth_weight,
                    )

                    sample_losses.append(metrics["loss_total"])

                    # Accumulate.
                    for k, v in metrics.items():
                        agg[k] += float(v.detach().item()) if torch.is_tensor(v) else float(v)
                    n_samples += 1

                    # Save preview from first sample.
                    if preview_data is None and wandb_run is not None:
                        preview_data = {
                            "pred_dra": out["dra"].detach(),
                            "pred_ddec": out["ddec"].detach(),
                            "gt_dra": gt_dra_grid.detach(),
                            "gt_ddec": gt_ddec_grid.detach(),
                            "rubin_band": rubin_band,
                            "mode": offset_mode,
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

            # Epoch summary.
            n = max(1, n_samples)
            epoch_metrics = {k: v / n for k, v in agg.items()}
            epoch_metrics["epoch"] = epoch
            epoch_metrics["lr"] = optimizer.param_groups[0]["lr"]
            epoch_metrics["temp"] = float(head.temperature.detach().item())
            epoch_metrics["time_sec"] = time.time() - t0
            epoch_metrics["samples"] = n_samples

            print(
                f"Epoch {epoch:03d} | "
                f"loss={epoch_metrics['loss_total']:.5f} "
                f"MAE_total={epoch_metrics['mae_total']*1000:.1f}mas "
                f"MAE_ra={epoch_metrics['mae_ra']*1000:.1f}mas "
                f"MAE_dec={epoch_metrics['mae_dec']*1000:.1f}mas "
                f"<0.1\"={epoch_metrics['frac_01arcsec']:.1%} "
                f"<0.2\"={epoch_metrics['frac_02arcsec']:.1%} "
                f"temp={epoch_metrics['temp']:.4f} "
                f"modes={dict(mode_counts)} "
                f"t={epoch_metrics['time_sec']:.1f}s"
            )

            if wandb_run is not None:
                wb = {f"train/{k}": v for k, v in epoch_metrics.items()}
                if preview_data is not None and epoch % args.vis_every == 0:
                    wb["train/preview"] = make_preview(
                        preview_data["pred_dra"], preview_data["pred_ddec"],
                        preview_data["gt_dra"], preview_data["gt_ddec"],
                        preview_data["rubin_band"], preview_data["mode"], epoch,
                    )
                wandb_run.log(wb, step=epoch)

            # Checkpoint.
            ckpt = {
                "epoch": epoch,
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": epoch_metrics,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "last_astrometry.pt")

            score = epoch_metrics["loss_total"]
            if score < best_loss:
                best_loss = score
                torch.save(ckpt, out_dir / "best_astrometry.pt")
                if wandb_run is not None:
                    wandb_run.summary["best_mae_total_mas"] = epoch_metrics["mae_total"] * 1000
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Train JAISP astrometry concordance head.")

    p.add_argument("--rubin-dir", type=str, required=True)
    p.add_argument("--euclid-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="checkpoints/jaisp_astrometry")
    p.add_argument("--backbone-ckpt", type=str, default="models/checkpoints/jaisp_v5/best.pt")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--patch-size", type=int, default=16)

    p.add_argument("--search-radius", type=int, default=3)
    p.add_argument("--softmax-temp", type=float, default=0.1)
    p.add_argument("--refine-hidden", type=int, default=32)
    p.add_argument("--refine-depth", type=int, default=4)

    p.add_argument("--max-offset", type=float, default=0.5,
                   help="Max synthetic offset amplitude in arcseconds")
    p.add_argument("--smooth-weight", type=float, default=0.1)
    p.add_argument("--curriculum-epochs", type=int, default=10)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--vis-every", type=int, default=1)

    p.add_argument("--wandb-project", type=str, default="JAISP-Astrometry")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])

    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
