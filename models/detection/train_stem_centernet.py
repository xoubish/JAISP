"""Train the high-resolution stem-based CenterNet detector.

This variant keeps the V7 BandStem modules at native resolution and fuses
them directly in the VIS frame, rather than training on the coarse 0.8"/px
bottleneck.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from detection.centernet_loss import CenterNetLoss
from detection.dataset import TileDetectionDataset, collate_fn
from detection.stem_centernet_detector import (
    StemCenterNetDetector,
    load_v7_foundation_from_checkpoint,
)
from detection.train_centernet import _log_tile


def _apply_extra_labels_inplace(ds: TileDetectionDataset, extra_labels_path: str) -> None:
    if not extra_labels_path:
        return
    extra_path = Path(extra_labels_path)
    if not extra_path.exists():
        raise FileNotFoundError(extra_path)

    extra = torch.load(extra_path, map_location="cpu", weights_only=False)
    promoted = extra.get("promoted", {})
    demoted = extra.get("demoted", {})

    tile_id_to_idx = {
        ds._base.tiles[idx]["tile_id"]: idx
        for idx in range(len(ds._base.tiles))
    }

    n_added = 0
    n_removed = 0

    for tile_id, remove_xy in demoted.items():
        idx = tile_id_to_idx.get(tile_id)
        if idx is None or len(remove_xy) == 0:
            continue
        centroids_np, classes_np, H, W = ds._label_cache[idx]
        if len(centroids_np) == 0:
            continue
        keep = np.ones(len(centroids_np), dtype=bool)
        for j in range(len(remove_xy)):
            d = np.sqrt(((centroids_np - remove_xy[j]) ** 2).sum(axis=1))
            keep &= (d > 0.005)
        n_before = len(centroids_np)
        ds._label_cache[idx] = (centroids_np[keep], classes_np[keep], H, W)
        n_removed += n_before - int(keep.sum())

    for tile_id, add_xy in promoted.items():
        idx = tile_id_to_idx.get(tile_id)
        if idx is None or len(add_xy) == 0:
            continue
        centroids_np, classes_np, H, W = ds._label_cache[idx]
        merged_c = np.concatenate([centroids_np, add_xy.astype(np.float32)], axis=0)
        merged_cls = np.zeros(len(merged_c), dtype=np.int64)
        ds._label_cache[idx] = (merged_c, merged_cls, H, W)
        n_added += len(add_xy)

    print(f"  Label refinement applied: +{n_added} promoted, -{n_removed} demoted")


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training stem-based CenterNet detector on {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    full_ds = TileDetectionDataset(
        rubin_dir=args.rubin_dir,
        euclid_dir=args.euclid_dir,
        nsig=args.nsig,
        max_sources=1000,
        use_all_bands=(args.euclid_dir is not None),
        augment=True,
    )
    _apply_extra_labels_inplace(full_ds, args.extra_labels)

    n_val = max(1, int(0.1 * len(full_ds)))
    n_tr = len(full_ds) - n_val
    tr_ds, val_ds = random_split(
        full_ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    val_workers = max(1, args.num_workers // 2) if args.num_workers > 0 else 0
    tr_loader = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=val_workers,
    )
    print(f"Train: {n_tr} samples   Val: {n_val} samples")

    foundation = load_v7_foundation_from_checkpoint(args.encoder_ckpt, device=device)
    model = StemCenterNetDetector(
        foundation=foundation,
        stream_ch=args.stream_ch,
        base_ch=args.base_ch,
        freeze_stems=not args.unfreeze_stems,
    ).to(device)

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"  [warn] Missing init-checkpoint keys: {missing}")
        if unexpected:
            print(f"  [warn] Unexpected init-checkpoint keys: {unexpected}")
        print(f"  Initialized detector weights from {args.init_checkpoint}")

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_total/1e6:.2f}M total, {n_train/1e6:.2f}M trainable")

    criterion = CenterNetLoss(sigma=args.sigma)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    step = 0

    prev_augment = full_ds._base.augment
    full_ds._base.augment = False
    try:
        viz_sample = collate_fn([val_ds[len(val_ds) // 2]])
    finally:
        full_ds._base.augment = prev_augment

    for epoch in range(args.epochs):
        model.train()
        tr_losses = []
        for batch in tr_loader:
            images = {b: v.to(device) for b, v in batch["images"].items()}
            rms = {b: v.to(device) for b, v in batch["rms"].items()}

            out = model(images, rms)
            losses = criterion(out, [c.to(device) for c in batch["centroids"]])

            optimizer.zero_grad()
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_losses.append(float(losses["loss_total"]))
            step += 1
            if use_wandb and step % 10 == 0:
                wandb.log({
                    "train/loss": float(losses["loss_total"]),
                    "train/loss_hm": float(losses["loss_hm"]),
                    "train/loss_off": float(losses["loss_off"]),
                    "train/loss_flux": float(losses["loss_flux"]),
                    "train/n_sources": float(losses["n_sources"]),
                }, step=step)

        scheduler.step()
        mean_tr = float(np.mean(tr_losses)) if tr_losses else float("nan")

        model.eval()
        val_losses = []
        prev_augment = full_ds._base.augment
        full_ds._base.augment = False
        with torch.no_grad():
            try:
                for batch in val_loader:
                    images = {b: v.to(device) for b, v in batch["images"].items()}
                    rms = {b: v.to(device) for b, v in batch["rms"].items()}
                    out = model(images, rms)
                    losses = criterion(out, [c.to(device) for c in batch["centroids"]])
                    val_losses.append(float(losses["loss_total"]))
            finally:
                full_ds._base.augment = prev_augment
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1:3d}/{args.epochs}  tr={mean_tr:.4f}  val={mean_val:.4f}  lr={lr_now:.2e}")

        if use_wandb:
            log = {
                "train/loss_epoch": mean_tr,
                "val/loss": mean_val,
                "train/lr": lr_now,
                "epoch": epoch + 1,
            }
            if (epoch + 1) % 5 == 0 or epoch == 0:
                try:
                    with torch.no_grad():
                        sim = {b: v.to(device) for b, v in viz_sample["images"].items()}
                        srm = {b: v.to(device) for b, v in viz_sample["rms"].items()}
                        sout = model(sim, srm)
                    log["viz/tile"] = _log_tile(viz_sample, sout, wandb, step, euclid_dir=args.euclid_dir)
                except Exception as exc:
                    print(f"  [warn] viz failed: {exc}")
            wandb.log(log, step=step)

        if mean_val < best_val:
            best_val = mean_val
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            model.save(args.out)
            print(f"  ✓ saved -> {args.out}")
            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val

    if use_wandb:
        wandb.finish()
    print(f"Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rubin_dir", required=True)
    p.add_argument("--euclid_dir", default=None)
    p.add_argument("--encoder_ckpt", required=True, help="Foundation V7 checkpoint")
    p.add_argument("--extra_labels", default=None, help="Refined labels from previous self-training round")
    p.add_argument("--init_checkpoint", default=None, help="Initialize from an existing stem detector checkpoint")
    p.add_argument("--out", default="../checkpoints/stem_centernet_v7.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--nsig", type=float, default=3.0)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--stream_ch", type=int, default=16)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--unfreeze_stems", action="store_true",
                   help="Allow the reused V7 BandStem weights to keep training")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run", default=None)
    p.add_argument("--device", default="", help='Device to use: "cuda", "cuda:1", "cpu" (default: auto)')
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    train(p.parse_args())
