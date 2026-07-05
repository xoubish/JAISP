"""Train the high-resolution stem-based CenterNet detector.

This variant keeps the V7 BandStem modules at native resolution and fuses
them directly in the VIS frame, rather than training on the coarse 0.8"/px
bottleneck.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from detection.centernet_loss import CenterNetLoss
from detection.dataset import TileDetectionDataset, _vis_bright_core_and_spike_mask, collate_fn
from jaisp_foundation_v10 import RUBIN_BANDS, EUCLID_BANDS
from detection.stem_centernet_detector import (
    StemCenterNetDetector,
    load_foundation_from_checkpoint,
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
        if hasattr(ds, "_weight_cache") and idx in ds._weight_cache:
            ds._weight_cache[idx] = ds._weight_cache[idx][keep]
        n_removed += n_before - int(keep.sum())

    for tile_id, add_xy in promoted.items():
        idx = tile_id_to_idx.get(tile_id)
        if idx is None or len(add_xy) == 0:
            continue
        centroids_np, classes_np, H, W = ds._label_cache[idx]
        merged_c = np.concatenate([centroids_np, add_xy.astype(np.float32)], axis=0)
        merged_cls = np.zeros(len(merged_c), dtype=np.int64)
        ds._label_cache[idx] = (merged_c, merged_cls, H, W)
        if hasattr(ds, "_weight_cache"):
            old_w = ds._weight_cache.get(idx, np.ones(len(centroids_np), dtype=np.float32))
            add_w = np.full(len(add_xy), 0.6, dtype=np.float32)
            ds._weight_cache[idx] = np.concatenate([old_w, add_w], axis=0)
        n_added += len(add_xy)

    print(f"  Label refinement applied: +{n_added} promoted, -{n_removed} demoted")


def _load_teacher_label_dict(path: str) -> dict[str, np.ndarray]:
    if not path:
        return {}
    data = torch.load(path, map_location="cpu", weights_only=False)
    labels = data.get("labels", data) if isinstance(data, dict) else data
    out = {}
    for tile_id, value in labels.items():
        xy = value[0] if isinstance(value, (tuple, list)) else value
        if torch.is_tensor(xy):
            xy = xy.detach().cpu().numpy()
        xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        out[str(tile_id)] = xy
    return out


def _teacher_for_tile(teacher: dict[str, np.ndarray], tile_id: str) -> np.ndarray:
    return teacher.get(tile_id, teacher.get(tile_id.replace("_euclid", ""), np.zeros((0, 2), dtype=np.float32)))


def _points_near(points: np.ndarray, refs: np.ndarray, radius: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    refs = np.asarray(refs, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    if len(refs) == 0 or radius <= 0:
        return np.zeros(len(points), dtype=bool)
    d2 = ((points[:, None, :] - refs[None, :, :]) ** 2).sum(axis=2)
    return d2.min(axis=1) <= float(radius) ** 2


def _points_in_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(points) == 0 or mask is None:
        return np.zeros(len(points), dtype=bool)
    h, w = mask.shape
    px = np.clip(np.round(points[:, 0] * (w - 1)).astype(int), 0, w - 1)
    py = np.clip(np.round(points[:, 1] * (h - 1)).astype(int), 0, h - 1)
    return mask[py, px]


def _disk_mask(shape: tuple[int, int], xs: np.ndarray, ys: np.ndarray, radius: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if radius <= 0:
        return mask
    r2 = float(radius) ** 2
    for x, y in zip(xs, ys):
        x0 = max(0, int(np.floor(x - radius)))
        x1 = min(w, int(np.ceil(x + radius)) + 1)
        y0 = max(0, int(np.floor(y - radius)))
        y1 = min(h, int(np.ceil(y + radius)) + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= ((xx - x) ** 2 + (yy - y) ** 2) <= r2
    return mask


def _dedupe_labels(labels: np.ndarray, radius: float = 0.0025) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32).reshape(-1, 2)
    if len(labels) <= 1:
        return labels
    keep = np.ones(len(labels), dtype=bool)
    for i in range(len(labels)):
        if not keep[i]:
            continue
        close = _points_near(labels[i + 1:], labels[i:i + 1], radius)
        if close.any():
            tail = keep[i + 1:]
            tail[close] = False
            keep[i + 1:] = tail
    return labels[keep]


def _apply_teacher_guidance_inplace(
    ds: TileDetectionDataset,
    teacher_labels_path: str = None,
    euclid_dir: str = None,
    teacher_match_radius: float = 0.012,
    teacher_filter_mode: str = "bright",
    teacher_add_labels: bool = True,
    bright_ignore_radius: int = 0,
    spike_radius: int = 40,
    spike_width: float = 3.0,
) -> None:
    """Clean StemCenterNet labels using fused CenterNet as a proposal teacher.

    Unmatched labels can be removed globally or only inside bright-star halo
    regions. Broad halo regions become ignore masks for negative heatmap loss,
    while the thin spike ridges remain hard negatives.
    """
    teacher = _load_teacher_label_dict(teacher_labels_path) if teacher_labels_path else {}
    if not teacher and bright_ignore_radius <= 0:
        return
    if teacher_filter_mode not in {"none", "bright", "global"}:
        raise ValueError("--teacher_filter_mode must be one of: none, bright, global")

    if getattr(ds, "_ignore_cache", None) is None:
        ds._ignore_cache = {}

    n_teacher_added = 0
    n_teacher_tiles = 0
    n_removed_teacher = 0
    n_removed_spike = 0
    n_ignore = 0

    euclid_dir_p = Path(euclid_dir) if euclid_dir else None
    for idx in range(len(ds._base.tiles)):
        tile = ds._base.tiles[idx]
        tile_id = tile["tile_id"]
        centroids_np, classes_np, h, w = ds._label_cache[idx]
        teacher_xy = _teacher_for_tile(teacher, tile_id)
        if len(teacher_xy) > 0:
            n_teacher_tiles += 1

        spike_mask = None
        broad_mask = None
        vis_path = tile.get("euclid_path")
        if (not vis_path or not Path(vis_path).exists()) and euclid_dir_p is not None:
            vis_path = euclid_dir_p / f"{tile_id}_euclid.npz"
        if vis_path and Path(vis_path).exists() and (bright_ignore_radius > 0 or spike_radius > 0):
            try:
                edata = np.load(str(vis_path), allow_pickle=True, mmap_mode="r")
                vis_img = np.nan_to_num(np.asarray(edata["img_VIS"], dtype=np.float32), nan=0.0)
                bright_xs, bright_ys, spike_mask = _vis_bright_core_and_spike_mask(
                    vis_img,
                    spike_radius=spike_radius,
                    spike_width=spike_width,
                    include_core=False,
                )
                if bright_ignore_radius > 0 and len(bright_xs) > 0:
                    broad_mask = _disk_mask(vis_img.shape, bright_xs, bright_ys, bright_ignore_radius)
                    # Ambiguous halo pixels are ignored, but spike ridges remain
                    # unignored so false detections on them are hard negatives.
                    ignore_mask = broad_mask & ~spike_mask
                    if idx in ds._ignore_cache:
                        ds._ignore_cache[idx] = ds._ignore_cache[idx] | ignore_mask
                    else:
                        ds._ignore_cache[idx] = ignore_mask
                    n_ignore += int(ignore_mask.sum())
            except Exception as exc:
                print(f"  [warn] teacher-guidance mask failed for {tile_id}: {exc}")

        labels = np.asarray(centroids_np, dtype=np.float32).reshape(-1, 2)
        keep = np.ones(len(labels), dtype=bool)

        if spike_mask is not None and len(labels) > 0:
            on_spike = _points_in_mask(labels, spike_mask)
            keep &= ~on_spike
            n_removed_spike += int(on_spike.sum())

        if teacher and teacher_filter_mode != "none" and len(labels) > 0:
            near_teacher = _points_near(labels, teacher_xy, teacher_match_radius)
            if teacher_filter_mode == "global":
                unsupported = ~near_teacher
            else:
                in_bright = _points_in_mask(labels, broad_mask) if broad_mask is not None else np.zeros(len(labels), dtype=bool)
                unsupported = in_bright & ~near_teacher
            n_removed_teacher += int((keep & unsupported).sum())
            keep &= ~unsupported

        labels = labels[keep]
        if hasattr(ds, "_weight_cache"):
            weights = ds._weight_cache.get(idx, np.ones(len(centroids_np), dtype=np.float32))
            weights = np.asarray(weights, dtype=np.float32).reshape(-1)
            if len(weights) == len(keep):
                weights = weights[keep]
            else:
                weights = np.ones(len(labels), dtype=np.float32)
        else:
            weights = np.ones(len(labels), dtype=np.float32)

        if teacher_add_labels and len(teacher_xy) > 0:
            teacher_keep = np.ones(len(teacher_xy), dtype=bool)
            if spike_mask is not None:
                teacher_keep &= ~_points_in_mask(teacher_xy, spike_mask)
            add_xy = teacher_xy[teacher_keep]
            if len(labels) > 0 and len(add_xy) > 0:
                add_xy = add_xy[~_points_near(add_xy, labels, radius=0.0025)]
            if len(add_xy) > 0:
                labels = np.concatenate([labels, add_xy.astype(np.float32)], axis=0)
                weights = np.concatenate([
                    weights,
                    np.full(len(add_xy), 0.8, dtype=np.float32),
                ], axis=0)
                n_teacher_added += len(add_xy)

        before_dedupe = len(labels)
        labels = _dedupe_labels(labels)
        if len(labels) != before_dedupe:
            # Fallback to uniform weights after rare teacher dedupe collisions.
            weights = np.ones(len(labels), dtype=np.float32)
        ds._label_cache[idx] = (labels.astype(np.float32), np.zeros(len(labels), dtype=np.int64), h, w)
        if hasattr(ds, "_weight_cache"):
            if len(weights) != len(labels):
                weights = np.ones(len(labels), dtype=np.float32)
            ds._weight_cache[idx] = weights.astype(np.float32)

    print(
        "  Teacher guidance: "
        f"teacher_tiles={n_teacher_tiles}, +{n_teacher_added} teacher labels, "
        f"-{n_removed_teacher} unsupported labels, -{n_removed_spike} spike labels, "
        f"ignore_pixels={n_ignore}"
    )


def _ddp_info(args) -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = world_size > 1
    if args.ddp and world_size == 1:
        raise ValueError(
            "--ddp was set but WORLD_SIZE=1. "
            "Launch with torchrun --nproc_per_node=<N> to enable DDP."
        )
    if use_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return use_ddp, rank, local_rank, world_size


def _ddp_needs_find_unused(ds: TileDetectionDataset) -> bool:
    if not getattr(ds, "use_all_bands", False):
        return False
    base = getattr(ds, "_base", None)
    if base is None or not hasattr(base, "tiles"):
        return True
    need_rubin = set(RUBIN_BANDS)
    need_euclid = set(EUCLID_BANDS)
    for tile in base.tiles:
        avail_r = set(tile.get("avail_rubin", []))
        avail_e = set(tile.get("avail_euclid", []))
        if not need_rubin.issubset(avail_r):
            return True
        if not need_euclid.issubset(avail_e):
            return True
    return False


def train(args):
    use_ddp, rank, local_rank, world_size = _ddp_info(args)
    if use_ddp and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if rank == 0:
        print(f"Training stem-based CenterNet detector on {device} (ddp={use_ddp})")

    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    use_wandb = args.wandb_project is not None
    if use_wandb and rank == 0:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    full_ds = TileDetectionDataset(
        rubin_dir=args.rubin_dir,
        euclid_dir=args.euclid_dir,
        nsig=args.nsig,
        max_sources=1000,
        use_all_bands=(args.euclid_dir is not None),
        augment=True,
        labels_mode=args.labels_mode,
        mer_fits=args.mer_fits,
        uncertain_ignore=args.uncertain_ignore,
        uncertain_nsig=args.uncertain_nsig,
        uncertain_radius_px=args.uncertain_radius_px,
        synthetic_sources_per_tile=args.synthetic_sources_per_tile,
        synthetic_prob=args.synthetic_prob,
        synthetic_min_snr=args.synthetic_min_snr,
        synthetic_max_snr=args.synthetic_max_snr,
        synthetic_min_sigma_px=args.synthetic_min_sigma_px,
        synthetic_max_sigma_px=args.synthetic_max_sigma_px,
        synthetic_weight=args.synthetic_weight,
    )
    _apply_extra_labels_inplace(full_ds, args.extra_labels)
    _apply_teacher_guidance_inplace(
        full_ds,
        teacher_labels_path=args.teacher_labels,
        euclid_dir=args.euclid_dir,
        teacher_match_radius=args.teacher_match_radius,
        teacher_filter_mode=args.teacher_filter_mode,
        teacher_add_labels=not args.no_teacher_add_labels,
        bright_ignore_radius=args.bright_ignore_radius,
        spike_radius=args.teacher_spike_radius,
        spike_width=args.teacher_spike_width,
    )

    if args.val_patches:
        # Patch-disjoint (spatially-disjoint) split; overlap duplicates make a
        # random split leak. TileDetectionDataset is one sample per tile
        # (augmented on the fly), so a tile-level split is index-level here.
        vp = tuple(f"_patch_{p.strip()}" for p in args.val_patches.split(",") if p.strip())
        val_idx = [i for i, t in enumerate(full_ds.tile_ids) if t.endswith(vp)]
        tr_idx = [i for i in range(len(full_ds)) if i not in set(val_idx)]
        tr_ds, val_ds = Subset(full_ds, tr_idx), Subset(full_ds, val_idx)
        n_tr, n_val = len(tr_idx), len(val_idx)
        if rank == 0:
            print(f"  Patch-disjoint split: held-out {args.val_patches} -> "
                  f"{n_val} val tiles, {n_tr} train tiles")
    else:
        n_val = max(1, int(0.1 * len(full_ds)))
        n_tr = len(full_ds) - n_val
        tr_ds, val_ds = random_split(
            full_ds, [n_tr, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )

    val_workers = max(1, args.num_workers // 2) if args.num_workers > 0 else 0
    if use_ddp:
        tr_sampler = DistributedSampler(tr_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        tr_loader = DataLoader(
            tr_ds,
            batch_size=args.batch_size,
            sampler=tr_sampler,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=val_workers,
        )
    else:
        tr_sampler = None
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
    if rank == 0:
        print(f"Train: {n_tr} samples   Val: {n_val} samples")

    foundation = load_foundation_from_checkpoint(args.encoder_ckpt, device=device)
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

    ddp_find_unused = _ddp_needs_find_unused(full_ds) if use_ddp else False
    if use_ddp:
        # Safer default: allow unused params to avoid rank crashes when a band
        # is missing on some samples. The warning is harmless.
        ddp_find_unused = True
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=ddp_find_unused,
        )
        if rank == 0:
            print(f"DDP: find_unused_parameters={ddp_find_unused}")

    model_to_save = model.module if use_ddp else model
    n_total = sum(p.numel() for p in model_to_save.parameters())
    n_train = sum(p.numel() for p in model_to_save.parameters() if p.requires_grad)
    if rank == 0:
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
        if tr_sampler is not None:
            tr_sampler.set_epoch(epoch)
        model.train()
        tr_loss_sum = 0.0
        tr_batches = 0
        for batch in tr_loader:
            images = {b: v.to(device) for b, v in batch["images"].items()}
            rms = {b: v.to(device) for b, v in batch["rms"].items()}

            out = model(images, rms)
            ignore_masks = batch.get("ignore_mask")
            if ignore_masks is not None:
                ignore_masks = ignore_masks.to(device)
            losses = criterion(
                out,
                [c.to(device) for c in batch["centroids"]],
                ignore_masks=ignore_masks,
                gt_weights=[w.to(device) for w in batch.get("source_weights", [])] or None,
            )

            optimizer.zero_grad()
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss_sum += float(losses["loss_total"])
            tr_batches += 1
            step += 1
            if use_wandb and rank == 0 and step % 10 == 0:
                wandb.log({
                    "train/loss": float(losses["loss_total"]),
                    "train/loss_hm": float(losses["loss_hm"]),
                    "train/loss_off": float(losses["loss_off"]),
                    "train/loss_flux": float(losses["loss_flux"]),
                    "train/n_sources": float(losses["n_sources"]),
                }, step=step)

        scheduler.step()
        if use_ddp:
            t = torch.tensor([tr_loss_sum, float(tr_batches)], device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            tr_loss_sum = float(t[0].item())
            tr_batches = int(t[1].item())
        mean_tr = tr_loss_sum / max(tr_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        prev_augment = full_ds._base.augment
        full_ds._base.augment = False
        with torch.no_grad():
            try:
                for batch in val_loader:
                    images = {b: v.to(device) for b, v in batch["images"].items()}
                    rms = {b: v.to(device) for b, v in batch["rms"].items()}
                    out = model(images, rms)
                    ignore_masks = batch.get("ignore_mask")
                    if ignore_masks is not None:
                        ignore_masks = ignore_masks.to(device)
                    losses = criterion(
                        out,
                        [c.to(device) for c in batch["centroids"]],
                        ignore_masks=ignore_masks,
                        gt_weights=[w.to(device) for w in batch.get("source_weights", [])] or None,
                    )
                    val_loss_sum += float(losses["loss_total"])
                    val_batches += 1
            finally:
                full_ds._base.augment = prev_augment
        if use_ddp:
            t = torch.tensor([val_loss_sum, float(val_batches)], device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            val_loss_sum = float(t[0].item())
            val_batches = int(t[1].item())
        mean_val = val_loss_sum / max(val_batches, 1)

        lr_now = scheduler.get_last_lr()[0]
        if rank == 0:
            print(f"Epoch {epoch + 1:3d}/{args.epochs}  tr={mean_tr:.4f}  val={mean_val:.4f}  lr={lr_now:.2e}")

        if use_wandb and rank == 0:
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
                    log["viz/tile"] = _log_tile(
                        viz_sample, sout, wandb, step,
                        euclid_dir=args.euclid_dir,
                        spike_veto_radius=args.viz_spike_veto_radius,
                        spike_veto_width=args.viz_spike_veto_width,
                    )
                except Exception as exc:
                    print(f"  [warn] viz failed: {exc}")
            wandb.log(log, step=step)

        if rank == 0 and mean_val < best_val:
            best_val = mean_val
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            model_to_save.save(args.out)
            print(f"  ✓ saved -> {args.out}")
            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val

    if use_wandb and rank == 0:
        wandb.finish()
    if rank == 0:
        print(f"Done. Best val loss: {best_val:.4f}")
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rubin_dir", required=True)
    p.add_argument("--euclid_dir", default=None)
    p.add_argument("--encoder_ckpt", required=True, help="Foundation v8/v9/v10 checkpoint")
    p.add_argument("--extra_labels", default=None, help="Refined labels from previous self-training round")
    p.add_argument("--init_checkpoint", default=None, help="Initialize from an existing stem detector checkpoint")
    p.add_argument("--out", default="../checkpoints/stem_centernet_v7.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--nsig", type=float, default=3.0)
    p.add_argument("--mer_fits", default=None, help="MER Q1 catalogue FITS (required for --labels_mode mer)")
    p.add_argument("--val_patches", default=None,
                   help="Comma-separated patch ids to hold out for a PATCH-DISJOINT split (e.g. '25')")
    p.add_argument("--labels_mode", default="vis_peak", choices=["vis_peak", "multiband", "vis_sep", "mer"],
                   help="Pseudo-label source: improved VIS classical labels, multi-band SEP labels, "
                        "or SEP-primary VIS labels with the classical cleaning wrapper (vis_sep)")
    p.add_argument("--uncertain_ignore", action="store_true",
                   help="Ignore negative heatmap loss around low-threshold uncertain source proposals")
    p.add_argument("--uncertain_nsig", type=float, default=1.8,
                   help="Low-threshold proposal significance for uncertainty ignore masks")
    p.add_argument("--uncertain_radius_px", type=float, default=5.0,
                   help="VIS-pixel radius ignored around uncertain proposals")
    p.add_argument("--synthetic_sources_per_tile", type=int, default=0,
                   help="Number of synthetic perfect-label sources injected per augmented training tile")
    p.add_argument("--synthetic_prob", type=float, default=1.0,
                   help="Probability of injecting synthetic sources into a training tile")
    p.add_argument("--synthetic_min_snr", type=float, default=5.0)
    p.add_argument("--synthetic_max_snr", type=float, default=20.0)
    p.add_argument("--synthetic_min_sigma_px", type=float, default=1.1)
    p.add_argument("--synthetic_max_sigma_px", type=float, default=3.5)
    p.add_argument("--synthetic_weight", type=float, default=1.5,
                   help="Positive loss weight for synthetic perfect-label sources")
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--stream_ch", type=int, default=16)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--unfreeze_stems", action="store_true",
                   help="Allow the reused V7 BandStem weights to keep training")
    p.add_argument("--teacher_labels", default=None,
                   help="Fused CenterNet label .pt file used as proposal teacher for StemCenterNet")
    p.add_argument("--teacher_match_radius", type=float, default=0.012,
                   help="Normalized radius for matching stem labels/detections to teacher proposals")
    p.add_argument("--teacher_filter_mode", default="bright", choices=["none", "bright", "global"],
                   help="Where to require teacher support: none, only bright-star zones, or globally")
    p.add_argument("--no_teacher_add_labels", action="store_true",
                   help="Do not merge teacher detections into the stem training labels")
    p.add_argument("--bright_ignore_radius", type=int, default=0,
                   help="VIS-pixel radius around bright cores to ignore negative loss except on thin spikes")
    p.add_argument("--teacher_spike_radius", type=int, default=40,
                   help="VIS-pixel radial search length for thin spike hard-negative cleaning")
    p.add_argument("--teacher_spike_width", type=float, default=3.0,
                   help="VIS-pixel half-width for thin spike hard-negative cleaning")
    p.add_argument("--viz_spike_veto_radius", type=int, default=0,
                   help="Apply thin VIS bright-star spike veto to W&B red-cross visualization; 0 disables")
    p.add_argument("--viz_spike_veto_width", type=float, default=0.0,
                   help="Thin spike veto half-width for W&B visualization")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run", default=None)
    p.add_argument("--device", default="", help='Device to use: "cuda", "cuda:1", "cpu" (default: auto)')
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ddp", action="store_true",
                   help="Enable DistributedDataParallel (requires torchrun).")
    train(p.parse_args())
