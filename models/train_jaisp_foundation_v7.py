"""Training script for JAISP Foundation v7 mixed-resolution pretraining."""

import argparse
import contextlib
import copy
from datetime import timedelta
import os
import random
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from jaisp_foundation_v7 import ALL_BANDS, JAISPFoundationV7, create_optimizer, create_scheduler
from jaisp_dataset_v6 import JAISPDatasetV6, collate_v6
from jaisp_dataset_v7 import sample_context_target, sample_context_target_phaseB_mixed

try:
    import wandb
except ImportError:
    wandb = None


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    c = np.corrcoef(a.ravel(), b.ravel())
    return float(c[0, 1])


def info_weighted_mask(info: np.ndarray, top_frac: float = 0.10) -> np.ndarray:
    thresh = np.nanpercentile(info, (1.0 - top_frac) * 100.0)
    return info >= thresh


def available_band_pool(sample: dict) -> dict:
    pool = {}
    pool.update(sample.get("rubin", {}))
    pool.update(sample.get("euclid", {}))
    return pool


def short_band_name(band: str) -> str:
    return band.split("_", 1)[1]


@contextlib.contextmanager
def suppress_stdout(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


class JAISPTrainerV7:
    def __init__(
        self,
        rubin_dir: str,
        euclid_dir: str,
        output_dir: str = "./checkpoints/jaisp_v7",
        stem_ch: int = 64,
        hidden_ch: int = 256,
        blocks_per_stage: int = 2,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        fused_pixel_scale_arcsec: float = 0.8,
        batch_size: int = 1,
        num_workers: int = 4,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        epochs: int = 80,
        warmup_epochs: int = 5,
        accum_steps: int = 4,
        grad_clip: float = 1.0,
        n_targets_per_step: int = 1,
        val_fraction: float = 0.05,
        vis_every_n_epochs: int = 2,
        cross_instrument_prob: float = 1.0,
        wandb_project: str = "JAISP-Foundation-v7",
        wandb_name: str = None,
        seed: int = 42,
        ddp_timeout_minutes: int = 120,
        checkpoint_every_n_epochs: int = 2,
        persistent_workers: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = self.world_size > 1
        self.ddp_timeout_minutes = int(ddp_timeout_minutes)

        requested_device = torch.device(device)
        if self.distributed and not dist.is_initialized():
            backend = "nccl" if requested_device.type == "cuda" else "gloo"
            init_method = os.environ.get("DIST_INIT_METHOD")
            if init_method:
                dist.init_process_group(
                    backend=backend,
                    init_method=init_method,
                    rank=self.rank,
                    world_size=self.world_size,
                    timeout=timedelta(minutes=self.ddp_timeout_minutes),
                )
            else:
                dist.init_process_group(
                    backend=backend,
                    timeout=timedelta(minutes=self.ddp_timeout_minutes),
                )

        if requested_device.type == "cuda" and torch.cuda.is_available():
            if self.distributed:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = requested_device
        else:
            self.device = requested_device

        self.is_main_process = self.rank == 0
        self.output_dir = Path(output_dir)
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()

        self.epochs = epochs
        self.accum_steps = accum_steps
        self.grad_clip = grad_clip
        self.n_targets = n_targets_per_step
        self.vis_every = vis_every_n_epochs
        self.cross_instrument_prob = float(cross_instrument_prob)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.checkpoint_every_n_epochs = int(checkpoint_every_n_epochs)
        self.persistent_workers = bool(persistent_workers)

        # AMP: use bfloat16 on CUDA (no GradScaler needed for bf16)
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.rng = np.random.RandomState(self.seed)

        if self.is_main_process:
            print("Loading training dataset...")
        with suppress_stdout(not self.is_main_process):
            self.train_dataset = JAISPDatasetV6(
                rubin_dir=rubin_dir,
                euclid_dir=euclid_dir,
                augment=True,
                load_euclid=True,
                seed=self.seed,
            )
        # Validation should be deterministic and augmentation-free.
        self.val_dataset = copy.deepcopy(self.train_dataset)
        self.val_dataset.augment = False
        self.val_dataset.rng = np.random.RandomState(self.seed)

        n_total = len(self.train_dataset)
        n_val = max(1, int(n_total * val_fraction))
        split_order = self.rng.permutation(n_total).tolist()
        self.val_indices = sorted(split_order[:n_val])
        self.train_indices = sorted(split_order[n_val:])
        self.train_loader, self.train_sampler = self._make_subset_loader(
            self.train_dataset,
            self.train_indices,
            shuffle=True,
            generator_seed=self.seed,
        )
        self.val_loader = self._make_subset_loader(self.val_dataset, self.val_indices, shuffle=False)
        if self.is_main_process:
            print(f"  Train tiles: {len(self.train_indices)}, Val tiles: {n_val}")

        if self.is_main_process:
            print("Initializing JAISPFoundationV7...")
        with suppress_stdout(not self.is_main_process):
            base_model = JAISPFoundationV7(
                band_names=ALL_BANDS,
                stem_ch=stem_ch,
                hidden_ch=hidden_ch,
                blocks_per_stage=blocks_per_stage,
                transformer_depth=transformer_depth,
                transformer_heads=transformer_heads,
                fused_pixel_scale_arcsec=fused_pixel_scale_arcsec,
            ).to(self.device)
        if self.distributed:
            # Each step touches only a subset of band-specific stems/decoders,
            # so some parameters legitimately receive no gradient on a rank.
            self.model = DDP(
                base_model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                find_unused_parameters=True,
            )
        else:
            self.model = base_model

        self.optimizer = create_optimizer(self.model, lr=lr, weight_decay=weight_decay)
        self.scheduler = create_scheduler(self.optimizer, warmup_epochs, epochs)

        wandb_mode = (os.environ.get("WANDB_MODE") or "").strip().lower()
        wandb_disabled = (os.environ.get("WANDB_DISABLED") or "").strip().lower() in {"1", "true", "yes"}
        self.use_wandb = (
            wandb is not None
            and self.is_main_process
            and not wandb_disabled
            and wandb_mode != "disabled"
        )
        if self.use_wandb:
            wandb_kwargs = dict(
                project=wandb_project,
                name=wandb_name or f"v7_h{hidden_ch}_depth{transformer_depth}",
                config={
                    "stem_ch": stem_ch,
                    "hidden_ch": hidden_ch,
                    "blocks_per_stage": blocks_per_stage,
                    "transformer_depth": transformer_depth,
                    "transformer_heads": transformer_heads,
                    "fused_pixel_scale_arcsec": fused_pixel_scale_arcsec,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "warmup_epochs": warmup_epochs,
                    "accum_steps": accum_steps,
                    "n_targets": n_targets_per_step,
                    "cross_instrument_prob": cross_instrument_prob,
                    "train_tiles": len(self.train_indices),
                    "val_tiles": n_val,
                    "seed": self.seed,
                },
            )
            if wandb_mode:
                wandb_kwargs["mode"] = wandb_mode
            wandb.init(**wandb_kwargs)
            # Keep step-based training curves separate from epoch-based
            # validation / visualization logs so W&B panels get cleaner axes.
            wandb.define_metric("global_step")
            wandb.define_metric("epoch")
            wandb.define_metric("train/loss", step_metric="global_step")
            wandb.define_metric("lr", step_metric="global_step")
            wandb.define_metric("train/epoch_loss", step_metric="epoch")
            wandb.define_metric("train/band_*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("vis/*", step_metric="epoch")

        self.config = {
            "band_names": ALL_BANDS,
            "stem_ch": stem_ch,
            "hidden_ch": hidden_ch,
            "blocks_per_stage": blocks_per_stage,
            "transformer_depth": transformer_depth,
            "transformer_heads": transformer_heads,
            "fused_pixel_scale_arcsec": fused_pixel_scale_arcsec,
            "seed": self.seed,
        }
        self.best_val_loss = float("inf")
        self.global_step = 0

    def _worker_init_fn(self, worker_id: int) -> None:
        base_seed = self.seed + 1000 * self.rank + worker_id
        random.seed(base_seed)
        np.random.seed(base_seed)
        torch.manual_seed(base_seed)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return
        dataset = worker_info.dataset
        base_dataset = getattr(dataset, "dataset", dataset)
        if hasattr(base_dataset, "rng"):
            base_dataset.rng = np.random.RandomState(base_seed)

    def _make_subset_loader(
        self,
        dataset,
        indices,
        shuffle: bool,
        generator_seed: int = None,
    ):
        subset = Subset(dataset, list(indices))
        generator = None
        sampler = None
        if self.distributed and shuffle:
            sampler = DistributedSampler(
                subset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=int(generator_seed if generator_seed is not None else self.seed),
                drop_last=False,
            )
        elif shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(generator_seed if generator_seed is not None else self.seed))
        use_persistent = self.num_workers > 0 and self.persistent_workers
        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None),
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=use_persistent,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=False,
            collate_fn=collate_v6,
            generator=generator,
            sampler=sampler,
            worker_init_fn=self._worker_init_fn,
        )
        if shuffle:
            return loader, sampler
        return loader

    def _raw_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _barrier(self) -> None:
        if not self.distributed or not dist.is_initialized():
            return
        if self.device.type == "cuda":
            dist.barrier(device_ids=[self.local_rank])
        else:
            dist.barrier()

    def _reduce_mean_scalar(self, value: float) -> float:
        if not self.distributed:
            return float(value)
        tensor = torch.tensor(float(value), device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return float(tensor.item())

    def _reduce_mean_stats(self, total: float, count: int) -> float:
        total_t = torch.tensor(float(total), device=self.device)
        count_t = torch.tensor(float(count), device=self.device)
        if self.distributed:
            dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
        denom = max(float(count_t.item()), 1.0)
        return float(total_t.item() / denom)

    def _reduce_band_losses(self, band_losses: dict) -> dict:
        reduced = {}
        for band in ALL_BANDS:
            values = band_losses.get(band, [])
            total = float(np.sum(values)) if values else 0.0
            count = len(values)
            total_t = torch.tensor(total, device=self.device)
            count_t = torch.tensor(float(count), device=self.device)
            if self.distributed:
                dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
            if count_t.item() > 0:
                reduced[band] = float(total_t.item() / count_t.item())
        return reduced

    def _broadcast_scalar(self, value: float) -> float:
        if not self.distributed:
            return float(value)
        tensor = torch.tensor(float(value), device=self.device)
        dist.broadcast(tensor, src=0)
        return float(tensor.item())

    def _prepare_batch(self, sample: dict, rng: np.random.RandomState, force_phase_b: bool = False) -> dict:
        use_phase_b = (
            sample.get("has_euclid", False)
            and (force_phase_b or self.cross_instrument_prob > 0.0 and rng.random() < self.cross_instrument_prob)
        )
        if use_phase_b:
            split = sample_context_target_phaseB_mixed(sample, rng, n_targets=self.n_targets)
        else:
            split = sample_context_target(sample, rng, n_targets=self.n_targets)
        if split is None:
            return None

        ctx_img = {b: v.unsqueeze(0).to(self.device) for b, v in split["context_images"].items()}
        ctx_rms = {b: v.unsqueeze(0).to(self.device) for b, v in split["context_rms"].items()}
        targets = [
            {
                "band": t["band"],
                "image": t["image"].unsqueeze(0).to(self.device),
                "rms": t["rms"].unsqueeze(0).to(self.device),
            }
            for t in split["targets"]
        ]
        return {"ctx_img": ctx_img, "ctx_rms": ctx_rms, "targets": targets}

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        epoch_loss_total = 0.0
        band_losses = defaultdict(list)
        band_norm_losses = defaultdict(list)   # v10: un-RMS-multiplied pixel loss for fair per-band comparison
        n_steps = 0

        self.optimizer.zero_grad(set_to_none=True)
        accum_count = 0
        microbatch_loss_sum = 0.0
        microbatch_count = 0

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.epochs}",
            leave=False,
            disable=not self.is_main_process,
        )
        for sample_list in pbar:
            for sample in sample_list:
                batch = self._prepare_batch(sample, self.rng)
                if batch is None:
                    continue

                step_loss_val = 0.0
                n_tgts = len(batch["targets"])
                for tgt in batch["targets"]:
                    with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                        out = self.model(
                            batch["ctx_img"],
                            batch["ctx_rms"],
                            tgt["band"],
                            tgt["image"],
                            tgt["rms"],
                        )
                    (out["loss"] / (n_tgts * self.accum_steps)).backward()
                    loss_scalar = float(out["loss"].detach())
                    step_loss_val += loss_scalar
                    band_losses[tgt["band"]].append(loss_scalar)
                    if "pixel_loss_norm" in out:
                        band_norm_losses[tgt["band"]].append(float(out["pixel_loss_norm"].detach()))

                step_loss_val /= n_tgts
                accum_count += 1
                microbatch_count += 1
                microbatch_loss_sum += step_loss_val

                if accum_count >= self.accum_steps:
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    self.global_step += 1
                    n_steps += 1

                    loss_val = microbatch_loss_sum / max(1, microbatch_count)
                    loss_val = self._reduce_mean_scalar(loss_val)
                    epoch_loss_total += loss_val
                    microbatch_loss_sum = 0.0
                    microbatch_count = 0

                    if self.use_wandb:
                        wandb.log({
                            "train/loss": loss_val,
                            "global_step": self.global_step,
                            "lr": self.optimizer.param_groups[0]["lr"],
                        })
                    pbar.set_postfix(loss=f"{loss_val:.4f}")

        if accum_count > 0:
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            n_steps += 1
            loss_val = microbatch_loss_sum / max(1, microbatch_count)
            loss_val = self._reduce_mean_scalar(loss_val)
            epoch_loss_total += loss_val
            if self.use_wandb:
                wandb.log({
                    "train/loss": loss_val,
                    "global_step": self.global_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

        avg_loss = self._reduce_mean_stats(epoch_loss_total, n_steps)
        per_band = self._reduce_band_losses(band_losses)
        per_band_norm = self._reduce_band_losses(band_norm_losses) if band_norm_losses else {}
        return {"loss": avg_loss, "per_band": per_band, "per_band_norm": per_band_norm}

    @torch.no_grad()
    def _validate(self) -> float:
        if not self.is_main_process:
            return float("nan")
        self.model.eval()
        eval_model = self._raw_model()
        eval_model.eval()
        val_losses = []
        val_rng = np.random.RandomState(0)
        n_seen = 0

        for sample_list in self.val_loader:
            for sample in sample_list:
                if n_seen >= 50:
                    break
                batch = self._prepare_batch(sample, val_rng, force_phase_b=sample.get("has_euclid", False))
                if batch is None:
                    continue

                for tgt in batch["targets"]:
                    with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                        out = eval_model(
                            batch["ctx_img"],
                            batch["ctx_rms"],
                            tgt["band"],
                            tgt["image"],
                            tgt["rms"],
                        )
                    val_losses.append(float(out["loss"]))
                n_seen += 1
            if n_seen >= 50:
                break

        return float(np.mean(val_losses)) if val_losses else float("inf")

    @torch.no_grad()
    def _visualize(self, epoch: int) -> None:
        if not self.use_wandb or not self.is_main_process:
            return

        self.model.eval()
        eval_model = self._raw_model()
        eval_model.eval()
        raw_sample = None
        pool = None
        for tile_idx in self.val_indices[:25]:
            candidate = self.val_dataset[tile_idx]
            candidate_pool = available_band_pool(candidate)
            if len(candidate_pool) < 2:
                continue
            if pool is None or len(candidate_pool) > len(pool):
                raw_sample = candidate
                pool = candidate_pool
            if pool is not None and len(pool) == len(ALL_BANDS):
                break

        if raw_sample is None or pool is None:
            return

        ordered_bands = [b for b in ALL_BANDS if b in pool]
        vis_rng = np.random.RandomState(epoch + 123)
        panel_data = []
        scatter_truth_all, scatter_pred_all = [], []
        pearsons, maes, std_ratios = [], [], []

        log_dict = {"epoch": epoch + 1}
        fused_hw_summary = None

        for target_band in ordered_bands:
            context_bands = [b for b in ordered_bands if b != target_band]
            if not context_bands:
                continue

            ctx_img = {b: pool[b]["image"].unsqueeze(0).to(self.device) for b in context_bands}
            ctx_rms = {b: pool[b]["rms"].unsqueeze(0).to(self.device) for b in context_bands}
            tgt_img = pool[target_band]["image"].unsqueeze(0).to(self.device)
            tgt_rms = pool[target_band]["rms"].unsqueeze(0).to(self.device)

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                out = eval_model(ctx_img, ctx_rms, target_band, tgt_img, tgt_rms)

            truth = out["target_norm"][0, 0].float().cpu().numpy()
            pred = out["pred"][0, 0].float().cpu().numpy()
            resid = pred - truth
            info = out["info_weights"][0, 0].float().cpu().numpy()
            fused_hw_summary = out["fused_hw"]

            bright = info_weighted_mask(info, top_frac=0.10)
            corr = float("nan")
            mae = float("nan")
            std_ratio = float("nan")
            if bright.sum() > 10:
                t_bright = truth[bright]
                p_bright = pred[bright]
                corr = pearson_r(t_bright, p_bright)
                mae = float(np.mean(np.abs(p_bright - t_bright)))
                std_ratio = float(np.std(p_bright) / max(np.std(t_bright), 1e-6))
                pearsons.append(corr)
                maes.append(mae)
                std_ratios.append(std_ratio)

                n = min(2000, len(t_bright))
                idx = vis_rng.choice(len(t_bright), n, replace=False)
                scatter_truth_all.append(t_bright[idx])
                scatter_pred_all.append(p_bright[idx])

            panel_data.append({
                "band": target_band,
                "truth": truth,
                "pred": pred,
                "resid": resid,
                "info": info,
                "corr": corr,
                "mae": mae,
                "std_ratio": std_ratio,
            })
            log_dict[f"vis/band_{target_band}_pearson_r_bright"] = corr
            log_dict[f"vis/band_{target_band}_mae_bright"] = mae
            log_dict[f"vis/band_{target_band}_std_ratio_bright"] = std_ratio

        if panel_data:
            n_cols = len(panel_data)
            fig, axes = plt.subplots(4, n_cols, figsize=(3.2 * n_cols, 11), squeeze=False)
            row_labels = [
                "Truth (noise units)",
                "Prediction",
                "Residual",
                "Info weights",
            ]
            for col, panel in enumerate(panel_data):
                truth = panel["truth"]
                pred = panel["pred"]
                resid = panel["resid"]
                info = panel["info"]

                lo = float(np.nanpercentile(truth, 1))
                hi = float(np.nanpercentile(truth, 99))
                lim = max(float(np.nanpercentile(np.abs(resid), 99)), 1e-3)
                info_hi = max(float(np.nanpercentile(info, 99.5)), float(info.max()), 1e-6)
                arrays = [truth, pred, resid, info]
                cmaps = ["gray", "gray", "RdBu_r", "inferno"]
                vranges = [(lo, hi), (lo, hi), (-lim, lim), (0.0, info_hi)]

                title = short_band_name(panel["band"])
                if np.isfinite(panel["corr"]):
                    title = (
                        f"{title}\n"
                        f"r={panel['corr']:.2f}  "
                        f"mae={panel['mae']:.2f}  "
                        f"std={panel['std_ratio']:.2f}"
                    )

                for row, (arr, cmap, vr) in enumerate(zip(arrays, cmaps, vranges)):
                    ax = axes[row, col]
                    ax.imshow(arr, origin="lower", cmap=cmap, vmin=vr[0], vmax=vr[1])
                    ax.axis("off")
                    if col == 0:
                        ax.set_ylabel(row_labels[row], fontsize=9)
                    if row == 0:
                        ax.set_title(title, fontsize=10, pad=4)

            tile_id = raw_sample.get("tile_id", "unknown")
            fused_txt = fused_hw_summary if fused_hw_summary is not None else "?"
            fig.suptitle(
                f"v7 reconstruction diagnostics | tile={tile_id} | bands={len(panel_data)} | fused={fused_txt}",
                fontsize=12,
                y=0.995,
            )
            plt.tight_layout(rect=(0, 0, 1, 0.97))
            log_dict["vis/all_band_grid"] = wandb.Image(fig)
            plt.close(fig)

        if scatter_truth_all:
            t_cat = np.concatenate(scatter_truth_all)
            p_cat = np.concatenate(scatter_pred_all)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.hexbin(t_cat, p_cat, gridsize=60, cmap="viridis", mincnt=1)
            lim = max(float(np.abs(t_cat).max()), float(np.abs(p_cat).max()))
            lim = min(lim, 30.0)
            ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
            pooled_r = pearson_r(t_cat, p_cat)
            ax.set_xlabel("Truth (noise units)")
            ax.set_ylabel("Prediction (noise units)")
            ax.set_title(f"Bright-pixel pred vs truth | pooled r = {pooled_r:.3f}")
            plt.tight_layout()
            log_dict["vis/scatter_pred_vs_truth"] = wandb.Image(fig)
            log_dict["vis/pearson_r_all"] = pooled_r
            plt.close(fig)

        if pearsons:
            log_dict["vis/pearson_r_mean"] = float(np.nanmean(pearsons))
        if maes:
            log_dict["vis/mae_bright_mean"] = float(np.nanmean(maes))
        if std_ratios:
            log_dict["vis/std_ratio_bright_mean"] = float(np.nanmean(std_ratios))

        wandb.log(log_dict)

    def _save_checkpoint(self, epoch: int, tag: str = "latest") -> None:
        if not self.is_main_process:
            return
        ckpt = {
            "epoch": epoch,
            "model": self._raw_model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "config": self.config,
            "train_indices": list(self.train_indices),
            "val_indices": list(self.val_indices),
            "rng_state": self.rng.get_state(),
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(ckpt, self.output_dir / f"checkpoint_{tag}.pt")

    def load_checkpoint(self, path: str, weights_only: bool = False) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._raw_model().load_state_dict(ckpt["model"])
        # Weights-only resume = warm-start a NEW training run from the loaded weights:
        # don't inherit the prior best_val_loss / step counter / epoch (which would
        # otherwise make `start_epoch >= self.epochs` and skip training entirely).
        if weights_only:
            self.best_val_loss = float("inf")
            self.global_step = 0
        else:
            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            self.global_step = ckpt.get("global_step", 0)
        if "train_indices" in ckpt and "val_indices" in ckpt:
            self.train_indices = list(ckpt["train_indices"])
            self.val_indices = list(ckpt["val_indices"])
            self.train_loader, self.train_sampler = self._make_subset_loader(
                self.train_dataset,
                self.train_indices,
                shuffle=True,
                generator_seed=self.seed,
            )
            self.val_loader = self._make_subset_loader(self.val_dataset, self.val_indices, shuffle=False)
        if not weights_only:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if "rng_state" in ckpt:
            if not weights_only:
                self.rng.set_state(ckpt["rng_state"])
        if "python_random_state" in ckpt:
            if not weights_only:
                random.setstate(ckpt["python_random_state"])
        if "torch_rng_state" in ckpt:
            if not weights_only:
                torch.set_rng_state(ckpt["torch_rng_state"])
        if torch.cuda.is_available() and ckpt.get("cuda_rng_state_all") is not None:
            if not weights_only:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
        # Weights-only resume returns -1 so start_epoch becomes 0 in train().
        # Don't pre-step the scheduler — we want a fresh cosine schedule for the
        # new run, not the tail end of the previous one.
        if weights_only:
            return -1
        return int(ckpt["epoch"])

    def train(self, resume_from: str = None, resume_weights_only: bool = False) -> None:
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from, weights_only=resume_weights_only) + 1
            if self.is_main_process:
                mode = "weights-only" if resume_weights_only else "full-state"
                print(f"Resumed from {resume_from} (epoch {start_epoch}, mode={mode})")

        for epoch in range(start_epoch, self.epochs):
            train_metrics = self._train_epoch(epoch)
            self.scheduler.step()
            self._barrier()

            is_best = False
            if self.is_main_process:
                val_loss = self._validate()
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, "best")

                if self.use_wandb:
                    log_dict = {
                        "epoch": epoch + 1,
                        "train/epoch_loss": train_metrics["loss"],
                        "val/loss": val_loss,
                        "val/best_loss": self.best_val_loss,
                    }
                    for band, loss in train_metrics["per_band"].items():
                        log_dict[f"train/band_{band}"] = loss
                    for band, loss in train_metrics.get("per_band_norm", {}).items():
                        log_dict[f"train/band_{band}_norm"] = loss
                    wandb.log(log_dict)

                print(
                    f"Epoch {epoch+1:3d}/{self.epochs} | "
                    f"train {train_metrics['loss']:.4f} | "
                    f"val {val_loss:.4f}{'*' if is_best else ''}"
                )

                if self.checkpoint_every_n_epochs > 0 and (epoch + 1) % self.checkpoint_every_n_epochs == 0:
                    self._save_checkpoint(epoch, "latest")
                if (epoch + 1) % self.vis_every == 0 or epoch == 0:
                    self._visualize(epoch)
            else:
                val_loss = 0.0

            self.best_val_loss = self._broadcast_scalar(self.best_val_loss if self.is_main_process else 0.0)
            self._barrier()

        self._save_checkpoint(self.epochs - 1, "final")
        if self.use_wandb:
            wandb.finish()
        if self.is_main_process:
            print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--rubin_dir", required=True)
    p.add_argument("--euclid_dir", required=True)
    p.add_argument("--output_dir", default="./checkpoints/jaisp_v7")
    p.add_argument("--resume", default=None)
    p.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="Resume model weights and split metadata, but reinitialize optimizer/scheduler/RNG state",
    )
    p.add_argument("--stem_ch", type=int, default=64)
    p.add_argument("--hidden_ch", type=int, default=256)
    p.add_argument("--blocks_per_stage", type=int, default=2)
    p.add_argument("--transformer_depth", type=int, default=4)
    p.add_argument("--transformer_heads", type=int, default=8)
    p.add_argument("--fused_pixel_scale_arcsec", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--n_targets_per_step", type=int, default=1)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--vis_every_n_epochs", type=int, default=2)
    p.add_argument("--cross_instrument_prob", type=float, default=1.0)
    p.add_argument("--wandb_project", default="JAISP-Foundation-v7")
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ddp_timeout_minutes", type=int, default=120)
    p.add_argument("--checkpoint_every_n_epochs", type=int, default=2)
    p.add_argument(
        "--persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs (faster, but sometimes less stable on shared filesystems)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    trainer_kwargs = vars(args).copy()
    trainer_kwargs.pop("resume", None)
    resume_weights_only = trainer_kwargs.pop("resume_weights_only", False)
    trainer = JAISPTrainerV7(**trainer_kwargs)
    trainer.train(resume_from=args.resume, resume_weights_only=resume_weights_only)
