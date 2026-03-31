"""Training script for JAISP Foundation v7 mixed-resolution pretraining."""

import argparse
import random
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from jaisp_foundation_v7 import ALL_BANDS, JAISPFoundationV7, create_optimizer, create_scheduler
from jaisp_dataset_v7 import make_loader_v7, sample_context_target, sample_context_target_phaseB_mixed

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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.accum_steps = accum_steps
        self.grad_clip = grad_clip
        self.n_targets = n_targets_per_step
        self.vis_every = vis_every_n_epochs
        self.cross_instrument_prob = float(cross_instrument_prob)
        self.rng = np.random.RandomState(42)

        print("Loading dataset...")
        full_dataset, _ = make_loader_v7(
            rubin_dir=rubin_dir,
            euclid_dir=euclid_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=True,
            load_euclid=True,
        )
        self.full_dataset = full_dataset

        n_total = len(full_dataset)
        n_val = max(1, int(n_total * val_fraction))
        idx = list(range(n_total))
        random.shuffle(idx)
        self.val_indices = set(idx[:n_val])
        self.train_indices = idx[n_val:]
        print(f"  Train tiles: {len(self.train_indices)}, Val tiles: {n_val}")

        print("Initializing JAISPFoundationV7...")
        self.model = JAISPFoundationV7(
            band_names=ALL_BANDS,
            stem_ch=stem_ch,
            hidden_ch=hidden_ch,
            blocks_per_stage=blocks_per_stage,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
            fused_pixel_scale_arcsec=fused_pixel_scale_arcsec,
        ).to(self.device)

        self.optimizer = create_optimizer(self.model, lr=lr, weight_decay=weight_decay)
        self.scheduler = create_scheduler(self.optimizer, warmup_epochs, epochs)

        self.use_wandb = wandb is not None
        if self.use_wandb:
            wandb.init(
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
                },
            )

        self.config = {
            "band_names": ALL_BANDS,
            "stem_ch": stem_ch,
            "hidden_ch": hidden_ch,
            "blocks_per_stage": blocks_per_stage,
            "transformer_depth": transformer_depth,
            "transformer_heads": transformer_heads,
            "fused_pixel_scale_arcsec": fused_pixel_scale_arcsec,
        }
        self.best_val_loss = float("inf")
        self.global_step = 0

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
        epoch_loss = 0.0
        band_losses = defaultdict(list)
        n_steps = 0

        self.optimizer.zero_grad(set_to_none=True)
        accum_count = 0
        train_order = self.rng.permutation(self.train_indices).tolist()

        pbar = tqdm(train_order, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
        for tile_idx in pbar:
            sample = self.full_dataset[tile_idx]
            batch = self._prepare_batch(sample, self.rng)
            if batch is None:
                continue

            step_loss_val = 0.0
            n_tgts = len(batch["targets"])
            for tgt in batch["targets"]:
                out = self.model(
                    batch["ctx_img"],
                    batch["ctx_rms"],
                    tgt["band"],
                    tgt["image"],
                    tgt["rms"],
                )
                (out["loss"] / (n_tgts * self.accum_steps)).backward()
                step_loss_val += float(out["loss"].detach())
                band_losses[tgt["band"]].append(float(out["loss"].detach()))

            step_loss_val /= n_tgts
            accum_count += 1

            if accum_count >= self.accum_steps:
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                n_steps += 1
                self.global_step += 1

                loss_val = step_loss_val
                epoch_loss += loss_val
                if self.use_wandb:
                    wandb.log({
                        "train/loss": loss_val,
                        "step": self.global_step,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    })
                pbar.set_postfix(loss=f"{loss_val:.4f}")

        if accum_count > 0:
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        avg_loss = epoch_loss / max(1, n_steps)
        per_band = {b: float(np.mean(v)) for b, v in band_losses.items()}
        return {"loss": avg_loss, "per_band": per_band}

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        val_losses = []
        val_rng = np.random.RandomState(0)

        for tile_idx in list(self.val_indices)[:50]:
            sample = self.full_dataset[tile_idx]
            batch = self._prepare_batch(sample, val_rng, force_phase_b=sample.get("has_euclid", False))
            if batch is None:
                continue

            for tgt in batch["targets"]:
                out = self.model(
                    batch["ctx_img"],
                    batch["ctx_rms"],
                    tgt["band"],
                    tgt["image"],
                    tgt["rms"],
                )
                val_losses.append(float(out["loss"]))

        return float(np.mean(val_losses)) if val_losses else float("inf")

    @torch.no_grad()
    def _visualize(self, epoch: int) -> None:
        if not self.use_wandb:
            return

        self.model.eval()
        raw_sample = None
        pool = None
        for tile_idx in list(self.val_indices)[:25]:
            candidate = self.full_dataset[tile_idx]
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

            out = self.model(ctx_img, ctx_rms, target_band, tgt_img, tgt_rms)

            truth = out["target_norm"][0, 0].cpu().numpy()
            pred = out["pred"][0, 0].cpu().numpy()
            resid = pred - truth
            info = out["info_weights"][0, 0].cpu().numpy()
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
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "config": self.config,
        }
        torch.save(ckpt, self.output_dir / f"checkpoint_{tag}.pt")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.global_step = ckpt.get("global_step", 0)
        return int(ckpt["epoch"])

    def train(self, resume_from: str = None) -> None:
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resumed from {resume_from} (epoch {start_epoch})")

        for epoch in range(start_epoch, self.epochs):
            train_metrics = self._train_epoch(epoch)
            self.scheduler.step()

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
                wandb.log(log_dict)

            print(
                f"Epoch {epoch+1:3d}/{self.epochs} | "
                f"train {train_metrics['loss']:.4f} | "
                f"val {val_loss:.4f}{'*' if is_best else ''}"
            )

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, "latest")
            if (epoch + 1) % self.vis_every == 0 or epoch == 0:
                self._visualize(epoch)

        self._save_checkpoint(self.epochs - 1, "final")
        if self.use_wandb:
            wandb.finish()
        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--rubin_dir", required=True)
    p.add_argument("--euclid_dir", required=True)
    p.add_argument("--output_dir", default="./checkpoints/jaisp_v7")
    p.add_argument("--resume", default=None)
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
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    trainer_kwargs = vars(args).copy()
    trainer_kwargs.pop("resume", None)
    trainer = JAISPTrainerV7(**trainer_kwargs)
    trainer.train(resume_from=args.resume)
