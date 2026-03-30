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

            step_loss = torch.tensor(0.0, device=self.device)
            for tgt in batch["targets"]:
                out = self.model(
                    batch["ctx_img"],
                    batch["ctx_rms"],
                    tgt["band"],
                    tgt["image"],
                    tgt["rms"],
                )
                step_loss = step_loss + out["loss"]
                band_losses[tgt["band"]].append(float(out["loss"].detach()))

            step_loss = step_loss / len(batch["targets"])
            (step_loss / self.accum_steps).backward()
            accum_count += 1

            if accum_count >= self.accum_steps:
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                n_steps += 1
                self.global_step += 1

                loss_val = float(step_loss.detach())
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
        sample = None
        rng = np.random.RandomState(epoch + 123)
        for tile_idx in list(self.val_indices)[:10]:
            candidate = self.full_dataset[tile_idx]
            sample = self._prepare_batch(candidate, rng, force_phase_b=candidate.get("has_euclid", False))
            if sample is not None:
                break
        if sample is None:
            return

        tgt = sample["targets"][0]
        out = self.model(
            sample["ctx_img"],
            sample["ctx_rms"],
            tgt["band"],
            tgt["image"],
            tgt["rms"],
        )

        truth = out["target_norm"][0, 0].cpu().numpy()
        pred = out["pred"][0, 0].cpu().numpy()
        resid = pred - truth

        lo = float(np.percentile(truth, 1))
        hi = float(np.percentile(truth, 99))
        lim = max(np.percentile(np.abs(resid), 99), 1e-3)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(truth, origin="lower", cmap="gray", vmin=lo, vmax=hi)
        axes[0].set_title(f"Truth {tgt['band']}")
        axes[1].imshow(pred, origin="lower", cmap="gray", vmin=lo, vmax=hi)
        axes[1].set_title(f"Pred {tgt['band']}")
        axes[2].imshow(resid, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim)
        axes[2].set_title(f"Residual  fused={out['fused_hw']}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        wandb.log({"vis/reconstruction": wandb.Image(fig), "epoch": epoch + 1})
        plt.close(fig)

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
