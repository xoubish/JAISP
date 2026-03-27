# train_jaisp_foundation_v6.py
#
# Training script for JAISP Foundation v6 - Masked Band Prediction
#
# Training objective:
#   For each tile, randomly hold out 1 Rubin band (the target).
#   Feed the remaining N-1 bands through the encoder-decoder.
#   Minimize InformationMap-weighted L1 between predicted and true target.
#
# Curriculum:
#   Phase A (this script): within-instrument only (all Rubin bands)
#   Phase B (future):      cross-instrument (Rubin → Euclid VIS)
#
# Key design choices:
#   - Gradient accumulation: accumulate 4 steps before optimizer step
#     (effective batch = 4 tiles even with batch_size=1)
#   - No teacher/EMA, no stop-gradient: reconstruction loss is sufficient
#   - W&B logging: per-band losses, LR, reconstruction visualizations
#   - Checkpoint: best model by validation loss + every 10 epochs

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import wandb
from typing import Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from jaisp_foundation_v6 import JAISPFoundationV6, create_optimizer, create_scheduler, ALL_BANDS
from jaisp_dataset_v6 import make_loader_v6, sample_context_target


class JAISPTrainerV6:
    def __init__(
        self,
        rubin_dir: str,
        euclid_dir: str,
        output_dir: str = './checkpoints/jaisp_v6',
        # Model
        stem_ch: int = 64,
        encoder_dims: tuple = (128, 256, 512),
        blocks_per_stage: int = 2,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        # Training
        batch_size: int = 1,
        num_workers: int = 4,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        epochs: int = 60,
        warmup_epochs: int = 5,
        accum_steps: int = 4,
        grad_clip: float = 1.0,
        n_targets_per_step: int = 1,   # mask 1 band per tile (increase for harder task)
        val_fraction: float = 0.05,    # fraction of tiles held for validation
        vis_every_n_epochs: int = 2,   # visualization frequency
        # W&B
        wandb_project: str = 'JAISP-Foundation-v6',
        wandb_name: str = None,
        # Device
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.accum_steps = accum_steps
        self.grad_clip = grad_clip
        self.n_targets = n_targets_per_step
        self.vis_every = vis_every_n_epochs

        self.rng = np.random.RandomState(42)

        # ---- Dataset -------------------------------------------------------
        print('Loading dataset...')
        full_dataset, full_loader = make_loader_v6(
            rubin_dir=rubin_dir,
            euclid_dir=euclid_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=True,
            load_euclid=False,   # Phase A: Rubin only
        )
        self.full_dataset = full_dataset

        # Train/val split by tile index
        n_total = len(full_dataset)
        n_val = max(1, int(n_total * val_fraction))
        idx = list(range(n_total))
        random.shuffle(idx)
        self.val_indices = set(idx[:n_val])
        self.train_indices = idx[n_val:]
        print(f'  Train tiles: {len(self.train_indices)}, Val tiles: {n_val}')

        # ---- Model ---------------------------------------------------------
        print('Initializing JAISPFoundationV6...')
        self.model = JAISPFoundationV6(
            band_names=ALL_BANDS,
            stem_ch=stem_ch,
            encoder_dims=encoder_dims,
            blocks_per_stage=blocks_per_stage,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
        ).to(self.device)

        # ---- Optimizer & Scheduler -----------------------------------------
        self.optimizer = create_optimizer(self.model, lr=lr, weight_decay=weight_decay)
        self.scheduler = create_scheduler(self.optimizer, warmup_epochs, epochs)

        # ---- W&B -----------------------------------------------------------
        wandb.init(
            project=wandb_project,
            name=wandb_name or f'v6_bs{batch_size}_dim{encoder_dims[-1]}_depth{transformer_depth}',
            config={
                'stem_ch': stem_ch, 'encoder_dims': encoder_dims,
                'blocks_per_stage': blocks_per_stage,
                'transformer_depth': transformer_depth, 'transformer_heads': transformer_heads,
                'lr': lr, 'weight_decay': weight_decay,
                'epochs': epochs, 'warmup_epochs': warmup_epochs,
                'accum_steps': accum_steps, 'n_targets': n_targets_per_step,
                'train_tiles': len(self.train_indices), 'val_tiles': n_val,
            }
        )

        self.best_val_loss = float('inf')
        self.global_step = 0

    # -----------------------------------------------------------------------
    # Core: prepare one forward pass from a dataset sample
    # -----------------------------------------------------------------------

    def _prepare_batch(self, sample: dict) -> dict:
        """
        Given a raw dataset sample, sample context/target split and move to device.
        Returns None if the tile can't produce a valid split.
        """
        split = sample_context_target(sample, self.rng, n_targets=self.n_targets)
        if split is None:
            return None

        # Move to device and add batch dimension ([1,H,W] → [1,1,H,W])
        ctx_img = {b: v.unsqueeze(0).to(self.device) for b, v in split['context_images'].items()}
        ctx_rms = {b: v.unsqueeze(0).to(self.device) for b, v in split['context_rms'].items()}
        targets = [
            {
                'band':  t['band'],
                'image': t['image'].unsqueeze(0).to(self.device),
                'rms':   t['rms'].unsqueeze(0).to(self.device),
            }
            for t in split['targets']
        ]
        return {'ctx_img': ctx_img, 'ctx_rms': ctx_rms, 'targets': targets}

    # -----------------------------------------------------------------------
    # Train one epoch
    # -----------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()

        epoch_loss = 0.0
        band_losses = defaultdict(list)
        n_steps = 0

        self.optimizer.zero_grad(set_to_none=True)
        accum_count = 0

        # Shuffle training indices
        train_order = self.rng.permutation(self.train_indices).tolist()

        pbar = tqdm(train_order, desc=f'Epoch {epoch+1}/{self.epochs}', leave=False)
        for tile_idx in pbar:
            sample_list = [self.full_dataset[tile_idx]]  # list of 1 sample (our batch)
            sample = sample_list[0]

            batch = self._prepare_batch(sample)
            if batch is None:
                continue

            # Forward pass for each target band in this tile
            step_loss = torch.tensor(0.0, device=self.device)
            for tgt in batch['targets']:
                out = self.model(
                    batch['ctx_img'], batch['ctx_rms'],
                    tgt['band'], tgt['image'], tgt['rms'],
                )
                step_loss = step_loss + out['loss']
                band_losses[tgt['band']].append(float(out['loss'].detach()))

            step_loss = step_loss / len(batch['targets'])
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
                wandb.log({'train/loss': loss_val, 'step': self.global_step,
                           'lr': self.optimizer.param_groups[0]['lr']})
                pbar.set_postfix(loss=f'{loss_val:.4f}')

        # Flush remaining gradients
        if accum_count > 0:
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        avg_loss = epoch_loss / max(1, n_steps)
        per_band = {b: np.mean(v) for b, v in band_losses.items()}
        return {'loss': avg_loss, 'per_band': per_band}

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()

        val_losses = []
        val_rng = np.random.RandomState(0)   # deterministic validation splits

        for tile_idx in list(self.val_indices)[:50]:   # cap at 50 for speed
            sample = self.full_dataset[tile_idx]
            split = sample_context_target(sample, val_rng, n_targets=1)
            if split is None:
                continue

            ctx_img = {b: v.unsqueeze(0).to(self.device) for b, v in split['context_images'].items()}
            ctx_rms = {b: v.unsqueeze(0).to(self.device) for b, v in split['context_rms'].items()}
            tgt = split['targets'][0]

            out = self.model(
                ctx_img, ctx_rms,
                tgt['band'],
                tgt['image'].unsqueeze(0).to(self.device),
                tgt['rms'].unsqueeze(0).to(self.device),
            )
            val_losses.append(float(out['loss']))

        return float(np.mean(val_losses)) if val_losses else float('inf')

    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _visualize(self, epoch: int) -> None:
        """
        Log reconstruction panels to W&B:
          - Per-example: context | truth | prediction | residual | infomap
            logged as individual wandb.Image panels so they appear as separate
            panels in the W&B UI (grouped under vis/ex0/, vis/ex1/, vis/ex2/).
          - Pixel scatter: predicted vs true SNR at bright pixels (info-weighted),
            gives a direct read on whether the model is learning the right values.
          - Reconstruction stats: per-example Pearson r and MAE at source pixels.
        """
        self.model.eval()

        # Use fixed seeds so the same tiles are shown every epoch (easy to compare)
        val_list = sorted(self.val_indices)
        rng = np.random.RandomState(0)
        vis_indices = rng.choice(val_list, size=min(3, len(val_list)), replace=False)

        log_dict = {'epoch': epoch + 1}

        # Accumulate scatter data across examples
        scatter_truth_all, scatter_pred_all = [], []

        for ex_idx, tile_idx in enumerate(vis_indices):
            sample = self.full_dataset[int(tile_idx)]
            split = sample_context_target(sample, np.random.RandomState(tile_idx), n_targets=1)
            if split is None:
                continue

            ctx_img = {b: v.unsqueeze(0).to(self.device) for b, v in split['context_images'].items()}
            ctx_rms = {b: v.unsqueeze(0).to(self.device) for b, v in split['context_rms'].items()}
            tgt = split['targets'][0]

            out = self.model(
                ctx_img, ctx_rms,
                tgt['band'],
                tgt['image'].unsqueeze(0).to(self.device),
                tgt['rms'].unsqueeze(0).to(self.device),
            )

            ctx_band = list(ctx_img.keys())[0]
            ctx_arr  = (ctx_img[ctx_band] / (ctx_rms[ctx_band] + 1e-10)).squeeze().cpu().numpy()
            truth    = out['target_norm'].squeeze().cpu().numpy()
            pred     = out['pred'].squeeze().cpu().numpy()
            resid    = pred - truth
            info     = out['info_weights'].squeeze().cpu().numpy()

            # ---- Individual image panels ------------------------------------
            def _render(arr, cmap='gray', vrange=None) -> np.ndarray:
                """Render a 2D array to an HxWx3 uint8 image for wandb.Image."""
                if vrange is None:
                    vlo, vhi = np.nanpercentile(arr, [1, 99])
                else:
                    vlo, vhi = vrange
                norm = plt.Normalize(vmin=vlo, vmax=vhi)
                cm = plt.get_cmap(cmap)
                rgba = cm(norm(arr))          # H×W×4
                return (rgba[:, :, :3] * 255).astype(np.uint8)

            rmax = float(np.nanpercentile(np.abs(resid), 99)) or 1.0
            prefix = f'vis/ex{ex_idx}'
            log_dict[f'{prefix}/context ({ctx_band})']       = wandb.Image(_render(ctx_arr),  caption=f'context: {ctx_band}')
            log_dict[f'{prefix}/truth ({tgt["band"]})']      = wandb.Image(_render(truth),    caption=f'truth: {tgt["band"]}')
            log_dict[f'{prefix}/prediction']                  = wandb.Image(_render(pred),     caption='prediction')
            log_dict[f'{prefix}/residual (pred − truth)']    = wandb.Image(_render(resid, cmap='RdBu_r', vrange=(-rmax, rmax)), caption='residual')
            log_dict[f'{prefix}/infomap']                     = wandb.Image(_render(info, cmap='inferno', vrange=(info.min(), info.max())), caption='InfoMap weights')

            # ---- Reconstruction stats at bright pixels ----------------------
            # Use the info map as a mask: keep the top 10% highest-weight pixels
            thresh = np.nanpercentile(info, 90)
            bright = info >= thresh
            if bright.sum() > 10:
                t_bright = truth[bright]
                p_bright = pred[bright]
                corr = float(np.corrcoef(t_bright, p_bright)[0, 1]) if len(t_bright) > 1 else 0.0
                mae  = float(np.mean(np.abs(p_bright - t_bright)))
                log_dict[f'{prefix}/pearson_r_bright'] = corr
                log_dict[f'{prefix}/mae_bright_px']    = mae

                # Collect for pooled scatter
                # Sub-sample to keep scatter manageable (max 2000 pts per example)
                n = min(2000, bright.sum())
                idx = np.random.choice(bright.sum(), n, replace=False)
                scatter_truth_all.append(t_bright[idx])
                scatter_pred_all.append(p_bright[idx])

        # ---- Pixel scatter: pred vs truth (all examples pooled) ------------
        if scatter_truth_all:
            t_cat = np.concatenate(scatter_truth_all)
            p_cat = np.concatenate(scatter_pred_all)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.hexbin(t_cat, p_cat, gridsize=60, cmap='viridis', mincnt=1)
            lim = max(np.abs(t_cat).max(), np.abs(p_cat).max())
            lim = min(lim, 30)  # cap at 30σ for display
            ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1, label='y=x')
            r = float(np.corrcoef(t_cat, p_cat)[0, 1])
            ax.set_xlabel('Truth (noise units)')
            ax.set_ylabel('Prediction (noise units)')
            ax.set_title(f'Pred vs Truth at bright pixels  |  r = {r:.3f}')
            ax.legend(fontsize=8)
            plt.tight_layout()
            log_dict['vis/scatter_pred_vs_truth'] = wandb.Image(fig)
            plt.close(fig)
            log_dict['vis/pearson_r_all'] = r

        wandb.log(log_dict)

    # -----------------------------------------------------------------------
    # Checkpoint save / load
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, tag: str = 'latest') -> None:
        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
        }
        torch.save(ckpt, self.output_dir / f'checkpoint_{tag}.pt')

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.global_step = ckpt.get('global_step', 0)
        return int(ckpt['epoch'])

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def train(self, resume_from: str = None) -> None:
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f'Resumed from {resume_from} (epoch {start_epoch})')

        for epoch in range(start_epoch, self.epochs):
            # ---- Train ----
            train_metrics = self._train_epoch(epoch)

            # ---- Scheduler step ----
            self.scheduler.step()

            # ---- Validate ----
            val_loss = self._validate()
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, 'best')

            # ---- Log epoch metrics ----
            log_dict = {
                'epoch': epoch + 1,
                'train/epoch_loss': train_metrics['loss'],
                'val/loss': val_loss,
                'val/best_loss': self.best_val_loss,
            }
            for band, loss in train_metrics['per_band'].items():
                log_dict[f'train/band_{band}'] = loss
            wandb.log(log_dict)

            print(
                f'Epoch {epoch+1:3d}/{self.epochs} | '
                f'train {train_metrics["loss"]:.4f} | '
                f'val {val_loss:.4f}{"*" if is_best else ""}'
            )

            # ---- Periodic checkpoint + visualization ----
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, 'latest')
            if (epoch + 1) % self.vis_every == 0 or epoch == 0:
                self._visualize(epoch)

        # Final save
        self._save_checkpoint(self.epochs - 1, 'final')
        wandb.finish()
        print(f'Training complete. Best val loss: {self.best_val_loss:.4f}')


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='JAISP Foundation v6 Training')
    # Paths
    parser.add_argument('--rubin_dir',  default='../data/rubin_tiles_ecdfs')
    parser.add_argument('--euclid_dir', default='../data/euclid_tiles_ecdfs')
    parser.add_argument('--output_dir', default='./checkpoints/jaisp_v6')
    parser.add_argument('--resume',     default=None, help='Path to checkpoint to resume from')
    # Model
    parser.add_argument('--stem_ch',           type=int, default=64)
    parser.add_argument('--encoder_dims',      type=int, nargs='+', default=[128, 256, 512])
    parser.add_argument('--blocks_per_stage',  type=int, default=2)
    parser.add_argument('--transformer_depth', type=int, default=4)
    parser.add_argument('--transformer_heads', type=int, default=8)
    # Training
    parser.add_argument('--epochs',       type=int,   default=60)
    parser.add_argument('--warmup',       type=int,   default=5)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--accum_steps',  type=int,   default=4)
    parser.add_argument('--grad_clip',    type=float, default=1.0)
    parser.add_argument('--n_targets',    type=int,   default=1,
                        help='Number of bands to mask per tile (1=easy, 2=harder)')
    parser.add_argument('--num_workers',  type=int,   default=4)
    # W&B
    parser.add_argument('--wandb_project', default='JAISP-Foundation-v6')
    parser.add_argument('--wandb_name',    default=None)

    args = parser.parse_args()

    trainer = JAISPTrainerV6(
        rubin_dir=args.rubin_dir,
        euclid_dir=args.euclid_dir,
        output_dir=args.output_dir,
        stem_ch=args.stem_ch,
        encoder_dims=tuple(args.encoder_dims),
        blocks_per_stage=args.blocks_per_stage,
        transformer_depth=args.transformer_depth,
        transformer_heads=args.transformer_heads,
        epochs=args.epochs,
        warmup_epochs=args.warmup,
        lr=args.lr,
        weight_decay=args.weight_decay,
        accum_steps=args.accum_steps,
        grad_clip=args.grad_clip,
        n_targets_per_step=args.n_targets,
        num_workers=args.num_workers,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )

    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
