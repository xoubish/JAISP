# train_jaisp_foundation_v5.py
#
# Training script for JAISP Foundation v5 - STRICT Position Encoding
#
# Key changes from v4:
#  - Uses JAISPFoundationV5 with strict position alignment (shift_px=0)
#  - Forces exact token-to-token matching to learn precise spatial correspondence
#  - All other components remain the same
#
# Inherited features from v4:
#  - Teacher is ALWAYS kept in eval() (BN stats stable) during both train + visualization
#  - Logs pairing diagnostics (cross-instrument vs within-instrument)
#  - Visualization: weights are shown BOTH at image-res and token-grid-res
#  - UMAP is paired + information-weighted (same indices for both views) + includes paired-cos stats
#  - Removes sklearn FutureWarning cleanly

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*"
)

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from collections import defaultdict
import gc

from jaisp_foundation_v5 import JAISPFoundationV5, create_optimizer, create_scheduler
from jaisp_dataset_v4 import make_loader


class JAISPTrainerV5:
    def __init__(self,
                 rubin_dir: str,
                 euclid_dir: str,
                 output_dir: str = "./checkpoints",
                 embed_dim: int = 256,
                 proj_dim: int = 256,
                 depth: int = 6,
                 patch_size: int = 16,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 wandb_project: str = "JAISP-Foundation-v5",
                 wandb_name: str = None):

        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Dataset
        print("Loading data...")
        self.dataset, self.dataloader = make_loader(
            rubin_dir=rubin_dir,
            euclid_dir=euclid_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=True
        )

        band_names = self.dataset.get_band_names()
        print(f"Tiles: {len(self.dataset)}, Bands: {len(band_names)}")

        # Model v5 - STRICT position encoding
        print("Initializing JAISPFoundationV5 (strict position encoding)...")
        self.model = JAISPFoundationV5(
            band_names=band_names,
            stem_ch=64,
            embed_dim=embed_dim,
            proj_dim=proj_dim,
            depth=depth,
            patch_size=patch_size,
            shift_px=0,  # v5: strict position matching
            shift_temp=0.07,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

        self.wandb_project = wandb_project
        self.wandb_name = wandb_name

        # Band / pairing tracking (epoch-local summaries are logged)
        self.band_tracker = defaultdict(int)
        self.pair_tracker = defaultdict(int)

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def _to_float(x, default: float = 0.0) -> float:
        if x is None:
            return default
        if torch.is_tensor(x):
            return float(x.detach().item())
        try:
            return float(x)
        except Exception:
            return default

    def _set_teacher_eval(self):
        # keep teacher in eval so BN stats don't drift
        self.model.teacher_stems.eval()
        self.model.teacher_encoder.eval()
        self.model.teacher_projector.eval()

    def _move_batch_to_device(self, batch: dict):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)
            elif isinstance(batch[k], list):
                batch[k] = [x.to(self.device) if torch.is_tensor(x) else x for x in batch[k]]
        return batch

    # ----------------------------
    # Visualization
    # ----------------------------
    def visualize(self, batch, outputs, epoch, step):
        """
        Visualization that is consistent about spatial grids:

        - weights_img_*: weights at IMAGE resolution (HxW of each view)
        - weights_tok_*: weights downsampled to TOKEN grid (Ht x Wt)
        - act_*: token activation norm on token grid
        - sim_map: cosine sim map on COMMON token grid
        - UMAP: paired token sampling (same idx for v1/v2) weighted by average weights
        """

        with torch.no_grad():
            def prepare_2d(data):
                # Handle list (variable res) vs Tensor
                x = data[0] if isinstance(data, list) else data
                x = x.detach().cpu().float().numpy().squeeze()
                if x.ndim == 3:
                    x = x[0]
                return x

            img1 = prepare_2d(batch['view1_image'])
            img2 = prepare_2d(batch['view2_image'])

            # weights come out of model as [1,1,H,W]
            w1_img = prepare_2d(outputs['weights1'])
            w2_img = prepare_2d(outputs['weights2'])

            z1 = outputs['z1'][0].detach().cpu().float().numpy()  # [N1, D]
            z2 = outputs['z2'][0].detach().cpu().float().numpy()  # [N2, D]

            loss_dict = {
                'total': self._to_float(outputs.get('loss', 0.0)),
                'align': self._to_float(outputs.get('align_loss', 0.0)),
                'var':   self._to_float(outputs.get('var_loss', 0.0)),
                'cov':   self._to_float(outputs.get('cov_loss', 0.0)),
                'tok_sim': self._to_float(outputs.get('token_sim', 0.0)),
                'glob_sim': self._to_float(outputs.get('global_sim', 0.0)),
            }

        band1, band2 = outputs.get('band1', 'V1'), outputs.get('band2', 'V2')

        # grid sizes should be provided by model; fallback is safe-ish
        def _infer_gs(z):
            N = z.shape[0]
            s = int(np.sqrt(N))
            if s * s == N:
                return (s, s)
            h = max(1, s)
            w = max(1, N // h)
            return (h, w)

        gs1 = outputs.get('grid_size1') or outputs.get('grid_hw1')
        gs2 = outputs.get('grid_size2') or outputs.get('grid_hw2')
        gs1 = (int(gs1[0]), int(gs1[1])) if gs1 is not None else _infer_gs(z1)
        gs2 = (int(gs2[0]), int(gs2[1])) if gs2 is not None else _infer_gs(z2)

        # Normalize images for display
        def norm_img(x):
            p1, p99 = np.nanpercentile(x, [1, 99])
            return np.clip((x - p1) / (p99 - p1 + 1e-10), 0, 1)

        # Token norms on native token grids
        z1_norm = z1 / (np.linalg.norm(z1, axis=-1, keepdims=True) + 1e-10)
        z2_norm = z2 / (np.linalg.norm(z2, axis=-1, keepdims=True) + 1e-10)
        act1 = np.linalg.norm(z1, axis=-1).reshape(gs1)
        act2 = np.linalg.norm(z2, axis=-1).reshape(gs2)

        # Build common token grid
        target_gs = (max(gs1[0], gs2[0]), max(gs1[1], gs2[1]))
        Hc, Wc = target_gs

        z1_t = torch.from_numpy(z1_norm).reshape(1, gs1[0], gs1[1], -1).permute(0, 3, 1, 2)  # [1,D,H,W]
        z2_t = torch.from_numpy(z2_norm).reshape(1, gs2[0], gs2[1], -1).permute(0, 3, 1, 2)
        z1_interp = F.interpolate(z1_t, size=target_gs, mode='bilinear', align_corners=False)
        z2_interp = F.interpolate(z2_t, size=target_gs, mode='bilinear', align_corners=False)
        sim_map = (z1_interp * z2_interp).sum(dim=1).squeeze(0).cpu().numpy()  # [Hc,Wc]

        # Downsample weights to token grids (THIS is what your loss sees after resampling)
        w1_tok = F.interpolate(torch.from_numpy(w1_img).view(1, 1, *w1_img.shape),
                               size=gs1, mode='bilinear', align_corners=False).view(gs1).numpy()
        w2_tok = F.interpolate(torch.from_numpy(w2_img).view(1, 1, *w2_img.shape),
                               size=gs2, mode='bilinear', align_corners=False).view(gs2).numpy()

        # Also weights on common grid (for sim/UMAP)
        w1_c = F.interpolate(torch.from_numpy(w1_img).view(1, 1, *w1_img.shape),
                             size=target_gs, mode='bilinear', align_corners=False).view(-1).numpy()
        w2_c = F.interpolate(torch.from_numpy(w2_img).view(1, 1, *w2_img.shape),
                             size=target_gs, mode='bilinear', align_corners=False).view(-1).numpy()

        # --- Figure ---
        fig = plt.figure(figsize=(24, 16))

        # Row 1: images + IMAGE-res weights
        ax = plt.subplot(4, 4, 1)
        ax.imshow(norm_img(img1), origin='lower', cmap='gray')
        ax.set_title(f'V1 image: {band1}\n{img1.shape}', fontsize=10, weight='bold')
        ax.axis('off')

        ax = plt.subplot(4, 4, 2)
        ax.imshow(w1_img, origin='lower', cmap='hot')
        ax.set_title(f'V1 weights (image-res)\n{w1_img.shape}', fontsize=10)
        ax.axis('off')

        ax = plt.subplot(4, 4, 3)
        ax.imshow(norm_img(img2), origin='lower', cmap='gray')
        ax.set_title(f'V2 image: {band2}\n{img2.shape}', fontsize=10, weight='bold')
        ax.axis('off')

        ax = plt.subplot(4, 4, 4)
        ax.imshow(w2_img, origin='lower', cmap='hot')
        ax.set_title(f'V2 weights (image-res)\n{w2_img.shape}', fontsize=10)
        ax.axis('off')

        # Row 2: TOKEN-res weights + activation norms
        ax = plt.subplot(4, 4, 5)
        ax.imshow(w1_tok, origin='lower', cmap='hot')
        ax.set_title(f'V1 weights (token grid)\n{gs1}', fontsize=10)
        ax.axis('off')

        ax = plt.subplot(4, 4, 6)
        im = ax.imshow(act1, origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f'Activation norm V1 (token grid)\n{gs1}', fontsize=10)

        ax = plt.subplot(4, 4, 7)
        ax.imshow(w2_tok, origin='lower', cmap='hot')
        ax.set_title(f'V2 weights (token grid)\n{gs2}', fontsize=10)
        ax.axis('off')

        ax = plt.subplot(4, 4, 8)
        im = ax.imshow(act2, origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f'Activation norm V2 (token grid)\n{gs2}', fontsize=10)

        # Row 3: sim map + sim hist + variance + cov
        ax = plt.subplot(4, 4, 9)
        im = ax.imshow(sim_map, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f'Spatial sim on common token grid\n{target_gs}  avg={sim_map.mean():.3f}', fontsize=10)

        ax = plt.subplot(4, 4, 10)
        ax.hist(sim_map.flatten(), bins=50, alpha=0.8)
        ax.set_title('Sim distribution (common grid)', fontsize=10)

        ax = plt.subplot(4, 4, 11)
        ax.plot(np.sort(z1.var(axis=0))[::-1], label='V1')
        ax.plot(np.sort(z2.var(axis=0))[::-1], label='V2')
        ax.set_yscale('log')
        ax.set_title('Per-dim variance (token embeddings)', fontsize=10)
        ax.legend(fontsize=8)

        ax = plt.subplot(4, 4, 12)
        # covariance on z1 tokens (first 64 dims) just as a cheap “are dims decorrelating?” check
        z_cent = z1 - z1.mean(axis=0)
        cov = (z_cent.T @ z_cent) / (max(1, z_cent.shape[0] - 1))
        ax.imshow(cov[:64, :64], cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax.set_title('Covariance (z1 dims 0..63)', fontsize=10)

        # Row 4: correlations + UMAP + summary
        def plot_corr(w_img, act, gs, subplot_idx, title):
            ax_c = plt.subplot(4, 4, subplot_idx)
            wt = torch.from_numpy(w_img).float().view(1, 1, *w_img.shape)
            w_down = F.interpolate(wt, size=gs, mode='bilinear', align_corners=False).view(-1).numpy()
            ax_c.scatter(w_down, act.flatten(), alpha=0.25, s=6)
            corr = np.corrcoef(w_down, act.flatten())[0, 1]
            ax_c.set_title(f'{title}: corr(weights,img→token)={corr:.3f}', fontsize=10)
            return corr

        _ = plot_corr(w1_img, act1, gs1, 13, 'V1')
        _ = plot_corr(w2_img, act2, gs2, 14, 'V2')

        ax_umap = plt.subplot(4, 4, 15)
        try:
            from umap import UMAP

            # paired tokens on common grid
            Z1 = z1_interp.squeeze(0).permute(1, 2, 0).reshape(Hc * Wc, -1).cpu().numpy()
            Z2 = z2_interp.squeeze(0).permute(1, 2, 0).reshape(Hc * Wc, -1).cpu().numpy()

            # paired weight distribution on common grid
            w_avg = 0.5 * (w1_c + w2_c)
            w_avg = np.asarray(w_avg, dtype=np.float64)
            w_avg = np.clip(w_avg, 1e-12, None)
            p = w_avg / (w_avg.sum() + 1e-12)

            n_samples = min(600, Hc * Wc)
            rng = np.random.default_rng(0)
            idx = rng.choice(Hc * Wc, size=n_samples, replace=False, p=p)

            Z1s = Z1[idx]
            Z2s = Z2[idx]

            combined = np.vstack([Z1s, Z2s])
            emb = UMAP(n_neighbors=20, min_dist=0.05, n_epochs=300).fit_transform(combined)

            ax_umap.scatter(emb[:n_samples, 0], emb[:n_samples, 1],
                            c='red', s=10, alpha=0.45, label='V1 (paired)')
            ax_umap.scatter(emb[n_samples:, 0], emb[n_samples:, 1],
                            c='blue', s=10, alpha=0.45, label='V2 (paired)')

            paired_cos = (Z1s * Z2s).sum(axis=1)  # they are cosine-like already (normalized)
            ax_umap.set_title(f'UMAP (paired+weighted)\npaired cos={paired_cos.mean():.3f}±{paired_cos.std():.3f}', fontsize=10)
            ax_umap.legend(loc='best', fontsize=8)
        except Exception:
            ax_umap.text(0.5, 0.5, "UMAP Fail", ha='center')
            ax_umap.set_title("UMAP", fontsize=10)

        ax_txt = plt.subplot(4, 4, 16)
        ax_txt.axis('off')
        summary = (
            f"EPOCH: {epoch}\n"
            f"Loss: {loss_dict['total']:.4f}\n"
            f"Align: {loss_dict['align']:.4f}\n"
            f"Var: {loss_dict['var']:.4f}\n"
            f"Cov: {loss_dict['cov']:.4f}\n\n"
            f"TokSim: {loss_dict['tok_sim']:.3f}\n"
            f"GlobSim: {loss_dict['glob_sim']:.3f}\n\n"
            f"V1 gs={gs1}  V2 gs={gs2}\n"
            f"Common gs={target_gs}"
        )
        ax_txt.text(0, 1, summary, family='monospace', va='top', fontsize=10)

        plt.tight_layout()
        wandb.log({"vis/overview": wandb.Image(plt.gcf())}, step=step)
        plt.close('all')
        gc.collect()

    # ----------------------------
    # Training
    # ----------------------------
    def train(self,
              epochs: int = 200,
              lr: float = 3e-4,
              weight_decay: float = 0.05,
              warmup_epochs: int = 10,
              save_freq: int = 10,
              vis_freq: int = 5,
              ema_m: float = 0.996,
              accum_steps: int = 4):

        optimizer = create_optimizer(self.model, lr, weight_decay)
        scheduler = create_scheduler(optimizer, warmup_epochs, epochs)

        wandb.init(
            project=self.wandb_project,
            name=self.wandb_name,
            config={
                "arch": "v4-direct-alignment-native-res",
                "epochs": epochs,
                "lr": lr,
                "ema_m": ema_m,
                "shift_px": getattr(self.model.align_loss, "shift_px", None),
                "shift_temp": getattr(self.model.align_loss, "shift_temp", None),
                "embed_dim": self.model.embed_dim,
                "n_tiles": len(self.dataset),
                "accum_steps": accum_steps,
            }
        )

        best_loss = float('inf')
        global_step = 0

        for epoch in range(epochs):
            self.model.train()
            self._set_teacher_eval()

            # epoch-local trackers
            self.band_tracker = defaultdict(int)
            self.pair_tracker = defaultdict(int)

            stats = defaultdict(float)
            n_batches = 0

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(pbar):
                batch = self._move_batch_to_device(batch)

                outputs = self.model(batch)

                loss_v  = self._to_float(outputs.get('loss', 0.0))
                align_v = self._to_float(outputs.get('align_loss', 0.0))
                var_v   = self._to_float(outputs.get('var_loss', 0.0))
                cov_v   = self._to_float(outputs.get('cov_loss', 0.0))
                tok_sim = self._to_float(outputs.get('token_sim', 0.0))
                glob_sim = self._to_float(outputs.get('global_sim', 0.0))

                # Backprop with accumulation
                loss = outputs['loss'] / float(accum_steps)
                loss.backward()

                do_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(self.dataloader))
                if do_step:
                    _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    self.model.update_teacher(m=ema_m)  # EMA update

                # Stats
                stats['loss'] += loss_v
                stats['align'] += align_v
                stats['var'] += var_v
                stats['cov'] += cov_v
                stats['tok_sim'] += tok_sim
                stats['glob_sim'] += glob_sim
                n_batches += 1

                # Pairing diagnostics
                b1 = outputs.get('band1', None)
                b2 = outputs.get('band2', None)
                if b1 is not None:
                    self.band_tracker[b1] += 1
                if b2 is not None:
                    self.band_tracker[b2] += 1
                if (b1 is not None) and (b2 is not None):
                    self.pair_tracker[f"{b1}__{b2}"] += 1
                    # instrument category
                    is_r1 = ('rubin' in b1)
                    is_r2 = ('rubin' in b2)
                    if is_r1 and is_r2:
                        self.pair_tracker["pair/within_rubin"] += 1
                    elif (not is_r1) and (not is_r2):
                        self.pair_tracker["pair/within_euclid"] += 1
                    else:
                        self.pair_tracker["pair/cross_instrument"] += 1

                # Logging
                wandb.log({
                    'train/loss': loss_v,
                    'train/align_loss': align_v,
                    'train/var_loss': var_v,
                    'train/cov_loss': cov_v,
                    'train/token_sim': tok_sim,
                    'train/global_sim': glob_sim,
                    'train/lr': optimizer.param_groups[0]['lr'],
                }, step=global_step)

                global_step += 1

            # Epoch summary logs
            for k in list(stats.keys()):
                stats[k] /= max(n_batches, 1)

            # pairing fractions
            total_pairs = (
                self.pair_tracker.get("pair/within_rubin", 0)
                + self.pair_tracker.get("pair/within_euclid", 0)
                + self.pair_tracker.get("pair/cross_instrument", 0)
            )
            if total_pairs > 0:
                wandb.log({
                    "epoch/pair_frac_within_rubin": self.pair_tracker.get("pair/within_rubin", 0) / total_pairs,
                    "epoch/pair_frac_within_euclid": self.pair_tracker.get("pair/within_euclid", 0) / total_pairs,
                    "epoch/pair_frac_cross_instrument": self.pair_tracker.get("pair/cross_instrument", 0) / total_pairs,
                }, step=global_step)

            # scheduler
            scheduler.step()

            # Visualization
            if (vis_freq and (epoch % vis_freq == 0)) or (epoch == epochs - 1):
                print(f"\n>>> Running visualization for Epoch {epoch}...")
                self.model.eval()
                self._set_teacher_eval()
                vis_batch = next(iter(self.dataloader))
                vis_batch = self._move_batch_to_device(vis_batch)
                with torch.no_grad():
                    vis_out = self.model(vis_batch)
                self.visualize(vis_batch, vis_out, epoch, global_step)
                self.model.train()
                self._set_teacher_eval()

            # Checkpoint
            if (epoch + 1) % save_freq == 0 or stats['loss'] < best_loss:
                ckpt = {
                    'model': self.model.state_dict(),
                    'loss': stats['loss'],
                    'epoch': epoch,
                }
                out_path = (self.output_dir / "best.pt") if stats['loss'] < best_loss else (self.output_dir / f"ckpt_{epoch+1:03d}.pt")
                torch.save(ckpt, out_path)
                if stats['loss'] < best_loss:
                    best_loss = stats['loss']

            # quick console print
            print(f"[epoch {epoch:03d}] loss={stats['loss']:.4f} align={stats['align']:.4f} tokSim={stats['tok_sim']:.3f} globSim={stats['glob_sim']:.3f}")

        wandb.finish()


def main():
    trainer = JAISPTrainerV5(
        rubin_dir="../data/rubin_tiles_ecdfs",
        euclid_dir="../data/euclid_tiles_ecdfs",
        output_dir="./checkpoints/jaisp_v5",
        batch_size=1,
        num_workers=0,  # debug-friendly; bump later
        wandb_project="JAISP-Foundation-v5",
        wandb_name="v5_strict_position_encoding"
    )

    trainer.train(epochs=200, vis_freq=5, accum_steps=4, ema_m=0.996)


if __name__ == "__main__":
    main()