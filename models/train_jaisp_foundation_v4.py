# train_jaisp_foundation_v4.py
#
# Training script for JAISP Foundation v4 - Direct Alignment

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from collections import defaultdict
import gc

from jaisp_foundation_v4 import JAISPFoundationV4, create_optimizer, create_scheduler
from jaisp_dataset_v4 import make_loader, BAND_WAVELENGTHS, ALL_BANDS
import warnings


class JAISPTrainerV4:
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
                 wandb_project: str = "JAISP-Foundation-v4",
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
        
        # Model
        self.model = JAISPFoundationV4(
            band_names=band_names,
            stem_ch=64,
            embed_dim=embed_dim,
            proj_dim=proj_dim,
            depth=depth,
            patch_size=patch_size,
            shift_px=2,
            shift_temp=0.07,
        ).to(self.device)

        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        
        self.band_tracker = defaultdict(int)
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name

    def visualize(self, batch, outputs, epoch, step):
        """
        Memory-efficient visualization with robust shape handling.
        UMAP is now: paired + information-weighted (high-weight tokens).
        """
    
        with torch.no_grad():
            def prepare_2d(data):
                # Handle list (variable res) vs Tensor
                x = data[0] if isinstance(data, list) else data
                # CPU + Float + Squeeze unit dims
                x = x.detach().cpu().float().numpy().squeeze()
                # If still 3D (e.g. C, H, W), take first channel
                if x.ndim == 3:
                    x = x[0]
                return x
    
            img1 = prepare_2d(batch['view1_image'])
            img2 = prepare_2d(batch['view2_image'])
    
            w1 = prepare_2d(outputs['weights1'])
            w2 = prepare_2d(outputs['weights2'])
    
            z1 = outputs['z1'][0].detach().cpu().float().numpy()  # [N1, D]
            z2 = outputs['z2'][0].detach().cpu().float().numpy()  # [N2, D]
    
            loss_dict = {
                'total': float(outputs['loss'].item()) if torch.is_tensor(outputs.get('loss')) else float(outputs.get('loss', 0.0)),
                'align': float(outputs['align_loss'].item()) if torch.is_tensor(outputs.get('align_loss')) else float(outputs.get('align_loss', 0.0)),
                'var':   float(outputs['var_loss'].item()) if torch.is_tensor(outputs.get('var_loss')) else float(outputs.get('var_loss', 0.0)),
                'cov':   float(outputs['cov_loss'].item()) if torch.is_tensor(outputs.get('cov_loss')) else float(outputs.get('cov_loss', 0.0)),
                'tok_sim': float(outputs.get('token_sim', 0.0)),
                'glob_sim': float(outputs.get('global_sim', 0.0)),
            }
    
        band1, band2 = outputs.get('band1', 'V1'), outputs.get('band2', 'V2')
    
        # --- Robust grid-size inference ---
        def _infer_gs_from_weights_or_tokens(w, z):
            if isinstance(w, np.ndarray) and w.ndim == 2:
                return (w.shape[0], w.shape[1])
            if isinstance(z, np.ndarray) and z.ndim == 2:
                N = z.shape[0]
                s = int(np.sqrt(N))
                if s * s == N:
                    return (s, s)
                # fallback: best near-square rectangle
                h = max(1, s)
                w_ = max(1, N // h)
                return (h, w_)
            return (1, 1)
    
        gs1 = outputs.get('grid_size1') or outputs.get('grid_hw1')
        gs2 = outputs.get('grid_size2') or outputs.get('grid_hw2')
    
        if gs1 is None:
            gs1 = _infer_gs_from_weights_or_tokens(w1, z1)
        if gs2 is None:
            gs2 = _infer_gs_from_weights_or_tokens(w2, z2)
    
        gs1 = (int(gs1[0]), int(gs1[1]))
        gs2 = (int(gs2[0]), int(gs2[1]))
    
        fig = plt.figure(figsize=(24, 16))
    
        def norm(x):
            p1, p99 = np.nanpercentile(x, [1, 99])
            return np.clip((x - p1) / (p99 - p1 + 1e-10), 0, 1)
    
        # Row 1: Images and Weights
        ax = plt.subplot(4, 4, 1)
        ax.imshow(norm(img1), origin='lower', cmap='gray')
        ax.set_title(f'V1: {band1}\n{img1.shape}', fontsize=10, weight='bold')
        ax.axis('off')
    
        ax = plt.subplot(4, 4, 2)
        ax.imshow(w1, origin='lower', cmap='hot')
        ax.set_title(f'Weights V1\nGrid: {gs1}', fontsize=10)
        ax.axis('off')
    
        ax = plt.subplot(4, 4, 3)
        ax.imshow(norm(img2), origin='lower', cmap='gray')
        ax.set_title(f'V2: {band2}\n{img2.shape}', fontsize=10, weight='bold')
        ax.axis('off')
    
        ax = plt.subplot(4, 4, 4)
        ax.imshow(w2, origin='lower', cmap='hot')
        ax.set_title(f'Weights V2\nGrid: {gs2}', fontsize=10)
        ax.axis('off')
    
        # Row 2: Latents
        z1_norm = z1 / (np.linalg.norm(z1, axis=-1, keepdims=True) + 1e-10)
        z2_norm = z2 / (np.linalg.norm(z2, axis=-1, keepdims=True) + 1e-10)
    
        ax = plt.subplot(4, 4, 5)
        act1 = np.linalg.norm(z1, axis=-1).reshape(gs1)
        im = ax.imshow(act1, origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title('Activation Norm (V1)')
    
        ax = plt.subplot(4, 4, 6)
        act2 = np.linalg.norm(z2, axis=-1).reshape(gs2)
        im = ax.imshow(act2, origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title('Activation Norm (V2)')
    
        # Common grid + similarity map (this is already paired)
        ax = plt.subplot(4, 4, 7)
        target_gs = (max(gs1[0], gs2[0]), max(gs1[1], gs2[1]))
    
        z1_t = torch.from_numpy(z1_norm).reshape(1, gs1[0], gs1[1], -1).permute(0, 3, 1, 2)  # [1,D,H,W]
        z2_t = torch.from_numpy(z2_norm).reshape(1, gs2[0], gs2[1], -1).permute(0, 3, 1, 2)  # [1,D,H,W]
        z1_interp = F.interpolate(z1_t, size=target_gs, mode='bilinear', align_corners=False)
        z2_interp = F.interpolate(z2_t, size=target_gs, mode='bilinear', align_corners=False)
        sim_map = (z1_interp * z2_interp).sum(dim=1).squeeze(0).numpy()  # [H,W]
    
        im = ax.imshow(sim_map, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f'Spatial Sim Map\nAvg: {sim_map.mean():.3f}')
    
        ax = plt.subplot(4, 4, 8)
        ax.hist(sim_map.flatten(), bins=50, alpha=0.7, color='purple')
        ax.set_title('Similarity Distribution')
    
        # Row 3: Stats
        ax = plt.subplot(4, 4, 9)
        ax.plot(np.sort(z1.var(axis=0))[::-1], label='V1')
        ax.plot(np.sort(z2.var(axis=0))[::-1], label='V2')
        ax.set_yscale('log')
        ax.set_title('Per-Dim Variance')
        ax.legend()
    
        def plot_corr(w, act, gs, subplot_idx, title):
            ax_c = plt.subplot(4, 4, subplot_idx)
            wt = torch.from_numpy(w).float().view(1, 1, *w.shape)
            w_down = F.interpolate(wt, size=gs, mode='bilinear', align_corners=False).view(-1).numpy()
            ax_c.scatter(w_down, act.flatten(), alpha=0.3, s=5)
            corr = np.corrcoef(w_down, act.flatten())[0, 1]
            ax_c.set_title(f'{title} Corr: {corr:.3f}')
            return corr
    
        _ = plot_corr(w1, act1, gs1, 10, 'V1')
        _ = plot_corr(w2, act2, gs2, 11, 'V2')
    
        ax = plt.subplot(4, 4, 12)
        z_cent = z1 - z1.mean(axis=0)
        cov = (z_cent.T @ z_cent) / (z_cent.shape[0] - 1)
        ax.imshow(cov[:64, :64], cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax.set_title('Covariance (First 64 dims)')
    
        # -------------------------------------------------------------------------
        # Row 4: FIXED UMAP (paired + weighted sampling)
        # -------------------------------------------------------------------------
        ax_umap = plt.subplot(4, 4, 14)
        try:
            from umap import UMAP
    
            # 1) Make paired token matrices on the SAME grid: [N, D]
            H, W = target_gs
            Z1 = z1_interp.squeeze(0).permute(1, 2, 0).reshape(H * W, -1).numpy()  # [N,D]
            Z2 = z2_interp.squeeze(0).permute(1, 2, 0).reshape(H * W, -1).numpy()  # [N,D]
    
            # 2) Build a paired weight distribution on the same grid (use BOTH weights)
            w1_t = torch.from_numpy(w1).float().view(1, 1, *w1.shape)
            w2_t = torch.from_numpy(w2).float().view(1, 1, *w2.shape)
            w1c = F.interpolate(w1_t, size=target_gs, mode='bilinear', align_corners=False).view(-1).numpy()
            w2c = F.interpolate(w2_t, size=target_gs, mode='bilinear', align_corners=False).view(-1).numpy()
            w_avg = 0.5 * (w1c + w2c)
    
            # Normalize to a probability distribution (with floor so background isn't impossible)
            w_avg = np.asarray(w_avg, dtype=np.float64)
            w_avg = np.clip(w_avg, 1e-12, None)
            p = w_avg / (w_avg.sum() + 1e-12)
    
            # 3) Sample indices from high-information tokens (PAIRED sampling)
            n_samples = 600  # a bit higher is OK; still cheap
            n_samples = min(n_samples, H * W)
    
            rng = np.random.default_rng(0)  # deterministic; change/remove if you want
            idx = rng.choice(H * W, size=n_samples, replace=False, p=p)
    
            Z1s = Z1[idx]
            Z2s = Z2[idx]
    
            # 4) UMAP on combined, but we KEEP pairing (same idx for both)
            combined = np.vstack([Z1s, Z2s])
            emb = UMAP(n_neighbors=20, min_dist=0.05, n_epochs=300).fit_transform(combined)
    
            ax_umap.scatter(emb[:n_samples, 0], emb[:n_samples, 1],
                            c='red', s=10, alpha=0.45, label='V1 (paired tokens)')
            ax_umap.scatter(emb[n_samples:, 0], emb[n_samples:, 1],
                            c='blue', s=10, alpha=0.45, label='V2 (paired tokens)')
            ax_umap.legend(loc='best', fontsize=8)
    
            # 5) Add paired cosine stats (this is the “truth check”)
            paired_cos = (Z1s * Z2s).sum(axis=1)  # since already normalized
            ax_umap.set_title(f'UMAP (paired + weighted)\npaired cos: {paired_cos.mean():.3f} ± {paired_cos.std():.3f}')
    
        except Exception:
            ax_umap.text(0.5, 0.5, "UMAP Fail", ha='center')
            ax_umap.set_title("UMAP")
    
        # Summary
        ax_txt = plt.subplot(4, 4, 15)
        ax_txt.axis('off')
        summary = (
            f"EPOCH: {epoch}\n"
            f"Loss: {loss_dict['total']:.4f}\n"
            f"Align: {loss_dict['align']:.4f}\n"
            f"Var: {loss_dict['var']:.4f}\n"
            f"Cov: {loss_dict['cov']:.4f}\n\n"
            f"Tok Sim: {loss_dict['tok_sim']:.3f}\n"
            f"Glob Sim: {loss_dict['glob_sim']:.3f}"
        )
        ax_txt.text(0, 1, summary, family='monospace', verticalalignment='top')
    
        plt.tight_layout()
        wandb.log({"vis/overview": wandb.Image(plt.gcf())}, step=step)
        plt.close('all')
        gc.collect()

    def train(self,
              epochs: int = 100,
              lr: float = 3e-4,
              weight_decay: float = 0.05,
              warmup_epochs: int = 10,
              save_freq: int = 10,
              vis_freq: int = 5):

        optimizer = create_optimizer(self.model, lr, weight_decay)
        scheduler = create_scheduler(optimizer, warmup_epochs, epochs)
    
        wandb.init(
            project=self.wandb_project,
            name=self.wandb_name,
            config={
                "arch": "v4-direct-alignment-native-res",
                "epochs": epochs,
                "lr": lr,
                "ema_m": 0.996,
                "shift_px": getattr(self.model.align_loss, "shift_px", None),
                "shift_temp": getattr(self.model.align_loss, "shift_temp", None),
                "embed_dim": self.model.embed_dim,
                "n_tiles": len(self.dataset)
            }
        )
    
        def _to_float(x, default: float = 0.0) -> float:
            if x is None:
                return default
            if torch.is_tensor(x):
                return float(x.detach().item())
            try:
                return float(x)
            except Exception:
                return default
    
        best_loss = float('inf')
        global_step = 0
        accum_steps = 4
    
        for epoch in range(epochs):
            self.model.train()
            # keep teacher in eval so BN stats don't drift
            self.model.teacher_stems.eval()
            self.model.teacher_encoder.eval()
            self.model.teacher_projector.eval()

            stats = defaultdict(float)
            n_batches = 0
    
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            optimizer.zero_grad(set_to_none=True)
    
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                if isinstance(batch, dict):
                    for k in batch:
                        if isinstance(batch[k], torch.Tensor):
                            batch[k] = batch[k].to(self.device)
                        elif isinstance(batch[k], list):
                            batch[k] = [x.to(self.device) if torch.is_tensor(x) else x for x in batch[k]]
    
                outputs = self.model(batch)
    
                # Robust scalar extraction
                loss_v  = _to_float(outputs.get('loss', 0.0))
                align_v = _to_float(outputs.get('align_loss', 0.0))
                var_v   = _to_float(outputs.get('var_loss', 0.0))
                cov_v   = _to_float(outputs.get('cov_loss', 0.0))
                tok_sim = _to_float(outputs.get('token_sim', 0.0))
                glob_sim = _to_float(outputs.get('global_sim', 0.0))
                glob_sim_w = _to_float(outputs.get('global_sim_weighted', 0.0))
    
                # Backprop with accumulation
                loss = outputs['loss'] / accum_steps
                loss.backward()
    
                do_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(self.dataloader))
                if do_step:
                    _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # EMA teacher update (REQUIRED)
                    self.model.update_teacher(m=0.996)
    
                # Stats
                stats['loss'] += loss_v
                stats['align'] += align_v
                stats['var'] += var_v
                stats['cov'] += cov_v
                stats['tok_sim'] += tok_sim
                stats['glob_sim'] += glob_sim
                stats['glob_sim_w'] += glob_sim_w
                n_batches += 1
    
                # Band usage tracking (guarded)
                if 'band1' in outputs:
                    self.band_tracker[outputs['band1']] += 1
                if 'band2' in outputs:
                    self.band_tracker[outputs['band2']] += 1
    
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
    
            # Visualization
            if (vis_freq and (epoch % vis_freq == 0)) or (epoch == epochs - 1):
                print(f"\n>>> Running visualization for Epoch {epoch}...")
                self.model.eval()
                vis_batch = next(iter(self.dataloader))
    
                # Move vis_batch to device
                for k in vis_batch:
                    if isinstance(vis_batch[k], torch.Tensor):
                        vis_batch[k] = vis_batch[k].to(self.device)
                    elif isinstance(vis_batch[k], list):
                        vis_batch[k] = [x.to(self.device) if torch.is_tensor(x) else x for x in vis_batch[k]]
    
                with torch.no_grad():
                    vis_out = self.model(vis_batch)
                self.visualize(vis_batch, vis_out, epoch, global_step)
                self.model.train()
                # keep teacher in eval so BN stats don't drift
                self.model.teacher_stems.eval()
                self.model.teacher_encoder.eval()
                self.model.teacher_projector.eval()

    
            # Epoch scheduler + checkpoint
            for k in stats:
                stats[k] /= max(n_batches, 1)
    
            scheduler.step()
    
            if (epoch + 1) % save_freq == 0 or stats['loss'] < best_loss:
                ckpt = {'model': self.model.state_dict(), 'loss': stats['loss']}
                out_path = (self.output_dir / "best.pt") if stats['loss'] < best_loss else (self.output_dir / f"ckpt_{epoch+1:03d}.pt")
                torch.save(ckpt, out_path)
                if stats['loss'] < best_loss:
                    best_loss = stats['loss']
    
        wandb.finish()

def main():
    import warnings
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*force_all_finite.*"
    )

    trainer = JAISPTrainerV4(
        rubin_dir="../data/rubin_tiles_ecdfs",
        euclid_dir="../data/euclid_tiles_ecdfs",
        output_dir="./checkpoints/jaisp_v4",
        batch_size=1,
        wandb_project="JAISP-Foundation-v4",
        wandb_name="v4_native_res"
    )

    trainer.train(epochs=100, vis_freq=5)

if __name__ == "__main__":
    main()
