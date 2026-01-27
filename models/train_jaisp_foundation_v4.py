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

from jaisp_foundation_v4 import JAISPFoundationV4, create_optimizer, create_scheduler
from jaisp_dataset_v4 import make_loader, BAND_WAVELENGTHS, ALL_BANDS


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
            temperature=0.1
        ).to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        
        self.band_tracker = defaultdict(int)
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
    
    def visualize(self, batch, outputs, epoch, step):
        """Visualization"""
        fig = plt.figure(figsize=(20, 12))
        
        img1 = batch['view1_image'][0, 0].cpu().numpy()
        img2 = batch['view2_image'][0, 0].cpu().numpy()
        band1, band2 = outputs['band1'], outputs['band2']
        w1 = outputs['weights1'][0, 0].cpu().numpy()
        w2 = outputs['weights2'][0, 0].cpu().numpy()
        z1 = outputs['z1'][0].cpu().numpy()
        z2 = outputs['z2'][0].cpu().numpy()
        gs = outputs['grid_size']
        
        def norm(x):
            p1, p99 = np.nanpercentile(x, [1, 99])
            return np.clip((x - p1) / (p99 - p1 + 1e-10), 0, 1)
        
        # Row 1: Images and weights
        ax = plt.subplot(3, 4, 1)
        ax.imshow(norm(img1), origin='lower', cmap='gray')
        ax.set_title(f'{band1}', fontsize=11, weight='bold')
        ax.axis('off')
        
        ax = plt.subplot(3, 4, 2)
        ax.imshow(w1, origin='lower', cmap='hot')
        ax.set_title('Weights (V1)', fontsize=11)
        ax.axis('off')
        
        ax = plt.subplot(3, 4, 3)
        ax.imshow(norm(img2), origin='lower', cmap='gray')
        ax.set_title(f'{band2}', fontsize=11, weight='bold')
        ax.axis('off')
        
        ax = plt.subplot(3, 4, 4)
        ax.imshow(w2, origin='lower', cmap='hot')
        ax.set_title('Weights (V2)', fontsize=11)
        ax.axis('off')
        
        # Row 2: Embedding analysis
        ax = plt.subplot(3, 4, 5)
        z1_norm = z1 / (np.linalg.norm(z1, axis=-1, keepdims=True) + 1e-10)
        z2_norm = z2 / (np.linalg.norm(z2, axis=-1, keepdims=True) + 1e-10)
        sim = (z1_norm * z2_norm).sum(axis=-1).reshape(gs)
        im = ax.imshow(sim, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f'Token Sim (μ={sim.mean():.3f})', fontsize=11)
        
        ax = plt.subplot(3, 4, 6)
        ax.hist(sim.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(sim.mean(), color='r', linestyle='--')
        ax.set_xlabel('Similarity')
        ax.set_title('Token Sim Distribution', fontsize=11)
        ax.grid(alpha=0.3)
        
        ax = plt.subplot(3, 4, 7)
        var1 = z1.var(axis=0)
        var2 = z2.var(axis=0)
        ax.plot(np.sort(var1)[::-1], alpha=0.7, label='V1')
        ax.plot(np.sort(var2)[::-1], alpha=0.7, label='V2')
        ax.axhline(1.0, color='r', linestyle='--', alpha=0.5)
        ax.set_yscale('log')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Variance')
        ax.legend()
        ax.set_title('Per-dim Variance', fontsize=11)
        ax.grid(alpha=0.3)
        
        # Weight-activation correlation
        ax = plt.subplot(3, 4, 8)
        w1_down = F.interpolate(
            torch.from_numpy(w1).unsqueeze(0).unsqueeze(0),
            size=gs, mode='bilinear'
        ).numpy().flatten()
        act1 = np.linalg.norm(z1, axis=-1)
        ax.scatter(w1_down, act1, alpha=0.5, s=10)
        corr = np.corrcoef(w1_down, act1)[0, 1]
        ax.set_xlabel('Info Weight')
        ax.set_ylabel('Embedding Norm')
        ax.set_title(f'Weight-Act Corr: {corr:.3f}', fontsize=11)
        ax.grid(alpha=0.3)
        
        # Row 3: More diagnostics
        # Band usage
        ax = plt.subplot(3, 4, 9)
        bands = list(self.band_tracker.keys())
        counts = [self.band_tracker[b] for b in bands]
        colors = ['#E74C3C' if 'rubin' in b else '#3498DB' for b in bands]
        ax.barh(range(len(bands)), counts, color=colors, alpha=0.7)
        ax.set_yticks(range(len(bands)))
        ax.set_yticklabels([b.split('_')[1] for b in bands], fontsize=8)
        ax.set_xlabel('Count')
        ax.set_title('Band Usage', fontsize=11)
        
        # Embedding covariance
        ax = plt.subplot(3, 4, 10)
        z_flat = z1 - z1.mean(axis=0)
        cov = (z_flat.T @ z_flat) / (z_flat.shape[0] - 1)
        n_show = min(32, cov.shape[0])
        im = ax.imshow(cov[:n_show, :n_show], cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title('Embedding Covariance', fontsize=11)
        
        # Summary
        ax = plt.subplot(3, 4, 11)
        ax.axis('off')
        wl1 = BAND_WAVELENGTHS.get(band1, 0)
        wl2 = BAND_WAVELENGTHS.get(band2, 0)
        summary = f"""Epoch {epoch}
{'='*25}

Bands: {band1} ↔ {band2}
λ gap: {abs(wl1-wl2)} nm

Loss: {outputs['loss'].item():.4f}
  Align: {outputs['align_loss'].item():.4f}
  Var:   {outputs['var_loss'].item():.4f}
  Cov:   {outputs['cov_loss'].item():.4f}

Similarities:
  Token: {outputs['token_sim']:.3f}
  Global: {outputs['global_sim']:.3f}
  Global (weighted): {outputs['global_sim_weighted']:.3f}

Weight-Act Corr: {corr:.3f}
"""
        ax.text(0.05, 0.95, summary, fontsize=9, family='monospace',
               va='top', transform=ax.transAxes)
        
        # UMAP if available
        ax = plt.subplot(3, 4, 12)
        try:
            from umap import UMAP
            combined = np.vstack([z1_norm[:100], z2_norm[:100]])
            reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            emb = reducer.fit_transform(combined)
            n = z1_norm[:100].shape[0]
            ax.scatter(emb[:n, 0], emb[:n, 1], c='#E74C3C', alpha=0.5, s=10, label='V1')
            ax.scatter(emb[n:, 0], emb[n:, 1], c='#3498DB', alpha=0.5, s=10, label='V2')
            ax.legend()
            ax.set_title('Token UMAP', fontsize=11)
        except:
            ax.text(0.5, 0.5, 'UMAP unavailable', ha='center', va='center')
        
        plt.tight_layout()
        wandb.log({"vis/overview": wandb.Image(fig)}, step=step)
        plt.close(fig)
    
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
                "embed_dim": self.model.embed_dim,
                "n_tiles": len(self.dataset)
            }
        )
        
        best_loss = float('inf')
        global_step = 0
        accum_steps = 4  # Gradient accumulation to simulate batch_size=4
        
        for epoch in range(epochs):
            self.model.train()
            stats = defaultdict(float)
            n_batches = 0
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(pbar):
                outputs = self.model(batch)
                loss = outputs['loss'] / accum_steps  # Scale loss for accumulation
                loss.backward()
                
                # Step optimizer every accum_steps
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(self.dataloader):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    grad_norm = torch.tensor(0.0)
                
                # Track (use unscaled loss)
                stats['loss'] += outputs['loss'].item()
                stats['align'] += outputs['align_loss'].item()
                stats['var'] += outputs['var_loss'].item()
                stats['cov'] += outputs['cov_loss'].item()
                stats['tok_sim'] += outputs['token_sim']
                stats['glob_sim'] += outputs['global_sim']
                stats['glob_sim_w'] += outputs['global_sim_weighted']
                n_batches += 1
                
                # Band tracking
                b1 = outputs['band1']
                b2 = outputs['band2']
                self.band_tracker[b1] += 1
                self.band_tracker[b2] += 1
                
                pbar.set_postfix({
                    'loss': f"{outputs['loss'].item():.3f}",
                    'tok_sim': f"{outputs['token_sim']:.3f}",
                    'glob_sim': f"{outputs['global_sim']:.3f}"
                })
                
                wandb.log({
                    'train/loss': outputs['loss'].item(),
                    'train/align_loss': outputs['align_loss'].item(),
                    'train/var_loss': outputs['var_loss'].item(),
                    'train/cov_loss': outputs['cov_loss'].item(),
                    'train/token_sim': outputs['token_sim'],
                    'train/global_sim': outputs['global_sim'],
                    'train/global_sim_weighted': outputs['global_sim_weighted'],
                    'train/grad_norm': grad_norm.item(),
                    'train/lr': optimizer.param_groups[0]['lr']
                }, step=global_step)
                
                global_step += 1
            
            # Visualization
            if epoch % vis_freq == 0:
                try:
                    self.model.eval()
                    with torch.no_grad():
                        vis_batch = next(iter(self.dataloader))
                        vis_out = self.model(vis_batch)
                    self.visualize(vis_batch, vis_out, epoch, global_step)
                    self.model.train()
                except Exception as e:
                    print(f"Vis failed: {e}")
            
            # Epoch summary
            for k in stats:
                stats[k] /= max(n_batches, 1)
            
            print(f"\nEpoch {epoch+1}: loss={stats['loss']:.4f}, "
                  f"tok_sim={stats['tok_sim']:.3f}, glob_sim={stats['glob_sim']:.3f}")
            
            wandb.log({
                'epoch/loss': stats['loss'],
                'epoch/token_sim': stats['tok_sim'],
                'epoch/global_sim': stats['glob_sim'],
                **{f'band_usage/{b}': c / sum(self.band_tracker.values()) 
                   for b, c in self.band_tracker.items()}
            }, step=global_step)
            
            scheduler.step()
            
            # Save
            if (epoch + 1) % save_freq == 0 or stats['loss'] < best_loss:
                ckpt = {
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': stats['loss']
                }
                torch.save(ckpt, self.output_dir / f"ckpt_{epoch+1:03d}.pt")
                if stats['loss'] < best_loss:
                    best_loss = stats['loss']
                    torch.save(ckpt, self.output_dir / "best.pt")
                    print(f"✨ New best: {best_loss:.4f}")
        
        wandb.finish()


def main():
    trainer = JAISPTrainerV4(
        rubin_dir="../data/rubin_tiles_ecdfs",
        euclid_dir="../data/euclid_tiles_ecdfs",
        output_dir="./checkpoints/jaisp_v4",
        embed_dim=256,
        proj_dim=256,
        depth=6,
        patch_size=16,
        batch_size=1,  # Reduced due to Euclid 1050x1050 memory requirements
        num_workers=4,
        wandb_project="JAISP-Foundation-v4",
        wandb_name="v4_native_res"
    )
    
    trainer.train(
        epochs=200,
        lr=3e-4,
        warmup_epochs=20,
        save_freq=20,
        vis_freq=5
    )


if __name__ == "__main__":
    main()