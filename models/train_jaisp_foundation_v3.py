# train_jaisp_foundation_v3.py
#
# Training script for JAISP Foundation v3 - Per-Band Views

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from collections import defaultdict

from jaisp_foundation_v3 import JAISPFoundationV3, create_optimizer, create_scheduler
from jaisp_dataset_v3 import make_loader, BAND_WAVELENGTHS


class BandUsageTracker:
    """Track which bands are being used in training"""
    def __init__(self, band_names):
        self.counts = {b: 0 for b in band_names}
        self.pair_counts = defaultdict(int)
    
    def update(self, band1: str, band2: str):
        self.counts[band1] += 1
        self.counts[band2] += 1
        pair = tuple(sorted([band1, band2]))
        self.pair_counts[pair] += 1
    
    def get_stats(self) -> dict:
        total = sum(self.counts.values())
        if total == 0:
            return {}
        return {f'band_usage/{b}': c / total for b, c in self.counts.items()}


class Diagnostics:
    """Diagnostic tools for v3"""
    
    @staticmethod
    @torch.no_grad()
    def cross_band_similarity_matrix(model, dataloader, device, n_samples=50):
        """
        Compute average similarity between all band pairs.
        
        This tells us: are all bands mapping to similar space?
        """
        model.eval()
        band_embeddings = defaultdict(list)
        
        for i, batch in enumerate(dataloader):
            if i >= n_samples:
                break
            
            for view in ['view1', 'view2']:
                img = batch[f'{view}_image'].to(device)
                rms = batch[f'{view}_rms'].to(device)
                band = batch[f'{view}_band']
                if isinstance(band, (list, tuple)):
                    band = band[0]
                
                out = model.encode_band(img, rms, band)
                # Global pooling
                global_emb = out['tokens'].mean(dim=1)  # (B, D)
                global_emb = F.normalize(global_emb, dim=-1)
                
                band_embeddings[band].append(global_emb.cpu())
        
        # Average embedding per band
        band_means = {}
        for band, embs in band_embeddings.items():
            if embs:
                stacked = torch.cat(embs, dim=0)
                band_means[band] = stacked.mean(dim=0)
        
        # Similarity matrix
        bands = sorted(band_means.keys())
        n = len(bands)
        sim_matrix = np.zeros((n, n))
        
        for i, b1 in enumerate(bands):
            for j, b2 in enumerate(bands):
                sim = F.cosine_similarity(
                    band_means[b1].unsqueeze(0),
                    band_means[b2].unsqueeze(0)
                ).item()
                sim_matrix[i, j] = sim
        
        model.train()
        return sim_matrix, bands
    
    @staticmethod
    @torch.no_grad()
    def per_band_collapse_check(model, dataloader, device, n_samples=20):
        """Check if any individual band has collapsed"""
        model.eval()
        band_variances = defaultdict(list)
        
        for i, batch in enumerate(dataloader):
            if i >= n_samples:
                break
            
            for view in ['view1', 'view2']:
                img = batch[f'{view}_image'].to(device)
                rms = batch[f'{view}_rms'].to(device)
                band = batch[f'{view}_band']
                if isinstance(band, (list, tuple)):
                    band = band[0]
                
                out = model.encode_band(img, rms, band)
                tokens = out['tokens']  # (B, N, D)
                
                # Per-dim variance
                var = tokens.var(dim=(0, 1))  # (D,)
                band_variances[band].append(var.cpu())
        
        # Average variance per band
        results = {}
        for band, vars in band_variances.items():
            if vars:
                avg_var = torch.stack(vars).mean(dim=0)
                results[band] = {
                    'var_mean': avg_var.mean().item(),
                    'var_min': avg_var.min().item(),
                    'effective_dim': (1.0 / (avg_var / avg_var.sum() + 1e-10).pow(2).sum()).item()
                }
        
        model.train()
        return results


class JAISPTrainerV3:
    def __init__(self,
                 rubin_dir: str,
                 euclid_dir: str,
                 output_dir: str = "./checkpoints",
                 embed_dim: int = 256,
                 trunk_depth: int = 6,
                 patch_size: int = 16,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 wandb_project: str = "JAISP-Foundation-v3",
                 wandb_name: str = None):
        
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create dataset
        print(f"Loading data...")
        self.dataset, self.dataloader = make_loader(
            rubin_dir=rubin_dir,
            euclid_dir=euclid_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            patch_size=512,
            augment=True,
            hard_pair_prob=0.3,
            cross_instrument_prob=0.5
        )
        
        self.band_names = self.dataset.get_band_names()
        print(f"Dataset: {len(self.dataset)} tiles, {len(self.band_names)} bands")
        
        # Create model
        self.model = JAISPFoundationV3(
            band_names=self.band_names,
            stem_channels=64,
            embed_dim=embed_dim,
            trunk_depth=trunk_depth,
            num_heads=8,
            patch_size=patch_size,
            use_ema=True,
            ema_decay=0.996
        ).to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        
        self.band_tracker = BandUsageTracker(self.band_names)
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
    
    def visualize_batch(self, batch, outputs, epoch, step):
        """Visualize training batch"""
        fig = plt.figure(figsize=(20, 15))
        
        # Get data
        img1 = batch['view1_image'][0, 0].cpu().numpy()
        img2 = batch['view2_image'][0, 0].cpu().numpy()
        band1 = outputs['band1']
        band2 = outputs['band2']
        weights1 = outputs['weights1'][0, 0].cpu().numpy()
        weights2 = outputs['weights2'][0, 0].cpu().numpy()
        tokens1 = outputs['tokens1'][0].cpu().numpy()
        tokens2 = outputs['tokens2'][0].cpu().numpy()
        grid_size = outputs['grid_size']
        
        # Normalize images for display
        def norm_img(x):
            p1, p99 = np.nanpercentile(x, [1, 99])
            return np.clip((x - p1) / (p99 - p1 + 1e-10), 0, 1)
        
        # Row 1: Images and weights
        ax1 = plt.subplot(3, 4, 1)
        ax1.imshow(norm_img(img1), origin='lower', cmap='gray')
        wl1 = BAND_WAVELENGTHS.get(band1, 0)
        ax1.set_title(f'{band1} ({wl1}nm)', fontsize=11, weight='bold')
        ax1.axis('off')
        
        ax2 = plt.subplot(3, 4, 2)
        im = ax2.imshow(weights1, origin='lower', cmap='hot')
        plt.colorbar(im, ax=ax2, fraction=0.046)
        ax2.set_title(f'Info Weights ({band1})', fontsize=11)
        
        ax3 = plt.subplot(3, 4, 3)
        ax3.imshow(norm_img(img2), origin='lower', cmap='gray')
        wl2 = BAND_WAVELENGTHS.get(band2, 0)
        ax3.set_title(f'{band2} ({wl2}nm)', fontsize=11, weight='bold')
        ax3.axis('off')
        
        ax4 = plt.subplot(3, 4, 4)
        im = ax4.imshow(weights2, origin='lower', cmap='hot')
        plt.colorbar(im, ax=ax4, fraction=0.046)
        ax4.set_title(f'Info Weights ({band2})', fontsize=11)
        
        # Row 2: Token analysis
        ax5 = plt.subplot(3, 4, 5)
        token_act1 = np.linalg.norm(tokens1, axis=-1).reshape(grid_size)
        im = ax5.imshow(token_act1, origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax5, fraction=0.046)
        ax5.set_title('Token Activation (V1)', fontsize=11)
        
        ax6 = plt.subplot(3, 4, 6)
        t1_norm = tokens1 / (np.linalg.norm(tokens1, axis=-1, keepdims=True) + 1e-10)
        t2_norm = tokens2 / (np.linalg.norm(tokens2, axis=-1, keepdims=True) + 1e-10)
        token_sim = (t1_norm * t2_norm).sum(axis=-1).reshape(grid_size)
        im = ax6.imshow(token_sim, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax6, fraction=0.046)
        ax6.set_title(f'Token Sim ({band1}↔{band2})', fontsize=11)
        
        ax7 = plt.subplot(3, 4, 7)
        ax7.hist(token_sim.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax7.axvline(token_sim.mean(), color='r', linestyle='--', 
                   label=f'μ={token_sim.mean():.3f}')
        ax7.set_xlabel('Cosine Similarity')
        ax7.set_title('Token Sim Distribution', fontsize=11)
        ax7.legend()
        ax7.grid(alpha=0.3)
        
        ax8 = plt.subplot(3, 4, 8)
        var1 = np.var(tokens1, axis=0)
        var2 = np.var(tokens2, axis=0)
        ax8.plot(np.sort(var1)[::-1], label=band1, alpha=0.7)
        ax8.plot(np.sort(var2)[::-1], label=band2, alpha=0.7)
        ax8.axhline(1.0, color='r', linestyle='--', alpha=0.5)
        ax8.set_xlabel('Dimension')
        ax8.set_ylabel('Variance')
        ax8.set_yscale('log')
        ax8.legend()
        ax8.set_title('Per-dim Variance', fontsize=11)
        ax8.grid(alpha=0.3)
        
        # Row 3: Summary and band usage
        ax9 = plt.subplot(3, 4, 9)
        usage = self.band_tracker.counts
        bands = list(usage.keys())
        counts = [usage[b] for b in bands]
        colors = ['#E74C3C' if 'rubin' in b else '#3498DB' for b in bands]
        ax9.barh(range(len(bands)), counts, color=colors, alpha=0.7)
        ax9.set_yticks(range(len(bands)))
        ax9.set_yticklabels([b.replace('rubin_', 'R:').replace('euclid_', 'E:') for b in bands])
        ax9.set_xlabel('Usage Count')
        ax9.set_title('Band Usage', fontsize=11)
        ax9.grid(alpha=0.3, axis='x')
        
        # Weight-activation correlation
        ax10 = plt.subplot(3, 4, 10)
        w1_down = F.interpolate(
            torch.from_numpy(weights1).unsqueeze(0).unsqueeze(0),
            size=grid_size, mode='bilinear', align_corners=False
        ).numpy().flatten()
        ax10.scatter(w1_down, token_act1.flatten(), alpha=0.5, s=10)
        corr = np.corrcoef(w1_down, token_act1.flatten())[0, 1]
        ax10.set_xlabel('Info Weight')
        ax10.set_ylabel('Token Activation')
        ax10.set_title(f'Weight-Act Corr: {corr:.3f}', fontsize=11)
        ax10.grid(alpha=0.3)
        
        # Wavelength gap vs similarity
        ax11 = plt.subplot(3, 4, 11)
        wl_gap = abs(wl1 - wl2)
        ax11.bar(['Gap (nm)', 'Similarity'], [wl_gap/10, outputs['token_similarity']*100],
                color=['#9B59B6', '#2ECC71'], alpha=0.7)
        ax11.set_title(f'λ Gap: {wl_gap}nm, Sim: {outputs["token_similarity"]:.3f}', fontsize=11)
        ax11.grid(alpha=0.3, axis='y')
        
        # Summary text
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        summary = f"""Epoch {epoch}
{'='*30}

Bands: {band1} ↔ {band2}
Wavelength gap: {wl_gap} nm

Loss: {outputs['loss'].item():.4f}
  JEPA:     {outputs['jepa_loss'].item():.4f}
  Variance: {outputs['var_loss'].item():.4f}
  Covar:    {outputs['cov_loss'].item():.4f}

Token Similarity: {outputs['token_similarity']:.3f}
Global Similarity: {outputs['global_similarity']:.3f}
Weight-Act Corr: {corr:.3f}
"""
        ax12.text(0.05, 0.95, summary, fontsize=9, family='monospace',
                 verticalalignment='top', transform=ax12.transAxes)
        
        plt.tight_layout()
        wandb.log({"visualizations/batch": wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    def run_diagnostics(self, epoch, step):
        """Run cross-band diagnostics"""
        print("\n📊 Running diagnostics...")
        
        # Cross-band similarity matrix
        try:
            sim_matrix, bands = Diagnostics.cross_band_similarity_matrix(
                self.model, self.dataloader, self.device, n_samples=30
            )
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(len(bands)))
            ax.set_yticks(range(len(bands)))
            ax.set_xticklabels([b.split('_')[1] for b in bands], rotation=45, ha='right')
            ax.set_yticklabels([b.split('_')[1] for b in bands])
            
            # Add values
            for i in range(len(bands)):
                for j in range(len(bands)):
                    ax.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)
            
            plt.colorbar(im, ax=ax, label='Cosine Similarity')
            ax.set_title(f'Cross-Band Similarity Matrix (Epoch {epoch})')
            
            # Color code by instrument
            for i, b in enumerate(bands):
                color = '#E74C3C' if 'rubin' in b else '#3498DB'
                ax.get_yticklabels()[i].set_color(color)
                ax.get_xticklabels()[i].set_color(color)
            
            plt.tight_layout()
            wandb.log({"diagnostics/cross_band_sim": wandb.Image(fig)}, step=step)
            plt.close(fig)
            
            # Log average cross-instrument similarity
            rubin_idx = [i for i, b in enumerate(bands) if 'rubin' in b]
            euclid_idx = [i for i, b in enumerate(bands) if 'euclid' in b]
            if rubin_idx and euclid_idx:
                cross_sim = sim_matrix[np.ix_(rubin_idx, euclid_idx)].mean()
                wandb.log({"diagnostics/cross_instrument_sim": cross_sim}, step=step)
                print(f"  Cross-instrument similarity: {cross_sim:.3f}")
        except Exception as e:
            print(f"  Cross-band sim failed: {e}")
        
        # Per-band collapse check
        try:
            collapse = Diagnostics.per_band_collapse_check(
                self.model, self.dataloader, self.device, n_samples=20
            )
            
            for band, stats in collapse.items():
                wandb.log({
                    f"diagnostics/var_mean_{band}": stats['var_mean'],
                    f"diagnostics/eff_dim_{band}": stats['effective_dim']
                }, step=step)
            
            # Alert if any band collapsed
            min_var = min(s['var_min'] for s in collapse.values())
            min_eff_dim = min(s['effective_dim'] for s in collapse.values())
            print(f"  Min variance across bands: {min_var:.4f}")
            print(f"  Min effective dim: {min_eff_dim:.1f}")
            
            if min_var < 0.01:
                print("  ⚠️ WARNING: Some band may be collapsing!")
        except Exception as e:
            print(f"  Collapse check failed: {e}")
    
    def train(self,
              epochs: int = 100,
              lr: float = 3e-4,
              weight_decay: float = 0.05,
              warmup_epochs: int = 10,
              save_freq: int = 10,
              vis_freq: int = 5,
              diag_freq: int = 10):
        
        optimizer = create_optimizer(self.model, lr=lr, weight_decay=weight_decay)
        scheduler = create_scheduler(optimizer, warmup_epochs, epochs)
        
        wandb.init(
            project=self.wandb_project,
            name=self.wandb_name,
            config={
                "architecture": "JAISP-Foundation-v3-PerBand",
                "epochs": epochs,
                "lr": lr,
                "embed_dim": self.model.embed_dim,
                "n_bands": len(self.band_names),
                "bands": self.band_names,
                "n_tiles": len(self.dataset),
            }
        )
        
        best_loss = float('inf')
        global_step = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_stats = defaultdict(float)
            n_batches = 0
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()
                
                outputs = self.model(batch)
                loss = outputs['loss']
                
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track band usage
                band1 = batch['view1_band'][0] if isinstance(batch['view1_band'], list) else batch['view1_band']
                band2 = batch['view2_band'][0] if isinstance(batch['view2_band'], list) else batch['view2_band']
                self.band_tracker.update(band1, band2)
                
                # Stats
                epoch_stats['loss'] += loss.item()
                epoch_stats['jepa'] += outputs['jepa_loss'].item()
                epoch_stats['var'] += outputs['var_loss'].item()
                epoch_stats['cov'] += outputs['cov_loss'].item()
                epoch_stats['tok_sim'] += outputs['token_similarity']
                epoch_stats['glob_sim'] += outputs['global_similarity']
                n_batches += 1
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.3f}",
                    'sim': f"{outputs['token_similarity']:.3f}",
                    'bands': f"{band1.split('_')[1][:2]}↔{band2.split('_')[1][:2]}"
                })
                
                # Log
                wandb.log({
                    'train/loss': loss.item(),
                    'train/jepa_loss': outputs['jepa_loss'].item(),
                    'train/var_loss': outputs['var_loss'].item(),
                    'train/cov_loss': outputs['cov_loss'].item(),
                    'train/token_sim': outputs['token_similarity'],
                    'train/global_sim': outputs['global_similarity'],
                    'train/grad_norm': grad_norm.item(),
                    'train/lr': optimizer.param_groups[0]['lr'],
                }, step=global_step)
                
                global_step += 1
            
            # Visualization
            if epoch % vis_freq == 0:
                try:
                    self.model.eval()
                    with torch.no_grad():
                        vis_batch = next(iter(self.dataloader))
                        vis_out = self.model(vis_batch)
                    self.visualize_batch(vis_batch, vis_out, epoch, global_step)
                    self.model.train()
                except Exception as e:
                    print(f"Vis failed: {e}")
            
            # Diagnostics
            if epoch % diag_freq == 0 and epoch > 0:
                self.run_diagnostics(epoch, global_step)
            
            # Epoch summary
            for k in epoch_stats:
                epoch_stats[k] /= max(n_batches, 1)
            
            print(f"\nEpoch {epoch+1}: loss={epoch_stats['loss']:.4f}, "
                  f"tok_sim={epoch_stats['tok_sim']:.3f}, glob_sim={epoch_stats['glob_sim']:.3f}")
            
            # Log band usage
            wandb.log(self.band_tracker.get_stats(), step=global_step)
            
            scheduler.step()
            
            # Save
            if (epoch + 1) % save_freq == 0 or epoch_stats['loss'] < best_loss:
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_stats['loss'],
                    'band_names': self.band_names
                }
                torch.save(ckpt, self.output_dir / f"checkpoint_{epoch+1:03d}.pt")
                
                if epoch_stats['loss'] < best_loss:
                    best_loss = epoch_stats['loss']
                    torch.save(ckpt, self.output_dir / "best_model.pt")
                    print(f"✨ New best: {best_loss:.4f}")
        
        wandb.finish()


def main():
    trainer = JAISPTrainerV3(
        rubin_dir="../data/rubin_tiles_ecdfs",
        euclid_dir="../data/euclid_tiles_ecdfs",
        output_dir="./checkpoints/jaisp_v3",
        embed_dim=256,
        trunk_depth=6,
        patch_size=16,
        batch_size=4,
        num_workers=4,
        wandb_project="JAISP-Foundation-v3",
        wandb_name="v3_perband"
    )
    
    trainer.train(
        epochs=100,
        lr=3e-4,
        warmup_epochs=10,
        save_freq=10,
        vis_freq=5,
        diag_freq=10
    )


if __name__ == "__main__":
    main()
