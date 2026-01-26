# train_stage1_detr_jepa.py
#
# Training script for DETR-JEPA Foundation Model

import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

from JAISP_dataset import make_loader
from stage1_detr_jepa_foundation import (
    DETRJEPA, create_optimizer, create_scheduler
)


class DETRJEPATrainer:
    def __init__(self, 
                 rubin_dir: str,
                 euclid_dir: str,
                 output_dir: str = "./checkpoints",
                 batch_size: int = 1,
                 num_workers: int = 4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 wandb_project: str = "JAISP-DETR-JEPA",
                 wandb_name: str = None):
        
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create dataloaders
        print(f"Loading data from:\n  Rubin: {rubin_dir}\n  Euclid: {euclid_dir}")
        self.dataset, self.dataloader = make_loader(
            rubin_dir=rubin_dir,
            euclid_dir=euclid_dir,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
        
        print(f"Dataset: {len(self.dataset)} tiles")
        print(f"Batches per epoch: {len(self.dataloader)}")
        
        # Create model (reduced complexity for stability)
        self.model = DETRJEPA(
            rubin_channels=6,
            euclid_channels=4,
            patch_size=16,
            embed_dim=256,
            backbone_depth=4,
            decoder_depth=3,
            num_queries=50,
            num_heads=8
        ).to(self.device)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nModel parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        
        print("\n" + "="*60)
        print("DETR-JEPA: Object-Centric Foundation Model")
        print("="*60)
        print("Architecture:")
        print("  1. ViT backbone: Image → Patch features")
        print("  2. DETR decoder: 50 learnable queries → 50 object slots")
        print("  3. Each query 'discovers' one object via cross-attention")
        print("  4. JEPA: Align Rubin object manifold ↔ Euclid object manifold")
        print("  5. Hungarian matching: Optimal 1-1 correspondence")
        print("\nKey insight: Learn in OBJECT SPACE, not pixel/patch space!")
        print("="*60)
        
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
    
    def visualize_objects(self, batch, outputs, epoch, step):
        """Visualize what the model is learning"""
        # Get first tile
        rubin_img = batch['x_rubin'][0].cpu().numpy()
        euclid_img = batch['x_euclid'][0].cpu().numpy()
        
        # Create RGB composites
        rubin_rgb = np.stack([rubin_img[3], rubin_img[2], rubin_img[1]], axis=-1)
        euclid_rgb = np.stack([euclid_img[2], euclid_img[1], euclid_img[0]], axis=-1)
        
        # Normalize
        for i in range(3):
            p1, p99 = np.nanpercentile(rubin_rgb[..., i], [1, 99])
            rubin_rgb[..., i] = np.clip((rubin_rgb[..., i] - p1) / (p99 - p1 + 1e-10), 0, 1)
            p1, p99 = np.nanpercentile(euclid_rgb[..., i], [1, 99])
            euclid_rgb[..., i] = np.clip((euclid_rgb[..., i] - p1) / (p99 - p1 + 1e-10), 0, 1)
        
        # Create figure
        fig = plt.figure(figsize=(20, 10))
        
        # Images
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(rubin_rgb, origin='lower')
        ax1.set_title(f'Rubin (Epoch {epoch})', fontsize=12, weight='bold')
        ax1.axis('off')
        
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(euclid_rgb, origin='lower')
        ax2.set_title(f'Euclid (Epoch {epoch})', fontsize=12, weight='bold')
        ax2.axis('off')
        
        # UMAP
        ax3 = plt.subplot(2, 3, 3)
        try:
            from umap import UMAP
            with torch.no_grad():
                rubin_norm = F.normalize(outputs['rubin_objects'][0], dim=-1).cpu().numpy()
                euclid_norm = F.normalize(outputs['euclid_objects'][0], dim=-1).cpu().numpy()
            
            combined = np.vstack([rubin_norm, euclid_norm])
            if combined.shape[0] > 5:
                reducer = UMAP(n_neighbors=min(15, combined.shape[0]-1), 
                              min_dist=0.1, n_components=2, random_state=42)
                embedding_2d = reducer.fit_transform(combined)
                
                n_queries = rubin_norm.shape[0]
                ax3.scatter(embedding_2d[:n_queries, 0], embedding_2d[:n_queries, 1],
                           c='#E74C3C', alpha=0.6, s=50, label='Rubin', edgecolors='white', linewidth=1)
                ax3.scatter(embedding_2d[n_queries:, 0], embedding_2d[n_queries:, 1],
                           c='#3498DB', alpha=0.6, s=50, label='Euclid', edgecolors='white', linewidth=1)
                
                if 'matches' in outputs and outputs['matches'] is not None:
                    matches = outputs['matches'][0].cpu().numpy()[:10]
                    for i, j in enumerate(matches):
                        ax3.plot([embedding_2d[i, 0], embedding_2d[n_queries + j, 0]],
                               [embedding_2d[i, 1], embedding_2d[n_queries + j, 1]],
                               'gray', alpha=0.3, linewidth=1)
                
                ax3.set_title('Object Embedding Space (UMAP)', fontsize=12, weight='bold')
                ax3.legend(fontsize=9)
                ax3.grid(alpha=0.3)
        except Exception as e:
            ax3.text(0.5, 0.5, f'UMAP failed: {str(e)}', ha='center', va='center', transform=ax3.transAxes)
        
        # Similarity matrix
        ax4 = plt.subplot(2, 3, 4)
        try:
            with torch.no_grad():
                rubin_norm = F.normalize(outputs['rubin_objects'][0], dim=-1).cpu().numpy()
                euclid_norm = F.normalize(outputs['euclid_objects'][0], dim=-1).cpu().numpy()
                sim_matrix = rubin_norm @ euclid_norm.T
                
                n_show = min(50, sim_matrix.shape[0])
                im = ax4.imshow(sim_matrix[:n_show, :n_show], cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                
                if 'matches' in outputs and outputs['matches'] is not None:
                    matches = outputs['matches'][0].cpu().numpy()[:n_show]
                    for i, j in enumerate(matches[:n_show]):
                        if j < n_show:
                            ax4.plot(j, i, 'g*', markersize=10, markeredgecolor='yellow', markeredgewidth=1)
                
                plt.colorbar(im, ax=ax4, label='Cosine Similarity')
                ax4.set_xlabel('Euclid Objects', fontsize=10)
                ax4.set_ylabel('Rubin Objects', fontsize=10)
                ax4.set_title(f'Similarity Matrix (Avg: {outputs["similarity"]:.3f})', fontsize=12, weight='bold')
        except Exception as e:
            ax4.text(0.5, 0.5, f'Failed: {str(e)}', ha='center', va='center', transform=ax4.transAxes)
        
        # Histogram
        ax5 = plt.subplot(2, 3, 5)
        try:
            if 'matches' in outputs and outputs['matches'] is not None:
                matches = outputs['matches'][0].cpu().numpy()
                matched_sim = [sim_matrix[i, j] for i, j in enumerate(matches)]
                mask = np.ones_like(sim_matrix, dtype=bool)
                for i, j in enumerate(matches):
                    mask[i, j] = False
                off_diag = sim_matrix[mask]
                
                ax5.hist(matched_sim, bins=30, alpha=0.7, label='Matched', color='#2ECC71', edgecolor='black')
                ax5.hist(off_diag, bins=30, alpha=0.7, label='Unmatched', color='#E74C3C', edgecolor='black')
                ax5.axvline(np.mean(matched_sim), color='#2ECC71', linestyle='--', linewidth=2)
                ax5.axvline(np.mean(off_diag), color='#E74C3C', linestyle='--', linewidth=2)
                ax5.set_xlabel('Cosine Similarity', fontsize=10)
                ax5.set_ylabel('Count', fontsize=10)
                ax5.set_title('Matching Quality', fontsize=12, weight='bold')
                ax5.legend(fontsize=8)
                ax5.grid(alpha=0.3)
        except Exception as e:
            pass
        
        # Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        info_text = f"""Epoch {epoch} Summary
{'='*30}

Loss: {outputs['loss'].item():.4f}
Avg Similarity: {outputs['similarity']:.3f}
Object Queries: {outputs['n_objects']}
"""
        ax6.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax6.transAxes)
        
        plt.tight_layout()
        wandb.log({f"visualizations/overview": wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    def train(self, 
              epochs: int = 100,
              lr: float = 3e-5,
              weight_decay: float = 0.01,
              warmup_epochs: int = 10,
              save_freq: int = 10,
              log_freq: int = 1):
        
        optimizer = create_optimizer(self.model, lr=lr, weight_decay=weight_decay)
        scheduler = create_scheduler(optimizer, warmup_epochs, epochs)
        
        wandb.init(
            project=self.wandb_project,
            name=self.wandb_name,
            config={
                "epochs": epochs, "lr": lr, "n_tiles": len(self.dataset),
                "embed_dim": 256, "num_queries": 50
            },
            settings=wandb.Settings(start_method="thread")
        )
        
        wandb.watch(self.model, log="all", log_freq=log_freq*10)
        
        best_loss = float('inf')
        global_step = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_similarity = 0.0
            n_valid_batches = 0
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs['loss']
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_similarity += outputs['similarity']
                n_valid_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'sim': f'{outputs["similarity"]:.3f}'})
                
                if batch_idx % log_freq == 0:
                    has_nan = (torch.isnan(outputs['rubin_objects']).any() or 
                              torch.isnan(outputs['euclid_objects']).any())
                    
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/similarity': outputs['similarity'] if not np.isnan(outputs['similarity']) else 0.0,
                        'train/n_objects': outputs['n_objects'],
                        'train/gradient_norm': grad_norm.item(),
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'train/has_nan_embeddings': float(has_nan),
                        'train/rubin_embed_mean': outputs['rubin_objects'].mean().item(),
                        'train/rubin_embed_std': outputs['rubin_objects'].std().item(),
                        'epoch': epoch,
                        'global_step': global_step
                    })
                
                global_step += 1
            
            if n_valid_batches > 0 and epoch % 5 == 0:
                try:
                    self.visualize_objects(batch, outputs, epoch, global_step)
                except Exception as e:
                    print(f"Visualization failed: {e}")
            
            if n_valid_batches > 0:
                avg_loss = epoch_loss / n_valid_batches
                avg_similarity = epoch_similarity / n_valid_batches
                
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Loss: {avg_loss:.4f}")
                print(f"  Avg similarity: {avg_similarity:.3f}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
                
                wandb.log({'train/epoch_loss': avg_loss, 'train/epoch_similarity': avg_similarity, 'epoch': epoch})
            else:
                avg_loss = float('inf')
            
            scheduler.step()
            
            if (epoch + 1) % save_freq == 0 or avg_loss < best_loss:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }
                
                ckpt_path = self.output_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
                torch.save(checkpoint, ckpt_path)
                wandb.save(str(ckpt_path))
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = self.output_dir / "best_model.pt"
                    torch.save(checkpoint, best_path)
                    wandb.save(str(best_path))
                    print(f"✨ New best! Loss: {best_loss:.4f}")
        
        wandb.finish()


def main():
    RUBIN_DIR = "../data/rubin_tiles_ecdfs"
    EUCLID_DIR = "../data/euclid_tiles_ecdfs"
    OUTPUT_DIR = "./checkpoints/stage1_detr_jepa"
    
    trainer = DETRJEPATrainer(
        rubin_dir=RUBIN_DIR,
        euclid_dir=EUCLID_DIR,
        output_dir=OUTPUT_DIR,
        batch_size=1,
        num_workers=4,
        wandb_project="JAISP-DETR-JEPA",
        wandb_name="stage1_detr_jepa_lr3e-5"
    )
    
    trainer.train(epochs=100, lr=3e-5, save_freq=10, log_freq=1)


if __name__ == "__main__":
    main()