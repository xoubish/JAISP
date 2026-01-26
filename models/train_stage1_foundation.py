# train_stage1_foundation.py
#
# Training script for JEPA Foundation Model (Stage 1)
# With comprehensive W&B logging

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from JAISP_dataset import make_loader
from stage1_jepa_foundation import JAISPFoundation, create_optimizer, create_scheduler


class Stage1Trainer:
    def __init__(self, 
                 rubin_dir: str,
                 euclid_dir: str,
                 output_dir: str = "./checkpoints",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 wandb_project: str = "JAISP-Foundation",
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
        
        # Create model
        self.model = JAISPFoundation(
            n_patches_per_tile=4,
            vit_patch_size=8,        # Match the 8x8 ViT patches we set for zoomed cuts
            embed_dim=384,
            depth=6,
            num_heads=6,
            projection_dim=256,
            temperature=0.05
        ).to(self.device)
        
        print("  Rubin patches:  128×128 pixels = 25.6\" × 25.6\" on sky")
        print("  Euclid patches: 256×256 pixels = 25.6\" × 25.6\" on sky (matched!)")
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        
        # W&B config
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
    
    def log_similarity_matrix_to_wandb(self, outputs, step):
        """Log similarity matrix visualization to W&B"""
        z_rubin = outputs['z_rubin'].detach().cpu().numpy()
        z_euclid = outputs['z_euclid'].detach().cpu().numpy()
        
        # Compute similarity (subsample if too large)
        n_samples = min(100, len(z_rubin))
        z_r = z_rubin[:n_samples]
        z_e = z_euclid[:n_samples]
        
        similarity = z_r @ z_e.T
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(similarity, cmap='RdBu_r', center=0, 
                   vmin=-1, vmax=1, square=True, ax=ax,
                   cbar_kws={'label': 'Cosine Similarity'})
        ax.set_xlabel('Euclid Patches')
        ax.set_ylabel('Rubin Patches')
        ax.set_title(f'Similarity Matrix (Step {step})')
        
        # Log to W&B
        wandb.log({"similarity_matrix": wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    def log_embedding_histogram(self, outputs, step):
        """Log embedding distribution histograms"""
        z_rubin = outputs['z_rubin'].detach().cpu().numpy()
        z_euclid = outputs['z_euclid'].detach().cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Rubin embeddings
        axes[0].hist(z_rubin.flatten(), bins=50, alpha=0.7, color='#E74C3C', edgecolor='black')
        axes[0].set_xlabel('Embedding Value')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Rubin Embedding Distribution')
        axes[0].grid(alpha=0.3)
        
        # Euclid embeddings
        axes[1].hist(z_euclid.flatten(), bins=50, alpha=0.7, color='#3498DB', edgecolor='black')
        axes[1].set_xlabel('Embedding Value')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Euclid Embedding Distribution')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        wandb.log({"embedding_distributions": wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    def train(self, 
              epochs: int = 100,
              lr: float = 1e-4,
              weight_decay: float = 0.05,
              warmup_epochs: int = 20,
              save_freq: int = 10,
              log_freq: int = 10,
              vis_freq: int = 50):  # Frequency for visual logs
        
        # Setup optimizer and scheduler
        optimizer = create_optimizer(self.model, lr=lr, weight_decay=weight_decay)
        scheduler = create_scheduler(optimizer, warmup_epochs, epochs)
        
        config = {
            "epochs": epochs,
            "lr": lr,
            "batch_size": self.dataloader.batch_size,
            "patch_size_rubin": 48,  # Updated
            "patch_size_euclid": 96, # Updated
            "vit_patch_size": 8,     # Updated
            "temperature": 0.05,     # Updated
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "n_tiles": len(self.dataset),
            "n_batches_per_epoch": len(self.dataloader),
            "patch_size": 128,
            "n_patches_per_tile": 4,
            "embed_dim": 384,
            "projection_dim": 256,
            "vit_depth": 6,
            "vit_heads": 6,
            "device": str(self.device),
            "model_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        wandb.init(
            project=self.wandb_project,
            name=self.wandb_name,
            config=config,
            settings=wandb.Settings(start_method="thread")
        )
        
        # Watch model (logs gradients and parameters)
        wandb.watch(self.model, log="all", log_freq=log_freq)
        
        # Training loop
        best_loss = float('inf')
        global_step = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_separation = 0.0
            epoch_diag_sim = 0.0
            epoch_off_diag_sim = 0.0
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch)
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Compute metrics
                with torch.no_grad():
                    similarity = torch.matmul(outputs['z_rubin'], outputs['z_euclid'].T)
                    diag_sim = torch.diag(similarity).mean().item()
                    
                    # Off-diagonal similarity
                    n = similarity.shape[0]
                    off_diag = similarity.clone()
                    off_diag.fill_diagonal_(0)
                    off_diag_sim = off_diag.sum().item() / (similarity.numel() - n) if n > 1 else 0.0
                    
                    separation = diag_sim - off_diag_sim
                    
                    # Top-1 accuracy (how many matches are correctly identified)
                    top1_acc = (similarity.argmax(dim=1) == torch.arange(n, device=similarity.device)).float().mean().item()
                
                # Accumulate for epoch stats
                epoch_loss += loss.item()
                epoch_separation += separation
                epoch_diag_sim += diag_sim
                epoch_off_diag_sim += off_diag_sim
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'sep': f'{separation:.4f}',
                    'top1': f'{top1_acc:.2%}'
                })
                
                # Log to W&B
                if batch_idx % log_freq == 0:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/separation': separation,
                        'train/diagonal_similarity': diag_sim,
                        'train/off_diagonal_similarity': off_diag_sim,
                        'train/top1_accuracy': top1_acc,
                        'train/gradient_norm': grad_norm.item(),
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'global_step': global_step
                    })
                
                # Log visualizations periodically
                if global_step % vis_freq == 0 and global_step > 0:
                    self.log_similarity_matrix_to_wandb(outputs, global_step)
                    self.log_embedding_histogram(outputs, global_step)
                
                global_step += 1
            
            # Epoch summary
            n_batches = len(self.dataloader)
            avg_loss = epoch_loss / n_batches
            avg_separation = epoch_separation / n_batches
            avg_diag_sim = epoch_diag_sim / n_batches
            avg_off_diag_sim = epoch_off_diag_sim / n_batches
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Separation: {avg_separation:.4f}")
            print(f"  Diagonal similarity: {avg_diag_sim:.4f}")
            print(f"  Off-diagonal similarity: {avg_off_diag_sim:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Log epoch metrics
            wandb.log({
                'train/epoch_loss': avg_loss,
                'train/epoch_separation': avg_separation,
                'train/epoch_diagonal_similarity': avg_diag_sim,
                'train/epoch_off_diagonal_similarity': avg_off_diag_sim,
                'epoch': epoch
            })
            
            # Step scheduler
            scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0 or avg_loss < best_loss:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'separation': avg_separation,
                }
                
                # Save regular checkpoint
                ckpt_path = self.output_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
                torch.save(checkpoint, ckpt_path)
                
                # Save to W&B
                wandb.save(str(ckpt_path))
                print(f"Saved checkpoint: {ckpt_path}")
                
                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = self.output_dir / "best_model.pt"
                    torch.save(checkpoint, best_path)
                    wandb.save(str(best_path))
                    print(f"✨ New best model! Loss: {best_loss:.4f}, Separation: {avg_separation:.4f}")
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best loss: {best_loss:.4f}")
        print(f"{'='*60}")
        
        # Final evaluation with visualizations
        print("\nGenerating final visualizations...")
        self.final_evaluation()
        
        wandb.finish()
    
    def final_evaluation(self):
        """Run final evaluation and log comprehensive visualizations"""
        self.model.eval()
        
        all_z_rubin = []
        all_z_euclid = []
        
        print("Extracting embeddings for final evaluation...")
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                outputs = self.model(batch)
                all_z_rubin.append(outputs['z_rubin'].cpu())
                all_z_euclid.append(outputs['z_euclid'].cpu())
        
        z_rubin = torch.cat(all_z_rubin, dim=0).numpy()
        z_euclid = torch.cat(all_z_euclid, dim=0).numpy()
        
        # Compute final metrics
        similarity = z_rubin @ z_euclid.T
        diag = np.diag(similarity)
        off_diag = similarity[~np.eye(similarity.shape[0], dtype=bool)]
        
        final_metrics = {
            'final/diagonal_mean': diag.mean(),
            'final/diagonal_std': diag.std(),
            'final/off_diagonal_mean': off_diag.mean(),
            'final/off_diagonal_std': off_diag.std(),
            'final/separation': diag.mean() - off_diag.mean(),
            'final/top1_accuracy': (similarity.argmax(axis=1) == np.arange(len(similarity))).mean(),
        }
        
        wandb.log(final_metrics)
        
        # Create final similarity matrix
        n_vis = min(200, len(similarity))
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Similarity matrix
        sns.heatmap(similarity[:n_vis, :n_vis], cmap='RdBu_r', center=0,
                   vmin=-1, vmax=1, square=True, ax=axes[0],
                   cbar_kws={'label': 'Cosine Similarity'})
        axes[0].set_title('Final Similarity Matrix')
        axes[0].set_xlabel('Euclid Patches')
        axes[0].set_ylabel('Rubin Patches')
        
        # Distribution comparison
        axes[1].hist(diag, bins=50, alpha=0.7, label='Matched pairs', color='#2ECC71', edgecolor='black')
        axes[1].hist(off_diag, bins=50, alpha=0.7, label='Unmatched pairs', color='#E74C3C', edgecolor='black')
        axes[1].axvline(diag.mean(), color='#2ECC71', linestyle='--', linewidth=2)
        axes[1].axvline(off_diag.mean(), color='#E74C3C', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Cosine Similarity')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Final Similarity Distribution')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        wandb.log({"final/similarity_analysis": wandb.Image(fig)})
        plt.close(fig)
        
        print("\n✓ Final evaluation complete")
        print(f"  Separation: {final_metrics['final/separation']:.4f}")
        print(f"  Top-1 Accuracy: {final_metrics['final/top1_accuracy']:.2%}")


def main():
    # Configuration
    RUBIN_DIR = "../data/rubin_tiles_ecdfs"
    EUCLID_DIR = "../data/euclid_tiles_ecdfs"
    OUTPUT_DIR = "./checkpoints/stage1_foundation"
    
    BATCH_SIZE = 8  # Adjust based on your GPU memory
    NUM_WORKERS = 4
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    
    # W&B configuration
    WANDB_PROJECT = "JAISP-Foundation"
    WANDB_NAME = f"stage1_jepa_b{BATCH_SIZE}_lr{LEARNING_RATE}"
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: No GPU detected, training will be slow!")
    
    # Create trainer
    trainer = Stage1Trainer(
        rubin_dir=RUBIN_DIR,
        euclid_dir=EUCLID_DIR,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        wandb_project=WANDB_PROJECT,
        wandb_name=WANDB_NAME
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    trainer.train(
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        save_freq=10,
        log_freq=10,
        vis_freq=50  # Log visualizations every 50 steps
    )


if __name__ == "__main__":
    main()