# train_stage1_foundation.py
#
# Training script for JEPA Foundation Model (Stage 1)

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb  # Optional: for logging

from JAISP_dataloader import make_loader
from stage1_jepa_foundation import JAISPFoundation, create_optimizer, create_scheduler


class Stage1Trainer:
    def __init__(self, 
                 rubin_dir: str,
                 euclid_dir: str,
                 output_dir: str = "./checkpoints",
                 batch_size: int = 4,
                 num_workers: int = 4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_wandb: bool = False):
        
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.use_wandb = use_wandb
        
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
            patch_size=128,
            n_patches_per_tile=4,
            vit_patch_size=16,
            embed_dim=384,
            depth=6,
            num_heads=6,
            projection_dim=256,
            temperature=0.07
        ).to(self.device)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    def train(self, 
              epochs: int = 100,
              lr: float = 1e-4,
              weight_decay: float = 0.05,
              warmup_epochs: int = 10,
              save_freq: int = 10,
              log_freq: int = 10):
        
        # Setup optimizer and scheduler
        optimizer = create_optimizer(self.model, lr=lr, weight_decay=weight_decay)
        scheduler = create_scheduler(optimizer, warmup_epochs, epochs)
        
        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project="JAISP-Foundation",
                config={
                    "epochs": epochs,
                    "lr": lr,
                    "batch_size": self.dataloader.batch_size,
                    "model": "JEPA-ViT",
                    "n_tiles": len(self.dataset)
                }
            )
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                # Move to device (only move tensors, not lists)
                # Lists stay as-is for the patch extractor
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch)
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Logging
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                if self.use_wandb and batch_idx % log_freq == 0:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'epoch': epoch
                    })
            
            # Epoch summary
            avg_loss = epoch_loss / len(self.dataloader)
            print(f"\nEpoch {epoch+1} Summary: Loss = {avg_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
            
            if self.use_wandb:
                wandb.log({'train/epoch_loss': avg_loss, 'epoch': epoch})
            
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
                }
                
                # Save regular checkpoint
                ckpt_path = self.output_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
                torch.save(checkpoint, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
                
                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = self.output_dir / "best_model.pt"
                    torch.save(checkpoint, best_path)
                    print(f"New best model! Loss: {best_loss:.4f}")
        
        print(f"\nTraining complete! Best loss: {best_loss:.4f}")
        
        if self.use_wandb:
            wandb.finish()
    
    def validate_sample(self):
        """Quick validation: visualize embeddings on a single batch"""
        self.model.eval()
        
        with torch.no_grad():
            batch = next(iter(self.dataloader))
            outputs = self.model(batch)
            
            print("\nSample batch validation:")
            print(f"  Loss: {outputs['loss'].item():.4f}")
            print(f"  Rubin embeddings: {outputs['z_rubin'].shape}")
            print(f"  Euclid embeddings: {outputs['z_euclid'].shape}")
            
            # Check embedding similarity
            similarity = torch.matmul(outputs['z_rubin'], outputs['z_euclid'].T)
            diag_sim = torch.diag(similarity).mean()
            off_diag_sim = (similarity.sum() - torch.diag(similarity).sum()) / (similarity.numel() - similarity.shape[0])
            
            print(f"  Avg diagonal similarity (matched pairs): {diag_sim:.4f}")
            print(f"  Avg off-diagonal similarity (non-matched): {off_diag_sim:.4f}")
            print(f"  Separation: {diag_sim - off_diag_sim:.4f} (higher is better)")


def main():
    # Configuration
    RUBIN_DIR = "./data/rubin_tiles"  # Update to your path
    EUCLID_DIR = "./data/euclid_tiles"  # Update to your path
    OUTPUT_DIR = "./checkpoints/stage1_foundation"
    
    BATCH_SIZE = 4  # Adjust based on your GPU memory
    NUM_WORKERS = 4
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    
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
        use_wandb=False  # Set to True if you want W&B logging
    )
    
    # Quick validation before training
    print("\n" + "="*60)
    print("Running validation on sample batch...")
    print("="*60)
    trainer.validate_sample()
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    trainer.train(
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        save_freq=10,
        log_freq=10
    )


if __name__ == "__main__":
    main()