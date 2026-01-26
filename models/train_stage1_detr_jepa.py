# train_stage1_detr_jepa.py
#
# Training script for DETR-JEPA Foundation Model

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb

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
        
        # Create model
        self.model = DETRJEPA(
            rubin_channels=6,
            euclid_channels=4,
            patch_size=16,
            embed_dim=384,
            backbone_depth=6,
            decoder_depth=6,
            num_queries=100,
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
        print("  2. DETR decoder: 100 learnable queries → 100 object slots")
        print("  3. Each query 'discovers' one object via cross-attention")
        print("  4. JEPA: Align Rubin object manifold ↔ Euclid object manifold")
        print("  5. Hungarian matching: Optimal 1-1 correspondence")
        print("\nKey insight: Learn in OBJECT SPACE, not pixel/patch space!")
        print("="*60)
        
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
    def __init__(self, 
                 rubin_dir: str,
                 euclid_dir: str,
                 output_dir: str = "./checkpoints",
                 batch_size: int = 1,  # Process one tile at a time for object detection
                 num_workers: int = 4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 wandb_project: str = "JAISP-ObjectCentric",
                 wandb_name: str = None):
        
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create dataloaders (batch_size=1 for object detection)
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
        self.model = ObjectCentricJAISP(
            rubin_channels=6,
            euclid_channels=4,
            embed_dim=256,
            temperature=0.1,
            detection_threshold=0.3,  # Lower = more sources detected
            max_sources=100  # Limit per tile
        ).to(self.device)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        
        print("\n" + "="*60)
        print("OBJECT-CENTRIC APPROACH")
        print("="*60)
        print("Key differences from patch-level matching:")
        print("  1. Detects sources in each tile (learnable detector)")
        print("  2. Extracts features for EACH source")
        print("  3. Matches individual sources across Rubin-Euclid")
        print("  4. Loss: match corresponding sources, not background")
        print("\nThis focuses learning on actual astronomical objects!")
        print("="*60)
        
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
    
    def train(self, 
              epochs: int = 100,
              lr: float = 1e-4,
              weight_decay: float = 0.01,
              warmup_epochs: int = 10,
              save_freq: int = 10,
              log_freq: int = 1):
        
        # Setup optimizer and scheduler
        optimizer = create_optimizer(self.model, lr=lr, weight_decay=weight_decay)
        scheduler = create_scheduler(optimizer, warmup_epochs, epochs)
        
        # Initialize wandb
        config = {
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "n_tiles": len(self.dataset),
            "embed_dim": 256,
            "temperature": 0.1,
            "detection_threshold": 0.3,
            "max_sources": 100,
            "device": str(self.device),
            "model_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        wandb.init(
            project=self.wandb_project,
            name=self.wandb_name,
            config=config,
            settings=wandb.Settings(start_method="thread")
        )
        
        wandb.watch(self.model, log="all", log_freq=log_freq*10)
        
        # Training loop
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
                
                # Forward pass
                outputs = self.model(batch)
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Logging
                epoch_loss += loss.item()
                epoch_similarity += outputs['similarity']
                n_valid_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'sim': f'{outputs["similarity"]:.3f}'
                })
                
                # Log to W&B
                if batch_idx % log_freq == 0:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/similarity': outputs['similarity'],
                        'train/n_objects': outputs['n_objects'],
                        'train/gradient_norm': grad_norm.item(),
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'global_step': global_step
                    })
                
                global_step += 1
            
            # Epoch summary
            if n_valid_batches > 0:
                avg_loss = epoch_loss / n_valid_batches
                avg_similarity = epoch_similarity / n_valid_batches
                
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Loss: {avg_loss:.4f}")
                print(f"  Avg matched similarity: {avg_similarity:.3f}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
                
                # Log epoch metrics
                wandb.log({
                    'train/epoch_loss': avg_loss,
                    'train/epoch_similarity': avg_similarity,
                    'epoch': epoch
                })
            else:
                print(f"\nEpoch {epoch+1}: No valid batches")
                avg_loss = float('inf')
            
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
                wandb.save(str(ckpt_path))
                print(f"Saved checkpoint: {ckpt_path}")
                
                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = self.output_dir / "best_model.pt"
                    torch.save(checkpoint, best_path)
                    wandb.save(str(best_path))
                    print(f"✨ New best model! Loss: {best_loss:.4f}")
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best loss: {best_loss:.4f}")
        print(f"{'='*60}")
        
        wandb.finish()


def main():
    # Configuration
    RUBIN_DIR = "../data/rubin_tiles_ecdfs"
    EUCLID_DIR = "../data/euclid_tiles_ecdfs"
    OUTPUT_DIR = "./checkpoints/stage1_detr_jepa"
    
    NUM_WORKERS = 4
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    
    # W&B configuration
    WANDB_PROJECT = "JAISP-DETR-JEPA"
    WANDB_NAME = f"stage1_detr_jepa_lr{LEARNING_RATE}"
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: No GPU detected, training will be slow!")
    
    # Create trainer
    trainer = DETRJEPATrainer(
        rubin_dir=RUBIN_DIR,
        euclid_dir=EUCLID_DIR,
        output_dir=OUTPUT_DIR,
        batch_size=1,
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
        log_freq=1
    )


if __name__ == "__main__":
    main()