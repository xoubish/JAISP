# visualize_stage1_embeddings.py
#
# Visualization tools for JEPA Foundation Model (Stage 1)
# - UMAP embeddings
# - Training curves
# - Similarity matrices
# - Embedding quality metrics

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import umap
from sklearn.decomposition import PCA

from JAISP_dataloader import make_loader
from stage1_jepa_foundation import JAISPFoundation


class EmbeddingVisualizer:
    """Visualize and analyze JEPA embeddings"""
    
    def __init__(self, checkpoint_path, rubin_dir, euclid_dir, 
                 output_dir="./visualizations", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load model
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
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
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        print(f"Loss at checkpoint: {checkpoint['loss']:.4f}")
        
        # Load data
        print(f"\nLoading data...")
        self.dataset, self.dataloader = make_loader(
            rubin_dir=rubin_dir,
            euclid_dir=euclid_dir,
            batch_size=8,
            shuffle=False,  # Don't shuffle for consistent visualization
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Dataset: {len(self.dataset)} tiles")
    
    def extract_embeddings(self, max_batches=None):
        """Extract embeddings for all data"""
        print("\nExtracting embeddings...")
        
        z_rubin_all = []
        z_euclid_all = []
        z_rubin_raw_all = []
        z_euclid_raw_all = []
        tile_ids = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.dataloader)):
                if max_batches and i >= max_batches:
                    break
                
                outputs = self.model(batch)
                
                z_rubin_all.append(outputs['z_rubin'].cpu())
                z_euclid_all.append(outputs['z_euclid'].cpu())
                z_rubin_raw_all.append(outputs['z_rubin_raw'].cpu())
                z_euclid_raw_all.append(outputs['z_euclid_raw'].cpu())
                
                # Store tile IDs
                for meta in batch['meta']:
                    tile_ids.append(meta['tile_id'])
        
        z_rubin = torch.cat(z_rubin_all, dim=0).numpy()
        z_euclid = torch.cat(z_euclid_all, dim=0).numpy()
        z_rubin_raw = torch.cat(z_rubin_raw_all, dim=0).numpy()
        z_euclid_raw = torch.cat(z_euclid_raw_all, dim=0).numpy()
        
        print(f"\nExtracted embeddings:")
        print(f"  Rubin (projected): {z_rubin.shape}")
        print(f"  Euclid (projected): {z_euclid.shape}")
        print(f"  Rubin (raw): {z_rubin_raw.shape}")
        print(f"  Euclid (raw): {z_euclid_raw.shape}")
        
        return {
            'z_rubin': z_rubin,
            'z_euclid': z_euclid,
            'z_rubin_raw': z_rubin_raw,
            'z_euclid_raw': z_euclid_raw,
            'tile_ids': tile_ids
        }
    
    def plot_umap(self, embeddings, save_name="umap_embeddings.png"):
        """Create UMAP visualization of embeddings"""
        print("\nComputing UMAP projection...")
        
        z_rubin = embeddings['z_rubin']
        z_euclid = embeddings['z_euclid']
        
        # Combine embeddings
        z_combined = np.vstack([z_rubin, z_euclid])
        labels = np.array(['Rubin'] * len(z_rubin) + ['Euclid'] * len(z_euclid))
        
        # Compute UMAP
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='cosine',
            random_state=42
        )
        embedding_2d = reducer.fit_transform(z_combined)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Colored by survey
        ax = axes[0]
        for label, color, marker in [('Rubin', '#E74C3C', 'o'), 
                                      ('Euclid', '#3498DB', '^')]:
            mask = labels == label
            ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                      c=color, label=label, alpha=0.6, s=30, marker=marker,
                      edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title('UMAP Projection: Rubin vs Euclid Embeddings', fontsize=14, weight='bold')
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(alpha=0.3)
        
        # Plot 2: Matched pairs connected
        ax = axes[1]
        n_pairs = min(len(z_rubin), len(z_euclid))
        
        # Plot points
        ax.scatter(embedding_2d[:n_pairs, 0], embedding_2d[:n_pairs, 1],
                  c='#E74C3C', alpha=0.6, s=30, marker='o', label='Rubin',
                  edgecolors='white', linewidth=0.5)
        ax.scatter(embedding_2d[len(z_rubin):len(z_rubin)+n_pairs, 0], 
                  embedding_2d[len(z_rubin):len(z_rubin)+n_pairs, 1],
                  c='#3498DB', alpha=0.6, s=30, marker='^', label='Euclid',
                  edgecolors='white', linewidth=0.5)
        
        # Draw lines connecting matched pairs (sample to avoid clutter)
        sample_pairs = np.random.choice(n_pairs, min(100, n_pairs), replace=False)
        for i in sample_pairs:
            ax.plot([embedding_2d[i, 0], embedding_2d[len(z_rubin) + i, 0]],
                   [embedding_2d[i, 1], embedding_2d[len(z_rubin) + i, 1]],
                   'k-', alpha=0.1, linewidth=0.5)
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title('UMAP: Matched Pairs Connected (sample)', fontsize=14, weight='bold')
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved UMAP plot: {save_path}")
        plt.close()
        
        return embedding_2d
    
    def plot_similarity_matrix(self, embeddings, save_name="similarity_matrix.png", n_samples=200):
        """Plot similarity matrix between Rubin and Euclid embeddings"""
        print("\nComputing similarity matrix...")
        
        z_rubin = embeddings['z_rubin'][:n_samples]
        z_euclid = embeddings['z_euclid'][:n_samples]
        
        # Compute cosine similarity
        z_rubin_norm = z_rubin / (np.linalg.norm(z_rubin, axis=1, keepdims=True) + 1e-8)
        z_euclid_norm = z_euclid / (np.linalg.norm(z_euclid, axis=1, keepdims=True) + 1e-8)
        similarity = z_rubin_norm @ z_euclid_norm.T
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Full similarity matrix
        ax = axes[0]
        im = ax.imshow(similarity, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xlabel('Euclid Patches', fontsize=12)
        ax.set_ylabel('Rubin Patches', fontsize=12)
        ax.set_title(f'Similarity Matrix (first {n_samples} patches)', fontsize=14, weight='bold')
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        
        # Diagonal vs off-diagonal
        ax = axes[1]
        diag = np.diag(similarity)
        off_diag = similarity[~np.eye(similarity.shape[0], dtype=bool)]
        
        ax.hist(diag, bins=50, alpha=0.7, label='Diagonal (matched pairs)', 
               color='#2ECC71', edgecolor='black', linewidth=0.5)
        ax.hist(off_diag, bins=50, alpha=0.7, label='Off-diagonal (unmatched)', 
               color='#E74C3C', edgecolor='black', linewidth=0.5)
        ax.axvline(diag.mean(), color='#2ECC71', linestyle='--', linewidth=2, 
                  label=f'Matched mean: {diag.mean():.3f}')
        ax.axvline(off_diag.mean(), color='#E74C3C', linestyle='--', linewidth=2,
                  label=f'Unmatched mean: {off_diag.mean():.3f}')
        
        ax.set_xlabel('Cosine Similarity', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Similarities', fontsize=14, weight='bold')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(alpha=0.3)
        
        # Add separation metric
        separation = diag.mean() - off_diag.mean()
        ax.text(0.05, 0.95, f'Separation: {separation:.3f}', 
               transform=ax.transAxes, fontsize=12, weight='bold',
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved similarity matrix: {save_path}")
        plt.close()
        
        return similarity
    
    def plot_pca_variance(self, embeddings, save_name="pca_variance.png"):
        """Plot PCA variance explained"""
        print("\nComputing PCA...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for i, (name, z) in enumerate([('Rubin', embeddings['z_rubin']), 
                                        ('Euclid', embeddings['z_euclid'])]):
            pca = PCA(n_components=min(50, z.shape[1]))
            pca.fit(z)
            
            ax = axes[i]
            ax.plot(np.cumsum(pca.explained_variance_ratio_), 'o-', linewidth=2)
            ax.axhline(0.95, color='r', linestyle='--', label='95% variance')
            ax.axhline(0.99, color='orange', linestyle='--', label='99% variance')
            ax.set_xlabel('Number of Components', fontsize=12)
            ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
            ax.set_title(f'{name} PCA Variance', fontsize=14, weight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PCA variance plot: {save_path}")
        plt.close()
    
    def compute_metrics(self, embeddings):
        """Compute embedding quality metrics"""
        print("\nComputing metrics...")
        
        z_rubin = embeddings['z_rubin']
        z_euclid = embeddings['z_euclid']
        
        # Normalize
        z_rubin_norm = z_rubin / (np.linalg.norm(z_rubin, axis=1, keepdims=True) + 1e-8)
        z_euclid_norm = z_euclid / (np.linalg.norm(z_euclid, axis=1, keepdims=True) + 1e-8)
        
        # Similarity matrix
        similarity = z_rubin_norm @ z_euclid_norm.T
        
        # Metrics
        diag = np.diag(similarity)
        off_diag = similarity[~np.eye(similarity.shape[0], dtype=bool)]
        
        metrics = {
            'diagonal_mean': diag.mean(),
            'diagonal_std': diag.std(),
            'off_diagonal_mean': off_diag.mean(),
            'off_diagonal_std': off_diag.std(),
            'separation': diag.mean() - off_diag.mean(),
            'top1_accuracy': (similarity.argmax(axis=1) == np.arange(len(similarity))).mean(),
            'top5_accuracy': np.mean([i in similarity[i].argsort()[-5:] for i in range(len(similarity))])
        }
        
        print("\nEmbedding Quality Metrics:")
        print(f"  Matched pairs similarity:   {metrics['diagonal_mean']:.4f} ± {metrics['diagonal_std']:.4f}")
        print(f"  Unmatched pairs similarity: {metrics['off_diagonal_mean']:.4f} ± {metrics['off_diagonal_std']:.4f}")
        print(f"  Separation:                 {metrics['separation']:.4f}")
        print(f"  Top-1 Accuracy:             {metrics['top1_accuracy']:.2%}")
        print(f"  Top-5 Accuracy:             {metrics['top5_accuracy']:.2%}")
        
        return metrics
    
    def visualize_all(self, max_batches=None, log_to_wandb=False, wandb_run_path=None):
        """Run all visualizations"""
        print("\n" + "="*60)
        print("RUNNING ALL VISUALIZATIONS")
        print("="*60)
        
        # Initialize W&B if requested
        if log_to_wandb:
            import wandb
            wandb.init(
                project="JAISP-Foundation",
                name="evaluation",
                job_type="evaluation",
                resume="allow" if wandb_run_path else None,
                id=wandb_run_path.split('/')[-1] if wandb_run_path else None
            )
        
        # Extract embeddings
        embeddings = self.extract_embeddings(max_batches=max_batches)
        
        # Generate plots
        umap_embedding = self.plot_umap(embeddings)
        similarity = self.plot_similarity_matrix(embeddings)
        self.plot_pca_variance(embeddings)
        
        # Compute metrics
        metrics = self.compute_metrics(embeddings)
        
        # Log to W&B
        if log_to_wandb:
            # Log metrics
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})
            
            # Log plots as images
            for img_file in self.output_dir.glob("*.png"):
                wandb.log({f"eval/{img_file.stem}": wandb.Image(str(img_file))})
            
            print("\n✓ Logged to W&B")
        
        # Save metrics
        metrics_path = self.output_dir / "metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write("Embedding Quality Metrics\n")
            f.write("="*40 + "\n")
            for k, v in metrics.items():
                f.write(f"{k:25s}: {v:.4f}\n")
        
        print(f"\n✓ All visualizations saved to: {self.output_dir}")
        print(f"✓ Metrics saved to: {metrics_path}")
        
        if log_to_wandb:
            wandb.finish()


def plot_training_curves(checkpoint_dir, save_path="training_curves.png"):
    """Plot training curves from multiple checkpoints"""
    print("\nPlotting training curves...")
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    
    if len(checkpoints) == 0:
        print("No checkpoints found!")
        return
    
    epochs = []
    losses = []
    
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        epochs.append(ckpt['epoch'])
        losses.append(ckpt['loss'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, losses, 'o-', linewidth=2, markersize=6, color='#3498DB')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curve', fontsize=14, weight='bold')
    ax.grid(alpha=0.3)
    
    # Add best loss marker
    best_idx = np.argmin(losses)
    ax.plot(epochs[best_idx], losses[best_idx], 'r*', markersize=20, 
           label=f'Best: {losses[best_idx]:.4f} @ epoch {epochs[best_idx]}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves: {save_path}")
    plt.close()


def main():
    # Configuration
    CHECKPOINT_PATH = "./checkpoints/stage1_foundation/best_model.pt"  # or specific epoch
    RUBIN_DIR = "../data/rubin_tiles_ecdfs"
    EUCLID_DIR = "../data/euclid_tiles_ecdfs"
    OUTPUT_DIR = "./visualizations/stage1"
    
    # W&B configuration
    LOG_TO_WANDB = True  # Set to True to log visualizations to W&B
    WANDB_RUN_PATH = None  # Set to "username/project/run_id" to link to existing run
    
    # Create visualizer
    visualizer = EmbeddingVisualizer(
        checkpoint_path=CHECKPOINT_PATH,
        rubin_dir=RUBIN_DIR,
        euclid_dir=EUCLID_DIR,
        output_dir=OUTPUT_DIR,
        device="cuda"
    )
    
    # Run all visualizations
    # max_batches=None means use all data
    # Set max_batches=10 for quick testing
    visualizer.visualize_all(
        max_batches=None,
        log_to_wandb=LOG_TO_WANDB,
        wandb_run_path=WANDB_RUN_PATH
    )
    
    # Plot training curves (from checkpoints)
    if LOG_TO_WANDB:
        import wandb
        wandb.init(
            project="JAISP-Foundation",
            name="evaluation_curves",
            job_type="evaluation"
        )
    
    plot_training_curves(
        checkpoint_dir="./checkpoints/stage1_foundation",
        save_path=OUTPUT_DIR + "/training_curves.png"
    )
    
    if LOG_TO_WANDB:
        wandb.log({"training_curves": wandb.Image(OUTPUT_DIR + "/training_curves.png")})
        wandb.finish()
    
    print("\n" + "="*60)
    print("✓ VISUALIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()