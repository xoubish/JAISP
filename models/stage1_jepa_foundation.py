# stage1_jepa_foundation.py
#
# JEPA Foundation Model for Rubin + Euclid
# Self-supervised learning of instrument-agnostic sky representations
#
# Architecture:
# - Dual ViT encoders (Rubin: 6 bands, Euclid: 4 bands)
# - Shared projection head to common embedding space
# - Contrastive loss with variance weighting
# - Handles variable-sized tiles via random patch extraction

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional


class PatchExtractor(nn.Module):
    """
    Extract fixed-size patches from variable-sized tiles.
    Handles RMS weighting and normalization.
    """
    def __init__(self, patch_size: int = 128, n_patches: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = n_patches
    
    def forward(self, img_list: List[torch.Tensor], 
                rms_list: List[torch.Tensor],
                valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_list: List of (C, H, W) tensors
            rms_list: List of (C, H, W) RMS maps
            valid_mask: Optional (B, C) mask indicating valid bands
        
        Returns:
            patches: (B*n_patches, C, patch_size, patch_size)
            weights: (B*n_patches, C, patch_size, patch_size) - inverse variance weights
        """
        all_patches = []
        all_weights = []
        
        for img, rms in zip(img_list, rms_list):
            C, H, W = img.shape
            
            # Extract n_patches from this tile
            for _ in range(self.n_patches):
                # Random crop position
                if H >= self.patch_size and W >= self.patch_size:
                    y = np.random.randint(0, H - self.patch_size + 1)
                    x = np.random.randint(0, W - self.patch_size + 1)
                    
                    patch = img[:, y:y+self.patch_size, x:x+self.patch_size]
                    rms_patch = rms[:, y:y+self.patch_size, x:x+self.patch_size]
                else:
                    # Tile too small: pad to patch_size
                    patch = F.pad(img, (0, max(0, self.patch_size-W), 
                                       0, max(0, self.patch_size-H)))
                    rms_patch = F.pad(rms, (0, max(0, self.patch_size-W), 
                                           0, max(0, self.patch_size-H)), 
                                     value=float('inf'))  # Infinite noise = zero weight
                    
                    # Crop if larger
                    patch = patch[:, :self.patch_size, :self.patch_size]
                    rms_patch = rms_patch[:, :self.patch_size, :self.patch_size]
                
                # Compute inverse variance weights
                # weight = 1 / (rms^2 + eps), but handle NaN/Inf carefully
                eps = 1e-10
                var = rms_patch ** 2
                weight = torch.where(
                    torch.isfinite(var),
                    1.0 / (var + eps),
                    torch.zeros_like(var)
                )
                
                all_patches.append(patch)
                all_weights.append(weight)
        
        patches = torch.stack(all_patches, dim=0)  # (B*n_patches, C, H, W)
        weights = torch.stack(all_weights, dim=0)
        
        return patches, weights


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder for multi-band astronomical images.
    Handles variance-weighted input normalization.
    """
    def __init__(self, 
                 in_channels: int,
                 patch_size: int = 16,
                 embed_dim: int = 384,
                 depth: int = 6,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding: (C, H, W) -> (N, D) where N = (H/p)*(W/p)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                     kernel_size=patch_size, 
                                     stride=patch_size)
        
        # Learnable position embedding
        # Max image size we expect: 128x128 -> 8x8 patches with p=16
        max_patches = (128 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, embed_dim) * 0.02)
        
        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            weights: (B, C, H, W) - inverse variance weights (optional)
        
        Returns:
            features: (B, embed_dim) - CLS token representation
        """
        B = x.shape[0]
        
        # Variance-weighted normalization per band
        if weights is not None:
            # Weighted mean and std per channel
            x_norm = self._weighted_normalize(x, weights)
        else:
            # Standard normalization
            x_norm = (x - x.mean(dim=(2,3), keepdim=True)) / (x.std(dim=(2,3), keepdim=True) + 1e-6)
        
        # Patch embedding
        x = self.patch_embed(x_norm)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # Add positional embedding
        N = x.shape[1]
        x = x + self.pos_embed[:, :N, :]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Return CLS token
        return x[:, 0]  # (B, D)
    
    def _weighted_normalize(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Normalize using inverse-variance weights"""
        # Compute weighted mean per channel
        w_sum = w.sum(dim=(2,3), keepdim=True) + 1e-10
        mean = (x * w).sum(dim=(2,3), keepdim=True) / w_sum
        
        # Compute weighted std
        var = ((x - mean)**2 * w).sum(dim=(2,3), keepdim=True) / w_sum
        std = torch.sqrt(var + 1e-10)
        
        return (x - mean) / std


class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ProjectionHead(nn.Module):
    """Project encoder features to shared embedding space"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)  # L2 normalize


class JAISPFoundation(nn.Module):
    """
    JAISP Foundation Model: Joint-Embedding Predictive Architecture
    for Self-Supervised Pre-training on Rubin + Euclid data
    """
    def __init__(self,
                 patch_size: int = 128,
                 n_patches_per_tile: int = 4,
                 vit_patch_size: int = 16,
                 embed_dim: int = 384,
                 depth: int = 6,
                 num_heads: int = 6,
                 projection_dim: int = 256,
                 temperature: float = 0.07):
        super().__init__()
        
        # Patch extraction
        self.patch_extractor = PatchExtractor(patch_size, n_patches_per_tile)
        
        # Separate encoders for each survey
        self.encoder_rubin = ViTEncoder(
            in_channels=6,  # ugrizy
            patch_size=vit_patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads
        )
        
        self.encoder_euclid = ViTEncoder(
            in_channels=4,  # VIS, Y, J, H
            patch_size=vit_patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads
        )
        
        # Projection heads to shared space
        self.proj_rubin = ProjectionHead(embed_dim, embed_dim, projection_dim)
        self.proj_euclid = ProjectionHead(embed_dim, embed_dim, projection_dim)
        
        self.temperature = temperature
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Dict with keys from JAISP_dataloader
        
        Returns:
            Dict with embeddings and loss
        """
        # Extract patches
        rubin_patches, rubin_weights = self.patch_extractor(
            batch['x_rubin'], batch['rms_rubin']
        )
        euclid_patches, euclid_weights = self.patch_extractor(
            batch['x_euclid'], batch['rms_euclid']
        )
        
        # Encode
        z_rubin_raw = self.encoder_rubin(rubin_patches, rubin_weights)
        z_euclid_raw = self.encoder_euclid(euclid_patches, euclid_weights)
        
        # Project to shared space
        z_rubin = self.proj_rubin(z_rubin_raw)
        z_euclid = self.proj_euclid(z_euclid_raw)
        
        # Compute contrastive loss
        loss = self.contrastive_loss(z_rubin, z_euclid)
        
        return {
            'loss': loss,
            'z_rubin': z_rubin,
            'z_euclid': z_euclid,
            'z_rubin_raw': z_rubin_raw,
            'z_euclid_raw': z_euclid_raw
        }
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE loss: pull together corresponding Rubin-Euclid patch pairs,
        push apart non-corresponding pairs
        """
        B = z1.shape[0]
        
        # Cosine similarity matrix
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature  # (B, B)
        
        # Labels: diagonal is positive pairs
        labels = torch.arange(B, device=z1.device)
        
        # Cross-entropy loss in both directions
        loss_12 = F.cross_entropy(sim_matrix, labels)
        loss_21 = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_12 + loss_21) / 2


# Training utilities
def create_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.05):
    """Create AdamW optimizer with layer-wise lr decay"""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Cosine annealing with warmup"""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])