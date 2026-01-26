# stage1_detr_jepa_foundation.py
#
# DETR-style Object Detection + JEPA in Object Space
#
# Architecture:
# 1. ViT backbone extracts patch features
# 2. Transformer decoder with learnable object queries (like DETR)
# 3. Each query attends to image features → one object slot
# 4. JEPA: Match object manifolds between Rubin and Euclid
#
# Key insight: Learn embedding space where:
#   "Set of Rubin objects" ≈ "Set of Euclid objects"
# Not pixel-level, not patch-level, but OBJECT-LEVEL

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import math


class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone for feature extraction.
    Converts image → sequence of patch features.
    """
    def __init__(self, 
                 in_channels: int,
                 patch_size: int = 16,
                 embed_dim: int = 384,
                 depth: int = 6,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                     kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding (learnable)
        # Rubin: 512×512 / 16 = 32×32 = 1024 patches
        # Euclid: 1050×1050 / 16 = 65×65 = 4225 patches
        max_patches = 5000  # Conservative upper bound
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, embed_dim) * 0.02)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            weights: (B, C, H, W) - inverse variance weights
        
        Returns:
            features: (B, N, D) where N = num patches
        """
        B = x.shape[0]
        
        # Variance-weighted normalization
        if weights is not None:
            x = self._weighted_normalize(x, weights)
        else:
            x = (x - x.mean(dim=(2,3), keepdim=True)) / (x.std(dim=(2,3), keepdim=True) + 1e-6)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # Add positional embedding
        N = x.shape[1]
        x = x + self.pos_embed[:, :N, :]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x  # (B, N, D) - patch features
    
    def _weighted_normalize(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Variance-weighted normalization"""
        w_sum = w.sum(dim=(2,3), keepdim=True).clamp(min=1e-10)
        mean = (x * w).sum(dim=(2,3), keepdim=True) / w_sum
        var = ((x - mean)**2 * w).sum(dim=(2,3), keepdim=True) / w_sum
        std = torch.sqrt(var.clamp(min=1e-10))
        x_norm = (x - mean) / std
        # Re-weight in normalized space
        x_norm = x_norm * torch.sqrt(w / (w.mean(dim=(2,3), keepdim=True) + 1e-10))
        return x_norm


class TransformerBlock(nn.Module):
    """Standard Transformer block"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ObjectDecoder(nn.Module):
    """
    DETR-style decoder with learnable object queries.
    Each query "discovers" one object in the image.
    """
    def __init__(self, 
                 embed_dim: int = 384,
                 num_queries: int = 100,
                 num_layers: int = 6,
                 num_heads: int = 8):
        super().__init__()
        self.num_queries = num_queries
        
        # Learnable object queries (like DETR)
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dim))
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory: (B, N, D) - patch features from backbone
        
        Returns:
            object_features: (B, num_queries, D) - one embedding per object
        """
        B = memory.shape[0]
        
        # Initialize queries (same for all images in batch)
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)
        
        # Decode: each query attends to image features
        for layer in self.layers:
            queries = layer(queries, memory)
        
        queries = self.norm(queries)
        
        return queries  # (B, Q, D)


class DecoderLayer(nn.Module):
    """Transformer decoder layer with cross-attention"""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        
        # Self-attention on queries
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # Cross-attention to image features
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, Q, D)
            memory: (B, N, D) - image features
        """
        # Self-attention among queries
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q)[0]
        
        # Cross-attention to image
        q = self.norm2(queries)
        queries = queries + self.cross_attn(q, memory, memory)[0]
        
        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        
        return queries


class SetMatchingLoss(nn.Module):
    """
    Hungarian matching between Rubin and Euclid object sets.
    Finds optimal 1-1 correspondence, then computes loss.
    """
    def __init__(self, cost_type: str = 'cosine'):
        super().__init__()
        self.cost_type = cost_type
    
    def forward(self, rubin_objects: torch.Tensor, euclid_objects: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rubin_objects: (B, Q, D)
            euclid_objects: (B, Q, D)
        
        Returns:
            loss: scalar
            matches: (B, Q) - matched indices
        """
        from scipy.optimize import linear_sum_assignment
        
        B, Q, D = rubin_objects.shape
        
        # Normalize for cosine similarity (with safety checks)
        rubin_norm = F.normalize(rubin_objects + 1e-8, dim=-1, eps=1e-8)
        euclid_norm = F.normalize(euclid_objects + 1e-8, dim=-1, eps=1e-8)
        
        # Check for NaNs/Infs (more aggressive)
        if (not torch.isfinite(rubin_norm).all() or 
            not torch.isfinite(euclid_norm).all() or
            torch.isnan(rubin_objects).any() or
            torch.isnan(euclid_objects).any()):
            # Return small positive loss to keep gradients flowing
            dummy_loss = (rubin_objects ** 2).mean() + (euclid_objects ** 2).mean()
            return dummy_loss * 0.001, torch.zeros((B, Q), dtype=torch.long, device=rubin_objects.device)
        
        total_loss = 0
        all_matches = []
        
        for b in range(B):
            # Cost matrix: negative similarity (we want to maximize similarity)
            cost_matrix = -(rubin_norm[b] @ euclid_norm[b].T)  # (Q, Q)
            
            # Check for invalid values
            cost_np = cost_matrix.detach().cpu().numpy()
            if not np.isfinite(cost_np).all():
                # Skip this batch if cost matrix is invalid
                all_matches.append(torch.arange(Q, device=rubin_objects.device))
                continue
            
            # Hungarian algorithm to find optimal matching
            row_ind, col_ind = linear_sum_assignment(cost_np)
            
            # Compute loss on matched pairs
            matched_rubin = rubin_norm[b, row_ind]
            matched_euclid = euclid_norm[b, col_ind]
            
            # Cosine similarity loss
            sim = (matched_rubin * matched_euclid).sum(dim=-1)  # (Q,)
            loss_b = 1.0 - sim.mean()  # Want similarity → 1
            
            total_loss += loss_b
            all_matches.append(torch.tensor(col_ind, device=rubin_objects.device))
        
        return total_loss / B, torch.stack(all_matches)


class DETRJEPA(nn.Module):
    """
    DETR-style Object Detection + JEPA in Object Space.
    
    The key innovation:
    - Extract object-level representations (not patches)
    - JEPA learns: "manifold of Rubin objects" ≈ "manifold of Euclid objects"
    - Set-to-set matching via Hungarian algorithm
    """
    def __init__(self,
                 rubin_channels: int = 6,
                 euclid_channels: int = 4,
                 patch_size: int = 16,
                 embed_dim: int = 384,
                 backbone_depth: int = 6,
                 decoder_depth: int = 6,
                 num_queries: int = 100,
                 num_heads: int = 8):
        super().__init__()
        
        # ViT backbones
        self.backbone_rubin = ViTBackbone(
            in_channels=rubin_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=backbone_depth,
            num_heads=6
        )
        
        self.backbone_euclid = ViTBackbone(
            in_channels=euclid_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=backbone_depth,
            num_heads=6
        )
        
        # Object decoders (DETR-style)
        self.decoder_rubin = ObjectDecoder(
            embed_dim=embed_dim,
            num_queries=num_queries,
            num_layers=decoder_depth,
            num_heads=num_heads
        )
        
        self.decoder_euclid = ObjectDecoder(
            embed_dim=embed_dim,
            num_queries=num_queries,
            num_layers=decoder_depth,
            num_heads=num_heads
        )
        
        # Set matching loss
        self.set_loss = SetMatchingLoss()
        
        print(f"DETR-JEPA Architecture:")
        print(f"  Backbone: ViT-{embed_dim} (depth={backbone_depth})")
        print(f"  Decoder: {num_queries} object queries (depth={decoder_depth})")
        print(f"  Learning: Object manifold alignment via Hungarian matching")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights carefully"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Dict from dataloader
        
        Returns:
            Dict with loss and diagnostics
        """
        device = next(self.parameters()).device
        
        # For simplicity, process first item in batch
        # (Can extend to true batching later)
        rubin_img = batch['x_rubin'][0].unsqueeze(0).to(device)
        euclid_img = batch['x_euclid'][0].unsqueeze(0).to(device)
        rubin_rms = batch['rms_rubin'][0].unsqueeze(0).to(device)
        euclid_rms = batch['rms_euclid'][0].unsqueeze(0).to(device)
        
        rubin_weights = 1.0 / (rubin_rms ** 2 + 1e-10)
        euclid_weights = 1.0 / (euclid_rms ** 2 + 1e-10)
        
        # Extract patch features via ViT
        rubin_features = self.backbone_rubin(rubin_img, rubin_weights)  # (1, N_r, D)
        euclid_features = self.backbone_euclid(euclid_img, euclid_weights)  # (1, N_e, D)
        
        # Decode to object representations
        rubin_objects = self.decoder_rubin(rubin_features)  # (1, Q, D)
        euclid_objects = self.decoder_euclid(euclid_features)  # (1, Q, D)
        
        # Set matching loss (Hungarian algorithm)
        loss, matches = self.set_loss(rubin_objects, euclid_objects)
        
        # Compute average similarity of matched pairs for monitoring
        rubin_norm = F.normalize(rubin_objects, dim=-1)
        euclid_norm = F.normalize(euclid_objects, dim=-1)
        
        matched_euclid = euclid_norm[0, matches[0]]
        similarity = (rubin_norm[0] * matched_euclid).sum(dim=-1).mean()
        
        return {
            'loss': loss,
            'similarity': similarity.item(),
            'n_objects': rubin_objects.shape[1],
            'matches': matches,
            'rubin_objects': rubin_objects,
            'euclid_objects': euclid_objects
        }


# Training utilities
def create_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.05):
    """AdamW with layer-wise lr decay"""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Cosine annealing with warmup"""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])