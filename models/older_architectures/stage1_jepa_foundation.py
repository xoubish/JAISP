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
# - UPDATED: Signal-based sampling to prioritize galaxies over background

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional


class PatchExtractor(nn.Module):
    """
    Extract small, high-signal patches centered on astronomical sources.
    Zooms in to increase galaxy-to-background ratio and force morphological learning.
    """
    def __init__(self, 
                 patch_size_rubin: int = 48,   # Zoomed from 128
                 patch_size_euclid: int = 96,  # Zoomed from 256 to match 0.1" vs 0.2" scale
                 n_patches_per_tile: int = 4): 
        super().__init__()
        self.patch_size_rubin = patch_size_rubin
        self.patch_size_euclid = patch_size_euclid
        self.n_patches = n_patches_per_tile
    
    def forward(self, img_list: List[torch.Tensor], 
                rms_list: List[torch.Tensor],
                device: Optional[torch.device] = None,
                valid_mask: Optional[torch.Tensor] = None,
                survey: str = 'rubin') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_list: List of (C, H, W) tensors
            rms_list: List of (C, H, W) RMS maps
            device: Device to move tensors to
            survey: 'rubin' or 'euclid' to determine physical patch size
        
        Returns:
            patches: (B*n_patches, C, H, W)
            weights: (B*n_patches, C, H, W) - inverse variance weights
        """
        all_patches = []
        all_weights = []
        p_size = self.patch_size_rubin if survey == 'rubin' else self.patch_size_euclid
        
        for img, rms in zip(img_list, rms_list):
            if device is not None:
                img, rms = img.to(device), rms.to(device)
            C, H, W = img.shape

            # Create a 2D Signal Map (absolute sum across bands) to find galaxies
            signal_map = img.abs().sum(dim=0) 
            
            for _ in range(self.n_patches):
                if H >= p_size and W >= p_size:
                    # Signal-Based Sampling: evaluate multiple random candidates
                    # and pick the one with the highest average flux.
                    n_candidates = 8
                    candidates = []
                    for _ in range(n_candidates):
                        y = np.random.randint(0, H - p_size + 1)
                        x = np.random.randint(0, W - p_size + 1)
                        flux = signal_map[y:y+p_size, x:x+p_size].mean().item()
                        candidates.append((flux, y, x))
                    
                    # Select candidate with max signal
                    _, best_y, best_x = max(candidates, key=lambda x: x[0])
                    
                    patch = img[:, best_y:best_y+p_size, best_x:best_x+p_size]
                    rms_patch = rms[:, best_y:best_y+p_size, best_x:best_x+p_size]
                else:
                    # Padding logic for tiles smaller than the requested patch size
                    patch = F.pad(img, (0, max(0, p_size-W), 0, max(0, p_size-H)))
                    rms_patch = F.pad(rms, (0, max(0, p_size-W), 0, max(0, p_size-H)), value=float('inf'))
                    patch = patch[:, :p_size, :p_size]
                    rms_patch = rms_patch[:, :p_size, :p_size]
                
                # Inverse variance weighting
                eps = 1e-10
                var = rms_patch ** 2
                weight = torch.where(
                    torch.isfinite(var),
                    1.0 / (var + eps),
                    torch.zeros_like(var)
                )
                
                all_patches.append(patch)
                all_weights.append(weight)
        
        return torch.stack(all_patches, dim=0), torch.stack(all_weights, dim=0)


class ViTEncoder(nn.Module):
    """
    Vision Transformer that returns both global and spatial features
    to allow for saliency-weighted pooling.
    """
    def __init__(self, in_channels, patch_size=8, embed_dim=384, depth=6, num_heads=6):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, weights=None):
        B, C, H, W = x.shape
        device = x.device
        
        # 1. Standard ViT processing
        x_norm = self._weighted_normalize(x, weights) if weights is not None else x
        x = self.patch_embed(x_norm).flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        # 2. Extract Spatial Features (ignore CLS token for the mask)
        # These represent local patches of the galaxy
        spatial_features = x[:, 1:] # (B, N_patches, D)
        
        # 3. Create a Saliency Mask from the raw input
        # We look for pixels that are significantly above the noise
        with torch.no_grad():
            # Sum across bands and find high-SNR regions
            signal = x_norm.abs().sum(dim=1) 
            # Downsample signal map to match ViT patch grid (e.g., 48x48 -> 6x6)
            saliency = F.avg_pool2d(signal.unsqueeze(1), self.patch_size).flatten(2).transpose(1, 2)
            # Soft-thresholding: highlight peaks, suppress background
            saliency = F.softmax(saliency / 0.1, dim=1) 
            
        # 4. Weighted Global Pooling
        # Instead of just taking the CLS token, we weight the patches by their signal
        weighted_embedding = (spatial_features * saliency).sum(dim=1)
        
        return weighted_embedding

    def _weighted_normalize(self, x, w):
        w_sum = w.sum(dim=(2,3), keepdim=True).clamp(min=1e-10)
        mean = (x * w).sum(dim=(2,3), keepdim=True) / w_sum
        var = ((x - mean)**2 * w).sum(dim=(2,3), keepdim=True) / w_sum
        std = torch.sqrt(var.clamp(min=1e-10))
        return (x - mean) / std * torch.sqrt(w / (w.mean(dim=(2,3), keepdim=True) + 1e-10))

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
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ProjectionHead(nn.Module):
    """Project encoder features to L2-normalized shared embedding space"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)


class JAISPFoundation(nn.Module):
    """
    JAISP Foundation Model: Joint-Embedding Predictive Architecture
    Optimized for high-signal zoomed patches.
    """
    def __init__(self,
                 n_patches_per_tile: int = 4,
                 vit_patch_size: int = 8,
                 embed_dim: int = 384,
                 depth: int = 6,
                 num_heads: int = 6,
                 projection_dim: int = 256,
                 temperature: float = 0.05, # Lowered for sharper contrast
                 use_band_masking: bool = False):
        super().__init__()
        
        self.use_band_masking = use_band_masking
        
        self.patch_extractor = PatchExtractor(
            patch_size_rubin=48,  
            patch_size_euclid=96, 
            n_patches_per_tile=n_patches_per_tile
        )
        
        self.encoder_rubin = ViTEncoder(
            in_channels=6, 
            patch_size=vit_patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads
        )
        
        self.encoder_euclid = ViTEncoder(
            in_channels=4, 
            patch_size=vit_patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads
        )
        
        self.proj_rubin = ProjectionHead(embed_dim, embed_dim, projection_dim)
        self.proj_euclid = ProjectionHead(embed_dim, embed_dim, projection_dim)
        
        self.temperature = temperature
        self._init_weights()
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        
        rubin_patches, rubin_weights = self.patch_extractor(
            batch['x_rubin'], batch['rms_rubin'], device=device, survey='rubin'
        )
        euclid_patches, euclid_weights = self.patch_extractor(
            batch['x_euclid'], batch['rms_euclid'], device=device, survey='euclid'
        )
        
        z_rubin_raw = self.encoder_rubin(rubin_patches, rubin_weights)
        z_euclid_raw = self.encoder_euclid(euclid_patches, euclid_weights)
        
        z_rubin = self.proj_rubin(z_rubin_raw)
        z_euclid = self.proj_euclid(z_euclid_raw)
        
        loss = self.contrastive_loss(z_rubin, z_euclid)
        
        return {
            'loss': loss,
            'z_rubin': z_rubin,
            'z_euclid': z_euclid,
            'z_rubin_raw': z_rubin_raw,
            'z_euclid_raw': z_euclid_raw
        }
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.shape[0]
        device = z1.device
        
        z = torch.cat([z1, z2], dim=0)  
        sim = torch.matmul(z, z.T) / self.temperature  
        
        sim = sim - torch.eye(2*B, device=device) * 1e9
        
        pos_indices = torch.arange(2*B, device=device)
        pos_indices[:B] = pos_indices[:B] + B
        pos_indices[B:] = pos_indices[B:] - B
        
        pos_sim = sim[torch.arange(2*B, device=device), pos_indices]
        loss = -pos_sim + torch.logsumexp(sim, dim=1)
        
        return loss.mean()
    
    def _init_weights(self):
        """Initialize weights with low variance for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def create_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.05):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])