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
    Handles different pixel scales: matches physical sky area, not pixel count.
    """
    def __init__(self, 
                 patch_size_rubin: int = 128,
                 patch_size_euclid: int = 256,  
                 n_patches_per_tile: int = 4): 
        super().__init__()
        self.patch_size_rubin = patch_size_rubin
        self.patch_size_euclid = patch_size_euclid
        self.n_patches = n_patches_per_tile
    
    def forward(self, img_list: List[torch.Tensor], 
                rms_list: List[torch.Tensor],
                device: Optional[torch.device] = None,
                valid_mask: Optional[torch.Tensor] = None,
                survey: str = 'rubin') -> Tuple[torch.Tensor, torch.Tensor]: # Fixed: Added survey keyword
        """
        Args:
            img_list: List of (C, H, W) tensors
            rms_list: List of (C, H, W) RMS maps
            device: Device to move tensors to
            valid_mask: Optional (B, C) mask indicating valid bands
            survey: 'rubin' or 'euclid' to determine physical patch size
        
        Returns:
            patches: (B*n_patches, C, patch_size, patch_size)
            weights: (B*n_patches, C, patch_size, patch_size) - inverse variance weights
        """
        all_patches = []
        all_weights = []
        
        # Select patch size based on survey to maintain constant sky area
        p_size = self.patch_size_rubin if survey == 'rubin' else self.patch_size_euclid
        
        for img, rms in zip(img_list, rms_list):
            # Move to device if specified
            if device is not None:
                img = img.to(device)
                rms = rms.to(device)
            C, H, W = img.shape
            
            # Extract n_patches from this tile
            for _ in range(self.n_patches):
                # Random crop position
                if H >= p_size and W >= p_size:
                    y = np.random.randint(0, H - p_size + 1)
                    x = np.random.randint(0, W - p_size + 1)
                    
                    patch = img[:, y:y+p_size, x:x+p_size]
                    rms_patch = rms[:, y:y+p_size, x:x+p_size]
                else:
                    # Tile too small: pad to patch_size
                    patch = F.pad(img, (0, max(0, p_size-W), 
                                       0, max(0, p_size-H)))
                    rms_patch = F.pad(rms, (0, max(0, p_size-W), 
                                           0, max(0, p_size-H)), 
                                     value=float('inf'))  # Infinite noise = zero weight
                    
                    # Crop if larger
                    patch = patch[:, :p_size, :p_size]
                    rms_patch = rms_patch[:, :p_size, :p_size]
                
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
    Handles variance-weighted input normalization and band masking.
    """
    def __init__(self, 
                 in_channels: int,
                 patch_size: int = 16,
                 embed_dim: int = 384,
                 depth: int = 6,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 band_dropout: float = 0.0):  # randomly drop bands during training
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.band_dropout = band_dropout
        
        # Patch embedding: (C, H, W) -> (N, D) where N = (H/p)*(W/p)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                     kernel_size=patch_size, 
                                     stride=patch_size)
        
        # Learnable position embedding (dynamic size)
        # Max patches we expect: 256/16 = 16, so 16*16 = 256 max patches for Euclid
        max_patches = 256  # Conservative upper bound
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, embed_dim) * 0.02)
        
        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None,
                band_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            weights: (B, C, H, W) - inverse variance weights (optional)
            band_mask: (B, C) - binary mask for valid bands (1=use, 0=ignore)
        
        Returns:
            features: (B, embed_dim) - CLS token representation
        """
        B = x.shape[0]
        
        # Ensure x and weights are on the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)
        if weights is not None:
            weights = weights.to(device)
        if band_mask is not None:
            band_mask = band_mask.to(device)
        
        # Apply band mask (zero out bad bands)
        if band_mask is not None:
            # Expand mask: (B, C) -> (B, C, 1, 1)
            mask_expanded = band_mask.view(B, -1, 1, 1)
            x = x * mask_expanded
            if weights is not None:
                weights = weights * mask_expanded
        
        # Apply band dropout during training (robustness)
        if self.training and self.band_dropout > 0:
            drop_mask = torch.bernoulli(
                torch.full((B, self.in_channels, 1, 1), 1 - self.band_dropout, device=device)
            )
            x = x * drop_mask
            if weights is not None:
                weights = weights * drop_mask
        
        # Variance-weighted normalization per band
        if weights is not None:
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
        """
        Normalize using inverse-variance weights.
        High variance (low weight) regions contribute less to normalization.
        """
        # Compute weighted mean per channel
        w_sum = w.sum(dim=(2,3), keepdim=True).clamp(min=1e-10)
        mean = (x * w).sum(dim=(2,3), keepdim=True) / w_sum
        
        # Compute weighted std
        var = ((x - mean)**2 * w).sum(dim=(2,3), keepdim=True) / w_sum
        std = torch.sqrt(var.clamp(min=1e-10))
        
        # Normalize
        x_norm = (x - mean) / std
        
        # Re-apply weights: down-weight noisy regions in the normalized space
        # This is KEY: noisy pixels contribute less to the final embedding
        x_norm = x_norm * torch.sqrt(w / (w.mean(dim=(2,3), keepdim=True) + 1e-10))
        
        return x_norm


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
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Do NOT normalize - let the loss handle similarity scaling
        return self.net(x)


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
                 temperature: float = 0.07,
                 use_band_masking: bool = False):
        super().__init__()
        
        self.use_band_masking = use_band_masking
        
        # Patch extraction
        self.patch_extractor = PatchExtractor(
            patch_size_rubin=128,   # 128 × 0.2" = 25.6" on sky
            patch_size_euclid=256,  # 256 × 0.1" = 25.6" on sky (MATCHED!)
            n_patches_per_tile=n_patches_per_tile # Fixed: Keyword name matches PatchExtractor.__init__
        )
        
        # Separate encoders for each survey
        self.encoder_rubin = ViTEncoder(
            in_channels=6,  # ugrizy
            patch_size=vit_patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            band_dropout=0.1  # Randomly drop 10% of bands for robustness
        )
        
        self.encoder_euclid = ViTEncoder(
            in_channels=4,  # VIS, Y, J, H
            patch_size=vit_patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            band_dropout=0.1
        )
        
        # Projection heads to shared space
        self.proj_rubin = ProjectionHead(embed_dim, embed_dim, projection_dim)
        self.proj_euclid = ProjectionHead(embed_dim, embed_dim, projection_dim)
        
        self.temperature = temperature
        
        # Initialize projection heads with smaller weights for stability
        self._init_weights()
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Dict with keys from JAISP_dataloader
        
        Returns:
            Dict with embeddings and loss
        """
        # Get model device
        device = next(self.parameters()).device
        
        # Extract patches (PatchExtractor now accepts 'survey' to select correct size)
        rubin_patches, rubin_weights = self.patch_extractor(
            batch['x_rubin'], batch['rms_rubin'], device=device, survey='rubin'
        )
        euclid_patches, euclid_weights = self.patch_extractor(
            batch['x_euclid'], batch['rms_euclid'], device=device, survey='euclid'
        )
        
        # Optional: Create band quality masks (only if catastrophic failures)
        if self.use_band_masking:
            rubin_band_mask = self._create_rubin_band_mask(rubin_patches, rubin_weights)
            euclid_band_mask = batch['mask_euclid'].to(device)
            
            # Expand euclid_band_mask to match number of patches
            n_patches = rubin_patches.shape[0]
            batch_size = len(batch['x_rubin'])
            patches_per_tile = n_patches // batch_size
            euclid_band_mask = euclid_band_mask.repeat_interleave(patches_per_tile, dim=0)
        else:
            # Trust the variance weighting completely
            rubin_band_mask = None
            euclid_band_mask = None
        
        # Encode with variance weighting (and optional band masks)
        z_rubin_raw = self.encoder_rubin(rubin_patches, rubin_weights, rubin_band_mask)
        z_euclid_raw = self.encoder_euclid(euclid_patches, euclid_weights, euclid_band_mask)
        
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
        NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
        Normalize embeddings for cosine similarity, but don't force unit norm in forward pass
        """
        B = z1.shape[0]
        device = z1.device
        
        # Normalize for cosine similarity
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # Concatenate both views
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        
        # Compute full similarity matrix with temperature
        sim = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)
        
        # Mask out self-similarity (diagonal)
        sim = sim - torch.eye(2*B, device=device) * 1e9
        
        # Build target indices: Rubin patch i matches Euclid patch i
        pos_indices = torch.arange(2*B, device=device)
        pos_indices[:B] = pos_indices[:B] + B
        pos_indices[B:] = pos_indices[B:] - B
        
        # Get positive similarities
        pos_sim = sim[torch.arange(2*B, device=device), pos_indices]
        
        # Negative log likelihood (log(exp(pos) / sum(exp(all))))
        loss = -pos_sim + torch.logsumexp(sim, dim=1)
        
        return loss.mean()
    
    def _init_weights(self):
        """Initialize projection heads with smaller weights"""
        for module in [self.proj_rubin, self.proj_euclid]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def _create_rubin_band_mask(self, patches: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Create band quality mask based on effective signal content.
        Uses variance weighting to determine if a band has usable information.
        """
        B, C = patches.shape[:2]
        # Compute effective SNR: weighted signal
        effective_weight = weights.mean(dim=(2, 3))  # (B, C)
        signal = patches.abs().mean(dim=(2, 3))      # (B, C)
        
        # A band is "bad" if weights are too low or signal is essentially zero
        weight_threshold = 0.01 
        signal_threshold = 1e-6
        
        mask = ((effective_weight > weight_threshold) & (signal > signal_threshold)).float()
        
        # Ensure at least 3 bands are active
        n_active = mask.sum(dim=1, keepdim=True)
        if (n_active < 3).any():
            _, top_indices = effective_weight.topk(min(3, C), dim=1)
            mask = torch.zeros_like(mask)
            mask.scatter_(1, top_indices, 1.0)
        
        return mask


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