# jaisp_foundation_v3.py
#
# JAISP Foundation Model v3 - Per-Band Views
#
# Key change from v2: Each band is a separate "view", not stacked multi-channel.
#
# Architecture:
#   1. Per-band stems (handle PSF/noise differences per wavelength)
#   2. Shared ViT trunk (forces band-agnostic representations)
#   3. Token grid output with information weighting
#   4. Multi-view JEPA: any pair of bands can be matched
#
# This learns:
#   - "Same object across wavelengths should have similar representation"
#   - Robustness to missing sources in some bands
#   - Robustness to per-band astrometric offsets
#   - PSF-invariant features

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import math
import copy


# =============================================================================
# INFORMATION MAP
# =============================================================================

class InformationMap(nn.Module):
    """
    Computes weight map for single-channel images.
    High weight = significant signal, low weight = noise/background.
    """
    def __init__(self, min_weight: float = 0.01, snr_threshold: float = 2.0):
        super().__init__()
        self.min_weight = min_weight
        self.snr_threshold = snr_threshold
        
        # Small learnable refinement
        self.refine = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, image: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 1, H, W) - single band
            rms: (B, 1, H, W) - noise map for this band
        
        Returns:
            weights: (B, 1, H, W) normalized weights
        """
        # S/N map
        snr = image.abs() / (rms + 1e-10)
        
        # Gradient magnitude (edges)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-10)
        
        # Normalize both to [0, 1]
        snr_norm = torch.sigmoid((snr - self.snr_threshold) / 1.0)
        grad_norm = grad_mag / (grad_mag.amax(dim=(2, 3), keepdim=True) + 1e-10)
        
        # Combine
        weights = torch.maximum(snr_norm, grad_norm)
        
        # Learnable refinement
        weights = weights * self.refine(weights)
        
        # Clamp and normalize
        weights = weights.clamp(min=self.min_weight, max=1.0)
        weights = weights / (weights.sum(dim=(2, 3), keepdim=True) + 1e-10)
        weights = weights * (image.shape[2] * image.shape[3])  # Scale so mean ≈ 1
        
        return weights


# =============================================================================
# PER-BAND STEMS
# =============================================================================

class BandStem(nn.Module):
    """
    Stem for a single band.
    
    Input: (B, 1, H, W) single-channel image + RMS
    Output: (B, C_out, H, W) features ready for shared trunk
    
    Handles:
    - S/N normalization
    - Band-specific PSF characteristics
    - Noise properties
    """
    def __init__(self, out_channels: int = 64, hidden_channels: int = 32):
        super().__init__()
        
        # Input is 1 channel (single band)
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 5, padding=2),  # Larger kernel for PSF
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.GroupNorm(8, out_channels),
        )
    
    def forward(self, image: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 1, H, W)
            rms: (B, 1, H, W)
        Returns:
            features: (B, C_out, H, W)
        """
        # S/N normalize
        x = image / (rms + 1e-10)
        x = x.clamp(-10, 100)
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-10.0)
        
        return self.net(x)


# =============================================================================
# SHARED TRUNK
# =============================================================================

class PatchEmbedding(nn.Module):
    """Patchify stem features into tokens"""
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)  # (B, D, H/p, W/p)
        grid_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = self.norm(x)
        return x, grid_size


class PositionEmbedding2D(nn.Module):
    """Learnable 2D position embeddings"""
    def __init__(self, embed_dim: int, max_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_size = max_size
        self.pos_embed = nn.Parameter(torch.randn(1, max_size * max_size, embed_dim) * 0.02)
    
    def forward(self, grid_size: Tuple[int, int]) -> torch.Tensor:
        H, W = grid_size
        # Simple: just take first H*W positions
        # For production, should interpolate if size differs from training
        return self.pos_embed[:, :H*W, :]


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
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


class SharedTrunk(nn.Module):
    """
    Shared ViT trunk - processes features from ANY band identically.
    This forces the representation to be band-agnostic.
    """
    def __init__(self,
                 stem_channels: int = 64,
                 embed_dim: int = 256,
                 depth: int = 6,
                 num_heads: int = 8,
                 patch_size: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(stem_channels, embed_dim, patch_size)
        self.pos_embed = PositionEmbedding2D(embed_dim, max_size=64)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        tokens, grid_size = self.patch_embed(x)
        tokens = tokens + self.pos_embed(grid_size)
        
        for block in self.blocks:
            tokens = block(tokens)
        
        return self.norm(tokens), grid_size


# =============================================================================
# PREDICTOR
# =============================================================================

class Predictor(nn.Module):
    """Asymmetric predictor for JEPA"""
    def __init__(self, embed_dim: int, hidden_dim: int = None, depth: int = 2):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 2
        
        layers = []
        for i in range(depth):
            in_dim = embed_dim if i == 0 else hidden_dim
            out_dim = embed_dim if i == depth - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU() if i < depth - 1 else nn.Identity()
            ])
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class TokenWiseJEPALoss(nn.Module):
    """Token-wise JEPA with information weighting"""
    def __init__(self):
        super().__init__()
    
    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weights: torch.Tensor,
                grid_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            pred: (B, N, D) predicted tokens
            target: (B, N, D) target tokens (stop-grad)
            weights: (B, 1, H, W) information weights at image resolution
            grid_size: (H_tok, W_tok)
        """
        B, N, D = pred.shape
        H_t, W_t = grid_size
        
        # Downsample weights to token grid
        w = F.interpolate(weights, size=(H_t, W_t), mode='bilinear', align_corners=False)
        w = w.view(B, N)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-10)
        
        # Cosine similarity
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        sim = (pred_norm * target_norm).sum(dim=-1)  # (B, N)
        
        # Weighted loss
        loss = ((1 - sim) * w).sum(dim=1).mean()
        
        return loss


class VICRegTokenLoss(nn.Module):
    """VICReg regularization for tokens"""
    def __init__(self, var_weight: float = 1.0, cov_weight: float = 0.04):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight
    
    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, D = tokens.shape
        flat = tokens.reshape(-1, D)
        
        # Variance: each dim should have var >= 1
        std = torch.sqrt(flat.var(dim=0) + 1e-4)
        var_loss = F.relu(1.0 - std).mean()
        
        # Covariance: dims should be uncorrelated
        flat_centered = flat - flat.mean(dim=0)
        cov = (flat_centered.T @ flat_centered) / (flat.shape[0] - 1 + 1e-10)
        off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=cov.device)
        cov_loss = cov[off_diag_mask].pow(2).mean()
        
        total = self.var_weight * var_loss + self.cov_weight * cov_loss
        
        return {'var_loss': var_loss, 'cov_loss': cov_loss, 'reg_loss': total}


# =============================================================================
# FULL MODEL
# =============================================================================

class JAISPFoundationV3(nn.Module):
    """
    JAISP Foundation v3 - Per-Band Views
    
    Each band is treated as a separate view with its own stem.
    The shared trunk learns band-agnostic representations.
    
    Training: sample pairs of bands from same sky patch, align their tokens.
    """
    def __init__(self,
                 band_names: List[str],  # e.g., ['rubin_u', 'rubin_g', ..., 'euclid_vis', ...]
                 stem_channels: int = 64,
                 embed_dim: int = 256,
                 trunk_depth: int = 6,
                 num_heads: int = 8,
                 patch_size: int = 16,
                 predictor_depth: int = 2,
                 use_ema: bool = True,
                 ema_decay: float = 0.996,
                 var_weight: float = 1.0,
                 cov_weight: float = 0.04):
        super().__init__()
        
        self.band_names = band_names
        self.embed_dim = embed_dim
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        
        # Per-band stems
        self.stems = nn.ModuleDict({
            name: BandStem(out_channels=stem_channels)
            for name in band_names
        })
        
        # Per-band information maps
        self.info_maps = nn.ModuleDict({
            name: InformationMap()
            for name in band_names
        })
        
        # Shared trunk (online)
        self.trunk = SharedTrunk(
            stem_channels=stem_channels,
            embed_dim=embed_dim,
            depth=trunk_depth,
            num_heads=num_heads,
            patch_size=patch_size
        )
        
        # Target trunk (EMA)
        if use_ema:
            self.target_trunk = copy.deepcopy(self.trunk)
            for p in self.target_trunk.parameters():
                p.requires_grad = False
        else:
            self.target_trunk = self.trunk
        
        # Predictor
        self.predictor = Predictor(embed_dim, embed_dim * 2, predictor_depth)
        
        # Losses
        self.jepa_loss = TokenWiseJEPALoss()
        self.vicreg_loss = VICRegTokenLoss(var_weight, cov_weight)
        
        self._init_weights()
        
        print(f"JAISP Foundation v3 - Per-Band Views")
        print(f"  Bands: {band_names}")
        print(f"  Embed dim: {embed_dim}, Trunk depth: {trunk_depth}")
        print(f"  Patch size: {patch_size}")
        print(f"  EMA: {use_ema} (decay={ema_decay})")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    @torch.no_grad()
    def update_ema(self):
        if not self.use_ema:
            return
        for online_p, target_p in zip(self.trunk.parameters(), self.target_trunk.parameters()):
            target_p.data = self.ema_decay * target_p.data + (1 - self.ema_decay) * online_p.data
    
    def encode_band(self, 
                    image: torch.Tensor, 
                    rms: torch.Tensor, 
                    band: str,
                    return_weights: bool = False,
                    use_target: bool = False) -> Dict[str, torch.Tensor]:
        """
        Encode a single band.
        
        Args:
            image: (B, 1, H, W) - single channel
            rms: (B, 1, H, W)
            band: band name
            return_weights: whether to return info weights
            use_target: use target (EMA) trunk instead of online
        """
        # Info weights
        weights = self.info_maps[band](image, rms)
        
        # Stem
        stem_out = self.stems[band](image, rms)
        
        # Trunk
        trunk = self.target_trunk if use_target else self.trunk
        tokens, grid_size = trunk(stem_out)
        
        result = {'tokens': tokens, 'grid_size': grid_size}
        if return_weights:
            result['weights'] = weights
        
        return result
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Batch should contain:
            - view1_image: (B, 1, H, W)
            - view1_rms: (B, 1, H, W)
            - view1_band: str (band name)
            - view2_image: (B, 1, H, W)
            - view2_rms: (B, 1, H, W)
            - view2_band: str (band name)
        """
        device = next(self.parameters()).device
        
        # Get views
        img1 = batch['view1_image'].to(device)
        rms1 = batch['view1_rms'].to(device)
        band1 = batch['view1_band']
        if isinstance(band1, (list, tuple)):
            band1 = band1[0]
        
        img2 = batch['view2_image'].to(device)
        rms2 = batch['view2_rms'].to(device)
        band2 = batch['view2_band']
        if isinstance(band2, (list, tuple)):
            band2 = band2[0]
        
        # Safety
        img1 = torch.nan_to_num(img1, nan=0.0)
        img2 = torch.nan_to_num(img2, nan=0.0)
        rms1 = torch.nan_to_num(rms1, nan=1.0).clamp(min=1e-10)
        rms2 = torch.nan_to_num(rms2, nan=1.0).clamp(min=1e-10)
        
        # Ensure single channel
        if img1.dim() == 3:
            img1 = img1.unsqueeze(1)
            rms1 = rms1.unsqueeze(1)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(1)
            rms2 = rms2.unsqueeze(1)
        
        # Encode online
        out1 = self.encode_band(img1, rms1, band1, return_weights=True)
        out2 = self.encode_band(img2, rms2, band2, return_weights=True)
        
        tokens1, weights1, grid_size = out1['tokens'], out1['weights'], out1['grid_size']
        tokens2, weights2 = out2['tokens'], out2['weights']
        
        # Encode target (no grad)
        with torch.no_grad():
            target1 = self.encode_band(img1, rms1, band1, use_target=True)['tokens']
            target2 = self.encode_band(img2, rms2, band2, use_target=True)['tokens']
        
        # Predict
        pred1 = self.predictor(tokens1)
        pred2 = self.predictor(tokens2)
        
        # JEPA losses (bidirectional)
        jepa_1to2 = self.jepa_loss(pred1, target2, weights1, grid_size)
        jepa_2to1 = self.jepa_loss(pred2, target1, weights2, grid_size)
        jepa_loss = (jepa_1to2 + jepa_2to1) / 2
        
        # VICReg on tokens
        vicreg1 = self.vicreg_loss(tokens1)
        vicreg2 = self.vicreg_loss(tokens2)
        var_loss = (vicreg1['var_loss'] + vicreg2['var_loss']) / 2
        cov_loss = (vicreg1['cov_loss'] + vicreg2['cov_loss']) / 2
        reg_loss = (vicreg1['reg_loss'] + vicreg2['reg_loss']) / 2
        
        # Total loss
        total_loss = jepa_loss + reg_loss
        
        # Monitoring
        with torch.no_grad():
            t1_norm = F.normalize(tokens1, dim=-1)
            t2_norm = F.normalize(target2, dim=-1)
            token_sim = (t1_norm * t2_norm).sum(dim=-1).mean()
            
            # Weighted global
            def weighted_global(tok, w, gs):
                B, N, D = tok.shape
                w_down = F.interpolate(w, size=gs, mode='bilinear', align_corners=False).view(B, N, 1)
                w_down = w_down / (w_down.sum(dim=1, keepdim=True) + 1e-10)
                return (tok * w_down).sum(dim=1)
            
            g1 = weighted_global(tokens1, weights1, grid_size)
            g2 = weighted_global(tokens2, weights2, grid_size)
            global_sim = F.cosine_similarity(g1, g2, dim=-1).mean()
        
        # EMA update
        self.update_ema()
        
        return {
            'loss': total_loss,
            'jepa_loss': jepa_loss,
            'var_loss': var_loss,
            'cov_loss': cov_loss,
            'reg_loss': reg_loss,
            'token_similarity': token_sim.item(),
            'global_similarity': global_sim.item(),
            'tokens1': tokens1,
            'tokens2': tokens2,
            'weights1': weights1,
            'weights2': weights2,
            'grid_size': grid_size,
            'band1': band1,
            'band2': band2
        }
    
    def get_representation(self, image: torch.Tensor, rms: torch.Tensor, band: str) -> Dict[str, torch.Tensor]:
        """Get representation for inference"""
        if image.dim() == 3:
            image = image.unsqueeze(1)
            rms = rms.unsqueeze(1)
        
        out = self.encode_band(image, rms, band, return_weights=True)
        
        # Weighted global
        B, N, D = out['tokens'].shape
        w = F.interpolate(out['weights'], size=out['grid_size'], mode='bilinear', align_corners=False)
        w = w.view(B, N, 1)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-10)
        global_emb = (out['tokens'] * w).sum(dim=1)
        
        return {
            'tokens': out['tokens'],
            'global': global_emb,
            'grid_size': out['grid_size'],
            'weights': out['weights']
        }


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def create_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.05):
    stem_params = []
    trunk_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'stems' in name or 'info_maps' in name:
            stem_params.append(param)
        elif 'trunk' in name or 'predictor' in name:
            trunk_params.append(param)
        else:
            other_params.append(param)
    
    return torch.optim.AdamW([
        {'params': stem_params, 'lr': lr * 0.5},
        {'params': trunk_params, 'lr': lr},
        {'params': other_params, 'lr': lr}
    ], lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr)
    
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
