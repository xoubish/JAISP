# jaisp_foundation_v4.py
#
# JAISP Foundation v4 - Simplified Direct Approach
#
# Key changes from v3:
# 1. Simpler loss: directly maximize similarity between corresponding tokens
# 2. Add contrastive component: non-corresponding tokens should be dissimilar  
# 3. Shared projection head after stems (like SimCLR/BYOL)
# 4. Remove complex EMA - use simpler siamese with stop-grad
# 5. Monitor the right metrics
#
# The v3 approach failed because:
# - JEPA with predictor wasn't creating useful gradients
# - Different stems → different representation spaces
# - Nothing forced cross-band alignment

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
import copy


# =============================================================================
# INFORMATION MAP (simplified, no learnable parts)
# =============================================================================

class InformationMap(nn.Module):
    """Pure signal-based weighting - high S/N or gradient = high weight"""
    def __init__(self, snr_threshold: float = 2.0, min_weight: float = 0.001):
        super().__init__()
        self.snr_threshold = snr_threshold
        self.min_weight = min_weight
    
    def forward(self, image: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
        # S/N
        snr = image.abs() / (rms + 1e-10)
        snr_weight = torch.sigmoid((snr - self.snr_threshold) * 2.0)
        
        # Gradient magnitude
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        gx = F.conv2d(image, sobel_x, padding=1)
        gy = F.conv2d(image, sobel_y, padding=1)
        grad = torch.sqrt(gx**2 + gy**2 + 1e-10)
        grad_weight = grad / (grad.amax(dim=(2,3), keepdim=True) + 1e-10)
        
        # Combine
        weights = torch.maximum(snr_weight, grad_weight * 0.5)
        weights = weights ** 2  # Increase contrast
        weights = weights.clamp(min=self.min_weight)
        
        # Normalize
        weights = weights / (weights.sum(dim=(2,3), keepdim=True) + 1e-10)
        weights = weights * (image.shape[2] * image.shape[3])
        
        return weights


# =============================================================================
# ENCODER COMPONENTS
# =============================================================================

class BandStem(nn.Module):
    """Per-band stem: normalizes by RMS and extracts initial features"""
    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
    
    def forward(self, image: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
        x = image / (rms + 1e-10)
        x = x.clamp(-10, 100)
        return self.net(x)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size
    
    def forward(self, x):
        x = self.proj(x)
        gs = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), gs


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SharedEncoder(nn.Module):
    """Shared transformer encoder"""
    def __init__(self, stem_ch: int = 64, embed_dim: int = 256, depth: int = 6, patch_size: int = 16):
        super().__init__()
        self.patch_embed = PatchEmbed(stem_ch, embed_dim, patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        tokens, gs = self.patch_embed(x)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1]]
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens), gs


class ProjectionHead(nn.Module):
    """Projects tokens to embedding space (like SimCLR)"""
    def __init__(self, in_dim: int, out_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        # x: (B, N, D) -> process each token
        B, N, D = x.shape
        x = x.reshape(B * N, D)
        x = self.net(x)
        return x.reshape(B, N, -1)


# =============================================================================
# LOSS: Simple but effective
# =============================================================================

class AlignmentLoss(nn.Module):
    """
    Direct token alignment loss with support for different grid sizes.
    
    When views have different resolutions (e.g., Rubin 512 vs Euclid 1050),
    their token grids will differ (32×32 vs 65×65). We interpolate both
    to a common grid size before computing the loss.
    
    This works because both cover the SAME sky area.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def _interpolate_tokens(self, z: torch.Tensor, grid_size: Tuple[int, int], 
                            target_size: Tuple[int, int]) -> torch.Tensor:
        """Interpolate token grid to target size"""
        if grid_size == target_size:
            return z
        
        B, N, D = z.shape
        H, W = grid_size
        
        # Reshape to spatial: (B, D, H, W)
        z_spatial = z.transpose(1, 2).view(B, D, H, W)
        
        # Interpolate
        z_interp = F.interpolate(z_spatial, size=target_size, mode='bilinear', align_corners=False)
        
        # Reshape back: (B, N_new, D)
        return z_interp.view(B, D, -1).transpose(1, 2)
    
    def forward(self, 
                z1: torch.Tensor, 
                z2: torch.Tensor,
                weights1: torch.Tensor,
                weights2: torch.Tensor,
                grid_size1: Tuple[int, int],
                grid_size2: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """
        Args:
            z1: (B, N1, D) - projected tokens from view 1
            z2: (B, N2, D) - projected tokens from view 2 (may differ from N1!)
            weights1: (B, 1, H1, W1) - information weights view 1
            weights2: (B, 1, H2, W2) - information weights view 2
            grid_size1: (H_tok1, W_tok1)
            grid_size2: (H_tok2, W_tok2)
        """
        B, _, D = z1.shape
        
        # Use smaller grid as target (more conservative, avoids upsampling artifacts)
        target_size = (min(grid_size1[0], grid_size2[0]), 
                       min(grid_size1[1], grid_size2[1]))
        
        # Interpolate tokens to common grid
        z1_common = self._interpolate_tokens(z1, grid_size1, target_size)
        z2_common = self._interpolate_tokens(z2, grid_size2, target_size)
        
        # Interpolate weights to common grid
        w1_common = F.interpolate(weights1, size=target_size, mode='bilinear', align_corners=False)
        w2_common = F.interpolate(weights2, size=target_size, mode='bilinear', align_corners=False)
        
        # Average weights (both should highlight same sky regions)
        w_avg = (w1_common + w2_common) / 2
        w_avg = w_avg.view(B, -1)
        w_avg = w_avg / (w_avg.sum(dim=1, keepdim=True) + 1e-10)
        
        # Normalize embeddings
        z1_norm = F.normalize(z1_common, dim=-1)
        z2_norm = F.normalize(z2_common, dim=-1)
        
        # Positive pairs: corresponding tokens (same sky location)
        pos_sim = (z1_norm * z2_norm).sum(dim=-1)  # (B, N_common)
        
        # Weighted alignment loss
        align_loss = ((1 - pos_sim) * w_avg).sum(dim=1).mean()
        
        # Uniformity regularization
        N = z1_norm.shape[1]
        n_sample = min(256, N)
        idx = torch.randperm(N, device=z1.device)[:n_sample]
        
        z1_sample = z1_norm[:, idx]
        sim1 = torch.bmm(z1_sample, z1_sample.transpose(1, 2))
        mask = ~torch.eye(n_sample, dtype=torch.bool, device=z1.device)
        uniform_loss = sim1[:, mask].view(B, -1).mean()
        
        total = align_loss + 0.1 * F.relu(uniform_loss - 0.1)
        
        return {
            'loss': total,
            'align_loss': align_loss,
            'uniform_loss': uniform_loss,
            'pos_similarity': pos_sim.mean(),
            'common_grid_size': target_size
        }


class VICReg(nn.Module):
    """Variance-Invariance-Covariance regularization"""
    def __init__(self, var_weight: float = 1.0, cov_weight: float = 0.04):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, D = z.shape
        z_flat = z.reshape(-1, D)
        
        # Variance
        std = torch.sqrt(z_flat.var(dim=0) + 1e-4)
        var_loss = F.relu(1.0 - std).mean()
        
        # Covariance
        z_centered = z_flat - z_flat.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (z_flat.shape[0] - 1)
        off_diag = cov.flatten()[:-1].view(D-1, D+1)[:, 1:].flatten()
        cov_loss = off_diag.pow(2).mean()
        
        return {
            'var_loss': var_loss,
            'cov_loss': cov_loss,
            'reg_loss': self.var_weight * var_loss + self.cov_weight * cov_loss
        }


# =============================================================================
# FULL MODEL
# =============================================================================

class JAISPFoundationV4(nn.Module):
    """
    JAISP Foundation v4 - Simplified Direct Alignment
    
    Key insight: Use a SHARED projection head after the encoder.
    This forces all bands into the SAME embedding space.
    
    Architecture:
        Image → Band Stem → Shared Encoder → Shared Projection → Embeddings
        
    Loss: Direct token alignment + VICReg
    """
    def __init__(self,
                 band_names: List[str],
                 stem_ch: int = 64,
                 embed_dim: int = 256,
                 proj_dim: int = 256,
                 depth: int = 6,
                 patch_size: int = 16,
                 temperature: float = 0.1):
        super().__init__()
        
        self.band_names = band_names
        self.embed_dim = embed_dim
        
        # Per-band stems
        self.stems = nn.ModuleDict({
            name: BandStem(stem_ch) for name in band_names
        })
        
        # Per-band info maps
        self.info_maps = nn.ModuleDict({
            name: InformationMap() for name in band_names
        })
        
        # SHARED encoder (same weights for all bands)
        self.encoder = SharedEncoder(stem_ch, embed_dim, depth, patch_size)
        
        # SHARED projection head (forces all bands to same space)
        self.projector = ProjectionHead(embed_dim, proj_dim)
        
        # Losses
        self.align_loss = AlignmentLoss(temperature)
        self.vicreg = VICReg()
        
        self._init_weights()
        print(f"JAISP Foundation v4 - Direct Alignment")
        print(f"  Bands: {len(band_names)}")
        print(f"  Embed dim: {embed_dim}, Proj dim: {proj_dim}")
        print(f"  Encoder depth: {depth}")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, image: torch.Tensor, rms: torch.Tensor, band: str) -> Dict:
        """Encode a single band to projected tokens"""
        # Get weights
        weights = self.info_maps[band](image, rms)
        
        # Stem
        feat = self.stems[band](image, rms)
        
        # Shared encoder
        tokens, grid_size = self.encoder(feat)
        
        # Shared projection
        z = self.projector(tokens)
        
        return {
            'tokens': tokens,  # Pre-projection (for monitoring)
            'z': z,            # Post-projection (for loss)
            'weights': weights,
            'grid_size': grid_size
        }
    
    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass handling variable-sized images.
        
        batch contains lists (not stacked tensors) when images have different sizes.
        We process each sample individually then aggregate.
        """
        device = next(self.parameters()).device
        
        # Check if batch contains lists (variable size) or tensors (same size)
        is_variable = isinstance(batch['view1_image'], list)
        
        if is_variable:
            return self._forward_variable(batch, device)
        else:
            return self._forward_fixed(batch, device)
    
    def _forward_fixed(self, batch: Dict, device) -> Dict:
        """Original forward for same-sized images"""
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
        
        # Clean
        img1 = torch.nan_to_num(img1, nan=0.0)
        img2 = torch.nan_to_num(img2, nan=0.0)
        rms1 = torch.nan_to_num(rms1, nan=1.0).clamp(min=1e-10)
        rms2 = torch.nan_to_num(rms2, nan=1.0).clamp(min=1e-10)
        
        if img1.dim() == 3:
            img1, rms1 = img1.unsqueeze(1), rms1.unsqueeze(1)
        if img2.dim() == 3:
            img2, rms2 = img2.unsqueeze(1), rms2.unsqueeze(1)
        
        # Encode
        out1 = self.encode(img1, rms1, band1)
        out2 = self.encode(img2, rms2, band2)
        
        z1, z2 = out1['z'], out2['z']
        w1, w2 = out1['weights'], out2['weights']
        gs1, gs2 = out1['grid_size'], out2['grid_size']
        
        # Loss with grid size handling
        align1 = self.align_loss(z1, z2.detach(), w1, w2, gs1, gs2)
        align2 = self.align_loss(z2, z1.detach(), w2, w1, gs2, gs1)
        
        align_loss = (align1['loss'] + align2['loss']) / 2
        pos_sim = (align1['pos_similarity'] + align2['pos_similarity']) / 2
        
        # VICReg
        vic1 = self.vicreg(z1)
        vic2 = self.vicreg(z2)
        var_loss = (vic1['var_loss'] + vic2['var_loss']) / 2
        cov_loss = (vic1['cov_loss'] + vic2['cov_loss']) / 2
        reg_loss = (vic1['reg_loss'] + vic2['reg_loss']) / 2
        
        total_loss = align_loss + reg_loss
        
        # Monitoring
        with torch.no_grad():
            z1_norm = F.normalize(z1, dim=-1)
            z2_norm = F.normalize(z2, dim=-1)
            
            # Need to interpolate for comparison if grids differ
            if gs1 != gs2:
                target = (min(gs1[0], gs2[0]), min(gs1[1], gs2[1]))
                z1_interp = self.align_loss._interpolate_tokens(z1_norm, gs1, target)
                z2_interp = self.align_loss._interpolate_tokens(z2_norm, gs2, target)
            else:
                z1_interp, z2_interp = z1_norm, z2_norm
            
            token_sim = (z1_interp * z2_interp).sum(dim=-1).mean()
            global_sim = F.cosine_similarity(z1_norm.mean(1), z2_norm.mean(1), dim=-1).mean()
        
        return {
            'loss': total_loss,
            'align_loss': align_loss,
            'var_loss': var_loss,
            'cov_loss': cov_loss,
            'pos_similarity': pos_sim.item() if torch.is_tensor(pos_sim) else pos_sim,
            'token_sim': token_sim.item(),
            'global_sim': global_sim.item(),
            'global_sim_weighted': global_sim.item(),  # Placeholder
            'z1': z1, 'z2': z2,
            'weights1': w1, 'weights2': w2,
            'grid_size': gs1,
            'grid_size1': gs1, 'grid_size2': gs2,
            'band1': band1, 'band2': band2
        }
    
    def _forward_variable(self, batch: Dict, device) -> Dict:
        """Forward for variable-sized images (batch size 1 per unique size)"""
        
        # Process each sample in the batch individually
        all_z1, all_z2 = [], []
        all_w1, all_w2 = [], []
        all_gs1, all_gs2 = [], []
        
        B = len(batch['view1_image'])
        band1 = batch['view1_band'][0]  # Assume same band for whole batch
        band2 = batch['view2_band'][0]
        
        for i in range(B):
            img1 = batch['view1_image'][i].to(device)
            rms1 = batch['view1_rms'][i].to(device)
            img2 = batch['view2_image'][i].to(device)
            rms2 = batch['view2_rms'][i].to(device)
            
            # Clean
            img1 = torch.nan_to_num(img1, nan=0.0).clamp(min=-100, max=1e6)
            img2 = torch.nan_to_num(img2, nan=0.0).clamp(min=-100, max=1e6)
            rms1 = torch.nan_to_num(rms1, nan=1.0).clamp(min=1e-10)
            rms2 = torch.nan_to_num(rms2, nan=1.0).clamp(min=1e-10)
            
            if img1.dim() == 3:
                img1, rms1 = img1.unsqueeze(0), rms1.unsqueeze(0)
            if img2.dim() == 3:
                img2, rms2 = img2.unsqueeze(0), rms2.unsqueeze(0)
            
            b1 = batch['view1_band'][i] if isinstance(batch['view1_band'], list) else band1
            b2 = batch['view2_band'][i] if isinstance(batch['view2_band'], list) else band2
            
            out1 = self.encode(img1, rms1, b1)
            out2 = self.encode(img2, rms2, b2)
            
            all_z1.append(out1['z'])
            all_z2.append(out2['z'])
            all_w1.append(out1['weights'])
            all_w2.append(out2['weights'])
            all_gs1.append(out1['grid_size'])
            all_gs2.append(out2['grid_size'])
        
        # Compute loss per sample and average
        total_align = 0
        total_pos_sim = 0
        total_var = 0
        total_cov = 0
        
        for i in range(B):
            z1, z2 = all_z1[i], all_z2[i]
            w1, w2 = all_w1[i], all_w2[i]
            gs1, gs2 = all_gs1[i], all_gs2[i]
            
            align1 = self.align_loss(z1, z2.detach(), w1, w2, gs1, gs2)
            align2 = self.align_loss(z2, z1.detach(), w2, w1, gs2, gs1)
            
            total_align += (align1['loss'] + align2['loss']) / 2
            total_pos_sim += (align1['pos_similarity'] + align2['pos_similarity']) / 2
            
            vic1 = self.vicreg(z1)
            vic2 = self.vicreg(z2)
            total_var += (vic1['var_loss'] + vic2['var_loss']) / 2
            total_cov += (vic1['cov_loss'] + vic2['cov_loss']) / 2
        
        align_loss = total_align / B
        pos_sim = total_pos_sim / B
        var_loss = total_var / B
        cov_loss = total_cov / B
        reg_loss = var_loss + 0.04 * cov_loss
        
        total_loss = align_loss + reg_loss
        
        # Monitoring (use first sample)
        with torch.no_grad():
            z1, z2 = all_z1[0], all_z2[0]
            gs1, gs2 = all_gs1[0], all_gs2[0]
            z1_norm = F.normalize(z1, dim=-1)
            z2_norm = F.normalize(z2, dim=-1)
            
            if gs1 != gs2:
                target = (min(gs1[0], gs2[0]), min(gs1[1], gs2[1]))
                z1_interp = self.align_loss._interpolate_tokens(z1_norm, gs1, target)
                z2_interp = self.align_loss._interpolate_tokens(z2_norm, gs2, target)
            else:
                z1_interp, z2_interp = z1_norm, z2_norm
            
            token_sim = (z1_interp * z2_interp).sum(dim=-1).mean()
            global_sim = F.cosine_similarity(z1_norm.mean(1), z2_norm.mean(1), dim=-1).mean()
        
        return {
            'loss': total_loss,
            'align_loss': align_loss,
            'var_loss': var_loss,
            'cov_loss': cov_loss,
            'pos_similarity': pos_sim.item() if torch.is_tensor(pos_sim) else float(pos_sim),
            'token_sim': token_sim.item(),
            'global_sim': global_sim.item(),
            'global_sim_weighted': global_sim.item(),
            'z1': all_z1[0], 'z2': all_z2[0],
            'weights1': all_w1[0], 'weights2': all_w2[0],
            'grid_size': all_gs1[0],
            'grid_size1': all_gs1[0], 'grid_size2': all_gs2[0],
            'band1': band1, 'band2': band2
        }
    
    def get_representation(self, image, rms, band):
        """Get embeddings for inference"""
        out = self.encode(image, rms, band)
        z = F.normalize(out['z'], dim=-1)
        
        # Weighted global
        w = F.interpolate(out['weights'], out['grid_size'], mode='bilinear')
        w = w.view(z.shape[0], -1, 1)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-10)
        global_emb = (z * w).sum(dim=1)
        
        return {
            'tokens': out['tokens'],
            'z': out['z'],
            'global': global_emb,
            'weights': out['weights'],
            'grid_size': out['grid_size']
        }


def create_optimizer(model, lr=3e-4, weight_decay=0.05):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, warmup_epochs, total_epochs):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])