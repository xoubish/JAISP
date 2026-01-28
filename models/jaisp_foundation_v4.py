# jaisp_foundation_v4.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List

# =============================================================================
# INFORMATION MAP (Hardened with registered buffers)
# =============================================================================

class InformationMap(nn.Module):
    """Pure signal-based weighting using registered Sobel buffers."""
    def __init__(self, snr_threshold: float = 2.0, min_weight: float = 0.001):
        super().__init__()
        self.snr_threshold = snr_threshold
        self.min_weight = min_weight
        
        # Register kernels as buffers so they move to device with the model
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
        sy = sx.transpose(-1, -2)
        self.register_buffer('sobel_x', sx)
        self.register_buffer('sobel_y', sy)
    
    def forward(self, image: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
        # Signal-to-Noise weighting
        snr = image.abs() / (rms + 1e-10)
        snr_weight = torch.sigmoid((snr - self.snr_threshold) * 2.0)
        
        # Gradient magnitude using buffers
        gx = F.conv2d(image, self.sobel_x, padding=1)
        gy = F.conv2d(image, self.sobel_y, padding=1)
        grad = torch.sqrt(gx**2 + gy**2 + 1e-10)
        
        # Normalize gradient per-image
        grad_max = grad.amax(dim=(2,3), keepdim=True) + 1e-10
        grad_weight = grad / grad_max
        
        # Combine and increase contrast
        weights = torch.maximum(snr_weight, grad_weight * 0.5) ** 2
        weights = weights.clamp(min=self.min_weight)
        
        # Area-normalized output
        return weights / (weights.sum(dim=(2,3), keepdim=True) + 1e-10) * (image.shape[2] * image.shape[3])


# =============================================================================
# ENCODER COMPONENTS
# =============================================================================

class BandStem(nn.Module):
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
        x = x.clamp(-10, 100)  # Robust range for Astro data
        return self.net(x)

class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.proj(x)
        gs = (x.shape[2], x.shape[3]) # Capture dynamic grid size
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
    def __init__(self, stem_ch: int = 64, embed_dim: int = 256, depth: int = 6, patch_size: int = 16):
        super().__init__()
        self.patch_embed = PatchEmbed(stem_ch, embed_dim, patch_size)
        self.base_grid_size = 32
        self.pos_embed = nn.Parameter(torch.randn(1, self.base_grid_size ** 2, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
    
    def _interpolate_pos_embed(self, grid_size: Tuple[int, int]) -> torch.Tensor:
        H, W = grid_size
        if H == self.base_grid_size and W == self.base_grid_size:
            return self.pos_embed
        
        pos_embed = self.pos_embed.reshape(1, self.base_grid_size, self.base_grid_size, -1)
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        return pos_embed.permute(0, 2, 3, 1).reshape(1, H * W, -1)
    
    def forward(self, x):
        tokens, gs = self.patch_embed(x)
        tokens = tokens + self._interpolate_pos_embed(gs)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens), gs

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        B, N, D = x.shape
        x = x.reshape(B * N, D)
        x = self.net(x)
        return x.reshape(B, N, -1)

# =============================================================================
# LOSS: ALIGNMENT & VICReg
# =============================================================================

class AlignmentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def _interpolate_tokens(self, z: torch.Tensor, grid_size: Tuple[int, int], target_size: Tuple[int, int]) -> torch.Tensor:
        if grid_size == target_size:
            return z
        B, N, D = z.shape
        z_spatial = z.transpose(1, 2).view(B, D, grid_size[0], grid_size[1])
        z_interp = F.interpolate(z_spatial, size=target_size, mode='bilinear', align_corners=False)
        return z_interp.view(B, D, -1).transpose(1, 2)
    
    def forward(self, z1, z2, w1, w2, gs1, gs2):
        B, _, D = z1.shape
        # Interpolate to finer (MAX) grid
        target_size = (max(gs1[0], gs2[0]), max(gs1[1], gs2[1]))
        
        z1_common = self._interpolate_tokens(z1, gs1, target_size)
        z2_common = self._interpolate_tokens(z2, gs2, target_size)
        w_avg = (F.interpolate(w1, size=target_size, mode='bilinear') + 
                 F.interpolate(w2, size=target_size, mode='bilinear')) / 2
        
        w_avg = w_avg.view(B, -1)
        w_avg = w_avg / (w_avg.sum(dim=1, keepdim=True) + 1e-10)
        
        z1_norm = F.normalize(z1_common, dim=-1)
        z2_norm = F.normalize(z2_common, dim=-1)
        
        pos_sim = (z1_norm * z2_norm).sum(dim=-1)
        align_loss = ((1 - pos_sim) * w_avg).sum(dim=1).mean()
        
        return {'loss': align_loss, 'pos_similarity': pos_sim.mean()}

class VICReg(nn.Module):
    def __init__(self, var_weight: float = 1.0, cov_weight: float = 0.04):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, D = z.shape
        z_flat = z.reshape(-1, D)
        
        # Variance: push std towards 1.0
        std = torch.sqrt(z_flat.var(dim=0) + 1e-4)
        var_loss = F.relu(1.0 - std).mean()
        
        # Covariance: push off-diagonals to 0 (Masked version)
        z_centered = z_flat - z_flat.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (z_flat.shape[0] - 1)
        mask = ~torch.eye(D, device=z.device).bool()
        cov_loss = cov[mask].pow(2).mean()
        
        return {
            'var_loss': var_loss,
            'cov_loss': cov_loss,
            'reg_loss': self.var_weight * var_loss + self.cov_weight * cov_loss
        }

# =============================================================================
# FULL MODEL
# =============================================================================

class JAISPFoundationV4(nn.Module):
    def __init__(self, band_names: List[str], stem_ch: int = 64, embed_dim: int = 256, 
                 proj_dim: int = 256, depth: int = 6, patch_size: int = 16, temperature: float = 0.1):
        super().__init__()
        self.band_names = band_names
        self.embed_dim = embed_dim
        
        self.stems = nn.ModuleDict({name: BandStem(stem_ch) for name in band_names})
        self.info_maps = nn.ModuleDict({name: InformationMap() for name in band_names})
        self.encoder = SharedEncoder(stem_ch, embed_dim, depth, patch_size)
        self.projector = ProjectionHead(embed_dim, proj_dim)
        
        self.align_loss = AlignmentLoss(temperature)
        self.vicreg = VICReg()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def encode(self, image: torch.Tensor, rms: torch.Tensor, band: str) -> Dict:
        weights = self.info_maps[band](image, rms)
        feat = self.stems[band](image, rms)
        tokens, grid_size = self.encoder(feat)
        z = self.projector(tokens)
        return {'z': z, 'weights': weights, 'grid_size': grid_size}

    def forward(self, batch: Dict) -> Dict:
        # Move all list/tensors to model device
        device = next(self.parameters()).device
        is_variable = isinstance(batch['view1_image'], list)
    
        if is_variable:
            B = len(batch['view1_image'])
            all_z, all_w, all_gs = [], [], []
    
            for i in range(B):
                # Individual processing for variable sizes
                img1 = torch.nan_to_num(batch['view1_image'][i].to(device)).clamp(min=-100)
                rms1 = torch.nan_to_num(batch['view1_rms'][i].to(device), nan=1.0).clamp(min=1e-10)
                band1 = batch['view1_band'][i]
    
                img2 = torch.nan_to_num(batch['view2_image'][i].to(device)).clamp(min=-100)
                rms2 = torch.nan_to_num(batch['view2_rms'][i].to(device), nan=1.0).clamp(min=1e-10)
                band2 = batch['view2_band'][i]
    
                out1 = self.encode(img1.unsqueeze(0), rms1.unsqueeze(0), band1)
                out2 = self.encode(img2.unsqueeze(0), rms2.unsqueeze(0), band2)
    
                all_z.append((out1['z'], out2['z']))
                all_w.append((out1['weights'], out2['weights']))
                all_gs.append((out1['grid_size'], out2['grid_size']))
    
            # Loss accumulation
            align_l, var_l, cov_l = 0.0, 0.0, 0.0
            tok_sim_l, glob_sim_l = 0.0, 0.0
    
            for i in range(B):
                z1, z2 = all_z[i]
                w1, w2 = all_w[i]
                gs1, gs2 = all_gs[i]
    
                # Alignment losses + sims (both directions)
                out12 = self.align_loss(z1, z2.detach(), w1, w2, gs1, gs2)
                out21 = self.align_loss(z2, z1.detach(), w2, w1, gs2, gs1)
    
                al = (out12['loss'] + out21['loss']) / 2.0
                tok_sim = (out12['pos_similarity'] + out21['pos_similarity']) / 2.0
    
                # VICReg
                vic1, vic2 = self.vicreg(z1), self.vicreg(z2)
                vloss = (vic1['var_loss'] + vic2['var_loss']) / 2.0
                closs = (vic1['cov_loss'] + vic2['cov_loss']) / 2.0
    
                # Global sim (mean pooled cosine sim)
                with torch.no_grad():
                    g1 = F.normalize(z1.mean(dim=1), dim=-1)  # [1, D]
                    g2 = F.normalize(z2.mean(dim=1), dim=-1)
                    glob_sim = (g1 * g2).sum(dim=-1).mean()
    
                align_l += al
                var_l += vloss
                cov_l += closs
                tok_sim_l += tok_sim
                glob_sim_l += glob_sim
    
            align_l = align_l / B
            var_l = var_l / B
            cov_l = cov_l / B
            tok_sim_l = tok_sim_l / B
            glob_sim_l = glob_sim_l / B
    
            total_loss = align_l + var_l + 0.04 * cov_l
    
            # Return a representative sample for visualization (index 0)
            return {
                'loss': total_loss,
                'align_loss': align_l,
                'var_loss': var_l,
                'cov_loss': cov_l,
                'token_sim': float(tok_sim_l.detach().item()),
                'global_sim': float(glob_sim_l.detach().item()),
    
                'z1': all_z[0][0],
                'z2': all_z[0][1],
                'weights1': all_w[0][0],
                'weights2': all_w[0][1],
                'grid_size1': all_gs[0][0],
                'grid_size2': all_gs[0][1],
                'band1': batch['view1_band'][0],
                'band2': batch['view2_band'][0],
            }
    
        else:
            # Fallback for fixed resolution if needed
            return self._forward_fixed(batch, device)


    def _forward_fixed(self, batch, device):
        raise NotImplementedError(
            "_forward_fixed is not implemented. "
            "This model currently expects variable-size (list-based) inputs."
        )

def create_optimizer(model, lr=3e-4, weight_decay=0.05):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def create_scheduler(optimizer, warmup_epochs, total_epochs):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])