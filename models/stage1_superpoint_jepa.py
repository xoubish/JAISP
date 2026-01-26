import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.optimize import linear_sum_assignment

class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape
        x = self.proj(x)
        grid_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), grid_size

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        attn_out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(attn_out)
        return x + self.mlp(self.norm2(x))

class SuperPointJEPA(nn.Module):
    def __init__(self, rubin_channels=6, euclid_channels=4, embed_dim=256, descriptor_dim=256):
        super().__init__()
        self.patch_embed_rubin = PatchEmbed(rubin_channels, embed_dim)
        self.patch_embed_euclid = PatchEmbed(euclid_channels, embed_dim)
        self.encoder = nn.ModuleList([TransformerBlock(embed_dim, 8) for _ in range(4)])
        self.keypoint_head = nn.Sequential(nn.Conv2d(embed_dim, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 1, 1), nn.Sigmoid())
        self.descriptor_head = nn.Sequential(nn.Conv2d(embed_dim, descriptor_dim, 1))
        self.predictor = nn.Sequential(nn.Linear(descriptor_dim, descriptor_dim), nn.LayerNorm(descriptor_dim), nn.GELU(), nn.Linear(descriptor_dim, descriptor_dim))

    def encode_image(self, x, patch_embed, mask=None):
        B, C, H, W = x.shape
        x = (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-8)
        patches, grid_size = patch_embed(torch.nan_to_num(x, nan=0.0))
        for block in self.encoder:
            patches = checkpoint(block, patches, use_reentrant=False) if self.training else block(patches)
        
        feat_map = patches.transpose(1, 2).view(B, -1, grid_size[0], grid_size[1])
        heatmap = self.keypoint_head(feat_map)
        
        if mask is not None:
            if mask.dim() == 2: mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3: mask = mask.unsqueeze(1)
            heatmap = heatmap * F.interpolate(mask.float(), size=grid_size, mode='nearest')
            
        descs = F.normalize(self.descriptor_head(feat_map).flatten(2).transpose(1, 2), dim=-1)
        return {'heatmap': heatmap, 'descriptors': descs}

    def compute_vicreg_loss(self, d1, d2, rms1=None, rms2=None):
        B, N1, D = d1.shape
        d1_p, d2_n = F.normalize(self.predictor(d1), dim=-1), d2.detach()
        cost = torch.clamp(-(d1_p @ d2_n.transpose(1, 2)), -10., 10.)
        total_inv, all_matches = 0, []
        for b in range(B):
            row, col = linear_sum_assignment(cost[b].detach().cpu().numpy())
            sim = (d1_p[b, row] * d2_n[b, col]).sum(-1)
            weight = 1.0 / (rms1[b].mean() + rms2[b].mean() + 1e-4) if rms1 is not None else 1.0
            total_inv += ((1 - sim) * weight).mean()
            all_matches.append(torch.tensor(col, device=d1.device))
        def var_loss(z):
            std = torch.sqrt(z.reshape(-1, D).var(dim=0) + 1e-6)
            return F.relu(1.0 - std).mean()
        return {'loss': 25.0 * (total_inv/B) + 25.0 * (var_loss(d1) + var_loss(d2))/2, 
                'similarity': (1 - (total_inv/B)).item(), 'matches': torch.stack(all_matches)}

    def forward(self, batch):
        dev = next(self.parameters()).device
        def prep(k):
            v = batch.get(k)
            if v is None: return None
            return torch.stack(v).to(dev) if isinstance(v, list) else v.to(dev)
        r_out = self.encode_image(prep('x_rubin'), self.patch_embed_rubin)
        e_out = self.encode_image(prep('x_euclid'), self.patch_embed_euclid, mask=prep('mask_euclid'))
        l_dict = self.compute_vicreg_loss(r_out['descriptors'], e_out['descriptors'], rms1=prep('rms_rubin'), rms2=prep('rms_euclid'))
        return {**l_dict, 'rubin_heatmap': r_out['heatmap'], 'euclid_heatmap': e_out['heatmap']}

def create_optimizer(model, lr=5e-5): return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
def create_scheduler(opt, warmup, total):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    return SequentialLR(opt, [LinearLR(opt, 0.01, total_iters=warmup), CosineAnnealingLR(opt, total-warmup)], [warmup])