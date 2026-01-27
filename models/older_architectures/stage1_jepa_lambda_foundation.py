# stage1_jepa_lambda_foundation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint #
from typing import List, Dict, Tuple, Optional


class ResidualBlock(nn.Module):
    """Uses Reflective Padding to eliminate artificial edge artifacts."""
    def __init__(self, dim: int):
        super().__init__()
        # Changed to padding_mode='reflect' to stop the 'box' effect
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
        self.norm = nn.InstanceNorm2d(dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.activation(self.norm(self.conv1(x)))
        out = self.norm(self.conv2(out))
        return self.activation(out + identity)

class DenseSpectralEncoder(nn.Module):
    """
    Encoder that preserves 1:1 resolution while suppressing boundary noise.
    """
    def __init__(self, in_channels: int, embed_dim: int = 256, depth: int = 4):
        super().__init__()
        self.spectral_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.spatial_blocks = nn.ModuleList([ResidualBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weights is not None:
            x = self._weighted_normalize(x, weights)
        
        x = self.spectral_proj(x)
        
        for block in self.spatial_blocks:
            if self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        # New: Border Masking. Zero out the latent vectors at the very edge 
        # to stop the model from learning padding artifacts.
        x[:, :, 0, :] = 0; x[:, :, -1, :] = 0
        x[:, :, :, 0] = 0; x[:, :, :, -1] = 0
            
        B, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, D)
        x = self.norm(x).reshape(B, H, W, D)
        return x
    
    def _weighted_normalize(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        w_sum = w.sum(dim=(2,3), keepdim=True).clamp(min=1e-10)
        mean = (x * w).sum(dim=(2,3), keepdim=True) / w_sum
        var = ((x - mean)**2 * w).sum(dim=(2,3), keepdim=True) / w_sum
        std = torch.sqrt(var.clamp(min=1e-10))
        return (x - mean) / std

class JAISPFoundation(nn.Module):
    def __init__(self, projection_dim: int = 256, temperature: float = 0.05):
        super().__init__()
        from stage1_jepa_foundation import PatchExtractor 
        self.patch_extractor = PatchExtractor(
            patch_size_rubin=48, patch_size_euclid=96, n_patches_per_tile=4
        )
        self.encoder_rubin = DenseSpectralEncoder(in_channels=6, embed_dim=projection_dim)
        self.encoder_euclid = DenseSpectralEncoder(in_channels=4, embed_dim=projection_dim)
        self.temperature = temperature
        self._init_weights()

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        r_p, r_w = self.patch_extractor(batch['x_rubin'], batch['rms_rubin'], device=device, survey='rubin')
        e_p, e_w = self.patch_extractor(batch['x_euclid'], batch['rms_euclid'], device=device, survey='euclid')
        
        z_r = self.encoder_rubin(r_p, r_w)
        z_e = self.encoder_euclid(e_p, e_w)
        z_e_matched = F.avg_pool3d(z_e.permute(0, 3, 1, 2), kernel_size=(1, 2, 2)).permute(0, 2, 3, 1)
        
        saliency = e_p.abs().sum(dim=1, keepdim=True)
        saliency = F.avg_pool2d(saliency, kernel_size=2)
        
        loss = self.dense_saliency_loss(z_r, z_e_matched, saliency)
        return {'loss': loss, 'z_rubin': z_r, 'z_euclid': z_e}

    def dense_saliency_loss(self, z1, z2, saliency):
        """
        Forces the model to ignore edges by cropping the sampling interior.
        Ensures gradients are driven by sources, not padding artifacts.
        """
        B, H, W, D = z1.shape
        
        # 1. Create a spatial mask that ignores the outer 3 pixels
        # This explicitly tells the loss to ignore the 'blue/green' edge bands
        edge_mask = torch.ones((H, W), device=z1.device)
        edge_mask[:3, :] = 0; edge_mask[-3:, :] = 0
        edge_mask[:, :3] = 0; edge_mask[:, -3:] = 0
        
        # 2. Apply edge mask to saliency so we only sample the interior
        # saliency is (B, 1, H, W), edge_mask is (H, W)
        sal_f = (saliency.squeeze(1) * edge_mask).reshape(-1)
        
        # 3. Normalize latents for cosine similarity
        z1_f = F.normalize(z1.reshape(-1, D), dim=-1)
        z2_f = F.normalize(z2.reshape(-1, D), dim=-1)
        
        # 4. Sample top-K highest signal pixels in the interior
        # Reducing K slightly helps memory and focuses on the strongest sources
        k = 256 * B 
        _, top_idx = torch.topk(sal_f, k=min(k, sal_f.size(0)))
        
        z1_s = z1_f[top_idx]
        z2_s = z2_f[top_idx]
        
        # 5. Contrastive Cross-Entropy
        logits = torch.matmul(z1_s, z2_s.T) / self.temperature
        labels = torch.arange(len(z1_s), device=z1.device)
        return F.cross_entropy(logits, labels)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

def create_optimizer(model, lr=1e-4, weight_decay=0.05):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def create_scheduler(optimizer, warmup_epochs, total_epochs):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])