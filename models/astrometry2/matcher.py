"""Native-resolution local patch matcher for Rubin<->VIS residual astrometry."""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: int = 32, depth: int = 4):
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(max(2, depth)):
            layers.extend([
                nn.Conv2d(c_in, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(4, channels),
                nn.GELU(),
            ])
            c_in = channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalAstrometryMatcher(nn.Module):
    """
    Predict local residual offset between a Rubin patch and a VIS patch.

    Inputs:
      rubin_patch: [B, C_r, H, W]
      vis_patch:   [B, 1,   H, W]
      pixel_to_sky: [B, 2, 2] local VIS-pixel -> sky Jacobian (arcsec / pixel)

    Outputs:
      dx_px, dy_px       residual shift in VIS pixel axes
      pred_offset_arcsec predicted (DRA*, DDec)
      log_sigma          learned scalar uncertainty in arcsec
    """

    def __init__(
        self,
        rubin_channels: int,
        hidden_channels: int = 32,
        encoder_depth: int = 4,
        search_radius: int = 3,
        softmax_temp: float = 0.05,
        mlp_hidden: int = 128,
    ):
        super().__init__()
        self.search_radius = int(max(0, search_radius))
        self.rubin_encoder = PatchEncoder(rubin_channels, channels=hidden_channels, depth=encoder_depth)
        self.vis_encoder = PatchEncoder(1, channels=hidden_channels, depth=encoder_depth)

        self.rubin_proj = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        self.vis_proj = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)

        self._log_temp = nn.Parameter(torch.tensor(math.log(max(float(softmax_temp), 1e-3)), dtype=torch.float32))

        feat_dim = hidden_channels * 3 + 2
        self.residual_head = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 3),  # residual dx, residual dy, log_sigma
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

        disp = []
        r = self.search_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                disp.append((float(dx), float(dy)))
        dx_lut = torch.tensor([d[0] for d in disp], dtype=torch.float32).view(1, -1)
        dy_lut = torch.tensor([d[1] for d in disp], dtype=torch.float32).view(1, -1)
        self.register_buffer('dx_lut', dx_lut, persistent=False)
        self.register_buffer('dy_lut', dy_lut, persistent=False)

    @property
    def temperature(self) -> torch.Tensor:
        return self._log_temp.exp().clamp_min(1e-3)

    def _global_cost_volume(self, rubin_feat: torch.Tensor, vis_feat: torch.Tensor) -> torch.Tensor:
        rubin_n = F.normalize(self.rubin_proj(rubin_feat), dim=1, eps=1e-6)
        vis_n = F.normalize(self.vis_proj(vis_feat), dim=1, eps=1e-6)
        r = self.search_radius
        if r == 0:
            sim = (rubin_n * vis_n).mean(dim=(1, 2, 3), keepdim=False)
            return sim[:, None]

        vis_pad = F.pad(vis_n, (r, r, r, r), mode='replicate')
        logits = []
        scale = 1.0 / math.sqrt(float(max(1, rubin_n.shape[1])))
        h, w = rubin_n.shape[-2:]
        for dy in range(-r, r + 1):
            y0 = r + dy
            y1 = y0 + h
            for dx in range(-r, r + 1):
                x0 = r + dx
                x1 = x0 + w
                shifted = vis_pad[:, :, y0:y1, x0:x1]
                corr = (rubin_n * shifted).sum(dim=1).mean(dim=(1, 2)) * scale
                logits.append(corr)
        return torch.stack(logits, dim=1)

    def forward(
        self,
        rubin_patch: torch.Tensor,
        vis_patch: torch.Tensor,
        pixel_to_sky: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        rubin_feat = self.rubin_encoder(rubin_patch)
        vis_feat = self.vis_encoder(vis_patch)

        logits = self._global_cost_volume(rubin_feat, vis_feat)
        probs = torch.softmax(logits / self.temperature, dim=1)
        coarse_dx = (probs * self.dx_lut[:, :probs.shape[1]]).sum(dim=1)
        coarse_dy = (probs * self.dy_lut[:, :probs.shape[1]]).sum(dim=1)
        coarse = torch.stack([coarse_dx, coarse_dy], dim=1)

        rubin_pool = rubin_feat.mean(dim=(2, 3))
        vis_pool = vis_feat.mean(dim=(2, 3))
        delta_pool = rubin_pool - vis_pool
        pooled = torch.cat([rubin_pool, vis_pool, delta_pool, coarse], dim=1)
        residual = self.residual_head(pooled)
        dx_px = coarse_dx + residual[:, 0]
        dy_px = coarse_dy + residual[:, 1]
        log_sigma = residual[:, 2].clamp(min=-6.0, max=3.0)

        pix = torch.stack([dx_px, dy_px], dim=1).unsqueeze(-1)  # [B,2,1]
        pred_sky = torch.bmm(pixel_to_sky, pix).squeeze(-1)     # [B,2] = (DRA*, DDec)

        return {
            'dx_px': dx_px,
            'dy_px': dy_px,
            'coarse_dx_px': coarse_dx,
            'coarse_dy_px': coarse_dy,
            'pred_offset_arcsec': pred_sky,
            'log_sigma': log_sigma,
            'confidence': probs.max(dim=1).values,
            'temperature': self.temperature.detach(),
        }
