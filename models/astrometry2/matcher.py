"""Native-resolution local patch matcher for Rubin<->VIS residual astrometry.

Changes from v1:
  - Cost volume computed via F.unfold instead of a Python loop over (2r+1)^2
    displacements.  Single tensor operation, cleaner, and faster at larger
    search radii.
"""

import math
from typing import Dict

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
      rubin_patch:  [B, C_r, H, W]
      vis_patch:    [B, 1,   H, W]
      pixel_to_sky: [B, 2, 2]  local VIS-pixel -> sky Jacobian (arcsec / pixel)

    Outputs:
      dx_px, dy_px         residual shift in VIS pixel axes
      pred_offset_arcsec   predicted (DRA*, DDec) in arcsec
      log_sigma            learned log uncertainty (arcsec)

    Architecture:
      1. Each patch is encoded independently by a lightweight CNN (PatchEncoder).
      2. A spatially-weighted cross-correlation cost volume finds the coarse
         displacement via soft-argmax. Spatial weights are the product of
         the local Rubin feature energy (bright/peaked sources dominate) and
         a Gaussian center prior (edges matter less). This prevents the
         background from diluting the astrometric signal.
      3. A small MLP refines the coarse estimate using center-biased pooled
         features from both encoders.
      4. The final pixel displacement is multiplied by the per-source
         pixel_to_sky Jacobian to produce (DRA*, DDec) in arcseconds.
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
        # Start sigma at ~50 mas rather than 1 arcsec so the uncertainty
        # loss is meaningful from the first epoch.
        with torch.no_grad():
            self.residual_head[-1].bias[2] = math.log(0.05)

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

    def _center_weights(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Gaussian weights peaked at the patch center, normalized to sum to 1.
        sigma=0.5 in normalized [-1,1] coords means the weight falls to ~e^{-0.5}
        at the patch edges, suppressing noisy boundary pixels.
        """
        y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        g = torch.exp(-(yy ** 2 + xx ** 2) / (2 * 0.5 ** 2))
        return (g / g.sum()).view(1, 1, h, w)

    def _weighted_cost_volume(self, rubin_feat: torch.Tensor, vis_feat: torch.Tensor) -> torch.Tensor:
        """
        Spatially-weighted cross-correlation cost volume via unfold.

        Instead of looping over (2r+1)^2 displacements, we pad the VIS
        features and use F.unfold to extract all shifted patches in one op,
        then compute the weighted dot product as a batched matmul.
        """
        rubin_n = F.normalize(self.rubin_proj(rubin_feat), dim=1, eps=1e-6)
        vis_n = F.normalize(self.vis_proj(vis_feat), dim=1, eps=1e-6)
        B, C, H, W = rubin_n.shape
        r = self.search_radius

        if r == 0:
            sim = (rubin_n * vis_n).sum(dim=1, keepdim=True)
            return sim.mean(dim=(2, 3))  # [B, 1]

        # Spatial attention: source brightness x Gaussian center prior.
        rubin_energy = (rubin_n * rubin_n).sum(dim=1, keepdim=True)        # [B, 1, H, W]
        gauss = self._center_weights(H, W, rubin_n.device, rubin_n.dtype)  # [1, 1, H, W]
        spatial_w = rubin_energy * gauss
        spatial_w = spatial_w / (spatial_w.sum(dim=(2, 3), keepdim=True) + 1e-8)

        # Weight the Rubin features by spatial attention, then flatten spatial dims.
        # rubin_weighted: [B, C, H*W]  (each pixel scaled by its spatial weight)
        sqrt_sw = torch.sqrt(spatial_w + 1e-10)  # [B, 1, H, W]
        rubin_w = rubin_n * sqrt_sw              # [B, C, H, W]

        # Pad VIS and unfold to get all (2r+1)^2 shifted views.
        vis_pad = F.pad(vis_n, (r, r, r, r), mode='replicate')
        K = (2 * r + 1)
        # unfold extracts K*K patches of size H x W from the padded VIS.
        # Result shape: [B, C*H*W, K*K]
        vis_unf = vis_pad.unfold(2, H, 1).unfold(3, W, 1)  # [B, C, K, K, H, W]
        vis_unf = vis_unf.reshape(B, C, K * K, H, W)        # [B, C, K^2, H, W]

        # Weight VIS patches by same spatial weights.
        vis_w = vis_unf * sqrt_sw.unsqueeze(2)  # [B, C, K^2, H, W]

        # Dot product: sum over C, H, W for each of K^2 shifts.
        # rubin_w: [B, C, 1, H, W]  *  vis_w: [B, C, K^2, H, W]  ->  sum -> [B, K^2]
        scale = 1.0 / math.sqrt(float(max(1, C)))
        logits = (rubin_w.unsqueeze(2) * vis_w).sum(dim=(1, 3, 4)) * scale  # [B, K^2]

        return logits

    def forward(
        self,
        rubin_patch: torch.Tensor,
        vis_patch: torch.Tensor,
        pixel_to_sky: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        rubin_feat = self.rubin_encoder(rubin_patch)
        vis_feat = self.vis_encoder(vis_patch)

        # --- Coarse offset via spatially-weighted cost volume + soft-argmax ---
        logits = self._weighted_cost_volume(rubin_feat, vis_feat)
        probs = torch.softmax(logits / self.temperature, dim=1)
        coarse_dx = (probs * self.dx_lut[:, :probs.shape[1]]).sum(dim=1)
        coarse_dy = (probs * self.dy_lut[:, :probs.shape[1]]).sum(dim=1)
        coarse = torch.stack([coarse_dx, coarse_dy], dim=1)

        # --- MLP refinement with center-biased Gaussian pooling ---
        B, C, H, W = rubin_feat.shape
        gauss = self._center_weights(H, W, rubin_feat.device, rubin_feat.dtype)
        rubin_pool = (rubin_feat * gauss).sum(dim=(2, 3))
        vis_pool = (vis_feat * gauss).sum(dim=(2, 3))
        delta_pool = rubin_pool - vis_pool
        pooled = torch.cat([rubin_pool, vis_pool, delta_pool, coarse], dim=1)
        residual = self.residual_head(pooled)

        dx_px = coarse_dx + residual[:, 0]
        dy_px = coarse_dy + residual[:, 1]
        log_sigma = residual[:, 2].clamp(min=-6.0, max=3.0)

        # --- Convert pixel shift to sky offset via local WCS Jacobian ---
        pix = torch.stack([dx_px, dy_px], dim=1).unsqueeze(-1)  # [B, 2, 1]
        pred_sky = torch.bmm(pixel_to_sky, pix).squeeze(-1)      # [B, 2] = (DRA*, DDec)

        return {
            'dx_px': dx_px,
            'dy_px': dy_px,
            'coarse_dx_px': coarse_dx,
            'coarse_dy_px': coarse_dy,
            'pred_offset_arcsec': pred_sky,
            'log_sigma': log_sigma,
            'confidence': probs.max(dim=1).values,
            'temperature': self.temperature.detach(),
            # Raw cost-volume logits [B, K] -- useful for visualization.
            # Reshape to [B, 2r+1, 2r+1] to see the matching signal as a 2D map.
            'logits': logits,
        }
