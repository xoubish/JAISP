"""Native-resolution local patch matcher for multi-instrument residual astrometry.

Multi-band version:
  - The Rubin encoder sees ALL available input channels.  For Rubin-only
    mode this is 6 channels (u/g/r/i/z/y).  With NISP included, the
    encoder takes 9 channels (6 Rubin + 3 NISP Y/J/H), all reprojected
    onto the VIS pixel grid.
  - A learned band embedding (one vector per target band) is concatenated
    to the MLP's pooled features.  This tells the refinement head which
    band's offset to predict, enabling band-specific corrections from a
    single forward pass of the encoder.  The embedding table covers all
    target bands: 6 for Rubin-only, 9 for Rubin+NISP.
  - The cost volume and coarse offset are band-agnostic (they measure
    overall spatial alignment).  Only the MLP refinement and sigma
    prediction are band-conditioned, because the sub-pixel correction
    depends on wavelength and instrument.
  - At inference, the encoder runs once per source.  The MLP head is then
    called once per target band with the appropriate band embedding.
  - Backward compatible: if n_target_bands=1 the band embedding is a
    single learned vector and behavior matches the single-band model.
"""

import math
from typing import Dict, List, Optional

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
      rubin_patch:  [B, C_r, H, W]   all Rubin bands
      vis_patch:    [B, 1,   H, W]
      pixel_to_sky: [B, 2, 2]         local VIS-pixel -> sky Jacobian
      band_idx:     [B] int           index into the band embedding table
                                      (which band's offset to predict)

    Outputs:
      dx_px, dy_px         residual shift in VIS pixel axes
      pred_offset_arcsec   predicted (DRA*, DDec) in arcsec
      log_sigma            learned log uncertainty (arcsec)
    """

    def __init__(
        self,
        rubin_channels: int,
        hidden_channels: int = 32,
        encoder_depth: int = 4,
        search_radius: int = 3,
        softmax_temp: float = 0.05,
        mlp_hidden: int = 128,
        n_target_bands: int = 1,
        band_embed_dim: int = 16,
    ):
        super().__init__()
        self.search_radius = int(max(0, search_radius))
        self.n_target_bands = int(max(1, n_target_bands))
        self.band_embed_dim = int(band_embed_dim) if self.n_target_bands > 1 else 0
        self.rubin_encoder = PatchEncoder(rubin_channels, channels=hidden_channels, depth=encoder_depth)
        self.vis_encoder = PatchEncoder(1, channels=hidden_channels, depth=encoder_depth)

        self.rubin_proj = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        self.vis_proj = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)

        self._log_temp = nn.Parameter(torch.tensor(math.log(max(float(softmax_temp), 1e-3)), dtype=torch.float32))

        # Band embedding table: one learned vector per target band.
        if self.band_embed_dim > 0:
            self.band_embedding = nn.Embedding(self.n_target_bands, self.band_embed_dim)
        else:
            self.band_embedding = None

        feat_dim = hidden_channels * 3 + 2 + self.band_embed_dim
        self.residual_head = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 3),  # residual dx, residual dy, log_sigma
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
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
        y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        g = torch.exp(-(yy ** 2 + xx ** 2) / (2 * 0.5 ** 2))
        return (g / g.sum()).view(1, 1, h, w)

    def _weighted_cost_volume(self, rubin_feat: torch.Tensor, vis_feat: torch.Tensor) -> torch.Tensor:
        rubin_n = F.normalize(self.rubin_proj(rubin_feat), dim=1, eps=1e-6)
        vis_n = F.normalize(self.vis_proj(vis_feat), dim=1, eps=1e-6)
        B, C, H, W = rubin_n.shape
        r = self.search_radius

        if r == 0:
            sim = (rubin_n * vis_n).sum(dim=1, keepdim=True)
            return sim.mean(dim=(2, 3))

        rubin_energy = (rubin_n * rubin_n).sum(dim=1, keepdim=True)
        gauss = self._center_weights(H, W, rubin_n.device, rubin_n.dtype)
        spatial_w = rubin_energy * gauss
        spatial_w = spatial_w / (spatial_w.sum(dim=(2, 3), keepdim=True) + 1e-8)

        sqrt_sw = torch.sqrt(spatial_w + 1e-10)
        rubin_w = rubin_n * sqrt_sw

        vis_pad = F.pad(vis_n, (r, r, r, r), mode='replicate')
        K = (2 * r + 1)
        vis_unf = vis_pad.unfold(2, H, 1).unfold(3, W, 1)
        vis_unf = vis_unf.reshape(B, C, K * K, H, W)
        vis_w = vis_unf * sqrt_sw.unsqueeze(2)

        scale = 1.0 / math.sqrt(float(max(1, C)))
        logits = (rubin_w.unsqueeze(2) * vis_w).sum(dim=(1, 3, 4)) * scale
        return logits

    def forward(
        self,
        rubin_patch: torch.Tensor,
        vis_patch: torch.Tensor,
        pixel_to_sky: torch.Tensor,
        band_idx: Optional[torch.Tensor] = None,
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

        # Concatenate band embedding if multi-band.
        parts = [rubin_pool, vis_pool, delta_pool, coarse]
        if self.band_embedding is not None and band_idx is not None:
            band_emb = self.band_embedding(band_idx)  # [B, band_embed_dim]
            parts.append(band_emb)
        elif self.band_embedding is not None:
            # Default to band 0 if not provided (backward compat).
            band_emb = self.band_embedding.weight[0:1].expand(B, -1)
            parts.append(band_emb)

        pooled = torch.cat(parts, dim=1)
        residual = self.residual_head(pooled)

        dx_px = coarse_dx + residual[:, 0]
        dy_px = coarse_dy + residual[:, 1]
        log_sigma = residual[:, 2].clamp(min=-6.0, max=3.0)

        # --- Convert pixel shift to sky offset via local WCS Jacobian ---
        pix = torch.stack([dx_px, dy_px], dim=1).unsqueeze(-1)
        pred_sky = torch.bmm(pixel_to_sky, pix).squeeze(-1)

        return {
            'dx_px': dx_px,
            'dy_px': dy_px,
            'coarse_dx_px': coarse_dx,
            'coarse_dy_px': coarse_dy,
            'pred_offset_arcsec': pred_sky,
            'log_sigma': log_sigma,
            'confidence': probs.max(dim=1).values,
            'temperature': self.temperature.detach(),
            'logits': logits,
        }

    def predict_all_bands(
        self,
        rubin_patch: torch.Tensor,
        vis_patch: torch.Tensor,
        pixel_to_sky: torch.Tensor,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Run encoder once, then predict offsets for every target band.

        Returns a dict mapping band_idx -> output dict.
        Useful at inference to get all 6 concordance fields from one
        encoder pass per source.
        """
        rubin_feat = self.rubin_encoder(rubin_patch)
        vis_feat = self.vis_encoder(vis_patch)

        logits = self._weighted_cost_volume(rubin_feat, vis_feat)
        probs = torch.softmax(logits / self.temperature, dim=1)
        coarse_dx = (probs * self.dx_lut[:, :probs.shape[1]]).sum(dim=1)
        coarse_dy = (probs * self.dy_lut[:, :probs.shape[1]]).sum(dim=1)
        coarse = torch.stack([coarse_dx, coarse_dy], dim=1)

        B, C, H, W = rubin_feat.shape
        gauss = self._center_weights(H, W, rubin_feat.device, rubin_feat.dtype)
        rubin_pool = (rubin_feat * gauss).sum(dim=(2, 3))
        vis_pool = (vis_feat * gauss).sum(dim=(2, 3))
        delta_pool = rubin_pool - vis_pool
        base_parts = [rubin_pool, vis_pool, delta_pool, coarse]

        results = {}
        for bi in range(self.n_target_bands):
            parts = list(base_parts)
            if self.band_embedding is not None:
                idx_t = torch.full((B,), bi, dtype=torch.long, device=rubin_patch.device)
                parts.append(self.band_embedding(idx_t))

            pooled = torch.cat(parts, dim=1)
            residual = self.residual_head(pooled)

            dx_px = coarse_dx + residual[:, 0]
            dy_px = coarse_dy + residual[:, 1]
            log_sigma = residual[:, 2].clamp(min=-6.0, max=3.0)

            pix = torch.stack([dx_px, dy_px], dim=1).unsqueeze(-1)
            pred_sky = torch.bmm(pixel_to_sky, pix).squeeze(-1)

            results[bi] = {
                'dx_px': dx_px,
                'dy_px': dy_px,
                'coarse_dx_px': coarse_dx,
                'coarse_dy_px': coarse_dy,
                'pred_offset_arcsec': pred_sky,
                'log_sigma': log_sigma,
                'confidence': probs.max(dim=1).values,
                'temperature': self.temperature.detach(),
                'logits': logits,
            }
        return results
