"""LatentPositionHead v2: adds an unclamped raw-window side-channel.

Motivation (nb23, bright-worsener mechanism): the foundation's BandStem feeds
the network ``(image/rms).clamp(-10, 100)``, so stars with peak per-pixel
S/N > 100 have their cores erased in feature space; the head then displaces
them 10-30 mas (causally verified by flux-sweep and rotation interventions).
This variant gives the head a direct view of its own 17x17 VIS window through
a smooth asinh compression instead of the hard clamp:

    soft(x) = knee * asinh(x / knee),   x = image/rms,   knee = 50

which is linear below the knee and logarithmic above, so bright-star cores
keep their sub-pixel structure. The foundation (and its caches) are untouched;
only the head is retrained.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from latent_position_head import (
    LatentPositionHead,
    extract_local_windows,
    vis_px_to_bottleneck_px,
)

SOFT_KNEE = 50.0


def soft_snr_map(vis_img: torch.Tensor, vis_rms: torch.Tensor, knee: float = SOFT_KNEE) -> torch.Tensor:
    """Smoothly compressed per-pixel S/N map: knee*asinh((img/rms)/knee)."""
    x = vis_img / (vis_rms + 1e-10)
    return knee * torch.asinh(x / knee)


class LatentPositionHeadV2(LatentPositionHead):
    """LatentPositionHead + raw asinh-compressed VIS window branch."""

    def __init__(
        self,
        hidden_ch: int = 256,
        stem_ch: int = 64,
        bottleneck_out: int = 128,
        stem_out: int = 64,
        mlp_hidden: int = 128,
        bottleneck_window: int = 5,
        stem_window: int = 17,
        fused_pixel_scale: float = 0.4,
        vis_pixel_scale: float = 0.1,
        raw_out: int = 32,
    ):
        super().__init__(
            hidden_ch=hidden_ch, stem_ch=stem_ch, bottleneck_out=bottleneck_out,
            stem_out=stem_out, mlp_hidden=mlp_hidden,
            bottleneck_window=bottleneck_window, stem_window=stem_window,
            fused_pixel_scale=fused_pixel_scale, vis_pixel_scale=vis_pixel_scale,
        )
        self.raw_out = raw_out
        self.raw_conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, raw_out, 3, padding=1), nn.GroupNorm(8, raw_out), nn.GELU(),
        )
        # Rebuild the MLP for the widened feature vector; same init convention.
        feat_dim = bottleneck_out + stem_out + raw_out
        self.head = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, 3),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            self.head[-1].bias[2] = math.log(0.05)

    def forward(
        self,
        bottleneck: torch.Tensor,
        vis_stem_features: torch.Tensor,
        source_positions_vis: torch.Tensor,
        pixel_to_sky: torch.Tensor,
        fused_hw: Tuple[int, int],
        vis_hw: Tuple[int, int],
        vis_soft: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        if vis_soft is None:
            raise ValueError('LatentPositionHeadV2 requires vis_soft (asinh-compressed S/N map)')

        positions_bn = vis_px_to_bottleneck_px(
            source_positions_vis, self.vis_pixel_scale, self.fused_pixel_scale,
            fused_hw, vis_hw,
        )
        bn_windows = extract_local_windows(bottleneck, positions_bn, self.bottleneck_window)
        bn_vec = self._gauss_pool(self.bn_conv(bn_windows), self.bn_gauss)

        stem_windows = extract_local_windows(vis_stem_features, source_positions_vis, self.stem_window)
        stem_vec = self._gauss_pool(self.stem_conv(stem_windows), self.stem_gauss)

        raw_windows = extract_local_windows(vis_soft, source_positions_vis, self.stem_window)
        raw_vec = self._gauss_pool(self.raw_conv(raw_windows), self.stem_gauss)

        out = self.head(torch.cat([bn_vec, stem_vec, raw_vec], dim=1))
        dx_px, dy_px = out[:, 0], out[:, 1]
        log_sigma = out[:, 2].clamp(-6.0, 3.0)
        pix = torch.stack([dx_px, dy_px], dim=1).unsqueeze(-1)
        pred_sky = torch.bmm(pixel_to_sky, pix).squeeze(-1)
        sigma = torch.exp(log_sigma)
        return {
            'pred_offset_arcsec': pred_sky,
            'dx_px': dx_px, 'dy_px': dy_px,
            'log_sigma': log_sigma,
            'confidence': 1.0 / sigma.clamp_min(1e-4),
        }
