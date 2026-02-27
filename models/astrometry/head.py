"""
Astrometry Concordance Head for JAISP — v2 (Direct Regression).

v1 used cross-correlation + soft-argmax, which fundamentally cannot resolve
sub-token offsets (0.5" ≈ 0.3 tokens — the correlation peak barely moves).

v2 uses DIRECT REGRESSION from token embeddings. The foundation model's
strict position encoding means that spatial shifts are encoded in the token
representations — the head just needs to learn the mapping from token
differences to astrometric offsets.

Architecture:
  1. Encode Rubin and VIS → token grids, interpolate to common grid
  2. Build features: concat(rubin, vis, rubin−vis) → [B, 3D, H, W]
  3. Global branch: pool → MLP → single (Δy, Δx) for entire field
  4. Local branch: CNN → per-position (Δy, Δx) residual
  5. Combined = global + local, convert token-units → arcseconds
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pixel scales (arcsec / pixel)
# ---------------------------------------------------------------------------

PIXEL_SCALES = {
    "euclid_VIS": 0.1,
    "euclid_Y": 0.3,
    "euclid_J": 0.3,
    "euclid_H": 0.3,
    "rubin_u": 0.2,
    "rubin_g": 0.2,
    "rubin_r": 0.2,
    "rubin_i": 0.2,
    "rubin_z": 0.2,
    "rubin_y": 0.2,
}


def interpolate_tokens(
    tokens: torch.Tensor,
    src_grid: Tuple[int, int],
    dst_grid: Tuple[int, int],
) -> torch.Tensor:
    """Resample token grid [B, N, D] from src_grid to dst_grid."""
    if src_grid == dst_grid:
        return tokens
    B, _, D = tokens.shape
    sh, sw = int(src_grid[0]), int(src_grid[1])
    dh, dw = int(dst_grid[0]), int(dst_grid[1])
    x = tokens.transpose(1, 2).contiguous().view(B, D, sh, sw)
    x = F.interpolate(x, size=(dh, dw), mode="bilinear", align_corners=False)
    return x.view(B, D, dh * dw).transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Global offset branch
# ---------------------------------------------------------------------------

class GlobalOffsetBranch(nn.Module):
    """
    Detect the uniform (constant) component of the offset field.

    Uses conv → adaptive pool → MLP so it sees spatial structure
    before pooling, not just raw channel statistics.
    """

    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),  # → [B, hidden, 4, 4]
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 16, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 2),  # (Δy, Δx) in token units
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        x = self.encoder(features)
        x = x.view(B, -1)
        return self.mlp(x).unsqueeze(-1).unsqueeze(-1)  # [B, 2, 1, 1]


# ---------------------------------------------------------------------------
# Local offset branch
# ---------------------------------------------------------------------------

class LocalOffsetBranch(nn.Module):
    """
    Predict spatially-varying component of the offset field.
    Large 5×5 kernels give each position context from neighbors.
    """

    def __init__(self, in_channels: int, hidden: int = 64, depth: int = 5):
        super().__init__()
        layers = [nn.Conv2d(in_channels, hidden, 5, padding=2), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(hidden, hidden, 5, padding=2), nn.GELU()]
        layers += [nn.Conv2d(hidden, 2, 5, padding=2)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


# ---------------------------------------------------------------------------
# Optional stem-space refinement (from user's edit)
# ---------------------------------------------------------------------------

class StemResidualRefineNet(nn.Module):
    def __init__(self, stem_channels: int = 64, hidden: int = 32, depth: int = 4):
        super().__init__()
        in_ch = 3 * stem_channels + 2
        layers = [nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU()]
        layers += [nn.Conv2d(hidden, 2, 3, padding=1)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rubin_feat, vis_feat, coarse_dra, coarse_ddec):
        x = torch.cat([rubin_feat, vis_feat, rubin_feat - vis_feat, coarse_dra, coarse_ddec], dim=1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Main head
# ---------------------------------------------------------------------------

class AstrometryConcordanceHead(nn.Module):
    """
    Predict astrometric offset field (ΔRA*, ΔDec) via direct regression.

    Two parallel branches on concat(rubin_tokens, vis_tokens, diff):
      - Global: pool → MLP → single (Δy, Δx) for uniform shifts
      - Local: CNN → per-position (Δy, Δx) for spatial variation

    Final = global + local, converted to arcseconds.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        global_hidden: int = 128,
        local_hidden: int = 64,
        local_depth: int = 5,
        patch_size: int = 16,
        use_stem_refine: bool = False,
        stem_channels: int = 64,
        stem_hidden: int = 32,
        stem_depth: int = 4,
        stem_stride: int = 4,
        # Accept old args for CLI compat (ignored internally).
        search_radius: int = 3,
        softmax_temp: float = 0.1,
        refine_hidden: int = 32,
        refine_depth: int = 4,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_stem_refine = bool(use_stem_refine)
        self.stem_stride = max(1, int(stem_stride))

        feat_dim = 3 * embed_dim  # concat(rubin, vis, rubin-vis)

        self.global_branch = GlobalOffsetBranch(feat_dim, hidden=global_hidden)
        self.local_branch = LocalOffsetBranch(feat_dim, hidden=local_hidden, depth=local_depth)

        self.stem_refine = None
        if self.use_stem_refine:
            self.stem_refine = StemResidualRefineNet(
                stem_channels=stem_channels,
                hidden=stem_hidden,
                depth=stem_depth,
            )

    def forward(
        self,
        rubin_tokens: torch.Tensor,
        vis_tokens: torch.Tensor,
        rubin_grid: Tuple[int, int],
        vis_grid: Tuple[int, int],
        vis_image_hw: Tuple[int, int],
        vis_pixel_scale: float = 0.1,
        mesh_step: Optional[int] = None,
        rubin_stem: Optional[torch.Tensor] = None,
        vis_stem: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # --- Common token grid ---
        common_grid = (max(rubin_grid[0], vis_grid[0]), max(rubin_grid[1], vis_grid[1]))
        Hc, Wc = int(common_grid[0]), int(common_grid[1])

        rubin_common = interpolate_tokens(rubin_tokens, rubin_grid, common_grid)
        vis_common = interpolate_tokens(vis_tokens, vis_grid, common_grid)

        B, N, D = rubin_common.shape
        rubin_2d = rubin_common.transpose(1, 2).contiguous().view(B, D, Hc, Wc)
        vis_2d = vis_common.transpose(1, 2).contiguous().view(B, D, Hc, Wc)

        # --- Combined features ---
        features = torch.cat([rubin_2d, vis_2d, rubin_2d - vis_2d], dim=1)

        # --- Two branches ---
        global_offset = self.global_branch(features)    # [B, 2, 1, 1]
        local_offset = self.local_branch(features)      # [B, 2, Hc, Wc]
        combined = global_offset + local_offset          # [B, 2, Hc, Wc]

        # --- Token units → arcseconds ---
        H_vis, W_vis = int(vis_image_hw[0]), int(vis_image_hw[1])
        sky_per_token_y = (H_vis * vis_pixel_scale) / Hc
        sky_per_token_x = (W_vis * vis_pixel_scale) / Wc

        dra = combined[:, 1:2] * sky_per_token_x
        ddec = combined[:, 0:1] * sky_per_token_y

        # --- Optional stem refinement ---
        stem_residual = None
        if self.use_stem_refine and self.stem_refine is not None:
            if rubin_stem is None or vis_stem is None:
                raise ValueError("use_stem_refine=True requires rubin_stem and vis_stem.")
            refine_h = max(1, H_vis // self.stem_stride)
            refine_w = max(1, W_vis // self.stem_stride)
            rubin_refine = F.interpolate(rubin_stem, size=(refine_h, refine_w), mode="bilinear", align_corners=False)
            vis_refine = F.interpolate(vis_stem, size=(refine_h, refine_w), mode="bilinear", align_corners=False)
            coarse_dra = F.interpolate(dra, size=(refine_h, refine_w), mode="bilinear", align_corners=False)
            coarse_ddec = F.interpolate(ddec, size=(refine_h, refine_w), mode="bilinear", align_corners=False)
            stem_residual = self.stem_refine(rubin_refine, vis_refine, coarse_dra, coarse_ddec)
            dra = coarse_dra + stem_residual[:, 0:1]
            ddec = coarse_ddec + stem_residual[:, 1:2]

        # --- Optional mesh upsampling ---
        if mesh_step is not None and mesh_step > 0:
            mesh_h = max(1, H_vis // mesh_step)
            mesh_w = max(1, W_vis // mesh_step)
            dra = F.interpolate(dra, size=(mesh_h, mesh_w), mode="bilinear", align_corners=False)
            ddec = F.interpolate(ddec, size=(mesh_h, mesh_w), mode="bilinear", align_corners=False)

        return {
            "dra": dra,
            "ddec": ddec,
            "global_offset_tokens": global_offset,
            "local_offset_tokens": local_offset,
            "stem_residual": stem_residual,
            # Compat keys for preview/loss code.
            "raw_dy": global_offset[:, 0:1],
            "raw_dx": global_offset[:, 1:2],
            "confidence": torch.ones(B, 1, Hc, Wc, device=dra.device),
        }
