"""
Astrometry Concordance Head for JAISP.

Predicts a smooth offset field (ΔRA*, ΔDec) in arcseconds that maps
Rubin astrometry onto the Euclid VIS reference frame.

Architecture:
  1. Encode Rubin band and Euclid VIS through frozen backbone → token grids
  2. Interpolate both to a common token grid
  3. Compute local cross-correlation volume between the two token sets
  4. Soft-argmax → raw sub-token offsets
  5. Learned refinement CNN → smooth offset field in arcseconds
  6. Optionally upsample to desired mesh resolution (DSTEP)

Design follows the concordance data product specification:
  - ΔRA*, ΔDec in arcseconds as function of position
  - Applied to Rubin sky coordinates
  - Perfect alignment ⇒ Δ = 0
  - VIS is the reference grid
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
# Correlation engine
# ---------------------------------------------------------------------------

def compute_correlation_volume(
    query: torch.Tensor,
    reference: torch.Tensor,
    search_radius: int,
) -> torch.Tensor:
    """
    Local cross-correlation between query and reference token grids.

    Args:
        query:     [B, D, H, W]  (e.g. Rubin tokens in spatial layout)
        reference: [B, D, H, W]  (e.g. VIS tokens in spatial layout)
        search_radius: search ±r positions in each direction

    Returns:
        [B, S², H, W] correlation volume, S = 2*search_radius + 1
        Each position (i,j) holds cosine similarities with its (2r+1)² neighborhood.
    """
    r = search_radius
    S = 2 * r + 1

    q = F.normalize(query, dim=1)
    ref = F.normalize(reference, dim=1)

    # Pad reference so we can extract full neighborhoods at every position.
    ref_padded = F.pad(ref, [r, r, r, r], mode="replicate")

    # Collect correlation at each offset.
    B, D, H, W = q.shape
    corr_slices = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            # Extract the reference patch shifted by (dy, dx).
            ref_shift = ref_padded[:, :, r + dy : r + dy + H, r + dx : r + dx + W]
            # Cosine similarity per spatial position.
            sim = (q * ref_shift).sum(dim=1, keepdim=True)  # [B, 1, H, W]
            corr_slices.append(sim)

    return torch.cat(corr_slices, dim=1)  # [B, S², H, W]


def soft_argmax_2d(
    corr_volume: torch.Tensor,
    search_radius: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable sub-pixel peak finding via soft-argmax.

    Args:
        corr_volume: [B, S², H, W]
        search_radius: r
        temperature: sharpness (lower = sharper peak, more like hard argmax)

    Returns:
        offsets: [B, 2, H, W] — (Δy, Δx) in token-grid units
        confidence: [B, 1, H, W] — peak sharpness (max weight)
    """
    r = search_radius
    S = 2 * r + 1
    device = corr_volume.device
    dtype = corr_volume.dtype

    coords = torch.arange(-r, r + 1, device=device, dtype=dtype)
    cy, cx = torch.meshgrid(coords, coords, indexing="ij")
    cy = cy.reshape(S * S, 1, 1)  # [S², 1, 1]
    cx = cx.reshape(S * S, 1, 1)

    weights = F.softmax(corr_volume / temperature, dim=1)  # [B, S², H, W]

    dy = (weights * cy).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    dx = (weights * cx).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    confidence = weights.max(dim=1, keepdim=True)[0]

    return torch.cat([dy, dx], dim=1), confidence


# ---------------------------------------------------------------------------
# Refinement network
# ---------------------------------------------------------------------------

class SmoothRefineNet(nn.Module):
    """
    Small CNN that refines raw correlation offsets into a smooth field.

    Input:  [B, 2 + S² + 1, H, W]  (raw offsets + correlation volume + confidence)
    Output: [B, 2, H, W]  residual correction

    The large kernel sizes enforce spatial smoothness — astrometric distortions
    are low-frequency by nature.
    """

    def __init__(self, corr_channels: int, hidden: int = 32, depth: int = 4):
        super().__init__()
        in_ch = 2 + corr_channels + 1  # offsets + corr volume + confidence
        layers = [nn.Conv2d(in_ch, hidden, 5, padding=2), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(hidden, hidden, 5, padding=2), nn.GELU()]
        layers += [nn.Conv2d(hidden, 2, 5, padding=2)]
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize last layer to near-zero so initial output ≈ raw offsets.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, raw_offsets, corr_volume, confidence):
        x = torch.cat([raw_offsets, corr_volume, confidence], dim=1)
        return self.net(x)


class StemResidualRefineNet(nn.Module):
    """
    Refine coarse offsets using stem features at a higher-resolution grid.

    Inputs:
      - rubin_feat: [B, C, H, W] Rubin stem features (resampled to common grid)
      - vis_feat:   [B, C, H, W] VIS stem features (resampled to common grid)
      - coarse_dra: [B, 1, H, W] coarse ΔRA* in arcsec
      - coarse_ddec:[B, 1, H, W] coarse ΔDec in arcsec

    Output:
      - residual: [B, 2, H, W] additive correction in arcsec
    """

    def __init__(self, stem_channels: int = 64, hidden: int = 32, depth: int = 4):
        super().__init__()
        in_ch = 3 * stem_channels + 2  # rubin, vis, difference, coarse dra/ddec
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
        # Keep initial behavior close to coarse prediction.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        rubin_feat: torch.Tensor,
        vis_feat: torch.Tensor,
        coarse_dra: torch.Tensor,
        coarse_ddec: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([rubin_feat, vis_feat, rubin_feat - vis_feat, coarse_dra, coarse_ddec], dim=1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Main head
# ---------------------------------------------------------------------------

class AstrometryConcordanceHead(nn.Module):
    """
    Predict astrometric offset field (ΔRA*, ΔDec) from a pair of
    (Rubin band, Euclid VIS) token embeddings.

    Output is in arcseconds on a spatial grid. The grid can optionally
    be upsampled to a finer mesh for the FITS concordance product.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        search_radius: int = 3,
        softmax_temp: float = 0.1,
        refine_hidden: int = 32,
        refine_depth: int = 4,
        patch_size: int = 16,
        learnable_temperature: bool = True,
        use_stem_refine: bool = False,
        stem_channels: int = 64,
        stem_hidden: int = 32,
        stem_depth: int = 4,
        stem_stride: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.search_radius = search_radius
        self.patch_size = patch_size
        self.use_stem_refine = bool(use_stem_refine)
        self.stem_stride = max(1, int(stem_stride))
        S = 2 * search_radius + 1

        # Learnable temperature for soft-argmax.
        if learnable_temperature:
            self.log_temp = nn.Parameter(torch.tensor(math.log(max(softmax_temp, 1e-6))))
        else:
            self.register_buffer("log_temp", torch.tensor(math.log(max(softmax_temp, 1e-6))))

        # Refinement network.
        self.refine = SmoothRefineNet(
            corr_channels=S * S,
            hidden=refine_hidden,
            depth=refine_depth,
        )
        self.stem_refine = None
        if self.use_stem_refine:
            self.stem_refine = StemResidualRefineNet(
                stem_channels=stem_channels,
                hidden=stem_hidden,
                depth=stem_depth,
            )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp()

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
        """
        Args:
            rubin_tokens: [B, N_r, D] encoder tokens from a Rubin band
            vis_tokens:   [B, N_v, D] encoder tokens from Euclid VIS
            rubin_grid:   (Hr, Wr) Rubin token grid size
            vis_grid:     (Hv, Wv) VIS token grid size
            vis_image_hw: (H_pix, W_pix) VIS native pixel dimensions
            vis_pixel_scale: arcsec/pixel for VIS (default 0.1)
            mesh_step:    if set, upsample output to this mesh spacing in VIS pixels
                          (e.g. mesh_step=8 → output every 8 VIS pixels)
            rubin_stem: optional Rubin stem feature map [B, C, H_r, W_r]
            vis_stem: optional VIS stem feature map [B, C, H_v, W_v]

        Returns:
            dict with:
                dra:  [B, 1, H_out, W_out] ΔRA*  in arcseconds
                ddec: [B, 1, H_out, W_out] ΔDec  in arcseconds
                raw_dy, raw_dx: [B, 1, ...] raw offsets in token units (before refinement)
                confidence: [B, 1, ...] peak sharpness
                corr_volume: [B, S², ...] full correlation volume (for diagnostics)
        """
        # --- Common token grid ---
        common_grid = (max(rubin_grid[0], vis_grid[0]), max(rubin_grid[1], vis_grid[1]))
        Hc, Wc = int(common_grid[0]), int(common_grid[1])

        rubin_common = interpolate_tokens(rubin_tokens, rubin_grid, common_grid)
        vis_common = interpolate_tokens(vis_tokens, vis_grid, common_grid)

        # Reshape to spatial: [B, D, Hc, Wc]
        B, N, D = rubin_common.shape
        q = rubin_common.transpose(1, 2).contiguous().view(B, D, Hc, Wc)
        k = vis_common.transpose(1, 2).contiguous().view(B, D, Hc, Wc)

        # --- Cross-correlation ---
        corr = compute_correlation_volume(q, k, self.search_radius)   # [B, S², Hc, Wc]

        # --- Soft-argmax → raw offsets in token units ---
        raw_offsets, confidence = soft_argmax_2d(
            corr, self.search_radius, self.temperature
        )  # [B, 2, Hc, Wc], [B, 1, Hc, Wc]

        # --- Refinement ---
        residual = self.refine(raw_offsets, corr, confidence)         # [B, 2, Hc, Wc]
        refined_offsets = raw_offsets + residual                       # [B, 2, Hc, Wc]

        # --- Convert from token units to arcseconds ---
        # Each token on the common grid covers:
        #   sky_per_token = vis_image_size * vis_pixel_scale / common_grid_size
        H_vis, W_vis = int(vis_image_hw[0]), int(vis_image_hw[1])
        sky_per_token_y = (H_vis * vis_pixel_scale) / Hc
        sky_per_token_x = (W_vis * vis_pixel_scale) / Wc

        dra = refined_offsets[:, 1:2] * sky_per_token_x    # dx → ΔRA*
        ddec = refined_offsets[:, 0:1] * sky_per_token_y    # dy → ΔDec

        # --- Optional high-resolution residual refinement in stem feature space ---
        stem_residual = None
        if self.use_stem_refine:
            if rubin_stem is None or vis_stem is None:
                raise ValueError("use_stem_refine=True requires rubin_stem and vis_stem inputs.")

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
            "raw_dy": raw_offsets[:, 0:1],
            "raw_dx": raw_offsets[:, 1:2],
            "confidence": confidence,
            "corr_volume": corr,
            "refined_offsets_tokens": refined_offsets,
            "stem_residual": stem_residual,
        }
