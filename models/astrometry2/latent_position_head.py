"""Latent-space canonical position head for per-object multi-band alignment.

Uses the frozen V7 foundation model encoder to extract multi-scale features
at each source position, then predicts a refined canonical position offset.

The key insight: the V7 bottleneck (post-transformer, 0.8"/px) encodes
chromatically-informed cross-band structure from all 10 bands, while the
VIS stem features (0.1"/px) provide high-resolution spatial detail.
Combining both scales enables per-object alignment that accounts for
chromatic effects (DCR, color-dependent morphology) while retaining
sub-pixel spatial precision.

Training target: offset from approximate detection position to VIS PSF-fit
centroid (the highest-resolution, most precise centroid available).

Architecture:

    Full tile (10 bands)
          │
    ┌─────▼──────┐
    │ Frozen V7   │
    │  Encoder    │──── bottleneck [1, 256, ~132, ~132]  @ 0.8"/px
    │             │──── VIS stem   [1,  64, ~1084,~1084] @ 0.1"/px
    └─────────────┘
          │
    For each source at position (x, y) in VIS pixels:
          │
    ┌─────▼──────┐   ┌───────▼────────┐
    │ Bottleneck  │   │  VIS stem      │
    │ 5×5 window  │   │ 17×17 window   │
    │ → ConvNeXt  │   │ → 2×Conv3      │
    │ → pool→128d │   │ → pool → 64d   │
    └──────┬──────┘   └───────┬────────┘
           └────────┬─────────┘
                    ▼
             MLP (192 → 128 → 3)
                    │
            (dx_px, dy_px, log_σ)
                    │
                Jacobian
                    ▼
            (dRA*, dDec, σ) arcsec
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jaisp_foundation_v7 import JAISPFoundationV7, ALL_BANDS
from jaisp_foundation_v6 import ConvNeXtBlock


# ============================================================
# Feature extraction utilities
# ============================================================

def extract_local_windows(
    feature_map: torch.Tensor,
    positions: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Extract local windows around each position via bilinear grid_sample.

    Parameters
    ----------
    feature_map : [1, C, H, W]
        Single-tile feature map.
    positions : [N, 2]
        (x, y) positions in feature-map pixel coordinates.
    window_size : int
        Odd integer, side length of the extracted window.

    Returns
    -------
    [N, C, window_size, window_size]
    """
    _, C, H, W = feature_map.shape
    N = positions.shape[0]
    half = window_size // 2

    offsets = torch.arange(
        -half, half + 1, device=positions.device, dtype=positions.dtype,
    )
    gy, gx = torch.meshgrid(offsets, offsets, indexing='ij')  # [ws, ws]

    # Broadcast source positions onto each grid cell.
    gx = gx[None].expand(N, -1, -1) + positions[:, 0:1, None]  # [N, ws, ws]
    gy = gy[None].expand(N, -1, -1) + positions[:, 1:2, None]

    # Normalize to [-1, 1] for grid_sample (align_corners=True).
    gx = 2.0 * gx / max(W - 1, 1) - 1.0
    gy = 2.0 * gy / max(H - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)  # [N, ws, ws, 2]

    fm = feature_map.expand(N, -1, -1, -1)
    return F.grid_sample(
        fm, grid, mode='bilinear', padding_mode='zeros', align_corners=True,
    )


def vis_px_to_bottleneck_px(
    positions_vis: torch.Tensor,
    vis_pixel_scale: float,
    fused_pixel_scale: float,
    fused_hw: Tuple[int, int],
    vis_hw: Tuple[int, int],
) -> torch.Tensor:
    """Convert VIS pixel coordinates to bottleneck feature-map coordinates.

    The bottleneck covers the same angular extent as the input images.
    Its spatial size is  input_arcsec / fused_pixel_scale.  We linearly
    map VIS pixel coords into that grid.
    """
    # Angular extent of the VIS image.
    vis_h, vis_w = vis_hw
    bn_h, bn_w = fused_hw

    # Scale factors: VIS pixels → bottleneck pixels.
    sx = bn_w / vis_w
    sy = bn_h / vis_h
    scale = torch.tensor([sx, sy], device=positions_vis.device, dtype=positions_vis.dtype)
    return positions_vis * scale


# ============================================================
# Latent Position Head
# ============================================================

class LatentPositionHead(nn.Module):
    """Predict canonical per-object position from frozen V7 encoder features.

    Combines:
      1. Fused bottleneck features (0.8"/px, 256 ch) — cross-band context,
         capturing chromatic morphology from the full 10-band representation.
      2. VIS BandStem features (0.1"/px, 64 ch) — high-resolution spatial
         detail for precise centroiding.

    For each source at an approximate position (x, y) in VIS pixels:
      - Extract a 5×5 window from the bottleneck → ConvNeXt + pool → 128-d
      - Extract a 17×17 window from VIS stem    → 2×Conv3  + pool →  64-d
      - Concatenate 192-d → MLP → (dx_px, dy_px, log_σ)
      - Convert pixel offset to (ΔRA*, ΔDec) arcsec via local Jacobian

    The head predicts in VIS pixel space and converts to sky at the end,
    matching the existing matcher convention.
    """

    def __init__(
        self,
        hidden_ch: int = 256,
        stem_ch: int = 64,
        bottleneck_out: int = 128,
        stem_out: int = 64,
        mlp_hidden: int = 128,
        bottleneck_window: int = 5,
        stem_window: int = 17,
        fused_pixel_scale: float = 0.8,
        vis_pixel_scale: float = 0.1,
    ):
        super().__init__()
        self.fused_pixel_scale = fused_pixel_scale
        self.vis_pixel_scale = vis_pixel_scale
        self.bottleneck_window = bottleneck_window
        self.stem_window = stem_window

        # ---- Bottleneck path: cross-band context --------------------------
        self.bn_conv = nn.Sequential(
            ConvNeXtBlock(hidden_ch),
            nn.Conv2d(hidden_ch, bottleneck_out, 1),
            nn.GELU(),
        )

        # ---- VIS stem path: fine spatial detail ---------------------------
        self.stem_conv = nn.Sequential(
            nn.Conv2d(stem_ch, stem_out, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_out, stem_out, 3, padding=1),
            nn.GELU(),
        )

        # ---- MLP head -----------------------------------------------------
        feat_dim = bottleneck_out + stem_out
        self.head = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 3),  # dx_px, dy_px, log_sigma
        )
        # Initialize near zero: predict no offset initially.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            self.head[-1].bias[2] = math.log(0.05)  # 50 mas initial σ

        # ---- Gaussian center weights for spatial pooling ------------------
        for name, ws in [('bn', bottleneck_window), ('stem', stem_window)]:
            y = torch.linspace(-1, 1, ws)
            x = torch.linspace(-1, 1, ws)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            g = torch.exp(-(xx ** 2 + yy ** 2) / (2 * 0.4 ** 2))
            g = g / g.sum()
            self.register_buffer(f'{name}_gauss', g.view(1, 1, ws, ws), persistent=False)

    def _gauss_pool(self, features: torch.Tensor, gauss: torch.Tensor) -> torch.Tensor:
        """Gaussian-weighted spatial pooling → [N, C]."""
        return (features * gauss).sum(dim=(-2, -1))

    def forward(
        self,
        bottleneck: torch.Tensor,
        vis_stem_features: torch.Tensor,
        source_positions_vis: torch.Tensor,
        pixel_to_sky: torch.Tensor,
        fused_hw: Tuple[int, int],
        vis_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Predict canonical position offsets for detected sources.

        Parameters
        ----------
        bottleneck : [1, hidden_ch, H_bn, W_bn]
            Fused transformer bottleneck from V7 encoder.
        vis_stem_features : [1, stem_ch, H_vis, W_vis]
            Raw VIS BandStem output (before stream fusion).
        source_positions_vis : [N, 2]
            Approximate source (x, y) in VIS pixel coordinates.
        pixel_to_sky : [N, 2, 2]
            Local Jacobian per source (VIS pixels → arcsec).
        fused_hw : (H_bn, W_bn)
            Spatial size of the bottleneck feature map.
        vis_hw : (H_vis, W_vis)
            Spatial size of the VIS image.

        Returns
        -------
        dict with keys:
            pred_offset_arcsec : [N, 2]  (ΔRA*, ΔDec) in arcsec
            dx_px, dy_px       : [N]     offset in VIS pixels
            log_sigma          : [N]     log predicted uncertainty (arcsec)
            confidence         : [N]     1/σ (higher = more confident)
        """
        N = source_positions_vis.shape[0]

        # --- Bottleneck features ---
        positions_bn = vis_px_to_bottleneck_px(
            source_positions_vis, self.vis_pixel_scale, self.fused_pixel_scale,
            fused_hw, vis_hw,
        )
        bn_windows = extract_local_windows(
            bottleneck, positions_bn, self.bottleneck_window,
        )  # [N, hidden_ch, bn_win, bn_win]
        bn_feat = self.bn_conv(bn_windows)
        bn_vec = self._gauss_pool(bn_feat, self.bn_gauss)  # [N, bottleneck_out]

        # --- VIS stem features ---
        stem_windows = extract_local_windows(
            vis_stem_features, source_positions_vis, self.stem_window,
        )  # [N, stem_ch, stem_win, stem_win]
        stem_feat = self.stem_conv(stem_windows)
        stem_vec = self._gauss_pool(stem_feat, self.stem_gauss)  # [N, stem_out]

        # --- Predict offset ---
        combined = torch.cat([bn_vec, stem_vec], dim=1)  # [N, feat_dim]
        out = self.head(combined)  # [N, 3]

        dx_px = out[:, 0]
        dy_px = out[:, 1]
        log_sigma = out[:, 2].clamp(-6.0, 3.0)

        # Convert pixel offset to sky via Jacobian.
        pix = torch.stack([dx_px, dy_px], dim=1).unsqueeze(-1)  # [N, 2, 1]
        pred_sky = torch.bmm(pixel_to_sky, pix).squeeze(-1)       # [N, 2]

        sigma = torch.exp(log_sigma)
        return {
            'pred_offset_arcsec': pred_sky,
            'dx_px': dx_px,
            'dy_px': dy_px,
            'log_sigma': log_sigma,
            'confidence': 1.0 / sigma.clamp_min(1e-4),
        }


# ============================================================
# Frozen encoder wrapper (runs once per tile)
# ============================================================

class FrozenV7Encoder(nn.Module):
    """Wrapper that runs the frozen V7 encoder and extracts the VIS stem.

    Call ``encode_tile()`` once per tile to get:
      - bottleneck  [1, 256, H_bn, W_bn]
      - vis_stem    [1, 64, H_vis, W_vis]
      - fused_hw    (H_bn, W_bn)

    All parameters are frozen; no gradients flow through this module.
    """

    def __init__(self, v7_model: JAISPFoundationV7):
        super().__init__()
        self.encoder = v7_model.encoder
        self.vis_stem = v7_model.encoder.stems['euclid_VIS']
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_tile(
        self,
        context_images: Dict[str, torch.Tensor],
        context_rms: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run full encoder + extract raw VIS stem features.

        Parameters
        ----------
        context_images : {band_name: [1, 1, H, W]} for all available bands
        context_rms    : {band_name: [1, 1, H, W]} matching RMS maps

        Returns
        -------
        dict with bottleneck, vis_stem, fused_hw, vis_hw
        """
        enc_out = self.encoder(context_images, context_rms)

        vis_img = context_images['euclid_VIS']
        vis_rms = context_rms['euclid_VIS']
        vis_stem = self.vis_stem(vis_img, vis_rms)  # [1, stem_ch, H_vis, W_vis]

        return {
            'bottleneck': enc_out['bottleneck'],
            'vis_stem': vis_stem,
            'fused_hw': enc_out['fused_hw'],
            'vis_hw': (vis_img.shape[-2], vis_img.shape[-1]),
        }


# ============================================================
# Factory
# ============================================================

def load_latent_position_head(
    v7_checkpoint: str,
    device: torch.device = None,
    bottleneck_out: int = 128,
    stem_out: int = 64,
    mlp_hidden: int = 128,
    bottleneck_window: int = 5,
    stem_window: int = 17,
) -> Tuple[FrozenV7Encoder, LatentPositionHead]:
    """Load frozen V7 encoder + fresh LatentPositionHead.

    Returns
    -------
    (frozen_encoder, head)
        frozen_encoder : FrozenV7Encoder  (all params frozen)
        head           : LatentPositionHead (trainable)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(v7_checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})

    v7 = JAISPFoundationV7(
        band_names=cfg.get('band_names', ALL_BANDS),
        stem_ch=cfg.get('stem_ch', 64),
        hidden_ch=cfg.get('hidden_ch', 256),
        blocks_per_stage=cfg.get('blocks_per_stage', 2),
        transformer_depth=cfg.get('transformer_depth', 4),
        transformer_heads=cfg.get('transformer_heads', 8),
        fused_pixel_scale_arcsec=cfg.get('fused_pixel_scale_arcsec', 0.8),
    )
    missing, unexpected = v7.load_state_dict(ckpt['model'], strict=False)
    enc_missing = [k for k in missing if not k.startswith(('encoder.skip_projs', 'target_decoders'))]
    if enc_missing:
        print(f'  [warn] Missing encoder keys: {enc_missing}')

    hidden_ch = cfg.get('hidden_ch', 256)
    stem_ch = cfg.get('stem_ch', 64)
    fused_scale = cfg.get('fused_pixel_scale_arcsec', 0.8)

    frozen_encoder = FrozenV7Encoder(v7).to(device)

    head = LatentPositionHead(
        hidden_ch=hidden_ch,
        stem_ch=stem_ch,
        bottleneck_out=bottleneck_out,
        stem_out=stem_out,
        mlp_hidden=mlp_hidden,
        bottleneck_window=bottleneck_window,
        stem_window=stem_window,
        fused_pixel_scale=fused_scale,
        vis_pixel_scale=0.1,
    ).to(device)

    n_frozen = sum(p.numel() for p in frozen_encoder.parameters())
    n_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f'LatentPositionHead: {n_trainable/1e6:.2f}M trainable, '
          f'{n_frozen/1e6:.1f}M frozen encoder')

    return frozen_encoder, head
