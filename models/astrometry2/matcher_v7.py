"""Astrometry matcher using the V7 mixed-resolution MAE backbone.

Drop-in replacement for matcher_v6.V6AstrometryMatcher.  The only
architectural change is how frozen BandStem weights are loaded:

  V6:  v6_model.encoder.stems[band].net   (JAISPEncoderV6 wrapper)
  V7:  v7_model.stems[band].net           (stems live directly on JAISPFoundationV7)

Everything downstream (cost volume, soft-argmax, MLP head, Jacobian) is
identical — the API matches V6AstrometryMatcher exactly so it can be
swapped into train_astro_v6.py / infer_concordance.py with no other
code changes.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jaisp_foundation_v7 import JAISPFoundationV7, ALL_BANDS, RUBIN_BANDS
from jaisp_foundation_v6 import ConvNeXtBlock

RUBIN_BAND_ORDER = ['u', 'g', 'r', 'i', 'z', 'y']


# ============================================================
# V7 Rubin Encoder (frozen stems + trainable adapter)
# ============================================================

class V7RubinEncoder(nn.Module):
    """Per-band frozen V7 BandStem CNNs → mean → optional stream stages → ConvNeXt adapter.

    Input : [B, n_bands, H, W]  pre-noise-normalized patches
    Output: [B, out_ch, H', W']  where H'=H/2^n_stream_stages

    Parameters
    ----------
    n_stream_stages : int
        Number of frozen V7 Rubin stream encoder stages to apply after the
        stem mean.  0 = stem only (original behavior).  Each stage halves
        spatial dims and enriches features via ConvNeXt blocks.  The Rubin
        stream has 2 stages; values > 2 are clamped.
    """

    def __init__(
        self,
        v7_model: JAISPFoundationV7,
        n_input_bands: int = 6,
        out_ch: int = 64,
        n_adapter_blocks: int = 2,
        freeze_stems: bool = True,
        n_stream_stages: int = 0,
    ):
        super().__init__()
        self.n_input_bands = n_input_bands
        encoder = v7_model.encoder if hasattr(v7_model, 'encoder') else v7_model
        stem_ch = encoder.stem_ch
        stems = encoder.stems

        band_names = [f'rubin_{b}' for b in RUBIN_BAND_ORDER[:n_input_bands]]
        self.band_stems = nn.ModuleList([
            stems[name].net  # nn.Sequential: Conv→GN→GELU→Conv→GN→GELU
            for name in band_names
        ])
        # Full BandStem objects (with image/rms normalization) for per-pixel RMS mode.
        self.band_stems_full = nn.ModuleList([stems[name] for name in band_names])

        # Optional frozen stream encoder stages for deeper features.
        n_stream_stages = min(int(max(0, n_stream_stages)), 2)
        self.stream_stages = nn.ModuleList()
        if n_stream_stages > 0:
            rubin_stream = encoder.stream_encoders['rubin']
            for i in range(n_stream_stages):
                self.stream_stages.append(rubin_stream.stages[i])
        adapter_in_ch = encoder.hidden_ch if n_stream_stages > 0 else stem_ch

        if freeze_stems:
            for stem in self.band_stems:
                for p in stem.parameters():
                    p.requires_grad = False
            for stage in self.stream_stages:
                for p in stage.parameters():
                    p.requires_grad = False

        adapter: list = [ConvNeXtBlock(adapter_in_ch) for _ in range(n_adapter_blocks)]
        if out_ch != adapter_in_ch:
            adapter.append(nn.Conv2d(adapter_in_ch, out_ch, kernel_size=1))
        self.adapter = nn.Sequential(*adapter)

    def forward(self, rubin_patch: torch.Tensor, rubin_rms: torch.Tensor = None) -> torch.Tensor:
        n = min(rubin_patch.shape[1], len(self.band_stems))
        if rubin_rms is not None:
            # Per-pixel RMS path: use full BandStem (image/rms normalization).
            feats = [
                self.band_stems_full[i](rubin_patch[:, i:i+1, :, :], rubin_rms[:, i:i+1, :, :])
                for i in range(n)
            ]
        else:
            # Legacy path: patches are already scalar-normalized by _normalize_patch.
            feats = [
                self.band_stems[i](rubin_patch[:, i:i+1, :, :])
                for i in range(n)
            ]
        x = torch.stack(feats, dim=0).mean(dim=0)
        for stage in self.stream_stages:
            x = stage(x)
        return self.adapter(x)


# ============================================================
# V7 VIS Encoder (frozen stem + trainable adapter)
# ============================================================

class V7VISEncoder(nn.Module):
    """Frozen V7 euclid_VIS BandStem CNN → optional stream stages → ConvNeXt adapter.

    Input : [B, 1, H, W]  pre-noise-normalized VIS patch
    Output: [B, out_ch, H', W']  where H'=H/2^n_stream_stages

    Parameters
    ----------
    n_stream_stages : int
        Number of frozen V7 VIS stream encoder stages to apply after the
        stem.  0 = stem only (original behavior).  Each stage halves spatial
        dims and enriches features.  The VIS stream has 3 stages; values > 3
        are clamped.  Using 1 stage on a 33×33 patch gives 16×16 features —
        still large enough for the cost volume while capturing richer
        cross-channel mixing that is the key V7 improvement over V6.
    """

    def __init__(
        self,
        v7_model: JAISPFoundationV7,
        out_ch: int = 64,
        n_adapter_blocks: int = 2,
        freeze_stem: bool = True,
        n_stream_stages: int = 0,
    ):
        super().__init__()
        encoder = v7_model.encoder if hasattr(v7_model, 'encoder') else v7_model
        stem_ch = encoder.stem_ch
        stems = encoder.stems

        self.vis_stem = stems['euclid_VIS'].net
        self.vis_stem_full = stems['euclid_VIS']  # Full BandStem with image/rms norm.

        # Optional frozen stream encoder stages for deeper features.
        n_stream_stages = min(int(max(0, n_stream_stages)), 3)
        self.stream_stages = nn.ModuleList()
        if n_stream_stages > 0:
            vis_stream = encoder.stream_encoders['vis']
            for i in range(n_stream_stages):
                self.stream_stages.append(vis_stream.stages[i])
        adapter_in_ch = encoder.hidden_ch if n_stream_stages > 0 else stem_ch

        if freeze_stem:
            for p in self.vis_stem.parameters():
                p.requires_grad = False
            for stage in self.stream_stages:
                for p in stage.parameters():
                    p.requires_grad = False

        adapter: list = [ConvNeXtBlock(adapter_in_ch) for _ in range(n_adapter_blocks)]
        if out_ch != adapter_in_ch:
            adapter.append(nn.Conv2d(adapter_in_ch, out_ch, kernel_size=1))
        self.adapter = nn.Sequential(*adapter)

    def forward(self, vis_patch: torch.Tensor, vis_rms: torch.Tensor = None) -> torch.Tensor:
        if vis_rms is not None:
            x = self.vis_stem_full(vis_patch, vis_rms)
        else:
            x = self.vis_stem(vis_patch)
        for stage in self.stream_stages:
            x = stage(x)
        return self.adapter(x)


# ============================================================
# V7 Astrometry Matcher
# ============================================================

class V7AstrometryMatcher(nn.Module):
    """Drop-in replacement for V6AstrometryMatcher using V7 stems.

    Rubin encoder  : V7RubinEncoder — frozen V7 Rubin BandStems + trainable adapter
    VIS encoder    : V7VISEncoder   — frozen V7 euclid_VIS BandStem + trainable adapter
    Cost volume    : spatially-weighted cross-correlation
    Soft-argmax    : differentiable coarse offset
    MLP refinement : coarse + pooled features → (dx, dy, log_sigma)
    Band embedding : learned per-band vector for wavelength-specific correction
    Jacobian       : pixel shift → sky arcsec

    API is identical to V6AstrometryMatcher.
    """

    def __init__(
        self,
        v7_model: JAISPFoundationV7,
        n_rubin_bands: int = 6,
        hidden_channels: int = 64,
        n_adapter_blocks: int = 2,
        freeze_stems: bool = True,
        search_radius: int = 3,
        softmax_temp: float = 0.05,
        mlp_hidden: int = 128,
        n_target_bands: int = 6,
        band_embed_dim: int = 16,
        n_stream_stages: int = 0,
    ):
        super().__init__()
        self.search_radius = int(max(0, search_radius))
        self.n_target_bands = int(max(1, n_target_bands))
        self.band_embed_dim = int(band_embed_dim) if n_target_bands > 1 else 0

        # ---- Encoders ------------------------------------------------------
        self.rubin_encoder = V7RubinEncoder(
            v7_model=v7_model,
            n_input_bands=n_rubin_bands,
            out_ch=hidden_channels,
            n_adapter_blocks=n_adapter_blocks,
            freeze_stems=freeze_stems,
            n_stream_stages=n_stream_stages,
        )
        self.vis_encoder = V7VISEncoder(
            v7_model=v7_model,
            out_ch=hidden_channels,
            n_adapter_blocks=n_adapter_blocks,
            freeze_stem=freeze_stems,
            n_stream_stages=n_stream_stages,
        )

        # ---- Cost volume projection heads ----------------------------------
        self.rubin_proj = nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False)
        self.vis_proj   = nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False)

        # Learnable softmax temperature
        self._log_temp = nn.Parameter(
            torch.tensor(math.log(max(float(softmax_temp), 1e-3)))
        )

        # ---- Band embedding ------------------------------------------------
        if self.band_embed_dim > 0:
            self.band_embedding = nn.Embedding(self.n_target_bands, self.band_embed_dim)
        else:
            self.band_embedding = None

        # ---- MLP refinement head -------------------------------------------
        feat_dim = hidden_channels * 3 + 2 + self.band_embed_dim
        self.residual_head = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 3),   # dx_residual, dy_residual, log_sigma
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        with torch.no_grad():
            self.residual_head[-1].bias[2] = math.log(0.05)

        # ---- Displacement lookup tables for soft-argmax --------------------
        r = self.search_radius
        disp = [(float(dx), float(dy)) for dy in range(-r, r+1) for dx in range(-r, r+1)]
        self.register_buffer('dx_lut',
            torch.tensor([d[0] for d in disp], dtype=torch.float32).view(1, -1),
            persistent=False)
        self.register_buffer('dy_lut',
            torch.tensor([d[1] for d in disp], dtype=torch.float32).view(1, -1),
            persistent=False)

    @property
    def temperature(self) -> torch.Tensor:
        return self._log_temp.exp().clamp_min(1e-3)

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _center_weights(self, h: int, w: int, device, dtype) -> torch.Tensor:
        y = torch.linspace(-1., 1., h, device=device, dtype=dtype)
        x = torch.linspace(-1., 1., w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        g = torch.exp(-(yy**2 + xx**2) / (2 * 0.5**2))
        return (g / g.sum()).view(1, 1, h, w)

    def _weighted_cost_volume(
        self, rubin_feat: torch.Tensor, vis_feat: torch.Tensor
    ) -> torch.Tensor:
        rubin_n = F.normalize(self.rubin_proj(rubin_feat), dim=1, eps=1e-6)
        vis_n   = F.normalize(self.vis_proj(vis_feat),   dim=1, eps=1e-6)
        B, C, H, W = rubin_n.shape
        r = self.search_radius

        if r == 0:
            return (rubin_n * vis_n).sum(dim=1, keepdim=True).mean(dim=(2, 3))

        rubin_energy = (rubin_n * rubin_n).sum(dim=1, keepdim=True)
        gauss = self._center_weights(H, W, rubin_n.device, rubin_n.dtype)
        spatial_w = rubin_energy * gauss
        spatial_w = spatial_w / (spatial_w.sum(dim=(2, 3), keepdim=True) + 1e-8)
        sqrt_sw = torch.sqrt(spatial_w + 1e-10)

        rubin_w = rubin_n * sqrt_sw
        vis_pad = F.pad(vis_n, (r, r, r, r), mode='replicate')
        K = 2 * r + 1
        vis_unf = vis_pad.unfold(2, H, 1).unfold(3, W, 1).reshape(B, C, K*K, H, W)
        vis_w   = vis_unf * sqrt_sw.unsqueeze(2)

        scale = 1.0 / math.sqrt(float(max(1, C)))
        return (rubin_w.unsqueeze(2) * vis_w).sum(dim=(1, 3, 4)) * scale

    def _encode(
        self, rubin_patch: torch.Tensor, vis_patch: torch.Tensor,
        rubin_rms: torch.Tensor = None, vis_rms: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        rubin_feat = self.rubin_encoder(rubin_patch, rubin_rms)
        vis_feat   = self.vis_encoder(vis_patch, vis_rms)

        logits = self._weighted_cost_volume(rubin_feat, vis_feat)
        probs  = torch.softmax(logits / self.temperature, dim=1)
        coarse_dx = (probs * self.dx_lut[:, :probs.shape[1]]).sum(dim=1)
        coarse_dy = (probs * self.dy_lut[:, :probs.shape[1]]).sum(dim=1)

        B, C, H, W = rubin_feat.shape
        gauss    = self._center_weights(H, W, rubin_feat.device, rubin_feat.dtype)
        r_pool   = (rubin_feat * gauss).sum(dim=(2, 3))
        v_pool   = (vis_feat   * gauss).sum(dim=(2, 3))

        return {
            'rubin_pool': r_pool,
            'vis_pool':   v_pool,
            'delta_pool': r_pool - v_pool,
            'coarse_dx':  coarse_dx,
            'coarse_dy':  coarse_dy,
            'probs':      probs,
            'logits':     logits,
        }

    def _mlp_head(
        self,
        enc: Dict[str, torch.Tensor],
        pixel_to_sky: torch.Tensor,
        band_idx: Optional[torch.Tensor],
        B: int,
        device,
    ) -> Dict[str, torch.Tensor]:
        coarse = torch.stack([enc['coarse_dx'], enc['coarse_dy']], dim=1)
        parts  = [enc['rubin_pool'], enc['vis_pool'], enc['delta_pool'], coarse]

        if self.band_embedding is not None:
            if band_idx is not None:
                parts.append(self.band_embedding(band_idx.long()))
            else:
                parts.append(self.band_embedding.weight[0:1].expand(B, -1))

        out     = self.residual_head(torch.cat(parts, dim=1))
        dx_px   = enc['coarse_dx'] + out[:, 0]
        dy_px   = enc['coarse_dy'] + out[:, 1]
        log_sig = out[:, 2].clamp(min=-6., max=3.)

        pix     = torch.stack([dx_px, dy_px], dim=1).unsqueeze(-1)
        pred_sky = torch.bmm(pixel_to_sky, pix).squeeze(-1)

        return {
            'dx_px':             dx_px,
            'dy_px':             dy_px,
            'coarse_dx_px':      enc['coarse_dx'],
            'coarse_dy_px':      enc['coarse_dy'],
            'pred_offset_arcsec': pred_sky,
            'log_sigma':         log_sig,
            'confidence':        enc['probs'].max(dim=1).values,
            'temperature':       self.temperature.detach(),
            'logits':            enc['logits'],
        }

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        rubin_patch: torch.Tensor,
        vis_patch: torch.Tensor,
        pixel_to_sky: torch.Tensor,
        band_idx: Optional[torch.Tensor] = None,
        rubin_rms: Optional[torch.Tensor] = None,
        vis_rms: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc = self._encode(rubin_patch, vis_patch, rubin_rms, vis_rms)
        return self._mlp_head(enc, pixel_to_sky, band_idx, rubin_patch.shape[0], rubin_patch.device)

    def predict_all_bands(
        self,
        rubin_patch: torch.Tensor,
        vis_patch: torch.Tensor,
        pixel_to_sky: torch.Tensor,
        rubin_rms: Optional[torch.Tensor] = None,
        vis_rms: Optional[torch.Tensor] = None,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Encode once, predict for every target band."""
        B   = rubin_patch.shape[0]
        enc = self._encode(rubin_patch, vis_patch, rubin_rms, vis_rms)
        return {
            bi: self._mlp_head(
                enc, pixel_to_sky,
                torch.full((B,), bi, dtype=torch.long, device=rubin_patch.device),
                B, rubin_patch.device,
            )
            for bi in range(self.n_target_bands)
        }


# ============================================================
# Factory
# ============================================================

def load_v7_matcher(
    v7_checkpoint: str,
    device: torch.device = None,
    n_rubin_bands: int = 6,
    hidden_channels: int = 64,
    n_adapter_blocks: int = 2,
    freeze_stems: bool = True,
    search_radius: int = 3,
    n_target_bands: int = 6,
    band_embed_dim: int = 16,
    mlp_hidden: int = 128,
    n_stream_stages: int = 0,
) -> V7AstrometryMatcher:
    """Load a V7 checkpoint and wrap it in V7AstrometryMatcher.

    Reused from V7 (frozen by default):
      - Rubin BandStem CNNs  (rubin_u/g/r/i/z/y)
      - euclid_VIS BandStem  (trained cross-instrument in Phase B)
      - Optionally: per-stream ConvNeXt encoder stages (n_stream_stages > 0)

    Freshly initialized:
      - ConvNeXt adapter blocks
      - Cost-volume projections, soft-argmax, MLP refinement, band embedding
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(v7_checkpoint, map_location='cpu', weights_only=False)
    cfg  = ckpt.get('config', {})

    v7 = JAISPFoundationV7(
        band_names               = cfg.get('band_names', ALL_BANDS),
        stem_ch                  = cfg.get('stem_ch', 64),
        hidden_ch                = cfg.get('hidden_ch', 256),
        blocks_per_stage         = cfg.get('blocks_per_stage', 2),
        transformer_depth        = cfg.get('transformer_depth', 4),
        transformer_heads        = cfg.get('transformer_heads', 8),
        fused_pixel_scale_arcsec = cfg.get('fused_pixel_scale_arcsec', 0.8),
    )
    missing, unexpected = v7.load_state_dict(ckpt['model'], strict=False)
    # skip_projs and target_decoders are not needed for stem/stream extraction
    enc_missing = [k for k in missing if not k.startswith(('encoder.skip_projs', 'target_decoders'))]
    if enc_missing:
        print(f'  [warn] Missing encoder keys: {enc_missing}')
    v7.eval()

    matcher = V7AstrometryMatcher(
        v7_model         = v7,
        n_rubin_bands    = n_rubin_bands,
        hidden_channels  = hidden_channels,
        n_adapter_blocks = n_adapter_blocks,
        freeze_stems     = freeze_stems,
        search_radius    = search_radius,
        n_target_bands   = n_target_bands,
        band_embed_dim   = band_embed_dim,
        mlp_hidden       = mlp_hidden,
        n_stream_stages  = n_stream_stages,
    ).to(device)

    n_total     = sum(p.numel() for p in matcher.parameters())
    n_trainable = sum(p.numel() for p in matcher.parameters() if p.requires_grad)
    n_frozen    = n_total - n_trainable
    stages_str  = f', {n_stream_stages} stream stages' if n_stream_stages > 0 else ''
    print(f'V7AstrometryMatcher: {n_total/1e6:.1f}M total, {n_trainable/1e6:.1f}M trainable')
    print(f'  (frozen v7 stems{stages_str}: {n_frozen/1e6:.1f}M)')

    return matcher
