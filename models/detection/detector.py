"""
JaispDetector: DETR-style source detection on top of the JAISP V7 MAE encoder.

Architecture
------------
  JAISPEncoderWrapper (frozen JAISPFoundationV7)
      input : {band_name: [B, 1, H, W]} images + rms dicts
      output: bottleneck [B, encoder_dim, h, w]
               encoder_dim = hidden_ch (default 256), h/w from fused physical scale
  Feature projection + 2D sinusoidal positional encoding
      → [h*w, B, d_model]
  Transformer decoder  (N_q learned object queries)
      → [N_q, B, d_model]
  Prediction heads
      → centroid   [B, N_q, 2]   (x, y) normalised to [0, 1]
      → logits     [B, N_q, C]   star / galaxy / artifact
      → conf       [B, N_q]      objectness logit
      → log_flux   [B, N_q]      log10(peak flux) proxy

Training uses Hungarian matching (like DETR) between predicted and GT positions.

Usage
-----
    from jaisp_foundation_v7 import JAISPFoundationV7
    model = JAISPFoundationV7(...)
    model.load_state_dict(ckpt['model'])
    encoder = JAISPEncoderWrapper(model, freeze=True)
    detector = JaispDetector(encoder, num_queries=300, encoder_dim=256)

    # batch from TileDetectionDataset
    out = detector(batch['images'], batch['rms'])
    # out['centroids']  [B, N_q, 2]
    # out['logits']     [B, N_q, 3]
    # out['conf']       [B, N_q]
    # out['log_flux']   [B, N_q]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


_HERE   = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int = 3) -> nn.Sequential:
    dims = [in_dim] + [hidden_dim] * (layers - 1) + [out_dim]
    mods = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.ReLU())
    return nn.Sequential(*mods)


def _sinusoidal_2d(h: int, w: int, d_model: int, device: torch.device) -> torch.Tensor:
    """2D sinusoidal positional encoding → [h*w, d_model]."""
    assert d_model % 4 == 0
    d = d_model // 2
    div = torch.exp(
        torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d)
    )
    y_pos = torch.arange(h, device=device).float().unsqueeze(1)
    x_pos = torch.arange(w, device=device).float().unsqueeze(1)
    pe_y = torch.zeros(h, d, device=device)
    pe_x = torch.zeros(w, d, device=device)
    pe_y[:, 0::2] = torch.sin(y_pos * div)
    pe_y[:, 1::2] = torch.cos(y_pos * div)
    pe_x[:, 0::2] = torch.sin(x_pos * div)
    pe_x[:, 1::2] = torch.cos(x_pos * div)
    pe = torch.cat([
        pe_y.unsqueeze(1).expand(h, w, d),
        pe_x.unsqueeze(0).expand(h, w, d),
    ], dim=-1)
    return pe.view(h * w, d_model)


# ---------------------------------------------------------------------------
# Stub encoder (used when no MAE checkpoint is available)
# ---------------------------------------------------------------------------

class _StubEncoder(nn.Module):
    """
    3-layer CNN stub that produces a bottleneck tensor [B, 512, H/8, W/8].
    Used for smoke tests when no V7 checkpoint is available.
    """
    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128,         256, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(256,         512, 3, stride=2, padding=1), nn.GELU(),
        )

    def forward(
        self,
        images: Dict[str, torch.Tensor],
        _rms:   Dict[str, torch.Tensor],    # unused in stub, kept for interface compat
    ) -> torch.Tensor:
        # Stack all bands along channel dim → [B, C, H, W]
        imgs = torch.cat(list(images.values()), dim=1)
        return self.net(imgs)              # [B, 512, H/8, W/8]


# ---------------------------------------------------------------------------
# Wrapper around JAISPEncoderV6
# ---------------------------------------------------------------------------

class JAISPEncoderWrapper(nn.Module):
    """
    Thin wrapper that calls JAISPFoundationV7.encode() and returns the bottleneck tensor.

    Parameters
    ----------
    encoder : JAISPFoundationV7 instance
    freeze  : disable gradients through the encoder
    """

    def __init__(self, encoder: nn.Module, freeze: bool = True):
        super().__init__()
        self.encoder = encoder
        if freeze:
            for p in encoder.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        images: Dict[str, torch.Tensor],   # {band: [B, 1, H, W]}
        rms:    Dict[str, torch.Tensor],   # {band: [B, 1, H, W]}
    ) -> torch.Tensor:
        grad_ctx = torch.no_grad() if not any(
            p.requires_grad for p in self.encoder.parameters()
        ) else torch.enable_grad()
        with grad_ctx:
            out = self.encoder.encode(images, rms)
        return out['bottleneck']


# ---------------------------------------------------------------------------
# Source classes
# ---------------------------------------------------------------------------

SOURCE_CLASSES = ['source']
N_CLASSES = len(SOURCE_CLASSES)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class JaispDetector(nn.Module):
    """
    DETR-style source detector built on JAISPFoundationV7 features.

    Parameters
    ----------
    encoder        : JAISPEncoderWrapper (or _StubEncoder) — must return
                     [B, encoder_dim, H/8, W/8] given (images_dict, rms_dict)
    num_queries    : max sources predicted per tile (default 300)
    d_model        : transformer hidden dim (default 256)
    nhead          : attention heads (default 8)
    num_dec_layers : decoder depth (default 6)
    encoder_dim    : channel dim of encoder output (default 512)
    """

    def __init__(
        self,
        encoder:        nn.Module,
        num_queries:    int = 300,
        d_model:        int = 256,
        nhead:          int = 8,
        num_dec_layers: int = 6,
        encoder_dim:    int = 512,
    ):
        super().__init__()
        self.encoder     = encoder
        self.num_queries = num_queries
        self.d_model     = d_model

        # Project encoder features → d_model
        self.feat_proj = nn.Sequential(
            nn.Conv2d(encoder_dim, d_model, kernel_size=1),
            nn.GroupNorm(8, d_model),
        )

        # Learned object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        # Prediction heads
        self.centroid_head = _mlp(d_model, d_model, 2,         layers=3)
        self.class_head    = nn.Linear(d_model, N_CLASSES)
        self.conf_head     = nn.Linear(d_model, 1)
        self.flux_head     = _mlp(d_model, d_model // 2, 1,    layers=2)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.centroid_head[-1].weight)
        nn.init.uniform_(self.centroid_head[-1].bias, -0.1, 0.1)

    # ------------------------------------------------------------------
    def forward(
        self,
        images: Dict[str, torch.Tensor],   # {band: [B, 1, H, W]}
        rms:    Dict[str, torch.Tensor],   # {band: [B, 1, H, W]}
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with:
            centroids  [B, N_q, 2]   normalised (x, y) ∈ [0, 1]
            logits     [B, N_q, 1]   class logits (single 'source' class)
            conf       [B, N_q]      objectness logit
            log_flux   [B, N_q]      log10 flux proxy
        """
        feats = self.encoder(images, rms)    # [B, encoder_dim, h, w]
        feats = self.feat_proj(feats)        # [B, d_model, h, w]
        B, D, h, w = feats.shape

        # Flatten + 2D positional encoding
        pos = _sinusoidal_2d(h, w, D, feats.device)          # [h*w, D]
        memory = feats.flatten(2).permute(2, 0, 1) + pos.unsqueeze(1)  # [h*w, B, D]

        # Object queries
        queries = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)  # [N_q, B, D]

        out = self.decoder(queries, memory)       # [N_q, B, D]
        out = out.permute(1, 0, 2)               # [B, N_q, D]

        return {
            'centroids': self.centroid_head(out).sigmoid(),   # [B, N_q, 2]
            'logits':    self.class_head(out),                # [B, N_q, 3]
            'conf':      self.conf_head(out).squeeze(-1),     # [B, N_q]
            'log_flux':  self.flux_head(out).squeeze(-1),     # [B, N_q]
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        images: Dict[str, torch.Tensor],
        rms:    Dict[str, torch.Tensor],
        conf_threshold: float = 0.5,
        tile_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference and filter to confident detections.

        Returns dict with centroids [N,2], classes [N], scores [N],
        log_flux [N], and positions_px [N,2] if tile_hw is given.
        """
        out    = self(images, rms)
        scores = out['conf'][0].sigmoid()
        keep   = scores > conf_threshold

        result = {
            'centroids': out['centroids'][0][keep],
            'classes':   out['logits'][0][keep].argmax(-1),
            'scores':    scores[keep],
            'log_flux':  out['log_flux'][0][keep],
        }
        if tile_hw is not None:
            H, W = tile_hw
            xy = result['centroids']
            result['positions_px'] = xy * xy.new_tensor([W - 1, H - 1])
        return result

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            'state_dict':   self.state_dict(),
            'num_queries':  self.num_queries,
            'd_model':      self.d_model,
            'encoder_dim':  self.feat_proj[0].in_channels,
            'num_dec_layers': len(self.decoder.layers),
        }, path)

    @classmethod
    def load(
        cls,
        path: str,
        encoder: nn.Module,
        device: Optional[torch.device] = None,
    ) -> 'JaispDetector':
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(
            encoder=encoder,
            num_queries=ckpt['num_queries'],
            d_model=ckpt['d_model'],
            encoder_dim=ckpt['encoder_dim'],
            num_dec_layers=ckpt.get('num_dec_layers', 6),
        )
        model.load_state_dict(ckpt['state_dict'])
        if device is not None:
            model = model.to(device)
        return model
