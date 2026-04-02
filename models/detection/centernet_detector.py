"""CenterNet-style source detector on top of the JAISP V7 MAE encoder.

Replaces the DETR decoder + Hungarian matching with dense per-pixel prediction:
  - Heatmap: probability of a source at each bottleneck pixel
  - Offset: sub-pixel (dx, dy) refinement from grid center
  - Log flux: brightness proxy
  - (Future) Profile: ellipticity (e1, e2), half-light radius, Sersic index

Inference: find peaks in the heatmap via local-max NMS, read off offset/flux
at peak locations.  The predict() API matches JaispDetector exactly so it can
be swapped into the astrometry pipeline without changes.
"""

from __future__ import annotations

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

from detection.detector import JAISPEncoderWrapper, _StubEncoder  # reuse encoder wrapper


def _head(in_ch: int, out_ch: int, hidden_ch: int = 256) -> nn.Sequential:
    """Small 2-layer conv head: 3x3 hidden + 1x1 output."""
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_ch, out_ch, 1),
    )


class CenterNetDetector(nn.Module):
    """Dense source detector: heatmap + offset + flux on V7 encoder features.

    Parameters
    ----------
    encoder      : JAISPEncoderWrapper or _StubEncoder
    encoder_dim  : channel dim of encoder bottleneck (default 256)
    neck_layers  : number of 3x3 Conv+BN+ReLU refinement layers (default 3)
    head_ch      : hidden channels in prediction heads (default 256)
    predict_profile : if True, add a profile head for (e1, e2, r_half, sersic_n)
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int = 256,
        neck_layers: int = 3,
        head_ch: int = 256,
        predict_profile: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.predict_profile = predict_profile

        # Refinement neck
        neck = []
        ch = encoder_dim
        for _ in range(neck_layers):
            neck.extend([
                nn.Conv2d(ch, head_ch, 3, padding=1),
                nn.BatchNorm2d(head_ch),
                nn.ReLU(inplace=True),
            ])
            ch = head_ch
        self.neck = nn.Sequential(*neck)

        # Prediction heads
        self.hm_head = _head(head_ch, 1, head_ch)
        self.off_head = _head(head_ch, 2, head_ch)
        self.flux_head = _head(head_ch, 1, head_ch)

        if predict_profile:
            # (e1, e2, log_r_half, sersic_n) per pixel
            self.profile_head = _head(head_ch, 4, head_ch)
        else:
            self.profile_head = None

        self._init_weights()

    def _init_weights(self):
        # Initialize heatmap bias so initial predictions are near zero
        # log(0.01 / 0.99) ~ -4.6, but -2.19 (CenterNet convention) works better
        nn.init.constant_(self.hm_head[-1].bias, -2.19)
        nn.init.zeros_(self.off_head[-1].bias)

    def forward(
        self,
        images: Dict[str, torch.Tensor],
        rms: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        feats = self.encoder(images, rms)     # [B, encoder_dim, H, W]
        feats = self.neck(feats)              # [B, head_ch, H, W]

        out = {
            'heatmap':  self.hm_head(feats).sigmoid(),  # [B, 1, H, W]
            'offset':   self.off_head(feats),            # [B, 2, H, W]
            'log_flux': self.flux_head(feats).squeeze(1),  # [B, H, W]
        }
        if self.profile_head is not None:
            out['profile'] = self.profile_head(feats)    # [B, 4, H, W]
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        images: Dict[str, torch.Tensor],
        rms: Dict[str, torch.Tensor],
        conf_threshold: float = 0.3,
        tile_hw: Optional[Tuple[int, int]] = None,
        nms_kernel: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """Run inference and return detected sources.

        Returns dict matching JaispDetector.predict() API:
            centroids    [N, 2]  normalized (x, y) in [0, 1]
            classes      [N]     all zeros (single 'source' class)
            scores       [N]     confidence
            log_flux     [N]     flux proxy
            positions_px [N, 2]  pixel coords (if tile_hw given)
            profile      [N, 4]  (e1, e2, log_r_half, sersic_n) if predict_profile
        """
        out = self(images, rms)
        hm = out['heatmap'][0, 0]  # [H, W]
        H, W = hm.shape

        # Local-max NMS: keep pixels where heatmap == max in neighborhood
        pad = nms_kernel // 2
        hm_max = F.max_pool2d(
            hm.unsqueeze(0).unsqueeze(0), nms_kernel, stride=1, padding=pad
        )[0, 0]
        keep = (hm == hm_max) & (hm > conf_threshold)

        if not keep.any():
            device = hm.device
            result = {
                'centroids': torch.zeros(0, 2, device=device),
                'classes':   torch.zeros(0, dtype=torch.long, device=device),
                'scores':    torch.zeros(0, device=device),
                'log_flux':  torch.zeros(0, device=device),
            }
            if tile_hw is not None:
                result['positions_px'] = torch.zeros(0, 2, device=device)
            if self.profile_head is not None:
                result['profile'] = torch.zeros(0, 4, device=device)
            return result

        # Extract peak positions
        yi, xi = torch.where(keep)  # grid positions in feature map
        scores = hm[yi, xi]

        # Add sub-pixel offsets
        off = out['offset'][0]  # [2, H, W]
        dx = off[0, yi, xi]
        dy = off[1, yi, xi]

        # Normalize to [0, 1] (x = col, y = row)
        cx = (xi.float() + dx) / max(W - 1, 1)
        cy = (yi.float() + dy) / max(H - 1, 1)
        centroids = torch.stack([cx, cy], dim=1)  # [N, 2]

        flux = out['log_flux'][0, yi, xi]

        result = {
            'centroids': centroids,
            'classes':   torch.zeros(len(scores), dtype=torch.long, device=hm.device),
            'scores':    scores,
            'log_flux':  flux,
        }
        if tile_hw is not None:
            Ht, Wt = tile_hw
            result['positions_px'] = centroids * centroids.new_tensor([Wt - 1, Ht - 1])
        if self.profile_head is not None:
            prof = out['profile'][0]  # [4, H, W]
            result['profile'] = prof[:, yi, xi].T  # [N, 4]

        return result

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            'state_dict':      self.state_dict(),
            'encoder_dim':     self.encoder_dim,
            'predict_profile': self.predict_profile,
            'model_type':      'centernet',
        }, path)

    @classmethod
    def load(
        cls,
        path: str,
        encoder: nn.Module,
        device: Optional[torch.device] = None,
    ) -> 'CenterNetDetector':
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(
            encoder=encoder,
            encoder_dim=ckpt['encoder_dim'],
            predict_profile=ckpt.get('predict_profile', False),
        )
        model.load_state_dict(ckpt['state_dict'])
        if device is not None:
            model = model.to(device)
        return model
