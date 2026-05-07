"""Shared foundation/data-IO helpers used by every head built on the JAISP encoder.

Three thin pieces of plumbing live here so heads (astrometry, PSF, photometry,
detection, future) all share a single implementation:

- ``discover_tile_pairs(rubin_dir, euclid_dir)`` — pair Rubin and Euclid NPZ
  tiles by id.
- ``load_tile_data(rubin_path, euclid_path, device)`` — read one tile, build
  the encoder-ready ``{band: [1, 1, H, W]}`` image / RMS dicts, and return
  the VIS WCS.
- ``FrozenEncoder(foundation_model)`` — a no-grad ``nn.Module`` wrapping a
  loaded foundation so callers can grab the bottleneck and the VIS stem
  features in one place.

None of these touch astrometry-specific weights or models. The astrometry2
package re-exports them so existing imports continue to work.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Tile pairing
# ============================================================

def discover_tile_pairs(rubin_dir: str, euclid_dir: str) -> List[Tuple[str, str, str]]:
    """Return ``[(tile_id, rubin_path, euclid_path), ...]`` for matching NPZs."""
    pairs = []
    for rubin_path in sorted(glob.glob(os.path.join(rubin_dir, 'tile_x*_y*.npz'))):
        basename = os.path.basename(rubin_path)
        if basename.endswith('_euclid.npz'):
            continue  # skip euclid files if rubin_dir happens to contain them
        tile_id = os.path.splitext(basename)[0]
        euclid_path = os.path.join(euclid_dir, f'{tile_id}_euclid.npz')
        if os.path.exists(euclid_path):
            pairs.append((tile_id, rubin_path, euclid_path))
    if not pairs:
        raise FileNotFoundError(
            f'No tile pairs found. Checked rubin_dir={rubin_dir} '
            f'(pattern tile_x*_y*.npz) and euclid_dir={euclid_dir} '
            f'(pattern {{tile_id}}_euclid.npz).'
        )
    return pairs


# ============================================================
# Tile loader
# ============================================================

def load_tile_data(
    rubin_path: str,
    euclid_path: str,
    device: torch.device,
):
    """Load one tile and build encoder-ready tensors.

    Returns
    -------
    context_images : {band: [1, 1, H, W]} on ``device``
    context_rms    : {band: [1, 1, H, W]} on ``device``
    vis_hw         : (H_vis, W_vis)
    vis_wcs        : astropy WCS for VIS

    The heavy lifting (building the 10-band image dict, parsing the VIS WCS
    card string) is delegated to ``astrometry2.dataset`` /
    ``astrometry2.source_matching`` via lazy imports to avoid a circular
    import chain when those modules re-export from this one.
    """
    # Lazy-imported to avoid circular import: astrometry2.dataset re-exports
    # symbols from this module.
    from astropy.wcs import WCS

    from astrometry2.dataset import build_full_context_detector_inputs
    from astrometry2.source_matching import safe_header_from_card_string

    rdata = np.load(rubin_path, allow_pickle=True)
    edata = np.load(euclid_path, allow_pickle=True)
    rubin_var = rdata['var'] if 'var' in rdata else None

    context_images, context_rms, vis_hw = build_full_context_detector_inputs(
        edata, rdata['img'], rubin_var=rubin_var,
    )

    vhdr = safe_header_from_card_string(edata['wcs_VIS'].item())
    vis_wcs = WCS(vhdr)

    img_t = {
        k: torch.from_numpy(v[None, None].copy()).float().to(device)
        for k, v in context_images.items()
    }
    rms_t = {
        k: torch.from_numpy(v[None, None].copy()).float().to(device)
        for k, v in context_rms.items()
    }
    return img_t, rms_t, vis_hw, vis_wcs


# ============================================================
# Frozen encoder wrapper
# ============================================================

class FrozenEncoder(nn.Module):
    """Wrapper that runs a frozen JAISP encoder and extracts the VIS stem.

    Call once per tile to access ``self.encoder(images, rms)['bottleneck']``
    and ``self.vis_stem(vis_img, vis_rms)``. Works with any foundation that
    exposes ``encoder`` and ``encoder.stems['euclid_VIS']`` (V7, V8, V10).
    All parameters are frozen; gradients do not flow through this module.
    """

    def __init__(self, foundation_model):
        super().__init__()
        self.encoder = foundation_model.encoder
        self.vis_stem = foundation_model.encoder.stems['euclid_VIS']
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_tile(
        self,
        context_images: Dict[str, torch.Tensor],
        context_rms: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run the encoder + extract raw VIS stem features for one tile."""
        enc_out = self.encoder(context_images, context_rms)
        vis_img = context_images['euclid_VIS']
        vis_rms = context_rms['euclid_VIS']
        vis_stem = self.vis_stem(vis_img, vis_rms)
        return {
            'bottleneck': enc_out['bottleneck'],
            'vis_stem': vis_stem,
            'fused_hw': enc_out['fused_hw'],
            'vis_hw': (vis_img.shape[-2], vis_img.shape[-1]),
        }


# Backward-compatible alias kept for old code that referenced the V7-only name.
FrozenV7Encoder = FrozenEncoder


__all__ = [
    'discover_tile_pairs',
    'load_tile_data',
    'FrozenEncoder',
    'FrozenV7Encoder',
]
