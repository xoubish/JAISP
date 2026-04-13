"""Load any JAISP foundation checkpoint (v7 or v8) by inspecting its config.

Usage:
    from load_foundation import load_foundation
    model = load_foundation('checkpoints/jaisp_v7_concat/checkpoint_best.pt')
    model = load_foundation('checkpoints/jaisp_v8_fine/checkpoint_best.pt')

The function reads ``fused_pixel_scale_arcsec`` from the checkpoint config
to decide which model class to instantiate:
  - 0.8  → JAISPFoundationV7  (hardcoded stream depths)
  - other → JAISPFoundationV8 (auto-computed stream depths)
"""

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from jaisp_foundation_v7 import JAISPFoundationV7, ALL_BANDS


def load_foundation(
    checkpoint_path: str,
    device: torch.device = None,
    freeze: bool = False,
) -> nn.Module:
    """Load a foundation model from any v7 or v8 checkpoint.

    Parameters
    ----------
    checkpoint_path : path to checkpoint_best.pt (or any saved checkpoint)
    device : target device (default: CPU)
    freeze : if True, freeze all parameters

    Returns
    -------
    model : JAISPFoundationV7 or JAISPFoundationV8 with weights loaded
    """
    if device is None:
        device = torch.device('cpu')

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})

    fused_scale = cfg.get('fused_pixel_scale_arcsec', 0.8)
    common_kwargs = dict(
        band_names=cfg.get('band_names', ALL_BANDS),
        stem_ch=cfg.get('stem_ch', 64),
        hidden_ch=cfg.get('hidden_ch', 256),
        blocks_per_stage=cfg.get('blocks_per_stage', 2),
        transformer_depth=cfg.get('transformer_depth', 4),
        transformer_heads=cfg.get('transformer_heads', 8),
        fused_pixel_scale_arcsec=fused_scale,
    )

    # V8 has auto-computed stream depths; V7 uses hardcoded depths.
    # Detect v8 by checking if the fused scale differs from the v7 default
    # or if the checkpoint model keys indicate different stream depths.
    use_v8 = abs(fused_scale - 0.8) > 0.01

    if use_v8:
        from jaisp_foundation_v8 import JAISPFoundationV8
        model = JAISPFoundationV8(**common_kwargs)
    else:
        model = JAISPFoundationV7(**common_kwargs)

    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    # Filter expected missing keys (skip_projs, target_decoders not always needed).
    real_missing = [
        k for k in missing
        if not k.startswith(('encoder.skip_projs', 'target_decoders'))
    ]
    if real_missing:
        print(f'  [warn] Missing keys after load: {real_missing[:5]}...')

    model = model.to(device)
    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    version = 'v8' if use_v8 else 'v7'
    print(f'Loaded {version} foundation (fused_scale={fused_scale}) from {checkpoint_path}')
    return model
