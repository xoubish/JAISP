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
    rubin_concat = bool(cfg.get('rubin_concat', False))
    # v10 loss-shaping params (default to v8/v9 behaviour if absent)
    loss_type           = str(cfg.get('loss_type', 'l1'))
    charbonnier_eps     = float(cfg.get('charbonnier_eps', 1e-3))
    core_l2_weight      = float(cfg.get('core_l2_weight', 0.0))
    core_info_threshold = float(cfg.get('core_info_threshold', 0.5))
    common_kwargs = dict(
        band_names=cfg.get('band_names', ALL_BANDS),
        stem_ch=cfg.get('stem_ch', 64),
        hidden_ch=cfg.get('hidden_ch', 256),
        blocks_per_stage=cfg.get('blocks_per_stage', 2),
        transformer_depth=cfg.get('transformer_depth', 4),
        transformer_heads=cfg.get('transformer_heads', 8),
        fused_pixel_scale_arcsec=fused_scale,
    )

    # V8/V9/V10 share the JAISPFoundationV8 class; V7 uses hardcoded stream depths.
    # Detect v8+ class by fused_scale differing from v7's 0.8 OR by any of the
    # v9/v10 markers being set.
    is_v9_plus = rubin_concat or (loss_type != 'l1') or (core_l2_weight > 0)
    use_v8_class = abs(fused_scale - 0.8) > 0.01 or is_v9_plus

    if use_v8_class:
        from jaisp_foundation_v8 import JAISPFoundationV8
        model = JAISPFoundationV8(
            rubin_concat=rubin_concat,
            loss_type=loss_type,
            charbonnier_eps=charbonnier_eps,
            core_l2_weight=core_l2_weight,
            core_info_threshold=core_info_threshold,
            **common_kwargs,
        )
    else:
        # V7 has no v9/v10 parameters; ignore them if set.
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

    if use_v8_class:
        if loss_type == 'charbonnier' or core_l2_weight > 0:
            version = 'v10'
        elif rubin_concat:
            version = 'v9'
        else:
            version = 'v8'
    else:
        version = 'v7'
    print(f'Loaded {version} foundation (fused_scale={fused_scale}, '
          f'rubin_concat={rubin_concat}, loss={loss_type}, '
          f'core_l2={core_l2_weight}) from {checkpoint_path}')
    return model
