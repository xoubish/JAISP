"""Load any JAISP foundation checkpoint (v7, v8, v9, or v10) by inspecting its config.

Usage:
    from load_foundation import load_foundation
    model = load_foundation('checkpoints/jaisp_v7_concat/checkpoint_best.pt')
    model = load_foundation('checkpoints/jaisp_v8_fine/checkpoint_best.pt')
    model = load_foundation('checkpoints/jaisp_v10_warmstart/checkpoint_best.pt')

The function reads checkpoint config markers to choose the model class:
  - fused_pixel_scale ≈ 0.8 and no v9/v10 markers → JAISPFoundationV7 (legacy)
  - everything else → JAISPFoundationV10 (standalone, current production)

V8/V9/V10 share the same architecture; only loss kwargs differ. The standalone
``jaisp_foundation_v10`` module supplies the model class for all three.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from jaisp_foundation_v10 import JAISPFoundationV10, ALL_BANDS


def load_foundation(
    checkpoint_path: str,
    device: torch.device = None,
    freeze: bool = False,
) -> nn.Module:
    """Load a foundation model from any v7+ checkpoint.

    Parameters
    ----------
    checkpoint_path : path to checkpoint_best.pt (or any saved checkpoint)
    device : target device (default: CPU)
    freeze : if True, freeze all parameters

    Returns
    -------
    model : JAISPFoundationV10 (covers v8/v9/v10) or JAISPFoundationV7 (legacy)
            with weights loaded.
    """
    if device is None:
        device = torch.device('cpu')

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})

    fused_scale = cfg.get('fused_pixel_scale_arcsec', 0.8)
    rubin_concat = bool(cfg.get('rubin_concat', False))
    # v10 loss-shaping params (default to v8 behaviour if absent).
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

    # V8/V9/V10 share the JAISPFoundationV10 class architecture; V7 uses
    # hardcoded stream depths and lives in older_architectures/.
    is_v9_plus = rubin_concat or (loss_type != 'l1') or (core_l2_weight > 0)
    use_v10_class = abs(fused_scale - 0.8) > 0.01 or is_v9_plus

    if use_v10_class:
        model = JAISPFoundationV10(
            rubin_concat=rubin_concat,
            loss_type=loss_type,
            charbonnier_eps=charbonnier_eps,
            core_l2_weight=core_l2_weight,
            core_info_threshold=core_info_threshold,
            **common_kwargs,
        )
    else:
        # Legacy V7 fallback: import lazily from older_architectures so the
        # default load path doesn't pull in the deprecated file unless needed.
        from models.older_architectures.jaisp_foundation_v7 import JAISPFoundationV7
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

    if use_v10_class:
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
