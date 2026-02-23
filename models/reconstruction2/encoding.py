"""
Updated encoding helpers for resolution-aware reconstruction.

Changes from the original encode_target_and_context:
  - Returns stem features (native resolution) alongside tokens.
  - New function signature that the training loop can call.

Drop this into the reconstruction package or inline it in the training script.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from head import interpolate_tokens


def encode_band_with_stem(
    backbone,
    image: torch.Tensor,
    rms: torch.Tensor,
    band: str,
    device: torch.device,
    freeze_backbone: bool,
    use_projector_tokens: bool,
) -> Dict[str, torch.Tensor]:
    """
    Encode a single band and return BOTH tokens and stem features.

    Returns:
        tokens:    [1, N, D]
        grid_size: (Ht, Wt)
        stem_feat: [1, stem_ch, H_native, W_native]  ← NEW
    """
    image = image.unsqueeze(0).to(device)    # [1, 1, H, W]
    rms = rms.unsqueeze(0).to(device)

    if freeze_backbone:
        with torch.no_grad():
            stem_feat = backbone.stems[band](image, rms)        # [1, 64, H, W]
            tokens, grid_size = backbone.encoder(stem_feat)
            if use_projector_tokens:
                tokens = backbone.projector(tokens)
    else:
        stem_feat = backbone.stems[band](image, rms)
        tokens, grid_size = backbone.encoder(stem_feat)
        if use_projector_tokens:
            tokens = backbone.projector(tokens)

    return {
        "tokens": tokens,
        "grid_size": grid_size,
        "stem_feat": stem_feat.half() if freeze_backbone else stem_feat.detach().half(),
        # float16 for stem skip connections halves memory (~140MB vs ~280MB per Euclid band).
        # These are detached anyway — no gradients flow through pixel skips.
    }


def encode_target_and_context_with_stems(
    backbone,
    target_masked: torch.Tensor,
    target_rms: torch.Tensor,
    target_band: str,
    context_images: List[torch.Tensor],
    context_rms: List[torch.Tensor],
    context_bands: List[str],
    device: torch.device,
    freeze_backbone: bool,
    use_projector_tokens: bool,
) -> Tuple[
    torch.Tensor,           # target_tokens     [1, N, D]
    torch.Tensor,           # context_tokens    [1, K, N, D]
    Tuple[int, int],        # target_grid       (Ht, Wt)
    torch.Tensor,           # target_stem_feat  [1, 64, H_tgt, W_tgt]
    List[torch.Tensor],     # context_stem_feats  list of [1, 64, H_k, W_k]
]:
    """
    Like the original encode_target_and_context but also returns stem features.
    """
    tgt = encode_band_with_stem(
        backbone, target_masked, target_rms, target_band,
        device, freeze_backbone, use_projector_tokens,
    )
    target_tokens = tgt["tokens"]
    target_grid = tgt["grid_size"]
    target_stem_feat = tgt["stem_feat"]

    aligned_context = []
    context_stem_feats = []

    for img, rms, band in zip(context_images, context_rms, context_bands):
        ctx = encode_band_with_stem(
            backbone, img, rms, str(band),
            device, freeze_backbone, use_projector_tokens,
        )
        # Align context tokens to target token grid (same as before).
        ctx_aligned = interpolate_tokens(ctx["tokens"], ctx["grid_size"], target_grid)
        aligned_context.append(ctx_aligned)

        # Keep stem features at their NATIVE resolution (not resampled yet).
        # The head's ContextAggregator handles resampling to target resolution.
        context_stem_feats.append(ctx["stem_feat"])

    if aligned_context:
        context_tokens = torch.stack(aligned_context, dim=1)   # [1, K, N, D]
    else:
        context_tokens = torch.zeros(
            1, 1, target_tokens.shape[1], target_tokens.shape[2],
            device=target_tokens.device, dtype=target_tokens.dtype,
        )

    return (
        target_tokens,
        context_tokens,
        target_grid,
        target_stem_feat,
        context_stem_feats,
    )
