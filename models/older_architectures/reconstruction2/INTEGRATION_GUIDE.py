"""
Training loop integration guide for ResolutionAwareReconstructionHead.
=====================================================================

This shows the minimal changes to train_masked_reconstruction.py.
Only the parts that change are shown — everything else stays identical.
"""

# =======================================================================
# 1. IMPORTS — replace old head, add new encoding helper
# =======================================================================

# OLD:
#   from head import MaskedReconstructionHead, interpolate_tokens
#
# NEW:
from head import ResolutionAwareReconstructionHead, interpolate_tokens
from encoding import encode_target_and_context_with_stems


# =======================================================================
# 2. HEAD CONSTRUCTION (in train() function, around line 577)
# =======================================================================

# OLD:
#   head = MaskedReconstructionHead(
#       embed_dim=(...),
#       patch_size=args.patch_size,
#       depth=args.head_depth,
#       num_heads=args.head_heads,
#       mlp_ratio=args.head_mlp_ratio,
#   ).to(device)
#
# NEW:
head = ResolutionAwareReconstructionHead(
    embed_dim=(args.proj_dim if args.use_projector_tokens else args.embed_dim),
    stem_ch=64,  # matches BandStem out_channels in foundation
    patch_size=args.patch_size,
    depth=args.head_depth,
    num_heads=args.head_heads,
    mlp_ratio=args.head_mlp_ratio,
    skip_proj=16,
).to(device)


# =======================================================================
# 3. ENCODING (in the per-sample loop, replaces encode_target_and_context)
# =======================================================================

# OLD:
#   target_tokens, context_tokens, target_grid = encode_target_and_context(
#       backbone=backbone,
#       target_masked=target_masked_raw,
#       ... )
#
# NEW — also returns stem features:
target_tokens, context_tokens, target_grid, target_stem_feat, context_stem_feats = \
    encode_target_and_context_with_stems(
        backbone=backbone,
        target_masked=target_masked_raw,
        target_rms=target_rms,
        target_band=target_band,
        context_images=context_images,
        context_rms=context_rms,
        context_bands=context_bands,
        device=device,
        freeze_backbone=args.freeze_backbone,
        use_projector_tokens=args.use_projector_tokens,
    )


# =======================================================================
# 4. HEAD FORWARD CALL (replaces the old head() call)
# =======================================================================

# We need the target image's native (H, W) for the decoder.
target_hw = (int(target_image.shape[-2]), int(target_image.shape[-1]))

# OLD:
#   head_out = head(
#       target_tokens=target_tokens,
#       context_tokens=context_tokens,
#       token_mask=token_mask,
#       grid_size=target_grid,
#       masked_input=target_masked_model.unsqueeze(0),
#       pixel_mask=pixel_mask.unsqueeze(0),
#   )
#
# NEW — pass native resolution and stem features:
head_out = head(
    target_tokens=target_tokens,
    context_tokens=context_tokens,
    token_mask=token_mask,
    grid_size=target_grid,
    target_hw=target_hw,                        # ← NEW
    target_stem_feat=target_stem_feat,           # ← NEW
    context_stem_feats=context_stem_feats,       # ← NEW
    masked_input=target_masked_model.unsqueeze(0),
    pixel_mask=pixel_mask.unsqueeze(0),
)

# Everything downstream (compute_losses, preview, etc.) is UNCHANGED.
# The head output dict has the same keys: "residual", "token_inpaint", "pred"
# but now pred is at the target band's native resolution.


# =======================================================================
# 5. VALIDATION (evaluate_fixed_validation) — same pattern of changes
# =======================================================================
# Replace encode_target_and_context → encode_target_and_context_with_stems
# Add target_hw, target_stem_feat, context_stem_feats to head() call.
# The rest is identical.


# =======================================================================
# NOTES
# =======================================================================
#
# Memory: the stem features are large tensors at native resolution:
#   - Rubin:  [1, 64, 512, 512]    = 67 MB  (float32)
#   - Euclid: [1, 64, 1050, 1050]  = 282 MB (float32)
#
# With a frozen backbone these are computed under torch.no_grad() and
# detached, so they don't consume gradient memory.  But if you have K=9
# context bands that are all Euclid, that's ~2.5 GB of stem features.
#
# Mitigation strategies if memory is tight:
#   a) Aggregate context stems eagerly (one at a time, running sum)
#      instead of storing all K and stacking. This reduces peak memory
#      from O(K * stem_size) to O(1 * stem_size).
#   b) Downsample stem features 2× before passing to the head
#      (e.g. F.avg_pool2d(..., 2)).  The head's ContextAggregator and
#      UpsampleFuseBlock will still resize to match decoder stages.
#   c) Use float16 for stem features (they're frozen / detached anyway).
#
# For option (a), replace the list accumulation in encoding.py with:
#
#   running_ctx = None
#   running_count = 0
#   for img, rms, band in zip(context_images, context_rms, context_bands):
#       ctx = encode_band_with_stem(...)
#       stem = ctx["stem_feat"]
#       # Resample to target native resolution immediately
#       stem = F.interpolate(stem, size=target_hw, mode="bilinear", align_corners=False)
#       if running_ctx is None:
#           running_ctx = stem
#       else:
#           running_ctx = running_ctx + stem
#       running_count += 1
#   # Then pass running_ctx / running_count as a single pre-aggregated context_stem
#   # and skip the ContextAggregator in the head.
