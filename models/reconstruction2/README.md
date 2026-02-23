# JAISP Masked Reconstruction Head v2 — Resolution-Aware

This folder adds a resolution-aware reconstruction pipeline to predict masked pixels in a target band from:
- the masked target image itself (with stem skip connections at native resolution), and
- a sampled subset of context bands (`k→1`, including `9→1`), injected at both token level and native pixel level.

## What changed from v1

The original head decoded each 16×16 patch independently via a single linear layer, producing visible block artifacts. This version replaces that with:

- **Progressive upsampling decoder** — 4 stages of PixelShuffle 2× upsampling (for `patch_size=16`), so neighboring patches share overlapping receptive fields at every scale. No more grid artifacts by construction.
- **Context stem injection** — stem features from context bands are kept at their native resolution (e.g. Euclid VIS at 1050×1050), resampled to the target band's pixel grid, and attention-weighted per-pixel before being fused into each decoder stage. This lets a low-resolution target band (Rubin at 512×512) learn from high-resolution context (Euclid at 1050×1050).
- **Target stem skip connections** — the target band's own stem features (zeroed inside the mask) provide high-frequency detail where available, with context stems filling the gap inside the masked region.
- **Native resolution output** — the decoder always produces output at the target band's native pixel grid, not a fixed size.

## Files

| File | Role |
|------|------|
| `head.py` | `ResolutionAwareReconstructionHead` with progressive decoder, `ContextAggregator`, `UpsampleFuseBlock` |
| `encoding.py` | `encode_target_and_context_with_stems()` — returns tokens AND stem features from the backbone |
| `dataset.py` | Multi-band dataset that samples one target band and random context bands per tile (unchanged from v1) |
| `masking.py` | Mixed mask generator: `random`, `object`, `hard` (unchanged from v1) |
| `train_masked_reconstruction.py` | End-to-end training script (updated to wire in stem features) |

## Quick Start

Run from repo root:

```bash
python3 models/reconstruction2/train_masked_reconstruction.py \
  --rubin-dir data/rubin_tiles_ecdfs \
  --euclid-dir data/euclid_tiles_ecdfs \
  --backbone-ckpt checkpoints/jaisp_v5/best.pt \
  --output-dir checkpoints/jaisp_reconstruction2 \
  --epochs 30 \
  --batch-size 1 \
  --min-context 1 \
  --max-context 9
```

## Recommended First Run

Start with frozen backbone, batch size 1, and a short run to verify everything works:

```bash
python3 models/reconstruction2/train_masked_reconstruction.py \
  --rubin-dir data/rubin_tiles_ecdfs \
  --euclid-dir data/euclid_tiles_ecdfs \
  --backbone-ckpt checkpoints/jaisp_v5/best.pt \
  --output-dir checkpoints/jaisp_reconstruction2_smoke \
  --epochs 3 \
  --batch-size 1 \
  --num-workers 0 \
  --wandb-mode disabled
```

## Key Arguments

| Arg | Default | Notes |
|-----|---------|-------|
| `--backbone-ckpt` | `models/checkpoints/jaisp_v5/best.pt` | Path to pretrained foundation checkpoint |
| `--freeze-backbone` | `True` | Trains only the head (recommended first) |
| `--train-backbone` | — | Flag to unfreeze backbone; uses `--backbone-lr` (default 3e-5) |
| `--batch-size` | 2 | Use 1 if memory-constrained (Euclid stems are ~280 MB each) |
| `--min-context` / `--max-context` | 1 / 9 | Randomly samples K context bands per sample |
| `--mask-random/object/hard` | 0.5 / 0.4 / 0.1 | With curriculum ramping hard masks over `--curriculum-epochs` (default 8) |
| `--predict-noise-units` | `True` | Works in image/rms space (recommended for cross-band consistency) |
| `--stem-ch` | 64 | Must match `BandStem.out_channels` in the foundation model |
| `--skip-proj` | 16 | Projection dim for stem skip connections (lower = less memory) |
| `--head-depth` | 2 | Transformer blocks in the token-level fusion stage |

## Memory Notes

Stem features at native resolution are large:
- Rubin: `[1, 64, 512, 512]` ≈ 67 MB per band
- Euclid: `[1, 64, 1050, 1050]` ≈ 280 MB per band

With frozen backbone these are computed under `torch.no_grad()` and detached, so they don't consume gradient memory. But with K=9 context bands, peak memory can be significant. Mitigations:
- Use `--batch-size 1`
- Reduce `--max-context` for initial experiments
- Reduce `--skip-proj` (e.g. 8 instead of 16)

## Architecture Overview

```
Context bands (native res)
    │
    ├── stem features ──► ContextAggregator ──► inject at each
    │   [64, H_k, W_k]    (attention-weighted    decoder stage
    │                       across K bands)
    │
    └── encoder tokens ──► token-level ──► progressive upsample ──► output at
        [N, 256]            cross-band     4× PixelShuffle stages    target native
                            transformer                               resolution
Target band (masked)
    │
    ├── stem features ──► skip connections (zeroed inside mask)
    │   [64, H_tgt, W_tgt]
    │
    └── encoder tokens
        [N, 256]
```
