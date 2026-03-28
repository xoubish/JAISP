## Overview

JAISP (Joint AI Survey Processing) is a **self-supervised, multi-instrument foundation model for astronomical imaging**.
It learns a shared pixel-precise representation of the same sky region across different telescopes, bands, resolutions, and noise regimes.

The core objective is dense reconstruction learning, not contrastive embedding.
The learned representations support downstream tasks such as:

- astrometric alignment between Rubin and Euclid
- cross-band/instrument matching and transfer
- deblending and morphology analysis
- photometric consistency checks

---

## Current Model (v6)

The active foundation model is **JAISP Foundation v6** (`models/jaisp_foundation_v6.py`).

v6 replaces the previous JEPA-based v5 with a **Masked Autoencoder (MAE)** architecture trained to reconstruct masked bands at pixel precision.

Key characteristics:

- **ConvNeXt encoder + U-Net decoder**
  No patch tokenization — full-resolution convolutional processing preserves sub-pixel spatial information critical for precision astrometry.

- **Band-specific stems with GroupNorm**
  Each band has its own shallow stem handling its noise/PSF characteristics. Stems are shared across instruments during Phase B.

- **FiLM conditioning in the decoder**
  The decoder is conditioned on the target band identity, enabling the same encoder to reconstruct any band.

- **InformationMap-weighted L1 loss**
  Loss is weighted by SNR + gradient-based pixel importance so training emphasizes sources over empty sky.

- **Fixed 2D sinusoidal position embeddings**
  Not learned, not interpolated — stable at any resolution.

### Training curriculum

| Phase | `cross_instrument_prob` | Pool | Objective |
|-------|------------------------|------|-----------|
| A | 0.0 | Rubin only (u/g/r/i/z/y) | Predict any Rubin band from the other 5 |
| B | 1.0 | All 10 bands (Rubin + Euclid VIS/Y/J/H) | Predict any band from any combination of the other 9 |

Phase A builds within-instrument spatial understanding; Phase B encodes cross-instrument correspondence.
Euclid bands are downsampled to Rubin resolution (512×512) during Phase B training.

---

## Training

```bash
# Phase A (Rubin only)
python models/train_jaisp_foundation_v6.py \
  --rubin_dir  data/rubin_tiles_ecdfs \
  --euclid_dir data/euclid_tiles_ecdfs \
  --output_dir models/checkpoints/jaisp_v6 \
  --cross_instrument_prob 0.0 \
  --epochs 60

# Phase B (all 10 bands, resume from Phase A)
python models/train_jaisp_foundation_v6.py \
  --rubin_dir  data/rubin_tiles_ecdfs \
  --euclid_dir data/euclid_tiles_ecdfs \
  --output_dir models/checkpoints/jaisp_v6_phaseB \
  --resume     models/checkpoints/jaisp_v6/checkpoint_best.pt \
  --cross_instrument_prob 1.0 \
  --epochs 120
```

---

## Downstream: Astrometry

`models/astrometry2/` contains the Rubin → Euclid VIS astrometric concordance pipeline.

The v6 astrometry matcher (`matcher_v6.py`) reuses frozen Phase B BandStem weights for both the Rubin encoder and the VIS encoder, so the cost volume compares representations trained jointly in the same cross-instrument feature space.

```bash
# Train astrometry matcher (requires Phase B checkpoint)
cd models/astrometry2
python train_astro_v6.py \
  --v6-checkpoint ../checkpoints/jaisp_v6_phaseB/checkpoint_best.pt \
  --rubin_dir  ../../data/rubin_tiles_ecdfs \
  --euclid_dir ../../data/euclid_tiles_ecdfs
```

See `models/astrometry2/README.md` for full pipeline documentation.

---

## Key files

| File | Description |
|------|-------------|
| `models/jaisp_foundation_v6.py` | v6 MAE foundation model (20.8M params) |
| `models/jaisp_dataset_v6.py` | Dataset: all bands per tile + Phase A/B sampling |
| `models/train_jaisp_foundation_v6.py` | Training loop with W&B visualization |
| `models/eval_foundation_v6.py` | Evaluation: reconstruction quality + spatial precision |
| `models/astrometry2/matcher_v6.py` | Astrometry matcher with v6 Phase B backbone |
| `models/astrometry2/train_astro_v6.py` | Astrometry training using v6 stems |

---

## Project scope

This repository focuses on building and validating the **foundation representation** and the astrometric concordance product.
It is an active research codebase; architecture and training procedures continue to evolve.

### Archive

Earlier approaches are preserved in `models/older_architectures/`:
- v5: JEPA-based student/teacher with VICReg (deprecated — patch tokenization lost sub-pixel precision)
- reconstruction2, astrometry (v1): earlier downstream heads
