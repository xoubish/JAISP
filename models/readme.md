# JAISP Foundation Models

This directory now contains two foundation-model tracks:

- `JAISPFoundationV6`: stable single-grid MAE where Phase B downsamples Euclid to Rubin resolution before fusion.
- `JAISPFoundationV7`: experimental mixed-resolution MAE with native-resolution Rubin/VIS/NISP branches and late latent fusion.

Both models learn to reconstruct held-out bands at pixel precision from the remaining context bands.

This document covers the foundation model only.
For downstream heads, see:

- `detection/README.md` (DETR-style source detection)
- `astrometry2/README.md` (Rubin -> Euclid VIS concordance)
- `photometry/README.md` (PSF modeling + forced photometry)

---

## Why MAE, not JEPA (v5 → v6)

v5 used a JEPA student/teacher with 16×16 patch tokens. Three problems:

1. **16×16 patches = 3.2" token resolution** — Rubin at 0.2"/px needs sub-pixel precision for astrometry.
2. **Latent-space cosine loss** doesn't force the network to care about where things are spatially.
3. **Strict token-to-token matching** (`shift_px=0`) was self-defeating with the real ~0.25–0.5px instrument misalignments present in the data.

v6 uses pixel-space L1 reconstruction. To reconstruct a band correctly, the encoder must preserve sub-pixel spatial information — there is no way to get a high Pearson r at the pixel level without it.

---

## v6 Architecture

```
Input: N available bands, each [1, 512, 512], noise-normalized (image / rms)
       (Euclid bands downsampled to 512×512 during Phase B)

Encoder
  BandStem (per band)     [1,H,W] → [stem_ch,H,W]   Conv5→GN→GELU→Conv3→GN→GELU
  Mean aggregation        N stems → [stem_ch,H,W]
  DownBlock × 3           stride-2 ConvNeXt          → H/8, W/8
  TransformerBlock × 4    MHSA + FFN on H/8 tokens   + sinusoidal pos embeddings

Decoder (U-Net + FiLM band conditioning)
  UpBlock × 3             bilinear upsample + ConvNeXt + skip connection
  FiLM                    target-band embedding → scale/shift each UpBlock
  Output conv             → [1, H, W] predicted band

Loss
  InformationMap(target_rms, target_image)  → pixel importance weights
  Weighted L1 in noise-normalized units
```

Total parameters: **20.8M**

---

## v6 Training Curriculum

### Phase A — within-instrument (`cross_instrument_prob=0.0`)
Pool: Rubin u/g/r/i/z/y (6 bands)
Task: randomly mask 1 Rubin band, reconstruct it from the other 5.

### Phase B — joint cross-instrument (`cross_instrument_prob=1.0`)
Pool: all 10 bands — Rubin u/g/r/i/z/y + Euclid VIS/Y/J/H
Task: randomly mask any 1 band, reconstruct it from the other 9.
Euclid bands are bilinearly downsampled to 512×512 before entering the pool.
Tiles without Euclid coverage fall back to Phase A automatically.

Phase B encodes cross-instrument spatial correspondence directly into the BandStem weights, which the astrometry matcher then reuses.

---

## v7 Mixed-Resolution Architecture

`JAISPFoundationV7` keeps each instrument stream native for early processing, then fuses streams on a shared latent physical grid.

High-level flow:

```
Rubin bands (512x512, 0.2"/px)  -> per-band stems -> Rubin branch  -> latent resize
VIS band   (~1050x1050, 0.1"/px) -> per-band stem  -> VIS branch    -> latent resize
NISP bands (~350x350, 0.3"/px)   -> per-band stems -> NISP branch   -> latent resize

Shared latent physical grid
  -> stream mean fusion + stream embeddings
  -> transformer bottleneck
  -> target-stream decoder
  -> native-resolution reconstruction of the held-out band
```

Design choices:

- Rubin, VIS, and NISP use different pre-fusion branch depths so they reach roughly comparable physical latent scales before fusion.
- Fusion happens in latent space, not by forcing raw images onto one input pixel grid.
- The decoder reconstructs to the target band's native size, so VIS can stay VIS-scale and NISP can stay NISP-scale.
- The loss stays the same InformationMap-weighted L1 objective as v6, so the training signal is comparable.

Current status:

- `v7` is experimental.
- There is a training entrypoint, but there is not yet a dedicated `eval_foundation_v7.py`.
- The main downstream question is whether retaining native VIS structure improves astrometry and dense detection.

---

## Why This Transfers To Downstream Tasks

The main representation design choices were made to support detection, astrometry, and photometry heads without task-specific re-encoding:

- No patch tokens: preserves sub-pixel spatial information needed by centroid-level tasks.
- Per-band stems + shared aggregation: allows band-specific noise handling while learning a common latent frame.
- Phase B cross-instrument masking: encourages Rubin/Euclid correspondence directly in encoder features.
- Dense latent feature maps: reusable for set prediction (detection) and local offset matching (astrometry), whether they come from the v6 single-grid bottleneck or the v7 shared physical latent grid.

---

## Files

| File | Description |
|------|-------------|
| `jaisp_foundation_v6.py` | Model definition: all components + `create_optimizer` / `create_scheduler` |
| `jaisp_dataset_v6.py` | `JAISPDatasetV6`, `sample_context_target` (Phase A), `sample_context_target_phaseB` (Phase B), `make_loader_v6` |
| `train_jaisp_foundation_v6.py` | `JAISPTrainerV6` training loop with W&B visualization |
| `eval_foundation_v6.py` | Three evaluation checks: reconstruction table, spatial precision, per-band grid |
| `jaisp_foundation_v7.py` | Experimental mixed-resolution foundation model with late latent fusion |
| `jaisp_dataset_v7.py` | Mixed-resolution Phase B sampler that keeps Euclid bands native |
| `train_jaisp_foundation_v7.py` | Training entrypoint for `JAISPFoundationV7` |

---

## Quick Start

```bash
# v6 Phase A
python train_jaisp_foundation_v6.py \
  --rubin_dir  ../data/rubin_tiles_ecdfs \
  --euclid_dir ../data/euclid_tiles_ecdfs \
  --output_dir ./checkpoints/jaisp_v6 \
  --cross_instrument_prob 0.0 \
  --epochs 60

# v6 Phase B (resume from Phase A checkpoint)
python train_jaisp_foundation_v6.py \
  --rubin_dir  ../data/rubin_tiles_ecdfs \
  --euclid_dir ../data/euclid_tiles_ecdfs \
  --output_dir ./checkpoints/jaisp_v6_phaseB \
  --resume     ./checkpoints/jaisp_v6/checkpoint_best.pt \
  --cross_instrument_prob 1.0 \
  --epochs 120

# v6 Evaluate
python eval_foundation_v6.py \
  --checkpoint ./checkpoints/jaisp_v6_phaseB/checkpoint_best.pt \
  --rubin_dir  ../data/rubin_tiles_ecdfs \
  --euclid_dir ../data/euclid_tiles_ecdfs

# v7 Experimental mixed-resolution training
python train_jaisp_foundation_v7.py \
  --rubin_dir  ../data/rubin_tiles_ecdfs \
  --euclid_dir ../data/euclid_tiles_ecdfs \
  --output_dir ./checkpoints/jaisp_v7 \
  --cross_instrument_prob 1.0 \
  --epochs 80
```

---

## Supported bands

- Rubin: `rubin_u`, `rubin_g`, `rubin_r`, `rubin_i`, `rubin_z`, `rubin_y`
- Euclid: `euclid_VIS`, `euclid_Y`, `euclid_J`, `euclid_H`

Total: 10 bands. Each has its own BandStem with independent weights.

---

## Data format

### Rubin NPZ (`tile_x*_y*.npz`)
- `img`: `[6, H, W]` float32 flux
- `var`: `[6, H, W]` float32 variance (converted to RMS inside dataset)
- Native resolution: 512×512, 0.2"/px

### Euclid NPZ (`tile_x*_y*_euclid.npz`)
- `img_VIS`, `img_Y`, `img_J`, `img_H`
- `var_VIS`, `var_Y`, `var_J`, `var_H` (optional)
- Native resolution: VIS ~1050×1050 at 0.1"/px; NISP ~350×350 at 0.3"/px

---

## Checkpoints

The training script writes:
- `checkpoint_best.pt` — best validation loss
- `checkpoint_latest.pt` — every 10 epochs
- `checkpoint_final.pt` — end of training

Each checkpoint contains `model`, `optimizer`, `scheduler`, `epoch`, `best_val_loss`, `global_step`.
