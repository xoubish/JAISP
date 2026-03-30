# JAISP Foundation v6

`JAISPFoundationV6` is a self-supervised masked autoencoder (MAE) foundation model for joint Rubin + Euclid image tiles.

It learns to reconstruct any masked band at pixel precision from the remaining bands, forcing the encoder to maintain exact spatial layout across instruments.

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

## Architecture

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

## Training curriculum

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

## Why This Transfers To Downstream Tasks

The main representation design choices were made to support detection, astrometry, and photometry heads without task-specific re-encoding:

- No patch tokens: preserves sub-pixel spatial information needed by centroid-level tasks.
- Per-band stems + shared aggregation: allows band-specific noise handling while learning a common latent frame.
- Phase B cross-instrument masking: encourages Rubin/Euclid correspondence directly in encoder features.
- Dense bottleneck features (`[B, 512, H/8, W/8]`): reusable for set prediction (detection) and local offset matching (astrometry).

---

## Files

| File | Description |
|------|-------------|
| `jaisp_foundation_v6.py` | Model definition: all components + `create_optimizer` / `create_scheduler` |
| `jaisp_dataset_v6.py` | `JAISPDatasetV6`, `sample_context_target` (Phase A), `sample_context_target_phaseB` (Phase B), `make_loader_v6` |
| `train_jaisp_foundation_v6.py` | `JAISPTrainerV6` training loop with W&B visualization |
| `eval_foundation_v6.py` | Three evaluation checks: reconstruction table, spatial precision, per-band grid |

---

## Quick start

```bash
# Phase A
python train_jaisp_foundation_v6.py \
  --rubin_dir  ../data/rubin_tiles_ecdfs \
  --euclid_dir ../data/euclid_tiles_ecdfs \
  --output_dir ./checkpoints/jaisp_v6 \
  --cross_instrument_prob 0.0 \
  --epochs 60

# Phase B (resume from Phase A checkpoint)
python train_jaisp_foundation_v6.py \
  --rubin_dir  ../data/rubin_tiles_ecdfs \
  --euclid_dir ../data/euclid_tiles_ecdfs \
  --output_dir ./checkpoints/jaisp_v6_phaseB \
  --resume     ./checkpoints/jaisp_v6/checkpoint_best.pt \
  --cross_instrument_prob 1.0 \
  --epochs 120

# Evaluate
python eval_foundation_v6.py \
  --checkpoint ./checkpoints/jaisp_v6_phaseB/checkpoint_best.pt \
  --rubin_dir  ../data/rubin_tiles_ecdfs \
  --euclid_dir ../data/euclid_tiles_ecdfs
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
