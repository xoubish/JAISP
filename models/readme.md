# JAISP Foundation Models

This directory contains two foundation-model tracks:

- `JAISPFoundationV7`: **current active path** -- mixed-resolution MAE with native-resolution Rubin/VIS/NISP branches and late latent fusion.
- `JAISPFoundationV6`: archived single-grid MAE. Kept as a library -- V7 imports primitives (`BandStem`, `ConvNeXtBlock`, `FiLM`, etc.) from it.

Both models learn to reconstruct held-out bands at pixel precision from the remaining context bands.

For downstream heads, see:

- `detection/README.md` (DETR source detection)
- `astrometry2/README.md` (Rubin <-> Euclid VIS concordance)
- `photometry/README.md` (PSF modeling + forced photometry)

---

## Why MAE, Not JEPA (v5 -> v6)

v5 used a JEPA student/teacher with 16x16 patch tokens. Three problems:

1. **16x16 patches = 3.2" token resolution** -- need <0.2" for astrometry.
2. **Latent-space cosine loss** doesn't force the network to preserve spatial layout.
3. **Strict token matching** (`shift_px=0`) broke on real ~0.25-0.5px instrument misalignments.

v6 replaced this with pixel-space L1 reconstruction. To reconstruct a band correctly, the encoder *must* preserve sub-pixel spatial information -- there is no shortcut.

---

## Why v7, Not v6

v6 Phase B force-downsampled Euclid VIS (1050x1050 at 0.1"/px) to Rubin's 512x512 grid before encoding. This discarded the 2x resolution advantage that makes VIS valuable for astrometry and deblending.

v7 keeps each instrument at native resolution through two encoder branches (Rubin + Euclid), then fuses at a shared physical scale in latent space. VIS and NISP resolution is preserved end-to-end.

---

## v7 Architecture

Two streams: Rubin (mean-pooled, 6 bands with similar PSFs) and Euclid (concat+project, 4 bands with different PSFs).

```
Rubin:  6 BandStems -> mean pool -> [64, 512, 512]      -> 2-stage encoder --\
                                                                               --> latent @ 0.8"/px
Euclid: 4 BandStems -> concat+1×1 proj -> [64, 1084, 1084] -> 3-stage encoder --/  (~132×132 tokens)
         (VIS/Y/J/H)  (zero-fill missing bands)                                          |
                                                                           Stream fusion +
                                                                           learned stream embeddings
                                                                                          |
                                                                           Transformer bottleneck
                                                                           (depth=4, heads=8)
                                                                                          |
                                                                           TargetDecoder with
                                                                           pyramid skip connections
                                                                                          |
                                                                           Native-resolution output
                                                                           (Euclid->~1084, Rubin->512)
```

NISP Y/J/H data comes from Euclid MER mosaics at 0.1"/px (same as VIS). All 4 Euclid
bands share one stream because they share the same pixel scale and tile size.

The Euclid stream uses `StreamFuser` with fixed-slot concatenation: each of the 4 BandStem
outputs is placed in its assigned slot (VIS=0, Y=1, J=2, H=3), missing bands get zero-filled
slots, and a learned 1×1 projection maps 4×64=256 channels to 64. This preserves per-band
PSF structure (VIS: 0.2" vs NISP: ~0.5") through the full encoder.

**Key design choices:**

- Two streams (Rubin + Euclid) with concat for Euclid, mean for Rubin.
- Branch depth differs per instrument so both streams arrive at ~0.8"/px before fusion.
- `_estimate_fused_hw` converts to angular arcsec scale for physics-grounded fusion.
- `build_target_skips` selects pyramid levels by closest physical scale match.
- Learned stream identity embeddings distinguish Rubin/Euclid feature statistics.
- Unified training: no Phase A/B. Tiles with Euclid use cross-instrument masking; Rubin-only tiles fall back to within-instrument prediction automatically.
- Loss: InformationMap-weighted L1 in noise-normalized units.
- ~13.3M parameters (down from 16M with the old 3-stream design).

---

## v6 Architecture (archived, kept as library)

```
Input: N bands, each [1, 512, 512], noise-normalized

Encoder
  BandStem (per band)     [1,H,W] -> [64,H,W]     Conv5->GN->GELU->Conv3->GN->GELU
  Mean aggregation        N stems -> [64,H,W]
  DownBlock x 3           stride-2 ConvNeXt         -> H/8, W/8
  TransformerBlock x 4    MHSA + FFN on H/8 tokens

Decoder (U-Net + FiLM)
  UpBlock x 3             bilinear upsample + ConvNeXt + skip
  FiLM conditioning       target-band embedding -> scale/shift
  Output conv             -> [1, H, W]

Loss: InformationMap-weighted L1
Total: 20.8M params
```

**v6 training curriculum:**
- Phase A (`cross_instrument_prob=0.0`): Rubin-only masked band prediction.
- Phase B (`cross_instrument_prob=1.0`): Joint Rubin + Euclid. Euclid downsampled to 512x512.

---

## Files

| File | Description |
|------|-------------|
| `jaisp_foundation_v7.py` | V7 mixed-resolution model with late latent fusion |
| `jaisp_dataset_v7.py` | V7 data loader (native-resolution per instrument) |
| `train_jaisp_foundation_v7.py` | V7 training entrypoint |
| `eval_foundation_v7.py` | V7 evaluation and diagnostics |
| `jaisp_foundation_v6.py` | V6 model primitives -- **kept as library**, imported by V7 |
| `jaisp_dataset_v6.py` | V6 dataset -- **kept as library**, used by V7 and downstream |

---

## Data Format

### Rubin NPZ (`tile_x*_y*.npz`)
- `img`: `[6, 512, 512]` float32 flux (u, g, r, i, z, y)
- `var`: `[6, 512, 512]` float32 variance
- `wcs_hdr`: FITS WCS header string

### Euclid NPZ (`tile_x*_y*_euclid.npz`)
- `img_VIS/Y/J/H`: float32 flux at native resolution
- `var_VIS/Y/J/H`: float32 variance (optional)
- `wcs_VIS/Y/J/H`: FITS WCS header string

---

## Supported Bands

Rubin: `rubin_u`, `rubin_g`, `rubin_r`, `rubin_i`, `rubin_z`, `rubin_y`
Euclid: `euclid_VIS`, `euclid_Y`, `euclid_J`, `euclid_H`
Total: 10 bands. Each has its own BandStem with independent weights.

---

## Quick Start

```bash
# V7 training (current)
python train_jaisp_foundation_v7.py \
    --rubin_dir  ../data/rubin_tiles_ecdfs \
    --euclid_dir ../data/euclid_tiles_ecdfs \
    --output_dir ./checkpoints/jaisp_v7_baseline \
    --hidden_ch 256 --transformer_depth 4 --transformer_heads 8 \
    --fused_pixel_scale_arcsec 0.8 --cross_instrument_prob 1.0 \
    --epochs 100 --wandb_name v7_h256_d4_fused0.8
```

---

## Checkpoints

The training script writes:
- `checkpoint_best.pt` -- best validation loss
- `checkpoint_latest.pt` -- every 10 epochs
- `checkpoint_final.pt` -- end of training

Each contains `model`, `optimizer`, `scheduler`, `epoch`, `best_val_loss`, `global_step`, `config`.
