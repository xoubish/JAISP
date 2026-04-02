# JAISP (Joint AI Survey Processing)

A self-supervised multi-instrument foundation model for precision cosmology with Rubin Observatory and Euclid, trained on overlapping imaging in the Extended Chandra Deep Field South (ECDFS).

---

## Table of Contents

1. [Motivation](#motivation)
2. [Data](#data)
3. [Foundation Model](#foundation-model)
   - [Architecture History (v5 -> v6 -> v7)](#architecture-history)
   - [v7 Mixed-Resolution MAE (Current)](#v7-mixed-resolution-mae)
4. [Downstream Heads](#downstream-heads)
   - [Detection (DETR)](#1-detection)
   - [Astrometry Concordance](#2-astrometry-concordance)
   - [PSF + Forced Photometry](#3-psf--forced-photometry)
5. [Project Structure](#project-structure)
6. [Quick Start](#quick-start)
7. [Checkpoints](#checkpoints)

---

## Motivation

Rubin Observatory's LSST and ESA's Euclid survey the same sky at different resolutions, wavelengths, and pixel scales. Jointly analyzing their imaging enables science neither can achieve alone -- sub-pixel astrometric alignment, deeper source detection, and more precise photometry.

JAISP learns a single spatially precise shared representation from both instruments, then attaches lightweight task-specific heads for detection, astrometry, and photometry. The central insight is that **pixel-space reconstruction** (masked autoencoding) forces the encoder to preserve the exact spatial layout needed by precision cosmology tasks, unlike latent-space objectives (JEPA) that discard sub-pixel information.

### The Pipeline

```
 Rubin tiles (6 bands, 512x512, 0.2"/px)
 Euclid tiles (VIS 1050x1050 @ 0.1"/px, NISP 350x350 @ 0.3"/px)
         |
    [ Foundation Model (self-supervised MAE) ]
         |
         +-- frozen encoder features
         |
    +----+----+--------------------+
    |         |                    |
 Detection  Astrometry         Photometry
 (DETR)     (patch matching)   (PSF + matched filter)
```

---

## Data

All data covers the ECDFS field, ~1700 overlapping tiles.

### Rubin NPZ files (`data/rubin_tiles_ecdfs/tile_x*_y*.npz`)

| Key      | Shape        | Description                        |
|----------|--------------|------------------------------------|
| `img`    | `[6, 512, 512]` | Flux in 6 bands (u, g, r, i, z, y) |
| `var`    | `[6, 512, 512]` | Variance (converted to RMS internally) |
| `wcs_hdr`| string       | FITS WCS header for astrometry     |

Pixel scale: 0.2 arcsec/pixel.

### Euclid NPZ files (`data/euclid_tiles_ecdfs/tile_x*_y*_euclid.npz`)

| Key       | Shape             | Pixel Scale   |
|-----------|-------------------|---------------|
| `img_VIS` | `[~1050, ~1050]`  | 0.1 arcsec/px |
| `img_Y/J/H` | `[~350, ~350]` | 0.3 arcsec/px |
| `var_VIS/Y/J/H` | same       | Variance      |
| `wcs_VIS/Y/J/H` | string     | FITS WCS      |

### Supported Bands

| Instrument | Bands                           | Count |
|------------|---------------------------------|-------|
| Rubin      | `rubin_u`, `rubin_g`, `rubin_r`, `rubin_i`, `rubin_z`, `rubin_y` | 6 |
| Euclid     | `euclid_VIS`, `euclid_Y`, `euclid_J`, `euclid_H` | 4 |
| **Total**  |                                 | **10** |

Each band has its own `BandStem` with independent learned weights.

---

## Foundation Model

### Architecture History

The foundation model went through seven major iterations. The overall arc is a progression from **latent-space alignment** (v1-v5) to **pixel-space reconstruction** (v6-v7) -- driven by the realization that precision cosmology tasks (astrometry, photometry) demand sub-pixel spatial fidelity that contrastive/JEPA objectives cannot enforce.

#### v1: Patch-Level Contrastive Learning

**Approach**: ViT encoders for Rubin and Euclid. Extract large patches (192x192 Rubin, 384x384 Euclid), encode each to a single embedding vector, train with contrastive loss (NT-Xent) to match co-located Rubin-Euclid pairs.

**Why it failed**: ~95% of pixels are empty sky background. The model learned "flat background = flat background" and embeddings collapsed (separation metric ~0.002). Actual sources averaged away in the global patch embedding.

#### v2: Signal-Based Patch Sampling

**Approach**: Same ViT architecture as v1, but improved data sampling -- evaluate multiple random patch candidates and select those with highest signal (inverse-variance weighted). Prioritize regions containing actual astronomical sources.

**Why we moved on**: Signal sampling helped but was a band-aid. The fundamental flaw was patch-level encoding -- compressing a 192x192 image to a single vector destroys the spatial information needed for astrometry. Led to rethinking the representation granularity.

#### v3: DETR-JEPA (Object-Centric)

**Approach**: ViT backbone producing spatial feature tokens, then a DETR-style decoder with 100 learnable object queries. Each query "discovers" one source via cross-attention. Hungarian matching finds optimal 1-1 correspondence between Rubin and Euclid object slots. Contrastive loss in object space.

**Motivation**: "Can we do object detection with DETR and then do JEPA on that manifold?" Move from patch-level to object-level representation where each embedding corresponds to one astronomical source.

**Why we moved on**: Methodologically interesting but practically complex. Hungarian matching added instability. The object queries didn't reliably converge to distinct sources in crowded fields. More importantly, object-level embeddings still don't give you pixel-level spatial precision.

#### v4: Native-Resolution JEPA with InformationMap

**Approach**: Moved to full-tile native resolution (512x512) instead of patches. Per-band CNN stems with noise normalization. BYOL/JEPA student-teacher architecture with EMA. InformationMap weighting (SNR-based thresholding + Sobel gradient signals) to focus learning on source pixels. Shift-tolerant alignment loss allowing +/-5px matching tolerance.

**Motivation**: Process the full tile at native resolution so no spatial information is lost to patching. Use signal-based weights to solve the background-dominance problem without patch sampling.

**Why we moved on**: The +/-5px shift tolerance was a crutch -- it allowed the model to be spatially imprecise, which is the opposite of what astrometry needs. The EMA teacher added complexity without clear benefit. InformationMap weighting was a good idea that survived into v6/v7.

#### v5: Strict-Position JEPA

**Approach**: Identical to v4 but with shift tolerance removed (`shift_px=0`). Forces exact token-to-token matching at corresponding spatial positions. Same InformationMap, band stems, ViT backbone, VICReg regularization.

**Motivation**: Fix v4's spatial imprecision by enforcing strict positional correspondence.

**Why it failed**: Three compounding problems:
1. **16x16 patch tokens = 3.2" resolution** -- astrometry needs <0.2".
2. **Latent-space cosine loss** doesn't force the network to care about exact pixel layout -- only that embeddings are "similar".
3. **Strict matching broke on real data** -- genuine 0.25-0.5px instrument misalignments between Rubin and Euclid mean corresponding tokens don't perfectly align.

A simple CNN baseline (astrometry2) achieved 38 mas vs JEPA's 47 mas, proving the representation wasn't learning useful spatial information.

**Key lesson from v1-v5**: Latent-space alignment objectives (contrastive, JEPA, BYOL) optimize for *feature similarity*, not *spatial precision*. For precision cosmology, you need the model to know exactly where things are at the sub-pixel level. No amount of contrastive loss engineering achieves this -- you need to reconstruct pixels.

#### v6: Masked Band Prediction (Dense Reconstruction)

**Approach**: Fundamental shift from JEPA to masked autoencoding. Per-band CNN stems (BandStem with GroupNorm), ConvNeXt encoder with 3 stride-2 stages, transformer bottleneck at H/8 resolution, U-Net decoder with FiLM band conditioning. Pixel-space L1 reconstruction loss weighted by InformationMap. 20.8M params.

Training in two phases:
- **Phase A** (`cross_instrument_prob=0.0`): mask 1 Rubin band, reconstruct from the other 5.
- **Phase B** (`cross_instrument_prob=1.0`): mask any band, reconstruct from the other 9 (joint Rubin + Euclid).

**Why it works**: To reconstruct a held-out band at the pixel level, the encoder *must* preserve sub-pixel spatial information. There is no shortcut -- you can't get a high reconstruction fidelity without knowing exactly where every source is. This is precisely the spatial precision astrometry and photometry need.

**Limitation**: Phase B downsampled Euclid VIS (1050x1050 at 0.1"/px) to Rubin's 512x512 grid before encoding. This discarded the 2x resolution advantage that makes VIS the most valuable instrument for astrometry and deblending.

#### v7: Mixed-Resolution MAE (current)

**Approach**: Fixes v6's downsampling assumption. Each instrument processes at native resolution through independent encoder branches with different depths, then fuses on a shared physical-scale latent grid. The decoder reconstructs back to the target band's native resolution.

**Why**: VIS at 0.1"/px is the sharpest imaging in the survey. Downsampling it to match Rubin before encoding throws away exactly the information that makes joint analysis valuable.

### Summary

| Version | Approach | Key Idea | Outcome |
|---------|----------|----------|---------|
| v1 | Patch contrastive | Match Rubin-Euclid patch embeddings | Failed: background collapse |
| v2 | Patch contrastive | Signal-based patch selection | Abandoned: patch-level still lossy |
| v3 | DETR-JEPA | Object-level manifold matching | Abandoned: complexity, no precision gain |
| v4 | Native-res JEPA | InformationMap + shift tolerance | Superseded: spatially imprecise |
| v5 | Native-res JEPA | Strict position matching | Failed: JEPA can't enforce pixel precision |
| v6 | Dense MAE | Pixel-space reconstruction | Works: 31.9 mas astrometry |
| v7 | Mixed-res MAE | Native resolution per instrument | **Current**: preserves VIS advantage |

### v7 Mixed-Resolution MAE

**File**: `models/jaisp_foundation_v7.py`

```
Rubin bands (512x512, 0.2"/px)   -> BandStems -> Rubin branch (2 stages) --\
VIS band   (1050x1050, 0.1"/px)  -> BandStem  -> VIS branch   (3 stages) --> latent @ 0.8"/px
NISP bands (350x350, 0.3"/px)    -> BandStems -> NISP branch  (1 stage)  --/     (~130x130 tokens)
                                                                                       |
                                                                           Transformer bottleneck
                                                                           (depth=4, heads=8)
                                                                                       |
                                                                           TargetDecoder with
                                                                           skip connections
                                                                                       |
                                                                           Native-resolution output
                                                                           (VIS->1050, NISP->350,
                                                                            Rubin->512)
```

**Key design choices:**

- **Per-stream branch depths** differ so all streams arrive at ~0.8 arcsec/px before fusion. Rubin needs 2 stride-2 stages (0.2 -> 0.8), VIS needs 3 (0.1 -> 0.8), NISP needs 1 (0.3 -> 0.6).
- **Physics-grounded fusion**: `_estimate_fused_hw` converts to angular arcsec scale so tokens fuse at matched physical resolution, not matched pixel count.
- **Skip connections**: `build_target_skips` routes encoder pyramid features to the decoder at the closest matching physical scale, fusing across all streams.
- **Learned stream identity embeddings** let the transformer distinguish Rubin/VIS/NISP feature statistics.
- **Unified training**: no Phase A/B split. Tiles with Euclid always use cross-instrument masking; Rubin-only tiles fall back to within-instrument prediction automatically.
- **Loss**: InformationMap-weighted L1 in noise-normalized units (same as v6).

**Why pixel-space reconstruction matters**: to reconstruct a held-out band at the pixel level, the encoder *must* preserve sub-pixel spatial information. There is no shortcut -- this is exactly the spatial precision needed for astrometry, detection, and photometry downstream.

**Config** (best checkpoint: `jaisp_v7_baseline`):

| Parameter | Value |
|-----------|-------|
| `stem_ch` | 64 |
| `hidden_ch` | 256 |
| `transformer_depth` | 4 |
| `transformer_heads` | 8 |
| `fused_pixel_scale_arcsec` | 0.8 |
| `cross_instrument_prob` | 1.0 |
| Total params | 16.0M |

---

## Downstream Heads

All downstream heads reuse the frozen V7 foundation encoder. Only lightweight task-specific layers are trained.

### 1. Detection

**Directory**: `models/detection/`

DETR-style source detector that predicts source positions, confidence, and flux from the V7 encoder bottleneck features.

#### Architecture

```
Frozen V7 encoder
  -> bottleneck [B, 256, ~130, ~130]
  -> 1x1 Conv projection + GroupNorm
  -> 2D sinusoidal positional encoding
  -> Transformer decoder (6 layers, 8 heads)
     with 500 learned object queries
  -> Prediction heads:
       centroid (x,y) in [0,1]  (3-layer MLP + sigmoid)
       confidence (objectness)   (Linear)
       log_flux proxy            (2-layer MLP)
```

#### Training

Uses Hungarian matching between 500 predicted queries and ground-truth pseudo-labels per tile:

- **Pseudo-labels** come from classical peak-finding on a Rubin g+r+i coadd image (3-sigma threshold). These are the "teacher" -- good enough to bootstrap the detector, but the DETR sees all 10 bands through the V7 encoder and can eventually surpass classical detection.
- **Loss**: position L1 (`lambda_pos=5.0`) + objectness BCE for matched queries (`lambda_conf=2.0`) + objectness suppression for unmatched queries (`lambda_noobj=0.5`).
- **Data**: 130 train / 14 val tiles with random 90-degree rotations and flips.

#### Files

| File | Description |
|------|-------------|
| `detector.py` | `JaispDetector` model, encoder wrapper, inference |
| `matcher.py` | Hungarian matcher + detection loss |
| `dataset.py` | Pseudo-label dataset with DETR-compatible collation |
| `train_detection.py` | Training loop with W&B logging |

### 2. Astrometry Concordance

**Directory**: `models/astrometry2/`

Measures and corrects the smooth astrometric distortion field between Rubin and Euclid VIS. Works by matching small image patches around detected sources, predicting per-source pixel offsets with uncertainty, then fitting a smooth spatial field.

#### Pipeline

```
For each tile:
  1. Detect sources in Rubin (classical or DETR) and VIS (classical)
  2. Match sources across instruments via WCS nearest-neighbor
  3. Extract 33x33 patches around each matched source
  4. Per-patch prediction:
       Frozen V7 BandStems -> ConvNeXt adapters
       -> spatially-weighted cost volume (cross-correlation)
       -> soft-argmax (differentiable coarse offset)
       -> MLP refinement -> (dx, dy, log_sigma) in pixels
       -> Jacobian transform -> (dRA*, dDec) in arcsec
  5. Fit smooth field from per-source offsets:
       Control-grid least squares (or MLP solver)
       with adaptive anchor regularization
  6. Export concordance FITS (dRA, dDec, coverage maps)
```

#### Matcher Architecture

Two versions available:

- **V6** (`matcher_v6.py`): uses frozen V6 Phase B BandStems. Best result: 31.9 mas.
- **V7** (`matcher_v7.py`): uses frozen V7 BandStems. Same architecture, different stem weights. API-compatible drop-in replacement.

Both support:
- Per-band embedding for wavelength-specific correction (multiband mode)
- Learnable softmax temperature for cost volume
- Separate learning rates: full LR for adapters/heads, 0.1x for stems if unfrozen

#### Source Detection Options

- **Classical** (default): `detect_sources()` -- median subtraction, Gaussian smoothing, local maxima above N-sigma threshold.
- **DETR** (optional): pass `--detr-checkpoint` to use the trained DETR detector for Rubin source finding. Produces more sources with potentially better positions than classical detection.

#### Field Solver

Two solvers for fitting the smooth concordance field:

- **Control grid** (`field_solver.py`): bilinear basis functions on a regular grid, least-squares with smoothness regularization and adaptive per-node anchor weights.
- **Neural network** (`nn_field_solver.py`): 4-layer MLP that maps (x,y) -> (dRA, dDec). No grid resolution hyperparameter; infinitely differentiable.

#### Files

| File | Description |
|------|-------------|
| `matcher_v6.py` | V6-based patch matcher (frozen V6 stems + adapters) |
| `matcher_v7.py` | V7-based patch matcher (frozen V7 stems + adapters) |
| `dataset.py` | Patch extraction, WCS matching, DETR integration |
| `source_matching.py` | Classical peak-finding, WCS-based source matching |
| `field_solver.py` | Control-grid least-squares field solver |
| `nn_field_solver.py` | MLP-based field solver |
| `train_astro_v6.py` | Training script (V6 backbone) |
| `train_astro_v7.py` | Training script (V7 backbone, optional DETR sources) |
| `infer_concordance.py` | Per-tile inference -> FITS export |
| `infer_global_concordance.py` | Global multi-tile concordance fitting |
| `apply_concordance.py` | Apply fitted concordance fields to data |
| `viz.py` | Diagnostic visualizations |

### 3. PSF + Forced Photometry

**Directory**: `models/photometry/`

Spatially-varying PSF model and Cramer-Rao optimal flux estimator.

- **PSFNet**: base Gaussian + learned residual in log-PSF space, conditioned on spatial position.
- **Matched filter**: `flux = (PSF^T W d) / (PSF^T W PSF)` where W is the inverse-variance weight matrix.
- **Pipeline**: precompute PSF grid per tile -> interpolate at source positions -> extract stamps -> subtract local background -> matched-filter flux.

See `models/photometry/README.md` for details.

---

## Project Structure

```
JAISP/
|
+-- README.md                          This file
+-- requirements.txt                   Python dependencies
|
+-- data/
|   +-- rubin_tiles_ecdfs/             Rubin survey tiles (*.npz)
|   +-- euclid_tiles_ecdfs/            Euclid survey tiles (*.npz)
|
+-- checkpoints/
|   +-- jaisp_v7_baseline/             Best V7 foundation checkpoint
|
+-- models/
|   +-- jaisp_foundation_v7.py         V7 mixed-resolution MAE (current)
|   +-- jaisp_foundation_v6.py         V6 single-grid MAE (library, used by V7)
|   +-- jaisp_dataset_v7.py            V7 data loader
|   +-- jaisp_dataset_v6.py            V6 data loader (library, used by downstream)
|   +-- train_jaisp_foundation_v7.py   V7 training entrypoint
|   +-- eval_foundation_v7.py          V7 evaluation/diagnostics
|   |
|   +-- detection/                     DETR source detection head
|   |   +-- detector.py
|   |   +-- matcher.py
|   |   +-- dataset.py
|   |   +-- train_detection.py
|   |
|   +-- astrometry2/                   Rubin<->Euclid concordance
|   |   +-- matcher_v7.py              V7 patch matcher
|   |   +-- matcher_v6.py              V6 patch matcher
|   |   +-- dataset.py                 Patch dataset + DETR integration
|   |   +-- source_matching.py         Classical detection utilities
|   |   +-- field_solver.py            Control-grid field solver
|   |   +-- nn_field_solver.py         MLP field solver
|   |   +-- train_astro_v7.py          V7 training script
|   |   +-- infer_concordance.py       Per-tile inference
|   |   +-- infer_global_concordance.py
|   |   +-- apply_concordance.py
|   |
|   +-- photometry/                    PSF + forced photometry
|   |   +-- psf_net.py
|   |   +-- train_psf_net.py
|   |   +-- forced_photometry.py
|   |   +-- pipeline.py
|   |
|   +-- older_architectures/           Archived experiments (v4, v5, etc.)
|   +-- checkpoints/                   Saved model weights
|
+-- io/                                Data I/O notebooks and scripts
+-- wandb/                             Experiment tracking logs
```

---

## Quick Start

### 1. Foundation Model Training

```bash
# V7 mixed-resolution MAE (current path)
python models/train_jaisp_foundation_v7.py \
    --rubin_dir  data/rubin_tiles_ecdfs \
    --euclid_dir data/euclid_tiles_ecdfs \
    --output_dir checkpoints/jaisp_v7_baseline \
    --hidden_ch 256 \
    --transformer_depth 4 \
    --transformer_heads 8 \
    --fused_pixel_scale_arcsec 0.8 \
    --cross_instrument_prob 1.0 \
    --epochs 100 \
    --wandb_name v7_h256_d4_fused0.8
```

### 2. Detection Head

```bash
# Train DETR detector on frozen V7 encoder
python models/detection/train_detection.py \
    --rubin_dir    data/rubin_tiles_ecdfs \
    --euclid_dir   data/euclid_tiles_ecdfs \
    --encoder_ckpt checkpoints/jaisp_v7_baseline/checkpoint_best.pt \
    --out          models/checkpoints/detector_v7.pt \
    --num_queries 500 \
    --epochs 100 \
    --wandb_project jaisp-detection
```

### 3. Astrometry Matcher

```bash
# V7 backbone with DETR source detection
python models/astrometry2/train_astro_v7.py \
    --v7-checkpoint  checkpoints/jaisp_v7_baseline/checkpoint_best.pt \
    --detr-checkpoint models/checkpoints/detector_v7.pt \
    --rubin-dir      data/rubin_tiles_ecdfs \
    --euclid-dir     data/euclid_tiles_ecdfs \
    --multiband \
    --output-dir     models/checkpoints/astro_v7 \
    --wandb-project  JAISP-Astrometry-v7

# V7 backbone with classical source detection (no DETR needed)
python models/astrometry2/train_astro_v7.py \
    --v7-checkpoint  checkpoints/jaisp_v7_baseline/checkpoint_best.pt \
    --rubin-dir      data/rubin_tiles_ecdfs \
    --euclid-dir     data/euclid_tiles_ecdfs \
    --multiband \
    --output-dir     models/checkpoints/astro_v7_classical
```

### 4. Concordance Inference

```bash
# Generate concordance FITS with V7 matcher + DETR sources
python models/astrometry2/infer_concordance.py \
    --checkpoint    models/checkpoints/astro_v7/checkpoint_best.pt \
    --v7-checkpoint checkpoints/jaisp_v7_baseline/checkpoint_best.pt \
    --detr-checkpoint models/checkpoints/detector_v7.pt \
    --rubin-dir     data/rubin_tiles_ecdfs \
    --euclid-dir    data/euclid_tiles_ecdfs \
    --output        concordance_v7.fits \
    --all-bands
```

### 5. PSF Training

```bash
python models/photometry/train_psf_net.py \
    --rubin_dir  data/rubin_tiles_ecdfs \
    --euclid_dir data/euclid_tiles_ecdfs \
    --out models/checkpoints/psf_net_v1.pt \
    --epochs 20
```

---

## Checkpoints

| Checkpoint | Location | Description |
|------------|----------|-------------|
| V7 foundation (best) | `checkpoints/jaisp_v7_baseline/checkpoint_best.pt` | hidden_ch=256, depth=4, epoch 86, val_loss=1.4689 |
| V7 detector | `models/checkpoints/detector_v7.pt` | DETR on V7 encoder, 500 queries |
| V6 astrometry | `models/checkpoints/astrometry_v6_phaseB2/checkpoint_best.pt` | Best astrometry result: 31.9 mas |

---

## Current Status

- **V7 foundation** is trained and stable. Best checkpoint: `jaisp_v7_baseline`.
- **Detection** is being retrained on V7 with improved pseudo-labels (3-sigma threshold, 500 queries, Euclid bands enabled).
- **Astrometry V7** matcher is implemented and ready for training once the detector converges.
- **Photometry** is functional but not yet integrated with V7.
- This is an active research codebase. Architecture and training defaults evolve with experiments.
