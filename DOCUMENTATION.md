# JAISP (Joint AI Survey Processing)

## Full Documentation

A self-supervised multi-instrument foundation model for precision cosmology with Rubin Observatory and Euclid, trained on overlapping imaging in the Extended Chandra Deep Field South (ECDFS).

---

## Table of Contents

1. [Motivation](#motivation)
2. [Data](#data)
3. [Foundation Model](#foundation-model)
   - [Architecture History (v1 through v7)](#architecture-history-v1-through-v7)
   - [v7 Mixed-Resolution MAE (Current)](#v7-mixed-resolution-mae-current)
4. [Downstream Heads](#downstream-heads)
   - [Detection (DETR)](#1-detection)
   - [Astrometry Concordance](#2-astrometry-concordance)
   - [PSF + Forced Photometry](#3-psf--forced-photometry)
5. [Project Structure](#project-structure)
6. [Quick Start](#quick-start)
7. [Checkpoints](#checkpoints)

---

## Motivation

Rubin Observatory's LSST and ESA's Euclid will together produce the deepest, widest multi-wavelength imaging survey ever conducted. They observe the same sky, but through very different eyes: Rubin captures six optical bands (u through y) at 0.2 arcsec/pixel over a 512x512 tile grid, while Euclid provides a single ultra-sharp visible channel (VIS) at 0.1 arcsec/pixel on a 1050x1050 grid, plus three near-infrared bands (Y, J, H) at 0.3 arcsec/pixel. Jointly analyzing these instruments enables science that neither can achieve alone -- sharper source detection by combining Euclid's resolution with Rubin's depth, sub-pixel astrometric alignment across surveys, and more precise photometric measurements that leverage all 10 wavelength channels simultaneously.

The challenge is that these instruments have different pixel scales, point-spread functions, noise properties, and coordinate systems. JAISP addresses this by learning a single spatially precise shared representation from both instruments through self-supervised pretraining, then attaching lightweight task-specific heads for detection, astrometry, and photometry. The central insight -- arrived at after five failed iterations -- is that **pixel-space reconstruction** via a Masked Autoencoder (MAE), where the model hides one band and learns to reconstruct it from the remaining bands, forces the encoder to preserve the exact spatial layout needed by precision cosmology tasks. Latent-space objectives like JEPA and contrastive learning optimize for "feature similarity" but allow the network to discard sub-pixel spatial information, which is precisely what astrometry and photometry demand.

### The Pipeline

The system works in two layers. First, a foundation model is trained once through self-supervised masked band prediction: given 9 of 10 bands, reconstruct the held-out band at pixel precision. This pretraining forces the encoder to learn cross-instrument spatial correspondence, noise properties, and spectral relationships without any labels. Second, the frozen encoder features are reused by three downstream heads, each of which trains only a small task-specific network on top.

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

This two-layer design means the expensive foundation pretraining only happens once. Each downstream task gets the benefit of 10-band multi-instrument features without paying the cost of encoding from scratch.

---

## Data

All data covers the Extended Chandra Deep Field South (ECDFS), a well-studied region of sky with deep multi-wavelength coverage. The field is tiled into approximately 1700 overlapping tiles. Each tile is stored as a compressed NumPy archive (.npz) containing the image data, variance maps, and WCS astrometric headers.

### Tiling and Overlap

Tiles are laid out on a regular grid with 256-pixel stride in both x and y, but each Rubin tile is 512x512 pixels. This means adjacent tiles overlap by **256 pixels (50%)** in each direction. A given point on the sky appears in up to 4 overlapping tiles.

This overlap has several benefits:

- **Foundation pretraining**: The same source appears in multiple tiles at different positions relative to tile edges. This acts as free data augmentation -- the model sees a galaxy near the center of one tile and near the edge of a neighbor, learning position-invariant features. This is particularly valuable given the current dataset size.
- **Detection**: Sources near tile edges (where detection is hardest) appear near the center of overlapping tiles, so the detector learns to find sources regardless of their position within a tile.
- **Astrometry**: Shared sources in overlap regions tie neighboring tiles' concordance solutions together in the global field fit (`infer_global_concordance.py`), enforcing continuity across the full survey footprint.

The downside is that the 144 tiles are not independent samples. With 50% overlap, there are roughly ~36 truly independent sky areas. This limits the diversity of source populations and noise realizations the model trains on, making the dataset expansion (currently in progress) important for generalization.

### Tile Size Rationale

The current 512x512 Rubin tile size (102" x 102" on sky) is a deliberate choice balancing several factors:

- **Transformer cost**: The V7 fused-scale bottleneck produces ~130x130 tokens (~17,000 per tile). Full self-attention is quadratic in token count, so doubling tile dimensions would increase attention cost ~9x. The current size is near the practical limit for dense self-attention without requiring sparse or windowed attention mechanisms.
- **Source density**: At 3-sigma detection in ECDFS, a 512x512 tile contains ~500 sources -- a good density for both detection training (enough targets per tile) and astrometry (enough anchors for the per-tile concordance fit).
- **Spatial context**: 102" spans ~50 PSF widths, which is more than enough context for learning source morphology, spectral relationships, and cross-instrument correspondence. Astronomical sources don't require arcminute-scale spatial context.
- **Astrometry field fitting**: While larger tiles would provide more baseline for per-tile concordance fits, this is not a constraint in practice because the global concordance solver already combines measurements from all tiles across the full survey footprint. The per-tile fit is just a local step; spatial coverage comes from having many tiles, not from making individual tiles larger.

As the dataset grows, the plan is to add more tiles at the same size rather than increase tile dimensions. More tiles provides more diverse training samples (different source populations, noise realizations, PSF conditions) without increasing per-sample computational cost. This is a more efficient use of additional data than larger tiles would be.

### Rubin NPZ files (`data/rubin_tiles_ecdfs/tile_x*_y*.npz`)

| Key      | Shape            | Description                        |
|----------|------------------|------------------------------------|
| `img`    | `[6, 512, 512]`  | Flux in 6 bands (u, g, r, i, z, y) |
| `var`    | `[6, 512, 512]`  | Variance per pixel (converted to RMS internally) |
| `wcs_hdr`| string           | FITS WCS header for astrometric calibration |

Pixel scale: 0.2 arcsec/pixel. Each tile covers roughly 102 x 102 arcsec on the sky.

### Euclid NPZ files (`data/euclid_tiles_ecdfs/tile_x*_y*_euclid.npz`)

| Key              | Shape             | Pixel Scale    |
|------------------|-------------------|----------------|
| `img_VIS`        | `[~1050, ~1050]`  | 0.1 arcsec/px  |
| `img_Y`, `img_J`, `img_H` | `[~350, ~350]` | 0.3 arcsec/px |
| `var_VIS/Y/J/H`  | same              | Variance       |
| `wcs_VIS/Y/J/H`  | string            | FITS WCS       |

Euclid VIS has twice the angular resolution of Rubin, which is why preserving it at native resolution (rather than downsampling to match Rubin) is so important for the foundation model design.

### Supported Bands

| Instrument | Bands                           | Wavelength Range | Count |
|------------|---------------------------------|------------------|-------|
| Rubin      | `rubin_u`, `rubin_g`, `rubin_r`, `rubin_i`, `rubin_z`, `rubin_y` | 320-1060 nm | 6 |
| Euclid     | `euclid_VIS`, `euclid_Y`, `euclid_J`, `euclid_H` | 550-2020 nm | 4 |
| **Total**  |                                 |                  | **10** |

Each band has its own `BandStem` -- a small per-band CNN that handles noise normalization and initial feature extraction. This per-band design allows the model to learn band-specific noise properties and PSF characteristics while producing a common feature representation for downstream fusion.

---

## Foundation Model

### Architecture History (v1 through v7)

The foundation model went through seven major iterations over the course of this project. Understanding this history is important because each version's failure revealed a specific insight about what self-supervised astronomical representations need. The overall arc is a progression from **latent-space alignment** (v1-v5) to **pixel-space reconstruction** (v6-v7), driven by the realization that precision cosmology demands sub-pixel spatial fidelity that contrastive and JEPA objectives fundamentally cannot enforce.

#### v1: Patch-Level Contrastive Learning

The first approach was conceptually straightforward: take large patches from Rubin and Euclid images at the same sky location, encode each through a Vision Transformer (ViT), and train with a contrastive loss (NT-Xent) to pull co-located patch pairs together while pushing non-overlapping pairs apart. Rubin patches were 192x192 pixels, Euclid patches 384x384 (accounting for the 2x pixel scale difference), and each was compressed to a single embedding vector.

This failed comprehensively. The core problem is that astronomical imaging is dominated by empty sky -- roughly 95% of pixels in any given patch are featureless background noise. When you compress an entire 192x192 patch to a single vector, the handful of galaxies and stars (occupying maybe 5% of the area) get averaged into the background. The model quickly learned that "flat Rubin background" matches "flat Euclid background" and collapsed, with separation metrics dropping to ~0.002. The embeddings carried no useful information about actual astronomical sources.

#### v2: Signal-Based Patch Sampling

Rather than abandon the patch-contrastive framework entirely, v2 attempted to fix the data problem. Instead of extracting random patches, it evaluated multiple candidate patches and selected those with the highest astronomical signal, weighted by inverse variance. The idea was to force the model to see patches containing actual galaxies and stars rather than empty sky.

This helped somewhat -- the model did learn slightly more meaningful embeddings -- but was ultimately a band-aid on a fundamental architectural flaw. The problem isn't just which patches you train on; it's that compressing any 192x192 astronomical image to a single vector inherently destroys the spatial information needed for astrometry. A galaxy's precise sub-pixel position cannot survive global average pooling. This realization led to rethinking the representation granularity entirely.

#### v3: DETR-JEPA (Object-Centric Learning)

The third approach asked: "What if we don't compress the whole patch, but instead let the model discover individual objects?" Inspired by DETR (Detection Transformer), v3 used a ViT backbone to produce spatial feature tokens, then fed them to a DETR-style decoder with 100 learnable object queries. Each query attended to the spatial features and was trained to "discover" one astronomical source. Hungarian matching found the optimal 1-to-1 correspondence between Rubin and Euclid object slots, and a contrastive loss pulled matched objects together.

This was a creative idea -- moving from patch-level to object-level learning, where each embedding corresponds to a single astronomical source. But in practice it was fragile. The Hungarian matching added instability during training, and the object queries didn't reliably converge to distinct sources in crowded deep-field images where hundreds of faint galaxies overlap. More fundamentally, even perfect per-object embeddings don't give you pixel-level spatial precision -- they tell you "these two objects are the same source" but not exactly where that source is to sub-pixel accuracy.

#### v4: Native-Resolution JEPA with InformationMap

v4 made a critical architectural shift: instead of extracting patches, process the full 512x512 tile at native resolution. This eliminated the information loss from patching entirely. The architecture used per-band CNN stems with noise normalization, a shared ViT-like trunk with positional encodings, and a BYOL/JEPA-style student-teacher framework with exponential moving average (EMA).

Two important innovations appeared in v4. First, **InformationMap weighting**: instead of treating all pixels equally, the loss was weighted by a signal-to-noise map combined with Sobel gradient magnitudes. This naturally focused learning on source pixels (high SNR, strong gradients) rather than empty background, solving the background-dominance problem without resorting to patch sampling. InformationMap weighting proved valuable enough to survive into v6 and v7.

Second, v4 introduced a **shift-tolerant alignment loss** that allowed tokens to match within a +/-5 pixel tolerance window. The reasoning was that Rubin and Euclid have genuine sub-pixel astrometric misalignments, so forcing exact positional matching would create conflicting gradients.

The shift tolerance turned out to be a mistake. By allowing 5 pixels of slack, the model had no incentive to learn precise spatial correspondence -- it could satisfy the loss with spatially imprecise features. This is the opposite of what astrometry needs. The EMA teacher also added complexity without clear benefit over simpler training schemes.

#### v5: Strict-Position JEPA

v5 was a targeted fix for v4's spatial imprecision: remove the shift tolerance entirely (`shift_px=0`) and force exact token-to-token matching at corresponding spatial positions. Everything else remained the same -- InformationMap weighting, per-band stems, ViT backbone, VICReg regularization to prevent collapse.

This version exposed the fundamental limits of the JEPA approach for precision cosmology. Three problems compounded:

1. **Resolution ceiling**: The ViT used 16x16 patch tokens, meaning each token covered 3.2 arcseconds on the sky. Astrometry needs precision below 0.2 arcseconds -- the tokenization itself is too coarse by an order of magnitude.
2. **Latent-space loss is the wrong objective**: Cosine similarity between token embeddings rewards "similar features" but doesn't require the network to preserve exact spatial layout. Two tokens can be highly similar in embedding space while differing in the precise sub-pixel positions of the sources they encode.
3. **Strict matching vs real misalignments**: Real Rubin-Euclid data has genuine 0.25-0.5 pixel instrument misalignments. Forcing exact token-to-token matching on misaligned data creates conflicting supervision signals that prevent convergence.

The decisive evidence came from a simple baseline comparison: a straightforward CNN with a cost volume (the astrometry2 module) achieved 38 milliarcsecond (mas) accuracy on the astrometry task, while v5's JEPA features only managed 47 mas. The expensive self-supervised representation was *worse* than a simple supervised CNN. This proved that the JEPA approach, regardless of how it was tuned, was not learning the spatial information that precision cosmology requires.

**The key lesson from v1-v5**: Latent-space alignment objectives -- whether contrastive (v1-v2), object-centric (v3), or JEPA-style (v4-v5) -- optimize for feature similarity, not spatial precision. They allow the network to learn "this region looks like that region" without knowing exactly where things are at the sub-pixel level. For precision cosmology, you need the encoder to preserve exact pixel positions. The only way to guarantee this is to require the network to actually reconstruct pixels.

#### v6: Masked Band Prediction (Dense Reconstruction)

v6 represents the fundamental paradigm shift from latent-space alignment to pixel-space reconstruction. Instead of making embeddings match across instruments, the model is trained to predict a held-out band's pixel values from the remaining bands. This is a masked autoencoder (MAE), but operating on wavelength bands rather than spatial patches.

The architecture replaced the ViT with a dense convolutional pipeline: per-band CNN stems (BandStem with GroupNorm for batch-size-1 compatibility), a ConvNeXt encoder with three stride-2 downsampling stages producing dense feature maps at H/8 resolution, a transformer bottleneck operating on these dense tokens, and a U-Net decoder with skip connections that reconstructs back to full resolution. FiLM (Feature-wise Linear Modulation) conditioning tells the decoder which band to predict. The loss is InformationMap-weighted L1 in noise-normalized units -- the same signal-aware weighting that proved valuable in v4, now applied to a reconstruction objective. Total: 20.8M parameters.

Training used a two-phase curriculum:
- **Phase A** (`cross_instrument_prob=0.0`): Rubin-only. Mask one Rubin band, reconstruct it from the other five. This teaches the model spectral relationships and spatial structure within one instrument.
- **Phase B** (`cross_instrument_prob=1.0`): Joint Rubin + Euclid. Mask any one of the 10 bands, reconstruct it from the other 9. This teaches cross-instrument spatial correspondence, since reconstructing a Euclid band from Rubin features (or vice versa) requires the encoder to learn precise alignment.

The reason this works is simple and powerful: to reconstruct a held-out band at the pixel level, the encoder *must* preserve sub-pixel spatial information. If a galaxy is at position (245.3, 167.8) in the input bands, the decoder needs to place reconstructed flux at exactly that position in the output. There is no shortcut -- you can't get high pixel-level fidelity without encoding precise positions. This is exactly the spatial precision that astrometry, detection, and photometry need downstream.

**Limitation**: Phase B downsampled Euclid VIS (1050x1050 at 0.1"/px) to Rubin's 512x512 grid before encoding. This was a pragmatic choice to avoid dealing with mixed resolutions, but it discarded the 2x resolution advantage that makes VIS the most valuable single channel for astrometry and deblending.

#### v7: Mixed-Resolution MAE (current)

v7 fixes v6's resolution bottleneck. Instead of forcing all instruments onto one pixel grid, each instrument processes at its native resolution through independent encoder branches with different depths. The branches are designed so that after their respective downsampling stages, all streams arrive at approximately the same physical angular scale (~0.8 arcsec/pixel). At this common physical scale they fuse into a shared latent representation, pass through a transformer bottleneck, and then decode back to the target band's native resolution.

This means VIS features are never downsampled to Rubin's coarser grid. When the model reconstructs a VIS band, it decodes to the full 1050x1050 resolution. When it reconstructs a Rubin band, it decodes to 512x512. The encoder learns to preserve each instrument's native spatial information throughout.

See the next section for the full v7 architecture.

### Version Summary

| Version | Approach | Key Idea | Outcome |
|---------|----------|----------|---------|
| v1 | Patch contrastive | Match Rubin-Euclid patch embeddings | Failed: background collapse |
| v2 | Patch contrastive | Signal-based patch selection | Abandoned: patch-level still lossy |
| v3 | DETR-JEPA | Object-level manifold matching | Abandoned: complexity, no precision gain |
| v4 | Native-res JEPA | InformationMap + shift tolerance | Superseded: spatially imprecise |
| v5 | Native-res JEPA | Strict position matching | Failed: JEPA can't enforce pixel precision |
| v6 | Dense MAE | Pixel-space reconstruction | Works: 31.9 mas astrometry |
| v7 | Mixed-res MAE | Native resolution per instrument | **Current**: preserves VIS advantage |

### v7 Mixed-Resolution MAE (Current)

**File**: `models/jaisp_foundation_v7.py`

The v7 architecture has three main stages: per-instrument encoding at native resolution, cross-instrument fusion at a shared physical scale, and target-specific decoding back to native resolution.

**Encoding**: Each instrument has its own encoder branch. Rubin's six bands each pass through a BandStem (a small CNN that noise-normalizes and extracts local features), then the stem outputs are averaged and fed through a StreamEncoder with 2 ConvNeXt downsampling stages. VIS uses 3 stages (because it starts at finer resolution and needs more downsampling to reach the common scale). NISP uses 1 stage. The branch depths are chosen so that all streams converge to approximately 0.8 arcsec/pixel -- this is a physics-grounded design where the fusion happens at matched angular resolution, not matched pixel count.

**Fusion**: The encoded streams are interpolated to a common spatial grid, summed with learned stream identity embeddings (so the transformer can distinguish Rubin features from VIS features from NISP features), and passed through a transformer bottleneck with 4 layers and 8 attention heads operating on approximately 130x130 tokens with 2D sinusoidal positional encodings.

**Decoding**: A TargetDecoder upsamples back to the target band's native resolution using transposed convolutions and skip connections. The skip connections are routed from whichever encoder pyramid level has the closest matching physical scale, fusing information across all instrument streams at each decoder stage. FiLM conditioning tells the decoder which specific band to reconstruct.

```
Rubin bands (512x512, 0.2"/px)   -> BandStems -> Rubin branch (2 stages) --\
VIS band   (1050x1050, 0.1"/px)  -> BandStem  -> VIS branch   (3 stages) --> latent @ 0.8"/px
NISP bands (350x350, 0.3"/px)    -> BandStems -> NISP branch  (1 stage)  --/    (~130x130 tokens)
                                                                                       |
                                                                        Stream mean fusion +
                                                                        learned stream embeddings
                                                                                       |
                                                                        Transformer bottleneck
                                                                        (depth=4, heads=8)
                                                                                       |
                                                                        TargetDecoder with
                                                                        pyramid skip connections
                                                                                       |
                                                                        Native-resolution output
                                                                        (VIS->1050, NISP->350,
                                                                         Rubin->512)
```

**Training**: Unlike v6's two-phase curriculum, v7 training is unified from epoch 1. Tiles with Euclid coverage always use cross-instrument masking; Rubin-only tiles (those without a matching Euclid file) automatically fall back to within-instrument prediction. The loss is the same InformationMap-weighted L1 as v6.

**Best checkpoint** (`jaisp_v7_baseline`):

| Parameter | Value |
|-----------|-------|
| `stem_ch` | 64 |
| `hidden_ch` | 256 |
| `transformer_depth` | 4 |
| `transformer_heads` | 8 |
| `fused_pixel_scale_arcsec` | 0.8 |
| `cross_instrument_prob` | 1.0 |
| Epoch | 86 |
| Val loss | 1.4689 |
| Total params | 16.0M |

---

## Downstream Heads

All downstream heads reuse the frozen V7 foundation encoder. Only lightweight task-specific layers are trained on top. This means each head gets the benefit of the full 10-band multi-instrument representation without the cost of encoding from scratch, and training each head is fast (hours, not days).

### 1. Detection

**Directory**: `models/detection/`

The detection head finds astronomical sources (galaxies and stars) in tiles using a DETR-style set prediction approach. Unlike classical detection pipelines that operate on a single coadded image, this detector sees all available bands through the V7 encoder's learned features, giving it access to sources that may only be visible in specific wavelength ranges.

#### How It Works

The frozen V7 encoder processes the multi-band input tile and produces a bottleneck feature map at approximately 130x130 spatial resolution. A 1x1 convolution projects these features to the detection head's working dimension, and 2D sinusoidal positional encodings are added so the transformer decoder knows where each feature is on the sky. The transformer decoder then takes 500 learned object queries -- think of these as 500 "slots" that each learn to claim one source -- and cross-attends to the spatial features. Each query outputs a predicted centroid position (normalized to [0,1]), a confidence score (is this slot pointing at a real source or empty sky?), and a log-flux estimate.

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

Since there is no curated source catalog for this field, the detector is trained on **pseudo-labels** generated by classical peak-finding on a Rubin g+r+i coadd image. This classical detector (median subtraction, Gaussian smoothing, local maxima above 3-sigma) serves as the "teacher". The DETR is the "student" that sees all 10 bands through the V7 encoder and can eventually surpass the teacher by detecting sources invisible in just g+r+i.

Training uses Hungarian matching to assign each of the 500 predicted queries to the best-matching ground-truth source (or to "no object" if the query doesn't match anything). The loss combines position accuracy for matched queries with confidence suppression for unmatched ones:

| Loss Term | Weight | Purpose |
|-----------|--------|---------|
| `loss_pos` | 5.0 | L1 on matched centroid positions -- teaches where sources are |
| `loss_conf_obj` | 2.0 | BCE pushing matched queries toward confidence=1 |
| `loss_conf_noobj` | 0.5 | BCE pushing unmatched queries toward confidence=0 |

The training data consists of 130 training tiles and 14 validation tiles with random 90-degree rotations and flips for augmentation.

#### Files

| File | Description |
|------|-------------|
| `detector.py` | `JaispDetector` model, encoder wrapper, inference |
| `matcher.py` | Hungarian matcher + detection loss |
| `dataset.py` | Pseudo-label dataset with DETR-compatible collation |
| `train_detection.py` | Training loop with W&B logging |

### 2. Astrometry Concordance

**Directory**: `models/astrometry2/`

Astrometry concordance measures and corrects the smooth spatial distortion between Rubin and Euclid coordinate systems. Even after standard WCS calibration, there are residual sub-arcsecond offsets that vary smoothly across a tile. These must be corrected for joint analysis -- stacking images, measuring galaxy shapes for weak lensing, or combining photometry across instruments.

The approach works at the individual source level: detect sources in both instruments, match them across surveys, extract small image patches around each matched pair, predict the precise pixel offset between the Rubin and VIS positions of each source, and then fit a smooth spatial field from all these per-source measurements.

#### Pipeline

For each tile, the pipeline proceeds through six stages:

1. **Source detection**: Find sources in the Rubin tile (using either classical peak-finding or the trained DETR detector) and in the Euclid VIS image (classical peak-finding). The DETR option is valuable because it can find fainter sources, giving more anchor points for the field fit.

2. **Cross-instrument matching**: Match Rubin and VIS source lists using WCS sky coordinates. Mutual nearest-neighbor matching with separation limits and sigma-clipping removes spurious matches. Typically yields 50-200 matched pairs per tile.

3. **Patch extraction**: For each matched source, extract a 33x33 pixel patch centered on the VIS position, and reproject the corresponding Rubin bands onto the VIS pixel grid. Each patch pair shows the same source as seen by both instruments.

4. **Per-patch offset prediction**: The core of the matcher. Frozen V7 BandStems extract features from the Rubin and VIS patches independently, then trainable ConvNeXt adapter blocks refine these features toward astrometry-relevant signals. A spatially-weighted cross-correlation (cost volume) between the two feature maps produces a coarse offset via differentiable soft-argmax. An MLP refines this with a residual correction and outputs `(dx, dy, log_sigma)` in pixels, which a Jacobian matrix transforms to `(dRA*, dDec)` in arcseconds. The uncertainty `sigma` is used to weight the field fit.

5. **Smooth field fitting**: The per-source offsets are noisy -- each individual measurement has ~30-50 mas scatter. But the true distortion varies smoothly across the tile. A control-grid least-squares solver (bilinear basis functions with smoothness regularization and adaptive per-node anchor weights) fits a smooth 2D field from the noisy per-source measurements. An alternative MLP-based solver is also available.

6. **Export**: The fitted concordance field is evaluated on a regular mesh and written to a FITS file as `dRA`, `dDec`, and coverage maps.

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

#### Matcher Versions

Two matcher versions are available, differing only in which foundation model's BandStems they reuse:

- **V6** (`matcher_v6.py`): Uses frozen V6 Phase B BandStems. These were trained with cross-instrument masking where VIS was downsampled to Rubin resolution. Best result: 31.9 mas median astrometric precision.
- **V7** (`matcher_v7.py`): Uses frozen V7 BandStems, where VIS was processed at native resolution. API-compatible drop-in replacement for V6.

Both versions support multiband mode (per-band learned embeddings for wavelength-specific chromatic correction) and separate learning rates (full LR for adapters and heads, 0.1x for stems if unfrozen for fine-tuning).

#### Source Detection Options

The first stage of the pipeline -- detecting sources in Rubin -- can use either:

- **Classical** (default): `detect_sources()` performs median background subtraction, Gaussian smoothing, and local maximum detection above an N-sigma threshold. Simple, fast, and well-understood.
- **DETR** (optional): Pass `--detr-checkpoint` to replace classical Rubin detection with the trained DETR detector. The DETR sees all 10 bands through the V7 encoder and can detect fainter sources that are invisible in a 3-band coadd, providing more anchor points for the concordance fit. VIS detection still uses the classical method.

#### Field Solvers

Two solvers are available for fitting the smooth concordance field from per-source offset measurements:

- **Control grid** (`field_solver.py`): Fits bilinear basis functions on a regular grid using weighted least squares. Includes finite-difference smoothness regularization and adaptive per-node anchor weights that prevent edge drift in regions with sparse source coverage. The grid resolution is automatically reduced for tiles with few matches to avoid underdetermined systems.
- **Neural network** (`nn_field_solver.py`): A 4-layer MLP that maps normalized (x, y) tile coordinates to (dRA, dDec) offsets, trained via Adam for 2000 steps. Has no grid resolution hyperparameter, is infinitely differentiable, and scales naturally to any source density.

#### Files

| File | Description |
|------|-------------|
| `matcher_v6.py` | V6-based patch matcher (frozen V6 stems + trainable adapters) |
| `matcher_v7.py` | V7-based patch matcher (frozen V7 stems + trainable adapters) |
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

Accurate photometry requires knowing the point-spread function (PSF) -- the shape of a point source as recorded by the detector. The PSF varies across the field of view due to optical distortions and detector effects. This module models the spatially-varying PSF and uses it for optimal flux extraction.

- **PSFNet**: A neural network that parameterizes the PSF as a base Gaussian plus a learned residual in log-PSF space, conditioned on position within the tile. This allows the model to capture complex PSF wings and asymmetries that vary smoothly across the field.
- **Matched filter**: Given the PSF model at a source's position, the optimal linear flux estimator is `flux = (PSF^T W d) / (PSF^T W PSF)` where W is the inverse-variance weight matrix and d is the observed pixel data. This is the Cramer-Rao optimal estimator -- no other linear method can achieve lower variance.
- **Pipeline**: For each tile, precompute a PSF grid at regular positions, interpolate to each detected source's location, extract small image stamps, subtract local background, and apply the matched-filter flux estimator.

See `models/photometry/README.md` for full details.

---

## Project Structure

```
JAISP/
|
+-- README.md                          Short project overview
+-- DOCUMENTATION.md                   This file (full documentation)
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
|   +-- older_architectures/           Archived experiments (v1-v5)
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

- **V7 foundation** is trained and stable. Best checkpoint: `jaisp_v7_baseline` (epoch 86, val_loss 1.4689).
- **Detection** is being retrained on V7 with improved pseudo-labels (3-sigma threshold, 500 queries, Euclid bands enabled). Early training shows steady val loss improvement.
- **Astrometry V7** matcher is implemented (`matcher_v7.py`, `train_astro_v7.py`) and ready for training once the detector converges. The V6 matcher achieved 31.9 mas; V7 should improve on this by preserving VIS native resolution.
- **Photometry** is functional but not yet integrated with V7.
- This is an active research codebase. Architecture and training defaults evolve with experiments.
