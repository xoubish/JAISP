# JAISP (Joint AI Survey Processing)

## Full Documentation

A self-supervised multi-instrument foundation model for precision cosmology with Rubin Observatory and Euclid. The project was developed first on a 144-tile ECDFS subset and is now being retrained on a larger flat Rubin+Euclid tile extraction (`data/rubin_tiles_all/`, `data/euclid_tiles_all/`) containing 790 matched pairs.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Data](#data)
3. [Foundation Model](#foundation-model)
   - [Architecture History (v1 through v7)](#architecture-history-v1-through-v7)
   - [v7 Mixed-Resolution MAE (Current)](#v7-mixed-resolution-mae-current)
4. [Downstream Heads](#downstream-heads)
   - [Detection](#1-detection)
   - [Astrometry Concordance](#2-astrometry-concordance)
   - [PSF + Forced Photometry](#3-psf--forced-photometry)
5. [Project Structure](#project-structure)
6. [Quick Start](#quick-start)
7. [Checkpoints](#checkpoints)

---

## Motivation

Rubin Observatory's LSST and ESA's Euclid will together produce the deepest, widest multi-wavelength imaging survey ever conducted. They observe the same sky, but through very different eyes: Rubin captures six optical bands (u through y) at 0.2 arcsec/pixel over a 512x512 tile grid, while Euclid provides a single ultra-sharp visible channel (VIS) at 0.1 arcsec/pixel on a ~1084x1084 grid, plus three near-infrared bands (Y, J, H) delivered as MER mosaics at the same 0.1 arcsec/pixel scale. Jointly analyzing these instruments enables science that neither can achieve alone -- sharper source detection by combining Euclid's resolution with Rubin's depth, sub-pixel astrometric alignment across surveys, and more precise photometric measurements that leverage all 10 wavelength channels simultaneously.

The challenge is that these instruments have different pixel scales, point-spread functions, noise properties, and coordinate systems. JAISP addresses this by learning a single spatially precise shared representation from both instruments through self-supervised pretraining, then attaching lightweight task-specific heads for detection, astrometry, and photometry. The central insight -- arrived at after five failed iterations -- is that **pixel-space reconstruction** via a Masked Autoencoder (MAE), where the model hides one band and learns to reconstruct it from the remaining bands, forces the encoder to preserve the exact spatial layout needed by precision cosmology tasks. Latent-space objectives like JEPA and contrastive learning optimize for "feature similarity" but allow the network to discard sub-pixel spatial information, which is precisely what astrometry and photometry demand.

### The Pipeline

The system works in two layers. First, a foundation model is trained once through self-supervised masked band prediction: given 9 of 10 bands, reconstruct the held-out band at pixel precision. This pretraining forces the encoder to learn cross-instrument spatial correspondence, noise properties, and spectral relationships without any labels. Second, the frozen encoder features are reused by three downstream heads, each of which trains only a small task-specific network on top.

```
 Rubin tiles (6 bands, 512x512, 0.2"/px)
 Euclid tiles (VIS + NISP Y/J/H, all ~1084x1084 @ 0.1"/px from MER mosaics)
         |
    [ Foundation Model (self-supervised MAE) ]
         |
         +-- frozen encoder features
         |
    +----+----+--------------------+
    |         |                    |
 Detection  Astrometry         Photometry
 (3 choices)(patch matching)   (PSF + matched filter)
```

This two-layer design means the expensive foundation pretraining only happens once. Each downstream task gets the benefit of 10-band multi-instrument features without paying the cost of encoding from scratch.

---

## Data

The repo currently contains two compatible tile products:

- **Legacy ECDFS development subset**: `data/rubin_tiles_ecdfs/` and `data/euclid_tiles_ecdfs/`, with 144 matched Rubin+Euclid pairs. These are the tiles many earlier experiments and downstream prototypes were built on.
- **Current flat training set**: `data/rubin_tiles_all/` and `data/euclid_tiles_all/`, extracted from `data/tiles_product.tar.gz`. This set currently contains 790 matched Rubin+Euclid pairs plus one extra readable Euclid-only tile that Rubin-driven loaders ignore.

Both products use the same NPZ schemas. In the flat set, filenames encode tract/patch metadata directly, for example `tile_x02816_y00512_tract5063_patch_14.npz` and `tile_x02816_y00512_tract5063_patch_14_euclid.npz`.

![Tile coverage map](docs/figures/downloaded_patches.png)
*Left: Spatial distribution of Rubin tile centers by patch, covering the ECDFS field. Right: Tile counts per patch showing matched Rubin+Euclid pair availability.*

![10-band science tile](docs/figures/random_10bandtile.png)
*All 10 bands for a sample matched tile. Top row: Rubin u/g/r/i/z/y (512x512, 0.2"/px). Bottom row: Euclid VIS and NISP Y/J/H (all ~1084x1084, 0.1"/px from MER mosaics). Note the different noise properties across bands.*

![10-band RMS tile](docs/figures/random_10bandtile_rms.png)
*Per-pixel RMS (noise) maps for the same tile, derived from the variance arrays in the NPZ files. Rubin RMS shows chip-edge effects and depth variations. Euclid NISP RMS reveals satellite trails and detector artifacts. These maps are used for per-pixel noise normalization in the foundation model BandStems and in the astrometry matcher.*

### Tiling and Overlap

Tiles are laid out on a regular grid with 256-pixel stride in both x and y, but each Rubin tile is 512x512 pixels. This means adjacent tiles overlap by **256 pixels (50%)** in each direction. A given point on the sky appears in up to 4 overlapping tiles.

This overlap has several benefits:

- **Foundation pretraining**: The same source appears in multiple tiles at different positions relative to tile edges. This acts as free data augmentation -- the model sees a galaxy near the center of one tile and near the edge of a neighbor, learning position-invariant features. This is particularly valuable given the current dataset size.
- **Detection**: Sources near tile edges (where detection is hardest) appear near the center of overlapping tiles, so the detector learns to find sources regardless of their position within a tile.
- **Astrometry**: Shared sources in overlap regions tie neighboring tiles' concordance solutions together in the global field fit (`infer_global_concordance.py`), enforcing continuity across the full survey footprint.

The downside is that tile count overstates statistical independence. In the legacy 144-tile ECDFS subset, 50% overlap means there are only roughly ~36 truly independent sky areas. The expanded flat set improves sample count substantially, but overlap still matters when designing train/val/test splits or making final downstream performance claims.

### Tile Size Rationale

The current 512x512 Rubin tile size (102" x 102" on sky) is a deliberate choice balancing several factors:

- **Transformer cost**: The V7 fused-scale bottleneck produces ~130x130 tokens (~17,000 per tile). Full self-attention is quadratic in token count, so doubling tile dimensions would increase attention cost ~9x. The current size is near the practical limit for dense self-attention without requiring sparse or windowed attention mechanisms.
- **Source density**: At 3-sigma detection in ECDFS, a 512x512 tile contains ~500 sources -- a good density for both detection training (enough targets per tile) and astrometry (enough anchors for the per-tile concordance fit).
- **Spatial context**: 102" spans ~50 PSF widths, which is more than enough context for learning source morphology, spectral relationships, and cross-instrument correspondence. Astronomical sources don't require arcminute-scale spatial context.
- **Astrometry field fitting**: While larger tiles would provide more baseline for per-tile concordance fits, this is not a constraint in practice because the global concordance solver already combines measurements from all tiles across the full survey footprint. The per-tile fit is just a local step; spatial coverage comes from having many tiles, not from making individual tiles larger.

As the dataset grows, the plan is to add more tiles at the same size rather than increase tile dimensions. More tiles provides more diverse training samples (different source populations, noise realizations, PSF conditions) without increasing per-sample computational cost. This is a more efficient use of additional data than larger tiles would be.

### Rubin NPZ files (`data/rubin_tiles_all/tile_x*_y*.npz`)

| Key      | Shape            | Description                        |
|----------|------------------|------------------------------------|
| `img`    | `[6, 512, 512]`  | Flux in 6 bands (u, g, r, i, z, y) |
| `var`    | `[6, 512, 512]`  | Variance per pixel (converted to RMS internally) |
| `wcs_hdr`| string           | FITS WCS header for astrometric calibration |

Pixel scale: 0.2 arcsec/pixel. Each tile covers roughly 102 x 102 arcsec on the sky. The legacy `data/rubin_tiles_ecdfs/` directory uses the same schema.

### Euclid NPZ files (`data/euclid_tiles_all/tile_x*_y*_euclid.npz`)

| Key              | Shape             | Pixel Scale    |
|------------------|-------------------|----------------|
| `img_VIS`        | `[~1084, ~1084]`  | 0.1 arcsec/px  |
| `img_Y`, `img_J`, `img_H` | `[~1084, ~1084]` | 0.1 arcsec/px (MER mosaics) |
| `var_VIS/Y/J/H`  | same              | Variance       |
| `wcs_VIS/Y/J/H`  | string            | FITS WCS       |

Euclid VIS has twice the angular resolution of Rubin, which is why preserving it at native resolution (rather than downsampling to match Rubin) is so important for the foundation model design. The legacy `data/euclid_tiles_ecdfs/` directory uses the same schema.

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

Two important innovations appeared in v4. First, **InformationMap weighting**: instead of treating all pixels equally, the loss was weighted by a signal-to-noise map combined with Sobel gradient magnitudes. This naturally focused learning on source pixels (high SNR, strong gradients) rather than empty background, solving the background-dominance problem without resorting to patch sampling. InformationMap weighting proved valuable enough to survive into v6 and v7, where it was extended with an RMS-adaptive minimum weight floor to prevent hallucination in noisy bands (see v7 Training).

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

v7 fixes v6's resolution bottleneck. Instead of forcing all instruments onto one pixel grid, each instrument processes at its native resolution through independent encoder branches with different depths. The branches are designed so that after their respective downsampling stages, both streams arrive at approximately the same physical angular scale (~0.8 arcsec/pixel). At this common physical scale they fuse into a shared latent representation, pass through a transformer bottleneck, and then decode back to the target band's native resolution.

A key design choice is how per-band features are aggregated within each stream. The Euclid stream uses **fixed-slot concatenation** followed by a learned 1×1 projection, preserving per-band PSF and color structure (VIS PSF: 0.2" vs NISP PSF: ~0.5") through the entire encoder. The Rubin stream uses mean pooling (all 6 optical bands have similar PSFs). This asymmetric design ensures the encoder can learn band-specific spatial features for Euclid while keeping the Rubin path efficient.

VIS features are never downsampled to Rubin's coarser grid. When the model reconstructs any Euclid band, it decodes to the full ~1084x1084 resolution. When it reconstructs a Rubin band, it decodes to 512x512. The encoder learns to preserve each instrument's native spatial information throughout.

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
| v7 | Mixed-res MAE | 2-stream (Rubin mean / Euclid concat), native resolution | **Current**: preserves per-band PSF structure |

### v7 Mixed-Resolution MAE (Current)

**File**: `models/jaisp_foundation_v7.py`

The v7 architecture has three main stages: per-instrument encoding at native resolution, cross-instrument fusion at a shared physical scale, and target-specific decoding back to native resolution.

**Encoding**: The model has two instrument streams, each with its own encoder branch:

- **Rubin stream**: Six BandStems (one per optical band) produce per-band feature maps. These are **mean-pooled** into a single tensor and fed through a StreamEncoder with 2 ConvNeXt downsampling stages. Mean pooling is acceptable here because all Rubin bands have similar PSFs (~0.7-1.0") and variable band availability (some tiles may lack u or y) is handled gracefully.
- **Euclid stream**: Four BandStems (VIS, Y, J, H) produce per-band feature maps. These are **concatenated** into fixed slots (4 × 64 = 256 channels, with zero-filled slots for masked bands during MAE training) and projected back to 64 channels via a learned 1×1 convolution. This preserves per-band structure through the entire encoder -- critical because the Euclid bands have very different PSFs (VIS: 0.2", NISP Y/J/H: ~0.5") and the encoder needs to learn band-specific spatial features for downstream photometry and deblending. The fused features pass through a StreamEncoder with 3 ConvNeXt downsampling stages.

The branch depths are chosen so that both streams converge to approximately 0.8 arcsec/pixel -- this is a physics-grounded design where the fusion happens at matched angular resolution, not matched pixel count.

**Fusion**: The encoded streams are interpolated to a common spatial grid, summed with learned stream identity embeddings (so the transformer can distinguish Rubin from Euclid features), and passed through a transformer bottleneck with 4 layers and 8 attention heads operating on approximately 132×132 tokens with 2D sinusoidal positional encodings.

**Decoding**: A per-stream TargetDecoder upsamples back to the target band's native resolution using bilinear interpolation and skip connections. The skip connections are routed from whichever encoder pyramid level has the closest matching physical scale, fusing information across both instrument streams at each decoder stage. FiLM conditioning tells the decoder which specific band to reconstruct.

```
Rubin:  6 BandStems -> mean pool -> [64, 512, 512]      -> 2-stage encoder --\
                                                                               --> latent @ 0.8"/px
Euclid: 4 BandStems -> concat+1×1 proj -> [64, 1084, 1084] -> 3-stage encoder --/    (~132×132 tokens)
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

The Euclid concat+project design (via the `StreamFuser` module) is the key architectural difference from earlier versions. By preserving per-band information through the encoder, the model can learn that the same galaxy looks different in VIS vs H-band due to PSF differences -- exactly the information that photometry and deblending need. During MAE training, when one Euclid band is masked as the reconstruction target, its slot is zero-filled; the 1×1 projection learns to ignore zeros, so the encoder gracefully handles variable band availability.

NISP Y/J/H data comes from Euclid MER mosaics, already resampled to 0.1"/px (same as VIS).

**Training**: Unlike v6's two-phase curriculum, v7 training is unified from epoch 1. Tiles with Euclid coverage use cross-instrument masking; Rubin-only tiles automatically fall back to within-instrument prediction. In the current flat training set, the Rubin side is effectively fully paired (790 matched pairs), so almost every sample participates in cross-instrument learning.

**Loss**: InformationMap-weighted L1 in noise-normalized (SNR) space, with two RMS-aware mechanisms:

1. **RMS-adaptive InformationMap floor**: The original InformationMap used a fixed minimum weight (`min_weight=0.001`) for blank-sky pixels. This meant that for noisy bands (u, y) where almost no pixels exceed the SNR threshold, the model could hallucinate sources at near-zero loss cost -- the info weights were negligible at blank-sky locations, so false sources went unpunished. The adaptive floor raises the minimum weight based on the tile's mean RMS: `adaptive_min = 0.001 + sigmoid(mean_rms - 1.0) * 0.3`. Bands with higher noise get a higher floor, ensuring blank-sky pixels contribute meaningfully to the loss and penalizing hallucinations.

2. **Tile-level RMS band weight**: The per-target loss is multiplied by the target band's mean RMS across the tile: `loss = mean_rms * pixel_loss`. In noise-normalized space, noisy bands naturally produce smaller loss magnitudes (targets are flatter). This multiplicative weight compensates, giving noisy bands proportionally larger gradients so the model cannot coast on the easy high-SNR bands (g/r/i/z).

Training uses mixed-precision (bfloat16 autocast) and supports multi-GPU via `torchrun` with DistributedDataParallel.

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
| Total params | 16.0M (old 3-stream) / 13.3M (current 2-stream concat) |

**`v7_rms_aware_loss` run** (2026-04-07, [wandb](https://wandb.ai/AI-Astro/JAISP-Foundation-v7/runs/x9y9os7r)):

| Metric | Value |
|--------|-------|
| val/best_loss | 4.0493 |
| val/loss | 4.0515 |
| train/epoch_loss | 4.7354 |

Per-band reconstruction quality (bright sources):

| Band | Train Loss | MAE | Pearson r | std ratio |
|------|-----------|-----|-----------|-----------|
| Rubin g | 1.83 | 1.206 | 0.998 | 1.011 |
| Rubin r | 1.39 | 1.020 | 0.998 | 1.004 |
| Rubin i | 2.52 | 1.265 | 0.997 | 1.013 |
| Rubin z | 4.32 | 0.802 | 0.989 | 1.001 |
| Rubin u | 5.33 | 0.808 | 0.968 | 0.959 |
| Rubin y | **27.51** | 0.849 | 0.969 | 0.935 |
| Euclid VIS | — | 0.598 | 0.870 | 0.919 |
| Euclid Y | — | 0.197 | 0.887 | 1.024 |
| Euclid J | — | 0.198 | 0.932 | 0.991 |
| Euclid H | — | 0.203 | 0.942 | 1.031 |
| **Mean** | — | **0.715** | **0.955** | **0.989** |

Notes: Rubin g/r/i/z reconstructions are near-perfect. Rubin u and Euclid NISP bands are solid. **Rubin y training loss is anomalously high** (27.5 vs 1.3–5.3 for other Rubin bands) despite reasonable reconstruction metrics — needs investigation. Euclid VIS is the weakest band (r=0.87, std_ratio=0.92), possibly under-reconstructed at native resolution.

---

## Downstream Heads

All downstream heads reuse the frozen V7 foundation encoder. Only lightweight task-specific layers are trained on top. This means each head gets the benefit of the full 10-band multi-instrument representation without the cost of encoding from scratch, and training each head is fast (hours, not days).

### 1. Detection

**Directory**: `models/detection/`

The detection stack now supports three complementary source-finding choices:

1. **Classical VIS baseline**: native-resolution Euclid VIS peak-finding with bright-star masking. This is fast, robust, and remains the bootstrap source list for pseudo-label generation.
2. **V7 + CenterNet**: a dense detector on top of the frozen V7 **fused bottleneck**. This is the strongest current option for broad 10-band semantic fusion, especially when the signal is spread across multiple bands rather than carried by one sharp VIS peak.
3. **V7 + StemCenterNet**: a dense detector on top of the frozen V7 **BandStems** at native resolution. This preserves more local spatial detail and is the highest-resolution neural option currently in the repo.

The point of keeping all three is scientific comparison, not redundancy. Classical VIS is the baseline and pseudo-label source. The fused-bottleneck detector tests whether the self-supervised latent has learned genuinely multi-band source evidence. The stem detector tests whether native-resolution V7 features improve local source finding beyond what the coarser bottleneck can express.

![Detection overview](models/detection/detect.png)

#### Detection Approach: Why CenterNet, Not DETR

The detection head went through two major neural iterations. Understanding why the first was abandoned helps explain the current design.

**DETR (Detection Transformer) -- tried first, abandoned.** DETR is a set-prediction architecture from natural image detection. It uses a transformer decoder with learned "object queries" -- fixed-size slots that each learn to claim one object through cross-attention to spatial features. Training requires Hungarian matching to find the optimal assignment between predicted slots and ground-truth objects, and a composite loss that teaches matched slots to predict positions while pushing unmatched slots toward zero confidence.

DETR was a poor fit for astronomical source detection for several reasons:

- **Designed for the wrong problem.** DETR's innovations (set prediction, no NMS, no anchor boxes) solve problems that don't exist in astronomical imaging. Sources at the 0.8"/px bottleneck resolution are effectively point-like -- there are no overlapping bounding boxes to deduplicate. The set-prediction framework adds complexity without corresponding benefit.
- **Data-hungry.** The original DETR paper trained for 500 epochs on 118,000 images. Our historical ECDFS detection experiments used only ~130 training tiles (from a 144-tile subset). With so little data, the model struggled to converge -- the 500 object queries exhibited "query collapse" where all predictions clustered at a single location for many epochs before slowly spreading out.
- **Expensive.** 500 queries cross-attending to ~17,000 memory tokens through 6 transformer decoder layers is computationally heavy for what is fundamentally "find bright spots in a feature map."
- **Slow convergence.** After fixing a critical bug where pseudo-labels were computed on unaugmented images while the model saw augmented images, DETR still took 25+ epochs to reach val loss 1.02, and the confidence head struggled to differentiate real sources from empty queries.

The DETR code is preserved in `detector.py`, `matcher.py`, and `train_detection.py` for reference.

#### Current Detection Choices

**Classical VIS** is the control baseline. It runs source detection directly on the Euclid VIS image at native 0.1"/px resolution and remains the pseudo-label source for both neural training loops. It is useful because it is simple, interpretable, and usually conservative around obvious sources, but it is limited to what is visible in VIS.

**V7 + CenterNet (fused bottleneck)** treats detection as a per-pixel prediction problem on the frozen V7 bottleneck. The foundation model has already fused Rubin, VIS, and NISP streams into a shared multi-band latent, so the detector head operates on a representation that has deep cross-band mixing built in. This is the learned version of classical peak-finding -- but operating on rich 10-band features instead of a simple coadd.

This approach is a natural fit for astronomical source detection because:

- **Every pixel gets direct supervision.** No Hungarian matching, no set prediction instability. Each pixel's heatmap target is simply a Gaussian centered at the nearest ground-truth source. The focal loss (from CornerNet/CenterNet) handles the extreme class imbalance between source pixels and empty sky.
- **Fast convergence.** With direct per-pixel supervision and only 3.5M trainable parameters, CenterNet converges much faster than DETR on the historical 144-tile subset. Val loss drops steadily from epoch 1 without the query-collapse plateau that plagued DETR.
- **Naturally extensible.** Additional per-pixel heads can be added cheaply by appending more output channels. The current architecture already supports an optional profile head for source shape parameters (ellipticity, half-light radius, Sersic index) that can be activated when training labels become available -- this is important for future integration with tools like Tractor that need shape priors for deblending and forced photometry.

**V7 + StemCenterNet (native stems)** reuses the pretrained V7 BandStems directly at native band resolution. Rubin, VIS, and NISP streams are projected into a common VIS-frame feature grid, fused with a lightweight residual encoder-decoder, and then converted to the same heatmap/offset-style outputs as the bottleneck detector. This path preserves more local spatial detail than the fused 0.8"/px bottleneck, which makes it appealing for high-resolution source finding and deblending, but it also makes the model more sensitive to sharp instrumental structure such as diffraction spikes and bright-star halos. The stem self-training loop therefore now uses a lighter bright-star veto during round-2 label promotion to reduce artifact pickup without masking too aggressively.

In practice, the two neural detectors test different hypotheses:

- **CenterNet on the fused bottleneck** asks whether the foundation model learned strong multi-band source evidence.
- **StemCenterNet on native-resolution stems** asks whether V7 pretraining improves local detection when the head keeps more of the original spatial detail.

#### How V7 + CenterNet Works

The frozen V7 encoder processes the multi-band input tile and produces a bottleneck feature map at approximately 130x130 spatial resolution. A decoder neck with three progressive 2x bilinear upsampling stages (each followed by Conv-BN-ReLU) upsamples the bottleneck 8x to ~1040x1040, matching Euclid VIS native resolution (0.1"/px). Channel widths narrow as spatial resolution increases (256 -> 128 -> 64 -> 64) to keep memory manageable. Four parallel prediction heads then produce dense per-pixel outputs at VIS resolution:

```
Frozen V7 encoder
  -> bottleneck [B, 256, ~130, ~130]        (0.8"/px)
  -> Flat conv: 256 -> 128 channels          (130x130)
  -> 3x bilinear 2x upsample + Conv-BN-ReLU:
       128 -> 64 channels                    (130 -> 260)
        64 -> 64 channels                    (260 -> 520)
        64 -> 64 channels                    (520 -> 1040)
  -> Dense prediction heads at VIS resolution (~0.1"/px):
       Heatmap  [B, 1, ~1040, ~1040]  -- source probability (sigmoid)
       Offset   [B, 2, ~1040, ~1040]  -- sub-pixel (dx, dy) refinement
       Log flux [B, 1, ~1040, ~1040]  -- brightness proxy
       Profile  [B, 4, ~1040, ~1040]  -- (e1, e2, r_half, sersic_n) [optional, future]
```

At inference, source detection is simple: find local maxima in the heatmap (via max-pooling NMS with kernel 7), threshold on confidence, and read off the offset, flux, and profile values at each peak location. Because the heatmap is at VIS resolution, each pixel corresponds to 0.1" on the sky -- the offset head only needs to cover tiny sub-pixel corrections (±0.05"), giving VIS-native centroid precision without relying on large sub-pixel offsets from a coarse bottleneck grid.

#### How V7 + StemCenterNet Works

StemCenterNet keeps the same CenterNet-style output heads but swaps the backbone:

```
Frozen V7 BandStems at native resolution
  -> Rubin bands (6 stems) -> learned weighted Rubin stream
  -> Euclid bands (4 stems: VIS/Y/J/H) -> learned weighted Euclid stream
  -> reproject Rubin to Euclid frame, concatenate streams
  -> shallow residual encoder-decoder
  -> Dense prediction heads at VIS resolution:
       Heatmap  [B, 1, ~1084, ~1084]
       Offset   [B, 2, ~1084, ~1084]
       Log flux [B, 1, ~1084, ~1084]
       Profile  [B, 4, ~1084, ~1084]  [optional]
```

Compared with the bottleneck detector, this path gives the head much more local spatial information but less deep cross-band fusion. That tradeoff is scientifically useful: if it wins, native-resolution pretraining is paying off directly for detection; if it loses on NIR-only or dropout-style sources, that tells us the fused latent is doing something genuinely important for multi-band reasoning.

#### Training: Self-Training Pipeline

Since there is no curated source catalog for this field, both neural detectors use a **self-training loop** that bootstraps from noisy classical pseudo-labels and progressively cleans them.

**Pseudo-labels**: When Euclid VIS is available, sources are detected in the VIS image at native 0.1"/px resolution using classical peak-finding (3-sigma threshold, Gaussian smoothing, subpixel centroiding). This preserves VIS's spatial precision. A **bright-star spike mask** (dilating saturated VIS cores by 40 VIS pixels, about 4 arcsec) suppresses obvious diffraction-spike detections during pseudo-label creation. When VIS is unavailable, Rubin g+r+i coadd pseudo-labels are used as a fallback.

**Precomputed features**: The fused-bottleneck CenterNet path can cache encoder outputs to disk. With 8 augmentation variants each (4 rotations × 2 flips), bottleneck tensors `[256, ~130, ~130]` are saved and training then runs only the lightweight decoder neck + heads -- no encoder forward pass needed per step. This is the fastest neural detection path in the repo.

**Live stem training**: StemCenterNet does not use cached bottleneck features. It runs directly from the frozen V7 BandStems at native resolution, which is more expensive but preserves more local structure.

**Self-training rounds**:
1. **Round 1**: Train on VIS pseudo-labels. The model learns what sources look like in 10-band feature space.
2. **Label refinement**: Run the trained detector on all tiles. High-confidence (>0.8) novel detections that don't match any VIS pseudo-label are **promoted** as new labels (sources visible in other bands but not VIS). Existing pseudo-labels where the model has low confidence (<0.3) are **demoted** (artifacts like diffraction spikes that appear only in VIS — the other 9 bands show nothing, so the model assigns low confidence). This is self-consistent: the model's multi-band understanding cleans its own training data.
3. **Round 2**: Retrain on VIS labels + promoted labels - demoted labels.

For the stem path, round-2 promotion now includes a **lighter bright-star veto mask** (`promotion_spike_radius=20` by default) so the model can still promote real novel sources near bright objects without freely promoting long spike chains as new training labels.

**Loss**: Each ground-truth source is rendered as a 2D Gaussian (sigma=2 pixels in VIS-resolution heatmap coordinates, ≈ 0.2") on the heatmap target. Gaussian rendering uses bounded per-source computation (only within 3σ radius) to avoid OOM at VIS resolution. The loss combines:

| Loss Term | Type | Where Applied | Weight | Purpose |
|-----------|------|---------------|--------|---------|
| `loss_hm` | Focal loss | All pixels | 1.0 | Teaches source vs background; focal weighting handles ~99% empty-sky imbalance |
| `loss_off` | L1 | Only at GT source positions | 1.0 | Sub-pixel offset refinement |
| `loss_flux` | L1 | Only at GT source positions | 0.1 | Flux estimation (lower weight: pseudo-labels are noisy) |

Historical detection development used 130 training tiles and 14 validation tiles from the 144-tile ECDFS subset, with random 90-degree rotations and flips (8 variants cached per tile). The same pipeline now needs to be rerun on the expanded flat dataset once the new foundation checkpoints are ready.

#### Files

| File | Description |
|------|-------------|
| `centernet_detector.py` | `CenterNetDetector` model: 8× decoder neck + heatmap/offset/flux/profile heads |
| `stem_centernet_detector.py` | `StemCenterNetDetector`: native-resolution V7 BandStem fusion + dense heads |
| `centernet_loss.py` | Focal loss, bounded Gaussian heatmap rendering (memory-safe at VIS scale), masked offset/flux L1 |
| `train_centernet.py` | CenterNet training loop (supports live encoder or cached features mode) |
| `train_stem_centernet.py` | StemCenterNet training loop (live native-resolution stem features) |
| `precompute_features.py` | One-time V7 encoder feature caching (all tiles × 8 augmentation variants) |
| `cached_dataset.py` | Dataset loading precomputed features + pseudo-labels with label refinement support |
| `self_train.py` | Self-training loop: VIS labels → train → promote/demote → retrain |
| `self_train_stem.py` | Self-training loop for StemCenterNet with lighter artifact-aware promotion |
| `dataset.py` | Pseudo-label generation (VIS with saturation mask, or Rubin fallback), tile dataset |
| `detect.png` | Example qualitative comparison figure for the three detection choices |
| `detector.py` | `JaispDetector` (DETR, archived -- kept for reference, see note below) |
| `matcher.py` | Hungarian matcher + DETR loss (archived) |
| `train_detection.py` | DETR training loop (archived) |

**Note on DETR (archived)**: DETR was the first detection approach tried, using a transformer decoder with 500 learned object queries and Hungarian matching. It was abandoned because: (1) DETR needs large datasets (trained for 500 epochs on 118k images; we have 130 tiles); (2) query collapse persisted for many epochs; (3) overkill for point-like sources at bottleneck resolution; (4) convergence was ~5× slower than CenterNet. The code is preserved for reference.

### 2. Astrometry Concordance

**Directory**: `models/astrometry2/`

Astrometry concordance measures and corrects the smooth spatial distortion between Rubin and Euclid coordinate systems. Even after standard WCS calibration, there are residual sub-arcsecond offsets that vary smoothly across a tile. These must be corrected for joint analysis -- stacking images, measuring galaxy shapes for weak lensing, or combining photometry across instruments.

The approach works at the individual source level: detect sources in one or both instruments, extract small image patches around each source, predict the precise pixel offset between the Rubin and VIS positions of each source, and then fit a smooth spatial field from all these per-source measurements. With multiband mode, each of the 10 bands gets its own per-source offset prediction (capturing band-specific chromatic effects like differential chromatic refraction), conditioned by a learned band embedding.

#### Pipeline

For each tile, the pipeline proceeds through six stages:

1. **Source detection**: Find sources using either classical peak-finding (Rubin g+r+i+z coadd and VIS independently, then WCS cross-match) or the trained CenterNet detector (10-band neural detection in the VIS frame, no cross-matching needed). The neural detector option is valuable because it sees all 10 bands through the V7 encoder and can find fainter sources than classical 3-band detection, giving more anchor points for the field fit.

2. **Cross-instrument matching** (classical path only): Match Rubin and VIS source lists using WCS sky coordinates. Mutual nearest-neighbor matching with separation limits, sigma-clipping, and object-level deduplication removes spurious matches. Typically yields 50-200 matched pairs per tile. The CenterNet path skips this step since detections are already in the VIS frame.

3. **Patch extraction**: For each matched source, extract a 33x33 pixel patch centered on the VIS position. All 10 bands (6 Rubin + VIS + 3 NISP) are reprojected onto the VIS pixel grid via WCS. When per-pixel variance maps are available in the NPZ files (which they are for both Rubin `var` and Euclid `var_VIS/Y/J/H`), per-pixel RMS patches are also extracted and reprojected alongside the image patches.

4. **Per-patch offset prediction**: The core of the matcher. Frozen V7 features extract representations from the Rubin/NISP and VIS patches independently. Two encoder modes are available:
   - **Stem-only** (`--stream-stages 0`, default): Uses only the V7 BandStem CNNs, producing 33x33 feature maps at full patch resolution. This is equivalent to the V6 matcher.
   - **Stem + stream stages** (`--stream-stages 1`): Applies one frozen V7 stream encoder ConvNeXt stage after the stems, producing 16x16 feature maps with richer cross-channel mixing. This leverages V7's deeper pretrained representations while maintaining enough spatial resolution for the cost volume.

   When per-pixel RMS patches are available, the full BandStem normalization (`image / rms`) is used -- matching the normalization the stems saw during foundation pretraining. When RMS is absent, the legacy scalar MAD normalization is used as a fallback.

   Trainable ConvNeXt adapter blocks then refine these features toward astrometry-relevant signals. A spatially-weighted cross-correlation (cost volume) between the two feature maps produces a coarse offset via differentiable soft-argmax. An MLP refines this with a residual correction and outputs `(dx, dy, log_sigma)` in pixels, which a Jacobian matrix transforms to `(dRA*, dDec)` in arcseconds. In multiband mode, a learned band embedding conditions the MLP so each band gets a wavelength-specific correction. The uncertainty `sigma` is used to weight the field fit.

5. **Smooth field fitting**: The per-source offsets are noisy -- each individual measurement has ~30-50 mas scatter. But the true distortion varies smoothly across the tile. A control-grid least-squares solver (bilinear basis functions with smoothness regularization and adaptive per-node anchor weights) fits a smooth 2D field from the noisy per-source measurements. An alternative MLP-based solver is also available.

6. **Export**: The fitted concordance field is evaluated on a regular mesh and written to a FITS file as `dRA`, `dDec`, and coverage maps.

```
For each tile:
  1. Detect sources (classical cross-match OR CenterNet 10-band)
  2. Extract 33x33 patches + optional per-pixel RMS patches
  3. Per-patch prediction:
       Frozen V7 BandStems (+ optional stream stages)
       -> per-pixel RMS normalization (or scalar MAD fallback)
       -> ConvNeXt adapters
       -> spatially-weighted cost volume (cross-correlation)
       -> soft-argmax (differentiable coarse offset)
       -> MLP refinement + band embedding -> (dx, dy, log_sigma) in pixels
       -> Jacobian transform -> (dRA*, dDec) in arcsec
  4. Fit smooth field from per-source offsets:
       Control-grid least squares (or MLP solver)
       with adaptive anchor regularization
  5. Export concordance FITS (dRA, dDec, coverage maps)
```

#### Matcher Versions

Two matcher versions are available, differing only in which foundation model's BandStems they reuse:

- **V6** (`matcher_v6.py`): Uses frozen V6 Phase B BandStems. These were trained with cross-instrument masking where VIS was downsampled to Rubin resolution. Best result: 31.9 mas median astrometric precision.
- **V7** (`matcher_v7.py`): Uses frozen V7 BandStems, where VIS was processed at native resolution. API-compatible drop-in replacement for V6. Additionally supports `--stream-stages N` to use frozen V7 stream encoder ConvNeXt stages after the stems for richer features.

Both versions support multiband mode (per-band learned embeddings for wavelength-specific chromatic correction) and separate learning rates (full LR for adapters and heads, 0.1x for stems and stream stages if unfrozen for fine-tuning).

#### Data Augmentation

Training patches are augmented with random 90/180/270-degree rotations and horizontal/vertical flips, giving up to 16 distinct orientations per source patch. The pixel-to-sky Jacobian matrix is correctly transformed for each augmentation so the sky-space loss remains valid. This is important for the data-limited regime -- with ~200 training tiles, augmentation significantly increases effective sample diversity.

#### Per-Pixel RMS Normalization

When variance maps are available in the tile NPZ files (Rubin `var`, Euclid `var_VIS/Y/J/H`), the dataset extracts and reprojects per-pixel RMS patches alongside each image patch. During training, the V7 matcher uses the full BandStem normalization (`image / rms`, matching what the stems saw during foundation pretraining) instead of the legacy scalar MAD normalization. This preserves spatial noise structure -- tile edges with higher variance, chip gaps, proximity to bright objects -- which becomes increasingly important as the dataset covers more varied sky regions with heterogeneous noise properties. When RMS data is absent, the pipeline falls back to scalar MAD normalization automatically.

#### Source Detection Options

The first stage of the pipeline -- detecting sources -- can use either:

- **Classical** (default): `detect_sources()` performs median background subtraction, Gaussian smoothing, and local maximum detection above an N-sigma threshold in both Rubin (g+r+i+z coadd) and VIS independently, then cross-matches via WCS. Simple, fast, and well-understood.
- **Neural** (optional): Pass `--detector-checkpoint` to replace classical detection with the trained CenterNet detector. The neural detector sees all 10 bands through the V7 encoder and produces source positions directly in the VIS frame -- no cross-matching step needed. It can find fainter sources that are invisible in a 3-band coadd, providing more anchor points for the concordance fit (typically 200+ anchors per tile vs 50-150 with classical).

#### Field Solvers

Two solvers are available for fitting the smooth concordance field from per-source offset measurements:

- **Control grid** (`field_solver.py`): Fits bilinear basis functions on a regular grid using weighted least squares. Includes finite-difference smoothness regularization and adaptive per-node anchor weights that prevent edge drift in regions with sparse source coverage. The grid resolution is automatically reduced for tiles with few matches to avoid underdetermined systems.
- **Neural network** (`nn_field_solver.py`): A 4-layer MLP that maps normalized (x, y) tile coordinates to (dRA, dDec) offsets, trained via Adam for 2000 steps. Has no grid resolution hyperparameter, is infinitely differentiable, and scales naturally to any source density.

#### Files

| File | Description |
|------|-------------|
| `matcher_v6.py` | V6-based patch matcher (frozen V6 stems + trainable adapters) |
| `matcher_v7.py` | V7-based patch matcher (frozen V7 stems + optional stream stages + trainable adapters, per-pixel RMS support) |
| `dataset.py` | Patch + RMS extraction, WCS matching, CenterNet detector integration, rotation/flip augmentation |
| `source_matching.py` | Classical peak-finding, WCS-based source matching |
| `field_solver.py` | Control-grid least-squares field solver |
| `nn_field_solver.py` | MLP-based field solver |
| `train_astro_v6.py` | Training script (V6 backbone) |
| `train_astro_v7.py` | Training script (V7 backbone, CenterNet or classical sources, saves full checkpoint metadata) |
| `infer_concordance.py` | Per-tile inference -> FITS export |
| `infer_global_concordance.py` | Global multi-tile concordance fitting |
| `apply_concordance.py` | Apply fitted concordance fields to data |
| `sky_cube.py` | Aligned 10-band sky cube extraction with concordance correction |
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
|   +-- rubin_tiles_all/               Current flat Rubin training tiles (*.npz)
|   +-- euclid_tiles_all/              Current flat Euclid training tiles (*.npz)
|   +-- rubin_tiles_ecdfs/             Legacy ECDFS Rubin tiles (*.npz)
|   +-- euclid_tiles_ecdfs/            Legacy ECDFS Euclid tiles (*.npz)
|   +-- tiles_product.tar.gz           Source archive for the expanded flat tile set
|
+-- checkpoints/
|   +-- jaisp_v7_baseline/             Historical 144-tile V7 checkpoint
|   +-- jaisp_v7_tiles_all_ddp*/       Current / future 790-tile V7 training outputs
|
+-- models/
|   +-- jaisp_foundation_v7.py         V7 mixed-resolution MAE (current)
|   +-- jaisp_foundation_v6.py         V6 single-grid MAE (library, used by V7)
|   +-- jaisp_dataset_v7.py            V7 mixed-resolution split helpers
|   +-- jaisp_dataset_v6.py            V6 data loader (library, used by downstream)
|   +-- train_jaisp_foundation_v7.py   V7 training entrypoint
|   +-- eval_foundation_v7.py          V7 evaluation/diagnostics
|   |
|   +-- detection/                     Source detection head
|   |   +-- centernet_detector.py      CenterNet model: 8x decoder + heads
|   |   +-- stem_centernet_detector.py Native-resolution stem-based CenterNet
|   |   +-- centernet_loss.py          Focal loss + bounded heatmap targets
|   |   +-- train_centernet.py         CenterNet training (live or cached features)
|   |   +-- train_stem_centernet.py    StemCenterNet training
|   |   +-- precompute_features.py     One-time V7 encoder feature caching
|   |   +-- cached_dataset.py          Dataset for cached features + labels
|   |   +-- self_train.py              Self-training: train -> refine -> retrain
|   |   +-- self_train_stem.py         Stem self-training: train -> refine -> retrain
|   |   +-- dataset.py                 Pseudo-labels (VIS + saturation mask)
|   |   +-- detect.png                 Example detection comparison figure
|   |   +-- detector.py               DETR model (archived)
|   |   +-- matcher.py                Hungarian matcher (archived)
|   |   +-- train_detection.py        DETR training script (archived)
|   |
|   +-- astrometry2/                   Rubin<->Euclid concordance
|   |   +-- matcher_v7.py              V7 patch matcher (stem + optional stream stages)
|   |   +-- matcher_v6.py              V6 patch matcher
|   |   +-- dataset.py                 Patch dataset + per-pixel RMS + detector integration
|   |   +-- source_matching.py         Classical detection utilities
|   |   +-- field_solver.py            Control-grid least-squares field solver
|   |   +-- nn_field_solver.py         MLP field solver
|   |   +-- train_astro_v6.py          V6 training script
|   |   +-- train_astro_v7.py          V7 training script (CenterNet or classical sources)
|   |   +-- infer_concordance.py       Per-tile inference -> FITS export
|   |   +-- infer_global_concordance.py  Global multi-tile concordance fitting
|   |   +-- apply_concordance.py       Apply fitted concordance fields to data
|   |   +-- sky_cube.py                Aligned 10-band sky cube extraction
|   |   +-- viz.py                     Diagnostic visualizations
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
# V7 mixed-resolution MAE with 2-stream concat architecture (recommended)
# Multi-GPU with bfloat16 AMP (adjust --nproc_per_node to number of GPUs)
cd models && torchrun --nproc_per_node=2 train_jaisp_foundation_v7.py \
    --rubin_dir  ../data/rubin_tiles_all \
    --euclid_dir ../data/euclid_tiles_all \
    --output_dir ./checkpoints/jaisp_v7_concat \
    --hidden_ch 256 --transformer_depth 4 --transformer_heads 8 \
    --fused_pixel_scale_arcsec 0.8 --cross_instrument_prob 1.0 \
    --epochs 100 --lr 3e-4 --accum_steps 2 \
    --persistent_workers --num_workers 4 \
    --wandb_name v7_rms_aware_loss

# Single-GPU fallback (plain python, no torchrun needed)
cd models && python train_jaisp_foundation_v7.py \
    --rubin_dir  ../data/rubin_tiles_all \
    --euclid_dir ../data/euclid_tiles_all \
    --output_dir ./checkpoints/jaisp_v7_concat \
    --hidden_ch 256 --transformer_depth 4 --transformer_heads 8 \
    --fused_pixel_scale_arcsec 0.8 --cross_instrument_prob 1.0 \
    --epochs 100 --lr 3e-4 --accum_steps 4 \
    --persistent_workers --num_workers 4 \
    --wandb_name v7_concat_euclid_fused
```

### 2. Detection Head

```bash
# Classical VIS baseline
# No training step required; the classical detector is built into
# models/detection/dataset.py and astrometry2/source_matching.py.

# Option A: fused-bottleneck CenterNet
# 200-tile subset is sufficient for detection; full 790 tiles add training
# time without significant accuracy gain.
# Step 1: Precompute encoder features (one-time)
python models/detection/precompute_features.py \
    --rubin_dir    data/rubin_tiles_200 \
    --euclid_dir   data/euclid_tiles_200 \
    --encoder_ckpt models/checkpoints/jaisp_v7_concat/checkpoint_best.pt \
    --out_dir      data/cached_features_v7_rms_aware \
    --n_augments   8

# Step 2: Self-training (runs round 1 + label refinement + round 2)
python models/detection/self_train.py \
    --feature_dir  data/cached_features_v7_rms_aware \
    --rubin_dir    data/rubin_tiles_200 \
    --euclid_dir   data/euclid_tiles_200 \
    --out_dir      checkpoints/centernet_v7_rms_aware \
    --rounds 2 --epochs 100 --batch_size 4 \
    --wandb_project jaisp-detection

# Option B: native-resolution StemCenterNet
python models/detection/self_train_stem.py \
    --encoder_ckpt models/checkpoints/jaisp_v7_concat/checkpoint_best.pt \
    --rubin_dir    data/rubin_tiles_200 \
    --euclid_dir   data/euclid_tiles_200 \
    --out_dir      checkpoints/stem_centernet_v7_rms_aware \
    --rounds 2 --epochs 60 --batch_size 1 \
    --stream_ch 16 --base_ch 32 \
    --promotion_spike_radius 20 \
    --wandb_project jaisp-detection
```

### 3. Astrometry Matcher

The matcher is a patch-level model -- it only needs to learn "given a 33x33 patch pair, predict the offset." Training on ~200 tiles is sufficient; inference and field solving then run on all 790. Use `--val-frac 0.75` to train on ~200 tiles while holding out the rest for validation.

```bash
# V7 backbone with CenterNet source detection (recommended)
python models/astrometry2/train_astro_v7.py \
    --v7-checkpoint       checkpoints/<v7-foundation>/checkpoint_best.pt \
    --detector-checkpoint checkpoints/<detector>/centernet_best.pt \
    --rubin-dir      data/rubin_tiles_all \
    --euclid-dir     data/euclid_tiles_all \
    --multiband \
    --stream-stages 1 \
    --val-frac 0.75 \
    --epochs 60 \
    --output-dir     checkpoints/astro_v7 \
    --wandb-project  JAISP-Astrometry-v7

# V7 backbone with classical source detection (no detector needed)
python models/astrometry2/train_astro_v7.py \
    --v7-checkpoint  checkpoints/<v7-foundation>/checkpoint_best.pt \
    --rubin-dir      data/rubin_tiles_all \
    --euclid-dir     data/euclid_tiles_all \
    --multiband \
    --stream-stages 1 \
    --val-frac 0.75 \
    --epochs 60 \
    --output-dir     checkpoints/astro_v7_classical
```

### 4. Concordance Inference

```bash
# Generate concordance FITS with V7 matcher + CenterNet sources
python models/astrometry2/infer_concordance.py \
    --checkpoint         checkpoints/astro_v7/checkpoint_best.pt \
    --v7-checkpoint      checkpoints/<v7-foundation>/checkpoint_best.pt \
    --detector-checkpoint checkpoints/<detector>/centernet_best.pt \
    --rubin-dir     data/rubin_tiles_all \
    --euclid-dir    data/euclid_tiles_all \
    --output        concordance_v7.fits \
    --all-bands
```

### 5. PSF Training

```bash
python models/photometry/train_psf_net.py \
    --rubin_dir  data/rubin_tiles_all \
    --euclid_dir data/euclid_tiles_all \
    --out checkpoints/psf_net_v1.pt \
    --epochs 20
```

---

## Checkpoints

| Checkpoint / Artifact | Location | Description |
|-----------------------|----------|-------------|
| Historical V7 foundation baseline | `checkpoints/jaisp_v7_baseline/checkpoint_best.pt` | Legacy 144-tile V7 baseline (epoch 86, val_loss 1.4689) |
| Previous flat-set V7 checkpoint | `checkpoints/jaisp_v7_tiles_all_ddp_online/checkpoint_best.pt` | Previous downstream backbone (before RMS-aware loss) |
| **Current V7 foundation (RMS-aware)** | `models/checkpoints/jaisp_v7_concat/checkpoint_best.pt` | **Latest**: v7_rms_aware_loss run (epoch 92, val_loss 4.0493). Use this for all new downstream training. |
| V7 detector (CenterNet bottleneck) | `checkpoints/centernet_v7_patch25_box16_round2/centernet_best.pt` | Trained on previous (non-RMS-aware) foundation; needs retraining on current checkpoint |
| V7 detector (StemCenterNet, artifact-aware) | `checkpoints/stem_centernet_v7_patch25_box16_round2_mask20/stem_centernet_best.pt` | Trained on previous foundation; needs retraining on current checkpoint |
| Cached features (previous) | `data/cached_features_v7_patch25_box16/` | Cached features from previous foundation checkpoint (stale) |

---

## Current Status

- **V7 foundation** current best checkpoint is `models/checkpoints/jaisp_v7_concat/checkpoint_best.pt` from the `v7_rms_aware_loss` run (epoch 92, val_loss 4.0493). This uses RMS-aware loss weighting and should be used for all new downstream training. The older `jaisp_v7_tiles_all_ddp_online` and `jaisp_v7_baseline` checkpoints are kept as historical references.
- **Detection** now has three supported choices in the repo: classical VIS, fused-bottleneck `CenterNet`, and native-resolution `StemCenterNet`. The DETR code remains archived for historical reference only. The two neural detectors are intentionally both kept because they test different hypotheses about whether deep multi-band fusion or native-resolution stem reuse is more valuable for science detection tasks.
- **Astrometry V7** matcher is actively being trained and evaluated. Recent improvements:
  - **Stream stages**: `--stream-stages 1` uses frozen V7 stream encoder ConvNeXt stages after the stems, giving richer features that leverage V7's deeper pretrained representations (not just the bare stems which are equivalent to V6).
  - **Per-pixel RMS**: When variance maps are available in the tile NPZ files, the full BandStem normalization (`image / rms`) is used instead of scalar MAD normalization, matching the foundation pretraining distribution.
  - **Rotation augmentation**: Random 90/180/270-degree rotations alongside flips give 16 augmentation variants (up from 4).
  - **Checkpoint metadata**: V7 training checkpoints now save full metadata (input bands, target bands, args) so inference scripts can correctly reconstruct the model configuration.
  - **Recommended workflow**: Train on ~200 tiles (`--val-frac 0.75`), then run inference on all 790 tiles and fit the global concordance field. The matcher learns a patch-level skill that generalizes across tiles; the field solver is where full-footprint coverage matters.
  - Early results on a 16-tile subset with CenterNet anchors: raw WCS 60 mas → NN 25 mas per-source, 15 mas field residual at epoch 27.
- **Photometry** is functional but still mostly documented against the older data paths and has not yet been rerun on the expanded flat dataset.
- This is an active research codebase. Architecture and training defaults evolve with experiments.
