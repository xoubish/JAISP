# Astrometry2: Per-Object Alignment + Concordance QA

## What this is for

Rubin and Euclid observe the same sky, but source centers do not land at exactly the same place in every band/instrument. In the current ECDFS data, the dominant offset is **not** a large smooth WCS distortion. It is source-by-source centering / centroid-definition scatter: a galaxy or faint source has a slightly different measured center in Rubin, VIS, NISP, and even different Rubin bands.

This directory therefore supports two related products:

1. **Latent position head**: the current per-object correction path. It predicts a source-specific offset from frozen foundation features and aligns all non-VIS bands to the VIS reference frame.
2. **Concordance fields**: smooth residual WCS fields fitted with PINN/NN/control-grid solvers. In ECDFS these fields are small and are mainly used for QA, diagnostics, and fallback smooth correction.

Key design principles:
- The final high-precision correction comes from the **object-level head**, not from stacking a large concordance field on top of WCS.
- The correction is stored in **sky coordinates** (arcsec), not pixel offsets.
- WCS handles geometry; the latent head handles object-level centering; the concordance field measures any remaining smooth residual.
- Perfect post-head alignment means the residual field and the median residual vector are consistent with zero.
- Concordance meshes are sampled coarsely and bilinearly interpolated when a smooth field correction is explicitly requested.

---

## Why a neural network?

Classical source matching gives you discrete (band source position, VIS source position, offset) triples. Notebook 09 shows that these offsets are dominated by the source center itself: changing centroid definitions moves the measured offset by ~54 mas, while the smooth per-tile field is only ~5 mas.

A learned head is useful because it sees the pixels and the multi-band context. It can learn that a blended galaxy, a faint source, a colour gradient, or a band-specific PSF should shift the usable source position differently from a simple catalog centroid. A smooth concordance file cannot express this source-by-source information.

The smooth field solvers still matter as diagnostics. They answer: after the head correction, is there any coherent residual WCS field left? In the latest run the answer is basically no: the head-residual PINN field has ~1 mas amplitude and barely changes median residuals.

---

## Current results

### Latest v8 latent head on 790 ECDFS tiles (current)

The current checkpoint is `models/checkpoints/latent_position_v8_no_psf/best.pt`, evaluated with `eval_latent_position.py` on all 790 ECDFS tiles. Sources require SNR >= 5 and raw offset < 200 mas.

| Band | N sources | Raw median | Head median | Head p68 |
|------|-----------|-----------:|------------:|---------:|
| rubin_u | 12,347 | 119.4 mas | 30.5 mas | 53.3 mas |
| rubin_g | 60,148 | 54.0 mas | 11.4 mas | 22.3 mas |
| rubin_r | 70,022 | 45.7 mas | 10.4 mas | 19.8 mas |
| rubin_i | 62,232 | 41.2 mas | 10.3 mas | 18.9 mas |
| rubin_z | 42,980 | 41.9 mas | 10.9 mas | 19.9 mas |
| rubin_y | 17,126 | 61.5 mas | 14.7 mas | 29.4 mas |
| nisp_Y | 116,572 | 41.3 mas | 9.4 mas | 17.0 mas |
| nisp_J | 126,352 | 41.9 mas | 9.4 mas | 17.0 mas |
| nisp_H | 122,341 | 42.2 mas | 9.5 mas | 17.0 mas |

The head improves median radial residuals by ~74-79% in most bands. Applying a residual PINN field after the head changes medians by only ~0.0-0.2 mas because the post-head smooth field amplitude is already ~1 mas.

### Centering diagnosis

Notebook 09 (`io/09_astrometry_diagnostics.ipynb`) directly tests the large-offset cause:

| Quantity | Median |
|---|---:|
| Smooth per-tile bulk field | 5.4 mas |
| Residual after bulk removal | 47.5 mas |
| Rubin detection -> PSF-fit recentering | 52.7 mas |
| VIS detection -> PSF-fit recentering | 7.0 mas |
| Offset change induced by recentering | 54.0 mas |
| Rubin r-vs-i band-centering offset | 47.8 mas |

Conclusion: the 40-50 mas raw residual is dominated by centering / centroid-definition scatter, not by an unmodelled smooth concordance field.

### Historical V7 matcher result

The older V7 per-patch matcher and smooth field solvers established the same lesson: raw offsets are tens of mas, but the smooth field is only a few mas. The matcher/field stack remains useful for experiments and FITS concordance exports, but it is not the current headline per-object correction.

| Regime | Typical median | Notes |
|--------|---------------:|-------|
| Raw WCS anchors | 41-119 mas | Band-dependent source-centering scatter |
| Raw-anchor PINN field amplitude | ~5-9 mas | Smooth WCS/concordance component |
| Head-residual PINN field amplitude | ~1 mas | Coherent residual after latent head |

### Historical V6 results (144-tile ECDFS subset, archived)

| Model | val MAE | p68 | frac < 0.1" | Notes |
|-------|---------|-----|-------------|-------|
| Raw WCS | ~47 mas | — | — | No correction |
| astrometry2 baseline CNN | ~38 mas | — | — | Single-band, random-init VIS encoder |
| v6 Phase A stems | ~60 mas | — | — | Frozen Rubin stems, random VIS CNN |
| **v6 Phase B all-bands** | **31.9 mas** | **34.6 mas** | **96.1%** | Frozen Rubin + VIS stems from joint Phase B |

---

## Pipeline overview

### Current per-object alignment flow

```
Tiles on disk
    │
    ▼
[train_latent_position.py / eval_latent_position.py]
    classical source detection + Gaussian centroid targets
    WCS match each non-VIS band to VIS
    frozen foundation encoder (v7 or v8)
    local bottleneck window + VIS stem window per source
    LatentPositionHead -> (dx, dy, log_sigma) in VIS pixels
    local WCS Jacobian -> residual sky offset in arcsec
    │
    ▼
    per-band metrics + optional anchors.npz
```

`anchors.npz` stores raw offsets and head residuals for every band. It can be passed directly to `fit_direct_pinn.py --cache` to fit raw or post-head residual fields.

### Concordance field flow (QA/fallback)

```
Tiles on disk
    │
    ▼
[source_matching.py]
    detect_sources()           — local-peak detection + subpixel centroiding in Rubin & VIS
    match_sources_wcs()        — mutual-nearest-neighbor match in sky coords + sigma-clip
    refine_centroids_in_band() — flux-weighted centroid refinement in the target Rubin band
    │
    ▼  per matched source pair:
[dataset.py]
    reproject_rubin_patch_to_vis()  — warp a 33×33 Rubin stamp onto the VIS pixel grid via WCS
    extract_vis_patch()             — cut the corresponding VIS stamp
    local_vis_pixel_to_sky_matrix() — 2×2 pixel→sky Jacobian at that position
    target_offset_arcsec            — (VIS_ra - Rubin_ra)*cos(dec)*3600, (VIS_dec - Rubin_dec)*3600
    │
    ▼  for each (rubin_patch, vis_patch, pixel_to_sky, target_offset):
[matcher_v6.py  V6AstrometryMatcher]
    V6RubinEncoder        — frozen v6 Phase B Rubin BandStem + trainable ConvNeXt adapter
    V6VISEncoder          — frozen v6 Phase B VIS BandStem + trainable ConvNeXt adapter
    _weighted_cost_volume — spatially-weighted cross-correlation (rubin_energy × Gaussian_prior)
                            → soft-argmax → coarse (dx_px, dy_px)
    residual MLP          — refine coarse estimate, predict log_sigma
    pixel_to_sky Jacobian — convert pixel shift to (ΔRA*, ΔDec) in arcsec
    │
    ▼  per tile at inference:
[field_solver.py  solve_control_grid_field]
    auto_grid_shape              — reduce grid resolution for sparse tiles
    bilinear control grid        — (anchor_xy, pred_offset, weight=1/σ²) → grid coefficients
    adaptive per-node anchor     — ridge regularization stronger where sources are sparse
    smoothness prior             — Tikhonov regularization on finite differences
    evaluate_control_grid_mesh() — regular mesh at DSTEP spacing + coverage map
    │
    ▼
[infer_concordance.py / fit_direct_pinn.py]
    FITS output  —  DRA + DDE + COV image HDUs per tile
```

### Global concordance (single smooth field, no tile boundaries)

```
[infer_global_concordance.py]
    collect_all_predictions()
        — runs the matcher on every tile
        — converts VIS pixel positions → (RA, Dec) via VIS WCS
    │
    ▼  all sources in a common sky frame
    sigma-clip: reject |pred| > clip_arcsec (default 300 mas)
    │
    ▼
    solve_global_field()   [--solver grid]
        convert (RA, Dec) → local tangent-plane arcsec offsets
        solve_control_grid_field() over the entire mosaic footprint
        → single smooth (ΔRA*, ΔDec) field covering all tiles
    ──── OR ────
    solve_global_field()   [--solver nn]
        fit DistortionMLP: (x_arcsec, y_arcsec) → (ΔRA*, ΔDec)
        Adam + cosine LR + weight-decay smoothness prior
        → infinitely differentiable field, no grid resolution to choose
    │
    ▼
    FITS output  —  GLOBAL.DRA + GLOBAL.DDE + GLOBAL.COV
                    with sky-coordinate WCS (CRVAL = field RA/Dec origin)
    │
    ▼
[GlobalConcordanceMap]
    correction_at_sky(ra, dec)
        — interpolates the global field at any sky position
        — no tile_id needed, no boundary discontinuities
    rubin_to_vis(rubin_x, rubin_y, rubin_wcs, vis_wcs)
        — same API as ConcordanceMap.rubin_to_vis
```

---

## Files

| File | What it does |
|------|-------------|
| `matcher.py` | `LocalAstrometryMatcher` — baseline neural network (single-band Rubin, random-init VIS CNN) |
| `matcher_v6.py` | `V6AstrometryMatcher` — Phase B backbone: frozen v6 Rubin + VIS BandStems + trainable adapters |
| `dataset.py` | `build_patch_samples` builds all training samples from tile pairs; `MatchedPatchDataset` serves them with optional augmentation |
| `field_solver.py` | Pure-numpy bilinear control-grid solver + evaluator with adaptive regularization — no learning |
| `nn_field_solver.py` | MLP-based global field solver — `DistortionMLP` + `fit_nn_field` + `evaluate_nn_mesh` |
| `train_local_matcher.py` | Training loop with W&B logging and tile-level diagnostic previews (used by both matchers) |
| `train_astro_v6.py` | Thin wrapper around `train_local_matcher.py` that instantiates `V6AstrometryMatcher` |
| `infer_concordance.py` | Run a trained checkpoint over all tiles and write a per-tile FITS concordance file |
| `infer_global_concordance.py` | Collect predictions from all tiles and fit a single global field (no tile boundaries); supports `--solver grid` and `--solver nn` |
| `apply_concordance.py` | `ConcordanceMap` — load per-tile FITS and apply corrections at any sky position |
| `sky_cube.py` | `SkyCubeExtractor` — given RA/Dec returns a 10-band [10, H, W] sky cube; can apply a smooth concordance field when requested |
| `latent_position_head.py` | `LatentPositionHead` — latent-space canonical position head for per-object multi-band alignment |
| `train_latent_position.py` | Training script for the latent position head (tile-level, jitter-based self-supervision) |
| `eval_latent_position.py` | Cross-instrument eval: align all 9 non-VIS bands to VIS, per-band metrics and diagnostics |
| `viz.py` | Diagnostic figures: per-tile offset maps, global field plots, coverage maps |

`source_matching.py` is in this directory (`astrometry2/`).

---

## Architecture details

### Spatially-weighted cost volume

Most pixels in a 33×33 patch are sky background, not source. If all pixels vote equally on the displacement, background noise dilutes the astrometric signal from the actual star or galaxy at the center.

The fix is to weight each pixel's contribution by:

```
spatial_w[i,j] = rubin_feature_energy[i,j] × gaussian_center_prior[i,j]
```

- `rubin_feature_energy`: L2 norm of the Rubin feature vector at pixel (i,j). High where the source is bright/detected.
- `gaussian_center_prior`: Gaussian peaked at the patch center (sigma = 0.5 in normalized [-1,1] coords). Suppresses edge pixels which are often noisier and less relevant.

The cost volume is computed via `F.unfold` over the padded VIS features, extracting all (2r+1)² shifted views in a single operation. The spatially-weighted dot product is then a broadcast multiply + sum. No Python loop over displacements.

The result: the bright point source at the anchor position dominates the offset vote.

### Center-biased pooling

The same Gaussian weights are used for the MLP's feature pooling. Instead of global mean-pool (which equally weights source and background), the pooled representation is dominated by the source center.

### Uncertainty output

`log_sigma` is the predicted log standard deviation of the offset error (in arcsec). The training loss is:

```
L = radial_error / sigma + log_sigma
```

This is the negative log-likelihood of a Rayleigh distribution. It forces the model to be honest: if sigma is large (uncertain), the penalty is smaller, but you pay `log_sigma`. If sigma is small but the error is large, you pay `radial_error / sigma`. At inference, `weight = 1 / σ²` is used to weight each anchor's contribution to the control-grid field solver.

Initial sigma is set to 50 mas (log(0.05) ≈ -3.0) so the loss is informative from epoch 1.

### Patch normalization

Raw flux patches are not cross-tile comparable: a source with flux=100 in a shallow tile may have the same S/N as a source with flux=10 in a deep tile.

Each patch is background-subtracted and MAD-normalized before reaching the encoder:
```
patch_norm = (patch - median(patch)) / (1.4826 * MAD(patch))
```
Using the median background and MAD noise estimate (rather than mean/std) ensures the bright source itself does not bias the normalization. After this, background pixels cluster around 0 with std ≈ 1, and the source is a clean high-amplitude S/N peak. This also makes the `rubin_feature_energy` weighting in the cost volume physically meaningful: high energy = high S/N, not just high flux.

Applied per-channel for Rubin (each band independently) and to the VIS channel. Applied in `__getitem__` after augmentation so stored patches remain as raw flux for visualization.

### Augmentation

Both Rubin and VIS patches are flipped (horizontal and/or vertical) synchronously with equal probability. Since the target sky-coord offset is a physical measurement independent of display orientation, it is unchanged. The pixel→sky Jacobian is updated consistently: a horizontal flip negates the x-column of the Jacobian (a +x pixel shift now maps to the opposite sky direction). This up to 4× multiplies the effective training set size with no extra data.

---

## Field solver details

### Adaptive grid resolution

The control grid has a fixed topology (default 12×12 = 144 nodes per axis). With only ~50 matched sources per tile, that's ~0.35 sources per node, heavily underdetermined. The solver would rely almost entirely on the regularizer, producing artifacts.

`auto_grid_shape` reduces the grid resolution so the number of nodes stays below half the number of anchors. A tile with 50 sources gets a 5×5 grid; a tile with 200+ sources keeps the full 12×12. This ensures the data constrains the field rather than the regularizer hallucinating it.

Enable with `--auto-grid`.

### Adaptive per-node anchor regularization

The smoothness prior (Tikhonov on finite differences) only penalizes differences between adjacent nodes. It's perfectly happy with a large constant offset in a source-sparse region, as long as it's locally smooth. A single noisy measurement near a corner can drag a cluster of nodes to a large value, and the smoothness prior propagates it unchecked.

The fix is adaptive per-node ridge regularization. Each grid node gets a regularization strength inversely proportional to its local data support:

```
lambda_i = anchor_lambda × max(1, base_support / local_support_i)
```

where `local_support_i` is the Gaussian-weighted sum of data weights near node i. Well-constrained interior nodes are barely affected. Unconstrained edge/corner nodes get pulled firmly toward zero.

Controlled by `--anchor-lambda` (base strength, default 1e-3) and `--anchor-radius-px` (Gaussian scale, auto-computed from grid cell spacing when using `--auto-grid`).

### Coverage map

Each tile now produces a coverage HDU (`{prefix}.COV`) recording the minimum distance (in VIS pixels) from each mesh point to the nearest anchor source. This tells downstream code where the concordance field is data-driven versus pure extrapolation. A threshold of ~100-150 VIS pixels is a reasonable cutoff for flagging unreliable regions.

---

## Precision target

For the per-object head, the practical ECDFS target is:
- Rubin g/r/i/z and NISP Y/J/H: median residual ~10 mas, p68 ~17-22 mas
- Rubin y: median residual < 15 mas
- Rubin u: substantially improved but lower-confidence because of SNR; current median is ~30 mas
- Smooth post-head field: ~1 mas amplitude, used as a QA check

Do not interpret the ~1 mas post-head PINN field amplitude as the per-source residual. It is the remaining coherent smooth component after the head, not the radial residual distribution of individual anchors.

---

## Training

### Latent position head (current)

For current astrometry work, use the v8 no-PSF latent head path:

```bash
# Train
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=models python models/astrometry2/train_latent_position.py \
  --rubin-dir        data/rubin_tiles_all \
  --euclid-dir       data/euclid_tiles_all \
  --foundation-checkpoint models/checkpoints/jaisp_v8_fine/checkpoint_best.pt \
  --output-dir       models/checkpoints/latent_position_v8_no_psf \
  --epochs 30 --bottleneck-window 5 --dual-gpu \
  --wandb-project JAISP-LatentPosition

# Evaluate and export anchors for QA fields
PYTHONPATH=models python models/astrometry2/eval_latent_position.py \
  --rubin-dir        data/rubin_tiles_all \
  --euclid-dir       data/euclid_tiles_all \
  --foundation-checkpoint models/checkpoints/jaisp_v8_fine/checkpoint_best.pt \
  --head-checkpoint  models/checkpoints/latent_position_v8_no_psf/best.pt \
  --save-anchors     models/checkpoints/latent_position_v8_no_psf/anchors.npz \
  --output-dir       models/checkpoints/latent_position_v8_no_psf/eval

# Fit the post-head residual field as a QA check
PYTHONPATH=models python models/astrometry2/fit_direct_pinn.py \
  --cache models/checkpoints/latent_position_v8_no_psf/anchors.npz \
  --use-head-resid \
  --output models/checkpoints/latent_position_v8_no_psf/concordance_pinn_head_resid_fixed.fits \
  --bands r i g z --include-nisp
```

Do not pass `--psf-checkpoint` for the current headline astrometry run. The PSFField path is available for experiments, but PSFField-refined centroids were worse latent-head labels in the v8 migration ablation.

### V7 matcher (historical / concordance experiments)

Uses frozen V7 BandStems from the RMS-aware foundation checkpoint. Supports multiband mode (all Rubin + NISP bands), CenterNet or classical source detection, and PSF-fit centroiding.

```bash
# V7 matcher + CenterNet detector + all bands
python train_astro_v7.py \
  --v7-checkpoint       ../checkpoints/jaisp_v7_concat/checkpoint_best.pt \
  --detector-checkpoint ../../checkpoints/centernet_v7_rms_aware/centernet_best.pt \
  --rubin-dir           ../../data/rubin_tiles_200 \
  --euclid-dir          ../../data/euclid_tiles_200 \
  --multiband \
  --epochs 120 \
  --output-dir ../checkpoints/astro_v7_psffit

# V7 matcher without CenterNet (classical source detection)
python train_astro_v7.py \
  --v7-checkpoint  ../checkpoints/jaisp_v7_concat/checkpoint_best.pt \
  --rubin-dir      ../../data/rubin_tiles_200 \
  --euclid-dir     ../../data/euclid_tiles_200 \
  --multiband \
  --epochs 120 \
  --output-dir ../checkpoints/astro_v7_classical

# Export concordance FITS for all bands
python infer_concordance.py \
  --checkpoint          ../checkpoints/astro_v7_psffit/checkpoint_best.pt \
  --v7-checkpoint       ../checkpoints/jaisp_v7_concat/checkpoint_best.pt \
  --detector-checkpoint ../../checkpoints/centernet_v7_rms_aware/centernet_best.pt \
  --rubin-dir      ../../data/rubin_tiles_all \
  --euclid-dir     ../../data/euclid_tiles_all \
  --output         concordance_v7.fits \
  --all-bands --auto-grid
```

### V7 latent position head (historical baseline)

For per-object alignment across all 10 bands (e.g. forced photometry), the latent position head uses the frozen V7 encoder's fused bottleneck + VIS stem features to predict chromatically-informed canonical source positions.

```bash
python train_latent_position.py \
  --rubin-dir  ../../data/rubin_tiles_200 \
  --euclid-dir ../../data/euclid_tiles_200 \
  --v7-checkpoint ../checkpoints/jaisp_v7_concat/checkpoint_best.pt \
  --epochs 30 --lr 3e-4 \
  --jitter-arcsec 0.03 --jitter-max-arcsec 0.1 \
  --output-dir ../checkpoints/latent_position_head
```

### V6 matcher (archived)

V6 commands are preserved for reference. Use V7 for new work.

```bash
python train_astro_v6.py \
  --v6-checkpoint ../checkpoints/jaisp_v6_phaseB/checkpoint_best.pt \
  --rubin-dir     ../../data/rubin_tiles_200 \
  --euclid-dir    ../../data/euclid_tiles_200 \
  --output-dir    ../checkpoints/astrometry_v6 \
  --epochs 30 --batch-size 64
```

The output FITS contains `{tile_id}.{band}.DRA`, `{tile_id}.{band}.DDE`, `{tile_id}.{band}.COV`
triplets for every tile × band combination (e.g. `tile_x0_y0.r.DRA`, `tile_x0_y0.i.DRA`, …).

Key arguments (both scripts):
- `--rubin-band`: target Rubin band (single-band mode, e.g. `r`, `i`, `z`)
- `--multiband`: train/infer for all Rubin bands simultaneously
- `--all-bands`: (infer only) export concordance for every band in a multiband checkpoint
- `--context-bands`: additional Rubin bands fed to the encoder (single-band mode)
- `--detect-bands`: bands used for multi-band source detection (default: g r i z)
- `--max-patches-per-tile`: max matched sources per tile (default 64)
- `--search-radius`: pixel-shift search half-width (default 3 = ±0.3" at VIS scale)
- `--patch-size`: must be odd (default 33 = 3.3" at VIS resolution)

Extra arguments for `train_astro_v6.py`:
- `--v6-checkpoint`: path to Phase B foundation checkpoint (required)
- `--adapter-blocks`: trainable ConvNeXt adapter blocks on top of frozen stems (default 2)
- `--unfreeze-stems`: fine-tune the v6 stems (default: frozen)

Training prints per-epoch metrics in mas: `train_MAE`, `train_p68`, `val_MAE`, `val_p68`.

---

## Applying a concordance field in practice

Use this when you explicitly want a smooth WCS/concordance correction. For the current per-object astrometry claim, use the latent head residuals from `eval_latent_position.py`; the concordance field is QA/fallback and does not remove the dominant centering scatter by itself.

`apply_concordance.py` provides `ConcordanceMap` — load once, apply anywhere.

### Point source / catalog matching

```python
import numpy as np
from astropy.wcs import WCS
from apply_concordance import ConcordanceMap

cmap = ConcordanceMap('concordance_r.fits')

# Project a Rubin source onto the VIS pixel grid
vis_x, vis_y = cmap.rubin_to_vis(
    rubin_x=247.3, rubin_y=183.1,
    rubin_wcs=rubin_wcs,          # astropy WCS for the Rubin tile
    vis_wcs=vis_wcs,              # astropy WCS for the VIS tile
    tile_id='tile_x01024_y00000',
    band='r',
)
```

### Full-image reprojection (Rubin → VIS pixel grid)

```python
from scipy.ndimage import map_coordinates

H, W = rubin_img.shape
yy, xx = np.mgrid[0:H, 0:W]

# Get corrected VIS coords for every Rubin pixel
vis_coords = cmap.rubin_grid_to_vis(
    rubin_xs=xx.ravel(), rubin_ys=yy.ravel(),
    rubin_wcs=rubin_wcs, vis_wcs=vis_wcs,
    tile_id='tile_x01024_y00000', band='r',
)
# vis_coords: [H*W, 2]  columns = (vis_x, vis_y)

# Resample Rubin image onto VIS pixel grid
vis_H, vis_W = vis_img.shape
rubin_on_vis = map_coordinates(
    rubin_img,
    [vis_coords[:, 1].reshape(vis_H, vis_W),   # row = y
     vis_coords[:, 0].reshape(vis_H, vis_W)],   # col = x
    order=1, mode='constant', cval=np.nan,
)
```

### Masking unreliable regions

```python
# Coverage = min distance (VIS px) to nearest anchor source
# High coverage means the field is extrapolated, not data-driven
vis_xy = np.stack([vis_coords[:, 0], vis_coords[:, 1]], axis=1).astype(np.float32)
cov = cmap.coverage('tile_x01024_y00000', 'r', vis_xy)
reliable_mask = cov < 150   # VIS pixels within 150px of an anchor
```

---

## Global concordance (recommended field product)

Per-tile fields are independent, so adjacent tiles can disagree at their shared boundary.
`infer_global_concordance.py` solves a **single** smooth field over the entire mosaic
in sky coordinates — no tile edges, no boundary artefacts.

For the current ECDFS data, the global field should be interpreted as the smooth WCS component. The raw-anchor field amplitude is only a few mas, and the head-residual field is ~1 mas.

### Running

```bash
python infer_global_concordance.py \
    --rubin-dir  ../../data/rubin_tiles_all \
    --euclid-dir ../../data/euclid_tiles_all \
    --checkpoint ../checkpoints/astro_v7_psffit/checkpoint_best.pt \
    --v7-checkpoint ../checkpoints/jaisp_v7_concat/checkpoint_best.pt \
    --output     ../checkpoints/astro_v7_psffit/global_concordance.fits \
    --dstep-arcsec 1.0 \
    --clip-arcsec 0.3 \
    --auto-grid \
    --solver nn \
    --plot       ../checkpoints/astro_v7_psffit/global_concordance_plot.png \
    --summary-json ../checkpoints/astro_v7_psffit/global_summary.json
```

### Solver options

| Flag | Description |
|------|-------------|
| `--solver grid` | Control-grid least-squares (fast, default). Uses `--grid-h-global`, `--grid-w-global`, `--smooth-lambda`, `--anchor-lambda`. |
| `--solver nn` | MLP trained with Adam + weight-decay smoothness prior. No grid resolution to choose; SiLU activations give an infinitely differentiable field. |
| `--nn-hidden-dim` | Neurons per hidden layer (default 64) |
| `--nn-layers` | Number of hidden layers (default 4) |
| `--nn-steps` | Adam training steps (default 2000) |
| `--nn-lr` | Initial learning rate (default 1e-3) |
| `--nn-weight-decay` | L2 weight decay — higher = smoother field (default 1e-4, try 1e-3 if spiky) |
| `--clip-arcsec` | Reject sources with \|pred offset\| > this value before solving (default 0.3" = 300 mas) |
| `--dstep-arcsec` | Output mesh resolution in arcsec (default 1.0) |

### Using the global field

```python
from infer_global_concordance import GlobalConcordanceMap

gcmap = GlobalConcordanceMap('global_concordance_r.fits')

# Apply correction at any sky position — no tile_id needed
vis_x, vis_y = gcmap.rubin_to_vis(
    rubin_x, rubin_y,
    rubin_wcs=rubin_wcs,
    vis_wcs=vis_wcs,
)

# Check coverage (arcsec to nearest source anchor)
dra, ddec = gcmap.correction_at_sky(ra_array, dec_array)
cov = gcmap.coverage_at_sky(ra_array, dec_array)
reliable = cov < 150   # arcsec
```

### Why the NN solver

The control-grid solver requires choosing a grid resolution: too coarse → can't resolve real spatial structure; too fine → underdetermined, spiky.  The MLP sidesteps this entirely.  Weight decay acts as a continuous smoothness prior — the same mechanism as Tikhonov regularization but without an explicit grid.  SiLU activations produce a field that is smooth to all orders, avoiding the piecewise-linear artefacts that can appear at control-grid cell boundaries.  Smoothness is tuned via `--nn-weight-decay` alone.

With the current data (~50,000+ sources over 790 tiles) both solvers produce comparable smooth-field diagnostics. The NN advantage may grow as data volume increases or on fields with larger coherent residuals. On ECDFS, solver choice does not change the main conclusion: the smooth field is small compared with source-centering scatter.

---

## Export per-tile concordance FITS

```bash
python models/astrometry2/infer_concordance.py \
  --rubin-dir data/rubin_tiles_all \
  --euclid-dir data/euclid_tiles_all \
  --checkpoint models/checkpoints/astro_v7_psffit/checkpoint_best.pt \
  --v7-checkpoint models/checkpoints/jaisp_v7_concat/checkpoint_best.pt \
  --output models/checkpoints/astro_v7_psffit/concordance.fits \
  --all-bands --auto-grid \
  --plot-dir models/checkpoints/astro_v7_psffit/plots
```

The output FITS file has three HDUs per tile:
- `{tile_id}.r.DRA` — ΔRA* field in arcsec, sampled every `dstep` VIS pixels
- `{tile_id}.r.DDE` — ΔDec field in arcsec, sampled every `dstep` VIS pixels
- `{tile_id}.r.COV` — coverage map: min distance (VIS px) to nearest anchor source

Each offset HDU carries `DSTEP`, `DUNIT=arcsec`, `INTERP=bilinear`, `CONCRDNC=True`, `RBNBAND`, and a scaled linear WCS (`CRPIX`/`CD`/`PC`/`CDELT`) so the pixel coordinates map correctly to the VIS frame. Higher-order distortion keywords are intentionally not propagated in this version.

Downstream usage:
```
Rubin pixel (x_r, y_r)
  → Rubin WCS → (RA_r, Dec_r)
  → interpolate concordance at (RA_r, Dec_r) → (dra, ddec)
  → corrected sky: (RA_r + dra/cos(dec)/3600, Dec_r + ddec/3600)
  → VIS WCS inverse → VIS pixel (x_v, y_v)
```

New inference flags:
- `--auto-grid`: automatically reduce grid resolution for tiles with few matches
- `--anchor-lambda`: ridge regularization base strength (default 1e-3, raised from 1e-4)
- `--anchor-radius-px`: Gaussian scale for adaptive anchor (0 = auto-computed from grid spacing)

---

## Tile size considerations

The defaults are tuned for ~1000×1000 VIS pixel tiles. Smaller tiles risk being source-starved: a 500×500 tile might only yield 20-30 matches, pushing the grid solver to its minimum 4×4 resolution. Larger tiles (4000×4000+) work fine but the 12×12 grid may be too coarse to capture real spatial structure; scale `--grid-h` and `--grid-w` proportionally.

The patch size (33×33) and search radius (±3 pixels) are fixed by the trained model and don't change with tile size. Source detection parameters assume a certain density per unit area, not per tile.

---

## Design decisions and current defaults

**VIS grid adherence**: The concordance product is defined as a residual sky-offset field on top of the VIS WCS, and exported with a scaled linear WCS for mesh indexing. We intentionally defer full distortion-aware mesh WCS propagation for now; if downstream validation shows this is limiting, distortion terms can be added in a follow-up.

**Mesh size (DSTEP)**: Default is 8 VIS pixels (0.8"). Rubin astrometric distortions from DCR vary on arcminute scales; 0.8" sampling is likely overkill. The `smooth_lambda` regularizer keeps the solved field smooth regardless. Empirically, DSTEP=16 or 32 may give equally good results with fewer mesh samples (smaller files), not fewer control-grid solve nodes.

**Band-specific fields**: Smooth concordance products are exported per band because DCR and centroid definitions are wavelength dependent. The latent head, however, is evaluated across all 9 non-VIS bands against the VIS reference.

**No epoch correction**: The surveys are assumed to handle proper motion and parallax internally. The concordance corrects for systematic WCS residuals only.

**Scaling to Roman/NISP**: The v6 VIS encoder reuses the `euclid_VIS` BandStem from the Phase B foundation model. To extend to NISP or Roman, a Phase B checkpoint with a BandStem trained on the target instrument's band is needed — or fall back to the baseline `LocalAstrometryMatcher` with a fresh CNN for that channel.

---

## Planned improvements

**Held-out / OOD validation**: The current numbers are ECDFS tract5063. The next strongest claim requires held-out tiles or a non-ECDFS field to verify transfer without retraining.

**Residual maps after the head**: Continue checking that post-head residual vectors have no coherent spatial pattern and that head-residual PINN fields stay at the ~1 mas level.

**Population breakdown**: Report head residuals separately for bright/faint, compact/extended, and high/low SNR sources. Notebook 09 already shows centering is the dominant raw offset; the next step is making the post-head outlier story equally explicit.

**Per-object alignment for photometry**: Use the latent position head's corrected positions as input to the forced photometry pipeline and measure whether multi-band-informed positions improve photometric precision compared to single-band centroiding.

**Joint PSF + astrometry training**: PSFField is not a drop-in latent-head label source yet. Revisit it with a target convention that the head can see in its features, or train PSF/position jointly so the centroid definition is shared.

**True super-resolution in sky_cube.py**: `SkyCubeExtractor` currently uses bicubic resampling (order=3) to bring Rubin from 0.2"/px to VIS resolution (0.1"/px). The V7 decoder conditioned on `euclid_VIS` would produce a physically motivated super-resolution; wire it in as an optional `mode='sr'` path once validated.
