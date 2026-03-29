# Astrometry2: Rubin → Euclid VIS Concordance

## What this is for

Rubin and Euclid observe the same sky but their astrometric solutions are not perfectly aligned. There are small residual offsets between the two instruments, varying slowly across the field. For JAISP to combine Rubin and VIS data at the pixel level, we need to know, at every position in the VIS frame: "if I project a Rubin sky coordinate through the VIS WCS, by how many arcseconds is it off?"

The answer is a smooth 2D field of corrections (ΔRA*, ΔDec) stored as a coarse mesh over the VIS tile. This is the **astrometry concordance product**.

Key design principles (from the JAISP data product spec):
- The correction is stored in **sky coordinates** (arcsec), not pixel offsets.
- At every VIS pixel position (x, y): apply the correction to Rubin's sky coord before projecting onto the VIS grid.
- Perfect alignment means ΔRA* = ΔDec = 0 everywhere.
- WCS handles geometry; concordance handles the residual astrometric error.
- The correction mesh is sampled every ~8 VIS pixels (DSTEP=8, i.e. 0.8"), keeping files compact. Downstream code bilinearly interpolates to native resolution at runtime.

---

## Why a neural network?

Classical source matching gives you a discrete set of (Rubin source position, VIS source position, offset) triples, typically a few hundred per tile. To turn those sparse noisy measurements into a smooth continuous field you need to suppress outliers (blended sources, bad detections) and interpolate coherently across the tile.

A CNN matcher operating at native VIS resolution does both better than traditional approaches. It looks at the actual image morphology of each matched source pair, not just the catalog centroids. A blended source that has a clean centroid in the Rubin catalog but is clearly resolved in VIS will be flagged through a high predicted uncertainty. A faint source sitting on a diffraction spike will similarly be downweighted. Classical matching has no access to this pixel-level context without hand-crafted quality flags.

The learned uncertainty (log_sigma) then weights the control-grid field solver, so reliable point sources dominate the field and bad matches are automatically suppressed. The model also trains jointly across all tiles, so it learns shared priors about what Rubin PSFs look like and what typical offset patterns are, rather than fitting each tile independently from scratch.

---

## Current results

On 144 ECDFS tiles:

| Model | val MAE | p68 | frac < 0.1" | Notes |
|-------|---------|-----|-------------|-------|
| Raw WCS | ~47 mas | — | — | No correction |
| astrometry2 baseline CNN | ~38 mas | — | — | Single-band, random-init VIS encoder |
| v6 Phase A stems | ~60 mas | — | — | Frozen Rubin stems, random VIS CNN — cost volume mismatched |
| **v6 Phase B all-bands** | **31.9 mas** | **34.6 mas** | **96.1%** | Frozen Rubin + VIS stems from joint Phase B training |

The Phase B improvement is the key result: both the Rubin and VIS encoders now share a cross-instrument feature space learned during Phase B masked-band reconstruction, so the cost volume compares representations that were trained to be spatially consistent with each other. Phase A v6 was actually worse than baseline because the VIS encoder was a random CNN being compared against frozen Rubin stems — incompatible feature spaces.

Target: sub-20 mas (limited by source density and tile count, not model capacity).
NISP-to-VIS offsets (same telescope) are 38–41 mas for comparison, confirming the cross-telescope residual is real and correctable.

The main bottleneck is data volume: ~50 matched sources per tile across 144 tiles. More tiles or deeper detection would directly improve field solver conditioning and astrometric precision.

---

## Pipeline overview

### Per-tile concordance (fast, independent tiles)

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
[infer_concordance.py]
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
| `sky_cube.py` | `SkyCubeExtractor` — given RA/Dec returns an aligned 10-band [10, H, W] sky cube with concordance applied |
| `viz.py` | Diagnostic figures: per-tile offset maps, global field plots, coverage maps |

`source_matching.py` lives in `../astrometry/` and is shared with the older pipeline.

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

Residuals should be below the native astrometric uncertainties of each survey:
- Rubin (LSST coadds): ~10-20 mas RMS
- Euclid VIS: ~5 mas RMS

Aim for `p68_total < 20 mas` on the validation set.

---

## Training

### Baseline matcher (single-band, no foundation model)

```bash
python train_local_matcher.py \
  --rubin-dir  ../../data/rubin_tiles_ecdfs \
  --euclid-dir ../../data/euclid_tiles_ecdfs \
  --rubin-band r \
  --output-dir ../checkpoints/astrometry2_r \
  --epochs 30 --batch-size 64
```

### v6 Phase B matcher — single band

Requires a Phase B foundation checkpoint (`jaisp_v6_phaseB*/checkpoint_best.pt`).
Uses frozen v6 BandStems for both Rubin and VIS encoders — both live in the same cross-instrument feature space learned during Phase B.

```bash
python train_astro_v6.py \
  --v6-checkpoint ../checkpoints/jaisp_v6_phaseB/checkpoint_best.pt \
  --rubin-dir     ../../data/rubin_tiles_ecdfs \
  --euclid-dir    ../../data/euclid_tiles_ecdfs \
  --output-dir    ../checkpoints/astrometry_v6 \
  --epochs 30 --batch-size 64 \
  --wandb-project JAISP-Astrometry-v6
```

### v6 Phase B matcher — all Rubin bands (recommended with more data)

Train a single model for all 6 Rubin bands simultaneously using `--multiband`.
A shared encoder sees all bands; a learned band embedding handles per-band DCR corrections.
Each Rubin band gets its own DRA/DDE/COV concordance field in the output FITS.

```bash
# 1. Train multiband matcher
python train_astro_v6.py \
  --v6-checkpoint ../checkpoints/jaisp_v6_phaseB/checkpoint_best.pt \
  --rubin-dir     ../../data/rubin_tiles_ecdfs \
  --euclid-dir    ../../data/euclid_tiles_ecdfs \
  --output-dir    ../checkpoints/astrometry_v6_multiband \
  --multiband \
  --epochs 30 --batch-size 64 \
  --wandb-project JAISP-Astrometry-v6 \
  --wandb-run-name v6_multiband

# 2. Export concordance FITS for all 6 Rubin bands in one run
python infer_concordance.py \
  --rubin-dir    ../../data/rubin_tiles_ecdfs \
  --euclid-dir   ../../data/euclid_tiles_ecdfs \
  --checkpoint   ../checkpoints/astrometry_v6_multiband/checkpoint_best.pt \
  --v6-checkpoint ../checkpoints/jaisp_v6_phaseB/checkpoint_best.pt \
  --output       ../checkpoints/astrometry_v6_multiband/concordance_all_bands.fits \
  --all-bands \
  --auto-grid \
  --plot-dir     ../checkpoints/astrometry_v6_multiband/plots \
  --summary-json ../checkpoints/astrometry_v6_multiband/summary.json
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

## Applying the concordance in practice

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

## Global concordance (recommended for production)

Per-tile fields are independent, so adjacent tiles can disagree at their shared boundary.
`infer_global_concordance.py` solves a **single** smooth field over the entire mosaic
in sky coordinates — no tile edges, no boundary artefacts.

### Running

```bash
python infer_global_concordance.py \
    --rubin-dir  ../../data/rubin_tiles_ecdfs \
    --euclid-dir ../../data/euclid_tiles_ecdfs \
    --checkpoint ../checkpoints/astrometry_v6_phaseB2/checkpoint_best.pt \
    --v6-checkpoint ../checkpoints/jaisp_v6_phaseB2/checkpoint_best.pt \
    --output     ../checkpoints/astrometry_v6_phaseB2/global_concordance_r.fits \
    --dstep-arcsec 1.0 \
    --clip-arcsec 0.3 \
    --auto-grid \
    --solver nn \
    --plot       ../checkpoints/astrometry_v6_phaseB2/global_concordance_plot.png \
    --summary-json ../checkpoints/astrometry_v6_phaseB2/global_summary.json
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

With the current data (~5000 sources over 48 tiles) both solvers produce comparable results.  The NN advantage will grow as data volume increases and the field develops real small-scale structure that a fixed grid resolution cannot resolve without also fitting noise.

---

## Export per-tile concordance FITS

```bash
/home/shemmati/venvs/superres/bin/python models/astrometry2/infer_concordance.py \
  --rubin-dir data/rubin_tiles_ecdfs \
  --euclid-dir data/euclid_tiles_ecdfs \
  --checkpoint models/checkpoints/astrometry2_r/best_matcher.pt \
  --output models/checkpoints/astrometry2_r/concordance_r.fits \
  --plot-dir models/checkpoints/astrometry2_r/plots \
  --auto-grid
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

**One band at a time**: Each checkpoint targets one Rubin band. DCR causes wavelength-dependent offsets, so each band should have its own concordance field. Use `--context-bands` to pass additional bands as encoder input context without making them the alignment target.

**No epoch correction**: The surveys are assumed to handle proper motion and parallax internally. The concordance corrects for systematic WCS residuals only.

**Scaling to Roman/NISP**: The v6 VIS encoder reuses the `euclid_VIS` BandStem from the Phase B foundation model. To extend to NISP or Roman, a Phase B checkpoint with a BandStem trained on the target instrument's band is needed — or fall back to the baseline `LocalAstrometryMatcher` with a fresh CNN for that channel.

---

## Planned improvements

**Multiband concordance**: Re-train with `--multiband` once more tile data arrives, then export per-band global concordance with `--all-bands`. Each Rubin band has its own DCR-driven offset pattern; a shared model with band embeddings handles this efficiently.

**Deeper source detection**: Lowering detection thresholds or using forced photometry at known catalog positions would increase the number of matched sources per tile, directly improving field solver conditioning in sparse regions. Target: 200+ sources/tile → enables a finer NN field or a denser control grid.

**Unfreeze stems after warmup**: The current v6 matcher freezes the Phase B stems throughout training. A two-stage schedule — freeze for the first N epochs to stabilize the adapter, then unfreeze with a lower LR — may improve astrometric precision by allowing the stems to specialize toward centroid localization.

**True super-resolution in sky_cube.py**: `SkyCubeExtractor` currently uses bicubic resampling (order=3) to bring Rubin from 0.2"/px to VIS resolution (0.1"/px). This is geometric resampling — no new information. The v6 Phase B decoder conditioned on `euclid_VIS` would produce a physically motivated super-resolution; wire it in as an optional `mode='sr'` path once the decoder is validated.
