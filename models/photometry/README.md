# JAISP Photometry

This module provides two photometry paths on Rubin+Euclid tiles:

- `PSFFieldPhotometryPipeline`: fast PSF-template forced photometry for compact or isolated sources.
- `scarlet_like.py`: a residual-trained, scarlet-like local scene model for extended sources and blends.
- `foundation_head.py`: the learned V8 downstream photometry head. It reads frozen
  V8 bottleneck + VIS-stem features at CenterNet/astrometry-corrected positions
  and is trained by scene residual chi-square.

Use `models/psf/` for PSF model training/validation. Use this directory for flux extraction and the legacy PSFNet reference code.

## Components

- `psf_net.py`: legacy `PSFNet` spatially varying PSF model, retained as reference
- `train_psf_net.py`: legacy star-driven PSF training loop
- `psf_field_pipeline.py`: current PSFField-backed forced-photometry pipeline
- `scarlet_like.py`: positive morphology + per-band flux scene optimizer, trained by residual chi-square
- `foundation_head.py`: trainable morphology-refinement head on top of frozen V8 features
- `train_foundation_photometry_head.py`: CenterNet -> astrometry head -> V8 photometry-head training loop
- `stamp_extractor.py`: batched stamp extraction + local background estimation
- `forced_photometry.py`: vectorized matched-filter flux estimator
- `pipeline.py`: legacy PSFNet end-to-end tile photometry wrapper

## Current Photometry Strategy

The matched-filter PSF path is still the right first diagnostic. It is fast,
stable, and gives a clear residual map. But large residuals for galaxies are
not automatically a PSF failure: an extended source is not a delta function
convolved with the PSF.

For a non-neural upper-bound/debugging path, use the scarlet-like optimizer:

1. Detect a master catalog, preferably in Euclid VIS via CenterNet.
2. Build a non-negative morphology initializer from VIS stamps.
3. Convolve each source morphology with the PSFField template in each band.
4. Fit non-negative per-band source fluxes by minimizing the noise-weighted
   pixel residual of the whole local blend scene.

The production direction is the V8 foundation photometry head:

1. CenterNet proposes VIS-frame detections from the 10-band tile.
2. The latent astrometry head corrects those positions object-by-object.
3. The frozen V8 encoder provides bottleneck and VIS-stem features.
4. `FoundationScarletPhotometryHead` predicts morphology refinements from those
   features.
5. PSFField renders per-band PSFs, fluxes are solved analytically, and the head
   weights are trained by the residual chi-square of local scenes.

The scarlet-like optimizer remains useful as a baseline/refinement reference,
but the trainable checkpoint to claim is produced by
`train_foundation_photometry_head.py`.

## Legacy PSFNet Design Choices

### 1) PSF parameterization in log space

`PSFNet` predicts per-source PSFs using:

- a per-band learnable base PSF (`base_psf`), initialized from approximate Gaussian FWHM
- a learned spatial residual from an MLP of `(x_norm, y_norm, band_embedding)`

The model applies `softplus(base + delta)` and then L1-normalizes each PSF stamp to sum to 1.

Why this choice:

- positivity is guaranteed
- initialization is physically sensible from epoch 0
- the MLP only learns residual structure, not the full PSF from scratch

### 2) Spatially varying PSFs at throughput scale

For inference, PSFs are precomputed on a regular grid:

- `precompute_grid(grid_size=8)` -> `[B, G, G, S, S]`
- per-source PSFs are then obtained by bilinear interpolation in `(x_norm, y_norm)`

This avoids per-source full forward passes and enables high-throughput forced photometry.

### 3) Robust star-driven training

`train_psf_net.py` detects compact, high-SNR sources and filters by concentration and central-pixel SNR.

Training loss uses analytic optimal per-source flux and optimizes PSF shape parameters by weighted residual chi-square.

### 4) Forced photometry with known-position estimator

`forced_photometry.matched_filter` implements the Cramer-Rao optimal linear estimator:

- `flux = sum(d * p / var) / sum(p^2 / var)`
- `flux_err = 1 / sqrt(sum(p^2 / var))`

This is fully vectorized over `[N_sources, N_bands, S, S]` tensors.

### 5) Local background subtraction before fitting

`TilePhotometryPipeline` estimates annulus background per source and subtracts it before matched filtering. This prevents background pedestal from biasing flux and chi-square.

## Quick Start

From repo root:

```python
import torch
from models.photometry import PSFFieldPhotometryPipeline

pipe = PSFFieldPhotometryPipeline.from_checkpoint(
    'models/checkpoints/psf_field_v3.pt',
    band_names=['rubin_g', 'rubin_r', 'rubin_i', 'rubin_z'],
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
)

# tile/rms: [B, H, W] tensors on one pixel grid.
# positions_px can be [N, 2] shared positions or [N, B, 2] per-band
# astrometry-head-corrected positions.
result = pipe.run(tile, rms, positions_px, sed_vec=None)

# result keys: flux, flux_err, chi2_dof, snr, bg
```

Train the V8 foundation photometry head:

```bash
python models/photometry/train_foundation_photometry_head.py \
  --rubin-dir data/rubin_tiles_all \
  --euclid-dir data/euclid_tiles_all \
  --foundation-checkpoint models/checkpoints/jaisp_v8_fine/checkpoint_best.pt \
  --detector-checkpoint checkpoints/centernet_v8_fine/centernet_best.pt \
  --astrometry-checkpoint models/checkpoints/latent_position_v8_no_psf/best.pt \
  --psf-checkpoint models/checkpoints/psf_field_v3.pt \
  --output-dir models/checkpoints/photometry_foundation_v1 \
  --epochs 20 \
  --max-sources 48 \
  --max-sources-per-step 24 \
  --wandb-project JAISP-photometry \
  --wandb-name photometry_foundation_v1 \
  --wandb-log-images
```

With W&B enabled, the trainer logs train/val scalar curves plus validation
residual galleries (`data-bg`, `learned model`, `residual`) every epoch by
default. Use `--wandb-image-every 5` to log images less often, or
`--wandb-image-band 1`/`2`/`3` to visualize Y/J/H instead of VIS.

Quick smoke test:

```bash
python models/photometry/train_foundation_photometry_head.py \
  --epochs 1 \
  --max-tiles 2 \
  --val-frac 0.5 \
  --max-sources 8 \
  --max-sources-per-step 6 \
  --min-sources 2 \
  --output-dir models/checkpoints/photometry_foundation_smoke \
  --sub-grid 1
```

Scarlet-like per-scene residual fit for a local tile:

```python
import torch
from models.photometry import (
    PSFFieldPhotometryPipeline,
    fit_scarlet_like_tile,
    make_positive_morphology_templates,
)

# euclid_tile/rms: [4, H, W]; vis_positions: [N, 2] in VIS pixels.
morph = make_positive_morphology_templates(
    euclid_tile[0],
    vis_positions,
    stamp_size=31,
)["templates"]

psf_pipe = PSFFieldPhotometryPipeline.from_checkpoint(
    'models/checkpoints/psf_field_v3.pt',
    band_names=['euclid_VIS', 'euclid_Y', 'euclid_J', 'euclid_H'],
    stamp_size=31,
    sub_grid=2,
)
psfs = psf_pipe.render_psfs(vis_positions, tile_hw=euclid_tile.shape[-2:])

scene_out = fit_scarlet_like_tile(
    euclid_tile,
    euclid_rms,
    vis_positions,
    morph,
    psfs=psfs,
    n_steps=120,
)

# scene_out keys include flux, chi2_dof, groups, and scene_results.
```

If all bands have been reprojected onto a VIS 0.1"/px grid, pass
`px_scales=[0.1] * len(band_names)` so PSFField renders templates on the
same pixel footprint as the data. If you are running on native Rubin images,
leave the default 0.2"/px Rubin scales.

Legacy PSFNet experiment:

```bash
python models/photometry/train_psf_net.py \
  --rubin_dir data/rubin_tiles_200 \
  --euclid_dir data/euclid_tiles_200 \
  --out models/checkpoints/psf_net_v1.pt \
  --epochs 20
```

Inference with saved checkpoint:

```python
import torch
from models.photometry.pipeline import TilePhotometryPipeline

pipe = TilePhotometryPipeline.from_checkpoint(
    'models/checkpoints/psf_net_v1.pt',
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
)

# tile: [C, H, W], rms: [C, H, W], positions_px: [N, 2]
# result = pipe.run(tile, rms, positions_px)
# result keys: flux, flux_err, chi2_dof, snr, bg
```

## Output Contract (`TilePhotometryPipeline.run`)

Returns a dict:

- `flux`: `[N, B]`
- `flux_err`: `[N, B]`
- `chi2_dof`: `[N, B]`
- `snr`: `[N, B]`
- `bg`: `[N, B]`

Where `N` is number of source positions and `B` is number of bands in the tile tensor.

`fit_scarlet_like_tile` returns a dict with:

- `flux`: `[N, B]` non-negative fitted source fluxes
- `chi2_dof`: `[N, B]` group-scene residual chi-square copied to each member source
- `groups`: local blend groups used for scene fitting
- `scene_results`: data/model/residual tensors and loss history for visual diagnostics

`io/13_foundation_photometry_head.ipynb` loads a trained
`FoundationScarletPhotometryHead` checkpoint and compares it against PSF-only
photometry on the same CenterNet + astrometry-corrected catalog. `io/12` remains
the per-scene scarlet-like optimizer diagnostic.

## PSF-Template Centroiding (Historical Bridge to Astrometry)

PSFNet can serve as an experimental centroiding engine for the astrometry pipeline, replacing the Gaussian-fit centroiding in `source_matching.refine_centroids_psf_fit`. This path is historical and should not be confused with the current astrometry result.

The current astrometry finding is that the large raw offsets are dominated by source centering / centroid-definition scatter. PSFField-refined centroids were tested as latent-head targets and degraded the head to a ~29-30 mas plateau, so PSF-template centroids are not currently the production astrometry label convention.

```python
from models.photometry.psf_net import load_psf_net, refine_centroids_psf_template

psf_net = load_psf_net('checkpoints/psf_net_v1.pt', device='cuda')
refined_xy, snr, fwhm = refine_centroids_psf_template(
    vis_image, seed_positions, psf_net,
    band_name='euclid_VIS', tile_hw=(1084, 1084),
)
```

## Practical Notes

- Bands follow `BAND_ORDER` in `psf_net.py` for multi-band training/inference consistency.
- Input `rms` maps should be strictly positive; code clamps very small values for numerical stability.
- For current PSF work, see `models/psf/PSFField`; for current astrometry, see `models/astrometry2/README.md`.
