# JAISP Photometry

This module provides the legacy matched-filter forced-photometry pipeline on Rubin+Euclid tiles. The original PSF model here, `PSFNet`, was never the production PSF path; the current PSF model is `models/psf/PSFField`.

Use this directory as the forced-photometry implementation and PSFNet reference code. Use `models/psf/` for current PSF modelling.

## Components

- `psf_net.py`: legacy `PSFNet` spatially varying PSF model, retained as reference
- `train_psf_net.py`: legacy star-driven PSF training loop
- `stamp_extractor.py`: batched stamp extraction + local background estimation
- `forced_photometry.py`: vectorized matched-filter flux estimator
- `pipeline.py`: end-to-end tile photometry wrapper

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

```bash
# Legacy PSFNet experiment. Current PSFField training lives in models/psf/.
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
