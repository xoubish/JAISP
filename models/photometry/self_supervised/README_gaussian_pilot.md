# Label-free multiband photometry pilot

This implementation fits **Rubin u/g/r/i/z/y and Euclid VIS/Y/J/H** together. No catalog flux labels enter preparation, PSF calibration, training, or checkpoint selection. It is a working first experiment, not a validated improvement in total-flux accuracy.

Open [nb_all_band_photometry.ipynb](nb_all_band_photometry.ipynb) for the executed all-band comparisons, running medians with central 68% distributions, and a crowded scene. The plots distinguish image-fit quality, changes in flux, and accuracy against known truth; only the first two are measured on real Q1 images here.

## Pilot result

The completed run uses 48 training, 11 validation, and 47 test scenes. Foundation-corrected pixel χ² improves in nine bands but worsens VIS by 9.9% relative to image moments. The constant-feature control reproduces almost all of this: per-band χ² differs by at most 0.23%. The foundation mostly learns a shared size expansion (median linear factor 1.801, 16th–84th percentiles 1.797–1.808).

**This experiment does not establish a useful foundation-specific gain or better real-source flux accuracy.** The first head is a working all-band control experiment. A stronger image-only shape fit should replace the biased weighted/truncated moment initializer before evaluating more flexible foundation morphology. The displayed holdout is exploratory, not a pristine test for further architecture selection.

## Model and controls

A detected source has one positive-definite intrinsic elliptical Gaussian covariance in tangent-plane arcsec². VIS image moments, with the approximate PSF covariance subtracted, initialize that shape. A small identity-initialized network receives a 3×3 window of frozen v11 features and predicts a bounded size/shear correction. It predicts no flux or color.

For each band, the renderer projects the covariance through the local WCS, adds a calibrated circular Gaussian PSF covariance, and integrates over native pixels with subpixel quadrature. Normalization refers to the full profile; cropped templates are **not** renormalized. The joint model is

`image_b = sky_b + sum_s flux_s,b × unit_flux_template_s,b`.

All source amplitudes and a constant background are fitted jointly by variance-weighted least squares. Fluxes are signed to avoid positivity bias near zero. Singular scenes fail explicitly; there is no clipped pseudo-NNLS or hidden ridge. The saved source covariance retains blend anticorrelations. Errors are conditional on shape/PSF and assume diagonal pixel noise.

The comparisons use the same detected objects, frozen astrometry, PSFs, scene pixels, solver, sky partitions, optimizer, and epoch budget:

- **Image moments:** no trained photometry correction.
- **Global correction:** the same network receives a fixed mean training feature vector for every source.
- **Foundation correction:** each source receives its own frozen features.

The global control tests whether feature dependence adds anything beyond a shared shape calibration. It is not a scratch image-CNN benchmark. The separate supervised aperture/CNN experiments remain in `../simple_regressor_review` and are not directly comparable to this native-unit, different-target scene likelihood.

## Run from the repository root

```bash
python -m unittest discover -s models/photometry/self_supervised/tests -v

python -m models.photometry.self_supervised.run

python -m models.photometry.self_supervised.run \
  --output models/photometry/self_supervised/runs/q1_all_bands_constant \
  --scenes-from models/photometry/self_supervised/runs/q1_all_bands \
  --feature-mode constant
```

Then execute the notebook. Defaults use four CPU threads, 12 training tiles, 4 validation tiles, 12 test tiles, up to four scenes per tile, and ten epochs. `--train-tiles`, `--val-tiles`, `--test-tiles`, `--scenes-per-tile`, and `--epochs` support larger runs. Preparation settings must match a reused cache. Use a new output directory or `--rebuild` when changing preparation. `--scenes-from` reuses exactly the same prepared experiment for controls.

Inputs are the existing Q1 Euclid tiles, Rubin tiles, v11 detection cache, identity-augmentation v11 feature cache, v11 Q1 foundation checkpoint, and compatible anchored astrometry checkpoint. All paths/provenance are recorded. Empty `--astrometry-checkpoint ''` runs a fixed-detection-position ablation; the default applies the frozen anchored head. This head supplies a canonical sky position, not independent per-band centroid corrections.

Each native scene spans roughly 12 arcsec. Every detection within 14 arcsec is considered, covering rotated corners and profile wings. Sources with negligible footprint in a particular band are explicitly unmeasurable there. Groups above 30 detections are skipped rather than truncated. Tile-edge scenes are rejected so the neighbor-search region is covered. Scenes are separated by at least 28 arcsec; RA guards separate training, validation, and test. Validation tiles may overlap a partition edge, but the full scene and its neighbor-search margin must be inside the validation region.

## Products

Each run writes:

- `scenes.pt`: native pixels, variances, masks, WCS Jacobians, positions, fixed PSFs, initial covariances, and detached feature windows (the control references the original file).
- `psf_calibration.json`: fitted PSF widths, calibrator counts, width distributions, and calibration residuals.
- `head.pt`, `history.json`, `summary.json`, `metadata.json`: selected checkpoint, training/validation history, held-out results, and provenance.
- `test_fluxes.csv`, `test_flux_covariances.pt`: fitted native-unit fluxes and conditional source covariance matrices. Rows in the raw CSV exist only where a template has measurable footprint.
- After notebook execution: `test_fluxes_flagged.csv`, with **all ten band rows for every scene source**, including missing/partial coverage and conditioning flags; and three all-band PNG figures.

Use `OK_CONDITIONAL` rows in the flagged catalog for ordinary inspection. Boundary neighbors are nuisance components, not automatically reliable catalog measurements. Negative fluxes remain valid measurements; do not convert them to magnitudes. Objects outside the selected scenes are not a full-field catalog.

## Scientific limits

- The approximate global PSF is calibrated from compact isolated training detections, without flux labels. Selecting the narrow VIS quartile does not certify stars; unresolved galaxies and spatial PSF changes can bias it. This is deliberately not the older DR1 learned ePSF combined with incompatible v11 features.
- One Gaussian shared across bands cannot describe galaxy wings, complex blends, or color gradients. The bounded correction can partly compensate for PSF or astrometry errors rather than recover true morphology.
- Pixels stay on the stored native grids. NISP Q1 MER tiles are already resampled to 0.1 arcsec; they are not independent raw 0.3-arcsec exposures. Their resampling correlations are not modeled.
- Finite positive variance defines usable pixels; Rubin BAD, SAT, and NO_DATA bits are also excluded using the assignments in `io/ingest_tiles.py`. Undetected neighbors are not modeled.
- Flux units are the native image units **per band**. No absolute zero points or microJy/AB conversions are assumed. S/N, not an invented magnitude scale, is used in the all-band change plots.
- The new head is label-free. Its existing foundation/astrometry components have their own training history, which may include the downstream held-out sky. Cached features contain the observed scene pixels; this is full-image likelihood fitting, not an independent held-out-pixel objective.
- Better pixel χ² does not prove better deblended flux. Independent Gaussian pixel-integral tests and noise Monte Carlo validate the renderer/solver; real-data foundation flux accuracy still requires source injection with encoder recomputation and/or independent calibrated multiband references. No injection result is claimed for the learned head.

The next scientific experiment should establish native-image zero points, use spatially calibrated ePSFs and per-band positions, introduce multiple shared morphology components with independent band amplitudes, and compare against a scratch image model. Injected scenes must be encoded again; cached clean-scene features would invalidate that test.
