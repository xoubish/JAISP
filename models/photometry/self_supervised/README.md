# Self-supervised multiband photometry

The current implementation fits **Rubin u/g/r/i/z/y and Euclid VIS/Y/J/H** with positive multiscale source profiles, image-likelihood refinement, and an optional foundation morphology prior. It uses no external catalog flux labels.

Open [nb_multiband_flux_recovery.ipynb](nb_multiband_flux_recovery.ipynb) for the executed real-scene comparison and the independent known-flux benchmark. The earlier Gaussian experiment is preserved in [nb_all_band_photometry.ipynb](nb_all_band_photometry.ipynb).

## Completed fresh-test result

On 128 new controlled blends (256 sources), the equal-band average of median absolute fractional flux error falls from **33.61% with the old Gaussian model to 10.76% with the new foundation-assisted mixture model**, a **68.0% reduction**. The stronger image-only mixture reaches **10.93%**. The foundation-minus-image difference is −0.17 percentage points, with a paired blend-bootstrap 95% interval of **−0.65 to +0.45 points**: the foundation-specific gain remains inconclusive.

The improvement over the old model is substantial and present in every band. It must not be attributed entirely to foundation features: improved profile fitting accounts for most of the gain. These percentages describe the specified synthetic benchmark, not independently verified real-survey flux accuracy.

## Tractor VIS-prior comparison

[nb_tractor_comparison.ipynb](nb_tractor_comparison.ipynb) contains the executed comparison with the actual Tractor renderer and VIS profile-selection ladder from [Nima Chartab's Euclid forced-photometry package](https://github.com/NimaChartab/euclid-forced-photometry). An unchanged MIT-licensed snapshot is under `vendor/euclid_forced_photometry`, pinned to `a6bde3d405fda2af980d898fc06790312d8f1c9c`. Tractor itself remains an external GPL dependency. `setup_tractor.sh` installs a separate environment with pinned Tractor/astrometry revisions and numerical dependencies; no packages in the foundation environment are replaced.

On the same 128 controlled blends, the mean of the three NISP per-band median absolute flux errors is **19.79% for the adapted Tractor VIS-profile baseline, 10.47% for the VIS-image mixture prior, and 10.28% for the foundation mixture prior**. Foundation minus Tractor is −9.51 percentage points, with a paired blend-bootstrap 95% interval of −12.71 to −6.32. All 256 sources have all ten measurements, with no failed scenes. Magnitude curves use common positive-flux samples; the companion fractional-flux plots retain signed measurements.

This is an adapted pipeline comparison, **not a demonstration that foundation features beat Tractor on real data**. The benchmark fixes positions, uses geometric source segments rather than SEP membership, retains the shared Gaussian PSFs, and solves final signed fluxes plus constant sky with full blend covariance. Upstream's spatial empirical PSFs and survey zero points are not used. The Tractor profiles stay fixed across bands; the mixture fitter allows band-dependent profiles, and its image-only control performs similarly to the foundation version. The notebook records these differences and has the all-band plots and per-band statistics. Native-unit instrumental magnitude axes remain explicitly labeled.

To reproduce, run `bash models/photometry/self_supervised/setup_tractor.sh python3.11`, then the export/fitting commands in the notebook. Export verifies every regenerated truth flux against the saved benchmark; it does not seed fitting with truth morphology or flux. Run the Tractor bridge tests in its separate environment. Real survey comparisons require matched original science/variance/flag images, spatial PSFs, source lists and photometric calibration for both methods.

## Archive PSF sensitivity

The PSF-sensitivity offset plots now use **true AB magnitude** on x and **measured minus true ΔAB magnitude** on y. Image-header zero points (VIS 24.6, Y 29.8, J 30.0, H 29.9) are recorded in `runs/psf_study/image_calibration.json`, with native cached pixels verified against the original archive images. Run `fetch_ab_calibration` to reproduce that check. Sources brighter than AB 20 are excluded from these running curves.

[nb_psf_sensitivity.ipynb](nb_psf_sensitivity.ipynb) runs the same three photometry methods on 128 new archive-PSF-rendered blends, with original Gaussian, local-FWHM Gaussian, and full GRID-PSF fitting kernels. The linked upstream notebook defaults to normalized **GRID-PSF** stamps, selecting the nearest sample independently at each source and band. We fetched 19 local samples per Euclid band from the matching EDF-S MER tile 102044185, retaining archive paths and sample coordinates.

The noisy images, positions, flux truth and frozen learned priors are identical across conditions. In NISP, changing the original Gaussian to the full stamp changes mean per-band median absolute fractional error from **21.34% to 20.00% for Tractor, 11.01% to 10.42% for the image-prior mixture, and 10.94% to 10.78% for the foundation mixture**. Paired 95% blend-bootstrap intervals for these differences are **[−3.46,+1.27]**, **[−1.16,+0.33]**, and **[−1.00,+0.41] percentage points**, respectively. All include zero. PSF choice alone does not explain the much larger pipeline gap in this local simulation, and foundation features still have no demonstrated extra gain over the image control.

The empirical-PSF renderer respects the delivered pixel response, preserves asymmetric wings and small negative interpolation lobes, normalizes finite stamps once, pads FFT convolution, and never renormalizes cropped source footprints. Tests check these conventions and agreement with Tractor's profile rendering. The original Gaussian code path remains supported. Rubin PSFs stay unchanged in this experiment, with its measurements retained in the all-band tables. Priors are not retrained, and the Gaussian ellipse initializer is held fixed. Sharing the same archive kernels between truth and the grid fitter is deliberate: this is a matched-PSF sensitivity test, not validation of PSF accuracy or real-survey flux accuracy. Finite-stamp normalization does not recover missing far-wing light.

Run `python -m models.photometry.self_supervised.fetch_psf_study` to obtain the small local archive cutouts, then `python -m models.photometry.self_supervised.psf_study --count 128`. Run `tractor_compare` in the isolated environment for each of `runs/psf_study/{gaussian,core_gaussian,grid}`, then execute the notebook. `input_identity_audit.json` records the byte-value equality check on every original input array across the three conditions.

## What changed

The first model's weighted/truncated VIS moments underestimated sizes, and its bounded Gaussian correction learned nearly the same expansion everywhere. A single shared Gaussian also could not represent cores, wings, and color gradients.

The new model:

1. Deweights Gaussian-windowed VIS moments before PSF subtraction; uses these only to initialize ellipticity. Actual sizes come from fitting the images.
2. Represents each source by a positive mixture of seven PSF-convolved profiles spanning 0.025–1.1 arcsec. Every detected neighbor in a scene is fitted jointly. Profile normalization refers to total flux, including light outside the cutout.
3. Learns object-dependent mixture priors from bright sources' image-fitted profiles. Raw VIS pixels, a frozen high-resolution VIS stem, and a frozen multiband bottleneck have separate training-fitted PCA transforms. A regularized predictor combines their information.
4. Adds independent VIS noise at RMS factors two and four during prior training. The original bright image-derived profile remains the target; no catalog photometry is introduced. The image-only control receives the same training examples.
5. Estimates prior precision from validation curve-of-growth errors, including bias, with covariance shrinkage and a 3% curve-of-growth uncertainty floor. Uncertain profile directions receive less weight than reliably predicted ones.
6. Refines VIS and per-band component mixtures against native pixels, allowing color gradients. Uses actual nonnegative least squares for profile components and an unconstrained constant background. The prior does not impose a target total flux.
7. Remeasures signed total fluxes jointly after choosing profiles. This permits negative faint-source flux estimates. Reported covariance is conditional on the estimated profiles/PSFs, not a fully marginalized uncertainty.

The population-prior and raw-VIS-prior controls use exactly the same image fitter. “Image-only” refers to the photometry prior: real-scene comparisons reuse the same foundation detection/astrometry inputs, while simulations use known positions. This does not measure the end-to-end benefit of foundation detection and astrometry. Most gains over the first model can come from improved image modeling. Foundation-specific claims must use the paired comparison against these controls, not just the old Gaussian model.

## Reproduce

Run from the repository root with the existing prepared `runs/q1_all_bands` scenes and `q1_all_bands_constant` baseline checkpoint:

```bash
python -m unittest discover -s models/photometry/self_supervised/tests -v
python -m models.photometry.self_supervised.run_mixture
python -m models.photometry.self_supervised.train_denoising_prior
python -m models.photometry.self_supervised.calibrate_prior
python -m models.photometry.self_supervised.injections \
  --run models/photometry/self_supervised/runs/q1_mixture_calibrated \
  --count 128 --seed 20261201 --weak-vis
```

Then execute `nb_multiband_flux_recovery.ipynb`. The earlier preparation commands are retained below for a fresh setup. All scripts use CPU by default; no GPU is required.

`run_mixture` recomputes the frozen encoder on each actual native scene. Training and simulation inference both use the same scene-cutout context, including high-resolution VIS stem features. Injected scenes are **encoded again**; no clean-image feature cache is reused. The initial training sample contains 51 independent bright source profiles and 17 independent validation profiles; the noise variants produce 153/51 training/validation examples, not additional independent galaxies.

The raw-VIS control was also checked with 40 and 64 PCA dimensions on validation data; these did not beat its selected 24-dimensional version. Model selection and prior precision use validation images, not the simulation flux truth. Earlier development benchmarks remain in `runs/q1_mixture` and `runs/q1_mixture_denoise`; the final precision-calibrated model and fresh test are in `runs/q1_mixture_calibrated`.

## Apply to a scene

```python
from models.photometry.self_supervised.predict import MixturePhotometry

photometer = MixturePhotometry(
    "models/photometry/self_supervised/runs/q1_mixture_calibrated/priors.pt",
    mode="foundation",  # alternatively "image" or "population"
)
measurements = photometer(scene)
vis_flux = measurements["euclid_VIS"]["flux"]
```

A scene has `sky[N,2]` and a `bands` mapping containing all ten bands. Each band supplies Torch tensors `image[H,W]`, `variance[H,W]`, boolean `mask[H,W]`, native pixel `positions[N,2]`, tangent-arcsec-to-pixel `sky_to_pixel[2,2]`, and scalar `psf_sigma` in native pixels. The existing scene preparer builds this structure with corrected astrometry and WCS information. Inference requires identical source lists across bands, but permits different image shapes and grids.

Outputs include signed native-unit fluxes, conditional errors/covariance, model footprints, source-index mappings, reconstructed pixels, and per-source component weights. Missing/partial footprints and ill-conditioned fits must be respected; outer neighbors are nuisance components, not automatically useful catalog measurements. `fluxes.csv` contains flags and all ten band rows for each scene source.

## Known-flux evaluation

The final benchmark contains 128 new two-source blends (256 sources, 2,560 source/band measurements per model) at separations of 0.4–1.4 arcsec. Half have VIS S/N reduced by three; half have correlated noise. Profiles are exponential core/disk galaxies with changing band fractions, rendered on a fine grid and convolved using SciPy independently of the Gaussian fitting dictionary. All models receive the same pixels, source positions, and assumed PSFs.

The evaluator reports fractional flux bias, NMAD, median absolute error, and running 16th/50th/84th percentiles versus true flux, S/N, and separation. A paired bootstrap resamples whole blends, preserving source and band correlations. It separates improvement over the old model from incremental improvement attributable to foundation information. Confidence intervals crossing zero are inconclusive.

These are controlled synthetic scenes at Q1 noise/PSF scales, **not additive source injections into real survey backgrounds**. Positions and PSFs are known; missing detections, position errors, spatial PSF mismatch, and irregular real-galaxy morphologies are not tested. The 47 real Q1 scenes are an exploratory reconstruction comparison; their flux accuracy still needs independently calibrated references or realistic injections.

Real flux units remain native per band. No microJy/AB zero points are assumed. NISP tiles are already resampled; reported variance does not include a full pixel-correlation model. The pre-existing foundation/astrometry may have seen the real held-out sky. The approximate global circular PSF, small prior-training sample, and conditional error estimates remain limitations.

## Earlier Gaussian pilot (preserved)


See [the original Gaussian pilot notes](README_gaussian_pilot.md) for initial scene preparation, the old controls, and their recorded results.
