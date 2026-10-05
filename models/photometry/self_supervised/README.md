# Self-supervised multiband photometry

The current implementation fits **Rubin u/g/r/i/z/y and Euclid VIS/Y/J/H** with positive multiscale source profiles, image-likelihood refinement, and an optional foundation morphology prior. It uses no external catalog flux labels.

Open [nb_multiband_flux_recovery.ipynb](nb_multiband_flux_recovery.ipynb) for the executed real-scene comparison and the independent known-flux benchmark. The earlier Gaussian experiment is preserved in [nb_all_band_photometry.ipynb](nb_all_band_photometry.ipynb).

## Empirical real-galaxy injection pilot

[nb_empirical_injection_pilot.ipynb](nb_empirical_injection_pilot.ipynb) is the executed independent-renderer experiment. A frozen library of 100 candidate galaxies (95 pass data-only identity, counterpart and centering checks) supplies native-band nonparametric morphologies and jointly dimmed empirical colors. Complete, disjoint regions supply real coadd backgrounds. Independent detections in **every band** protect blank-sky selection from sources missing in VIS catalogs; secure isophotes are expanded before source-core masking. The initial catalog-only sky pilot remains in `runs/empirical_injection_pilot` as a diagnostic of this contamination problem. Use **`runs/empirical_injection_pilot_v2`** for the revised comparison.

The pilot renders 1000 scenes / 1750 sources with GalSim, local archive Euclid pixel PSFs and approximate Rubin Gaussian PSFs. It preserves clumps and supported color gradients, includes weak-band morphology flags, rotations, slight PSF broadening, separations 0.35/0.7/1.4 arcsec, and neighbors with VIS ratios 1/3/10 and their own empirical SEDs. Intrinsic and PSF mass normalize before rendering; missing masked/cropped flux does not normalize. Foundation/image priors and Tractor see identical saved data, masks, variance and positions. Fluxes remain signed, and all failures are retained. Donor- and sky-field-grouped bootstrap intervals accompany paired metrics; an exact-template oracle diagnoses sky and noise limitations.

The revised run uses 310 sky patches from seven regions; all 1000 scenes succeed for all competitors and input hashes agree. On 5650 qualified central source-band measurements with nominal true S/N 1–10, foundation/image/Tractor mean absolute errors are **1.691 / 1.706 / 2.673** in exact-template noise units. Foundation reduces this error by **36.7%** versus the adapted Tractor baseline (paired difference −0.982, joint donor/background bootstrap 95% [−1.281, −0.701]), but its difference versus image-only is not established (−0.0149, interval [−0.0617, +0.0281]). Median fractional biases are **−17.2% / −12.2% / −35.7%**, so faint-end morphology bias still needs work. GalSim rendered mass closes within 0.003%; the oracle is close to unbiased. This supports a fitter-level gain over this Tractor configuration, not an additional foundation-feature gain or a MER claim.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m models.photometry.self_supervised.empirical all --count 1000 --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 models/photometry/self_supervised/runs/tractor_env/bin/python models/photometry/self_supervised/empirical_tractor.py --run models/photometry/self_supervised/runs/empirical_injection_pilot_v2 --workers 16
MPLCONFIGDIR=/tmp/jaisp-mpl IPYTHONDIR=/tmp/jaisp-ipython OPENBLAS_NUM_THREADS=1 python -m models.photometry.self_supervised.empirical_report --notebook
python -m unittest models.photometry.self_supervised.tests.test_empirical -v
```

These are **forced-position, sky-noise tests**, not end-to-end detection or MER tests. No calibrated effective gain exists in the cached mosaic headers: the default adds no source shot noise, which matters for bright neighbors. `--gain-json` accepts a calibrated native-unit effective gain for each band and enables an explicit post-coadd Poisson approximation. The nonparametric donor reconstruction is approximate; weak bands use flagged VIS-shape fallbacks and positive amplitude posterior estimates. The library is held out from photometry-prior training; independence from foundation pretraining is not established. Catalog fluxes are never truth or fitting seeds. Models remain frozen, and this pilot must not be reused as a final test after tuning a head on its results.

## Flux-path audit and stellar profiles

[nb_flux_path_audit.ipynb](nb_flux_path_audit.ipynb) records the normalization and centering audit. Prepared science pixels match 30 source-array cutouts exactly; native WCS positions agree within 0.000007 pixel. All ten production v11 stems use `asinh50`. Compression is confined to feature extraction, and the final flux fit uses original native-unit science pixels. The compression/inverse round trip, signed flux recovery, masked cores, cropped wings, column preconditioning, source ordering, unit scaling and repeat calls pass their checks. Repeated A→changed-input→A calls return identical features and fluxes on CPU.

**The unmodified mixture is inappropriate for stars.** Its minimum intrinsic Gaussian sigma is 0.025 arcsec, rather than zero. Independently rendered noiseless stellar tests give +3.2% VIS flux error with the calibrated Gaussian PSF and +3.5% with one archive GRID stamp even without a learned prior. At peak pixel S/N 5, the foundation galaxy-profile prior gives about +28%/+31% VIS error. These are deterministic diagnostics with supplied variance, not ensemble faint-end bias estimates or measurements of real stars.

Mixture photometers now accept a supplied boolean `scene["point_sources"]` mask, one flag per source. Flagged stars have zero intrinsic size and use PSF-only profiles in every band; galaxy profiles retain their seven-component dictionary. This removes the stellar bias to below 0.001% in all 14 audit scenes, including blended stars and archive PSFs. It requires independent stellar classifications; there is no automatic classifier, and this restricted test is not an end-to-end performance claim. Existing checkpoints and historical benchmark files are retained.

```python
# Supply independently established stellar classes in the scene's source order.
scene["point_sources"] = np.asarray(stellar_flags, dtype=bool)
measurements = photometer(scene)  # MixturePhotometry or BandPriorPhotometry
```

Two additional diagnostics need follow-up before a realistic benchmark. The foundation fuses bands by resizing arrays, without WCS registration; central Rubin/VIS offsets from independently rounded scene crops have median 0.081 arcsec and maximum 0.164 arcsec in the prepared scenes. These are feature-alignment offsets, not errors in the native flux-template centers. The separate experimental amortised-scarlet renderer conserves total mass to approximately 0.1% in the tested cases but adds compact-profile smoothing: flux errors range from +0.3% to +5.3% against independently pixel-integrated Gaussians. It integrates intrinsic morphology before convolving a PSF stamp that already includes pixel response. Same-grid point-sampling controls isolate that extra smoothing; using a single sample is not a general repair on rotated/coarser grids. Its renderer and checkpoint convention require a deliberate revision before comparison.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m models.photometry.self_supervised.audit_flux_path
python -m unittest discover -s models/photometry/self_supervised/tests -v
```

Diagnostics are saved separately in `runs/flux_path_audit`, with per-source measurements, source-pixel identity and compression checks, encoder-alignment offsets, and renderer controls. There is no new claim of beating Tractor or MER. A future realistic test needs independent rendering, clumpy morphology, observed joint SED distributions, local empirical PSFs and PSF mismatch, real backgrounds/correlated noise, blends and detection/centroid errors, with models frozen before testing. MER must be run on the injected images to be a truth-scored simulation competitor; its catalog alone is a real-data reference.

## Expanded band-specific prior experiment

[nb_expanded_multiband_prior.ipynb](nb_expanded_multiband_prior.ipynb) reports a separate experiment in `runs/q1_multiband_expanded`. It prepares 300 training scenes and all 142 available nonoverlapping validation scenes within the original guarded sky partitions. There are 580 independent reliable VIS training profiles, versus 51 in the earlier prior. Whole validation scenes separate regression selection from precision calibration; noise augmentation does not increase the independent-source counts.

The head predicts seven positive profile weights independently for every band. Each band's teacher is a bright, jointly image-fitted profile, with no catalog flux labels. Inputs include 17×17 raw S/N and coverage windows in all ten bands, a 9×9 frozen VIS-stem window, and a 5×5 multiband bottleneck window. Controls use population profiles, VIS pixels, or raw ten-band pixels. The ten-band pixel and foundation heads have the same number of latent regression features. Every control uses the same native-pixel morphology fitter, background treatment and signed final flux solver. The earlier VIS-first foundation prior is also evaluated on the same fresh known-flux blends.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m models.photometry.self_supervised.try_multiband_prior prepare
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m models.photometry.self_supervised.try_multiband_prior train
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m models.photometry.self_supervised.try_multiband_prior benchmark
```

Preparation is resumable. Scene-count arguments are upper limits: overlapping footprints and guarded sky coverage can yield fewer scenes, with actual teacher coverage checked before training. The benchmark uses 256 new blends, including weak VIS and correlated noise, after all priors are frozen. `results.json`, per-source signed fluxes, paired whole-blend bootstrap intervals, and PNG/PDF plots are saved in the experiment directory. This remains a controlled simulation with known positions and approximate Gaussian PSFs, not a measurement of real-survey flux accuracy. u and y have especially small independent teacher samples; profile and PSF uncertainty are not included in conditional flux errors.

The completed fresh benchmark **does not justify replacing the existing foundation head**. On 256 blends (512 sources), equal-band average median absolute fractional flux error is **10.39% for the existing foundation prior, 12.21% for expanded VIS pixels, 12.05% for expanded ten-band pixels, and 11.78% for the expanded foundation head**. The new head is worse than the existing head by **+1.39 percentage points (paired 95% interval +0.93 to +1.95)**. Its improvement over ten-band pixels is **−0.27 points (−0.53 to +0.16)**, so an extra benefit from foundation features is inconclusive. It does improve over the VIS-only control by **−0.42 points (−0.73 to −0.09)**.

The expanded head predicts validation teacher profiles better, but this does not transfer into better simulated flux recovery. This experiment changes both training coverage and the VIS-first coupling, so it does not identify which change causes the regression. The existing `MixturePhotometry` checkpoint remains the recommended baseline; `BandPriorPhotometry` exposes the experimental controls for further study. No benchmark truth was used to select or refit these heads.

```python
from models.photometry.self_supervised.predict import BandPriorPhotometry
photometer = BandPriorPhotometry(
    "models/photometry/self_supervised/runs/q1_multiband_expanded/priors.pt",
    mode="foundation",  # also population, vis_image, all_images
)
measurements = photometer(scene)
```

## Detection-head catalog comparison on real tiles

[nb_detection_catalog_comparison.ipynb](nb_detection_catalog_comparison.ipynb) measures **the detection head's own source list** on 28 non-overlapping real Q1 tiles (19 in the patch-25 holdout, 9 in the prior's RA test partition; 6558 detections, 5339 photometered) with three methods and compares them with position-matched MER fluxes. Sources come from the production v11 CenterNet export, every position is the frozen anchored-astrometry VIS-canonical centroid projected through each band's WCS, and the pipeline in `detcat/` fetches the archive science/RMS/flag cutouts and GRID PSFs so that our calibrated mixture photometer (foundation and image priors, all ten bands) and the upstream Tractor VIS profile ladder with positions frozen at the same centroids see identical pixels, masks and PSFs. The archive cutouts are verified to be pixel-identical to the local tiles.

On the 4732 MER-matched sources measured by all methods, NMAD scatter against MER is **0.125 / 0.125 mag for foundation / Tractor in VIS** and **0.151 / 0.185 (Y), 0.139 / 0.165 (J), 0.137 / 0.156 (H)**; the NISP differences have tile-bootstrap 95% intervals excluding zero, the VIS difference does not. The foundation prior beats the image-prior control by 0.012 mag in VIS and by 0.004–0.006 mag in J and H, with intervals excluding zero. Both JAISP priors measure systematically more flux than MER (medians −0.07 VIS, −0.03 to −0.04 NISP, growing faint-ward and present for isolated galaxies) while Tractor shows no offset; the free scene background and positive-only profile selection are the suspected causes, and this needs an injection test before it is called a bias. Detection completeness against MER is 0.90 at VIS 24, purity 0.97 to VIS 25. Rubin ugrizy fluxes are in the catalog with no external reference (nanojansky pixel units assumed).

```bash
python -m models.photometry.self_supervised.detcat.select_tiles
python -m models.photometry.self_supervised.detcat.sources
python -m models.photometry.self_supervised.detcat.fetch
python -m models.photometry.self_supervised.detcat.prepare
python -m models.photometry.self_supervised.detcat.mer
OMP_NUM_THREADS=1 python -m models.photometry.self_supervised.detcat.photometer --workers 40
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 runs/tractor_env/bin/python -m models.photometry.self_supervised.detcat.tractor_run
python -m models.photometry.self_supervised.detcat.report
```

Products are in `runs/detection_catalog`: `catalog_long.csv`, `catalog_wide.csv`, `summary.csv`, `paired_bootstrap.csv`, `detection_statistics.csv`, the figures, and per-region inputs, models and MER matches. The Tractor environment must be run with single-threaded BLAS or it starves the other steps. Offline tests are in `tests/test_detcat.py`.

## Amortised scarlet head (experimental)

[amortised_scarlet.py](amortised_scarlet.py) is a learned scarlet-style photometer on the frozen foundation encoder: positive source morphologies on a 0.1″ tangent-plane grid, explicit per-source PSFs (GRID stamps for Euclid, calibrated Gaussians for Rubin), and the same signed linear flux solve as the mixture photometer. The head reads the fused bottleneck window, the VIS stem window and the raw ten-band pixels at each source and outputs two centred monotone elliptical radial profiles with a per-band mixing weight and a bounded perturbation map; two unrolled steps feed the exact residual gradient back into the head. It trains on 43 real tiles (`runs/amortised_scarlet/training_tiles`, prepared with the `detcat` pipeline) with the mean over bands of log χ²/dof as the only signal and a fixed robust background. Checkpoints, histories and logs are in `runs/amortised_scarlet/{fixedbg,fixedbg_nostep,lr1e-3,v1_freeform}`.

```bash
JAISP_DETCAT_OUT=models/photometry/self_supervised/runs/amortised_scarlet/training_tiles python -m models.photometry.self_supervised.detcat.select_tiles --training 48 --exclude models/photometry/self_supervised/runs/detection_catalog/tiles.json
# then detcat.sources / fetch / prepare with the same JAISP_DETCAT_OUT
python -m models.photometry.self_supervised.amortised_scarlet --epochs 4 --lr 1e-3 --fixed-background --output models/photometry/self_supervised/runs/amortised_scarlet/fixedbg
python -m models.photometry.self_supervised.injections --run models/photometry/self_supervised/runs/q1_mixture_calibrated --count 128 --seed 20261201 --weak-vis --scarlet <checkpoint> --suffix _scarlet
python -m models.photometry.self_supervised.detcat.photometer --scarlet <checkpoint> --device cuda:0 --workers 3
```

Result of the first iteration: on the 128 known-flux blends the fixed-background head reaches **0.138** mean median |fractional error| (mixture 0.108; its epoch-1 checkpoint 0.109), better than the mixture below S/N 20 and worse above S/N 40 and for pairs closer than 0.7″. On the 28 real tiles it is worse than the mixture against MER in every band (NMAD VIS 0.242 vs 0.125, NISP 0.17–0.21 vs 0.14–0.15) and worse than Tractor in NISP; it matches the mixture at the bright end in NISP but under-fits bright galaxies in VIS. The epoch-1 checkpoint removes most of the offset on real tiles (VIS 0.000, NISP −0.03 mag) and lowers the NISP scatter to 0.167–0.189, still above the mixture, with the VIS scatter unchanged at 0.240. Three ablations determined the design: free-form morphologies let blended sources trade light freely (0.280 on injections despite matching the mixture's validation χ²), a free constant background inflates fluxes by about 10 % with any slightly too-broad template, and removing the unrolled refinement steps gives 0.227. Validation χ² improved monotonically while injection accuracy degraded after epoch 1, so χ² alone must not be used to select such a head. The notebook section "Amortised scarlet head" and `tests/test_amortised_scarlet.py` document the renderer conventions (flux conservation on rotated and coarser grids, placement, PSF convolution).

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
