# Euclid forced photometry

Measure fluxes on Euclid Q1 VIS and NISP images using Tractor source models,
with optional unWISE W1/W2 photometry. VIS constrains the source profiles;
the lower-resolution fits keep those profiles fixed and solve for flux.

This is an educational analysis package. It provides explicit diagnostics
and limited validation tests; users must assess the PSF, residuals, source
list and uncertainty assumptions for their own science sample.

## Notebooks

- [00: Modeling a galaxy with the Tractor](notebooks/00_how_tractor_works.ipynb): one galaxy,
  with image construction, model parameters and optimization written out.
- [01: Multi-band forced photometry on Euclid Q1 data](notebooks/01_multiband_forced_photometry.ipynb):
  a 200-arcsec field, source models on VIS, forced fluxes in Y, J and H,
  residuals and catalog checks.
- [02: Forced photometry on unWISE W1 and W2](notebooks/02_wise_forced_photometry.ipynb):
  the same field on unWISE W1 and W2, with residuals and a catalog cross-check.
- [03: Multi-band photometry in one call](notebooks/03_one_call.ipynb): a compact 50-arcsec
  Euclid run and the resulting catalog. WISE is optional.
- [04: Photometry at supplied coordinates](notebooks/04_photometry_at_coordinates.ipynb):
  photometry at user positions without a MER source list.
- [Supporting: Model selection](notebooks/supporting_notebooks/model_selection.ipynb):
  the profile trials for one pair of neighbouring sources.
- [Supporting: Injection and recovery](notebooks/supporting_notebooks/injection_recovery.ipynb):
  matching-PSF point-source checks on real backgrounds.

Start with 00 to learn the fitting model, 01 to inspect the full workflow,
or 03 for a short working example.

## Install and run

From the repository directory:

```bash
./scripts/install.sh
# For optional WISE support, use: ./scripts/install.sh --wise
# If the script created .venv:
source .venv/bin/activate
jupyter lab notebooks/
```

The install script uses an active virtualenv or conda environment, or creates
`.venv`. It installs this package plus the required Tractor and astrometry.net
components. WISE additionally requires `unwise_psf` and its dependencies.
Use a Jupyter kernel from that environment. If the script created `.venv`,
activate it with `source .venv/bin/activate` before starting Jupyter.

The notebooks locate the repository from their working directory and work
from either `notebooks/` or `notebooks/supporting_notebooks/`. Inputs are
fetched from public archives on first use; image data are not bundled with
this source distribution. The notebooks cache them under
`examples/native_data/`. Set `EUCLID_PHOT_DATA_DIR` to use another cache.
Network access is required when an input is absent.

## One-call API

```python
from pathlib import Path
import euclid_phot as ep

result = ep.run_forced_photometry(
    269.48, 67.30, 50.0,  # RA, Dec in degrees; box side in arcsec
    prior={"band": "VIS", "objects": "mer", "model_selection": "tree"},
    target_bands={"euclid": ("Y", "J", "H"), "wise": ()},
    psf_product="grid", persource_psf=True,
    mask_bright_stars=True, with_flag=True, calibrate_errors=True,
    data_dir=Path("examples/native_data"), n_workers=4,
)
catalog = result.to_table()
catalog.write("photometry.csv", overwrite=True)
```

`model_selection="prior"` starts from MER classifications and shapes;
`"tree"` compares profile classes on VIS and also refines positions.
The tree adapts [The Farmer](https://ui.adsabs.harvard.edu/abs/2023ApJS..269...20W) (Weaver et al. 2023), with different
selection details and an additional Sérsic trial. Its shapes are already
fitted, so `free_shapes` does not add another shape fit on that path.

`psf_product="grid"` uses PSF samples on a regular grid; `"catalog"` uses
samples at MER catalog positions. With `persource_psf=True`, each source
uses its nearest sample inside a joint fit. Images retain their delivered
MER pixels. A cutout spanning multiple MER tiles needs separate tile fits.

The package also supports supplied coordinates and selected profile classes;
see [notebook 04](notebooks/04_photometry_at_coordinates.ipynb) and
`help(ep.run_forced_photometry)`. With the tree, unlisted neighbours detected
in the same blob are added to the model; with an explicit profile class the
supplied list must include them. The command-line entry point is
`euclid-phot run --help`.

## Interpreting results

Flux densities are in microJansky. AB magnitudes are given for positive
fluxes; a flux the fit could not constrain is NaN. The output metadata
records PSF conventions, uncertainty definitions and error-scale factors.

`reliable` is False when the source is near a bright star, on masked pixels,
or at the cutout edge. `blended` marks a close neighbour.

Euclid errors are conditional template errors, optionally scaled using
empty-position point-source fits. WISE uses a residual-based scale factor.
Neither treatment captures all blending, morphology, PSF or sky uncertainty.
The reported five-times-median-error magnitude is an error summary, not a
measured detection-completeness limit.

Shared morphology can fail for color gradients or infrared-only emission.
Structured residuals and sensitivity to neighbouring models require inspection.
The example retains a few-percent NISP/MER difference; it should not be removed
by tuning a PSF to force catalog agreement. WISE blends also remain sensitive
to PSF shape and source-list completeness.

The [Schlafly et al. (2019) unWISE catalog](https://ui.adsabs.harvard.edu/abs/2019ApJS..240...30S) uses
a different PSF normalization. Notebook 02 transfers our fitted amplitudes
to that convention for its comparison only; exported fluxes retain their
fitting convention. Agreement with that catalog is a cross-check, not an
absolute calibration or proof of individual blended flux accuracy.

Matching-PSF injections test rendering and recovery consistency. They do not
validate the real PSF, galaxy morphology, blind detection completeness, or
all uncertainty contributions. The injection notebook states the tested scope.

## References

The image modelling uses [The Tractor](https://github.com/dstndstn/tractor).
Euclid reference flux definitions are documented in the
[MER photometry cookbook](https://euclid.esac.esa.int/dr/q1/dpdd/merdpd/merphotometrycookbook.html).
Please cite the relevant survey and method papers when using their data or methods.
