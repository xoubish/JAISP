"""euclid_phot: forced photometry on Euclid Q1 with Tractor.

The package exposes both the individual pipeline steps (used by the
notebooks) and a one-call driver (used by everything else). See the module
docstrings for the inputs and outputs of each step.

Pipeline order:

    query_mer_catalog        catalog.py
    fetch_cutout             cutouts.py
    extract_catalog_psf      psf.py
    build_tractor_image      images.py
    build_sources_from_mer   models.py        (default)
       or
    detect_blobs + ModelSelector.fit_blob     selection.py (alternative)
    fit_forced_photometry    fit.py
    fit_free_shapes          fit.py
    fit_nisp_forced          nisp.py
    calibrate_result_errors  calibrate.py   (empirical error calibration)
    fit_wise_forced          wise.py

Beyond the single-cutout pipeline: ``run_injection_recovery``
(injection.py) validates the photometry and errors on synthetic sources;
the ``euclid-phot`` command-line script (cli.py) runs the driver.

The package imports Tractor and matplotlib only when first used, so
importing euclid_phot does not require them. The notebooks download their
demo cutouts from IRSA/S3 on first run and cache them locally under
``examples/data/``.
"""
from __future__ import annotations

__version__ = "0.5.0"

# Names that resolve via __getattr__ (PEP 562). Listed here so IDEs and
# `from euclid_phot import *` find them.
_LAZY = {
    # catalog
    "query_mer_catalog": ("catalog", "query_mer_catalog"),
    # cutouts
    "discover_mer_mosaics": ("cutouts", "discover_mer_mosaics"),
    "fetch_cutout": ("cutouts", "fetch_cutout"),
    "trim_catalog_to_cutout": ("cutouts", "trim_catalog_to_cutout"),
    # psf
    "extract_catalog_psf": ("psf", "extract_catalog_psf"),
    "extract_grid_psf": ("psf", "extract_grid_psf"),
    "get_psf_for_source": ("psf", "get_psf_for_source"),
    "psf_summary": ("psf", "psf_summary"),
    # images
    "AstropyWCSAdapter": ("images", "AstropyWCSAdapter"),
    "build_tractor_image": ("images", "build_tractor_image"),
    # models
    "build_sources_from_mer": ("models", "build_sources_from_mer"),
    "build_sources_from_coords": ("models", "build_sources_from_coords"),
    # quality flags
    "flag_sources": ("flags", "flag_sources"),
    "blend_flags": ("flags", "blend_flags"),
    # extinction
    "query_ebv": ("extinction", "query_ebv"),
    "add_extinction_columns": ("extinction", "add_extinction_columns"),
    "EXTINCTION_COEFF": ("config", "EXTINCTION_COEFF"),
    "bright_star_pixel_mask": ("flags", "bright_star_pixel_mask"),
    "starsignal_pixel_mask": ("flags", "starsignal_pixel_mask"),
    "probable_bright_stars": ("flags", "probable_bright_stars"),
    "MER_VIS_BAD_BITS": ("config", "MER_VIS_BAD_BITS"),
    "MER_NISP_BAD_BITS": ("config", "MER_NISP_BAD_BITS"),
    "MER_VIS_FLAG_BITS": ("config", "MER_VIS_FLAG_BITS"),
    # selection
    "SimpleGalaxy": ("selection", "SimpleGalaxy"),
    "ModelSelector": ("selection", "ModelSelector"),
    "detect_blobs": ("selection", "detect_blobs"),
    "assign_mer_to_blobs": ("selection", "assign_mer_to_blobs"),
    "run_model_selection": ("selection", "run_model_selection"),
    "reproduce_figure3": ("selection", "reproduce_figure3"),
    # injection-recovery validation
    "make_truth_table": ("injection", "make_truth_table"),
    "inject_sources": ("injection", "inject_sources"),
    "run_injection_recovery": ("injection", "run_injection_recovery"),
    "summarize_recovery": ("injection", "summarize_recovery"),
    # error calibration
    "draw_empty_positions": ("calibrate", "draw_empty_positions"),
    "measure_error_inflation": ("calibrate", "measure_error_inflation"),
    "calibrate_result_errors": ("calibrate", "calibrate_result_errors"),
    # fit
    "fit_forced_photometry": ("fit", "fit_forced_photometry"),
    "conditional_flux_errors": ("fit", "conditional_flux_errors"),
    "fit_free_shapes": ("fit", "fit_free_shapes"),
    "refine_positions": ("fit", "refine_positions"),
    "measure_residual": ("fit", "measure_residual"),
    # nisp
    "fit_nisp_forced": ("nisp", "fit_nisp_forced"),
    # wise
    "fetch_unwise_cutouts": ("wise", "fetch_unwise_cutouts"),
    "get_wise_psf": ("wise", "get_wise_psf"),
    "fit_wise_forced": ("wise", "fit_wise_forced"),
    "query_unwise_2019": ("wise", "query_unwise_2019"),
    "vega_mag_to_ujy": ("wise", "vega_mag_to_ujy"),
    "select_isolated_sources": ("wise", "select_isolated_sources"),
    # pipeline
    "run_forced_photometry": ("pipeline", "run_forced_photometry"),
    "ForcedPhotometryResult": ("pipeline", "ForcedPhotometryResult"),
    # catalog assembly
    "build_catalog": ("catalog_table", "build_catalog"),
    "flux_to_ab_mag": ("catalog_table", "flux_to_ab_mag"),
    "flux_err_to_mag_err": ("catalog_table", "flux_err_to_mag_err"),
    # submodules
    "config": ("config", None),
    "viz": ("viz", None),
}

__all__ = list(_LAZY.keys()) + ["__version__"]


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module 'euclid_phot' has no attribute {name!r}")
    import importlib
    module_name, attr_name = _LAZY[name]
    module = importlib.import_module(f".{module_name}", __name__)
    if attr_name is None:
        return module
    return getattr(module, attr_name)


def __dir__():
    return sorted(__all__)
