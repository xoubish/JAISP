"""JAISP photometry package.

Each submodule is imported in its own try/except so a broken legacy module
(e.g. the deprecated ``psf_field_pipeline``) does not take the whole package
down. Active heads — ``FoundationScarletPhotometryHead``, the rendered-stamp
head — must always import.
"""

import warnings as _warnings


def _try_import(callback):
    try:
        callback()
    except ImportError as exc:
        _warnings.warn(
            f"photometry: skipping a submodule that failed to import: {exc}",
            stacklevel=2,
        )


_exports = {}


def _add(*names_and_values):
    for name, value in names_and_values:
        globals()[name] = value
        _exports[name] = value


def _legacy_psf_net():
    from .psf_net import PSFNet, BAND_ORDER  # noqa: WPS433
    _add(("PSFNet", PSFNet), ("BAND_ORDER", BAND_ORDER))


def _stamp_extractor():
    from .stamp_extractor import extract_stamps, estimate_local_background  # noqa: WPS433
    _add(("extract_stamps", extract_stamps), ("estimate_local_background", estimate_local_background))


def _forced_photometry():
    from .forced_photometry import matched_filter, aperture_flux, snr  # noqa: WPS433
    _add(("matched_filter", matched_filter), ("aperture_flux", aperture_flux), ("snr", snr))


def _pipeline():
    from .pipeline import TilePhotometryPipeline  # noqa: WPS433
    _add(("TilePhotometryPipeline", TilePhotometryPipeline))


def _psf_field_pipeline():
    # Deprecated: depends on the old PSFField class (moved to
    # ``older_architectures/psf``). Kept importable when present.
    from .psf_field_pipeline import (  # noqa: WPS433
        PSFFieldPhotometryPipeline,
        load_psf_field_checkpoint,
        normalise_band_name,
    )
    _add(
        ("PSFFieldPhotometryPipeline", PSFFieldPhotometryPipeline),
        ("load_psf_field_checkpoint", load_psf_field_checkpoint),
        ("normalise_band_name", normalise_band_name),
    )


def _scarlet_like():
    from .scarlet_like import (  # noqa: WPS433
        ScarletLikePhotometryHead,
        SceneFitResult,
        build_neighbor_groups,
        convolve_morphology_with_psf,
        fit_scarlet_like_scene,
        fit_scarlet_like_tile,
        make_positive_morphology_templates,
        normalise_templates,
        place_templates_in_scene,
    )
    _add(
        ("ScarletLikePhotometryHead", ScarletLikePhotometryHead),
        ("SceneFitResult", SceneFitResult),
        ("build_neighbor_groups", build_neighbor_groups),
        ("convolve_morphology_with_psf", convolve_morphology_with_psf),
        ("fit_scarlet_like_scene", fit_scarlet_like_scene),
        ("fit_scarlet_like_tile", fit_scarlet_like_tile),
        ("make_positive_morphology_templates", make_positive_morphology_templates),
        ("normalise_templates", normalise_templates),
        ("place_templates_in_scene", place_templates_in_scene),
    )


def _foundation_head():
    from .foundation_head import (  # noqa: WPS433
        FoundationScarletPhotometryHead,
        LearnedSceneResult,
        photometry_head_loss,
        render_learned_photometry_tile,
    )
    _add(
        ("FoundationScarletPhotometryHead", FoundationScarletPhotometryHead),
        ("LearnedSceneResult", LearnedSceneResult),
        ("photometry_head_loss", photometry_head_loss),
        ("render_learned_photometry_tile", render_learned_photometry_tile),
    )


def _rendered_stamp_head():
    from .rendered_stamp_head import RenderedStampHead  # noqa: WPS433
    _add(("RenderedStampHead", RenderedStampHead))


for _step in (
    _legacy_psf_net,
    _stamp_extractor,
    _forced_photometry,
    _pipeline,
    _psf_field_pipeline,
    _scarlet_like,
    _foundation_head,
    _rendered_stamp_head,
):
    _try_import(_step)


__all__ = list(_exports.keys())
