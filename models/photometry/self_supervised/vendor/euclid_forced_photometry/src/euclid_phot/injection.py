"""Synthetic-source injection-recovery validation.

Inject sources of known flux into the real pixels, run the unmodified
pipeline, and check that the recovered fluxes are unbiased and that
(recovered - true) / reported error scatters like N(0, 1).

    make_truth_table        truth rows in the MER schema at source-free positions
    inject_sources          add Tractor-rendered sources to a real cutout
    run_injection_recovery  inject, run the pipeline, join with the truth
    summarize_recovery      bias / scatter / pull / completeness per flux bin

By default, injection and recovery both use the nearest GRID-PSF stamp.
Different products can be requested explicitly to study PSF mismatch.
No source photon noise is added: these experiments test recovery of a
deterministic added signal on the existing sky, not a complete noise model.
"""
from __future__ import annotations

import warnings

import numpy as np

# MER catalog flux column per Euclid band (matches pipeline._MER_FLUX_COL).
_FLUX_COL = {"VIS": "flux_vis_sersic", "Y": "flux_y_sersic",
             "J": "flux_j_sersic", "H": "flux_h_sersic"}


def make_truth_table(cutout, mer_cat=None, *,
                     n_inject: int = 30,
                     flux_range_ujy: tuple = (2.0, 100.0),
                     model: str = "point",
                     re_arcsec: float = 0.5,
                     axis_ratio: float = 0.7,
                     position_angle_deg: float = 30.0,
                     sersic_n: float = 1.5,
                     avoid_radius_arcsec: float = 2.0,
                     bands: tuple = ("VIS", "Y", "J", "H"),
                     rng=None):
    """Truth rows in the MER catalog schema, at source-free positions.

    Positions come from :func:`euclid_phot.calibrate.draw_empty_positions`
    (so injections never sit on a known source); fluxes are log-uniform in
    ``flux_range_ujy``. Pass a tuple for a flat-SED source or a
    ``{band: (lo, hi)}`` dict to draw each band independently (VIS is much
    deeper than NISP, so one flat range cannot test both at comparable S/N).

    ``object_id`` is negative (-1, -2, ...) so injected rows never collide
    with real MER identifiers.

    ``model='point'`` sets ``point_like_prob = 1.0`` (built as PointSource);
    ``model='sersic'`` writes the MER Sersic shape columns instead.
    """
    from astropy.table import Table

    from .calibrate import draw_empty_positions

    if model not in ("point", "sersic"):
        raise ValueError(f"model must be 'point' or 'sersic', got {model!r}")
    rng = np.random.default_rng(rng)
    ra, dec = draw_empty_positions(
        cutout, mer_cat, n_positions=n_inject,
        avoid_radius_arcsec=avoid_radius_arcsec, rng=rng)
    n = len(ra)
    if n < n_inject:
        warnings.warn(
            f"make_truth_table: placed {n}/{n_inject} injections (crowded "
            "or small field).", stacklevel=2)

    def _draw(lo, hi):
        return np.exp(rng.uniform(np.log(float(lo)), np.log(float(hi)),
                                  size=n))

    if isinstance(flux_range_ujy, dict):
        flux_by_band = {b: _draw(*flux_range_ujy[b]) for b in bands}
    else:
        flux = _draw(*flux_range_ujy)
        flux_by_band = {b: flux for b in bands}

    from astropy.table import MaskedColumn

    def _masked():
        return MaskedColumn(np.zeros(n), mask=np.ones(n, dtype=bool))

    tab = Table()
    tab["object_id"] = -(np.arange(n, dtype=np.int64) + 1)
    tab["ra"] = ra
    tab["dec"] = dec
    for b in bands:
        tab[_FLUX_COL[b]] = flux_by_band[b]
    # build_sources_from_mer reads every morphology column up front; fully
    # masked columns are skipped by its masked-aware reader.
    if model == "point":
        tab["point_like_prob"] = np.ones(n)
        for col in ("sersic_sersic_vis_radius", "sersic_sersic_vis_axis_ratio",
                    "sersic_angle", "sersic_sersic_vis_index"):
            tab[col] = _masked()
    else:
        tab["point_like_prob"] = np.zeros(n)
        tab["sersic_sersic_vis_radius"] = np.full(n, float(re_arcsec))
        tab["sersic_sersic_vis_axis_ratio"] = np.full(n, float(axis_ratio))
        tab["sersic_angle"] = np.full(n, float(position_angle_deg))
        tab["sersic_sersic_vis_index"] = np.full(n, float(sersic_n))
    for col in ("semimajor_axis", "ellipticity", "position_angle"):
        tab[col] = _masked()
    tab.meta["injected"] = True
    tab.meta["model"] = model
    return tab


def inject_sources(cutout, psf, truth, *,
                   band: str = "VIS",
                   flux_col: str | None = None,
                   mag_zero: float | None = None,
                   psf_fwhm_arcsec: float = 0.16):
    """Return a copy of ``cutout`` with the truth sources rendered into it.

    Parameters
    ----------
    cutout : Cutout
        The real cutout to inject into (its pixels are not modified; a
        shallow copy with a new ``data`` array is returned).
    psf : ndarray or psf_data dict
        A single PSF stamp (used for every injection), or a psf_data dict
        (from ``extract_grid_psf`` / ``extract_catalog_psf``), in which case
        each injection is rendered with its nearest stamp, matching what a
        real source at that position would look like.
    truth : Table
        Output of :func:`make_truth_table`.
    flux_col : str, optional
        Truth flux column; defaults to the band's MER column.
    psf_fwhm_arcsec : float
        Passed to ``build_sources_from_mer`` for the point/galaxy decision
        (irrelevant for ``model='point'`` truths).
    """
    import copy as _copy

    from tractor import Tractor

    from .images import build_tractor_image
    from .models import build_sources_from_mer
    from .psf import psf_summary

    flux_col = flux_col or _FLUX_COL[band]
    if flux_col not in truth.colnames:
        raise ValueError(f"truth table has no {flux_col!r} column")

    per_source = isinstance(psf, dict)
    if per_source:
        avg_stamp, _ = psf_summary(psf)
    else:
        avg_stamp = np.asarray(psf)

    sources = build_sources_from_mer(
        truth, band=band, flux_col=flux_col,
        psf_fwhm_arcsec=psf_fwhm_arcsec,
        pixel_scale_arcsec=cutout.pixel_scale_arcsec)

    model_sum = np.zeros_like(np.asarray(cutout.data, dtype=np.float32))
    if per_source:
        # Group injections by nearest stamp; one render per group.
        nearest = np.empty(len(sources), dtype=int)
        for i, src in enumerate(sources):
            p = src.getPosition()
            cosd = np.cos(np.radians(float(p.dec)))
            # Wrap the RA difference into [-180, 180].
            d_ra = ((np.asarray(psf["ra"]) - float(p.ra) + 540.0) % 360.0) - 180.0
            d = ((d_ra * cosd) ** 2
                 + (psf["dec"] - float(p.dec)) ** 2)
            nearest[i] = int(np.argmin(d))
        for pi in np.unique(nearest):
            stamp = psf["stamps"][int(pi)].astype(np.float32).copy()
            stamp /= stamp.sum()
            tim = build_tractor_image(cutout, stamp, mag_zero=mag_zero)
            grp = [sources[j] for j in np.where(nearest == pi)[0]]
            model_sum += Tractor([tim], grp).getModelImage(0)
    else:
        tim = build_tractor_image(cutout, avg_stamp, mag_zero=mag_zero)
        model_sum = Tractor([tim], sources).getModelImage(0)

    out = _copy.copy(cutout)
    out.data = (np.asarray(cutout.data, dtype=np.float32) + model_sum)
    return out


def run_injection_recovery(*,
                           data_dir,
                           target_ra: float,
                           target_dec: float,
                           cutout_size_arcsec: float = 50.0,
                           bands: tuple = ("VIS", "Y", "J", "H"),
                           n_inject: int = 30,
                           flux_range_ujy: tuple = (2.0, 100.0),
                           model: str = "point",
                           rng=0,
                           verbose: bool = False,
                           return_result: bool = False,
                           injection_psf_product: str = "grid",
                           **run_kwargs):
    """Run the full injection-recovery test on a (cached) field.

    Loads the field's cutouts and MER catalog from ``data_dir`` (fetching
    live on a cache miss), builds a truth table at source-free positions,
    injects into every requested band, and runs the unmodified
    :func:`euclid_phot.pipeline.run_forced_photometry` on the modified
    pixels via its ``cutouts=`` / ``mer_catalog=`` overrides.

    Returns ``(catalog, truth)`` where ``catalog`` is the standard output
    table (injected rows have ``object_id < 0``) and ``truth`` carries the
    true fluxes. Feed both to :func:`summarize_recovery`. With
    ``return_result=True``, returns ``(catalog, truth, result)``; the
    ``ForcedPhotometryResult`` carries the injected cutouts
    (``result.cutouts``), fitted sources and PSF stamps for visual
    inspection.

    ``rng`` defaults to a fixed seed for reproducibility; pass fresh entropy
    for independent realizations.

    ``injection_psf_product`` selects ``'grid'`` or ``'catalog'`` for the
    added signal. Recovery uses the same product unless ``psf_product``
    is supplied explicitly in ``run_kwargs``. Missing products raise an
    error instead of silently changing the experiment.
    """
    from pathlib import Path

    from astropy.table import Table, vstack

    from .cutouts import fetch_cutout
    from .pipeline import run_forced_photometry
    from .psf import extract_catalog_psf, extract_grid_psf

    extractors = {"grid": extract_grid_psf, "catalog": extract_catalog_psf}
    if injection_psf_product not in extractors:
        raise ValueError("injection_psf_product must be 'grid' or 'catalog'")
    run_kwargs.setdefault("psf_product", injection_psf_product)
    data_dir = Path(data_dir)
    prior_band = bands[0]
    if prior_band != "VIS":
        raise ValueError("the injection driver assumes a VIS prior band")

    cuts = {b: fetch_cutout(b, target_ra, target_dec, cutout_size_arcsec,
                            data_dir=data_dir / "cutouts",
                            with_flag=((bool(run_kwargs.get("with_flag", False))
                                        and b == prior_band)
                                       or run_kwargs.get("mask_bright_stars") is True))
            for b in bands}

    mer_cache = data_dir / (f"mer_catalog_{target_ra:.4f}_{target_dec:.4f}"
                            f"_{int(round(cutout_size_arcsec))}.fits")
    if mer_cache.exists():
        mer = Table.read(mer_cache)
    else:
        from .catalog import query_mer_catalog
        mer = query_mer_catalog(target_ra, target_dec,
                                cutout_size_arcsec / 2.0 / 3600.0)

    truth = make_truth_table(cuts[prior_band], mer, n_inject=n_inject,
                             flux_range_ujy=flux_range_ujy, model=model,
                             bands=bands, rng=rng)

    psf_radius = max(60.0, float(cutout_size_arcsec) * 0.75)
    injected = {}
    for b in bands:
        pd = extractors[injection_psf_product](
            b, target_ra, target_dec, radius_arcsec=psf_radius,
            tile_id=cuts[b].header.get("MERTILE"), data_dir=data_dir / "psf")
        if len(pd.get("stamps", ())) == 0:
            raise ValueError(f"No {injection_psf_product} PSF stamps for {b}")
        injected[b] = inject_sources(cuts[b], pd, truth, band=b)

    merged = vstack([mer, truth], join_type="outer",
                    metadata_conflicts="silent")

    result = run_forced_photometry(
        target_ra, target_dec, cutout_size_arcsec,
        prior={"band": prior_band, "objects": "mer"},
        target_bands={"euclid": tuple(b for b in bands if b != prior_band),
                      "wise": ()},
        data_dir=data_dir,
        cutouts=injected,
        mer_catalog=merged,
        verbose=verbose,
        **run_kwargs)
    if return_result:
        return result.to_table(), truth, result
    return result.to_table(), truth


def summarize_recovery(catalog, truth, *,
                       band: str = "VIS",
                       n_bins: int = 4,
                       snr_min: float = 5.0):
    """Bias / scatter / pull / completeness of the injected sources.

    Joins ``catalog`` (pipeline output; injected rows have negative
    ``object_id``) to ``truth`` on ``object_id`` and returns a dict:

    - ``ratio_median``: median(recovered / true), the flux bias,
    - ``ratio_mad``: robust scatter of the ratio,
    - ``pull``: (recovered - true) / reported error, per injection,
    - ``pull_std``: robust std of the pull, ~1 when the reported errors
      are correct (this is the acceptance test for the error calibration),
    - ``completeness``: fraction of injections recovered with
      ``flux_quality`` and S/N > ``snr_min``,
    - ``bins``: per-flux-bin Table of the same statistics.
    """
    from astropy.stats import mad_std
    from astropy.table import Table

    fcol, ecol = f"flux_{band}_ujy", f"flux_err_{band}_ujy"
    tcol = _FLUX_COL[band]

    cat_ids = np.asarray(catalog["object_id"])
    idx_by_id = {int(i): k for k, i in enumerate(cat_ids) if i < 0}
    if len(truth) == 0:
        raise ValueError("truth must contain at least one injected source")
    rows = np.array([idx_by_id.get(int(i), -1) for i in truth["object_id"]])
    matched = rows >= 0
    true_flux = np.asarray(truth[tcol], dtype=float)
    flux = np.full(len(truth), np.nan)
    err = np.full(len(truth), np.nan)
    flux[matched] = np.asarray(catalog[fcol], dtype=float)[rows[matched]]
    err[matched] = np.asarray(catalog[ecol], dtype=float)[rows[matched]]

    det = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    snr = np.where(det, flux / np.where(err > 0, err, np.inf), 0.0)
    completeness = float(np.mean(det & (snr > snr_min)))

    ratio = flux[det] / true_flux[det]
    pull = (flux[det] - true_flux[det]) / err[det]

    edges = np.exp(np.linspace(np.log(true_flux.min() * 0.999),
                               np.log(true_flux.max() * 1.001), n_bins + 1))
    brows = []
    for i in range(n_bins):
        m = det & (true_flux >= edges[i]) & (true_flux < edges[i + 1])
        if m.sum() == 0:
            continue
        r = flux[m] / true_flux[m]
        p = (flux[m] - true_flux[m]) / err[m]
        brows.append((edges[i], edges[i + 1], int(m.sum()),
                      float(np.median(r)),
                      float(mad_std(r)) if m.sum() > 2 else np.nan,
                      float(mad_std(p)) if m.sum() > 2 else np.nan))
    bins = Table(rows=brows or None,
                 names=("flux_lo_ujy", "flux_hi_ujy", "n", "ratio_median",
                        "ratio_mad", "pull_std"))

    return {
        "band": band,
        "n_injected": int(len(rows)),
        "n_recovered": int(det.sum()),
        "ratio_median": float(np.median(ratio)) if ratio.size else np.nan,
        "ratio_mad": float(mad_std(ratio)) if ratio.size > 2 else np.nan,
        "pull": pull,
        "pull_std": float(mad_std(pull)) if pull.size > 2 else np.nan,
        "completeness": completeness,
        "bins": bins,
    }
