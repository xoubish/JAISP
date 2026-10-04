"""Build Tractor source models from MER catalog priors.

Per MER row:

- ``PointSource`` when ``point_like_prob > 0.96`` (see
  ``star_prob_threshold``) or when the Sersic effective radius is smaller
  than the PSF half-width.
- ``SersicGalaxy(pos, brightness, shape, n)`` otherwise, preserving MER's
  continuous Sersic index ``n``.
- ``ExpGalaxy`` fallback when no Sersic fit is available, seeded from the
  SE++ isophotal semimajor axis.

MER fluxes are microJansky; Tractor's ``NanoMaggies`` uses an AB zero
point of 22.5 (1 nMgy = 3.631 microJy), so seeds divide by 3.631.
"""
from __future__ import annotations

import numpy as np
from tractor import DevGalaxy, ExpGalaxy, GalaxyShape, NanoMaggies, PointSource, RaDecPos
from tractor.sersic import SersicGalaxy, SersicIndex

_UJY_PER_NMGY = 3.631
# Tractor's Sersic mixture-of-Gaussians is valid for 0.29 <= n <= 6.3;
# clip slightly inside to leave LSQR stepping margin.
SERSIC_N_MIN = 0.4
SERSIC_N_MAX = 6.0


def _row_value(row, col, default=None):
    val = row[col]
    if hasattr(val, "mask") and np.ma.is_masked(val):
        return default
    val = float(val)
    return val if np.isfinite(val) else default


def _configure_shape_steps(shape):
    """Use local derivatives for Tractor's radius/axis-ratio representation.

    GalaxyShape otherwise inherits unit steps: +1 arcsec and +1 in b/a.
    Those secants are not useful derivatives of compact Euclid galaxies.
    EllipseE/ESoft supply their own step sizes and are left unchanged.
    """
    if isinstance(shape, GalaxyShape):
        shape.setStepSizes([max(0.001, 0.01 * abs(float(shape.re))), 0.01, 1.0])
    return shape


def _galaxy_shape(re, ab, phi):
    return _configure_shape_steps(GalaxyShape(re, ab, phi))


def build_sources_from_mer(mer_cat, *, band: str = "VIS",
                           psf_fwhm_arcsec: float = 0.16,
                           flux_col: str = "flux_vis_sersic",
                           star_prob_threshold: float = 0.96,
                           pixel_scale_arcsec: float = 0.10,
                           force_model: str | None = None) -> list:
    """Construct one Tractor source per MER row.

    Parameters
    ----------
    mer_cat : astropy.table.Table
    band : str
        Band name to register with NanoMaggies (must match the photocal
        band).
    psf_fwhm_arcsec : float
        Full-width-half-max of the PSF in this band (arcsec). Galaxies
        with ``re < psf_fwhm/2`` are downgraded to PointSource.
    flux_col : str
        Catalog column used to seed brightness.
    star_prob_threshold : float
        A source with ``point_like_prob`` above this is built as a
        PointSource. The default 0.96 follows the Euclid MER paper's
        high-purity star cut. Sources below the cut still collapse to
        PointSource when their effective radius is below the PSF
        half-width.
    pixel_scale_arcsec : float
        Pixel scale of the prior image, used to convert the MER isophotal
        ``semimajor_axis`` (stored in pixels) to arcsec in the no-Sersic
        fallback.
    force_model : {'point', 'sersic', 'exp', 'dev'}, optional
        Build every source as this Tractor class instead of following the
        MER classification. Galaxy shapes are seeded from the catalog
        values where available (0.3 arcsec circular otherwise); the
        step-2 shape refit adjusts them.

    Returns
    -------
    list[tractor.PointSource | tractor.sersic.SersicGalaxy | tractor.ExpGalaxy]
        Aligned 1:1 with ``mer_cat`` rows.
    """
    psf_hwhm = psf_fwhm_arcsec / 2.0
    sources = []
    for row in mer_cat:
        pos = RaDecPos(row["ra"], row["dec"])
        # A masked or non-finite catalog flux would seed a NaN brightness.
        flux_seed = _row_value(row, flux_col, 0.0)
        if flux_seed is None or not np.isfinite(flux_seed):
            flux_seed = 0.0
        brightness = NanoMaggies(**{band: flux_seed / _UJY_PER_NMGY})

        plp = _row_value(row, "point_like_prob", 0.0)
        re  = _row_value(row, "sersic_sersic_vis_radius")
        ab  = _row_value(row, "sersic_sersic_vis_axis_ratio")
        pa  = _row_value(row, "sersic_angle")
        n   = _row_value(row, "sersic_sersic_vis_index")

        if force_model is not None:
            if force_model == "point":
                sources.append(PointSource(pos, brightness)); continue
            semimaj_pix = _row_value(row, "semimajor_axis", 0.0) or 0.0
            re_seed = re if re is not None and re > 0 else \
                semimaj_pix * pixel_scale_arcsec
            re_seed = max(0.05, min(re_seed if re_seed > 0 else 0.3, 20.0))
            ab_seed = max(0.05, min(ab if ab is not None else 1.0, 1.0))
            shape = _galaxy_shape(re_seed, ab_seed, -(pa or 0.0))
            if force_model == "sersic":
                n_val = max(SERSIC_N_MIN,
                            min(n if n is not None else 1.0, SERSIC_N_MAX))
                sources.append(SersicGalaxy(pos, brightness, shape,
                                            SersicIndex(n_val)))
            elif force_model == "exp":
                sources.append(ExpGalaxy(pos, brightness, shape))
            elif force_model == "dev":
                sources.append(DevGalaxy(pos, brightness, shape))
            else:
                raise ValueError(
                    f"force_model must be one of 'point', 'sersic', 'exp', "
                    f"'dev'; got {force_model!r}")
            continue

        if plp is not None and plp > star_prob_threshold:
            sources.append(PointSource(pos, brightness)); continue

        if re is not None:
            if re < psf_hwhm:
                sources.append(PointSource(pos, brightness)); continue
            re_clipped = max(0.05, min(re, 20.0))
            ab_clipped = max(0.05, min(ab if ab is not None else 1.0, 1.0))
            shape = _galaxy_shape(re_clipped, ab_clipped, -(pa or 0.0))
            n_val = n if n is not None else 1.0
            n_clipped = max(SERSIC_N_MIN, min(n_val, SERSIC_N_MAX))
            sources.append(
                SersicGalaxy(pos, brightness, shape, SersicIndex(n_clipped)))
            continue

        # No Sersic fit: ExpGalaxy seeded from the SE++ isophotal semimajor
        # axis. MER stores semimajor_axis in pixels.
        semimaj_pix = _row_value(row, "semimajor_axis", 0.0) or 0.0
        semimaj = semimaj_pix * pixel_scale_arcsec
        if semimaj < psf_hwhm:
            sources.append(PointSource(pos, brightness)); continue
        ecc = _row_value(row, "ellipticity", 0.0) or 0.0
        ab_iso = max(0.05, 1.0 - ecc)
        re_iso = max(0.05, min(semimaj, 20.0))
        shape = _galaxy_shape(re_iso, ab_iso,
                            -(_row_value(row, "position_angle", 0.0) or 0.0))
        sources.append(ExpGalaxy(pos, brightness, shape))
    return sources


def build_sources_from_coords(ra, dec, *, band: str = "VIS",
                              flux_guess_ujy=None,
                              model: str = "point",
                              re_arcsec=None, axis_ratio=None,
                              position_angle_deg=None, sersic_n=None) -> list:
    """Build Tractor sources at user-supplied sky positions.

    Positions are returned thawed; the fitting step freezes them for forced
    photometry exactly as it does for MER-prior sources.

    Parameters
    ----------
    ra, dec : array-like (deg)
        Source positions.
    band : str
        Band to register with NanoMaggies (must match the image photocal).
    flux_guess_ujy : array-like or float, optional
        Initial brightness in microJansky. Defaults to 1.0 uJy; the value
        only seeds the linear flux fit.
    model : {'point', 'sersic', 'exp', 'dev'}
        ``'point'`` builds a PointSource at each position. The galaxy
        classes build a
        SersicGalaxy / ExpGalaxy / DevGalaxy and require ``re_arcsec``
        (and optionally ``axis_ratio``, ``position_angle_deg`` in the MER
        THETA_IMAGE convention, and ``sersic_n`` for the Sersic class).
    re_arcsec, axis_ratio, position_angle_deg, sersic_n : array-like, optional
        Per-source shape seeds used when ``model='sersic'``.

    Returns
    -------
    list[tractor source]  aligned 1:1 with ``ra``/``dec``.
    """
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    nsrc = len(ra)

    def _per_source(val, default):
        if val is None:
            return np.full(nsrc, default, dtype=float)
        arr = np.atleast_1d(np.asarray(val, dtype=float))
        return np.full(nsrc, float(arr[0])) if arr.size == 1 else arr

    flux = _per_source(flux_guess_ujy, 1.0)
    sources = []
    for i in range(nsrc):
        pos = RaDecPos(ra[i], dec[i])
        brightness = NanoMaggies(**{band: float(flux[i]) / _UJY_PER_NMGY})
        if model == "point":
            sources.append(PointSource(pos, brightness))
        elif model in ("sersic", "exp", "dev"):
            if re_arcsec is None:
                raise ValueError(f"model={model!r} needs re_arcsec")
            re_i = _per_source(re_arcsec, 0.3)[i]
            ab_i = _per_source(axis_ratio, 1.0)[i]
            pa_i = _per_source(position_angle_deg, 0.0)[i]
            shape = _galaxy_shape(max(0.05, min(re_i, 20.0)),
                                max(0.05, min(ab_i, 1.0)), -pa_i)
            if model == "sersic":
                n_i = _per_source(sersic_n, 1.0)[i]
                sources.append(SersicGalaxy(
                    pos, brightness, shape,
                    SersicIndex(max(SERSIC_N_MIN, min(n_i, SERSIC_N_MAX)))))
            elif model == "exp":
                sources.append(ExpGalaxy(pos, brightness, shape))
            else:
                sources.append(DevGalaxy(pos, brightness, shape))
        else:
            raise ValueError(
                f"model must be 'point', 'sersic', 'exp', or 'dev'; "
                f"got {model!r}")
    return sources
