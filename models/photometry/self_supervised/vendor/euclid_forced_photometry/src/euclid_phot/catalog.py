"""MER catalog retrieval via IRSA TAP.

Public:
    query_mer_catalog(ra, dec, half_size_deg, *, brightness_cut_ujy=0.0)
        Joins ``euclid_q1_mer_catalogue`` with ``euclid_q1_mer_morphology`` on
        ``object_id`` and returns the rows whose centers land inside a
        cos(dec)-corrected RA/Dec box centered on ``(ra, dec)``. Adds an
        ``is_star`` boolean column derived from ``point_like_prob``.

Columns returned (microJansky fluxes throughout):
    object_id, ra, dec
    flux_vis_psf / _sersic + errors
    flux_{h,j,y}_templfit / _sersic + errors
    flux_detection_total
    semimajor_axis, ellipticity, position_angle, point_like_prob
    gal_ebv, gal_ebv_err (per-source Galactic E(B-V), Planck R1.20 map)
    det_quality_flag (MER source-quality bits; bits 7/8 mark sources inside
        the MER VIS/NIR bright-star polygon masks)
    sersic_sersic_vis_radius / _index / _axis_ratio, sersic_angle
    is_star (derived)
"""
from __future__ import annotations

import numpy as np
from astroquery.ipac.irsa import Irsa

# Row cap passed to IRSA TAP; a result that reaches it is flagged as
# likely truncated.
_MER_QUERY_MAXREC = 200_000

_ADQL = """
SELECT
    m.object_id, m.ra, m.dec,
    m.flux_vis_psf, m.fluxerr_vis_psf,
    m.flux_vis_sersic, m.fluxerr_vis_sersic,
    m.flux_h_templfit, m.fluxerr_h_templfit,
    m.flux_h_sersic, m.fluxerr_h_sersic,
    m.flux_j_templfit, m.fluxerr_j_templfit,
    m.flux_j_sersic, m.fluxerr_j_sersic,
    m.flux_y_templfit, m.fluxerr_y_templfit,
    m.flux_y_sersic, m.fluxerr_y_sersic,
    m.flux_detection_total,
    m.semimajor_axis, m.ellipticity, m.position_angle,
    m.point_like_prob,
    m.det_quality_flag,
    m.gal_ebv, m.gal_ebv_err,
    morph.sersic_sersic_vis_radius,
    morph.sersic_sersic_vis_index,
    morph.sersic_sersic_vis_axis_ratio,
    morph.sersic_angle
FROM euclid_q1_mer_catalogue AS m
LEFT JOIN euclid_q1_mer_morphology AS morph ON m.object_id = morph.object_id
WHERE {ra_where}
  AND m.dec BETWEEN {dec_min} AND {dec_max}
  AND m.flux_vis_psf  > {flux_min}{extra_where}
ORDER BY m.flux_vis_psf DESC
"""


def _ra_where(ra: float, half_size_ra_deg: float) -> str:
    """ADQL RA box constraint, handling the RA = 0/360 wrap.

    A box that stays inside [0, 360] is a plain BETWEEN; one that straddles
    the wrap becomes the union of the two arcs (``ra >= a OR ra <= b``).
    """
    ra_min = ra - half_size_ra_deg
    ra_max = ra + half_size_ra_deg
    if ra_min >= 0.0 and ra_max <= 360.0:
        return f"m.ra BETWEEN {ra_min} AND {ra_max}"
    a = ra_min % 360.0
    b = ra_max % 360.0
    return f"(m.ra >= {a} OR m.ra <= {b})"


def query_mer_catalog(ra: float, dec: float, half_size_deg: float,
                      *, brightness_cut_ujy: float = 0.0,
                      require_detection_band: str | None = None,
                      star_prob_threshold: float = 0.96):
    """Query the MER catalog inside a cos(dec)-corrected RA/Dec box.

    Parameters
    ----------
    ra, dec : float
        Box center in degrees.
    half_size_deg : float
        Half-width of the box in degrees (DEC); RA is widened by 1/cos(dec).
    brightness_cut_ujy : float
        Lower bound on ``flux_vis_psf`` (microJansky). The default of 0 keeps
        every detection with a positive VIS PSF flux.
    require_detection_band : {'VIS','Y','J','H'} or None
        If given, additionally require a positive ``flux_<band>_templfit``
        (``flux_vis_psf`` for VIS, which has no TEMPLFIT photometry).
    star_prob_threshold : float
        ``is_star`` is set where ``point_like_prob`` exceeds this. The default
        0.96 follows the Euclid MER paper's high-purity star cut.

    Returns
    -------
    astropy.table.Table
    """
    extra = ""
    if require_detection_band is not None:
        band = require_detection_band.lower()
        col = "flux_vis_psf" if band == "vis" else f"flux_{band}_templfit"
        extra = f"\n  AND m.{col} > 0"
    cos_dec = float(np.cos(np.radians(dec)))
    cos_edge = max(np.cos(np.radians(dec - half_size_deg)),
                   np.cos(np.radians(dec + half_size_deg)))
    radius_deg = float(np.hypot(half_size_deg,
                                half_size_deg * cos_edge / cos_dec)) + 1e-3
    circle = (f"CONTAINS(POINT('ICRS', m.ra, m.dec), "
              f"CIRCLE('ICRS', {ra % 360.0}, {dec}, {radius_deg}))=1")
    adql = _ADQL.format(
        ra_where=f"{circle}\n  AND {_ra_where(ra % 360.0, half_size_deg / cos_dec)}",
        dec_min=dec - half_size_deg,
        dec_max=dec + half_size_deg,
        flux_min=float(brightness_cut_ujy),
        extra_where=extra,
    )
    # IRSA synchronous TAP silently truncates at MAXREC=2000 by default.
    import warnings

    from .netutils import retry
    mer_cat = retry(
        lambda: Irsa.query_tap(adql, maxrec=_MER_QUERY_MAXREC).to_table(),
        what="IRSA TAP MER query")
    if len(mer_cat) == 0:
        warnings.warn(
            f"MER query returned no rows for the box at ({ra}, {dec}); the "
            "field may be outside Euclid Q1 coverage or fully masked.",
            stacklevel=2)
    elif len(mer_cat) >= _MER_QUERY_MAXREC:
        warnings.warn(
            f"MER query returned {len(mer_cat)} rows, the MAXREC cap; the "
            "catalog is likely truncated. Use a smaller cutout or raise "
            "euclid_phot.catalog._MER_QUERY_MAXREC.",
            stacklevel=2)
    # Masked point_like_prob fills as ~1e20 and would compare as a star.
    plp = mer_cat["point_like_prob"]
    plp_filled = plp.filled(0.0) if hasattr(plp, "filled") else np.asarray(plp, dtype=float)
    mer_cat["is_star"] = np.asarray(plp_filled, dtype=float) > star_prob_threshold
    return mer_cat
