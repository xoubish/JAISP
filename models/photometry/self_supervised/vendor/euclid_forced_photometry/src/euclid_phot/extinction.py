"""Galactic extinction columns for the output catalog.

Adds m_corrected = m_observed - R_band * E(B-V) with ``R_band`` from
:data:`euclid_phot.config.EXTINCTION_COEFF`. E(B-V) comes from the MER
per-source ``gal_ebv`` column (Planck R1.20) when available, else one IRSA
dust-service field value (:func:`query_ebv`, cached on disk); gradients are
not resolved by the single field fallback. Flux columns stay observed-frame; corrected
magnitudes, ``ebv``, and per-band ``a_<band>_mag`` are added alongside.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from .config import EXTINCTION_COEFF


def query_ebv(ra: float, dec: float, *, data_dir=None) -> float:
    """Field E(B-V) from the IRSA Galactic dust service (SFD-calibrated).

    Returns the Schlafly & Finkbeiner (2011) rescaled mean E(B-V) at
    ``(ra, dec)``. The value is cached to
    ``data_dir/ebv_<ra>_<dec>.json`` so a field is queried
    once; offline runs on a cached field never touch the network.
    """
    cache = None
    if data_dir is not None:
        cache = Path(data_dir) / f"ebv_{ra:.4f}_{dec:.4f}.json"
        if cache.exists():
            return float(json.loads(cache.read_text())["ebv"])

    from astroquery.ipac.irsa.irsa_dust import IrsaDust

    from .netutils import retry

    tab = retry(
        lambda: IrsaDust.get_query_table(f"{ra} {dec} equ j2000",
                                         section="ebv"),
        what="IRSA dust E(B-V) query")
    ebv = float(tab["ext SandF mean"][0])

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"ebv": ebv, "ra": ra, "dec": dec,
                                     "source": "IRSA dust, ext SandF mean"}))
    return ebv


def ebv_for_result(result, *, data_dir=None):
    """Per-source E(B-V) for a ``ForcedPhotometryResult``.

    Returns ``(ebv_array, source_str)`` aligned with ``result.sources``, or
    ``(None, reason)`` when no value is available. Prefers the MER
    ``gal_ebv`` column; falls back to one IRSA dust-service value for the
    whole field (user-coordinate runs have no MER rows).
    """
    n = len(result.sources)
    mer = result.mer_cat
    if mer is not None and "gal_ebv" in getattr(mer, "colnames", []):
        col = mer["gal_ebv"]
        ebv = np.asarray(col.filled(np.nan) if hasattr(col, "filled")
                         else col, dtype=float)
        if len(ebv) == n and np.isfinite(ebv).any():
            med = float(np.nanmedian(ebv))
            return np.where(np.isfinite(ebv), ebv, med), \
                "MER gal_ebv (Planck R1.20, per source)"
    try:
        ra, dec = float(result.target[0]), float(result.target[1])
        ebv = query_ebv(ra, dec, data_dir=data_dir)
        return np.full(n, ebv), "IRSA dust service (SandF mean, per field)"
    except Exception as exc:
        return None, f"unavailable ({exc!r})"


def add_extinction_columns(tab, ebv, *, ebv_source: str = ""):
    """Add ``ebv``, ``a_<band>_mag`` and ``mag_<band>_ab_extcorr`` columns.

    ``ebv`` may be a scalar (applied to every row) or an array aligned with
    the table. Bands are discovered from the existing ``mag_<band>_ab``
    columns; flux columns are left observed-frame, as recorded in
    ``tab.meta``.
    """
    import astropy.units as u

    n = len(tab)
    ebv = np.broadcast_to(np.asarray(ebv, dtype=float), (n,)).copy()
    tab["ebv"] = ebv * u.mag

    bands = [c[len("mag_"):-len("_ab")] for c in tab.colnames
             if c.startswith("mag_") and c.endswith("_ab")
             and not c.startswith("mag_err_")]
    for b in bands:
        r = EXTINCTION_COEFF.get(b)
        if r is None:
            warnings.warn(f"no extinction coefficient for band {b!r}; "
                          "skipping its corrected magnitude.", stacklevel=2)
            continue
        a = r * ebv
        tab[f"a_{b}_mag"] = a * u.mag
        tab[f"mag_{b}_ab_extcorr"] = (
            np.asarray(tab[f"mag_{b}_ab"], dtype=float) - a) * u.mag

    tab.meta["extinction"] = {
        "law": ("Gordon et al. 2023 at Euclid effective wavelengths, "
                "R_V = 3.1 (config.EXTINCTION_COEFF); "
                "Yuan, Liu & Xiang 2013 Table 2, third-column empirical "
                "coefficients for W1/W2"),
        "ebv_source": ebv_source or "caller-supplied",
        "limitations": ("Approximate foreground screen with fixed coefficients; "
                        "does not propagate extinction-law or dust-map uncertainty. "
                        "The full dust column is not appropriate for every foreground star."),
        "convention": ("flux columns are OBSERVED-frame; "
                       "mag_<band>_ab_extcorr = mag_<band>_ab - "
                       "R_band * E(B-V)"),
        "coefficients": {b: round(float(v), 4)
                         for b, v in EXTINCTION_COEFF.items()},
    }
    return tab
