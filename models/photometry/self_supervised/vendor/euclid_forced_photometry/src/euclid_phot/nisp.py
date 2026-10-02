"""Forced photometry on NISP Y/J/H bands given VIS-fitted sources.

For each NISP band we build a Tractor ``Image`` with the supplied PSF field,
clone the VIS-fitted sources keeping positions and shapes, and run flux-only
optimization. Returns fluxes in microJanskys, aligned 1:1 with the input
source list.
"""
from __future__ import annotations

import copy as _copy

import numpy as np
from tractor import NanoMaggies, Tractor

from .images import _invvar_from_rms, build_tractor_image

_UJY_PER_NMGY = 3.631


def _clone_for_band(vis_src, band: str):
    """Deep-copy a VIS-fitted source and reset its brightness for ``band``.

    Works uniformly for every Tractor source class the model-selection
    tree can produce.
    """
    cloned = _copy.deepcopy(vis_src)
    cloned.brightness = NanoMaggies(**{band: 1.0})
    # halfsize is a render-patch size in pixels carried over from the VIS
    # fit; clear it so each band's image sizes its own render patch.
    if hasattr(cloned, "halfsize"):
        cloned.halfsize = None
    return cloned


def _fit_one_nisp_band(args):
    """Worker for ``fit_nisp_forced``: one band, runnable in a thread."""
    sources, band, cutout, psf_stamp, pixel_mask, psf_data = args
    if pixel_mask is not None:
        invvar = np.where(np.asarray(pixel_mask, bool), 0.0,
                          _invvar_from_rms(cutout.rms, cutout.data))
        tim = build_tractor_image(cutout, psf_stamp, invvar=invvar, psf_data=psf_data)
    else:
        tim = build_tractor_image(cutout, psf_stamp, psf_data=psf_data)
    band_sources = [_clone_for_band(s, band) for s in sources]
    for s in band_sources:
        s.freezeAllBut("brightness")
    tim.freezeAllParams()
    tr = Tractor([tim], band_sources)
    # With only brightness thawed and the sky frozen, R.IV aligns 1:1 with
    # the sources; the flux error is 1/sqrt(IV).
    R = tr.optimize_forced_photometry(
        minsb=0.0, mindlnp=1.0, sky=False, variance=True)
    flux_ujy = (np.array([s.brightness.getFlux(band) for s in band_sources])
                * _UJY_PER_NMGY)
    iv = np.asarray(R.IV, dtype=float) if getattr(R, "IV", None) is not None \
        else np.zeros(len(band_sources))
    if iv.shape[0] != len(band_sources):
        iv = np.zeros(len(band_sources))
    with np.errstate(divide="ignore", invalid="ignore"):
        flux_err_ujy = np.where(iv > 0, 1.0 / np.sqrt(iv), np.nan) * _UJY_PER_NMGY
    return band, {"flux_ujy": flux_ujy, "flux_err_ujy": flux_err_ujy}


def fit_nisp_forced(sources, cutouts: dict, psf_stamps: dict | None = None,
                    *,
                    bands: tuple = ("Y", "J", "H"),
                    psf_data: dict | None = None,
                    pixel_mask: np.ndarray | None = None,
                    n_workers: int = 1) -> dict:
    """Run forced photometry on each NISP band.

    Parameters
    ----------
    sources : list of Tractor sources (from the VIS step)
    cutouts : dict[band] -> Cutout (output of fetch_cutout)
    psf_stamps : dict[band] -> ndarray, optional
        Constant PSFs; not needed when ``psf_data`` supplies local samples.
    psf_data : dict[band] -> PSF samples, optional
        Render every source through its nearest sample in that band.
        If omitted, use the constant ``psf_stamps`` as before.
    bands : tuple
        Subset of NISP bands to fit.
    pixel_mask : ndarray of bool, optional
        Pixels to exclude from every band fit (inverse variance zeroed
        there). All MER mosaics are resampled onto the common 0.1 arcsec
        VIS grid, so the prior-band bright-star mask (the STARSIGNAL
        footprints) applies to the NISP cutouts unchanged.
    n_workers : int
        If > 1, fit the bands in parallel threads. The speedup depends on
        your Tractor build, since its compiled FFT may or may not run on
        several threads at once (~1.4x at n_workers=4 on the reference build).

    Returns
    -------
    dict[band] -> dict with keys ``flux_ujy`` and ``flux_err_ujy``, each an
        ndarray in microJansky aligned 1:1 with ``sources``. ``flux_err_ujy``
        is the formal 1-sigma uncertainty (1/sqrt of the Tractor
        inverse-variance) from the flux-only fit.
    """
    work = [(sources, band, cutouts[band],
             psf_stamps[band] if psf_data is None and psf_stamps is not None else None, pixel_mask,
             psf_data[band] if psf_data is not None else None)
            for band in bands if band in cutouts]
    if n_workers <= 1 or len(work) <= 1:
        results = [_fit_one_nisp_band(w) for w in work]
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(
                min(n_workers, len(work))) as exec_:
            results = list(exec_.map(_fit_one_nisp_band, work))
    return dict(results)
