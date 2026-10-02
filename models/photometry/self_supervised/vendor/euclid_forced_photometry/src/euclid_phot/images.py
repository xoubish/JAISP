"""Tractor ``Image`` construction from a Cutout + PSF.

Tractor's ``ConstantFitsWcs`` expects astrometry.net's WCS interface;
``AstropyWCSAdapter`` wraps an astropy WCS to match it.

Public: ``AstropyWCSAdapter``, ``build_tractor_image``.
"""
from __future__ import annotations

import numpy as np
from tractor import ConstantSky, Image, LinearPhotoCal
from tractor.psf import PixelizedPSF
from tractor.wcs import ConstantFitsWcs

from .config import AB_MAG_ZP, mer_bad_bits_for_band

# Distinguishes the default bad-bit set from an explicit flag_bad_bits=None.
_USE_DEFAULT_BAD_BITS = object()


class AstropyWCSAdapter:
    """Wrap an astropy WCS so Tractor's ConstantFitsWcs can call into it.

    Uses the low-level ``*_values`` API: the high-level SkyCoord path
    silently transforms frames for headers that omit RADESYS (like unWISE),
    giving wrong pixel coordinates.
    """
    def __init__(self, astropy_wcs):
        self._wcs = astropy_wcs

    def radec2pixelxy(self, ra, dec):
        px, py = self._wcs.world_to_pixel_values(ra, dec)
        return True, float(px) + 1.0, float(py) + 1.0

    def pixelxy2radec(self, x, y):
        ra, dec = self._wcs.pixel_to_world_values(x - 1.0, y - 1.0)
        return float(ra), float(dec)

    def pixel_scale(self):
        ps = self._wcs.proj_plane_pixel_scales()[0]
        ps_deg = float(ps.value) if hasattr(ps, "value") else float(ps)
        return ps_deg * 3600.0

    def get_cd(self):
        cd = self._wcs.pixel_scale_matrix
        return [cd[0, 0], cd[0, 1], cd[1, 0], cd[1, 1]]


def _invvar_from_rms(rms: np.ndarray, data: np.ndarray | None = None) -> np.ndarray:
    """Inverse variance from an RMS map.

    Pixels with non-positive or non-finite RMS get invvar=0. If ``data`` is
    given, pixels with non-finite data are also zeroed.
    """
    invvar = np.where((rms > 0) & np.isfinite(rms), 1.0 / rms ** 2, 0.0)
    if data is not None:
        invvar = np.where(np.isfinite(data), invvar, 0.0)
    return invvar


def _apply_flag(invvar: np.ndarray, flag: np.ndarray | None,
                bad_bits: int | None) -> np.ndarray:
    """Zero the inverse variance on flagged pixels.

    ``bad_bits`` is a bitmask: a pixel is dropped where ``flag & bad_bits``
    is non-zero. If ``bad_bits`` is None, any non-zero flag is treated as bad
    (the MER FLG plane's convention, where 0 means a clean pixel)."""
    if flag is None:
        return invvar
    flag = np.asarray(flag)
    bad = (flag != 0) if bad_bits is None else ((flag & int(bad_bits)) != 0)
    return np.where(bad, 0.0, invvar)


def build_tractor_image(cutout, psf_stamp: np.ndarray | None = None,
                        *,
                        psf_data: dict | None = None,
                        invvar: np.ndarray | None = None,
                        flag: np.ndarray | None = None,
                        flag_bad_bits=_USE_DEFAULT_BAD_BITS,
                        pixel_mask: np.ndarray | None = None,
                        name: str | None = None,
                        sky: float = 0.0,
                        mag_zero: float | None = None) -> Image:
    """Build a Tractor ``Image`` from a ``Cutout`` and a PSF stamp.

    Parameters
    ----------
    cutout : Cutout
        Output of ``fetch_cutout``; provides data, rms, wcs, header.
    psf_stamp : ndarray (H, W) float32, sum=1, optional
        The PSF used when ``psf_data`` is omitted. Not needed for local PSFs.
    psf_data : dict, optional
        CATALOG-PSF or GRID-PSF samples. When supplied, each source is
        rendered with its nearest sample, throughout fitting and residuals.
    invvar : ndarray, optional
        Override the inverse-variance map. If not given we compute it from
        ``cutout.rms`` as ``1/rms^2`` (with rms<=0 mapped to invvar=0).
    pixel_mask : ndarray of bool, optional
        Extra per-pixel veto applied on top of the FLG bits: the inverse
        variance is zeroed where True, e.g. the STARSIGNAL bright-star
        mask from :func:`euclid_phot.flags.starsignal_pixel_mask`.
    name : str, optional
        Image name string. Defaults to ``f"Euclid-{cutout.band}"``.
    sky : float
        Constant sky level for ConstantSky.
    mag_zero : float, optional
        AB magnitude zero point of the IMAGE. If None, read from
        ``cutout.header['MAGZERO']``; if that is missing, falls back to 23.9
        (the MER *catalog* flux-to-mag zero point) and emits a UserWarning.
        Pass ``mag_zero`` explicitly when the header lacks ``MAGZERO``.

    Returns
    -------
    tractor.Image
    """
    if invvar is None:
        if cutout.rms is None:
            raise ValueError("cutout.rms is None; pass an explicit invvar=")
        invvar = _invvar_from_rms(cutout.rms, cutout.data)
    else:
        # Caller-supplied invvar must not pair finite weight with
        # non-finite data.
        invvar = np.where(np.isfinite(cutout.data), invvar, 0.0)

    flag = flag if flag is not None else getattr(cutout, "flag", None)
    bad_bits = (mer_bad_bits_for_band(cutout.band)
                if flag_bad_bits is _USE_DEFAULT_BAD_BITS else flag_bad_bits)
    invvar = _apply_flag(invvar, flag, bad_bits)

    if pixel_mask is not None:
        invvar = np.where(np.asarray(pixel_mask, dtype=bool), 0.0, invvar)

    if mag_zero is None:
        if "MAGZERO" in cutout.header:
            mag_zero = cutout.header["MAGZERO"]
        else:
            # MER mosaics carry a band- and tile-dependent AB zeropoint in
            # MAGZERO (the mosaics are in ADU/s or electrons), several mag
            # from the catalog's 23.9.
            import warnings
            warnings.warn(
                f"no MAGZERO in header for band {getattr(cutout, 'band', '?')!r}; "
                "falling back to 23.9, which is almost certainly wrong for a "
                "Euclid MER mosaic. Pass mag_zero= explicitly.",
                stacklevel=2)
            mag_zero = 23.9
    scale = 10 ** (0.4 * (float(mag_zero) - AB_MAG_ZP))

    wcs = ConstantFitsWcs(AstropyWCSAdapter(cutout.wcs))
    if psf_data is not None:
        psf_tile = psf_data.get("tile_id")
        science_tile = cutout.header.get("MERTILE")
        if psf_tile and science_tile and str(psf_tile) != str(science_tile):
            raise ValueError("PSF samples and science cutout come from different MER tiles")
    if psf_data is None:
        if psf_stamp is None:
            raise ValueError('Supply psf_data for local PSFs or a constant psf_stamp.')
        # PixelizedPSF assumes a unit-sum PSF.
        psf_sum = float(np.sum(psf_stamp))
        if not np.isfinite(psf_sum) or psf_sum <= 0:
            raise ValueError(
                f"psf_stamp has non-finite or non-positive sum ({psf_sum}); "
                "cannot build a PixelizedPSF from it.")
        psf_stamp = np.asarray(psf_stamp, dtype=np.float32) / psf_sum
        psf = PixelizedPSF(psf_stamp)
    else:
        from .spatial_psf import SpatialPixelizedPSF
        psf = SpatialPixelizedPSF(psf_data, wcs)

    return Image(
        data=np.where(np.isfinite(cutout.data), cutout.data, 0.0),
        invvar=invvar,
        psf=psf,
        wcs=wcs,
        photocal=LinearPhotoCal(scale, band=cutout.band),
        sky=ConstantSky(sky),
        name=name or f"Euclid-{cutout.band}",
    )
