"""Two-step fitting on the VIS image, then helpers for forced photometry.

Step 1 (flux-only forced photometry) gives every source a reasonable
brightness estimate. Sources whose Step-1 flux is negative or absurdly far
from the MER reference are flagged and held frozen for Step 2.

Step 2 (free-shape optimization) thaws the shape of every resolved
galaxy at least ``margin_pix`` from the cutout edge (positions stay
frozen), then runs Tractor's joint optimizer for up to ``n_iter`` steps
with a ``dlnp < 0.1`` early-out. In the default MER path the galaxies are
``SersicGalaxy``, so this thaws their shape *and* their continuous Sersic
index ``n``. ``PointSource``, ``SimpleGalaxy`` (frozen circular shape by
definition), and ``FixedCompositeGalaxy`` are left untouched.
"""
from __future__ import annotations

import copy as _copy
import warnings

import numpy as np
from tractor import PointSource, Tractor
from tractor.galaxy import FixedCompositeGalaxy

from .models import _configure_shape_steps

_UJY_PER_NMGY = 3.631


def _flux_ujy(src, band):
    return src.brightness.getFlux(band) * _UJY_PER_NMGY


def conditional_flux_errors(tim, sources) -> np.ndarray:
    """Flux uncertainties in microJy at the sources' current geometry.

    For each single-amplitude source, return ``1/sqrt(sum(t**2 * weight))``
    with ``t`` its unit-flux template including the image photocalibration.
    Positions, shapes, neighbours and sky are held fixed; blend/shape/sky
    covariance and correlated image noise are not included. No parameter
    or flux is fitted or changed. A source with no weighted template
    support receives NaN. Requires the linear NanoMaggies image convention
    used by :func:`build_tractor_image`.
    """
    from tractor import LinearPhotoCal

    photocal = tim.getPhotoCal()
    if not isinstance(photocal, LinearPhotoCal):
        raise ValueError("conditional_flux_errors requires LinearPhotoCal")
    scale = float(photocal.getScale())
    weights = tim.getInvvar()
    errors = np.full(len(sources), np.nan)
    for i, src in enumerate(sources):
        patches = src.getUnitFluxModelPatches(tim, minval=0.)
        if len(patches) != 1:
            raise ValueError("conditional_flux_errors requires one flux amplitude per source")
        patch = patches[0]
        if patch is None or patch.patch is None:
            continue
        spatch, simage = patch.getSlices(tim.shape)
        template = np.asarray(patch.patch[spatch], dtype=float) * scale
        iv = float(np.sum(template**2 * weights[simage]))
        if np.isfinite(iv) and iv > 0:
            errors[i] = _UJY_PER_NMGY / np.sqrt(iv)
    return errors


def _revert_railed_shapes(sources, seed_shapes, *,
                          growth_cap: float = 5.0,
                          re_floor_arcsec: float = 2.72,
                          max_re_arcsec: float = 20.0) -> int:
    """Return step-2 shapes that ran away along the radius degeneracy to
    their seed value, and freeze them.

    Uses the same per-source ceiling as the model-selection tree
    (selection.py): ``clip(growth_cap * seed_re, re_floor, max_re)``.
    Returns the number reverted.
    """
    n = 0
    for i, seed in seed_shapes.items():
        src = sources[i]
        try:
            re_fit = float(src.getShape().re)
            re_seed = float(seed.re)
        except (AttributeError, TypeError):
            continue
        cap = float(np.clip(growth_cap * re_seed,
                            re_floor_arcsec, max_re_arcsec))
        if not np.isfinite(re_fit) or re_fit > cap:
            src.shape = _copy.deepcopy(seed)
            src.freezeParam("shape")
            n += 1
    return n


def fit_forced_photometry(tim, sources, *,
                          band: str = "VIS",
                          initial_fluxes_ujy: np.ndarray | None = None,
                          flag_unphysical: bool = True,
                          return_errors: bool = False):
    """Step 1: freeze everything except brightness; optimize flux.

    Parameters
    ----------
    return_errors : bool
        If True, also return ``flux_err_ujy``, the formal 1-sigma flux
        uncertainty (1/sqrt of the Tractor inverse-variance) from this
        flux-only fit, aligned 1:1 with ``sources``.

    Returns
    -------
    tractor : Tractor
    fit_quality : ndarray of bool, shape (len(sources),)
        True for sources that came out with a physical flux. Coarse
        catastrophe guard only: rejects a negative flux or a flux more than
        100x the MER reference, not a goodness-of-fit cut. Rejected sources
        are frozen in place so Step 2 cannot touch them.
    flux_err_ujy : ndarray, optional
        Only when ``return_errors=True``.
    """
    tractor = Tractor([tim], sources)
    tim.freezeAllParams()
    for src in sources:
        src.freezeAllBut("brightness")

    R = tractor.optimize_forced_photometry(
        minsb=0, mindlnp=1, sky=False, variance=return_errors)

    flux_err_ujy = None
    if return_errors:
        iv = np.asarray(R.IV, dtype=float) if getattr(R, "IV", None) is not None \
            else np.zeros(len(sources))
        if iv.shape[0] != len(sources):
            iv = np.zeros(len(sources))
        with np.errstate(divide="ignore", invalid="ignore"):
            flux_err_ujy = np.where(iv > 0, 1.0 / np.sqrt(iv), np.nan) * _UJY_PER_NMGY

    fit_quality = np.ones(len(sources), dtype=bool)
    if flag_unphysical:
        for i, src in enumerate(sources):
            flux = _flux_ujy(src, band)
            ref = (initial_fluxes_ujy[i]
                   if initial_fluxes_ujy is not None else None)
            bad = flux < 0
            if ref is not None and ref > 0 and abs(flux) > 100 * ref:
                bad = True
            if bad:
                src.freezeAllParams()
                fit_quality[i] = False
    if return_errors:
        return tractor, fit_quality, flux_err_ujy
    return tractor, fit_quality


def fit_free_shapes(tractor, tim, sources, fit_quality, *,
                    band: str = "VIS",
                    initial_fluxes_ujy: np.ndarray | None = None,
                    margin_pix: int = 50,
                    n_iter: int = 30,
                    dlnp_min: float = 0.1):
    """Step 2: thaw shapes for interior, well-behaved sources; jointly optimize.

    Re-uses the Tractor object from Step 1 (positions frozen in Step 1 stay
    frozen). Thaws the shape of every resolved galaxy at least ``margin_pix``
    from the edge; for a ``SersicGalaxy`` (the default MER class) that includes
    the Sersic index ``n``. ``PointSource``, ``SimpleGalaxy``, and
    ``FixedCompositeGalaxy`` are skipped. Returns the updated ``fit_quality``
    array.
    """
    # SimpleGalaxy (Weaver et al. 2023 bridge tier) subclasses ExpGalaxy;
    # its defining frozen circular shape must survive step 2.
    from .selection import SimpleGalaxy

    H, W = tim.shape
    # Cap the margin so an interior region exists on small cutouts.
    margin_pix = min(margin_pix, max(1, min(H, W) // 4))
    wcs = tim.getWcs()
    n_thawed = 0
    seed_shapes: dict[int, object] = {}
    for i, src in enumerate(sources):
        if not fit_quality[i] or isinstance(src, (PointSource, SimpleGalaxy)):
            continue
        # LSQR can step a FixedCompositeGalaxy logre to where exp(logre)
        # overflows; the selection tree already fit its shape.
        if isinstance(src, FixedCompositeGalaxy):
            continue
        px, py = wcs.positionToPixel(src.getPosition())
        if margin_pix < px < W - margin_pix and margin_pix < py < H - margin_pix:
            _configure_shape_steps(src.getShape())
            src.thawAllParams()
            src.freezeParam("pos")
            seed_shapes[i] = _copy.deepcopy(src.getShape())
            n_thawed += 1

    def _optimize_loop():
        for _ in range(n_iter):
            try:
                dlnp, _, _ = tractor.optimize()
            except OverflowError:
                # EllipseESoft.re = exp(logre); LSQR can overflow it in
                # degenerate shape directions.
                warnings.warn(
                    "LSQR shape optimization overflowed; abandoning the "
                    "remaining step-2 iterations. Step-1 (forced-phot) "
                    "fluxes are unaffected.",
                    stacklevel=2,
                )
                break
            if dlnp < dlnp_min:
                break

    _optimize_loop()

    # The optimizer enforces no shape bounds; a runaway radius hides its
    # flux in extrapolated wings.
    n_railed = _revert_railed_shapes(sources, seed_shapes)
    if n_railed:
        warnings.warn(
            f"step 2 reverted {n_railed} runaway shape(s) to the prior "
            "value (radius past the per-source ceiling); fluxes refit "
            "with those shapes frozen.",
            stacklevel=2,
        )
        _optimize_loop()

    for i, src in enumerate(sources):
        if not fit_quality[i]:
            continue
        flux = _flux_ujy(src, band)
        ref = (initial_fluxes_ujy[i]
               if initial_fluxes_ujy is not None else None)
        bad = flux < 0
        if ref is not None and ref > 0 and abs(flux) > 100 * ref:
            bad = True
        if bad:
            fit_quality[i] = False

    return fit_quality, n_thawed


def refine_positions(tractor, tim, sources, fit_quality, *,
                     band: str = "VIS",
                     flux_floor_ujy: float = 10.0,
                     max_shift_arcsec: float = 0.2,
                     margin_pix: int = 50,
                     n_iter: int = 30,
                     dlnp_min: float = 0.1):
    """Optional step: let bright, interior sources recenter, with a hard bound.

    Thaws the centroid of sources with ``flux > flux_floor_ujy`` at least
    ``margin_pix`` from the edge, then jointly re-optimizes flux + position.
    Moves beyond ``max_shift_arcsec`` revert to the prior position: the
    genuine Q1 catalog-to-centroid offset is sub-pixel, so a larger move is
    a runaway. Fluxes are
    re-settled at the final positions, which later NISP/unWISE fits share.

    Returns ``(fit_quality, n_refined, shift_arcsec)`` where ``shift_arcsec``
    is aligned 1:1 with ``sources`` (0 where not thawed or reverted).
    """
    from tractor import RaDecPos

    H, W = tim.shape
    margin_pix = min(margin_pix, max(1, min(H, W) // 4))
    wcs = tim.getWcs()
    orig = [(float(s.getPosition().ra), float(s.getPosition().dec))
            for s in sources]

    thawed = []
    for src in sources:
        src.freezeAllParams()
        src.thawParam("brightness")
    for i, src in enumerate(sources):
        if not fit_quality[i] or _flux_ujy(src, band) < flux_floor_ujy:
            continue
        px, py = wcs.positionToPixel(src.getPosition())
        if margin_pix < px < W - margin_pix and margin_pix < py < H - margin_pix:
            src.thawParam("pos")
            thawed.append(i)

    shift_arcsec = np.zeros(len(sources))
    if not thawed:
        return fit_quality, 0, shift_arcsec

    for _ in range(n_iter):
        try:
            dlnp, _, _ = tractor.optimize()
        except OverflowError:
            warnings.warn(
                "LSQR position optimization overflowed; abandoning the "
                "remaining iterations. Step-1 fluxes are unaffected.",
                stacklevel=2)
            break
        if dlnp < dlnp_min:
            break

    cosd = float(np.cos(np.radians(np.median([o[1] for o in orig]))))
    n_reverted = 0
    for i in thawed:
        p = sources[i].getPosition()
        o_ra, o_dec = orig[i]
        dra = (float(p.ra) - o_ra) * cosd * 3600.0
        ddec = (float(p.dec) - o_dec) * 3600.0
        s = float(np.hypot(dra, ddec))
        if s > max_shift_arcsec:
            sources[i].pos = RaDecPos(o_ra, o_dec)
            n_reverted += 1
        else:
            shift_arcsec[i] = s

    for src in sources:
        src.freezeAllParams()
        src.thawParam("brightness")
    for _ in range(10):
        dlnp, _, _ = tractor.optimize()
        if dlnp < dlnp_min:
            break

    return fit_quality, len(thawed) - n_reverted, shift_arcsec


def measure_residual(tractor, tim, footprint=None):
    """Compute model + residual arrays.

    Returns
    -------
    dict with ``model``, ``residual``, ``data``, ``chi`` ndarrays and (if
    ``footprint`` is given) the sigma-clipped sigma outside ``footprint``.
    """
    from astropy.stats import sigma_clipped_stats
    model = tractor.getModelImage(0)
    data = tim.getImage()
    residual = data - model
    chi = tractor.getChiImage(imgi=0)
    out = {"model": model, "residual": residual, "data": data, "chi": chi}
    if footprint is not None:
        outside = ~footprint
        if outside.sum() > 100:
            _, _, sigma = sigma_clipped_stats(residual[outside], sigma=3.0)
            out["sigma_outside"] = float(sigma)
    return out
