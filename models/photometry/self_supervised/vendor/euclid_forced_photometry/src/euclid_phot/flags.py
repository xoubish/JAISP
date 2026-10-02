"""Per-source quality flags for the output catalog.

The brightest stars are biased high (the compact CATALOG-PSF cannot model
their spikes and halo), faint sources in those halos inherit the
contamination, and FLG-flagged pixels carry no usable signal. These flags
mark the affected rows.
"""
from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from .config import MER_VIS_BAD_BITS, MER_VIS_STARSIGNAL

_USE_DEFAULT_BAD_BITS = object()


def flag_sources(ra, dec, fluxes_ujy, is_star, *,
                 wcs=None, flag_plane=None, shape=None,
                 bright_star_ujy: float = 80.0,
                 neighbor_radius_arcsec: float = 6.0,
                 core_radius_pix: int = 3,
                 edge_margin_pix: int = 10,
                 flag_bad_bits=_USE_DEFAULT_BAD_BITS,
                 pixel_mask=None) -> Table:
    """Compute per-source quality flags.

    Parameters
    ----------
    ra, dec : array-like (deg)
        Source positions.
    fluxes_ujy : array-like
        Per-source flux in the band that fixes the positions (e.g. VIS),
        microJansky. Used to find the bright stars.
    is_star : array-like of bool
        Star/galaxy classification (e.g. the MER ``is_star`` column).
    wcs, flag_plane : optional
        The cutout WCS and the MER FLG bitmask. With a WCS, a source is
        flagged ``masked`` if any pixel within ``core_radius_pix`` of its
        center is set in the flag plane or in ``pixel_mask`` (the boolean
        mask of pixels given zero weight in the fit).
    shape : tuple, optional
        Cutout shape ``(H, W)`` for the ``edge`` flag; defaults to
        ``flag_plane.shape`` when a flag plane is given.
    bright_star_ujy : float
        Stars brighter than this are flagged ``bright_star``; at Euclid VIS
        depth this is roughly where the unmodeled spikes/halo bias the flux
        by more than a percent.
    neighbor_radius_arcsec : float
        A (non-bright) source within this distance of a bright star is flagged
        ``near_bright_star``.
    core_radius_pix : int
        Half-size of the square pixel box checked against ``flag_plane``.
    edge_margin_pix : int
        A source whose center lies within this many pixels of the cutout
        boundary (or off it) is flagged ``edge``; its clipped profile biases
        the flux low. The default 10 pixels (1 arcsec on the MER grid)
        covers typical Q1 galaxies.
    flag_bad_bits : int or None
        Bitmask of FLG bits to treat as bad. Defaults to ``MER_VIS_BAD_BITS``
        (INVALID | SAT | NO_DATA). ``None`` means any non-zero bit; with a
        real MER FLG plane the OBJECTS bit (24) then flags essentially every
        catalog row as ``masked``.

    Returns
    -------
    astropy.table.Table
        Columns ``bright_star``, ``near_bright_star``, ``masked``, ``edge``
        (all bool), and ``reliable`` (bool) = none of the four set.
    """
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    flux = np.atleast_1d(np.asarray(fluxes_ujy, dtype=float))
    is_star = np.atleast_1d(np.asarray(is_star, dtype=bool))
    n = len(ra)

    bright_star = is_star & np.isfinite(flux) & (flux > bright_star_ujy)

    near_bright_star = np.zeros(n, dtype=bool)
    if bright_star.any():
        coords = SkyCoord(ra, dec, unit="deg")
        _, sep, _ = coords.match_to_catalog_sky(coords[bright_star])
        near_bright_star = (sep.arcsec < neighbor_radius_arcsec) & ~bright_star

    masked = np.zeros(n, dtype=bool)
    bad = None
    if flag_plane is not None:
        fp = np.asarray(flag_plane)
        bits = MER_VIS_BAD_BITS if flag_bad_bits is _USE_DEFAULT_BAD_BITS else flag_bad_bits
        bad = (fp != 0) if bits is None else ((fp & int(bits)) != 0)
    if pixel_mask is not None:
        pm = np.asarray(pixel_mask, dtype=bool)
        bad = pm if bad is None else (bad | pm)
    if wcs is not None and bad is not None:
        H, W = bad.shape
        r = int(core_radius_pix)
        for i in range(n):
            px, py = wcs.world_to_pixel_values(ra[i], dec[i])
            ix, iy = int(round(float(px))), int(round(float(py)))
            y0, y1 = max(0, iy - r), min(H, iy + r + 1)
            x0, x1 = max(0, ix - r), min(W, ix + r + 1)
            if y0 < y1 and x0 < x1:
                masked[i] = bool(bad[y0:y1, x0:x1].any())

    edge = np.zeros(n, dtype=bool)
    if shape is None and flag_plane is not None:
        shape = np.asarray(flag_plane).shape
    if wcs is not None and shape is not None:
        H, W = shape
        m = float(edge_margin_pix)
        for i in range(n):
            px, py = wcs.world_to_pixel_values(ra[i], dec[i])
            edge[i] = not (m <= float(px) < W - m
                           and m <= float(py) < H - m)

    reliable = ~(bright_star | near_bright_star | masked | edge)
    return Table({
        "bright_star": bright_star,
        "near_bright_star": near_bright_star,
        "masked": masked,
        "edge": edge,
        "reliable": reliable,
    })


def blend_flags(ra, dec, fluxes_ujy, *,
                radius_arcsec: float = 1.0,
                flux_ratio: float = 0.1,
                profile_re_arcsec=None) -> Table:
    """Per-source blending / crowding flags.

    The exported conditional errors omit off-diagonal information in the
    normal matrix, understating uncertainty in strongly blended pairs;
    these flags mark the affected rows.

    Parameters
    ----------
    ra, dec : array-like (deg)
    fluxes_ujy : array-like
        Fluxes in the band that defined the source list (e.g. VIS). NaN
        fluxes are treated as 0 for the ratio test (an undetected neighbor
        cannot meaningfully contaminate).
    radius_arcsec : float
        Neighbor search radius. The default 1.0 arcsec is ~6x the VIS PSF
        FWHM; pairs closer than this share significant PSF overlap at VIS
        resolution. Scale it up for a NISP-prior fit.
    flux_ratio : float
        A source is ``blended`` when a neighbor within ``radius_arcsec``
        has flux > ``flux_ratio`` x its own (default 0.1: a 10x fainter
        neighbor perturbs the fit at or below the percent level).
    profile_re_arcsec : array-like, optional
        Per-source model effective radius (arcsec). When given, a source is
        additionally flagged ``blended`` if any other source lies within its
        effective radius: the flux split is then set by the deblending
        model, not by distinct pixels, so no flux-ratio test applies.

    Returns
    -------
    astropy.table.Table
        Columns ``blended`` (bool), ``n_neighbors`` (int, within radius),
        ``nearest_arcsec`` (float, inf when no neighbor in radius).
    """
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    flux = np.nan_to_num(
        np.atleast_1d(np.asarray(fluxes_ujy, dtype=float)), nan=0.0)
    n = len(ra)

    blended = np.zeros(n, dtype=bool)
    n_neighbors = np.zeros(n, dtype=int)
    nearest = np.full(n, np.inf)
    if n > 1:
        import astropy.units as u
        coords = SkyCoord(ra, dec, unit="deg")
        i1, i2, sep, _ = coords.search_around_sky(
            coords, radius_arcsec * u.arcsec)
        off = i1 != i2                      # drop self-matches
        i1, i2, sep = i1[off], i2[off], sep.arcsec[off]
        for a, b, s in zip(i1, i2, sep, strict=True):
            n_neighbors[a] += 1
            if s < nearest[a]:
                nearest[a] = s
            if flux[b] > flux_ratio * flux[a]:
                blended[a] = True

        if profile_re_arcsec is not None:
            pre = np.clip(np.nan_to_num(np.atleast_1d(
                np.asarray(profile_re_arcsec, dtype=float))), 0.0, None)
            r_search = float(pre.max())
            if r_search > 0:
                j1, j2, sep2, _ = coords.search_around_sky(
                    coords, max(r_search, radius_arcsec) * u.arcsec)
                off2 = j1 != j2
                j1, sep2 = j1[off2], sep2.arcsec[off2]
                inside = sep2 < pre[j1]
                np.logical_or.at(blended, j1[inside], True)

    return Table({
        "blended": blended,
        "n_neighbors": n_neighbors,
        "nearest_arcsec": nearest,
    })


def probable_bright_stars(mer_cat) -> np.ndarray:
    """Boolean flag of MER rows to treat as stars for bright-star masking.

    ``point_like_prob > 0.96``, plus abstained rows (masked
    ``point_like_prob``, which includes the brightest stars) where
    ``flux_vis_psf >= flux_vis_sersic``: a point source's PSF-model flux
    captures essentially all its light while an extended galaxy's is
    core-only.

    Returns a bool array aligned with ``mer_cat``; pass to
    :func:`bright_star_pixel_mask` with ``flux_vis_psf`` as the brightness.
    """
    plp = mer_cat["point_like_prob"]
    abstained = (np.asarray(plp.mask, dtype=bool)
                 if hasattr(plp, "mask") else np.zeros(len(mer_cat), bool))
    plp_val = np.asarray(plp.filled(0.0) if hasattr(plp, "filled") else plp,
                         dtype=float)
    classified_star = ~abstained & (plp_val > 0.96)

    def _col(name):
        c = mer_cat[name]
        return np.asarray(c.filled(np.nan) if hasattr(c, "filled") else c,
                          dtype=float)

    fpsf = _col("flux_vis_psf")
    fser = _col("flux_vis_sersic")
    with np.errstate(invalid="ignore"):
        abstained_star = (abstained & np.isfinite(fpsf) & np.isfinite(fser)
                          & (fpsf >= fser))
    return classified_star | abstained_star


def starsignal_pixel_mask(cutout) -> np.ndarray | None:
    """Boolean bright-star pixel mask (True = drop from the fit) read from
    the MER FLG plane's STARSIGNAL bit.

    STARSIGNAL marks the footprints of bright-star halos and diffraction
    spikes -- real light the compact CATALOG-PSF stamps cannot model.
    Zeroing the inverse variance on these pixels (pass the mask to
    ``build_tractor_image(..., pixel_mask=...)``) keeps that light from
    biasing neighboring sources; the covered stars are themselves not
    measured.

    Returns None when the cutout carries no FLG plane (fetch it with
    ``fetch_cutout(..., with_flag=True)``); ``bright_star_pixel_mask``
    below is the geometric fallback for that case.
    """
    if str(cutout.band).upper() != "VIS":
        raise ValueError("STARSIGNAL is a VIS flag; NISP bit 18 is GHOST, "
                         "not a bright-star footprint")
    flag_plane = getattr(cutout, "flag", None)
    if flag_plane is None:
        return None
    return (np.asarray(flag_plane) & MER_VIS_STARSIGNAL) != 0


def bright_star_pixel_mask(shape, wcs, ra, dec, fluxes_ujy, is_star, *,
                           bright_ujy: float = 80.0,
                           core_keep_arcsec: float = 0.5,
                           halo_radius_arcsec: float = 1.5,
                           halo_max_arcsec: float = 6.0,
                           spike_angles_deg=None,
                           spike_length_factor: float = 2.5,
                           spike_halfwidth_arcsec: float = 0.15) -> np.ndarray:
    """Boolean pixel mask (True = drop from the fit) over bright-star halos
    and, optionally, their diffraction spikes.

    The MER CATALOG-PSF stamps span only ~±1 arcsec, so a bright star's
    halo and spikes are unmodeled light: the star's own flux comes out ~10%
    high and a neighbor on a spike inherits the bias. The recommended
    defense is the MER FLG STARSIGNAL bit (set ``mask_bright_stars=True``);
    this function is the geometric fallback for data without the FLG plane
    (cf. the Legacy Surveys BRIGHT/MEDIUM maskbits).

    The star's core (r <= ``core_keep_arcsec``) is not masked, since the PSF
    model is good there and the flux must stay constrained. Masked is the
    annulus ``core_keep < r <= R`` plus optional spike rectangles, with

        R = halo_radius_arcsec * sqrt(flux / bright_ujy),  capped at
        ``halo_max_arcsec``.

    Parameters
    ----------
    shape : (H, W)
        Image shape the mask is built for.
    wcs : astropy.wcs.WCS
        WCS of that image.
    ra, dec, fluxes_ujy, is_star : array-like
        Source positions, prior-band fluxes (microJansky; e.g. MER
        ``flux_vis_psf`` for stars), and the star classification.
    bright_ujy : float
        Stars brighter than this get masked (same default as
        :func:`flag_sources`, where the halo bias exceeds ~1%).
    spike_angles_deg : sequence of float, optional
        Position angles of the diffraction spikes, degrees counter-clockwise
        from the +x image axis; each angle masks a full two-sided arm. The
        VIS spike orientation depends on the observation roll angle, so
        there is no universal default. ``None`` masks halos only.
    spike_length_factor : float
        Spike arm half-length, in units of the star's halo radius ``R``.
    spike_halfwidth_arcsec : float
        Spike arm half-width.

    Returns
    -------
    ndarray of bool, ``shape``, True where the pixel should get invvar = 0.
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    flux = np.atleast_1d(np.asarray(fluxes_ujy, dtype=float))
    is_star = np.atleast_1d(np.asarray(is_star, dtype=bool))

    pixscale = float(np.abs(wcs.pixel_scale_matrix[0, 0]) * 3600.0)
    bright = is_star & np.isfinite(flux) & (flux > bright_ujy)
    if not bright.any():
        return mask

    for i in np.where(bright)[0]:
        px, py = wcs.world_to_pixel_values(ra[i], dec[i])
        px, py = float(px), float(py)
        r_halo = min(halo_radius_arcsec * np.sqrt(flux[i] / bright_ujy),
                     halo_max_arcsec) / pixscale
        r_core = core_keep_arcsec / pixscale
        r_max = (spike_length_factor * r_halo
                 if spike_angles_deg is not None else r_halo)
        x0 = max(0, int(np.floor(px - r_max - 1)))
        x1 = min(W, int(np.ceil(px + r_max + 2)))
        y0 = max(0, int(np.floor(py - r_max - 1)))
        y1 = min(H, int(np.ceil(py + r_max + 2)))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx = xx - px
        dy = yy - py
        rr = np.hypot(dx, dy)
        local = (rr > r_core) & (rr <= r_halo)
        if spike_angles_deg is not None:
            hw = max(spike_halfwidth_arcsec / pixscale, 1.0)
            arm_len = spike_length_factor * r_halo
            for ang in np.atleast_1d(spike_angles_deg):
                t = np.radians(float(ang))
                # Distance along (u) and across (v) the two-sided arm.
                u = dx * np.cos(t) + dy * np.sin(t)
                v = -dx * np.sin(t) + dy * np.cos(t)
                local |= ((np.abs(v) <= hw) & (np.abs(u) <= arm_len)
                          & (rr > r_core))
        mask[y0:y1, x0:x1] |= local
    return mask
