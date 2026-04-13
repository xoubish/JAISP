"""
VIS-based star selection for PSF training.

Pipeline
--------
1. Load a Rubin+Euclid tile pair.
2. Detect compact sources in the Euclid VIS mosaic (highest resolution).
3. Fit a 2D Gaussian (via iterative moments) to each detection for FWHM +
   sub-pixel centroid.
4. Apply physical cuts:
     - saturation  (peak below empirical limit, no flagged pixels)
     - stellar locus (FWHM within ±`stellar_locus_width` of median)
     - isolation  (no neighbour within `isolation_arcsec`)
5. Cross-match VIS sub-pixel centroid → Rubin pixel coords via WCS.
6. Extract 10-band stamps at sub-pixel positions (all centred on same sky
   point) and return a StarCatalog.

Output
------
`StarCatalog` carries everything `train_psf_field` needs: stamps, rms maps,
sub-pixel centroid init (≈0 since stamps are already sub-pixel-centred),
normalised tile position, and per-star diagnostics (FWHM, SNR, peak).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from astropy.io.fits import Header
from astropy.wcs import WCS

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
# Put the two sibling modules' directories (not the package inits) on sys.path
# so their internal absolute imports (e.g. `from psf_net import ...`) resolve.
for _p in (_MODELS / 'photometry', _MODELS / 'astrometry2', _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stamp_extractor import extract_stamps                                # noqa: E402
from source_matching import (                                             # noqa: E402
    detect_sources,
    safe_header_from_card_string,
)

from .psf_field import BAND_ORDER, N_BANDS                                # noqa: E402


_EUCLID_BANDS = ['VIS', 'Y', 'J', 'H']
VIS_IDX = BAND_ORDER.index('euclid_VIS')

# Arcsec per pixel — must match PSFField conventions
RUBIN_PX_ARCSEC = 0.2
EUCLID_PX_ARCSEC = 0.1


# ---------------------------------------------------------------------------
# StarCatalog — what select_stars returns
# ---------------------------------------------------------------------------

@dataclass
class StarCatalog:
    """All per-star data needed for PSFField training."""
    stamps:              torch.Tensor   # [N, 10, S, S]
    rms_stamps:          torch.Tensor   # [N, 10, S, S]
    centroid_init_arcsec: torch.Tensor  # [N, 2]   sub-pixel offset init (≈0)
    tile_pos:            torch.Tensor   # [N, 2]   normalised to [0, 1]²
    sed_init:            torch.Tensor   # [N, 10]  log-flux per band (crude)
    # Diagnostics
    vis_fwhm_arcsec:     torch.Tensor   # [N]
    vis_peak:            torch.Tensor   # [N]
    vis_snr:             torch.Tensor   # [N]
    rubin_xy_px:         torch.Tensor   # [N, 2]   sub-pixel Rubin coords
    vis_xy_px:           torch.Tensor   # [N, 2]   sub-pixel VIS coords

    def __len__(self) -> int:
        return self.stamps.shape[0]


# ---------------------------------------------------------------------------
# Tile loading
# ---------------------------------------------------------------------------

def _var_to_rms(var: np.ndarray) -> np.ndarray:
    """sqrt(var) with bad/zero pixels filled by per-band median rms."""
    rms = np.sqrt(np.clip(var, 0.0, None))
    bands = rms if rms.ndim == 3 else rms[None]
    out = bands.copy()
    for bi in range(bands.shape[0]):
        good = bands[bi] > 1e-6
        med = float(np.median(bands[bi][good])) if good.any() else 1.0
        out[bi] = np.where(good, bands[bi], med)
    return out if rms.ndim == 3 else out[0]


def _load_rubin(path: Path) -> Tuple[torch.Tensor, torch.Tensor, WCS]:
    """Returns (img [6,H,W], rms [6,H,W], WCS) as float32 tensors."""
    data = np.load(path, allow_pickle=True, mmap_mode='r')
    img = np.nan_to_num(np.asarray(data['img'], dtype=np.float32), nan=0.0)
    var = np.nan_to_num(np.asarray(data['var'], dtype=np.float32), nan=0.0)
    rms = _var_to_rms(var)
    wcs = WCS(Header(data['wcs_hdr'].item()))
    return torch.from_numpy(img), torch.from_numpy(rms), wcs


def _load_euclid(path: Path) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Returns (img [4,H,W], rms [4,H,W], list-of-WCS)."""
    data = np.load(path, allow_pickle=True, mmap_mode='r')
    imgs, rmss, wcss = [], [], []
    for band in _EUCLID_BANDS:
        img = np.nan_to_num(np.asarray(data[f'img_{band}'], dtype=np.float32), nan=0.0)
        var = np.nan_to_num(np.asarray(data[f'var_{band}'], dtype=np.float32), nan=0.0)
        imgs.append(img)
        rmss.append(_var_to_rms(var))
        wcss.append(WCS(safe_header_from_card_string(str(data[f'wcs_{band}']))))
    return (
        torch.from_numpy(np.stack(imgs)),
        torch.from_numpy(np.stack(rmss)),
        wcss,
    )


# ---------------------------------------------------------------------------
# Batched 2D moment fit on VIS stamps
# ---------------------------------------------------------------------------

def _fit_moments_2d(
    stamps: torch.Tensor,       # [N, S, S]  background-subtracted
    inner_radius_px: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched first- and second-moment fit within a circular weighting region.

    Returns
    -------
    centroid_offset : [N, 2]  (dx, dy) from stamp centre in pixels
    sigma_px        : [N]      quadrature mean σ in pixels
    peak            : [N]      central-pixel value
    """
    N, S, _ = stamps.shape
    half = (S - 1) / 2.0
    device = stamps.device
    dtype = stamps.dtype

    yy, xx = torch.meshgrid(
        torch.arange(S, device=device, dtype=dtype) - half,
        torch.arange(S, device=device, dtype=dtype) - half,
        indexing='ij',
    )
    r2 = xx * xx + yy * yy                              # [S, S]
    mask = (r2 <= inner_radius_px * inner_radius_px).to(dtype)  # [S, S]

    # Clip negative pixels (noise) so they don't pull moments
    w = stamps.clamp(min=0.0) * mask                    # [N, S, S]
    total = w.sum(dim=(-2, -1)).clamp(min=1e-12)        # [N]

    x0 = (w * xx).sum(dim=(-2, -1)) / total             # [N] pixel offset x
    y0 = (w * yy).sum(dim=(-2, -1)) / total             # [N] pixel offset y

    # Re-centre for second moment
    dx = xx.unsqueeze(0) - x0.view(-1, 1, 1)
    dy = yy.unsqueeze(0) - y0.view(-1, 1, 1)
    r2c = dx * dx + dy * dy
    sig2 = (w * r2c).sum(dim=(-2, -1)) / (2.0 * total)  # quadrature σ²
    sigma = sig2.clamp(min=1e-8).sqrt()                 # [N]

    peak = stamps[:, int(half), int(half)]              # [N] central pixel

    return torch.stack([x0, y0], dim=-1), sigma, peak


# ---------------------------------------------------------------------------
# Isolation cut: reject stars with a neighbour within min_sep_arcsec
# ---------------------------------------------------------------------------

def _isolation_mask(
    xy_px: np.ndarray,
    px_scale_arcsec: float,
    min_sep_arcsec: float,
) -> np.ndarray:
    """
    Boolean mask: True if the source has no neighbour within min_sep_arcsec.
    O(N²) but N is few hundred — trivial.
    """
    if xy_px.shape[0] <= 1:
        return np.ones(xy_px.shape[0], dtype=bool)
    min_sep_px = min_sep_arcsec / px_scale_arcsec
    dx = xy_px[:, None, 0] - xy_px[None, :, 0]
    dy = xy_px[:, None, 1] - xy_px[None, :, 1]
    r2 = dx * dx + dy * dy
    np.fill_diagonal(r2, np.inf)
    return r2.min(axis=-1) > min_sep_px * min_sep_px


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def select_stars(
    rubin_path: Path,
    euclid_path: Path,
    stamp_size: int = 21,
    max_stars: int = 80,
    vis_detect_nsig: float = 20.0,
    vis_fit_stamp: int = 15,
    fit_radius_px: float = 4.0,           # moment fit aperture in VIS pixels
    stellar_locus_width: float = 0.08,    # ±8 % of median FWHM
    isolation_arcsec: float = 3.0,
    peak_saturation_quantile: float = 0.995,  # drop brightest 0.5 %
    detections_vis_px: Optional[np.ndarray] = None,   # [N, 2] float, optional
    device: Optional[torch.device] = None,
) -> Optional[StarCatalog]:
    """
    Select isolated unsaturated point sources in VIS and return 10-band stamps.

    Parameters
    ----------
    detections_vis_px : pre-computed VIS detection positions [N, 2] in
        VIS pixel coordinates (x, y). If provided, these replace the internal
        classical peak-finder — typically the CenterNet pseudo-labels which
        already include bright-core + spike-mask handling.
        If None, the classical `detect_sources` is run on the VIS image.

    Returns None if no stars survive the cuts.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load both tiles ----------------------------------------------------
    rubin_img, rubin_rms, rubin_wcs = _load_rubin(Path(rubin_path))
    euclid_img, euclid_rms, euclid_wcs_list = _load_euclid(Path(euclid_path))
    eH, eW = euclid_img.shape[1], euclid_img.shape[2]
    rH, rW = rubin_img.shape[1], rubin_img.shape[2]

    vis_np = euclid_img[0].numpy()   # VIS is first Euclid band
    vis_rms_np = euclid_rms[0].numpy()

    # --- Detection candidates in VIS ---------------------------------------
    # Either: use externally-provided detections (e.g. CenterNet pseudo-
    # labels, which are VIS-based with bright-core + spike-mask handling),
    # or fall back to a classical peak-finder on the VIS image.
    if detections_vis_px is not None and detections_vis_px.size > 0:
        xs_int = detections_vis_px[:, 0].astype(np.float32)
        ys_int = detections_vis_px[:, 1].astype(np.float32)
    else:
        xs_int, ys_int = detect_sources(
            vis_np, nsig=vis_detect_nsig, smooth_sigma=1.0,
            min_dist=7, max_sources=max_stars * 5,
        )
    if xs_int.size == 0:
        return None

    margin = max(vis_fit_stamp, stamp_size) // 2 + 2
    in_bounds = (
        (xs_int >= margin) & (xs_int < eW - margin) &
        (ys_int >= margin) & (ys_int < eH - margin)
    )
    xs_int, ys_int = xs_int[in_bounds], ys_int[in_bounds]
    if xs_int.size == 0:
        return None

    # --- Extract small VIS stamps for moment fit ----------------------------
    # Subtract a per-detection local background (annulus median) first
    fit_positions = torch.from_numpy(
        np.stack([xs_int, ys_int], axis=1).astype(np.float32)
    )
    vis_tensor = euclid_img[0:1].to(device)                       # [1, H, W]
    fit_stamps = extract_stamps(
        vis_tensor, fit_positions.to(device), vis_fit_stamp
    ).squeeze(1)                                                   # [N, S_fit, S_fit]

    # Annulus background
    S_fit = vis_fit_stamp
    half_f = (S_fit - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(S_fit, device=device, dtype=torch.float32) - half_f,
        torch.arange(S_fit, device=device, dtype=torch.float32) - half_f,
        indexing='ij',
    )
    r = (xx * xx + yy * yy).sqrt()
    ann = (r >= fit_radius_px + 1.0) & (r <= fit_radius_px + 3.0)
    bg = fit_stamps[:, ann].median(dim=-1).values                  # [N]
    fit_stamps_bg = fit_stamps - bg.view(-1, 1, 1)

    # Moment fit
    centroid_off, sigma_px, peak = _fit_moments_2d(
        fit_stamps_bg, inner_radius_px=fit_radius_px
    )
    centroid_off = centroid_off.cpu().numpy()
    sigma_px = sigma_px.cpu().numpy()
    peak = peak.cpu().numpy()

    # FWHM in arcsec (VIS)
    fwhm_arcsec = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_px * EUCLID_PX_ARCSEC
    snr = peak / np.maximum(vis_rms_np[ys_int.astype(int), xs_int.astype(int)], 1e-6)

    # --- Sub-pixel VIS position ---------------------------------------------
    vis_xy = np.stack([
        xs_int.astype(np.float32) + centroid_off[:, 0],
        ys_int.astype(np.float32) + centroid_off[:, 1],
    ], axis=1)

    # Reject moments that ran away (non-compact detections)
    sane = (np.abs(centroid_off).max(axis=1) < 1.5) & np.isfinite(fwhm_arcsec)
    vis_xy = vis_xy[sane]
    fwhm_arcsec = fwhm_arcsec[sane]
    peak = peak[sane]
    snr = snr[sane]
    if vis_xy.shape[0] == 0:
        return None

    # --- Cuts ---------------------------------------------------------------
    # 1. Saturation: drop top quantile peaks AND require SNR > 15
    peak_thresh = float(np.quantile(peak, peak_saturation_quantile))
    keep_sat = (peak < peak_thresh) & (snr > 15.0) & (peak > 0)

    # 2. Stellar locus — tight clip around median FWHM of the unsaturated set
    if keep_sat.sum() < 5:
        return None
    median_fwhm = float(np.median(fwhm_arcsec[keep_sat]))
    locus_lo = median_fwhm * (1.0 - stellar_locus_width)
    locus_hi = median_fwhm * (1.0 + stellar_locus_width)
    keep_locus = (fwhm_arcsec >= locus_lo) & (fwhm_arcsec <= locus_hi)

    # 3. Isolation (in VIS arcsec space)
    keep_iso = _isolation_mask(vis_xy, EUCLID_PX_ARCSEC, isolation_arcsec)

    keep = keep_sat & keep_locus & keep_iso
    if keep.sum() == 0:
        return None

    vis_xy      = vis_xy[keep]
    fwhm_arcsec = fwhm_arcsec[keep]
    peak        = peak[keep]
    snr         = snr[keep]

    # Cap to max_stars (sort by SNR desc to keep best signal)
    if vis_xy.shape[0] > max_stars:
        order = np.argsort(-snr)[:max_stars]
        vis_xy, fwhm_arcsec, peak, snr = (
            vis_xy[order], fwhm_arcsec[order], peak[order], snr[order]
        )

    N = vis_xy.shape[0]

    # --- Project VIS sub-pixel position → Rubin pixel coords ----------------
    ra, dec = euclid_wcs_list[0].wcs_pix2world(vis_xy[:, 0], vis_xy[:, 1], 0)
    rx, ry = rubin_wcs.wcs_world2pix(ra, dec, 0)
    rubin_xy = np.stack([rx, ry], axis=1).astype(np.float32)

    # Reject stars that fall outside the Rubin tile
    rmargin = stamp_size // 2 + 2
    in_rubin = (
        (rubin_xy[:, 0] >= rmargin) & (rubin_xy[:, 0] < rW - rmargin) &
        (rubin_xy[:, 1] >= rmargin) & (rubin_xy[:, 1] < rH - rmargin)
    )
    if in_rubin.sum() == 0:
        return None
    vis_xy      = vis_xy[in_rubin]
    rubin_xy    = rubin_xy[in_rubin]
    fwhm_arcsec = fwhm_arcsec[in_rubin]
    peak        = peak[in_rubin]
    snr         = snr[in_rubin]
    N = vis_xy.shape[0]

    # --- Extract 10-band stamps at sub-pixel positions ----------------------
    rubin_img_d = rubin_img.to(device)
    rubin_rms_d = rubin_rms.to(device)
    euclid_img_d = euclid_img.to(device)
    euclid_rms_d = euclid_rms.to(device)

    r_pos = torch.from_numpy(rubin_xy).to(device)
    rubin_stamps = extract_stamps(rubin_img_d, r_pos, stamp_size)        # [N,6,S,S]
    rubin_rms_s  = extract_stamps(rubin_rms_d, r_pos, stamp_size)

    # Project VIS sub-pixel → each Euclid band's pixel grid separately.
    # In MER mosaics all 4 Euclid bands share the same WCS grid → identity,
    # but handle the general case.
    euclid_stamps_list = []
    euclid_rms_list = []
    for bi, wcs in enumerate(euclid_wcs_list):
        if bi == 0:
            ex, ey = vis_xy[:, 0], vis_xy[:, 1]
        else:
            ex, ey = wcs.wcs_world2pix(ra[in_rubin], dec[in_rubin], 0)
        e_pos = torch.from_numpy(
            np.stack([ex, ey], axis=1).astype(np.float32)
        ).to(device)
        euclid_stamps_list.append(
            extract_stamps(euclid_img_d[bi:bi+1], e_pos, stamp_size).squeeze(1)
        )
        euclid_rms_list.append(
            extract_stamps(euclid_rms_d[bi:bi+1], e_pos, stamp_size).squeeze(1)
        )
    euclid_stamps = torch.stack(euclid_stamps_list, dim=1)               # [N,4,S,S]
    euclid_rms_s  = torch.stack(euclid_rms_list,    dim=1)

    stamps     = torch.cat([rubin_stamps, euclid_stamps], dim=1)          # [N,10,S,S]
    rms_stamps = torch.cat([rubin_rms_s,  euclid_rms_s],  dim=1)

    # --- Background-subtract stamps (annulus median per-band) ----------------
    S = stamp_size
    half_s = (S - 1) / 2.0
    yy2, xx2 = torch.meshgrid(
        torch.arange(S, device=device, dtype=torch.float32) - half_s,
        torch.arange(S, device=device, dtype=torch.float32) - half_s,
        indexing='ij',
    )
    r_full = (xx2 * xx2 + yy2 * yy2).sqrt()
    ann_full = (r_full > 7.0) & (r_full <= 9.5)
    bg_all = stamps[..., ann_full].median(dim=-1).values                  # [N, 10]
    stamps = stamps - bg_all.unsqueeze(-1).unsqueeze(-1)

    # --- Crude per-band fluxes for SED init (sum within 3-px radius) --------
    inner_mask = (r_full < 3.0).to(stamps.dtype)
    flux_crude = (stamps * inner_mask).sum(dim=(-2, -1)).clamp(min=1e-3)  # [N,10]
    log_flux = torch.log10(flux_crude)
    # Normalise per star (zero-mean across bands) — SED encoder expects this
    sed_init = log_flux - log_flux.mean(dim=-1, keepdim=True)

    # --- Centroid init: 0 since stamps are already sub-pixel-centred -------
    centroid_init_arcsec = torch.zeros(N, 2, device=device, dtype=torch.float32)

    # --- Tile position (use VIS coords for consistency) --------------------
    tile_pos = torch.from_numpy(np.stack([
        vis_xy[:, 0] / max(eW - 1, 1),
        vis_xy[:, 1] / max(eH - 1, 1),
    ], axis=1).astype(np.float32)).to(device)

    # All numpy arrays below have already been filtered by `in_rubin`
    return StarCatalog(
        stamps=stamps.detach(),
        rms_stamps=rms_stamps.detach(),
        centroid_init_arcsec=centroid_init_arcsec,
        tile_pos=tile_pos.detach(),
        sed_init=sed_init.detach(),
        vis_fwhm_arcsec=torch.from_numpy(fwhm_arcsec.astype(np.float32)).to(device),
        vis_peak=torch.from_numpy(peak.astype(np.float32)).to(device),
        vis_snr=torch.from_numpy(snr.astype(np.float32)).to(device),
        rubin_xy_px=torch.from_numpy(rubin_xy.astype(np.float32)).to(device),
        vis_xy_px=torch.from_numpy(vis_xy.astype(np.float32)).to(device),
    )
