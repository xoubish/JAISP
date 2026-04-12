"""Direct global PINN concordance from raw centroid offsets.

Bypasses the per-patch neural matcher entirely.  Uses raw centroid-based
offsets (VIS position − Rubin position) from matched sources and fits a
physics-informed smooth field directly.

Key insight (diagnostic analysis, 2026-04-11): the concordance field
signal is ~5 mas while per-source centroid noise is ~47 mas at typical
SNR=10–30.  The per-patch NN matcher cannot beat centroid noise.  The
smooth PINN field naturally averages over many sources, recovering the
sub-noise signal through spatial coherence and physics constraints.

Two modes
---------
Quick mode (--cache):
    Uses existing prediction cache (raw centroid offsets already computed).
    Computes weights from per-tile residual scatter.  Fast (~30 s).

Full mode (--rubin-dir / --euclid-dir):
    Reprocesses all tiles with PSF-fit centroiding to get per-source SNR.
    Weights from King (1983) formula: w = 1/σ²_centroid.  Slow (~20 min).

Usage
-----
    # Quick
    python fit_direct_pinn.py \\
        --cache ../checkpoints/astro_v7_psffit_all/predictions_cache_790.npz \\
        --output ../checkpoints/astro_v7_psffit_all/concordance_direct_pinn.fits

    # Full (proper SNR weights)
    python fit_direct_pinn.py \\
        --rubin-dir ../../data/rubin_tiles_all \\
        --euclid-dir ../../data/euclid_tiles_all \\
        --output concordance_direct_pinn.fits \\
        --save-anchors anchors_790.npz
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_MODELS, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from astrometry2.source_matching import (
    RUBIN_BAND_ORDER,
    _to_float32,
    build_detection_image,
    detect_sources,
    match_sources_wcs,
    refine_centroids_psf_fit,
    robust_sigma,
    safe_header_from_card_string,
)
from astrometry2.dataset import (
    BAND_TO_IDX,
    NISP_BAND_ORDER,
    discover_tile_pairs,
)
from astrometry2.pinn_field_solver import fit_pinn_field, evaluate_pinn_mesh


# ============================================================
# Anchor collection: full reprocessing with PSF-fit SNR
# ============================================================

def _process_band_anchors(
    band_img, band_wcs, vis_refined, vis_snr, vis_csig,
    vwcs, matched_band_xy, psf_radius, fwhm_guess, pixel_scale,
    band_id, tile_id,
):
    """Compute offsets for one band at matched positions."""
    VIS_PX = 0.1
    band_refined, band_snr, band_csig = refine_centroids_psf_fit(
        band_img, matched_band_xy, radius=psf_radius, fwhm_guess=fwhm_guess)
    b_ra, b_dec = band_wcs.wcs_pix2world(band_refined[:, 0], band_refined[:, 1], 0)
    v_ra, v_dec = vwcs.wcs_pix2world(vis_refined[:, 0], vis_refined[:, 1], 0)
    dra = (v_ra - b_ra) * np.cos(np.deg2rad(v_dec)) * 3600.0
    ddec = (v_dec - b_dec) * 3600.0
    sigma_b = band_csig * pixel_scale
    sigma_v = vis_csig * VIS_PX
    sigma_total = np.sqrt(sigma_b**2 + sigma_v**2 + 0.005**2).astype(np.float32)
    N = len(dra)
    return (v_ra, v_dec, dra.astype(np.float32), ddec.astype(np.float32),
            band_snr.astype(np.float32), vis_snr.astype(np.float32),
            sigma_total, np.full(N, band_id, dtype=np.int32),
            np.array([tile_id] * N, dtype=object))


def collect_raw_anchors(
    pairs: List[Tuple[str, str, str]],
    target_bands: List[str] = ('u', 'g', 'r', 'i', 'z', 'y'),
    include_nisp: bool = True,
    detect_bands: List[str] = ('g', 'r', 'i', 'z'),
    max_sep_arcsec: float = 0.12,
    clip_sigma: float = 3.5,
    max_matches: int = 256,
    psf_radius: int = 5,
    fwhm_guess_rubin: float = 3.0,
    fwhm_guess_vis: float = 4.0,
    fwhm_guess_nisp: float = 4.0,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Process all tiles: detect, match, PSF-fit → anchor table with SNR.

    Handles 6 Rubin bands + 3 NISP bands, all compared against VIS.
    """
    from astropy.wcs import WCS

    detect_bands_full = [f'rubin_{b}' for b in detect_bands]
    acc = {k: [] for k in ('ra', 'dec', 'dra', 'ddec', 'snr_band',
                            'snr_vis', 'sigma', 'band_idx', 'tile_id')}

    def _append(result):
        ra, dec, dra, ddec, snr_b, snr_v, sig, bidx, tids = result
        acc['ra'].append(ra); acc['dec'].append(dec)
        acc['dra'].append(dra); acc['ddec'].append(ddec)
        acc['snr_band'].append(snr_b); acc['snr_vis'].append(snr_v)
        acc['sigma'].append(sig); acc['band_idx'].append(bidx)
        acc['tile_id'].append(tids)

    RUBIN_PX, NISP_PX = 0.2, 0.1
    n_processed = 0
    t0 = time.time()

    for tile_id, rubin_path, euclid_path in pairs:
        try:
            rdata = np.load(rubin_path, allow_pickle=True)
            edata = np.load(euclid_path, allow_pickle=True)
            rubin_cube = rdata['img']
            vis_img = np.nan_to_num(_to_float32(edata['img_VIS']), nan=0.0)
            rwcs = WCS(rdata['wcs_hdr'].item())
            vwcs = WCS(safe_header_from_card_string(edata['wcs_VIS'].item()))
        except Exception:
            continue

        # VIS detections (shared across all bands)
        vx, vy = detect_sources(vis_img, nsig=4.0, smooth_sigma=1.2,
                                min_dist=9, max_sources=800)

        # ── Rubin bands ──────────────────────────────────────────────
        rubin_det = build_detection_image(rubin_cube, detect_bands_full)
        rx, ry = detect_sources(rubin_det, nsig=4.5, smooth_sigma=1.0,
                                min_dist=7, max_sources=600)
        matched = match_sources_wcs(rx, ry, vx, vy, rwcs, vwcs,
                                    max_sep_arcsec=max_sep_arcsec,
                                    clip_sigma=clip_sigma,
                                    max_matches=max_matches)
        if matched['vis_xy'].shape[0] >= 5:
            vis_ref, vis_snr_arr, vis_csig_arr = refine_centroids_psf_fit(
                vis_img, matched['vis_xy'], radius=psf_radius,
                fwhm_guess=fwhm_guess_vis)
            for band in target_bands:
                if band not in RUBIN_BAND_ORDER:
                    continue
                tidx = RUBIN_BAND_ORDER.index(band)
                if tidx >= rubin_cube.shape[0]:
                    continue
                rimg = np.nan_to_num(_to_float32(rubin_cube[tidx]), nan=0.0)
                bid = BAND_TO_IDX.get(f'rubin_{band}', 0)
                _append(_process_band_anchors(
                    rimg, rwcs, vis_ref, vis_snr_arr, vis_csig_arr,
                    vwcs, matched['rubin_xy'], psf_radius,
                    fwhm_guess_rubin, RUBIN_PX, bid, tile_id))

        # ── NISP bands ───────────────────────────────────────────────
        if include_nisp:
            for nisp_short in NISP_BAND_ORDER:
                img_key, wcs_key = f'img_{nisp_short}', f'wcs_{nisp_short}'
                if img_key not in edata or wcs_key not in edata:
                    continue
                try:
                    nisp_img = np.nan_to_num(_to_float32(edata[img_key]), nan=0.0)
                    nisp_wcs = WCS(safe_header_from_card_string(edata[wcs_key].item()))
                except Exception:
                    continue
                nx, ny = detect_sources(nisp_img, nsig=4.0, smooth_sigma=1.2,
                                        min_dist=9, max_sources=800)
                matched_n = match_sources_wcs(nx, ny, vx, vy, nisp_wcs, vwcs,
                                              max_sep_arcsec=max_sep_arcsec,
                                              clip_sigma=clip_sigma,
                                              max_matches=max_matches)
                if matched_n['vis_xy'].shape[0] < 5:
                    continue
                vis_ref_n, vis_snr_n, vis_csig_n = refine_centroids_psf_fit(
                    vis_img, matched_n['vis_xy'], radius=psf_radius,
                    fwhm_guess=fwhm_guess_vis)
                bid = BAND_TO_IDX.get(f'nisp_{nisp_short}', 0)
                _append(_process_band_anchors(
                    nisp_img, nisp_wcs, vis_ref_n, vis_snr_n, vis_csig_n,
                    vwcs, matched_n['rubin_xy'], psf_radius,
                    fwhm_guess_nisp, NISP_PX, bid, tile_id))

        n_processed += 1
        if verbose and n_processed % 100 == 0:
            print(f'  {n_processed}/{len(pairs)} tiles, {time.time()-t0:.0f}s')

    if not acc['ra']:
        raise RuntimeError('No anchors collected from any tile.')

    anchors = {
        'ra': np.concatenate(acc['ra']).astype(np.float64),
        'dec': np.concatenate(acc['dec']).astype(np.float64),
        'dra': np.concatenate(acc['dra']),
        'ddec': np.concatenate(acc['ddec']),
        'snr_band': np.concatenate(acc['snr_band']),
        'snr_vis': np.concatenate(acc['snr_vis']),
        'sigma_centroid': np.concatenate(acc['sigma']),
        'band_idx': np.concatenate(acc['band_idx']),
        'tile_id': np.concatenate(acc['tile_id']),
    }
    if verbose:
        N = len(anchors['ra'])
        print(f'\nCollected {N} anchors from {n_processed} tiles in {time.time()-t0:.0f}s')
        print(f'  σ_centroid median: {np.median(anchors["sigma_centroid"])*1000:.1f} mas')
        for bi in sorted(np.unique(anchors['band_idx'])):
            n_bi = (anchors['band_idx'] == bi).sum()
            print(f'    band_idx={bi}: {n_bi} anchors')
    return anchors


# ============================================================
# Anchor collection: from existing prediction cache
# ============================================================

def load_anchors_from_cache(
    cache_path: str,
    bands: List[str] = ('r', 'i', 'g', 'z'),
    include_nisp: bool = False,
) -> Dict[str, np.ndarray]:
    """Load raw centroid offsets from an existing prediction cache.

    Since the cache doesn't have PSF-fit SNR, we estimate per-source
    quality from per-tile residual scatter (sources whose offset is
    close to the tile median are likely better measured).
    """
    cache = np.load(cache_path, allow_pickle=True)

    all_bands = list(bands)
    if include_nisp:
        all_bands += [f'nisp_{b}' for b in NISP_BAND_ORDER]

    all_ra, all_dec = [], []
    all_dra, all_ddec = [], []
    all_sigma = []
    all_band_idx = []
    all_tiles = []

    for band in all_bands:
        key = band if band.startswith('nisp_') else band
        if f'{key}_ra' not in cache:
            print(f'  Band {band} not in cache, skipping')
            continue

        ra = cache[f'{key}_ra']
        dec = cache[f'{key}_dec']
        raw = cache[f'{key}_raw']        # [N, 2] arcsec
        tiles = cache[f'{key}_tiles']    # [N] tile IDs

        # Compute per-tile median offset and residual
        unique_tiles = np.unique(tiles)
        sigma_est = np.full(len(ra), 0.050, dtype=np.float32)  # 50 mas default

        for tid in unique_tiles:
            mask = tiles == tid
            if mask.sum() < 5:
                continue
            tile_dra = raw[mask, 0]
            tile_ddec = raw[mask, 1]
            # Per-tile scatter = robust estimate of centroid noise
            scatter_ra = robust_sigma(tile_dra)
            scatter_dec = robust_sigma(tile_ddec)
            tile_sigma = np.sqrt(scatter_ra**2 + scatter_dec**2)
            sigma_est[mask] = max(tile_sigma, 0.005)

        band_id = BAND_TO_IDX.get(
            f'rubin_{band}' if not band.startswith('nisp_') else band,
            0)

        all_ra.append(ra)
        all_dec.append(dec)
        all_dra.append(raw[:, 0].astype(np.float32))
        all_ddec.append(raw[:, 1].astype(np.float32))
        all_sigma.append(sigma_est)
        all_band_idx.append(np.full(len(ra), band_id, dtype=np.int32))
        all_tiles.append(tiles)

    anchors = {
        'ra': np.concatenate(all_ra).astype(np.float64),
        'dec': np.concatenate(all_dec).astype(np.float64),
        'dra': np.concatenate(all_dra),
        'ddec': np.concatenate(all_ddec),
        'sigma_centroid': np.concatenate(all_sigma),
        'band_idx': np.concatenate(all_band_idx),
        'tile_id': np.concatenate(all_tiles),
    }

    N = len(anchors['ra'])
    print(f'Loaded {N} anchors from cache ({len(all_bands)} bands)')
    print(f'  σ_centroid (tile-based) median: '
          f'{np.median(anchors["sigma_centroid"])*1000:.1f} mas')

    return anchors


# ============================================================
# Sky → tangent-plane conversion
# ============================================================

def sky_to_tangent_plane(
    ra: np.ndarray,
    dec: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Convert (RA, Dec) to tangent-plane (x, y) in arcsec."""
    ra0 = float(np.median(ra))
    dec0 = float(np.median(dec))
    cosdec = np.cos(np.deg2rad(dec0))
    x = (ra - ra0) * cosdec * 3600.0    # arcsec
    y = (dec - dec0) * 3600.0
    return np.stack([x, y], axis=1).astype(np.float32), ra0, dec0


# ============================================================
# Outlier rejection
# ============================================================

def sigma_clip_anchors(
    anchors: Dict[str, np.ndarray],
    clip_arcsec: float = 0.3,
    clip_sigma: float = 4.0,
) -> Dict[str, np.ndarray]:
    """Remove outlier anchors by offset magnitude and per-tile scatter."""
    dra = anchors['dra']
    ddec = anchors['ddec']
    tiles = anchors['tile_id']
    mag = np.hypot(dra, ddec)

    # Hard clip on absolute offset
    keep = mag < clip_arcsec

    # Per-tile sigma clip on residual
    unique_tiles = np.unique(tiles)
    for tid in unique_tiles:
        tmask = tiles == tid
        if tmask.sum() < 5:
            continue
        tile_dra = dra[tmask]
        tile_ddec = ddec[tmask]
        med_dra = np.median(tile_dra)
        med_ddec = np.median(tile_ddec)
        resid = np.hypot(tile_dra - med_dra, tile_ddec - med_ddec)
        resid_sig = robust_sigma(resid)
        tile_keep = resid < (np.median(resid) + clip_sigma * resid_sig)
        # Update keep mask for this tile's sources
        tile_indices = np.where(tmask)[0]
        keep[tile_indices[~tile_keep]] = False

    n_before = len(dra)
    n_after = keep.sum()
    print(f'Sigma clip: {n_before} → {n_after} '
          f'({n_before - n_after} removed, {(1 - n_after/n_before)*100:.1f}%)')

    return {k: v[keep] for k, v in anchors.items()}


# ============================================================
# PINN fitting + FITS export
# ============================================================

def fit_and_export(
    anchors: Dict[str, np.ndarray],
    output_path: str,
    dstep_arcsec: float = 1.0,
    n_steps: int = 8000,
    hidden_dim: int = 128,
    n_layers: int = 5,
    lambda_curl: float = 1.0,
    lambda_lapl: float = 0.1,
    lambda_band: float = 0.1,
    n_collocation: int = 15000,
    device: str = None,
):
    """Fit global PINN and export concordance FITS."""
    import torch
    from astropy.io import fits
    from astropy.wcs import WCS

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Convert to tangent plane
    pos_arcsec, ra0, dec0 = sky_to_tangent_plane(anchors['ra'], anchors['dec'])
    offsets = np.stack([anchors['dra'], anchors['ddec']], axis=1).astype(np.float32)
    sigma = anchors['sigma_centroid']
    band_idx = anchors['band_idx']

    # Weights from centroid noise
    weights = 1.0 / np.maximum(sigma, 0.003)**2
    weights = (weights / weights.mean()).astype(np.float32)

    # Determine number of bands
    unique_bands = np.unique(band_idx)
    n_bands = int(unique_bands.max()) + 1

    print(f'\nFitting PINN:')
    print(f'  {len(pos_arcsec)} anchors, {len(unique_bands)} bands')
    print(f'  Field extent: {pos_arcsec[:,0].ptp():.0f} × {pos_arcsec[:,1].ptp():.0f} arcsec')
    print(f'  Weight range: [{weights.min():.2f}, {weights.max():.2f}]')
    print(f'  Architecture: {n_layers} layers × {hidden_dim} hidden')
    print(f'  Physics: λ_curl={lambda_curl}, λ_lapl={lambda_lapl}, '
          f'λ_band={lambda_band}')
    print(f'  Device: {device}')
    print()

    model, meta = fit_pinn_field(
        pos_arcsec=pos_arcsec,
        offsets_arcsec=offsets,
        weights=weights,
        band_indices=band_idx,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_bands=n_bands,
        n_steps=n_steps,
        lr=1e-3,
        lambda_curl=lambda_curl,
        lambda_lapl=lambda_lapl,
        lambda_band=lambda_band,
        n_collocation=n_collocation,
        device=device,
    )

    # ── Evaluate on mesh and export FITS ──────────────────────────────
    x_range = pos_arcsec[:, 0].max() - pos_arcsec[:, 0].min()
    y_range = pos_arcsec[:, 1].max() - pos_arcsec[:, 1].min()
    field_w = int(np.ceil(x_range / dstep_arcsec)) + 2
    field_h = int(np.ceil(y_range / dstep_arcsec)) + 2

    # Build a sky WCS for the output mesh
    cosdec0 = np.cos(np.deg2rad(dec0))
    out_wcs = WCS(naxis=2)
    out_wcs.wcs.crpix = [field_w / 2 + 1, field_h / 2 + 1]
    out_wcs.wcs.crval = [ra0, dec0]
    out_wcs.wcs.cdelt = [
        -dstep_arcsec / 3600.0 / cosdec0,
        dstep_arcsec / 3600.0,
    ]
    out_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

    # Band name mapping: band_idx int → FITS extension prefix
    band_names_map = {}
    for short in RUBIN_BAND_ORDER:
        band_names_map[BAND_TO_IDX.get(f'rubin_{short}', -1)] = short.upper()
    for short in NISP_BAND_ORDER:
        band_names_map[BAND_TO_IDX.get(f'nisp_{short}', -1)] = f'NISP_{short}'

    hdr = out_wcs.to_header()
    hdr['BUNIT'] = 'arcsec'
    hdr['RA0'] = (ra0, 'Reference RA [deg]')
    hdr['DEC0'] = (dec0, 'Reference Dec [deg]')
    hdr['DSTEP'] = (dstep_arcsec, 'Mesh step [arcsec]')
    hdr['NANCHORS'] = (len(pos_arcsec), 'Total anchor sources')
    hdr['METHOD'] = ('direct_pinn', 'Concordance method')

    hdul = fits.HDUList([fits.PrimaryHDU()])

    # Per-band fields (geometric + chromatic) — matches existing FITS format
    # Extension names: {BAND}.DRA, {BAND}.DDE  (e.g. R.DRA, NISP_Y.DDE)
    first_result = None
    for bi in unique_bands:
        bname = band_names_map.get(int(bi), f'BAND{bi}')
        result = evaluate_pinn_mesh(
            model, meta, field_h, field_w, dstep=1,
            pos_arcsec_anchors=pos_arcsec[band_idx == bi],
            band_idx=int(bi),
        )
        if first_result is None:
            first_result = result
        hdul.append(fits.ImageHDU(result['dra'], header=hdr,
                                  name=f'{bname}.DRA'))
        hdul.append(fits.ImageHDU(result['ddec'], header=hdr,
                                  name=f'{bname}.DDE'))

    # Shared coverage map (min distance to nearest anchor from ALL bands)
    cov_result = evaluate_pinn_mesh(
        model, meta, field_h, field_w, dstep=1,
        pos_arcsec_anchors=pos_arcsec, band_idx=None,
    )
    if cov_result['coverage'] is not None:
        hdul.append(fits.ImageHDU(cov_result['coverage'], header=hdr,
                                  name='COVERAGE'))
    geo_result = cov_result  # for stats below

    hdul.writeto(output_path, overwrite=True)
    print(f'\nFITS written: {output_path}')
    print(f'  Extensions: {[h.name for h in hdul[1:]]}')

    # ── Compute quality metrics ───────────────────────────────────────
    print('\n' + '=' * 60)
    print('QUALITY METRICS')
    print('=' * 60)

    # Evaluate PINN at anchor positions
    pos_norm = (pos_arcsec - meta['pos_min']) / meta['pos_scale'] * 2.0 - 1.0
    model.eval()
    with torch.no_grad():
        xy_t = torch.tensor(pos_norm, dtype=torch.float32)
        bi_t = torch.tensor(band_idx, dtype=torch.long)
        pred = model(xy_t, band_idx=bi_t).numpy() * meta['off_scale']

    resid = offsets - pred
    resid_mag = np.hypot(resid[:, 0], resid[:, 1]) * 1000  # mas
    raw_mag = np.hypot(offsets[:, 0], offsets[:, 1]) * 1000

    print(f'Raw offsets:            median={np.median(raw_mag):.1f} mas, '
          f'p68={np.percentile(raw_mag, 68):.1f} mas')
    print(f'After PINN subtraction: median={np.median(resid_mag):.1f} mas, '
          f'p68={np.percentile(resid_mag, 68):.1f} mas')
    improvement = (1 - np.median(resid_mag) / np.median(raw_mag)) * 100
    print(f'Improvement: {improvement:+.1f}%')

    # Per-band breakdown
    print(f'\nPer-band:')
    for bi in unique_bands:
        bname = band_names_map.get(int(bi), f'band{bi}')
        mask = band_idx == bi
        raw_b = np.median(raw_mag[mask])
        res_b = np.median(resid_mag[mask])
        print(f'  {bname:>6}: raw={raw_b:.1f}  resid={res_b:.1f}  '
              f'Δ={raw_b - res_b:+.1f} mas  ({mask.sum()} srcs)')

    # Geometric field stats
    geo_dra = geo_result['dra'] * 1000  # mas
    geo_ddec = geo_result['ddec'] * 1000
    geo_mag = np.hypot(geo_dra, geo_ddec)
    print(f'\nGeometric field amplitude:')
    print(f'  median={np.median(geo_mag):.1f} mas, '
          f'max={geo_mag.max():.1f} mas, '
          f'range={geo_mag.max() - geo_mag.min():.1f} mas')

    return model, meta


# ============================================================
# CLI
# ============================================================

def build_parser():
    p = argparse.ArgumentParser(
        description='Fit direct PINN concordance from raw centroid offsets.')

    src = p.add_argument_group('data source (choose one)')
    src.add_argument('--cache', type=str, default=None,
                     help='Path to predictions_cache.npz (quick mode)')
    src.add_argument('--rubin-dir', type=str, default=None,
                     help='Path to Rubin tile directory (full mode)')
    src.add_argument('--euclid-dir', type=str, default=None,
                     help='Path to Euclid tile directory (full mode)')

    out = p.add_argument_group('output')
    out.add_argument('--output', type=str, required=True,
                     help='Output FITS path')
    out.add_argument('--save-anchors', type=str, default=None,
                     help='Save anchor table as .npz for reuse')

    bands = p.add_argument_group('bands')
    bands.add_argument('--bands', nargs='+',
                       default=['u', 'g', 'r', 'i', 'z', 'y'],
                       help='Rubin target bands (default: all 6)')
    bands.add_argument('--include-nisp', action='store_true', default=True,
                       help='Include NISP Y/J/H bands (default: True)')
    bands.add_argument('--no-nisp', action='store_true',
                       help='Exclude NISP bands')

    pinn = p.add_argument_group('PINN')
    pinn.add_argument('--n-steps', type=int, default=8000)
    pinn.add_argument('--hidden-dim', type=int, default=128)
    pinn.add_argument('--n-layers', type=int, default=5)
    pinn.add_argument('--lambda-curl', type=float, default=1.0)
    pinn.add_argument('--lambda-lapl', type=float, default=0.1)
    pinn.add_argument('--lambda-band', type=float, default=0.1)
    pinn.add_argument('--n-collocation', type=int, default=15000)
    pinn.add_argument('--dstep-arcsec', type=float, default=1.0)

    clip = p.add_argument_group('outlier rejection')
    clip.add_argument('--clip-arcsec', type=float, default=0.3,
                      help='Hard clip on |offset| (arcsec)')
    clip.add_argument('--clip-sigma', type=float, default=4.0,
                      help='Per-tile sigma clip on residual')

    p.add_argument('--device', type=str, default=None)
    return p


def main():
    args = build_parser().parse_args()

    use_nisp = args.include_nisp and not args.no_nisp

    # ── Collect anchors ───────────────────────────────────────────────
    if args.cache:
        print(f'Loading from cache: {args.cache}')
        anchors = load_anchors_from_cache(
            args.cache,
            bands=args.bands,
            include_nisp=use_nisp,
        )
    elif args.rubin_dir and args.euclid_dir:
        print(f'Full reprocessing: {args.rubin_dir} + {args.euclid_dir}')
        pairs = discover_tile_pairs(args.rubin_dir, args.euclid_dir)
        print(f'  {len(pairs)} tile pairs')
        anchors = collect_raw_anchors(
            pairs,
            target_bands=args.bands,
            include_nisp=use_nisp,
        )
    else:
        print('ERROR: specify --cache or both --rubin-dir and --euclid-dir')
        sys.exit(1)

    # ── Save anchors if requested ─────────────────────────────────────
    if args.save_anchors:
        np.savez_compressed(args.save_anchors, **anchors)
        print(f'Anchors saved: {args.save_anchors}')

    # ── Sigma clip ────────────────────────────────────────────────────
    anchors = sigma_clip_anchors(
        anchors,
        clip_arcsec=args.clip_arcsec,
        clip_sigma=args.clip_sigma,
    )

    # ── Fit and export ────────────────────────────────────────────────
    fit_and_export(
        anchors,
        output_path=args.output,
        dstep_arcsec=args.dstep_arcsec,
        n_steps=args.n_steps,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        lambda_curl=args.lambda_curl,
        lambda_lapl=args.lambda_lapl,
        lambda_band=args.lambda_band,
        n_collocation=args.n_collocation,
        device=args.device,
    )


if __name__ == '__main__':
    main()
