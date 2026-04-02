"""Matched local-patch dataset for standalone astrometry matching.

Multi-band version:
  - build_patch_samples_multiband generates one sample per source per
    target band, each containing the full 6-band Rubin stamp.
  - Each sample carries a 'band_idx' integer identifying the target band.
  - The target offset is computed from that specific band's refined centroid,
    so ground truth is band-specific (capturing DCR differences).
  - The original build_patch_samples is preserved for backward compatibility.
"""

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from astropy.wcs import WCS

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, MODELS_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from source_matching import (
    RUBIN_BAND_ORDER,
    _to_float32,
    build_detection_image,
    detect_sources,
    match_sources_wcs,
    refine_centroids_in_band,
    safe_header_from_card_string,
)

VIS_PIXEL_SCALE_ARCSEC = 0.1


# ============================================================
# DETR-based source detection (optional, replaces classical)
# ============================================================

def detect_sources_detr(
    tile_images: Dict[str, np.ndarray],
    tile_rms: Dict[str, np.ndarray],
    detector,
    device,
    conf_threshold: float = 0.3,
    tile_hw: tuple = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the DETR detector on a tile and return (x, y) pixel coordinates.

    Parameters
    ----------
    tile_images : {band: [H, W]} numpy arrays for available bands
    tile_rms    : {band: [H, W]} RMS noise maps
    detector    : JaispDetector instance
    device      : torch device
    conf_threshold : minimum confidence for detections
    tile_hw     : (H, W) of the tile (for denormalization)

    Returns
    -------
    xs, ys : 1D arrays of pixel coordinates
    """
    import torch as _torch

    # Build batch dicts: {band: [1, 1, H, W]}
    imgs_d = {}
    rms_d = {}
    for band, img in tile_images.items():
        imgs_d[band] = _torch.from_numpy(img[None, None]).float().to(device)
        rms_d[band] = _torch.from_numpy(tile_rms[band][None, None]).float().to(device)

    with _torch.no_grad():
        result = detector.predict(imgs_d, rms_d,
                                  conf_threshold=conf_threshold,
                                  tile_hw=tile_hw)

    if result['positions_px'].shape[0] == 0:
        return np.zeros(0), np.zeros(0)

    pos = result['positions_px'].cpu().numpy()  # [N, 2] (x, y)
    return pos[:, 0], pos[:, 1]

# Unified band ordering for the multi-instrument model.
# Rubin bands come first (matching RUBIN_BAND_ORDER), then NISP bands.
NISP_BAND_ORDER = ['Y', 'J', 'H']
ALL_BAND_ORDER = [f'rubin_{b}' for b in RUBIN_BAND_ORDER] + [f'nisp_{b}' for b in NISP_BAND_ORDER]
BAND_TO_IDX = {b: i for i, b in enumerate(ALL_BAND_ORDER)}


def _load_nisp_band(edata, band: str):
    """Load a NISP band image and WCS from the euclid npz.

    Returns (image, wcs) or (None, None) if the band is missing.
    """
    img_key = f'img_{band}'
    wcs_key = f'wcs_{band}'
    if img_key not in edata or wcs_key not in edata:
        return None, None
    try:
        img = np.nan_to_num(_to_float32(edata[img_key]), nan=0.0)
        wcs = WCS(safe_header_from_card_string(edata[wcs_key].item()))
        return img, wcs
    except Exception as exc:
        print(f'[dataset] Failed to load NISP band {band}: {exc}')
        return None, None


def reproject_nisp_patch_to_vis(
    nisp_img: np.ndarray,
    nisp_wcs: WCS,
    vis_wcs: WCS,
    center_vis_xy: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """Reproject a NISP stamp onto the VIS pixel grid, same as Rubin reprojection."""
    gx, gy = _patch_grid(center_vis_xy, patch_size)
    ra, dec = vis_wcs.wcs_pix2world(gx.ravel(), gy.ravel(), 0)
    nx, ny = nisp_wcs.wcs_world2pix(ra, dec, 0)
    patch = _bilinear_sample(nisp_img, nx.reshape(gx.shape), ny.reshape(gy.shape))
    return patch.reshape(patch_size, patch_size)


def _normalize_patch(arr: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """
    Background-subtract and noise-normalize a patch to ~unit background noise.

    Uses robust statistics (median background, MAD noise estimate) so the
    bright source itself does not bias the normalization.
    """
    bg = float(np.median(arr))
    diff = arr.astype(np.float32) - bg
    mad = float(np.median(np.abs(diff)))
    scale = max(1.4826 * mad, float(floor))
    return (diff / scale).astype(np.float32)


def normalize_rubin_band(name: str) -> str:
    band = str(name).strip().lower()
    if band in RUBIN_BAND_ORDER:
        band = f'rubin_{band}'
    if not band.startswith('rubin_'):
        band = f'rubin_{band}'
    short = band.split('_', 1)[1]
    if short not in RUBIN_BAND_ORDER:
        raise ValueError(f'Invalid Rubin band: {name}')
    return band


def normalize_rubin_bands(names: Sequence[str]) -> List[str]:
    raw = [str(x).strip().lower() for x in names if str(x).strip()]
    if not raw:
        return []
    if any(x == 'all' for x in raw):
        return [f'rubin_{b}' for b in RUBIN_BAND_ORDER]
    out: List[str] = []
    seen = set()
    for name in raw:
        band = normalize_rubin_band(name)
        if band not in seen:
            seen.add(band)
            out.append(band)
    return out


def normalize_nisp_band(name: str) -> str:
    band = str(name).strip()
    if not band:
        raise ValueError('Empty NISP band name.')
    band_lc = band.lower()
    if band_lc.startswith('nisp_'):
        short = band.split('_', 1)[1].upper()
    else:
        raise ValueError(f'Invalid NISP band: {name}')
    if short not in NISP_BAND_ORDER:
        raise ValueError(f'Invalid NISP band: {name}')
    return f'nisp_{short}'


def discover_tile_pairs(rubin_dir: str, euclid_dir: str) -> List[Tuple[str, str, str]]:
    pairs = []
    for rubin_path in sorted(glob.glob(os.path.join(rubin_dir, 'tile_x*_y*.npz'))):
        tile_id = os.path.splitext(os.path.basename(rubin_path))[0]
        euclid_path = os.path.join(euclid_dir, f'{tile_id}_euclid.npz')
        if os.path.exists(euclid_path):
            pairs.append((tile_id, rubin_path, euclid_path))
    if not pairs:
        raise FileNotFoundError('No tile pairs found.')
    return pairs


def split_tile_pairs(
    pairs: Sequence[Tuple[str, str, str]],
    val_frac: float,
    seed: int,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    pairs = list(pairs)
    if val_frac <= 0.0 or len(pairs) <= 1:
        return pairs, []
    rng = np.random.RandomState(int(seed))
    order = np.arange(len(pairs))
    rng.shuffle(order)
    n_val = max(1, int(round(len(pairs) * float(val_frac))))
    n_val = min(n_val, len(pairs) - 1)
    val_idx = set(order[:n_val].tolist())
    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in val_idx]
    val_pairs = [pairs[i] for i in range(len(pairs)) if i in val_idx]
    return train_pairs, val_pairs


def _bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    out = (
        wa * img[y0c, x0c]
        + wb * img[y0c, x1c]
        + wc * img[y1c, x0c]
        + wd * img[y1c, x1c]
    )

    invalid = (x < 0) | (x > (w - 1)) | (y < 0) | (y > (h - 1))
    if np.any(invalid):
        out = out.astype(np.float32, copy=False)
        out[invalid] = 0.0
    return out.astype(np.float32, copy=False)


def _patch_grid(center_xy: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    half = int(patch_size) // 2
    offs = np.arange(-half, half + 1, dtype=np.float32)
    yy, xx = np.meshgrid(offs, offs, indexing='ij')
    x = np.asarray(center_xy[0], dtype=np.float32) + xx
    y = np.asarray(center_xy[1], dtype=np.float32) + yy
    return x, y


def extract_vis_patch(vis_img: np.ndarray, center_xy: np.ndarray, patch_size: int) -> np.ndarray:
    gx, gy = _patch_grid(center_xy, patch_size)
    return _bilinear_sample(vis_img, gx, gy).reshape(patch_size, patch_size)


def reproject_rubin_patch_to_vis(
    rubin_img: np.ndarray,
    rubin_wcs: WCS,
    vis_wcs: WCS,
    center_vis_xy: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    gx, gy = _patch_grid(center_vis_xy, patch_size)
    ra, dec = vis_wcs.wcs_pix2world(gx.ravel(), gy.ravel(), 0)
    rx, ry = rubin_wcs.wcs_world2pix(ra, dec, 0)
    patch = _bilinear_sample(rubin_img, rx.reshape(gx.shape), ry.reshape(gy.shape))
    return patch.reshape(patch_size, patch_size)


def local_vis_pixel_to_sky_matrix(vis_wcs: WCS, center_xy: np.ndarray) -> np.ndarray:
    x0 = float(center_xy[0])
    y0 = float(center_xy[1])
    ra0, dec0 = vis_wcs.wcs_pix2world([x0], [y0], 0)
    rax, decx = vis_wcs.wcs_pix2world([x0 + 1.0], [y0], 0)
    ray, decy = vis_wcs.wcs_pix2world([x0], [y0 + 1.0], 0)
    cosdec = np.cos(np.deg2rad(float(dec0[0])))
    j = np.zeros((2, 2), dtype=np.float32)
    j[0, 0] = float((rax[0] - ra0[0]) * cosdec * 3600.0)
    j[1, 0] = float((decx[0] - dec0[0]) * 3600.0)
    j[0, 1] = float((ray[0] - ra0[0]) * cosdec * 3600.0)
    j[1, 1] = float((decy[0] - dec0[0]) * 3600.0)
    return j


def _select_input_bands(target_band: str, context_bands: Sequence[str]) -> List[str]:
    out = [normalize_rubin_band(target_band)]
    for band in normalize_rubin_bands(context_bands):
        if band not in out:
            out.append(band)
    return out


# ---------------------------------------------------------------------------
# Original single-band sample builder (preserved for backward compatibility)
# ---------------------------------------------------------------------------

def build_patch_samples(
    pairs: Sequence[Tuple[str, str, str]],
    target_band: str,
    context_bands: Sequence[str],
    detect_bands: Sequence[str],
    *,
    patch_size: int = 33,
    max_patches_per_tile: int = 64,
    min_matches: int = 20,
    max_matches: int = 256,
    max_sep_arcsec: float = 0.12,
    clip_sigma: float = 3.5,
    rubin_nsig: float = 4.5,
    vis_nsig: float = 4.0,
    rubin_smooth: float = 1.0,
    vis_smooth: float = 1.2,
    rubin_min_dist: int = 7,
    vis_min_dist: int = 9,
    max_sources_rubin: int = 600,
    max_sources_vis: int = 800,
    detect_clip_sigma: float = 8.0,
    refine_radius: int = 3,
    refine_flux_floor_sigma: float = 1.5,
    seed: int = 42,
    split_name: str = 'train',
    detr_detector=None,
    detr_device=None,
    detr_conf_threshold: float = 0.3,
) -> List[Dict]:
    if patch_size % 2 == 0:
        raise ValueError('patch_size must be odd.')

    target_band = normalize_rubin_band(target_band)
    input_bands = _select_input_bands(target_band, context_bands)
    detect_bands = normalize_rubin_bands(detect_bands) or [f'rubin_{b}' for b in ('g', 'r', 'i', 'z')]
    target_idx = RUBIN_BAND_ORDER.index(target_band.split('_', 1)[1])
    rng = np.random.RandomState(int(seed))
    use_detr = detr_detector is not None

    samples: List[Dict] = []
    kept_tiles = 0
    for tile_id, rubin_path, euclid_path in pairs:
        try:
            rdata = np.load(rubin_path, allow_pickle=True)
            edata = np.load(euclid_path, allow_pickle=True)
            rubin_cube = rdata['img']
            vis_img = np.nan_to_num(_to_float32(edata['img_VIS']), nan=0.0)
            rubin_target = np.nan_to_num(_to_float32(rubin_cube[target_idx]), nan=0.0)
            rwcs = WCS(rdata['wcs_hdr'].item())
            vhdr = safe_header_from_card_string(edata['wcs_VIS'].item())
            vwcs = WCS(vhdr)
        except Exception as exc:
            print(f'[{split_name}] skip {tile_id}: load/wcs failed ({exc})')
            continue

        # --- Source detection (DETR or classical) --------------------------
        if use_detr:
            # Build band dicts for the DETR detector
            rubin_bands_list = [f'rubin_{b}' for b in RUBIN_BAND_ORDER]
            tile_images = {}
            tile_rms_d = {}
            for i, band in enumerate(rubin_bands_list):
                if i < rubin_cube.shape[0]:
                    img_np = np.nan_to_num(_to_float32(rubin_cube[i]), nan=0.0)
                    tile_images[band] = img_np
                    # Use robust sigma as RMS estimate
                    med = float(np.median(img_np))
                    sig = float(1.4826 * np.median(np.abs(img_np - med)))
                    tile_rms_d[band] = np.full_like(img_np, max(sig, 1e-10))

            H_r, W_r = rubin_cube.shape[1], rubin_cube.shape[2]
            rx, ry = detect_sources_detr(
                tile_images, tile_rms_d, detr_detector, detr_device,
                conf_threshold=detr_conf_threshold,
                tile_hw=(H_r, W_r),
            )
            # VIS: still use classical detection (DETR is Rubin-only for now)
            vx, vy = detect_sources(
                vis_img,
                nsig=vis_nsig,
                smooth_sigma=vis_smooth,
                min_dist=vis_min_dist,
                max_sources=max_sources_vis,
            )
        else:
            rubin_det = build_detection_image(rubin_cube, detect_bands, clip_sigma=detect_clip_sigma)
            rx, ry = detect_sources(
                rubin_det,
                nsig=rubin_nsig,
                smooth_sigma=rubin_smooth,
                min_dist=rubin_min_dist,
                max_sources=max_sources_rubin,
            )
            vx, vy = detect_sources(
                vis_img,
                nsig=vis_nsig,
                smooth_sigma=vis_smooth,
                min_dist=vis_min_dist,
                max_sources=max_sources_vis,
            )
        matched = match_sources_wcs(
            rx,
            ry,
            vx,
            vy,
            rwcs,
            vwcs,
            max_sep_arcsec=max_sep_arcsec,
            clip_sigma=clip_sigma,
            max_matches=max_matches,
        )
        if matched['vis_xy'].shape[0] == 0:
            continue

        rubin_xy_target = refine_centroids_in_band(
            rubin_target,
            matched['rubin_xy'],
            radius=refine_radius,
            flux_floor_sigma=refine_flux_floor_sigma,
        )
        r_ra, r_dec = rwcs.wcs_pix2world(rubin_xy_target[:, 0], rubin_xy_target[:, 1], 0)
        v_ra, v_dec = vwcs.wcs_pix2world(matched['vis_xy'][:, 0], matched['vis_xy'][:, 1], 0)
        dra = (v_ra - r_ra) * np.cos(np.deg2rad(v_dec)) * 3600.0
        ddec = (v_dec - r_dec) * 3600.0
        offsets = np.stack([dra, ddec], axis=1).astype(np.float32)
        vis_xy = matched['vis_xy'].astype(np.float32)

        if vis_xy.shape[0] < int(min_matches):
            continue

        keep = np.arange(vis_xy.shape[0])
        if keep.size > int(max_patches_per_tile):
            keep = rng.choice(keep, int(max_patches_per_tile), replace=False)
            keep.sort()

        tile_samples = []
        for idx in keep:
            anchor_xy = vis_xy[idx]
            vis_patch = extract_vis_patch(vis_img, anchor_xy, patch_size)
            rubin_patches = []
            for band in input_bands:
                band_idx = RUBIN_BAND_ORDER.index(band.split('_', 1)[1])
                if band_idx >= rubin_cube.shape[0]:
                    continue
                rubin_band_img = np.nan_to_num(_to_float32(rubin_cube[band_idx]), nan=0.0)
                rubin_patch = reproject_rubin_patch_to_vis(rubin_band_img, rwcs, vwcs, anchor_xy, patch_size)
                rubin_patches.append(rubin_patch)
            if not rubin_patches:
                continue

            pix2sky = local_vis_pixel_to_sky_matrix(vwcs, anchor_xy)
            tile_samples.append(
                {
                    'tile_id': tile_id,
                    'anchor_xy': anchor_xy.astype(np.float32),
                    'rubin_patch': np.stack(rubin_patches, axis=0).astype(np.float32),
                    'vis_patch': vis_patch[None].astype(np.float32),
                    'target_offset_arcsec': offsets[idx].astype(np.float32),
                    'pixel_to_sky': pix2sky.astype(np.float32),
                    'input_bands': list(input_bands),
                    'target_band': target_band,
                }
            )

        if tile_samples:
            samples.extend(tile_samples)
            kept_tiles += 1

    if not samples:
        raise RuntimeError(f'No patch samples created for split {split_name}.')

    mags = np.array([np.hypot(s['target_offset_arcsec'][0], s['target_offset_arcsec'][1]) * 1000.0 for s in samples], dtype=np.float32)
    print(
        f'[{split_name}] patch samples={len(samples)} from tiles={kept_tiles} | '
        f'|offset| median={np.median(mags):.1f} mas p68={np.percentile(mags, 68):.1f} mas'
    )
    return samples


# ---------------------------------------------------------------------------
# Multi-band sample builder
# ---------------------------------------------------------------------------

def build_patch_samples_multiband(
    pairs: Sequence[Tuple[str, str, str]],
    target_bands: Sequence[str],
    detect_bands: Sequence[str],
    *,
    include_nisp: bool = False,
    patch_size: int = 33,
    max_patches_per_tile: int = 64,
    min_matches: int = 20,
    max_matches: int = 256,
    max_sep_arcsec: float = 0.12,
    clip_sigma: float = 3.5,
    rubin_nsig: float = 4.5,
    vis_nsig: float = 4.0,
    rubin_smooth: float = 1.0,
    vis_smooth: float = 1.2,
    rubin_min_dist: int = 7,
    vis_min_dist: int = 9,
    max_sources_rubin: int = 600,
    max_sources_vis: int = 800,
    detect_clip_sigma: float = 8.0,
    refine_radius: int = 3,
    refine_flux_floor_sigma: float = 1.5,
    seed: int = 42,
    split_name: str = 'train',
    detr_detector=None,
    detr_device=None,
    detr_conf_threshold: float = 0.3,
) -> List[Dict]:
    """Build training samples with multi-instrument input and per-band targets.

    For each matched source, the input stamp contains ALL available bands
    reprojected onto the VIS pixel grid:
      - 6 Rubin channels (u/g/r/i/z/y)
      - Optionally 3 NISP channels (Y/J/H) if include_nisp=True

    For each target_band, a separate sample is created with that band's
    refined centroid offset as the ground truth and a 'band_idx' integer
    for conditioning the MLP head.

    Parameters
    ----------
    target_bands : list of band names to produce targets for.
        Rubin bands: 'r', 'i', etc. or 'rubin_r', 'rubin_i'.
        NISP bands: 'nisp_Y', 'nisp_J', 'nisp_H'.
        Use ['all'] for all 6 Rubin bands, ['all_nisp'] to include NISP targets too.
    detect_bands : bands used for Rubin source detection.
    include_nisp : if True, include reprojected NISP Y/J/H as encoder input channels.
        This is independent of whether NISP bands appear in target_bands.
    detr_detector : optional JaispDetector for DETR-based Rubin source detection.
    detr_device : torch device for DETR inference.
    detr_conf_threshold : confidence threshold for DETR detections.
    """
    if patch_size % 2 == 0:
        raise ValueError('patch_size must be odd.')

    # Parse target bands.
    target_bands_raw = [str(b).strip() for b in target_bands if str(b).strip()]
    target_bands_raw_lc = [b.lower() for b in target_bands_raw]
    target_bands_norm = []
    if any(b in ('all', 'all_rubin') for b in target_bands_raw_lc):
        target_bands_norm.extend([f'rubin_{b}' for b in RUBIN_BAND_ORDER])
    if any(b == 'all_nisp' for b in target_bands_raw_lc):
        target_bands_norm.extend([f'nisp_{b}' for b in NISP_BAND_ORDER])
    # Also parse individual band names.
    for b in target_bands_raw:
        b_lc = b.lower()
        if b_lc.startswith('nisp_'):
            nb = normalize_nisp_band(b)
            if nb not in target_bands_norm:
                target_bands_norm.append(nb)
        elif b_lc not in ('all', 'all_rubin', 'all_nisp'):
            nb = normalize_rubin_band(b)
            if nb not in target_bands_norm:
                target_bands_norm.append(nb)
    if not target_bands_norm:
        target_bands_norm = [f'rubin_{b}' for b in RUBIN_BAND_ORDER]

    detect_bands_norm = normalize_rubin_bands(detect_bands) or [f'rubin_{b}' for b in ('g', 'r', 'i', 'z')]

    # Determine input channel layout.
    rubin_input_bands = [f'rubin_{b}' for b in RUBIN_BAND_ORDER]
    nisp_input_bands = [f'nisp_{b}' for b in NISP_BAND_ORDER] if include_nisp else []
    all_input_bands = rubin_input_bands + nisp_input_bands
    n_input_channels = len(all_input_bands)

    # Band embedding indices: use the global ALL_BAND_ORDER table
    # so indices are consistent across Rubin-only and Rubin+NISP configs.
    rubin_targets = [b for b in target_bands_norm if b.startswith('rubin_')]
    nisp_targets = [b for b in target_bands_norm if b.startswith('nisp_')]

    rng = np.random.RandomState(int(seed))
    samples: List[Dict] = []
    kept_tiles = 0

    for tile_id, rubin_path, euclid_path in pairs:
        try:
            rdata = np.load(rubin_path, allow_pickle=True)
            edata = np.load(euclid_path, allow_pickle=True)
            rubin_cube = rdata['img']
            vis_img = np.nan_to_num(_to_float32(edata['img_VIS']), nan=0.0)
            rwcs = WCS(rdata['wcs_hdr'].item())
            vhdr = safe_header_from_card_string(edata['wcs_VIS'].item())
            vwcs = WCS(vhdr)
        except Exception as exc:
            print(f'[{split_name}] skip {tile_id}: load/wcs failed ({exc})')
            continue

        # Load NISP data if needed.
        nisp_data = {}  # band_letter -> (img, wcs)
        if include_nisp or nisp_targets:
            for nb in NISP_BAND_ORDER:
                img, nwcs = _load_nisp_band(edata, nb)
                if img is not None:
                    nisp_data[nb] = (img, nwcs)

        # Source detection (DETR or classical).
        if detr_detector is not None:
            rubin_bands_list = [f'rubin_{b}' for b in RUBIN_BAND_ORDER]
            tile_images = {}
            tile_rms_d = {}
            for i, band in enumerate(rubin_bands_list):
                if i < rubin_cube.shape[0]:
                    img_np = np.nan_to_num(_to_float32(rubin_cube[i]), nan=0.0)
                    tile_images[band] = img_np
                    med = float(np.median(img_np))
                    sig = float(1.4826 * np.median(np.abs(img_np - med)))
                    tile_rms_d[band] = np.full_like(img_np, max(sig, 1e-10))
            H_r, W_r = rubin_cube.shape[1], rubin_cube.shape[2]
            rx, ry = detect_sources_detr(
                tile_images, tile_rms_d, detr_detector, detr_device,
                conf_threshold=detr_conf_threshold, tile_hw=(H_r, W_r),
            )
        else:
            rubin_det = build_detection_image(rubin_cube, detect_bands_norm, clip_sigma=detect_clip_sigma)
            rx, ry = detect_sources(
                rubin_det, nsig=rubin_nsig, smooth_sigma=rubin_smooth,
                min_dist=rubin_min_dist, max_sources=max_sources_rubin,
            )
        vx, vy = detect_sources(
            vis_img, nsig=vis_nsig, smooth_sigma=vis_smooth,
            min_dist=vis_min_dist, max_sources=max_sources_vis,
        )
        matched = match_sources_wcs(
            rx, ry, vx, vy, rwcs, vwcs,
            max_sep_arcsec=max_sep_arcsec,
            clip_sigma=clip_sigma,
            max_matches=max_matches,
        )
        if matched['vis_xy'].shape[0] < int(min_matches):
            continue

        vis_xy = matched['vis_xy'].astype(np.float32)

        # Pre-compute per-band refined centroids and offsets.
        per_band_offsets = {}  # band_name -> (N, 2) arcsec

        # Rubin target bands.
        for tband in rubin_targets:
            short = tband.split('_', 1)[1]
            bidx = RUBIN_BAND_ORDER.index(short)
            if bidx >= rubin_cube.shape[0]:
                continue
            rubin_band_img = np.nan_to_num(_to_float32(rubin_cube[bidx]), nan=0.0)
            rubin_xy_refined = refine_centroids_in_band(
                rubin_band_img, matched['rubin_xy'],
                radius=refine_radius, flux_floor_sigma=refine_flux_floor_sigma,
            )
            r_ra, r_dec = rwcs.wcs_pix2world(rubin_xy_refined[:, 0], rubin_xy_refined[:, 1], 0)
            v_ra, v_dec = vwcs.wcs_pix2world(vis_xy[:, 0], vis_xy[:, 1], 0)
            dra = (v_ra - r_ra) * np.cos(np.deg2rad(v_dec)) * 3600.0
            ddec = (v_dec - r_dec) * 3600.0
            per_band_offsets[tband] = np.stack([dra, ddec], axis=1).astype(np.float32)

        # NISP target bands: offset = VIS position - NISP position.
        # NISP sources are detected independently and matched to VIS.
        # For simplicity, we use the existing VIS-matched positions and
        # compute the NISP centroid offset in sky coords.
        for tband in nisp_targets:
            nb = tband.split('_', 1)[1]  # 'Y', 'J', or 'H'
            if nb not in nisp_data:
                continue
            nisp_img, nwcs = nisp_data[nb]
            # Refine NISP centroids: project VIS positions into NISP pixel coords.
            nisp_x, nisp_y = nwcs.wcs_world2pix(
                *vwcs.wcs_pix2world(vis_xy[:, 0], vis_xy[:, 1], 0), 0)
            nisp_xy_init = np.stack([nisp_x, nisp_y], axis=1).astype(np.float32)
            nisp_xy_refined = refine_centroids_in_band(
                nisp_img, nisp_xy_init,
                radius=max(1, refine_radius // 3),  # NISP pixels are 3x coarser
                flux_floor_sigma=refine_flux_floor_sigma,
            )
            n_ra, n_dec = nwcs.wcs_pix2world(nisp_xy_refined[:, 0], nisp_xy_refined[:, 1], 0)
            v_ra, v_dec = vwcs.wcs_pix2world(vis_xy[:, 0], vis_xy[:, 1], 0)
            dra = (v_ra - n_ra) * np.cos(np.deg2rad(v_dec)) * 3600.0
            ddec = (v_dec - n_dec) * 3600.0
            per_band_offsets[tband] = np.stack([dra, ddec], axis=1).astype(np.float32)

        if not per_band_offsets:
            continue

        # Subsample sources.
        keep = np.arange(vis_xy.shape[0])
        if keep.size > int(max_patches_per_tile):
            keep = rng.choice(keep, int(max_patches_per_tile), replace=False)
            keep.sort()

        tile_samples = []
        for src_idx in keep:
            anchor_xy = vis_xy[src_idx]
            vis_patch = extract_vis_patch(vis_img, anchor_xy, patch_size)

            # Build full multi-instrument input stamp.
            input_patches = []

            # Rubin channels.
            for band in rubin_input_bands:
                short = band.split('_', 1)[1]
                bidx = RUBIN_BAND_ORDER.index(short)
                if bidx >= rubin_cube.shape[0]:
                    input_patches.append(np.zeros((patch_size, patch_size), dtype=np.float32))
                else:
                    rubin_band_img = np.nan_to_num(_to_float32(rubin_cube[bidx]), nan=0.0)
                    input_patches.append(
                        reproject_rubin_patch_to_vis(rubin_band_img, rwcs, vwcs, anchor_xy, patch_size)
                    )

            # NISP channels (if included as input).
            for band in nisp_input_bands:
                nb = band.split('_', 1)[1]
                if nb in nisp_data:
                    nisp_img, nwcs = nisp_data[nb]
                    input_patches.append(
                        reproject_nisp_patch_to_vis(nisp_img, nwcs, vwcs, anchor_xy, patch_size)
                    )
                else:
                    input_patches.append(np.zeros((patch_size, patch_size), dtype=np.float32))

            input_stamp = np.stack(input_patches, axis=0).astype(np.float32)  # [n_channels, H, W]
            vis_stamp = vis_patch[None].astype(np.float32)                     # [1, H, W]
            pix2sky = local_vis_pixel_to_sky_matrix(vwcs, anchor_xy)

            # One sample per target band.
            for tband, offsets in per_band_offsets.items():
                tile_samples.append({
                    'tile_id': tile_id,
                    'anchor_xy': anchor_xy.astype(np.float32),
                    'rubin_patch': input_stamp,  # name kept for compat, but now multi-instrument
                    'vis_patch': vis_stamp,
                    'target_offset_arcsec': offsets[src_idx].astype(np.float32),
                    'pixel_to_sky': pix2sky.astype(np.float32),
                    'input_bands': list(all_input_bands),
                    'target_band': tband,
                    'band_idx': BAND_TO_IDX[tband],
                })

        if tile_samples:
            samples.extend(tile_samples)
            kept_tiles += 1

    if not samples:
        raise RuntimeError(f'No patch samples created for split {split_name}.')

    mags = np.array([
        np.hypot(s['target_offset_arcsec'][0], s['target_offset_arcsec'][1]) * 1000.0
        for s in samples
    ], dtype=np.float32)
    n_per_band = {}
    for s in samples:
        b = s['target_band']
        n_per_band[b] = n_per_band.get(b, 0) + 1
    band_str = '  '.join(f'{b}:{n}' for b, n in sorted(n_per_band.items()))
    print(
        f'[{split_name}] multiband samples={len(samples)} from tiles={kept_tiles} | '
        f'|offset| median={np.median(mags):.1f} mas p68={np.percentile(mags, 68):.1f} mas\n'
        f'  per-band: {band_str}'
    )
    return samples


# ---------------------------------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------------------------------

class MatchedPatchDataset(torch.utils.data.Dataset):
    def __init__(self, samples: Sequence[Dict], augment: bool = False):
        self.samples = list(samples)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        rubin_patch = s['rubin_patch'].copy()           # [C, H, W]
        vis_patch = s['vis_patch'].copy()               # [1, H, W]
        pix2sky = s['pixel_to_sky'].copy()              # [2, 2]

        if self.augment:
            if np.random.rand() > 0.5:
                rubin_patch = rubin_patch[:, :, ::-1].copy()
                vis_patch = vis_patch[:, :, ::-1].copy()
                pix2sky = pix2sky.copy()
                pix2sky[:, 0] = -pix2sky[:, 0]

            if np.random.rand() > 0.5:
                rubin_patch = rubin_patch[:, ::-1, :].copy()
                vis_patch = vis_patch[:, ::-1, :].copy()
                pix2sky = pix2sky.copy()
                pix2sky[:, 1] = -pix2sky[:, 1]

        rubin_patch = np.stack(
            [_normalize_patch(rubin_patch[c]) for c in range(rubin_patch.shape[0])], axis=0
        )
        vis_patch = np.stack(
            [_normalize_patch(vis_patch[c]) for c in range(vis_patch.shape[0])], axis=0
        )

        out = {
            'tile_id': s['tile_id'],
            'anchor_xy': torch.from_numpy(s['anchor_xy'].copy()),
            'rubin_patch': torch.from_numpy(rubin_patch),
            'vis_patch': torch.from_numpy(vis_patch),
            'target_offset_arcsec': torch.from_numpy(s['target_offset_arcsec'].copy()),
            'pixel_to_sky': torch.from_numpy(pix2sky),
            'input_bands': list(s['input_bands']),
            'target_band': s['target_band'],
        }
        # Include band_idx if present (multi-band mode).
        if 'band_idx' in s:
            out['band_idx'] = int(s['band_idx'])
        return out


def collate_matched_patches(batch: List[Dict]) -> Dict:
    out = {
        'tile_id': [b['tile_id'] for b in batch],
        'anchor_xy': torch.stack([b['anchor_xy'] for b in batch], dim=0),
        'rubin_patch': torch.stack([b['rubin_patch'] for b in batch], dim=0),
        'vis_patch': torch.stack([b['vis_patch'] for b in batch], dim=0),
        'target_offset_arcsec': torch.stack([b['target_offset_arcsec'] for b in batch], dim=0),
        'pixel_to_sky': torch.stack([b['pixel_to_sky'] for b in batch], dim=0),
        'input_bands': [b['input_bands'] for b in batch],
        'target_band': [b['target_band'] for b in batch],
    }
    # Collate band_idx if present.
    if 'band_idx' in batch[0]:
        out['band_idx'] = torch.tensor([b['band_idx'] for b in batch], dtype=torch.long)
    return out


def make_loader(dataset: MatchedPatchDataset, batch_size: int, num_workers: int, shuffle: bool):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_matched_patches,
    )
