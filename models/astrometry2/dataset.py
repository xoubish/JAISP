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
    compute_encoder_label_sigma,
    detect_sources,
    estimate_source_snr,
    expected_centroid_sigma_arcsec,
    match_sources_wcs,
    refine_centroids_encoder,
    refine_centroids_in_band,
    refine_centroids_psf_fit,
    safe_header_from_card_string,
)

VIS_PIXEL_SCALE_ARCSEC = 0.1
RUBIN_PIXEL_SCALE_ARCSEC = 0.2


# ============================================================
# V7 BandStem loader for encoder-based centroiding
# ============================================================

def load_v7_stems(v7_checkpoint: str, device='cpu'):
    """Load frozen V7 BandStems from a foundation checkpoint.

    Returns a dict mapping band names (e.g. 'rubin_r', 'euclid_VIS')
    to frozen BandStem modules on the given device.
    """
    import torch
    from jaisp_foundation_v7 import JAISPFoundationV7, ALL_BANDS

    ckpt = torch.load(v7_checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    v7 = JAISPFoundationV7(
        band_names=cfg.get('band_names', ALL_BANDS),
        stem_ch=cfg.get('stem_ch', 64),
        hidden_ch=cfg.get('hidden_ch', 256),
        blocks_per_stage=cfg.get('blocks_per_stage', 2),
        transformer_depth=cfg.get('transformer_depth', 4),
        transformer_heads=cfg.get('transformer_heads', 8),
        fused_pixel_scale_arcsec=cfg.get('fused_pixel_scale_arcsec', 0.8),
    )
    v7.load_state_dict(ckpt['model'], strict=False)

    encoder = v7.encoder if hasattr(v7, 'encoder') else v7
    stems = {}
    for name, stem in encoder.stems.items():
        stem.eval()
        for p in stem.parameters():
            p.requires_grad = False
        stems[name] = stem.to(device)
    return stems


# ============================================================
# Neural source detection (DETR or CenterNet, replaces classical)
# ============================================================

def detect_sources_neural(
    tile_images: Dict[str, np.ndarray],
    tile_rms: Dict[str, np.ndarray],
    detector,
    device,
    conf_threshold: float = 0.3,
    tile_hw: tuple = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a neural detector on a tile and return (x, y) pixel coordinates.

    Works with any detector that implements predict(images, rms, conf_threshold, tile_hw)
    returning a dict with 'positions_px' -- both JaispDetector (DETR) and
    CenterNetDetector are compatible.

    Parameters
    ----------
    tile_images : {band: [H, W]} numpy arrays for available bands
    tile_rms    : {band: [H, W]} RMS noise maps
    detector    : JaispDetector or CenterNetDetector instance
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


def _constant_rms_image(image: np.ndarray) -> np.ndarray:
    """Cheap per-band RMS proxy for detector inference."""
    img = np.asarray(image, dtype=np.float32)
    med = float(np.median(img))
    sig = float(1.4826 * np.median(np.abs(img - med)))
    return np.full_like(img, max(sig, 1e-10), dtype=np.float32)


def _rms_from_var_or_image(var: Optional[np.ndarray], img: np.ndarray) -> np.ndarray:
    """Return RMS from variance, with robust fallback to image-based estimate."""
    if var is None:
        return _constant_rms_image(img)
    v = np.asarray(var, dtype=np.float32)
    rms = np.sqrt(np.maximum(np.nan_to_num(v, nan=0.0), 0.0)).astype(np.float32)
    good = np.isfinite(rms) & (rms > 0)
    if good.all():
        return rms
    fallback = _constant_rms_image(img)
    rms = np.where(good, rms, fallback)
    return rms.astype(np.float32)


def build_full_context_detector_inputs(
    edata,
    rubin_cube: np.ndarray,
    rubin_var: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Tuple[int, int]]:
    """Build the full 10-band input dict expected by the V7 detector."""
    tile_images: Dict[str, np.ndarray] = {}
    tile_rms: Dict[str, np.ndarray] = {}

    for i, short in enumerate(RUBIN_BAND_ORDER):
        if i >= rubin_cube.shape[0]:
            continue
        band = f'rubin_{short}'
        img = np.nan_to_num(_to_float32(rubin_cube[i]), nan=0.0)
        tile_images[band] = img
        band_var = None
        if rubin_var is not None and i < rubin_var.shape[0]:
            band_var = rubin_var[i]
        tile_rms[band] = _rms_from_var_or_image(band_var, img)

    vis_img = np.nan_to_num(_to_float32(edata['img_VIS']), nan=0.0)
    tile_images['euclid_VIS'] = vis_img
    vis_var = edata['var_VIS'] if 'var_VIS' in edata else None
    tile_rms['euclid_VIS'] = _rms_from_var_or_image(vis_var, vis_img)

    for short in NISP_BAND_ORDER:
        key = f'img_{short}'
        if key not in edata:
            continue
        img = np.nan_to_num(_to_float32(edata[key]), nan=0.0)
        tile_images[f'euclid_{short}'] = img
        var_key = f'var_{short}'
        band_var = edata[var_key] if var_key in edata else None
        tile_rms[f'euclid_{short}'] = _rms_from_var_or_image(band_var, img)

    return tile_images, tile_rms, vis_img.shape


def detect_sources_multiband(
    edata,
    rubin_cube: np.ndarray,
    rubin_var: Optional[np.ndarray],
    detector,
    device,
    conf_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the neural detector on the full 10-band tile and return VIS-frame anchors."""
    tile_images, tile_rms, tile_hw = build_full_context_detector_inputs(edata, rubin_cube, rubin_var=rubin_var)
    return detect_sources_neural(
        tile_images,
        tile_rms,
        detector,
        device,
        conf_threshold=conf_threshold,
        tile_hw=tile_hw,
    )


def signal_mask_in_band(
    image: np.ndarray,
    seed_xy: np.ndarray,
    radius: int = 3,
    flux_floor_sigma: float = 1.5,
) -> np.ndarray:
    """Return which seed positions have enough local positive flux to refine."""
    img = np.asarray(image, dtype=np.float32)
    H, W = img.shape
    seed_xy = np.asarray(seed_xy, dtype=np.float32)
    out = np.zeros((seed_xy.shape[0],), dtype=bool)
    global_sig = float(np.median(np.abs(img - np.median(img))) * 1.4826)
    global_sig = max(global_sig, 1e-8)
    r = max(1, int(radius))
    for i, (x0f, y0f) in enumerate(seed_xy):
        x0 = int(round(float(x0f)))
        y0 = int(round(float(y0f)))
        xa = max(0, x0 - r)
        xb = min(W, x0 + r + 1)
        ya = max(0, y0 - r)
        yb = min(H, y0 + r + 1)
        patch = img[ya:yb, xa:xb]
        if patch.size == 0:
            continue
        bg = float(np.percentile(patch, 30))
        w = np.clip(patch - bg, 0.0, None)
        out[i] = float(w.sum()) > float(flux_floor_sigma) * global_sig
    return out


def project_vis_to_band_xy(vis_xy: np.ndarray, vis_wcs: WCS, band_wcs: WCS) -> np.ndarray:
    """Project VIS pixel positions into another band's pixel frame."""
    ra, dec = vis_wcs.wcs_pix2world(vis_xy[:, 0], vis_xy[:, 1], 0)
    bx, by = band_wcs.wcs_world2pix(ra, dec, 0)
    return np.stack([bx, by], axis=1).astype(np.float32)


# Backward-compatible alias
detect_sources_detr = detect_sources_neural

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
        basename = os.path.basename(rubin_path)
        if basename.endswith('_euclid.npz'):
            continue  # skip euclid files if rubin_dir happens to contain them
        tile_id = os.path.splitext(basename)[0]
        euclid_path = os.path.join(euclid_dir, f'{tile_id}_euclid.npz')
        if os.path.exists(euclid_path):
            pairs.append((tile_id, rubin_path, euclid_path))
    if not pairs:
        raise FileNotFoundError(
            f'No tile pairs found. Checked rubin_dir={rubin_dir} '
            f'(pattern tile_x*_y*.npz) and euclid_dir={euclid_dir} '
            f'(pattern {{tile_id}}_euclid.npz).'
        )
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


def _shift_patch(patch: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift a 2D patch by (dx, dy) pixels using bilinear sampling."""
    h, w = patch.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return _bilinear_sample(patch, gx - float(dx), gy - float(dy)).reshape(h, w)


def _shift_patch_stack(stack: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift a [C, H, W] stack by (dx, dy) pixels."""
    return np.stack([_shift_patch(stack[c], dx, dy) for c in range(stack.shape[0])], axis=0)


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


def _biased_sample_indices(
    rng: np.random.RandomState,
    n_total: int,
    n_keep: int,
    mags_arcsec: Optional[np.ndarray],
    bias: float,
    power: float,
    floor_mas: float,
) -> np.ndarray:
    """Sample indices with optional bias toward larger magnitudes.

    bias=0 -> uniform sampling. bias=1 -> fully magnitude-weighted sampling.
    """
    if n_keep >= n_total:
        return np.arange(n_total)
    if mags_arcsec is None or bias <= 0:
        keep = rng.choice(n_total, n_keep, replace=False)
        keep.sort()
        return keep

    bias = float(np.clip(bias, 0.0, 1.0))
    mags_mas = np.clip(np.asarray(mags_arcsec, dtype=np.float32) * 1000.0, 0.0, None)
    weights = mags_mas + float(max(0.0, floor_mas))
    weights = np.power(weights, float(max(0.0, power)))
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        keep = rng.choice(n_total, n_keep, replace=False)
        keep.sort()
        return keep

    p = weights / weights.sum()
    if bias < 1.0:
        p = bias * p + (1.0 - bias) / float(n_total)
    keep = rng.choice(n_total, n_keep, replace=False, p=p)
    keep.sort()
    return keep


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
    offset_bias: float = 0.0,
    offset_bias_power: float = 1.0,
    offset_bias_floor_mas: float = 5.0,
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
            rubin_var = rdata['var'] if 'var' in rdata else None
            vis_img = np.nan_to_num(_to_float32(edata['img_VIS']), nan=0.0)
            vis_var = _to_float32(edata['var_VIS']) if 'var_VIS' in edata else None
            rubin_target = np.nan_to_num(_to_float32(rubin_cube[target_idx]), nan=0.0)
            rwcs = WCS(rdata['wcs_hdr'].item())
            vhdr = safe_header_from_card_string(edata['wcs_VIS'].item())
            vwcs = WCS(vhdr)
        except Exception as exc:
            print(f'[{split_name}] skip {tile_id}: load/wcs failed ({exc})')
            continue

        # --- Source detection (DETR or classical) --------------------------
        if use_detr:
            ax, ay = detect_sources_multiband(
                edata,
                rubin_cube,
                rubin_var,
                detr_detector,
                detr_device,
                conf_threshold=detr_conf_threshold,
            )
            if ax.size == 0:
                continue
            vis_seed_xy = np.stack([ax, ay], axis=1).astype(np.float32)
            vis_keep = signal_mask_in_band(
                vis_img,
                vis_seed_xy,
                radius=refine_radius,
                flux_floor_sigma=refine_flux_floor_sigma,
            )
            if not vis_keep.any():
                continue
            vis_xy = refine_centroids_in_band(
                vis_img,
                vis_seed_xy[vis_keep],
                radius=refine_radius,
                flux_floor_sigma=refine_flux_floor_sigma,
            )
            rubin_xy_seed = project_vis_to_band_xy(vis_xy, vwcs, rwcs)
            rubin_refine_radius = max(1, refine_radius // 3)
            target_keep = signal_mask_in_band(
                rubin_target,
                rubin_xy_seed,
                radius=rubin_refine_radius,
                flux_floor_sigma=refine_flux_floor_sigma,
            )
            if int(target_keep.sum()) < int(min_matches):
                continue
            vis_xy = vis_xy[target_keep]
            rubin_xy_target = refine_centroids_in_band(
                rubin_target,
                rubin_xy_seed[target_keep],
                radius=rubin_refine_radius,
                flux_floor_sigma=refine_flux_floor_sigma,
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
            vis_xy = matched['vis_xy'].astype(np.float32)
            if vis_xy.shape[0] < int(min_matches):
                continue

        r_ra, r_dec = rwcs.wcs_pix2world(rubin_xy_target[:, 0], rubin_xy_target[:, 1], 0)
        v_ra, v_dec = vwcs.wcs_pix2world(vis_xy[:, 0], vis_xy[:, 1], 0)
        dra = (v_ra - r_ra) * np.cos(np.deg2rad(v_dec)) * 3600.0
        ddec = (v_dec - r_dec) * 3600.0
        offsets = np.stack([dra, ddec], axis=1).astype(np.float32)

        # Drop matches with large raw WCS offsets (likely mismatches).
        raw_mag = np.hypot(offsets[:, 0], offsets[:, 1])
        keep_mask = raw_mag <= float(max_sep_arcsec)
        if int(keep_mask.sum()) < int(min_matches):
            continue
        vis_xy = vis_xy[keep_mask]
        rubin_xy_target = rubin_xy_target[keep_mask]
        offsets = offsets[keep_mask]

        keep = np.arange(vis_xy.shape[0])
        if keep.size > int(max_patches_per_tile):
            mags = np.hypot(offsets[:, 0], offsets[:, 1])
            keep = _biased_sample_indices(
                rng,
                keep.size,
                int(max_patches_per_tile),
                mags,
                offset_bias,
                offset_bias_power,
                offset_bias_floor_mas,
            )

        tile_samples = []
        # Pre-compute per-band RMS images (sqrt of variance, clamped).
        has_rms = rubin_var is not None and vis_var is not None
        if has_rms:
            vis_rms = np.sqrt(np.maximum(np.nan_to_num(vis_var, nan=0.0), 1e-20)).astype(np.float32)

        for idx in keep:
            anchor_xy = vis_xy[idx]
            vis_patch = extract_vis_patch(vis_img, anchor_xy, patch_size)
            rubin_patches = []
            rubin_rms_patches = [] if has_rms else None
            for band in input_bands:
                band_idx = RUBIN_BAND_ORDER.index(band.split('_', 1)[1])
                if band_idx >= rubin_cube.shape[0]:
                    continue
                rubin_band_img = np.nan_to_num(_to_float32(rubin_cube[band_idx]), nan=0.0)
                rubin_patch = reproject_rubin_patch_to_vis(rubin_band_img, rwcs, vwcs, anchor_xy, patch_size)
                rubin_patches.append(rubin_patch)
                if rubin_rms_patches is not None:
                    band_rms = np.sqrt(np.maximum(
                        np.nan_to_num(_to_float32(rubin_var[band_idx]), nan=0.0), 1e-20,
                    )).astype(np.float32)
                    rubin_rms_patches.append(
                        reproject_rubin_patch_to_vis(band_rms, rwcs, vwcs, anchor_xy, patch_size)
                    )
            if not rubin_patches:
                continue

            pix2sky = local_vis_pixel_to_sky_matrix(vwcs, anchor_xy)
            sample = {
                'tile_id': tile_id,
                'anchor_xy': anchor_xy.astype(np.float32),
                'rubin_patch': np.stack(rubin_patches, axis=0).astype(np.float32),
                'vis_patch': vis_patch[None].astype(np.float32),
                'target_offset_arcsec': offsets[idx].astype(np.float32),
                'pixel_to_sky': pix2sky.astype(np.float32),
                'input_bands': list(input_bands),
                'target_band': target_band,
            }
            if has_rms:
                sample['rubin_rms_patch'] = np.stack(rubin_rms_patches, axis=0).astype(np.float32)
                sample['vis_rms_patch'] = extract_vis_patch(vis_rms, anchor_xy, patch_size)[None].astype(np.float32)
            tile_samples.append(sample)

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
    offset_bias: float = 0.0,
    offset_bias_power: float = 1.0,
    offset_bias_floor_mas: float = 5.0,
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
    v7_stems: dict = None,
    stems_device=None,
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
        b_str = str(b).strip()
        if not b_str:
            continue
        b_lc = b_str.lower()
        # Allow shorthand NISP bands as uppercase Y/J/H.
        if b_str in ('Y', 'J', 'H'):
            nb = normalize_nisp_band(f'nisp_{b_str}')
            if nb not in target_bands_norm:
                target_bands_norm.append(nb)
        elif b_lc.startswith('nisp_'):
            nb = normalize_nisp_band(b_str)
            if nb not in target_bands_norm:
                target_bands_norm.append(nb)
        elif b_lc not in ('all', 'all_rubin', 'all_nisp'):
            nb = normalize_rubin_band(b_str)
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
            rubin_var = rdata['var'] if 'var' in rdata else None
            vis_img = np.nan_to_num(_to_float32(edata['img_VIS']), nan=0.0)
            vis_var = _to_float32(edata['var_VIS']) if 'var_VIS' in edata else None
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

        # Determine centroiding mode: encoder-based (when stems available) or PSF-fit.
        use_encoder_centroids = v7_stems is not None and stems_device is not None

        # Source detection (DETR or classical).
        if detr_detector is not None:
            ax, ay = detect_sources_multiband(
                edata,
                rubin_cube,
                rubin_var,
                detr_detector,
                detr_device,
                conf_threshold=detr_conf_threshold,
            )
            if ax.size == 0:
                continue
            vis_seed_xy = np.stack([ax, ay], axis=1).astype(np.float32)
            vis_keep = signal_mask_in_band(
                vis_img,
                vis_seed_xy,
                radius=refine_radius,
                flux_floor_sigma=refine_flux_floor_sigma,
            )
            if not vis_keep.any():
                continue
            # CenterNet positions already come from the V7 encoder's offset head.
            # When encoder stems are available, refine using the VIS BandStem
            # feature-energy peak; otherwise trust CenterNet positions directly
            # (they're already encoder-based — classical refinement would replace
            # learned features with parametric assumptions).
            vis_seed_kept = vis_seed_xy[vis_keep]
            if use_encoder_centroids and 'euclid_VIS' in v7_stems:
                vis_rms_img = _rms_from_var_or_image(
                    _to_float32(edata['var_VIS']) if 'var_VIS' in edata else None, vis_img,
                )
                vis_xy, _, _ = refine_centroids_encoder(
                    vis_img, vis_rms_img, vis_seed_kept,
                    v7_stems['euclid_VIS'], stems_device,
                    radius=refine_radius,
                )
            else:
                # Trust CenterNet positions directly — no classical refinement.
                vis_xy = vis_seed_kept.copy()
            vis_xy = vis_xy.astype(np.float32)
            rubin_xy_seed_default = project_vis_to_band_xy(vis_xy, vwcs, rwcs)
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
            rubin_xy_seed_default = matched['rubin_xy'].astype(np.float32)

        # Pre-compute per-band refined centroids and offsets.
        per_band_offsets = {}  # band_name -> (N, 2) arcsec
        per_band_valid = {}    # band_name -> (N,) bool

        # Per-source SNR and expected label uncertainty.
        per_band_snr = {}          # band_name -> [N] float32
        per_band_label_sigma = {}  # band_name -> [N] float32 (arcsec)

        # Rubin target bands.
        RUBIN_WCS_SYSTEMATIC = 0.005  # 5 mas (SITCOMTN-159)
        rubin_refine_radius = max(2, refine_radius)
        for tband in rubin_targets:
            short = tband.split('_', 1)[1]
            bidx = RUBIN_BAND_ORDER.index(short)
            if bidx >= rubin_cube.shape[0]:
                continue
            rubin_band_img = np.nan_to_num(_to_float32(rubin_cube[bidx]), nan=0.0)
            rubin_xy_seed = rubin_xy_seed_default
            valid = signal_mask_in_band(
                rubin_band_img,
                rubin_xy_seed,
                radius=rubin_refine_radius,
                flux_floor_sigma=refine_flux_floor_sigma,
            )
            if not valid.any():
                per_band_valid[tband] = valid
                continue

            # Centroid refinement: encoder-based (data-driven) or PSF-fit (fallback).
            if use_encoder_centroids and tband in v7_stems:
                rubin_rms_band = _rms_from_var_or_image(
                    _to_float32(rubin_var[bidx]) if rubin_var is not None and bidx < rubin_var.shape[0] else None,
                    rubin_band_img,
                )
                rubin_xy_refined, rubin_peak_snr, rubin_sharpness = refine_centroids_encoder(
                    rubin_band_img, rubin_rms_band, rubin_xy_seed,
                    v7_stems[tband], stems_device,
                    radius=rubin_refine_radius,
                )
                label_sigma = compute_encoder_label_sigma(
                    rubin_peak_snr, rubin_sharpness,
                    RUBIN_PIXEL_SCALE_ARCSEC,
                    systematic_floor_arcsec=RUBIN_WCS_SYSTEMATIC,
                )
            else:
                rubin_xy_refined, rubin_snr, _ = refine_centroids_psf_fit(
                    rubin_band_img, rubin_xy_seed,
                    radius=rubin_refine_radius,
                    flux_floor_sigma=refine_flux_floor_sigma,
                    fwhm_guess=3.0,
                )
                label_sigma = expected_centroid_sigma_arcsec(
                    rubin_snr, RUBIN_PIXEL_SCALE_ARCSEC, fwhm_px=3.0,
                    systematic_floor_arcsec=RUBIN_WCS_SYSTEMATIC,
                )
                rubin_peak_snr = rubin_snr

            band_snr = np.ones(vis_xy.shape[0], dtype=np.float32)
            band_snr[valid] = rubin_peak_snr[valid]
            per_band_snr[tband] = band_snr
            band_label_sigma = np.full(vis_xy.shape[0], 0.01, dtype=np.float32)
            band_label_sigma[valid] = label_sigma[valid]
            per_band_label_sigma[tband] = band_label_sigma

            r_ra, r_dec = rwcs.wcs_pix2world(rubin_xy_refined[valid, 0], rubin_xy_refined[valid, 1], 0)
            v_ra, v_dec = vwcs.wcs_pix2world(vis_xy[valid, 0], vis_xy[valid, 1], 0)
            dra = (v_ra - r_ra) * np.cos(np.deg2rad(v_dec)) * 3600.0
            ddec = (v_dec - r_dec) * 3600.0
            offsets = np.zeros((vis_xy.shape[0], 2), dtype=np.float32)
            offsets[valid] = np.stack([dra, ddec], axis=1).astype(np.float32)
            if max_sep_arcsec is not None and float(max_sep_arcsec) > 0:
                mag = np.hypot(dra, ddec)
                valid_keep = valid.copy()
                valid_keep[valid] = mag <= float(max_sep_arcsec)
            else:
                valid_keep = valid
            per_band_valid[tband] = valid_keep
            if not valid_keep.any():
                per_band_offsets[tband] = offsets
                continue
            if not np.all(valid_keep == valid):
                offsets[~valid_keep] = 0.0
            per_band_offsets[tband] = offsets

        # NISP target bands: offset = VIS position - NISP position.
        NISP_WCS_SYSTEMATIC = 0.002  # 2 mas (same telescope as VIS)
        for tband in nisp_targets:
            nb = tband.split('_', 1)[1]  # 'Y', 'J', or 'H'
            if nb not in nisp_data:
                continue
            nisp_img, nwcs = nisp_data[nb]
            nisp_xy_init = project_vis_to_band_xy(vis_xy, vwcs, nwcs)
            nisp_radius = refine_radius
            valid = signal_mask_in_band(
                nisp_img,
                nisp_xy_init,
                radius=nisp_radius,
                flux_floor_sigma=refine_flux_floor_sigma,
            )
            if not valid.any():
                per_band_valid[tband] = valid
                continue

            # Encoder-based or PSF-fit centroiding for NISP.
            euclid_stem_name = f'euclid_{nb}'
            if use_encoder_centroids and euclid_stem_name in v7_stems:
                nisp_rms = _rms_from_var_or_image(
                    _to_float32(edata[f'var_{nb}']) if f'var_{nb}' in edata else None,
                    nisp_img,
                )
                nisp_xy_refined, nisp_peak_snr, nisp_sharpness = refine_centroids_encoder(
                    nisp_img, nisp_rms, nisp_xy_init,
                    v7_stems[euclid_stem_name], stems_device,
                    radius=nisp_radius,
                )
                label_sigma_nisp = compute_encoder_label_sigma(
                    nisp_peak_snr, nisp_sharpness,
                    VIS_PIXEL_SCALE_ARCSEC,  # NISP MER at 0.1"/px
                    systematic_floor_arcsec=NISP_WCS_SYSTEMATIC,
                )
            else:
                nisp_xy_refined, nisp_snr, _ = refine_centroids_psf_fit(
                    nisp_img, nisp_xy_init,
                    radius=nisp_radius,
                    flux_floor_sigma=refine_flux_floor_sigma,
                    fwhm_guess=2.5,
                )
                label_sigma_nisp = expected_centroid_sigma_arcsec(
                    nisp_snr, VIS_PIXEL_SCALE_ARCSEC, fwhm_px=2.5,
                    systematic_floor_arcsec=NISP_WCS_SYSTEMATIC,
                )
                nisp_peak_snr = nisp_snr

            band_snr = np.ones(vis_xy.shape[0], dtype=np.float32)
            band_snr[valid] = nisp_peak_snr[valid]
            per_band_snr[tband] = band_snr
            band_label_sigma = np.full(vis_xy.shape[0], 0.005, dtype=np.float32)
            band_label_sigma[valid] = label_sigma_nisp[valid]
            per_band_label_sigma[tband] = band_label_sigma
            n_ra, n_dec = nwcs.wcs_pix2world(nisp_xy_refined[valid, 0], nisp_xy_refined[valid, 1], 0)
            v_ra, v_dec = vwcs.wcs_pix2world(vis_xy[valid, 0], vis_xy[valid, 1], 0)
            dra = (v_ra - n_ra) * np.cos(np.deg2rad(v_dec)) * 3600.0
            ddec = (v_dec - n_dec) * 3600.0
            offsets = np.zeros((vis_xy.shape[0], 2), dtype=np.float32)
            offsets[valid] = np.stack([dra, ddec], axis=1).astype(np.float32)
            if max_sep_arcsec is not None and float(max_sep_arcsec) > 0:
                mag = np.hypot(dra, ddec)
                valid_keep = valid.copy()
                valid_keep[valid] = mag <= float(max_sep_arcsec)
            else:
                valid_keep = valid
            per_band_valid[tband] = valid_keep
            if not valid_keep.any():
                per_band_offsets[tband] = offsets
                continue
            if not np.all(valid_keep == valid):
                offsets[~valid_keep] = 0.0
            per_band_offsets[tband] = offsets

        if not per_band_offsets:
            continue

        valid_any = np.zeros((vis_xy.shape[0],), dtype=bool)
        for valid in per_band_valid.values():
            valid_any |= valid
        if int(valid_any.sum()) < int(min_matches):
            continue

        source_mag = None
        if offset_bias > 0:
            source_mag = np.zeros((vis_xy.shape[0],), dtype=np.float32)
            for tband, offsets in per_band_offsets.items():
                valid = per_band_valid.get(tband)
                if valid is None or not valid.any():
                    continue
                mag = np.hypot(offsets[:, 0], offsets[:, 1]).astype(np.float32)
                source_mag = np.maximum(source_mag, mag * valid.astype(np.float32))

        # Subsample sources.
        keep = np.where(valid_any)[0]
        if keep.size > int(max_patches_per_tile):
            mags_keep = source_mag[keep] if source_mag is not None else None
            sel = _biased_sample_indices(
                rng,
                keep.size,
                int(max_patches_per_tile),
                mags_keep,
                offset_bias,
                offset_bias_power,
                offset_bias_floor_mas,
            )
            keep = keep[sel]

        tile_samples = []
        # Pre-compute per-band RMS images (sqrt of variance, clamped).
        has_rms = rubin_var is not None and vis_var is not None
        if has_rms:
            vis_rms = np.sqrt(np.maximum(np.nan_to_num(vis_var, nan=0.0), 1e-20)).astype(np.float32)

        for src_idx in keep:
            anchor_xy = vis_xy[src_idx]
            vis_patch = extract_vis_patch(vis_img, anchor_xy, patch_size)

            # Build full multi-instrument input stamp.
            input_patches = []
            rms_patches = [] if has_rms else None

            # Rubin channels.
            for band in rubin_input_bands:
                short = band.split('_', 1)[1]
                bidx = RUBIN_BAND_ORDER.index(short)
                if bidx >= rubin_cube.shape[0]:
                    input_patches.append(np.zeros((patch_size, patch_size), dtype=np.float32))
                    if rms_patches is not None:
                        rms_patches.append(np.ones((patch_size, patch_size), dtype=np.float32))
                else:
                    rubin_band_img = np.nan_to_num(_to_float32(rubin_cube[bidx]), nan=0.0)
                    input_patches.append(
                        reproject_rubin_patch_to_vis(rubin_band_img, rwcs, vwcs, anchor_xy, patch_size)
                    )
                    if rms_patches is not None:
                        band_rms = np.sqrt(np.maximum(
                            np.nan_to_num(_to_float32(rubin_var[bidx]), nan=0.0), 1e-20,
                        )).astype(np.float32)
                        rms_patches.append(
                            reproject_rubin_patch_to_vis(band_rms, rwcs, vwcs, anchor_xy, patch_size)
                        )

            # NISP channels (if included as input).
            for band in nisp_input_bands:
                nb = band.split('_', 1)[1]
                if nb in nisp_data:
                    nisp_img, nwcs = nisp_data[nb]
                    input_patches.append(
                        reproject_nisp_patch_to_vis(nisp_img, nwcs, vwcs, anchor_xy, patch_size)
                    )
                    if rms_patches is not None:
                        var_key = f'var_{nb}'
                        if var_key in edata:
                            nisp_rms = np.sqrt(np.maximum(
                                np.nan_to_num(_to_float32(edata[var_key]), nan=0.0), 1e-20,
                            )).astype(np.float32)
                            rms_patches.append(
                                reproject_nisp_patch_to_vis(nisp_rms, nwcs, vwcs, anchor_xy, patch_size)
                            )
                        else:
                            rms_patches.append(np.ones((patch_size, patch_size), dtype=np.float32))
                else:
                    input_patches.append(np.zeros((patch_size, patch_size), dtype=np.float32))
                    if rms_patches is not None:
                        rms_patches.append(np.ones((patch_size, patch_size), dtype=np.float32))

            input_stamp = np.stack(input_patches, axis=0).astype(np.float32)  # [n_channels, H, W]
            vis_stamp = vis_patch[None].astype(np.float32)                     # [1, H, W]
            pix2sky = local_vis_pixel_to_sky_matrix(vwcs, anchor_xy)

            sample_base = {
                'tile_id': tile_id,
                'anchor_xy': anchor_xy.astype(np.float32),
                'rubin_patch': input_stamp,  # name kept for compat, but now multi-instrument
                'vis_patch': vis_stamp,
                'pixel_to_sky': pix2sky.astype(np.float32),
                'input_bands': list(all_input_bands),
            }
            if has_rms:
                sample_base['rubin_rms_patch'] = np.stack(rms_patches, axis=0).astype(np.float32)
                vis_rms_patch = extract_vis_patch(vis_rms, anchor_xy, patch_size)
                sample_base['vis_rms_patch'] = vis_rms_patch[None].astype(np.float32)

            # One sample per target band.
            for tband, offsets in per_band_offsets.items():
                if not per_band_valid.get(tband, np.zeros((vis_xy.shape[0],), dtype=bool))[src_idx]:
                    continue
                sample = {
                    **sample_base,
                    'target_offset_arcsec': offsets[src_idx].astype(np.float32),
                    'target_band': tband,
                    'band_idx': BAND_TO_IDX[tband],
                }
                # Per-source SNR and label noise sigma (SITCOMTN-159).
                if tband in per_band_snr:
                    sample['source_snr'] = np.float32(per_band_snr[tband][src_idx])
                if tband in per_band_label_sigma:
                    sample['label_sigma_arcsec'] = np.float32(per_band_label_sigma[tband][src_idx])
                tile_samples.append(sample)

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
    def __init__(
        self,
        samples: Sequence[Dict],
        augment: bool = False,
        jitter_arcsec: float = 0.0,
        jitter_max_arcsec: float = 0.0,
        jitter_prob: float = 1.0,
    ):
        self.samples = list(samples)
        self.augment = bool(augment)
        self.jitter_arcsec = float(jitter_arcsec)
        self.jitter_max_arcsec = float(jitter_max_arcsec)
        self.jitter_prob = float(jitter_prob)

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_spatial(self, *arrays, pix2sky):
        """Apply the same random spatial augmentation to all arrays and the Jacobian."""
        pix2sky = pix2sky.copy()

        # Random 90-degree rotation (k=0,1,2,3 for 0°,90°,180°,270° CCW).
        k = np.random.randint(4)
        if k > 0:
            arrays = tuple(np.rot90(a, k=k, axes=(1, 2)).copy() for a in arrays)
            if k == 1:
                pix2sky = np.stack([pix2sky[:, 1], -pix2sky[:, 0]], axis=1)
            elif k == 2:
                pix2sky = np.stack([-pix2sky[:, 0], -pix2sky[:, 1]], axis=1)
            elif k == 3:
                pix2sky = np.stack([-pix2sky[:, 1], pix2sky[:, 0]], axis=1)

        # Random horizontal flip.
        if np.random.rand() > 0.5:
            arrays = tuple(a[:, :, ::-1].copy() for a in arrays)
            pix2sky = pix2sky.copy()
            pix2sky[:, 0] = -pix2sky[:, 0]

        # Random vertical flip.
        if np.random.rand() > 0.5:
            arrays = tuple(a[:, ::-1, :].copy() for a in arrays)
            pix2sky = pix2sky.copy()
            pix2sky[:, 1] = -pix2sky[:, 1]

        return arrays, pix2sky

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        rubin_patch = s['rubin_patch'].copy()           # [C, H, W]
        vis_patch = s['vis_patch'].copy()               # [1, H, W]
        pix2sky = s['pixel_to_sky'].copy()              # [2, 2]
        target_offset = s['target_offset_arcsec'].copy()
        has_rms = 'rubin_rms_patch' in s and 'vis_rms_patch' in s

        if has_rms:
            rubin_rms = s['rubin_rms_patch'].copy()
            vis_rms = s['vis_rms_patch'].copy()

        if self.augment:
            if has_rms:
                (rubin_patch, vis_patch, rubin_rms, vis_rms), pix2sky = \
                    self._augment_spatial(rubin_patch, vis_patch, rubin_rms, vis_rms, pix2sky=pix2sky)
            else:
                (rubin_patch, vis_patch), pix2sky = \
                    self._augment_spatial(rubin_patch, vis_patch, pix2sky=pix2sky)

        if self.jitter_arcsec > 0 and np.random.rand() < self.jitter_prob:
            jitter = np.random.normal(scale=self.jitter_arcsec, size=2).astype(np.float32)
            if self.jitter_max_arcsec > 0:
                jitter = np.clip(jitter, -self.jitter_max_arcsec, self.jitter_max_arcsec)
            inv = np.linalg.pinv(pix2sky).astype(np.float32)
            dx_px, dy_px = (inv @ jitter.reshape(2, 1)).reshape(2).tolist()
            rubin_patch = _shift_patch_stack(rubin_patch, dx_px, dy_px)
            if has_rms:
                rubin_rms = _shift_patch_stack(rubin_rms, dx_px, dy_px)
            target_offset = (target_offset - jitter).astype(np.float32, copy=False)

        if not has_rms:
            # Legacy path: scalar MAD normalization per channel.
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
            'target_offset_arcsec': torch.from_numpy(target_offset),
            'pixel_to_sky': torch.from_numpy(pix2sky),
            'input_bands': list(s['input_bands']),
            'target_band': s['target_band'],
        }
        if has_rms:
            out['rubin_rms_patch'] = torch.from_numpy(rubin_rms)
            out['vis_rms_patch'] = torch.from_numpy(vis_rms)
        if 'band_idx' in s:
            out['band_idx'] = int(s['band_idx'])
        # Per-source label noise (SITCOMTN-159).
        if 'source_snr' in s:
            out['source_snr'] = torch.tensor(float(s['source_snr']), dtype=torch.float32)
        if 'label_sigma_arcsec' in s:
            out['label_sigma_arcsec'] = torch.tensor(float(s['label_sigma_arcsec']), dtype=torch.float32)
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
    if 'band_idx' in batch[0]:
        out['band_idx'] = torch.tensor([b['band_idx'] for b in batch], dtype=torch.long)
    if 'rubin_rms_patch' in batch[0]:
        out['rubin_rms_patch'] = torch.stack([b['rubin_rms_patch'] for b in batch], dim=0)
        out['vis_rms_patch'] = torch.stack([b['vis_rms_patch'] for b in batch], dim=0)
    if 'source_snr' in batch[0]:
        out['source_snr'] = torch.stack([b['source_snr'] for b in batch], dim=0)
    if 'label_sigma_arcsec' in batch[0]:
        out['label_sigma_arcsec'] = torch.stack([b['label_sigma_arcsec'] for b in batch], dim=0)
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
