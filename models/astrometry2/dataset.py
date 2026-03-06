"""Matched local-patch dataset for standalone astrometry matching."""

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
ASTROMETRY_DIR = MODELS_DIR / 'astrometry'
for p in (MODELS_DIR, ASTROMETRY_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from jaisp_dataset_v4 import RUBIN_BAND_ORDER, _to_float32
from source_matching import (
    build_detection_image,
    detect_sources,
    match_sources_wcs,
    refine_centroids_in_band,
    safe_header_from_card_string,
)

VIS_PIXEL_SCALE_ARCSEC = 0.1


def _normalize_patch(arr: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """
    Background-subtract and noise-normalize a patch to ~unit background noise.

    Uses robust statistics (median background, MAD noise estimate) so the
    bright source itself does not bias the normalization.

    After normalization:
      - Background pixels → values clustered around 0 with std ≈ 1
      - Source pixels     → large positive values (S/N >> 1)

    This makes patches cross-tile and cross-instrument comparable regardless
    of exposure depth, sky level, or flux calibration differences.
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
) -> List[Dict]:
    if patch_size % 2 == 0:
        raise ValueError('patch_size must be odd.')

    target_band = normalize_rubin_band(target_band)
    input_bands = _select_input_bands(target_band, context_bands)
    detect_bands = normalize_rubin_bands(detect_bands) or [f'rubin_{b}' for b in ('g', 'r', 'i', 'z')]
    target_idx = RUBIN_BAND_ORDER.index(target_band.split('_', 1)[1])
    rng = np.random.RandomState(int(seed))

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
            # Horizontal flip (W-axis):
            #   Both patches are flipped so relative source morphology is preserved.
            #   The sky-coord target offset is unchanged (it's a physical measurement).
            #   The pixel_to_sky Jacobian's x-column (col 0) must be negated because
            #   after the flip, a +x pixel displacement maps to the opposite sky direction.
            if np.random.rand() > 0.5:
                rubin_patch = rubin_patch[:, :, ::-1].copy()
                vis_patch = vis_patch[:, :, ::-1].copy()
                pix2sky = pix2sky.copy()
                pix2sky[:, 0] = -pix2sky[:, 0]

            # Vertical flip (H-axis): same logic, y-column (col 1) is negated.
            if np.random.rand() > 0.5:
                rubin_patch = rubin_patch[:, ::-1, :].copy()
                vis_patch = vis_patch[:, ::-1, :].copy()
                pix2sky = pix2sky.copy()
                pix2sky[:, 1] = -pix2sky[:, 1]

        # Normalize after augmentation: per-patch background subtraction +
        # MAD noise scaling. Applied per-channel for Rubin (each band
        # independently) and to the single VIS channel.
        # This makes patches cross-tile comparable regardless of depth,
        # and turns sources into clean S/N peaks for the encoder.
        rubin_patch = np.stack(
            [_normalize_patch(rubin_patch[c]) for c in range(rubin_patch.shape[0])], axis=0
        )
        vis_patch = np.stack(
            [_normalize_patch(vis_patch[c]) for c in range(vis_patch.shape[0])], axis=0
        )

        return {
            'tile_id': s['tile_id'],
            'anchor_xy': torch.from_numpy(s['anchor_xy'].copy()),
            'rubin_patch': torch.from_numpy(rubin_patch),
            'vis_patch': torch.from_numpy(vis_patch),
            'target_offset_arcsec': torch.from_numpy(s['target_offset_arcsec'].copy()),
            'pixel_to_sky': torch.from_numpy(pix2sky),
            'input_bands': list(s['input_bands']),
            'target_band': s['target_band'],
        }


def collate_matched_patches(batch: List[Dict]) -> Dict:
    return {
        'tile_id': [b['tile_id'] for b in batch],
        'anchor_xy': torch.stack([b['anchor_xy'] for b in batch], dim=0),
        'rubin_patch': torch.stack([b['rubin_patch'] for b in batch], dim=0),
        'vis_patch': torch.stack([b['vis_patch'] for b in batch], dim=0),
        'target_offset_arcsec': torch.stack([b['target_offset_arcsec'] for b in batch], dim=0),
        'pixel_to_sky': torch.stack([b['pixel_to_sky'] for b in batch], dim=0),
        'input_bands': [b['input_bands'] for b in batch],
        'target_band': [b['target_band'] for b in batch],
    }


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
