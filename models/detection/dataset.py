"""
TileDetectionDataset: wraps JAISPDatasetV6 and adds pseudo-label source detection.

Each item returns:
    images    : {band_name: [1, H, W]}  — same format as JAISPEncoderV6 expects
    rms       : {band_name: [1, H, W]}
    centroids : [M, 2]  float32  (x, y) normalised to [0, 1]
    classes   : [M]     int64    0=star, 1=galaxy
    tile_id   : str
    tile_hw   : (H, W)

When a real source catalog becomes available, replace _pseudo_labels() with a
catalog lookup and drop the classical detection dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_HERE   = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jaisp_foundation_v6 import RUBIN_BANDS, EUCLID_BANDS, ALL_BANDS
from jaisp_dataset_v6 import JAISPDatasetV6
from astrometry2.source_matching import detect_sources, build_detection_image


_INNER_R = 2.0
_OUTER_R = 5.0
_STAR_CONC_THRESHOLD = 0.5


def _concentration_index(
    img_band: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """C = flux(r<2px) / flux(r<5px) per source. Stars: C > 0.5."""
    S = 11
    half = S // 2
    H, W = img_band.shape
    yy, xx = np.mgrid[:S, :S] - half
    r = np.sqrt(xx ** 2 + yy ** 2)
    inner = r < _INNER_R
    outer = r < _OUTER_R
    C = np.zeros(len(xs))
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        x0, y0 = int(xi) - half, int(yi) - half
        if x0 < 0 or y0 < 0 or x0 + S > W or y0 + S > H:
            continue
        stamp = img_band[y0:y0+S, x0:x0+S].clip(0)
        f_outer = stamp[outer].sum()
        C[i] = stamp[inner].sum() / max(f_outer, 1e-6)
    return C


def _pseudo_labels(
    rubin_img: np.ndarray,   # [6, H, W]
    nsig: float,
    max_sources: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Detect sources via classical peak-finding and classify star/galaxy by
    concentration index.

    Returns (centroids_norm [M,2], classes [M], H, W).
    """
    H, W = rubin_img.shape[1], rubin_img.shape[2]
    det = build_detection_image(rubin_img, ['rubin_g', 'rubin_r', 'rubin_i'])
    xs, ys = detect_sources(det, nsig=nsig, max_sources=max_sources)

    if xs.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,),   dtype=np.int64),
            H, W,
        )

    conc    = _concentration_index(rubin_img[2], xs, ys)   # r-band
    classes = (conc < _STAR_CONC_THRESHOLD).astype(np.int64)  # 0=star, 1=galaxy

    centroids = np.stack([
        xs / max(W - 1, 1),
        ys / max(H - 1, 1),
    ], axis=1).astype(np.float32)

    return centroids, classes, H, W


class TileDetectionDataset(Dataset):
    """
    Parameters
    ----------
    rubin_dir      : Rubin tile directory (tile_x*_y*.npz)
    euclid_dir     : Euclid tile directory (optional)
    nsig           : detection significance for pseudo-labels
    max_sources    : cap on pseudo-labels per tile
    use_all_bands  : if True include Euclid bands; else Rubin-only
    augment        : random 90° rotations + flips (matches MAE training)
    """

    def __init__(
        self,
        rubin_dir:     str,
        euclid_dir:    Optional[str] = None,
        nsig:          float = 5.0,
        max_sources:   int = 500,
        use_all_bands: bool = False,
        augment:       bool = True,
    ):
        self.nsig         = nsig
        self.max_sources  = max_sources
        self.use_all_bands = use_all_bands and (euclid_dir is not None)
        self.bands        = ALL_BANDS if self.use_all_bands else RUBIN_BANDS

        # Reuse JAISPDatasetV6 for consistent tile loading + augmentation.
        # euclid_dir is required by JAISPDatasetV6; fall back to rubin_dir
        # (tiles without Euclid will simply have has_euclid=False).
        self._base = JAISPDatasetV6(
            rubin_dir=rubin_dir,
            euclid_dir=euclid_dir or rubin_dir,
            augment=augment,
        )

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict:
        item = self._base[idx]   # uses JAISPDatasetV6 format

        # Build per-band image/rms dicts in JAISPEncoderV6 format
        images: dict = {}
        rms:    dict = {}
        rubin_img_np = None

        for band in RUBIN_BANDS:
            if band in item['rubin']:
                img_t = item['rubin'][band]['image']   # [1, H, W]
                rms_t = item['rubin'][band]['rms']
                images[band] = img_t
                rms[band]    = rms_t
                if rubin_img_np is None:
                    # Collect Rubin bands for detection (numpy, no-aug copy)
                    pass

        if self.use_all_bands:
            for band in EUCLID_BANDS:
                if band in item.get('euclid', {}):
                    images[band] = item['euclid'][band]['image']
                    rms[band]    = item['euclid'][band]['rms']

        # Pseudo-label detection on raw (pre-aug) Rubin tile for consistency
        # Re-load the raw tile for detection (augmented tile has rotated coords)
        raw_path = self._base.tiles[idx]['rubin_path']
        raw_data = np.load(raw_path, allow_pickle=True, mmap_mode='r')
        raw_img  = np.nan_to_num(
            np.asarray(raw_data['img'], dtype=np.float32), nan=0.0
        )
        centroids_np, classes_np, H, W = _pseudo_labels(
            raw_img, self.nsig, self.max_sources
        )

        return {
            'images':    images,                              # {band: [1,H,W]}
            'rms':       rms,                                 # {band: [1,H,W]}
            'centroids': torch.from_numpy(centroids_np),     # [M, 2]
            'classes':   torch.from_numpy(classes_np),       # [M]
            'tile_id':   item['tile_id'],
            'tile_hw':   (H, W),
        }


def collate_fn(batch: List[dict]) -> dict:
    """
    Stack images/rms per band across the batch; keep centroids/classes as lists
    (variable number of sources per tile).
    """
    bands = list(batch[0]['images'].keys())
    images = {b: torch.stack([s['images'][b] for s in batch]) for b in bands}
    rms    = {b: torch.stack([s['rms'][b]    for s in batch]) for b in bands}
    return {
        'images':    images,
        'rms':       rms,
        'centroids': [s['centroids'] for s in batch],
        'classes':   [s['classes']   for s in batch],
        'tile_id':   [s['tile_id']   for s in batch],
        'tile_hw':   [s['tile_hw']   for s in batch],
    }
