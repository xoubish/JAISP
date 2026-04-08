"""
TileDetectionDataset: tile loader with pseudo-label source detection.

Each item returns:
    images    : {band_name: [1, H, W]}
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

from jaisp_foundation_v7 import RUBIN_BANDS, EUCLID_BANDS, ALL_BANDS
from jaisp_dataset_v6 import JAISPDatasetV6
from astrometry2.source_matching import detect_sources, build_detection_image



def _pseudo_labels(
    rubin_img: np.ndarray,   # [6, H, W]
    nsig: float,
    max_sources: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Detect sources from Rubin g+r+i coadd.

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

    classes = np.zeros(len(xs), dtype=np.int64)  # all class 0 = source

    centroids = np.stack([
        xs / max(W - 1, 1),
        ys / max(H - 1, 1),
    ], axis=1).astype(np.float32)

    return centroids, classes, H, W


def _vis_bright_core_and_spike_mask(
    vis_img: np.ndarray,
    spike_radius: int = 40,
    min_star_area: int = 20,
    max_spike_radius: int = 400,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bright-source centroids and a dilated mask around their cores.

    The mask radius scales with each star's saturated area:
        r = spike_radius * sqrt(area / min_star_area)
    so faint saturated stars get the base radius while bright stars with
    large saturated cores get proportionally larger masks that cover their
    longer diffraction spikes.

    This is used both by the classical VIS pseudo-labeler and by self-training
    refinement so artifact suppression stays consistent across stages.
    """
    from scipy.ndimage import binary_dilation, center_of_mass, label as ndlabel

    H, W = vis_img.shape
    bright_xs = np.empty(0, dtype=np.float64)
    bright_ys = np.empty(0, dtype=np.float64)
    spike_mask = np.zeros((H, W), dtype=bool)

    nonzero_vals = vis_img[vis_img > 0]
    if len(nonzero_vals) <= 100:
        return bright_xs, bright_ys, spike_mask

    sat_thresh = np.percentile(nonzero_vals, 99.5)
    sat_mask = vis_img > sat_thresh
    labeled, n_blobs = ndlabel(sat_mask)
    if n_blobs <= 0:
        return bright_xs, bright_ys, spike_mask

    areas = np.bincount(labeled.ravel())  # areas[0] = background
    core_labels = [lb for lb in range(1, n_blobs + 1) if areas[lb] >= min_star_area]
    if not core_labels:
        return bright_xs, bright_ys, spike_mask

    coms = center_of_mass(np.ones_like(vis_img), labeled, core_labels)
    if isinstance(coms, tuple):
        coms = [coms]  # single blob: center_of_mass returns a bare tuple
    bright_ys = np.array([c[0] for c in coms], dtype=np.float64)
    bright_xs = np.array([c[1] for c in coms], dtype=np.float64)

    # Per-star brightness-scaled dilation: brighter stars (larger saturated
    # area) get proportionally larger masks.
    for lb in core_labels:
        r = int(spike_radius * np.sqrt(areas[lb] / min_star_area))
        r = min(max(r, 1), max_spike_radius)
        blob_mask = labeled == lb
        yg, xg = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (xg ** 2 + yg ** 2) <= r ** 2
        spike_mask |= binary_dilation(blob_mask, structure=disk)

    return bright_xs, bright_ys, spike_mask


def _pseudo_labels_vis(
    vis_img: np.ndarray,       # [H_vis, W_vis]
    nsig: float = 3.0,
    max_sources: int = 1000,
    spike_radius: int = 40,
    min_star_area: int = 20,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Detect sources from Euclid VIS at native 0.1"/px resolution.

    Returns centroids normalized to VIS frame [0,1], preserving the full
    VIS spatial precision without projecting through a coarser grid.

    Saturated blobs with area >= min_star_area pixels are identified as star
    or galaxy cores. Their centroids are recorded as detections, then the blobs
    are dilated by spike_radius to mask diffraction spike regions. Classical
    peak-finding runs on the full image but its results are filtered to exclude
    positions inside the spike mask. This means bright sources are always
    detected (from blob centroids) while fake spike sources are suppressed.
    Small saturated blobs (hot pixels, CRs) are ignored.

    Returns (centroids_norm [M,2], classes [M], H_vis, W_vis).
    """
    H, W = vis_img.shape

    bright_xs, bright_ys, spike_mask = _vis_bright_core_and_spike_mask(
        vis_img,
        spike_radius=spike_radius,
        min_star_area=min_star_area,
    )

    # Classical peak-finding on the full image, then filter spike regions
    xs, ys = detect_sources(vis_img, nsig=nsig, max_sources=max_sources,
                            smooth_sigma=1.2, min_dist=9)
    if xs.size > 0:
        xi = np.clip(xs.round().astype(int), 0, W - 1)
        yi = np.clip(ys.round().astype(int), 0, H - 1)
        keep = ~spike_mask[yi, xi]
        xs, ys = xs[keep], ys[keep]

    # Prepend bright-core centroids (always kept regardless of spike mask)
    xs = np.concatenate([bright_xs, xs])
    ys = np.concatenate([bright_ys, ys])

    if xs.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            H, W,
        )

    classes = np.zeros(len(xs), dtype=np.int64)
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

        self._base = JAISPDatasetV6(
            rubin_dir=rubin_dir,
            euclid_dir=euclid_dir or rubin_dir,
            augment=augment,
            load_euclid=self.use_all_bands,
        )

        # Cache pseudo-labels from raw (unaugmented) tiles — run detection once.
        # Use VIS when Euclid data is available (0.1"/px, sharper centroids);
        # fall back to Rubin g+r+i coadd when not.
        self._label_cache = {}
        n_vis, n_rubin = 0, 0
        print(f'  Pre-computing pseudo-labels (nsig={nsig}) ...', end=' ', flush=True)
        for idx in range(len(self._base)):
            tile = self._base.tiles[idx]

            # Try VIS first
            euclid_path = tile.get('euclid_path')
            used_vis = False
            if euclid_path and Path(euclid_path).exists():
                try:
                    edata = np.load(euclid_path, allow_pickle=True, mmap_mode='r')
                    vis_img = np.nan_to_num(
                        np.asarray(edata['img_VIS'], dtype=np.float32), nan=0.0
                    )
                    centroids_np, classes_np, H, W = _pseudo_labels_vis(
                        vis_img, self.nsig, self.max_sources
                    )
                    used_vis = True
                    n_vis += 1
                except Exception:
                    pass

            if not used_vis:
                raw_path = tile['rubin_path']
                raw_data = np.load(raw_path, allow_pickle=True, mmap_mode='r')
                raw_img = np.nan_to_num(
                    np.asarray(raw_data['img'], dtype=np.float32), nan=0.0
                )
                centroids_np, classes_np, H, W = _pseudo_labels(
                    raw_img, self.nsig, self.max_sources
                )
                n_rubin += 1

            self._label_cache[idx] = (centroids_np, classes_np, H, W)
        print(f'done ({len(self._label_cache)} tiles: {n_vis} VIS, {n_rubin} Rubin)')

    @staticmethod
    def _transform_centroids(
        centroids: np.ndarray, n_rot: int, flip_ud: bool, flip_lr: bool,
    ) -> np.ndarray:
        """Apply the same augmentation to normalized (x, y) centroids."""
        if centroids.shape[0] == 0:
            return centroids
        xy = centroids.copy()
        # rot90: each 90° rotation maps (x, y) -> (y, 1-x)
        for _ in range(n_rot % 4):
            xy = np.stack([xy[:, 1], 1.0 - xy[:, 0]], axis=1)
        if flip_ud:
            xy[:, 1] = 1.0 - xy[:, 1]
        if flip_lr:
            xy[:, 0] = 1.0 - xy[:, 0]
        return xy

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict:
        item = self._base[idx]

        # Build per-band image/rms dicts
        images: dict = {}
        rms:    dict = {}

        for band in RUBIN_BANDS:
            if band in item['rubin']:
                images[band] = item['rubin'][band]['image']
                rms[band]    = item['rubin'][band]['rms']

        if self.use_all_bands:
            for band in EUCLID_BANDS:
                if band in item.get('euclid', {}):
                    images[band] = item['euclid'][band]['image']
                    rms[band]    = item['euclid'][band]['rms']

        # Retrieve cached pseudo-labels and transform to match augmentation
        centroids_np, classes_np, H, W = self._label_cache[idx]
        aug_params = item.get('aug_params', (0, False, False))
        centroids_np = self._transform_centroids(centroids_np, *aug_params)

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
