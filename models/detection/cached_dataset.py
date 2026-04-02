"""Dataset that loads precomputed V7 encoder features for fast CenterNet training.

Instead of running the frozen encoder every step, this dataset loads
cached bottleneck tensors from disk. Training only runs the lightweight
CenterNet head (a few conv layers), making each step ~10x faster.

The pseudo-labels are still generated at init time (from VIS when available,
Rubin fallback) and cached. Augmentation is handled by loading different
precomputed augmentation variants.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from astrometry2.source_matching import detect_sources
from detection.dataset import _pseudo_labels, _pseudo_labels_vis, TileDetectionDataset


class CachedFeatureDataset(Dataset):
    """Load precomputed encoder features + pseudo-labels for CenterNet training.

    Parameters
    ----------
    feature_dir   : directory with tile_*_aug*.pt files from precompute_features.py
    rubin_dir     : Rubin tile directory (for Rubin-fallback pseudo-labels)
    euclid_dir    : Euclid tile directory (for VIS pseudo-labels)
    nsig          : detection threshold for pseudo-labels
    max_sources   : max pseudo-labels per tile
    extra_labels  : optional path to .pt file with extra pseudo-labels from
                    self-training (promoted high-confidence detections)
    """

    def __init__(
        self,
        feature_dir:  str,
        rubin_dir:    str,
        euclid_dir:   Optional[str] = None,
        nsig:         float = 3.0,
        max_sources:  int = 1000,
        extra_labels: Optional[str] = None,
    ):
        self.feature_dir = Path(feature_dir)
        self.nsig = nsig
        self.max_sources = max_sources

        # Discover all cached feature files
        feat_files = sorted(self.feature_dir.glob('tile_*_aug*.pt'))
        if not feat_files:
            raise FileNotFoundError(f'No cached features found in {feature_dir}')

        # Group by tile_id
        self._tile_feats: Dict[str, List[Path]] = {}
        for f in feat_files:
            # Filename: tile_x00000_y00000_aug0.pt
            parts = f.stem.rsplit('_aug', 1)
            tile_id = parts[0]
            if tile_id not in self._tile_feats:
                self._tile_feats[tile_id] = []
            self._tile_feats[tile_id].append(f)

        self._tile_ids = sorted(self._tile_feats.keys())
        n_augs = [len(v) for v in self._tile_feats.values()]
        print(f'CachedFeatureDataset: {len(self._tile_ids)} tiles, '
              f'{min(n_augs)}-{max(n_augs)} augs each, '
              f'{len(feat_files)} total samples')

        # Build flat index: (tile_id, feature_path)
        self._samples: List[Tuple[str, Path]] = []
        for tid in self._tile_ids:
            for fp in self._tile_feats[tid]:
                self._samples.append((tid, fp))

        # Cache pseudo-labels per tile (VIS when available, Rubin fallback)
        self._label_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._compute_labels(rubin_dir, euclid_dir)

        # Apply label refinement from self-training if provided
        if extra_labels and Path(extra_labels).exists():
            extra = torch.load(extra_labels, map_location='cpu', weights_only=False)
            promoted = extra.get('promoted', {})
            demoted_labels = extra.get('demoted', {})

            # First remove demoted labels (artifacts the model rejected)
            n_removed = 0
            for tid, remove_xy in demoted_labels.items():
                if tid in self._label_cache and remove_xy.shape[0] > 0:
                    old_c, old_cls = self._label_cache[tid]
                    if old_c.shape[0] == 0:
                        continue
                    # Remove labels that are close to any demoted position
                    keep = np.ones(len(old_c), dtype=bool)
                    for j in range(len(remove_xy)):
                        dists = np.sqrt(((old_c - remove_xy[j]) ** 2).sum(axis=1))
                        keep &= (dists > 0.005)  # ~0.5 bottleneck pixels
                    n_before = len(old_c)
                    old_c = old_c[keep]
                    old_cls = old_cls[keep]
                    self._label_cache[tid] = (old_c, old_cls)
                    n_removed += n_before - len(old_c)

            # Then add promoted labels (novel detections)
            n_added = 0
            for tid, new_centroids in promoted.items():
                if tid in self._label_cache:
                    old_c, old_cls = self._label_cache[tid]
                    merged_c = np.concatenate([old_c, new_centroids], axis=0)
                    merged_cls = np.zeros(len(merged_c), dtype=np.int64)
                    self._label_cache[tid] = (merged_c, merged_cls)
                    n_added += len(new_centroids)

            print(f'  Label refinement: +{n_added} promoted, -{n_removed} demoted')

    def _compute_labels(self, rubin_dir: str, euclid_dir: Optional[str]):
        """Compute pseudo-labels from VIS (preferred) or Rubin (fallback)."""
        rubin_dir = Path(rubin_dir)
        euclid_dir = Path(euclid_dir) if euclid_dir else None
        n_vis, n_rubin = 0, 0

        print(f'  Computing pseudo-labels (nsig={self.nsig}) ...', end=' ', flush=True)
        for tid in self._tile_ids:
            # Try VIS
            used_vis = False
            if euclid_dir:
                euclid_path = euclid_dir / f'{tid}_euclid.npz'
                if euclid_path.exists():
                    try:
                        edata = np.load(str(euclid_path), allow_pickle=True, mmap_mode='r')
                        vis_img = np.nan_to_num(
                            np.asarray(edata['img_VIS'], dtype=np.float32), nan=0.0
                        )
                        centroids, classes, _, _ = _pseudo_labels_vis(
                            vis_img, self.nsig, self.max_sources
                        )
                        used_vis = True
                        n_vis += 1
                    except Exception:
                        pass

            if not used_vis:
                rubin_path = rubin_dir / f'{tid}.npz'
                if not rubin_path.exists():
                    # Try to find it
                    candidates = list(rubin_dir.glob(f'{tid}*.npz'))
                    rubin_path = candidates[0] if candidates else rubin_path
                try:
                    rdata = np.load(str(rubin_path), allow_pickle=True, mmap_mode='r')
                    raw_img = np.nan_to_num(
                        np.asarray(rdata['img'], dtype=np.float32), nan=0.0
                    )
                    centroids, classes, _, _ = _pseudo_labels(
                        raw_img, self.nsig, self.max_sources
                    )
                    n_rubin += 1
                except Exception as exc:
                    print(f'\n  [warn] skip {tid}: {exc}')
                    centroids = np.zeros((0, 2), dtype=np.float32)
                    classes = np.zeros(0, dtype=np.int64)

            self._label_cache[tid] = (centroids, classes)

        print(f'done ({n_vis} VIS, {n_rubin} Rubin)')

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        tile_id, feat_path = self._samples[idx]

        # Load cached features
        cached = torch.load(feat_path, map_location='cpu', weights_only=True)
        features = cached['features']          # [256, H, W]
        aug_params = cached['aug_params']       # (n_rot, flip_ud, flip_lr)

        # Get pseudo-labels and transform to match this augmentation
        centroids_np, classes_np = self._label_cache[tile_id]
        centroids_np = TileDetectionDataset._transform_centroids(
            centroids_np, *aug_params
        )

        return {
            'features':  features,                                # [256, H, W]
            'centroids': torch.from_numpy(centroids_np),         # [M, 2]
            'classes':   torch.from_numpy(classes_np),           # [M]
            'tile_id':   tile_id,
            'aug_idx':   cached['aug_idx'],
        }


def collate_cached(batch: List[dict]) -> dict:
    """Collate for CachedFeatureDataset: stack features, keep labels as lists."""
    return {
        'features':  torch.stack([s['features'] for s in batch]),  # [B, 256, H, W]
        'centroids': [s['centroids'] for s in batch],
        'classes':   [s['classes'] for s in batch],
        'tile_id':   [s['tile_id'] for s in batch],
        'aug_idx':   [s['aug_idx'] for s in batch],
    }
