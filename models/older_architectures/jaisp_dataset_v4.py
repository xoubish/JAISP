# jaisp_dataset_v4.py
#
# Dataset for JAISP Foundation - Native resolutions preserved
#
# Rubin: 512×512, Euclid: 1050×1050 - SAME sky area
# We keep native sizes; the model handles alignment via token grid interpolation.

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


RUBIN_BANDS = ['rubin_u', 'rubin_g', 'rubin_r', 'rubin_i', 'rubin_z', 'rubin_y']
RUBIN_BAND_ORDER = ['u', 'g', 'r', 'i', 'z', 'y']

EUCLID_BANDS = ['euclid_VIS', 'euclid_Y', 'euclid_J', 'euclid_H']
EUCLID_BAND_KEYS = ['VIS', 'Y', 'J', 'H']

ALL_BANDS = RUBIN_BANDS + EUCLID_BANDS

BAND_WAVELENGTHS = {
    'rubin_u': 367, 'rubin_g': 482, 'rubin_r': 622,
    'rubin_i': 755, 'rubin_z': 869, 'rubin_y': 971,
    'euclid_VIS': 700, 'euclid_Y': 1020, 'euclid_J': 1250, 'euclid_H': 1650,
}


def _to_float32(x):
    return np.asarray(x).astype(np.float32, copy=False)


def _safe_sqrt_var(var):
    var = _to_float32(var)
    out = var.copy()
    m = np.isfinite(out)
    out[m] = np.maximum(out[m], 0.0)
    return np.sqrt(out, dtype=np.float32)


class PairingSampler:
    """
    Picks (b1, b2) from available bands.

    Key knobs:
      - cross_prob: probability of sampling a Rubin↔Euclid pair when both exist.
      - balance_usage: bias sampling toward less-used bands to keep distribution healthy.
    """
    def __init__(
        self,
        bands: List[str],
        cross_prob: float = 0.9,
        seed: int = 42,
        balance_usage: bool = True,
    ):
        self.bands = list(bands)
        self.cross_prob = float(cross_prob)
        self.rng = np.random.RandomState(seed)
        self.balance_usage = bool(balance_usage)
        self.usage = {b: 0 for b in self.bands}

    @staticmethod
    def _is_rubin(b: str) -> bool:
        return b.startswith("rubin_")

    @staticmethod
    def _is_euclid(b: str) -> bool:
        return b.startswith("euclid_")

    def _choice_balanced(self, candidates: List[str]) -> str:
        """Pick a candidate, biased toward the least used ones."""
        if not candidates:
            return None
        if not self.balance_usage:
            return self.rng.choice(candidates)
        counts = np.array([self.usage.get(b, 0) for b in candidates], dtype=np.float32)
        m = counts.min()
        least = [b for b, c in zip(candidates, counts) if c == m]
        return self.rng.choice(least)

    def sample(self, available: List[str]) -> Tuple[Optional[str], Optional[str]]:
        avail = [b for b in available if b in self.bands]
        if len(avail) == 0:
            return None, None
        if len(avail) == 1:
            return avail[0], avail[0]

        rubin_avail = [b for b in avail if self._is_rubin(b)]
        euclid_avail = [b for b in avail if self._is_euclid(b)]

        do_cross = (self.rng.rand() < self.cross_prob) and (len(rubin_avail) > 0) and (len(euclid_avail) > 0)

        if do_cross:
            b1 = self._choice_balanced(rubin_avail)
            b2 = self._choice_balanced(euclid_avail)
        else:
            # within-set: still try to avoid picking two from the exact same most-used band subset
            # pick first balanced, second balanced from remaining
            b1 = self._choice_balanced(avail)
            remaining = [b for b in avail if b != b1]
            if not remaining:
                b2 = b1
            else:
                b2 = self._choice_balanced(remaining)

        self.usage[b1] = self.usage.get(b1, 0) + 1
        self.usage[b2] = self.usage.get(b2, 0) + 1
        return b1, b2


class JAISPDatasetV4(Dataset):
    """
    Dataset preserving native resolutions.

    Rubin: 512×512 (coarser pixels)
    Euclid: 1050×1050 (finer pixels)
    Both cover the SAME sky area.

    Returns images at their native sizes. The model handles
    token grid alignment via interpolation.
    """
    def __init__(
        self,
        rubin_dir: str,
        euclid_dir: str,
        augment: bool = True,
        mmap: bool = True,
        seed: int = 42,
        cross_prob: float = 0.9,        # NEW: strongly encourage Rubin↔Euclid pairs
        balance_usage: bool = True,     # NEW: keep band sampling distribution healthier
        precompute_available: bool = True,  # NEW: avoid re-opening NPZ every __getitem__
        finite_frac_thresh: float = 0.30,   # NEW: keep your previous 0.3 but configurable
    ):
        self.rubin_dir = Path(rubin_dir)
        self.euclid_dir = Path(euclid_dir)
        self.augment = augment
        self.mmap = mmap
        self.rng = np.random.RandomState(seed)
        self.finite_frac_thresh = float(finite_frac_thresh)

        # Find tiles
        rubin_files = sorted(glob.glob(os.path.join(rubin_dir, "tile_x*_y*.npz")))
        self.tile_ids = [os.path.splitext(os.path.basename(p))[0] for p in rubin_files]

        self.pairs = []
        for tid in self.tile_ids:
            rp = os.path.join(rubin_dir, f"{tid}.npz")
            ep = os.path.join(euclid_dir, f"{tid}_euclid.npz")
            if os.path.exists(rp):
                self.pairs.append({
                    'tile_id': tid,
                    'rubin_path': rp,
                    'euclid_path': ep,
                    'has_euclid': os.path.exists(ep),
                })

        if not self.pairs:
            raise FileNotFoundError(f"No tiles found in {rubin_dir}")

        self.sampler = PairingSampler(
            ALL_BANDS,
            cross_prob=cross_prob,
            seed=seed,
            balance_usage=balance_usage,
        )

        # Optional: precompute available bands once (makes pairing distribution stable & faster)
        self._avail_cache: Optional[List[List[str]]] = None
        if precompute_available:
            self._avail_cache = [self._get_available_bands(p) for p in self.pairs]

        print(f"JAISPDatasetV4: {len(self.pairs)} tiles (native resolutions)")
        print(f"  pairing: cross_prob={cross_prob}, balance_usage={balance_usage}, precompute_available={precompute_available}")

    def _load_rubin_band(self, path: str, band: str) -> Tuple[np.ndarray, np.ndarray]:
        band_idx = RUBIN_BAND_ORDER.index(band.split('_')[1])
        data = np.load(path, mmap_mode='r' if self.mmap else None, allow_pickle=True)
        img = _to_float32(data['img'][band_idx])
        rms = _safe_sqrt_var(data['var'][band_idx]) if 'var' in data else (np.ones_like(img) * np.nanstd(img))
        return img, rms

    def _load_euclid_band(self, path: str, band: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not os.path.exists(path):
            return None, None

        band_key = band.split('_')[1]
        data = np.load(path, mmap_mode='r' if self.mmap else None, allow_pickle=True)

        img_key = f'img_{band_key}'
        if img_key not in data:
            return None, None

        img = _to_float32(data[img_key])
        var_key = f'var_{band_key}'
        rms = _safe_sqrt_var(data[var_key]) if var_key in data else (np.ones_like(img) * np.nanstd(img))
        return img, rms

    def _finite_frac_ok(self, arr: np.ndarray) -> bool:
        frac = np.isfinite(arr).sum() / float(arr.size)
        return frac > self.finite_frac_thresh

    def _get_available_bands(self, pair: dict) -> List[str]:
        available: List[str] = []

        # Rubin availability
        try:
            data = np.load(pair['rubin_path'], mmap_mode='r' if self.mmap else None, allow_pickle=True)
            imgs = data['img']
            for i, band in enumerate(RUBIN_BANDS):
                if i < imgs.shape[0]:
                    if self._finite_frac_ok(imgs[i]):
                        available.append(band)
        except Exception:
            pass

        # Euclid availability
        if pair['has_euclid']:
            try:
                data = np.load(pair['euclid_path'], mmap_mode='r' if self.mmap else None, allow_pickle=True)
                for band, key in zip(EUCLID_BANDS, EUCLID_BAND_KEYS):
                    k = f'img_{key}'
                    if k in data and self._finite_frac_ok(data[k]):
                        available.append(band)
            except Exception:
                pass

        return available

    def _load_band(self, pair: dict, band: str):
        if band.startswith('rubin_'):
            return self._load_rubin_band(pair['rubin_path'], band)
        else:
            return self._load_euclid_band(pair['euclid_path'], band)

    def _augment(self, img: np.ndarray, rms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Augment single image (same random state should be used for pairs)."""
        k = self.rng.randint(4)
        img = np.rot90(img, k).copy()
        rms = np.rot90(rms, k).copy()

        if self.rng.rand() > 0.5:
            img = np.flip(img, 0).copy()
            rms = np.flip(rms, 0).copy()
        if self.rng.rand() > 0.5:
            img = np.flip(img, 1).copy()
            rms = np.flip(rms, 1).copy()

        return img, rms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]

        if self._avail_cache is not None:
            available = self._avail_cache[idx]
        else:
            available = self._get_available_bands(pair)

        # Fallback: if something is weird, at least return a Rubin-Rubin pair
        if len(available) < 2:
            available = RUBIN_BANDS[:2]

        b1, b2 = self.sampler.sample(available)
        if b1 is None:
            b1 = available[0]
            b2 = available[1] if len(available) > 1 else available[0]

        try:
            img1, rms1 = self._load_band(pair, b1)
            img2, rms2 = self._load_band(pair, b2)
            if img1 is None or img2 is None:
                raise ValueError("Band not available")
        except Exception:
            img1, rms1 = self._load_rubin_band(pair['rubin_path'], RUBIN_BANDS[0])
            img2, rms2 = self._load_rubin_band(pair['rubin_path'], RUBIN_BANDS[1])
            b1, b2 = RUBIN_BANDS[0], RUBIN_BANDS[1]

        # Clean NaNs
        img1 = np.nan_to_num(img1, nan=0.0)
        img2 = np.nan_to_num(img2, nan=0.0)
        rms1 = np.maximum(np.nan_to_num(rms1, nan=1.0), 1e-10)
        rms2 = np.maximum(np.nan_to_num(rms2, nan=1.0), 1e-10)

        # Augment with same random seed for both views
        if self.augment:
            state = self.rng.get_state()
            img1, rms1 = self._augment(img1, rms1)
            self.rng.set_state(state)
            img2, rms2 = self._augment(img2, rms2)

        is_cross = (b1.startswith("rubin_") and b2.startswith("euclid_")) or (b1.startswith("euclid_") and b2.startswith("rubin_"))

        return {
            'view1_image': torch.from_numpy(img1[None].copy()),
            'view1_rms': torch.from_numpy(rms1[None].copy()),
            'view1_band': b1,
            'view1_size': img1.shape,  # (H, W)

            'view2_image': torch.from_numpy(img2[None].copy()),
            'view2_rms': torch.from_numpy(rms2[None].copy()),
            'view2_band': b2,
            'view2_size': img2.shape,

            'tile_id': pair['tile_id'],
            'is_cross_pair': bool(is_cross),  # helpful for logging / debugging
        }

    def get_band_names(self):
        return ALL_BANDS


def collate_variable_size(batch):
    """
    Custom collate for variable-sized images.
    Returns lists instead of stacked tensors for images.
    """
    return {
        'view1_image': [b['view1_image'] for b in batch],
        'view1_rms': [b['view1_rms'] for b in batch],
        'view1_band': [b['view1_band'] for b in batch],
        'view1_size': [b['view1_size'] for b in batch],

        'view2_image': [b['view2_image'] for b in batch],
        'view2_rms': [b['view2_rms'] for b in batch],
        'view2_band': [b['view2_band'] for b in batch],
        'view2_size': [b['view2_size'] for b in batch],

        'tile_id': [b['tile_id'] for b in batch],
        'is_cross_pair': [b.get('is_cross_pair', False) for b in batch],
    }


def make_loader(
    rubin_dir: str,
    euclid_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    **kwargs
):
    """
    kwargs are passed to JAISPDatasetV4, including:
      - cross_prob (recommended: 0.9–1.0 for debugging)
      - balance_usage
      - precompute_available
      - augment, mmap, seed, ...
    """
    dataset = JAISPDatasetV4(rubin_dir=rubin_dir, euclid_dir=euclid_dir, **kwargs)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
        collate_fn=collate_variable_size,
    )

    return dataset, loader
