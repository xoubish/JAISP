# jaisp_dataset_v6.py
#
# Dataset for JAISP Foundation v6 - Masked Band Prediction
#
# Key difference from v4:
#   v4 returned a PAIR of views (two bands from a tile).
#   v6 returns ALL available bands from a tile.
#   The training loop decides which bands are context and which are target.
#
# This separation of concerns makes it easy to experiment with different
# masking strategies (mask 1 band, mask 2 bands, cross-instrument, etc.)
# without changing the dataset.

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple


RUBIN_BANDS = ['rubin_u', 'rubin_g', 'rubin_r', 'rubin_i', 'rubin_z', 'rubin_y']
RUBIN_BAND_ORDER = ['u', 'g', 'r', 'i', 'z', 'y']

EUCLID_BANDS = ['euclid_VIS', 'euclid_Y', 'euclid_J', 'euclid_H']
EUCLID_BAND_KEYS = ['VIS', 'Y', 'J', 'H']

ALL_BANDS = RUBIN_BANDS + EUCLID_BANDS


def _to_float32(x) -> np.ndarray:
    return np.asarray(x).astype(np.float32, copy=False)


def _safe_rms(var: np.ndarray, fallback_image: np.ndarray) -> np.ndarray:
    """sqrt(variance), replacing non-positive/nan with MAD-based estimate."""
    var = _to_float32(var)
    rms = np.zeros_like(var)
    good = np.isfinite(var) & (var > 0)
    rms[good] = np.sqrt(var[good])
    if not good.all():
        # MAD-based fallback for bad pixels
        finite = fallback_image[np.isfinite(fallback_image)]
        if len(finite) > 10:
            mad = np.median(np.abs(finite - np.median(finite)))
            rms[~good] = max(1.4826 * mad, 1e-10)
        else:
            rms[~good] = 1.0
    return rms


class JAISPDatasetV6(Dataset):
    """
    Loads all available bands for each Rubin+Euclid tile pair.

    Each __getitem__ returns a dict with:
        rubin:   {band_name: {'image': Tensor[1,H,W], 'rms': Tensor[1,H,W]}}
                 for each available Rubin band (H=W=512)
        euclid:  {band_name: {'image': Tensor[1,H,W], 'rms': Tensor[1,H,W]}}
                 for each available Euclid band (H=W=1050), or empty dict
        tile_id: str
        has_euclid: bool

    The training loop is responsible for choosing context vs target bands.

    Parameters
    ----------
    rubin_dir:           directory with tile_x*_y*.npz files (Rubin)
    euclid_dir:          directory with tile_x*_y*_euclid.npz files
    load_euclid:         if True, load Euclid bands when available (Phase B)
    augment:             apply 4-fold rotation + flip (same transform for all bands)
    mmap:                use numpy memory-mapping (faster startup, lower RAM)
    min_rubin_bands:     skip tiles with fewer than this many Rubin bands
    finite_frac_thresh:  band is considered available only if this fraction of pixels is finite
    seed:                random seed for augmentation
    """
    def __init__(
        self,
        rubin_dir: str,
        euclid_dir: str,
        load_euclid: bool = False,
        augment: bool = True,
        mmap: bool = True,
        min_rubin_bands: int = 2,
        finite_frac_thresh: float = 0.30,
        seed: int = 42,
    ):
        self.rubin_dir = Path(rubin_dir)
        self.euclid_dir = Path(euclid_dir)
        self.load_euclid = load_euclid
        self.augment = augment
        self.mmap = mmap
        self.min_rubin_bands = min_rubin_bands
        self.finite_frac_thresh = float(finite_frac_thresh)
        self.rng = np.random.RandomState(seed)

        # Discover tiles
        rubin_files = sorted(glob.glob(os.path.join(rubin_dir, 'tile_x*_y*.npz')))
        tile_ids = [os.path.splitext(os.path.basename(p))[0] for p in rubin_files]

        # Pre-scan which bands are available per tile (avoids slow __getitem__ failures)
        print(f"JAISPDatasetV6: scanning {len(tile_ids)} tiles...")
        self.tiles = []
        for tid in tile_ids:
            rp = os.path.join(rubin_dir, f'{tid}.npz')
            ep = os.path.join(euclid_dir, f'{tid}_euclid.npz')
            avail_r = self._scan_rubin(rp)
            if len(avail_r) < min_rubin_bands:
                continue
            avail_e = self._scan_euclid(ep) if (load_euclid and os.path.exists(ep)) else []
            self.tiles.append({
                'tile_id': tid,
                'rubin_path': rp,
                'euclid_path': ep if os.path.exists(ep) else None,
                'avail_rubin': avail_r,
                'avail_euclid': avail_e,
            })

        if not self.tiles:
            raise FileNotFoundError(
                f"No usable tiles found in {rubin_dir} "
                f"(min_rubin_bands={min_rubin_bands})"
            )
        print(f"  {len(self.tiles)} tiles passed quality cuts")
        n_with_euclid = sum(1 for t in self.tiles if t['euclid_path'] and t['avail_euclid'])
        print(f"  {n_with_euclid} tiles have Euclid coverage")

    # ------------------------------------------------------------------
    # Band availability scanning (done once at startup)
    # ------------------------------------------------------------------

    def _finite_ok(self, arr: np.ndarray) -> bool:
        return np.isfinite(arr).mean() > self.finite_frac_thresh

    def _scan_rubin(self, path: str) -> List[str]:
        """Return list of band names with sufficient finite coverage."""
        try:
            data = np.load(path, mmap_mode='r' if self.mmap else None, allow_pickle=True)
            imgs = data['img']  # [6, H, W]
            return [
                RUBIN_BANDS[i]
                for i in range(min(len(RUBIN_BAND_ORDER), imgs.shape[0]))
                if self._finite_ok(imgs[i])
            ]
        except Exception:
            return []

    def _scan_euclid(self, path: str) -> List[str]:
        try:
            data = np.load(path, mmap_mode='r' if self.mmap else None, allow_pickle=True)
            return [
                b for b, key in zip(EUCLID_BANDS, EUCLID_BAND_KEYS)
                if f'img_{key}' in data and self._finite_ok(data[f'img_{key}'])
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Band loading
    # ------------------------------------------------------------------

    def _load_rubin_band(self, path: str, band: str) -> Tuple[np.ndarray, np.ndarray]:
        band_key = band.split('_')[1]
        idx = RUBIN_BAND_ORDER.index(band_key)
        data = np.load(path, mmap_mode='r' if self.mmap else None, allow_pickle=True)
        img = _to_float32(data['img'][idx])
        rms = _safe_rms(data['var'][idx], img) if 'var' in data else np.ones_like(img)
        return img, rms

    def _load_euclid_band(self, path: str, band: str) -> Tuple[np.ndarray, np.ndarray]:
        band_key = band.split('_')[1]
        data = np.load(path, mmap_mode='r' if self.mmap else None, allow_pickle=True)
        img = _to_float32(data[f'img_{band_key}'])
        var_key = f'var_{band_key}'
        rms = _safe_rms(data[var_key], img) if var_key in data else np.ones_like(img)
        return img, rms

    # ------------------------------------------------------------------
    # Augmentation (same random state applied to all bands of a tile)
    # ------------------------------------------------------------------

    def _get_aug_state(self) -> Tuple[int, bool, bool]:
        """Sample augmentation parameters: (n_rot90, flip_ud, flip_lr)."""
        return (
            int(self.rng.randint(4)),
            bool(self.rng.rand() > 0.5),
            bool(self.rng.rand() > 0.5),
        )

    @staticmethod
    def _apply_aug(img: np.ndarray, rms: np.ndarray,
                   n_rot: int, flip_ud: bool, flip_lr: bool
                   ) -> Tuple[np.ndarray, np.ndarray]:
        img = np.rot90(img, n_rot).copy()
        rms = np.rot90(rms, n_rot).copy()
        if flip_ud:
            img = np.flipud(img).copy()
            rms = np.flipud(rms).copy()
        if flip_lr:
            img = np.fliplr(img).copy()
            rms = np.fliplr(rms).copy()
        return img, rms

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Dict:
        tile = self.tiles[idx]
        aug_params = self._get_aug_state() if self.augment else (0, False, False)

        rubin_data = {}
        for band in tile['avail_rubin']:
            try:
                img, rms = self._load_rubin_band(tile['rubin_path'], band)
            except Exception:
                continue
            img = np.nan_to_num(img, nan=0.0)
            rms = np.maximum(np.nan_to_num(rms, nan=1.0), 1e-10)
            if self.augment:
                img, rms = self._apply_aug(img, rms, *aug_params)
            rubin_data[band] = {
                'image': torch.from_numpy(img[None].copy()),   # [1, H, W]
                'rms':   torch.from_numpy(rms[None].copy()),   # [1, H, W]
            }

        euclid_data = {}
        if self.load_euclid and tile['euclid_path'] and tile['avail_euclid']:
            for band in tile['avail_euclid']:
                try:
                    img, rms = self._load_euclid_band(tile['euclid_path'], band)
                except Exception:
                    continue
                img = np.nan_to_num(img, nan=0.0)
                rms = np.maximum(np.nan_to_num(rms, nan=1.0), 1e-10)
                if self.augment:
                    img, rms = self._apply_aug(img, rms, *aug_params)
                euclid_data[band] = {
                    'image': torch.from_numpy(img[None].copy()),
                    'rms':   torch.from_numpy(rms[None].copy()),
                }

        return {
            'tile_id':    tile['tile_id'],
            'rubin':      rubin_data,
            'euclid':     euclid_data,
            'has_euclid': bool(euclid_data),
        }

    def get_band_names(self) -> List[str]:
        return ALL_BANDS


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_v6(batch: List[Dict]) -> List[Dict]:
    """
    Returns a plain list of per-sample dicts.
    Images have variable spatial sizes (Rubin 512 vs Euclid 1050), so we don't
    stack them. The training loop iterates over the list directly.
    """
    return batch


# ============================================================
# MASKING UTILITIES (used by the training loop)
# ============================================================

def sample_context_target(
    sample: Dict,
    rng: np.random.RandomState,
    n_targets: int = 1,
) -> Optional[Dict]:
    """
    Given a dataset sample, randomly select context and target bands.

    Parameters
    ----------
    sample:    output of JAISPDatasetV6.__getitem__
    rng:       random state (caller controls reproducibility)
    n_targets: number of bands to mask (1 or 2)

    Returns None if not enough bands are available.
    Returns dict:
        context_images: {band: Tensor[1, H, W]}
        context_rms:    {band: Tensor[1, H, W]}
        targets:        list of {'band': str, 'image': Tensor, 'rms': Tensor}
    """
    rubin = sample['rubin']
    avail = list(rubin.keys())

    if len(avail) < n_targets + 1:
        return None  # need at least 1 context + n_targets

    # Shuffle and pick targets
    order = rng.permutation(len(avail)).tolist()
    target_bands = [avail[order[i]] for i in range(n_targets)]
    context_bands = [avail[order[i]] for i in range(n_targets, len(avail))]

    context_images = {b: rubin[b]['image'] for b in context_bands}
    context_rms    = {b: rubin[b]['rms']   for b in context_bands}
    targets = [
        {'band': b, 'image': rubin[b]['image'], 'rms': rubin[b]['rms']}
        for b in target_bands
    ]

    return {
        'context_images': context_images,
        'context_rms':    context_rms,
        'targets':        targets,
    }


def sample_context_target_phaseB(
    sample: Dict,
    rng: np.random.RandomState,
    n_targets: int = 1,
) -> Optional[Dict]:
    """
    Phase B: joint Rubin+Euclid masked band prediction.

    Pools ALL available bands (Rubin 512×512 + Euclid downsampled to 512×512),
    randomly holds out n_targets bands as prediction targets, and uses the rest
    as context — same logic as Phase A but across both instruments.

    This forces the encoder to learn cross-instrument spatial correspondence:
    it must predict any band from any combination of the others.

    Returns None if fewer than n_targets+1 bands are available.
    """
    if not sample.get('has_euclid'):
        return None

    rubin  = sample['rubin']
    euclid = sample.get('euclid', {})
    if not rubin or not euclid:
        return None

    # Downsample all Euclid bands to Rubin resolution once
    rubin_h, rubin_w = next(iter(rubin.values()))['image'].shape[-2:]
    euclid_ds = {}
    for band, data in euclid.items():
        img_ds = F.interpolate(
            data['image'].unsqueeze(0).float(), size=(rubin_h, rubin_w),
            mode='bilinear', align_corners=False,
        ).squeeze(0)
        rms_ds = F.interpolate(
            data['rms'].unsqueeze(0).float(), size=(rubin_h, rubin_w),
            mode='bilinear', align_corners=False,
        ).squeeze(0).clamp(min=1e-10)
        euclid_ds[band] = {'image': img_ds, 'rms': rms_ds}

    # Merge into one pool
    pool = {**rubin, **euclid_ds}
    avail = list(pool.keys())

    if len(avail) < n_targets + 1:
        return None

    # Random split: n_targets as targets, rest as context
    order = rng.permutation(len(avail)).tolist()
    target_bands  = [avail[order[i]] for i in range(n_targets)]
    context_bands = [avail[order[i]] for i in range(n_targets, len(avail))]

    return {
        'context_images': {b: pool[b]['image'] for b in context_bands},
        'context_rms':    {b: pool[b]['rms']   for b in context_bands},
        'targets': [
            {'band': b, 'image': pool[b]['image'], 'rms': pool[b]['rms']}
            for b in target_bands
        ],
    }


# ============================================================
# DATALOADER FACTORY
# ============================================================

def make_loader_v6(
    rubin_dir: str,
    euclid_dir: str,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    load_euclid: bool = False,
    **kwargs,
) -> Tuple['JAISPDatasetV6', DataLoader]:
    """
    batch_size > 1 is supported only for within-instrument Rubin-only mode
    where all images are 512×512. With load_euclid=True, use batch_size=1.
    """
    dataset = JAISPDatasetV6(
        rubin_dir=rubin_dir,
        euclid_dir=euclid_dir,
        load_euclid=load_euclid,
        **kwargs,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
        collate_fn=collate_v6,
    )
    return dataset, loader
