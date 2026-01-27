# jaisp_dataset_v3.py
#
# Dataset for JAISP Foundation v3 - Per-Band Views
# Updated to match actual JAISP data format:
#   - Rubin: tile_x*_y*.npz with 'img' (6,H,W), 'var' (6,H,W)
#   - Euclid: tile_x*_y*_euclid.npz with 'img_VIS', 'img_Y', etc.

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random


# Band definitions matching your data
RUBIN_BANDS = ['rubin_u', 'rubin_g', 'rubin_r', 'rubin_i', 'rubin_z', 'rubin_y']
RUBIN_BAND_ORDER = ['u', 'g', 'r', 'i', 'z', 'y']  # Order in the (6,H,W) array

EUCLID_BANDS = ['euclid_VIS', 'euclid_Y', 'euclid_J', 'euclid_H']
EUCLID_BAND_KEYS = ['VIS', 'Y', 'J', 'H']  # Keys in the npz file

ALL_BANDS = RUBIN_BANDS + EUCLID_BANDS

# Wavelength ordering (approximate central wavelength in nm)
BAND_WAVELENGTHS = {
    'rubin_u': 367,
    'rubin_g': 482,
    'rubin_r': 622,
    'rubin_i': 755,
    'rubin_z': 869,
    'rubin_y': 971,
    'euclid_VIS': 700,
    'euclid_Y': 1020,
    'euclid_J': 1250,
    'euclid_H': 1650,
}


def _to_float32(x):
    x = np.asarray(x)
    return x.astype(np.float32, copy=False) if x.dtype != np.float32 else x


def _safe_sqrt_var(var: np.ndarray) -> np.ndarray:
    """sqrt(max(var,0)) while preserving NaNs."""
    var = _to_float32(var)
    out = var.copy()
    m = np.isfinite(out)
    out[m] = np.maximum(out[m], 0.0)
    return np.sqrt(out, dtype=np.float32)


class PairingSampler:
    """
    Intelligent band pair sampling.
    
    Ensures:
    - All bands participate regularly
    - Hard pairs (cross-instrument, large wavelength gap) included
    - Easy pairs (same instrument, adjacent bands) for stability
    """
    def __init__(self, 
                 available_bands: List[str],
                 hard_pair_prob: float = 0.3,
                 cross_instrument_prob: float = 0.5,
                 seed: int = 42):
        self.available_bands = available_bands
        self.hard_pair_prob = hard_pair_prob
        self.cross_instrument_prob = cross_instrument_prob
        self.rng = np.random.RandomState(seed)
        
        # Categorize bands
        self.rubin_bands = [b for b in available_bands if b.startswith('rubin')]
        self.euclid_bands = [b for b in available_bands if b.startswith('euclid')]
        
        # Track usage to ensure balance
        self.band_usage = {b: 0 for b in available_bands}
    
    def sample_pair(self, tile_available_bands: List[str]) -> Tuple[str, str]:
        """Sample a pair of bands for training."""
        available = [b for b in tile_available_bands if b in self.available_bands]
        
        if len(available) < 2:
            if len(available) == 1:
                return available[0], available[0]
            return None, None
        
        r = self.rng.rand()
        
        rubin_avail = [b for b in available if b.startswith('rubin')]
        euclid_avail = [b for b in available if b.startswith('euclid')]
        
        if r < self.cross_instrument_prob and rubin_avail and euclid_avail:
            # Cross-instrument pair
            band1 = self._sample_least_used(rubin_avail)
            band2 = self._sample_least_used(euclid_avail)
        elif r < self.cross_instrument_prob + self.hard_pair_prob:
            # Hard pair: large wavelength gap
            band1, band2 = self._sample_wavelength_distant(available)
        else:
            # Random pair
            band1, band2 = self.rng.choice(available, 2, replace=False)
        
        self.band_usage[band1] += 1
        self.band_usage[band2] += 1
        
        return band1, band2
    
    def _sample_least_used(self, bands: List[str]) -> str:
        usage = [self.band_usage[b] for b in bands]
        min_usage = min(usage)
        least_used = [b for b, u in zip(bands, usage) if u == min_usage]
        return self.rng.choice(least_used)
    
    def _sample_wavelength_distant(self, bands: List[str]) -> Tuple[str, str]:
        wavelengths = [(b, BAND_WAVELENGTHS.get(b, 500)) for b in bands]
        wavelengths.sort(key=lambda x: x[1])
        n = len(wavelengths)
        if n >= 4:
            blue_half = [b for b, _ in wavelengths[:n//2]]
            red_half = [b for b, _ in wavelengths[n//2:]]
            return self.rng.choice(blue_half), self.rng.choice(red_half)
        else:
            return wavelengths[0][0], wavelengths[-1][0]


class JAISPPerBandDataset(Dataset):
    """
    Dataset that treats each band as a separate view.
    
    Data format:
        rubin_dir/tile_x*_y*.npz:
            'img': (6, H, W) - 6 bands stacked [u,g,r,i,z,y]
            'var': (6, H, W) - variance per band
        
        euclid_dir/tile_x*_y*_euclid.npz:
            'img_VIS', 'img_Y', 'img_J', 'img_H': (H, W) each
            'var_VIS', 'var_Y', 'var_J', 'var_H': (H, W) each
    
    Returns single-band pairs for training.
    """
    def __init__(self,
                 rubin_dir: str,
                 euclid_dir: str,
                 patch_size: int = 512,
                 hard_pair_prob: float = 0.3,
                 cross_instrument_prob: float = 0.5,
                 augment: bool = True,
                 mmap: bool = True,
                 seed: int = 42):
        
        self.rubin_dir = Path(rubin_dir)
        self.euclid_dir = Path(euclid_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.mmap = mmap
        self.rng = np.random.RandomState(seed)
        
        # Find tiles (Rubin naming: tile_x*_y*.npz)
        rubin_files = sorted(glob.glob(os.path.join(rubin_dir, "tile_x*_y*.npz")))
        self.tile_ids = [os.path.splitext(os.path.basename(p))[0] for p in rubin_files]
        
        # Build pairs list
        self.pairs = []
        for tid in self.tile_ids:
            rubin_path = os.path.join(rubin_dir, f"{tid}.npz")
            euclid_path = os.path.join(euclid_dir, f"{tid}_euclid.npz")
            if os.path.exists(rubin_path):
                self.pairs.append({
                    'tile_id': tid,
                    'rubin_path': rubin_path,
                    'euclid_path': euclid_path,
                    'has_euclid': os.path.exists(euclid_path)
                })
        
        if len(self.pairs) == 0:
            raise FileNotFoundError(f"No Rubin tiles found (tile_x*_y*.npz) in {rubin_dir}")
        
        # Pairing sampler
        self.sampler = PairingSampler(
            ALL_BANDS,
            hard_pair_prob=hard_pair_prob,
            cross_instrument_prob=cross_instrument_prob,
            seed=seed
        )
        
        print(f"JAISPPerBandDataset:")
        print(f"  Tiles: {len(self.pairs)}")
        print(f"  Rubin bands: {RUBIN_BANDS}")
        print(f"  Euclid bands: {EUCLID_BANDS}")
        print(f"  Total bands (views): {len(ALL_BANDS)}")
    
    def _load_rubin_band(self, rubin_path: str, band: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single Rubin band."""
        # band is like 'rubin_u', extract 'u'
        band_letter = band.split('_')[1]
        band_idx = RUBIN_BAND_ORDER.index(band_letter)
        
        data = np.load(rubin_path, mmap_mode='r' if self.mmap else None, allow_pickle=True)
        img = _to_float32(data['img'][band_idx])  # (H, W)
        
        if 'var' in data:
            var = _to_float32(data['var'][band_idx])
            rms = _safe_sqrt_var(var)
        else:
            rms = np.ones_like(img) * np.nanstd(img)
        
        return img, rms
    
    def _load_euclid_band(self, euclid_path: str, band: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load a single Euclid band. Returns None if band missing."""
        # band is like 'euclid_VIS', extract 'VIS'
        band_key = band.split('_')[1]
        
        if not os.path.exists(euclid_path):
            return None, None
        
        data = np.load(euclid_path, mmap_mode='r' if self.mmap else None, allow_pickle=True)
        
        img_key = f'img_{band_key}'
        var_key = f'var_{band_key}'
        
        if img_key not in data:
            return None, None
        
        img = _to_float32(data[img_key])  # (H, W)
        
        if var_key in data:
            var = _to_float32(data[var_key])
            rms = _safe_sqrt_var(var)
        else:
            rms = np.ones_like(img) * np.nanstd(img)
        
        return img, rms
    
    def _get_available_bands(self, pair_info: dict) -> List[str]:
        """Check which bands have valid data for this tile."""
        available = []
        
        # Check Rubin bands
        try:
            data = np.load(pair_info['rubin_path'], mmap_mode='r' if self.mmap else None, allow_pickle=True)
            rubin_img = data['img']
            for i, band in enumerate(RUBIN_BANDS):
                if i < rubin_img.shape[0]:
                    band_data = rubin_img[i]
                    # Check if band has valid data
                    if np.isfinite(band_data).sum() > 0.3 * band_data.size:
                        available.append(band)
        except Exception as e:
            pass
        
        # Check Euclid bands
        if pair_info['has_euclid']:
            try:
                data = np.load(pair_info['euclid_path'], mmap_mode='r' if self.mmap else None, allow_pickle=True)
                for band, band_key in zip(EUCLID_BANDS, EUCLID_BAND_KEYS):
                    img_key = f'img_{band_key}'
                    if img_key in data:
                        band_data = data[img_key]
                        if np.isfinite(band_data).sum() > 0.3 * band_data.size:
                            available.append(band)
            except Exception as e:
                pass
        
        return available
    
    def _center_crop_or_pad(self, image: np.ndarray, rms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Center crop or pad to patch_size."""
        H, W = image.shape
        target = self.patch_size
        
        # Pad if needed
        if H < target or W < target:
            pad_h = max(0, target - H)
            pad_w = max(0, target - W)
            image = np.pad(image, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)),
                          mode='reflect')
            rms = np.pad(rms, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)),
                        mode='reflect')
            H, W = image.shape
        
        # Center crop
        start_h = (H - target) // 2
        start_w = (W - target) // 2
        
        return (image[start_h:start_h+target, start_w:start_w+target],
                rms[start_h:start_h+target, start_w:start_w+target])
    
    def _augment(self, img1: np.ndarray, rms1: np.ndarray, 
                 img2: np.ndarray, rms2: np.ndarray) -> Tuple:
        """Apply same augmentation to both views."""
        k = self.rng.randint(4)
        img1 = np.rot90(img1, k).copy()
        rms1 = np.rot90(rms1, k).copy()
        img2 = np.rot90(img2, k).copy()
        rms2 = np.rot90(rms2, k).copy()
        
        if self.rng.rand() > 0.5:
            img1 = np.flip(img1, axis=0).copy()
            rms1 = np.flip(rms1, axis=0).copy()
            img2 = np.flip(img2, axis=0).copy()
            rms2 = np.flip(rms2, axis=0).copy()
        
        if self.rng.rand() > 0.5:
            img1 = np.flip(img1, axis=1).copy()
            rms1 = np.flip(rms1, axis=1).copy()
            img2 = np.flip(img2, axis=1).copy()
            rms2 = np.flip(rms2, axis=1).copy()
        
        return img1, rms1, img2, rms2
    
    def _load_band(self, pair_info: dict, band: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a band (either Rubin or Euclid)."""
        if band.startswith('rubin'):
            return self._load_rubin_band(pair_info['rubin_path'], band)
        else:
            img, rms = self._load_euclid_band(pair_info['euclid_path'], band)
            if img is None:
                raise ValueError(f"Band {band} not available")
            return img, rms
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        pair_info = self.pairs[idx]
        
        # Get available bands for this tile
        available = self._get_available_bands(pair_info)
        
        if len(available) < 2:
            # Fallback: use first two Rubin bands
            available = RUBIN_BANDS[:2]
        
        # Sample band pair
        band1, band2 = self.sampler.sample_pair(available)
        
        if band1 is None:
            band1, band2 = available[0], available[1] if len(available) > 1 else available[0]
        
        # Load bands
        try:
            img1, rms1 = self._load_band(pair_info, band1)
            img2, rms2 = self._load_band(pair_info, band2)
        except Exception as e:
            # Fallback
            print(f"Warning: Failed to load {band1}/{band2} for {pair_info['tile_id']}: {e}")
            img1, rms1 = self._load_rubin_band(pair_info['rubin_path'], RUBIN_BANDS[0])
            img2, rms2 = self._load_rubin_band(pair_info['rubin_path'], RUBIN_BANDS[1])
            band1, band2 = RUBIN_BANDS[0], RUBIN_BANDS[1]
        
        # Handle NaNs
        img1 = np.nan_to_num(img1, nan=0.0)
        img2 = np.nan_to_num(img2, nan=0.0)
        rms1 = np.nan_to_num(rms1, nan=1.0)
        rms2 = np.nan_to_num(rms2, nan=1.0)
        rms1 = np.maximum(rms1, 1e-10)
        rms2 = np.maximum(rms2, 1e-10)
        
        # Center crop/pad
        img1, rms1 = self._center_crop_or_pad(img1, rms1)
        img2, rms2 = self._center_crop_or_pad(img2, rms2)
        
        # Augment
        if self.augment:
            img1, rms1, img2, rms2 = self._augment(img1, rms1, img2, rms2)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        img1 = img1[np.newaxis, ...]
        rms1 = rms1[np.newaxis, ...]
        img2 = img2[np.newaxis, ...]
        rms2 = rms2[np.newaxis, ...]
        
        return {
            'view1_image': torch.from_numpy(img1.copy()),
            'view1_rms': torch.from_numpy(rms1.copy()),
            'view1_band': band1,
            'view2_image': torch.from_numpy(img2.copy()),
            'view2_rms': torch.from_numpy(rms2.copy()),
            'view2_band': band2,
            'tile_id': pair_info['tile_id']
        }
    
    def get_band_names(self) -> List[str]:
        """Return all band names for model initialization."""
        return ALL_BANDS


def make_loader(rubin_dir: str,
                euclid_dir: str,
                batch_size: int = 4,
                num_workers: int = 4,
                **kwargs) -> Tuple[Dataset, DataLoader]:
    """Create dataset and dataloader."""
    
    dataset = JAISPPerBandDataset(
        rubin_dir=rubin_dir,
        euclid_dir=euclid_dir,
        **kwargs
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True
    )
    
    return dataset, loader


# =============================================================================
# MULTI-VIEW EXTENSION (3+ views at once)
# =============================================================================

class JAISPMultiViewDataset(JAISPPerBandDataset):
    """
    Extension that can return 3+ views at once for multi-way consistency.
    """
    def __init__(self, *args, n_views: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_views = n_views
    
    def __getitem__(self, idx: int) -> Dict:
        pair_info = self.pairs[idx]
        available = self._get_available_bands(pair_info)
        
        n = min(self.n_views, len(available))
        if n < 2:
            n = 2
            available = RUBIN_BANDS[:2]
        
        bands = list(self.rng.choice(available, n, replace=False))
        
        views = []
        for band in bands:
            try:
                img, rms = self._load_band(pair_info, band)
                img = np.nan_to_num(img, nan=0.0)
                rms = np.nan_to_num(rms, nan=1.0)
                rms = np.maximum(rms, 1e-10)
                img, rms = self._center_crop_or_pad(img, rms)
                views.append({'image': img, 'rms': rms, 'band': band})
            except:
                continue
        
        # Apply same augmentation to all
        if self.augment and len(views) >= 2:
            k = self.rng.randint(4)
            flip_h = self.rng.rand() > 0.5
            flip_w = self.rng.rand() > 0.5
            
            for v in views:
                v['image'] = np.rot90(v['image'], k).copy()
                v['rms'] = np.rot90(v['rms'], k).copy()
                if flip_h:
                    v['image'] = np.flip(v['image'], axis=0).copy()
                    v['rms'] = np.flip(v['rms'], axis=0).copy()
                if flip_w:
                    v['image'] = np.flip(v['image'], axis=1).copy()
                    v['rms'] = np.flip(v['rms'], axis=1).copy()
        
        result = {'tile_id': pair_info['tile_id'], 'n_views': len(views)}
        for i, v in enumerate(views):
            result[f'view{i+1}_image'] = torch.from_numpy(v['image'][np.newaxis, ...].copy())
            result[f'view{i+1}_rms'] = torch.from_numpy(v['rms'][np.newaxis, ...].copy())
            result[f'view{i+1}_band'] = v['band']
        
        return result