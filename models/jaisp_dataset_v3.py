# jaisp_dataset_v3.py
#
# Dataset for JAISP Foundation v3 - Per-Band Views
#
# Each band is a separate view. Dataset handles:
# - Extracting individual bands from multi-band files
# - Smart pairing: cross-instrument, cross-wavelength, same-instrument
# - Missing band handling
# - Per-band RMS

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random


# Band definitions
RUBIN_BANDS = ['rubin_u', 'rubin_g', 'rubin_r', 'rubin_i', 'rubin_z', 'rubin_y']
EUCLID_BANDS = ['euclid_vis', 'euclid_y', 'euclid_j', 'euclid_h']
ALL_BANDS = RUBIN_BANDS + EUCLID_BANDS

# Wavelength ordering (approximate central wavelength in nm)
BAND_WAVELENGTHS = {
    'rubin_u': 367,
    'rubin_g': 482,
    'rubin_r': 622,
    'rubin_i': 755,
    'rubin_z': 869,
    'rubin_y': 971,
    'euclid_vis': 700,  # Broad optical
    'euclid_y': 1020,
    'euclid_j': 1250,
    'euclid_h': 1650,
}


class PairingSampler:
    """
    Intelligent band pair sampling following the roadmap.
    
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
        """
        Sample a pair of bands for training.
        
        Args:
            tile_available_bands: which bands are available for this tile
        
        Returns:
            (band1, band2)
        """
        available = [b for b in tile_available_bands if b in self.available_bands]
        
        if len(available) < 2:
            # Not enough bands, return what we have (will be handled upstream)
            return available[0], available[0] if len(available) == 1 else (None, None)
        
        # Decide pairing strategy
        r = self.rng.rand()
        
        rubin_avail = [b for b in available if b.startswith('rubin')]
        euclid_avail = [b for b in available if b.startswith('euclid')]
        
        if r < self.cross_instrument_prob and rubin_avail and euclid_avail:
            # Cross-instrument pair
            band1 = self._sample_least_used(rubin_avail)
            band2 = self._sample_least_used(euclid_avail)
        elif r < self.cross_instrument_prob + self.hard_pair_prob:
            # Hard pair: large wavelength gap within available
            band1, band2 = self._sample_wavelength_distant(available)
        else:
            # Random pair
            band1, band2 = self.rng.choice(available, 2, replace=False)
        
        self.band_usage[band1] += 1
        self.band_usage[band2] += 1
        
        return band1, band2
    
    def _sample_least_used(self, bands: List[str]) -> str:
        """Sample from least-used bands to ensure balance"""
        usage = [self.band_usage[b] for b in bands]
        min_usage = min(usage)
        least_used = [b for b, u in zip(bands, usage) if u == min_usage]
        return self.rng.choice(least_used)
    
    def _sample_wavelength_distant(self, bands: List[str]) -> Tuple[str, str]:
        """Sample pair with large wavelength difference"""
        wavelengths = [(b, BAND_WAVELENGTHS.get(b, 500)) for b in bands]
        wavelengths.sort(key=lambda x: x[1])
        
        # Pick from ends (blue and red)
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
    
    Expected data format:
        rubin_dir/tile_001.npz:
            'image': (6, H, W) - 6 bands stacked
            'rms': (6, H, W) - per-band RMS
        
        euclid_dir/tile_001.npz:
            'image': (4, H, W) - 4 bands stacked
            'rms': (4, H, W)
    
    Returns single-band pairs for training.
    """
    def __init__(self,
                 rubin_dir: str,
                 euclid_dir: str,
                 rubin_bands: List[str] = RUBIN_BANDS,
                 euclid_bands: List[str] = EUCLID_BANDS,
                 patch_size: int = 512,
                 hard_pair_prob: float = 0.3,
                 cross_instrument_prob: float = 0.5,
                 augment: bool = True,
                 seed: int = 42):
        
        self.rubin_dir = Path(rubin_dir)
        self.euclid_dir = Path(euclid_dir)
        self.rubin_bands = rubin_bands
        self.euclid_bands = euclid_bands
        self.all_bands = rubin_bands + euclid_bands
        self.patch_size = patch_size
        self.augment = augment
        self.rng = np.random.RandomState(seed)
        
        # Band index mapping
        self.rubin_band_idx = {b: i for i, b in enumerate(rubin_bands)}
        self.euclid_band_idx = {b: i for i, b in enumerate(euclid_bands)}
        
        # Find matching tiles
        rubin_tiles = {f.stem for f in self.rubin_dir.glob("*.npz")} if self.rubin_dir.exists() else set()
        euclid_tiles = {f.stem for f in self.euclid_dir.glob("*.npz")} if self.euclid_dir.exists() else set()
        self.tiles = sorted(rubin_tiles & euclid_tiles)
        
        # Pairing sampler
        self.sampler = PairingSampler(
            self.all_bands,
            hard_pair_prob=hard_pair_prob,
            cross_instrument_prob=cross_instrument_prob,
            seed=seed
        )
        
        print(f"JAISPPerBandDataset:")
        print(f"  Tiles: {len(self.tiles)}")
        print(f"  Rubin bands: {rubin_bands}")
        print(f"  Euclid bands: {euclid_bands}")
        print(f"  Total bands (views): {len(self.all_bands)}")
    
    def _load_band(self, tile_id: str, band: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single band from the appropriate file.
        
        Returns:
            image: (H, W) single band
            rms: (H, W) noise for this band
        """
        if band.startswith('rubin'):
            path = self.rubin_dir / f"{tile_id}.npz"
            band_idx = self.rubin_band_idx[band]
        else:
            path = self.euclid_dir / f"{tile_id}.npz"
            band_idx = self.euclid_band_idx[band]
        
        data = np.load(path)
        
        # Handle different key conventions
        if 'image' in data:
            image = data['image'][band_idx]
        elif 'data' in data:
            image = data['data'][band_idx]
        else:
            image = data[list(data.keys())[0]][band_idx]
        
        if 'rms' in data:
            rms = data['rms'][band_idx]
        elif 'noise' in data:
            rms = data['noise'][band_idx]
        elif 'weight' in data:
            w = data['weight'][band_idx]
            rms = 1.0 / np.sqrt(np.maximum(w, 1e-10))
        else:
            # Estimate from image
            rms = np.ones_like(image) * np.std(image)
        
        return image.astype(np.float32), rms.astype(np.float32)
    
    def _get_available_bands(self, tile_id: str) -> List[str]:
        """Check which bands have valid data for this tile"""
        available = []
        
        # Check Rubin
        rubin_path = self.rubin_dir / f"{tile_id}.npz"
        if rubin_path.exists():
            data = np.load(rubin_path)
            img = data['image'] if 'image' in data else data[list(data.keys())[0]]
            for i, band in enumerate(self.rubin_bands):
                if i < img.shape[0]:
                    # Check if band has valid data (not all NaN or zero)
                    band_data = img[i]
                    if np.isfinite(band_data).sum() > 0.5 * band_data.size:
                        available.append(band)
        
        # Check Euclid
        euclid_path = self.euclid_dir / f"{tile_id}.npz"
        if euclid_path.exists():
            data = np.load(euclid_path)
            img = data['image'] if 'image' in data else data[list(data.keys())[0]]
            for i, band in enumerate(self.euclid_bands):
                if i < img.shape[0]:
                    band_data = img[i]
                    if np.isfinite(band_data).sum() > 0.5 * band_data.size:
                        available.append(band)
        
        return available
    
    def _center_crop(self, image: np.ndarray, rms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Center crop to patch_size"""
        H, W = image.shape
        
        # Pad if needed
        if H < self.patch_size or W < self.patch_size:
            pad_h = max(0, self.patch_size - H)
            pad_w = max(0, self.patch_size - W)
            image = np.pad(image, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)),
                          mode='reflect')
            rms = np.pad(rms, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)),
                        mode='reflect')
            H, W = image.shape
        
        start_h = (H - self.patch_size) // 2
        start_w = (W - self.patch_size) // 2
        
        return (image[start_h:start_h+self.patch_size, start_w:start_w+self.patch_size],
                rms[start_h:start_h+self.patch_size, start_w:start_w+self.patch_size])
    
    def _augment(self, img1: np.ndarray, rms1: np.ndarray, 
                 img2: np.ndarray, rms2: np.ndarray) -> Tuple:
        """Apply same augmentation to both views"""
        # Random rotation
        k = self.rng.randint(4)
        img1 = np.rot90(img1, k).copy()
        rms1 = np.rot90(rms1, k).copy()
        img2 = np.rot90(img2, k).copy()
        rms2 = np.rot90(rms2, k).copy()
        
        # Random flip
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
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx: int) -> Dict:
        tile_id = self.tiles[idx]
        
        # Get available bands for this tile
        available = self._get_available_bands(tile_id)
        
        if len(available) < 2:
            # Fallback: try to at least get something
            available = self.all_bands[:2]
        
        # Sample band pair
        band1, band2 = self.sampler.sample_pair(available)
        
        # Load bands
        try:
            img1, rms1 = self._load_band(tile_id, band1)
            img2, rms2 = self._load_band(tile_id, band2)
        except Exception as e:
            # Fallback to first available
            print(f"Warning: Failed to load {band1}/{band2} for {tile_id}: {e}")
            img1, rms1 = self._load_band(tile_id, available[0])
            img2, rms2 = self._load_band(tile_id, available[1] if len(available) > 1 else available[0])
            band1, band2 = available[0], available[1] if len(available) > 1 else available[0]
        
        # Center crop (handle different sizes)
        img1, rms1 = self._center_crop(img1, rms1)
        img2, rms2 = self._center_crop(img2, rms2)
        
        # Augment
        if self.augment:
            img1, rms1, img2, rms2 = self._augment(img1, rms1, img2, rms2)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        img1 = img1[np.newaxis, ...]
        rms1 = rms1[np.newaxis, ...]
        img2 = img2[np.newaxis, ...]
        rms2 = rms2[np.newaxis, ...]
        
        return {
            'view1_image': torch.from_numpy(img1),
            'view1_rms': torch.from_numpy(rms1),
            'view1_band': band1,
            'view2_image': torch.from_numpy(img2),
            'view2_rms': torch.from_numpy(rms2),
            'view2_band': band2,
            'tile_id': tile_id
        }
    
    def get_band_names(self) -> List[str]:
        """Return all band names for model initialization"""
        return self.all_bands


def make_loader(rubin_dir: str,
                euclid_dir: str,
                batch_size: int = 4,
                num_workers: int = 4,
                **kwargs) -> Tuple[Dataset, DataLoader]:
    """Create dataset and dataloader"""
    
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
    
    Useful for stronger constraints: "all N views of the same patch should
    map to similar representations"
    """
    def __init__(self, *args, n_views: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_views = n_views
    
    def __getitem__(self, idx: int) -> Dict:
        tile_id = self.tiles[idx]
        available = self._get_available_bands(tile_id)
        
        # Sample n_views bands
        n = min(self.n_views, len(available))
        if n < 2:
            n = 2
            available = self.all_bands[:2]
        
        bands = list(self.rng.choice(available, n, replace=False))
        
        views = []
        for band in bands:
            try:
                img, rms = self._load_band(tile_id, band)
                img, rms = self._center_crop(img, rms)
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
        
        # Format output
        result = {'tile_id': tile_id, 'n_views': len(views)}
        for i, v in enumerate(views):
            result[f'view{i+1}_image'] = torch.from_numpy(v['image'][np.newaxis, ...])
            result[f'view{i+1}_rms'] = torch.from_numpy(v['rms'][np.newaxis, ...])
            result[f'view{i+1}_band'] = v['band']
        
        return result
