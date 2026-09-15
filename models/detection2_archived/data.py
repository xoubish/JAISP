from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .common import resolve
from detection.dataset import TileDetectionDataset
from detection.masks import bright_star_saturation_mask, load_gaia_cache


def artifact_mask(cfg, tid):
    with np.load(resolve(cfg['euclid_dir']) / f'{tid}_euclid.npz', allow_pickle=True) as ed:
        gaia = load_gaia_cache(str(resolve(cfg['gaia'])))
        return bright_star_saturation_mask(np.nan_to_num(ed['img_VIS']), str(ed['wcs_VIS']), gaia)


def novel_candidates(xy, seeds_px, radius):
    if not len(seeds_px):
        return np.ones(len(xy), dtype=bool)
    return cKDTree(seeds_px).query(xy)[0] > radius


def disk_mask(xy_normalized, shape, radius):
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for x, y in xy_normalized * np.array([w - 1, h - 1]):
        x0, x1 = max(0, int(np.floor(x - radius))), min(w, int(np.ceil(x + radius)) + 1)
        y0, y1 = max(0, int(np.floor(y - radius))), min(h, int(np.ceil(y + radius)) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    return mask


class TrainingDataset(Dataset):
    def __init__(self, cfg, labels, arm):
        self.cfg, self.labels, self.arm = cfg, labels, arm
        self.samples = []
        for tid in sorted(labels):
            paths = sorted(resolve(cfg['feature_dir']).glob(f'{tid}_aug*.pt'))
            if len(paths) != 4:
                raise ValueError(f'Expected four cached augmentations for {tid}')
            self.samples.extend((tid, p) for p in paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        tid, path = self.samples[index]
        cached = torch.load(path, map_location='cpu', weights_only=True)
        aug = cached['aug_params']
        row = self.labels[tid]
        xy = TileDetectionDataset._transform_centroids(row['seeds'], *aug)
        mask = disk_mask(row['unknown'], row['shape'], self.cfg['ignore_radius_vis_px'])
        mask = TileDetectionDataset._transform_mask(mask, *aug)
        if self.arm == 'control':
            mask[:] = False
        return {'features': cached['features'], 'centroids': torch.from_numpy(xy),
                'ignore': torch.from_numpy(mask.copy()), 'tile_id': tid}


def collate(batch):
    return {'features': torch.stack([s['features'] for s in batch]),
            'centroids': [s['centroids'] for s in batch],
            'ignore': torch.stack([s['ignore'] for s in batch]),
            'tile_id': [s['tile_id'] for s in batch]}
