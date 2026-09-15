"""Read one prepared tile (a batch of fixed crops) per training step."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

BANDS = ['rubin_'+b for b in 'ugrizy']+['euclid_'+b for b in ('VIS', 'Y', 'J', 'H')]


class CropTiles(Dataset):
    def __init__(self, prepared, tile_ids):
        self.prepared, self.tile_ids = Path(prepared), list(tile_ids)

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, index):
        tid = self.tile_ids[index]
        with np.load(self.prepared/'tiles'/f'{tid}.npz', allow_pickle=False) as z:
            ids = z['object_indices']
            present = ids >= 0
            features = z['object_features'][np.maximum(ids, 0)].astype(np.float32)
            features[~present] = 0
            result = {'tile_id': tid, 'features': torch.from_numpy(features),
                      'present': torch.from_numpy(present), 'positions': torch.from_numpy(z['positions']),
                      'targets': [torch.from_numpy(z[f'target_{i}']) for i in range(len(BANDS))],
                      'valid': [torch.from_numpy(z[f'valid_{i}']) for i in range(len(BANDS))],
                      'source': [torch.from_numpy(z[f'source_{i}']) for i in range(len(BANDS))]}
        return result


def to_device(batch, device):
    return {k: [t.to(device, non_blocking=True) for t in v] if isinstance(v, list)
            else v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}
