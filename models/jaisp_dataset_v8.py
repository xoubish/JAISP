"""Dataset helpers for JAISP Foundation v8 — random crop for fine-scale training.

Wraps the v7/v6 tile loader and adds a spatially-consistent random crop
across Rubin and Euclid bands.  Crop is specified in Rubin pixels
(e.g. 256×256 from 512×512) and the corresponding Euclid crop is
computed from the pixel-scale ratio (Rubin 0.2"/px, Euclid 0.1"/px).

This lets v8 train at finer fused scales (0.4"/px) without re-tiling
the data: a 256×256 Rubin crop at 0.4"/px gives ~128×128 bottleneck
tokens, same as v7's 512×512 at 0.8"/px.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from jaisp_dataset_v6 import (
    ALL_BANDS,
    EUCLID_BANDS,
    JAISPDatasetV6,
    RUBIN_BANDS,
    collate_v6,
)
from jaisp_dataset_v7 import (
    band_group,
    sample_context_target_phaseB_mixed,
)


def random_crop_sample(
    sample: Dict,
    crop_size_rubin: int,
    rng: np.random.RandomState,
) -> Dict:
    """Apply a spatially-consistent random crop to all bands in a sample.

    The crop origin is chosen in Rubin pixel coordinates.  Euclid bands
    are cropped at the corresponding angular region using the 2:1 pixel
    scale ratio (Rubin 0.2"/px → Euclid 0.1"/px).

    Parameters
    ----------
    sample : dict from JAISPDatasetV6.__getitem__
    crop_size_rubin : int, crop side in Rubin pixels (e.g. 256)
    rng : random state for reproducible crop positions
    """
    rubin = sample.get('rubin', {})
    euclid = sample.get('euclid', {})

    if not rubin:
        return sample

    # Get Rubin spatial size from any band.
    ref = next(iter(rubin.values()))
    _, H_r, W_r = ref['image'].shape
    cs = min(crop_size_rubin, H_r, W_r)

    # Random crop origin in Rubin pixels.
    y0_r = int(rng.randint(0, max(1, H_r - cs)))
    x0_r = int(rng.randint(0, max(1, W_r - cs)))

    # Crop Rubin bands.
    cropped_rubin = {}
    for band, data in rubin.items():
        cropped_rubin[band] = {
            'image': data['image'][:, y0_r:y0_r + cs, x0_r:x0_r + cs].contiguous(),
            'rms':   data['rms'][:, y0_r:y0_r + cs, x0_r:x0_r + cs].contiguous(),
        }

    # Crop Euclid bands at the corresponding angular region.
    # Euclid pixel scale is 0.1"/px, Rubin is 0.2"/px → ratio = 2.
    scale_ratio = 2  # Euclid pixels per Rubin pixel
    cs_e = cs * scale_ratio
    x0_e = x0_r * scale_ratio
    y0_e = y0_r * scale_ratio

    cropped_euclid = {}
    for band, data in euclid.items():
        _, H_e, W_e = data['image'].shape
        # Clamp to Euclid image bounds.
        x0_ec = min(x0_e, max(0, W_e - cs_e))
        y0_ec = min(y0_e, max(0, H_e - cs_e))
        cs_ex = min(cs_e, W_e - x0_ec)
        cs_ey = min(cs_e, H_e - y0_ec)
        cropped_euclid[band] = {
            'image': data['image'][:, y0_ec:y0_ec + cs_ey, x0_ec:x0_ec + cs_ex].contiguous(),
            'rms':   data['rms'][:, y0_ec:y0_ec + cs_ey, x0_ec:x0_ec + cs_ex].contiguous(),
        }

    return {
        'tile_id': sample['tile_id'],
        'rubin': cropped_rubin,
        'euclid': cropped_euclid,
        'has_euclid': bool(cropped_euclid),
        'aug_params': sample.get('aug_params', (0, False, False)),
        'crop_origin_rubin': (x0_r, y0_r),
    }


def sample_context_target_v8(
    sample: Dict,
    rng: np.random.RandomState,
    crop_size_rubin: Optional[int] = None,
    n_targets: int = 1,
) -> Optional[Dict]:
    """Sample context/target split with optional random crop.

    If ``crop_size_rubin`` is set, applies a random crop before splitting.
    Then delegates to the v7 Phase B mixed-resolution splitter.
    """
    if crop_size_rubin is not None and crop_size_rubin > 0:
        sample = random_crop_sample(sample, crop_size_rubin, rng)

    return sample_context_target_phaseB_mixed(sample, rng, n_targets=n_targets)


def make_loader_v8(
    rubin_dir: str,
    euclid_dir: str,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    load_euclid: bool = True,
    **kwargs,
) -> Tuple[JAISPDatasetV6, DataLoader]:
    """Create a dataloader for v8.

    Same as v7 but the training loop should call ``sample_context_target_v8``
    with a ``crop_size_rubin`` parameter to apply random cropping.
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


__all__ = [
    "ALL_BANDS",
    "EUCLID_BANDS",
    "JAISPDatasetV6",
    "RUBIN_BANDS",
    "band_group",
    "make_loader_v8",
    "random_crop_sample",
    "sample_context_target_v8",
]
