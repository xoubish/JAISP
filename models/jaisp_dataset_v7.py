"""Dataset helpers for JAISP Foundation v7 mixed-resolution training.

v7 reuses the v6 tile loader but changes the Phase B split:
Euclid bands stay at native resolution instead of being downsampled to the
Rubin 512x512 grid before entering the context/target pool.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
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
    sample_context_target,
)


def band_group(band_name: str) -> str:
    """Map a band name to its native-resolution stream."""
    if band_name in RUBIN_BANDS:
        return "rubin"
    if band_name == "euclid_VIS":
        return "vis"
    if band_name in ("euclid_Y", "euclid_J", "euclid_H"):
        return "nisp"
    raise KeyError(f"Unknown band: {band_name}")


def sample_context_target_phaseB_mixed(
    sample: Dict,
    rng: np.random.RandomState,
    n_targets: int = 1,
) -> Optional[Dict]:
    """Sample a mixed-resolution Phase B context/target split.

    Rubin bands remain at 512x512. All Euclid bands (VIS and NISP Y/J/H)
    are ~1084x1084 at 0.1"/px from MER mosaics. The model is responsible
    for resolution-aware fusion downstream.
    """
    if not sample.get("has_euclid"):
        return None

    rubin = sample["rubin"]
    euclid = sample.get("euclid", {})
    if not rubin or not euclid:
        return None

    pool = {**rubin, **euclid}
    avail = list(pool.keys())
    if len(avail) < n_targets + 1:
        return None

    order = rng.permutation(len(avail)).tolist()
    target_bands = [avail[order[i]] for i in range(n_targets)]
    context_bands = [avail[order[i]] for i in range(n_targets, len(avail))]

    return {
        "context_images": {b: pool[b]["image"] for b in context_bands},
        "context_rms": {b: pool[b]["rms"] for b in context_bands},
        "targets": [
            {"band": b, "image": pool[b]["image"], "rms": pool[b]["rms"]}
            for b in target_bands
        ],
    }


def make_loader_v7(
    rubin_dir: str,
    euclid_dir: str,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    load_euclid: bool = True,
    **kwargs,
) -> Tuple[JAISPDatasetV6, DataLoader]:
    """Create a dataloader for v7.

    Mixed native resolutions require `batch_size=1`, so we keep the same
    list-based collate function as v6.
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
    "make_loader_v7",
    "sample_context_target",
    "sample_context_target_phaseB_mixed",
]
