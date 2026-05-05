"""Dataset helpers for JAISP Foundation v9 — adversarial cross-instrument masking.

v9 reuses the v8 random-crop tile loader and adds **adversarial masking**:
with probability ``p_adversarial`` the loss masks the target band PLUS one or
two wavelength-adjacent within-instrument neighbours. This forces the encoder
to draw on cross-instrument and far-wavelength inputs at training time, rather
than collapsing to the immediate within-instrument shortcut diagnosed in
notebook 13 (where v8's encoder routed ~100% of attribution through Euclid VIS
as a universal spatial scaffold).

The standard one-band masking from v7/v8 is preserved for ``1 - p_adversarial``
of training steps so the model still sees its inference-time distribution.

Wavelength-adjacent neighbours (used to pick which extra bands to mask):

    Rubin (ordered by wavelength): u, g, r, i, z, y
    Euclid (ordered by wavelength): VIS, Y, J, H

For target ``rubin_g`` an adversarial step might additionally mask ``rubin_u``
and ``rubin_r`` (its two wavelength-nearest neighbours), forcing the encoder
to recover g from rubin_i/z/y or from any Euclid band.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
)
from jaisp_dataset_v7 import band_group, sample_context_target_phaseB_mixed
from jaisp_dataset_v8 import random_crop_sample


# Wavelength-ordered band lists used to pick adversarial neighbours.
# Order matches RUBIN_BANDS / EUCLID_BANDS in jaisp_foundation_v6.py.
RUBIN_BY_WAVELENGTH = list(RUBIN_BANDS)   # u, g, r, i, z, y
EUCLID_BY_WAVELENGTH = list(EUCLID_BANDS) # VIS, Y, J, H


def _wavelength_neighbours(band: str, n: int = 2) -> List[str]:
    """Return up to ``n`` wavelength-nearest within-instrument neighbours.

    Picks the closest neighbours in wavelength order. For end-of-spectrum
    targets (e.g. rubin_u) this returns only the inner-side neighbours.
    """
    if band in RUBIN_BY_WAVELENGTH:
        ordered = RUBIN_BY_WAVELENGTH
    elif band in EUCLID_BY_WAVELENGTH:
        ordered = EUCLID_BY_WAVELENGTH
    else:
        return []
    idx = ordered.index(band)
    candidates: List[Tuple[int, str]] = []
    for offset in (1, -1, 2, -2):
        j = idx + offset
        if 0 <= j < len(ordered) and ordered[j] != band:
            candidates.append((abs(offset), ordered[j]))
    candidates.sort()
    return [b for _, b in candidates[:n]]


def adversarial_drop(
    split: Dict,
    rng: np.random.RandomState,
    p_adversarial: float = 0.25,
    n_extra_max: int = 2,
) -> Dict:
    """Optionally drop wavelength-adjacent context bands alongside the target.

    Operates on the dict returned by ``sample_context_target_phaseB_mixed``.
    With probability ``p_adversarial``, removes 1..n_extra_max bands from
    ``context_images``/``context_rms`` that are wavelength-adjacent to one
    of the target bands. Targets are unchanged.

    Caveats:
    - We never drop a band that would leave the encoder with fewer than two
      available context bands total (the encoder needs at least one usable
      stream to produce a bottleneck).
    - We never drop a band of the *opposite* instrument from the target — the
      adversarial pressure is meant to force *cross-instrument* coupling, so
      we hide same-instrument neighbours and keep the cross-instrument bands
      visible.
    """
    if rng.random() >= p_adversarial:
        return split
    if not split or not split.get("targets"):
        return split

    target_bands = {t["band"] for t in split["targets"]}
    candidates: List[str] = []
    for tb in target_bands:
        for nb in _wavelength_neighbours(tb, n=n_extra_max):
            if nb in split["context_images"]:
                candidates.append(nb)
    candidates = list(dict.fromkeys(candidates))   # de-dup, preserve order
    if not candidates:
        return split

    n_drop = int(rng.randint(1, min(n_extra_max, len(candidates)) + 1))
    drop = list(rng.choice(candidates, size=n_drop, replace=False))

    # Make sure we don't strand the encoder with too few context bands.
    remaining = len(split["context_images"]) - len(drop)
    if remaining < 2:
        return split

    new_split = {
        "context_images": {b: v for b, v in split["context_images"].items() if b not in drop},
        "context_rms":    {b: v for b, v in split["context_rms"].items()    if b not in drop},
        "targets": split["targets"],
        # Keep diagnostic info for downstream telemetry / debugging.
        "adversarial_dropped": list(drop),
    }
    return new_split


def sample_context_target_v9(
    sample: Dict,
    rng: np.random.RandomState,
    n_targets: int = 1,
    crop_size_rubin: Optional[int] = None,
    p_adversarial: float = 0.25,
    n_extra_max: int = 2,
) -> Optional[Dict]:
    """v9 sampling: optional random crop, Phase-B split, then adversarial drop."""
    if crop_size_rubin is not None and crop_size_rubin > 0:
        sample = random_crop_sample(sample, crop_size_rubin, rng)
    split = sample_context_target_phaseB_mixed(sample, rng, n_targets=n_targets)
    if split is None:
        return None
    return adversarial_drop(split, rng, p_adversarial=p_adversarial, n_extra_max=n_extra_max)


def make_loader_v9(
    rubin_dir: str,
    euclid_dir: str,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    load_euclid: bool = True,
    **kwargs,
) -> Tuple[JAISPDatasetV6, DataLoader]:
    """Same DataLoader as v8; the v9 trainer applies the adversarial drop in
    its ``_prepare_batch`` override."""
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
    "adversarial_drop",
    "band_group",
    "make_loader_v9",
    "random_crop_sample",
    "sample_context_target_v9",
    "_wavelength_neighbours",
]
