"""JAISP Dataset v10 — standalone tile dataset for fine-scale foundation training.

This file is a self-contained consolidation of v6's tile loader, v7's Phase B
mixed-resolution split, v8's random crop, and v9's adversarial cross-instrument
masking. It does not import from any older v6/v7/v8/v9 dataset module.

Per __getitem__ the dataset returns a dict with all available bands for one
Rubin+Euclid tile pair (no batching across tiles — Rubin and Euclid have
different native shapes, so the collate function leaves the list intact).
The training loop then chooses context vs target bands per step.

Sampling pipeline (call ``sample_context_target_v10`` from the trainer):
  1. Optional spatially-consistent random crop across all bands
     (Rubin pixel-coords; Euclid cropped at the matching angular region).
  2. Phase B mixed-resolution split: pool Rubin (0.2"/px) + Euclid (0.1"/px),
     hold out ``n_targets`` bands as prediction targets.
  3. With probability ``p_adversarial``, drop 1-2 wavelength-adjacent
     within-instrument context neighbours of the target — forces the encoder
     to use cross-instrument and far-wavelength inputs instead of collapsing
     to the immediate within-instrument shortcut.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Band metadata
# ============================================================

RUBIN_BANDS = ["rubin_u", "rubin_g", "rubin_r", "rubin_i", "rubin_z", "rubin_y"]
RUBIN_BAND_ORDER = ["u", "g", "r", "i", "z", "y"]
EUCLID_BANDS = ["euclid_VIS", "euclid_Y", "euclid_J", "euclid_H"]
EUCLID_BAND_KEYS = ["VIS", "Y", "J", "H"]
ALL_BANDS = RUBIN_BANDS + EUCLID_BANDS

# Wavelength-ordered band lists used to pick adversarial neighbours.
RUBIN_BY_WAVELENGTH = list(RUBIN_BANDS)    # u, g, r, i, z, y
EUCLID_BY_WAVELENGTH = list(EUCLID_BANDS)  # VIS, Y, J, H


def band_group(band_name: str) -> str:
    if band_name in RUBIN_BANDS:
        return "rubin"
    if band_name in EUCLID_BANDS:
        return "euclid"
    raise KeyError(f"Unknown band: {band_name}")


# ============================================================
# Pixel utilities
# ============================================================

MAX_REASONABLE_RMS = np.float32(1.0e10)
MAX_REASONABLE_VAR = np.float32(MAX_REASONABLE_RMS * MAX_REASONABLE_RMS)


def _to_float32(x) -> np.ndarray:
    return np.asarray(x).astype(np.float32, copy=False)


def _safe_rms(var: np.ndarray, fallback_image: np.ndarray) -> np.ndarray:
    """sqrt(variance), replacing bad or sentinel values with MAD fallback."""
    var = _to_float32(var)
    rms = np.zeros_like(var)
    # Some Euclid products carry huge finite sentinel variance values (~1e32);
    # treat them as invalid so they don't collapse normalised inputs to zero.
    good = np.isfinite(var) & (var > 0) & (var <= MAX_REASONABLE_VAR)
    rms[good] = np.sqrt(var[good])
    if not good.all():
        finite = fallback_image[np.isfinite(fallback_image)]
        if len(finite) > 10:
            mad = np.median(np.abs(finite - np.median(finite)))
            rms[~good] = max(1.4826 * mad, 1e-10)
        else:
            rms[~good] = 1.0
    return rms


# ============================================================
# Dataset
# ============================================================

class JAISPDatasetV10(Dataset):
    """Per-tile dataset returning all available Rubin+Euclid bands.

    Each ``__getitem__`` returns a dict with::

        rubin:   {band: {'image': [1, H, W], 'rms': [1, H, W]}}  Rubin 512×512
        euclid:  {band: {'image': [1, H, W], 'rms': [1, H, W]}}  Euclid ~1084×1084
        tile_id: str
        has_euclid: bool
        aug_params: (n_rot90, flip_ud, flip_lr)

    Variable native resolutions across instruments mean batches must stay as
    Python lists (see :func:`collate_v10`). The training loop handles
    context/target masking and (optionally) random cropping per step.
    """

    def __init__(
        self,
        rubin_dir: str,
        euclid_dir: str,
        load_euclid: bool = True,
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
        self.min_rubin_bands = int(min_rubin_bands)
        self.finite_frac_thresh = float(finite_frac_thresh)
        self.rng = np.random.RandomState(seed)

        rubin_files = sorted(glob.glob(os.path.join(rubin_dir, "tile_x*_y*.npz")))
        tile_ids = [os.path.splitext(os.path.basename(p))[0] for p in rubin_files]

        print(f"JAISPDatasetV10: scanning {len(tile_ids)} tiles...")
        self.tiles: List[Dict] = []
        for tid in tile_ids:
            rp = os.path.join(rubin_dir, f"{tid}.npz")
            ep = os.path.join(euclid_dir, f"{tid}_euclid.npz")
            avail_r = self._scan_rubin(rp)
            if len(avail_r) < self.min_rubin_bands:
                continue
            avail_e = self._scan_euclid(ep) if (load_euclid and os.path.exists(ep)) else []
            self.tiles.append({
                "tile_id": tid,
                "rubin_path": rp,
                "euclid_path": ep if os.path.exists(ep) else None,
                "avail_rubin": avail_r,
                "avail_euclid": avail_e,
            })

        if not self.tiles:
            raise FileNotFoundError(
                f"No usable tiles found in {rubin_dir} (min_rubin_bands={self.min_rubin_bands})"
            )
        print(f"  {len(self.tiles)} tiles passed quality cuts")
        n_with_euclid = sum(1 for t in self.tiles if t["euclid_path"] and t["avail_euclid"])
        print(f"  {n_with_euclid} tiles have Euclid coverage")

    # --- band availability scanning ---------------------------------------

    def _finite_ok(self, arr: np.ndarray) -> bool:
        return np.isfinite(arr).mean() > self.finite_frac_thresh

    def _scan_rubin(self, path: str) -> List[str]:
        try:
            data = np.load(path, mmap_mode="r" if self.mmap else None, allow_pickle=True)
            imgs = data["img"]
            return [
                RUBIN_BANDS[i]
                for i in range(min(len(RUBIN_BAND_ORDER), imgs.shape[0]))
                if self._finite_ok(imgs[i])
            ]
        except Exception:
            return []

    def _scan_euclid(self, path: str) -> List[str]:
        try:
            data = np.load(path, mmap_mode="r" if self.mmap else None, allow_pickle=True)
            return [
                b for b, key in zip(EUCLID_BANDS, EUCLID_BAND_KEYS)
                if f"img_{key}" in data and self._finite_ok(data[f"img_{key}"])
            ]
        except Exception:
            return []

    # --- band loading -----------------------------------------------------

    def _load_rubin_band(self, path: str, band: str) -> Tuple[np.ndarray, np.ndarray]:
        band_key = band.split("_")[1]
        idx = RUBIN_BAND_ORDER.index(band_key)
        data = np.load(path, mmap_mode="r" if self.mmap else None, allow_pickle=True)
        img = _to_float32(data["img"][idx])
        rms = _safe_rms(data["var"][idx], img) if "var" in data else np.ones_like(img)
        return img, rms

    def _load_euclid_band(self, path: str, band: str) -> Tuple[np.ndarray, np.ndarray]:
        band_key = band.split("_")[1]
        data = np.load(path, mmap_mode="r" if self.mmap else None, allow_pickle=True)
        img = _to_float32(data[f"img_{band_key}"])
        var_key = f"var_{band_key}"
        rms = _safe_rms(data[var_key], img) if var_key in data else np.ones_like(img)
        return img, rms

    # --- augmentation -----------------------------------------------------

    def _get_aug_state(self) -> Tuple[int, bool, bool]:
        return (
            int(self.rng.randint(4)),
            bool(self.rng.rand() > 0.5),
            bool(self.rng.rand() > 0.5),
        )

    @staticmethod
    def _apply_aug(
        img: np.ndarray,
        rms: np.ndarray,
        n_rot: int,
        flip_ud: bool,
        flip_lr: bool,
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

    # --- Dataset interface ------------------------------------------------

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Dict:
        tile = self.tiles[idx]
        aug_params = self._get_aug_state() if self.augment else (0, False, False)

        rubin_data = {}
        for band in tile["avail_rubin"]:
            try:
                img, rms = self._load_rubin_band(tile["rubin_path"], band)
            except Exception:
                continue
            img = np.nan_to_num(img, nan=0.0)
            rms = np.maximum(np.nan_to_num(rms, nan=1.0), 1e-10)
            if self.augment:
                img, rms = self._apply_aug(img, rms, *aug_params)
            rubin_data[band] = {
                "image": torch.from_numpy(img[None].copy()),
                "rms":   torch.from_numpy(rms[None].copy()),
            }

        euclid_data = {}
        if self.load_euclid and tile["euclid_path"] and tile["avail_euclid"]:
            for band in tile["avail_euclid"]:
                try:
                    img, rms = self._load_euclid_band(tile["euclid_path"], band)
                except Exception:
                    continue
                img = np.nan_to_num(img, nan=0.0)
                rms = np.maximum(np.nan_to_num(rms, nan=1.0), 1e-10)
                if self.augment:
                    img, rms = self._apply_aug(img, rms, *aug_params)
                euclid_data[band] = {
                    "image": torch.from_numpy(img[None].copy()),
                    "rms":   torch.from_numpy(rms[None].copy()),
                }

        return {
            "tile_id":    tile["tile_id"],
            "rubin":      rubin_data,
            "euclid":     euclid_data,
            "has_euclid": bool(euclid_data),
            "aug_params": aug_params,
        }

    def get_band_names(self) -> List[str]:
        return ALL_BANDS


# ============================================================
# Collate
# ============================================================

def collate_v10(batch: List[Dict]) -> List[Dict]:
    """Identity collate: Rubin (512) and Euclid (~1084) have different native
    shapes, so we leave each tile as a separate dict and let the trainer iterate."""
    return batch


# ============================================================
# Sampling pipeline
# ============================================================

def random_crop_sample(
    sample: Dict,
    crop_size_rubin: int,
    rng: np.random.RandomState,
) -> Dict:
    """Spatially-consistent random crop across Rubin + Euclid bands.

    Crop origin is chosen in Rubin pixel coordinates; Euclid bands are cropped
    at the corresponding angular region using the 2:1 scale ratio
    (Rubin 0.2"/px → Euclid 0.1"/px).
    """
    rubin = sample.get("rubin", {})
    euclid = sample.get("euclid", {})
    if not rubin:
        return sample

    ref = next(iter(rubin.values()))
    _, H_r, W_r = ref["image"].shape
    cs = min(int(crop_size_rubin), H_r, W_r)

    y0_r = int(rng.randint(0, max(1, H_r - cs)))
    x0_r = int(rng.randint(0, max(1, W_r - cs)))

    cropped_rubin = {}
    for band, data in rubin.items():
        cropped_rubin[band] = {
            "image": data["image"][:, y0_r:y0_r + cs, x0_r:x0_r + cs].contiguous(),
            "rms":   data["rms"][:, y0_r:y0_r + cs, x0_r:x0_r + cs].contiguous(),
        }

    scale_ratio = 2  # Euclid pixels per Rubin pixel
    cs_e = cs * scale_ratio
    x0_e = x0_r * scale_ratio
    y0_e = y0_r * scale_ratio

    cropped_euclid = {}
    for band, data in euclid.items():
        _, H_e, W_e = data["image"].shape
        x0_ec = min(x0_e, max(0, W_e - cs_e))
        y0_ec = min(y0_e, max(0, H_e - cs_e))
        cs_ex = min(cs_e, W_e - x0_ec)
        cs_ey = min(cs_e, H_e - y0_ec)
        cropped_euclid[band] = {
            "image": data["image"][:, y0_ec:y0_ec + cs_ey, x0_ec:x0_ec + cs_ex].contiguous(),
            "rms":   data["rms"][:, y0_ec:y0_ec + cs_ey, x0_ec:x0_ec + cs_ex].contiguous(),
        }

    return {
        "tile_id": sample["tile_id"],
        "rubin": cropped_rubin,
        "euclid": cropped_euclid,
        "has_euclid": bool(cropped_euclid),
        "aug_params": sample.get("aug_params", (0, False, False)),
        "crop_origin_rubin": (x0_r, y0_r),
    }


def _phase_b_split(
    sample: Dict,
    rng: np.random.RandomState,
    n_targets: int = 1,
) -> Optional[Dict]:
    """Pool Rubin + Euclid (each at native resolution) and randomly hold out
    ``n_targets`` bands as prediction targets."""
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
        "context_rms":    {b: pool[b]["rms"] for b in context_bands},
        "targets": [
            {"band": b, "image": pool[b]["image"], "rms": pool[b]["rms"]}
            for b in target_bands
        ],
    }


def _wavelength_neighbours(band: str, n: int = 2) -> List[str]:
    """Up to ``n`` wavelength-nearest within-instrument neighbours of ``band``."""
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

    With probability ``p_adversarial`` removes 1..n_extra_max bands from
    ``context_images``/``context_rms`` that are wavelength-adjacent to one of
    the target bands. Targets are unchanged. Never drops a band that would
    leave fewer than two context bands total.
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
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return split

    n_drop = int(rng.randint(1, min(n_extra_max, len(candidates)) + 1))
    drop = list(rng.choice(candidates, size=n_drop, replace=False))

    if len(split["context_images"]) - len(drop) < 2:
        return split

    return {
        "context_images": {b: v for b, v in split["context_images"].items() if b not in drop},
        "context_rms":    {b: v for b, v in split["context_rms"].items()    if b not in drop},
        "targets": split["targets"],
        "adversarial_dropped": list(drop),
    }


def sample_context_target_v10(
    sample: Dict,
    rng: np.random.RandomState,
    n_targets: int = 1,
    crop_size_rubin: Optional[int] = None,
    p_adversarial: float = 0.25,
    n_extra_max: int = 2,
) -> Optional[Dict]:
    """Full v10 sampling: random crop → Phase-B mixed-resolution split → adversarial drop."""
    if crop_size_rubin is not None and crop_size_rubin > 0:
        sample = random_crop_sample(sample, int(crop_size_rubin), rng)
    split = _phase_b_split(sample, rng, n_targets=n_targets)
    if split is None:
        return None
    return adversarial_drop(split, rng, p_adversarial=p_adversarial, n_extra_max=n_extra_max)


# ============================================================
# DataLoader factory
# ============================================================

def make_loader_v10(
    rubin_dir: str,
    euclid_dir: str,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    load_euclid: bool = True,
    **kwargs,
) -> Tuple[JAISPDatasetV10, DataLoader]:
    """Mixed native resolutions force ``batch_size=1``; we use a list-based collate."""
    dataset = JAISPDatasetV10(
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
        collate_fn=collate_v10,
    )
    return dataset, loader


__all__ = [
    "ALL_BANDS",
    "RUBIN_BANDS",
    "EUCLID_BANDS",
    "JAISPDatasetV10",
    "adversarial_drop",
    "band_group",
    "collate_v10",
    "make_loader_v10",
    "random_crop_sample",
    "sample_context_target_v10",
]
