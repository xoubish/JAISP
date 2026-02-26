"""
Dataset for astrometry concordance training.

Always pairs a Rubin band with Euclid VIS (the reference frame).
Applies synthetic astrometric offsets to the Rubin image for self-supervised training.
"""

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from jaisp_dataset_v4 import (
    RUBIN_BANDS, RUBIN_BAND_ORDER,
    EUCLID_BANDS, EUCLID_BAND_KEYS,
    _to_float32, _safe_sqrt_var,
)

from offsets import generate_offset_field, sample_offset_mode


class AstrometryDataset(Dataset):
    """
    Yields (Rubin_band, Euclid_VIS) pairs with synthetic offsets applied to Rubin.

    Each sample provides:
      - shifted_rubin: Rubin image warped by a synthetic offset
      - rubin_rms: noise map (unshifted — conservative choice)
      - vis_image, vis_rms: original VIS data (reference, unshifted)
      - gt_dra, gt_ddec: ground truth offset field in arcseconds [H_rubin, W_rubin]
      - offset_mode: which generator was used
      - rubin_band: which Rubin band was selected
    """

    def __init__(
        self,
        rubin_dir: str,
        euclid_dir: str,
        max_offset_arcsec: float = 0.5,
        curriculum_epochs: int = 10,
        seed: int = 42,
        augment: bool = True,
        mmap: bool = True,
        finite_frac_thresh: float = 0.30,
    ):
        self.rubin_dir = Path(rubin_dir)
        self.euclid_dir = Path(euclid_dir)
        self.max_offset = float(max_offset_arcsec)
        self.curriculum_epochs = int(curriculum_epochs)
        self.augment = augment
        self.mmap = mmap
        self.rng = np.random.RandomState(seed)
        self.finite_frac_thresh = float(finite_frac_thresh)
        self.epoch = 1  # updated externally by training loop

        # Find tiles that have BOTH Rubin and Euclid.
        rubin_files = sorted(glob.glob(os.path.join(rubin_dir, "tile_x*_y*.npz")))
        self.pairs = []
        for rp in rubin_files:
            tid = os.path.splitext(os.path.basename(rp))[0]
            ep = os.path.join(euclid_dir, f"{tid}_euclid.npz")
            if os.path.exists(ep):
                self.pairs.append({"tile_id": tid, "rubin_path": rp, "euclid_path": ep})

        if not self.pairs:
            raise FileNotFoundError(f"No tiles with both Rubin and Euclid in {rubin_dir} / {euclid_dir}")

        # Precompute which Rubin bands are available per tile.
        self._rubin_avail: List[List[str]] = []
        for pair in self.pairs:
            avail = []
            try:
                data = np.load(pair["rubin_path"], mmap_mode="r" if mmap else None, allow_pickle=True)
                imgs = data["img"]
                for i, band in enumerate(RUBIN_BANDS):
                    if i < imgs.shape[0]:
                        frac = np.isfinite(imgs[i]).sum() / float(imgs[i].size)
                        if frac > self.finite_frac_thresh:
                            avail.append(band)
            except Exception:
                pass
            self._rubin_avail.append(avail if avail else RUBIN_BANDS[:1])

        # Check VIS availability.
        self._has_vis: List[bool] = []
        for pair in self.pairs:
            try:
                data = np.load(pair["euclid_path"], mmap_mode="r" if mmap else None, allow_pickle=True)
                has = "img_VIS" in data
            except Exception:
                has = False
            self._has_vis.append(has)

        n_with_vis = sum(self._has_vis)
        print(f"AstrometryDataset: {len(self.pairs)} tiles, {n_with_vis} with VIS")
        print(f"  max_offset={self.max_offset}\" curriculum={self.curriculum_epochs} epochs")

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_rubin(self, path: str, band: str) -> Tuple[np.ndarray, np.ndarray]:
        idx = RUBIN_BAND_ORDER.index(band.split("_")[1])
        data = np.load(path, mmap_mode="r" if self.mmap else None, allow_pickle=True)
        img = _to_float32(data["img"][idx])
        rms = _safe_sqrt_var(data["var"][idx]) if "var" in data else np.ones_like(img) * np.nanstd(img)
        return img, rms

    def _load_vis(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(path, mmap_mode="r" if self.mmap else None, allow_pickle=True)
        img = _to_float32(data["img_VIS"])
        rms = _safe_sqrt_var(data["var_VIS"]) if "var_VIS" in data else np.ones_like(img) * np.nanstd(img)
        return img, rms

    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]

        # Pick a random Rubin band.
        rubin_band = self.rng.choice(self._rubin_avail[idx])
        rubin_img, rubin_rms = self._load_rubin(pair["rubin_path"], rubin_band)

        # Load VIS.
        if self._has_vis[idx]:
            vis_img, vis_rms = self._load_vis(pair["euclid_path"])
        else:
            # Fallback: skip (should be filtered in practice).
            vis_img = np.zeros((64, 64), dtype=np.float32)
            vis_rms = np.ones((64, 64), dtype=np.float32)

        # Clean NaNs.
        rubin_img = np.nan_to_num(rubin_img, nan=0.0)
        rubin_rms = np.maximum(np.nan_to_num(rubin_rms, nan=1.0), 1e-10)
        vis_img = np.nan_to_num(vis_img, nan=0.0)
        vis_rms = np.maximum(np.nan_to_num(vis_rms, nan=1.0), 1e-10)

        # Synchronized augmentation.
        if self.augment:
            k = self.rng.randint(4)
            flip_ud = self.rng.rand() > 0.5
            flip_lr = self.rng.rand() > 0.5
            for arr_list in [[rubin_img, rubin_rms], [vis_img, vis_rms]]:
                for j in range(len(arr_list)):
                    arr_list[j] = np.rot90(arr_list[j], k).copy()
                    if flip_ud:
                        arr_list[j] = np.flip(arr_list[j], 0).copy()
                    if flip_lr:
                        arr_list[j] = np.flip(arr_list[j], 1).copy()

        # Generate synthetic offset field on Rubin pixel grid.
        H_r, W_r = rubin_img.shape
        mode = sample_offset_mode(self.epoch, self.curriculum_epochs, self.rng)
        dra, ddec = generate_offset_field(H_r, W_r, self.max_offset, mode=mode, rng=self.rng)

        return {
            "tile_id": pair["tile_id"],
            "rubin_band": rubin_band,
            "rubin_image": torch.from_numpy(rubin_img[None].copy()),    # [1, H, W]
            "rubin_rms": torch.from_numpy(rubin_rms[None].copy()),
            "vis_image": torch.from_numpy(vis_img[None].copy()),
            "vis_rms": torch.from_numpy(vis_rms[None].copy()),
            "gt_dra": torch.from_numpy(dra.copy()),                     # [H_r, W_r]
            "gt_ddec": torch.from_numpy(ddec.copy()),
            "offset_mode": mode,
            "has_vis": self._has_vis[idx],
        }


def collate_astrometry(batch: List[Dict]) -> Dict:
    """List-based collation for variable-sized images."""
    return {
        "tile_id": [b["tile_id"] for b in batch],
        "rubin_band": [b["rubin_band"] for b in batch],
        "rubin_image": [b["rubin_image"] for b in batch],
        "rubin_rms": [b["rubin_rms"] for b in batch],
        "vis_image": [b["vis_image"] for b in batch],
        "vis_rms": [b["vis_rms"] for b in batch],
        "gt_dra": [b["gt_dra"] for b in batch],
        "gt_ddec": [b["gt_ddec"] for b in batch],
        "offset_mode": [b["offset_mode"] for b in batch],
        "has_vis": [b["has_vis"] for b in batch],
    }


def make_astrometry_loader(
    rubin_dir: str,
    euclid_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs,
):
    dataset = AstrometryDataset(rubin_dir=rubin_dir, euclid_dir=euclid_dir, **kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
        collate_fn=collate_astrometry,
    )
    return dataset, loader
