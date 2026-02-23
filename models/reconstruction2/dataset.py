import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Keep one source of truth for band names.
from jaisp_dataset_v4 import ALL_BANDS, EUCLID_BANDS, EUCLID_BAND_KEYS, RUBIN_BANDS, RUBIN_BAND_ORDER


def _to_float32(x):
    return np.asarray(x).astype(np.float32, copy=False)


def _safe_sqrt_var(var):
    var = _to_float32(var)
    out = var.copy()
    m = np.isfinite(out)
    out[m] = np.maximum(out[m], 0.0)
    return np.sqrt(out, dtype=np.float32)


class JAISPMultiBandReconstructionDataset(Dataset):
    """
    Samples one target band plus K context bands from the same tile.

    This supports 9->1 (and any k->1) reconstruction training by sampling
    context subset sizes at runtime.
    """

    def __init__(
        self,
        rubin_dir: str,
        euclid_dir: str,
        min_context_bands: int = 1,
        max_context_bands: int = 9,
        augment: bool = True,
        mmap: bool = True,
        seed: int = 42,
        finite_frac_thresh: float = 0.30,
        precompute_available: bool = True,
    ):
        self.rubin_dir = Path(rubin_dir)
        self.euclid_dir = Path(euclid_dir)
        self.min_context_bands = int(min_context_bands)
        self.max_context_bands = int(max_context_bands)
        self.augment = bool(augment)
        self.mmap = bool(mmap)
        self.rng = np.random.RandomState(seed)
        self.finite_frac_thresh = float(finite_frac_thresh)

        rubin_files = sorted(glob.glob(os.path.join(rubin_dir, "tile_x*_y*.npz")))
        self.tile_ids = [os.path.splitext(os.path.basename(p))[0] for p in rubin_files]

        self.pairs = []
        for tid in self.tile_ids:
            rp = os.path.join(rubin_dir, f"{tid}.npz")
            ep = os.path.join(euclid_dir, f"{tid}_euclid.npz")
            if os.path.exists(rp):
                self.pairs.append(
                    {
                        "tile_id": tid,
                        "rubin_path": rp,
                        "euclid_path": ep,
                        "has_euclid": os.path.exists(ep),
                    }
                )

        if not self.pairs:
            raise FileNotFoundError(f"No tiles found in {rubin_dir}")

        self._avail_cache: Optional[List[List[str]]] = None
        if precompute_available:
            self._avail_cache = [self._get_available_bands(pair) for pair in self.pairs]

        print(f"JAISPMultiBandReconstructionDataset: {len(self.pairs)} tiles")
        print(
            "  context range: "
            f"[{self.min_context_bands}, {self.max_context_bands}], "
            f"augment={self.augment}, precompute_available={precompute_available}"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def _finite_frac_ok(self, arr: np.ndarray) -> bool:
        frac = np.isfinite(arr).sum() / float(arr.size)
        return frac > self.finite_frac_thresh

    def _load_rubin_band(self, path: str, band: str) -> Tuple[np.ndarray, np.ndarray]:
        band_idx = RUBIN_BAND_ORDER.index(band.split("_")[1])
        data = np.load(path, mmap_mode="r" if self.mmap else None, allow_pickle=True)
        img = _to_float32(data["img"][band_idx])
        rms = _safe_sqrt_var(data["var"][band_idx]) if "var" in data else (np.ones_like(img) * np.nanstd(img))
        return img, rms

    def _load_euclid_band(self, path: str, band: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not os.path.exists(path):
            return None, None

        band_key = band.split("_")[1]
        data = np.load(path, mmap_mode="r" if self.mmap else None, allow_pickle=True)

        img_key = f"img_{band_key}"
        if img_key not in data:
            return None, None

        img = _to_float32(data[img_key])
        var_key = f"var_{band_key}"
        rms = _safe_sqrt_var(data[var_key]) if var_key in data else (np.ones_like(img) * np.nanstd(img))
        return img, rms

    def _load_band(self, pair: Dict, band: str) -> Tuple[np.ndarray, np.ndarray]:
        if band.startswith("rubin_"):
            return self._load_rubin_band(pair["rubin_path"], band)
        return self._load_euclid_band(pair["euclid_path"], band)

    def _get_available_bands(self, pair: Dict) -> List[str]:
        available: List[str] = []

        try:
            data = np.load(pair["rubin_path"], mmap_mode="r" if self.mmap else None, allow_pickle=True)
            imgs = data["img"]
            for i, band in enumerate(RUBIN_BANDS):
                if i < imgs.shape[0] and self._finite_frac_ok(imgs[i]):
                    available.append(band)
        except Exception:
            pass

        if pair["has_euclid"]:
            try:
                data = np.load(pair["euclid_path"], mmap_mode="r" if self.mmap else None, allow_pickle=True)
                for band, key in zip(EUCLID_BANDS, EUCLID_BAND_KEYS):
                    img_key = f"img_{key}"
                    if img_key in data and self._finite_frac_ok(data[img_key]):
                        available.append(band)
            except Exception:
                pass

        return [b for b in available if b in ALL_BANDS]

    def _augment_with_params(
        self,
        img: np.ndarray,
        rms: np.ndarray,
        k_rot: int,
        flip_ud: bool,
        flip_lr: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        img = np.rot90(img, k_rot).copy()
        rms = np.rot90(rms, k_rot).copy()

        if flip_ud:
            img = np.flip(img, 0).copy()
            rms = np.flip(rms, 0).copy()
        if flip_lr:
            img = np.flip(img, 1).copy()
            rms = np.flip(rms, 1).copy()

        return img, rms

    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]
        available = self._avail_cache[idx] if self._avail_cache is not None else self._get_available_bands(pair)

        if len(available) < 2:
            available = RUBIN_BANDS[:2]

        target_band = self.rng.choice(available)
        context_candidates = [b for b in available if b != target_band]
        if not context_candidates:
            context_candidates = [target_band]

        max_k = min(self.max_context_bands, len(context_candidates))
        min_k = min(self.min_context_bands, max_k)
        if max_k <= 0:
            context_bands = [target_band]
        else:
            k = self.rng.randint(min_k, max_k + 1)
            context_bands = list(self.rng.choice(context_candidates, size=k, replace=False))

        # Synchronized augment params for all bands in this sample.
        k_rot = self.rng.randint(4)
        flip_ud = self.rng.rand() > 0.5
        flip_lr = self.rng.rand() > 0.5

        tgt_img, tgt_rms = self._load_band(pair, target_band)
        if tgt_img is None:
            tgt_img, tgt_rms = self._load_rubin_band(pair["rubin_path"], RUBIN_BANDS[0])
            target_band = RUBIN_BANDS[0]

        tgt_img = np.nan_to_num(tgt_img, nan=0.0)
        tgt_rms = np.maximum(np.nan_to_num(tgt_rms, nan=1.0), 1e-10)

        if self.augment:
            tgt_img, tgt_rms = self._augment_with_params(tgt_img, tgt_rms, k_rot, flip_ud, flip_lr)

        context_images: List[torch.Tensor] = []
        context_rms: List[torch.Tensor] = []
        context_names: List[str] = []

        for band in context_bands:
            c_img, c_rms = self._load_band(pair, band)
            if c_img is None:
                continue
            c_img = np.nan_to_num(c_img, nan=0.0)
            c_rms = np.maximum(np.nan_to_num(c_rms, nan=1.0), 1e-10)
            if self.augment:
                c_img, c_rms = self._augment_with_params(c_img, c_rms, k_rot, flip_ud, flip_lr)

            context_images.append(torch.from_numpy(c_img[None].copy()))
            context_rms.append(torch.from_numpy(c_rms[None].copy()))
            context_names.append(band)

        # Keep at least one context view.
        if not context_images:
            fallback_band = target_band
            f_img, f_rms = self._load_band(pair, fallback_band)
            f_img = np.nan_to_num(f_img, nan=0.0)
            f_rms = np.maximum(np.nan_to_num(f_rms, nan=1.0), 1e-10)
            if self.augment:
                f_img, f_rms = self._augment_with_params(f_img, f_rms, k_rot, flip_ud, flip_lr)
            context_images.append(torch.from_numpy(f_img[None].copy()))
            context_rms.append(torch.from_numpy(f_rms[None].copy()))
            context_names.append(fallback_band)

        return {
            "tile_id": pair["tile_id"],
            "target_band": target_band,
            "target_image": torch.from_numpy(tgt_img[None].copy()),
            "target_rms": torch.from_numpy(tgt_rms[None].copy()),
            "context_bands": context_names,
            "context_images": context_images,
            "context_rms": context_rms,
        }


def collate_multiband_reconstruction(batch: List[Dict]) -> Dict:
    return {
        "tile_id": [b["tile_id"] for b in batch],
        "target_band": [b["target_band"] for b in batch],
        "target_image": [b["target_image"] for b in batch],
        "target_rms": [b["target_rms"] for b in batch],
        "context_bands": [b["context_bands"] for b in batch],
        "context_images": [b["context_images"] for b in batch],
        "context_rms": [b["context_rms"] for b in batch],
    }


def make_reconstruction_loader(
    rubin_dir: str,
    euclid_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    **kwargs,
):
    dataset = JAISPMultiBandReconstructionDataset(
        rubin_dir=rubin_dir,
        euclid_dir=euclid_dir,
        **kwargs,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
        collate_fn=collate_multiband_reconstruction,
    )
    return dataset, loader
