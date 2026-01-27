# JAISP_dataloader.py
#
# Paired Rubin + Euclid tiles with EXACT sizes (no crop/pad/resample).
# Returns variable-sized images as lists via a custom collate_fn.
#
# Rubin NPZ schema (per tile):
#   - img: (6,Hr,Wr)
#   - var: (6,Hr,Wr)
#   - mask: (6,Hr,Wr)  [optional, not used here]
#   - wcs_hdr or similar (optional)
#   - ra_center, dec_center, tile_id, ...
#
# Euclid NPZ schema (per tile):
#   - img_VIS/img_Y/img_J/img_H
#   - var_VIS/var_Y/var_J/var_H
#   - wcs_VIS, wcs_Y, ...
#   - ra_center, dec_center, tile_id, ...

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

RUBIN_BANDS  = ["u", "g", "r", "i", "z", "y"]
EUCLID_BANDS = ["VIS", "Y", "J", "H"]


def _to_float32(x):
    x = np.asarray(x)
    return x.astype(np.float32, copy=False) if x.dtype != np.float32 else x


def _safe_sqrt_var(var: np.ndarray) -> np.ndarray:
    """sqrt(max(var,0)) while preserving NaNs."""
    var = _to_float32(var)
    # keep NaNs; only clamp finite negatives to 0
    out = var.copy()
    m = np.isfinite(out)
    out[m] = np.maximum(out[m], 0.0)
    return np.sqrt(out, dtype=np.float32)


def _extract_wcs_like(npz_obj, band=None):
    """
    WCS is stored as strings (to_header_string) in your Euclid saver.
    Rubin may store a header-ish blob under various keys.
    We keep this loose and return whatever is present.
    """
    if band is None:
        for k in ("wcs_hdr", "wcs", "WCS", "header", "fits_header"):
            if k in npz_obj:
                return npz_obj[k]
        return None

    for k in (f"wcs_{band}", f"WCS_{band}", f"header_{band}", f"fits_header_{band}"):
        if k in npz_obj:
            return npz_obj[k]
    return None


def jaisp_collate_variable(batch):
    """
    Variable-size collate: keep images/var/rms as lists (no stacking across batch),
    because Rubin and Euclid have different shapes and tiles can vary.
    """
    out = {
        "x_rubin":    [b["x_rubin"] for b in batch],      # list of (6,Hr,Wr) tensors
        "x_euclid":   [b["x_euclid"] for b in batch],     # list of (4,He,We) tensors
        "mask_euclid": torch.stack([b["mask_euclid"] for b in batch], dim=0),  # (B,4)
        "meta":       [b["meta"] for b in batch],
    }

    # Always present in this implementation
    out["var_rubin"]  = [b["var_rubin"] for b in batch]   # list of (6,Hr,Wr)
    out["var_euclid"] = [b["var_euclid"] for b in batch]  # list of (4,He,We)

    out["rms_rubin"]  = [b["rms_rubin"] for b in batch]   # list of (6,Hr,Wr)
    out["rms_euclid"] = [b["rms_euclid"] for b in batch]  # list of (4,He,We)

    return out


class RubinEuclidTiles(Dataset):
    """
    EXACT-SIZE paired dataset (no crop/pad/resample, no augmentation).

    Rubin file:  <tile_id>.npz
      - img: (6,Hr,Wr)
      - var: (6,Hr,Wr)
      - ra_center, dec_center
      - optional: wcs_hdr (or similar)

    Euclid file: <tile_id>_euclid.npz
      - img_VIS/img_Y/img_J/img_H
      - var_VIS/var_Y/var_J/var_H
      - optional: wcs_VIS, etc.

    Notes on missing Euclid:
      - if a Euclid file exists but one band is missing, we fill placeholders with ref_shape.
      - if the whole Euclid file is missing, we fall back to Rubin shape as placeholders
        (debug convenience only). mask_euclid will be all zeros in that case.
    """

    def __init__(
        self,
        rubin_dir,
        euclid_dir,
        tile_ids=None,
        euclid_missing="zeros",        # missing SCI: "zeros" | "nan"
        euclid_missing_var="nan",      # missing VAR: "nan" | "ones" | "zeros"
        return_wcs=False,
        mmap=True,
    ):
        self.rubin_dir = rubin_dir
        self.euclid_dir = euclid_dir
        self.euclid_missing = euclid_missing
        self.euclid_missing_var = euclid_missing_var
        self.return_wcs = return_wcs
        self.mmap = mmap

        if tile_ids is None:
            rubin_files = sorted(glob.glob(os.path.join(rubin_dir, "tile_x*_y*.npz")))
            tile_ids = [os.path.splitext(os.path.basename(p))[0] for p in rubin_files]
        self.tile_ids = list(tile_ids)

        self.pairs = []
        for tid in self.tile_ids:
            rp = os.path.join(self.rubin_dir, f"{tid}.npz")
            ep = os.path.join(self.euclid_dir, f"{tid}_euclid.npz")
            if os.path.exists(rp):
                self.pairs.append((tid, rp, ep))

        if len(self.pairs) == 0:
            raise FileNotFoundError("No Rubin tiles found (tile_x*_y*.npz).")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tile_id, rubin_path, euclid_path = self.pairs[idx]

        # ---- Rubin ----
        r = np.load(rubin_path, mmap_mode="r" if self.mmap else None, allow_pickle=True)
        rubin_img = _to_float32(r["img"])  # (6,Hr,Wr)

        ra = float(r["ra_center"]) if "ra_center" in r else np.nan
        dec = float(r["dec_center"]) if "dec_center" in r else np.nan

        rubin_var = _to_float32(r["var"]) if "var" in r else None
        if rubin_var is None:
            # This should not happen in your current schema, but keep it robust.
            rubin_var = np.full_like(rubin_img, np.nan, dtype=np.float32)

        rubin_rms = _safe_sqrt_var(rubin_var)

        # ---- Euclid ----
        mask_e = np.zeros((len(EUCLID_BANDS),), dtype=np.float32)

        e_img_stack, e_var_stack = [], []
        wcs_e = {}

        def fill_sci(shape):
            if self.euclid_missing == "nan":
                return np.full(shape, np.nan, np.float32)
            return np.zeros(shape, np.float32)

        def fill_var(shape):
            if self.euclid_missing_var == "zeros":
                return np.zeros(shape, np.float32)
            if self.euclid_missing_var == "ones":
                return np.ones(shape, np.float32)
            # default "nan": safest; downstream can mask by finite(var)
            return np.full(shape, np.nan, np.float32)

        if os.path.exists(euclid_path):
            e = np.load(euclid_path, mmap_mode="r" if self.mmap else None, allow_pickle=True)

            # reference shape for placeholders
            ref_shape = None
            for b in EUCLID_BANDS:
                k_img = f"img_{b}"
                if k_img in e:
                    ref_shape = e[k_img].shape
                    break
            if ref_shape is None:
                # fallback only
                ref_shape = rubin_img.shape[-2:]

            for j, b in enumerate(EUCLID_BANDS):
                k_img = f"img_{b}"
                if k_img in e:
                    img = _to_float32(e[k_img])
                    mask_e[j] = 1.0
                else:
                    img = fill_sci(ref_shape)
                e_img_stack.append(img)

                k_var = f"var_{b}"
                if k_var in e:
                    v = _to_float32(e[k_var])
                else:
                    v = fill_var(ref_shape)
                e_var_stack.append(v)

                if self.return_wcs:
                    w = _extract_wcs_like(e, band=b)
                    if w is not None:
                        wcs_e[b] = w

        else:
            # No Euclid file at all: placeholders (debug convenience only)
            ref_shape = rubin_img.shape[-2:]
            for _ in EUCLID_BANDS:
                e_img_stack.append(fill_sci(ref_shape))
                e_var_stack.append(fill_var(ref_shape))

        euclid_img = np.stack(e_img_stack, axis=0)   # (4,He,We)
        euclid_var = np.stack(e_var_stack, axis=0)   # (4,He,We)
        euclid_rms = _safe_sqrt_var(euclid_var)      # (4,He,We)

        sample = {
            "x_rubin": torch.from_numpy(rubin_img),
            "x_euclid": torch.from_numpy(euclid_img),
            "var_rubin": torch.from_numpy(rubin_var),
            "var_euclid": torch.from_numpy(euclid_var),
            "rms_rubin": torch.from_numpy(rubin_rms),
            "rms_euclid": torch.from_numpy(euclid_rms),
            "mask_euclid": torch.from_numpy(mask_e),
            "meta": {
                "tile_id": tile_id,
                "ra_center": ra,
                "dec_center": dec,
                "rubin_path": rubin_path,
                "euclid_path": euclid_path,
                "rubin_bands": RUBIN_BANDS,
                "euclid_bands": EUCLID_BANDS,
                "rubin_hw": tuple(map(int, rubin_img.shape[-2:])),
                "euclid_hw": tuple(map(int, euclid_img.shape[-2:])),
            },
        }

        if self.return_wcs:
            sample["meta"]["wcs_rubin"] = _extract_wcs_like(r, band=None)
            sample["meta"]["wcs_euclid"] = wcs_e if len(wcs_e) else None

        return sample


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def make_loader(
    rubin_dir,
    euclid_dir,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    **dataset_kwargs,
):
    ds = RubinEuclidTiles(rubin_dir, euclid_dir, **dataset_kwargs)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and (num_workers > 0),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        drop_last=False,
        collate_fn=jaisp_collate_variable,
    )
    return ds, dl
