import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

RUBIN_BANDS  = ["u", "g", "r", "i", "z", "y"]
EUCLID_BANDS = ["VIS", "Y", "J", "H"]

def _to_float32(x):
    x = np.asarray(x)
    return x.astype(np.float32, copy=False) if x.dtype != np.float32 else x

def _extract_wcs_like(npz_obj, band=None):
    if band is None:
        for k in ("wcs", "WCS", "header", "fits_header"):
            if k in npz_obj:
                return npz_obj[k]
        return None
    for k in (f"wcs_{band}", f"WCS_{band}", f"header_{band}", f"fits_header_{band}"):
        if k in npz_obj:
            return npz_obj[k]
    return None

def _per_band_percentiles(x, qlo=1.0, qhi=99.0):
    C = x.shape[0]
    lo = np.zeros((C,), dtype=np.float32)
    hi = np.zeros((C,), dtype=np.float32)
    for c in range(C):
        v = x[c]
        lo[c] = np.nanpercentile(v, qlo)
        hi[c] = np.nanpercentile(v, qhi)
        if not np.isfinite(lo[c]) or not np.isfinite(hi[c]) or hi[c] <= lo[c]:
            mu = np.nanmean(v)
            sig = np.nanstd(v)
            lo[c] = np.float32(mu - 3.0 * sig)
            hi[c] = np.float32(mu + 3.0 * sig + 1e-6)
    return lo, hi

def normalize_affine(x, lo, hi, eps=1e-6):
    lo = lo[:, None, None]
    hi = hi[:, None, None]
    return (x - lo) / (hi - lo + eps)

def denormalize_affine(xn, lo, hi, eps=1e-6):
    lo = lo[:, None, None]
    hi = hi[:, None, None]
    return xn * (hi - lo + eps) + lo

def normalize_robust01(x, lo, hi, eps=1e-6):
    xn = normalize_affine(x, lo, hi, eps=eps)
    return np.clip(xn, 0.0, 1.0)

def jaisp_collate_variable(batch):
    """
    Variable-size collate:
      - keep images as lists (no stacking across batch)
      - stack only mask_euclid (fixed length 4)
      - meta stays list[dict]
    """
    return {
        "x_rubin": [b["x_rubin"] for b in batch],     # list of (6,Hr,Wr)
        "x_euclid": [b["x_euclid"] for b in batch],   # list of (4,He,We)
        "mask_euclid": torch.stack([b["mask_euclid"] for b in batch], dim=0),  # (B,4)
        "meta": [b["meta"] for b in batch],
    }

class RubinEuclidTiles(Dataset):
    """
    EXACT-SIZE paired dataset. No augmentation. No resizing/cropping/padding.

    Rubin:  <tile_id>.npz
      - img: (6,Hr,Wr) in RUBIN_BANDS order
      - ra_center, dec_center
      - optional: wcs/header

    Euclid: <tile_id>_euclid.npz
      - img_VIS/img_Y/img_J/img_H each (He,We)
      - optional: wcs_VIS, etc.
    """
    def __init__(
        self,
        rubin_dir,
        euclid_dir,
        tile_ids=None,
        normalize="none",          # "none" | "affine" | "robust01"
        norm_qlo=1.0,
        norm_qhi=99.0,
        euclid_missing="zeros",    # "zeros" | "nan"
        return_wcs=False,
        mmap=True,
    ):
        self.rubin_dir = rubin_dir
        self.euclid_dir = euclid_dir
        self.normalize = normalize
        self.norm_qlo = norm_qlo
        self.norm_qhi = norm_qhi
        self.euclid_missing = euclid_missing
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

        # ---- Euclid ----
        mask_e = np.zeros((len(EUCLID_BANDS),), dtype=np.float32)
        e_stack = []
        wcs_e = {}

        if os.path.exists(euclid_path):
            e = np.load(euclid_path, mmap_mode="r" if self.mmap else None, allow_pickle=True)

            # reference shape for missing band placeholders
            ref_shape = None
            for b in EUCLID_BANDS:
                k = f"img_{b}"
                if k in e:
                    ref_shape = e[k].shape
                    break
            if ref_shape is None:
                # if no Euclid images exist, fall back to Rubin shape for placeholders
                ref_shape = rubin_img.shape[-2:]

            for j, b in enumerate(EUCLID_BANDS):
                k = f"img_{b}"
                if k in e:
                    img = _to_float32(e[k])
                    mask_e[j] = 1.0
                else:
                    img = (np.full(ref_shape, np.nan, np.float32)
                           if self.euclid_missing == "nan"
                           else np.zeros(ref_shape, np.float32))
                e_stack.append(img)

                if self.return_wcs:
                    w = _extract_wcs_like(e, band=b)
                    if w is not None:
                        wcs_e[b] = w
        else:
            # Entire Euclid file missing: placeholders in Rubin shape
            ref_shape = rubin_img.shape[-2:]
            for _ in EUCLID_BANDS:
                img = (np.full(ref_shape, np.nan, np.float32)
                       if self.euclid_missing == "nan"
                       else np.zeros(ref_shape, np.float32))
                e_stack.append(img)

        euclid_img = np.stack(e_stack, axis=0)  # (4,He,We) exact native

        # ---- Optional normalization (stored for reversibility) ----
        norm = {"rubin": None, "euclid": None}
        if self.normalize in ("affine", "robust01"):
            r_lo, r_hi = _per_band_percentiles(rubin_img, qlo=self.norm_qlo, qhi=self.norm_qhi)
            rubin_img = normalize_affine(rubin_img, r_lo, r_hi) if self.normalize == "affine" \
                        else normalize_robust01(rubin_img, r_lo, r_hi)
            norm["rubin"] = {"method": self.normalize, "qlo": float(self.norm_qlo), "qhi": float(self.norm_qhi),
                             "lo": r_lo, "hi": r_hi}

            e_lo, e_hi = _per_band_percentiles(euclid_img, qlo=self.norm_qlo, qhi=self.norm_qhi)
            euclid_img = normalize_affine(euclid_img, e_lo, e_hi) if self.normalize == "affine" \
                         else normalize_robust01(euclid_img, e_lo, e_hi)
            norm["euclid"] = {"method": self.normalize, "qlo": float(self.norm_qlo), "qhi": float(self.norm_qhi),
                              "lo": e_lo, "hi": e_hi}

        sample = {
            "x_rubin": torch.from_numpy(rubin_img),     # (6,Hr,Wr)
            "x_euclid": torch.from_numpy(euclid_img),   # (4,He,We)
            "mask_euclid": torch.from_numpy(mask_e),    # (4,)
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
                "norm": norm,
            }
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
