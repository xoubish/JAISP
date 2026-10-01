"""Mask-aware image context and native linear aperture inputs.

V2 subtracts an annulus median, excludes invalid image/variance pixels, and adds
valid coverage as the fourth CNN channel. Aperture measurements use native pixels
regardless of CNN binning. Variance sums assume independent input pixels.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def load_cache(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def block_sum(a, f):
    """Sum f x f blocks. a: [...,H,W] -> [...,H/f,W/f] (flux-conserving downsample)."""
    if f == 1:
        return a
    H, W = a.shape[-2:]
    a = a[..., : H // f * f, : W // f * f]
    new = a.shape[:-2] + (H // f, f, W // f, f)
    return a.reshape(new).sum(axis=(-3, -1))


def centre_prior(size, sigma_frac):
    c = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size]
    r2 = (x - c) ** 2 + (y - c) ** 2
    sig = max(sigma_frac * size, 1.0)
    return np.exp(-r2 / (2 * sig * sig)).astype(np.float32)


def fit_input_scale(stamps_img, bin_factor):
    """Global asinh scale for the (binned) image channel; scalar, amplitude-preserving."""
    b = block_sum(np.asarray(stamps_img, dtype=np.float32), bin_factor)
    s = float(np.nanstd(b))
    return s if np.isfinite(s) and s > 0 else 1.0


def _dihedral(x, k):
    if k & 4:
        x = torch.flip(x, dims=[2])
    return torch.rot90(x, k & 3, dims=[1, 2])


class StampDataset(Dataset):
    def __init__(self, cache, mask, input_scale, bin_factor=2, centre_sigma_frac=0.10,
                 augment=False, model_version=2):
        self.img = cache["stamps"][mask][:, 0]      # [N,128,128] native
        self.rms = cache["stamps"][mask][:, 1]      # [N,128,128] native (0 = masked)
        self.flux = cache["flux"][mask].astype(np.float32)
        self.fluxerr = cache["fluxerr"][mask].astype(np.float32)
        self.mag = cache["mag"][mask].astype(np.float32)
        self.input_scale = float(input_scale)
        self.f = int(bin_factor)
        self.augment = bool(augment)
        self.model_version = model_version
        if self.f < 1 or self.img.shape[-1] % self.f or self.img.shape[-2] % self.f:
            raise ValueError("bin_factor must divide the stamp dimensions")
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError("input_scale must be finite and positive")
        self.S = self.img.shape[-1] // self.f
        self.cp = torch.from_numpy(centre_prior(self.S, centre_sigma_frac))
        size = self.img.shape[-1]
        yy, xx = np.mgrid[:size, :size]
        rr = np.hypot(xx - (size - 1) / 2, yy - (size - 1) / 2)
        self.sky = (rr > 0.86 * size / 2) & (rr < 0.98 * size / 2)

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, i):
        raw = self.img[i].astype(np.float32)
        rms = self.rms[i].astype(np.float32)
        valid = np.isfinite(raw) & np.isfinite(rms) & (rms > 0)
        if not valid.any():
            raise ValueError("Stamp has no valid variance pixels; rebuild the cache")
        sky = valid & self.sky
        if not sky.any():
            raise ValueError("Stamp has no valid background annulus; rebuild the cache")
        background = float(np.median(raw[sky]))
        clean = np.where(valid, raw - background, 0.0)
        var = block_sum(np.where(valid, rms, 0.0) ** 2, self.f)
        img = block_sum(clean, self.f)
        snr = np.divide(img, np.sqrt(var), out=np.zeros_like(img), where=var > 0)
        coverage = block_sum(valid.astype(np.float32), self.f) / self.f ** 2
        feat = torch.stack([torch.asinh(torch.from_numpy(img) / self.input_scale),
                            torch.asinh(torch.from_numpy(snr)), self.cp,
                            torch.from_numpy(coverage)])
        img_lin = torch.from_numpy(np.stack([clean, valid.astype(np.float32)]))
        if self.model_version == 1:
            # Reproduce the original input convention for old checkpoints only.
            img = block_sum(raw, self.f)
            var = block_sum(rms ** 2, self.f)
            feat = torch.stack([torch.asinh(torch.from_numpy(img) / self.input_scale),
                                torch.asinh(torch.from_numpy(img / np.sqrt(var + 1e-12))), self.cp])
            feat = torch.nan_to_num(feat, nan=0., posinf=0., neginf=0.)
        if self.augment:
            k = int(torch.randint(0, 8, (1,)).item())
            feat, img_lin = _dihedral(feat, k), _dihedral(img_lin, k)
        return feat, img_lin, torch.tensor(self.flux[i]), torch.tensor(self.fluxerr[i])


def balanced_weights(mag, bin_width=0.75, max_ratio=5.0):
    """1/sqrt(count) per 0.75-mag bin, capped -> sqrt magnitude-balanced sampling."""
    if len(mag) == 0:
        raise ValueError("Cannot balance an empty sample")
    if np.ptp(mag) == 0:
        return np.ones(len(mag), dtype=np.float64)
    edges = np.arange(mag.min(), mag.max() + bin_width, bin_width)
    idx = np.clip(np.digitize(mag, edges) - 1, 0, len(edges) - 2)
    counts = np.bincount(idx, minlength=len(edges) - 1).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts[idx], 1.0))
    w = w / w.min()
    return np.minimum(w, max_ratio).astype(np.float64)
