"""Empirical ePSF model per band, built with photutils.EPSFBuilder.

Iterative shift+stack with sigma-clipping. For each band:
  1. Background-subtract each stamp from a sigma-clipped median over an outer
     annulus (8-14 native px).
  2. Wrap each stamp as a ``photutils.psf.EPSFStar`` with its sub-pixel
     centroid (data-derived ``frac_xy`` from the build).
  3. Run ``EPSFBuilder``: oversampled ePSF is built by sigma-clipped median
     stacking of recentred stamps; centroids are refined using the running
     ePSF and the build is iterated until convergence.

There is intentionally **no spatial variation** in this model. PCA + polynomial
fits on the same data showed near-zero spatial R², i.e. the PSF is essentially
constant across the field for every band. One ePSF per band keeps the model
honest and the inference trivial.

Build:

    PYTHONPATH=models python models/psf/psf_field_pca.py \\
        --train-dir   data/psf_training_v4 \\
        --output-path models/checkpoints/psf_field_pca/psf_field_pca.pt \\
        --psf-size 99 --oversampling 5 \\
        --rubin-snr-min 80 --euclid-snr-min 30 \\
        --morpho-tol 0.4 --max-stars 2000
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Photutils complains about a deprecation in astropy 5+; not relevant here.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="photutils")

from photutils.psf import EPSFBuilder, EPSFStar, EPSFStars  # noqa: E402


ALL_BANDS = ["rubin_u", "rubin_g", "rubin_r", "rubin_i", "rubin_z", "rubin_y",
             "euclid_VIS", "euclid_Y", "euclid_J", "euclid_H"]


# ============================================================
# Helpers
# ============================================================

def _sigma_clipped_median(values: np.ndarray, n_sigma: float = 3.0,
                          n_iter: int = 3) -> float:
    v = np.asarray(values, dtype=np.float64)
    for _ in range(n_iter):
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        sigma = 1.4826 * mad
        if sigma <= 0 or v.size < 5:
            break
        keep = np.abs(v - med) < n_sigma * sigma
        if keep.sum() < 5:
            break
        v = v[keep]
    return float(np.median(v))


def _bg_subtract(stamp: np.ndarray, frac_xy: np.ndarray,
                 bg_inner: float = 8.0, bg_outer: float = 14.0,
                 ) -> Optional[np.ndarray]:
    """Sigma-clipped median bg from an outer annulus, subtract."""
    H, W = stamp.shape[-2:]
    yy, xx = np.indices((H, W), dtype=np.float32)
    cx = W // 2 + float(frac_xy[0])
    cy = H // 2 + float(frac_xy[1])
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    bg_mask = (r2 > bg_inner ** 2) & (r2 < bg_outer ** 2)
    if bg_mask.sum() < 10:
        return None
    bg = _sigma_clipped_median(stamp[bg_mask])
    return stamp.astype(np.float64) - bg


def _measure_sigma_native(stamp_bs: np.ndarray, frac_xy: np.ndarray,
                          ap_radius: float = 5.0) -> Optional[float]:
    """Intensity-weighted sigma (quadrature mean of σx, σy) inside a small
    aperture around the source. ``stamp_bs`` should already be background-
    subtracted. Returns ``None`` if the aperture has no signal."""
    H, W = stamp_bs.shape[-2:]
    yy, xx = np.indices((H, W), dtype=np.float32)
    cx = W // 2 + float(frac_xy[0])
    cy = H // 2 + float(frac_xy[1])
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    s = np.clip(stamp_bs, 0.0, None)
    in_ap = r2 < ap_radius ** 2
    w = s * in_ap
    total = float(w.sum())
    if total <= 1e-9:
        return None
    cx_m = float((xx * w).sum() / total)
    cy_m = float((yy * w).sum() / total)
    sx2 = max(float(((xx - cx_m) ** 2 * w).sum() / total), 1e-9)
    sy2 = max(float(((yy - cy_m) ** 2 * w).sum() / total), 1e-9)
    return 0.5 * (float(np.sqrt(sx2)) + float(np.sqrt(sy2)))


def _core_sigma_mas(psf_oversampled: np.ndarray, oversampling: int, band: str,
                    core_radius_native: float = 5.0) -> Tuple[float, float]:
    """Intensity-weighted σ measured *only inside the PSF core* (radius
    ``core_radius_native`` native pixels), reported in milli-arcseconds.

    Reporting full-stamp σ inflates the number with contamination from
    galaxy/neighbour residuals out in the wings — it is not the σ the science
    pipeline cares about. Core σ (within ~5σ of a stellar PSF) is the
    meaningful comparison to expected band PSF widths.

    Returns ``(sigma_mas, core_flux_fraction)``.
    """
    P = psf_oversampled.shape[0]
    yy, xx = np.indices((P, P), dtype=np.float32)
    centre = (P - 1) / 2.0
    r = np.sqrt((xx - centre) ** 2 + (yy - centre) ** 2)  # oversampled px
    core_mask = r < core_radius_native * oversampling
    core = psf_oversampled * core_mask
    s_core = float(core.sum())
    if s_core <= 0:
        return float("nan"), 0.0
    cx = float((xx * core).sum() / s_core)
    cy = float((yy * core).sum() / s_core)
    sigma_ovs = float(np.sqrt(((xx - cx) ** 2 * core).sum() / s_core))
    sigma_native = sigma_ovs / oversampling
    px_native = 0.2 if band.startswith("rubin") else 0.1
    sigma_mas = sigma_native * px_native * 1000.0
    core_frac = s_core / max(float(psf_oversampled.sum()), 1e-12)
    return sigma_mas, core_frac


def _crop_or_pad_centred(arr: np.ndarray, target: int) -> np.ndarray:
    """Crop or zero-pad a 2-D array to ``target × target`` keeping the
    geometric centre stable. Used to fit photutils' ePSF output to the
    declared ``psf_size`` without shifting the centroid."""
    H, W = arr.shape
    out = np.zeros((target, target), dtype=arr.dtype)
    # Source pixel index (centre of input) and target pixel index (centre of output).
    cy_src = (H - 1) / 2.0
    cx_src = (W - 1) / 2.0
    cy_dst = (target - 1) / 2.0
    cx_dst = (target - 1) / 2.0
    # Valid src/dst pixel ranges that overlap, indexed off the centre.
    half = min((H - 1) // 2, (W - 1) // 2, (target - 1) // 2)
    y_src0 = int(round(cy_src - half))
    x_src0 = int(round(cx_src - half))
    y_dst0 = int(round(cy_dst - half))
    x_dst0 = int(round(cx_dst - half))
    side = 2 * half + 1
    out[y_dst0:y_dst0 + side, x_dst0:x_dst0 + side] = \
        arr[y_src0:y_src0 + side, x_src0:x_src0 + side]
    return out


# ============================================================
# Per-band fit
# ============================================================

def fit_band_epsf(stamps_native: np.ndarray, frac_xy: np.ndarray,
                  snr: np.ndarray, *,
                  oversampling: int = 5, psf_size: int = 99,
                  snr_min: float = 20.0, max_stars: int = 2000,
                  norm_radius: float = 5.5,
                  maxiters: int = 10,
                  progress_bar: bool = False,
                  morpho_tol: float = 0.4,
                  morpho_ap_radius: float = 5.0,
                  ) -> Dict:
    """Build one band's ePSF via photutils.EPSFBuilder.

    Cuts applied (in order):
      1. SNR ≥ ``snr_min``.
      2. Cap to top-``max_stars`` by SNR (compute is O(N·maxiters)).
      3. Morphology cut: per-stamp σ_native must lie within
         ``[Q25 · (1 − morpho_tol), Q25 · (1 + morpho_tol)]`` where Q25 is the
         lower-quartile of σ across the surviving sample. The lower quartile
         (rather than median) is robust to galaxy contamination — galaxies are
         systematically wider, so Q25 sits inside the stellar locus.

    Returns a dict with:
        ``data``: oversampled ePSF stamp ``[psf_size, psf_size]`` (unit-flux)
        ``oversampling``: int
        ``n_train``: number of stars actually used after photutils' filtering
        ``sigma_q25``: per-stamp σ Q25 used for the morphology cut (native px)
        ``n_morpho_rejected``: stars dropped by morphology cut
    """
    keep = snr >= snr_min
    stamps_f = stamps_native[keep]
    fxy_f = frac_xy[keep]
    snr_f = snr[keep]

    # Cap to top-`max_stars` by SNR — photutils does sigma-clipping internally
    # but its compute scales with N_stars × maxiters, so 2k is a sensible cap.
    if len(stamps_f) > max_stars:
        order = np.argsort(snr_f)[::-1][:max_stars]
        stamps_f = stamps_f[order]
        fxy_f = fxy_f[order]

    # Pass 1: bg-subtract each stamp + measure its σ_native. We keep both so
    # we can morpho-cut without re-doing the bg subtraction.
    bs_stamps: List[np.ndarray] = []
    bs_fracs: List[np.ndarray] = []
    sigmas: List[float] = []
    for k in range(len(stamps_f)):
        s_bs = _bg_subtract(stamps_f[k].astype(np.float32), fxy_f[k])
        if s_bs is None:
            continue
        sig = _measure_sigma_native(s_bs, fxy_f[k],
                                    ap_radius=morpho_ap_radius)
        if sig is None or not np.isfinite(sig):
            continue
        bs_stamps.append(s_bs)
        bs_fracs.append(fxy_f[k])
        sigmas.append(sig)

    if len(sigmas) < 10:
        raise RuntimeError(f"Too few stars after bg/sigma measurement: {len(sigmas)}")

    sigmas_arr = np.array(sigmas, dtype=np.float32)
    sigma_q25 = float(np.quantile(sigmas_arr, 0.25))
    sigma_lo = sigma_q25 * (1.0 - morpho_tol)
    sigma_hi = sigma_q25 * (1.0 + morpho_tol)
    morpho_keep = (sigmas_arr >= sigma_lo) & (sigmas_arr <= sigma_hi)
    n_morpho_rejected = int((~morpho_keep).sum())

    # Pass 2: build EPSFStars from the morpho-clean subset.
    stars: List[EPSFStar] = []
    for k in range(len(bs_stamps)):
        if not morpho_keep[k]:
            continue
        s_bs = bs_stamps[k]
        H, W = s_bs.shape
        cx = W // 2 + float(bs_fracs[k][0])
        cy = H // 2 + float(bs_fracs[k][1])
        try:
            stars.append(EPSFStar(s_bs.astype(np.float64),
                                  cutout_center=(cx, cy)))
        except Exception:
            continue

    if len(stars) < 5:
        raise RuntimeError(f"Too few stars after morpho cut: {len(stars)} "
                           f"(σ_q25={sigma_q25:.2f}, kept σ∈[{sigma_lo:.2f}, {sigma_hi:.2f}])")
    epsf_stars = EPSFStars(stars)

    builder = EPSFBuilder(
        oversampling=oversampling,
        shape=(psf_size, psf_size),
        maxiters=maxiters,
        progress_bar=progress_bar,
        norm_radius=norm_radius,
        smoothing_kernel="quartic",
    )
    epsf, fitted_stars = builder(epsf_stars)

    data = np.asarray(epsf.data, dtype=np.float32)
    if data.shape != (psf_size, psf_size):
        # Should match because we set shape=, but pad/crop defensively.
        data = _crop_or_pad_centred(data, psf_size).astype(np.float32)

    # Normalise to unit flux (photutils normalises within ``norm_radius`` only,
    # which is fine for pipeline use — but we want unit-flux for compatibility
    # with the renderer's sum=1 assumption).
    total = float(data.sum())
    if total > 0:
        data = data / total

    return {
        "data": data,
        "oversampling": int(oversampling),
        "n_train": int(len(fitted_stars)),
        "sigma_q25": float(sigma_q25),
        "sigma_lo": float(sigma_lo),
        "sigma_hi": float(sigma_hi),
        "n_morpho_rejected": n_morpho_rejected,
    }


# ============================================================
# Inference class
# ============================================================

class PSFFieldEPSF:
    """Empirical per-band ePSF model. No spatial variation — one ePSF per
    band, looked up by ``band_idx``. Same forward/render API as PSFFieldV4."""

    def __init__(self, models: Dict[str, Dict], oversampling: int,
                 psf_size: int, band_names: List[str]):
        self.models = models
        self.oversampling = int(oversampling)
        self.psf_size = int(psf_size)
        self.band_names = list(band_names)
        self.band_to_idx = {b: i for i, b in enumerate(self.band_names)}

    def __call__(self, pos_norm: torch.Tensor,
                 band_idx: torch.Tensor) -> torch.Tensor:
        """[B, 2] in [-1, 1] (ignored — no spatial variation), [B] long
        → [B, 1, P, P] unit-flux ePSF."""
        device = pos_norm.device
        band_np = band_idx.detach().cpu().numpy()
        B = band_np.shape[0]
        P = self.psf_size
        out = np.empty((B, 1, P, P), dtype=np.float32)
        for k in range(B):
            band = self.band_names[int(band_np[k])]
            mdl = self.models[band]
            out[k, 0] = mdl["data"]
        return torch.from_numpy(out).to(device)

    def render_at_native(self, psf_oversampled: torch.Tensor,
                         frac_xy: torch.Tensor,
                         stamp_size: int = 32) -> torch.Tensor:
        """Same convention as PSFFieldV4.render_at_native."""
        B, _, P, _ = psf_oversampled.shape
        ovs = self.oversampling
        device = psf_oversampled.device

        half = stamp_size // 2
        coords = torch.arange(stamp_size, device=device, dtype=torch.float32) - float(half)
        sub = (torch.arange(ovs, device=device, dtype=torch.float32)
               - (ovs - 1) / 2.0) / ovs
        ys = coords.view(stamp_size, 1, 1, 1) + sub.view(1, ovs, 1, 1)
        xs = coords.view(1, 1, stamp_size, 1) + sub.view(1, 1, 1, ovs)

        ys_b = ys.unsqueeze(0).expand(B, -1, -1, -1, -1)
        xs_b = xs.unsqueeze(0).expand(B, -1, -1, -1, -1)
        ys_b = ys_b - frac_xy[:, 1].view(B, 1, 1, 1, 1)
        xs_b = xs_b - frac_xy[:, 0].view(B, 1, 1, 1, 1)

        ys_idx = ys_b * ovs
        xs_idx = xs_b * ovs
        centre = (P - 1) / 2.0
        ys_pix = ys_idx + centre
        xs_pix = xs_idx + centre
        ys_n = 2.0 * ys_pix / (P - 1) - 1.0
        xs_n = 2.0 * xs_pix / (P - 1) - 1.0

        grid = torch.stack([
            xs_n.expand(B, stamp_size, ovs, stamp_size, ovs),
            ys_n.expand(B, stamp_size, ovs, stamp_size, ovs),
        ], dim=-1)
        grid = grid.reshape(B, stamp_size * ovs, stamp_size * ovs, 2)

        sampled = F.grid_sample(psf_oversampled, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True)
        sampled = sampled.view(B, 1, stamp_size, ovs, stamp_size, ovs).sum(dim=(3, 5))
        return sampled

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "models": self.models,
            "oversampling": self.oversampling,
            "psf_size": self.psf_size,
            "band_names": self.band_names,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "PSFFieldEPSF":
        blob = torch.load(str(path), weights_only=False, map_location="cpu")
        return cls(models=blob["models"],
                   oversampling=blob["oversampling"],
                   psf_size=blob["psf_size"],
                   band_names=blob["band_names"])


# Backwards-compatible alias for any code/notebooks that import the old name.
PSFFieldPCA = PSFFieldEPSF


# ============================================================
# CLI build
# ============================================================

def _build(args: argparse.Namespace) -> None:
    bands = ALL_BANDS if args.bands is None else [b.strip() for b in args.bands.split(",")]
    print(f"Building EPSF model: psf_size={args.psf_size}, "
          f"oversampling={args.oversampling}, "
          f"rubin_snr_min={args.rubin_snr_min}, euclid_snr_min={args.euclid_snr_min}, "
          f"max_stars={args.max_stars}, morpho_tol={args.morpho_tol}, "
          f"maxiters={args.maxiters}")

    models: Dict[str, Dict] = {}
    for band in bands:
        path = args.train_dir / f"{band}.npz"
        if not path.exists():
            print(f"  {band:14s}  skipping (no {path.name})")
            continue
        d = np.load(path, allow_pickle=False)
        N0 = d["stamps"].shape[0]
        if N0 < 50:
            print(f"  {band:14s}  skipping (only {N0} stamps)")
            continue
        snr_min = args.rubin_snr_min if band.startswith("rubin") else args.euclid_snr_min
        try:
            mdl = fit_band_epsf(
                stamps_native=d["stamps"],
                frac_xy=d["frac_xy"],
                snr=d["snr"],
                oversampling=args.oversampling,
                psf_size=args.psf_size,
                snr_min=snr_min,
                max_stars=args.max_stars,
                norm_radius=args.norm_radius,
                maxiters=args.maxiters,
                progress_bar=False,
                morpho_tol=args.morpho_tol,
            )
        except Exception as e:
            print(f"  {band:14s}  FAILED: {type(e).__name__}: {e}")
            continue
        sig_core_mas, core_frac = _core_sigma_mas(
            mdl["data"], oversampling=args.oversampling, band=band,
            core_radius_native=5.0,
        )
        print(f"  {band:14s}  N={mdl['n_train']:5d}  "
              f"σ_q25={mdl['sigma_q25']:.2f}±{args.morpho_tol*100:.0f}%  "
              f"morpho_rej={mdl['n_morpho_rejected']:4d}  "
              f"core σ={sig_core_mas:.0f} mas  core/total flux={core_frac:.2f}")
        models[band] = mdl

    if not models:
        raise RuntimeError("No bands successfully fit — check inputs.")

    PSFFieldEPSF(
        models=models, oversampling=args.oversampling,
        psf_size=args.psf_size, band_names=ALL_BANDS,
    ).save(args.output_path)
    print(f"\nSaved {len(models)} bands → {args.output_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", required=True, type=Path)
    p.add_argument("--output-path", required=True, type=Path)
    p.add_argument("--psf-size", type=int, default=99)
    p.add_argument("--oversampling", type=int, default=5)
    p.add_argument("--rubin-snr-min", type=float, default=80.0,
                   help="Minimum stamp SNR for Rubin bands (default 80). Rubin "
                        "is shallower so a strict SNR cut is needed to keep "
                        "galaxy contamination low.")
    p.add_argument("--euclid-snr-min", type=float, default=30.0,
                   help="Minimum stamp SNR for Euclid bands (default 30).")
    p.add_argument("--max-stars", type=int, default=2000,
                   help="Cap on stars per band — top-SNR are kept (default 2000).")
    p.add_argument("--morpho-tol", type=float, default=0.4,
                   help="Morphology cut tolerance: keep stars with σ_native in "
                        "[Q25 · (1−tol), Q25 · (1+tol)]. Default 0.4 (±40%%). "
                        "Set 0.0 to disable.")
    p.add_argument("--norm-radius", type=float, default=5.5,
                   help="EPSFBuilder norm_radius in native pixels (default 5.5).")
    p.add_argument("--maxiters", type=int, default=10,
                   help="EPSFBuilder iterations (default 10).")
    p.add_argument("--bands", type=str, default=None,
                   help="Comma-separated band names. Default: all 10.")
    return p


__all__ = ["PSFFieldEPSF", "PSFFieldPCA", "fit_band_epsf",
           "_core_sigma_mas", "ALL_BANDS"]


if __name__ == "__main__":
    _build(build_argparser().parse_args())
