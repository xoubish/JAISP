"""Quick validation for psf_field_pca: render predictions on held-out stamps
per band, report chi²/dof + Pearson r, and save a per-band residual figure
mirroring the v4-NN training visualisation.

Run::

    PYTHONPATH=models python models/psf/validate_psf_pca.py \\
        --train-dir   data/psf_training_v4 \\
        --model       models/checkpoints/psf_field_pca/psf_field_pca.pt \\
        --out-dir     models/checkpoints/psf_field_pca/viz \\
        --n-per-band  5
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch

from psf.psf_field_pca import PSFFieldPCA, ALL_BANDS  # noqa: E402


def _chi2_per_dof(pred, obs, rms, alpha=1.0):
    inv_var = 1.0 / np.maximum(rms, 1e-6) ** 2
    resid = alpha * pred - obs
    chi2 = float((resid ** 2 * inv_var).sum())
    dof = max(int(obs.size - 1), 1)
    return chi2 / dof


def _pearson(a, b):
    a = a.flatten().astype(np.float64); b = b.flatten().astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / max(den, 1e-12))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-per-band", type=int, default=5)
    p.add_argument("--holdout-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = PSFFieldPCA.load(args.model)
    rng = np.random.RandomState(args.seed)

    all_chi2, all_r = {}, {}
    for band in ALL_BANDS:
        path = args.train_dir / f"{band}.npz"
        if not path.exists() or band not in model.models:
            continue
        d = np.load(path, allow_pickle=False)
        N = d["stamps"].shape[0]
        if N < 5:
            continue

        # Tile-disjoint holdout for fair eval, matching train_psf_v4.
        tiles = d["tile_id"]
        unique = np.unique(tiles)
        rng_b = np.random.RandomState(args.seed)
        rng_b.shuffle(unique)
        n_hold = max(1, int(round(args.holdout_frac * len(unique))))
        hold = set(unique[:n_hold])
        mask = np.array([t in hold for t in tiles])
        idx = np.where(mask)[0]
        if len(idx) < args.n_per_band:
            idx = np.arange(N)
        snr = d["snr"][idx]
        order = np.argsort(snr)[::-1]
        pick = idx[order[:args.n_per_band]]

        stamps = torch.tensor(d["stamps"][pick], dtype=torch.float32)
        rms = torch.tensor(d["rms"][pick], dtype=torch.float32)
        frac = torch.tensor(d["frac_xy"][pick], dtype=torch.float32)
        pos = torch.tensor(d["pos_norm"][pick], dtype=torch.float32)
        bidx = torch.full((len(pick),), ALL_BANDS.index(band), dtype=torch.long)

        psf_o = model(pos, bidx)
        rendered = model.render_at_native(psf_o, frac, stamp_size=stamps.shape[-1])

        pred = rendered.squeeze(1)
        inv_var = 1.0 / rms.clamp(min=1e-6) ** 2
        num = (pred * stamps * inv_var).sum(dim=(-2, -1))
        den = (pred * pred * inv_var).sum(dim=(-2, -1)).clamp(min=1e-12)
        alpha = num / den
        scaled = alpha.view(-1, 1, 1) * pred
        resid = stamps - scaled

        chi2s = [_chi2_per_dof(scaled[k].cpu().numpy(),
                               stamps[k].cpu().numpy(),
                               rms[k].cpu().numpy())
                 for k in range(len(pick))]
        rs = [_pearson(scaled[k].cpu().numpy(), stamps[k].cpu().numpy())
              for k in range(len(pick))]
        all_chi2[band] = float(np.median(chi2s))
        all_r[band] = float(np.median(rs))
        print(f"  {band:14s}  median χ²/dof = {all_chi2[band]:6.2f}   "
              f"median r = {all_r[band]:.4f}")

        # Per-band figure mirroring train_psf_v4 viz.
        fig, axes = plt.subplots(4, len(pick), figsize=(2.2 * len(pick), 8.4),
                                 squeeze=False)
        for col in range(len(pick)):
            obs_k = stamps[col].cpu().numpy()
            pred_k = scaled[col].cpu().numpy()
            resid_k = resid[col].cpu().numpy()
            psfo_k = psf_o[col, 0].cpu().numpy()
            vmax_obs = max(np.nanpercentile(obs_k, 99), 1e-6)
            rlim = max(np.nanpercentile(np.abs(resid_k), 99), 1e-6)
            axes[0, col].imshow(obs_k, origin="lower", cmap="gray",
                                vmin=0.0, vmax=vmax_obs)
            axes[0, col].set_title(
                f"SNR={d['snr'][pick[col]]:.0f}  α={float(alpha[col]):.3g}",
                fontsize=8,
            )
            axes[1, col].imshow(pred_k, origin="lower", cmap="gray",
                                vmin=0.0, vmax=vmax_obs)
            axes[2, col].imshow(resid_k, origin="lower", cmap="RdBu_r",
                                vmin=-rlim, vmax=+rlim)
            axes[3, col].imshow(psfo_k, origin="lower", cmap="inferno")
            for r in range(4):
                axes[r, col].axis("off")
        for r, name in enumerate(["Observed", "Pred (α·PSF)", "Residual",
                                  "Oversampled ePSF"]):
            axes[r, 0].set_ylabel(name, fontsize=10)
            axes[r, 0].axis("on")
            axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        fig.suptitle(
            f"PSF PCA — {band}   χ²/dof={all_chi2[band]:.2f}   r={all_r[band]:.3f}   "
            f"N_train={model.models[band]['n_train']}",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        out_path = args.out_dir / f"viz_{band}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    if all_chi2:
        print(f"\nMean across bands:  χ²/dof = "
              f"{np.mean(list(all_chi2.values())):.2f}   "
              f"r = {np.mean(list(all_r.values())):.4f}")


if __name__ == "__main__":
    main()
