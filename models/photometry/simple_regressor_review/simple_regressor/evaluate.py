"""Evaluate the CNN flux regressor: metrics + diagnostic plot (with MER x-error bars)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import load_cache, StampDataset
from .losses import scaler_from_state
from .model import build_model



def nmad(x):
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def _metrics(pred, true, ferr, mag_true):
    pred, true, ferr, mag_true = map(lambda a: np.asarray(a, float), (pred, true, ferr, mag_true))
    q = (pred - true) / ferr
    frac = (pred - true) / true
    pos = np.isfinite(pred) & (pred > 0)
    dmag = np.full_like(pred, np.nan)
    dmag[pos] = -2.5 * np.log10(pred[pos] / true[pos])   # zeropoint-free
    return dict(
        n=int(len(pred)),
        median_fractional_flux_error=float(np.median(frac)),
        nmad_fractional_flux_error=nmad(frac),
        p95_absolute_fractional_flux_error=float(np.percentile(np.abs(frac), 95)),
        fraction_above_20pct_flux_error=float((np.abs(frac) > 0.2).mean()),
        flux_rmse_ujy=float(np.sqrt(np.mean((pred - true) ** 2))),
        nonpositive_prediction_fraction=float((~pos).mean()),
        median_delta_mag=float(np.nanmedian(dmag)),
        nmad_delta_mag=nmad(dmag),
        median_abs_q=float(np.median(np.abs(q))),
        fraction_within_1sigma=float((np.abs(q) <= 1).mean()),
        fraction_within_2sigma=float((np.abs(q) <= 2).mean()),
    )


def _load(output_dir, checkpoint):
    ckpt = torch.load(Path(output_dir) / checkpoint, map_location="cpu", weights_only=False)
    saved_cfg = dict(ckpt["config"])
    saved_cfg.setdefault("model_version", 1)
    cfg = Config(**saved_cfg); cfg.input_scale = float(ckpt["input_scale"])
    scaler = scaler_from_state(ckpt["target_scaler"])
    model = build_model(cfg); model.load_state_dict(ckpt["model"]); model.eval()
    return model, cfg, scaler


@torch.no_grad()
def predict(model, scaler, cache, mask, cfg, device, baseline=False):
    ds = StampDataset(cache, mask, cfg.input_scale, cfg.bin_factor, cfg.centre_sigma_frac, augment=False, model_version=cfg.model_version)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    preds = []
    for feat, linear, _, _ in dl:
        linear = linear.to(device)
        flux = (model.baseline_flux(linear) if baseline else
                scaler.to_flux(model(feat.to(device), linear)))
        preds.append(flux.cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


def evaluate(output_dir, split="test", checkpoint="best.pt", save_plot=True):
    out = Path(output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, scaler = _load(out, checkpoint); model.to(device)
    torch.set_num_threads(max(1, cfg.torch_threads))
    cache = load_cache(str(out / "stamps_cache.npz"))
    mask = cache["split"].astype(str) == split
    if mask.sum() == 0:
        raise RuntimeError(f"No sources in split '{split}'")
    pred = predict(model, scaler, cache, mask, cfg, device)
    true = cache["flux"][mask].astype(np.float64)
    ferr = cache["fluxerr"][mask].astype(np.float64)
    mag = cache["mag"][mask].astype(np.float64)

    if not np.isfinite(pred).all():
        raise RuntimeError("Non-finite predictions")
    baseline = (predict(model, scaler, cache, mask, cfg, device, baseline=True)
                if cfg.model_version == 2 else None)
    baseline_metrics = _metrics(baseline, true, ferr, mag) if baseline is not None else None
    overall = _metrics(pred, true, ferr, mag)
    bins = []
    edges = np.arange(np.floor(mag.min()), np.ceil(mag.max()) + 1e-6, 1.0)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mag >= lo) & (mag < hi)
        if m.sum() >= 3:
            d = _metrics(pred[m], true[m], ferr[m], mag[m]); d.update(mag_min=float(lo), mag_max=float(hi))
            if baseline is not None:
                d["aperture_baseline"] = _metrics(baseline[m], true[m], ferr[m], mag[m])
            bins.append(d)
    (out / f"metrics_{split}.json").write_text(json.dumps(
        dict(split=split, pool=cfg.pool, overall=overall, magnitude_bins=bins,
             aperture_baseline=baseline_metrics, q_note="q uses catalog error only; not predicted uncertainty"), indent=2))
    with open(out / f"predictions_{split}.csv", "w", newline="") as f:
        wtr = csv.writer(f); wtr.writerow(["object_id", "mag", "flux_true_ujy", "fluxerr_ujy", "flux_pred_ujy", "q", "aperture_flux_ujy"])
        oid = cache["object_id"][mask]
        for i in range(len(pred)):
            wtr.writerow([int(oid[i]), f"{mag[i]:.4f}", f"{true[i]:.6g}", f"{ferr[i]:.6g}",
                          f"{pred[i]:.6g}", f"{(pred[i]-true[i])/ferr[i]:.4f}",
                          "" if baseline is None else f"{baseline[i]:.6g}"])
    if save_plot:
        _plot(pred, true, ferr, mag, overall, cfg.pool, out / f"diagnostic_{split}.png")
    print(f"[evaluate] pool={cfg.pool} {split}: n={overall['n']} med|q|={overall['median_abs_q']:.2f} "
          f"within1sig={overall['fraction_within_1sigma']:.1%} NMAD(frac)={overall['nmad_fractional_flux_error']:.3f} "
          f"medfracerr={overall['median_fractional_flux_error']:+.3f}")
    return dict(split=split, pool=cfg.pool, overall=overall, magnitude_bins=bins,
             aperture_baseline=baseline_metrics, q_note="q uses catalog error only; not predicted uncertainty")


def _plot(pred, true, ferr, mag, overall, pool, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    q = (pred - true) / ferr
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    lim = [max(1e-3, np.nanmin(true) * 0.5), np.nanmax(true) * 2]
    # x error bars = MER quoted flux uncertainty -> shows the intrinsic (label) scatter.
    ax[0].errorbar(true, np.clip(pred, lim[0], None), xerr=ferr, fmt="o", ms=3,
                   alpha=0.35, elinewidth=0.6, capsize=0, mec="none")
    ax[0].plot(lim, lim, "k--", lw=1); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlim(lim); ax[0].set_ylim(lim); ax[0].set_xlabel("MER flux +/- err [uJy]")
    ax[0].set_ylabel("predicted flux [uJy]"); ax[0].set_title(f"flux vs flux  (pool={pool}, x-err=MER)")
    ax[1].scatter(mag, np.clip((pred - true) / true, -2, 2), s=6, alpha=0.4)
    ax[1].axhline(0, color="k", ls="--", lw=1); ax[1].set_ylim(-2, 2)
    ax[1].set_xlabel("MER VIS mag"); ax[1].set_ylabel("(pred - MER)/MER")
    ax[1].set_title(f"fractional error (median {overall['median_fractional_flux_error']:+.3f}, "
                    f"NMAD {overall['nmad_fractional_flux_error']:.3f})")
    ax[2].hist(np.clip(q, -6, 6), bins=41, alpha=0.8); ax[2].axvline(0, color="k", ls="--", lw=1)
    ax[2].set_xlabel("q = (pred - MER)/MER_err")
    ax[2].set_title(f"within 1sig {overall['fraction_within_1sigma']:.0%}, med|q| {overall['median_abs_q']:.2f}")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    evaluate(a.output_dir, a.split, a.checkpoint, save_plot=not a.no_plot)


if __name__ == "__main__":
    main()
