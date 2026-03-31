"""Post-training evaluation for JAISP Foundation v7 mixed-resolution reconstruction.

This evaluator scores every available target band on a tile using all other
available bands as context. On Euclid-covered tiles that means all 10 bands:
Rubin u/g/r/i/z/y plus Euclid VIS/Y/J/H.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from jaisp_dataset_v6 import JAISPDatasetV6
from jaisp_foundation_v7 import ALL_BANDS, JAISPFoundationV7


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    c = np.corrcoef(a.ravel(), b.ravel())
    return float(c[0, 1])


def psnr(truth: np.ndarray, pred: np.ndarray, max_val: float = None) -> float:
    mse = np.mean((truth - pred) ** 2)
    if mse < 1e-12:
        return float("inf")
    if max_val is None:
        max_val = np.nanpercentile(np.abs(truth), 99)
    max_val = max(float(max_val), 1e-6)
    return float(10.0 * np.log10(max_val ** 2 / mse))


def info_weighted_mask(info: np.ndarray, top_frac: float = 0.10) -> np.ndarray:
    thresh = np.nanpercentile(info, (1.0 - top_frac) * 100.0)
    return info >= thresh


def available_band_pool(sample: dict) -> dict:
    pool = {}
    pool.update(sample.get("rubin", {}))
    pool.update(sample.get("euclid", {}))
    return pool


def short_band_name(band: str) -> str:
    return band.split("_", 1)[1]


def load_model(checkpoint_path: str, device: torch.device) -> JAISPFoundationV7:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = JAISPFoundationV7(
        band_names=cfg.get("band_names", ALL_BANDS),
        stem_ch=cfg.get("stem_ch", 64),
        hidden_ch=cfg.get("hidden_ch", 256),
        blocks_per_stage=cfg.get("blocks_per_stage", 2),
        transformer_depth=cfg.get("transformer_depth", 4),
        transformer_heads=cfg.get("transformer_heads", 8),
        fused_pixel_scale_arcsec=cfg.get("fused_pixel_scale_arcsec", 0.8),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint from epoch {epoch} ({checkpoint_path})")
    return model


@torch.no_grad()
def eval_reconstruction(
    model: JAISPFoundationV7,
    dataset: JAISPDatasetV6,
    tile_indices: list,
    device: torch.device,
    n_tiles: int = None,
    top_frac: float = 0.10,
) -> dict:
    if n_tiles is not None:
        tile_indices = tile_indices[:n_tiles]

    per_band = defaultdict(lambda: defaultdict(list))

    for tile_idx in tqdm(tile_indices, desc="Reconstruction eval v7"):
        sample = dataset[tile_idx]
        pool = available_band_pool(sample)
        avail = [b for b in ALL_BANDS if b in pool]
        if len(avail) < 2:
            continue

        for target_band in avail:
            context_bands = [b for b in avail if b != target_band]
            if not context_bands:
                continue

            ctx_img = {b: pool[b]["image"].unsqueeze(0).to(device) for b in context_bands}
            ctx_rms = {b: pool[b]["rms"].unsqueeze(0).to(device) for b in context_bands}
            tgt_img = pool[target_band]["image"].unsqueeze(0).to(device)
            tgt_rms = pool[target_band]["rms"].unsqueeze(0).to(device)

            out = model(ctx_img, ctx_rms, target_band, tgt_img, tgt_rms)

            truth = out["target_norm"].squeeze().cpu().numpy()
            pred = out["pred"].squeeze().cpu().numpy()
            info = out["info_weights"].squeeze().cpu().numpy()
            mask = info_weighted_mask(info, top_frac=top_frac)
            if mask.sum() < 10:
                continue

            t_bright = truth[mask]
            p_bright = pred[mask]
            truth_std = float(np.std(t_bright))
            pred_std = float(np.std(p_bright))

            per_band[target_band]["r"].append(pearson_r(t_bright, p_bright))
            per_band[target_band]["mae"].append(float(np.mean(np.abs(p_bright - t_bright))))
            per_band[target_band]["psnr"].append(psnr(truth, pred))
            per_band[target_band]["std_ratio"].append(pred_std / max(truth_std, 1e-6))
            per_band[target_band]["r_baseline"].append(pearson_r(t_bright, np.zeros_like(t_bright)))
            per_band[target_band]["mae_baseline"].append(float(np.mean(np.abs(t_bright))))

    summary = {}
    for band in ALL_BANDS:
        if band not in per_band:
            continue
        d = per_band[band]
        r_arr = np.asarray(d["r"], dtype=np.float32)
        summary[band] = {
            "r_mean": float(np.nanmean(r_arr)),
            "r_std": float(np.nanstd(r_arr)),
            "mae_mean": float(np.nanmean(d["mae"])),
            "mae_baseline": float(np.nanmean(d["mae_baseline"])),
            "psnr_mean": float(np.nanmean(d["psnr"])),
            "std_ratio_mean": float(np.nanmean(d["std_ratio"])),
            "r_baseline_mean": float(np.nanmean(d["r_baseline"])),
            "improvement_r": float(np.nanmean(r_arr) - np.nanmean(d["r_baseline"])),
            "n_tiles": int(len(r_arr)),
        }

    return {"per_band": dict(per_band), "summary": summary}


def print_reconstruction_table(summary: dict) -> None:
    print("\n" + "=" * 92)
    print("V7 RECONSTRUCTION QUALITY (bright pixels, info-weighted top-10%)")
    print("=" * 92)
    print(
        f'{"Band":<14} {"r(model)":>10} {"r(base)":>9} {"Δr":>8} '
        f'{"MAE":>9} {"MAE base":>10} {"PSNR":>8} {"std(pred/truth)":>16} {"N":>5}'
    )
    print("-" * 92)
    for band in ALL_BANDS:
        if band not in summary:
            continue
        s = summary[band]
        print(
            f'{band:<14} {s["r_mean"]:>10.3f} {s["r_baseline_mean"]:>9.3f} {s["improvement_r"]:>+8.3f} '
            f'{s["mae_mean"]:>9.2f} {s["mae_baseline"]:>10.2f} {s["psnr_mean"]:>8.1f} '
            f'{s["std_ratio_mean"]:>16.3f} {s["n_tiles"]:>5}'
        )
    print("=" * 92)
    print("`std(pred/truth)` near 1.0 is healthy; far below 1.0 usually means washed-out predictions.\n")


@torch.no_grad()
def plot_band_grid(
    model: JAISPFoundationV7,
    dataset: JAISPDatasetV6,
    tile_idx: int,
    device: torch.device,
    save_path: str = None,
    top_frac: float = 0.10,
) -> None:
    sample = dataset[tile_idx]
    pool = available_band_pool(sample)
    avail = [b for b in ALL_BANDS if b in pool]
    if len(avail) < 2:
        print(f"Tile {sample['tile_id']} does not have enough bands for plotting.")
        return

    fig, axes = plt.subplots(4, len(avail), figsize=(3.2 * len(avail), 11), squeeze=False)
    row_labels = [
        "Truth (noise units)",
        "Prediction",
        "Residual",
        "Info weights",
    ]

    for col, target_band in enumerate(avail):
        context_bands = [b for b in avail if b != target_band]
        ctx_img = {b: pool[b]["image"].unsqueeze(0).to(device) for b in context_bands}
        ctx_rms = {b: pool[b]["rms"].unsqueeze(0).to(device) for b in context_bands}
        tgt_img = pool[target_band]["image"].unsqueeze(0).to(device)
        tgt_rms = pool[target_band]["rms"].unsqueeze(0).to(device)

        out = model(ctx_img, ctx_rms, target_band, tgt_img, tgt_rms)
        truth = out["target_norm"].squeeze().cpu().numpy()
        pred = out["pred"].squeeze().cpu().numpy()
        resid = pred - truth
        info = out["info_weights"].squeeze().cpu().numpy()
        mask = info_weighted_mask(info, top_frac=top_frac)

        corr = float("nan")
        mae = float("nan")
        std_ratio = float("nan")
        if mask.sum() > 10:
            t_bright = truth[mask]
            p_bright = pred[mask]
            corr = pearson_r(t_bright, p_bright)
            mae = float(np.mean(np.abs(p_bright - t_bright)))
            std_ratio = float(np.std(p_bright) / max(np.std(t_bright), 1e-6))

        lo = float(np.nanpercentile(truth, 1))
        hi = float(np.nanpercentile(truth, 99))
        lim = max(float(np.nanpercentile(np.abs(resid), 99)), 1e-3)
        info_hi = max(float(np.nanpercentile(info, 99.5)), float(info.max()), 1e-6)
        arrays = [truth, pred, resid, info]
        cmaps = ["gray", "gray", "RdBu_r", "inferno"]
        vranges = [(lo, hi), (lo, hi), (-lim, lim), (0.0, info_hi)]

        title = short_band_name(target_band)
        if np.isfinite(corr):
            title = f"{title}\nr={corr:.2f}  mae={mae:.2f}  std={std_ratio:.2f}"

        for row, (arr, cmap, vr) in enumerate(zip(arrays, cmaps, vranges)):
            ax = axes[row, col]
            ax.imshow(arr, cmap=cmap, vmin=vr[0], vmax=vr[1], origin="lower")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
            if row == 0:
                ax.set_title(title, fontsize=10, pad=4)

    fig.suptitle(
        f"v7 all-band reconstruction | tile={sample['tile_id']} | bands={len(avail)}",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.close(fig)


def choose_plot_tiles(dataset: JAISPDatasetV6, n_plot_tiles: int) -> list:
    scored = []
    for tile_idx in range(len(dataset)):
        count = len(available_band_pool(dataset[tile_idx]))
        if count >= 2:
            scored.append((count, tile_idx))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [tile_idx for _, tile_idx in scored[:n_plot_tiles]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JAISP Foundation v7")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rubin_dir", default="../data/rubin_tiles_ecdfs")
    parser.add_argument("--euclid_dir", default="../data/euclid_tiles_ecdfs")
    parser.add_argument("--n_eval_tiles", type=int, default=None, help="Cap number of eval tiles (default: all)")
    parser.add_argument("--n_plot_tiles", type=int, default=2, help="Number of diagnostic grids to save")
    parser.add_argument("--top_frac", type=float, default=0.10, help="Bright-pixel top fraction from info weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default=None, help="Directory to save figures and metric files")
    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir) if args.save_dir else Path(args.checkpoint).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)

    print("Loading dataset...")
    dataset = JAISPDatasetV6(
        rubin_dir=args.rubin_dir,
        euclid_dir=args.euclid_dir,
        augment=False,
        load_euclid=True,
    )
    all_indices = list(range(len(dataset)))

    recon = eval_reconstruction(
        model,
        dataset,
        all_indices,
        device,
        n_tiles=args.n_eval_tiles,
        top_frac=args.top_frac,
    )
    print_reconstruction_table(recon["summary"])

    np.savez(
        save_dir / "eval_reconstruction_v7.npz",
        summary=recon["summary"],
        per_band={band: dict(vals) for band, vals in recon["per_band"].items()},
    )
    with open(save_dir / "eval_reconstruction_v7_summary.json", "w", encoding="utf-8") as f:
        json.dump(recon["summary"], f, indent=2)

    plot_indices = choose_plot_tiles(dataset, args.n_plot_tiles)
    for rank, tile_idx in enumerate(plot_indices):
        tile_id = dataset[tile_idx]["tile_id"]
        save_path = save_dir / f"band_grid_v7_{rank:02d}_{tile_id}.png"
        plot_band_grid(
            model,
            dataset,
            tile_idx,
            device,
            save_path=str(save_path),
            top_frac=args.top_frac,
        )

    print(f"Saved metrics and plots in {save_dir}")


if __name__ == "__main__":
    main()
