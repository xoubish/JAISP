"""Post-training evaluation for JAISP Foundation v7 mixed-resolution reconstruction.

Four checks:

  1. Reconstruction quality (leave-one-band-out)
     Pearson r, MAE, PSNR at bright pixels for every band.
     Includes std_ratio to catch washed-out predictions.

  2. Cross-instrument reconstruction
     Rubin-only context → predict each Euclid band (and vice versa).
     This is the key v7 check: does the model generalise across instruments?

  3. Spatial precision
     r at correct position vs r at pixel-shifted position.
     Tests whether encoder features preserve sub-pixel spatial layout.
     Run on VIS (0.1"/px, the hardest case) and one Rubin band.

  4. Output resolution check
     Confirms each band is predicted at its native resolution,
     not accidentally downsampled.

Usage:
  python eval_foundation_v7.py \\
      --checkpoint ./checkpoints/jaisp_v7_baseline/checkpoint_best.pt \\
      --rubin_dir  ../data/rubin_tiles_ecdfs \\
      --euclid_dir ../data/euclid_tiles_ecdfs

  # With W&B logging (resume training run):
  python eval_foundation_v7.py \\
      --checkpoint ./checkpoints/jaisp_v7_baseline/checkpoint_best.pt \\
      --rubin_dir  ../data/rubin_tiles_ecdfs \\
      --euclid_dir ../data/euclid_tiles_ecdfs \\
      --wandb_run_id <run-id-from-training>
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
from jaisp_foundation_v7 import ALL_BANDS, EUCLID_BANDS, RUBIN_BANDS, JAISPFoundationV7


# ============================================================
# Helpers
# ============================================================

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    c = np.corrcoef(a.ravel(), b.ravel())
    return float(c[0, 1])


def psnr(truth: np.ndarray, pred: np.ndarray) -> float:
    mse = np.mean((truth - pred) ** 2)
    if mse < 1e-12:
        return float("inf")
    max_val = max(float(np.nanpercentile(np.abs(truth), 99)), 1e-6)
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


def native_resolution(band: str) -> str:
    if band in RUBIN_BANDS:
        return "512×512 @ 0.2\"/px"
    # MER mosaics: all Euclid bands (VIS + NISP) are at 0.1"/px
    return "~1084×1084 @ 0.1\"/px"


# ============================================================
# Load model
# ============================================================

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


# ============================================================
# Check 1: Leave-one-band-out reconstruction quality
# ============================================================

@torch.no_grad()
def eval_reconstruction(
    model: JAISPFoundationV7,
    dataset: JAISPDatasetV6,
    tile_indices: list,
    device: torch.device,
    n_tiles: int = None,
    top_frac: float = 0.10,
) -> dict:
    """
    For every tile, leave each band out and predict it from the rest.
    Records Pearson r, MAE, PSNR, std_ratio, and predicted output shape.
    """
    if n_tiles is not None:
        tile_indices = tile_indices[:n_tiles]

    per_band = defaultdict(lambda: defaultdict(list))

    for tile_idx in tqdm(tile_indices, desc="[1] Reconstruction eval"):
        sample = dataset[tile_idx]
        pool = available_band_pool(sample)
        avail = [b for b in ALL_BANDS if b in pool]
        if len(avail) < 2:
            continue

        for target_band in avail:
            context_bands = [b for b in avail if b != target_band]
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

            per_band[target_band]["r"].append(pearson_r(t_bright, p_bright))
            per_band[target_band]["mae"].append(float(np.mean(np.abs(p_bright - t_bright))))
            per_band[target_band]["mae_baseline"].append(float(np.mean(np.abs(t_bright))))
            per_band[target_band]["psnr"].append(psnr(truth, pred))
            per_band[target_band]["std_ratio"].append(
                float(np.std(p_bright) / max(np.std(t_bright), 1e-6))
            )
            per_band[target_band]["r_baseline"].append(
                pearson_r(t_bright, np.zeros_like(t_bright))
            )
            # Record predicted output shape (should match native resolution)
            per_band[target_band]["pred_h"].append(int(pred.shape[0]))
            per_band[target_band]["pred_w"].append(int(pred.shape[1]))

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
            "pred_h_mean": float(np.mean(d["pred_h"])),
            "pred_w_mean": float(np.mean(d["pred_w"])),
            "n_tiles": int(len(r_arr)),
        }

    return {"per_band": {b: dict(v) for b, v in per_band.items()}, "summary": summary}


def print_reconstruction_table(summary: dict) -> None:
    print("\n" + "=" * 105)
    print("CHECK 1: RECONSTRUCTION QUALITY — leave-one-band-out (bright pixels, info-weighted top-10%)")
    print("=" * 105)
    print(
        f'{"Band":<14} {"r":>7} {"Δr":>+7} {"MAE":>7} {"MAE_b":>7} '
        f'{"PSNR":>7} {"std_r":>7} {"pred shape":>14} {"native":>22} {"N":>4}'
    )
    print("-" * 105)
    for band in ALL_BANDS:
        if band not in summary:
            continue
        s = summary[band]
        pred_shape = f'{s["pred_h_mean"]:.0f}×{s["pred_w_mean"]:.0f}'
        print(
            f'{band:<14} {s["r_mean"]:>7.3f} {s["improvement_r"]:>+7.3f} '
            f'{s["mae_mean"]:>7.2f} {s["mae_baseline"]:>7.2f} '
            f'{s["psnr_mean"]:>7.1f} {s["std_ratio_mean"]:>7.3f} '
            f'{pred_shape:>14} {native_resolution(band):>22} {s["n_tiles"]:>4}'
        )
    print("=" * 105)
    print(
        "std_ratio near 1.0 = healthy. Far below 1.0 = washed-out predictions.\n"
        "pred shape should match native resolution — if VIS shows 512×512 something is wrong.\n"
    )


# ============================================================
# Check 2: Cross-instrument reconstruction
# ============================================================

@torch.no_grad()
def eval_cross_instrument(
    model: JAISPFoundationV7,
    dataset: JAISPDatasetV6,
    tile_indices: list,
    device: torch.device,
    n_tiles: int = None,
    top_frac: float = 0.10,
) -> dict:
    """
    Two directions:
      rubin_to_euclid: context = all Rubin bands, target = each Euclid band
      euclid_to_rubin: context = all Euclid bands, target = each Rubin band

    Tiles without Euclid coverage are skipped.
    """
    if n_tiles is not None:
        tile_indices = tile_indices[:n_tiles]

    rubin_to_euclid = defaultdict(lambda: defaultdict(list))
    euclid_to_rubin = defaultdict(lambda: defaultdict(list))

    for tile_idx in tqdm(tile_indices, desc="[2] Cross-instrument eval"):
        sample = dataset[tile_idx]
        if not sample.get("has_euclid"):
            continue
        pool = available_band_pool(sample)
        rubin_avail = [b for b in RUBIN_BANDS if b in pool]
        euclid_avail = [b for b in EUCLID_BANDS if b in pool]
        if not rubin_avail or not euclid_avail:
            continue

        # Rubin → Euclid
        ctx_img = {b: pool[b]["image"].unsqueeze(0).to(device) for b in rubin_avail}
        ctx_rms = {b: pool[b]["rms"].unsqueeze(0).to(device) for b in rubin_avail}
        for target_band in euclid_avail:
            tgt_img = pool[target_band]["image"].unsqueeze(0).to(device)
            tgt_rms = pool[target_band]["rms"].unsqueeze(0).to(device)
            out = model(ctx_img, ctx_rms, target_band, tgt_img, tgt_rms)
            truth = out["target_norm"].squeeze().cpu().numpy()
            pred = out["pred"].squeeze().cpu().numpy()
            info = out["info_weights"].squeeze().cpu().numpy()
            mask = info_weighted_mask(info, top_frac=top_frac)
            if mask.sum() < 10:
                continue
            t_b, p_b = truth[mask], pred[mask]
            rubin_to_euclid[target_band]["r"].append(pearson_r(t_b, p_b))
            rubin_to_euclid[target_band]["mae"].append(float(np.mean(np.abs(p_b - t_b))))
            rubin_to_euclid[target_band]["std_ratio"].append(
                float(np.std(p_b) / max(np.std(t_b), 1e-6))
            )

        # Euclid → Rubin
        ctx_img = {b: pool[b]["image"].unsqueeze(0).to(device) for b in euclid_avail}
        ctx_rms = {b: pool[b]["rms"].unsqueeze(0).to(device) for b in euclid_avail}
        for target_band in rubin_avail:
            tgt_img = pool[target_band]["image"].unsqueeze(0).to(device)
            tgt_rms = pool[target_band]["rms"].unsqueeze(0).to(device)
            out = model(ctx_img, ctx_rms, target_band, tgt_img, tgt_rms)
            truth = out["target_norm"].squeeze().cpu().numpy()
            pred = out["pred"].squeeze().cpu().numpy()
            info = out["info_weights"].squeeze().cpu().numpy()
            mask = info_weighted_mask(info, top_frac=top_frac)
            if mask.sum() < 10:
                continue
            t_b, p_b = truth[mask], pred[mask]
            euclid_to_rubin[target_band]["r"].append(pearson_r(t_b, p_b))
            euclid_to_rubin[target_band]["mae"].append(float(np.mean(np.abs(p_b - t_b))))
            euclid_to_rubin[target_band]["std_ratio"].append(
                float(np.std(p_b) / max(np.std(t_b), 1e-6))
            )

    def _summarise(store):
        out = {}
        for band, d in store.items():
            r_arr = np.asarray(d["r"], dtype=np.float32)
            out[band] = {
                "r_mean": float(np.nanmean(r_arr)),
                "mae_mean": float(np.nanmean(d["mae"])),
                "std_ratio_mean": float(np.nanmean(d["std_ratio"])),
                "n_tiles": int(len(r_arr)),
            }
        return out

    return {
        "rubin_to_euclid": _summarise(rubin_to_euclid),
        "euclid_to_rubin": _summarise(euclid_to_rubin),
    }


def print_cross_instrument_table(cross: dict) -> None:
    print("=" * 70)
    print("CHECK 2: CROSS-INSTRUMENT RECONSTRUCTION")
    print("=" * 70)
    for direction, summary in [
        ("Rubin → Euclid (Rubin context, predict each Euclid band)", cross["rubin_to_euclid"]),
        ("Euclid → Rubin (Euclid context, predict each Rubin band)", cross["euclid_to_rubin"]),
    ]:
        print(f"\n  {direction}")
        print(f'  {"Band":<14} {"r":>7} {"MAE":>7} {"std_ratio":>10} {"N":>5}')
        print("  " + "-" * 42)
        band_list = EUCLID_BANDS if "Euclid" in direction.split("→")[1] else RUBIN_BANDS
        for band in band_list:
            if band not in summary:
                continue
            s = summary[band]
            print(
                f'  {band:<14} {s["r_mean"]:>7.3f} {s["mae_mean"]:>7.2f} '
                f'{s["std_ratio_mean"]:>10.3f} {s["n_tiles"]:>5}'
            )
    print("=" * 70)
    print(
        "Cross-instrument r should be lower than leave-one-out (harder task).\n"
        "But r > 0 means the model is transferring information across instruments.\n"
    )


# ============================================================
# Check 3: Spatial precision
# ============================================================

@torch.no_grad()
def eval_spatial_precision(
    model: JAISPFoundationV7,
    dataset: JAISPDatasetV6,
    tile_indices: list,
    device: torch.device,
    n_tiles: int = 20,
) -> dict:
    """
    Tests whether encoder features preserve spatial layout.

    Runs two sub-checks:
      - VIS band (0.1"/px): offsets in VIS pixels (2, 5, 10, 20px = 0.2", 0.5", 1", 2")
      - Rubin r-band (0.2"/px): offsets (4, 8, 16, 32px = 0.8", 1.6", 3.2", 6.4")

    For each: r_correct = Pearson r at correct position
              r_offset  = Pearson r after rolling prediction by N pixels

    r_correct >> r_offset means spatial information is preserved.
    r_correct ≈ r_offset means the model predicts a spatially uniform average.
    """
    tile_indices = list(tile_indices)[:n_tiles]

    vis_offsets_px = [2, 5, 10, 20]
    rubin_offsets_px = [4, 8, 16, 32]

    vis_results = defaultdict(list)
    rubin_results = defaultdict(list)

    for tile_idx in tqdm(tile_indices, desc="[3] Spatial precision eval"):
        sample = dataset[tile_idx]
        pool = available_band_pool(sample)
        avail = [b for b in ALL_BANDS if b in pool]
        if len(avail) < 2:
            continue

        # --- VIS spatial precision ---
        if "euclid_VIS" in pool and len([b for b in avail if b != "euclid_VIS"]) >= 1:
            context_bands = [b for b in avail if b != "euclid_VIS"]
            ctx_img = {b: pool[b]["image"].unsqueeze(0).to(device) for b in context_bands}
            ctx_rms = {b: pool[b]["rms"].unsqueeze(0).to(device) for b in context_bands}
            tgt_img = pool["euclid_VIS"]["image"].unsqueeze(0).to(device)
            tgt_rms = pool["euclid_VIS"]["rms"].unsqueeze(0).to(device)
            out = model(ctx_img, ctx_rms, "euclid_VIS", tgt_img, tgt_rms)
            truth = out["target_norm"].squeeze().cpu().numpy()
            pred = out["pred"].squeeze().cpu().numpy()
            info = out["info_weights"].squeeze().cpu().numpy()
            mask = info_weighted_mask(info, top_frac=0.10)
            if mask.sum() >= 10:
                vis_results["r_correct"].append(pearson_r(truth[mask], pred[mask]))
                H, W = truth.shape
                for dx in vis_offsets_px:
                    pred_shifted = np.roll(pred, dx, axis=1)
                    crop = slice(0, W - dx)
                    t_c, p_c, m_c = truth[:, crop], pred_shifted[:, crop], mask[:, crop]
                    if m_c.sum() >= 10:
                        vis_results[f"r_offset_{dx}px"].append(
                            pearson_r(t_c[m_c], p_c[m_c])
                        )

        # --- Rubin r-band spatial precision ---
        rubin_r = "rubin_r"
        if rubin_r in pool and len([b for b in avail if b != rubin_r]) >= 1:
            context_bands = [b for b in avail if b != rubin_r]
            ctx_img = {b: pool[b]["image"].unsqueeze(0).to(device) for b in context_bands}
            ctx_rms = {b: pool[b]["rms"].unsqueeze(0).to(device) for b in context_bands}
            tgt_img = pool[rubin_r]["image"].unsqueeze(0).to(device)
            tgt_rms = pool[rubin_r]["rms"].unsqueeze(0).to(device)
            out = model(ctx_img, ctx_rms, rubin_r, tgt_img, tgt_rms)
            truth = out["target_norm"].squeeze().cpu().numpy()
            pred = out["pred"].squeeze().cpu().numpy()
            info = out["info_weights"].squeeze().cpu().numpy()
            mask = info_weighted_mask(info, top_frac=0.10)
            if mask.sum() >= 10:
                rubin_results["r_correct"].append(pearson_r(truth[mask], pred[mask]))
                H, W = truth.shape
                for dx in rubin_offsets_px:
                    pred_shifted = np.roll(pred, dx, axis=1)
                    crop = slice(0, W - dx)
                    t_c, p_c, m_c = truth[:, crop], pred_shifted[:, crop], mask[:, crop]
                    if m_c.sum() >= 10:
                        rubin_results[f"r_offset_{dx}px"].append(
                            pearson_r(t_c[m_c], p_c[m_c])
                        )

    def _mean(d):
        return {k: float(np.nanmean(v)) for k, v in d.items()}

    return {"vis": _mean(vis_results), "rubin_r": _mean(rubin_results)}


def print_spatial_table(spatial: dict) -> None:
    print("=" * 60)
    print("CHECK 3: SPATIAL PRECISION")
    print("=" * 60)
    for stream, label, px_scale in [
        ("vis", "euclid_VIS (0.1\"/px)", 0.1),
        ("rubin_r", "rubin_r   (0.2\"/px)", 0.2),
    ]:
        d = spatial.get(stream, {})
        if not d:
            continue
        print(f"\n  {label}")
        print(f'  {"Condition":<28} {"Pearson r":>10}')
        print("  " + "-" * 40)
        r_correct = d.get("r_correct", float("nan"))
        print(f'  {"At correct position":<28} {r_correct:>10.3f}')
        for key in sorted(d):
            if key == "r_correct":
                continue
            px = int(key.split("_")[2].replace("px", ""))
            arcsec = px * px_scale
            label_str = f'Shifted {px}px ({arcsec:.2f}")'
            print(f'  {label_str:<28} {d[key]:>10.3f}')
    print("=" * 60)
    print(
        "r_correct >> r_offset = spatial layout preserved.\n"
        "r_correct ≈ r_offset = model predicts a spatially uniform average.\n"
    )


# ============================================================
# Check 4 (visual): Per-band reconstruction grid
# ============================================================

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

    n_cols = len(avail)
    fig, axes = plt.subplots(4, n_cols, figsize=(3.2 * n_cols, 11), squeeze=False)
    row_labels = ["Truth (noise units)", "Prediction", "Residual", "Info weights"]

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

        corr = mae = std_ratio = float("nan")
        if mask.sum() > 10:
            t_b, p_b = truth[mask], pred[mask]
            corr = pearson_r(t_b, p_b)
            mae = float(np.mean(np.abs(p_b - t_b)))
            std_ratio = float(np.std(p_b) / max(np.std(t_b), 1e-6))

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
        title += f"\n{pred.shape[0]}×{pred.shape[1]}"

        for row, (arr, cmap, vr) in enumerate(zip(arrays, cmaps, vranges)):
            ax = axes[row, col]
            ax.imshow(arr, cmap=cmap, vmin=vr[0], vmax=vr[1], origin="lower")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
            if row == 0:
                ax.set_title(title, fontsize=9, pad=4)

    fig.suptitle(
        f"v7 all-band reconstruction | tile={sample['tile_id']} | bands={len(avail)}",
        fontsize=12, y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.close(fig)


def choose_plot_tiles(dataset: JAISPDatasetV6, n_plot_tiles: int) -> list:
    scored = []
    for tile_idx in range(len(dataset)):
        pool = available_band_pool(dataset[tile_idx])
        count = len(pool)
        if count >= 2:
            scored.append((count, tile_idx))
    scored.sort(reverse=True)
    return [tile_idx for _, tile_idx in scored[:n_plot_tiles]]


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JAISP Foundation v7")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rubin_dir", default="../data/rubin_tiles_ecdfs")
    parser.add_argument("--euclid_dir", default="../data/euclid_tiles_ecdfs")
    parser.add_argument("--n_eval_tiles", type=int, default=None, help="Cap eval tiles (default: all)")
    parser.add_argument("--n_spatial_tiles", type=int, default=20, help="Tiles for spatial precision check")
    parser.add_argument("--n_plot_tiles", type=int, default=2, help="Tiles for band grid plots")
    parser.add_argument("--top_frac", type=float, default=0.10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--wandb_run_id", default=None, help="W&B run ID to log summary metrics to")
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

    # ---- Check 1: Leave-one-band-out reconstruction ------------------------
    print("\n[1/4] Reconstruction quality (leave-one-band-out)...")
    recon = eval_reconstruction(
        model, dataset, all_indices, device,
        n_tiles=args.n_eval_tiles, top_frac=args.top_frac,
    )
    print_reconstruction_table(recon["summary"])
    with open(save_dir / "eval_reconstruction_v7.json", "w") as f:
        json.dump(recon["summary"], f, indent=2)

    # ---- Check 2: Cross-instrument reconstruction --------------------------
    print("\n[2/4] Cross-instrument reconstruction (Rubin↔Euclid)...")
    cross = eval_cross_instrument(
        model, dataset, all_indices, device,
        n_tiles=args.n_eval_tiles, top_frac=args.top_frac,
    )
    print_cross_instrument_table(cross)
    with open(save_dir / "eval_cross_instrument_v7.json", "w") as f:
        json.dump(cross, f, indent=2)

    # ---- Check 3: Spatial precision ----------------------------------------
    print(f"\n[3/4] Spatial precision (VIS + rubin_r, {args.n_spatial_tiles} tiles)...")
    spatial = eval_spatial_precision(
        model, dataset, all_indices, device, n_tiles=args.n_spatial_tiles,
    )
    print_spatial_table(spatial)
    with open(save_dir / "eval_spatial_v7.json", "w") as f:
        json.dump(spatial, f, indent=2)

    # ---- Check 4: Band grid plots ------------------------------------------
    print(f"\n[4/4] Reconstruction grid plots ({args.n_plot_tiles} tiles)...")
    plot_indices = choose_plot_tiles(dataset, args.n_plot_tiles)
    for rank, tile_idx in enumerate(plot_indices):
        tile_id = dataset[tile_idx]["tile_id"]
        plot_band_grid(
            model, dataset, tile_idx, device,
            save_path=str(save_dir / f"band_grid_v7_{rank:02d}_{tile_id}.png"),
            top_frac=args.top_frac,
        )

    # ---- Optional: W&B logging ---------------------------------------------
    if args.wandb_run_id:
        try:
            import wandb
            wandb.init(id=args.wandb_run_id, resume="must")
            log = {}
            for band, s in recon["summary"].items():
                log[f"eval/{band}/r_mean"] = s["r_mean"]
                log[f"eval/{band}/improvement_r"] = s["improvement_r"]
                log[f"eval/{band}/std_ratio"] = s["std_ratio_mean"]
                log[f"eval/{band}/pred_h"] = s["pred_h_mean"]
                log[f"eval/{band}/pred_w"] = s["pred_w_mean"]
            for band, s in cross["rubin_to_euclid"].items():
                log[f"eval/rubin_to_euclid/{band}/r_mean"] = s["r_mean"]
            for band, s in cross["euclid_to_rubin"].items():
                log[f"eval/euclid_to_rubin/{band}/r_mean"] = s["r_mean"]
            log["eval/spatial/vis_r_correct"] = spatial["vis"].get("r_correct", float("nan"))
            log["eval/spatial/vis_r_offset_10px"] = spatial["vis"].get("r_offset_10px", float("nan"))
            log["eval/spatial/rubin_r_correct"] = spatial["rubin_r"].get("r_correct", float("nan"))
            log["eval/spatial/rubin_r_offset_16px"] = spatial["rubin_r"].get("r_offset_16px", float("nan"))
            wandb.log(log)
            wandb.finish()
            print("Metrics logged to W&B.")
        except Exception as e:
            print(f"W&B logging failed: {e}")

    # ---- Overall judgement -------------------------------------------------
    print("\n=== OVERALL JUDGEMENT ===")
    rubin_r_vals = [s["r_mean"] for b, s in recon["summary"].items() if b in RUBIN_BANDS]
    euclid_r_vals = [s["r_mean"] for b, s in recon["summary"].items() if b in EUCLID_BANDS]
    avg_r_rubin = float(np.nanmean(rubin_r_vals)) if rubin_r_vals else float("nan")
    avg_r_euclid = float(np.nanmean(euclid_r_vals)) if euclid_r_vals else float("nan")
    avg_dr = float(np.nanmean([s["improvement_r"] for s in recon["summary"].values()]))
    avg_std_ratio = float(np.nanmean([s["std_ratio_mean"] for s in recon["summary"].values()]))

    vis_r_correct = spatial["vis"].get("r_correct", float("nan"))
    vis_r_offset = spatial["vis"].get("r_offset_10px", float("nan"))
    vis_spatial_gap = vis_r_correct - vis_r_offset

    vis_pred_h = recon["summary"].get("euclid_VIS", {}).get("pred_h_mean", 0)

    cross_r2e = float(np.nanmean([s["r_mean"] for s in cross["rubin_to_euclid"].values()])) if cross["rubin_to_euclid"] else float("nan")

    print(f"  Rubin leave-one-out r:       {avg_r_rubin:.3f}   (target: >0.75)")
    print(f"  Euclid leave-one-out r:      {avg_r_euclid:.3f}   (target: >0.65)")
    print(f"  Mean improvement over zero:  {avg_dr:+.3f}  (target: >0.30)")
    print(f"  Mean std_ratio:              {avg_std_ratio:.3f}   (target: 0.7–1.1)")
    print(f"  VIS pred resolution h:       {vis_pred_h:.0f}   (target: ~1050)")
    print(f"  VIS spatial gap (10px):      {vis_spatial_gap:+.3f}  (target: >0.15)")
    print(f"  Rubin→Euclid cross-instr r:  {cross_r2e:.3f}   (target: >0.30)")

    ok_recon = avg_r_rubin > 0.75 and avg_r_euclid > 0.60
    ok_spatial = vis_spatial_gap > 0.15
    ok_resolution = vis_pred_h > 900
    ok_cross = cross_r2e > 0.30

    if ok_recon and ok_spatial and ok_resolution and ok_cross:
        print("\n  PASS — Ready to test v7 encoder on downstream astrometry/detection.")
    elif ok_recon and ok_resolution:
        print("\n  PARTIAL — Reconstruction OK, but cross-instrument or spatial precision needs work.")
        if not ok_spatial:
            print("    Spatial gap too small: encoder may not preserve sub-pixel layout.")
        if not ok_cross:
            print("    Low cross-instrument r: Rubin/Euclid streams not well-aligned yet.")
        print("    Consider training longer.")
    elif not ok_resolution:
        print("\n  FAIL (resolution) — VIS is not being decoded at native resolution.")
        print(f"    Got pred_h={vis_pred_h:.0f}, expected ~1050. Check TargetDecoder stage_sizes.")
    else:
        print("\n  FAIL — Model is not learning useful representations.")
        print("    Check: loss curve, LR warmup, whether Euclid tiles are loading correctly.")

    print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    main()
