"""Train the foundation-conditioned low-rank ePSF head.

The trainer consumes the star stamps produced by ``build_psf_v4_training_set.py``
and the original Rubin/Euclid tile files. For each tile it:

  1. Runs the frozen 10-band JAISP foundation encoder once.
  2. Samples the tile's held star stamps across all bands.
  3. Predicts a constrained oversampled ePSF per star from frozen features,
     tile position, and band id.
  4. Renders to native pixels at the recorded sub-pixel phase.
  5. Solves flux and a local constant background analytically, then optimizes
     residual chi/Charbonnier loss plus small calibration regularizers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from load_foundation import load_foundation
    from astrometry2.dataset import discover_tile_pairs, split_tile_pairs
    from astrometry2.latent_position_head import FrozenEncoder
    from astrometry2.train_latent_position import load_tile_data
    from psf.foundation_epsf_head import (
        ALL_BANDS,
        DEFAULT_CORE_SIGMA_MAS,
        RUBIN_BANDS,
        FoundationEPSFHead,
        analytic_epsf_bank,
        load_base_epsf_bank,
    )
except ImportError:
    from models.load_foundation import load_foundation
    from models.astrometry2.dataset import discover_tile_pairs, split_tile_pairs
    from models.astrometry2.latent_position_head import FrozenEncoder
    from models.astrometry2.train_latent_position import load_tile_data
    from models.psf.foundation_epsf_head import (
        ALL_BANDS,
        DEFAULT_CORE_SIGMA_MAS,
        RUBIN_BANDS,
        FoundationEPSFHead,
        analytic_epsf_bank,
        load_base_epsf_bank,
    )

try:
    import wandb
except ImportError:
    wandb = None


Record = Mapping[str, object]


def parse_band_float_overrides(spec: Optional[str], *, name: str) -> Dict[str, float]:
    """Parse ``band=value,band=value`` CLI overrides."""
    if spec is None or not str(spec).strip():
        return {}
    out: Dict[str, float] = {}
    for raw_part in str(spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"{name} override {part!r} must look like band=value")
        band, value = [x.strip() for x in part.split("=", 1)]
        if band not in ALL_BANDS:
            raise ValueError(f"{name} override has unknown band {band!r}; valid bands: {ALL_BANDS}")
        out[band] = float(value)
    return out


def load_psf_records_by_tile(
    train_dir: Path,
    band_names: Sequence[str] = ALL_BANDS,
    min_snr: float = 0.0,
) -> Dict[str, List[Record]]:
    """Load v4 per-band NPZ stamps and group records by tile id."""
    by_tile: Dict[str, List[Record]] = {}
    counts: Dict[str, int] = {}
    for band_idx, band in enumerate(band_names):
        path = train_dir / f"{band}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        required = ("stamps", "rms", "frac_xy", "pos_norm", "pos_pix", "snr", "tile_id")
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"{path} is missing keys: {missing}")

        stamps = np.asarray(data["stamps"], dtype=np.float32)
        rms = np.asarray(data["rms"], dtype=np.float32)
        frac_xy = np.asarray(data["frac_xy"], dtype=np.float32)
        pos_norm = np.asarray(data["pos_norm"], dtype=np.float32)
        pos_pix = np.asarray(data["pos_pix"], dtype=np.float32)
        tile_ids = np.asarray(data["tile_id"])
        snr = np.asarray(data["snr"], dtype=np.float32)
        keep = np.where(snr >= float(min_snr))[0]
        counts[band] = int(len(keep))
        for k in keep:
            tile_id = str(tile_ids[k])
            by_tile.setdefault(tile_id, []).append(
                {
                    "band": band,
                    "band_idx": int(band_idx),
                    "stamp": stamps[k],
                    "rms": rms[k],
                    "frac_xy": frac_xy[k],
                    "pos_norm": pos_norm[k],
                    "pos_pix": pos_pix[k],
                    "snr": float(snr[k]),
                }
            )
    if not by_tile:
        raise RuntimeError(f"No PSF records found under {train_dir}")
    print(
        f"Loaded {sum(counts.values())} PSF stamps from {len(by_tile)} tiles: "
        + "  ".join(f"{b}={n}" for b, n in counts.items())
    )
    return by_tile


def make_record_batch(
    records: Sequence[Record],
    device: torch.device,
    rng: np.random.RandomState,
    max_stars: int = 256,
) -> Dict[str, torch.Tensor]:
    """Stack a tile's records into tensors, optionally subsampling stars."""
    if len(records) == 0:
        raise ValueError("records must not be empty")
    if max_stars > 0 and len(records) > max_stars:
        idx = rng.choice(len(records), size=max_stars, replace=False)
        records = [records[int(i)] for i in idx]

    stamp = torch.from_numpy(np.stack([r["stamp"] for r in records]).astype(np.float32)).to(device)
    rms = torch.from_numpy(np.stack([r["rms"] for r in records]).astype(np.float32)).to(device)
    frac_xy = torch.from_numpy(np.stack([r["frac_xy"] for r in records]).astype(np.float32)).to(device)
    pos_norm = torch.from_numpy(np.stack([r["pos_norm"] for r in records]).astype(np.float32)).to(device)
    pos_pix = np.stack([r["pos_pix"] for r in records]).astype(np.float32)
    band_idx_np = np.array([r["band_idx"] for r in records], dtype=np.int64)

    source_positions_vis = pos_pix.copy()
    rubin_mask = band_idx_np < len(RUBIN_BANDS)
    source_positions_vis[rubin_mask] *= 2.0

    return {
        "stamp": stamp,
        "rms": rms,
        "frac_xy": frac_xy,
        "pos_norm": pos_norm,
        "pos_pix": torch.from_numpy(pos_pix).to(device),
        "source_positions_vis": torch.from_numpy(source_positions_vis).to(device),
        "band_idx": torch.from_numpy(band_idx_np).long().to(device),
        "snr": torch.tensor([float(r["snr"]) for r in records], dtype=torch.float32, device=device),
    }


def load_cached_bottleneck(
    cache_dir: Optional[Path],
    tile_id: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if cache_dir is None:
        return None
    for name in (f"{tile_id}_aug0.pt", f"{tile_id}.pt"):
        path = cache_dir / name
        if not path.exists():
            continue
        payload = torch.load(str(path), map_location=device, weights_only=False)
        feat = payload["features"] if isinstance(payload, dict) and "features" in payload else payload
        feat = feat.float().to(device)
        if feat.ndim == 3:
            feat = feat.unsqueeze(0)
        return feat
    return None


@torch.no_grad()
def encode_tile(
    frozen_encoder: FrozenEncoder,
    tile_id: str,
    rubin_path: str,
    euclid_path: str,
    device: torch.device,
    features_cache_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Load image arrays and return frozen encoder features for one tile."""
    context_images, context_rms, _vis_hw, _vis_wcs = load_tile_data(rubin_path, euclid_path, device)
    vis_img = context_images["euclid_VIS"]
    vis_rms = context_rms["euclid_VIS"]
    vis_stem = frozen_encoder.vis_stem(vis_img, vis_rms)
    cached = load_cached_bottleneck(features_cache_dir, tile_id, device)
    if cached is None:
        bottleneck = frozen_encoder.encoder(context_images, context_rms)["bottleneck"]
    else:
        bottleneck = cached
    return {
        "bottleneck": bottleneck,
        "vis_stem": vis_stem,
        "fused_hw": (bottleneck.shape[-2], bottleneck.shape[-1]),
        "vis_hw": (vis_img.shape[-2], vis_img.shape[-1]),
    }


def solve_flux_background(
    model_unit: torch.Tensor,
    data: torch.Tensor,
    rms: torch.Tensor,
    fit_background: bool = True,
    nonnegative_flux: bool = True,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted least-squares solve for flux and optional constant background."""
    pred = model_unit.squeeze(1)
    weight = 1.0 / rms.clamp(min=1e-6).pow(2)

    if not fit_background:
        num = (weight * pred * data).sum(dim=(-2, -1))
        den = (weight * pred * pred).sum(dim=(-2, -1)).clamp(min=eps)
        flux = num / den
        if nonnegative_flux:
            flux = flux.clamp_min(0.0)
        bg = torch.zeros_like(flux)
        return flux, bg

    a = (weight * pred * pred).sum(dim=(-2, -1))
    b = (weight * pred).sum(dim=(-2, -1))
    c = weight.sum(dim=(-2, -1))
    d = (weight * pred * data).sum(dim=(-2, -1))
    e = (weight * data).sum(dim=(-2, -1))
    det = (a * c - b * b).clamp(min=eps)
    flux = (d * c - b * e) / det
    bg = (a * e - b * d) / det
    if nonnegative_flux:
        flux = flux.clamp_min(0.0)
    return flux, bg


def psf_residual_loss(
    model_unit: torch.Tensor,
    data: torch.Tensor,
    rms: torch.Tensor,
    *,
    charbonnier_eps: float = 1e-3,
    fit_background: bool = True,
    nonnegative_flux: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    flux, bg = solve_flux_background(
        model_unit,
        data,
        rms,
        fit_background=fit_background,
        nonnegative_flux=nonnegative_flux,
    )
    pred = model_unit.squeeze(1)
    model = flux.view(-1, 1, 1) * pred + bg.view(-1, 1, 1)
    z = (model - data) / rms.clamp(min=1e-6)
    loss_pix = torch.sqrt(z * z + float(charbonnier_eps) ** 2) - float(charbonnier_eps)
    loss_per_star = loss_pix.mean(dim=(-2, -1))
    return loss_per_star.mean(), {
        "flux": flux,
        "background": bg,
        "chi_abs_median": z.abs().median(),
        "flux_median": flux.median(),
        "background_median": bg.median(),
    }


def _float_dict(row: Mapping[str, object]) -> Dict[str, float]:
    out = {}
    for key, value in row.items():
        if isinstance(value, torch.Tensor):
            out[key] = float(value.detach().cpu())
        elif isinstance(value, (int, float, np.number)):
            out[key] = float(value)
    return out


def aggregate(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row.keys() if isinstance(row[k], (int, float, np.number))})
    return {k: float(np.mean([float(row[k]) for row in rows if k in row])) for k in keys}


def _robust_limits(x: np.ndarray, q: float = 99.0) -> Tuple[float, float]:
    finite = np.asarray(x)[np.isfinite(x)]
    if finite.size == 0:
        return 0.0, 1.0
    vmax = float(np.percentile(finite, q))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(finite)))
    vmax = max(vmax, 1e-8)
    return 0.0, vmax


def _pearson_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nan_to_num(a.flatten(1), nan=0.0, posinf=0.0, neginf=0.0)
    b = torch.nan_to_num(b.flatten(1), nan=0.0, posinf=0.0, neginf=0.0)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    den = (a.pow(2).sum(dim=1).sqrt() * b.pow(2).sum(dim=1).sqrt()).clamp(min=1e-12)
    return (a * b).sum(dim=1) / den


def _make_visual_batch(
    records: Sequence[Record],
    device: torch.device,
    max_examples: int,
) -> Dict[str, torch.Tensor]:
    """Deterministic highest-SNR visual batch."""
    finite_records = [r for r in records if np.isfinite(float(r["snr"]))]
    ordered = sorted(finite_records, key=lambda r: float(r["snr"]), reverse=True)
    if max_examples > 0:
        ordered = ordered[:max_examples]
    if not ordered:
        ordered = list(records[:max_examples])
    rng = np.random.RandomState(0)
    return make_record_batch(ordered, device, rng, max_stars=0)


@torch.no_grad()
def _run_visual_tile(
    pair: Tuple[str, str, str],
    records: Sequence[Record],
    frozen_encoder: FrozenEncoder,
    head: FoundationEPSFHead,
    args: argparse.Namespace,
    device: torch.device,
    max_examples: int,
) -> Optional[Dict[str, torch.Tensor]]:
    if not records:
        return None
    tile_id, rubin_path, euclid_path = pair
    enc = None
    if not args.no_foundation_features:
        enc = encode_tile(
            frozen_encoder,
            tile_id,
            rubin_path,
            euclid_path,
            device,
            features_cache_dir=args.features_cache_dir,
        )
    batch = _make_visual_batch(records, device, max_examples=max_examples)
    feature_kwargs = {}
    if enc is not None:
        feature_kwargs = {
            "bottleneck": enc["bottleneck"],
            "vis_stem_features": enc["vis_stem"],
            "source_positions_vis": batch["source_positions_vis"],
            "fused_hw": enc["fused_hw"],
            "vis_hw": enc["vis_hw"],
        }
    pred = head(
        batch["pos_norm"],
        batch["band_idx"],
        return_dict=True,
        **feature_kwargs,
    )
    native = head.render_at_native(pred["epsf"], batch["frac_xy"], stamp_size=batch["stamp"].shape[-1])
    flux, bg = solve_flux_background(
        native,
        batch["stamp"],
        batch["rms"],
        fit_background=args.fit_background,
        nonnegative_flux=args.nonnegative_flux,
    )
    model = flux.view(-1, 1, 1) * native.squeeze(1) + bg.view(-1, 1, 1)
    resid = batch["stamp"] - model
    chi = resid / batch["rms"].clamp(min=1e-6)
    corr = _pearson_torch(model, batch["stamp"])
    return {
        "tile_id": tile_id,
        "batch": batch,
        "pred": pred,
        "native": native,
        "model": model,
        "resid": resid,
        "chi": chi,
        "corr": corr,
        "flux": flux,
        "background": bg,
    }


def _plot_fit_examples(vis: Mapping[str, object], epoch: int, max_examples: int):
    batch = vis["batch"]
    model = vis["model"].detach().cpu().numpy()
    resid = vis["resid"].detach().cpu().numpy()
    epsf = vis["pred"]["epsf"].detach().cpu().numpy()[:, 0]
    stamps = batch["stamp"].detach().cpu().numpy()
    rms = batch["rms"].detach().cpu().numpy()
    band_idx = batch["band_idx"].detach().cpu().numpy()
    snr = batch["snr"].detach().cpu().numpy()
    corr = vis["corr"].detach().cpu().numpy()
    n = min(max_examples, stamps.shape[0])
    fig, axes = plt.subplots(4, n, figsize=(2.2 * n, 8.2), squeeze=False)
    for col in range(n):
        obs = stamps[col]
        mod = model[col]
        res = resid[col]
        eps = epsf[col]
        _, vmax = _robust_limits(obs)
        rlim = max(float(np.nanpercentile(np.abs(res[np.isfinite(res)]), 99)) if np.isfinite(res).any() else 1e-8, 1e-8)
        chi_abs = np.nanmedian(np.abs(res / np.maximum(rms[col], 1e-6)))
        axes[0, col].imshow(obs, origin="lower", cmap="gray", vmin=0.0, vmax=vmax)
        axes[1, col].imshow(mod, origin="lower", cmap="gray", vmin=0.0, vmax=vmax)
        axes[2, col].imshow(res, origin="lower", cmap="RdBu_r", vmin=-rlim, vmax=rlim)
        axes[3, col].imshow(eps, origin="lower", cmap="inferno")
        axes[0, col].set_title(
            f"{ALL_BANDS[int(band_idx[col])]}\nSNR={snr[col]:.0f} r={corr[col]:.2f} |chi|={chi_abs:.1f}",
            fontsize=7,
        )
        for row in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    for row, label in enumerate(["Observed", "Model", "Residual", "ePSF"]):
        axes[row, 0].set_ylabel(label, fontsize=9)
    fig.suptitle(f"Held-out Gaia star fits - epoch {epoch} - tile {vis['tile_id']}", fontsize=10)
    fig.tight_layout()
    return fig


def _plot_epsf_gallery(vis_list: Sequence[Mapping[str, object]], head: FoundationEPSFHead, epoch: int):
    by_band: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}
    base = head.base_epsf().detach().cpu().numpy()
    for vis in vis_list:
        batch = vis["batch"]
        epsf = vis["pred"]["epsf"].detach().cpu().numpy()[:, 0]
        coeff = vis["pred"]["coeff"].detach().cpu().numpy()
        snr = batch["snr"].detach().cpu().numpy()
        band_idx = batch["band_idx"].detach().cpu().numpy()
        order = np.argsort(snr)[::-1]
        for idx in order:
            bi = int(band_idx[idx])
            if bi not in by_band:
                by_band[bi] = (epsf[idx], coeff[idx], float(snr[idx]))
    if not by_band:
        return None
    fig, axes = plt.subplots(2, len(ALL_BANDS), figsize=(1.6 * len(ALL_BANDS), 3.4), squeeze=False)
    for bi, band in enumerate(ALL_BANDS):
        axes[0, bi].imshow(base[bi], origin="lower", cmap="inferno")
        axes[0, bi].set_title(band, fontsize=7)
        if bi in by_band:
            eps, coeff, snr = by_band[bi]
            axes[1, bi].imshow(eps, origin="lower", cmap="inferno")
            axes[1, bi].set_title(f"SNR={snr:.0f} |c|={np.mean(np.abs(coeff)):.2f}", fontsize=7)
        else:
            axes[1, bi].set_facecolor("#222")
        for row in range(2):
            axes[row, bi].set_xticks([])
            axes[row, bi].set_yticks([])
    axes[0, 0].set_ylabel("Base", fontsize=9)
    axes[1, 0].set_ylabel("Head", fontsize=9)
    fig.suptitle(f"Per-band ePSF base vs foundation head - epoch {epoch}", fontsize=10)
    fig.tight_layout()
    return fig


def _plot_coeff_hist(vis_list: Sequence[Mapping[str, object]], epoch: int):
    coeffs = []
    band_ids = []
    for vis in vis_list:
        coeffs.append(vis["pred"]["coeff"].detach().cpu().numpy())
        band_ids.append(vis["batch"]["band_idx"].detach().cpu().numpy())
    if not coeffs:
        return None
    coeff = np.concatenate(coeffs, axis=0)
    band_idx = np.concatenate(band_ids, axis=0)
    rank = coeff.shape[1]
    fig, axes = plt.subplots(1, rank, figsize=(2.0 * rank, 2.4), squeeze=False)
    for k in range(rank):
        ax = axes[0, k]
        ax.hist(coeff[:, k], bins=24, color="steelblue", alpha=0.8)
        ax.axvline(0.0, color="black", lw=0.8)
        ax.set_title(f"c{k}", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Residual-basis coefficient histograms - epoch {epoch} - N={len(band_idx)}", fontsize=10)
    fig.tight_layout()
    return fig


@torch.no_grad()
def build_wandb_visuals(
    head: FoundationEPSFHead,
    val_pairs: Sequence[Tuple[str, str, str]],
    records_by_tile: Mapping[str, Sequence[Record]],
    frozen_encoder: FrozenEncoder,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> Dict[str, object]:
    if wandb is None or not val_pairs or args.wandb_image_every <= 0:
        return {}
    head.eval()
    vis_list = []
    for pair in val_pairs[: max(1, args.wandb_max_visual_tiles)]:
        try:
            vis = _run_visual_tile(
                pair,
                records_by_tile.get(pair[0], []),
                frozen_encoder,
                head,
                args,
                device,
                max_examples=args.wandb_n_examples,
            )
        except Exception as exc:
            print(f"  [warn] W&B visual skip {pair[0]}: {type(exc).__name__}: {exc}")
            continue
        if vis is not None:
            vis_list.append(vis)
    if not vis_list:
        return {}

    payload: Dict[str, object] = {}
    fig = _plot_fit_examples(vis_list[0], epoch, max_examples=args.wandb_n_examples)
    payload["vis/heldout_fits"] = wandb.Image(fig)
    plt.close(fig)

    fig = _plot_epsf_gallery(vis_list, head, epoch)
    if fig is not None:
        payload["vis/epsf_base_vs_head"] = wandb.Image(fig)
        plt.close(fig)

    fig = _plot_coeff_hist(vis_list, epoch)
    if fig is not None:
        payload["vis/coeff_hist"] = wandb.Image(fig)
        plt.close(fig)

    corrs = torch.cat([v["corr"].detach().cpu() for v in vis_list]).numpy()
    chi_abs = torch.cat([v["chi"].abs().flatten(1).median(dim=1).values.detach().cpu() for v in vis_list]).numpy()
    payload["vis/median_pearson"] = float(np.nanmedian(corrs))
    payload["vis/median_abs_chi"] = float(np.nanmedian(chi_abs))
    return payload


def run_one_tile(
    pair: Tuple[str, str, str],
    records: Sequence[Record],
    frozen_encoder: FrozenEncoder,
    head: FoundationEPSFHead,
    optimizer: Optional[torch.optim.Optimizer],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.RandomState,
) -> Optional[Dict[str, float]]:
    tile_id, rubin_path, euclid_path = pair
    if not records:
        return None

    train = optimizer is not None
    head.train(mode=train)
    enc = None
    if not args.no_foundation_features:
        enc = encode_tile(
            frozen_encoder,
            tile_id,
            rubin_path,
            euclid_path,
            device,
            features_cache_dir=args.features_cache_dir,
        )
    batch = make_record_batch(records, device, rng, max_stars=args.max_stars_per_tile)
    stamp_size = int(batch["stamp"].shape[-1])

    with torch.set_grad_enabled(train):
        feature_kwargs = {}
        if enc is not None:
            feature_kwargs = {
                "bottleneck": enc["bottleneck"],
                "vis_stem_features": enc["vis_stem"],
                "source_positions_vis": batch["source_positions_vis"],
                "fused_hw": enc["fused_hw"],
                "vis_hw": enc["vis_hw"],
            }
        pred = head(
            batch["pos_norm"],
            batch["band_idx"],
            return_dict=True,
            **feature_kwargs,
        )
        native = head.render_at_native(pred["epsf"], batch["frac_xy"], stamp_size=stamp_size)
        data_loss, stats = psf_residual_loss(
            native,
            batch["stamp"],
            batch["rms"],
            charbonnier_eps=args.charbonnier_eps,
            fit_background=args.fit_background,
            nonnegative_flux=args.nonnegative_flux,
        )
        reg = head.regularization(
            pred,
            coeff_l2_weight=args.coeff_l2_weight,
            residual_l2_weight=args.residual_l2_weight,
            basis_l2_weight=args.basis_l2_weight,
            basis_tv_weight=args.basis_tv_weight,
            epsf_tv_weight=args.epsf_tv_weight,
        )
        loss = data_loss + reg["loss"]

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()

    row = {
        "loss": loss,
        "loss_data": data_loss,
        "loss_reg": reg["loss"],
        "chi_abs_median": stats["chi_abs_median"],
        "flux_median": stats["flux_median"],
        "background_median": stats["background_median"],
        "n_star": int(batch["stamp"].shape[0]),
    }
    for key, value in reg.items():
        if key != "loss":
            row[f"reg_{key}"] = value
    return _float_dict(row)


def save_checkpoint(
    path: Path,
    head: FoundationEPSFHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Mapping[str, object],
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "config": dict(config),
            "train_metrics": dict(train_metrics),
            "val_metrics": dict(val_metrics),
            "base_epsf": head.base_epsf().detach().cpu(),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    records_by_tile = load_psf_records_by_tile(args.train_dir, min_snr=args.min_snr)
    pairs = discover_tile_pairs(str(args.rubin_dir), str(args.euclid_dir))
    pairs = [p for p in pairs if p[0] in records_by_tile]
    if not pairs:
        raise RuntimeError("No discovered tile pairs have PSF training records.")
    train_pairs, val_pairs = split_tile_pairs(pairs, val_frac=args.val_frac, seed=args.seed)
    if args.max_train_tiles > 0:
        train_pairs = train_pairs[:args.max_train_tiles]
    if args.max_val_tiles > 0:
        val_pairs = val_pairs[:args.max_val_tiles]
    print(f"Train tiles={len(train_pairs)} val tiles={len(val_pairs)}")

    foundation_ckpt = torch.load(args.foundation_checkpoint, map_location="cpu", weights_only=False)
    fcfg = foundation_ckpt.get("config", {})
    fused_scale = float(fcfg.get("fused_pixel_scale_arcsec", args.fused_pixel_scale))
    hidden_ch = int(fcfg.get("hidden_ch", args.hidden_ch))
    stem_ch = int(fcfg.get("stem_ch", args.stem_ch))

    foundation = load_foundation(args.foundation_checkpoint, device=torch.device("cpu"), freeze=True)
    frozen_encoder = FrozenEncoder(foundation).to(device).eval()

    base_mode = str(args.base_epsf_mode).lower()
    if base_mode == "auto":
        base_mode = "checkpoint" if args.base_epsf_checkpoint else "gaussian"

    base_sigmas_mas = parse_band_float_overrides(args.base_sigma_mas, name="base-sigma-mas")
    base_epsf = None
    if base_mode == "checkpoint":
        if not args.base_epsf_checkpoint:
            raise ValueError("--base-epsf-mode checkpoint requires --base-epsf-checkpoint")
        base_epsf = load_base_epsf_bank(
            args.base_epsf_checkpoint,
            band_names=ALL_BANDS,
            psf_size=args.psf_size,
            oversampling=args.oversampling,
        )
        print(f"Loaded base ePSF bank: {args.base_epsf_checkpoint}")
    elif base_mode in ("gaussian", "moffat"):
        base_epsf = analytic_epsf_bank(
            band_names=ALL_BANDS,
            psf_size=args.psf_size,
            oversampling=args.oversampling,
            kind=base_mode,
            sigmas_mas=base_sigmas_mas or None,
            sigma_scale=args.base_sigma_scale,
            moffat_beta=args.base_moffat_beta,
        )
        used_sigmas = {
            band: float(base_sigmas_mas.get(band, DEFAULT_CORE_SIGMA_MAS[band])) * float(args.base_sigma_scale)
            for band in ALL_BANDS
        }
        print(
            f"Initialized analytic {base_mode} base ePSF "
            f"(sigma_scale={args.base_sigma_scale:g}, sigmas_mas={used_sigmas})"
        )
    else:
        raise ValueError(f"unknown --base-epsf-mode {args.base_epsf_mode!r}")

    head = FoundationEPSFHead(
        psf_size=args.psf_size,
        oversampling=args.oversampling,
        band_names=ALL_BANDS,
        basis_rank=args.basis_rank,
        hidden_ch=hidden_ch,
        stem_ch=stem_ch,
        bottleneck_out=args.bottleneck_out,
        stem_out=args.stem_out,
        bottleneck_window=args.bottleneck_window,
        stem_window=args.stem_window,
        fused_pixel_scale=fused_scale,
        vis_pixel_scale=args.vis_pixel_scale,
        band_embed_dim=args.band_embed_dim,
        mlp_hidden=args.mlp_hidden,
        pos_freqs=args.pos_freqs,
        coeff_scale=args.coeff_scale,
        delta_scale=args.delta_scale,
        basis_init=args.basis_init,
        base_epsf=base_epsf,
        train_base=args.train_base,
        use_foundation_features=not args.no_foundation_features,
    ).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    config = {
        "foundation_checkpoint": str(args.foundation_checkpoint),
        "base_epsf_mode": base_mode,
        "base_epsf_checkpoint": str(args.base_epsf_checkpoint) if args.base_epsf_checkpoint else None,
        "base_sigma_mas": {
            band: float(base_sigmas_mas.get(band, DEFAULT_CORE_SIGMA_MAS[band]))
            for band in ALL_BANDS
        },
        "base_sigma_scale": args.base_sigma_scale,
        "base_moffat_beta": args.base_moffat_beta,
        "psf_size": args.psf_size,
        "oversampling": args.oversampling,
        "band_names": ALL_BANDS,
        "basis_rank": args.basis_rank,
        "hidden_ch": hidden_ch,
        "stem_ch": stem_ch,
        "bottleneck_out": args.bottleneck_out,
        "stem_out": args.stem_out,
        "bottleneck_window": args.bottleneck_window,
        "stem_window": args.stem_window,
        "fused_pixel_scale": fused_scale,
        "vis_pixel_scale": args.vis_pixel_scale,
        "band_embed_dim": args.band_embed_dim,
        "mlp_hidden": args.mlp_hidden,
        "pos_freqs": args.pos_freqs,
        "coeff_scale": args.coeff_scale,
        "delta_scale": args.delta_scale,
        "basis_init": args.basis_init,
        "train_base": args.train_base,
        "use_foundation_features": not args.no_foundation_features,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = None
    if args.wandb_project:
        if wandb is None:
            print("[warn] wandb not installed; continuing without W&B logging.")
        else:
            run = wandb.init(project=args.wandb_project, name=args.wandb_name, config={**vars(args), **config})

    best_val = float("inf")
    rng = np.random.RandomState(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = list(rng.permutation(len(train_pairs)))
        train_rows = []
        for idx in order:
            pair = train_pairs[int(idx)]
            try:
                row = run_one_tile(
                    pair,
                    records_by_tile.get(pair[0], []),
                    frozen_encoder,
                    head,
                    optimizer,
                    args,
                    device,
                    rng,
                )
            except Exception as exc:
                print(f"train skip {pair[0]}: {type(exc).__name__}: {exc}")
                continue
            if row is not None:
                train_rows.append(row)
        if not train_rows:
            raise RuntimeError(
                "No training tiles completed successfully in this epoch. "
                "Check the skipped-tile messages above."
            )
        scheduler.step()

        val_rows = []
        with torch.no_grad():
            for pair in val_pairs:
                try:
                    row = run_one_tile(
                        pair,
                        records_by_tile.get(pair[0], []),
                        frozen_encoder,
                        head,
                        None,
                        args,
                        device,
                        rng,
                    )
                except Exception as exc:
                    print(f"val skip {pair[0]}: {type(exc).__name__}: {exc}")
                    continue
                if row is not None:
                    val_rows.append(row)

        train_metrics = aggregate(train_rows)
        val_metrics = aggregate(val_rows)
        train_loss = train_metrics.get("loss", float("nan"))
        val_loss = val_metrics.get("loss", train_loss)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train={train_loss:.5f} val={val_loss:.5f} "
            f"n_train_tiles={len(train_rows)} n_val_tiles={len(val_rows)}"
        )

        save_checkpoint(
            out_dir / "checkpoint_latest.pt",
            head,
            optimizer,
            epoch,
            config,
            train_metrics,
            val_metrics,
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                out_dir / "checkpoint_best.pt",
                head,
                optimizer,
                epoch,
                config,
                train_metrics,
                val_metrics,
            )
            print(f"  saved best: {out_dir / 'checkpoint_best.pt'}")

        if run is not None:
            payload = {f"train/{k}": v for k, v in train_metrics.items()}
            payload.update({f"val/{k}": v for k, v in val_metrics.items()})
            payload["epoch"] = epoch
            payload["lr"] = optimizer.param_groups[0]["lr"]
            if (
                args.wandb_image_every > 0
                and (epoch == 1 or epoch == args.epochs or epoch % args.wandb_image_every == 0)
            ):
                payload.update(
                    build_wandb_visuals(
                        head,
                        val_pairs,
                        records_by_tile,
                        frozen_encoder,
                        args,
                        device,
                        epoch,
                    )
                )
            wandb.log(payload)

    summary = {
        "best_val_loss": best_val,
        "epochs_run": args.epochs,
        "n_train_tiles": len(train_pairs),
        "n_val_tiles": len(val_pairs),
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Best val loss: {best_val:.5f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", required=True, type=Path)
    p.add_argument("--rubin-dir", required=True, type=Path)
    p.add_argument("--euclid-dir", required=True, type=Path)
    p.add_argument("--foundation-checkpoint", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--base-epsf-mode",
        choices=("auto", "gaussian", "moffat", "checkpoint"),
        default="auto",
        help="Base ePSF prior. auto uses checkpoint when provided, otherwise analytic Gaussian.",
    )
    p.add_argument("--base-epsf-checkpoint", type=Path, default=None)
    p.add_argument(
        "--base-sigma-mas",
        type=str,
        default="",
        help=(
            "Optional comma-separated analytic core sigma overrides in mas, "
            "e.g. rubin_r=390,euclid_VIS=120."
        ),
    )
    p.add_argument(
        "--base-sigma-scale",
        type=float,
        default=1.0,
        help="Multiplicative scale applied to analytic base widths.",
    )
    p.add_argument(
        "--base-moffat-beta",
        type=float,
        default=3.5,
        help="Moffat beta for --base-epsf-mode moffat.",
    )
    p.add_argument("--features-cache-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--min-snr", type=float, default=10.0)
    p.add_argument("--max-stars-per-tile", type=int, default=256)
    p.add_argument(
        "--max-train-tiles",
        type=int,
        default=0,
        help="Optional cap on train tiles after the deterministic split. 0 = all.",
    )
    p.add_argument(
        "--max-val-tiles",
        type=int,
        default=0,
        help="Optional cap on validation tiles after the deterministic split. 0 = all.",
    )

    p.add_argument("--psf-size", type=int, default=99)
    p.add_argument("--oversampling", type=int, default=5)
    p.add_argument("--basis-rank", type=int, default=8)
    p.add_argument("--hidden-ch", type=int, default=256)
    p.add_argument("--stem-ch", type=int, default=64)
    p.add_argument("--bottleneck-out", type=int, default=128)
    p.add_argument("--stem-out", type=int, default=64)
    p.add_argument("--bottleneck-window", type=int, default=11)
    p.add_argument("--stem-window", type=int, default=31)
    p.add_argument("--fused-pixel-scale", type=float, default=0.4)
    p.add_argument("--vis-pixel-scale", type=float, default=0.1)
    p.add_argument("--band-embed-dim", type=int, default=16)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--pos-freqs", type=int, default=6)
    p.add_argument("--coeff-scale", type=float, default=1.0)
    p.add_argument("--delta-scale", type=float, default=0.35)
    p.add_argument("--basis-init", type=float, default=1e-2)
    p.add_argument("--train-base", action="store_true")
    p.add_argument("--no-foundation-features", action="store_true")

    p.add_argument("--charbonnier-eps", type=float, default=1e-3)
    p.add_argument("--no-fit-background", action="store_false", dest="fit_background")
    p.set_defaults(fit_background=True)
    p.add_argument("--allow-negative-flux", action="store_false", dest="nonnegative_flux")
    p.set_defaults(nonnegative_flux=True)
    p.add_argument("--coeff-l2-weight", type=float, default=1e-4)
    p.add_argument("--residual-l2-weight", type=float, default=1e-5)
    p.add_argument("--basis-l2-weight", type=float, default=1e-6)
    p.add_argument("--basis-tv-weight", type=float, default=1e-5)
    p.add_argument("--epsf-tv-weight", type=float, default=0.0)

    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument(
        "--wandb-image-every",
        type=int,
        default=5,
        help="Log W&B diagnostic images every N epochs, plus first/last. 0 disables images.",
    )
    p.add_argument(
        "--wandb-n-examples",
        type=int,
        default=6,
        help="Number of held-out star examples shown in the W&B fit gallery.",
    )
    p.add_argument(
        "--wandb-max-visual-tiles",
        type=int,
        default=2,
        help="Maximum validation tiles to encode for W&B images.",
    )
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
