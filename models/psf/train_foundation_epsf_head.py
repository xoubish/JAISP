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
        RUBIN_BANDS,
        FoundationEPSFHead,
        load_base_epsf_bank,
    )
except ImportError:
    from models.load_foundation import load_foundation
    from models.astrometry2.dataset import discover_tile_pairs, split_tile_pairs
    from models.astrometry2.latent_position_head import FrozenEncoder
    from models.astrometry2.train_latent_position import load_tile_data
    from models.psf.foundation_epsf_head import (
        ALL_BANDS,
        RUBIN_BANDS,
        FoundationEPSFHead,
        load_base_epsf_bank,
    )

try:
    import wandb
except ImportError:
    wandb = None


Record = Mapping[str, object]


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
    print(f"Train tiles={len(train_pairs)} val tiles={len(val_pairs)}")

    foundation_ckpt = torch.load(args.foundation_checkpoint, map_location="cpu", weights_only=False)
    fcfg = foundation_ckpt.get("config", {})
    fused_scale = float(fcfg.get("fused_pixel_scale_arcsec", args.fused_pixel_scale))
    hidden_ch = int(fcfg.get("hidden_ch", args.hidden_ch))
    stem_ch = int(fcfg.get("stem_ch", args.stem_ch))

    foundation = load_foundation(args.foundation_checkpoint, device=torch.device("cpu"), freeze=True)
    frozen_encoder = FrozenEncoder(foundation).to(device).eval()

    base_epsf = None
    if args.base_epsf_checkpoint:
        base_epsf = load_base_epsf_bank(
            args.base_epsf_checkpoint,
            band_names=ALL_BANDS,
            psf_size=args.psf_size,
            oversampling=args.oversampling,
        )
        print(f"Loaded base ePSF bank: {args.base_epsf_checkpoint}")

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
        "base_epsf_checkpoint": str(args.base_epsf_checkpoint) if args.base_epsf_checkpoint else None,
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
    p.add_argument("--base-epsf-checkpoint", type=Path, default=None)
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
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
