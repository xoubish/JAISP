"""Fit a no-NN empirical ePSF bank from Gaia star stamps.

This is the deliberately simple baseline for the foundation ePSF work:

  * one positive, unit-flux oversampled ePSF per band
  * no foundation features
  * no spatial variation inside a JAISP tile
  * flux/background solved analytically per star
  * native-pixel rendering uses the same renderer as ``FoundationEPSFHead``

The output checkpoint contains ``base_epsf`` in the normal ALL_BANDS order and
can be used directly by ``train_foundation_epsf_head.py`` with
``--base-epsf-mode checkpoint --base-epsf-checkpoint <output>``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_THIS = Path(__file__).resolve()
for _path in (_THIS.parents[1], _THIS.parents[2]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:  # PYTHONPATH=models
    from psf.foundation_epsf_head import (
        ALL_BANDS,
        DEFAULT_CORE_SIGMA_MAS,
        FoundationEPSFHead,
        analytic_epsf_bank,
        load_base_epsf_bank,
    )
    from psf.train_foundation_epsf_head import (
        aggregate,
        parse_band_float_overrides,
        psf_residual_loss_with_centroid_nuisance,
    )
except ImportError:  # Package import from repo root
    from models.psf.foundation_epsf_head import (
        ALL_BANDS,
        DEFAULT_CORE_SIGMA_MAS,
        FoundationEPSFHead,
        analytic_epsf_bank,
        load_base_epsf_bank,
    )
    from models.psf.train_foundation_epsf_head import (
        aggregate,
        parse_band_float_overrides,
        psf_residual_loss_with_centroid_nuisance,
    )


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _epsf_tv(epsf: torch.Tensor) -> torch.Tensor:
    dy = epsf[..., 1:, :] - epsf[..., :-1, :]
    dx = epsf[..., :, 1:] - epsf[..., :, :-1]
    return dx.abs().mean() + dy.abs().mean()


def _edge_flux(epsf: torch.Tensor, edge: int = 3) -> torch.Tensor:
    if edge <= 0:
        return epsf.new_zeros(())
    edge = min(int(edge), epsf.shape[-1] // 2)
    mask = torch.zeros_like(epsf, dtype=torch.bool)
    mask[..., :edge, :] = True
    mask[..., -edge:, :] = True
    mask[..., :, :edge] = True
    mask[..., :, -edge:] = True
    return epsf[mask].sum() / epsf.shape[0]


def _float_row(row: Mapping[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in row.items():
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                out[key] = float(value.detach().cpu())
        elif isinstance(value, (int, float, np.number)):
            out[key] = float(value)
    return out


def _load_band_npz(
    train_dir: Path,
    band: str,
    *,
    min_snr: float,
    max_snr: float,
    max_stars: int,
    seed: int,
) -> Optional[Dict[str, np.ndarray]]:
    path = train_dir / f"{band}.npz"
    if not path.exists():
        print(f"{band:14s} missing {path}; using analytic/checkpoint initial base")
        return None
    data = np.load(path, allow_pickle=False)
    required = ("stamps", "rms", "frac_xy", "snr")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{path} is missing keys: {missing}")

    stamps = np.asarray(data["stamps"], dtype=np.float32)
    rms = np.asarray(data["rms"], dtype=np.float32)
    frac_xy = np.asarray(data["frac_xy"], dtype=np.float32)
    snr = np.asarray(data["snr"], dtype=np.float32)

    finite = (
        np.isfinite(stamps).reshape(stamps.shape[0], -1).all(axis=1)
        & np.isfinite(rms).reshape(rms.shape[0], -1).all(axis=1)
        & np.isfinite(frac_xy).all(axis=1)
        & np.isfinite(snr)
        & (snr >= float(min_snr))
    )
    if float(max_snr) > 0:
        finite &= snr <= float(max_snr)
    keep = np.where(finite)[0]
    if max_stars > 0 and len(keep) > int(max_stars):
        rng = np.random.RandomState(int(seed))
        keep = rng.choice(keep, size=int(max_stars), replace=False)
    if len(keep) == 0:
        print(f"{band:14s} no stamps after filters; using initial base")
        return None

    return {
        "stamps": stamps[keep],
        "rms": np.maximum(rms[keep], 1e-6),
        "frac_xy": frac_xy[keep],
        "snr": snr[keep],
    }


def _make_tensor_batch(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stamps = torch.from_numpy(arrays["stamps"][indices].astype(np.float32)).to(device)
    rms = torch.from_numpy(arrays["rms"][indices].astype(np.float32)).to(device)
    frac_xy = torch.from_numpy(arrays["frac_xy"][indices].astype(np.float32)).to(device)
    return stamps, rms, frac_xy


def _iter_batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), int(batch_size)):
        yield indices[start:start + int(batch_size)]


@torch.no_grad()
def _render_sum_stats(head: FoundationEPSFHead, device: torch.device) -> Dict[str, float]:
    vals = torch.linspace(-0.45, 0.45, 7, device=device)
    yy, xx = torch.meshgrid(vals, vals, indexing="ij")
    frac = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    epsf = head.base_epsf().unsqueeze(1).expand(frac.shape[0], -1, -1, -1)
    native = head.render_at_native(epsf, frac, stamp_size=32)
    sums = native.flatten(1).sum(dim=1)
    return {
        "render_sum_min": float(sums.min().cpu()),
        "render_sum_median": float(sums.median().cpu()),
        "render_sum_max": float(sums.max().cpu()),
    }


@torch.no_grad()
def _evaluate(
    head: FoundationEPSFHead,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    rows = []
    head.eval()
    for batch_idx in _iter_batches(indices, args.batch_size):
        stamp, rms, frac_xy = _make_tensor_batch(arrays, batch_idx, device)
        epsf = head.base_epsf().unsqueeze(1).expand(stamp.shape[0], -1, -1, -1)
        loss, stats, _native = psf_residual_loss_with_centroid_nuisance(
            head,
            epsf,
            frac_xy,
            stamp,
            rms,
            stamp_size=stamp.shape[-1],
            centroid_fit_max_px=args.centroid_fit_max_px,
            centroid_fit_steps=args.centroid_fit_steps,
            loss_snr_cap=args.loss_snr_cap,
            loss_radius_px=args.loss_radius_px,
            loss_taper_px=args.loss_taper_px,
            core_loss_radius_px=args.core_loss_radius_px,
            core_loss_weight=args.core_loss_weight,
            charbonnier_eps=args.charbonnier_eps,
            fit_background=not args.no_fit_background,
            nonnegative_flux=not args.allow_negative_flux,
        )
        row = {"loss": loss}
        for key, value in stats.items():
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                row[key] = value
        rows.append(_float_row(row))
    out = aggregate(rows)
    out.update(_render_sum_stats(head, device))
    return out


def _fit_one_band(
    band: str,
    band_idx: int,
    init_bank: torch.Tensor,
    arrays: Optional[Mapping[str, np.ndarray]],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    init = init_bank[band_idx:band_idx + 1].to(device)
    head = FoundationEPSFHead(
        psf_size=args.psf_size,
        oversampling=args.oversampling,
        band_names=[band],
        basis_rank=1,
        base_epsf=init,
        train_base=True,
        use_foundation_features=False,
        delta_scale=0.0,
    ).to(device)
    for param in head.parameters():
        param.requires_grad_(False)
    head.base_logits.requires_grad_(True)

    if arrays is None:
        metrics = _render_sum_stats(head, device)
        return head.base_epsf().detach().cpu()[0], metrics

    n = int(arrays["stamps"].shape[0])
    rng = np.random.RandomState(int(args.seed) + band_idx * 1009)
    order = rng.permutation(n)
    n_val = max(1, int(round(float(args.val_frac) * n))) if n > 1 else 0
    val_idx = np.sort(order[:n_val]) if n_val > 0 else np.array([], dtype=np.int64)
    train_idx = np.sort(order[n_val:]) if n_val < n else order
    if len(train_idx) == 0:
        train_idx = order
    if len(val_idx) == 0:
        val_idx = train_idx

    opt = torch.optim.AdamW([head.base_logits], lr=args.lr, weight_decay=0.0)
    best_loss = float("inf")
    best_epsf = head.base_epsf().detach().cpu()[0]
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, int(args.epochs) + 1):
        head.train()
        shuffled = rng.permutation(train_idx)
        train_rows = []
        for batch_idx in _iter_batches(shuffled, args.batch_size):
            stamp, rms, frac_xy = _make_tensor_batch(arrays, batch_idx, device)
            epsf = head.base_epsf().unsqueeze(1).expand(stamp.shape[0], -1, -1, -1)
            loss_data, stats, _native = psf_residual_loss_with_centroid_nuisance(
                head,
                epsf,
                frac_xy,
                stamp,
                rms,
                stamp_size=stamp.shape[-1],
                centroid_fit_max_px=args.centroid_fit_max_px,
                centroid_fit_steps=args.centroid_fit_steps,
                loss_snr_cap=args.loss_snr_cap,
                loss_radius_px=args.loss_radius_px,
                loss_taper_px=args.loss_taper_px,
                core_loss_radius_px=args.core_loss_radius_px,
                core_loss_weight=args.core_loss_weight,
                charbonnier_eps=args.charbonnier_eps,
                fit_background=not args.no_fit_background,
                nonnegative_flux=not args.allow_negative_flux,
            )
            epsf_one = head.base_epsf()
            loss_reg = (
                float(args.epsf_tv_weight) * _epsf_tv(epsf_one)
                + float(args.edge_flux_weight) * _edge_flux(epsf_one, edge=args.edge_pixels)
            )
            loss = loss_data + loss_reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([head.base_logits], args.grad_clip)
            opt.step()

            row = {
                "loss": loss.detach(),
                "loss_data": loss_data.detach(),
                "loss_reg": loss_reg.detach(),
            }
            for key, value in stats.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    row[key] = value.detach()
            train_rows.append(_float_row(row))

        val_metrics = _evaluate(head, arrays, val_idx, args, device)
        val_loss = val_metrics.get("loss", float("inf"))
        if val_loss < best_loss:
            best_loss = val_loss
            best_epsf = head.base_epsf().detach().cpu()[0]
            best_metrics = dict(val_metrics)

        if epoch == 1 or epoch == int(args.epochs) or epoch % int(args.log_every) == 0:
            train_metrics = aggregate(train_rows)
            print(
                f"{band:14s} epoch {epoch:03d}/{args.epochs} "
                f"train={train_metrics.get('loss', float('nan')):.4f} "
                f"val={val_loss:.4f} "
                f"val_rawchi={val_metrics.get('chi_abs_median_raw', float('nan')):.3f} "
                f"val_corechi={val_metrics.get('chi_abs_core_median_raw', float('nan')):.3f} "
                f"render_sum={val_metrics.get('render_sum_median', float('nan')):.5f}"
            )

    best_metrics["best_val_loss"] = best_loss
    best_metrics["n_train"] = float(len(train_idx))
    best_metrics["n_val"] = float(len(val_idx))
    return best_epsf, best_metrics


def _initial_bank(args: argparse.Namespace) -> torch.Tensor:
    mode = str(args.base_epsf_mode).lower()
    if mode == "auto":
        mode = "checkpoint" if args.base_epsf_checkpoint else "gaussian"
    sigmas = parse_band_float_overrides(args.base_sigma_mas, name="base-sigma-mas")
    if mode == "checkpoint":
        if not args.base_epsf_checkpoint:
            raise ValueError("--base-epsf-mode checkpoint requires --base-epsf-checkpoint")
        return load_base_epsf_bank(
            args.base_epsf_checkpoint,
            band_names=ALL_BANDS,
            psf_size=args.psf_size,
            oversampling=args.oversampling,
        )
    if mode not in ("gaussian", "moffat"):
        raise ValueError(f"unknown --base-epsf-mode {args.base_epsf_mode!r}")
    return analytic_epsf_bank(
        band_names=ALL_BANDS,
        psf_size=args.psf_size,
        oversampling=args.oversampling,
        kind=mode,
        sigmas_mas=sigmas or None,
        sigma_scale=args.base_sigma_scale,
        moffat_beta=args.base_moffat_beta,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    init_bank = _initial_bank(args)
    fitted = init_bank.clone()
    metrics: Dict[str, Dict[str, float]] = {}

    bands = ALL_BANDS if not args.bands else [b.strip() for b in args.bands.split(",") if b.strip()]
    unknown = [band for band in bands if band not in ALL_BANDS]
    if unknown:
        raise ValueError(f"unknown bands: {unknown}; valid bands are {ALL_BANDS}")

    print(f"Fitting empirical ePSF bank on {device}; output={args.output}")
    for band in bands:
        band_idx = ALL_BANDS.index(band)
        arrays = _load_band_npz(
            args.train_dir,
            band,
            min_snr=args.min_snr,
            max_snr=args.max_snr,
            max_stars=args.max_stars_per_band,
            seed=args.seed + band_idx,
        )
        if arrays is not None:
            print(
                f"{band:14s} using {arrays['stamps'].shape[0]} stamps "
                f"(SNR med={np.median(arrays['snr']):.1f}, p90={np.percentile(arrays['snr'], 90):.1f})"
            )
        epsf, band_metrics = _fit_one_band(band, band_idx, init_bank, arrays, args, device)
        fitted[band_idx] = epsf
        metrics[band] = band_metrics

    payload = {
        "base_epsf": fitted.cpu(),
        "band_names": list(ALL_BANDS),
        "config": {key: _jsonable(value) for key, value in vars(args).items()},
        "metrics": metrics,
    }
    torch.save(payload, args.output)
    summary_path = args.output.with_suffix(".json")
    with summary_path.open("w") as f:
        json.dump(
            {
                "output": str(args.output),
                "band_names": list(ALL_BANDS),
                "metrics": metrics,
                "config": {key: _jsonable(value) for key, value in vars(args).items()},
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--bands", type=str, default="", help="Comma-separated subset of bands; default fits all.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--base-epsf-mode", choices=("auto", "gaussian", "moffat", "checkpoint"), default="auto")
    p.add_argument("--base-epsf-checkpoint", type=Path, default=None)
    p.add_argument("--base-sigma-mas", type=str, default="")
    p.add_argument("--base-sigma-scale", type=float, default=1.0)
    p.add_argument("--base-moffat-beta", type=float, default=3.5)

    p.add_argument("--psf-size", type=int, default=99)
    p.add_argument("--oversampling", type=int, default=5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--min-snr", type=float, default=20.0)
    p.add_argument("--max-snr", type=float, default=0.0)
    p.add_argument("--max-stars-per-band", type=int, default=0)

    p.add_argument("--charbonnier-eps", type=float, default=1e-3)
    p.add_argument("--loss-snr-cap", type=float, default=1000.0)
    p.add_argument("--loss-radius-px", type=float, default=15.0)
    p.add_argument("--loss-taper-px", type=float, default=2.0)
    p.add_argument("--core-loss-radius-px", type=float, default=3.0)
    p.add_argument("--core-loss-weight", type=float, default=3.0)
    p.add_argument("--centroid-fit-max-px", type=float, default=0.20)
    p.add_argument("--centroid-fit-steps", type=int, default=5)
    p.add_argument("--no-fit-background", action="store_true")
    p.add_argument("--allow-negative-flux", action="store_true")

    p.add_argument("--epsf-tv-weight", type=float, default=2e-4)
    p.add_argument("--edge-flux-weight", type=float, default=1e-3)
    p.add_argument("--edge-pixels", type=int, default=3)
    p.add_argument("--log-every", type=int, default=10)
    return p


if __name__ == "__main__":
    main()
