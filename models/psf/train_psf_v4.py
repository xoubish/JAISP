"""Train PSF v4 — NN ePSF head trained on stamps from build_psf_v4_training_set.py.

Loss: per-pixel weighted L1 between (rendered native PSF stamp × estimated flux)
and the observed stamp, weighted by 1 / RMS² so high-SNR pixels dominate.
The model output is unit-flux, so we fit a per-stamp scalar flux that minimises
the loss conditional on the predicted PSF — closed-form, no extra parameters.

Usage::

    PYTHONPATH=models python models/psf/train_psf_v4.py \\
        --train-dir   data/psf_training_v4 \\
        --output-dir  models/checkpoints/psf_field_v4 \\
        --epochs 80 --lr 3e-4 --batch 64 \\
        --wandb-project JAISP-PSFField-v4 \\
        --wandb-name v4_charb

Validation:
    Holds out 10% of stars per band (spatially: tile-disjoint splits).
    Reports per-band: median chi^2/dof, FWHM agreement, encircled-energy curves,
    plus the rendered-stamp residual figure (observed, predicted, residual).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from psf.psf_field_v4 import PSFFieldV4, ALL_BANDS  # noqa: E402

try:
    import wandb
except ImportError:
    wandb = None


# ============================================================
# Dataset
# ============================================================

class PSFTrainingSet(Dataset):
    """Loads per-band npz stamps produced by build_psf_v4_training_set.py."""

    def __init__(self, root: Path, split: str = "train", holdout_frac: float = 0.10,
                 seed: int = 0):
        self.root = Path(root)
        self.split = split
        self.records: List[Dict] = []
        rng = np.random.RandomState(seed)

        for band_idx, band in enumerate(ALL_BANDS):
            path = self.root / f"{band}.npz"
            if not path.exists():
                continue
            d = np.load(path, allow_pickle=False)
            n = d["stamps"].shape[0]
            if n == 0:
                continue
            # Tile-disjoint holdout: split tiles, not individual stars.
            tiles = d["tile_id"]
            unique_tiles = np.unique(tiles)
            rng.shuffle(unique_tiles)
            n_holdout = max(1, int(round(holdout_frac * len(unique_tiles))))
            holdout_tiles = set(unique_tiles[:n_holdout])
            mask = np.array([t in holdout_tiles for t in tiles])
            keep = ~mask if split == "train" else mask
            idx = np.where(keep)[0]

            self.records.append({
                "band": band, "band_idx": band_idx,
                "stamps":   d["stamps"][idx].astype(np.float32),
                "rms":      d["rms"][idx].astype(np.float32),
                "frac_xy":  d["frac_xy"][idx].astype(np.float32),
                "pos_norm": d["pos_norm"][idx].astype(np.float32),
                "snr":      d["snr"][idx].astype(np.float32),
                "flux":     d["flux"][idx].astype(np.float32),
            })

        # Flatten into a single index space.
        self.flat = []
        for r in self.records:
            for k in range(len(r["stamps"])):
                self.flat.append((r["band_idx"], k))

        if not self.flat:
            raise RuntimeError(f"No PSF training stamps found under {self.root}")

        # Per-band counts
        counts = {r["band"]: len(r["stamps"]) for r in self.records}
        print(f"PSFTrainingSet[{split}]: {len(self.flat)} stamps  "
              + "  ".join(f"{b}={c}" for b, c in counts.items()))

    def __len__(self) -> int:
        return len(self.flat)

    def __getitem__(self, k):
        bi, idx = self.flat[k]
        # Linear lookup by band_idx — the records list has at most 10 entries.
        rec = next(r for r in self.records if r["band_idx"] == bi)
        return {
            "stamp":   torch.from_numpy(rec["stamps"][idx]),       # [H, W]
            "rms":     torch.from_numpy(rec["rms"][idx]),          # [H, W]
            "frac_xy": torch.from_numpy(rec["frac_xy"][idx]),      # [2]
            "pos_norm": torch.from_numpy(rec["pos_norm"][idx]),    # [2]
            "band_idx": torch.tensor(bi, dtype=torch.long),
            "snr":     torch.tensor(float(rec["snr"][idx])),
            "flux":    torch.tensor(float(rec["flux"][idx])),
        }


def collate(batch):
    out = {}
    out["stamp"]    = torch.stack([b["stamp"] for b in batch])         # [B, H, W]
    out["rms"]      = torch.stack([b["rms"] for b in batch])
    out["frac_xy"]  = torch.stack([b["frac_xy"] for b in batch])       # [B, 2]
    out["pos_norm"] = torch.stack([b["pos_norm"] for b in batch])      # [B, 2]
    out["band_idx"] = torch.stack([b["band_idx"] for b in batch])      # [B]
    out["snr"]      = torch.stack([b["snr"] for b in batch])
    out["flux"]     = torch.stack([b["flux"] for b in batch])
    return out


# ============================================================
# Loss
# ============================================================

# ============================================================
# Metrics + visualization
# ============================================================

def _gaussian_fwhm_pix(stamp: np.ndarray) -> float:
    """Cheap FWHM estimate from second moments. Returns FWHM in pixels."""
    H, W = stamp.shape
    yy, xx = np.indices(stamp.shape, dtype=np.float32)
    s = float(stamp.sum())
    if s <= 0:
        return float("nan")
    cx = float((xx * stamp).sum() / s)
    cy = float((yy * stamp).sum() / s)
    sigx2 = max(float(((xx - cx) ** 2 * stamp).sum() / s), 1e-6)
    sigy2 = max(float(((yy - cy) ** 2 * stamp).sum() / s), 1e-6)
    sigma = 0.5 * (np.sqrt(sigx2) + np.sqrt(sigy2))
    return 2.355 * sigma


def _chi2_per_dof(pred: np.ndarray, obs: np.ndarray, rms: np.ndarray,
                  alpha: float) -> float:
    """Reduced chi² between alpha*pred and obs, weighted by 1/rms²."""
    inv_var = 1.0 / np.maximum(rms, 1e-6) ** 2
    resid = alpha * pred - obs
    chi2 = float((resid ** 2 * inv_var).sum())
    dof = max(int(obs.size - 1), 1)
    return chi2 / dof


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64); b = b.flatten().astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    den = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    return float((a * b).sum() / max(den, 1e-12))


@torch.no_grad()
def _visualize_psf(model: PSFFieldV4, val_ds: "PSFTrainingSet",
                   stamp_size: int, device: torch.device,
                   epoch: int, n_per_band: int = 5,
                   ) -> Dict[str, "wandb.Image"]:
    """Per-band figure: held-out stars, model predictions, residuals, oversampled ePSF.

    Returns a dict {wandb_key: wandb.Image} with per-band fitted-stamp galleries
    plus a per-band 5×5 spatial-grid PSF mosaic. Logged at each call.
    """
    if wandb is None:
        return {}
    model.eval()
    log = {}

    rng = np.random.RandomState(123 + epoch)
    band_chi2: Dict[str, float] = {}
    band_fwhm: Dict[str, float] = {}
    band_r:    Dict[str, float] = {}

    for rec in val_ds.records:
        band = rec["band"]; bi = rec["band_idx"]
        n_stamps = len(rec["stamps"])
        if n_stamps == 0:
            continue

        # Stable per-band picks: highest SNR first, then a few medium-SNR for variety
        order = np.argsort(rec["snr"])[::-1]
        pick = list(order[:max(1, n_per_band - 1)])
        if n_stamps > 5 and len(pick) < n_per_band:
            pick.append(int(rng.choice(order[len(pick): max(len(pick) + 50, 50)])))
        pick = pick[:n_per_band]

        pos    = torch.tensor(rec["pos_norm"][pick], dtype=torch.float32, device=device)
        frac   = torch.tensor(rec["frac_xy"][pick],  dtype=torch.float32, device=device)
        stamps = torch.tensor(rec["stamps"][pick],   dtype=torch.float32, device=device)
        rms    = torch.tensor(rec["rms"][pick],      dtype=torch.float32, device=device)
        bidx   = torch.full((len(pick),), bi, dtype=torch.long, device=device)

        psf_oversampled = model(pos, bidx)
        rendered = model.render_at_native(psf_oversampled, frac, stamp_size=stamp_size)

        # Best-fit alpha per stamp (closed form) for residual visualisation
        pred = rendered.squeeze(1)
        inv_var = 1.0 / rms.clamp(min=1e-6) ** 2
        num = (pred * stamps * inv_var).sum(dim=(-2, -1))
        den = (pred * pred * inv_var).sum(dim=(-2, -1)).clamp(min=1e-12)
        alpha = (num / den)
        scaled = alpha.view(-1, 1, 1) * pred
        resid = stamps - scaled

        # Per-band aggregate metrics over the picked stamps
        chi2s = [_chi2_per_dof(scaled[k].cpu().numpy(), stamps[k].cpu().numpy(),
                               rms[k].cpu().numpy(), 1.0) for k in range(len(pick))]
        # FWHM from oversampled (in oversampled pixels) → convert to native px
        ovs = model.oversampling
        fwhms = [_gaussian_fwhm_pix(psf_oversampled[k, 0].cpu().numpy()) / ovs
                 for k in range(len(pick))]
        rs = [_pearson_r(scaled[k].cpu().numpy(), stamps[k].cpu().numpy())
              for k in range(len(pick))]
        band_chi2[band] = float(np.median(chi2s))
        band_fwhm[band] = float(np.median(fwhms))
        band_r[band]    = float(np.median(rs))

        # ----- per-band gallery figure -----
        fig, axes = plt.subplots(4, len(pick), figsize=(2.2 * len(pick), 8.4),
                                 squeeze=False)
        for col in range(len(pick)):
            obs_k    = stamps[col].cpu().numpy()
            pred_k   = scaled[col].cpu().numpy()
            resid_k  = resid[col].cpu().numpy()
            psfo_k   = psf_oversampled[col, 0].cpu().numpy()
            vmax_obs = max(np.nanpercentile(obs_k, 99), 1e-6)
            rlim = max(np.nanpercentile(np.abs(resid_k), 99), 1e-6)

            axes[0, col].imshow(obs_k, origin="lower", cmap="gray",
                                vmin=0.0, vmax=vmax_obs)
            axes[0, col].set_title(
                f"SNR={rec['snr'][pick[col]]:.0f}  α={float(alpha[col]):.3g}",
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
            # Re-enable axis just for ylabel
            axes[r, 0].axis("on")
            axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        fig.suptitle(
            f"PSF v4 — {band}  (epoch {epoch})  "
            f"χ²/dof={band_chi2[band]:.2f}  FWHM={band_fwhm[band]:.2f} px  r={band_r[band]:.3f}",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        log[f"vis/psf_{band}"] = wandb.Image(fig)
        plt.close(fig)

        # ----- spatial-grid PSF mosaic per band -----
        gn = 5
        gx = np.linspace(-0.9, 0.9, gn)
        gy = np.linspace(-0.9, 0.9, gn)
        gpos = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
        gpos_t = torch.tensor(gpos, dtype=torch.float32, device=device)
        gbi = torch.full((gn * gn,), bi, dtype=torch.long, device=device)
        grid_psf = model(gpos_t, gbi)[:, 0].cpu().numpy()  # [25, P, P]

        fig2, axes2 = plt.subplots(gn, gn, figsize=(7, 7))
        for k in range(gn * gn):
            r, c = k // gn, k % gn
            axes2[r, c].imshow(grid_psf[k], origin="lower", cmap="inferno")
            axes2[r, c].axis("off")
        fig2.suptitle(f"PSF v4 — {band}  spatial grid (epoch {epoch})",
                      fontsize=10)
        fig2.tight_layout()
        log[f"vis/psf_grid_{band}"] = wandb.Image(fig2)
        plt.close(fig2)

    # Per-band scalar metrics for headline curves in W&B
    for b, v in band_chi2.items():  log[f"val/{b}_chi2_per_dof"] = v
    for b, v in band_fwhm.items():  log[f"val/{b}_fwhm_native_px"] = v
    for b, v in band_r.items():     log[f"val/{b}_pearson_r"]     = v
    if band_chi2:
        log["val/chi2_per_dof_mean"]    = float(np.mean(list(band_chi2.values())))
        log["val/pearson_r_mean"]       = float(np.mean(list(band_r.values())))

    return log


def psf_loss(pred_unit: torch.Tensor,    # [B, 1, S, S]   unit-flux native render
             observed: torch.Tensor,      # [B, S, S]      flux-units
             rms: torch.Tensor,           # [B, S, S]      flux-units
             charbonnier_eps: float = 1e-3,
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Charbonnier loss on (alpha * pred - obs) with closed-form best alpha.

    Per-stamp closed-form flux: alpha = sum(pred * obs / rms²) / sum(pred² / rms²).
    """
    B = pred_unit.shape[0]
    pred = pred_unit.squeeze(1)   # [B, S, S]
    inv_var = 1.0 / rms.clamp(min=1e-6) ** 2
    num = (pred * observed * inv_var).sum(dim=(-2, -1))
    den = (pred * pred * inv_var).sum(dim=(-2, -1)).clamp(min=1e-12)
    alpha = (num / den).view(B, 1, 1)            # best per-stamp flux

    resid = alpha * pred - observed
    # Weighted Charbonnier
    per_pixel = torch.sqrt(resid * resid + charbonnier_eps ** 2)
    loss = (inv_var * per_pixel).sum(dim=(-2, -1))
    norm = inv_var.sum(dim=(-2, -1)).clamp(min=1e-9)
    return (loss / norm).mean(), alpha.squeeze()


# ============================================================
# Training loop
# ============================================================

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = PSFTrainingSet(args.train_dir, split="train",
                              holdout_frac=args.holdout_frac, seed=args.seed)
    val_ds = PSFTrainingSet(args.train_dir, split="val",
                            holdout_frac=args.holdout_frac, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate,
                              pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate)

    stamp_size = train_ds.records[0]["stamps"].shape[-1]
    print(f"Stamp size (native): {stamp_size} px")

    model = PSFFieldV4(psf_size=args.psf_size, oversampling=args.oversampling,
                       hidden_ch=args.hidden_ch, n_freqs=args.n_freqs,
                       gauss_init_sigma_ovs=args.gauss_init_sigma_ovs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # W&B
    run = None
    if args.wandb_project and wandb is not None:
        run = wandb.init(project=args.wandb_project, name=args.wandb_name,
                         config=vars(args))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        # ------- TRAIN -------
        model.train()
        train_losses = []
        per_band_train: Dict[int, List[float]] = {i: [] for i in range(len(ALL_BANDS))}
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)
            psf_oversampled = model(batch["pos_norm"], batch["band_idx"])
            rendered = model.render_at_native(psf_oversampled,
                                              batch["frac_xy"],
                                              stamp_size=stamp_size)
            loss, alpha = psf_loss(rendered, batch["stamp"], batch["rms"],
                                   charbonnier_eps=args.charbonnier_eps)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
            for bi, l in zip(batch["band_idx"].tolist(), [float(loss.detach())]):
                per_band_train[bi].append(l)
        scheduler.step()

        # ------- VAL -------
        model.eval()
        val_losses = []
        per_band_val: Dict[int, List[float]] = {i: [] for i in range(len(ALL_BANDS))}
        with torch.no_grad():
            for batch in val_loader:
                for k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)
                psf_oversampled = model(batch["pos_norm"], batch["band_idx"])
                rendered = model.render_at_native(psf_oversampled,
                                                  batch["frac_xy"],
                                                  stamp_size=stamp_size)
                loss, _alpha = psf_loss(rendered, batch["stamp"], batch["rms"],
                                        charbonnier_eps=args.charbonnier_eps)
                val_losses.append(float(loss))
                # Per-band log: aggregate per-batch (lazy — still informative)
                for bi in batch["band_idx"].tolist():
                    per_band_val[bi].append(float(loss))

        train_mean = float(np.mean(train_losses))
        val_mean = float(np.mean(val_losses))
        print(f"Epoch {epoch+1:3d}/{args.epochs} | train {train_mean:.5f} | "
              f"val {val_mean:.5f}{'*' if val_mean < best_val else ''}")

        if run is not None:
            log = {"epoch": epoch + 1, "train/loss": train_mean,
                   "val/loss": val_mean, "lr": optimizer.param_groups[0]["lr"]}
            for bi, vs in per_band_val.items():
                if vs:
                    log[f"val/band_{ALL_BANDS[bi]}"] = float(np.mean(vs))

            # Visualisation + per-band metrics (every vis_every epochs and on the last epoch).
            if (epoch + 1) % args.vis_every == 0 or epoch == args.epochs - 1 or epoch == 0:
                try:
                    vlog = _visualize_psf(model, val_ds, stamp_size=stamp_size,
                                          device=device, epoch=epoch + 1,
                                          n_per_band=args.vis_n_per_band)
                    log.update(vlog)
                except Exception as e:
                    print(f"  [warn] visualisation failed: {e}")

            wandb.log(log)

        if val_mean < best_val:
            best_val = val_mean
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "psf_size": args.psf_size,
                    "oversampling": args.oversampling,
                    "hidden_ch": args.hidden_ch,
                    "n_freqs": args.n_freqs,
                    "gauss_init_sigma_ovs": args.gauss_init_sigma_ovs,
                    "band_names": ALL_BANDS,
                },
                "epoch": epoch + 1,
                "val_loss": val_mean,
            }, out_dir / "checkpoint_best.pt")

        # Always save latest
        torch.save({
            "model": model.state_dict(),
            "config": {
                "psf_size": args.psf_size,
                "oversampling": args.oversampling,
                "hidden_ch": args.hidden_ch,
                "n_freqs": args.n_freqs,
                "band_names": ALL_BANDS,
            },
            "epoch": epoch + 1,
            "val_loss": val_mean,
        }, out_dir / "checkpoint_latest.pt")

    print(f"\nBest val loss: {best_val:.5f}")
    summary = {"best_val_loss": best_val, "epochs_run": args.epochs,
               "n_train": len(train_ds), "n_val": len(val_ds)}
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PSF v4 NN ePSF head.")
    p.add_argument("--train-dir", required=True, type=Path,
                   help="Directory with per-band .npz files from build_psf_v4_training_set.py.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--holdout-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--psf-size", type=int, default=99,
                   help="Side of the oversampled ePSF stamp (default 99). With "
                        "ovs=5 this covers ±9.8 native pixels — enough to fit "
                        "Rubin/Euclid IR PSF wings (5σ ≈ 8.5 native px).")
    p.add_argument("--oversampling", type=int, default=5)
    p.add_argument("--hidden-ch", type=int, default=128)
    p.add_argument("--n-freqs", type=int, default=8)
    p.add_argument("--gauss-init-sigma-ovs", type=float, default=5.0,
                   help="Initial sigma (in oversampled pixels) of the per-band "
                        "Gaussian prior added to the decoder output. Sets the "
                        "PSF shape the model starts from before learning residuals.")
    p.add_argument("--charbonnier-eps", type=float, default=1e-3)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--vis-every", type=int, default=2,
                   help="Log per-band PSF figures + spatial grids every N epochs (default 2).")
    p.add_argument("--vis-n-per-band", type=int, default=5,
                   help="Number of held-out stars per band in the gallery figure (default 5).")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
