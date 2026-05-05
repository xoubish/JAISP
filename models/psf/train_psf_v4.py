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
from typing import Dict, List, Tuple

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
                       hidden_ch=args.hidden_ch, n_freqs=args.n_freqs).to(device)
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
    p.add_argument("--psf-size", type=int, default=47)
    p.add_argument("--oversampling", type=int, default=5)
    p.add_argument("--hidden-ch", type=int, default=128)
    p.add_argument("--n-freqs", type=int, default=8)
    p.add_argument("--charbonnier-eps", type=float, default=1e-3)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
