"""Train an aperture-initialized supervised flux regressor without foundation features."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import load_config
from .data import load_cache, fit_input_scale, StampDataset, balanced_weights
from .losses import TargetScaler, LogScaler, uncertainty_huber, mag_huber
from .model import build_model
from .baseline import fit_baseline


def pick_device(name):
    try:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    except Exception:
        pass
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_epoch(model, loader, scaler, cfg, device, opt=None):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    for feat, img_lin, flux, ferr in loader:
        feat, flux, ferr = feat.to(device), flux.to(device), ferr.to(device)
        with torch.set_grad_enabled(train):
            z = model(feat, img_lin.to(device))
            if cfg.loss == "mag":
                loss = mag_huber(z, scaler, flux, ferr, weighted=cfg.mag_weighted,
                                 floor_mag=cfg.mag_sigma_floor, delta=cfg.huber_delta,
                                 delta_mag=cfg.mag_huber_delta,
                                 shape=cfg.mag_loss_shape, power=cfg.mag_loss_power)
            else:
                loss = uncertainty_huber(scaler.to_flux(z), flux, ferr,
                                     a=cfg.absolute_sigma_floor_ujy,
                                     b=cfg.fractional_sigma_floor, delta=cfg.huber_delta)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; inspect inputs, labels and learning rate")
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
                opt.step()
        tot += float(loss.detach()) * len(flux); n += len(flux)
    return tot / max(n, 1)


def _ckpt(model, cfg, input_scale, scaler, epoch):
    return dict(model=model.state_dict(), config=cfg.to_json(), input_scale=input_scale,
                target_scaler=scaler.state(), epoch=epoch)


def train(cfg):
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    torch.set_num_threads(max(1, cfg.torch_threads))
    if cfg.epochs < 1:
        raise ValueError("epochs must be positive")
    cache = load_cache(str(out / "stamps_cache.npz"))
    if cfg.model_version == 2:
        meta_path = out / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta.get("cache_version") != 2:
            raise ValueError("V2 requires a fresh mask-correct cache: rerun prepare")
        old = meta.get("config", {})
        for key in ("flux_column", "fluxerr_column", "stamp", "native_pixscale"):
            if key in old and old[key] != getattr(cfg, key):
                raise ValueError(f"Cache/config mismatch for {key}; rerun prepare")
    split = cache["split"].astype(str)
    tr, va = split == "train", split == "val"
    if tr.sum() == 0 or va.sum() == 0:
        raise RuntimeError(f"Empty split: train={tr.sum()} val={va.sum()}")

    input_scale = cfg.input_scale if cfg.input_scale > 0 else \
        fit_input_scale(cache["stamps"][tr][:, 0], cfg.bin_factor)
    if cfg.loss not in ("mag", "flux"):
        raise ValueError("cfg.loss must be 'mag' or 'flux'")
    scaler = (LogScaler.fit(cache["flux"][tr]) if cfg.loss == "mag"
              else TargetScaler.fit(cache["flux"][tr], f0=cfg.f0_ujy))
    print(f"[train] device={device} pool={cfg.pool} input_scale={input_scale:.5g} "
          f"target mean={scaler.mean:.3f} std={scaler.std:.3f}")

    ds_tr = StampDataset(cache, tr, input_scale, cfg.bin_factor, cfg.centre_sigma_frac, augment=cfg.augment, model_version=cfg.model_version)
    ds_va = StampDataset(cache, va, input_scale, cfg.bin_factor, cfg.centre_sigma_frac, augment=False, model_version=cfg.model_version)
    if cfg.balanced_sampling:
        w = balanced_weights(cache["mag"][tr])
        sampler = WeightedRandomSampler(torch.as_tensor(w), len(w), replacement=True)
        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = build_model(cfg).to(device)
    print(f"[train] model=FluxCNN(pool={cfg.pool}) params: {model.n_params():,}  train={tr.sum()} val={va.sum()}")
    if cfg.model_version == 2:
        calibration_ds = StampDataset(cache, tr, input_scale, cfg.bin_factor, cfg.centre_sigma_frac,
                                      augment=False, model_version=2)
        calibration = fit_baseline(model, DataLoader(calibration_ds, batch_size=cfg.batch_size), scaler, device)
        (out / "baseline.json").write_text(json.dumps(calibration, indent=2))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4, min_lr=1e-5)

    history, best_val, best_epoch, since = [], float("inf"), -1, 0
    t0 = time.time()
    # Include the unmodified aperture estimate as an eligible validation checkpoint.
    best_val = run_epoch(model, dl_va, scaler, cfg, device)
    torch.save(_ckpt(model, cfg, input_scale, scaler, -1), out / "best.pt")
    initial_val = best_val
    print(f"[train] initial validation loss {initial_val:.5g}")
    for epoch in range(cfg.epochs):
        tl = run_epoch(model, dl_tr, scaler, cfg, device, opt)
        vl = run_epoch(model, dl_va, scaler, cfg, device, None)
        sched.step(vl); lr = opt.param_groups[0]["lr"]
        history.append(dict(epoch=epoch, train_loss=tl, val_loss=vl, lr=lr))
        improved = vl < best_val - 1e-6
        if improved:
            best_val, best_epoch, since = vl, epoch, 0
            torch.save(_ckpt(model, cfg, input_scale, scaler, epoch), out / "best.pt")
        else:
            since += 1
        print(f"[train] epoch {epoch:3d}  train {tl:.4f}  val {vl:.4f}  lr {lr:.1e}{'  *' if improved else ''}")
        if since >= cfg.patience:
            print(f"[train] early stop at epoch {epoch} (best {best_epoch}, val {best_val:.4f})")
            break

    torch.save(_ckpt(model, cfg, input_scale, scaler, history[-1]["epoch"]), out / "last.pt")
    (out / "history.json").write_text(json.dumps(
        dict(history=history, best_epoch=best_epoch, best_val_loss=best_val,
             elapsed_sec=time.time() - t0, device=device, pool=cfg.pool, initial_val_loss=initial_val), indent=2))
    print(f"[train] done in {time.time()-t0:.1f}s; best val {best_val:.4f} @ epoch {best_epoch}")
    return str(out / "best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, output_dir=a.output_dir, pool=a.pool, epochs=a.epochs, device=a.device)
    train(cfg)


if __name__ == "__main__":
    main()
