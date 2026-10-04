"""Training-only robust calibration of native aperture sums to catalog flux."""
import numpy as np
import torch


@torch.no_grad()
def fit_baseline(model, loader, scaler, device):
    sums, fluxes = [], []
    for _, linear, flux, _ in loader:
        a, _ = model.aperture_features(linear.to(device))
        sums.append(a.cpu().numpy())
        fluxes.append(flux.numpy())
    a = np.concatenate(sums).astype(np.float64)
    y = np.concatenate(fluxes).astype(np.float64)
    scale = np.maximum(np.median(np.abs(a), axis=0), 1e-12)
    x = a / scale
    # Fractional residual fit, with robust reweighting; uses training labels only.
    design = x / y[:, None]
    weight = np.ones(len(y))
    for _ in range(12):
        root = np.sqrt(weight)
        beta = np.linalg.lstsq(design * root[:, None], root, rcond=None)[0]
        residual = design @ beta - 1
        spread = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-6)
        weight = np.minimum(1., 1.5 * spread / np.maximum(np.abs(residual), 1e-12))
    coefficients = beta / scale
    if not np.isfinite(coefficients).all():
        raise RuntimeError("Non-finite aperture calibration")
    model.coefficients.copy_(torch.as_tensor(coefficients, device=device))
    model.aperture_scale.copy_(torch.as_tensor(scale, device=device))
    model.target_mean.fill_(scaler.mean)
    model.target_std.fill_(scaler.std)
    model.flux_floor.fill_(max(float(np.median(y)) * 1e-6, 1e-12))
    return dict(coefficients=coefficients.tolist(), n_train=len(y),
                train_nonpositive_fraction=float(np.mean(a @ coefficients <= 0)))
