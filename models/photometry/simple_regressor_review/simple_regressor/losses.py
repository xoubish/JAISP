"""Target transform and the uncertainty-aware supervised loss.

Matches the convention used by the existing JAISP photometry heads:
  target  z = (asinh(flux/F0) - mean) / std        (mean/std fit on train only)
  loss    Huber_delta( (pred_flux - MER_flux) / sigma_eff )
  sigma_eff^2 = MER_err^2 + a^2 + (b * |MER_flux|)^2
The (b*flux) and floor terms use the *reference* flux/error, never the prediction,
so the network cannot lower the loss by inflating a predicted uncertainty (there is
no predicted uncertainty in this baseline).
"""
import numpy as np
import torch


class TargetScaler:
    def __init__(self, f0=1.0, mean=0.0, std=1.0):
        self.f0 = float(f0)
        self.mean = float(mean)
        self.std = float(std)

    @classmethod
    def fit(cls, flux_ujy, f0=1.0):
        a = np.arcsinh(np.asarray(flux_ujy, dtype=np.float64) / f0)
        return cls(f0=f0, mean=float(a.mean()), std=float(a.std() + 1e-8))

    def to_z(self, flux):
        return (torch.asinh(flux / self.f0) - self.mean) / self.std

    def to_flux(self, z):
        return self.f0 * torch.sinh(z * self.std + self.mean)

    def state(self):
        return dict(f0=self.f0, mean=self.mean, std=self.std)

    @classmethod
    def from_state(cls, s):
        s = {k: v for k, v in s.items() if k != "kind"}
        return cls(**s)


def uncertainty_huber(pred_flux, true_flux, fluxerr, a=0.01, b=0.02, delta=1.0):
    sigma_eff = torch.sqrt(fluxerr ** 2 + a ** 2 + (b * torch.abs(true_flux)) ** 2)
    r = (pred_flux - true_flux) / sigma_eff
    return torch.nn.functional.huber_loss(r, torch.zeros_like(r), delta=delta)


MAG_PER_LN = 2.5 / np.log(10.0)   # 1.0857: d(mag) = 1.0857 d(ln F)


class LogScaler:
    """Target for the magnitude loss: z = (ln F - mean)/std. Predicted flux = exp(...)
    is always positive, so Delta-mag is always defined."""
    kind = "log"

    def __init__(self, mean=0.0, std=1.0):
        self.mean = float(mean); self.std = float(std)

    @classmethod
    def fit(cls, flux_ujy):
        a = np.log(np.asarray(flux_ujy, dtype=np.float64))
        return cls(mean=float(a.mean()), std=float(a.std() + 1e-8))

    def to_lnflux(self, z):
        return z * self.std + self.mean

    def to_flux(self, z):
        return torch.exp(self.to_lnflux(z))

    def state(self):
        return dict(kind="log", mean=self.mean, std=self.std)

    @classmethod
    def from_state(cls, s):
        return cls(mean=s["mean"], std=s["std"])


def scaler_from_state(s):
    """Rebuild the target scaler stored in a checkpoint (old checkpoints = asinh)."""
    return LogScaler.from_state(s) if s.get("kind") == "log" else TargetScaler.from_state(s)


def mag_huber(z, scaler, true_flux, fluxerr, weighted=True, floor_mag=0.02,
              delta=1.0, delta_mag=0.2, shape="huber", power=2.0):
    """Huber loss on dmag = 2.5 log10(pred/true) = mag_true - mag_pred (ln space; the
    sign does not matter for the symmetric Huber).

    weighted=True : r = dmag / sqrt(sigma_mag^2 + floor^2) -- MER-error aware.
    weighted=False: r = dmag (mag)                         -- every object equal.
    shape="huber" : quadratic up to delta (delta_mag if unweighted), then LINEAR (robust).
    shape="power" : mean(|r|^power); power=2 = least squares, >2 = punish large offsets
                    even harder (aggressive; outliers dominate).
    """
    dmag = MAG_PER_LN * (scaler.to_lnflux(z) - torch.log(true_flux))
    if shape == "power":
        r = dmag
        if weighted:
            r = dmag / torch.sqrt((MAG_PER_LN * fluxerr / true_flux) ** 2 + floor_mag ** 2)
        return (r.abs() ** power).mean()
    if weighted:
        sig = torch.sqrt((MAG_PER_LN * fluxerr / true_flux) ** 2 + floor_mag ** 2)
        r = dmag / sig
        return torch.nn.functional.huber_loss(r, torch.zeros_like(r), delta=delta)
    return torch.nn.functional.huber_loss(dmag, torch.zeros_like(dmag), delta=delta_mag)
