"""Fit a two-Gaussian NISP-S PSF kernel (defined on the 0.1"/px MER grid) to the stacked
empirical cross-dispersion profiles, forward-modelling the 0.3" detector-pixel integration.

Model: kernel(r) = (1-f2) G(sigma1) + f2 G(sigma2)   [sigmas in MER px = 0.1"]
Cross-dispersion profile seen on the detector = kernel integrated along the dispersion,
convolved with a 0.3" box (3 MER px) and sampled at the 0.3" grid, then normalised
like the data (sum over |y|<=2 px).
Writes experiments/grism/nisps_kernel.npz with per-lambda (sigma1, sigma2, f2).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
emp = np.load(REPO / "experiments/grism/nisps_psf_empirical.npz", allow_pickle=True)
lam, ygrid, prof_emp = emp["lam"], emp["ygrid"], emp["profile"]
DET_PX = 3.0   # MER px per detector px


def model_profile(params, y_det):
    s1, s2, f2 = params
    s1, s2 = abs(s1), abs(s2)
    # 1D (dispersion-integrated) Gaussian mixture in MER px, box-integrated over 3 MER px around each y
    yy = y_det * DET_PX
    from scipy.special import erf
    def boxint(s):
        a = (yy - DET_PX / 2) / (np.sqrt(2) * s); b = (yy + DET_PX / 2) / (np.sqrt(2) * s)
        return 0.5 * (erf(b) - erf(a))
    p = (1 - f2) * boxint(s1) + f2 * boxint(s2)
    # the empirical stack was built by linear interpolation of 1-px samples onto this grid:
    # that convolves the true pixel profile with a unit triangle (width 1 det px). Apply the same here.
    ov = int(round(1 / (ygrid[1] - ygrid[0])))
    tri = np.array([max(0.0, 1 - abs(k) / ov) for k in range(-ov, ov + 1)]); tri /= tri.sum()
    p = np.convolve(p, tri, mode="same")
    core = p[np.abs(y_det) <= 2].sum() / ov
    return p / core


rows = []
for b in range(len(lam)):
    data = prof_emp[b]
    ok = np.isfinite(data) & (np.abs(ygrid) <= 6)
    res = least_squares(lambda q: (model_profile(q, ygrid[ok]) - data[ok]) * (1 + 3 * (np.abs(ygrid[ok]) > 1.5)),
                        x0=[2.0, 6.0, 0.3], bounds=([0.5, 2.0, 0.0], [5.0, 25.0, 0.8]))
    s1, s2, f2 = res.x
    rms = np.sqrt(np.mean((model_profile(res.x, ygrid[ok]) - data[ok]) ** 2))
    rows.append((lam[b], s1, s2, f2, rms))
    print(f"lambda {lam[b]:6.0f}: sigma1 {s1:.2f} MER px ({0.1 * s1 * 2.355:.3f}\" FWHM), sigma2 {s2:.2f} ({0.1 * s2 * 2.355:.2f}\" FWHM), wing frac {f2:.3f}, rms {rms:.4f}")
rows = np.array(rows)
np.savez(REPO / "experiments/grism/nisps_kernel.npz", lam=rows[:, 0], sigma1=rows[:, 1], sigma2=rows[:, 2], f2=rows[:, 3], unit="MER px (0.1 arcsec)")
print("wrote experiments/grism/nisps_kernel.npz")
