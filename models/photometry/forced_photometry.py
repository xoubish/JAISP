"""
Vectorised matched-filter forced photometry (Tractor-like linear estimator).

For a source with known PSF p(x,y) the optimal linear flux estimator is:

    flux     = Σ(d·p/σ²) / Σ(p²/σ²)          (Cramér-Rao optimal)
    flux_err = 1 / sqrt(Σ(p²/σ²))
    chi2_dof = Σ((d - flux·p)² / σ²) / (S²-1)

All operations are fully vectorised over [N, B, S, S] tensors — no Python loops.

Usage
-----
    flux, flux_err, chi2_dof = matched_filter(stamps, psfs, var)
"""

from typing import Tuple

import torch


def matched_filter(
    stamps:  torch.Tensor,   # [N, B, S, S]  sky-subtracted
    psfs:    torch.Tensor,   # [N, B, S, S]  normalised (sum=1)
    var:     torch.Tensor,   # [N, B, S, S]  pixel variance (rms²), > 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimal linear flux estimation under a known PSF model.

    Parameters
    ----------
    stamps  : sky-subtracted data [N, B, S, S]
    psfs    : normalised PSF stamps [N, B, S, S]
    var     : per-pixel variance [N, B, S, S]

    Returns
    -------
    flux     : [N, B]  in the same units as `stamps`
    flux_err : [N, B]  1-sigma Cramér-Rao bound
    chi2_dof : [N, B]  reduced chi² of the PSF fit
    """
    var = var.clamp(min=1e-20)  # guard against zero/negative variance

    psf_ov    = psfs / var                                   # [N, B, S, S]
    numerator = (stamps * psf_ov).sum(dim=(-2, -1))          # [N, B]
    denominator = (psfs * psf_ov).sum(dim=(-2, -1))          # [N, B]

    denom_safe = denominator.clamp(min=1e-20)
    flux     = numerator / denom_safe                        # [N, B]
    flux_err = denom_safe.rsqrt()                            # [N, B]

    S = stamps.shape[-1]
    dof = max(1, S * S - 1)
    resid = stamps - flux.unsqueeze(-1).unsqueeze(-1) * psfs # [N, B, S, S]
    chi2_dof = (resid ** 2 / var).sum(dim=(-2, -1)) / dof   # [N, B]

    return flux, flux_err, chi2_dof


def snr(
    flux:     torch.Tensor,  # [N, B]
    flux_err: torch.Tensor,  # [N, B]
) -> torch.Tensor:
    """Signal-to-noise ratio [N, B]."""
    return flux / flux_err.clamp(min=1e-20)


def aperture_flux(
    stamps: torch.Tensor,   # [N, B, S, S]  sky-subtracted
    var:    torch.Tensor,   # [N, B, S, S]
    radius_px: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Circular aperture photometry (no PSF model, useful for bright sources
    or sanity checks).

    Returns
    -------
    flux     : [N, B]  sum within aperture
    flux_err : [N, B]  sqrt(Σ var) within aperture
    """
    N, B, S, _ = stamps.shape
    device = stamps.device
    half = (S - 1) / 2.0

    y, x = torch.meshgrid(
        torch.arange(S, dtype=torch.float32, device=device),
        torch.arange(S, dtype=torch.float32, device=device),
        indexing='ij',
    )
    mask = ((x - half) ** 2 + (y - half) ** 2) <= radius_px ** 2  # [S, S]
    mask4 = mask.view(1, 1, S, S).float()

    flux     = (stamps * mask4).sum(dim=(-2, -1))             # [N, B]
    flux_err = ((var * mask4).sum(dim=(-2, -1))).clamp(min=0).sqrt()

    return flux, flux_err
