"""
PSFNet: spatially-varying PSF model per band, fit from isolated stars.

Architecture
------------
  base_psf   : nn.Parameter [B, S, S]  — per-band Gaussian initialisation
  band_embed : nn.Embedding(B, E)       — learned band token
  mlp        : (2 + E) → 64 → 64 → S² — spatial residual on log-PSF
  output     : softplus(base + delta), then L1-normalised

Forward
-------
  psf_net(x_norm, y_norm, band_idx) → [N, S, S]
  x_norm, y_norm ∈ [0, 1] (fractional tile position)

PSF grid for fast inference
---------------------------
  psf_grid = psf_net.precompute_grid(grid_size=8)  # [B, G, G, S, S]
  psfs = PSFNet.get_psfs_for_sources(psf_grid, x_norm, y_norm)  # [N, B, S, S]
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BAND_ORDER = [
    'rubin_u', 'rubin_g', 'rubin_r', 'rubin_i', 'rubin_z', 'rubin_y',
    'euclid_VIS', 'euclid_Y', 'euclid_J', 'euclid_H',
]

# Approximate PSF FWHM in pixels for each band at native resolution
# (Rubin at 0.2"/px, Euclid at 0.1"/px)
_BAND_FWHM_PX = {
    'rubin_u':    3.5,
    'rubin_g':    3.2,
    'rubin_r':    3.0,
    'rubin_i':    2.8,
    'rubin_z':    2.7,
    'rubin_y':    2.7,
    'euclid_VIS': 1.8,
    'euclid_Y':   2.5,
    'euclid_J':   2.5,
    'euclid_H':   2.8,
}


def _gaussian_psf(stamp_size: int, fwhm_px: float) -> torch.Tensor:
    """Normalised 2D Gaussian PSF stamp."""
    sigma = fwhm_px / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    c = (stamp_size - 1) / 2.0
    y, x = torch.meshgrid(
        torch.arange(stamp_size, dtype=torch.float32),
        torch.arange(stamp_size, dtype=torch.float32),
        indexing='ij',
    )
    g = torch.exp(-0.5 * ((x - c) ** 2 + (y - c) ** 2) / sigma ** 2)
    return g / g.sum()


class PSFNet(nn.Module):
    """
    Spatially-varying PSF model, one prediction per (position, band).

    Parameters
    ----------
    n_bands      : number of bands (default 10 for Rubin+Euclid)
    stamp_size   : PSF stamp side length in pixels (default 21)
    hidden_dim   : MLP hidden width
    band_embed_dim: dimension of per-band embedding
    """

    def __init__(
        self,
        n_bands: int = 10,
        stamp_size: int = 21,
        hidden_dim: int = 64,
        band_embed_dim: int = 8,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.stamp_size = stamp_size
        s2 = stamp_size * stamp_size

        # Per-band Gaussian base PSF (in log space so softplus keeps it positive)
        base = torch.stack([
            _gaussian_psf(stamp_size, _BAND_FWHM_PX.get(b, 3.0))
            for b in BAND_ORDER[:n_bands]
        ])  # [B, S, S]
        # Store in log space; add small floor before log
        self.base_psf = nn.Parameter(torch.log(base.clamp(min=1e-9)))

        # Band embedding
        self.band_embed = nn.Embedding(n_bands, band_embed_dim)

        # Spatial MLP: (x, y) + band_embed → residual on log-PSF
        in_dim = 2 + band_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, s2),
        )
        # Zero-init last layer → network starts at pure Gaussian base
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_norm: torch.Tensor,   # [N] positions in [0, 1]
        y_norm: torch.Tensor,   # [N]
        band_idx: torch.Tensor, # [N] int64
    ) -> torch.Tensor:
        """Return normalised PSF stamps [N, S, S]."""
        embed = self.band_embed(band_idx)               # [N, E]
        feat = torch.cat([
            x_norm.unsqueeze(-1),
            y_norm.unsqueeze(-1),
            embed,
        ], dim=-1)                                      # [N, 2+E]
        delta = self.mlp(feat).view(-1, self.stamp_size, self.stamp_size)  # [N, S, S]
        base  = self.base_psf[band_idx]                 # [N, S, S]
        psf   = F.softplus(base + delta)
        return psf / psf.sum(dim=(-2, -1), keepdim=True)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def precompute_grid(
        self,
        grid_size: int = 8,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Evaluate PSF on a regular (grid_size × grid_size) spatial grid for
        all bands.

        Returns
        -------
        psf_grid : [B, G, G, S, S] float32 on `device`
        """
        if device is None:
            device = next(self.parameters()).device
        G = grid_size
        coords = torch.linspace(0.0, 1.0, G, device=device)
        gy, gx = torch.meshgrid(coords, coords, indexing='ij')  # [G, G]
        gy_flat = gy.reshape(-1)  # [G²]
        gx_flat = gx.reshape(-1)

        B = self.n_bands
        # Evaluate all bands at all grid points in one batched forward pass
        # Repeat each grid point for every band: [B*G², ...]
        x_rep = gx_flat.repeat(B)                    # [B*G²]
        y_rep = gy_flat.repeat(B)
        band_rep = torch.arange(B, device=device).repeat_interleave(G * G)

        psfs = self(x_rep, y_rep, band_rep)          # [B*G², S, S]
        return psfs.view(B, G, G, self.stamp_size, self.stamp_size)

    # ------------------------------------------------------------------
    @staticmethod
    def get_psfs_for_sources(
        psf_grid: torch.Tensor,          # [B, G, G, S, S]
        x_norm: torch.Tensor,            # [N] float32  ∈ [0,1]
        y_norm: torch.Tensor,            # [N] float32  ∈ [0,1]
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        """
        Bilinear interpolation of psf_grid at arbitrary source positions.

        Returns
        -------
        psfs : [N, B, S, S] float32
        """
        B, G, _, S, _ = psf_grid.shape
        N = x_norm.shape[0]
        device = psf_grid.device
        x_norm = x_norm.to(device)
        y_norm = y_norm.to(device)

        out = torch.empty(N, B, S, S, dtype=torch.float32, device=device)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            xn = x_norm[start:end]   # [n]
            yn = y_norm[start:end]   # [n]

            # Integer grid indices and fractional weights
            gx = xn * (G - 1)
            gy = yn * (G - 1)
            x0 = gx.long().clamp(0, G - 2)
            y0 = gy.long().clamp(0, G - 2)
            x1 = x0 + 1
            y1 = y0 + 1
            wx1 = (gx - x0.float()).view(-1, 1, 1, 1)  # [n, 1, 1, 1]
            wy1 = (gy - y0.float()).view(-1, 1, 1, 1)
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1

            # Gather four corners: psf_grid[b, y, x] → [n, B, S, S]
            p00 = psf_grid[:, y0, x0].permute(1, 0, 2, 3)  # [n, B, S, S]
            p01 = psf_grid[:, y0, x1].permute(1, 0, 2, 3)
            p10 = psf_grid[:, y1, x0].permute(1, 0, 2, 3)
            p11 = psf_grid[:, y1, x1].permute(1, 0, 2, 3)

            interp = wy0 * wx0 * p00 + wy0 * wx1 * p01 + wy1 * wx0 * p10 + wy1 * wx1 * p11
            # Re-normalise after interpolation (bilinear can break exact normalisation slightly)
            interp = interp / interp.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
            out[start:end] = interp

        return out

    # ------------------------------------------------------------------
    def training_loss(
        self,
        star_stamps: torch.Tensor,  # [N, B, S, S]
        star_rms:    torch.Tensor,  # [N, B, S, S]
        x_norm:      torch.Tensor,  # [N]
        y_norm:      torch.Tensor,  # [N]
    ) -> torch.Tensor:
        """
        Fit PSF to isolated star stamps.

        For each (source, band) the analytic optimal flux is:
            flux = Σ(stamp * psf / var) / Σ(psf² / var)
        Loss is the mean weighted residual chi².
        """
        N, B, S, _ = star_stamps.shape
        var = (star_rms ** 2).clamp(min=1e-12)

        losses = []
        for bi in range(B):
            band_idx = torch.full((N,), bi, dtype=torch.long, device=x_norm.device)
            psf = self(x_norm, y_norm, band_idx)            # [N, S, S]
            psf_ov = psf / var[:, bi]
            numer  = (star_stamps[:, bi] * psf_ov).sum((-2, -1))  # [N]
            denom  = (psf * psf_ov).sum((-2, -1))                  # [N]
            flux   = (numer / denom.clamp(min=1e-12)).detach()     # analytic, no grad through flux

            resid = star_stamps[:, bi] - flux.view(N, 1, 1) * psf
            losses.append((resid ** 2 / var[:, bi]).mean())

        return torch.stack(losses).mean()
