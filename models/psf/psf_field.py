"""
PSFField: continuous, chromatic, spatially-varying PSF model for 10 bands.

Design
------
One SIREN-based implicit representation of the PSF intensity as a continuous
function of angular position, tile position, band, and source SED.

    f(x_sub, y_sub, tile_pos, band, sed) -> intensity

All spatial coordinates are in ARCSEC (instrument-agnostic). When fitting data,
pixels are integrated over their finite response area via a K×K sub-grid
(Monte-Carlo quadrature). Rubin pixels integrate 0.2"×0.2" boxes, Euclid
0.1"×0.1" — the same PSFField handles both with correct physical sampling.

Joint inference
---------------
Per-star latent parameters — learnable alongside PSFField weights:
  - centroid  ∈ ℝ²  (shared across 10 bands: one sky position per star)
  - SED       ∈ ℝ¹⁰ (derived from analytic optimal fluxes each iteration)

Global learnable parameters inside this module:
  - SIREN weights, band embedding, SED encoder weights
  - DCR coefficients: 6 Rubin bands × 2 (dx, dy) — chromatic astrometric shift

Chromatic handling
------------------
Two distinct effects, factored cleanly:
  1. PSF *shape* depends on source SED → SIREN conditioned on sed embedding
  2. Centroid *position* has a color-dependent shift in Rubin bands only (DCR)

For Euclid (space-based) bands, DCR is identically zero.

Band order (10 total)
---------------------
    0..5 : rubin_u, rubin_g, rubin_r, rubin_i, rubin_z, rubin_y
    6..9 : euclid_VIS, euclid_Y, euclid_J, euclid_H
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Band metadata
# ---------------------------------------------------------------------------

BAND_ORDER = [
    'rubin_u', 'rubin_g', 'rubin_r', 'rubin_i', 'rubin_z', 'rubin_y',
    'euclid_VIS', 'euclid_Y', 'euclid_J', 'euclid_H',
]
BAND_TO_IDX = {b: i for i, b in enumerate(BAND_ORDER)}
N_BANDS = 10
N_RUBIN = 6  # bands 0..5 are Rubin (ground-based → DCR applies)

# Pixel scales in arcsec / pixel
_BAND_PX_SCALE = {
    'rubin_u': 0.2, 'rubin_g': 0.2, 'rubin_r': 0.2,
    'rubin_i': 0.2, 'rubin_z': 0.2, 'rubin_y': 0.2,
    'euclid_VIS': 0.1, 'euclid_Y': 0.1, 'euclid_J': 0.1, 'euclid_H': 0.1,
}
BAND_PX_SCALE = torch.tensor(
    [_BAND_PX_SCALE[b] for b in BAND_ORDER], dtype=torch.float32
)  # [10]


# ---------------------------------------------------------------------------
# SIREN primitives (Sitzmann et al. 2020)
# ---------------------------------------------------------------------------

class SineLayer(nn.Module):
    """
    Linear + sin activation with SIREN-specific init.

    For the first layer, weights are initialised uniformly in [-1/in, 1/in].
    For subsequent layers, in [-sqrt(6/in)/w0, sqrt(6/in)/w0] so that the
    distribution of activations is unchanged after each sin.
    """

    def __init__(self, in_features: int, out_features: int,
                 w0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / w0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class SIREN(nn.Module):
    """Sequential SIREN: N SineLayers + final linear."""

    def __init__(self, in_dim: int, hidden: int, depth: int,
                 out_dim: int = 1, w0_first: float = 30.0,
                 w0_hidden: float = 1.0):
        super().__init__()
        layers = [SineLayer(in_dim, hidden, w0=w0_first, is_first=True)]
        for _ in range(depth - 1):
            layers.append(SineLayer(hidden, hidden, w0=w0_hidden))
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, out_dim)
        # Small init on final layer so PSF starts near-zero and grows during training
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden) / w0_hidden
            self.head.weight.uniform_(-bound, bound)
            self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


# ---------------------------------------------------------------------------
# Fourier features for tile-position encoding
# ---------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    """
    Random-frequency Fourier feature encoding (Tancik et al. 2020).

    Maps a coordinate in ℝ^D to ℝ^(2·F·D) via {sin, cos} of D·F random linear
    projections. Logarithmically-spaced frequencies capture both low-frequency
    (smooth) and high-frequency (detailed) spatial structure.
    """

    def __init__(self, in_dim: int = 2, num_freqs: int = 6,
                 max_freq: float = 8.0, min_freq: float = 1.0):
        super().__init__()
        # Logarithmically spaced frequencies → multi-scale
        freqs = torch.logspace(
            math.log10(min_freq), math.log10(max_freq), num_freqs
        )  # [F]
        self.register_buffer('freqs', freqs)
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.out_dim = 2 * num_freqs * in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]. Broadcast with freqs: [..., D, F]
        proj = x.unsqueeze(-1) * self.freqs * (2.0 * math.pi)
        proj = proj.flatten(-2)  # [..., D*F]
        return torch.cat([proj.sin(), proj.cos()], dim=-1)  # [..., 2*D*F]


# ---------------------------------------------------------------------------
# SED encoder — 10-band fluxes → low-dim embedding + scalar color
# ---------------------------------------------------------------------------

class SEDEncoder(nn.Module):
    """
    Encode a 10-band log-flux vector into:
      - `embed`  : ℝ^E for PSF-shape conditioning
      - `color`  : scalar g−i proxy for DCR term

    Input is assumed pre-normalised (log10 flux, star-mean-subtracted) so
    roughly zero-mean across stars. This keeps the MLP well-conditioned.
    """

    def __init__(self, embed_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(N_BANDS, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, embed_dim),
        )
        # Fixed g−i color projection: log10(F_g) − log10(F_i)
        # (Rubin g = idx 1, Rubin i = idx 3 in BAND_ORDER)
        w = torch.zeros(N_BANDS)
        w[BAND_TO_IDX['rubin_g']] = 1.0
        w[BAND_TO_IDX['rubin_i']] = -1.0
        self.register_buffer('color_weights', w)

    def forward(self, sed_vec: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sed_vec : [N, 10] — normalised per-band log-flux

        Returns
        -------
        embed : [N, E]
        color : [N]       — scalar g−i color
        """
        embed = self.mlp(sed_vec)
        color = (sed_vec * self.color_weights).sum(dim=-1)  # [N]
        return embed, color


# ---------------------------------------------------------------------------
# DCR — color-dependent centroid shift for Rubin bands only
# ---------------------------------------------------------------------------

class DCRTerm(nn.Module):
    """
    Residual Differential Chromatic Refraction in stacked Rubin mosaics.

    Models a per-band, color-proportional centroid shift:
        Δxy_b = dcr_coeff[b] · color        (Rubin bands, b=0..5)
        Δxy_b = 0                            (Euclid bands, b=6..9)

    `dcr_coeff` : [6, 2] learnable arcsec / unit-color.
    Captures the residual after averaging observations at various parallactic
    angles — a small but systematic effect for precision astrometry.
    """

    def __init__(self):
        super().__init__()
        self.dcr_coeff = nn.Parameter(torch.zeros(N_RUBIN, 2))

    def forward(self, band_idx: torch.Tensor, color: torch.Tensor
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        band_idx : [N] long
        color    : [N] float   (scalar g−i)

        Returns
        -------
        shift_arcsec : [N, 2] — add to nominal centroid in this band
        """
        N = band_idx.shape[0]
        shift = torch.zeros(N, 2, device=color.device, dtype=color.dtype)
        rubin_mask = band_idx < N_RUBIN
        if rubin_mask.any():
            coef = self.dcr_coeff[band_idx[rubin_mask]]           # [N_rubin, 2]
            shift[rubin_mask] = coef * color[rubin_mask].unsqueeze(-1)
        return shift


# ---------------------------------------------------------------------------
# Main PSFField
# ---------------------------------------------------------------------------

class PSFField(nn.Module):
    """
    Continuous, chromatic, spatially-varying PSF field for 10-band imaging.

    Call `render_stamps(...)` to produce pixel-integrated PSF stamps for a
    batch of (star, band) pairs.

    Key arguments
    -------------
    sed_embed_dim  : width of per-star SED embedding (default 8)
    tile_freqs     : number of Fourier frequencies for tile-position
                     encoding (default 6)
    siren_hidden   : SIREN hidden width (default 128)
    siren_depth    : number of SineLayers (default 5)
    """

    def __init__(
        self,
        sed_embed_dim: int = 8,
        band_embed_dim: int = 16,
        tile_freqs: int = 6,
        siren_hidden: int = 128,
        siren_depth: int = 5,
        w0_first: float = 30.0,
    ):
        super().__init__()
        self.sed_embed_dim = sed_embed_dim
        self.band_embed_dim = band_embed_dim

        self.tile_pos_enc = FourierFeatures(
            in_dim=2, num_freqs=tile_freqs, max_freq=8.0
        )
        self.band_embed   = nn.Embedding(N_BANDS, band_embed_dim)
        self.sed_encoder  = SEDEncoder(embed_dim=sed_embed_dim)
        self.dcr          = DCRTerm()

        # SIREN input: (xy_sub in arcsec) + tile ctx + band embed + sed embed
        in_dim = 2 + self.tile_pos_enc.out_dim + band_embed_dim + sed_embed_dim
        self.siren = SIREN(
            in_dim=in_dim,
            hidden=siren_hidden,
            depth=siren_depth,
            out_dim=1,
            w0_first=w0_first,
        )

        # Cache the per-band pixel scale (in arcsec) as a buffer
        self.register_buffer('band_px_scale', BAND_PX_SCALE.clone())

    # ------------------------------------------------------------------
    # Low-level: evaluate PSF at arbitrary (x_sub, y_sub) query points
    # ------------------------------------------------------------------
    def evaluate(
        self,
        xy_sub: torch.Tensor,        # [..., 2]  arcsec relative to centroid
        tile_pos: torch.Tensor,      # [..., 2]  ∈ [0, 1]²
        band_idx: torch.Tensor,      # [...]     long
        sed_vec: torch.Tensor,       # [..., 10]
    ) -> torch.Tensor:
        """
        Evaluate PSF intensity at query points. All inputs share a common
        leading shape (broadcastable).

        Returns
        -------
        intensity : [...] — non-negative intensity at each query point
        """
        tile_ctx           = self.tile_pos_enc(tile_pos)
        band_ctx           = self.band_embed(band_idx)
        sed_embed, _color  = self.sed_encoder(sed_vec)

        feat = torch.cat([xy_sub, tile_ctx, band_ctx, sed_embed], dim=-1)
        raw  = self.siren(feat).squeeze(-1)
        return F.softplus(raw)

    # ------------------------------------------------------------------
    # High-level: render a pixel-integrated stamp for a batch of stars
    # ------------------------------------------------------------------
    def render_stamps(
        self,
        centroids_arcsec: torch.Tensor,  # [N, 2]  offset of star within stamp
        tile_pos: torch.Tensor,          # [N, 2]  normalised
        band_idx: torch.Tensor,          # [N]     long — all same band OK too
        sed_vec: torch.Tensor,           # [N, 10]
        stamp_size: int,
        px_scale: float,
        sub_grid: int = 4,
        apply_dcr: bool = True,
    ) -> torch.Tensor:
        """
        Render a pixel-integrated PSF stamp per star, with proper angular-
        coordinate sampling and K×K sub-pixel quadrature.

        Parameters
        ----------
        centroids_arcsec : [N, 2] — (dx, dy) in arcsec of the star's true
                          centroid relative to the geometric stamp centre
        tile_pos         : [N, 2] ∈ [0, 1]
        band_idx         : [N]    — band index per star (may be identical)
        sed_vec          : [N, 10]
        stamp_size       : S (odd integer recommended)
        px_scale         : arcsec/pixel for this set of stars
        sub_grid         : K — K×K points per pixel for quadrature
        apply_dcr        : if True, add the DCR color-dependent shift

        Returns
        -------
        stamps : [N, S, S] — pixel-integrated intensities
        """
        device = centroids_arcsec.device
        dtype  = centroids_arcsec.dtype
        N      = centroids_arcsec.shape[0]
        S      = stamp_size
        K      = sub_grid
        half_s = (S - 1) / 2.0

        # --- DCR shift per band ---
        if apply_dcr:
            _, color = self.sed_encoder(sed_vec)
            dcr_shift = self.dcr(band_idx, color)        # [N, 2] arcsec
            effective_centroid = centroids_arcsec + dcr_shift
        else:
            effective_centroid = centroids_arcsec

        # --- Pixel-centre grid in arcsec relative to stamp centre ---
        i_idx = torch.arange(S, device=device, dtype=dtype)
        pix_centres = (i_idx - half_s) * px_scale            # [S]
        # [S, S, 2] — (x, y) at each pixel
        py, px = torch.meshgrid(pix_centres, pix_centres, indexing='ij')
        pix_xy = torch.stack([px, py], dim=-1)               # [S, S, 2]

        # --- Sub-pixel quadrature grid ---
        # K points per pixel, at the centre of each sub-cell (uniform quadrature).
        # offsets within pixel: ((k+0.5)/K − 0.5) * px_scale, k=0..K-1
        sub = ((torch.arange(K, device=device, dtype=dtype) + 0.5) / K - 0.5) * px_scale
        sy, sx = torch.meshgrid(sub, sub, indexing='ij')
        sub_xy = torch.stack([sx, sy], dim=-1)               # [K, K, 2]

        # --- Build full query grid: [N, S, S, K, K, 2] ---
        # query = pix_xy + sub_xy − effective_centroid
        pix_q = pix_xy.view(1, S, S, 1, 1, 2)                # [1, S, S, 1, 1, 2]
        sub_q = sub_xy.view(1, 1, 1, K, K, 2)                # [1, 1, 1, K, K, 2]
        cen_q = effective_centroid.view(N, 1, 1, 1, 1, 2)    # [N, 1, 1, 1, 1, 2]
        xy_sub = pix_q + sub_q - cen_q                       # [N, S, S, K, K, 2]

        # --- Broadcast context to every query point ---
        tile_q = tile_pos.view(N, 1, 1, 1, 1, 2).expand(N, S, S, K, K, 2)
        band_q = band_idx.view(N, 1, 1, 1, 1).expand(N, S, S, K, K)
        sed_q  = sed_vec.view(N, 1, 1, 1, 1, N_BANDS).expand(N, S, S, K, K, N_BANDS)

        # Flatten for batched SIREN evaluation
        M = N * S * S * K * K
        intensity = self.evaluate(
            xy_sub.reshape(M, 2),
            tile_q.reshape(M, 2),
            band_q.reshape(M),
            sed_q.reshape(M, N_BANDS),
        ).view(N, S, S, K, K)

        # --- Integrate over pixel (mean over sub-grid) ---
        stamps = intensity.mean(dim=(-2, -1))                # [N, S, S]
        return stamps


# ---------------------------------------------------------------------------
# Analytic optimal flux and chi² loss
# ---------------------------------------------------------------------------

def analytic_optimal_flux(
    data: torch.Tensor,    # [N, S, S]   data stamps (background-subtracted)
    model: torch.Tensor,   # [N, S, S]   unnormalised PSF stamps
    var:   torch.Tensor,   # [N, S, S]   pixel variance
    eps:   float = 1e-12,
) -> torch.Tensor:
    """
    Closed-form maximum-likelihood flux estimate assuming Gaussian pixel noise:
        F* = Σ (D · M / σ²) / Σ (M² / σ²)
    """
    inv_var = 1.0 / var.clamp(min=eps)
    num = (data * model * inv_var).sum(dim=(-2, -1))
    den = (model * model * inv_var).sum(dim=(-2, -1)).clamp(min=eps)
    return num / den                                         # [N]


def chi2_loss(
    data: torch.Tensor,   # [N, S, S]
    model: torch.Tensor,  # [N, S, S]   PSF stamps
    var:   torch.Tensor,  # [N, S, S]
    flux:  Optional[torch.Tensor] = None,  # [N] — if None, compute analytically
    reduce: str = 'mean',
) -> torch.Tensor:
    """
    Heteroscedastic chi² loss summed over pixels, then reduced over stars.

    Flux is detached from the graph when computed analytically — we don't
    want gradients to flow through the flux estimate because it's an
    implicit function of the PSF (would create a two-sided dependence).
    """
    if flux is None:
        flux = analytic_optimal_flux(data, model, var).detach()  # [N]
    resid = data - flux.view(-1, 1, 1) * model
    chi2_per_pix = resid ** 2 / var.clamp(min=1e-12)
    chi2_per_star = chi2_per_pix.sum(dim=(-2, -1))               # [N]
    if reduce == 'mean':
        return chi2_per_star.mean()
    elif reduce == 'sum':
        return chi2_per_star.sum()
    else:
        return chi2_per_star


# ---------------------------------------------------------------------------
# Convenience: normalise a rendered PSF stamp to unit integral
# ---------------------------------------------------------------------------

def normalise_psf(psf: torch.Tensor, px_scale: float) -> torch.Tensor:
    """
    Normalise PSF stamp to ∫ PSF dA = 1 (integrating over angular area).

    Uses sum × pixel_area as a discrete approximation to the integral.
    `px_scale` is in arcsec/pixel; pixel area is px_scale².
    """
    pix_area = px_scale * px_scale
    total = psf.sum(dim=(-2, -1), keepdim=True) * pix_area
    return psf / total.clamp(min=1e-12)
