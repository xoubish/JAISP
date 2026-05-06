"""PSF v4: NN ePSF head — predicts an oversampled effective PSF stamp at any
spatial position and band.

This is a *learned* version of the SPHEREx-style tabulated ePSF. Where SPHEREx
stores 21×21 zones with a fixed ePSF per zone, we train a small neural network
that takes ``(x, y, band)`` and outputs an oversampled (5×) PSF stamp on demand.
The continuous parameterisation lets the PSF vary smoothly across each tile, no
zone boundaries.

Output convention (matches photutils.psf.EPSFModel):
    - oversampled stamp of shape (psf_size, psf_size) — default 47×47
    - oversampling factor (5×) baked in via the rendering helper
    - normalised to unit total flux

Once trained, an instance can be wrapped as a photutils EPSFModel via
``epsf_v4_photutils.PSFFieldV4ToEPSF`` and dropped into Tractor / Photutils
forced photometry.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Module-level band ordering — matches jaisp_foundation_v6/v7.
RUBIN_BANDS = ["rubin_u", "rubin_g", "rubin_r", "rubin_i", "rubin_z", "rubin_y"]
EUCLID_BANDS = ["euclid_VIS", "euclid_Y", "euclid_J", "euclid_H"]
ALL_BANDS = RUBIN_BANDS + EUCLID_BANDS


class PSFFieldV4(nn.Module):
    """Continuous NN ePSF head.

    Parameters
    ----------
    psf_size : int
        Side of the oversampled PSF stamp produced by the network. Default 47
        matches SPHEREx ePSF / photutils convention.
    oversampling : int
        Oversampling factor of the produced stamp relative to the band's
        native pixel scale. Default 5 matches SPHEREx.
    band_names : list of str
        Band ordering used by the ``band_idx`` input (default = ALL_BANDS).
    hidden_ch : int
        Hidden channel width of the per-band MLP+upsampler.
    n_freqs : int
        Number of Fourier-feature frequencies for the (x, y) input
        (Rahaman et al. 2019 / Tancik et al. 2020 spectral bias mitigation).
    """

    def __init__(
        self,
        psf_size: int = 47,
        oversampling: int = 5,
        band_names: List[str] = ALL_BANDS,
        hidden_ch: int = 128,
        n_freqs: int = 8,
        gauss_init_sigma_ovs: float = 5.0,
    ):
        super().__init__()
        self.psf_size = int(psf_size)
        self.oversampling = int(oversampling)
        self.band_names = list(band_names)
        self.band_to_idx = {b: i for i, b in enumerate(self.band_names)}
        self.gauss_init_sigma_ovs = float(gauss_init_sigma_ovs)
        n_bands = len(self.band_names)

        self.n_freqs = int(n_freqs)
        # Learnable Fourier features for the spatial position (x, y).
        # Position is in [-1, 1]; frequencies span roughly 1..2^(n_freqs-1).
        freqs = 2 ** torch.arange(self.n_freqs).float() * math.pi
        self.register_buffer("ff_freqs", freqs, persistent=False)
        # Per-band embedding (small) so the network can produce different
        # PSF profiles per band without sharing all parameters.
        self.band_embed = nn.Embedding(n_bands, 32)

        in_dim = 2 * self.n_freqs * 2 + 32  # sin + cos × 2 axes × n_freqs + band_embed

        # Map (Fourier(x,y) ⊕ band_embed) → a low-resolution PSF latent, then
        # upsample to the oversampled stamp via a small ConvTranspose stack.
        # 12×12 → 24×24 → 47×47 (cropped from a 48 upsample).
        base_h = 12
        self.base_h = base_h
        self.head_to_grid = nn.Sequential(
            nn.Linear(in_dim, hidden_ch),
            nn.GELU(),
            nn.Linear(hidden_ch, hidden_ch),
            nn.GELU(),
            nn.Linear(hidden_ch, hidden_ch * base_h * base_h // 4),
        )

        # Bilinear upsample + Conv (instead of ConvTranspose2d with stride 2).
        # ConvTranspose2d has a well-known "checkerboard" failure mode that
        # produces lattice ridges in the oversampled output (Odena et al. 2016,
        # https://distill.pub/2016/deconv-checkerboard/). Replacing it with
        # bilinear upsampling + regular Conv2d removes those artifacts entirely.
        self.upsample_net = nn.Sequential(
            # Stage 1: base_h × base_h  → 2·base_h × 2·base_h
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_ch // 4, hidden_ch // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch // 2, hidden_ch // 2, kernel_size=3, padding=1),
            nn.GELU(),
            # Stage 2: 2·base_h × 2·base_h  → 4·base_h × 4·base_h
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_ch // 2, hidden_ch // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch // 4, 1, kernel_size=3, padding=1),
        )

        with torch.no_grad():
            convs = [m for m in self.upsample_net if isinstance(m, nn.Conv2d)]
            for m in convs:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # Scale the LAST conv down so the decoder's initial output is small
            # compared to the per-band Gaussian prior below. This makes the
            # initial PSF a centred Gaussian that the decoder gradually refines,
            # instead of a near-uniform blob (the failure mode v4 was hitting
            # without a prior).
            if convs:
                convs[-1].weight.data *= 0.01

        # Per-band learnable Gaussian prior, in pre-softplus log-space. The
        # decoder's output is added to this before softplus + flux normalisation,
        # so each band starts with a centred 2-D Gaussian PSF and the network
        # only has to learn deviations.
        P = self.psf_size
        c = (P - 1) / 2.0
        coords = torch.arange(P, dtype=torch.float32) - c
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        log_gauss = -(xx * xx + yy * yy) / (2.0 * self.gauss_init_sigma_ovs ** 2)
        self.psf_bias = nn.Parameter(
            log_gauss.view(1, 1, P, P).repeat(n_bands, 1, 1, 1)
        )  # [n_bands, 1, P, P]

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"PSFFieldV4: {n_params/1e6:.2f}M trainable params  "
              f"(psf_size={self.psf_size}, oversampling={self.oversampling}, "
              f"n_bands={n_bands}, gauss_init_sigma_ovs={self.gauss_init_sigma_ovs})")

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def forward(
        self,
        pos_norm: torch.Tensor,   # [B, 2] in [-1, 1]
        band_idx: torch.Tensor,   # [B] int64 indices into self.band_names
    ) -> torch.Tensor:
        """Produce oversampled, unit-flux PSF stamps for a batch of (pos, band)."""
        B = pos_norm.shape[0]

        # Fourier features
        xy = pos_norm  # [B, 2]
        # [B, 2, n_freqs]
        proj = xy.unsqueeze(-1) * self.ff_freqs.view(1, 1, -1)
        ff = torch.cat([proj.sin(), proj.cos()], dim=-1)  # [B, 2, 2*n_freqs]
        ff = ff.flatten(1)                                 # [B, 4*n_freqs]

        # Band embedding
        be = self.band_embed(band_idx)  # [B, 32]

        feat = torch.cat([ff, be], dim=-1)
        latent = self.head_to_grid(feat)
        latent = latent.view(B, -1, self.base_h, self.base_h)

        psf = self.upsample_net(latent)  # [B, 1, ~48, ~48]
        # Crop / pad to exact (psf_size, psf_size).
        Hp, Wp = psf.shape[-2:]
        if Hp >= self.psf_size and Wp >= self.psf_size:
            offh = (Hp - self.psf_size) // 2
            offw = (Wp - self.psf_size) // 2
            psf = psf[..., offh:offh + self.psf_size, offw:offw + self.psf_size]
        else:
            ph = (self.psf_size - Hp) // 2
            pw = (self.psf_size - Wp) // 2
            psf = F.pad(psf, (pw, self.psf_size - Wp - pw,
                              ph, self.psf_size - Hp - ph))

        # Add the per-band Gaussian prior (in pre-softplus space) before
        # rectifying. This anchors the model at a centred Gaussian shape.
        psf = psf + self.psf_bias[band_idx]
        # Force non-negative (softplus is smooth, won't kill gradients) and
        # normalise to unit total flux per stamp.
        psf = F.softplus(psf)
        psf = psf / psf.flatten(1).sum(dim=-1).clamp(min=1e-8).view(B, 1, 1, 1)
        return psf  # [B, 1, psf_size, psf_size], unit-flux, oversampled

    # ----------------------------------------------------------
    # Render to native resolution at sub-pixel position
    # ----------------------------------------------------------

    def render_at_native(
        self,
        psf_oversampled: torch.Tensor,  # [B, 1, P, P], unit-flux
        frac_xy: torch.Tensor,           # [B, 2] in pixels (sub-pixel offset)
        stamp_size: int = 32,
    ) -> torch.Tensor:
        """Render an oversampled PSF onto a native-resolution stamp_size × stamp_size grid.

        The source's sub-pixel position within the stamp is given by ``frac_xy``;
        ``(0, 0)`` means the source is centred on the central pixel of the stamp,
        ``(0.5, 0.5)`` means it sits at the intersection of the four central pixels.
        The native stamp is sampled by averaging the oversampled PSF over each
        native pixel (i.e. exact pixel integration, not just bilinear interpolation).
        """
        B, _, P, _ = psf_oversampled.shape
        ovs = self.oversampling
        assert P == self.psf_size, f"expected oversampled stamp side {self.psf_size}, got {P}"

        # The oversampled stamp covers a (P/ovs) × (P/ovs) native-pixel region.
        # We want to integrate it over the requested native stamp (stamp_size × stamp_size)
        # centred at the sub-pixel offset frac_xy.
        # Build a sampling grid in oversampled-pixel coordinates and use
        # grid_sample with 'bilinear' on the integrated grid (sum-pool over ovs).

        # Convention: the source's "central pixel" in the stamp is at integer index
        #   ix = stamp_size // 2 (= 16 for stamp_size=32, = 11 for stamp_size=23, etc).
        # This matches the build_psf_v4_training_set.py stamp-extraction convention,
        # where the stamp covers [ix - half, ix - half + stamp) and the source sits
        # at stamp-pixel index `half + frac`. Native-pixel coords relative to the
        # source's central pixel are then `(arange(stamp_size) - half)`, which
        # yields zero exactly at the central pixel.
        half = stamp_size // 2
        coords = torch.arange(stamp_size, device=psf_oversampled.device,
                              dtype=torch.float32) - float(half)
        # For each native pixel, we want to sample ovs×ovs sub-pixel points and average.
        sub = (torch.arange(ovs, device=psf_oversampled.device, dtype=torch.float32)
               - (ovs - 1) / 2.0) / ovs
        # Native pixel centres are at integer offsets from stamp centre. Source position
        # within the stamp is frac_xy (in native pixels). So in oversampled-PSF
        # coordinates, the centre of pixel (px, py) of the stamp is at
        # ovs * (px - frac_xy_x), ovs * (py - frac_xy_y) plus the half-pixel offset.

        # Build sampling locations in oversampled-PSF pixel index space, then
        # convert to grid_sample's [-1, 1] normalised coords.
        # Shape gymnastics: end up with [B, stamp_size*ovs, stamp_size*ovs, 2].
        ys = coords.view(stamp_size, 1, 1, 1) + sub.view(1, ovs, 1, 1)  # [stamp_size, ovs, 1, 1]
        xs = coords.view(1, 1, stamp_size, 1) + sub.view(1, 1, 1, ovs)  # [1, 1, stamp_size, ovs]

        # Subtract per-batch frac_xy (broadcasts over the spatial dims).
        # frac_xy in native pixels -> in oversampled-PSF pixels -> normalised.
        # First in *native* coords:
        ys_b = ys.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, sH, ovs, 1, 1]
        xs_b = xs.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, 1, 1, sW, ovs]
        ys_b = ys_b - frac_xy[:, 1].view(B, 1, 1, 1, 1)
        xs_b = xs_b - frac_xy[:, 0].view(B, 1, 1, 1, 1)

        # Convert to oversampled-PSF index space: each native pixel = ovs PSF pixels.
        ys_idx = ys_b * ovs
        xs_idx = xs_b * ovs

        # PSF pixel centre indexing: index 0 is the top-left, (P-1)/2 is the centre.
        centre = (P - 1) / 2.0
        ys_pix = ys_idx + centre
        xs_pix = xs_idx + centre

        # Normalise to [-1, +1] for grid_sample.
        ys_n = 2.0 * ys_pix / (P - 1) - 1.0
        xs_n = 2.0 * xs_pix / (P - 1) - 1.0

        grid = torch.stack([
            xs_n.expand(B, stamp_size, ovs, stamp_size, ovs),
            ys_n.expand(B, stamp_size, ovs, stamp_size, ovs),
        ], dim=-1)  # [B, sH, ovs, sW, ovs, 2]
        grid = grid.reshape(B, stamp_size * ovs, stamp_size * ovs, 2)

        sampled = F.grid_sample(psf_oversampled, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True)
        # sampled: [B, 1, sH*ovs, sW*ovs] — sum (not mean) over each ovs×ovs native pixel
        # so that total flux is preserved: each oversampled pixel carries 1/ovs² of the
        # native pixel area, and the unit-flux PSF was normalised in oversampled space.
        sampled = sampled.view(B, 1, stamp_size, ovs, stamp_size, ovs).sum(dim=(3, 5))
        return sampled  # [B, 1, stamp_size, stamp_size]


__all__ = ["PSFFieldV4", "ALL_BANDS", "RUBIN_BANDS", "EUCLID_BANDS"]
