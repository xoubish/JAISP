"""
TilePhotometryPipeline: end-to-end forced PSF photometry for a full tile.

Processing order (all on GPU, tile loaded once)
------------------------------------------------
1.  precompute PSF grid [B, G, G, S, S] from PSFNet
2.  compute normalised source positions (x_norm, y_norm ∈ [0,1])
3.  bilinear-interpolate PSFs for all sources → [N, B, S, S]
4.  extract stamps from tile [C, H, W] → [N, B, S, S]
5.  extract stamps from rms map → [N, B, S, S] (rms)
6.  estimate local sky background → [N, B]; subtract from stamps
7.  var = rms_stamps²
8.  matched-filter forced photometry → flux, flux_err, chi2_dof [N, B]

Throughput (A100, stamp_size=21, B=10): ~700K sources / sec / GPU.

Usage
-----
    pipe = TilePhotometryPipeline(psf_net, device='cuda')
    result = pipe.run(tile, rms, positions_px)
    # result['flux']     [N, B]
    # result['flux_err'] [N, B]
    # result['chi2_dof'] [N, B]
    # result['snr']      [N, B]
    # result['bg']       [N, B]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from psf_net import PSFNet
from stamp_extractor import estimate_local_background, extract_stamps
from forced_photometry import matched_filter, snr as compute_snr


class TilePhotometryPipeline:
    """
    Forced PSF photometry pipeline for a single tile.

    Parameters
    ----------
    psf_net          : trained PSFNet (or freshly initialised for inference
                       without training)
    stamp_size       : PSF/stamp side length in pixels
    psf_grid_size    : precomputed PSF grid resolution (default 8 = 8×8 grid)
    bg_inner_radius  : inner sky-annulus radius in pixels
    bg_outer_radius  : outer sky-annulus radius in pixels
    stamp_chunk_size : sources per chunk for stamp extraction
    psf_chunk_size   : sources per chunk for PSF interpolation
    device           : torch device; inferred from psf_net if None
    """

    def __init__(
        self,
        psf_net: PSFNet,
        stamp_size: int = 21,
        psf_grid_size: int = 8,
        bg_inner_radius: float = 7.0,
        bg_outer_radius: float = 9.5,
        stamp_chunk_size: int = 16384,
        psf_chunk_size: int = 4096,
        device: Optional[torch.device] = None,
    ):
        self.psf_net = psf_net
        self.stamp_size = stamp_size
        self.psf_grid_size = psf_grid_size
        self.bg_inner_radius = bg_inner_radius
        self.bg_outer_radius = bg_outer_radius
        self.stamp_chunk_size = stamp_chunk_size
        self.psf_chunk_size = psf_chunk_size
        self.device = device or next(psf_net.parameters()).device

    # ------------------------------------------------------------------
    @torch.no_grad()
    def run(
        self,
        tile: torch.Tensor,          # [B_bands, H, W]  or [C, H, W]
        rms:  torch.Tensor,          # [B_bands, H, W]
        positions_px: torch.Tensor,  # [N, 2]  (x, y) in tile pixel coords
    ) -> Dict[str, torch.Tensor]:
        """
        Run photometry for all sources in `positions_px`.

        Parameters
        ----------
        tile         : [C, H, W] float32 — sky image (all bands)
        rms          : [C, H, W] float32 — per-pixel noise (same shape as tile)
        positions_px : [N, 2] float32 — (x, y) source positions in pixels

        Returns
        -------
        dict with keys:
            flux      [N, B]  — optimal flux in data units
            flux_err  [N, B]  — 1-sigma Cramér-Rao uncertainty
            chi2_dof  [N, B]  — reduced chi² of PSF fit (≈1 for well-fit sources)
            snr       [N, B]  — flux / flux_err
            bg        [N, B]  — sky background subtracted per source per band
        """
        tile = tile.to(self.device)
        rms  = rms.to(self.device)
        positions_px = positions_px.to(self.device)

        C, H, W = tile.shape
        N = positions_px.shape[0]

        # 1. Precompute PSF grid [B, G, G, S, S]
        psf_grid = self.psf_net.precompute_grid(
            grid_size=self.psf_grid_size, device=self.device
        )

        # 2. Normalised source positions
        x_norm = positions_px[:, 0] / max(1, W - 1)
        y_norm = positions_px[:, 1] / max(1, H - 1)

        # 3. Interpolate PSFs for all sources [N, B, S, S]
        psfs = PSFNet.get_psfs_for_sources(
            psf_grid, x_norm, y_norm,
            chunk_size=self.psf_chunk_size,
        )

        # 4. Extract stamps [N, C, S, S]
        stamps_all = extract_stamps(
            tile, positions_px, self.stamp_size,
            chunk_size=self.stamp_chunk_size,
        )
        rms_all = extract_stamps(
            rms, positions_px, self.stamp_size,
            chunk_size=self.stamp_chunk_size,
        )

        # 5. Local sky background [N, C]; subtract
        bg = estimate_local_background(
            stamps_all, self.bg_inner_radius, self.bg_outer_radius
        )
        stamps_sub = stamps_all - bg.unsqueeze(-1).unsqueeze(-1)

        # 6. Variance and matched filter
        var = rms_all.pow(2).clamp(min=1e-20)

        flux, flux_err, chi2_dof = matched_filter(stamps_sub, psfs, var)
        snr_vals = compute_snr(flux, flux_err)

        return {
            'flux':     flux,
            'flux_err': flux_err,
            'chi2_dof': chi2_dof,
            'snr':      snr_vals,
            'bg':       bg,
        }

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        """Save PSFNet weights and pipeline config."""
        torch.save({
            'psf_net_state': self.psf_net.state_dict(),
            'config': {
                'n_bands':        self.psf_net.n_bands,
                'stamp_size':     self.stamp_size,
                'psf_grid_size':  self.psf_grid_size,
                'bg_inner_radius': self.bg_inner_radius,
                'bg_outer_radius': self.bg_outer_radius,
                'hidden_dim':     self.psf_net.mlp[0].out_features,
                'band_embed_dim': self.psf_net.band_embed.embedding_dim,
            },
        }, path)
        print(f'Saved photometry checkpoint → {path}')

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> 'TilePhotometryPipeline':
        """Load PSFNet from checkpoint and return a ready pipeline."""
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        cfg  = ckpt['config']
        psf_net = PSFNet(
            n_bands=cfg['n_bands'],
            stamp_size=cfg['stamp_size'],
            hidden_dim=cfg['hidden_dim'],
            band_embed_dim=cfg['band_embed_dim'],
        )
        psf_net.load_state_dict(ckpt['psf_net_state'])
        if device is not None:
            psf_net = psf_net.to(device)
        psf_net.eval()
        return cls(
            psf_net=psf_net,
            stamp_size=cfg['stamp_size'],
            psf_grid_size=cfg.get('psf_grid_size', 8),
            bg_inner_radius=cfg.get('bg_inner_radius', 7.0),
            bg_outer_radius=cfg.get('bg_outer_radius', 9.5),
            device=device,
            **kwargs,
        )
