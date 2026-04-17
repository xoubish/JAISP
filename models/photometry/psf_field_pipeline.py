"""
PSFField-backed forced photometry.

This is the production-facing successor to the legacy PSFNet photometry
pipeline.  It reuses the existing matched-filter estimator but renders PSF
templates with ``models.psf.PSFField``.

The pipeline accepts either one shared source position per object or one
position per object per band.  The latter is the natural input from the
astrometry head, where each band has its own corrected source position.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

try:  # Package import: models.photometry.psf_field_pipeline
    from .forced_photometry import matched_filter, snr as compute_snr
    from .stamp_extractor import estimate_local_background, extract_stamps
except ImportError:  # Script import from models/photometry
    from forced_photometry import matched_filter, snr as compute_snr
    from stamp_extractor import estimate_local_background, extract_stamps

try:
    from models.psf import BAND_ORDER, BAND_PX_SCALE, BAND_TO_IDX, N_BANDS, PSFField
except ImportError:
    from psf import BAND_ORDER, BAND_PX_SCALE, BAND_TO_IDX, N_BANDS, PSFField


_RUBIN_SHORT = {"u", "g", "r", "i", "z", "y"}
_NISP_SHORT = {"Y", "J", "H"}


def normalise_band_name(name: str) -> str:
    """Map common local aliases to PSFField band names."""
    if name in BAND_TO_IDX:
        return name
    if name in _RUBIN_SHORT:
        return f"rubin_{name}"
    if name in _NISP_SHORT:
        return f"euclid_{name}"
    if name == "VIS":
        return "euclid_VIS"
    if name.startswith("nisp_"):
        return "euclid_" + name.split("_", 1)[1]
    if name.startswith("NISP_"):
        return "euclid_" + name.split("_", 1)[1]
    raise KeyError(f"Unknown band name {name!r}; expected one of {BAND_ORDER}")


def _sum_normalise_psf(psf: torch.Tensor) -> torch.Tensor:
    """Normalise PSF stamps to unit discrete pixel sum for flux estimation."""
    return psf / psf.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-20)


def load_psf_field_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> PSFField:
    """
    Load a ``train_psf_field.py`` checkpoint and return an eval-mode PSFField.

    The checkpoint stores additional training state, but photometry only needs
    the PSFField weights and architecture config.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    psf_field = PSFField(
        sed_embed_dim=cfg.get("sed_embed_dim", 8),
        band_embed_dim=cfg.get("band_embed_dim", 16),
        tile_freqs=cfg.get("tile_freqs", 6),
        siren_hidden=cfg.get("siren_hidden", 128),
        siren_depth=cfg.get("siren_depth", 5),
        w0_first=cfg.get("w0_first", 30.0),
        envelope_r_rubin=cfg.get("envelope_r_rubin", 1.7),
        envelope_r_euclid=cfg.get("envelope_r_euclid", 0.85),
        envelope_power=cfg.get("envelope_power", 4.0),
    ).to(device)
    state = ckpt.get("psf_field_state")
    if state is None:
        raise KeyError(f"{checkpoint_path} does not contain 'psf_field_state'")
    psf_field.load_state_dict(state)
    psf_field.eval()
    return psf_field


class PSFFieldPhotometryPipeline:
    """
    Forced photometry using PSFField-rendered templates.

    Parameters
    ----------
    psf_field
        Trained PSFField model.
    band_names
        Bands represented by the input tile tensor. Defaults to all PSFField
        bands in canonical order.
    stamp_size
        Odd stamp side length in pixels.
    sub_grid
        K for KxK PSF pixel-integration samples.
    px_scales
        Optional data pixel scale per band. Defaults to the native PSFField
        scale for each band. Override this when the input images have been
        reprojected to a common grid, for example all bands on 0.1"/px VIS.
    """

    def __init__(
        self,
        psf_field: PSFField,
        band_names: Optional[Sequence[str]] = None,
        stamp_size: int = 25,
        sub_grid: int = 4,
        px_scales: Optional[Sequence[float]] = None,
        bg_inner_radius: float = 8.0,
        bg_outer_radius: float = 11.5,
        stamp_chunk_size: int = 16384,
        psf_chunk_size: int = 1024,
        apply_dcr: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.psf_field = psf_field
        self.device = device or next(psf_field.parameters()).device
        self.psf_field = self.psf_field.to(self.device).eval()

        self.band_names = [normalise_band_name(b) for b in (band_names or BAND_ORDER)]
        self.band_indices = torch.tensor(
            [BAND_TO_IDX[b] for b in self.band_names],
            dtype=torch.long,
            device=self.device,
        )
        if px_scales is None:
            self.px_scales = [float(BAND_PX_SCALE[BAND_TO_IDX[b]]) for b in self.band_names]
        else:
            if len(px_scales) != len(self.band_names):
                raise ValueError("px_scales must have the same length as band_names")
            self.px_scales = [float(x) for x in px_scales]

        self.stamp_size = int(stamp_size)
        if self.stamp_size % 2 != 1:
            raise ValueError("stamp_size must be odd")
        self.sub_grid = int(sub_grid)
        self.bg_inner_radius = float(bg_inner_radius)
        self.bg_outer_radius = float(bg_outer_radius)
        self.stamp_chunk_size = int(stamp_chunk_size)
        self.psf_chunk_size = int(psf_chunk_size)
        self.apply_dcr = bool(apply_dcr)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "PSFFieldPhotometryPipeline":
        """Load PSFField from checkpoint and return a ready pipeline."""
        psf_field = load_psf_field_checkpoint(checkpoint_path, device=device)
        return cls(psf_field=psf_field, device=device, **kwargs)

    @torch.no_grad()
    def render_psfs(
        self,
        positions_px: torch.Tensor,
        tile_hw: Sequence[int],
        sed_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Render unit-sum PSF templates for each source and band.

        Parameters
        ----------
        positions_px
            [N, 2] shared positions or [N, B, 2] per-band positions.
        tile_hw
            (H, W) of the image grid that positions_px refers to.
        sed_vec
            Optional [N, 10] PSFField SED conditioning vector. Defaults to the
            mean-SED convention, i.e. zeros.
        """
        positions = self._prepare_positions(positions_px)
        N, C, _ = positions.shape
        H, W = int(tile_hw[0]), int(tile_hw[1])
        if sed_vec is None:
            sed = torch.zeros(N, N_BANDS, dtype=torch.float32, device=self.device)
        else:
            sed = sed_vec.to(self.device, dtype=torch.float32)
            if sed.shape != (N, N_BANDS):
                raise ValueError(f"sed_vec must have shape {(N, N_BANDS)}, got {tuple(sed.shape)}")

        psfs = torch.empty(
            N, C, self.stamp_size, self.stamp_size,
            dtype=torch.float32,
            device=self.device,
        )
        zeros = None

        for b in range(C):
            pos_b = positions[:, b, :]
            band_idx = self.band_indices[b]
            for start in range(0, N, self.psf_chunk_size):
                end = min(start + self.psf_chunk_size, N)
                pos_chunk = pos_b[start:end]
                tile_pos = torch.stack(
                    [
                        pos_chunk[:, 0] / max(W - 1, 1),
                        pos_chunk[:, 1] / max(H - 1, 1),
                    ],
                    dim=-1,
                ).clamp(0.0, 1.0)
                n_chunk = end - start
                if zeros is None or zeros.shape[0] < n_chunk:
                    zeros = torch.zeros(self.psf_chunk_size, 2, device=self.device)
                stamp = self.psf_field.render_stamps(
                    centroids_arcsec=zeros[:n_chunk],
                    tile_pos=tile_pos,
                    band_idx=torch.full(
                        (n_chunk,),
                        int(band_idx.item()),
                        dtype=torch.long,
                        device=self.device,
                    ),
                    sed_vec=sed[start:end],
                    stamp_size=self.stamp_size,
                    px_scale=self.px_scales[b],
                    sub_grid=self.sub_grid,
                    apply_dcr=self.apply_dcr,
                )
                psfs[start:end, b] = _sum_normalise_psf(stamp)

        return psfs

    @torch.no_grad()
    def run(
        self,
        tile: torch.Tensor,
        rms: torch.Tensor,
        positions_px: torch.Tensor,
        sed_vec: Optional[torch.Tensor] = None,
        return_psfs: bool = False,
    ) -> Dict[str, torch.Tensor | List[str]]:
        """
        Run forced photometry on a tile.

        Parameters
        ----------
        tile, rms
            [B, H, W] image and RMS tensors. B must match ``band_names``.
        positions_px
            [N, 2] shared source positions or [N, B, 2] per-band corrected
            positions in the same pixel grid as ``tile``.
        sed_vec
            Optional [N, 10] SED conditioning vector for PSFField.
        return_psfs
            Include rendered PSF templates in the returned dictionary.

        Returns
        -------
        dict containing ``flux``, ``flux_err``, ``chi2_dof``, ``snr``, ``bg``,
        ``positions_px``, and ``band_names``.
        """
        tile = tile.to(self.device, dtype=torch.float32)
        rms = rms.to(self.device, dtype=torch.float32)
        if tile.ndim != 3 or rms.shape != tile.shape:
            raise ValueError("tile and rms must both have shape [B, H, W]")
        C, H, W = tile.shape
        if C != len(self.band_names):
            raise ValueError(f"tile has {C} bands but pipeline has {len(self.band_names)} band_names")

        positions = self._prepare_positions(positions_px)
        if positions.shape[1] != C:
            raise ValueError(f"positions have {positions.shape[1]} bands but tile has {C}")

        psfs = self.render_psfs(positions, tile_hw=(H, W), sed_vec=sed_vec)
        stamps = self._extract_stamps(tile, positions)
        rms_stamps = self._extract_stamps(rms, positions)

        bg = estimate_local_background(
            stamps,
            inner_radius=self.bg_inner_radius,
            outer_radius=self.bg_outer_radius,
        )
        stamps_sub = stamps - bg.unsqueeze(-1).unsqueeze(-1)
        var = rms_stamps.pow(2).clamp(min=1e-20)

        flux, flux_err, chi2_dof = matched_filter(stamps_sub, psfs, var)
        result: Dict[str, torch.Tensor | List[str]] = {
            "flux": flux,
            "flux_err": flux_err,
            "chi2_dof": chi2_dof,
            "snr": compute_snr(flux, flux_err),
            "bg": bg,
            "positions_px": positions,
            "band_names": list(self.band_names),
        }
        if return_psfs:
            result["psfs"] = psfs
        return result

    def _prepare_positions(self, positions_px: torch.Tensor) -> torch.Tensor:
        positions = positions_px.to(self.device, dtype=torch.float32)
        if positions.ndim == 2:
            if positions.shape[-1] != 2:
                raise ValueError("positions_px must have shape [N, 2] or [N, B, 2]")
            positions = positions[:, None, :].expand(-1, len(self.band_names), -1)
        elif positions.ndim == 3:
            if positions.shape[-1] != 2:
                raise ValueError("positions_px must have shape [N, B, 2]")
            if positions.shape[1] != len(self.band_names):
                raise ValueError(
                    f"positions band axis has length {positions.shape[1]}, "
                    f"expected {len(self.band_names)}"
                )
        else:
            raise ValueError("positions_px must have shape [N, 2] or [N, B, 2]")
        return positions.contiguous()

    def _extract_stamps(self, image: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Extract [N, B, S, S] stamps, supporting per-band positions."""
        N, C, _ = positions.shape
        shared = torch.allclose(
            positions,
            positions[:, :1, :].expand_as(positions),
            rtol=0.0,
            atol=1e-6,
        )
        if shared:
            return extract_stamps(
                image,
                positions[:, 0, :],
                self.stamp_size,
                chunk_size=self.stamp_chunk_size,
            )

        out = torch.empty(
            N, C, self.stamp_size, self.stamp_size,
            dtype=torch.float32,
            device=self.device,
        )
        for b in range(C):
            out[:, b:b + 1] = extract_stamps(
                image[b:b + 1],
                positions[:, b, :],
                self.stamp_size,
                chunk_size=self.stamp_chunk_size,
            )
        return out


__all__ = [
    "PSFFieldPhotometryPipeline",
    "load_psf_field_checkpoint",
    "normalise_band_name",
]
