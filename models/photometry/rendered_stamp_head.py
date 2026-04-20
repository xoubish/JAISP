"""
End-to-end rendered-stamp head for JAISP photometry.

No explicit PSF, no unit-sum morphology template, no convolution step. The head
reads frozen V8 features at each source's VIS position and outputs, for every
(source, band) pair, a 31x31 unit-sum positive stamp. Fluxes are then solved
analytically per band and the scene residual chi-square is the only training
signal.

Design constraints baked in:

* One VIS position per source (astrometry-head corrected); the WCS projects
  that single position into the other bands. No per-band position refinement.
* A shared intrinsic spatial feature map is decoded per band via FiLM, so one
  wavelength-independent source representation drives all bands.
* Per-band sub-pixel offset and tile position are fed as conditioning so the
  band decoder can absorb spatial PSF variation and under-sampling phase
  without PSF being an explicit concept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from astrometry2.latent_position_head import (
        extract_local_windows,
        vis_px_to_bottleneck_px,
    )
    from jaisp_foundation_v6 import ConvNeXtBlock
except ImportError:
    from ..astrometry2.latent_position_head import (
        extract_local_windows,
        vis_px_to_bottleneck_px,
    )
    from ..jaisp_foundation_v6 import ConvNeXtBlock

try:
    from .scarlet_like import (
        _extract_band_scene,
        _initial_flux_from_templates,
        _odd_at_least,
        _prepare_positions,
        build_neighbor_groups,
        place_templates_in_scene,
    )
    from .stamp_extractor import estimate_local_background
except ImportError:
    from scarlet_like import (
        _extract_band_scene,
        _initial_flux_from_templates,
        _odd_at_least,
        _prepare_positions,
        build_neighbor_groups,
        place_templates_in_scene,
    )
    from stamp_extractor import estimate_local_background


@dataclass
class RenderedSceneResult:
    group_id: int
    indices: torch.Tensor
    center_xy: torch.Tensor
    scene_size: int
    data: torch.Tensor
    model: torch.Tensor
    resid: torch.Tensor
    background: torch.Tensor
    flux: torch.Tensor
    chi2_dof: torch.Tensor


class RenderedStampHead(nn.Module):
    """Predict per-source per-band rendered stamps end-to-end.

    Inputs (per tile):
        bottleneck           [1, C_bn, H_bn, W_bn]  frozen V8 fused features
        vis_stem             [1, C_stem, H_vis, W_vis]
        source_positions_vis [N, 2]  VIS pixel coordinates
        positions_per_band   [N, B, 2]  WCS-projected per-band pixel coords
        tile_hw, fused_hw, vis_hw  spatial shapes

    Output: [N, B, S, S] positive stamps, sum-normalised per (source, band).
    """

    def __init__(
        self,
        n_bands: int = 4,
        hidden_ch: int = 256,
        stem_ch: int = 64,
        bottleneck_ctx_dim: int = 64,
        intrinsic_ch: int = 32,
        decoder_ch: int = 32,
        stamp_size: int = 31,
        bottleneck_window: int = 11,
        stem_window: int = 31,
        fused_pixel_scale: float = 0.4,
        vis_pixel_scale: float = 0.1,
        band_embed_dim: int = 16,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if stamp_size % 2 != 1:
            raise ValueError("stamp_size must be odd")
        if bottleneck_window % 2 != 1 or stem_window % 2 != 1:
            raise ValueError("feature windows must be odd")
        if stem_window != stamp_size:
            raise ValueError(
                "for the MVP stem_window and stamp_size must match so the conv "
                f"intrinsic map is the stamp size (got {stem_window} vs {stamp_size})"
            )

        self.n_bands = int(n_bands)
        self.stamp_size = int(stamp_size)
        self.bottleneck_window = int(bottleneck_window)
        self.stem_window = int(stem_window)
        self.fused_pixel_scale = float(fused_pixel_scale)
        self.vis_pixel_scale = float(vis_pixel_scale)
        self.eps = float(eps)

        # Spatial intrinsic profile from the VIS stem window — preserves the
        # full 31x31 structure so spiral arms / tails / bars can propagate.
        self.intrinsic_conv = nn.Sequential(
            nn.Conv2d(stem_ch, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, intrinsic_ch, 3, padding=1),
        )

        # Pooled bottleneck context (all-band multi-channel) as a conditioning
        # vector — lets the per-band decoder see colour / broadband context.
        self.bn_conv = nn.Sequential(
            ConvNeXtBlock(hidden_ch),
            nn.Conv2d(hidden_ch, bottleneck_ctx_dim, 1),
            nn.GELU(),
        )
        y = torch.linspace(-1, 1, bottleneck_window)
        x = torch.linspace(-1, 1, bottleneck_window)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        g = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2 * 0.4 ** 2))
        g = g / g.sum()
        self.register_buffer(
            "bn_gauss",
            g.view(1, 1, bottleneck_window, bottleneck_window),
            persistent=False,
        )

        self.band_embed = nn.Embedding(self.n_bands, band_embed_dim)
        cond_dim = band_embed_dim + 2 + 2 + bottleneck_ctx_dim  # band + tile_pos + sub_px + ctx
        self.film = nn.Linear(cond_dim, 2 * intrinsic_ch)

        self.decoder = nn.Sequential(
            nn.Conv2d(intrinsic_ch, decoder_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_ch, decoder_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_ch, 1, 1),
        )

        self.intrinsic_ch = int(intrinsic_ch)

    def forward(
        self,
        bottleneck: torch.Tensor,
        vis_stem: torch.Tensor,
        source_positions_vis: torch.Tensor,
        positions_per_band: torch.Tensor,
        tile_hw: Tuple[int, int],
        fused_hw: Tuple[int, int],
        vis_hw: Tuple[int, int],
    ) -> torch.Tensor:
        n = source_positions_vis.shape[0]
        b = self.n_bands
        S = self.stamp_size
        if positions_per_band.numel() and positions_per_band.shape[1] != b:
            raise ValueError(
                f"positions_per_band has {positions_per_band.shape[1]} bands, head expects {b}"
            )
        if n == 0:
            return torch.empty(
                0, b, S, S, device=vis_stem.device, dtype=vis_stem.dtype,
            )

        stem_windows = extract_local_windows(
            vis_stem, source_positions_vis, self.stem_window,
        )
        intrinsic = self.intrinsic_conv(stem_windows)  # [N, C_int, S, S]

        pos_bn = vis_px_to_bottleneck_px(
            source_positions_vis,
            self.vis_pixel_scale,
            self.fused_pixel_scale,
            fused_hw,
            vis_hw,
        )
        bn_windows = extract_local_windows(
            bottleneck, pos_bn, self.bottleneck_window,
        )
        bn_feat = self.bn_conv(bn_windows)
        bn_vec = (bn_feat * self.bn_gauss).sum(dim=(-2, -1))  # [N, D_ctx]

        H, W = float(tile_hw[0]), float(tile_hw[1])
        tile_pos = positions_per_band.clone().float()
        tile_pos[..., 0] = tile_pos[..., 0] / max(W - 1.0, 1.0)
        tile_pos[..., 1] = tile_pos[..., 1] / max(H - 1.0, 1.0)
        tile_pos = tile_pos.clamp(0.0, 1.0)
        sub_px = (positions_per_band - positions_per_band.round()).float()

        band_idx = torch.arange(b, device=intrinsic.device)
        band_e = self.band_embed(band_idx)  # [B, E]
        band_e_nb = band_e.unsqueeze(0).expand(n, b, -1)
        bn_nb = bn_vec.unsqueeze(1).expand(n, b, -1)
        cond = torch.cat([band_e_nb, tile_pos, sub_px, bn_nb], dim=-1)  # [N, B, cond_dim]

        film = self.film(cond)  # [N, B, 2*C_int]
        gamma = 1.0 + film[..., : self.intrinsic_ch]
        beta = film[..., self.intrinsic_ch :]

        intrinsic_nb = intrinsic.unsqueeze(1).expand(n, b, -1, -1, -1)
        mod = intrinsic_nb * gamma[..., None, None] + beta[..., None, None]
        mod = F.gelu(mod)

        flat = mod.reshape(n * b, self.intrinsic_ch, S, S)
        out = self.decoder(flat).reshape(n, b, S, S)
        out = F.softplus(out)
        sums = out.sum(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        return out / sums


def render_end_to_end_tile(
    tile: torch.Tensor,
    rms: torch.Tensor,
    positions_per_band: torch.Tensor,
    stamps: torch.Tensor,
    groups: Optional[Sequence[Sequence[int]]] = None,
    group_radius_px: float = 10.0,
    min_scene_size: int = 51,
    max_scene_size: int = 91,
    return_scenes: bool = False,
) -> Dict[str, object]:
    """Place pre-rendered per-band stamps in each scene, solve fluxes, compute chi2.

    No morphology and no PSF inputs — the stamps are the full convolved renders.
    """
    tile = tile.float()
    rms = rms.float()
    if tile.ndim != 3 or rms.shape != tile.shape:
        raise ValueError("tile and rms must have shape [B, H, W]")
    n_band = tile.shape[0]
    pos = _prepare_positions(positions_per_band.to(tile.device), n_band)
    n_src = pos.shape[0]
    if stamps.shape[:2] != (n_src, n_band):
        raise ValueError(
            f"stamps shape {tuple(stamps.shape)} must start with (N={n_src}, B={n_band})"
        )

    if groups is None:
        groups = build_neighbor_groups(pos[:, 0, :], radius_px=group_radius_px)
    groups = [list(map(int, g)) for g in groups]

    stamp_size = stamps.shape[-1]
    flux = torch.zeros(n_src, n_band, dtype=torch.float32, device=tile.device)
    chi2_dof = torch.full(
        (n_src, n_band), float("nan"), dtype=torch.float32, device=tile.device
    )
    group_id = torch.full((n_src,), -1, dtype=torch.long, device=tile.device)
    losses = []
    scenes: List[RenderedSceneResult] = []

    for gid, idx_list in enumerate(groups):
        idx = torch.tensor(idx_list, dtype=torch.long, device=tile.device)
        group_id[idx] = gid
        group_pos = pos[idx]
        center = group_pos.mean(dim=0)
        offset = (group_pos - center[None]).abs().amax().item()
        scene_size = _odd_at_least(
            min(max_scene_size, int(round(stamp_size + 2 * offset + 6))),
            min_scene_size,
        )
        scene_size = min(int(max_scene_size), scene_size)
        if scene_size % 2 == 0:
            scene_size -= 1

        data_patch, rms_patch = _extract_band_scene(tile, rms, center, scene_size)
        background = estimate_local_background(
            data_patch[None],
            inner_radius=max(2.0, scene_size * 0.35),
            outer_radius=max(3.0, scene_size * 0.48),
        )[0]
        data_sub = data_patch - background[:, None, None]
        variance = rms_patch.pow(2).clamp_min(1e-20)

        placed = place_templates_in_scene(
            stamps[idx],
            group_pos,
            center,
            scene_size=scene_size,
        )
        group_flux = _initial_flux_from_templates(data_sub, variance, placed)
        model = (placed * group_flux[:, :, None, None]).sum(dim=0) + background[:, None, None]
        resid = data_patch - model
        chi2_pix = resid.pow(2) / variance
        loss = chi2_pix.mean()
        losses.append(loss)

        per_band_dof = max(1, scene_size * scene_size - len(idx_list) - 1)
        group_chi2 = chi2_pix.sum(dim=(-2, -1)) / per_band_dof
        flux[idx] = group_flux
        chi2_dof[idx] = group_chi2[None].expand(len(idx_list), -1)

        if return_scenes:
            scenes.append(
                RenderedSceneResult(
                    group_id=gid,
                    indices=idx.detach().cpu(),
                    center_xy=center.detach().cpu(),
                    scene_size=scene_size,
                    data=data_patch.detach().cpu(),
                    model=model.detach().cpu(),
                    resid=resid.detach().cpu(),
                    background=background.detach().cpu(),
                    flux=group_flux.detach().cpu(),
                    chi2_dof=group_chi2.detach().cpu(),
                )
            )

    data_loss = torch.stack(losses).mean() if losses else tile.sum() * 0.0
    return {
        "loss_data": data_loss,
        "flux": flux,
        "chi2_dof": chi2_dof,
        "positions_px": pos,
        "groups": groups,
        "group_id": group_id,
        "scene_results": scenes,
    }


def rendered_head_loss(
    head: RenderedStampHead,
    bottleneck: torch.Tensor,
    vis_stem: torch.Tensor,
    source_positions_vis: torch.Tensor,
    positions_per_band: torch.Tensor,
    tile: torch.Tensor,
    rms: torch.Tensor,
    fused_hw: Tuple[int, int],
    vis_hw: Tuple[int, int],
    groups: Optional[Sequence[Sequence[int]]] = None,
    group_radius_px: float = 10.0,
    min_scene_size: int = 51,
    max_scene_size: int = 91,
    return_scenes: bool = False,
) -> Dict[str, object]:
    stamps = head(
        bottleneck=bottleneck,
        vis_stem=vis_stem,
        source_positions_vis=source_positions_vis,
        positions_per_band=positions_per_band,
        tile_hw=tile.shape[-2:],
        fused_hw=fused_hw,
        vis_hw=vis_hw,
    )
    render = render_end_to_end_tile(
        tile=tile,
        rms=rms,
        positions_per_band=positions_per_band,
        stamps=stamps,
        groups=groups,
        group_radius_px=group_radius_px,
        min_scene_size=min_scene_size,
        max_scene_size=max_scene_size,
        return_scenes=return_scenes,
    )
    loss = render["loss_data"]
    return {
        "loss": loss,
        "loss_data": loss,
        "stamps": stamps,
        **render,
    }


__all__ = [
    "RenderedStampHead",
    "RenderedSceneResult",
    "render_end_to_end_tile",
    "rendered_head_loss",
]
