"""
Foundation-feature photometry head trained by image residuals.

This module is the learned counterpart to ``scarlet_like.py``.  It does not
optimize morphology pixels per scene.  Instead, a small trainable head reads
frozen V8 foundation features at CenterNet/astrometry-corrected source
positions and predicts a morphology refinement.  The renderer then solves
non-negative per-band fluxes and trains the head by the residual chi-square of
the local scene.

The first supported science grid is Euclid-native VIS/NISP at 0.1"/px.  Rubin
support should resample the learned VIS morphology to 0.2"/px or reproject
Rubin data to the VIS grid before sharing this exact morphology template.
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
        convolve_morphology_with_psf,
        normalise_templates,
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
        convolve_morphology_with_psf,
        normalise_templates,
        place_templates_in_scene,
    )
    from stamp_extractor import estimate_local_background


def _softplus_inverse(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp_min(eps)
    return x + torch.log(-torch.expm1(-x))


@dataclass
class LearnedSceneResult:
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


class FoundationScarletPhotometryHead(nn.Module):
    """
    Predict source morphologies from frozen V8 features.

    The head reads:

    - fused bottleneck windows for multi-band context
    - high-resolution VIS stem windows for morphology detail
    - VIS positive morphology initializers from image stamps

    It outputs positive, unit-sum morphology templates.  Fluxes are not learned
    directly; they are solved inside the differentiable renderer so the training
    signal remains pixel residual chi-square.
    """

    def __init__(
        self,
        hidden_ch: int = 256,
        stem_ch: int = 64,
        bottleneck_out: int = 128,
        stem_out: int = 64,
        mlp_hidden: int = 256,
        morph_size: int = 31,
        bottleneck_window: int = 11,
        stem_window: int = 31,
        fused_pixel_scale: float = 0.4,
        vis_pixel_scale: float = 0.1,
        delta_scale: float = 1.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        if morph_size % 2 != 1:
            raise ValueError("morph_size must be odd")
        if bottleneck_window % 2 != 1 or stem_window % 2 != 1:
            raise ValueError("feature windows must be odd")

        self.morph_size = int(morph_size)
        self.bottleneck_window = int(bottleneck_window)
        self.stem_window = int(stem_window)
        self.fused_pixel_scale = float(fused_pixel_scale)
        self.vis_pixel_scale = float(vis_pixel_scale)
        self.delta_scale = float(delta_scale)
        self.eps = float(eps)

        self.bn_conv = nn.Sequential(
            ConvNeXtBlock(hidden_ch),
            nn.Conv2d(hidden_ch, bottleneck_out, 1),
            nn.GELU(),
        )
        self.stem_conv = nn.Sequential(
            nn.Conv2d(stem_ch, stem_out, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_out, stem_out, 3, padding=1),
            nn.GELU(),
        )

        feat_dim = bottleneck_out + stem_out
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
        )
        self.morph_delta = nn.Linear(mlp_hidden, morph_size * morph_size)

        # Start exactly at the VIS initializer.  Early training is therefore a
        # residual improvement over the measured VIS morphology, not a random
        # generative model.
        nn.init.zeros_(self.morph_delta.weight)
        nn.init.zeros_(self.morph_delta.bias)

        for name, ws in (("bn", bottleneck_window), ("stem", stem_window)):
            y = torch.linspace(-1, 1, ws)
            x = torch.linspace(-1, 1, ws)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            g = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2 * 0.4 ** 2))
            g = g / g.sum()
            self.register_buffer(f"{name}_gauss", g.view(1, 1, ws, ws), persistent=False)

    @staticmethod
    def _gauss_pool(features: torch.Tensor, gauss: torch.Tensor) -> torch.Tensor:
        return (features * gauss).sum(dim=(-2, -1))

    def feature_vectors(
        self,
        bottleneck: torch.Tensor,
        vis_stem_features: torch.Tensor,
        source_positions_vis: torch.Tensor,
        fused_hw: Tuple[int, int],
        vis_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if source_positions_vis.numel() == 0:
            return torch.empty(
                0,
                self.trunk[0].in_features,
                dtype=bottleneck.dtype,
                device=bottleneck.device,
            )

        positions_bn = vis_px_to_bottleneck_px(
            source_positions_vis,
            self.vis_pixel_scale,
            self.fused_pixel_scale,
            fused_hw,
            vis_hw,
        )
        bn_windows = extract_local_windows(
            bottleneck,
            positions_bn,
            self.bottleneck_window,
        )
        bn_feat = self.bn_conv(bn_windows)
        bn_vec = self._gauss_pool(bn_feat, self.bn_gauss)

        stem_windows = extract_local_windows(
            vis_stem_features,
            source_positions_vis,
            self.stem_window,
        )
        stem_feat = self.stem_conv(stem_windows)
        stem_vec = self._gauss_pool(stem_feat, self.stem_gauss)
        return torch.cat([bn_vec, stem_vec], dim=1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        vis_stem_features: torch.Tensor,
        source_positions_vis: torch.Tensor,
        init_morphology: torch.Tensor,
        fused_hw: Tuple[int, int],
        vis_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Predict positive source morphologies.

        ``init_morphology`` must be [N, M, M], positive and unit-normalized.
        """
        if init_morphology.ndim != 3:
            raise ValueError("init_morphology must have shape [N, M, M]")
        if init_morphology.shape[-1] != self.morph_size:
            raise ValueError(
                f"init_morphology size {init_morphology.shape[-1]} does not "
                f"match head morph_size {self.morph_size}"
            )
        init_morph = normalise_templates(init_morphology, eps=self.eps)
        feat = self.feature_vectors(
            bottleneck,
            vis_stem_features,
            source_positions_vis,
            fused_hw,
            vis_hw,
        )
        hidden = self.trunk(feat)
        delta = torch.tanh(self.morph_delta(hidden)).reshape_as(init_morph)
        base_logits = _softplus_inverse(init_morph.clamp_min(self.eps), eps=self.eps)
        morphology = normalise_templates(
            F.softplus(base_logits + self.delta_scale * delta),
            eps=self.eps,
        )
        return {
            "morphology": morphology,
            "delta": delta,
            "features": feat,
        }


def morphology_regularization(
    morphology: torch.Tensor,
    init_morphology: torch.Tensor,
    tv_weight: float = 1e-4,
    anchor_weight: float = 2e-2,
) -> Dict[str, torch.Tensor]:
    morph = normalise_templates(morphology)
    init = normalise_templates(init_morphology)
    tv = (
        (morph[:, :, 1:] - morph[:, :, :-1]).abs().mean()
        + (morph[:, 1:, :] - morph[:, :-1, :]).abs().mean()
    )
    anchor = (morph - init).abs().sum(dim=(-2, -1)).mean()
    loss = float(tv_weight) * tv + float(anchor_weight) * anchor
    return {"loss": loss, "tv": tv, "anchor": anchor}


def render_learned_photometry_tile(
    tile: torch.Tensor,
    rms: torch.Tensor,
    positions_px: torch.Tensor,
    morphology: torch.Tensor,
    psfs: torch.Tensor,
    groups: Optional[Sequence[Sequence[int]]] = None,
    group_radius_px: float = 10.0,
    min_scene_size: int = 51,
    max_scene_size: int = 91,
    return_scenes: bool = False,
) -> Dict[str, object]:
    """
    Render a learned-morphology photometry model and solve fluxes.

    Parameters are intentionally parallel to ``fit_scarlet_like_tile``, except
    morphology is produced by a learned head and is not optimized here.
    """
    tile = tile.float()
    rms = rms.float()
    if tile.ndim != 3 or rms.shape != tile.shape:
        raise ValueError("tile and rms must have shape [B, H, W]")
    n_band = tile.shape[0]
    pos = _prepare_positions(positions_px.to(tile.device), n_band)
    n_src = pos.shape[0]
    morph = morphology.to(tile.device, dtype=torch.float32)
    if morph.shape[0] != n_src:
        raise ValueError("morphology and positions disagree on source count")
    psfs = psfs.to(tile.device, dtype=torch.float32)
    if psfs.shape[:2] != (n_src, n_band):
        raise ValueError("psfs must have shape [N, B, S, S]")

    if groups is None:
        groups = build_neighbor_groups(pos[:, 0, :], radius_px=group_radius_px)
    groups = [list(map(int, g)) for g in groups]

    flux = torch.zeros(n_src, n_band, dtype=torch.float32, device=tile.device)
    chi2_dof = torch.full((n_src, n_band), float("nan"), dtype=torch.float32, device=tile.device)
    group_id = torch.full((n_src,), -1, dtype=torch.long, device=tile.device)
    losses = []
    scenes: List[LearnedSceneResult] = []

    for gid, idx_list in enumerate(groups):
        idx = torch.tensor(idx_list, dtype=torch.long, device=tile.device)
        group_id[idx] = gid
        group_pos = pos[idx]
        center = group_pos.mean(dim=0)
        offset = (group_pos - center[None]).abs().amax().item()
        scene_size = _odd_at_least(
            min(max_scene_size, int(round(morph.shape[-1] + 2 * offset + 6))),
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

        templates = convolve_morphology_with_psf(morph[idx], psfs[idx])
        placed = place_templates_in_scene(
            templates,
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
                LearnedSceneResult(
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


def photometry_head_loss(
    head: FoundationScarletPhotometryHead,
    bottleneck: torch.Tensor,
    vis_stem: torch.Tensor,
    source_positions_vis: torch.Tensor,
    init_morphology: torch.Tensor,
    tile: torch.Tensor,
    rms: torch.Tensor,
    positions_px: torch.Tensor,
    psfs: torch.Tensor,
    fused_hw: Tuple[int, int],
    vis_hw: Tuple[int, int],
    groups: Optional[Sequence[Sequence[int]]] = None,
    group_radius_px: float = 10.0,
    min_scene_size: int = 51,
    max_scene_size: int = 91,
    tv_weight: float = 1e-4,
    anchor_weight: float = 2e-2,
    return_scenes: bool = False,
) -> Dict[str, object]:
    pred = head(
        bottleneck=bottleneck,
        vis_stem_features=vis_stem,
        source_positions_vis=source_positions_vis,
        init_morphology=init_morphology,
        fused_hw=fused_hw,
        vis_hw=vis_hw,
    )
    render = render_learned_photometry_tile(
        tile=tile,
        rms=rms,
        positions_px=positions_px,
        morphology=pred["morphology"],
        psfs=psfs,
        groups=groups,
        group_radius_px=group_radius_px,
        min_scene_size=min_scene_size,
        max_scene_size=max_scene_size,
        return_scenes=return_scenes,
    )
    reg = morphology_regularization(
        pred["morphology"],
        init_morphology,
        tv_weight=tv_weight,
        anchor_weight=anchor_weight,
    )
    loss = render["loss_data"] + reg["loss"]
    return {
        "loss": loss,
        "loss_data": render["loss_data"],
        "loss_reg": reg["loss"],
        "loss_tv": reg["tv"],
        "loss_anchor": reg["anchor"],
        "morphology": pred["morphology"],
        "delta": pred["delta"],
        **render,
    }


__all__ = [
    "FoundationScarletPhotometryHead",
    "LearnedSceneResult",
    "morphology_regularization",
    "photometry_head_loss",
    "render_learned_photometry_tile",
]
