"""
Scarlet-like residual-trained photometry head.

This module is intentionally separate from the matched-filter PSF photometry
path.  The model here is closer to scarlet:

    data_b(x, y) ~= background_b
                  + sum_i flux_{i,b} * [morphology_i (*) PSF_{i,b}](x, y)

where morphologies are non-negative and shared across bands, fluxes are
non-negative per band, and all source parameters in a local group are optimized
against the pixel residuals.

The first use case is per-scene optimization, not a globally trained neural
network.  Later, a learned head can predict the initial morphology logits or
regularize them, while this renderer/loss remains the differentiable photometry
surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .stamp_extractor import estimate_local_background, extract_stamps
except ImportError:
    from stamp_extractor import estimate_local_background, extract_stamps


def _softplus_inverse(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically stable inverse of softplus for positive initial values."""
    x = x.clamp_min(eps)
    return x + torch.log(-torch.expm1(-x))


def _odd_at_least(value: int, minimum: int) -> int:
    value = max(int(value), int(minimum))
    if value % 2 == 0:
        value += 1
    return value


def normalise_templates(templates: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Clip to non-negative values and L1-normalize each morphology template."""
    templates = templates.clamp_min(0.0)
    norm = templates.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return templates / norm


def build_neighbor_groups(
    positions_px: torch.Tensor,
    radius_px: float = 10.0,
) -> List[List[int]]:
    """
    Connected-component grouping by source distance.

    Sources closer than ``radius_px`` are connected; transitive chains become a
    single local scene.  This is the small, explicit analogue of scarlet's
    blend groups.
    """
    pos = positions_px.detach().float().cpu()
    n_src = int(pos.shape[0])
    if n_src == 0:
        return []

    parent = list(range(n_src))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    d = torch.cdist(pos, pos)
    pairs = torch.nonzero((d <= float(radius_px)) & (d > 0), as_tuple=False)
    for i, j in pairs.tolist():
        union(int(i), int(j))

    groups: Dict[int, List[int]] = {}
    for i in range(n_src):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def make_positive_morphology_templates(
    reference_image: torch.Tensor,
    positions_px: torch.Tensor,
    stamp_size: int = 31,
    bg_inner_radius: float = 11.0,
    bg_outer_radius: float = 15.0,
    smooth_sigma: float = 0.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Build non-negative morphology initializers from a reference image.

    The current default reference is Euclid VIS.  The stamps are background
    subtracted, clipped to positive flux, optionally smoothed, and normalized
    to unit sum.  These are initial morphologies; the scarlet-like head can
    train them further by minimizing residuals across all bands.
    """
    if reference_image.ndim != 2:
        raise ValueError("reference_image must have shape [H, W]")
    if stamp_size % 2 != 1:
        raise ValueError("stamp_size must be odd")

    device = reference_image.device
    ref = reference_image.to(device=device, dtype=torch.float32).unsqueeze(0)
    pos = positions_px.to(device=device, dtype=torch.float32)
    stamps = extract_stamps(ref, pos, stamp_size=stamp_size)[:, 0]
    bg = estimate_local_background(
        stamps[:, None],
        inner_radius=bg_inner_radius,
        outer_radius=bg_outer_radius,
    )[:, 0]
    templates = (stamps - bg[:, None, None]).clamp_min(0.0)

    if smooth_sigma > 0:
        # Small Gaussian smoothing can suppress VIS pixel noise before
        # optimization.  Kept local to avoid adding scipy/skimage here.
        radius = max(1, int(round(3.0 * smooth_sigma)))
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        kernel1 = torch.exp(-0.5 * (x / float(smooth_sigma)) ** 2)
        kernel1 = kernel1 / kernel1.sum().clamp_min(eps)
        kx = kernel1.view(1, 1, 1, -1)
        ky = kernel1.view(1, 1, -1, 1)
        t = templates[:, None]
        t = F.conv2d(t, kx, padding=(0, radius), groups=1)
        t = F.conv2d(t, ky, padding=(radius, 0), groups=1)
        templates = t[:, 0]

    templates = normalise_templates(templates, eps=eps)
    return {
        "templates": templates,
        "stamps": stamps,
        "background": bg,
    }


def convolve_morphology_with_psf(
    morphology: torch.Tensor,
    psfs: Optional[torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Convolve each source morphology with each source/band PSF.

    Parameters
    ----------
    morphology
        [K, M, M] non-negative, unit-sum source morphologies.
    psfs
        [K, B, P, P] unit-sum PSF kernels, or None to reuse morphology
        directly in every band.

    Returns
    -------
    templates
        [K, B, M, M] unit-sum rendered templates.
    """
    morphology = normalise_templates(morphology, eps=eps)
    if psfs is None:
        # The caller may expand the singleton band dimension when the number
        # of output bands is known.  Keeping this helper source-local makes it
        # useful for a single-band pure morphology model too.
        return morphology[:, None]

    if psfs.ndim != 4:
        raise ValueError("psfs must have shape [K, B, P, P]")
    k_src, n_band, psize, _ = psfs.shape
    if morphology.shape[0] != k_src:
        raise ValueError("morphology and psfs disagree on source count")
    if psize % 2 != 1:
        raise ValueError("PSF size must be odd")

    psfs = normalise_templates(psfs.reshape(k_src * n_band, psize, psize), eps=eps)
    psfs = psfs.reshape(k_src, n_band, psize, psize)

    out = []
    inp = morphology[:, None].reshape(1, k_src, *morphology.shape[-2:])
    pad = psize // 2
    for b in range(n_band):
        # conv2d is cross-correlation, so flip kernels for true convolution.
        weight = torch.flip(psfs[:, b:b + 1], dims=(-2, -1))
        y = F.conv2d(inp, weight, padding=pad, groups=k_src)[0]
        out.append(y)
    templates = torch.stack(out, dim=1)
    return normalise_templates(
        templates.reshape(k_src * n_band, *templates.shape[-2:]),
        eps=eps,
    ).reshape_as(templates)


def _prepare_positions(positions_px: torch.Tensor, n_bands: int) -> torch.Tensor:
    pos = positions_px.float()
    if pos.ndim == 2:
        if pos.shape[-1] != 2:
            raise ValueError("positions_px must have shape [N, 2] or [N, B, 2]")
        pos = pos[:, None, :].expand(-1, n_bands, -1)
    elif pos.ndim == 3:
        if pos.shape[-1] != 2:
            raise ValueError("positions_px must have shape [N, B, 2]")
        if pos.shape[1] != n_bands:
            raise ValueError(f"positions have {pos.shape[1]} bands, expected {n_bands}")
    else:
        raise ValueError("positions_px must have shape [N, 2] or [N, B, 2]")
    return pos.contiguous()


def _sampling_grid_for_templates(
    source_xy: torch.Tensor,
    scene_center_xy: torch.Tensor,
    scene_size: int,
    template_size: int,
) -> torch.Tensor:
    """Grid-sample coordinates that place source templates into one scene."""
    device = source_xy.device
    k_src = int(source_xy.shape[0])
    half_scene = (scene_size - 1) / 2.0
    half_template = (template_size - 1) / 2.0

    offsets = torch.arange(scene_size, dtype=torch.float32, device=device) - half_scene
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    patch_x = scene_center_xy[0] + xx
    patch_y = scene_center_xy[1] + yy

    local_x = patch_x[None] - source_xy[:, 0, None, None] + half_template
    local_y = patch_y[None] - source_xy[:, 1, None, None] + half_template
    x_norm = (local_x / max(template_size - 1, 1)) * 2.0 - 1.0
    y_norm = (local_y / max(template_size - 1, 1)) * 2.0 - 1.0
    return torch.stack([x_norm, y_norm], dim=-1).reshape(k_src, scene_size, scene_size, 2)


def place_templates_in_scene(
    templates: torch.Tensor,
    source_xy: torch.Tensor,
    scene_center_xy: torch.Tensor,
    scene_size: int,
) -> torch.Tensor:
    """
    Place source/band templates into a common scene patch.

    Parameters
    ----------
    templates
        [K, B, M, M] unit-sum templates.
    source_xy
        [K, B, 2] absolute pixel positions.
    scene_center_xy
        [B, 2] absolute pixel centers for the scene patch in each band.
    scene_size
        Odd scene side length.

    Returns
    -------
    placed
        [K, B, scene_size, scene_size] source templates sampled into the scene.
    """
    if scene_size % 2 != 1:
        raise ValueError("scene_size must be odd")
    k_src, n_band, template_size, _ = templates.shape
    placed = torch.empty(
        k_src,
        n_band,
        scene_size,
        scene_size,
        dtype=templates.dtype,
        device=templates.device,
    )
    for b in range(n_band):
        grid = _sampling_grid_for_templates(
            source_xy[:, b],
            scene_center_xy[b],
            scene_size=scene_size,
            template_size=template_size,
        )
        sampled = F.grid_sample(
            templates[:, b:b + 1],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0]
        placed[:, b] = sampled
    return placed


class ScarletLikePhotometryHead(nn.Module):
    """
    Trainable scarlet-like scene model.

    Each instance represents one local source group.  Parameters are scene
    parameters, not global network weights:

    - morphology logits [K, M, M], softplus-positive and unit-normalized
    - flux logits [K, B], softplus-positive
    - optional per-band background [B]

    Training minimizes weighted pixel residuals.  This is deliberately close
    to scarlet's optimization view, while remaining small enough to inspect in
    a notebook.
    """

    def __init__(
        self,
        source_xy: torch.Tensor,
        scene_center_xy: torch.Tensor,
        init_morphology: torch.Tensor,
        init_flux: torch.Tensor,
        init_background: Optional[torch.Tensor] = None,
        scene_size: int = 51,
        train_morphology: bool = True,
        train_background: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        if scene_size % 2 != 1:
            raise ValueError("scene_size must be odd")
        if init_morphology.ndim != 3:
            raise ValueError("init_morphology must have shape [K, M, M]")
        if init_flux.ndim != 2:
            raise ValueError("init_flux must have shape [K, B]")

        k_src, n_band = int(init_flux.shape[0]), int(init_flux.shape[1])
        if init_morphology.shape[0] != k_src:
            raise ValueError("init_morphology and init_flux disagree on source count")

        source_xy = _prepare_positions(source_xy, n_band)
        if source_xy.shape[0] != k_src:
            raise ValueError("source_xy and init_flux disagree on source count")
        if scene_center_xy.ndim == 1:
            scene_center_xy = scene_center_xy[None, :].expand(n_band, -1)
        if scene_center_xy.shape != (n_band, 2):
            raise ValueError(f"scene_center_xy must have shape {(n_band, 2)}")

        self.scene_size = int(scene_size)
        self.eps = float(eps)
        self.train_morphology = bool(train_morphology)
        self.train_background = bool(train_background)

        self.register_buffer("source_xy", source_xy.float())
        self.register_buffer("scene_center_xy", scene_center_xy.float())

        init_morph = normalise_templates(init_morphology.float(), eps=self.eps)
        self.register_buffer("init_morphology", init_morph.detach().clone())
        morph_logits = _softplus_inverse(init_morph.clamp_min(self.eps), eps=self.eps)
        if train_morphology:
            self.morph_logits = nn.Parameter(morph_logits)
        else:
            self.register_buffer("morph_logits", morph_logits)

        init_flux = init_flux.float().clamp_min(self.eps)
        self.flux_logits = nn.Parameter(_softplus_inverse(init_flux, eps=self.eps))

        if init_background is None:
            init_background = torch.zeros(n_band, dtype=torch.float32, device=init_flux.device)
        init_background = init_background.float()
        self.register_buffer("init_background", init_background.detach().clone())
        if train_background:
            self.background = nn.Parameter(init_background)
        else:
            self.register_buffer("background", init_background)

    @property
    def n_sources(self) -> int:
        return int(self.flux_logits.shape[0])

    @property
    def n_bands(self) -> int:
        return int(self.flux_logits.shape[1])

    def morphology(self) -> torch.Tensor:
        return normalise_templates(F.softplus(self.morph_logits), eps=self.eps)

    def flux(self) -> torch.Tensor:
        return F.softplus(self.flux_logits)

    def rendered_templates(self, psfs: Optional[torch.Tensor] = None) -> torch.Tensor:
        templates = convolve_morphology_with_psf(self.morphology(), psfs, eps=self.eps)
        if templates.shape[1] == 1 and self.n_bands != 1:
            templates = templates.expand(-1, self.n_bands, -1, -1)
        return templates

    def source_cube(self, psfs: Optional[torch.Tensor] = None) -> torch.Tensor:
        templates = self.rendered_templates(psfs)
        placed = place_templates_in_scene(
            templates,
            self.source_xy,
            self.scene_center_xy,
            scene_size=self.scene_size,
        )
        return placed * self.flux()[:, :, None, None]

    def forward(self, psfs: Optional[torch.Tensor] = None) -> torch.Tensor:
        scene = self.source_cube(psfs).sum(dim=0)
        return scene + self.background[:, None, None]

    def loss(
        self,
        data: torch.Tensor,
        variance: torch.Tensor,
        psfs: Optional[torch.Tensor] = None,
        tv_weight: float = 1e-4,
        center_weight: float = 1e-3,
        morph_anchor_weight: float = 1e-2,
        background_anchor_weight: float = 1e-2,
    ) -> Dict[str, torch.Tensor]:
        model = self(psfs)
        variance = variance.clamp_min(1e-20)
        resid = data - model
        chi2 = (resid.pow(2) / variance).mean()

        morph = self.morphology()
        tv = (
            (morph[:, :, 1:] - morph[:, :, :-1]).abs().mean()
            + (morph[:, 1:, :] - morph[:, :-1, :]).abs().mean()
        )

        size = morph.shape[-1]
        half = (size - 1) / 2.0
        y, x = torch.meshgrid(
            torch.arange(size, dtype=torch.float32, device=morph.device),
            torch.arange(size, dtype=torch.float32, device=morph.device),
            indexing="ij",
        )
        mx = (morph * x).sum(dim=(-2, -1))
        my = (morph * y).sum(dim=(-2, -1))
        center = ((mx - half).pow(2) + (my - half).pow(2)).mean() / max(half * half, 1.0)

        # Scarlet-style constraints keep morphology updates positive and local,
        # but a weak initializer anchor prevents flux/background degeneracies
        # from explaining residuals with unphysical diffuse templates.
        morph_anchor = (morph - self.init_morphology).abs().sum(dim=(-2, -1)).mean()
        bg_sigma = variance.sqrt().flatten(1).median(dim=1).values.clamp_min(self.eps)
        background_anchor = ((self.background - self.init_background) / bg_sigma).pow(2).mean()

        total = (
            chi2
            + float(tv_weight) * tv
            + float(center_weight) * center
            + float(morph_anchor_weight) * morph_anchor
            + float(background_anchor_weight) * background_anchor
        )
        return {
            "loss": total,
            "chi2": chi2,
            "tv": tv,
            "center": center,
            "morph_anchor": morph_anchor,
            "background_anchor": background_anchor,
            "model": model,
            "resid": resid,
        }


@dataclass
class SceneFitResult:
    flux: torch.Tensor
    morphology: torch.Tensor
    model: torch.Tensor
    resid: torch.Tensor
    background: torch.Tensor
    chi2_dof: torch.Tensor
    loss_history: torch.Tensor
    source_cube: torch.Tensor
    rendered_templates: torch.Tensor


def _initial_flux_from_templates(
    data_sub: torch.Tensor,
    variance: torch.Tensor,
    unit_source_templates: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Simultaneous weighted linear flux initializer for a local blend."""
    k_src, n_band, _, _ = unit_source_templates.shape
    flux = torch.zeros(k_src, n_band, dtype=data_sub.dtype, device=data_sub.device)
    var = variance.clamp_min(1e-20)
    for b in range(n_band):
        design = unit_source_templates[:, b].reshape(k_src, -1).transpose(0, 1)
        y = data_sub[b].reshape(-1)
        w = var[b].reshape(-1).reciprocal()
        ata = design.transpose(0, 1) @ (design * w[:, None])
        aty = design.transpose(0, 1) @ (y * w)
        ridge = eps * ata.diagonal().mean().clamp_min(1.0)
        ata = ata + torch.eye(k_src, dtype=ata.dtype, device=ata.device) * ridge
        try:
            sol = torch.linalg.solve(ata, aty)
        except RuntimeError:
            sol = torch.linalg.lstsq(ata, aty[:, None]).solution[:, 0]
        flux[:, b] = sol.clamp_min(eps)
    return flux


def fit_scarlet_like_scene(
    data: torch.Tensor,
    rms: torch.Tensor,
    source_xy: torch.Tensor,
    scene_center_xy: torch.Tensor,
    init_morphology: torch.Tensor,
    psfs: Optional[torch.Tensor] = None,
    scene_size: Optional[int] = None,
    init_flux: Optional[torch.Tensor] = None,
    init_background: Optional[torch.Tensor] = None,
    n_steps: int = 150,
    lr: float = 5e-2,
    tv_weight: float = 1e-4,
    center_weight: float = 1e-3,
    morph_anchor_weight: float = 1e-2,
    background_anchor_weight: float = 1e-2,
    train_morphology: bool = True,
    train_background: bool = False,
    verbose: bool = False,
) -> SceneFitResult:
    """
    Optimize one scarlet-like local scene.

    ``data`` and ``rms`` are [B, S, S] scene patches.  ``source_xy`` and
    ``scene_center_xy`` are absolute pixel coordinates in the same image grid.
    """
    if data.ndim != 3 or rms.shape != data.shape:
        raise ValueError("data and rms must both have shape [B, S, S]")
    n_band, patch_size, _ = data.shape
    scene_size = int(scene_size or patch_size)
    if scene_size != patch_size:
        raise ValueError("scene_size must match the data patch size")

    device = data.device
    source_xy = _prepare_positions(source_xy.to(device), n_band)
    if scene_center_xy.ndim == 1:
        scene_center_xy = scene_center_xy[None, :].expand(n_band, -1)
    scene_center_xy = scene_center_xy.to(device=device, dtype=torch.float32)
    init_morphology = init_morphology.to(device=device, dtype=torch.float32)
    if psfs is not None:
        psfs = psfs.to(device=device, dtype=torch.float32)

    if init_background is None:
        init_background = estimate_local_background(
            data[None],
            inner_radius=max(2.0, patch_size * 0.35),
            outer_radius=max(3.0, patch_size * 0.48),
        )[0]
    init_background = init_background.to(device=device, dtype=torch.float32)

    data_sub = data - init_background[:, None, None]
    variance = rms.pow(2).clamp_min(1e-20)

    if init_flux is None:
        init_templates = convolve_morphology_with_psf(init_morphology, psfs)
        if init_templates.shape[1] == 1 and n_band != 1:
            init_templates = init_templates.expand(-1, n_band, -1, -1)
        placed = place_templates_in_scene(
            init_templates,
            source_xy,
            scene_center_xy,
            scene_size=patch_size,
        )
        init_flux = _initial_flux_from_templates(data_sub, variance, placed)
    init_flux = init_flux.to(device=device, dtype=torch.float32).clamp_min(1e-8)

    head = ScarletLikePhotometryHead(
        source_xy=source_xy,
        scene_center_xy=scene_center_xy,
        init_morphology=init_morphology,
        init_flux=init_flux,
        init_background=init_background,
        scene_size=patch_size,
        train_morphology=train_morphology,
        train_background=train_background,
    ).to(device)

    opt = torch.optim.Adam(head.parameters(), lr=float(lr))
    history = []
    for step in range(int(n_steps)):
        opt.zero_grad(set_to_none=True)
        losses = head.loss(
            data,
            variance,
            psfs=psfs,
            tv_weight=tv_weight,
            center_weight=center_weight,
            morph_anchor_weight=morph_anchor_weight,
            background_anchor_weight=background_anchor_weight,
        )
        losses["loss"].backward()
        opt.step()
        history.append([
            float(losses["loss"].detach().cpu()),
            float(losses["chi2"].detach().cpu()),
            float(losses["tv"].detach().cpu()),
            float(losses["center"].detach().cpu()),
            float(losses["morph_anchor"].detach().cpu()),
            float(losses["background_anchor"].detach().cpu()),
        ])
        if verbose and (step == 0 or (step + 1) % 50 == 0):
            print(f"step {step + 1:4d}: loss={history[-1][0]:.4g} chi2={history[-1][1]:.4g}")

    with torch.no_grad():
        final = head.loss(data, variance, psfs=psfs, tv_weight=0.0, center_weight=0.0)
        model = final["model"]
        resid = final["resid"]
        per_band_dof = max(1, patch_size * patch_size - head.n_sources - 1)
        chi2_dof = (resid.pow(2) / variance).sum(dim=(-2, -1)) / per_band_dof
        source_cube = head.source_cube(psfs)
        rendered_templates = head.rendered_templates(psfs)
        return SceneFitResult(
            flux=head.flux().detach(),
            morphology=head.morphology().detach(),
            model=model.detach(),
            resid=resid.detach(),
            background=head.background.detach(),
            chi2_dof=chi2_dof.detach(),
            loss_history=torch.tensor(history, dtype=torch.float32),
            source_cube=source_cube.detach(),
            rendered_templates=rendered_templates.detach(),
        )


def _extract_band_scene(
    tile: torch.Tensor,
    rms: torch.Tensor,
    center_xy_by_band: torch.Tensor,
    scene_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract one scene patch per band, allowing band-specific centers."""
    n_band = tile.shape[0]
    patches = []
    rms_patches = []
    for b in range(n_band):
        patches.append(extract_stamps(tile[b:b + 1], center_xy_by_band[b:b + 1], scene_size)[0, 0])
        rms_patches.append(extract_stamps(rms[b:b + 1], center_xy_by_band[b:b + 1], scene_size)[0, 0])
    return torch.stack(patches, dim=0), torch.stack(rms_patches, dim=0)


def fit_scarlet_like_tile(
    tile: torch.Tensor,
    rms: torch.Tensor,
    positions_px: torch.Tensor,
    init_morphology: torch.Tensor,
    psfs: Optional[torch.Tensor] = None,
    groups: Optional[Sequence[Sequence[int]]] = None,
    group_radius_px: float = 10.0,
    min_scene_size: int = 51,
    max_scene_size: int = 91,
    n_steps: int = 120,
    lr: float = 5e-2,
    tv_weight: float = 1e-4,
    center_weight: float = 1e-3,
    morph_anchor_weight: float = 1e-2,
    background_anchor_weight: float = 1e-2,
    train_morphology: bool = True,
    train_background: bool = False,
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Fit scarlet-like local scenes for all sources in a tile.

    This is intentionally explicit and notebook-friendly.  It loops over local
    blend groups, optimizes each scene, and returns per-source fluxes plus the
    scene-level products needed for residual galleries.
    """
    tile = tile.float()
    rms = rms.float()
    if tile.ndim != 3 or rms.shape != tile.shape:
        raise ValueError("tile and rms must both have shape [B, H, W]")
    n_band = tile.shape[0]
    pos = _prepare_positions(positions_px.to(tile.device), n_band)
    n_src = pos.shape[0]
    morph = init_morphology.to(tile.device, dtype=torch.float32)
    if morph.shape[0] != n_src:
        raise ValueError("init_morphology and positions disagree on source count")
    if psfs is not None:
        psfs = psfs.to(tile.device, dtype=torch.float32)
        if psfs.shape[:2] != (n_src, n_band):
            raise ValueError("psfs must have shape [N, B, S, S]")

    if groups is None:
        groups = build_neighbor_groups(pos[:, 0, :], radius_px=group_radius_px)
    groups = [list(map(int, g)) for g in groups]

    flux = torch.zeros(n_src, n_band, dtype=torch.float32, device=tile.device)
    chi2_dof = torch.full((n_src, n_band), float("nan"), dtype=torch.float32, device=tile.device)
    group_id = torch.full((n_src,), -1, dtype=torch.long, device=tile.device)
    scene_results = []

    for gid, idx_list in enumerate(groups):
        idx = torch.tensor(idx_list, dtype=torch.long, device=tile.device)
        group_id[idx] = gid
        group_pos = pos[idx]
        center = group_pos.mean(dim=0)  # [B, 2]
        offset = (group_pos - center[None]).abs().amax().item()
        scene_size = _odd_at_least(
            min(max_scene_size, int(round(morph.shape[-1] + 2 * offset + 6))),
            min_scene_size,
        )
        scene_size = min(int(max_scene_size), scene_size)
        if scene_size % 2 == 0:
            scene_size -= 1

        data_patch, rms_patch = _extract_band_scene(tile, rms, center, scene_size)
        result = fit_scarlet_like_scene(
            data=data_patch,
            rms=rms_patch,
            source_xy=group_pos,
            scene_center_xy=center,
            init_morphology=morph[idx],
            psfs=psfs[idx] if psfs is not None else None,
            scene_size=scene_size,
            n_steps=n_steps,
            lr=lr,
            tv_weight=tv_weight,
            center_weight=center_weight,
            morph_anchor_weight=morph_anchor_weight,
            background_anchor_weight=background_anchor_weight,
            train_morphology=train_morphology,
            train_background=train_background,
            verbose=verbose,
        )
        flux[idx] = result.flux
        chi2_dof[idx] = result.chi2_dof[None].expand(len(idx), -1)
        scene_results.append({
            "group_id": gid,
            "indices": idx.detach().cpu(),
            "center_xy": center.detach().cpu(),
            "scene_size": scene_size,
            "result": result,
            "data": data_patch.detach().cpu(),
            "rms": rms_patch.detach().cpu(),
        })

    return {
        "flux": flux,
        "chi2_dof": chi2_dof,
        "snr_proxy": flux / rms.median(dim=-1).values.median(dim=-1).values.clamp_min(1e-8)[None, :],
        "positions_px": pos,
        "groups": groups,
        "group_id": group_id,
        "scene_results": scene_results,
    }


__all__ = [
    "ScarletLikePhotometryHead",
    "SceneFitResult",
    "build_neighbor_groups",
    "convolve_morphology_with_psf",
    "fit_scarlet_like_scene",
    "fit_scarlet_like_tile",
    "make_positive_morphology_templates",
    "normalise_templates",
    "place_templates_in_scene",
]
