"""Foundation-conditioned low-rank ePSF head.

This module implements the constrained head sketched in the discussion:

    ePSF = per-band base ePSF + small low-rank residual

The residual coefficients are predicted from frozen JAISP foundation features,
tile position, and band id. The output is always positive and unit-flux, and
the head can render the oversampled ePSF back to native pixels at a requested
sub-pixel phase.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # PYTHONPATH=models
    from astrometry2.latent_position_head import (
        extract_local_windows,
        vis_px_to_bottleneck_px,
    )
    from jaisp_foundation_v6 import ConvNeXtBlock
except ImportError:  # Package import from repo root
    from models.astrometry2.latent_position_head import (
        extract_local_windows,
        vis_px_to_bottleneck_px,
    )
    from models.jaisp_foundation_v6 import ConvNeXtBlock


RUBIN_BANDS = ["rubin_u", "rubin_g", "rubin_r", "rubin_i", "rubin_z", "rubin_y"]
EUCLID_BANDS = ["euclid_VIS", "euclid_Y", "euclid_J", "euclid_H"]
ALL_BANDS = RUBIN_BANDS + EUCLID_BANDS

BAND_PIXEL_SCALE_ARCSEC = {
    "rubin_u": 0.2,
    "rubin_g": 0.2,
    "rubin_r": 0.2,
    "rubin_i": 0.2,
    "rubin_z": 0.2,
    "rubin_y": 0.2,
    "euclid_VIS": 0.1,
    "euclid_Y": 0.1,
    "euclid_J": 0.1,
    "euclid_H": 0.1,
}

# Gaussian-equivalent core sigmas in mas. These are intentionally simple,
# per-band physical priors rather than empirical ePSF templates.
DEFAULT_CORE_SIGMA_MAS = {
    "rubin_u": 430.0,
    "rubin_g": 400.0,
    "rubin_r": 390.0,
    "rubin_i": 380.0,
    "rubin_z": 385.0,
    "rubin_y": 395.0,
    "euclid_VIS": 120.0,
    "euclid_Y": 190.0,
    "euclid_J": 195.0,
    "euclid_H": 200.0,
}


def _softplus_inverse(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.clamp_min(eps)
    return x + torch.log(-torch.expm1(-x))


def _normalise_unit_flux(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.flatten(-2).sum(dim=-1).clamp(min=eps).view(*x.shape[:-2], 1, 1)


def _crop_or_pad_centered_2d(x: torch.Tensor, size: int) -> torch.Tensor:
    """Return a centred ``size x size`` view/copy of a 2-D tensor."""
    if x.ndim != 2:
        raise ValueError("expected a 2-D tensor")
    h, w = x.shape
    out = x.new_zeros((size, size))
    src_y0 = max((h - size) // 2, 0)
    src_x0 = max((w - size) // 2, 0)
    src_y1 = min(src_y0 + size, h)
    src_x1 = min(src_x0 + size, w)
    copy_h = src_y1 - src_y0
    copy_w = src_x1 - src_x0
    dst_y0 = max((size - h) // 2, 0)
    dst_x0 = max((size - w) // 2, 0)
    out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = x[src_y0:src_y1, src_x0:src_x1]
    return out


def default_sigma_native(band: str) -> float:
    """Default Gaussian-equivalent core width in native pixels."""
    sigma_mas = float(DEFAULT_CORE_SIGMA_MAS.get(band, 400.0))
    px_scale = float(BAND_PIXEL_SCALE_ARCSEC.get(band, 0.2 if band.startswith("rubin") else 0.1))
    return sigma_mas / (px_scale * 1000.0)


def gaussian_epsf(
    psf_size: int,
    oversampling: int,
    sigma_native: float,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a centred, unit-flux oversampled Gaussian ePSF."""
    c = (int(psf_size) - 1) / 2.0
    coords = torch.arange(psf_size, device=device, dtype=dtype) - c
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    sigma_ovs = float(sigma_native) * float(oversampling)
    psf = torch.exp(-(xx * xx + yy * yy) / max(2.0 * sigma_ovs * sigma_ovs, 1e-12))
    return _normalise_unit_flux(psf)


def moffat_epsf(
    psf_size: int,
    oversampling: int,
    sigma_native: float,
    beta: float = 3.5,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a centred, unit-flux oversampled Moffat ePSF.

    ``sigma_native`` is interpreted as a Gaussian-equivalent core sigma. We
    convert it to a FWHM and choose the Moffat alpha that gives that FWHM.
    """
    beta = max(float(beta), 1.01)
    c = (int(psf_size) - 1) / 2.0
    coords = torch.arange(psf_size, device=device, dtype=dtype) - c
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    fwhm_ovs = 2.354820045 * float(sigma_native) * float(oversampling)
    alpha = fwhm_ovs / max(2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0), 1e-12)
    rr2 = xx * xx + yy * yy
    psf = torch.pow(1.0 + rr2 / max(alpha * alpha, 1e-12), -beta)
    return _normalise_unit_flux(psf)


def gaussian_epsf_bank(
    band_names: Sequence[str] = ALL_BANDS,
    psf_size: int = 99,
    oversampling: int = 5,
    sigmas_native: Optional[Mapping[str, float]] = None,
    sigmas_mas: Optional[Mapping[str, float]] = None,
    sigma_scale: float = 1.0,
) -> torch.Tensor:
    """Build ``[B, P, P]`` Gaussian fallback base ePSFs."""
    stamps = []
    for band in band_names:
        sigma = default_sigma_native(band)
        if sigmas_mas is not None and band in sigmas_mas:
            px_scale = float(BAND_PIXEL_SCALE_ARCSEC.get(band, 0.2 if band.startswith("rubin") else 0.1))
            sigma = float(sigmas_mas[band]) / (px_scale * 1000.0)
        if sigmas_native is not None and band in sigmas_native:
            sigma = float(sigmas_native[band])
        stamps.append(gaussian_epsf(psf_size, oversampling, sigma * float(sigma_scale)))
    return torch.stack(stamps, dim=0)


def analytic_epsf_bank(
    band_names: Sequence[str] = ALL_BANDS,
    psf_size: int = 99,
    oversampling: int = 5,
    *,
    kind: str = "gaussian",
    sigmas_mas: Optional[Mapping[str, float]] = None,
    sigma_scale: float = 1.0,
    moffat_beta: float = 3.5,
) -> torch.Tensor:
    """Build an analytic per-band base ePSF bank.

    ``kind='gaussian'`` is the clean default. ``kind='moffat'`` gives heavier
    wings while preserving the same Gaussian-equivalent core widths.
    """
    kind = str(kind).lower()
    if kind == "gaussian":
        return gaussian_epsf_bank(
            band_names=band_names,
            psf_size=psf_size,
            oversampling=oversampling,
            sigmas_mas=sigmas_mas,
            sigma_scale=sigma_scale,
        )
    if kind != "moffat":
        raise ValueError(f"unknown analytic ePSF kind {kind!r}")

    stamps = []
    for band in band_names:
        sigma = default_sigma_native(band)
        if sigmas_mas is not None and band in sigmas_mas:
            px_scale = float(BAND_PIXEL_SCALE_ARCSEC.get(band, 0.2 if band.startswith("rubin") else 0.1))
            sigma = float(sigmas_mas[band]) / (px_scale * 1000.0)
        stamps.append(moffat_epsf(psf_size, oversampling, sigma * float(sigma_scale), beta=moffat_beta))
    return torch.stack(stamps, dim=0)


def load_base_epsf_bank(
    checkpoint_path: str | Path,
    band_names: Sequence[str] = ALL_BANDS,
    psf_size: int = 99,
    oversampling: int = 5,
) -> torch.Tensor:
    """Load a per-band base ePSF bank.

    Supports the empirical ``PSFFieldEPSF`` checkpoint produced by
    ``psf_field_pca.py`` as well as checkpoints from this module.
    Missing bands fall back to approximate Gaussian bases.
    """
    blob = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    fallback = gaussian_epsf_bank(band_names, psf_size, oversampling)

    if "models" in blob:
        out = fallback.clone()
        models = blob["models"]
        for i, band in enumerate(band_names):
            if band not in models:
                continue
            data = torch.as_tensor(models[band]["data"], dtype=torch.float32)
            data = _crop_or_pad_centered_2d(data, psf_size)
            out[i] = _normalise_unit_flux(data.clamp_min(0.0))
        return out

    for key in ("base_epsf", "base_epsf_bank"):
        if key in blob:
            data = torch.as_tensor(blob[key], dtype=torch.float32)
            if data.ndim != 3:
                raise ValueError(f"{checkpoint_path} key {key!r} must have shape [B, P, P]")
            out = fallback.clone()
            n = min(out.shape[0], data.shape[0])
            for i in range(n):
                out[i] = _normalise_unit_flux(
                    _crop_or_pad_centered_2d(data[i], psf_size).clamp_min(0.0)
                )
            return out

    raise KeyError(
        f"{checkpoint_path} does not look like an empirical ePSF or "
        "FoundationEPSFHead checkpoint"
    )


class FoundationEPSFHead(nn.Module):
    """Low-rank residual ePSF head on top of frozen foundation features."""

    def __init__(
        self,
        psf_size: int = 99,
        oversampling: int = 5,
        band_names: Sequence[str] = ALL_BANDS,
        basis_rank: int = 8,
        hidden_ch: int = 256,
        stem_ch: int = 64,
        bottleneck_out: int = 128,
        stem_out: int = 64,
        bottleneck_window: int = 11,
        stem_window: int = 31,
        fused_pixel_scale: float = 0.4,
        vis_pixel_scale: float = 0.1,
        band_embed_dim: int = 16,
        mlp_hidden: int = 256,
        pos_freqs: int = 6,
        coeff_scale: float = 1.0,
        delta_scale: float = 0.35,
        basis_init: float = 1e-2,
        base_epsf: Optional[torch.Tensor] = None,
        train_base: bool = False,
        use_foundation_features: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        if psf_size % 2 != 1:
            raise ValueError("psf_size must be odd")
        if bottleneck_window % 2 != 1 or stem_window % 2 != 1:
            raise ValueError("feature windows must be odd")
        if basis_rank < 1:
            raise ValueError("basis_rank must be >= 1")

        self.psf_size = int(psf_size)
        self.oversampling = int(oversampling)
        self.band_names = list(band_names)
        self.band_to_idx = {b: i for i, b in enumerate(self.band_names)}
        self.basis_rank = int(basis_rank)
        self.bottleneck_window = int(bottleneck_window)
        self.stem_window = int(stem_window)
        self.fused_pixel_scale = float(fused_pixel_scale)
        self.vis_pixel_scale = float(vis_pixel_scale)
        self.coeff_scale = float(coeff_scale)
        self.delta_scale = float(delta_scale)
        self.basis_init = float(basis_init)
        self.use_foundation_features = bool(use_foundation_features)
        self.eps = float(eps)

        n_bands = len(self.band_names)
        if base_epsf is None:
            base_epsf = gaussian_epsf_bank(self.band_names, self.psf_size, self.oversampling)
        base_epsf = torch.as_tensor(base_epsf, dtype=torch.float32)
        if base_epsf.shape != (n_bands, self.psf_size, self.psf_size):
            raise ValueError(
                "base_epsf must have shape "
                f"{(n_bands, self.psf_size, self.psf_size)}, got {tuple(base_epsf.shape)}"
            )
        base_logits = _softplus_inverse(_normalise_unit_flux(base_epsf.clamp_min(self.eps)))
        if train_base:
            self.base_logits = nn.Parameter(base_logits)
        else:
            self.register_buffer("base_logits", base_logits)

        self.basis_logits = nn.Parameter(
            torch.randn(n_bands, self.basis_rank, self.psf_size, self.psf_size)
            * self.basis_init
        )

        feat_dim = 0
        if self.use_foundation_features:
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
            feat_dim = int(bottleneck_out) + int(stem_out)
            for name, ws in (("bn", self.bottleneck_window), ("stem", self.stem_window)):
                y = torch.linspace(-1, 1, ws)
                x = torch.linspace(-1, 1, ws)
                yy, xx = torch.meshgrid(y, x, indexing="ij")
                g = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2 * 0.4 ** 2))
                g = g / g.sum()
                self.register_buffer(f"{name}_gauss", g.view(1, 1, ws, ws), persistent=False)
        else:
            self.bn_conv = None
            self.stem_conv = None

        pos_freq_tensor = 2 ** torch.arange(int(pos_freqs), dtype=torch.float32) * math.pi
        self.register_buffer("pos_freq_tensor", pos_freq_tensor, persistent=False)
        pos_dim = 2 + 4 * int(pos_freqs)

        self.band_embed = nn.Embedding(n_bands, int(band_embed_dim))
        mlp_in = feat_dim + pos_dim + int(band_embed_dim)
        self.coeff_head = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, self.basis_rank),
        )
        nn.init.zeros_(self.coeff_head[-1].weight)
        nn.init.zeros_(self.coeff_head[-1].bias)

    @staticmethod
    def _gauss_pool(features: torch.Tensor, gauss: torch.Tensor) -> torch.Tensor:
        return (features * gauss).sum(dim=(-2, -1))

    def _pos_encoding(self, pos_norm: torch.Tensor) -> torch.Tensor:
        freq = self.pos_freq_tensor.to(device=pos_norm.device, dtype=pos_norm.dtype)
        proj = pos_norm.unsqueeze(-1) * freq.view(1, 1, -1)
        fourier = torch.cat([proj.sin(), proj.cos()], dim=-1).flatten(1)
        return torch.cat([pos_norm, fourier], dim=-1)

    def feature_vectors(
        self,
        bottleneck: torch.Tensor,
        vis_stem_features: torch.Tensor,
        source_positions_vis: torch.Tensor,
        fused_hw: Tuple[int, int],
        vis_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Extract pooled foundation features at VIS-frame source positions."""
        if not self.use_foundation_features:
            return torch.empty(
                source_positions_vis.shape[0],
                0,
                dtype=source_positions_vis.dtype,
                device=source_positions_vis.device,
            )
        if source_positions_vis.numel() == 0:
            feat_dim = self.coeff_head[0].in_features - self.band_embed.embedding_dim
            feat_dim -= 2 + 4 * int(self.pos_freq_tensor.numel())
            return torch.empty(0, feat_dim, dtype=bottleneck.dtype, device=bottleneck.device)

        positions_bn = vis_px_to_bottleneck_px(
            source_positions_vis,
            self.vis_pixel_scale,
            self.fused_pixel_scale,
            fused_hw,
            vis_hw,
        )
        bn_windows = extract_local_windows(bottleneck, positions_bn, self.bottleneck_window)
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

    def base_epsf(self) -> torch.Tensor:
        """Current positive, unit-flux per-band base ePSF bank."""
        return _normalise_unit_flux(F.softplus(self.base_logits))

    def forward(
        self,
        pos_norm: torch.Tensor,
        band_idx: torch.Tensor,
        *,
        bottleneck: Optional[torch.Tensor] = None,
        vis_stem_features: Optional[torch.Tensor] = None,
        source_positions_vis: Optional[torch.Tensor] = None,
        fused_hw: Optional[Tuple[int, int]] = None,
        vis_hw: Optional[Tuple[int, int]] = None,
        feature_vec: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        """Predict oversampled ePSFs.

        ``pos_norm`` is ``[N, 2]`` in ``[-1, 1]`` over the tile. ``band_idx`` is
        ``[N]`` in the local ``band_names`` order.
        """
        if pos_norm.ndim != 2 or pos_norm.shape[-1] != 2:
            raise ValueError("pos_norm must have shape [N, 2]")
        if band_idx.ndim != 1 or band_idx.shape[0] != pos_norm.shape[0]:
            raise ValueError("band_idx must have shape [N]")
        n = pos_norm.shape[0]

        if feature_vec is None:
            if self.use_foundation_features:
                missing = [
                    name for name, value in (
                        ("bottleneck", bottleneck),
                        ("vis_stem_features", vis_stem_features),
                        ("source_positions_vis", source_positions_vis),
                        ("fused_hw", fused_hw),
                        ("vis_hw", vis_hw),
                    )
                    if value is None
                ]
                if missing:
                    raise ValueError(
                        "foundation features requested but missing: " + ", ".join(missing)
                    )
                feature_vec = self.feature_vectors(
                    bottleneck,
                    vis_stem_features,
                    source_positions_vis,
                    fused_hw,
                    vis_hw,
                )
            else:
                feature_vec = torch.empty(n, 0, device=pos_norm.device, dtype=pos_norm.dtype)

        pos_feat = self._pos_encoding(pos_norm)
        band_feat = self.band_embed(band_idx)
        coeff_in = torch.cat([feature_vec, pos_feat, band_feat], dim=-1)
        coeff_raw = self.coeff_head(coeff_in)
        coeff = self.coeff_scale * torch.tanh(coeff_raw)

        basis = self.basis_logits[band_idx]  # [N, K, P, P]
        residual_logits = torch.einsum("nk,nkij->nij", coeff, basis)
        logits = self.base_logits[band_idx] + self.delta_scale * residual_logits
        epsf = _normalise_unit_flux(F.softplus(logits))
        epsf = epsf.unsqueeze(1)

        if return_dict:
            return {
                "epsf": epsf,
                "coeff": coeff,
                "coeff_raw": coeff_raw,
                "residual_logits": residual_logits,
                "features": feature_vec,
            }
        return epsf

    def render_at_native(
        self,
        psf_oversampled: torch.Tensor,
        frac_xy: torch.Tensor,
        stamp_size: int = 32,
    ) -> torch.Tensor:
        """Render oversampled ePSFs onto native pixels at sub-pixel phase."""
        bsz, _, p, _ = psf_oversampled.shape
        ovs = self.oversampling
        if p != self.psf_size:
            raise ValueError(f"expected oversampled side {self.psf_size}, got {p}")

        half = stamp_size // 2
        coords = (
            torch.arange(stamp_size, device=psf_oversampled.device, dtype=torch.float32)
            - float(half)
        )
        coords = coords.to(dtype=psf_oversampled.dtype)
        sub = (
            torch.arange(ovs, device=psf_oversampled.device, dtype=psf_oversampled.dtype)
            - (ovs - 1) / 2.0
        ) / ovs
        ys = coords.view(stamp_size, 1, 1, 1) + sub.view(1, ovs, 1, 1)
        xs = coords.view(1, 1, stamp_size, 1) + sub.view(1, 1, 1, ovs)

        ys_b = ys.unsqueeze(0).expand(bsz, -1, -1, -1, -1)
        xs_b = xs.unsqueeze(0).expand(bsz, -1, -1, -1, -1)
        ys_b = ys_b - frac_xy[:, 1].view(bsz, 1, 1, 1, 1)
        xs_b = xs_b - frac_xy[:, 0].view(bsz, 1, 1, 1, 1)

        centre = (p - 1) / 2.0
        ys_pix = ys_b * ovs + centre
        xs_pix = xs_b * ovs + centre
        ys_n = 2.0 * ys_pix / (p - 1) - 1.0
        xs_n = 2.0 * xs_pix / (p - 1) - 1.0
        grid = torch.stack(
            [
                xs_n.expand(bsz, stamp_size, ovs, stamp_size, ovs),
                ys_n.expand(bsz, stamp_size, ovs, stamp_size, ovs),
            ],
            dim=-1,
        ).reshape(bsz, stamp_size * ovs, stamp_size * ovs, 2)

        sampled = F.grid_sample(
            psf_oversampled,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.view(bsz, 1, stamp_size, ovs, stamp_size, ovs).sum(dim=(3, 5))

    def render_native(
        self,
        pos_norm: torch.Tensor,
        band_idx: torch.Tensor,
        frac_xy: torch.Tensor,
        stamp_size: int = 32,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(pos_norm, band_idx, return_dict=True, **kwargs)
        out["native"] = self.render_at_native(out["epsf"], frac_xy, stamp_size=stamp_size)
        return out

    def regularization(
        self,
        pred: Optional[Mapping[str, torch.Tensor]] = None,
        coeff_l2_weight: float = 1e-4,
        residual_l2_weight: float = 1e-5,
        basis_l2_weight: float = 1e-6,
        basis_tv_weight: float = 1e-5,
        epsf_tv_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Small regularizers that keep the residual head in calibration mode."""
        device = self.basis_logits.device
        loss = torch.zeros((), device=device)
        parts: Dict[str, torch.Tensor] = {}

        if basis_l2_weight > 0:
            v = self.basis_logits.pow(2).mean()
            parts["basis_l2"] = v
            loss = loss + float(basis_l2_weight) * v

        if basis_tv_weight > 0:
            dy = self.basis_logits[..., 1:, :] - self.basis_logits[..., :-1, :]
            dx = self.basis_logits[..., :, 1:] - self.basis_logits[..., :, :-1]
            v = dx.pow(2).mean() + dy.pow(2).mean()
            parts["basis_tv"] = v
            loss = loss + float(basis_tv_weight) * v

        if pred is not None and coeff_l2_weight > 0 and "coeff" in pred:
            v = pred["coeff"].pow(2).mean()
            parts["coeff_l2"] = v
            loss = loss + float(coeff_l2_weight) * v

        if pred is not None and residual_l2_weight > 0 and "residual_logits" in pred:
            v = pred["residual_logits"].pow(2).mean()
            parts["residual_l2"] = v
            loss = loss + float(residual_l2_weight) * v

        if pred is not None and epsf_tv_weight > 0 and "epsf" in pred:
            epsf = pred["epsf"]
            dy = epsf[..., 1:, :] - epsf[..., :-1, :]
            dx = epsf[..., :, 1:] - epsf[..., :, :-1]
            v = dx.abs().mean() + dy.abs().mean()
            parts["epsf_tv"] = v
            loss = loss + float(epsf_tv_weight) * v

        parts["loss"] = loss
        return parts


def load_foundation_epsf_head(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> FoundationEPSFHead:
    """Load a saved ``FoundationEPSFHead`` checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config", {}))
    head = FoundationEPSFHead(
        psf_size=cfg.get("psf_size", 99),
        oversampling=cfg.get("oversampling", 5),
        band_names=cfg.get("band_names", ALL_BANDS),
        basis_rank=cfg.get("basis_rank", 8),
        hidden_ch=cfg.get("hidden_ch", 256),
        stem_ch=cfg.get("stem_ch", 64),
        bottleneck_out=cfg.get("bottleneck_out", 128),
        stem_out=cfg.get("stem_out", 64),
        bottleneck_window=cfg.get("bottleneck_window", 11),
        stem_window=cfg.get("stem_window", 31),
        fused_pixel_scale=cfg.get("fused_pixel_scale", 0.4),
        vis_pixel_scale=cfg.get("vis_pixel_scale", 0.1),
        band_embed_dim=cfg.get("band_embed_dim", 16),
        mlp_hidden=cfg.get("mlp_hidden", 256),
        pos_freqs=cfg.get("pos_freqs", 6),
        coeff_scale=cfg.get("coeff_scale", 1.0),
        delta_scale=cfg.get("delta_scale", 0.35),
        basis_init=cfg.get("basis_init", 1e-2),
        train_base=cfg.get("train_base", False),
        use_foundation_features=cfg.get("use_foundation_features", True),
    )
    state = ckpt.get("head_state", ckpt.get("model", ckpt))
    head.load_state_dict(state)
    head.to(device).eval()
    return head


__all__ = [
    "FoundationEPSFHead",
    "load_foundation_epsf_head",
    "load_base_epsf_bank",
    "gaussian_epsf",
    "gaussian_epsf_bank",
    "ALL_BANDS",
    "RUBIN_BANDS",
    "EUCLID_BANDS",
]
