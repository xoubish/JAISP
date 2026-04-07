"""
Astrometry Concordance Head for JAISP - Hybrid correlation + regression.

This version uses:
  1) Coarse token-level local cross-correlation (robust to larger shifts)
  2) Residual regression branch (sub-token refinement)
  3) Optional stem-space residual refinement (higher spatial resolution)

The coarse stage prevents the "predict ~0 everywhere" collapse seen with
pure direct regression when synthetic offsets are large.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pixel scales (arcsec / pixel)
# ---------------------------------------------------------------------------

PIXEL_SCALES = {
    "euclid_VIS": 0.1,
    "euclid_Y": 0.1,   # MER mosaics: resampled to VIS pixel scale
    "euclid_J": 0.1,   # MER mosaics: resampled to VIS pixel scale
    "euclid_H": 0.1,   # MER mosaics: resampled to VIS pixel scale
    "rubin_u": 0.2,
    "rubin_g": 0.2,
    "rubin_r": 0.2,
    "rubin_i": 0.2,
    "rubin_z": 0.2,
    "rubin_y": 0.2,
}


def interpolate_tokens(
    tokens: torch.Tensor,
    src_grid: Tuple[int, int],
    dst_grid: Tuple[int, int],
) -> torch.Tensor:
    """Resample token grid [B, N, D] from src_grid to dst_grid."""
    if src_grid == dst_grid:
        return tokens
    B, _, D = tokens.shape
    sh, sw = int(src_grid[0]), int(src_grid[1])
    dh, dw = int(dst_grid[0]), int(dst_grid[1])
    x = tokens.transpose(1, 2).contiguous().view(B, D, sh, sw)
    x = F.interpolate(x, size=(dh, dw), mode="bilinear", align_corners=False)
    return x.view(B, D, dh * dw).transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Coarse local correlation branch
# ---------------------------------------------------------------------------

class CoarseCorrelationBranch(nn.Module):
    """
    Estimate per-token coarse shifts via local cross-correlation + soft-argmax.

    Returns offsets in token units:
      dy_tokens: [B, 1, H, W]
      dx_tokens: [B, 1, H, W]
      confidence: [B, 1, H, W] local-match confidence
      global_confidence: [B, 1, 1, 1] global-match confidence
    """

    def __init__(
        self,
        search_radius: int = 3,
        softmax_temp: float = 0.1,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.search_radius = int(max(0, search_radius))
        self.learnable_temperature = bool(learnable_temperature)

        t0 = max(float(softmax_temp), 1e-3)
        if self.learnable_temperature:
            self._log_temperature = nn.Parameter(torch.tensor(math.log(t0), dtype=torch.float32))
        else:
            self.register_buffer("_fixed_temperature", torch.tensor(t0, dtype=torch.float32), persistent=False)

        disp = []
        r = self.search_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                disp.append((float(dy), float(dx)))
        dy_lut = torch.tensor([d[0] for d in disp], dtype=torch.float32).view(1, -1, 1, 1)
        dx_lut = torch.tensor([d[1] for d in disp], dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("dy_lut", dy_lut, persistent=False)
        self.register_buffer("dx_lut", dx_lut, persistent=False)

    @property
    def temperature(self) -> torch.Tensor:
        if self.learnable_temperature:
            return self._log_temperature.exp()
        return self._fixed_temperature

    def forward(self, rubin_2d: torch.Tensor, vis_2d: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, D, H, W = rubin_2d.shape
        r = self.search_radius
        K = (2 * r + 1) ** 2

        # Cosine-style correlation over channels improves stability.
        rubin_n = F.normalize(rubin_2d, dim=1, eps=1e-6)
        vis_n = F.normalize(vis_2d, dim=1, eps=1e-6)
        vis_pad = F.pad(vis_n, (r, r, r, r), mode="replicate")

        logits = []
        scale = 1.0 / math.sqrt(float(max(1, D)))
        for dy in range(-r, r + 1):
            yy0 = r + dy
            yy1 = yy0 + H
            for dx in range(-r, r + 1):
                xx0 = r + dx
                xx1 = xx0 + W
                shifted = vis_pad[:, :, yy0:yy1, xx0:xx1]
                corr = (rubin_n * shifted).sum(dim=1, keepdim=True) * scale
                logits.append(corr)

        logits = torch.cat(logits, dim=1)  # [B, K, H, W]
        tau = self.temperature.clamp_min(1e-3)
        probs_local = torch.softmax(logits / tau, dim=1)
        dy_local = (probs_local * self.dy_lut).sum(dim=1, keepdim=True)
        dx_local = (probs_local * self.dx_lut).sum(dim=1, keepdim=True)
        conf_local = probs_local.max(dim=1, keepdim=True).values

        # Global fallback: aggregate evidence over all positions, then soft-argmax.
        logits_global = logits.mean(dim=(2, 3), keepdim=True)  # [B, K, 1, 1]
        probs_global = torch.softmax(logits_global / tau, dim=1)
        dy_global = (probs_global * self.dy_lut).sum(dim=1, keepdim=True)  # [B,1,1,1]
        dx_global = (probs_global * self.dx_lut).sum(dim=1, keepdim=True)  # [B,1,1,1]
        conf_global = probs_global.max(dim=1, keepdim=True).values

        # If local confidence is near-uniform, trust global estimate more.
        uniform = 1.0 / float(max(1, K))
        local_weight = ((conf_local - uniform) / max(1e-6, 1.0 - uniform)).clamp(0.0, 1.0)

        dy = local_weight * dy_local + (1.0 - local_weight) * dy_global
        dx = local_weight * dx_local + (1.0 - local_weight) * dx_global

        return {
            "dy_tokens": dy,
            "dx_tokens": dx,
            "confidence": conf_local,
            "global_confidence": conf_global,
            "local_weight": local_weight,
            "logits": logits,
        }


# ---------------------------------------------------------------------------
# Global offset branch
# ---------------------------------------------------------------------------

class GlobalAffineBranch(nn.Module):
    """
    Predict a low-order affine field over token coordinates.

    Uses conv → adaptive pool → MLP so it sees spatial structure
    before pooling, not just raw channel statistics.
    """

    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),  # → [B, hidden, 4, 4]
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 16, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 6),  # dy = a0+a1*x+a2*y ; dx = b0+b1*x+b2*y
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        x = self.encoder(features)
        x = x.view(B, -1)
        return self.mlp(x)  # [B, 6]


# ---------------------------------------------------------------------------
# Local offset branch
# ---------------------------------------------------------------------------

class LocalOffsetBranch(nn.Module):
    """
    Predict spatially-varying component of the offset field.
    Large 5×5 kernels give each position context from neighbors.
    """

    def __init__(self, in_channels: int, hidden: int = 64, depth: int = 5):
        super().__init__()
        layers = [nn.Conv2d(in_channels, hidden, 5, padding=2), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(hidden, hidden, 5, padding=2), nn.GELU()]
        layers += [nn.Conv2d(hidden, 2, 5, padding=2)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


# ---------------------------------------------------------------------------
# Optional stem-space refinement (from user's edit)
# ---------------------------------------------------------------------------

class StemResidualRefineNet(nn.Module):
    def __init__(self, stem_channels: int = 64, hidden: int = 32, depth: int = 4):
        super().__init__()
        in_ch = 3 * stem_channels + 2
        layers = [nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU()]
        layers += [nn.Conv2d(hidden, 2, 3, padding=1)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rubin_feat, vis_feat, coarse_dra, coarse_ddec):
        x = torch.cat([rubin_feat, vis_feat, rubin_feat - vis_feat, coarse_dra, coarse_ddec], dim=1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Main head
# ---------------------------------------------------------------------------

class AstrometryConcordanceHead(nn.Module):
    """
    Predict astrometric offset field (ΔRA*, ΔDec) with a hybrid model.

    Token-space decomposition:
      coarse_tokens = correlation(rubin, vis)
      residual_tokens = global(features) + local(features)
      final_tokens = coarse_tokens + gain * residual_tokens
    """

    def __init__(
        self,
        embed_dim: int = 256,
        global_hidden: int = 128,
        local_hidden: int = 64,
        local_depth: int = 5,
        patch_size: int = 16,
        use_stem_refine: bool = False,
        match_dim: int = 64,
        stem_channels: int = 64,
        stem_hidden: int = 32,
        stem_depth: int = 4,
        stem_stride: int = 4,
        # Correlation knobs.
        search_radius: int = 3,
        softmax_temp: float = 0.1,
        learnable_temperature: bool = True,
        residual_gain_init: float = 1.0,
        # Legacy args for strict CLI/checkpoint compatibility.
        refine_hidden: int = 32,
        refine_depth: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_stem_refine = bool(use_stem_refine)
        self.stem_stride = max(1, int(stem_stride))
        self.match_dim = int(max(8, match_dim))

        self.coarse_branch = CoarseCorrelationBranch(
            search_radius=search_radius,
            softmax_temp=softmax_temp,
            learnable_temperature=learnable_temperature,
        )

        # Learnable modality adapters for correlation space.
        self.rubin_match_proj = nn.Conv2d(embed_dim, self.match_dim, kernel_size=1, bias=False)
        self.vis_match_proj = nn.Conv2d(embed_dim, self.match_dim, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.rubin_match_proj.weight, a=math.sqrt(5.0))
        nn.init.kaiming_uniform_(self.vis_match_proj.weight, a=math.sqrt(5.0))

        feat_dim = 3 * embed_dim  # concat(rubin, vis, rubin-vis)
        self.affine_branch = GlobalAffineBranch(feat_dim, hidden=global_hidden)
        self.local_branch = LocalOffsetBranch(feat_dim, hidden=local_hidden, depth=local_depth)

        # Keep residual contribution significant; do not allow it to collapse to ~0.
        self._residual_gain_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        with torch.no_grad():
            rg = float(min(max(residual_gain_init, 0.55), 1.45))
            x = min(max(rg - 0.5, 1e-3), 1.0 - 1e-3)
            self._residual_gain_logit.copy_(torch.tensor(math.log(x / (1.0 - x)), dtype=torch.float32))

        self.stem_refine = None
        if self.use_stem_refine:
            self.stem_refine = StemResidualRefineNet(
                stem_channels=stem_channels,
                hidden=stem_hidden,
                depth=stem_depth,
            )

    @property
    def residual_gain(self) -> torch.Tensor:
        return 0.5 + torch.sigmoid(self._residual_gain_logit)

    @property
    def temperature(self) -> torch.Tensor:
        # Kept for compatibility with existing logs/tools.
        return self.coarse_branch.temperature

    def forward(
        self,
        rubin_tokens: torch.Tensor,
        vis_tokens: torch.Tensor,
        rubin_grid: Tuple[int, int],
        vis_grid: Tuple[int, int],
        vis_image_hw: Tuple[int, int],
        vis_pixel_scale: float = 0.1,
        mesh_step: Optional[int] = None,
        rubin_stem: Optional[torch.Tensor] = None,
        vis_stem: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # --- Common token grid ---
        common_grid = (max(rubin_grid[0], vis_grid[0]), max(rubin_grid[1], vis_grid[1]))
        Hc, Wc = int(common_grid[0]), int(common_grid[1])

        rubin_common = interpolate_tokens(rubin_tokens, rubin_grid, common_grid)
        vis_common = interpolate_tokens(vis_tokens, vis_grid, common_grid)

        B, N, D = rubin_common.shape
        rubin_2d = rubin_common.transpose(1, 2).contiguous().view(B, D, Hc, Wc)
        vis_2d = vis_common.transpose(1, 2).contiguous().view(B, D, Hc, Wc)

        # --- Coarse match from local/global token correlation ---
        rubin_match = self.rubin_match_proj(rubin_2d)
        vis_match = self.vis_match_proj(vis_2d)
        coarse = self.coarse_branch(rubin_match, vis_match)
        coarse_dy = coarse["dy_tokens"]          # [B, 1, Hc, Wc]
        coarse_dx = coarse["dx_tokens"]          # [B, 1, Hc, Wc]
        coarse_conf = coarse["confidence"]       # [B, 1, Hc, Wc]
        global_conf = coarse["global_confidence"]
        local_weight = coarse["local_weight"]

        # --- Residual branch features ---
        features = torch.cat([rubin_2d, vis_2d, rubin_2d - vis_2d], dim=1)

        # --- Residual (global affine + local) ---
        affine_params = self.affine_branch(features)    # [B, 6]
        yy = torch.linspace(-1, 1, Hc, device=features.device, dtype=features.dtype).view(1, 1, Hc, 1)
        xx = torch.linspace(-1, 1, Wc, device=features.device, dtype=features.dtype).view(1, 1, 1, Wc)
        dy_aff = (
            affine_params[:, 0:1].unsqueeze(-1).unsqueeze(-1)
            + affine_params[:, 1:2].unsqueeze(-1).unsqueeze(-1) * xx
            + affine_params[:, 2:3].unsqueeze(-1).unsqueeze(-1) * yy
        )
        dx_aff = (
            affine_params[:, 3:4].unsqueeze(-1).unsqueeze(-1)
            + affine_params[:, 4:5].unsqueeze(-1).unsqueeze(-1) * xx
            + affine_params[:, 5:6].unsqueeze(-1).unsqueeze(-1) * yy
        )
        affine_field = torch.cat([dy_aff, dx_aff], dim=1)  # [B,2,Hc,Wc]
        local_offset = self.local_branch(features)          # [B,2,Hc,Wc]
        residual_tokens = affine_field + local_offset       # [B,2,Hc,Wc]

        # Final token offsets = coarse correlation + scaled residual regression.
        gain = self.residual_gain
        final_dy = coarse_dy + gain * residual_tokens[:, 0:1]
        final_dx = coarse_dx + gain * residual_tokens[:, 1:2]

        # --- Token units → arcseconds ---
        H_vis, W_vis = int(vis_image_hw[0]), int(vis_image_hw[1])
        sky_per_token_y = (H_vis * vis_pixel_scale) / Hc
        sky_per_token_x = (W_vis * vis_pixel_scale) / Wc

        coarse_dra = coarse_dx * sky_per_token_x
        coarse_ddec = coarse_dy * sky_per_token_y
        dra = final_dx * sky_per_token_x
        ddec = final_dy * sky_per_token_y

        # --- Optional stem refinement ---
        stem_residual = None
        if self.use_stem_refine and self.stem_refine is not None:
            if rubin_stem is None or vis_stem is None:
                raise ValueError("use_stem_refine=True requires rubin_stem and vis_stem.")
            refine_h = max(1, H_vis // self.stem_stride)
            refine_w = max(1, W_vis // self.stem_stride)
            rubin_refine = F.interpolate(rubin_stem, size=(refine_h, refine_w), mode="bilinear", align_corners=False)
            vis_refine = F.interpolate(vis_stem, size=(refine_h, refine_w), mode="bilinear", align_corners=False)
            coarse_dra = F.interpolate(dra, size=(refine_h, refine_w), mode="bilinear", align_corners=False)
            coarse_ddec = F.interpolate(ddec, size=(refine_h, refine_w), mode="bilinear", align_corners=False)
            stem_residual = self.stem_refine(rubin_refine, vis_refine, coarse_dra, coarse_ddec)
            dra = coarse_dra + stem_residual[:, 0:1]
            ddec = coarse_ddec + stem_residual[:, 1:2]

        # --- Optional mesh upsampling ---
        if mesh_step is not None and mesh_step > 0:
            mesh_h = max(1, H_vis // mesh_step)
            mesh_w = max(1, W_vis // mesh_step)
            dra = F.interpolate(dra, size=(mesh_h, mesh_w), mode="bilinear", align_corners=False)
            ddec = F.interpolate(ddec, size=(mesh_h, mesh_w), mode="bilinear", align_corners=False)

        return {
            "dra": dra,
            "ddec": ddec,
            "coarse_dra": coarse_dra,
            "coarse_ddec": coarse_ddec,
            "coarse_dy_tokens": coarse_dy,
            "coarse_dx_tokens": coarse_dx,
            "coarse_confidence": coarse_conf,
            "coarse_global_confidence": global_conf,
            "coarse_local_weight": local_weight,
            "residual_dy_tokens": residual_tokens[:, 0:1],
            "residual_dx_tokens": residual_tokens[:, 1:2],
            "global_offset_tokens": affine_field,
            "affine_params_tokens": affine_params.view(B, 6, 1, 1),
            "local_offset_tokens": local_offset,
            "stem_residual": stem_residual,
            # Compat keys for preview/loss code.
            "raw_dy": final_dy,
            "raw_dx": final_dx,
            "confidence": coarse_conf,
            "residual_gain": gain.detach().view(1, 1, 1, 1),
        }


class NonParametricConcordanceHead(nn.Module):
    """
    Parameter-free baseline:
      - interpolate Rubin/VIS tokens to a common grid
      - local correlation + soft-argmax for displacement
      - optional fixed spatial smoothing on the displacement field

    No learnable parameters are used.
    """

    def __init__(
        self,
        patch_size: int = 16,
        search_radius: int = 3,
        softmax_temp: float = 0.1,
        smooth_kernel: int = 3,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.smooth_kernel = int(max(0, smooth_kernel))
        self.coarse_branch = CoarseCorrelationBranch(
            search_radius=search_radius,
            softmax_temp=softmax_temp,
            learnable_temperature=False,
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.coarse_branch.temperature

    @property
    def residual_gain(self) -> torch.Tensor:
        # Compatibility key for logging.
        return torch.tensor(0.0, dtype=torch.float32, device=self.temperature.device)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        if self.smooth_kernel <= 1:
            return x
        k = self.smooth_kernel
        if k % 2 == 0:
            k += 1
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    def forward(
        self,
        rubin_tokens: torch.Tensor,
        vis_tokens: torch.Tensor,
        rubin_grid: Tuple[int, int],
        vis_grid: Tuple[int, int],
        vis_image_hw: Tuple[int, int],
        vis_pixel_scale: float = 0.1,
        mesh_step: Optional[int] = None,
        rubin_stem: Optional[torch.Tensor] = None,
        vis_stem: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del rubin_stem, vis_stem  # unused for non-parametric baseline

        common_grid = (max(rubin_grid[0], vis_grid[0]), max(rubin_grid[1], vis_grid[1]))
        Hc, Wc = int(common_grid[0]), int(common_grid[1])
        rubin_common = interpolate_tokens(rubin_tokens, rubin_grid, common_grid)
        vis_common = interpolate_tokens(vis_tokens, vis_grid, common_grid)
        B, _, D = rubin_common.shape
        rubin_2d = rubin_common.transpose(1, 2).contiguous().view(B, D, Hc, Wc)
        vis_2d = vis_common.transpose(1, 2).contiguous().view(B, D, Hc, Wc)

        coarse = self.coarse_branch(rubin_2d, vis_2d)
        dy = self._smooth(coarse["dy_tokens"])
        dx = self._smooth(coarse["dx_tokens"])
        conf = coarse["confidence"]

        H_vis, W_vis = int(vis_image_hw[0]), int(vis_image_hw[1])
        sky_per_token_y = (H_vis * vis_pixel_scale) / Hc
        sky_per_token_x = (W_vis * vis_pixel_scale) / Wc
        dra = dx * sky_per_token_x
        ddec = dy * sky_per_token_y

        if mesh_step is not None and mesh_step > 0:
            mesh_h = max(1, H_vis // mesh_step)
            mesh_w = max(1, W_vis // mesh_step)
            dra = F.interpolate(dra, size=(mesh_h, mesh_w), mode="bilinear", align_corners=False)
            ddec = F.interpolate(ddec, size=(mesh_h, mesh_w), mode="bilinear", align_corners=False)

        return {
            "dra": dra,
            "ddec": ddec,
            "coarse_dra": dra,
            "coarse_ddec": ddec,
            "coarse_dy_tokens": dy,
            "coarse_dx_tokens": dx,
            "coarse_confidence": conf,
            "coarse_global_confidence": coarse["global_confidence"],
            "coarse_local_weight": coarse["local_weight"],
            "residual_dy_tokens": torch.zeros_like(dy),
            "residual_dx_tokens": torch.zeros_like(dx),
            "global_offset_tokens": torch.zeros_like(torch.cat([dy, dx], dim=1)),
            "local_offset_tokens": torch.zeros_like(torch.cat([dy, dx], dim=1)),
            "stem_residual": None,
            "raw_dy": dy,
            "raw_dx": dx,
            "confidence": conf,
            "residual_gain": torch.zeros(1, 1, 1, 1, device=dra.device, dtype=dra.dtype),
        }
