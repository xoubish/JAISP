"""JAISP Foundation v10 — standalone fine-scale mixed-resolution foundation.

This file is a self-contained consolidation of the v6 building blocks, the
v7 stream architecture, and the v8 fine-scale model class. It does not
import from any older v6/v7/v8/v9 module — everything required to construct,
train, and load v10 lives here.

Architecture:
  - Per-band ``BandStem`` + ``InformationMap``.
  - Two instrument streams:
      * Rubin (0.2"/px, 6 bands) — concat+project fusion (``rubin_concat=True``
        is the v9/v10 default).
      * Euclid (0.1"/px, 4 bands: VIS/Y/J/H from MER mosaics) — concat+project
        fusion.
  - ``StreamEncoder`` per stream (auto-computed depth so each stream lands at
    the chosen ``fused_pixel_scale_arcsec`` — typically 0.4"/px for v10).
  - Cross-stream fusion + transformer bottleneck.
  - Per-stream ``TargetDecoder`` upsamples the bottleneck back to the target
    band's native resolution with FiLM band conditioning.

Loss objective (v10 default = ``charbonnier`` with ``core_l2_weight``):
  - Charbonnier per-pixel loss (sqrt(diff^2 + eps^2)) — L1-like for large
    residuals, L2-like near zero so source cores stay sharp.
  - Optional extra L2 penalty on high-info pixels (``core_l2_weight``)
    that explicitly rewards getting the peak right.
  - Multiplied by tile-mean RMS so noisy bands are not ignored.

Submodule attribute names (``self.encoder``, ``self.target_decoders``, etc.)
are identical to v8 so existing v10 checkpoints
(``jaisp_v10_warmstart/checkpoint_best.pt``) load via state-dict without
modification.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ============================================================
# Band metadata
# ============================================================

RUBIN_BANDS = ["rubin_u", "rubin_g", "rubin_r", "rubin_i", "rubin_z", "rubin_y"]
EUCLID_BANDS = ["euclid_VIS", "euclid_Y", "euclid_J", "euclid_H"]
ALL_BANDS = RUBIN_BANDS + EUCLID_BANDS

STREAM_ORDER = ["rubin", "euclid"]
STREAM_PIXEL_SCALES = {
    "rubin": 0.2,
    "euclid": 0.1,  # MER mosaics: all Euclid bands at 0.1"/px
}
STREAM_BAND_SLOTS = {
    "rubin": RUBIN_BANDS,
    "euclid": EUCLID_BANDS,
}


def band_group(band_name: str) -> str:
    if band_name in RUBIN_BANDS:
        return "rubin"
    if band_name in EUCLID_BANDS:
        return "euclid"
    raise KeyError(f"Unknown band: {band_name}")


# ============================================================
# Information map (signal-based pixel weighting)
# ============================================================

class InformationMap(nn.Module):
    """Signal-based pixel weighting: focuses loss on sources, edges, not blank sky.

    Uses an RMS-adaptive floor so that noisy bands (high RMS) retain a
    meaningful minimum weight on blank-sky pixels, penalising hallucinated
    sources that would otherwise go unpunished.
    """

    def __init__(
        self,
        snr_threshold: float = 2.0,
        min_weight: float = 0.001,
        adaptive_floor_scale: float = 0.3,
    ):
        super().__init__()
        self.snr_threshold = float(snr_threshold)
        self.min_weight = float(min_weight)
        self.adaptive_floor_scale = float(adaptive_floor_scale)
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sx)
        self.register_buffer("sobel_y", sx.transpose(-1, -2).contiguous())

    def forward(self, image: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
        x = image / (rms + 1e-10)
        snr_weight = torch.sigmoid((x.abs() - self.snr_threshold) * 2.0)
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-10)
        grad_max = grad.amax(dim=(2, 3), keepdim=True) + 1e-10
        grad_weight = grad / grad_max
        weights = torch.maximum(snr_weight, grad_weight * 0.5) ** 2
        mean_rms = rms.mean(dim=(2, 3), keepdim=True)
        adaptive_min = self.min_weight + torch.sigmoid(mean_rms - 1.0) * self.adaptive_floor_scale
        weights = weights.clamp(min=adaptive_min)
        return weights / (weights.sum(dim=(2, 3), keepdim=True) + 1e-10) * (image.shape[2] * image.shape[3])


# ============================================================
# Per-band stem
# ============================================================

class BandStem(nn.Module):
    """Per-band CNN stem.

    Noise-normalises input, clamps, then extracts local features. GroupNorm
    is used instead of BatchNorm to handle batch_size=1 cleanly.
    """

    def __init__(self, out_channels: int = 64, clamp_min: float = -10.0, clamp_max: float = 100.0):
        super().__init__()
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
        x = image / (rms + 1e-10)
        x = x.clamp(self.clamp_min, self.clamp_max)
        return self.net(x)


# ============================================================
# Building blocks: norm, ConvNeXt, downsample, transformer, FiLM
# ============================================================

class LayerNorm2d(nn.Module):
    """LayerNorm applied over the channel dim of a [B, C, H, W] tensor."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block: depthwise 7x7 conv, LayerNorm, channel-MLP."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        hidden = dim * expansion
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        x = x.permute(0, 3, 1, 2)
        return residual + x


class DownBlock(nn.Module):
    """Downsample 2x (LayerNorm + stride-2 conv) then N ConvNeXt blocks."""

    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2):
        super().__init__()
        self.downsample = nn.Sequential(
            LayerNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2),
        )
        self.blocks = nn.Sequential(*[ConvNeXtBlock(out_ch) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.downsample(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block; uses MultiheadAttention (Flash Attention in PyTorch 2)."""

    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: scale + shift features by a learned band embedding."""

    def __init__(self, num_bands: int, channels: int):
        super().__init__()
        self.embed = nn.Embedding(num_bands, channels * 2)
        nn.init.zeros_(self.embed.weight)

    def forward(self, x: torch.Tensor, band_idx: torch.Tensor) -> torch.Tensor:
        gb = self.embed(band_idx)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.view(x.shape[0], x.shape[1], 1, 1)
        beta = beta.view(x.shape[0], x.shape[1], 1, 1)
        return x * (1.0 + gamma) + beta


# ============================================================
# Stream fusion + encoder + decoder
# ============================================================

class StreamFuser(nn.Module):
    """Fuse per-band BandStem outputs within a stream.

    For streams with more than one band, fixed-slot concat (zero-fill missing
    bands) followed by a 1x1 projection preserves per-band information through
    the encoder. ``use_concat=False`` falls back to mean pooling.
    """

    def __init__(self, band_names: List[str], stem_ch: int, out_ch: int, use_concat: bool):
        super().__init__()
        self.band_names = list(band_names)
        self.band_to_slot = {b: i for i, b in enumerate(self.band_names)}
        self.n_slots = len(self.band_names)
        self.use_concat = use_concat and self.n_slots > 1

        if self.use_concat:
            self.proj = nn.Sequential(
                nn.Conv2d(self.n_slots * stem_ch, out_ch, kernel_size=1, bias=False),
                LayerNorm2d(out_ch),
                nn.GELU(),
            )
        self.stem_ch = stem_ch
        self.out_ch = out_ch

    def forward(self, band_feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not band_feats:
            raise ValueError("StreamFuser received no band features.")

        if not self.use_concat:
            feats = list(band_feats.values())
            return torch.stack(feats, dim=0).mean(dim=0)

        ref = next(iter(band_feats.values()))
        B, C, H, W = ref.shape
        slots = ref.new_zeros(B, self.n_slots * C, H, W)
        for band_name, feat in band_feats.items():
            idx = self.band_to_slot[band_name]
            slots[:, idx * C : (idx + 1) * C] = feat
        return self.proj(slots)


class StreamEncoder(nn.Module):
    """Per-stream ConvNeXt encoder before cross-stream fusion."""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        depth: int,
        blocks_per_stage: int = 2,
    ):
        super().__init__()
        stages = []
        cur_ch = in_ch
        for _ in range(depth):
            stages.append(DownBlock(cur_ch, hidden_ch, blocks_per_stage))
            cur_ch = hidden_ch
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = [x]
        for stage in self.stages:
            x = checkpoint(stage, x, use_reentrant=False)
            feats.append(x)
        return feats


class DecoderStage(nn.Module):
    """Interpolate to a target size, fuse skip features, then refine + FiLM-condition."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_bands: int, blocks: int = 2):
        super().__init__()
        self.up_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, kernel_size=1)
        self.blocks = nn.Sequential(*[ConvNeXtBlock(out_ch) for _ in range(blocks)])
        self.film = FiLM(num_bands, out_ch)

    def forward(
        self,
        x: torch.Tensor,
        band_idx: torch.Tensor,
        size: Tuple[int, int],
        skip: torch.Tensor = None,
    ) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = self.up_proj(x)
        if skip is not None:
            skip = F.interpolate(skip, size=size, mode="bilinear", align_corners=False)
            x = x + self.skip_proj(skip)
        x = self.blocks(x)
        return self.film(x, band_idx)


class TargetDecoder(nn.Module):
    """Target-resolution decoder without requiring same-stream context skips."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        stage_channels: List[int],
        num_bands: int,
        blocks_per_stage: int = 2,
    ):
        super().__init__()
        stages = []
        cur_ch = in_ch
        for out_ch in stage_channels:
            stages.append(
                DecoderStage(cur_ch, skip_ch, out_ch, num_bands=num_bands, blocks=blocks_per_stage)
            )
            cur_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.out_head = nn.Sequential(
            LayerNorm2d(cur_ch),
            nn.Conv2d(cur_ch, 1, kernel_size=1),
        )

    def stage_sizes(self, bottleneck_hw: Tuple[int, int], target_hw: Tuple[int, int]) -> List[Tuple[int, int]]:
        cur_h, cur_w = bottleneck_hw
        h_t, w_t = target_hw
        sizes = []
        n_stages = len(self.stages)
        for i in range(n_stages):
            if i == n_stages - 1:
                next_size = (h_t, w_t)
            else:
                next_size = (
                    min(h_t, cur_h * 2),
                    min(w_t, cur_w * 2),
                )
            sizes.append(next_size)
            cur_h, cur_w = next_size
        return sizes

    def forward(
        self,
        x: torch.Tensor,
        band_idx: torch.Tensor,
        target_hw: Tuple[int, int],
        skip_maps: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_t, w_t = target_hw
        stage_sizes = self.stage_sizes(x.shape[-2:], (h_t, w_t))
        if skip_maps is None:
            skip_maps = [None] * len(stage_sizes)
        for stage, next_size, skip in zip(self.stages, stage_sizes, skip_maps):
            if self.training:
                x = checkpoint(stage, x, band_idx, next_size, skip, use_reentrant=False)
            else:
                x = stage(x, band_idx, next_size, skip=skip)
        return self.out_head(x)


# ============================================================
# Auto depth / decoder channel computation
# ============================================================

def compute_stream_depths(fused_pixel_scale: float) -> Dict[str, int]:
    """Compute stream encoder depths to reach the target fused scale.

    Each stride-2 ConvNeXt DownBlock doubles the effective pixel scale.
    ``depth = round(log2(fused_pixel_scale / instrument_pixel_scale))``.

    Examples::
        0.8"/px → Rubin depth=2 (0.2→0.4→0.8), Euclid depth=3 (0.1→0.2→0.4→0.8)
        0.4"/px → Rubin depth=1 (0.2→0.4),     Euclid depth=2 (0.1→0.2→0.4)
        0.2"/px → Rubin depth=0 (stays at 0.2), Euclid depth=1 (0.1→0.2)
    """
    depths = {}
    for stream, base_scale in STREAM_PIXEL_SCALES.items():
        d = max(1, round(math.log2(fused_pixel_scale / base_scale)))
        depths[stream] = d
    return depths


def decoder_stage_channels(stream_depth: int, hidden_ch: int) -> List[int]:
    """Derive TargetDecoder stage channels from encoder depth.

    Examples (hidden_ch=256):
        depth=1 → [128]
        depth=2 → [256, 128]
        depth=3 → [256, 128, 64]
    """
    if stream_depth <= 1:
        return [hidden_ch // 2]
    if stream_depth == 2:
        return [hidden_ch, hidden_ch // 2]
    channels = [hidden_ch]
    ch = hidden_ch
    for _ in range(stream_depth - 1):
        ch = max(ch // 2, 64)
        channels.append(ch)
    return channels


# ============================================================
# V10 Encoder
# ============================================================

class JAISPMixedEncoderV10(nn.Module):
    """Mixed-resolution encoder with configurable stream depths.

    Submodule attribute names match V8 so existing v10 checkpoints load via
    state-dict without remapping.
    """

    def __init__(
        self,
        band_names: List[str],
        stem_ch: int = 64,
        hidden_ch: int = 256,
        blocks_per_stage: int = 2,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        fused_pixel_scale_arcsec: float = 0.4,
        stream_branch_depths: Optional[Dict[str, int]] = None,
        rubin_concat: bool = True,
    ):
        super().__init__()
        self.band_names = list(band_names)
        self.stem_ch = stem_ch
        self.hidden_ch = hidden_ch
        self.fused_pixel_scale_arcsec = float(fused_pixel_scale_arcsec)
        self.rubin_concat = bool(rubin_concat)

        if stream_branch_depths is None:
            stream_branch_depths = compute_stream_depths(fused_pixel_scale_arcsec)
        self.stream_branch_depths = dict(stream_branch_depths)

        self.stems = nn.ModuleDict({b: BandStem(stem_ch) for b in band_names})
        self.info_maps = nn.ModuleDict({b: InformationMap() for b in band_names})

        self.stream_fusers = nn.ModuleDict({
            "rubin":  StreamFuser(RUBIN_BANDS,  stem_ch, stem_ch, use_concat=self.rubin_concat),
            "euclid": StreamFuser(EUCLID_BANDS, stem_ch, stem_ch, use_concat=True),
        })

        self.stream_encoders = nn.ModuleDict({
            stream: StreamEncoder(
                in_ch=stem_ch,
                hidden_ch=hidden_ch,
                depth=self.stream_branch_depths[stream],
                blocks_per_stage=blocks_per_stage,
            )
            for stream in STREAM_ORDER
        })
        self.skip_projs = nn.ModuleDict({
            stream: nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(stem_ch if level == 0 else hidden_ch, hidden_ch, kernel_size=1),
                    LayerNorm2d(hidden_ch),
                    nn.GELU(),
                )
                for level in range(self.stream_branch_depths[stream] + 1)
            ])
            for stream in STREAM_ORDER
        })
        self.stream_embeddings = nn.Embedding(len(STREAM_ORDER), hidden_ch)

        self.transformer = nn.ModuleList(
            [TransformerBlock(hidden_ch, transformer_heads) for _ in range(transformer_depth)]
        )
        self.transformer_norm = nn.LayerNorm(hidden_ch)

    @staticmethod
    def _make_2d_sincos(H: int, W: int, dim: int, device: torch.device) -> torch.Tensor:
        assert dim % 4 == 0
        d = dim // 2
        freq = torch.pow(10000.0, -torch.arange(0, d, 2, device=device, dtype=torch.float32) / d)
        y_pos = torch.arange(H, device=device, dtype=torch.float32)
        x_pos = torch.arange(W, device=device, dtype=torch.float32)
        pe_y = torch.cat([torch.outer(y_pos, freq).sin(), torch.outer(y_pos, freq).cos()], dim=-1)
        pe_x = torch.cat([torch.outer(x_pos, freq).sin(), torch.outer(x_pos, freq).cos()], dim=-1)
        pe = torch.cat([
            pe_y.unsqueeze(1).expand(-1, W, -1),
            pe_x.unsqueeze(0).expand(H, -1, -1),
        ], dim=-1)
        return pe.reshape(1, H * W, dim)

    def _estimate_fused_hw(self, context_images: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        heights, widths = [], []
        for band, img in context_images.items():
            scale = STREAM_PIXEL_SCALES[band_group(band)]
            heights.append(int(round(img.shape[-2] * scale / self.fused_pixel_scale_arcsec)))
            widths.append(int(round(img.shape[-1] * scale / self.fused_pixel_scale_arcsec)))
        h = max(1, int(round(sum(heights) / max(1, len(heights)))))
        w = max(1, int(round(sum(widths) / max(1, len(widths)))))
        return h, w

    @staticmethod
    def _stream_feature_scales(stream: str, n_levels: int) -> List[float]:
        base = STREAM_PIXEL_SCALES[stream]
        return [base * (2 ** level) for level in range(n_levels)]

    @staticmethod
    def _closest_scale_index(scales: List[float], desired_scale: float) -> int:
        desired_scale = max(float(desired_scale), 1e-6)
        return min(
            range(len(scales)),
            key=lambda idx: abs(math.log2(max(scales[idx], 1e-6) / desired_scale)),
        )

    def build_target_skips(
        self,
        stream_pyramids: Dict[str, List[torch.Tensor]],
        target_band: str,
        target_hw: Tuple[int, int],
        stage_sizes: List[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        target_stream = band_group(target_band)
        target_scale = STREAM_PIXEL_SCALES[target_stream]
        target_h, target_w = target_hw
        skip_maps = []
        for size in stage_sizes:
            stage_h, stage_w = size
            desired_scale = 0.5 * (
                target_scale * (target_h / max(stage_h, 1))
                + target_scale * (target_w / max(stage_w, 1))
            )
            fused = []
            for stream_idx, stream in enumerate(STREAM_ORDER):
                pyramids = stream_pyramids.get(stream)
                if not pyramids:
                    continue
                scales = self._stream_feature_scales(stream, len(pyramids))
                level_idx = self._closest_scale_index(scales, desired_scale)
                feat = self.skip_projs[stream][level_idx](pyramids[level_idx])
                feat = F.interpolate(feat, size=size, mode="bilinear", align_corners=False)
                feat = feat + self.stream_embeddings.weight[stream_idx].view(1, -1, 1, 1)
                fused.append(feat)
            skip = torch.stack(fused, dim=0).mean(dim=0) if fused else None
            skip_maps.append(skip)
        return skip_maps

    def forward(
        self,
        context_images: Dict[str, torch.Tensor],
        context_rms: Dict[str, torch.Tensor],
    ) -> Dict:
        stem_feats = {stream: {} for stream in STREAM_ORDER}
        for band, img in context_images.items():
            rms = context_rms[band]
            stream = band_group(band)
            stem_feats[stream][band] = self.stems[band](img, rms)

        fused_hw = self._estimate_fused_hw(context_images)
        stream_maps = {}
        stream_pyramids = {}
        stream_encoded = []

        for stream_idx, stream in enumerate(STREAM_ORDER):
            band_feats = stem_feats[stream]
            if not band_feats:
                continue
            x = self.stream_fusers[stream](band_feats)
            pyramids = self.stream_encoders[stream](x)
            stream_pyramids[stream] = pyramids
            latent = pyramids[-1]
            latent = F.interpolate(latent, size=fused_hw, mode="bilinear", align_corners=False)
            latent = latent + self.stream_embeddings.weight[stream_idx].view(1, -1, 1, 1)
            stream_maps[stream] = latent
            stream_encoded.append(latent)

        if not stream_encoded:
            raise ValueError("No context streams available.")

        x = torch.stack(stream_encoded, dim=0).mean(dim=0)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        pe = self._make_2d_sincos(H, W, C, x.device)
        tokens = tokens + pe
        for blk in self.transformer:
            tokens = blk(tokens)
        tokens = self.transformer_norm(tokens)
        bottleneck = tokens.transpose(1, 2).view(B, C, H, W)

        return {
            "bottleneck": bottleneck,
            "stream_maps": stream_maps,
            "stream_pyramids": stream_pyramids,
            "fused_hw": fused_hw,
        }


# ============================================================
# V10 Foundation Model
# ============================================================

class JAISPFoundationV10(nn.Module):
    """Fine-scale mixed-resolution foundation with v10 loss-shaping.

    Default loss is Charbonnier with a small core-L2 term — these are the
    settings used to train ``jaisp_v10_warmstart``. Pass ``loss_type='l1'``
    and ``core_l2_weight=0`` to recover v8/v9 behaviour.

    Submodule attribute names (``self.encoder``, ``self.target_decoders``)
    match v8 so existing v10 checkpoints load via state-dict.
    """

    def __init__(
        self,
        band_names: List[str] = ALL_BANDS,
        stem_ch: int = 64,
        hidden_ch: int = 256,
        blocks_per_stage: int = 2,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        fused_pixel_scale_arcsec: float = 0.4,
        rubin_concat: bool = True,
        loss_type: str = "charbonnier",
        charbonnier_eps: float = 1e-3,
        core_l2_weight: float = 0.5,
        core_info_threshold: float = 0.5,
    ):
        super().__init__()
        self.band_names = list(band_names)
        self.band_to_idx = {b: i for i, b in enumerate(self.band_names)}
        self.rubin_concat = bool(rubin_concat)

        self.loss_type = str(loss_type).lower()
        if self.loss_type not in ("l1", "charbonnier"):
            raise ValueError(f"loss_type must be 'l1' or 'charbonnier', got {loss_type!r}")
        self.charbonnier_eps = float(charbonnier_eps)
        self.core_l2_weight = float(core_l2_weight)
        self.core_info_threshold = float(core_info_threshold)

        stream_depths = compute_stream_depths(fused_pixel_scale_arcsec)

        self.encoder = JAISPMixedEncoderV10(
            band_names=self.band_names,
            stem_ch=stem_ch,
            hidden_ch=hidden_ch,
            blocks_per_stage=blocks_per_stage,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
            fused_pixel_scale_arcsec=fused_pixel_scale_arcsec,
            stream_branch_depths=stream_depths,
            rubin_concat=self.rubin_concat,
        )

        num_bands = len(self.band_names)
        self.target_decoders = nn.ModuleDict({
            stream: TargetDecoder(
                hidden_ch, hidden_ch,
                decoder_stage_channels(stream_depths[stream], hidden_ch),
                num_bands, blocks_per_stage,
            )
            for stream in STREAM_ORDER
        })

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"JAISPFoundationV10: {n_params/1e6:.1f}M trainable parameters")
        print(f"  stem_ch={stem_ch}, hidden_ch={hidden_ch}, "
              f"fused_scale={fused_pixel_scale_arcsec:.2f}\"/px")
        print(f"  stream_depths={stream_depths}")
        print(f"  rubin_concat={self.rubin_concat}  "
              f"({'symmetric concat fusion' if self.rubin_concat else 'mean fusion'})")
        print(f"  loss_type={self.loss_type}  charbonnier_eps={self.charbonnier_eps:g}  "
              f"core_l2_weight={self.core_l2_weight:g}  core_info_thr={self.core_info_threshold:g}")
        for stream in STREAM_ORDER:
            chs = decoder_stage_channels(stream_depths[stream], hidden_ch)
            print(f"  {stream} decoder channels: {chs}")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(
        self,
        context_images: Dict[str, torch.Tensor],
        context_rms: Dict[str, torch.Tensor],
    ) -> Dict:
        return self.encoder(context_images, context_rms)

    def forward(
        self,
        context_images: Dict[str, torch.Tensor],
        context_rms: Dict[str, torch.Tensor],
        target_band: str,
        target_image: torch.Tensor,
        target_rms: torch.Tensor,
    ) -> Dict:
        enc_out = self.encoder(context_images, context_rms)

        B = target_image.shape[0]
        band_idx = torch.full(
            (B,), self.band_to_idx[target_band],
            dtype=torch.long, device=target_image.device,
        )
        target_stream = band_group(target_band)
        decoder = self.target_decoders[target_stream]
        stage_sizes = decoder.stage_sizes(
            enc_out["bottleneck"].shape[-2:], target_image.shape[-2:],
        )
        skip_maps = self.encoder.build_target_skips(
            enc_out["stream_pyramids"], target_band,
            target_image.shape[-2:], stage_sizes,
        )
        pred = decoder(
            enc_out["bottleneck"], band_idx,
            target_image.shape[-2:], skip_maps=skip_maps,
        )

        target_norm = (target_image / (target_rms + 1e-10)).clamp(-10.0, 100.0)
        info_w = self.encoder.info_maps[target_band](target_image, target_rms)
        pixel_diff = pred - target_norm

        if self.loss_type == "charbonnier":
            per_pixel = torch.sqrt(pixel_diff * pixel_diff + self.charbonnier_eps ** 2)
        else:
            per_pixel = pixel_diff.abs()
        pixel_loss = (info_w * per_pixel).mean()

        if self.core_l2_weight > 0:
            high_info = (info_w > self.core_info_threshold).float()
            n_high = high_info.sum().clamp(min=1.0)
            core_loss = (high_info * pixel_diff * pixel_diff).sum() / n_high
        else:
            core_loss = pixel_diff.new_zeros(())

        rms_weight = target_rms.mean().clamp(min=0.1)
        loss = rms_weight * (pixel_loss + self.core_l2_weight * core_loss)

        return {
            "loss": loss,
            "pred": pred.detach(),
            "target_norm": target_norm.detach(),
            "info_weights": info_w.detach(),
            "rms_weight": rms_weight.detach(),
            "pixel_loss_norm": pixel_loss.detach(),
            "core_loss_norm": core_loss.detach(),
            "fused_hw": enc_out["fused_hw"],
        }


# ============================================================
# Optimizer & scheduler factories
# ============================================================

def create_optimizer(model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.05) -> torch.optim.Optimizer:
    """AdamW with separate weight decay groups (no decay on biases / norms / embeddings)."""
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name or "embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr,
    )


def create_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup → cosine decay."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=1e-6)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


__all__ = [
    # Constants
    "ALL_BANDS",
    "RUBIN_BANDS",
    "EUCLID_BANDS",
    "STREAM_ORDER",
    "STREAM_PIXEL_SCALES",
    "STREAM_BAND_SLOTS",
    # Helpers
    "band_group",
    "compute_stream_depths",
    "decoder_stage_channels",
    "create_optimizer",
    "create_scheduler",
    # Building blocks
    "InformationMap",
    "BandStem",
    "LayerNorm2d",
    "ConvNeXtBlock",
    "DownBlock",
    "TransformerBlock",
    "FiLM",
    "StreamFuser",
    "StreamEncoder",
    "DecoderStage",
    "TargetDecoder",
    # Model
    "JAISPMixedEncoderV10",
    "JAISPFoundationV10",
]
