"""JAISP Foundation v7 - mixed-resolution masked band prediction.

Key ideas:
  - Two instrument streams: Rubin (0.2"/px) and Euclid (0.1"/px from MER mosaics).
  - Euclid stream uses concat+project to preserve per-band (VIS/Y/J/H) structure
    through the encoder, enabling PSF-aware and color-aware representations.
  - Rubin stream uses mean pooling (6 bands with similar PSFs, variable availability).
  - Fuse streams at a shared latent physical scale (~0.8"/px).
  - Decode back to the target band's native resolution with FiLM conditioning.

NISP Y/J/H data comes from Euclid MER mosaics at 0.1"/px (~1084x1084),
identical to VIS. There is no separate NISP stream.
"""

import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from jaisp_foundation_v6 import (
    ALL_BANDS,
    BandStem,
    ConvNeXtBlock,
    DownBlock,
    FiLM,
    InformationMap,
    LayerNorm2d,
    TransformerBlock,
    create_optimizer,
    create_scheduler,
)


RUBIN_BANDS = ["rubin_u", "rubin_g", "rubin_r", "rubin_i", "rubin_z", "rubin_y"]
EUCLID_BANDS = ["euclid_VIS", "euclid_Y", "euclid_J", "euclid_H"]
STREAM_ORDER = ["rubin", "euclid"]
STREAM_PIXEL_SCALES = {
    "rubin": 0.2,
    "euclid": 0.1,  # MER mosaics: all Euclid bands at 0.1"/px
}
STREAM_BRANCH_DEPTHS = {
    "rubin": 2,   # 512 -> 128 at 0.8"/px
    "euclid": 3,  # ~1084 -> ~135 at 0.8"/px
}
# Ordered band slots within each stream (used by StreamFuser for concat).
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


class StreamFuser(nn.Module):
    """Fuse per-band BandStem outputs within a stream.

    For streams with <=2 bands (or where concat is too expensive), uses mean.
    For streams with >2 bands at the same resolution, uses fixed-slot concat
    with zero-filled missing bands, followed by a 1x1 projection.

    This preserves per-band information (PSF differences, color structure)
    through the encoder instead of discarding it via mean pooling.
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
        """Fuse available band features.

        Args:
            band_feats: {band_name: [B, stem_ch, H, W]} for available bands.
        Returns:
            [B, out_ch, H, W] fused features.
        """
        if not band_feats:
            raise ValueError("StreamFuser received no band features.")

        if not self.use_concat:
            # Mean pooling fallback.
            feats = list(band_feats.values())
            return torch.stack(feats, dim=0).mean(dim=0)

        # Get reference shape from any available feature.
        ref = next(iter(band_feats.values()))
        B, C, H, W = ref.shape

        # Allocate zero-filled slot tensor.
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
    """Interpolate to a target size, fuse skip features, then refine."""

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


class JAISPMixedEncoderV7(nn.Module):
    """Encode mixed-resolution context bands and fuse them on a common latent grid.

    Two streams:
      - rubin:  6 BandStems → mean pool → StreamEncoder(depth=2)
      - euclid: 4 BandStems → concat+project → StreamEncoder(depth=3)

    The Euclid stream preserves per-band structure (VIS/Y/J/H have different
    PSFs: 0.2" vs 0.5") through the full encoder via fixed-slot concatenation.
    """

    def __init__(
        self,
        band_names: List[str],
        stem_ch: int = 64,
        hidden_ch: int = 256,
        blocks_per_stage: int = 2,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        fused_pixel_scale_arcsec: float = 0.8,
    ):
        super().__init__()
        self.band_names = list(band_names)
        self.stem_ch = stem_ch
        self.hidden_ch = hidden_ch
        self.fused_pixel_scale_arcsec = float(fused_pixel_scale_arcsec)

        self.stems = nn.ModuleDict({b: BandStem(stem_ch) for b in band_names})
        self.info_maps = nn.ModuleDict({b: InformationMap() for b in band_names})

        # StreamFusers: Rubin uses mean, Euclid uses concat+project.
        self.stream_fusers = nn.ModuleDict({
            "rubin": StreamFuser(RUBIN_BANDS, stem_ch, stem_ch, use_concat=False),
            "euclid": StreamFuser(EUCLID_BANDS, stem_ch, stem_ch, use_concat=True),
        })

        self.stream_encoders = nn.ModuleDict({
            stream: StreamEncoder(
                in_ch=stem_ch,
                hidden_ch=hidden_ch,
                depth=STREAM_BRANCH_DEPTHS[stream],
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
                for level in range(STREAM_BRANCH_DEPTHS[stream] + 1)
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
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D sincos PE"
        d = dim // 2
        freq = torch.pow(10000.0, -torch.arange(0, d, 2, device=device, dtype=torch.float32) / d)

        y_pos = torch.arange(H, device=device, dtype=torch.float32)
        x_pos = torch.arange(W, device=device, dtype=torch.float32)
        y_sin = torch.outer(y_pos, freq).sin()
        y_cos = torch.outer(y_pos, freq).cos()
        x_sin = torch.outer(x_pos, freq).sin()
        x_cos = torch.outer(x_pos, freq).cos()

        pe_y = torch.cat([y_sin, y_cos], dim=-1)
        pe_x = torch.cat([x_sin, x_cos], dim=-1)
        pe = torch.cat([
            pe_y.unsqueeze(1).expand(-1, W, -1),
            pe_x.unsqueeze(0).expand(H, -1, -1),
        ], dim=-1)
        return pe.reshape(1, H * W, dim)

    def _estimate_fused_hw(self, context_images: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        heights = []
        widths = []
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
            desired_scale_y = target_scale * (target_h / max(stage_h, 1))
            desired_scale_x = target_scale * (target_w / max(stage_w, 1))
            desired_scale = 0.5 * (desired_scale_y + desired_scale_x)

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
        # Run BandStems and group by stream.
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
            raise ValueError("No context streams available for v7 encoder.")

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


class JAISPFoundationV7(nn.Module):
    """Mixed-resolution foundation model for Rubin + Euclid."""

    def __init__(
        self,
        band_names: List[str] = ALL_BANDS,
        stem_ch: int = 64,
        hidden_ch: int = 256,
        blocks_per_stage: int = 2,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        fused_pixel_scale_arcsec: float = 0.8,
    ):
        super().__init__()
        self.band_names = list(band_names)
        self.band_to_idx = {b: i for i, b in enumerate(self.band_names)}

        self.encoder = JAISPMixedEncoderV7(
            band_names=self.band_names,
            stem_ch=stem_ch,
            hidden_ch=hidden_ch,
            blocks_per_stage=blocks_per_stage,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
            fused_pixel_scale_arcsec=fused_pixel_scale_arcsec,
        )

        num_bands = len(self.band_names)
        self.target_decoders = nn.ModuleDict({
            "rubin": TargetDecoder(hidden_ch, hidden_ch, [256, 128], num_bands, blocks_per_stage),
            "euclid": TargetDecoder(hidden_ch, hidden_ch, [256, 128, 64], num_bands, blocks_per_stage),
        })

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"JAISPFoundationV7: {n_params/1e6:.1f}M trainable parameters")
        print(f"  stem_ch={stem_ch}, hidden_ch={hidden_ch}, fused_scale={fused_pixel_scale_arcsec:.2f}\"/px")

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
            (B,),
            self.band_to_idx[target_band],
            dtype=torch.long,
            device=target_image.device,
        )
        target_stream = band_group(target_band)
        decoder = self.target_decoders[target_stream]
        stage_sizes = decoder.stage_sizes(enc_out["bottleneck"].shape[-2:], target_image.shape[-2:])
        skip_maps = self.encoder.build_target_skips(
            enc_out["stream_pyramids"],
            target_band,
            target_image.shape[-2:],
            stage_sizes,
        )
        pred = decoder(
            enc_out["bottleneck"],
            band_idx,
            target_image.shape[-2:],
            skip_maps=skip_maps,
        )

        target_norm = (target_image / (target_rms + 1e-10)).clamp(-10.0, 100.0)
        info_w = self.encoder.info_maps[target_band](target_image, target_rms)
        pixel_loss = (info_w * (pred - target_norm).abs()).mean()
        # Tile-level RMS band weight: noisy bands (high mean RMS) get
        # proportionally larger loss so the model can't ignore them.
        rms_weight = target_rms.mean().clamp(min=0.1)
        loss = rms_weight * pixel_loss

        return {
            "loss": loss,
            "pred": pred.detach(),
            "target_norm": target_norm.detach(),
            "info_weights": info_w.detach(),
            "rms_weight": rms_weight.detach(),
            "fused_hw": enc_out["fused_hw"],
        }


__all__ = [
    "ALL_BANDS",
    "EUCLID_BANDS",
    "JAISPFoundationV7",
    "RUBIN_BANDS",
    "band_group",
    "create_optimizer",
    "create_scheduler",
]
