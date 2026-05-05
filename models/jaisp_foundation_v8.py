"""JAISP Foundation v8 — fine-scale bottleneck with random crop.

Inherits the v7 mixed-resolution MAE architecture but makes the fused
pixel scale and stream encoder depths fully configurable, enabling
finer bottleneck resolution (e.g. 0.4"/px instead of 0.8"/px) without
changing tile size.

Key changes from v7:
  - ``stream_branch_depths`` is computed automatically from the target
    ``fused_pixel_scale_arcsec`` instead of being hardcoded.
  - ``TargetDecoder`` stage channels are derived from the stream depths
    so the decoder matches the encoder depth.
  - Designed to work with random-cropped tiles at training time:
    256×256 Rubin crops (from existing 512×512 tiles) at 0.4"/px
    gives ~128×128 bottleneck tokens — same cost as v7.

Everything else (BandStem, StreamFuser, transformer, FiLM, loss)
is inherited from v7/v6 unchanged.

Typical configurations:
    v7-equivalent:  fused_scale=0.8, crop_size=None  (full 512 tiles)
    v8 fine-scale:  fused_scale=0.4, crop_size=256   (random 256 crops)
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Reuse v7 constants and building blocks.
from jaisp_foundation_v7 import (
    RUBIN_BANDS,
    EUCLID_BANDS,
    STREAM_ORDER,
    STREAM_PIXEL_SCALES,
    STREAM_BAND_SLOTS,
    StreamFuser,
    StreamEncoder,
    DecoderStage,
    TargetDecoder,
    band_group,
)


# ============================================================
# Auto depth computation
# ============================================================

def compute_stream_depths(fused_pixel_scale: float) -> Dict[str, int]:
    """Compute stream encoder depths to reach the target fused scale.

    Each stride-2 ConvNeXt DownBlock doubles the effective pixel scale.
    We pick the depth that brings each stream closest to
    ``fused_pixel_scale`` arcsec/px.

        depth = round(log2(fused_pixel_scale / instrument_pixel_scale))

    Examples::

        0.8"/px → Rubin depth=2 (0.2→0.4→0.8), Euclid depth=3 (0.1→0.2→0.4→0.8)
        0.4"/px → Rubin depth=1 (0.2→0.4),      Euclid depth=2 (0.1→0.2→0.4)
        0.2"/px → Rubin depth=0 (stays at 0.2),  Euclid depth=1 (0.1→0.2)
    """
    depths = {}
    for stream, base_scale in STREAM_PIXEL_SCALES.items():
        d = max(1, round(math.log2(fused_pixel_scale / base_scale)))
        depths[stream] = d
    return depths


def decoder_stage_channels(stream_depth: int, hidden_ch: int) -> List[int]:
    """Derive TargetDecoder stage channels from encoder depth.

    The decoder should have the same number of upsampling stages as the
    encoder has downsampling stages, progressively narrowing channels.

    Examples (hidden_ch=256):
        depth=1 → [128]           (1 upsample stage)
        depth=2 → [256, 128]      (2 stages, same as v7 Rubin)
        depth=3 → [256, 128, 64]  (3 stages, same as v7 Euclid)
    """
    if stream_depth <= 1:
        return [hidden_ch // 2]
    if stream_depth == 2:
        return [hidden_ch, hidden_ch // 2]
    # depth >= 3
    channels = [hidden_ch]
    ch = hidden_ch
    for _ in range(stream_depth - 1):
        ch = max(ch // 2, 64)
        channels.append(ch)
    return channels


# ============================================================
# V8 Encoder
# ============================================================

class JAISPMixedEncoderV8(nn.Module):
    """Mixed-resolution encoder with configurable stream depths.

    Identical to V7's encoder except ``stream_branch_depths`` is passed
    as a parameter rather than read from a module-level constant.

    The ``rubin_concat`` flag (default ``False``, matching v8 production)
    selects how Rubin BandStem outputs are combined.  Setting it to
    ``True`` switches Rubin to concat+project fusion (matching the Euclid
    stream) — used by v9 to remove the gradient-attenuation asymmetry
    between Rubin and Euclid that notebook 13 diagnosed.
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
        rubin_concat: bool = False,
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
# V8 Foundation Model
# ============================================================

class JAISPFoundationV8(nn.Module):
    """Fine-scale mixed-resolution foundation model.

    Same masked-band-prediction objective as V7, but with configurable
    fused scale and auto-computed stream depths + decoder channels.

    Parameters
    ----------
    fused_pixel_scale_arcsec : float
        Target bottleneck resolution.  0.4 gives 2× finer features than
        the v7 default of 0.8, at the same token count when using
        256×256 Rubin crops.
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
        rubin_concat: bool = False,
    ):
        super().__init__()
        self.band_names = list(band_names)
        self.band_to_idx = {b: i for i, b in enumerate(self.band_names)}
        self.rubin_concat = bool(rubin_concat)

        stream_depths = compute_stream_depths(fused_pixel_scale_arcsec)

        self.encoder = JAISPMixedEncoderV8(
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
        print(f"JAISPFoundationV8: {n_params/1e6:.1f}M trainable parameters")
        print(f"  stem_ch={stem_ch}, hidden_ch={hidden_ch}, "
              f"fused_scale={fused_pixel_scale_arcsec:.2f}\"/px")
        print(f"  stream_depths={stream_depths}")
        print(f"  rubin_concat={self.rubin_concat}  "
              f"({'symmetric concat fusion (v9)' if self.rubin_concat else 'mean fusion (v8 default)'})")
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
        pixel_loss = (info_w * (pred - target_norm).abs()).mean()
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
    "JAISPFoundationV8",
    "RUBIN_BANDS",
    "band_group",
    "compute_stream_depths",
    "create_optimizer",
    "create_scheduler",
]
