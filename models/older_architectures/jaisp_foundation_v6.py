# jaisp_foundation_v6.py
#
# JAISP Foundation v6 - Masked Band Prediction with Dense Reconstruction
#
# Fundamental departure from v5 (JEPA):
#   - NO token-grid patching (16×16 patches lose sub-pixel precision)
#   - NO teacher/EMA (not needed for reconstruction objectives)
#   - NO latent-space alignment loss
#
# Architecture:
#   - Band-specific CNN stems (per v5, but GroupNorm instead of BatchNorm)
#   - ConvNeXt encoder: 3 stride-2 stages → dense feature maps at H/8
#   - Transformer bottleneck: 4 blocks of full self-attention on H/8 tokens
#   - U-Net decoder: skip connections → pixel-level output
#   - FiLM conditioning: decoder knows WHICH band to reconstruct
#
# Self-supervised objective:
#   Phase A (this file): Masked band prediction
#     - Context: N-1 available Rubin bands
#     - Target: 1 masked Rubin band
#     - Loss: InformationMap-weighted L1 in noise-normalized pixel space
#   Phase B (future): Cross-instrument reconstruction
#     - Context: Rubin bands → predict Euclid VIS (downsampled to Rubin res)
#
# Why pixel-space reconstruction beats JEPA for precision cosmology:
#   - Forces the network to care about exact spatial layout (can't cheat)
#   - No resolution loss from patching
#   - Dense features naturally support astrometry, photometry, weak lensing

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


# ============================================================
# BAND METADATA
# ============================================================

RUBIN_BANDS = ['rubin_u', 'rubin_g', 'rubin_r', 'rubin_i', 'rubin_z', 'rubin_y']
EUCLID_BANDS = ['euclid_VIS', 'euclid_Y', 'euclid_J', 'euclid_H']
ALL_BANDS = RUBIN_BANDS + EUCLID_BANDS


# ============================================================
# INFORMATION MAP (identical to v5 - it's good)
# ============================================================

class InformationMap(nn.Module):
    """Signal-based pixel weighting: focuses loss on sources, edges, not blank sky.

    Uses an RMS-adaptive floor so that noisy bands (high RMS) retain a
    meaningful minimum weight on blank-sky pixels, penalising hallucinated
    sources that would otherwise go unpunished.
    """
    def __init__(self, snr_threshold: float = 2.0, min_weight: float = 0.001,
                 adaptive_floor_scale: float = 0.3):
        super().__init__()
        self.snr_threshold = float(snr_threshold)
        self.min_weight = float(min_weight)
        self.adaptive_floor_scale = float(adaptive_floor_scale)
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sx)
        self.register_buffer('sobel_y', sx.transpose(-1, -2).contiguous())

    def forward(self, image: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
        x = image / (rms + 1e-10)
        snr_weight = torch.sigmoid((x.abs() - self.snr_threshold) * 2.0)
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        grad = torch.sqrt(gx**2 + gy**2 + 1e-10)
        grad_max = grad.amax(dim=(2, 3), keepdim=True) + 1e-10
        grad_weight = grad / grad_max
        weights = torch.maximum(snr_weight, grad_weight * 0.5) ** 2
        # RMS-adaptive floor: noisier bands get a higher minimum weight
        # so blank-sky pixels still contribute to the loss
        mean_rms = rms.mean(dim=(2, 3), keepdim=True)
        adaptive_min = self.min_weight + torch.sigmoid(mean_rms - 1.0) * self.adaptive_floor_scale
        weights = weights.clamp(min=adaptive_min)
        return weights / (weights.sum(dim=(2, 3), keepdim=True) + 1e-10) * (image.shape[2] * image.shape[3])


# ============================================================
# BAND STEM (GroupNorm instead of BatchNorm for batch_size=1)
# ============================================================

class BandStem(nn.Module):
    """
    Per-band CNN stem. Noise-normalizes input, then extracts local features.
    GroupNorm is used instead of BatchNorm to handle batch_size=1 cleanly.
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
# CONVNEXT BLOCK (spatial backbone building block)
# ============================================================

class LayerNorm2d(nn.Module):
    """LayerNorm applied over the channel dim of a [B,C,H,W] tensor."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style block: depthwise 7×7 conv, LayerNorm, channel-MLP.
    Uses residual connection. Works at any spatial resolution.
    """
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
        x = x.permute(0, 2, 3, 1)       # B,C,H,W → B,H,W,C
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        x = x.permute(0, 3, 1, 2)       # B,H,W,C → B,C,H,W
        return residual + x


# ============================================================
# ENCODER STAGES
# ============================================================

class DownBlock(nn.Module):
    """
    Downsample 2× (ConvNeXt-style: LayerNorm + stride-2 conv) then N ConvNeXt blocks.
    Uses kernel_size=2 stride-2 conv for exact halving; bilinear upsample in decoder
    recovers odd dimensions correctly via size= argument.
    """
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2):
        super().__init__()
        self.downsample = nn.Sequential(
            LayerNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2),
        )
        self.blocks = nn.Sequential(*[ConvNeXtBlock(out_ch) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.downsample(x))


# ============================================================
# DECODER STAGES
# ============================================================

class UpBlock(nn.Module):
    """
    Bilinear 2× upsample, project to out_ch, add skip connection (also projected),
    then N ConvNeXt blocks. Upsamples to exact skip spatial size (handles odd dims).
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_blocks: int = 2):
        super().__init__()
        self.up_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, kernel_size=1)
        self.blocks = nn.Sequential(*[ConvNeXtBlock(out_ch) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up_proj(x) + self.skip_proj(skip)
        return self.blocks(x)


# ============================================================
# TRANSFORMER BLOCK (for bottleneck global context)
# ============================================================

class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block.
    Uses nn.MultiheadAttention which in PyTorch 2.0+ automatically dispatches
    to Flash Attention (memory-efficient) when no mask is provided.
    """
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


# ============================================================
# ENCODER
# ============================================================

class JAISPEncoderV6(nn.Module):
    """
    Multi-band encoder:
      1. Each context band independently through its BandStem → [B, stem_ch, H, W]
      2. Mean aggregation across bands → [B, stem_ch, H, W]
         (physically meaningful: all bands are in noise-normalized units)
      3. ConvNeXt stages: H → H/2 → H/4 → H/8 (with skip connections stored)
      4. Transformer bottleneck: global self-attention on H/8 × W/8 tokens
         (fixed 2D sincos position embedding so spatial layout is preserved)
    """
    def __init__(
        self,
        band_names: List[str],
        stem_ch: int = 64,
        dims: Tuple[int, ...] = (128, 256, 512),
        blocks_per_stage: int = 2,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
    ):
        super().__init__()
        self.band_names = list(band_names)
        self.stem_ch = stem_ch
        self.dims = dims

        # Per-band stems and info maps
        self.stems = nn.ModuleDict({b: BandStem(stem_ch) for b in band_names})
        self.info_maps = nn.ModuleDict({b: InformationMap() for b in band_names})

        # ConvNeXt encoder stages
        in_ch = stem_ch
        self.stages = nn.ModuleList()
        for out_ch in dims:
            self.stages.append(DownBlock(in_ch, out_ch, blocks_per_stage))
            in_ch = out_ch

        # Transformer bottleneck (on H/8 tokens)
        self.transformer = nn.ModuleList(
            [TransformerBlock(dims[-1], transformer_heads) for _ in range(transformer_depth)]
        )
        self.transformer_norm = nn.LayerNorm(dims[-1])

    @staticmethod
    def _make_2d_sincos(H: int, W: int, dim: int, device: torch.device) -> torch.Tensor:
        """
        Fixed 2D sinusoidal position embedding → [1, H*W, dim].
        First dim//2 channels encode Y position, last dim//2 encode X position.
        Not learned: spatial meaning is preserved regardless of input size.
        """
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D sincos PE"
        d = dim // 2  # half for y, half for x
        freq = torch.pow(10000.0, -torch.arange(0, d, 2, device=device, dtype=torch.float32) / d)

        y_pos = torch.arange(H, device=device, dtype=torch.float32)
        x_pos = torch.arange(W, device=device, dtype=torch.float32)

        y_sin = torch.outer(y_pos, freq).sin()  # [H, d/2]
        y_cos = torch.outer(y_pos, freq).cos()  # [H, d/2]
        x_sin = torch.outer(x_pos, freq).sin()  # [W, d/2]
        x_cos = torch.outer(x_pos, freq).cos()  # [W, d/2]

        pe_y = torch.cat([y_sin, y_cos], dim=-1)               # [H, d]
        pe_x = torch.cat([x_sin, x_cos], dim=-1)               # [W, d]
        pe = torch.cat([
            pe_y.unsqueeze(1).expand(-1, W, -1),                # [H, W, d]
            pe_x.unsqueeze(0).expand(H, -1, -1),                # [H, W, d]
        ], dim=-1)                                              # [H, W, dim]
        return pe.reshape(1, H * W, dim)

    def forward(
        self,
        context_images: Dict[str, torch.Tensor],
        context_rms: Dict[str, torch.Tensor],
    ) -> Dict:
        """
        context_images: {band_name: [B, 1, H, W]} — all context bands (same spatial size)
        context_rms:    {band_name: [B, 1, H, W]}

        Returns:
            bottleneck:   [B, dims[-1], H/8, W/8]  — transformer output
            skips:        list of [B, C, H', W']     — skip connection features
            info_weights: {band: [B, 1, H, W]}       — per-band information maps
        """
        # 1. Encode each context band through its stem
        stem_feats = []
        info_weights = {}
        for band, img in context_images.items():
            rms = context_rms[band]
            info_weights[band] = self.info_maps[band](img, rms)
            stem_feats.append(self.stems[band](img, rms))  # [B, stem_ch, H, W]

        # 2. Aggregate across bands (mean in noise-normalized feature space)
        x = torch.stack(stem_feats, dim=0).mean(dim=0)  # [B, stem_ch, H, W]

        # 3. ConvNeXt encoder stages (store skips before downsampling)
        skips = [x]  # skip[0]: stem-resolution features
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        # skips[0] = [B, stem_ch, H, W]
        # skips[1] = [B, dims[0], H/2, W/2]
        # skips[2] = [B, dims[1], H/4, W/4]
        # skips[3] = [B, dims[2], H/8, W/8]   ← same as x before transformer

        # 4. Transformer bottleneck
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        pe = self._make_2d_sincos(H, W, C, x.device)
        tokens = tokens + pe
        for blk in self.transformer:
            tokens = blk(tokens)
        tokens = self.transformer_norm(tokens)
        bottleneck = tokens.transpose(1, 2).view(B, C, H, W)  # [B, C, H/8, W/8]

        return {
            'bottleneck': bottleneck,
            'skips': skips,
            'info_weights': info_weights,
        }


# ============================================================
# FiLM CONDITIONING (target-band conditioning in decoder)
# ============================================================

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation: scale + shift features by a learned band embedding.
    Allows a single shared decoder to specialize for any target band.
    Initialized to identity (scale=1, shift=0) via zero-init of embedding weights.
    """
    def __init__(self, num_bands: int, channels: int):
        super().__init__()
        self.embed = nn.Embedding(num_bands, channels * 2)
        nn.init.zeros_(self.embed.weight)

    def forward(self, x: torch.Tensor, band_idx: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W], band_idx: [B] LongTensor"""
        gb = self.embed(band_idx)                         # [B, 2C]
        gamma, beta = gb.chunk(2, dim=-1)                 # [B, C] each
        gamma = gamma.view(x.shape[0], x.shape[1], 1, 1)
        beta = beta.view(x.shape[0], x.shape[1], 1, 1)
        return x * (1.0 + gamma) + beta


# ============================================================
# DECODER
# ============================================================

class JAISPDecoderV6(nn.Module):
    """
    U-Net decoder with FiLM band conditioning.
    Upsamples bottleneck features back to full resolution using encoder skip connections.
    FiLM layers condition each decoder stage on which band to reconstruct,
    allowing one decoder to predict any of the 10 bands.

    Output is in noise-normalized units (flux / rms), matching the loss target.
    """
    def __init__(
        self,
        num_bands: int,
        all_dims: Tuple[int, ...],   # (stem_ch, stage1_ch, stage2_ch, stage3_ch)
        blocks_per_stage: int = 2,
    ):
        super().__init__()
        dims = list(all_dims)         # e.g., [64, 128, 256, 512]

        self.up_blocks = nn.ModuleList()
        self.film_layers = nn.ModuleList()

        # Build in reverse: from dims[-1] back down to dims[0]
        for i in range(len(dims) - 1, 0, -1):
            in_ch   = dims[i]         # input from previous decoder stage
            skip_ch = dims[i - 1]     # skip connection channels from encoder
            out_ch  = dims[i - 1]     # output channels
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch, blocks_per_stage))
            self.film_layers.append(FiLM(num_bands, out_ch))

        # Final output head: predict in noise-normalized units
        self.out_head = nn.Sequential(
            LayerNorm2d(dims[0]),
            nn.Conv2d(dims[0], 1, kernel_size=1),
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: List[torch.Tensor],
        band_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        bottleneck: [B, dims[-1], H/8, W/8]
        skips:      from encoder — [stem, stage1, stage2, stage3]; we use all but last
        band_idx:   [B] LongTensor — index of band to reconstruct (for FiLM)
        returns:    [B, 1, H, W] predicted image in noise-normalized units
        """
        x = bottleneck
        # Reverse skip list (exclude last, which is same resolution as bottleneck)
        # skip_list[0] = coarsest skip (dims[-2], H/4)
        # skip_list[-1] = finest skip (stem_ch, H)
        skip_list = list(reversed(skips[:-1]))

        for up_block, film, skip in zip(self.up_blocks, self.film_layers, skip_list):
            x = up_block(x, skip)
            x = film(x, band_idx)

        return self.out_head(x)  # [B, 1, H, W]


# ============================================================
# FULL MODEL
# ============================================================

class JAISPFoundationV6(nn.Module):
    """
    JAISP Foundation v6: Dense Masked Band Prediction.

    Given N-1 context bands from the same sky tile, predicts the remaining
    (masked) band at pixel resolution. The reconstruction loss in pixel space
    forces the encoder to preserve precise spatial information — exactly what
    JEPA could not do because its loss lived in latent space.

    Encoder:  BandStems → mean aggregation → ConvNeXt(3 stages) → Transformer(4 blocks)
    Decoder:  U-Net (skip connections) + FiLM band conditioning → pixel output

    No teacher, no EMA, no predictor head. Just encode-and-reconstruct.
    """
    def __init__(
        self,
        band_names: List[str] = ALL_BANDS,
        stem_ch: int = 64,
        encoder_dims: Tuple[int, ...] = (128, 256, 512),
        blocks_per_stage: int = 2,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
    ):
        super().__init__()
        self.band_names = list(band_names)
        self.band_to_idx = {b: i for i, b in enumerate(band_names)}

        self.encoder = JAISPEncoderV6(
            band_names=band_names,
            stem_ch=stem_ch,
            dims=encoder_dims,
            blocks_per_stage=blocks_per_stage,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
        )

        all_dims = (stem_ch,) + tuple(encoder_dims)
        self.decoder = JAISPDecoderV6(
            num_bands=len(band_names),
            all_dims=all_dims,
            blocks_per_stage=blocks_per_stage,
        )

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"JAISPFoundationV6: {n_params/1e6:.1f}M trainable parameters")
        print(f"  stem_ch={stem_ch}, encoder_dims={encoder_dims}")
        print(f"  blocks_per_stage={blocks_per_stage}, transformer_depth={transformer_depth}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        """
        Run encoder only (for feature extraction after training).
        Returns bottleneck and skip features — dense spatial representations.
        """
        return self.encoder(context_images, context_rms)

    def forward(
        self,
        context_images: Dict[str, torch.Tensor],
        context_rms: Dict[str, torch.Tensor],
        target_band: str,
        target_image: torch.Tensor,
        target_rms: torch.Tensor,
    ) -> Dict:
        """
        context_images: {band: [B, 1, H, W]} — input context bands (not masked)
        context_rms:    {band: [B, 1, H, W]} — RMS noise maps for context bands
        target_band:    str — name of band to reconstruct (must be in self.band_names)
        target_image:   [B, 1, H, W] — ground truth flux values (held out)
        target_rms:     [B, 1, H, W] — RMS for target band (for normalization)

        Returns dict with:
            loss:         scalar reconstruction loss (InformationMap-weighted L1)
            pred:         [B, 1, H, W] predicted image in noise-normalized units
            target_norm:  [B, 1, H, W] ground truth in noise-normalized units
            info_weights: [B, 1, H, W] InformationMap weights for target band
        """
        # Encode context bands
        enc_out = self.encoder(context_images, context_rms)

        # Band index tensor for FiLM conditioning
        B = target_image.shape[0]
        band_idx = torch.full((B,), self.band_to_idx[target_band],
                              dtype=torch.long, device=target_image.device)

        # Decode → predicted target in noise-normalized units
        pred = self.decoder(enc_out['bottleneck'], enc_out['skips'], band_idx)

        # Ground truth in noise-normalized units (same domain as prediction)
        target_norm = (target_image / (target_rms + 1e-10)).clamp(-10.0, 100.0)

        # InformationMap weights for target band (focus loss on sources)
        info_w = self.encoder.info_maps[target_band](target_image, target_rms)

        # Weighted L1 reconstruction loss
        loss = (info_w * (pred - target_norm).abs()).mean()

        return {
            'loss': loss,
            'pred': pred.detach(),
            'target_norm': target_norm.detach(),
            'info_weights': info_w.detach(),
        }


# ============================================================
# OPTIMIZER & SCHEDULER
# ============================================================

def create_optimizer(model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.05) -> torch.optim.Optimizer:
    # Separate weight decay: don't decay biases, norms, embeddings
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or 'bias' in name or 'norm' in name or 'embed' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [{'params': decay_params, 'weight_decay': weight_decay},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=lr,
    )


def create_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=1e-6)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
