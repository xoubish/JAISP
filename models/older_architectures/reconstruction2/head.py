"""
Resolution-Aware Masked Reconstruction Head for JAISP.

Key design principles:
  - Decode to the TARGET BAND's native pixel resolution, not a fixed grid.
  - Inject stem features (64-ch, full native res) from context bands at every
    decoder scale via learned attention-weighted aggregation.
  - Inject target band's own stem features (masked outside the hole) as skip
    connections so the decoder preserves high-frequency detail where available.
  - Progressive 2× upsampling (4 stages for patch_size=16) eliminates block
    artifacts by design — neighboring patches share overlapping conv receptive
    fields at every scale.

Drop-in replacement for the original MaskedReconstructionHead.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def interpolate_tokens(
    tokens: torch.Tensor,
    src_grid: Tuple[int, int],
    dst_grid: Tuple[int, int],
) -> torch.Tensor:
    """Resample token grid [B,N,D] from src_grid to dst_grid."""
    if src_grid == dst_grid:
        return tokens
    B, _, D = tokens.shape
    sh, sw = int(src_grid[0]), int(src_grid[1])
    dh, dw = int(dst_grid[0]), int(dst_grid[1])
    x = tokens.transpose(1, 2).contiguous().view(B, D, sh, sw)
    x = F.interpolate(x, size=(dh, dw), mode="bilinear", align_corners=False)
    return x.view(B, D, dh * dw).transpose(1, 2).contiguous()


def _num_upsample_stages(patch_size: int) -> int:
    """How many 2× stages to go from token grid to pixel grid."""
    return int(round(math.log2(patch_size)))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm1(x)
        x = x + self.attn(xn, xn, xn)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class UpsampleFuseBlock(nn.Module):
    """
    Single 2× upsample stage with skip-connection injection.

    Input:  decoder features  [B, C_in,  h,   w  ]
    Skips:  target_stem       [B, C_stem, h*2, w*2]  (resampled by caller)
            context_stem      [B, C_stem, h*2, w*2]  (resampled by caller)
            mask              [B, 1,      h*2, w*2]  (resampled by caller)
    Output: fused features    [B, C_out,  h*2, w*2]
    """

    def __init__(self, ch_in: int, ch_out: int, stem_ch: int = 64, skip_proj: int = 16):
        super().__init__()
        # Project high-dim stem features to compact size before concat.
        self.tgt_proj = nn.Conv2d(stem_ch, skip_proj, 1)
        self.ctx_proj = nn.Conv2d(stem_ch, skip_proj, 1)

        # PixelShuffle-based 2× upsample (avoids checkerboard vs ConvTranspose2d).
        self.up = nn.Sequential(
            nn.Conv2d(ch_in, ch_out * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )

        # Fusion conv after concatenation: ch_out + 2*skip_proj + 1 (mask).
        fuse_in = ch_out + 2 * skip_proj + 1
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, ch_out, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch_out, ch_out, 3, padding=1),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        tgt_stem: torch.Tensor,
        ctx_stem: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)                                         # [B, C_out, 2h, 2w]
        h, w = x.shape[2], x.shape[3]

        # Resize skips to match (handles non-power-of-2 sizes).
        ts = F.interpolate(tgt_stem, size=(h, w), mode="bilinear", align_corners=False)
        cs = F.interpolate(ctx_stem, size=(h, w), mode="bilinear", align_corners=False)
        m = F.interpolate(mask, size=(h, w), mode="nearest")

        ts = self.tgt_proj(ts)                                  # [B, skip_proj, h, w]
        cs = self.ctx_proj(cs)

        return self.fuse(torch.cat([x, ts, cs, m], dim=1))     # [B, C_out, 2h, 2w]


class ContextAggregator(nn.Module):
    """
    Attention-weighted aggregation of K context stem feature maps.

    Operates at a *single* spatial scale — call it once per decoder stage
    with features resampled to that scale, or once at native res and let
    the decoder stages resize.

    For efficiency we do the latter: aggregate once at target native res,
    then each UpsampleFuseBlock just calls F.interpolate.
    """

    def __init__(self, stem_ch: int = 64):
        super().__init__()
        # Per-pixel, per-band attention score from feature content.
        self.score_net = nn.Sequential(
            nn.Conv2d(stem_ch, stem_ch // 2, 1),
            nn.GELU(),
            nn.Conv2d(stem_ch // 2, 1, 1),
        )

    def forward(
        self,
        context_feats: List[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            context_feats: list of K tensors [B, stem_ch, H_k, W_k] at various resolutions
            target_hw: (H_target, W_target) — the native resolution we aggregate onto

        Returns:
            [B, stem_ch, H_target, W_target] attention-weighted context features
        """
        if not context_feats:
            B = 1
            C = self.score_net[0].in_channels
            device = self.score_net[0].weight.device
            return torch.zeros(B, C, target_hw[0], target_hw[1], device=device)

        H, W = int(target_hw[0]), int(target_hw[1])

        # Pass 1: compute attention scores only (small tensors [B,1,H,W] each).
        scores = []
        for feat in context_feats:
            r = F.interpolate(feat.float(), size=(H, W), mode="bilinear", align_corners=False)
            scores.append(self.score_net(r))                         # [B, 1, H, W]
            # r is discarded here — not stored

        score_stack = torch.stack(scores, dim=1)                     # [B, K, 1, H, W]
        weights = torch.softmax(score_stack, dim=1)
        del scores, score_stack

        # Pass 2: resample again and accumulate weighted sum one band at a time.
        aggregated = None
        for k, feat in enumerate(context_feats):
            r = F.interpolate(feat.float(), size=(H, W), mode="bilinear", align_corners=False)
            wr = weights[:, k] * r                                   # [B, 1, H, W] * [B, C, H, W]
            if aggregated is None:
                aggregated = wr
            else:
                aggregated = aggregated + wr

        return aggregated


# ---------------------------------------------------------------------------
# Main head
# ---------------------------------------------------------------------------

class ResolutionAwareReconstructionHead(nn.Module):
    """
    Predict masked target-band pixels using:
      - Token-level cross-band reasoning (transformer blocks)
      - Progressive upsampling to target band's native pixel grid
      - Multi-scale skip connections from context and target stem features

    Replaces MaskedReconstructionHead with resolution-aware decoding.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        stem_ch: int = 64,
        patch_size: int = 16,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        skip_proj: int = 16,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.stem_ch = int(stem_ch)
        self.patch_size = int(patch_size)
        self.n_stages = _num_upsample_stages(self.patch_size)   # 4 for ps=16

        # --- Token-level cross-band fusion (same role as original head) ---
        self.context_score = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 1),
        )
        self.context_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)

        # --- Context stem aggregator ---
        self.context_agg = ContextAggregator(self.stem_ch)

        # --- Progressive pixel decoder ---
        # Channel schedule: embed_dim → ... → small → 1
        # 4 stages for patch_size=16:  256 → 128 → 64 → 32 → 16
        chs = self._channel_schedule()
        self.token_to_spatial = nn.Conv2d(self.embed_dim, chs[0], 1)

        self.up_stages = nn.ModuleList()
        for i in range(self.n_stages):
            self.up_stages.append(
                UpsampleFuseBlock(chs[i], chs[i + 1], self.stem_ch, skip_proj)
            )

        self.to_pixels = nn.Conv2d(chs[-1], 1, 3, padding=1)

        self._init_weights()

    def _channel_schedule(self) -> List[int]:
        """Produce [C_0, C_1, ..., C_n_stages] channel counts."""
        # Start at embed_dim, halve each stage, floor at 16.
        chs = [self.embed_dim]
        c = self.embed_dim
        for _ in range(self.n_stages):
            c = max(16, c // 2)
            chs.append(c)
        return chs

    def _init_weights(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------
    # Token-level fusion (unchanged from original head)
    # -----------------------------------------------------------------
    def _fuse_tokens(
        self,
        target_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        target_tokens: [B, N, D]
        context_tokens: [B, K, N, D]
        token_mask: [B, N]
        Returns: processed tokens [B, N, D]
        """
        B, K, N, D = context_tokens.shape

        target_rep = target_tokens.unsqueeze(1).expand(-1, K, -1, -1)
        score_in = torch.cat([target_rep, context_tokens], dim=-1)
        scores = self.context_score(score_in).squeeze(-1)           # [B, K, N]
        weights = torch.softmax(scores, dim=1)
        fused = (weights.unsqueeze(-1) * context_tokens).sum(dim=1) # [B, N, D]

        x = target_tokens + self.context_proj(fused)
        x = x + token_mask.unsqueeze(-1) * self.mask_token

        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    # -----------------------------------------------------------------
    # Progressive pixel decoder
    # -----------------------------------------------------------------
    def _decode_pixels(
        self,
        tokens: torch.Tensor,
        grid_size: Tuple[int, int],
        target_hw: Tuple[int, int],
        target_stem: torch.Tensor,
        context_stem_agg: torch.Tensor,
        pixel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        tokens:           [B, N, D]
        grid_size:        (Ht, Wt) token grid
        target_hw:        (H_pix, W_pix) target band native resolution
        target_stem:      [B, stem_ch, H_pix, W_pix] (masked target stem features)
        context_stem_agg: [B, stem_ch, H_pix, W_pix] (aggregated context stems)
        pixel_mask:       [B, 1, H_pix, W_pix]

        Returns: residual prediction [B, 1, H_pix, W_pix]
        """
        B, N, D = tokens.shape
        Ht, Wt = int(grid_size[0]), int(grid_size[1])

        x = tokens.transpose(1, 2).contiguous().view(B, D, Ht, Wt)
        x = self.token_to_spatial(x)                               # [B, C0, Ht, Wt]

        for stage in self.up_stages:
            x = stage(x, target_stem, context_stem_agg, pixel_mask)

        # Final interpolation to exact target resolution (handles non-power-of-2).
        H_tgt, W_tgt = int(target_hw[0]), int(target_hw[1])
        if x.shape[2] != H_tgt or x.shape[3] != W_tgt:
            x = F.interpolate(x, size=(H_tgt, W_tgt), mode="bilinear", align_corners=False)

        return self.to_pixels(x)                                   # [B, 1, H, W]

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(
        self,
        target_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        grid_size: Tuple[int, int],
        target_hw: Tuple[int, int],
        target_stem_feat: torch.Tensor,
        context_stem_feats: List[torch.Tensor],
        masked_input: Optional[torch.Tensor] = None,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            target_tokens:      [B, N, D]  encoder tokens from masked target
            context_tokens:     [B, N, D] or [B, K, N, D]  encoder tokens from context bands
            token_mask:         [B, N]  1.0 for masked tokens
            grid_size:          (Ht, Wt)  target token grid
            target_hw:          (H_pix, W_pix)  target band native pixel resolution
            target_stem_feat:   [B, stem_ch, H_pix, W_pix]  stem features of masked target image
            context_stem_feats: list of [B, stem_ch, H_k, W_k]  stem features per context band
            masked_input:       [B, 1, H_pix, W_pix]  masked target in model space (optional)
            pixel_mask:         [B, 1, H_pix, W_pix]  (optional)

        Returns dict with:
            residual:       [B, 1, H_pix, W_pix]
            token_inpaint:  masked_input + residual * mask  (or just residual)
            pred:           same as token_inpaint (no separate refine — the progressive
                            decoder already produces smooth output)
        """
        if token_mask.dim() != 2:
            raise ValueError(f"token_mask must be [B,N], got {tuple(token_mask.shape)}")

        # Accept [B,N,D] or [B,K,N,D] context tokens.
        if context_tokens.dim() == 3:
            context_tokens = context_tokens.unsqueeze(1)

        # --- Token-level fusion ---
        processed = self._fuse_tokens(target_tokens, context_tokens, token_mask)

        # --- Aggregate context stem features at target native resolution ---
        H_tgt, W_tgt = int(target_hw[0]), int(target_hw[1])
        # Stems may be float16 for memory; aggregator casts to float32 one at a time.
        ctx_agg = self.context_agg(context_stem_feats, (H_tgt, W_tgt))

        # --- Mask the target stem features (zero inside hole) ---
        target_stem_f32 = target_stem_feat.float()
        if pixel_mask is not None:
            # pixel_mask: [B,1,H,W] with 1=masked.  We zero out masked regions.
            pm = pixel_mask
            if pm.shape[-2] != target_stem_f32.shape[-2] or pm.shape[-1] != target_stem_f32.shape[-1]:
                pm = F.interpolate(pm, size=target_stem_f32.shape[-2:], mode="nearest")
            target_stem_masked = target_stem_f32 * (1.0 - pm)
        else:
            target_stem_masked = target_stem_f32

        # Provide mask for decoder (default to zeros = no mask info).
        if pixel_mask is None:
            pixel_mask = torch.zeros(
                processed.shape[0], 1, H_tgt, W_tgt,
                device=processed.device, dtype=processed.dtype,
            )
        # Ensure pixel_mask is [B,1,H,W].
        if pixel_mask.dim() == 3:
            pixel_mask = pixel_mask.unsqueeze(1)

        # --- Progressive decode ---
        residual = self._decode_pixels(
            processed, grid_size, (H_tgt, W_tgt),
            target_stem_masked, ctx_agg, pixel_mask,
        )

        # --- Compose output ---
        if masked_input is None:
            return {"residual": residual, "token_inpaint": residual, "pred": residual}

        h, w = residual.shape[-2], residual.shape[-1]
        mi = masked_input[:, :, :h, :w]
        pm = pixel_mask[:, :, :h, :w]

        pred = mi + residual * pm
        return {"residual": residual, "token_inpaint": pred, "pred": pred}
