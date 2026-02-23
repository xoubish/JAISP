import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def interpolate_tokens(
    tokens: torch.Tensor,
    src_grid: Tuple[int, int],
    dst_grid: Tuple[int, int],
) -> torch.Tensor:
    """Resample token grid [B,N,D] from src_grid to dst_grid."""
    if src_grid == dst_grid:
        return tokens

    bsz, _, dim = tokens.shape
    src_h, src_w = int(src_grid[0]), int(src_grid[1])
    dst_h, dst_w = int(dst_grid[0]), int(dst_grid[1])

    x = tokens.transpose(1, 2).contiguous().view(bsz, dim, src_h, src_w)
    x = F.interpolate(x, size=(dst_h, dst_w), mode="bilinear", align_corners=False)
    return x.view(bsz, dim, dst_h * dst_w).transpose(1, 2).contiguous()


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MaskedReconstructionHead(nn.Module):
    """
    Predict masked target-band pixels from:
      - target tokens encoded from masked target image
      - aggregated context tokens from other available bands
    """

    def __init__(
        self,
        embed_dim: int = 256,
        patch_size: int = 16,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)

        # Learn token-wise context fusion instead of plain averaging over context bands.
        self.context_score = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 1),
        )
        self.context_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.blocks = nn.ModuleList(
            [TransformerBlock(self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.patch_decoder = nn.Linear(self.embed_dim, self.patch_size * self.patch_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        target_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        grid_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            target_tokens: [B, N, D]
            context_tokens: [B, N, D] (already on target grid)
            token_mask: [B, N] with 1.0 for masked tokens
            grid_size: (H_tokens, W_tokens)

        Returns:
            Predicted target image on patch grid: [B, 1, H_tokens*patch, W_tokens*patch]
        """
        if token_mask.dim() != 2:
            raise ValueError(f"token_mask must be [B,N], got {tuple(token_mask.shape)}")

        # Accept either [B,N,D] or [B,K,N,D] context tokens.
        if context_tokens.dim() == 3:
            context_tokens = context_tokens.unsqueeze(1)
        if context_tokens.dim() != 4:
            raise ValueError(f"context_tokens must be [B,N,D] or [B,K,N,D], got {tuple(context_tokens.shape)}")

        bsz, kctx, ntok, dim = context_tokens.shape
        if target_tokens.shape[0] != bsz or target_tokens.shape[1] != ntok or target_tokens.shape[2] != dim:
            raise ValueError(
                "Token shape mismatch: "
                f"target={tuple(target_tokens.shape)}, context={tuple(context_tokens.shape)}"
            )

        target_rep = target_tokens.unsqueeze(1).expand(-1, kctx, -1, -1)  # [B,K,N,D]
        score_in = torch.cat([target_rep, context_tokens], dim=-1)  # [B,K,N,2D]
        scores = self.context_score(score_in).squeeze(-1)  # [B,K,N]
        weights = torch.softmax(scores, dim=1)
        fused_context = (weights.unsqueeze(-1) * context_tokens).sum(dim=1)  # [B,N,D]

        x = target_tokens + self.context_proj(fused_context)
        x = x + token_mask.unsqueeze(-1) * self.mask_token

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        patch_pixels = self.patch_decoder(x)  # [B,N,P*P]

        bsz, _, _ = patch_pixels.shape
        ht, wt = int(grid_size[0]), int(grid_size[1])
        p = self.patch_size

        patch_pixels = patch_pixels.view(bsz, ht, wt, p, p)
        patch_pixels = patch_pixels.permute(0, 1, 3, 2, 4).contiguous()
        return patch_pixels.view(bsz, 1, ht * p, wt * p)
