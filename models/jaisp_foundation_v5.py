# jaisp_foundation_v5.py
#
# JAISP Foundation v5 - STRICT POSITION ENCODING
#
# Key changes from v4:
#  - AlignmentLoss: REMOVED shift tolerance (shift_px=0)
#  - Forces exact token-to-token matching at corresponding spatial positions
#  - This should teach the model to preserve precise spatial correspondence
#
# All other components remain the same:
#  - signal-based InformationMap weights (SNR + gradients, device-safe)
#  - band-specific stems + shared ViT-like trunk with positional encodings
#  - VICReg regularization
#  - BYOL/JEPA-style Student + EMA Teacher + Predictor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
import copy


# =============================================================================
# INFORMATION MAP (Hardened with registered buffers)
# =============================================================================

class InformationMap(nn.Module):
    """Pure signal-based weighting using registered Sobel buffers."""
    def __init__(self, snr_threshold: float = 2.0, min_weight: float = 0.001):
        super().__init__()
        self.snr_threshold = float(snr_threshold)
        self.min_weight = float(min_weight)

        sx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sy = sx.transpose(-1, -2).contiguous()
        self.register_buffer('sobel_x', sx)
        self.register_buffer('sobel_y', sy)

    def forward(self, image: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
        # Work in noise units so weights don't learn instrument background texture
        x = image / (rms + 1e-10)

        # Signal-to-Noise weighting
        snr = x.abs()
        snr_weight = torch.sigmoid((snr - self.snr_threshold) * 2.0)

        # Gradient magnitude on SNR-normalized image
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        grad = torch.sqrt(gx**2 + gy**2 + 1e-10)

        # Normalize gradient per-image
        grad_max = grad.amax(dim=(2, 3), keepdim=True) + 1e-10
        grad_weight = grad / grad_max

        # Combine and increase contrast
        weights = torch.maximum(snr_weight, grad_weight * 0.5) ** 2
        weights = weights.clamp(min=self.min_weight)

        # Area-normalized output (so average weight ~1)
        return weights / (weights.sum(dim=(2, 3), keepdim=True) + 1e-10) * (image.shape[2] * image.shape[3])


# =============================================================================
# ENCODER COMPONENTS
# =============================================================================

class BandStem(nn.Module):
    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
        x = image / (rms + 1e-10)
        x = x.clamp(-10, 100)  # robust noise-unit range
        return self.net(x)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(in_ch, embed_dim, self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B,C,H,W] -> tokens: [B,N,D], gs=(Ht,Wt)
        x = self.proj(x)
        gs = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), gs


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SharedEncoder(nn.Module):
    def __init__(self, stem_ch: int = 64, embed_dim: int = 256, depth: int = 6, patch_size: int = 16):
        super().__init__()
        self.patch_embed = PatchEmbed(stem_ch, embed_dim, patch_size)
        self.base_grid_size = 32
        self.pos_embed = nn.Parameter(torch.randn(1, self.base_grid_size ** 2, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def _interpolate_pos_embed(self, grid_size: Tuple[int, int]) -> torch.Tensor:
        H, W = int(grid_size[0]), int(grid_size[1])
        if H == self.base_grid_size and W == self.base_grid_size:
            return self.pos_embed

        pe = self.pos_embed.reshape(1, self.base_grid_size, self.base_grid_size, -1).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(H, W), mode='bilinear', align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, H * W, -1)

    def forward(self, x):
        tokens, gs = self.patch_embed(x)
        tokens = tokens + self._interpolate_pos_embed(gs)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens), gs


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        B, N, D = x.shape
        x = x.reshape(B * N, D)
        x = self.net(x)
        return x.reshape(B, N, -1)


# =============================================================================
# LOSS: SHIFT-TOLERANT ALIGNMENT & VICReg
# =============================================================================

class AlignmentLoss(nn.Module):
    """
    STRICT position-aware token alignment (v5):
    Each token at position (i,j) must match the corresponding token at (i,j) in other view.
    NO shift tolerance - forces the model to learn precise spatial correspondence.
    """
    def __init__(self, shift_px: int = 0, shift_temp: float = 0.07):
        super().__init__()
        self.shift_px = 0  # FORCE to 0 for strict alignment
        self.shift_temp = float(shift_temp)
        print("⚠️  AlignmentLoss v5: STRICT position matching (shift_px=0)")

    def _interpolate_tokens(self, z: torch.Tensor, grid_size: Tuple[int, int], target_size: Tuple[int, int]) -> torch.Tensor:
        if grid_size == target_size:
            return z
        B, N, D = z.shape
        z_spatial = z.transpose(1, 2).contiguous().view(B, D, grid_size[0], grid_size[1])
        z_interp = F.interpolate(z_spatial, size=target_size, mode='bilinear', align_corners=False)
        return z_interp.view(B, D, -1).transpose(1, 2).contiguous()

    @staticmethod
    def _weights_to_tokens(w: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        # w: [B,1,H,W] -> resample -> flatten -> normalize
        w_r = F.interpolate(w, size=target_hw, mode='bilinear', align_corners=False)
        B = w_r.shape[0]
        w_flat = w_r.view(B, -1)
        w_flat = w_flat / (w_flat.sum(dim=1, keepdim=True) + 1e-10)
        return w_flat

    def forward(self, z1, z2, w1, w2, gs1, gs2):
        target_hw = (max(int(gs1[0]), int(gs2[0])), max(int(gs1[1]), int(gs2[1])))

        z1_common = self._interpolate_tokens(z1, gs1, target_hw)
        z2_common = self._interpolate_tokens(z2, gs2, target_hw)

        z1n = F.normalize(z1_common, dim=-1)
        z2n = F.normalize(z2_common, dim=-1)

        B, N, D = z1n.shape
        H, W = target_hw
        if N != H * W:
            raise RuntimeError(f"Token count mismatch: N={N}, H*W={H*W}. Check patching/interpolation.")

        # STRICT: Direct token-to-token similarity at EXACT same positions
        # z1[i,j] must match z2[i,j], no shift tolerance
        sim = (z1n * z2n).sum(dim=-1)  # [B, N] - cosine similarity per token
        
        # Weight by information maps
        w_avg = 0.5 * (self._weights_to_tokens(w1, target_hw) + self._weights_to_tokens(w2, target_hw))
        
        align_loss = ((1.0 - sim) * w_avg).sum(dim=1).mean()

        return {'loss': align_loss, 'pos_similarity': sim.mean()}


class VICReg(nn.Module):
    def __init__(self, var_weight: float = 1.0, cov_weight: float = 0.04):
        super().__init__()
        self.var_weight = float(var_weight)
        self.cov_weight = float(cov_weight)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, D = z.shape
        z_flat = z.reshape(-1, D)

        std = torch.sqrt(z_flat.var(dim=0) + 1e-4)
        var_loss = F.relu(1.0 - std).mean()

        z_centered = z_flat - z_flat.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (z_flat.shape[0] - 1)
        mask = ~torch.eye(D, device=z.device).bool()
        cov_loss = cov[mask].pow(2).mean()

        return {
            'var_loss': var_loss,
            'cov_loss': cov_loss,
            'reg_loss': self.var_weight * var_loss + self.cov_weight * cov_loss
        }


# =============================================================================
# FULL MODEL (Student + EMA Teacher + Predictor)
# =============================================================================

class JAISPFoundationV5(nn.Module):
    """
    v5: STRICT position encoding via exact token alignment.
    
    Key idea:
      - Student produces (z, p) where p = predictor(z)
      - Teacher (EMA) produces targets z_t (no grad)
      - Alignment matches student p(view1) -> teacher z(view2) AT EXACT positions
      - NO shift tolerance - token[i,j] must match token[i,j]
      - Info-map weights focus the loss on informative tokens (objects, edges)
    """
    def __init__(self,
                 band_names: List[str],
                 stem_ch: int = 64,
                 embed_dim: int = 256,
                 proj_dim: int = 256,
                 depth: int = 6,
                 patch_size: int = 16,
                 shift_px: int = 0,  # FORCED to 0 in v5
                 shift_temp: float = 0.07,
                 var_weight: float = 1.0,
                 cov_weight: float = 0.04):
        super().__init__()
        self.band_names = list(band_names)
        self.embed_dim = int(embed_dim)
        self.proj_dim = int(proj_dim)
        
        if shift_px != 0:
            print(f"⚠️  v5 WARNING: shift_px={shift_px} ignored, forcing to 0 for strict position matching")

        # Student
        self.stems = nn.ModuleDict({name: BandStem(stem_ch) for name in self.band_names})
        self.info_maps = nn.ModuleDict({name: InformationMap() for name in self.band_names})
        self.encoder = SharedEncoder(stem_ch, self.embed_dim, depth, patch_size)
        self.projector = ProjectionHead(self.embed_dim, self.proj_dim)

        # Predictor (student only)
        self.predictor = nn.Sequential(
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.GELU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        )

        # Losses
        self.align_loss = AlignmentLoss(shift_px=0, shift_temp=shift_temp)  # STRICT v5
        self.vicreg = VICReg(var_weight=var_weight, cov_weight=cov_weight)

        # Init student weights first
        self._init_weights()

        # Teacher (EMA copies of student; no grads)
        self.teacher_stems = copy.deepcopy(self.stems)
        self.teacher_encoder = copy.deepcopy(self.encoder)
        self.teacher_projector = copy.deepcopy(self.projector)
        self._freeze_teacher()

    def _freeze_teacher(self):
        for m in [self.teacher_stems, self.teacher_encoder, self.teacher_projector]:
            for p in m.parameters():
                p.requires_grad = False

    def _init_weights(self):
        # only init learnable modules once; teacher is deep-copied from initialized student
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def update_teacher(self, m: float = 0.996):
        """EMA update: teacher = m*teacher + (1-m)*student"""
        m = float(m)

        def _ema_update(teacher, student):
            for pt, ps in zip(teacher.parameters(), student.parameters()):
                pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))

        _ema_update(self.teacher_stems, self.stems)
        _ema_update(self.teacher_encoder, self.encoder)
        _ema_update(self.teacher_projector, self.projector)

    def encode(self, image: torch.Tensor, rms: torch.Tensor, band: str) -> Dict:
        weights = self.info_maps[band](image, rms)                 # [B,1,H,W]
        feat = self.stems[band](image, rms)
        tokens, grid_size = self.encoder(feat)
        z = self.projector(tokens)                                 # [B,N,D]
        p = self.predictor(z)                                      # [B,N,D]
        return {'z': z, 'p': p, 'weights': weights, 'grid_size': grid_size}

    @torch.no_grad()
    def encode_teacher(self, image: torch.Tensor, rms: torch.Tensor, band: str) -> Dict:
        feat = self.teacher_stems[band](image, rms)
        tokens, grid_size = self.teacher_encoder(feat)
        z = self.teacher_projector(tokens)
        return {'z': z, 'grid_size': grid_size}

    def forward(self, batch: Dict) -> Dict:
        device = next(self.parameters()).device
        is_variable = isinstance(batch['view1_image'], list)

        if not is_variable:
            return self._forward_fixed(batch, device)

        B = len(batch['view1_image'])
        packs = []  # (out1_s, out2_s, out1_t, out2_t, band1, band2)

        for i in range(B):
            img1 = torch.nan_to_num(batch['view1_image'][i].to(device)).clamp(min=-100)
            rms1 = torch.nan_to_num(batch['view1_rms'][i].to(device), nan=1.0).clamp(min=1e-10)
            band1 = batch['view1_band'][i]

            img2 = torch.nan_to_num(batch['view2_image'][i].to(device)).clamp(min=-100)
            rms2 = torch.nan_to_num(batch['view2_rms'][i].to(device), nan=1.0).clamp(min=1e-10)
            band2 = batch['view2_band'][i]

            out1 = self.encode(img1.unsqueeze(0), rms1.unsqueeze(0), band1)               # student
            out2 = self.encode(img2.unsqueeze(0), rms2.unsqueeze(0), band2)               # student
            t1 = self.encode_teacher(img1.unsqueeze(0), rms1.unsqueeze(0), band1)         # teacher
            t2 = self.encode_teacher(img2.unsqueeze(0), rms2.unsqueeze(0), band2)         # teacher

            packs.append((out1, out2, t1, t2, band1, band2))

        align_l = torch.tensor(0.0, device=device)
        var_l = torch.tensor(0.0, device=device)
        cov_l = torch.tensor(0.0, device=device)
        tok_sim_l = torch.tensor(0.0, device=device)
        glob_sim_l = torch.tensor(0.0, device=device)

        for i in range(B):
            out1, out2, t1, t2, band1, band2 = packs[i]

            w1, w2 = out1['weights'], out2['weights']
            gs1s, gs2s = out1['grid_size'], out2['grid_size']
            gs1t, gs2t = t1['grid_size'], t2['grid_size']

            p1, p2 = out1['p'], out2['p']
            z1t, z2t = t1['z'], t2['z']

            # alignment: student predicts teacher target from the other view (stop-grad is implicit via teacher)
            out12 = self.align_loss(p1, z2t, w1, w2, gs1s, gs2t)
            out21 = self.align_loss(p2, z1t, w2, w1, gs2s, gs1t)

            al = 0.5 * (out12['loss'] + out21['loss'])
            tok_sim = 0.5 * (out12['pos_similarity'] + out21['pos_similarity'])

            # regularization on student z (not predictor)
            z1s, z2s = out1['z'], out2['z']
            vic1, vic2 = self.vicreg(z1s), self.vicreg(z2s)
            vloss = 0.5 * (vic1['var_loss'] + vic2['var_loss'])
            closs = 0.5 * (vic1['cov_loss'] + vic2['cov_loss'])

            with torch.no_grad():
                g12 = (F.normalize(p1.mean(dim=1), dim=-1) * F.normalize(z2t.mean(dim=1), dim=-1)).sum(dim=-1).mean()
                g21 = (F.normalize(p2.mean(dim=1), dim=-1) * F.normalize(z1t.mean(dim=1), dim=-1)).sum(dim=-1).mean()
                glob_sim = 0.5 * (g12 + g21)

            align_l += al
            var_l += vloss
            cov_l += closs
            tok_sim_l += tok_sim
            glob_sim_l += glob_sim

        align_l = align_l / B
        var_l = var_l / B
        cov_l = cov_l / B
        tok_sim_l = tok_sim_l / B
        glob_sim_l = glob_sim_l / B

        total_loss = align_l + var_l + self.vicreg.cov_weight * cov_l

        # representative sample for visualization
        out1_0, out2_0, _, _, band1_0, band2_0 = packs[0]
        return {
            'loss': total_loss,
            'align_loss': align_l,
            'var_loss': var_l,
            'cov_loss': cov_l,
            'token_sim': float(tok_sim_l.detach().item()),
            'global_sim': float(glob_sim_l.detach().item()),

            'z1': out1_0['z'],                  # student z
            'z2': out2_0['z'],
            'weights1': out1_0['weights'],
            'weights2': out2_0['weights'],
            'grid_size1': out1_0['grid_size'],
            'grid_size2': out2_0['grid_size'],
            'band1': band1_0,
            'band2': band2_0,
        }

    def _forward_fixed(self, batch, device):
        raise NotImplementedError(
            "_forward_fixed is not implemented. "
            "This model currently expects variable-size (list-based) inputs."
        )


def create_optimizer(model, lr=3e-4, weight_decay=0.05):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, warmup_epochs, total_epochs):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=1e-6)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])