"""CenterNet loss: focal loss on heatmap + L1 on offsets/flux at GT positions.

No Hungarian matching -- each pixel gets direct supervision from the
nearest ground-truth source via Gaussian-splat heatmap targets.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def render_heatmap_targets(
    gt_centroids: List[torch.Tensor],   # list of [M_i, 2] normalized (x, y)
    feat_h: int,
    feat_w: int,
    sigma: float = 2.0,
    device: torch.device = None,
    gt_weights: List[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render Gaussian-splat heatmap targets from normalized GT centroids.

    Returns
    -------
    hm_target  : [B, 1, H, W]  heatmap target (max of Gaussians, in [0, 1])
    off_target : [B, 2, H, W]  sub-pixel offset target (fractional part)
    off_mask   : [B, 1, H, W]  mask: 1 at GT integer positions, 0 elsewhere
    weight_map : [B, 1, H, W]  source confidence weight at GT positions
    n_sources  : [B]            number of GT sources per sample
    """
    B = len(gt_centroids)
    hm = torch.zeros(B, 1, feat_h, feat_w, device=device)
    off = torch.zeros(B, 2, feat_h, feat_w, device=device)
    mask = torch.zeros(B, 1, feat_h, feat_w, device=device)
    weight_map = torch.zeros(B, 1, feat_h, feat_w, device=device)
    n_src = torch.zeros(B, device=device)

    # Pre-compute 1-D coordinate vectors (reused for bounding-box sub-grids)
    gy = torch.arange(feat_h, device=device, dtype=torch.float32)
    gx = torch.arange(feat_w, device=device, dtype=torch.float32)

    inv_2sig2 = 1.0 / (2.0 * sigma ** 2)
    # Cutoff at 3σ: outside this radius the Gaussian is < 0.01 anyway
    radius = int(3.0 * sigma) + 1

    for b in range(B):
        pts = gt_centroids[b]  # [M, 2] normalized (x, y)
        if pts.shape[0] == 0:
            continue

        pts = pts.to(device=device, dtype=torch.float32)
        M = pts.shape[0]
        n_src[b] = M
        if gt_weights is not None and gt_weights[b] is not None and gt_weights[b].shape[0] == M:
            weights = gt_weights[b].to(device=device, dtype=torch.float32).clamp(min=0.0)
        else:
            weights = torch.ones(M, device=device, dtype=torch.float32)

        # Convert normalized coords to feature-map pixel coords
        cx = pts[:, 0] * (feat_w - 1)   # [M]
        cy = pts[:, 1] * (feat_h - 1)   # [M]
        cx_int = cx.round().long().clamp(0, feat_w - 1)
        cy_int = cy.round().long().clamp(0, feat_h - 1)

        # Bounded Gaussian rendering: only evaluate within a (2r+1)² bounding
        # box around each source.  At VIS resolution (1040×1040) the full
        # vectorised [M, H, W] approach would allocate ~2 GB per sample;
        # bounded rendering keeps each sub-grid to (2r+1)² ≈ 196 elements.
        for i in range(M):
            x0 = max(0,       cx_int[i].item() - radius)
            x1 = min(feat_w,  cx_int[i].item() + radius + 1)
            y0 = max(0,       cy_int[i].item() - radius)
            y1 = min(feat_h,  cy_int[i].item() + radius + 1)

            sub_x = gx[x0:x1]          # [w]
            sub_y = gy[y0:y1]           # [h]
            sub_gy, sub_gx = torch.meshgrid(sub_y, sub_x, indexing='ij')  # [h, w]
            gauss = torch.exp(
                -((sub_gx - cx[i]) ** 2 + (sub_gy - cy[i]) ** 2) * inv_2sig2
            )
            hm[b, 0, y0:y1, x0:x1] = torch.max(hm[b, 0, y0:y1, x0:x1], gauss)

        # Force exact 1.0 at integer centers so focal loss has positive pixels
        hm[b, 0, cy_int, cx_int] = 1.0

        off[b, 0, cy_int, cx_int] = cx - cx_int.float()
        off[b, 1, cy_int, cx_int] = cy - cy_int.float()
        mask[b, 0, cy_int, cx_int] = 1.0
        weight_map[b, 0, cy_int, cx_int] = torch.maximum(
            weight_map[b, 0, cy_int, cx_int],
            weights,
        )

    return hm, off, mask, weight_map, n_src


def focal_loss(
    pred: torch.Tensor,   # [B, 1, H, W] in (0, 1)
    target: torch.Tensor, # [B, 1, H, W] in [0, 1]
    alpha: float = 2.0,
    beta: float = 4.0,
    ignore_mask: torch.Tensor = None,  # [B, 1, H, W] bool; suppress negative loss only
    pos_weight: torch.Tensor = None,    # [B, 1, H, W] float; weights positive centers
) -> torch.Tensor:
    """Modified focal loss from CornerNet / CenterNet.

    At positive locations (target == 1):
        loss = -(1 - pred)^alpha * log(pred)
    At negative locations (target < 1):
        loss = -(1 - target)^beta * pred^alpha * log(1 - pred)

    Returns scalar loss normalized by number of positive locations.
    """
    pred = pred.clamp(1e-6, 1.0 - 1e-6)

    pos_mask = target.eq(1).float()
    neg_mask = 1.0 - pos_mask
    if ignore_mask is not None:
        neg_mask = neg_mask * (~ignore_mask.bool()).float()

    if pos_weight is None:
        pos_weight = pos_mask
    else:
        pos_weight = pos_weight.to(device=pred.device, dtype=pred.dtype) * pos_mask

    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_weight
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg_mask

    n_pos = pos_weight.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


class CenterNetLoss(nn.Module):
    """Combined CenterNet loss: focal (heatmap) + L1 (offset, flux).

    Parameters
    ----------
    lambda_hm   : weight on heatmap focal loss
    lambda_off  : weight on offset L1 loss
    lambda_flux : weight on flux L1 loss
    sigma       : Gaussian sigma for heatmap target rendering
    """

    def __init__(
        self,
        lambda_hm: float = 1.0,
        lambda_off: float = 1.0,
        lambda_flux: float = 0.1,
        sigma: float = 2.0,
    ):
        super().__init__()
        self.lambda_hm = lambda_hm
        self.lambda_off = lambda_off
        self.lambda_flux = lambda_flux
        self.sigma = sigma

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt_centroids: List[torch.Tensor],
        gt_log_flux: List[torch.Tensor] = None,
        ignore_masks: torch.Tensor = None,
        gt_weights: List[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hm_pred = pred['heatmap']                    # [B, 1, H, W]
        off_pred = pred['offset']                    # [B, 2, H, W]
        flux_pred = pred['log_flux']                 # [B, H, W]
        B, _, H, W = hm_pred.shape

        hm_target, off_target, off_mask, weight_map, n_src = render_heatmap_targets(
            gt_centroids, H, W, sigma=self.sigma, device=hm_pred.device,
            gt_weights=gt_weights,
        )

        ignore = None
        if ignore_masks is not None:
            ignore = ignore_masks.to(device=hm_pred.device)
            if ignore.ndim == 3:
                ignore = ignore.unsqueeze(1)
            elif ignore.ndim != 4:
                raise ValueError('ignore_masks must have shape [B,H,W] or [B,1,H,W]')
            if ignore.shape[-2:] != (H, W):
                import torch.nn.functional as F
                ignore = F.interpolate(ignore.float(), size=(H, W), mode='nearest') > 0.5
            else:
                ignore = ignore.bool()

        # Heatmap focal loss
        loss_hm = focal_loss(hm_pred, hm_target, ignore_mask=ignore, pos_weight=weight_map)

        # Offset L1 (only at GT positions)
        weighted_mask = off_mask * weight_map.clamp(min=0.0)
        n_pos = weighted_mask.sum().clamp(min=1.0)
        loss_off = (torch.abs(off_pred - off_target) * weighted_mask).sum() / n_pos

        # Flux L1 (only at GT positions, if available)
        loss_flux = hm_pred.new_zeros(1)
        if gt_log_flux is not None:
            flux_target = torch.zeros(B, H, W, device=hm_pred.device)
            for b in range(B):
                if gt_log_flux[b] is not None and gt_log_flux[b].shape[0] > 0:
                    pts = gt_centroids[b].to(device=hm_pred.device, dtype=torch.float32)
                    cx_int = (pts[:, 0] * (W - 1)).round().long().clamp(0, W - 1)
                    cy_int = (pts[:, 1] * (H - 1)).round().long().clamp(0, H - 1)
                    fl = gt_log_flux[b].to(device=hm_pred.device, dtype=torch.float32)
                    flux_target[b, cy_int, cx_int] = fl
            loss_flux = (torch.abs(flux_pred - flux_target) * weighted_mask.squeeze(1)).sum() / n_pos

        losses = {
            'loss_hm':    self.lambda_hm   * loss_hm,
            'loss_off':   self.lambda_off  * loss_off,
            'loss_flux':  self.lambda_flux * loss_flux,
        }
        losses['loss_total'] = sum(losses.values())
        losses['n_sources'] = n_src.mean()
        losses['mean_source_weight'] = (
            weight_map.sum() / off_mask.sum().clamp(min=1.0)
        )
        return losses
