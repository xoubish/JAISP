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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render Gaussian-splat heatmap targets from normalized GT centroids.

    Returns
    -------
    hm_target  : [B, 1, H, W]  heatmap target (max of Gaussians, in [0, 1])
    off_target : [B, 2, H, W]  sub-pixel offset target (fractional part)
    off_mask   : [B, 1, H, W]  mask: 1 at GT integer positions, 0 elsewhere
    n_sources  : [B]            number of GT sources per sample
    """
    B = len(gt_centroids)
    hm = torch.zeros(B, 1, feat_h, feat_w, device=device)
    off = torch.zeros(B, 2, feat_h, feat_w, device=device)
    mask = torch.zeros(B, 1, feat_h, feat_w, device=device)
    n_src = torch.zeros(B, device=device)

    # Pre-compute coordinate grids
    gy = torch.arange(feat_h, device=device, dtype=torch.float32)
    gx = torch.arange(feat_w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # [H, W]

    radius = int(3 * sigma + 1)  # Gaussian clamp radius

    for b in range(B):
        pts = gt_centroids[b]  # [M, 2] normalized (x, y)
        if pts.shape[0] == 0:
            continue

        pts = pts.to(device=device, dtype=torch.float32)
        n_src[b] = pts.shape[0]

        # Convert normalized coords to feature-map pixel coords
        cx = pts[:, 0] * (feat_w - 1)   # [M]
        cy = pts[:, 1] * (feat_h - 1)   # [M]

        # Integer positions and fractional offsets
        cx_int = cx.round().long().clamp(0, feat_w - 1)
        cy_int = cy.round().long().clamp(0, feat_h - 1)
        dx = cx - cx_int.float()
        dy = cy - cy_int.float()

        for i in range(pts.shape[0]):
            # Clamp Gaussian to a local window for efficiency
            y_lo = max(0, cy_int[i].item() - radius)
            y_hi = min(feat_h, cy_int[i].item() + radius + 1)
            x_lo = max(0, cx_int[i].item() - radius)
            x_hi = min(feat_w, cx_int[i].item() + radius + 1)

            local_y = grid_y[y_lo:y_hi, x_lo:x_hi]
            local_x = grid_x[y_lo:y_hi, x_lo:x_hi]

            gauss = torch.exp(
                -((local_x - cx[i]) ** 2 + (local_y - cy[i]) ** 2)
                / (2.0 * sigma ** 2)
            )
            # Element-wise max (CenterNet convention: overlapping sources)
            hm[b, 0, y_lo:y_hi, x_lo:x_hi] = torch.max(
                hm[b, 0, y_lo:y_hi, x_lo:x_hi], gauss
            )

            # Offset target at the integer position
            off[b, 0, cy_int[i], cx_int[i]] = dx[i]
            off[b, 1, cy_int[i], cx_int[i]] = dy[i]
            mask[b, 0, cy_int[i], cx_int[i]] = 1.0

    return hm, off, mask, n_src


def focal_loss(
    pred: torch.Tensor,   # [B, 1, H, W] in (0, 1)
    target: torch.Tensor, # [B, 1, H, W] in [0, 1]
    alpha: float = 2.0,
    beta: float = 4.0,
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

    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg_mask

    n_pos = pos_mask.sum().clamp(min=1.0)
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
    ) -> Dict[str, torch.Tensor]:
        hm_pred = pred['heatmap']                    # [B, 1, H, W]
        off_pred = pred['offset']                    # [B, 2, H, W]
        flux_pred = pred['log_flux']                 # [B, H, W]
        B, _, H, W = hm_pred.shape

        hm_target, off_target, off_mask, n_src = render_heatmap_targets(
            gt_centroids, H, W, sigma=self.sigma, device=hm_pred.device,
        )

        # Heatmap focal loss
        loss_hm = focal_loss(hm_pred, hm_target)

        # Offset L1 (only at GT positions)
        n_pos = off_mask.sum().clamp(min=1.0)
        loss_off = (torch.abs(off_pred - off_target) * off_mask).sum() / n_pos

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
            loss_flux = (torch.abs(flux_pred - flux_target) * off_mask.squeeze(1)).sum() / n_pos

        losses = {
            'loss_hm':    self.lambda_hm   * loss_hm,
            'loss_off':   self.lambda_off  * loss_off,
            'loss_flux':  self.lambda_flux * loss_flux,
        }
        losses['loss_total'] = sum(losses.values())
        losses['n_sources'] = n_src.mean()
        return losses
