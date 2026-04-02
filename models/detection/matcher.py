"""
Hungarian matcher for DETR-style training.

Matches N_q predictions to M ground-truth sources by minimising:
    cost = λ_pos * L1(pred_xy, gt_xy)
         + λ_conf * (-sigmoid(pred_conf))

Unmatched predictions are penalised by λ_noobj * BCE(pred_conf, 0).

Returns per-sample index pairs (pred_idx, gt_idx) for the matched set.
"""

from typing import Dict, List, Tuple

import torch
from scipy.optimize import linear_sum_assignment


class HungarianMatcher:
    """
    Compute optimal assignment between predictions and ground truth.

    Parameters
    ----------
    cost_pos  : weight on L1 centroid distance
    cost_conf : weight on objectness for matched pairs
    """

    def __init__(
        self,
        cost_pos:  float = 5.0,
        cost_conf: float = 1.0,
    ):
        self.cost_pos  = cost_pos
        self.cost_conf = cost_conf

    @torch.no_grad()
    def __call__(
        self,
        pred_centroids: torch.Tensor,   # [N_q, 2]  normalised
        pred_conf:      torch.Tensor,   # [N_q]     logits
        gt_centroids:   torch.Tensor,   # [M, 2]    normalised
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        pred_idx : [K] matched prediction indices
        gt_idx   : [K] matched ground-truth indices
        (K ≤ min(N_q, M))
        """
        M = gt_centroids.shape[0]
        if M == 0:
            device = pred_centroids.device
            return torch.zeros(0, dtype=torch.long, device=device), \
                   torch.zeros(0, dtype=torch.long, device=device)

        # Cost matrix [N_q, M]
        c_pos = torch.cdist(pred_centroids.float(), gt_centroids.float(), p=1)  # [N_q, M]

        # Objectness: push conf high for matched predictions
        c_conf = -pred_conf.float().sigmoid().unsqueeze(1).expand(-1, M)  # [N_q, M]

        cost = self.cost_pos * c_pos + self.cost_conf * c_conf  # [N_q, M]

        cost_np = cost.cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(cost_np)
        device = pred_centroids.device
        return (torch.as_tensor(row_idx, dtype=torch.long, device=device),
                torch.as_tensor(col_idx, dtype=torch.long, device=device))


class DetectionLoss(torch.nn.Module):
    """
    DETR-style loss: position L1 + objectness BCE (matched/unmatched).

    Classification is dropped — with a single 'source' class the conf head
    already handles object-vs-background.

    Parameters
    ----------
    cost_pos      : position L1 weight in Hungarian cost matrix
    cost_conf     : conf weight in Hungarian cost matrix
    lambda_pos    : position L1 weight in final loss
    lambda_conf   : objectness BCE weight for matched queries
    lambda_noobj  : objectness BCE weight for unmatched queries
    """

    def __init__(
        self,
        cost_pos:     float = 5.0,
        cost_conf:    float = 1.0,
        lambda_pos:   float = 5.0,
        lambda_conf:  float = 2.0,
        lambda_noobj: float = 0.5,
    ):
        super().__init__()
        self.matcher      = HungarianMatcher(cost_pos, cost_conf)
        self.lambda_pos   = lambda_pos
        self.lambda_conf  = lambda_conf
        self.lambda_noobj = lambda_noobj

    def forward(
        self,
        pred_centroids: torch.Tensor,   # [B, N_q, 2]
        pred_logits:    torch.Tensor,   # [B, N_q, C]  (unused, kept for API compat)
        pred_conf:      torch.Tensor,   # [B, N_q]
        gt_centroids:   List[torch.Tensor],  # list of [M_i, 2]
        gt_classes:     List[torch.Tensor],  # list of [M_i] int64  (unused)
    ) -> Dict[str, torch.Tensor]:
        B = pred_centroids.shape[0]
        loss_pos        = pred_centroids.new_zeros(1)
        loss_conf_obj   = pred_centroids.new_zeros(1)
        loss_conf_noobj = pred_centroids.new_zeros(1)
        n_matched_total = 0

        for i in range(B):
            pi = pred_centroids[i]   # [N_q, 2]
            ci = pred_conf[i]        # [N_q]
            gi = gt_centroids[i]     # [M, 2]

            pred_idx, gt_idx = self.matcher(pi, ci, gi)
            n_matched = pred_idx.shape[0]
            n_matched_total += n_matched

            if n_matched > 0:
                # Position L1
                loss_pos = loss_pos + F_l1(pi[pred_idx], gi[gt_idx])

                # Objectness: matched → 1
                loss_conf_obj = loss_conf_obj + torch.nn.functional.binary_cross_entropy_with_logits(
                    ci[pred_idx],
                    torch.ones(n_matched, device=ci.device),
                    reduction='mean',
                )

            # Objectness: unmatched → 0
            unmatched = torch.ones(ci.shape[0], dtype=torch.bool, device=ci.device)
            if n_matched > 0:
                unmatched[pred_idx] = False
            if unmatched.any():
                loss_conf_noobj = loss_conf_noobj + torch.nn.functional.binary_cross_entropy_with_logits(
                    ci[unmatched],
                    torch.zeros(unmatched.sum(), device=ci.device),
                    reduction='mean',
                )

        denom = max(B, 1)
        losses = {
            'loss_pos':        self.lambda_pos   * loss_pos        / denom,
            'loss_conf_obj':   self.lambda_conf  * loss_conf_obj   / denom,
            'loss_conf_noobj': self.lambda_noobj * loss_conf_noobj / denom,
        }
        losses['loss_total'] = sum(losses.values())
        losses['n_matched']  = torch.tensor(float(n_matched_total) / denom)
        return losses


def F_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.l1_loss(pred, target, reduction='mean')
