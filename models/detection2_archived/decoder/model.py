"""Compact object vectors → learned templates → summed multiband images.

Positions and object presence are fixed in this warm-up. There is no supplied
PSF, analytic profile, target-pixel input, or dense feature-to-image connection.
The cached foundation features do see the full input image; this is ordinary
reconstruction, not held-out-pixel prediction.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..object_decoder import splat


class ObjectRenderer(nn.Module):
    def __init__(self, feature_dim, cfg, band_shapes, stamp_sizes, pixel_scales):
        super().__init__()
        self.band_shapes = [tuple(s) for s in band_shapes]
        self.stamp_sizes = list(stamp_sizes)
        self.pixel_scales = list(pixel_scales)
        self.flux_scale = cfg['flux_scale']
        self.n_bands = len(band_shapes)
        self.grid_side = max(stamp_sizes)
        dim = cfg['appearance_dim']
        self.projector = nn.Sequential(nn.Linear(feature_dim, cfg['hidden_dim']), nn.GELU(),
                                       nn.Linear(cfg['hidden_dim'], dim))
        self.band_embedding = nn.Embedding(self.n_bands, 8)
        self.shape_net = nn.Sequential(nn.Linear(dim+8, 64), nn.GELU(), nn.Linear(64, self.grid_side**2))
        self.flux_net = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, self.n_bands))
        self.background_net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, self.n_bands))
        nn.init.zeros_(self.background_net[-1].weight)
        nn.init.zeros_(self.background_net[-1].bias)

    def forward(self, features, present, positions):
        b, k, _ = features.shape
        if positions.shape != (b, k, self.n_bands, 2) or present.shape != (b, k):
            raise ValueError('Object arrays have inconsistent shapes')
        z = self.projector(features.float())
        gates = present.float()
        mean_z = (z*gates[..., None]).sum(1)/gates.sum(1, keepdim=True).clamp_min(1)
        background = self.background_net(mean_z)
        flux = F.softplus(self.flux_net(z))*self.flux_scale
        predictions, moments = [], []
        for band, (shape, side, scale) in enumerate(zip(self.band_shapes, self.stamp_sizes, self.pixel_scales)):
            emb = self.band_embedding.weight[band].expand(b, k, -1)
            logits = self.shape_net(torch.cat([z, emb], -1))
            templates = logits.softmax(-1).reshape(b*k, 1, self.grid_side, self.grid_side)
            if side != self.grid_side:
                templates = F.interpolate(templates, size=(side, side), mode='bilinear', align_corners=False)
                templates = templates/templates.sum((-2, -1), keepdim=True).clamp_min(1e-12)
            templates = templates.reshape(b, k, side, side)
            coordinates = (torch.arange(side, device=z.device, dtype=z.dtype)-(side-1)/2)*scale
            cx = (templates*coordinates[None, None, None, :]).sum((-2, -1))
            cy = (templates*coordinates[None, None, :, None]).sum((-2, -1))
            moments.append(cx.square()+cy.square())
            stamps = templates*(flux[..., band]*gates)[..., None, None]
            predictions.append(splat(stamps, positions[:, :, band], shape)+background[:, band, None, None])
        centering = (torch.stack(moments, -1)*gates[..., None]).sum()/(gates.sum()*self.n_bands).clamp_min(1)
        return {'images': predictions, 'background': background, 'flux': flux,
                'centering_arcsec2': centering}


def objective(output, batch, cfg):
    losses = []
    for prediction, target, valid in zip(output['images'], batch['targets'], batch['valid']):
        residual = prediction-target
        loss = F.huber_loss(residual, torch.zeros_like(residual), reduction='none', delta=cfg['huber_delta'])
        losses.append(((loss*valid).sum((-2, -1))/valid.sum((-2, -1)).clamp_min(1)).mean())
    reconstruction = torch.stack(losses).mean()
    return reconstruction+cfg['centering_weight']*output['centering_arcsec2'], reconstruction
