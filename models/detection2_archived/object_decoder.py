"""Untrained prototype of a learned object-to-image bottleneck.

No supplied PSFs, analytic source profiles, or dense feature-to-image shortcut.
This module is NOT yet connected to the first ablation's training loss.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def splat(stamps, centers_xy, image_shape):
    """Differentiable bilinear placement; gradients reach subpixel positions.

    stamps: [B,K,S,S], centers_xy: [B,K,2] in native-band pixels.
    Out-of-frame flux is discarded, without renormalizing boundary objects.
    """
    b, k, side, _ = stamps.shape
    h, w = image_shape
    offsets = torch.arange(side, device=stamps.device, dtype=stamps.dtype) - (side-1)/2
    yy, xx = torch.meshgrid(offsets, offsets, indexing='ij')
    x = centers_xy[..., 0, None, None] + xx
    y = centers_xy[..., 1, None, None] + yy
    x0, y0 = x.floor(), y.floor()
    dx, dy = x-x0, y-y0
    out = stamps.new_zeros(b, h*w)
    for ox, oy, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                           (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        xi, yi = x0.long()+ox, y0.long()+oy
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        index = (yi.clamp(0, h-1)*w + xi.clamp(0, w-1)).reshape(b, -1)
        out = out.scatter_add(1, index, (stamps*weight*valid).reshape(b, -1))
    return out.reshape(b, h, w)


class LearnedObjectDecoder(nn.Module):
    """Learn band-dependent appearance from compact object vectors.

    Input is an explicit catalogue: appearance [B,K,D], gates [B,K], and
    band_positions [B,K,n_bands,2]. Only constant per-image/band backgrounds
    are allowed. Native grid registration comes from existing image metadata.

    Gates are supplied by the caller. Learning object birth/death and preventing
    soft-gate/flux degeneracy require a separate objective and are not solved
    by this renderer. One decoded component is not automatically one real object.
    """
    def __init__(self, appearance_dim=16, n_bands=10, stamp_size=25):
        super().__init__()
        if stamp_size % 2 != 1:
            raise ValueError('Use odd stamp size for a defined central pixel')
        self.n_bands, self.stamp_size = n_bands, stamp_size
        self.band_embedding = nn.Embedding(n_bands, 8)
        self.shape_net = nn.Sequential(nn.Linear(appearance_dim+8, 64), nn.GELU(),
                                       nn.Linear(64, stamp_size**2))
        self.flux_net = nn.Sequential(nn.Linear(appearance_dim, 32), nn.GELU(),
                                      nn.Linear(32, n_bands))

    def forward(self, appearance, gates, band_positions, image_shapes, background=None):
        b, k, _ = appearance.shape
        if gates.shape != (b, k) or band_positions.shape != (b, k, self.n_bands, 2):
            raise ValueError('Catalogue arrays have inconsistent shapes')
        if len(image_shapes) != self.n_bands:
            raise ValueError('Provide a native image shape for every band')
        if background is not None and background.shape != (b, self.n_bands):
            raise ValueError('Background is one constant per image and band; no dense bypass')
        flux = F.softplus(self.flux_net(appearance))
        images = []
        for band in range(self.n_bands):
            embedding = self.band_embedding.weight[band].expand(b, k, -1)
            logits = self.shape_net(torch.cat([appearance, embedding], dim=-1))
            morphology = logits.softmax(-1).reshape(b, k, self.stamp_size, self.stamp_size)
            stamps = morphology * (gates*flux[..., band])[..., None, None]
            im = splat(stamps, band_positions[:, :, band], image_shapes[band])
            if background is not None:
                im = im + background[:, band, None, None]
            images.append(im)
        return images
