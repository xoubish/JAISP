"""Differentiable variable-projection photometry; amplitudes are measured from pixels.

Covariances use tangent-plane arcsec². Fluxes retain each input image's units.
No normalization after cropping: wings outside the cutout remain unobserved.
"""
import math
import numpy as np
import torch
from torch import nn

BANDS = tuple('rubin_' + b for b in 'ugrizy') + tuple('euclid_' + b for b in ('VIS', 'Y', 'J', 'H'))


class MorphologyHead(nn.Module):
    """Bounded correction to a VIS moment ellipse from frozen local features.

    Unit-determinant shear plus size correction preserve positive definiteness.
    Identity initialization makes the initial model exactly the image baseline.
    """
    def __init__(self, channels=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(channels * 9), nn.Linear(channels * 9, 64),
                                 nn.GELU(), nn.Linear(64, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, covariance):
        p = .6 * torch.tanh(self.net(features.detach().flatten(1)))
        zero = torch.zeros_like(p[:, 0])
        transform = torch.stack((torch.exp(p[:, 0]), zero,
                                 p[:, 2], torch.exp(p[:, 1])), -1).reshape(-1, 2, 2)
        return transform @ covariance @ transform.transpose(-1, -2)


def templates(positions, covariance_sky, sky_to_pixel, psf_sigma, shape, oversample=3):
    """Pixel-integrated elliptical Gaussian sources convolved with circular PSF.

    PSF sigma is in native pixels; positions are cutout-relative native pixels.
    Gauss–Legendre quadrature integrates over pixel area. PSF is an explicit pilot
    approximation, not the legacy learned ePSF. Returns [H*W, N].
    """
    dtype, device = covariance_sky.dtype, covariance_sky.device
    cov = sky_to_pixel @ covariance_sky @ sky_to_pixel.transpose(-1, -2)
    cov = cov + torch.eye(2, dtype=dtype, device=device) * psf_sigma ** 2
    inv = torch.linalg.inv(cov)
    norm = 2 * math.pi * torch.sqrt(torch.linalg.det(cov))
    yy, xx = torch.meshgrid(torch.arange(shape[0], dtype=dtype, device=device),
                            torch.arange(shape[1], dtype=dtype, device=device), indexing='ij')
    grid = torch.stack((xx.flatten(), yy.flatten()), -1)
    result = torch.zeros((grid.shape[0], len(positions)), dtype=dtype, device=device)
    nodes, weights = np.polynomial.legendre.leggauss(oversample)
    for oy, wy in zip(nodes / 2, weights / 2):
        for ox, wx in zip(nodes / 2, weights / 2):
            offset = grid.new_tensor([ox, oy])
            delta = grid[:, None, :] + offset - positions[None, :, :]
            exponent = torch.einsum('pni,nij,pnj->pn', delta, inv, delta)
            result = result + (wx * wy) * torch.exp(-.5 * exponent) / norm
    return result



def fit_flux(image, variance, template, mask=None, fit_background=True):
    """Joint signed flux + constant sky WLS, with conditional covariance.

    Signed amplitudes avoid positivity bias at low S/N. Singular blends are
    flagged by condition number; no hidden ridge or clipped pseudo-NNLS.
    Covariance ignores morphology/PSF uncertainty and correlated image noise.
    """
    # Float64 linear algebra protects strongly blended, differently scaled columns.
    template = template.double()
    y, v = image.flatten().double(), variance.flatten().double()
    valid = torch.isfinite(y) & torch.isfinite(v) & (v > 0)
    if mask is not None:
        valid = valid & mask.flatten().bool()
    # Optional fixed background: callers subtract their own robust estimate and drop the constant column.
    a = (torch.cat((template, torch.ones_like(template[:, :1])), 1) if fit_background else template)[valid]
    if int(valid.sum()) <= a.shape[1]:
        raise ValueError('Not enough valid pixels to fit scene')
    weight = torch.rsqrt(v[valid])
    aw, yw = a * weight[:, None], y[valid] * weight
    # Scale columns to avoid flux/background numerical conditioning artifacts.
    scale = torch.linalg.vector_norm(aw, dim=0).clamp_min(1e-15)
    normalized = aw / scale
    normal = normalized.T @ normalized
    condition = torch.linalg.cond(normal.detach())
    if not torch.isfinite(condition) or condition > 1e8:
        raise ValueError('Degenerate blend (normalized normal-matrix condition > 1e8)')
    inverse = torch.linalg.inv(normal)
    coeff = torch.linalg.solve(normal, normalized.T @ yw) / scale
    cov = inverse / scale[:, None] / scale[None, :]
    residual = (aw @ coeff - yw)
    if not fit_background:
        return dict(flux=coeff, error=torch.sqrt(cov.diagonal().clamp_min(0)), covariance=cov, background=coeff.new_tensor(0.),
                    loss=residual.square().mean(), chi2=residual.square().sum(), dof=int(valid.sum())-a.shape[1], condition=condition,
                    model=(template @ coeff).reshape(image.shape))
    return dict(flux=coeff[:-1], error=torch.sqrt(cov.diagonal()[:-1].clamp_min(0)),
                covariance=cov[:-1, :-1], background=coeff[-1],
                loss=residual.square().mean(), chi2=residual.square().sum(),
                dof=int(valid.sum())-a.shape[1], condition=condition,
                model=(template @ coeff[:-1] + coeff[-1]).reshape(image.shape))


def fit_scene(scene, covariance):
    results = {}
    for band, d in scene['bands'].items():
        t = templates(d['positions'], covariance, d['sky_to_pixel'],
                      d['psf_sigma'], d['image'].shape)
        # Sources beyond a native footprint have no measurable flux in that band.
        # Keep every template with >= 1e-5 of its total flux inside the cutout.
        active = (t.detach().sum(0) > 1e-5)
        result = fit_flux(d['image'], d['variance'], t[:, active], d['mask'])
        result['source_indices'] = torch.where(active)[0]
        result['footprint_fraction'] = t[:, active].detach().sum(0)
        results[band] = result
    return results


class ConstantMorphologyHead(MorphologyHead):
    """Same trainable network, one fixed training-mean representation for everyone.

    Separates image-likelihood calibration from object-dependent feature value.
    """
    def __init__(self, mean_features):
        super().__init__(mean_features.shape[1])
        self.register_buffer('mean_features',mean_features.detach().clone())

    def forward(self, features, covariance):
        return super().forward(self.mean_features.expand(len(features),-1,-1,-1),covariance)
