"""Amortised-scarlet head with an optional per-band branch and an optional implicit PSF.

Variants (see architecture_brainstorm/README.md):
    A0  ScarletHead as trained on chi-square (existing checkpoint, evaluated only)
    A1  same head, truth-supervised loss
    A2  A1 + per-band branch (band pixels, band residual, band PSF on the morphology grid)
    A3  A2 without foundation inputs (bottleneck and VIS stem replaced by zeros; encoder not run)
    A4  A2 without any PSF: the head outputs observed-space templates, no convolution, no PSF input

Fluxes are never predicted: every variant ends in the same signed linear solve
with a fixed robust background (amortised_scarlet.scene_fit).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import BANDS
from ..amortised_scarlet import (ScarletHead, MORPH_SIZE, MORPH_SCALE, N_KNOTS, BN_WINDOW, monotone_profile,
                                 render_unconvolved, fft_convolve, band_kernels, raw_pixel_windows, scene_fit, scene_to)

VARIANTS = dict(A0=dict(per_band=False, foundation=True, implicit_psf=False),
                A1=dict(per_band=False, foundation=True, implicit_psf=False),
                A2=dict(per_band=True, foundation=True, implicit_psf=False),
                A3=dict(per_band=True, foundation=False, implicit_psf=False),
                A4=dict(per_band=True, foundation=True, implicit_psf=True))


def morph_grid_points(d, device):
    """Band-pixel coordinates [N,K,K,2] of every morphology-grid point around every source."""
    k = MORPH_SIZE; g = (torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2) * MORPH_SCALE
    vv, uu = torch.meshgrid(g, g, indexing='ij'); sky = torch.stack((uu, vv), -1)
    return d['positions'].to(device)[:, None, None, :] + sky[None] @ d['sky_to_pixel'].to(device).T


def sample_map(value, points):
    """Bilinear sample of a band map [H,W] at points [N,K,K,2] (band pixels) -> [N,1,K,K]; zero outside."""
    grid = points / points.new_tensor([value.shape[1] - 1, value.shape[0] - 1]) * 2 - 1
    return F.grid_sample(value[None, None].expand(len(points), -1, -1, -1), grid, mode='bilinear', padding_mode='zeros', align_corners=True)


def psf_windows(scene, device):
    """Each source's PSF in every band resampled onto the morphology grid, peak-normalised: [N,10,K,K]."""
    out = []
    for band in BANDS:
        d = scene['bands'][band]; n = len(d['positions'])
        kernels = band_kernels(d, n, device); k = kernels.shape[-1]
        offsets = morph_grid_points(d, device) - d['positions'].to(device)[:, None, None, :]
        grid = (offsets + (k - 1) / 2) / (k - 1) * 2 - 1
        w = F.grid_sample(kernels[:, None], grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        out.append(w / w.flatten(1).amax(1).clamp_min(1e-12)[:, None, None, None])
    return torch.cat(out, 1)


def residual_windows(scene, results, device):
    """asinh-compressed chi residual of the current fit in every band, on the morphology grid: [N,10,K,K]."""
    out = []
    for band in BANDS:
        d = scene['bands'][band]; r = results[band]
        chi = torch.where(d['mask'], (d['image'] - r['model'].float()) / torch.sqrt(d['variance'].clamp_min(1e-20)), torch.zeros_like(d['image']))
        out.append(sample_map(torch.asinh(chi.detach() / 3), morph_grid_points(d, device)))
    return torch.cat(out, 1)


class TruthScarletHead(ScarletHead):
    version = 'truth_v1'

    def __init__(self, width=64, steps=2, per_band=False, implicit_psf=False, band_width=48, embed=8,
                 size_bound=.3, eps_bound=.2):
        super().__init__(width=width, steps=steps)
        if implicit_psf and not per_band: raise ValueError('The implicit-PSF head needs the per-band branch for band-dependent widths')
        self.per_band = per_band; self.implicit_psf = implicit_psf; self.psf_input = per_band and not implicit_psf
        # Without a PSF the band widths differ by up to ~4x (Rubin vs VIS): allow a wide size range.
        self.size_bound = 1.5 if implicit_psf else size_bound; self.eps_bound = eps_bound
        if per_band:
            self.band_embed = nn.Embedding(len(BANDS), embed)
            cin = width + 2 + embed + (1 if self.psf_input else 0)
            self.band_net = nn.Sequential(nn.Conv2d(cin, band_width, 3, padding=1), nn.GELU(),
                                          nn.Conv2d(band_width, band_width, 3, padding=2, dilation=2), nn.GELU(),
                                          nn.Conv2d(band_width, band_width, 3, padding=4, dilation=4), nn.GELU())
            self.band_eps = nn.Conv2d(band_width, 1, 1)
            self.band_vector = nn.Linear(2 * band_width, 3)      # mix-logit shift, log size, aperture factor
            for m in (self.band_eps, self.band_vector): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
            with torch.no_grad(): self.band_vector.bias[2] = -4.  # aperture factor starts at ~0.995

    def band_branch(self, feat, raw, residual, psf=None):
        n, w, k, _ = feat.shape; nb = len(BANDS)
        parts = [feat[:, None].expand(n, nb, w, k, k).reshape(n * nb, w, k, k),
                 raw.reshape(n * nb, 1, k, k), residual.reshape(n * nb, 1, k, k),
                 self.band_embed.weight[None, :, :, None, None].expand(n, nb, -1, k, k).reshape(n * nb, -1, k, k)]
        if self.psf_input: parts.append(psf.reshape(n * nb, 1, k, k))
        h = self.band_net(torch.cat(parts, 1))
        eps = self.band_eps(h)[:, 0].reshape(n, nb, k, k); vector = self.band_vector(self.pooled(h)).reshape(n, nb, 3)
        return dict(eps=eps, vector=vector)

    def band_morphologies(self, params, band_out=None):
        """({band: [N,1,K,K] unit-sum morphology}, {band: [N] template normalisation})."""
        if band_out is None:
            morph = self.morphologies(params)[0]
            return morph, {b: None for b in BANDS}
        v = params['vector']
        knots_c, knots_e = v[:, :N_KNOTS], v[:, N_KNOTS:2 * N_KNOTS]
        q = torch.exp(-F.softplus(v[:, 2 * N_KNOTS]) * .5)
        theta = torch.atan2(v[:, 2 * N_KNOTS + 2], v[:, 2 * N_KNOTS + 1] + 1.)
        c, s = torch.cos(theta), torch.sin(theta)
        x = self.sky[None, ..., 0] * c[:, None, None] + self.sky[None, ..., 1] * s[:, None, None]
        y = -self.sky[None, ..., 0] * s[:, None, None] + self.sky[None, ..., 1] * c[:, None, None]
        radius = torch.sqrt(x ** 2 + (y / q[:, None, None]) ** 2 + 1e-8)
        perturb = torch.exp(self.perturbation * torch.tanh(params['eps'][:, 0]))
        morph, norms = {}, {}
        for j, band in enumerate(BANDS):
            bv = band_out['vector'][:, j]
            r = radius / torch.exp(self.size_bound * torch.tanh(bv[:, 1]))[:, None, None]
            comps = torch.stack((monotone_profile(knots_c, r), monotone_profile(knots_e, r)), 1)
            comps = comps * (perturb * torch.exp(self.eps_bound * torch.tanh(band_out['eps'][:, j])))[:, None]
            comps = comps / comps.sum((-2, -1), keepdim=True).clamp_min(1e-12)
            mix = torch.sigmoid(v[:, 2 * N_KNOTS + 4 + j] + bv[:, 0])[:, None, None, None]
            morph[band] = mix * comps[:, :1] + (1 - mix) * comps[:, 1:]
            # Observed-space templates on a finite grid miss PSF wings beyond R_MAX: learn that fraction.
            norms[band] = torch.exp(-.3 * torch.sigmoid(bv[:, 2])) if self.implicit_psf else None
        return morph, norms


def render(scene, morph, norms, device, convolve=True):
    """{band: [H*W, N]} templates; no PSF convolution for the implicit-PSF head."""
    templates = {}
    for band, d in scene['bands'].items():
        m = morph[band]; n = m.shape[0]; s2p = d['sky_to_pixel'].to(device); pos = d['positions'].to(device)
        scale = math.sqrt(abs(float(torch.linalg.det(torch.linalg.inv(s2p)))))
        oversample = max(2, int(math.ceil(2 * scale / MORPH_SCALE)))
        kernels = band_kernels(d, n, device) if convolve else None
        pad = int(math.ceil(MORPH_SIZE * MORPH_SCALE / scale / 2 * 1.42)) + (kernels.shape[-1] // 2 if convolve else 0)
        canvas = render_unconvolved(m, pos, s2p, tuple(d['image'].shape), pad, oversample)
        if convolve: canvas = fft_convolve(canvas, kernels)
        t = canvas[:, 0, pad:pad + d['image'].shape[0], pad:pad + d['image'].shape[1]].reshape(n, -1).T
        templates[band] = t * norms[band][None] if norms[band] is not None else t
    return templates


class TruthScarlet:
    """Encoder + head + renderer + linear solve, shared by training and evaluation."""
    def __init__(self, head, encoder, device, foundation=True):
        self.head = head; self.encoder = encoder; self.device = device; self.foundation = foundation; self.refine_scale = .1

    def encode(self, scene):
        raw = raw_pixel_windows(scene, self.device); n = raw.shape[0]
        if not self.foundation:
            return raw.new_zeros(n, 256, BN_WINDOW, BN_WINDOW), raw.new_zeros(n, 64, MORPH_SIZE, MORPH_SIZE), raw
        with torch.no_grad():
            views = self.encoder.feature_views(scene, bottleneck_window=BN_WINDOW, stem_window=MORPH_SIZE)
        return torch.as_tensor(views['bottleneck'], device=self.device), torch.as_tensor(views['vis_stem'], device=self.device), raw

    def run(self, scene, steps=None):
        head = self.head; steps = head.steps if steps is None else steps
        scene = scene_to(scene, self.device)
        bn, stem, raw = self.encode(scene)
        params, feat = head(bn, stem, raw)
        psf = psf_windows(scene, self.device) if head.psf_input else None
        residual = torch.zeros_like(raw)
        for k in range(steps + 1):
            band_out = head.band_branch(feat, raw, residual, psf) if head.per_band else None
            morph, norms = head.band_morphologies(params, band_out)
            templates = render(scene, morph, norms, self.device, convolve=not head.implicit_psf)
            results, total = scene_fit(scene, templates, self.device, fixed_background=True)
            if k == steps: break
            grad, = torch.autograd.grad(total, params['eps'], retain_graph=True)
            g = grad / grad.flatten(1).abs().amax(1).clamp_min(1e-30)[:, None, None, None]
            params = head.refine_step(params, g.detach(), feat, self.refine_scale)
            if head.per_band: residual = residual_windows(scene, results, self.device)
        return dict(results=results, total=total, templates=templates, params=params)
