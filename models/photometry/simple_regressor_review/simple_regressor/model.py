"""Supervised flux models without foundation features.

Version 1 retains the original CNN for checkpoint compatibility. Version 2 adds
native-pixel aperture measurements, a training-fitted linear baseline, and a
zero-initialized CNN correction in transformed flux space. GroupNorm is confined
to the image-context branch; absolute amplitude bypasses it.

At fixed spatial size sum and average pooling are equivalent up to scale.
"""
import torch
from torch import nn


def _norm(ch):
    g = 8
    while ch % g:
        g -= 1
    return nn.GroupNorm(g, ch)


class ConvBlock(nn.Module):
    """Two 3x3 convs then an average-pool /2 (for context features)."""
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), _norm(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), _norm(cout), nn.GELU(),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class FluxCNN(nn.Module):
    POOLS = ("avg", "max", "avgmax", "sum", "gatedsum", "mix")

    def __init__(self, in_channels=3, pool="avgmax", stem_ch=16, widths=(32, 64, 128),
                 head_hidden=128, dropout=0.1):
        super().__init__()
        if pool not in self.POOLS:
            raise ValueError(f"pool must be one of {self.POOLS}")
        self.pool = pool
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_ch, 5, stride=2, padding=2), _norm(stem_ch), nn.GELU())
        chans = [stem_ch, *widths]
        self.blocks = nn.Sequential(*[ConvBlock(chans[i], chans[i + 1]) for i in range(len(widths))])
        cf = widths[-1]
        # Context projection; this alone does not preserve input amplitude.
        self.reduce = nn.Conv2d(cf, cf, 1)
        if pool in ("gatedsum", "mix"):
            self.gate = nn.Conv2d(cf, 1, 1)
        head_in = {"avgmax": 2 * cf, "mix": 4 * cf}.get(pool, cf)
        H = head_hidden
        self.head = nn.Sequential(
            nn.Linear(head_in, H), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(H, H), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(H, 1))

    def _aggregate(self, f):
        if self.pool == "avg":
            return f.mean(dim=(2, 3))
        if self.pool == "max":
            return f.amax(dim=(2, 3))
        if self.pool == "avgmax":
            return torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        if self.pool == "sum":
            return f.sum(dim=(2, 3))
        if self.pool == "gatedsum":
            g = torch.sigmoid(self.gate(f))
            return (f * g).sum(dim=(2, 3))
        # mix: all four pooling operators co-exist as parallel branches, concatenated
        # (NOT blended into one) -> the MLP head learns which to use.
        g = torch.sigmoid(self.gate(f))
        return torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3)),
                          f.sum(dim=(2, 3)), (f * g).sum(dim=(2, 3))], dim=1)

    def forward(self, feat, img_lin=None):
        f = self.reduce(self.blocks(self.stem(feat)))
        return self.head(self._aggregate(f)).squeeze(-1)   # standardised asinh flux z

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ApertureFluxCNN(FluxCNN):
    """Calibrated aperture flux plus a learned correction, in target z units."""
    def __init__(self, cfg):
        super().__init__(in_channels=4, pool=cfg.pool, head_hidden=cfg.head_hidden,
                         dropout=cfg.dropout)
        size = cfg.stamp
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        r = torch.hypot(x - (size - 1) / 2, y - (size - 1) / 2) * cfg.native_pixscale
        radii = cfg.aperture_radii_arcsec
        if not radii or min(radii) <= 0 or max(radii) >= 0.86 * size / 2 * cfg.native_pixscale:
            raise ValueError("Apertures must fit inside the background annulus")
        self.register_buffer("apertures", torch.stack([r <= a for a in radii]).float())
        self.register_buffer("coefficients", torch.zeros(len(radii)))
        self.register_buffer("aperture_scale", torch.ones(len(radii)))
        self.register_buffer("target_mean", torch.tensor(0.))
        self.register_buffer("target_std", torch.tensor(1.))
        self.register_buffer("target_f0", torch.tensor(cfg.f0_ujy))
        self.register_buffer("flux_floor", torch.tensor(1e-8))
        self.log_target = cfg.loss == "mag"
        # Include aperture amplitudes, coverage fractions and baseline z explicitly.
        dim = self.head[0].in_features + 2 * len(radii) + 1
        self.head[0] = nn.Linear(dim, cfg.head_hidden)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def aperture_features(self, img_lin):
        if img_lin is None or img_lin.shape[1:] != (2, *self.apertures.shape[-2:]):
            raise ValueError("Expected native background-subtracted image and mask [N,2,S,S]")
        sums = torch.einsum("nhw,ahw->na", img_lin[:, 0], self.apertures)
        coverage = torch.einsum("nhw,ahw->na", img_lin[:, 1], self.apertures)
        coverage = coverage / self.apertures.sum((-2, -1)).clamp_min(1)
        return sums, coverage

    def baseline_flux(self, img_lin):
        sums, _ = self.aperture_features(img_lin)
        return sums @ self.coefficients

    def forward(self, feat, img_lin=None):
        sums, coverage = self.aperture_features(img_lin)
        base = sums @ self.coefficients
        transformed = (base.clamp_min(self.flux_floor).log() if self.log_target
                       else torch.asinh(base / self.target_f0))
        base_z = (transformed - self.target_mean) / self.target_std
        f = self.reduce(self.blocks(self.stem(feat)))
        context = torch.cat([self._aggregate(f), torch.asinh(sums / self.aperture_scale),
                             coverage, base_z[:, None]], dim=1)
        return base_z + self.head(context).squeeze(-1)


def build_model(cfg):
    if cfg.model_version == 2:
        return ApertureFluxCNN(cfg)
    if cfg.model_version != 1:
        raise ValueError("model_version must be 1 or 2")
    return FluxCNN(in_channels=3, pool=cfg.pool,
                   head_hidden=cfg.head_hidden, dropout=cfg.dropout)
