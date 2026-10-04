"""Amortised scarlet photometry on top of the frozen foundation encoder.

Model (scarlet): image_b = sum_i f_ib * [ M_ib (*) PSF_b ] + background_b, with a
positive morphology per source on a 0.1-arcsec tangent-plane grid and signed
amplitudes solved linearly per band. A small head reads the frozen fused
bottleneck, the VIS stem and the raw ten-band pixels at each source and outputs
two positive unit-sum components (compact, extended) plus a per-band mixing
weight, so colour gradients are allowed without a second amplitude. Optional
unrolled refinement steps feed the exact residual gradient back into the head
(the gradient is treated as an input, first-order training).
PSFs stay explicit (per-source GRID stamps or calibrated Gaussians) and the
flux is never predicted: it is the weighted least-squares amplitude with a
free constant background, exactly as in the mixture photometer. Training uses
only the scene residual chi-square; no catalog flux labels.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import BANDS, fit_flux
from .scene_features import SceneEncoder, windows
from .pixel_psf import normalize_kernel, gaussian_kernel

MORPH_SIZE = 61          # morphology grid, 0.1 arcsec sampling, 6.1 arcsec across
MORPH_SCALE = .1         # arcsec per morphology pixel
BN_WINDOW = 15           # bottleneck window (0.4 arcsec per pixel): 6 arcsec
ROOT = Path(__file__).resolve().parents[3]


# ----------------------------------------------------------------------------- geometry
def band_to_morph_coords(xy, position, sky_to_pixel):
    """Band pixel coordinates [...,2] -> morphology grid coordinates [...,2] for one source."""
    pixel_to_sky = torch.linalg.inv(sky_to_pixel)
    offset = xy - position
    sky = offset @ pixel_to_sky.T                       # arcsec east/north
    return sky / MORPH_SCALE + (MORPH_SIZE - 1) / 2


def sample_morphology(morph, coords):
    """Bilinear sample of morph [N,C,K,K] at coords [N,H,W,2] (grid units); zero outside."""
    grid = coords / (MORPH_SIZE - 1) * 2 - 1
    return F.grid_sample(morph, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


def render_unconvolved(morph, positions, sky_to_pixel, shape, pad, oversample):
    """Flux-conserving resampling of unit-sum morphologies [N,C,K,K] onto a padded band canvas.

    Returns [N,C,H+2pad,W+2pad]; each pixel integrates the morphology density over its area."""
    n = morph.shape[0]; h, w = shape; device = morph.device
    o = oversample
    ys = (torch.arange(h + 2 * pad, device=device, dtype=torch.float32) - pad)[:, None, None] + (torch.arange(o, device=device) + .5) / o - .5
    xs = (torch.arange(w + 2 * pad, device=device, dtype=torch.float32) - pad)[None, :, None] + (torch.arange(o, device=device) + .5) / o - .5
    # sub-pixel centres: [H',o] and [W',o] -> full grid [H'*o, W'*o]
    yy = ys.expand(-1, w + 2 * pad, -1).reshape(h + 2 * pad, w + 2 * pad, o, 1).expand(-1, -1, -1, o)
    xx = xs.expand(h + 2 * pad, -1, -1).reshape(h + 2 * pad, w + 2 * pad, 1, o).expand(-1, -1, o, -1)
    # [H', W', oy, ox, 2] -> [H', oy, W', ox, 2] so that flattening interleaves sub-pixels with their parent pixel.
    pts = torch.stack((xx, yy), -1).permute(0, 2, 1, 3, 4).reshape(1, (h + 2 * pad) * o, (w + 2 * pad) * o, 2).expand(n, -1, -1, -1)
    coords = band_to_morph_coords(pts, positions[:, None, None, :], sky_to_pixel)
    sampled = sample_morphology(morph, coords)          # density per morphology pixel
    area = abs(float(torch.linalg.det(torch.linalg.inv(sky_to_pixel)))) / (MORPH_SCALE ** 2)  # morph pixels per band pixel
    out = sampled.reshape(n, morph.shape[1], h + 2 * pad, o, w + 2 * pad, o).mean((3, 5)) * area
    return out


def fft_convolve(images, kernels):
    """Convolve images [N,C,H,W] with per-source kernels [N,k,k] (zero-padded, 'same' output)."""
    from scipy.fft import next_fast_len
    n, c, h, w = images.shape; k = kernels.shape[-1]; half = k // 2
    hp, wp = next_fast_len(h + k - 1), next_fast_len(w + k - 1)
    fi = torch.fft.rfft2(images, s=(hp, wp))
    ker = torch.zeros(n, 1, hp, wp, dtype=images.dtype, device=images.device)
    ker[:, 0, :k, :k] = kernels
    ker = torch.roll(ker, shifts=(-half, -half), dims=(-2, -1))
    fk = torch.fft.rfft2(ker)
    out = torch.fft.irfft2(fi * fk, s=(hp, wp))
    return out[..., :h, :w]


def band_kernels(d, n, device):
    """Per-source PSF kernels [N,k,k]: GRID stamps when supplied, else pixel-integrated Gaussian."""
    if 'psf_kernels' in d:
        kernels = np.asarray(d['psf_kernels'], dtype='float32')
        return torch.tensor(np.stack([normalize_kernel(k) for k in kernels]), dtype=torch.float32, device=device)
    sigma = float(d['psf_sigma']); size = int(2 * math.ceil(4 * sigma) + 1)
    kernel = torch.tensor(gaussian_kernel((size, size), sigma), dtype=torch.float32, device=device)
    return kernel[None].expand(n, -1, -1)


def render_templates(scene, morph_bands, device):
    """Rendered unit-flux templates per band: {band: [H*W, N]} for morph_bands {band: [N,1,K,K]}."""
    templates = {}
    for band, d in scene['bands'].items():
        morph = morph_bands[band]; n = morph.shape[0]
        s2p = d['sky_to_pixel'].to(device); pos = d['positions'].to(device)
        scale = math.sqrt(abs(float(torch.linalg.det(torch.linalg.inv(s2p)))))
        oversample = max(2, int(math.ceil(2 * scale / MORPH_SCALE)))
        kernels = band_kernels(d, n, device)
        pad = int(math.ceil(MORPH_SIZE * MORPH_SCALE / scale / 2 * 1.42)) + kernels.shape[-1] // 2
        canvas = render_unconvolved(morph, pos, s2p, tuple(d['image'].shape), pad, oversample)
        convolved = fft_convolve(canvas, kernels)
        cropped = convolved[:, 0, pad:pad + d['image'].shape[0], pad:pad + d['image'].shape[1]]
        templates[band] = cropped.reshape(n, -1).T
    return templates


def raw_pixel_windows(scene, device):
    """Ten asinh-compressed S/N windows per source on the morphology grid: [N,10,K,K]."""
    out = []
    for band in BANDS:
        d = scene['bands'][band]; image = d['image'].to(device); var = d['variance'].to(device); mask = d['mask'].to(device)
        bg = image[mask].median() if mask.any() else image.new_tensor(0.)
        snr = torch.where(mask, (image - bg) / torch.sqrt(var.clamp_min(1e-20)), image.new_tensor(0.))
        snr = torch.asinh(snr / 5)[None, None]
        n = len(d['positions']); s2p = d['sky_to_pixel'].to(device); pos = d['positions'].to(device)
        k = MORPH_SIZE; g = (torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2) * MORPH_SCALE
        vv, uu = torch.meshgrid(g, g, indexing='ij'); sky = torch.stack((uu, vv), -1)   # arcsec east/north
        xy = pos[:, None, None, :] + sky[None] @ s2p.T
        grid = xy / xy.new_tensor([image.shape[1] - 1, image.shape[0] - 1]) * 2 - 1
        out.append(F.grid_sample(snr.expand(n, -1, -1, -1), grid, mode='bilinear', padding_mode='zeros', align_corners=True))
    return torch.cat(out, 1)


# ----------------------------------------------------------------------------- head
N_KNOTS = 24
R_MAX = 3.0  # arcsec: profiles vanish beyond this elliptical radius


def knot_radii(device):
    k = torch.arange(N_KNOTS, device=device, dtype=torch.float32) / (N_KNOTS - 1)
    return R_MAX * k ** 2


def monotone_profile(knot_logits, radius):
    """Decreasing radial profile from knot increments; radius [N,K,K] arcsec -> values [N,K,K].

    P_k = sum_{j>=k} softplus(a_j) is non-increasing in k by construction and zero beyond R_MAX."""
    increments = F.softplus(knot_logits)                              # [N,n]
    profile = torch.flip(torch.cumsum(torch.flip(increments, (1,)), 1), (1,))
    profile = torch.cat((profile, profile.new_zeros(profile.shape[0], 1)), 1)   # [N,n+1], last = 0
    index = (N_KNOTS - 1) * torch.sqrt((radius / R_MAX).clamp_min(0))          # fractional knot index
    index = index.clamp(max=N_KNOTS - 1e-4)
    x = index / (N_KNOTS) * 2 - 1                                                # grid coordinate over n+1 samples
    grid = torch.stack((x, torch.zeros_like(x)), -1)                             # [N,K,K,2]
    return F.grid_sample(profile[:, None, None, :], grid, mode='bilinear', padding_mode='border', align_corners=True)[:, 0]


class ScarletHead(nn.Module):
    """Monotone elliptical radial profiles (compact + extended) times a bounded perturbation map.

    Centred, monotonically decreasing and finite by construction, so two overlapping
    sources cannot trade light arbitrarily; the perturbation (|log| <= 0.5) keeps room
    for arms, bars and clumps. A per-band weight mixes the two profiles for colour gradients."""
    version = 'v2_monotone'

    def __init__(self, bottleneck_ch=256, stem_ch=64, width=64, steps=2, perturbation=.5):
        super().__init__()
        self.steps = steps; self.perturbation = perturbation
        self.bn_proj = nn.Sequential(nn.Conv2d(bottleneck_ch, width, 1), nn.GELU())
        self.stem_proj = nn.Sequential(nn.Conv2d(stem_ch, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, width, 3, padding=1), nn.GELU())
        self.raw_proj = nn.Sequential(nn.Conv2d(len(BANDS), width // 2, 3, padding=1), nn.GELU())
        fused = width * 2 + width // 2 + 2
        self.body = nn.Sequential(nn.Conv2d(fused, width, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU(),
                                  nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.GELU(),
                                  nn.Conv2d(width, width, 3, padding=8, dilation=8), nn.GELU())
        self.n_vector = 2 * N_KNOTS + 4 + len(BANDS)
        self.vector_out = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, self.n_vector))
        self.eps_out = nn.Conv2d(width, 1, 1)
        self.refine = nn.Sequential(nn.Conv2d(3 + width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU())
        self.refine_vector = nn.Linear(2 * width, self.n_vector); self.refine_eps = nn.Conv2d(width, 1, 1)
        for m in (self.refine_vector, self.refine_eps, self.eps_out): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.vector_out[-1].weight); nn.init.zeros_(self.vector_out[-1].bias)
        g = (torch.arange(MORPH_SIZE, dtype=torch.float32) - (MORPH_SIZE - 1) / 2) / ((MORPH_SIZE - 1) / 2)
        vv, uu = torch.meshgrid(g, g, indexing='ij'); self.register_buffer('coords', torch.stack((uu, vv))[None])
        self.register_buffer('sky', torch.stack((uu, vv), -1) * (MORPH_SIZE - 1) / 2 * MORPH_SCALE)   # [K,K,2] arcsec
        # Initial knots: compact ~ 0.15" exponential-like, extended ~ 0.6"; so untrained profiles are sane.
        r = knot_radii(torch.device('cpu'))
        init = torch.cat((torch.log(torch.expm1(torch.exp(-r / .15) * .3 + 1e-3)), torch.log(torch.expm1(torch.exp(-r / .6) * .05 + 1e-3)),
                          torch.zeros(4), torch.zeros(len(BANDS))))
        self.register_buffer('init_vector', init[None])

    def features(self, bottleneck, stem, raw):
        bn = self.bn_proj(bottleneck)
        bn = F.interpolate(bn, size=(57, 57), mode='bilinear', align_corners=True)
        bn = F.pad(bn, (2, 2, 2, 2))
        n = bn.shape[0]
        return self.body(torch.cat((bn, self.stem_proj(stem), self.raw_proj(raw), self.coords.expand(n, -1, -1, -1)), 1))

    @staticmethod
    def pooled(feat): return torch.cat((feat.mean((-2, -1)), feat.amax((-2, -1))), 1)

    def forward(self, bottleneck, stem, raw):
        feat = self.features(bottleneck, stem, raw)
        vector = self.vector_out(self.pooled(feat)) + self.init_vector
        eps = self.eps_out(feat)
        return dict(vector=vector, eps=eps), feat

    def morphologies(self, params):
        """{band: [N,1,K,K]} unit-sum band morphologies, plus the two components [N,2,K,K]."""
        v = params['vector']; n = v.shape[0]
        knots_c, knots_e = v[:, :N_KNOTS], v[:, N_KNOTS:2 * N_KNOTS]
        log_q = -F.softplus(v[:, 2 * N_KNOTS]) * .5          # axis ratio q in (0,1]
        theta = torch.atan2(v[:, 2 * N_KNOTS + 2], v[:, 2 * N_KNOTS + 1] + 1.)
        mix = torch.sigmoid(v[:, 2 * N_KNOTS + 4:])          # [N,10] compact fraction per band
        c, s_ = torch.cos(theta), torch.sin(theta); q = torch.exp(log_q)
        # elliptical radius: rotate sky offsets into the ellipse frame, scale minor axis by 1/q
        x = self.sky[None, ..., 0] * c[:, None, None] + self.sky[None, ..., 1] * s_[:, None, None]
        y = -self.sky[None, ..., 0] * s_[:, None, None] + self.sky[None, ..., 1] * c[:, None, None]
        radius = torch.sqrt(x ** 2 + (y / q[:, None, None]) ** 2 + 1e-8)
        perturb = torch.exp(self.perturbation * torch.tanh(params['eps'][:, 0]))
        comps = torch.stack((monotone_profile(knots_c, radius), monotone_profile(knots_e, radius)), 1) * perturb[:, None]
        comps = comps / comps.sum((-2, -1), keepdim=True).clamp_min(1e-12)
        bands = {band: (mix[:, j, None, None, None] * comps[:, :1] + (1 - mix[:, j, None, None, None]) * comps[:, 1:]) for j, band in enumerate(BANDS)}
        return bands, comps, mix

    def refine_step(self, params, g, feat, scale):
        bands, comps, mix = self.morphologies(params)
        h = self.refine(torch.cat((comps * MORPH_SIZE, g, feat), 1))
        return dict(vector=params['vector'] + scale * self.refine_vector(self.pooled(h)),
                    eps=params['eps'] + scale * self.refine_eps(h))


def band_morphologies(head, params):
    return head.morphologies(params)[0]


def robust_background(d, exclusion_arcsec=1.5, minimum_pixels=100):
    """Median of valid pixels farther than exclusion_arcsec from every source; None if too few."""
    image = d['image']; mask = d['mask']; pos = d['positions']
    scale = math.sqrt(abs(float(torch.linalg.det(torch.linalg.inv(d['sky_to_pixel'])))))
    yy, xx = torch.meshgrid(torch.arange(image.shape[0], device=image.device, dtype=torch.float32),
                            torch.arange(image.shape[1], device=image.device, dtype=torch.float32), indexing='ij')
    distance = torch.sqrt((xx[..., None] - pos[:, 0]) ** 2 + (yy[..., None] - pos[:, 1]) ** 2).amin(-1)
    far = mask & torch.isfinite(image) & (distance > exclusion_arcsec / scale)
    if int(far.sum()) < minimum_pixels: return None
    return image[far].median()


def scene_fit(scene, templates, device, fixed_background=False):
    """Signed flux + constant background per band through core.fit_flux.

    Returns per-band results and the objective: the mean over bands of log(chi2/dof).
    The log equalises bands whose variance maps are mis-scaled (Rubin reduced chi2
    is 10-25 on real tiles against 0.07 in NISP), so no single band dominates;
    it is the Gaussian likelihood profiled over an unknown per-band noise scale."""
    results = {}; terms = []
    for band, d in scene['bands'].items():
        t = templates[band]
        active = t.detach().sum(0) > 1e-5
        if not bool(active.any()): raise ValueError(f'{band}: no template inside the image')
        background = robust_background(d) if fixed_background else None
        if background is not None:
            r = fit_flux(d['image'].to(device) - background, d['variance'].to(device), t[:, active], d['mask'].to(device), fit_background=False)
            r['background'] = background; r['model'] = r['model'] + background
        else:
            r = fit_flux(d['image'].to(device), d['variance'].to(device), t[:, active], d['mask'].to(device))
        r['source_indices'] = torch.where(active)[0]; r['footprint_fraction'] = t[:, active].detach().sum(0)
        results[band] = r; terms.append(torch.log(r['chi2'] / max(r['dof'], 1)))
    return results, torch.stack(terms).mean()


def scene_to(scene, device):
    """Copy of the scene with every band tensor on the device (PSF kernels stay numpy)."""
    bands = {b: {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()} for b, d in scene['bands'].items()}
    return dict(scene, bands=bands)


class AmortisedScarlet:
    """Encoder + head + renderer + linear solve; shared by training and inference."""
    def __init__(self, head, encoder, device, fixed_background=False):
        self.head = head; self.encoder = encoder; self.device = device
        self.refine_scale = .1; self.fixed_background = fixed_background

    def encode(self, scene):
        with torch.no_grad():
            views = self.encoder.feature_views(scene, bottleneck_window=BN_WINDOW, stem_window=MORPH_SIZE)
        bn = torch.as_tensor(views['bottleneck'], device=self.device); stem = torch.as_tensor(views['vis_stem'], device=self.device)
        return bn, stem, raw_pixel_windows(scene, self.device)

    def run(self, scene, steps=None, create_graph=False):
        steps = self.head.steps if steps is None else steps
        scene = scene_to(scene, self.device)
        bn, stem, raw = self.encode(scene)
        params, feat = self.head(bn, stem, raw)
        results, total = None, None
        for k in range(steps + 1):
            morph = band_morphologies(self.head, params)
            templates = render_templates(scene, morph, self.device)
            results, total = scene_fit(scene, templates, self.device, self.fixed_background)
            if k == steps: break
            # exact residual gradient w.r.t. the perturbation map, treated as an input to the refinement net
            grad, = torch.autograd.grad(total, params['eps'], create_graph=create_graph, retain_graph=True)
            g = grad / grad.flatten(1).abs().amax(1).clamp_min(1e-30)[:, None, None, None]
            params = self.head.refine_step(params, g.detach() if not create_graph else g, feat, self.refine_scale)
        return dict(results=results, total=total, params=params)


# ----------------------------------------------------------------------------- data
def scene_list(roots, split_fraction=.1, seed=0):
    """(region_dir, source index) pairs from prepared detcat tile roots; validation = whole tiles."""
    folders = sorted(f.parent for root in roots for f in Path(root).glob('region_*/tile_inputs.npz'))
    rng = np.random.default_rng(seed); order = rng.permutation(len(folders))
    n_val = max(1, int(round(split_fraction * len(folders))))
    val = {folders[i] for i in order[:n_val]}
    items = {'train': [], 'val': []}
    for f in folders:
        with np.load(f / 'tile_inputs.npz', allow_pickle=True) as z: usable = np.flatnonzero(z['scene_inside'] & z['central_valid'])
        items['val' if f in val else 'train'].extend((f, int(i)) for i in usable)
    return items, sorted(str(f) for f in val)


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self, items, psf_sigmas):
        self.items = items; self.psf_sigmas = psf_sigmas; self.cache = {}
    def __len__(self): return len(self.items)
    def __getitem__(self, k):
        from .detcat.prepare import load_inputs, scene_for_source
        folder, i = self.items[k]
        if folder not in self.cache: self.cache = {folder: load_inputs(folder)}
        scene, info = scene_for_source(self.cache[folder], i, self.psf_sigmas)
        return scene, info, (str(folder), i)


def collate(batch): return batch


# ----------------------------------------------------------------------------- training
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--roots', nargs='+', default=['models/photometry/self_supervised/runs/amortised_scarlet/training_tiles'])
    p.add_argument('--output', default='models/photometry/self_supervised/runs/amortised_scarlet')
    p.add_argument('--foundation', default='models/checkpoints/jaisp_v11_q1_soft/checkpoint_best.pt')
    p.add_argument('--psf-calibration', default='models/photometry/self_supervised/runs/q1_all_bands/psf_calibration.json')
    p.add_argument('--epochs', type=int, default=8); p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--steps', type=int, default=2); p.add_argument('--width', type=int, default=64)
    p.add_argument('--accumulate', type=int, default=8); p.add_argument('--workers', type=int, default=6)
    p.add_argument('--limit', type=int, help='scenes per epoch, for smoke tests'); p.add_argument('--device', default='cuda')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--fixed-background', action='store_true', help='Subtract a robust per-band background (median beyond 1.5 arcsec of every source) instead of fitting a free constant')
    a = p.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device if torch.cuda.is_available() or a.device == 'cpu' else 'cpu'); torch.manual_seed(0)
    psf_sigmas = {b: v['sigma_px'] for b, v in json.loads(Path(a.psf_calibration).read_text()).items()}
    items, val_tiles = scene_list(a.roots)
    if a.limit: items = {k: v[:a.limit] for k, v in items.items()}
    print(f"train scenes {len(items['train'])}, validation scenes {len(items['val'])} from tiles {val_tiles}", flush=True)
    encoder = SceneEncoder(ROOT / a.foundation); encoder.encoder.to(device)
    head = ScarletHead(width=a.width, steps=a.steps).to(device)
    model = AmortisedScarlet(head, encoder, device, fixed_background=a.fixed_background)
    optimizer = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    start_epoch = 0; history = []; best = float('inf')
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False); head.load_state_dict(ck['head']); optimizer.load_state_dict(ck['optimizer'])
        start_epoch = ck['epoch'] + 1; history = ck['history']; best = min([h['val'] for h in history] or [float('inf')])
    loaders = {k: torch.utils.data.DataLoader(SceneDataset(v, psf_sigmas), batch_size=1, shuffle=(k == 'train'), num_workers=a.workers,
                                              collate_fn=collate, persistent_workers=a.workers > 0) for k, v in items.items()}
    metadata = dict(foundation=a.foundation, psf_calibration=a.psf_calibration, roots=a.roots, validation_tiles=val_tiles, steps=a.steps, width=a.width, head=ScarletHead.version,
                    fixed_background=a.fixed_background, lr=a.lr,
                    morph_size=MORPH_SIZE, morph_scale_arcsec=MORPH_SCALE, bottleneck_window=BN_WINDOW, bands=list(BANDS),
                    training='mean over bands of log(chi2/dof) of the scene residual, signed linear fluxes + constant background; no flux labels')
    for epoch in range(start_epoch, a.epochs):
        head.train(); losses = []; skipped = 0; t0 = time.time(); optimizer.zero_grad()
        for k, batch in enumerate(loaders['train']):
            scene, info, key = batch[0]
            if scene is None: skipped += 1; continue
            try:
                out_ = model.run(scene, create_graph=False); loss = out_['total'].float()
            except (ValueError, RuntimeError) as exc:
                skipped += 1; continue
            if not torch.isfinite(loss): skipped += 1; continue
            (loss / a.accumulate).backward(); losses.append(float(loss))
            if (k + 1) % a.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); optimizer.step(); optimizer.zero_grad()
            if k % 200 == 0: print(f'epoch {epoch + 1} step {k} loss {np.mean(losses[-200:]) if losses else float("nan"):.4f} ({time.time() - t0:.0f}s)', flush=True)
        head.eval(); val = []; val_steps0 = []
        for batch in loaders['val']:
            scene, info, key = batch[0]
            if scene is None: continue
            try:
                with torch.enable_grad():
                    r = model.run(scene, create_graph=False); val.append(float(r['total']))
                    r0 = model.run(scene, steps=0, create_graph=False); val_steps0.append(float(r0['total']))
            except (ValueError, RuntimeError): continue
        record = dict(epoch=epoch, train=float(np.mean(losses)), val=float(np.mean(val)), val_no_refinement=float(np.mean(val_steps0)), skipped=skipped, seconds=time.time() - t0)
        history.append(record); print(json.dumps(record), flush=True)
        state = dict(head=head.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, history=history, metadata=metadata)
        torch.save(state, out / 'last.pt')
        if record['val'] < best: best = record['val']; torch.save(state, out / 'best.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2))


# ----------------------------------------------------------------------------- inference
class AmortisedScarletPhotometry:
    """Drop-in for MixturePhotometry: returns the same per-band result dictionaries."""
    def __init__(self, checkpoint, device=None, steps=None):
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False); meta = ck['metadata']
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        head = ScarletHead(width=meta['width'], steps=meta['steps'] if steps is None else steps); head.load_state_dict(ck['head']); head.eval().to(self.device)
        encoder = SceneEncoder(ROOT / meta['foundation']); encoder.encoder.to(self.device)
        self.model = AmortisedScarlet(head, encoder, self.device, fixed_background=meta.get('fixed_background', False)); self.metadata = meta

    def __call__(self, scene):
        if set(scene['bands']) != set(BANDS): raise ValueError('All ten native bands are required')
        with torch.enable_grad():
            out = self.model.run(scene, create_graph=False)
        results = {}
        n = len(scene['sky'])
        for band, r in out['results'].items():
            flux = np.full(n, np.nan); error = flux.copy(); footprint = np.zeros(n)
            idx = r['source_indices'].cpu().numpy()
            flux[idx] = r['flux'].detach().cpu().numpy(); error[idx] = r['error'].detach().cpu().numpy(); footprint[idx] = r['footprint_fraction'].cpu().numpy()
            results[band] = dict(flux=flux, error=error, footprint=footprint, reduced_chi2=float(r['chi2'] / max(r['dof'], 1)), chi2=float(r['chi2']), dof=r['dof'],
                                 model=r['model'].detach().cpu().numpy(), condition=float(r['condition']), covariance=r['covariance'].detach().cpu().numpy(),
                                 source_indices=idx, background=float(r['background']))
        bands_, comps, mix = self.model.head.morphologies(out['params'])
        results['_morphology'] = comps.detach().cpu().numpy(); results['_mix'] = mix.detach().cpu().numpy()
        return results


if __name__ == '__main__': main()
