"""
W&B diagnostic figures for PSFField training.

All renderings stay inside the training angular extent (no SIREN extrapolation
artefacts) and use linear scales with a fixed percent-of-peak ceiling so the
core shape dominates — the same conventions as the static validation notebook.

Functions return `{wandb_key: wandb.Image}` dicts that the training script
merges into its per-epoch `wandb.log(...)`.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .psf_field import (
    PSFField, BAND_ORDER, BAND_PX_SCALE, N_BANDS,
    normalise_psf, analytic_optimal_flux,
)
from .star_selection import StarCatalog


# ---------------------------------------------------------------------------
# Shared rendering helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _render_mean_sed(
    psf: PSFField,
    band_idx: int,
    tile_xy=(0.5, 0.5),
    stamp_size: int = 25,
    sub_grid: int = 6,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Normalised PSF stamp at a tile position with mean-SED conditioning."""
    if device is None:
        device = next(psf.parameters()).device
    tp = torch.tensor([tile_xy], device=device, dtype=torch.float32)
    sed = torch.zeros(1, N_BANDS, device=device)
    stamp = psf.render_stamps(
        centroids_arcsec=torch.zeros(1, 2, device=device),
        tile_pos=tp,
        band_idx=torch.tensor([band_idx], device=device, dtype=torch.long),
        sed_vec=sed,
        stamp_size=stamp_size,
        px_scale=float(BAND_PX_SCALE[band_idx]),
        sub_grid=sub_grid,
        apply_dcr=False,
    )[0]
    return normalise_psf(stamp, float(BAND_PX_SCALE[band_idx])).cpu().numpy()


def _radial_profile(stamp: np.ndarray, px_scale: float, n_bins: Optional[int] = None):
    """Azimuthal mean, skipping empty bins.

    Default `n_bins = (S-1)/2` so each bin contains at least one pixel ring
    (avoids log-scale aliasing when alternating bins are empty).
    """
    S = stamp.shape[-1]
    half = (S - 1) / 2.0
    if n_bins is None:
        n_bins = max(4, int(half))
    y, x = np.mgrid[:S, :S]
    r_arc = np.sqrt((x - half) ** 2 + (y - half) ** 2) * px_scale
    r_max = half * px_scale
    edges = np.linspace(0, r_max, n_bins + 1)
    idx = np.clip(np.digitize(r_arc.ravel(), edges) - 1, 0, n_bins - 1)
    centres, means = [], []
    for k in range(n_bins):
        mask = idx == k
        if mask.any():
            centres.append(0.5 * (edges[k] + edges[k + 1]))
            means.append(stamp.ravel()[mask].mean())
    return np.asarray(centres), np.asarray(means)


# ---------------------------------------------------------------------------
# Panel 1 — per-band median PSF
# ---------------------------------------------------------------------------

def per_band_gallery(psf: PSFField, stamp_size: int = 25) -> Dict:
    import wandb, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    half_arc = ((stamp_size - 1) / 2.0) * BAND_PX_SCALE.numpy()
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for bi, (band, ax) in enumerate(zip(BAND_ORDER, axes.flat)):
        p = _render_mean_sed(psf, bi, stamp_size=stamp_size)
        ext = float(half_arc[bi])
        vmax = 0.02 * p.max()
        ax.imshow(p, origin='lower', cmap='magma',
                  extent=[-ext, ext, -ext, ext],
                  vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_title(f'{band} (±{ext:.2f}")', fontsize=9)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        # Common display window across Rubin (±2.4") and Euclid (±1.2")
        ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.0, 1.0)
    fig.suptitle('Median PSF per band (linear, clipped at 2 % of peak)', fontsize=10)
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return {'psf/gallery': img}


# ---------------------------------------------------------------------------
# Panel 2 — model vs data radial profiles
# ---------------------------------------------------------------------------

@torch.no_grad()
def radial_profiles_model_vs_data(
    psf: PSFField,
    cat: StarCatalog,
    sed_vec: torch.Tensor,
    centroid: torch.nn.Parameter,
    stamp_size: int,
    sub_grid: int,
) -> Dict:
    """
    Average radial profile across all stars in `cat`, model vs data, per band.
    Data profiles come from the real stamps; model profiles come from rendering
    each star with its learned centroid + SED.
    """
    import wandb, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    N = len(cat)
    device = cat.stamps.device
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for bi, (band, ax) in enumerate(zip(BAND_ORDER, axes.flat)):
        px = float(BAND_PX_SCALE[bi])
        # Render model stamps for all stars in this band
        model_b = psf.render_stamps(
            centroids_arcsec=centroid,
            tile_pos=cat.tile_pos,
            band_idx=torch.full((N,), bi, dtype=torch.long, device=device),
            sed_vec=sed_vec,
            stamp_size=stamp_size,
            px_scale=px,
            sub_grid=sub_grid,
            apply_dcr=True,
        )                                                        # [N, S, S]
        # Analytic optimal flux per star → match model amplitude to data
        rms = cat.rms_stamps[:, bi].clamp(min=1e-8)
        var = rms * rms
        flux = analytic_optimal_flux(cat.stamps[:, bi], model_b, var)
        model_scaled = flux.view(-1, 1, 1) * model_b

        # Peak-normalise each stamp before averaging → comparable profiles
        d = cat.stamps[:, bi]
        d_norm = d / d.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        m_norm = model_scaled / model_scaled.amax(
            dim=(-2, -1), keepdim=True).clamp(min=1e-6)

        d_np = d_norm.mean(dim=0).cpu().numpy()
        m_np = m_norm.mean(dim=0).cpu().numpy()
        r_d, prof_d = _radial_profile(d_np, px)
        r_m, prof_m = _radial_profile(m_np, px)
        prof_d /= prof_d.max() + 1e-12
        prof_m /= prof_m.max() + 1e-12

        train_half = ((stamp_size - 1) / 2.0) * px
        keep = r_d <= train_half

        ax.plot(r_d[keep], prof_d[keep], color='black', lw=1.8, label='data')
        ax.plot(r_m[keep], prof_m[keep], color='crimson',
                lw=1.2, ls='--', label='model')
        ax.set_yscale('log')
        ax.set_ylim(5e-3, 2)
        ax.set_xlim(0, train_half)
        ax.set_title(band, fontsize=9)
        ax.grid(alpha=0.3)
        if bi == 0:
            ax.legend(fontsize=7)

    fig.supxlabel('r (arcsec)')
    fig.supylabel('normalised intensity')
    fig.suptitle('Radial profile: model (dashed) vs data (solid)')
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return {'psf/radial_profiles': img}


# ---------------------------------------------------------------------------
# Panel 3 — example star: data / model / residual across all 10 bands
# ---------------------------------------------------------------------------

@torch.no_grad()
def example_star_stamps(
    psf: PSFField,
    cat: StarCatalog,
    sed_vec: torch.Tensor,
    centroid: torch.nn.Parameter,
    stamp_size: int,
    sub_grid: int,
    star_idx: int = 0,
) -> Dict:
    import wandb, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    N = len(cat)
    star_idx = min(star_idx, N - 1)
    device = cat.stamps.device

    fig, axes = plt.subplots(3, N_BANDS, figsize=(18, 4.5))
    for bi in range(N_BANDS):
        m = psf.render_stamps(
            centroids_arcsec=centroid,
            tile_pos=cat.tile_pos,
            band_idx=torch.full((N,), bi, dtype=torch.long, device=device),
            sed_vec=sed_vec,
            stamp_size=stamp_size,
            px_scale=float(BAND_PX_SCALE[bi]),
            sub_grid=sub_grid,
            apply_dcr=True,
        )
        rms = cat.rms_stamps[:, bi].clamp(min=1e-8)
        var = rms * rms
        flux = analytic_optimal_flux(cat.stamps[:, bi], m, var)
        m_scaled = (flux.view(-1, 1, 1) * m)[star_idx].cpu().numpy()
        d = cat.stamps[star_idx, bi].cpu().numpy()
        res = d - m_scaled

        vmax = max(abs(d).max(), 1e-6)
        for row, img, cmap, title in (
            (0, d,        'magma',  f'{BAND_ORDER[bi]}'),
            (1, m_scaled, 'magma',  'model'),
            (2, res,      'RdBu_r', 'resid'),
        ):
            ax = axes[row, bi]
            if cmap == 'RdBu_r':
                ax.imshow(img, origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax)
            else:
                ax.imshow(img, origin='lower', cmap=cmap, vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=8)
            if bi == 0:
                ax.set_ylabel(['data', 'model', 'resid'][row], fontsize=8)
    fig.suptitle(f'Example star {star_idx}: data / model / residual', fontsize=10)
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return {'psf/example_star': img}


# ---------------------------------------------------------------------------
# Panel 4 — centroid drift distribution
# ---------------------------------------------------------------------------

def centroid_drift_histogram(centroid_reg) -> Dict:
    import wandb, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_centroids = []
    for p in centroid_reg.parameters():
        all_centroids.append(p.detach().cpu().numpy())
    if not all_centroids:
        return {}
    cat = np.concatenate(all_centroids, axis=0)                # [N_total, 2]
    mag_mas = np.linalg.norm(cat, axis=-1) * 1000.0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mag_mas, bins=50, color='seagreen', alpha=0.8)
    ax.axvline(np.median(mag_mas), color='crimson', ls='--',
               label=f'median={np.median(mag_mas):.1f} mas')
    ax.axvline(np.percentile(mag_mas, 90), color='steelblue', ls=':',
               label=f'p90={np.percentile(mag_mas, 90):.1f} mas')
    ax.set_xlabel('|Δ centroid| (mas)')
    ax.set_ylabel('N stars')
    ax.set_title('Learned sub-pixel centroid drift')
    ax.legend()
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return {'psf/centroid_hist': img}


# ---------------------------------------------------------------------------
# Top-level helper — compute everything a training epoch should log
# ---------------------------------------------------------------------------

def build_epoch_visuals(
    psf: PSFField,
    catalogs: Dict[str, StarCatalog],
    sed_state: Dict[str, torch.Tensor],
    centroid_reg,
    stamp_size: int,
    sub_grid: int,
    sample_stems: int = 6,
) -> Dict:
    """
    Assemble all diagnostic panels. Uses a small random sample of tiles for
    radial-profile averaging to keep per-epoch cost low.
    """
    stems = list(catalogs.keys())
    sample = np.random.choice(stems, size=min(sample_stems, len(stems)),
                              replace=False)

    out: Dict = {}

    # 1. Gallery of median PSF per band
    out.update(per_band_gallery(psf, stamp_size=stamp_size))

    # 2. Radial profiles — concat stamps from sampled tiles
    data_stamps, rms_stamps, sed_vecs, tile_pos, centroids = [], [], [], [], []
    for stem in sample:
        cat = catalogs[stem]
        data_stamps.append(cat.stamps)
        rms_stamps.append(cat.rms_stamps)
        sed_vecs.append(sed_state[stem])
        tile_pos.append(cat.tile_pos)
        centroids.append(centroid_reg.get(stem).detach())
    if data_stamps:
        class _PseudoCat:
            pass
        merged = _PseudoCat()
        merged.stamps     = torch.cat(data_stamps, dim=0)
        merged.rms_stamps = torch.cat(rms_stamps, dim=0)
        merged.tile_pos   = torch.cat(tile_pos,   dim=0)
        merged_sed = torch.cat(sed_vecs, dim=0)
        merged_centroid = torch.cat(centroids, dim=0)
        # Tell Python len() works
        merged.__class__.__len__ = lambda self: self.stamps.shape[0]

        out.update(radial_profiles_model_vs_data(
            psf, merged, merged_sed, merged_centroid,
            stamp_size=stamp_size, sub_grid=sub_grid,
        ))

        # 3. Example star from the first sampled tile
        first_cat = catalogs[sample[0]]
        first_sed = sed_state[sample[0]]
        first_cen = centroid_reg.get(sample[0]).detach()
        out.update(example_star_stamps(
            psf, first_cat, first_sed, first_cen,
            stamp_size=stamp_size, sub_grid=sub_grid,
            star_idx=0,
        ))

    # 4. Centroid drift histogram
    out.update(centroid_drift_histogram(centroid_reg))

    return out
