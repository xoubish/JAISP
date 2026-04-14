"""
Post-training diagnostics for a trained PSFField checkpoint.

Loads the checkpoint, re-selects stars on the same tiles, renders model PSF
stamps at the learned centroids, and produces the full diagnostic suite:

  • chi²/ndof per band — histogram + per-band median
  • centroid drift distribution — how far centroids moved from init
  • radial profile model-vs-data per band — reveals PSF-shape errors
  • stamp gallery for a handful of stars — visual check per band
  • DCR coefficients table (mas per mag of colour, Rubin bands only)
  • effective FWHM of the trained PSF vs moment FWHM of the data

Outputs both a text summary and PNG figures in `--out_dir`.

Usage
-----
    python models/psf/validate_psf_field.py \
        --checkpoint       models/checkpoints/psf_field_v1.pt \
        --rubin_dir        data/rubin_tiles_200 \
        --euclid_dir       data/euclid_tiles_200 \
        --centernet_labels data/cached_features_v8_fine/pseudo_labels.pt \
        --out_dir          models/checkpoints/psf_field_v1_diag/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_MODELS,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from psf.psf_field import (                                                # noqa: E402
    PSFField, BAND_ORDER, BAND_PX_SCALE, N_BANDS, N_RUBIN,
    analytic_optimal_flux,
)
from psf.star_selection import select_stars, StarCatalog                   # noqa: E402
from psf.train_psf_field import _build_tile_index, CentroidRegistry        # noqa: E402


# ---------------------------------------------------------------------------
# Load checkpoint + rebuild model
# ---------------------------------------------------------------------------

def _load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt['config']

    psf = PSFField(
        sed_embed_dim=cfg['sed_embed_dim'],
        band_embed_dim=cfg['band_embed_dim'],
        tile_freqs=cfg['tile_freqs'],
        siren_hidden=cfg['siren_hidden'],
        siren_depth=cfg['siren_depth'],
        w0_first=cfg['w0_first'],
        envelope_r_rubin=cfg.get('envelope_r_rubin', 0.0),
        envelope_r_euclid=cfg.get('envelope_r_euclid', 0.0),
        envelope_power=cfg.get('envelope_power', 4.0),
    ).to(device)
    psf.load_state_dict(ckpt['psf_field_state'])
    psf.eval()

    cen = CentroidRegistry().to(device)
    # Empty ParameterDict cannot `load_state_dict` into existing empty — we
    # need to first register parameters with correct shapes. Use state-dict
    # keys (shape comes via the tensors) to rebuild.
    for key, val in ckpt['centroid_state'].items():
        if key.startswith('params.'):
            stem_key = key.split('.', 1)[1]
            cen.params[stem_key] = torch.nn.Parameter(val.to(device).clone())
    cen.eval()
    return psf, cen, cfg, ckpt.get('tiles', []), ckpt.get('epoch', -1)


# ---------------------------------------------------------------------------
# Radial profile helper (both model & data)
# ---------------------------------------------------------------------------

def _radial_profile(stamps: torch.Tensor, n_bins: int = 20) -> torch.Tensor:
    """
    Azimuthally-averaged profile, pixel units.

    Parameters
    ----------
    stamps : [..., S, S] stamp or batch of stamps
    Returns
    -------
    profile : [..., n_bins] averaged over all (..., S, S) inputs
    """
    S = stamps.shape[-1]
    half = (S - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(S, device=stamps.device, dtype=stamps.dtype) - half,
        torch.arange(S, device=stamps.device, dtype=stamps.dtype) - half,
        indexing='ij',
    )
    r = (xx * xx + yy * yy).sqrt()                                 # [S, S]
    r_max = half * np.sqrt(2)
    edges = torch.linspace(0, r_max, n_bins + 1, device=stamps.device)
    bin_idx = torch.bucketize(r.flatten(), edges) - 1
    bin_idx = bin_idx.clamp(0, n_bins - 1)

    flat = stamps.reshape(*stamps.shape[:-2], -1)                  # [..., S²]
    out = torch.zeros(*stamps.shape[:-2], n_bins, device=stamps.device)
    counts = torch.zeros(n_bins, device=stamps.device)
    for b in range(n_bins):
        mask = (bin_idx == b)
        if mask.any():
            out[..., b] = flat[..., mask].mean(dim=-1)
            counts[b] = mask.sum()
    return out, 0.5 * (edges[:-1] + edges[1:])


# ---------------------------------------------------------------------------
# Per-tile evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_tile(
    psf: PSFField,
    centroid: torch.nn.Parameter,
    cat: StarCatalog,
    stamp_size: int,
    sub_grid: int,
) -> Dict:
    """
    For each (star, band): compute model stamp, fit flux, report chi² and
    per-star residual statistics. Returns dict of aggregated tensors.
    """
    N = len(cat)
    device = cat.stamps.device
    rms = cat.rms_stamps.clamp(min=1e-8)
    var = rms * rms                                                # [N, 10, S, S]

    chi2 = torch.zeros(N, N_BANDS, device=device)
    ndof = stamp_size * stamp_size - 1   # sub 1 for fitted flux
    model_stamps = torch.zeros_like(cat.stamps)                    # [N, 10, S, S]

    for bi in range(N_BANDS):
        data_b = cat.stamps[:, bi]                                 # [N, S, S]
        var_b  = var[:, bi]
        m = psf.render_stamps(
            centroids_arcsec=centroid,
            tile_pos=cat.tile_pos,
            band_idx=torch.full((N,), bi, dtype=torch.long, device=device),
            sed_vec=cat.sed_init,
            stamp_size=stamp_size,
            px_scale=float(BAND_PX_SCALE[bi]),
            sub_grid=sub_grid,
            apply_dcr=True,
        )                                                          # [N, S, S]
        flux = analytic_optimal_flux(data_b, m, var_b)             # [N]
        resid = data_b - flux.view(-1, 1, 1) * m
        chi2[:, bi] = (resid ** 2 / var_b.clamp(min=1e-12)).sum(dim=(-2, -1))
        model_stamps[:, bi] = flux.view(-1, 1, 1) * m

    return {
        'chi2':          chi2.cpu(),                   # [N, 10]
        'ndof':          ndof,
        'model_stamps':  model_stamps.cpu(),           # [N, 10, S, S]
        'data_stamps':   cat.stamps.cpu(),
        'var_stamps':    var.cpu(),
        'centroid':      centroid.detach().cpu(),      # [N, 2] arcsec
        'vis_fwhm':      cat.vis_fwhm_arcsec.cpu(),
    }


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _plot_chi2_histogram(chi2_per_band: np.ndarray, ndof: int, out: Path):
    """Per-band histograms of chi²/ndof across all stars."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True)
    for bi, (band, ax) in enumerate(zip(BAND_ORDER, axes.flat)):
        vals = chi2_per_band[:, bi] / ndof
        med = float(np.nanmedian(vals))
        ax.hist(np.clip(vals, 0, 10), bins=40, color='steelblue', alpha=0.8)
        ax.axvline(med, color='crimson', linestyle='--',
                   label=f'median={med:.2f}')
        ax.axvline(1.0, color='black', linestyle=':', alpha=0.5,
                   label='χ²/ndof=1')
        ax.set_title(band, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle('χ²/ndof per band (clipped at 10)', fontsize=11)
    fig.supxlabel('χ²/ndof')
    fig.supylabel('N stars')
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _plot_centroid_drift(centroids_arcsec: np.ndarray, out: Path):
    """
    Histograms of learned centroid magnitude and (dx, dy) distributions.
    Centroids were initialised at 0, so magnitude = drift.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mag_mas = np.linalg.norm(centroids_arcsec, axis=-1) * 1000.0   # [N] mas
    dx_mas = centroids_arcsec[:, 0] * 1000.0
    dy_mas = centroids_arcsec[:, 1] * 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].hist(mag_mas, bins=40, color='seagreen', alpha=0.8)
    axes[0].axvline(np.median(mag_mas), color='crimson', linestyle='--',
                    label=f'median={np.median(mag_mas):.1f} mas')
    axes[0].set_xlabel('|centroid| (mas)')
    axes[0].set_ylabel('N stars')
    axes[0].set_title('centroid drift magnitude')
    axes[0].legend()

    axes[1].hist(dx_mas, bins=40, color='steelblue', alpha=0.8)
    axes[1].axvline(0, color='black', linestyle=':')
    axes[1].set_xlabel('dx (mas)')
    axes[1].set_title(f'dx: mean={dx_mas.mean():.2f}, rms={dx_mas.std():.2f}')

    axes[2].hist(dy_mas, bins=40, color='sandybrown', alpha=0.8)
    axes[2].axvline(0, color='black', linestyle=':')
    axes[2].set_xlabel('dy (mas)')
    axes[2].set_title(f'dy: mean={dy_mas.mean():.2f}, rms={dy_mas.std():.2f}')

    fig.suptitle('Learned sub-pixel centroid drift (shared across 10 bands)')
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _plot_radial_profiles(
    model_stamps: torch.Tensor,  # [N, 10, S, S]
    data_stamps:  torch.Tensor,  # [N, 10, S, S]
    out: Path,
):
    """Mean radial profile per band for model and data (normalised to peak)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for bi, (band, ax) in enumerate(zip(BAND_ORDER, axes.flat)):
        d = data_stamps[:, bi]    # [N, S, S]
        m = model_stamps[:, bi]
        # Normalise each stamp to peak so profiles are comparable
        d_norm = d / d.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        m_norm = m / m.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)

        data_p, r_centres = _radial_profile(d_norm)
        model_p, _        = _radial_profile(m_norm)
        # Mean across stars
        data_mean  = data_p.mean(dim=0).numpy()
        model_mean = model_p.mean(dim=0).numpy()

        px_arcsec = float(BAND_PX_SCALE[bi])
        r_arcsec = r_centres.numpy() * px_arcsec

        ax.plot(r_arcsec, data_mean,  color='black',   lw=1.8, label='data')
        ax.plot(r_arcsec, model_mean, color='crimson', lw=1.5, linestyle='--',
                label='model')
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 2)
        ax.set_xlim(0, min(1.5, r_arcsec.max()))
        ax.set_title(band, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.supxlabel('r (arcsec)')
    fig.supylabel('normalised intensity')
    fig.suptitle('Radial profile  (model dashed, data solid)')
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _plot_stamp_gallery(
    model_stamps: torch.Tensor,  # [N, 10, S, S]
    data_stamps:  torch.Tensor,
    n_show: int,
    out: Path,
):
    """
    For each of `n_show` randomly-chosen stars, plot a 3×10 panel
    (data | model | residual) across all bands.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    N = data_stamps.shape[0]
    pick = np.random.choice(N, size=min(n_show, N), replace=False)

    fig, axes = plt.subplots(3 * len(pick), N_BANDS,
                             figsize=(18, 3.5 * len(pick)),
                             squeeze=False)
    for ri, star in enumerate(pick):
        d = data_stamps[star]
        m = model_stamps[star]
        res = d - m
        for bi in range(N_BANDS):
            vmax = max(float(d[bi].abs().max()), 1e-6)
            for row_off, img, title in (
                (0, d[bi].numpy(),  f'{BAND_ORDER[bi]}\ndata'),
                (1, m[bi].numpy(),  'model'),
                (2, res[bi].numpy(),'resid'),
            ):
                ax = axes[3 * ri + row_off, bi]
                cmap = 'RdBu_r' if row_off == 2 else 'magma'
                ax.imshow(img, origin='lower', cmap=cmap,
                          vmin=-vmax if row_off == 2 else 0,
                          vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if bi == 0:
                    ax.set_ylabel(f'star {star}\n{title.split(chr(10))[-1]}',
                                  fontsize=8)
                if row_off == 0:
                    ax.set_title(title, fontsize=8)
    fig.suptitle('Stamp gallery — data / model / residual', fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Effective FWHM via half-max area on a densely-sampled model stamp
# ---------------------------------------------------------------------------

@torch.no_grad()
def _psf_fwhm_per_band(psf: PSFField, device: torch.device,
                       tile_pos: tuple = (0.5, 0.5),
                       sub_grid: int = 6,
                       stamp_size: int = 61) -> Dict[str, float]:
    out = {}
    sed = torch.zeros(1, N_BANDS, device=device)
    tp = torch.tensor([tile_pos], device=device, dtype=torch.float32)
    for bi, band in enumerate(BAND_ORDER):
        px = float(BAND_PX_SCALE[bi])
        stamp = psf.render_stamps(
            centroids_arcsec=torch.zeros(1, 2, device=device),
            tile_pos=tp,
            band_idx=torch.tensor([bi], device=device, dtype=torch.long),
            sed_vec=sed,
            stamp_size=stamp_size,
            px_scale=px,
            sub_grid=sub_grid,
            apply_dcr=False,
        )[0]
        peak = stamp.max()
        above = (stamp >= 0.5 * peak).float().sum().item()
        fwhm_px = 2.0 * (above / np.pi) ** 0.5
        out[band] = fwhm_px * px
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--rubin_dir',  required=True)
    p.add_argument('--euclid_dir', required=True)
    p.add_argument('--centernet_labels', default=None)
    p.add_argument('--out_dir',    required=True)
    p.add_argument('--max_tiles',  type=int, default=0,
                   help='0 = all tiles present in the checkpoint')
    p.add_argument('--n_gallery',  type=int, default=4,
                   help='Number of stars to show in the stamp gallery')
    p.add_argument('--stamp_size', type=int, default=None)
    p.add_argument('--sub_grid',   type=int, default=4)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading checkpoint on {device} ...')
    psf, cen, cfg, train_tiles, epoch = _load_checkpoint(
        Path(args.checkpoint), device
    )
    print(f'  epoch={epoch}  tiles in checkpoint={len(train_tiles)}')

    stamp_size = args.stamp_size or cfg['stamp_size']
    sub_grid   = args.sub_grid    or cfg['sub_grid']

    # --- Pseudo-labels ------------------------------------------------------
    pseudo_labels = None
    if args.centernet_labels:
        pl = torch.load(args.centernet_labels, map_location='cpu', weights_only=False)
        pseudo_labels = pl['labels'] if 'labels' in pl else pl

    # --- Tile index ---------------------------------------------------------
    all_tiles = _build_tile_index(Path(args.rubin_dir), Path(args.euclid_dir))
    tile_by_stem = {t['stem']: t for t in all_tiles}
    selected_stems = [s for s in train_tiles if s in tile_by_stem]
    if args.max_tiles > 0:
        selected_stems = selected_stems[:args.max_tiles]
    print(f'  evaluating on {len(selected_stems)} tiles')

    # --- Loop tiles ---------------------------------------------------------
    all_chi2 = []
    all_centroid = []
    all_vis_fwhm = []
    model_bag: List[torch.Tensor] = []
    data_bag:  List[torch.Tensor] = []
    n_stars_kept = 0

    for stem in selected_stems:
        t = tile_by_stem[stem]
        # Same detections as training
        det_px = None
        if pseudo_labels is not None and stem in pseudo_labels:
            entry = pseudo_labels[stem]
            xy_norm = entry[0] if isinstance(entry, tuple) else entry
            eu = np.load(t['euclid_path'], allow_pickle=True, mmap_mode='r')
            H_vis, W_vis = eu['img_VIS'].shape
            det_px = np.stack([
                xy_norm[:, 0] * W_vis, xy_norm[:, 1] * H_vis,
            ], axis=1).astype(np.float32)

        cat = select_stars(
            t['rubin_path'], t['euclid_path'],
            stamp_size=stamp_size,
            max_stars=cfg.get('max_stars_per_tile', 80),
            stellar_locus_width=cfg.get('stellar_locus_width', 0.08),
            isolation_arcsec=cfg.get('isolation_arcsec', 3.0),
            detections_vis_px=det_px,
            device=device,
        )
        if cat is None:
            continue
        centroid = cen.get(stem)
        if centroid is None:
            continue
        if centroid.shape[0] != len(cat):
            # Selection returned a different N than at training time — skip
            continue

        result = _evaluate_tile(psf, centroid, cat, stamp_size, sub_grid)
        all_chi2.append(result['chi2'])
        all_centroid.append(result['centroid'])
        all_vis_fwhm.append(result['vis_fwhm'])
        model_bag.append(result['model_stamps'])
        data_bag.append(result['data_stamps'])
        n_stars_kept += len(cat)

    if n_stars_kept == 0:
        raise SystemExit('No stars evaluated — nothing to report')

    chi2_all    = torch.cat(all_chi2,     dim=0).numpy()
    centroid_all = torch.cat(all_centroid, dim=0).numpy()
    vis_fwhm_all = torch.cat(all_vis_fwhm, dim=0).numpy()
    model_all = torch.cat(model_bag, dim=0)
    data_all  = torch.cat(data_bag,  dim=0)
    ndof = stamp_size * stamp_size - 1

    print(f'  evaluated {n_stars_kept} stars across '
          f'{len(all_chi2)} tiles\n')

    # --- Plots --------------------------------------------------------------
    print('Rendering diagnostic plots ...')
    _plot_chi2_histogram(chi2_all, ndof, out_dir / 'chi2_per_band.png')
    _plot_centroid_drift(centroid_all, out_dir / 'centroid_drift.png')
    _plot_radial_profiles(model_all, data_all, out_dir / 'radial_profiles.png')
    _plot_stamp_gallery(model_all, data_all, args.n_gallery,
                        out_dir / 'stamp_gallery.png')

    # --- FWHM comparison ----------------------------------------------------
    model_fwhm = _psf_fwhm_per_band(psf, device)

    # --- DCR table ----------------------------------------------------------
    dcr_arcsec = psf.dcr.dcr_coeff.detach().cpu().numpy()   # [6, 2]

    # --- Text summary -------------------------------------------------------
    summary = []
    summary.append(f'PSFField diagnostic summary')
    summary.append(f'  checkpoint : {args.checkpoint}')
    summary.append(f'  epoch      : {epoch}')
    summary.append(f'  stars      : {n_stars_kept}   tiles: {len(all_chi2)}')
    summary.append('')
    summary.append('chi²/ndof per band   (target ≈ 1 for a well-fit PSF)')
    summary.append(f'  {"band":<14}{"median":>10}{"mean":>10}{"frac>2":>10}')
    for bi, band in enumerate(BAND_ORDER):
        vals = chi2_all[:, bi] / ndof
        med = float(np.median(vals))
        mn  = float(np.mean(vals))
        frac = float((vals > 2).mean())
        summary.append(f'  {band:<14}{med:>10.3f}{mn:>10.3f}{frac:>10.2%}')
    summary.append('')

    mag_mas = np.linalg.norm(centroid_all, axis=-1) * 1000.0
    summary.append(f'Centroid drift (|Δ| in mas)')
    summary.append(f'  median={np.median(mag_mas):.2f}  mean={np.mean(mag_mas):.2f}  '
                   f'p90={np.percentile(mag_mas, 90):.2f}  '
                   f'max={np.max(mag_mas):.2f}')
    summary.append('')

    summary.append('Model FWHM per band (area-based half-max)')
    for band, f in model_fwhm.items():
        summary.append(f'  {band:<14}{f:.4f} arcsec')
    summary.append(f'  [data median VIS FWHM: {np.median(vis_fwhm_all):.4f} arcsec]')
    summary.append('')

    summary.append('DCR coefficients (mas per mag of g−i colour, Rubin only)')
    for bi, band in enumerate(BAND_ORDER[:N_RUBIN]):
        dx, dy = dcr_arcsec[bi]
        summary.append(f'  {band:<14} dx={dx*1000:+8.2f}   dy={dy*1000:+8.2f}')
    summary.append('')

    text = '\n'.join(summary)
    print(text)
    (out_dir / 'summary.txt').write_text(text)

    with open(out_dir / 'diagnostics.json', 'w') as f:
        json.dump({
            'n_stars':          n_stars_kept,
            'n_tiles':          len(all_chi2),
            'chi2_median_per_band': {b: float(np.median(chi2_all[:, bi]) / ndof)
                                     for bi, b in enumerate(BAND_ORDER)},
            'centroid_drift_mas_median': float(np.median(mag_mas)),
            'centroid_drift_mas_p90':    float(np.percentile(mag_mas, 90)),
            'model_fwhm_arcsec':         model_fwhm,
            'dcr_coeff_arcsec':          dcr_arcsec.tolist(),
        }, f, indent=2)

    print(f'\nAll outputs → {out_dir}')


if __name__ == '__main__':
    main()
