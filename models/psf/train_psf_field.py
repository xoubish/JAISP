"""
Joint PSFField training: optimise PSF model *and* per-star sub-pixel centroids
simultaneously, with chromatic SED conditioning and Rubin-only DCR.

Usage
-----
    python models/psf/train_psf_field.py \
        --rubin_dir  data/rubin_tiles_200 \
        --euclid_dir data/euclid_tiles_200 \
        --out        models/checkpoints/psf_field_v1.pt \
        --epochs     30 \
        --stamp_size 21 \
        --wandb_project jaisp-psf

Design
------
- Per-tile learnable centroid parameters stored in a `ParameterDict` keyed
  by tile stem. These are physical quantities (sky position refinements)
  that persist across epochs.
- Two parameter groups with different learning rates: PSFField slow, per-star
  centroids faster (they need to move ~0.02" to settle; PSFField needs
  gentle refinement of ~100k neural-net params).
- SED vector is initialised crudely from inner-aperture fluxes during star
  selection and refreshed every `--sed_refresh_every` epochs from the
  analytic optimal fluxes — gives the chromatic PSF model a consistent
  picture of each star's colour as the PSF converges.
- Star catalogues are computed once and cached — selection is
  deterministic given the tile data.
- All 10 bands are fit jointly on every tile pass (loss summed over bands).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_MODELS,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from psf.psf_field import (                                                # noqa: E402
    PSFField, BAND_PX_SCALE, N_BANDS, N_RUBIN,
    analytic_optimal_flux, chi2_loss, BAND_ORDER,
)
from psf.star_selection import select_stars, StarCatalog                   # noqa: E402


# ---------------------------------------------------------------------------
# Tile index
# ---------------------------------------------------------------------------

def _build_tile_index(rubin_dir: Path, euclid_dir: Path) -> List[dict]:
    euclid_map = {
        p.stem.replace('_euclid', ''): p
        for p in euclid_dir.glob('tile_x*_y*_euclid.npz')
    }
    tiles = []
    for rp in sorted(rubin_dir.glob('tile_x*_y*.npz')):
        ep = euclid_map.get(rp.stem)
        if ep is not None:
            tiles.append({'stem': rp.stem, 'rubin_path': rp, 'euclid_path': ep})
    return tiles


# ---------------------------------------------------------------------------
# Per-tile centroid registry
# ---------------------------------------------------------------------------

class CentroidRegistry(nn.Module):
    """
    Persistent `nn.Parameter` for every (tile, star_index) sub-pixel centroid.

    Keyed by tile stem. Each tile gets a [N_i, 2] arcsec offset tensor.
    """

    def __init__(self):
        super().__init__()
        self.params = nn.ParameterDict()

    def register(self, tile_stem: str, init_arcsec: torch.Tensor) -> nn.Parameter:
        key = tile_stem.replace('.', '_')   # ParameterDict keys cannot contain '.'
        if key not in self.params:
            self.params[key] = nn.Parameter(init_arcsec.detach().clone())
        return self.params[key]

    def get(self, tile_stem: str) -> Optional[nn.Parameter]:
        return self.params.get(tile_stem.replace('.', '_'))

    def total_params(self) -> int:
        return sum(p.numel() for p in self.params.values())


# ---------------------------------------------------------------------------
# One-tile training step
# ---------------------------------------------------------------------------

def _tile_step(
    psf_field: PSFField,
    centroid: nn.Parameter,
    cat: StarCatalog,
    stamp_size: int,
    sub_grid: int,
) -> torch.Tensor:
    """
    Compute summed chi² over all 10 bands for one tile.
    Returns a scalar loss tensor (no reduction across bands applied
    beyond equal weighting per band so faint bands don't dominate).
    """
    N = len(cat)
    device = cat.stamps.device
    rms = cat.rms_stamps.clamp(min=1e-8)
    var = rms * rms                                          # [N, 10, S, S]

    losses = []
    for bi in range(N_BANDS):
        data_b = cat.stamps[:, bi]                           # [N, S, S]
        var_b  = var[:, bi]
        model_b = psf_field.render_stamps(
            centroids_arcsec=centroid,
            tile_pos=cat.tile_pos,
            band_idx=torch.full((N,), bi, dtype=torch.long, device=device),
            sed_vec=cat.sed_init,
            stamp_size=stamp_size,
            px_scale=float(BAND_PX_SCALE[bi]),
            sub_grid=sub_grid,
            apply_dcr=True,
        )
        losses.append(chi2_loss(data_b, model_b, var_b, reduce='mean'))
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Diagnostic: measure PSF FWHM at tile centre for every band
# ---------------------------------------------------------------------------

@torch.no_grad()
def _measure_band_fwhm(
    psf_field: PSFField,
    sed_vec: torch.Tensor,     # [1, 10]
    tile_pos: torch.Tensor,    # [1, 2]
    stamp_size: int = 41,
    sub_grid: int = 4,
) -> Dict[str, float]:
    """Return per-band FWHM in arcsec using half-max contour on a large stamp."""
    out = {}
    device = sed_vec.device
    for bi, band in enumerate(BAND_ORDER):
        px = float(BAND_PX_SCALE[bi])
        stamp = psf_field.render_stamps(
            centroids_arcsec=torch.zeros(1, 2, device=device),
            tile_pos=tile_pos,
            band_idx=torch.tensor([bi], device=device, dtype=torch.long),
            sed_vec=sed_vec,
            stamp_size=stamp_size,
            px_scale=px,
            sub_grid=sub_grid,
            apply_dcr=False,
        )[0]
        # Half-maximum-area estimator of FWHM
        peak = stamp.max()
        if peak <= 0:
            out[band] = float('nan')
            continue
        above = (stamp >= 0.5 * peak).float().sum().item()
        # Area → equivalent circle diameter → FWHM in arcsec
        fwhm_px = 2.0 * (above / np.pi) ** 0.5
        out[band] = fwhm_px * px
    return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training PSFField on {device}')

    # --- W&B (optional) -----------------------------------------------------
    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )

    # --- Data ---------------------------------------------------------------
    tiles = _build_tile_index(Path(args.rubin_dir), Path(args.euclid_dir))
    print(f'Found {len(tiles)} paired Rubin+Euclid tiles')
    if len(tiles) == 0:
        raise SystemExit('No tiles found')

    # --- Model --------------------------------------------------------------
    psf_field = PSFField(
        sed_embed_dim=args.sed_embed_dim,
        band_embed_dim=args.band_embed_dim,
        tile_freqs=args.tile_freqs,
        siren_hidden=args.siren_hidden,
        siren_depth=args.siren_depth,
        w0_first=args.w0_first,
    ).to(device)

    centroid_reg = CentroidRegistry().to(device)

    n_psf = sum(p.numel() for p in psf_field.parameters())
    print(f'PSFField params: {n_psf/1e3:.1f}k')

    # --- CenterNet pseudo-labels (optional) --------------------------------
    # These are VIS-based detections with bright-core + diffraction-spike
    # masking — what CenterNet is trained to reproduce. Coordinates are
    # normalised [0, 1] w.r.t. VIS tile dimensions; we scale back to pixels
    # per-tile when passing to select_stars.
    pseudo_labels = None
    if args.centernet_labels:
        pl = torch.load(args.centernet_labels, map_location='cpu', weights_only=False)
        pseudo_labels = pl['labels'] if 'labels' in pl else pl
        print(f'Loaded CenterNet pseudo-labels for {len(pseudo_labels)} tiles '
              f'from {args.centernet_labels}')

    # --- Star selection (cache per tile) -----------------------------------
    print('Selecting stars (one-time) ...')
    t0 = time.time()
    catalogs: Dict[str, StarCatalog] = {}
    for ti, t in enumerate(tiles):
        # Resolve CenterNet detections for this tile if available
        det_px = None
        if pseudo_labels is not None and t['stem'] in pseudo_labels:
            entry = pseudo_labels[t['stem']]
            xy_norm = entry[0] if isinstance(entry, tuple) else entry
            eu = np.load(t['euclid_path'], allow_pickle=True, mmap_mode='r')
            H_vis, W_vis = eu['img_VIS'].shape
            det_px = np.stack([
                xy_norm[:, 0] * W_vis,
                xy_norm[:, 1] * H_vis,
            ], axis=1).astype(np.float32)

        cat = select_stars(
            t['rubin_path'], t['euclid_path'],
            stamp_size=args.stamp_size,
            max_stars=args.max_stars_per_tile,
            vis_detect_nsig=args.vis_detect_nsig,
            stellar_locus_width=args.stellar_locus_width,
            isolation_arcsec=args.isolation_arcsec,
            detections_vis_px=det_px,
            device=device,
        )
        if cat is None or len(cat) < args.min_stars_per_tile:
            continue
        catalogs[t['stem']] = cat
        centroid_reg.register(t['stem'], cat.centroid_init_arcsec)

    elapsed = time.time() - t0
    n_stars_total = sum(len(c) for c in catalogs.values())
    print(f'  selected {n_stars_total} stars across {len(catalogs)} tiles in {elapsed:.1f}s')
    print(f'  {centroid_reg.total_params()/1e3:.1f}k centroid parameters')
    if use_wandb:
        wandb.run.summary['n_tiles_with_stars'] = len(catalogs)
        wandb.run.summary['n_stars_total'] = n_stars_total

    if len(catalogs) == 0:
        raise SystemExit('No tiles survived star selection')

    # --- Optimiser (two parameter groups) ----------------------------------
    optimizer = optim.Adam(
        [
            {'params': psf_field.parameters(),   'lr': args.lr_psf},
            {'params': centroid_reg.parameters(), 'lr': args.lr_centroid},
        ]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ------------------------------------------------------
    best_loss = float('inf')
    tile_stems = list(catalogs.keys())
    global_step = 0

    for epoch in range(args.epochs):
        np.random.shuffle(tile_stems)
        epoch_losses = []

        for stem in tile_stems:
            cat = catalogs[stem]
            centroid = centroid_reg.get(stem)

            optimizer.zero_grad()
            loss = _tile_step(
                psf_field, centroid, cat,
                stamp_size=args.stamp_size,
                sub_grid=args.sub_grid,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(psf_field.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(centroid_reg.parameters(), 0.05)
            optimizer.step()

            epoch_losses.append(float(loss))
            global_step += 1

            if use_wandb and global_step % args.log_every == 0:
                wandb.log({
                    'train/loss_step': float(loss),
                    'step': global_step,
                })

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')

        # Sample centroid statistics (how far they've moved from init)
        cent_rms = float(torch.sqrt(
            torch.stack([(p ** 2).mean() for p in centroid_reg.parameters()]).mean()
        )) if centroid_reg.total_params() > 0 else 0.0

        lr_psf = optimizer.param_groups[0]['lr']
        lr_cen = optimizer.param_groups[1]['lr']
        print(f'Epoch {epoch+1:3d}/{args.epochs}  loss={mean_loss:.4f}  '
              f'cent_rms={cent_rms*1000:.1f} mas  '
              f'lr_psf={lr_psf:.2e}  lr_cen={lr_cen:.2e}')

        if use_wandb:
            log = {
                'train/loss_epoch': mean_loss,
                'train/centroid_rms_mas': cent_rms * 1000.0,
                'train/lr_psf': lr_psf,
                'train/lr_cen': lr_cen,
                'train/dcr_norm': float(psf_field.dcr.dcr_coeff.abs().mean()) * 1000.0,
                'epoch': epoch + 1,
            }
            if (epoch + 1) % args.diag_every == 0 or epoch == args.epochs - 1:
                # PSF FWHM per band at a representative tile position with
                # a mean-SED star (sed_vec ≈ 0 → average stellar colour)
                fwhms = _measure_band_fwhm(
                    psf_field,
                    sed_vec=torch.zeros(1, N_BANDS, device=device),
                    tile_pos=torch.tensor([[0.5, 0.5]], device=device),
                )
                for band, f in fwhms.items():
                    log[f'fwhm/{band}_arcsec'] = f
            wandb.log(log)

        if mean_loss < best_loss:
            best_loss = mean_loss
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'psf_field_state':  psf_field.state_dict(),
                'centroid_state':   centroid_reg.state_dict(),
                'config': {
                    'sed_embed_dim':  args.sed_embed_dim,
                    'band_embed_dim': args.band_embed_dim,
                    'tile_freqs':     args.tile_freqs,
                    'siren_hidden':   args.siren_hidden,
                    'siren_depth':    args.siren_depth,
                    'w0_first':       args.w0_first,
                    'stamp_size':     args.stamp_size,
                    'sub_grid':       args.sub_grid,
                    'max_stars_per_tile': args.max_stars_per_tile,
                    'stellar_locus_width': args.stellar_locus_width,
                    'isolation_arcsec':    args.isolation_arcsec,
                },
                'tiles':  list(catalogs.keys()),
                'epoch':  epoch + 1,
            }, args.out)
            if use_wandb:
                wandb.run.summary['best_loss'] = best_loss

    print(f'Training complete. Best loss: {best_loss:.4f}')
    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument('--rubin_dir',  required=True)
    p.add_argument('--euclid_dir', required=True)
    p.add_argument('--out',        default='models/checkpoints/psf_field_v1.pt')
    p.add_argument('--centernet_labels', default=None,
                   help='Path to CenterNet pseudo_labels.pt '
                        '(e.g. data/cached_features_v8_fine/pseudo_labels.pt). '
                        'If omitted, falls back to classical VIS peak detection.')
    # Architecture
    p.add_argument('--sed_embed_dim',  type=int,   default=8)
    p.add_argument('--band_embed_dim', type=int,   default=16)
    p.add_argument('--tile_freqs',     type=int,   default=6)
    p.add_argument('--siren_hidden',   type=int,   default=128)
    p.add_argument('--siren_depth',    type=int,   default=5)
    p.add_argument('--w0_first',       type=float, default=30.0)
    # Stamp / forward-model
    p.add_argument('--stamp_size', type=int, default=21)
    p.add_argument('--sub_grid',   type=int, default=4)
    # Star selection
    p.add_argument('--max_stars_per_tile', type=int,   default=80)
    p.add_argument('--min_stars_per_tile', type=int,   default=5)
    p.add_argument('--vis_detect_nsig',    type=float, default=20.0)
    p.add_argument('--stellar_locus_width',type=float, default=0.08)
    p.add_argument('--isolation_arcsec',   type=float, default=3.0)
    # Optimiser
    p.add_argument('--epochs',       type=int,   default=30)
    p.add_argument('--lr_psf',       type=float, default=3e-4)
    p.add_argument('--lr_centroid',  type=float, default=5e-3)
    # Logging
    p.add_argument('--wandb_project', default=None)
    p.add_argument('--wandb_run',     default=None)
    p.add_argument('--log_every',     type=int, default=25)
    p.add_argument('--diag_every',    type=int, default=2)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
