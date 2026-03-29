"""Global sky-coordinate astrometric concordance solve.

Instead of fitting an independent control-grid field per tile (which causes
discontinuities at tile boundaries), this script:

  1. Runs the NN matcher on every tile to collect per-source predictions
     (sky position, predicted offset, uncertainty).
  2. Converts all source positions to a common sky frame (arcsec offsets
     from a field reference point).
  3. Fits ONE smooth 2D field over the entire mosaic footprint using the
     same control-grid solver — no tile boundaries, no edge artefacts.
  4. Exports a single FITS with the global concordance field sampled on a
     regular RA/Dec grid covering the full footprint, plus a coverage map.

The global field is stored at angular resolution DSTEP_ARCSEC (default 1"),
so downstream code interpolates in sky coordinates rather than tile pixels.

Usage
-----
    python infer_global_concordance.py \
        --rubin-dir  ../../data/rubin_tiles_ecdfs \
        --euclid-dir ../../data/euclid_tiles_ecdfs \
        --checkpoint ../checkpoints/astrometry_v6_phaseB2/checkpoint_best.pt \
        --v6-checkpoint ../checkpoints/jaisp_v6_phaseB2/checkpoint_best.pt \
        --output     ../checkpoints/astrometry_v6_phaseB2/global_concordance_r.fits \
        --dstep-arcsec 1.0 \
        --auto-grid \
        --plot-dir   ../checkpoints/astrometry_v6_phaseB2/global_plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def _setup_imports():
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir.parent
    for p in (models_dir, script_dir):
        sp = str(p)
        if sp in sys.path:
            sys.path.remove(sp)
        sys.path.insert(0, sp)

_setup_imports()

import torch
from infer_concordance import (
    _normalize_any_band,
    build_parser as build_infer_parser,
    load_model,
    predict_tile,
)
from dataset import (
    discover_tile_pairs,
    normalize_rubin_bands,
)
from field_solver import auto_grid_shape, evaluate_control_grid_mesh, solve_control_grid_field


# ============================================================
# Global collection
# ============================================================

def collect_all_predictions(
    model,
    device: torch.device,
    pairs: list,
    target_band: str,
    input_bands: list,
    detect_bands: list,
    args,
) -> dict:
    """
    Run the NN on every tile and return all source predictions in a single
    sky-coordinate frame.

    Returns
    -------
    dict with:
      ra, dec         : [N] source sky positions (degrees)
      pred_offsets    : [N, 2] predicted (dRA*, dDec) in arcsec
      raw_offsets     : [N, 2] raw WCS offsets in arcsec
      sigma           : [N] predicted uncertainty in arcsec
      tile_ids        : [N] which tile each source came from
    """
    all_ra, all_dec = [], []
    all_pred, all_raw, all_sigma = [], [], []
    all_tile = []

    for tile_id, rubin_path, euclid_path in pairs:
        item = predict_tile(
            model, device, rubin_path, euclid_path,
            target_band, input_bands, detect_bands, args,
        )
        if item is None:
            continue

        vis_xy       = item['vis_xy']           # [M, 2] VIS pixel positions
        pred_offsets = item['pred_offsets']      # [M, 2] arcsec
        raw_offsets  = item['raw_offsets']       # [M, 2] arcsec
        sigma        = item['sigma_arcsec']      # [M]

        # Convert VIS pixel positions → sky coordinates using the VIS WCS
        try:
            from astropy.wcs import WCS
            from source_matching import safe_header_from_card_string
            import numpy as np
            edata = np.load(euclid_path, allow_pickle=True)
            vwcs  = WCS(safe_header_from_card_string(edata['wcs_VIS'].item()))
            ra, dec = vwcs.wcs_pix2world(vis_xy[:, 0], vis_xy[:, 1], 0)
        except Exception as exc:
            print(f'[skip] {tile_id}: WCS conversion failed ({exc})')
            continue

        all_ra.append(ra)
        all_dec.append(dec)
        all_pred.append(pred_offsets)
        all_raw.append(raw_offsets)
        all_sigma.append(sigma)
        all_tile.extend([tile_id] * len(ra))
        print(f'  {tile_id}: {len(ra)} sources')

    if not all_ra:
        raise RuntimeError('No sources collected from any tile.')

    return {
        'ra':           np.concatenate(all_ra),
        'dec':          np.concatenate(all_dec),
        'pred_offsets': np.concatenate(all_pred, axis=0),
        'raw_offsets':  np.concatenate(all_raw,  axis=0),
        'sigma':        np.concatenate(all_sigma),
        'tile_ids':     all_tile,
    }


# ============================================================
# Sky-coordinate field solve
# ============================================================

def solve_global_field(
    sources: dict,
    dstep_arcsec: float = 1.0,
    grid_h: int = 32,
    grid_w: int = 32,
    smooth_lambda: float = 1e-2,
    anchor_lambda: float = 1e-3,
    auto_grid: bool = True,
    clip_arcsec: float = 0.3,
    solver: str = 'grid',
    nn_hidden_dim: int = 64,
    nn_layers: int = 4,
    nn_steps: int = 2000,
    nn_lr: float = 1e-3,
    nn_weight_decay: float = 1e-4,
) -> dict:
    """
    Fit a single smooth concordance field over the full mosaic in sky coords.

    The solver works in a local tangent-plane frame (arcsec offsets from the
    field centroid), so there is no pixel-scale ambiguity. The output mesh is
    sampled on a regular RA/Dec grid at DSTEP_ARCSEC resolution.

    Parameters
    ----------
    clip_arcsec : float
        Reject sources whose predicted offset magnitude exceeds this value in
        arcsec before solving (default 0.3" = 300 mas).  Set to np.inf to
        disable.
    solver : {'grid', 'nn'}
        'grid' — regularised control-grid least-squares (fast, default).
        'nn'   — small MLP trained with Adam + weight-decay smoothness prior.
                 No grid resolution to choose; SiLU activations give a
                 differentiable interpolant.  Use nn_* parameters to tune.
    nn_hidden_dim : neurons per hidden layer (default 64)
    nn_layers     : number of hidden layers (default 4)
    nn_steps      : Adam training steps (default 2000)
    nn_lr         : initial learning rate (default 1e-3)
    nn_weight_decay : L2 weight-decay — higher = smoother field (default 1e-4)

    Returns
    -------
    dict with:
      field        : solver object (control-grid field OR DistortionMLP)
      mesh         : {'dra', 'ddec', 'coverage'} arrays on regular sky grid
      ra_grid      : [W_mesh] RA values of the mesh columns (degrees)
      dec_grid     : [H_mesh] Dec values of the mesh rows (degrees)
      ra_ref       : reference RA (field centroid, degrees)
      dec_ref      : reference Dec (field centroid, degrees)
      dstep_arcsec : float
      n_sources    : int
      solver       : str — which solver was used
    """
    ra   = sources['ra']
    dec  = sources['dec']
    pred = sources['pred_offsets']
    sig  = np.maximum(sources['sigma'], 1e-4)

    # Sigma-clip: reject sources with |pred| > clip_arcsec
    pred_mag = np.hypot(pred[:, 0], pred[:, 1])
    keep = pred_mag <= clip_arcsec
    n_total = len(ra)
    ra, dec, pred, sig = ra[keep], dec[keep], pred[keep], sig[keep]
    print(f'  Outlier clip ({clip_arcsec*1000:.0f} mas): kept {keep.sum()}/{n_total} sources')

    weights = 1.0 / sig ** 2

    # Field centroid as reference point
    ra_ref  = float(np.median(ra))
    dec_ref = float(np.median(dec))
    cos_dec = np.cos(np.deg2rad(dec_ref))

    # Convert sky positions to local tangent-plane arcsec offsets from centroid
    x_arcsec = (ra  - ra_ref)  * cos_dec * 3600.0
    y_arcsec = (dec - dec_ref) * 3600.0

    pos_xy  = np.stack([x_arcsec, y_arcsec], axis=1).astype(np.float32)
    x_min, x_max = float(pos_xy[:, 0].min()), float(pos_xy[:, 0].max())
    y_min, y_max = float(pos_xy[:, 1].min()), float(pos_xy[:, 1].max())
    field_shape_arcsec = (y_max - y_min, x_max - x_min)

    # Shift to non-negative coordinates
    pos_shifted = pos_xy - np.array([[x_min, y_min]], dtype=np.float32)
    field_h = int(y_max - y_min + 1)
    field_w = int(x_max - x_min + 1)
    dstep_px = max(1, int(round(dstep_arcsec)))   # 1 px = 1 arcsec in this frame

    # ── Solve ─────────────────────────────────────────────────────────────────
    if solver == 'nn':
        from nn_field_solver import fit_nn_field, evaluate_nn_mesh
        print(f'  NN solver: {nn_hidden_dim}×{nn_layers} layers, '
              f'{nn_steps} steps, wd={nn_weight_decay}')
        field, nn_meta = fit_nn_field(
            pos_arcsec     = pos_shifted,
            offsets_arcsec = pred.astype(np.float32),
            weights        = weights.astype(np.float32),
            hidden_dim     = nn_hidden_dim,
            n_layers       = nn_layers,
            n_steps        = nn_steps,
            lr             = nn_lr,
            weight_decay   = nn_weight_decay,
        )
        mesh = evaluate_nn_mesh(
            model              = field,
            meta               = nn_meta,
            field_h            = field_h,
            field_w            = field_w,
            dstep              = dstep_px,
            pos_arcsec_anchors = pos_shifted,
        )
        grid_shape = (nn_layers, nn_hidden_dim)   # informational only
        anchor_radius = float('nan')
    else:
        # Control-grid (default)
        if auto_grid:
            grid_h, grid_w = auto_grid_shape(
                n_anchors=len(ra),
                default=(grid_h, grid_w),
                min_shape=(6, 6),
            )
            print(f'  Auto grid: {grid_h}×{grid_w} for {len(ra)} sources')

        cell_y = (y_max - y_min) / max(1, grid_h - 1)
        cell_x = (x_max - x_min) / max(1, grid_w - 1)
        anchor_radius = 2.0 * 0.5 * (cell_y + cell_x)

        field = solve_control_grid_field(
            vis_xy          = pos_shifted,
            offsets_arcsec  = pred.astype(np.float32),
            weights         = weights.astype(np.float32),
            vis_shape       = (field_h, field_w),
            grid_shape      = (grid_h, grid_w),
            smooth_lambda   = smooth_lambda,
            anchor_lambda   = anchor_lambda,
            anchor_radius_px= anchor_radius,
        )
        mesh = evaluate_control_grid_mesh(
            field,
            vis_shape = (field_h, field_w),
            dstep     = dstep_px,
            anchor_xy = pos_shifted,
        )
        grid_shape = (grid_h, grid_w)

    # Build RA/Dec coordinate arrays for the mesh
    mesh_h, mesh_w = mesh['dra'].shape
    dec_grid = dec_ref + (np.arange(mesh_h) * dstep_arcsec + y_min) / 3600.0
    ra_grid  = ra_ref  + (np.arange(mesh_w) * dstep_arcsec + x_min) / (cos_dec * 3600.0)

    return {
        'field':               field,
        'mesh':                mesh,
        'ra_grid':             ra_grid,
        'dec_grid':            dec_grid,
        'ra_ref':              ra_ref,
        'dec_ref':             dec_ref,
        'dstep_arcsec':        dstep_arcsec,
        'n_sources':           len(ra),
        'grid_shape':          grid_shape,
        'anchor_radius_arcsec': anchor_radius,
        'field_shape_arcsec':  field_shape_arcsec,
        'solver':              solver,
    }


# ============================================================
# FITS output
# ============================================================

def write_global_fits(
    result: dict,
    output_path: str,
    target_band: str,
) -> None:
    """Write the global concordance field to a FITS file."""
    mesh      = result['mesh']
    ra_grid   = result['ra_grid']
    dec_grid  = result['dec_grid']
    ra_ref    = result['ra_ref']
    dec_ref   = result['dec_ref']
    dstep     = result['dstep_arcsec']

    # Build a WCS so downstream code can locate the mesh on the sky
    # Axes: axis 1 = RA (mesh columns), axis 2 = Dec (mesh rows)
    w = WCS(naxis=2)
    w.wcs.crpix  = [1.0, 1.0]
    w.wcs.crval  = [float(ra_grid[0]), float(dec_grid[0])]
    w.wcs.cdelt  = [float(ra_grid[1]  - ra_grid[0])  if len(ra_grid)  > 1 else dstep / 3600.0,
                    float(dec_grid[1] - dec_grid[0]) if len(dec_grid) > 1 else dstep / 3600.0]
    w.wcs.ctype  = ['RA---TAN', 'DEC--TAN']
    wcs_header   = w.to_header()

    def _make_hdu(data, name, comment):
        hdu = fits.ImageHDU(data=data.astype(np.float32), name=name)
        hdu.header.update(wcs_header)
        hdu.header['DSTEP']    = (float(dstep),   'Mesh step size in arcsec')
        hdu.header['DUNIT']    = ('arcsec',        'Unit of offset values')
        hdu.header['INTERP']   = ('bilinear',      'Recommended interpolation method')
        hdu.header['CONCRDNC'] = (True,            'Global concordance field')
        hdu.header['RBNBAND']  = (target_band,     'Rubin band')
        hdu.header['REFFRAME'] = ('euclid_VIS',    'Reference astrometric frame')
        hdu.header['SOLVETYP'] = ('global_sky',    'Global sky-coord field solve')
        hdu.header['RA_REF']   = (float(ra_ref),   'Field centroid RA (deg)')
        hdu.header['DEC_REF']  = (float(dec_ref),  'Field centroid Dec (deg)')
        hdu.header['NSRC']     = (result['n_sources'], 'Total sources used in solve')
        hdu.header['GRIDH']    = (result['grid_shape'][0], 'Control grid height')
        hdu.header['GRIDW']    = (result['grid_shape'][1], 'Control grid width')
        hdu.header['COMMENT']  = comment
        return hdu

    hdus = [fits.PrimaryHDU()]
    hdus[0].header['CONCRDNC'] = (True, 'JAISP global sky-coord concordance product')
    hdus[0].header['SOLVETYP'] = ('global_sky', 'Single field fitted over full mosaic')
    hdus[0].header['RBNBAND']  = (target_band,  'Rubin band')
    hdus[0].header['NSRC']     = (result['n_sources'], 'Total sources in solve')

    hdus.append(_make_hdu(mesh['dra'],  'GLOBAL.DRA', 'DeltaRA* offset field (arcsec)'))
    hdus.append(_make_hdu(mesh['ddec'], 'GLOBAL.DDE', 'DeltaDec offset field (arcsec)'))
    if 'coverage' in mesh and mesh['coverage'] is not None:
        hdus.append(_make_hdu(mesh['coverage'], 'GLOBAL.COV',
                              'Min distance to nearest source (arcsec)'))

    fits.HDUList(hdus).writeto(output_path, overwrite=True)
    print(f'Wrote global concordance: {output_path}')
    print(f'  Mesh shape: {mesh["dra"].shape}  ({dstep}" per pixel)')
    print(f'  Sources:    {result["n_sources"]}')
    print(f'  Grid:       {result["grid_shape"][0]}×{result["grid_shape"][1]}')


# ============================================================
# Global ConcordanceMap for apply_concordance.py
# ============================================================

class GlobalConcordanceMap:
    """
    Load and apply a global sky-coordinate concordance FITS.

    Usage is identical to ConcordanceMap but works in sky coords —
    no tile_id needed, no boundary discontinuities.

        gcmap = GlobalConcordanceMap('global_concordance_r.fits')
        vis_x, vis_y = gcmap.rubin_to_vis(
            rubin_x, rubin_y, rubin_wcs, vis_wcs, band='r'
        )
    """

    def __init__(self, fits_path: str):
        with fits.open(fits_path) as hdul:
            dra_hdu  = hdul['GLOBAL.DRA']
            dde_hdu  = hdul['GLOBAL.DDE']
            self.dra  = dra_hdu.data.astype(np.float32)
            self.dde  = dde_hdu.data.astype(np.float32)
            self.cov  = hdul['GLOBAL.COV'].data.astype(np.float32) if 'GLOBAL.COV' in hdul else None
            h = dra_hdu.header
            self.dstep_arcsec = float(h.get('DSTEP', 1.0))
            self.ra_ref       = float(h.get('RA_REF',  0.0))
            self.dec_ref      = float(h.get('DEC_REF', 0.0))
            # Use the WCS to get the grid origin
            self.wcs = WCS(h, naxis=2)
        print(f'GlobalConcordanceMap: {self.dra.shape} mesh at {self.dstep_arcsec}"/px  '
              f'from global_concordance FITS')

    def _sky_to_mesh_xy(self, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
        """Convert sky (RA, Dec) → fractional mesh pixel indices [N, 2] (col, row)."""
        mx, my = self.wcs.wcs_world2pix(ra, dec, 0)
        return np.stack([mx.astype(np.float32), my.astype(np.float32)], axis=1)

    def _interp(self, field: np.ndarray, mesh_xy: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates
        return map_coordinates(
            field.astype(np.float64),
            [mesh_xy[:, 1], mesh_xy[:, 0]],   # row, col
            order=1, mode='nearest',
        ).astype(np.float32)

    def correction_at_sky(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
    ):
        """Return (dra_arcsec, ddec_arcsec) at sky positions."""
        mesh_xy = self._sky_to_mesh_xy(np.atleast_1d(ra), np.atleast_1d(dec))
        return self._interp(self.dra, mesh_xy), self._interp(self.dde, mesh_xy)

    def coverage_at_sky(self, ra: np.ndarray, dec: np.ndarray) -> Optional[np.ndarray]:
        if self.cov is None:
            return None
        mesh_xy = self._sky_to_mesh_xy(np.atleast_1d(ra), np.atleast_1d(dec))
        return self._interp(self.cov, mesh_xy)

    def rubin_to_vis(
        self,
        rubin_x,
        rubin_y,
        rubin_wcs: WCS,
        vis_wcs: WCS,
        band: str = 'r',   # noqa: ARG002 — kept for API compatibility with ConcordanceMap
    ):
        """Project Rubin pixel(s) onto the VIS grid with global concordance applied."""
        rubin_x = np.atleast_1d(np.asarray(rubin_x, dtype=np.float64))
        rubin_y = np.atleast_1d(np.asarray(rubin_y, dtype=np.float64))
        ra, dec = rubin_wcs.wcs_pix2world(rubin_x, rubin_y, 0)
        dra, ddec = self.correction_at_sky(ra, dec)
        cos_dec  = np.cos(np.deg2rad(dec))
        ra_corr  = ra  + (dra  / 3600.0) / cos_dec
        dec_corr = dec + (ddec / 3600.0)
        vis_x, vis_y = vis_wcs.wcs_world2pix(ra_corr, dec_corr, 0)
        return vis_x.squeeze(), vis_y.squeeze()


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = build_infer_parser()
    p.description = 'Fit a single global sky-coordinate concordance field over all tiles.'
    # Remove per-tile output, add global-specific args
    p.add_argument('--dstep-arcsec', type=float, default=1.0,
                   help='Output mesh resolution in arcsec (default: 1.0)')
    p.add_argument('--grid-h-global', type=int, default=32,
                   help='Control grid height for global solve (default: 32)')
    p.add_argument('--grid-w-global', type=int, default=32,
                   help='Control grid width for global solve (default: 32)')
    p.add_argument('--plot', type=str, default='',
                   help='Optional path to save the diagnostic PNG (e.g. global_plot.png)')
    p.add_argument('--clip-arcsec', type=float, default=0.3,
                   help='Reject sources with |pred offset| > this value before solving '
                        '(arcsec, default 0.3 = 300 mas). Set to a large value to disable.')
    # ── Solver choice ──────────────────────────────────────────────────────────
    p.add_argument('--solver', choices=['grid', 'nn'], default='grid',
                   help='Field solver: "grid" = control-grid (default), "nn" = MLP.')
    p.add_argument('--nn-hidden-dim', type=int, default=64,
                   help='[nn solver] neurons per hidden layer (default 64)')
    p.add_argument('--nn-layers', type=int, default=4,
                   help='[nn solver] number of hidden layers (default 4)')
    p.add_argument('--nn-steps', type=int, default=2000,
                   help='[nn solver] Adam training steps (default 2000)')
    p.add_argument('--nn-lr', type=float, default=1e-3,
                   help='[nn solver] initial learning rate (default 1e-3)')
    p.add_argument('--nn-weight-decay', type=float, default=1e-4,
                   help='[nn solver] L2 weight decay — higher = smoother field (default 1e-4)')
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    v6_ckpt = getattr(args, 'v6_checkpoint', '') or ''
    model, ckpt = load_model(args.checkpoint, device, v6_checkpoint=v6_ckpt)

    target_band = _normalize_any_band(
        str(ckpt.get('target_band', ckpt.get('args', {}).get('rubin_band', 'r')))
    )
    if target_band == 'multiband':
        target_band = 'rubin_r'

    input_bands_raw = [str(x) for x in ckpt.get('input_bands', [target_band])]
    input_bands = []
    for b in input_bands_raw:
        nb = _normalize_any_band(b)
        if nb not in input_bands:
            input_bands.append(nb)
    detect_bands = normalize_rubin_bands(args.detect_bands) or [f'rubin_{b}' for b in ('g','r','i','z')]

    pairs = discover_tile_pairs(args.rubin_dir, args.euclid_dir)
    if args.tile_id:
        pairs = [(t, r, e) for t, r, e in pairs if t == args.tile_id]

    print(f'Collecting predictions from {len(pairs)} tiles...')
    sources = collect_all_predictions(
        model, device, pairs, target_band, input_bands, detect_bands, args,
    )
    print(f'Total sources: {sources["n_sources"] if "n_sources" in sources else len(sources["ra"])}')

    print('\nFitting global field...')
    result = solve_global_field(
        sources,
        dstep_arcsec    = args.dstep_arcsec,
        grid_h          = getattr(args, 'grid_h_global', 32),
        grid_w          = getattr(args, 'grid_w_global', 32),
        smooth_lambda   = args.smooth_lambda,
        anchor_lambda   = args.anchor_lambda,
        auto_grid       = getattr(args, 'auto_grid', False),
        clip_arcsec     = getattr(args, 'clip_arcsec', 0.3),
        solver          = getattr(args, 'solver', 'grid'),
        nn_hidden_dim   = getattr(args, 'nn_hidden_dim', 64),
        nn_layers       = getattr(args, 'nn_layers', 4),
        nn_steps        = getattr(args, 'nn_steps', 2000),
        nn_lr           = getattr(args, 'nn_lr', 1e-3),
        nn_weight_decay = getattr(args, 'nn_weight_decay', 1e-4),
    )

    # Summary stats
    pred = sources['pred_offsets']
    raw  = sources['raw_offsets']
    pred_mag = np.hypot(pred[:, 0], pred[:, 1]) * 1000.0
    raw_mag  = np.hypot(raw[:,  0], raw[:,  1]) * 1000.0
    print(f'\nRaw WCS  median: {np.median(raw_mag):.1f} mas')
    print(f'NN pred  median: {np.median(pred_mag):.1f} mas')
    print(f'Field footprint: {result["field_shape_arcsec"][1]:.0f}" × {result["field_shape_arcsec"][0]:.0f}"')

    write_global_fits(result, args.output, target_band)

    if getattr(args, 'plot', ''):
        from viz import plot_global_concordance
        plot_global_concordance(result, sources, args.plot, target_band=target_band)

    if args.summary_json:
        summary = {
            'n_sources':    int(len(sources['ra'])),
            'solver':       result['solver'],
            'grid_shape':   list(result['grid_shape']),
            'dstep_arcsec': result['dstep_arcsec'],
            'raw_median_mas':  float(np.median(raw_mag)),
            'pred_median_mas': float(np.median(pred_mag)),
            'ra_ref':  result['ra_ref'],
            'dec_ref': result['dec_ref'],
            'field_shape_arcsec': list(result['field_shape_arcsec']),
        }
        with open(args.summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'Summary: {args.summary_json}')


if __name__ == '__main__':
    main()
