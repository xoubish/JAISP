"""Run the standalone local matcher and export smooth concordance FITS fields.

Changes from v1:
  - --auto-grid: automatically reduce grid resolution for low-match tiles
    so the solver is never heavily underdetermined.
  - --anchor-radius-px: adaptive per-node ridge regularization. Nodes far
    from any matched source get a stronger pull toward zero, eliminating
    the clumpy red/blue artifacts in source-sparse regions.
  - Coverage HDU: each tile gets a third extension ({prefix}.COV) recording
    the minimum distance (VIS pixels) from each mesh point to the nearest
    anchor.  Downstream code can threshold this to mask unreliable regions.
  - anchor_lambda default raised from 1e-4 to 1e-3.
  - Cleaner sys.path setup via a helper function.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS


def _setup_imports():
    """Add project directories to sys.path for cross-module imports."""
    import sys
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir.parent
    astrometry_dir = models_dir / 'astrometry'
    for p in (astrometry_dir, models_dir, script_dir):
        sp = str(p)
        if sp in sys.path:
            sys.path.remove(sp)
        sys.path.insert(0, sp)

_setup_imports()

from dataset import (
    discover_tile_pairs,
    extract_vis_patch,
    local_vis_pixel_to_sky_matrix,
    normalize_rubin_band,
    normalize_rubin_bands,
    reproject_rubin_patch_to_vis,
)
from field_solver import auto_grid_shape, evaluate_control_grid_mesh, solve_control_grid_field
from matcher import LocalAstrometryMatcher
from viz import save_tile_diagnostic
from jaisp_dataset_v4 import RUBIN_BAND_ORDER, _to_float32
from source_matching import (
    build_detection_image,
    detect_sources,
    match_sources_wcs,
    refine_centroids_in_band,
    safe_header_from_card_string,
)


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt.get('args', {})
    model = LocalAstrometryMatcher(
        rubin_channels=int(ckpt.get('rubin_channels', len(ckpt.get('input_bands', ['rubin_r'])))),
        hidden_channels=args.get('hidden_channels', 32),
        encoder_depth=args.get('encoder_depth', 4),
        search_radius=args.get('search_radius', 3),
        softmax_temp=args.get('softmax_temp', 0.05),
        mlp_hidden=args.get('mlp_hidden', 128),
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    return model, ckpt


def make_concordance_hdu(
    data: np.ndarray,
    extname: str,
    dstep: int,
    rubin_band: str,
    tile_id: str,
    vis_wcs_header: Optional['fits.Header'],
) -> 'fits.ImageHDU':
    hdu = fits.ImageHDU(data=data.astype(np.float32), name=extname)
    hdu.header['DSTEP'] = (int(dstep), 'Mesh sampling step in VIS pixels')
    hdu.header['DUNIT'] = ('arcsec', 'Unit of offset values')
    hdu.header['INTERP'] = ('bilinear', 'Recommended interpolation method')
    hdu.header['CONCRDNC'] = (True, 'This is a concordance offset extension')
    hdu.header['RBNBAND'] = (rubin_band, 'Rubin band for this concordance')
    hdu.header['REFFRAME'] = ('euclid_VIS', 'Reference frame')
    hdu.header['TILEID'] = (tile_id, 'Source tile identifier')
    hdu.header['FITMETH'] = ('control_grid', 'Standalone DL field fit')
    if vis_wcs_header is not None:
        for key in [
            'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
            'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
            'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
            'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
        ]:
            if key in vis_wcs_header:
                val = vis_wcs_header[key]
                if key in ('CRPIX1', 'CRPIX2'):
                    val = val / float(max(1, dstep))
                if key.startswith('CD') or key.startswith('CDELT'):
                    val = val * float(max(1, dstep))
                hdu.header[key] = val
    return hdu


def make_coverage_hdu(
    coverage: np.ndarray,
    extname: str,
    dstep: int,
    tile_id: str,
) -> 'fits.ImageHDU':
    """Coverage HDU: min distance (VIS px) from each mesh point to nearest anchor."""
    hdu = fits.ImageHDU(data=coverage.astype(np.float32), name=extname)
    hdu.header['DSTEP'] = (int(dstep), 'Mesh sampling step in VIS pixels')
    hdu.header['DUNIT'] = ('VIS_px', 'Unit of coverage values')
    hdu.header['COVTYPE'] = ('min_dist', 'Min distance to nearest anchor source')
    hdu.header['TILEID'] = (tile_id, 'Source tile identifier')
    return hdu


@torch.no_grad()
def predict_tile(
    model,
    device: torch.device,
    rubin_path: str,
    euclid_path: str,
    target_band: str,
    input_bands: List[str],
    detect_bands: List[str],
    args,
) -> Optional[Dict]:
    target_band = normalize_rubin_band(target_band)
    try:
        rdata = np.load(rubin_path, allow_pickle=True)
        edata = np.load(euclid_path, allow_pickle=True)
        rubin_cube = rdata['img']
        vis_img = np.nan_to_num(_to_float32(edata['img_VIS']), nan=0.0)
        rwcs = WCS(rdata['wcs_hdr'].item())
        vhdr = safe_header_from_card_string(edata['wcs_VIS'].item())
        vwcs = WCS(vhdr)
    except Exception as exc:
        print(f'[skip] {os.path.basename(rubin_path)}: load/wcs failed ({exc})')
        return None

    rubin_det = build_detection_image(rubin_cube, detect_bands, clip_sigma=args.detect_clip_sigma)
    rx, ry = detect_sources(rubin_det, nsig=args.rubin_nsig, smooth_sigma=args.rubin_smooth, min_dist=args.rubin_min_dist, max_sources=args.max_sources_rubin)
    vx, vy = detect_sources(vis_img, nsig=args.vis_nsig, smooth_sigma=args.vis_smooth, min_dist=args.vis_min_dist, max_sources=args.max_sources_vis)
    matched = match_sources_wcs(
        rx, ry, vx, vy, rwcs, vwcs,
        max_sep_arcsec=args.max_sep_arcsec,
        clip_sigma=args.clip_sigma,
        max_matches=args.max_matches,
    )
    if matched['vis_xy'].shape[0] < int(args.min_matches):
        return None

    target_idx = RUBIN_BAND_ORDER.index(target_band.split('_', 1)[1])
    rubin_target = np.nan_to_num(_to_float32(rubin_cube[target_idx]), nan=0.0)
    rubin_xy_target = refine_centroids_in_band(
        rubin_target,
        matched['rubin_xy'],
        radius=args.refine_radius,
        flux_floor_sigma=args.refine_flux_floor_sigma,
    )
    r_ra, r_dec = rwcs.wcs_pix2world(rubin_xy_target[:, 0], rubin_xy_target[:, 1], 0)
    v_ra, v_dec = vwcs.wcs_pix2world(matched['vis_xy'][:, 0], matched['vis_xy'][:, 1], 0)
    raw_dra = (v_ra - r_ra) * np.cos(np.deg2rad(v_dec)) * 3600.0
    raw_ddec = (v_dec - r_dec) * 3600.0
    raw_offsets = np.stack([raw_dra, raw_ddec], axis=1).astype(np.float32)

    rubin_patches = []
    vis_patches = []
    pix2sky_list = []
    kept_xy = []
    kept_raw = []
    vis_xy = matched['vis_xy'].astype(np.float32)
    for idx_anchor, anchor_xy in enumerate(vis_xy):
        vis_patch = extract_vis_patch(vis_img, anchor_xy, args.patch_size)
        band_patches = []
        for band in input_bands:
            idx = RUBIN_BAND_ORDER.index(band.split('_', 1)[1])
            if idx >= rubin_cube.shape[0]:
                continue
            rubin_img = np.nan_to_num(_to_float32(rubin_cube[idx]), nan=0.0)
            band_patches.append(reproject_rubin_patch_to_vis(rubin_img, rwcs, vwcs, anchor_xy, args.patch_size))
        if len(band_patches) != len(input_bands):
            continue
        rubin_patches.append(np.stack(band_patches, axis=0))
        vis_patches.append(vis_patch[None])
        pix2sky_list.append(local_vis_pixel_to_sky_matrix(vwcs, anchor_xy))
        kept_xy.append(anchor_xy)
        kept_raw.append(raw_offsets[idx_anchor])

    if not rubin_patches:
        return None

    rubin_t = torch.from_numpy(np.stack(rubin_patches, axis=0)).float().to(device)
    vis_t = torch.from_numpy(np.stack(vis_patches, axis=0)).float().to(device)
    pix2sky_t = torch.from_numpy(np.stack(pix2sky_list, axis=0)).float().to(device)

    preds = []
    sigmas = []
    confs = []
    for i in range(0, rubin_t.shape[0], int(args.batch_size)):
        out = model(rubin_t[i:i + args.batch_size], vis_t[i:i + args.batch_size], pix2sky_t[i:i + args.batch_size])
        preds.append(out['pred_offset_arcsec'].cpu().numpy())
        sigmas.append(torch.exp(out['log_sigma']).cpu().numpy())
        confs.append(out['confidence'].cpu().numpy())
    pred_offsets = np.concatenate(preds, axis=0).astype(np.float32)
    sigma = np.concatenate(sigmas, axis=0).astype(np.float32)
    conf = np.concatenate(confs, axis=0).astype(np.float32)
    kept_xy = np.asarray(kept_xy, dtype=np.float32)
    kept_raw = np.asarray(kept_raw, dtype=np.float32)
    weights = 1.0 / np.maximum(sigma, 1e-4) ** 2

    # Auto grid shape: reduce resolution for sparse tiles.
    grid_h, grid_w = int(args.grid_h), int(args.grid_w)
    if getattr(args, 'auto_grid', False):
        grid_h, grid_w = auto_grid_shape(
            n_anchors=len(kept_xy),
            default=(grid_h, grid_w),
            min_shape=(4, 4),
        )

    # Compute adaptive anchor radius from grid cell spacing.
    anchor_radius = float(getattr(args, 'anchor_radius_px', 0.0))
    if anchor_radius <= 0 and getattr(args, 'auto_grid', False):
        # Default: ~2x the mean grid cell spacing.
        h, w = vis_img.shape
        cell_y = h / max(1, grid_h - 1)
        cell_x = w / max(1, grid_w - 1)
        anchor_radius = 2.0 * 0.5 * (cell_y + cell_x)

    field = solve_control_grid_field(
        vis_xy=kept_xy,
        offsets_arcsec=pred_offsets,
        weights=weights,
        vis_shape=vis_img.shape,
        grid_shape=(grid_h, grid_w),
        smooth_lambda=args.smooth_lambda,
        anchor_lambda=args.anchor_lambda,
        anchor_radius_px=anchor_radius,
    )
    mesh = evaluate_control_grid_mesh(
        field, vis_shape=vis_img.shape, dstep=args.dstep,
        anchor_xy=kept_xy,
    )

    raw_mag = np.hypot(kept_raw[:, 0], kept_raw[:, 1]) * 1000.0
    pred_mag = np.hypot(pred_offsets[:, 0], pred_offsets[:, 1]) * 1000.0
    return {
        'vis_image': vis_img,
        'vis_wcs_header': vhdr,
        'vis_shape': vis_img.shape,
        'vis_xy': kept_xy,
        'raw_offsets': kept_raw,
        'pred_offsets': pred_offsets,
        'sigma_arcsec': sigma,
        'confidence': conf,
        'field': field,
        'mesh': mesh,
        'grid_shape_used': (grid_h, grid_w),
        'anchor_radius_used': anchor_radius,
        'summary': {
            'matches': int(pred_offsets.shape[0]),
            'raw_median_mas': float(np.median(raw_mag)),
            'pred_median_mas': float(np.median(pred_mag)),
            'sigma_median_mas': float(np.median(sigma) * 1000.0),
            'grid_shape': [grid_h, grid_w],
            'anchor_radius_px': anchor_radius,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run the standalone local matcher and export smooth concordance FITS fields.')
    p.add_argument('--rubin-dir', type=str, required=True)
    p.add_argument('--euclid-dir', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--output', type=str, required=True)
    p.add_argument('--summary-json', type=str, default='')
    p.add_argument('--tile-id', type=str, default='')
    p.add_argument('--plot-dir', type=str, default='',
                   help='Optional directory to save one diagnostic PNG per exported tile.')
    p.add_argument('--input-bands', type=str, nargs='+', default=[], help='Optional Rubin input bands. Defaults to checkpoint config.')
    p.add_argument('--detect-bands', type=str, nargs='+', default=['g', 'r', 'i', 'z'])

    p.add_argument('--patch-size', type=int, default=33)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--min-matches', type=int, default=20)
    p.add_argument('--max-matches', type=int, default=256)
    p.add_argument('--max-sep-arcsec', type=float, default=0.12)
    p.add_argument('--clip-sigma', type=float, default=3.5)
    p.add_argument('--rubin-nsig', type=float, default=4.5)
    p.add_argument('--vis-nsig', type=float, default=4.0)
    p.add_argument('--rubin-smooth', type=float, default=1.0)
    p.add_argument('--vis-smooth', type=float, default=1.2)
    p.add_argument('--rubin-min-dist', type=int, default=7)
    p.add_argument('--vis-min-dist', type=int, default=9)
    p.add_argument('--max-sources-rubin', type=int, default=600)
    p.add_argument('--max-sources-vis', type=int, default=800)
    p.add_argument('--detect-clip-sigma', type=float, default=8.0)
    p.add_argument('--refine-radius', type=int, default=3)
    p.add_argument('--refine-flux-floor-sigma', type=float, default=1.5)

    p.add_argument('--grid-h', type=int, default=12)
    p.add_argument('--grid-w', type=int, default=12)
    p.add_argument('--auto-grid', action='store_true', default=False,
                   help='Automatically reduce grid resolution for tiles with few matches.')
    p.add_argument('--smooth-lambda', type=float, default=1e-2)
    p.add_argument('--anchor-lambda', type=float, default=1e-3,
                   help='Ridge regularisation base strength (raised from 1e-4).')
    p.add_argument('--anchor-radius-px', type=float, default=0.0,
                   help='Gaussian scale for adaptive per-node anchor. 0 = auto if --auto-grid.')
    p.add_argument('--dstep', type=int, default=8)
    p.add_argument('--device', type=str, default='')
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, ckpt = load_model(args.checkpoint, device)

    target_band = normalize_rubin_band(ckpt.get('target_band', ckpt.get('args', {}).get('rubin_band', 'r')))
    input_bands = normalize_rubin_bands(args.input_bands) or [normalize_rubin_band(x) for x in ckpt.get('input_bands', [target_band])]
    detect_bands = normalize_rubin_bands(args.detect_bands) or [f'rubin_{b}' for b in ('g', 'r', 'i', 'z')]

    pairs = discover_tile_pairs(args.rubin_dir, args.euclid_dir)
    results = []
    hdus = [fits.PrimaryHDU()]
    hdus[0].header['CONCRDNC'] = (True, 'Standalone DL astrometry concordance product')
    hdus[0].header['DSTEP'] = (int(args.dstep), 'Mesh sampling step in VIS pixels')
    hdus[0].header['DUNIT'] = ('arcsec', 'Offset unit')
    hdus[0].header['REFFRAME'] = ('euclid_VIS', 'Reference astrometric frame')
    hdus[0].header['FITMETH'] = ('dl_local+grid', 'Patch matcher + control grid')
    hdus[0].header['AUTOGRID'] = (bool(args.auto_grid), 'Auto grid shape enabled')
    hdus[0].header['ANCHLAM'] = (float(args.anchor_lambda), 'Anchor lambda base')
    hdus[0].header['ANCHRAD'] = (float(args.anchor_radius_px), 'Anchor radius (0=auto)')

    for tile_id, rubin_path, euclid_path in pairs:
        if args.tile_id and tile_id != args.tile_id:
            continue
        item = predict_tile(model, device, rubin_path, euclid_path, target_band, input_bands, detect_bands, args)
        if item is None:
            continue
        band_key = target_band.split('_', 1)[1]
        prefix = f'{tile_id}.{band_key}'
        hdus.append(make_concordance_hdu(item['mesh']['dra'], f'{prefix}.DRA', args.dstep, target_band, tile_id, item['vis_wcs_header']))
        hdus.append(make_concordance_hdu(item['mesh']['ddec'], f'{prefix}.DDE', args.dstep, target_band, tile_id, item['vis_wcs_header']))
        # Coverage HDU.
        if 'coverage' in item['mesh']:
            hdus.append(make_coverage_hdu(item['mesh']['coverage'], f'{prefix}.COV', args.dstep, tile_id))

        row = {'tile_id': tile_id, 'rubin_band': target_band}
        row.update(item['summary'])
        results.append(row)
        gs = item.get('grid_shape_used', (args.grid_h, args.grid_w))
        ar = item.get('anchor_radius_used', 0.0)
        print(
            f"[tile] {tile_id}:{target_band} matches={row['matches']} "
            f"grid={gs[0]}x{gs[1]} anchor_r={ar:.0f}px "
            f"raw_med={row['raw_median_mas']:.1f}mas "
            f"pred_med={row['pred_median_mas']:.1f}mas sigma_med={row['sigma_median_mas']:.1f}mas"
        )
        if args.plot_dir:
            plot_path = os.path.join(args.plot_dir, f'{prefix}.png')
            save_tile_diagnostic(item, tile_id, target_band, input_bands, plot_path)

    if len(hdus) == 1:
        raise RuntimeError('No tiles were exported.')
    fits.HDUList(hdus).writeto(args.output, overwrite=True)
    print(f'Wrote {args.output}')

    if args.summary_json:
        with open(args.summary_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Wrote summary: {args.summary_json}')


if __name__ == '__main__':
    main()
