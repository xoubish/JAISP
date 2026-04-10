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
    for p in (models_dir, script_dir):
        sp = str(p)
        if sp in sys.path:
            sys.path.remove(sp)
        sys.path.insert(0, sp)

_setup_imports()

from astrometry2.dataset import (
    BAND_TO_IDX,
    NISP_BAND_ORDER,
    detect_sources_multiband,
    discover_tile_pairs,
    extract_vis_patch,
    local_vis_pixel_to_sky_matrix,
    normalize_nisp_band,
    normalize_rubin_band,
    normalize_rubin_bands,
    project_vis_to_band_xy,
    reproject_nisp_patch_to_vis,
    reproject_rubin_patch_to_vis,
    signal_mask_in_band,
)
from astrometry2.field_solver import auto_grid_shape, evaluate_control_grid_mesh, solve_control_grid_field
from older_architectures.matcher import LocalAstrometryMatcher
try:
    from astrometry2.matcher_v6 import load_v6_matcher
    _V6_AVAILABLE = True
except ImportError:
    _V6_AVAILABLE = False
try:
    from astrometry2.matcher_v7 import load_v7_matcher
    _V7_AVAILABLE = True
except ImportError:
    _V7_AVAILABLE = False
from astrometry2.viz import save_tile_diagnostic
from astrometry2.source_matching import (
    RUBIN_BAND_ORDER,
    _to_float32,
    build_detection_image,
    detect_sources,
    match_sources_wcs,
    refine_centroids_in_band,
    safe_header_from_card_string,
)


def load_model(checkpoint_path: str, device: torch.device,
               v6_checkpoint: str = '', v7_checkpoint: str = ''):
    """Load a matcher checkpoint.

    If v7_checkpoint is provided, loads a V7AstrometryMatcher.
    Elif v6_checkpoint is provided, loads a V6AstrometryMatcher.
    Otherwise loads a baseline LocalAstrometryMatcher.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if v7_checkpoint:
        if not _V7_AVAILABLE:
            raise ImportError('matcher_v7.py not found — cannot load v7 model.')
        cfg = ckpt.get('args', {})
        n_rubin = int(ckpt.get('rubin_channels', len(ckpt.get('input_bands', ['rubin_r']))))
        model = load_v7_matcher(
            v7_checkpoint    = v7_checkpoint,
            device           = device,
            n_rubin_bands    = n_rubin,
            hidden_channels  = cfg.get('hidden_channels', 64),
            n_adapter_blocks = cfg.get('adapter_blocks', 2),
            freeze_stems     = True,
            search_radius    = cfg.get('search_radius', 3),
            n_target_bands   = int(ckpt.get('n_target_bands', 1)),
            band_embed_dim   = int(cfg.get('band_embed_dim', 16)),
            mlp_hidden       = cfg.get('mlp_hidden', 128),
            n_stream_stages  = int(cfg.get('stream_stages', 0)),
        )
        model.load_state_dict(ckpt['model'], strict=True)
    elif v6_checkpoint:
        if not _V6_AVAILABLE:
            raise ImportError('matcher_v6.py not found — cannot load v6 model.')
        cfg = ckpt.get('args', {})
        n_rubin = int(ckpt.get('rubin_channels', len(ckpt.get('input_bands', ['rubin_r']))))
        model = load_v6_matcher(
            v6_checkpoint   = v6_checkpoint,
            device          = device,
            n_rubin_bands   = n_rubin,
            hidden_channels = cfg.get('hidden_channels', 64),
            n_adapter_blocks= cfg.get('adapter_blocks', 2),
            freeze_stems    = True,
            search_radius   = cfg.get('search_radius', 3),
            n_target_bands  = int(ckpt.get('n_target_bands', 1)),
            band_embed_dim  = int(cfg.get('band_embed_dim', 16)),
            mlp_hidden      = cfg.get('mlp_hidden', 128),
        )
        model.load_state_dict(ckpt['model'], strict=True)
    else:
        args = ckpt.get('args', {})
        model = LocalAstrometryMatcher(
            rubin_channels=int(ckpt.get('rubin_channels', len(ckpt.get('input_bands', ['rubin_r'])))),
            hidden_channels=args.get('hidden_channels', 32),
            encoder_depth=args.get('encoder_depth', 4),
            search_radius=args.get('search_radius', 3),
            softmax_temp=args.get('softmax_temp', 0.05),
            mlp_hidden=args.get('mlp_hidden', 128),
            n_target_bands=int(ckpt.get('n_target_bands', 1)),
            band_embed_dim=int(args.get('band_embed_dim', 16)),
        ).to(device)
        model.load_state_dict(ckpt['model'], strict=True)

    model.eval()
    return model, ckpt


def _normalize_any_band(name: str) -> str:
    s = str(name).strip()
    if not s:
        raise ValueError('Empty band name.')
    if s.lower().startswith('nisp_'):
        return normalize_nisp_band(s)
    return normalize_rubin_band(s)


def _load_nisp_data(edata) -> Dict[str, tuple[np.ndarray, WCS]]:
    nisp_data: Dict[str, tuple[np.ndarray, WCS]] = {}
    for nb in NISP_BAND_ORDER:
        img_key = f'img_{nb}'
        wcs_key = f'wcs_{nb}'
        if img_key not in edata or wcs_key not in edata:
            continue
        try:
            img = np.nan_to_num(_to_float32(edata[img_key]), nan=0.0)
            wcs = WCS(safe_header_from_card_string(edata[wcs_key].item()))
            nisp_data[nb] = (img, wcs)
        except Exception as exc:
            print(f'[infer] Failed to load NISP band {nb}: {exc}')
            continue
    return nisp_data


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
    try:
        target_band = _normalize_any_band(target_band)
    except Exception as exc:
        print(f'[skip] {os.path.basename(rubin_path)}: invalid target band "{target_band}" ({exc})')
        return None

    input_bands_norm: List[str] = []
    for b in input_bands:
        try:
            nb = _normalize_any_band(b)
        except Exception as exc:
            print(f'[skip] {os.path.basename(rubin_path)}: invalid input band "{b}" ({exc})')
            return None
        if nb not in input_bands_norm:
            input_bands_norm.append(nb)
    if not input_bands_norm:
        return None

    # Validate input channel count matches the model before touching any data.
    try:
        model_in_ch = model.rubin_encoder.net[0].in_channels
        if len(input_bands_norm) != model_in_ch:
            print(
                f'[skip] {os.path.basename(rubin_path)}: input band count mismatch — '
                f'model expects {model_in_ch} channels but got {len(input_bands_norm)} '
                f'({input_bands_norm}). Use --include-nisp or adjust --input-bands.'
            )
            return None
    except AttributeError:
        pass  # Non-standard model; skip the check.

    try:
        rdata = np.load(rubin_path, allow_pickle=True)
        edata = np.load(euclid_path, allow_pickle=True)
        rubin_cube = rdata['img']
        rubin_var = rdata['var'] if 'var' in rdata else None
        vis_img = np.nan_to_num(_to_float32(edata['img_VIS']), nan=0.0)
        rwcs = WCS(rdata['wcs_hdr'].item())
        vhdr = safe_header_from_card_string(edata['wcs_VIS'].item())
        vwcs = WCS(vhdr)
    except Exception as exc:
        print(f'[skip] {os.path.basename(rubin_path)}: load/wcs failed ({exc})')
        return None

    need_nisp = target_band.startswith('nisp_') or any(b.startswith('nisp_') for b in input_bands_norm)
    nisp_data = _load_nisp_data(edata) if need_nisp else {}

    # Source detection: either classical Rubin<->VIS matching, or 10-band
    # neural anchors in the VIS frame.
    _detr = getattr(args, '_detr_detector', None)
    raw_anchor_xy = None
    vis_anchor_xy = None
    if _detr is not None:
        ax, ay = detect_sources_multiband(
            edata,
            rubin_cube,
            rubin_var,
            _detr,
            device,
            conf_threshold=getattr(args, 'detector_conf_threshold', 0.3),
        )
        if ax.size == 0:
            return None
        vis_seed_xy = np.stack([ax, ay], axis=1).astype(np.float32)
        raw_anchor_xy = vis_seed_xy.copy()
        vis_keep = signal_mask_in_band(
            vis_img,
            vis_seed_xy,
            radius=args.refine_radius,
            flux_floor_sigma=args.refine_flux_floor_sigma,
        )
        if not vis_keep.any():
            return None
        vis_xy = refine_centroids_in_band(
            vis_img,
            vis_seed_xy[vis_keep],
            radius=args.refine_radius,
            flux_floor_sigma=args.refine_flux_floor_sigma,
        ).astype(np.float32)
        vis_anchor_xy = vis_xy.copy()
    else:
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
        vis_xy = matched['vis_xy'].astype(np.float32)
        raw_anchor_xy = vis_xy.copy()
        vis_anchor_xy = vis_xy.copy()
    if target_band.startswith('rubin_'):
        target_idx = RUBIN_BAND_ORDER.index(target_band.split('_', 1)[1])
        rubin_target = np.nan_to_num(_to_float32(rubin_cube[target_idx]), nan=0.0)
        rubin_refine_radius = max(1, int(args.refine_radius) // 3)
        if _detr is not None:
            rubin_xy_seed = project_vis_to_band_xy(vis_xy, vwcs, rwcs)
            target_keep = signal_mask_in_band(
                rubin_target,
                rubin_xy_seed,
                radius=rubin_refine_radius,
                flux_floor_sigma=args.refine_flux_floor_sigma,
            )
            if int(target_keep.sum()) < int(args.min_matches):
                return None
            vis_xy = vis_xy[target_keep]
            rubin_xy_seed = rubin_xy_seed[target_keep]
        else:
            rubin_xy_seed = matched['rubin_xy']
        rubin_xy_target = refine_centroids_in_band(
            rubin_target,
            rubin_xy_seed,
            radius=rubin_refine_radius,
            flux_floor_sigma=args.refine_flux_floor_sigma,
        )
        r_ra, r_dec = rwcs.wcs_pix2world(rubin_xy_target[:, 0], rubin_xy_target[:, 1], 0)
        v_ra, v_dec = vwcs.wcs_pix2world(vis_xy[:, 0], vis_xy[:, 1], 0)
        raw_dra = (v_ra - r_ra) * np.cos(np.deg2rad(v_dec)) * 3600.0
        raw_ddec = (v_dec - r_dec) * 3600.0
        raw_offsets = np.stack([raw_dra, raw_ddec], axis=1).astype(np.float32)
    else:
        nb = target_band.split('_', 1)[1]
        if nb not in nisp_data:
            return None
        nisp_img, nwcs = nisp_data[nb]
        nisp_xy_init = project_vis_to_band_xy(vis_xy, vwcs, nwcs)
        nisp_radius = int(args.refine_radius)  # MER mosaics: NISP at 0.1"/px, same as VIS
        target_keep = signal_mask_in_band(
            nisp_img,
            nisp_xy_init,
            radius=nisp_radius,
            flux_floor_sigma=args.refine_flux_floor_sigma,
        )
        if int(target_keep.sum()) < int(args.min_matches):
            return None
        vis_xy = vis_xy[target_keep]
        nisp_xy_init = nisp_xy_init[target_keep]
        nisp_xy_refined = refine_centroids_in_band(
            nisp_img,
            nisp_xy_init,
            radius=nisp_radius,
            flux_floor_sigma=args.refine_flux_floor_sigma,
        )
        n_ra, n_dec = nwcs.wcs_pix2world(nisp_xy_refined[:, 0], nisp_xy_refined[:, 1], 0)
        v_ra, v_dec = vwcs.wcs_pix2world(vis_xy[:, 0], vis_xy[:, 1], 0)
        raw_dra = (v_ra - n_ra) * np.cos(np.deg2rad(v_dec)) * 3600.0
        raw_ddec = (v_dec - n_dec) * 3600.0
        raw_offsets = np.stack([raw_dra, raw_ddec], axis=1).astype(np.float32)

    # Drop large raw offsets (likely mismatches) before inference.
    max_sep = float(args.max_sep_arcsec)
    if max_sep > 0:
        raw_mag = np.hypot(raw_offsets[:, 0], raw_offsets[:, 1])
        keep = raw_mag <= max_sep
        if int(keep.sum()) < int(args.min_matches):
            return None
        vis_xy = vis_xy[keep]
        raw_offsets = raw_offsets[keep]
        vis_anchor_xy = vis_xy.copy()

    rubin_patches = []
    vis_patches = []
    pix2sky_list = []
    kept_xy = []
    kept_raw = []
    for idx_anchor, anchor_xy in enumerate(vis_xy):
        vis_patch = extract_vis_patch(vis_img, anchor_xy, args.patch_size)
        band_patches = []
        for band in input_bands_norm:
            if band.startswith('rubin_'):
                idx = RUBIN_BAND_ORDER.index(band.split('_', 1)[1])
                if idx < rubin_cube.shape[0]:
                    rubin_img = np.nan_to_num(_to_float32(rubin_cube[idx]), nan=0.0)
                    band_patches.append(reproject_rubin_patch_to_vis(rubin_img, rwcs, vwcs, anchor_xy, args.patch_size))
                else:
                    band_patches.append(np.zeros((args.patch_size, args.patch_size), dtype=np.float32))
            else:
                nb = band.split('_', 1)[1]
                if nb in nisp_data:
                    nisp_img, nwcs = nisp_data[nb]
                    band_patches.append(reproject_nisp_patch_to_vis(nisp_img, nwcs, vwcs, anchor_xy, args.patch_size))
                else:
                    band_patches.append(np.zeros((args.patch_size, args.patch_size), dtype=np.float32))
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
    band_idx_value = BAND_TO_IDX.get(target_band)
    model_n_target_bands = int(getattr(model, 'n_target_bands', 1))
    use_band_idx = model_n_target_bands > 1
    if use_band_idx and (band_idx_value is None or int(band_idx_value) >= model_n_target_bands):
        print(
            f'[skip] {os.path.basename(rubin_path)}: target band {target_band} not available in model '
            f'(band_idx={band_idx_value}, n_target_bands={model_n_target_bands})'
        )
        return None
    for i in range(0, rubin_t.shape[0], int(args.batch_size)):
        batch_rubin = rubin_t[i:i + args.batch_size]
        batch_vis = vis_t[i:i + args.batch_size]
        batch_pix2sky = pix2sky_t[i:i + args.batch_size]
        if use_band_idx:
            batch_band_idx = torch.full(
                (batch_rubin.shape[0],),
                int(band_idx_value),
                dtype=torch.long,
                device=device,
            )
            out = model(batch_rubin, batch_vis, batch_pix2sky, band_idx=batch_band_idx)
        else:
            out = model(batch_rubin, batch_vis, batch_pix2sky)
        preds.append(out['pred_offset_arcsec'].cpu().numpy())
        sigmas.append(torch.exp(out['log_sigma']).cpu().numpy())
        confs.append(out['confidence'].cpu().numpy())
    pred_offsets = np.concatenate(preds, axis=0).astype(np.float32)
    sigma = np.concatenate(sigmas, axis=0).astype(np.float32)
    conf = np.concatenate(confs, axis=0).astype(np.float32)
    kept_xy = np.asarray(kept_xy, dtype=np.float32)
    kept_raw = np.asarray(kept_raw, dtype=np.float32)
    if kept_xy.shape[0] < 4:
        return None
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
    if raw_anchor_xy is None:
        raw_anchor_xy = kept_xy.copy()
    if vis_anchor_xy is None:
        vis_anchor_xy = kept_xy.copy()

    return {
        'vis_image': vis_img,
        'vis_wcs_header': vhdr,
        'vis_shape': vis_img.shape,
        'raw_anchor_xy': raw_anchor_xy,
        'vis_anchor_xy': vis_anchor_xy,
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
            'raw_anchor_count': int(len(raw_anchor_xy)),
            'vis_anchor_count': int(len(vis_anchor_xy)),
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
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to astrometry matcher checkpoint_best.pt')
    p.add_argument('--v6-checkpoint', type=str, default='',
                   help='Path to jaisp_foundation_v6 Phase B checkpoint. '
                        'Required when the matcher checkpoint is a V6AstrometryMatcher.')
    p.add_argument('--v7-checkpoint', type=str, default='',
                   help='Path to jaisp_v7_baseline checkpoint. '
                        'Required when the matcher checkpoint is a V7AstrometryMatcher.')
    p.add_argument('--detector-checkpoint', type=str, default='',
                   help='Path to centernet_best.pt for neural source detection '
                        '(omit for classical peak-finding).')
    p.add_argument('--detector-conf-threshold', type=float, default=0.3,
                   help='CenterNet confidence threshold (default: 0.3).')
    p.add_argument('--all-bands', action='store_true', default=False,
                   help='Export concordance for every target band in a multiband checkpoint '
                        'in one FITS file (one DRA/DDE/COV triplet per band per tile).')
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
    v6_ckpt = getattr(args, 'v6_checkpoint', '') or ''
    v7_ckpt = getattr(args, 'v7_checkpoint', '') or ''
    model, ckpt = load_model(args.checkpoint, device,
                             v6_checkpoint=v6_ckpt, v7_checkpoint=v7_ckpt)

    # Optional neural detector for Rubin source detection
    detr_ckpt = getattr(args, 'detector_checkpoint', '') or ''
    if detr_ckpt and v7_ckpt:
        from train_astro_v7 import _load_detector
        args._detr_detector = _load_detector(detr_ckpt, v7_ckpt, device)
    else:
        args._detr_detector = None

    # Determine which target bands to export
    target_band_raw = ckpt.get('target_band', ckpt.get('args', {}).get('rubin_band', 'r'))
    is_multiband = str(target_band_raw) == 'multiband'

    ckpt_target_bands_raw = []
    if is_multiband:
        ckpt_target_bands_raw = [str(b) for b in ckpt.get('target_bands', []) if str(b).strip()]
        if ckpt_target_bands_raw:
            ckpt_target_bands_raw = sorted(
                ckpt_target_bands_raw,
                key=lambda b: (
                    0 if b.startswith('rubin_') else 1,
                    BAND_TO_IDX.get(b, 10_000),
                ),
            )

    if getattr(args, 'all_bands', False) and ckpt_target_bands_raw:
        target_bands_to_export = [_normalize_any_band(b) for b in ckpt_target_bands_raw]
        print(f'Exporting concordance for all {len(target_bands_to_export)} bands: {target_bands_to_export}')
    else:
        # Single band: use first from multiband ckpt or the stored target band
        single = ckpt_target_bands_raw[0] if ckpt_target_bands_raw else str(target_band_raw)
        if not single or single == 'multiband':
            single = 'rubin_r'
        target_bands_to_export = [_normalize_any_band(single)]

    target_band = target_bands_to_export[0]  # kept for legacy single-band path below

    if args.input_bands:
        input_bands_raw = [str(x) for x in args.input_bands]
    else:
        input_bands_raw = [str(x) for x in ckpt.get('input_bands', [target_band])]
    input_bands = []
    for b in input_bands_raw:
        nb = _normalize_any_band(b)
        if nb not in input_bands:
            input_bands.append(nb)
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
        for tband in target_bands_to_export:
            item = predict_tile(model, device, rubin_path, euclid_path, tband, input_bands, detect_bands, args)
            if item is None:
                continue
            band_key = tband.split('_', 1)[1]
            prefix = f'{tile_id}.{band_key}'
            hdus.append(make_concordance_hdu(item['mesh']['dra'], f'{prefix}.DRA', args.dstep, tband, tile_id, item['vis_wcs_header']))
            hdus.append(make_concordance_hdu(item['mesh']['ddec'], f'{prefix}.DDE', args.dstep, tband, tile_id, item['vis_wcs_header']))
            if 'coverage' in item['mesh']:
                hdus.append(make_coverage_hdu(item['mesh']['coverage'], f'{prefix}.COV', args.dstep, tile_id))

            row = {'tile_id': tile_id, 'rubin_band': tband}
            row.update(item['summary'])
            results.append(row)
            gs = item.get('grid_shape_used', (args.grid_h, args.grid_w))
            ar = item.get('anchor_radius_used', 0.0)
            print(
                f"[tile] {tile_id}:{tband} matches={row['matches']} "
                f"grid={gs[0]}x{gs[1]} anchor_r={ar:.0f}px "
                f"raw_med={row['raw_median_mas']:.1f}mas "
                f"pred_med={row['pred_median_mas']:.1f}mas sigma_med={row['sigma_median_mas']:.1f}mas"
            )
            if args.plot_dir:
                plot_path = os.path.join(args.plot_dir, f'{prefix}.png')
                save_tile_diagnostic(item, tile_id, tband, input_bands, plot_path)

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
