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
import os
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
from astrometry2.infer_concordance import (
    _normalize_any_band,
    build_parser as build_infer_parser,
    load_model,
    predict_tile,
)
from astrometry2.dataset import (
    BAND_TO_IDX,
    NISP_BAND_ORDER,
    discover_tile_pairs,
    normalize_rubin_bands,
    extract_vis_patch,
    local_vis_pixel_to_sky_matrix,
    reproject_rubin_patch_to_vis,
    reproject_nisp_patch_to_vis,
    signal_mask_in_band,
    project_vis_to_band_xy,
    _normalize_patch,
)
from astrometry2.source_matching import (
    RUBIN_BAND_ORDER,
    _to_float32,
    build_detection_image,
    detect_sources,
    match_sources_wcs,
    refine_centroids_in_band,
    safe_header_from_card_string,
)
from astrometry2.field_solver import auto_grid_shape, evaluate_control_grid_mesh, solve_control_grid_field


# ============================================================
# Global collection (legacy single-band, kept for compatibility)
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
    """Single-band collection: runs predict_tile per tile. Kept for single-band use."""
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

        vis_xy       = item['vis_xy']
        pred_offsets = item['pred_offsets']
        raw_offsets  = item['raw_offsets']
        sigma        = item['sigma_arcsec']

        try:
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
# Multi-band collection: detect once, predict all bands
# ============================================================

def _load_nisp_data_local(edata) -> dict:
    nisp_data = {}
    for nb in NISP_BAND_ORDER:
        img_key, wcs_key = f'img_{nb}', f'wcs_{nb}'
        if img_key not in edata or wcs_key not in edata:
            continue
        try:
            img = np.nan_to_num(_to_float32(edata[img_key]), nan=0.0)
            wcs = WCS(safe_header_from_card_string(edata[wcs_key].item()))
            nisp_data[nb] = (img, wcs)
        except Exception:
            continue
    return nisp_data


@torch.no_grad()
def collect_all_predictions_multiband(
    model,
    device: torch.device,
    pairs: list,
    target_bands: list,
    input_bands: list,
    detect_bands: list,
    args,
) -> dict:
    """Detect once per tile, predict all target bands from cached features.

    Returns dict: target_band -> {ra, dec, pred_offsets, raw_offsets, sigma, tile_ids}
    """
    from astrometry2.dataset import detect_sources_multiband

    batch_size = int(getattr(args, 'batch_size', 128))
    patch_size = int(getattr(args, 'patch_size', 33))
    model_n_target_bands = int(getattr(model, 'n_target_bands', 1))
    use_band_idx = model_n_target_bands > 1

    # Pre-validate band indices
    band_idx_map = {}
    for tb in target_bands:
        bidx = BAND_TO_IDX.get(tb)
        if use_band_idx and (bidx is None or bidx >= model_n_target_bands):
            print(f'[warn] Skipping target band {tb}: band_idx={bidx} out of range')
            continue
        band_idx_map[tb] = bidx

    # Accumulators per band
    results = {tb: {'ra': [], 'dec': [], 'pred': [], 'raw': [], 'sigma': [], 'tiles': []}
               for tb in band_idx_map}

    _detr = getattr(args, '_detr_detector', None)

    for ti, (tile_id, rubin_path, euclid_path) in enumerate(pairs):
        try:
            rdata = np.load(rubin_path, allow_pickle=True)
            edata = np.load(euclid_path, allow_pickle=True)
            rubin_cube = rdata['img']
            vis_img = np.nan_to_num(_to_float32(edata['img_VIS']), nan=0.0)
            rwcs = WCS(rdata['wcs_hdr'].item())
            vhdr = safe_header_from_card_string(edata['wcs_VIS'].item())
            vwcs = WCS(vhdr)
        except Exception as exc:
            continue

        nisp_data = _load_nisp_data_local(edata)

        # ---- DETECT ONCE ----
        if _detr is not None:
            ax, ay = detect_sources_multiband(
                edata, rubin_cube, _detr, device,
                conf_threshold=getattr(args, 'detector_conf_threshold', 0.3),
            )
            if ax.size == 0:
                continue
            vis_seed_xy = np.stack([ax, ay], axis=1).astype(np.float32)
            vis_keep = signal_mask_in_band(
                vis_img, vis_seed_xy,
                radius=getattr(args, 'refine_radius', 3),
                flux_floor_sigma=getattr(args, 'refine_flux_floor_sigma', 1.5),
            )
            if not vis_keep.any():
                continue
            vis_xy_base = refine_centroids_in_band(
                vis_img, vis_seed_xy[vis_keep],
                radius=getattr(args, 'refine_radius', 3),
                flux_floor_sigma=getattr(args, 'refine_flux_floor_sigma', 1.5),
            ).astype(np.float32)
        else:
            detect_bands_norm = [f'rubin_{b}' if not b.startswith('rubin_') else b for b in detect_bands]
            rubin_det = build_detection_image(rubin_cube, detect_bands_norm,
                                              clip_sigma=getattr(args, 'detect_clip_sigma', 8.0))
            rx, ry = detect_sources(rubin_det,
                                    nsig=getattr(args, 'rubin_nsig', 4.5),
                                    smooth_sigma=getattr(args, 'rubin_smooth', 1.0),
                                    min_dist=getattr(args, 'rubin_min_dist', 7),
                                    max_sources=getattr(args, 'max_sources_rubin', 600))
            vx, vy = detect_sources(vis_img,
                                    nsig=getattr(args, 'vis_nsig', 4.0),
                                    smooth_sigma=getattr(args, 'vis_smooth', 1.2),
                                    min_dist=getattr(args, 'vis_min_dist', 9),
                                    max_sources=getattr(args, 'max_sources_vis', 800))
            matched = match_sources_wcs(
                rx, ry, vx, vy, rwcs, vwcs,
                max_sep_arcsec=getattr(args, 'max_sep_arcsec', 0.12),
                clip_sigma=getattr(args, 'clip_sigma', 3.5),
                max_matches=getattr(args, 'max_matches', 256),
            )
            if matched['vis_xy'].shape[0] < int(getattr(args, 'min_matches', 20)):
                continue
            vis_xy_base = matched['vis_xy'].astype(np.float32)

        # ---- EXTRACT PATCHES ONCE (shared across all bands) ----
        input_bands_norm = []
        for b in input_bands:
            nb = _normalize_any_band(b)
            if nb not in input_bands_norm:
                input_bands_norm.append(nb)

        all_rubin_patches = []
        all_vis_patches = []
        all_pix2sky = []
        for anchor_xy in vis_xy_base:
            vis_patch = extract_vis_patch(vis_img, anchor_xy, patch_size)
            band_patches = []
            for band in input_bands_norm:
                if band.startswith('rubin_'):
                    idx = RUBIN_BAND_ORDER.index(band.split('_', 1)[1])
                    if idx < rubin_cube.shape[0]:
                        rubin_img = np.nan_to_num(_to_float32(rubin_cube[idx]), nan=0.0)
                        band_patches.append(reproject_rubin_patch_to_vis(rubin_img, rwcs, vwcs, anchor_xy, patch_size))
                    else:
                        band_patches.append(np.zeros((patch_size, patch_size), dtype=np.float32))
                else:
                    nb = band.split('_', 1)[1]
                    if nb in nisp_data:
                        nisp_img, nwcs = nisp_data[nb]
                        band_patches.append(reproject_nisp_patch_to_vis(nisp_img, nwcs, vwcs, anchor_xy, patch_size))
                    else:
                        band_patches.append(np.zeros((patch_size, patch_size), dtype=np.float32))
            all_rubin_patches.append(np.stack(band_patches, axis=0))
            all_vis_patches.append(vis_patch[None])
            all_pix2sky.append(local_vis_pixel_to_sky_matrix(vwcs, anchor_xy))

        if not all_rubin_patches:
            continue

        # Normalize patches
        rubin_arr = np.stack(all_rubin_patches, axis=0)  # [N, C, H, W]
        vis_arr = np.stack(all_vis_patches, axis=0)       # [N, 1, H, W]
        for i in range(rubin_arr.shape[0]):
            for c in range(rubin_arr.shape[1]):
                rubin_arr[i, c] = _normalize_patch(rubin_arr[i, c])
            vis_arr[i, 0] = _normalize_patch(vis_arr[i, 0])

        rubin_t = torch.from_numpy(rubin_arr).float().to(device)
        vis_t = torch.from_numpy(vis_arr).float().to(device)
        pix2sky_t = torch.from_numpy(np.stack(all_pix2sky, axis=0)).float().to(device)

        # ---- ENCODE ONCE ----
        # Run encoder in batches, cache the encoded features
        all_enc = {}
        for i in range(0, rubin_t.shape[0], batch_size):
            br = rubin_t[i:i+batch_size]
            bv = vis_t[i:i+batch_size]
            enc = model._encode(br, bv)
            for key, val in enc.items():
                if key not in all_enc:
                    all_enc[key] = []
                all_enc[key].append(val.cpu() if isinstance(val, torch.Tensor) else val)
        # Concatenate
        for key in all_enc:
            if isinstance(all_enc[key][0], torch.Tensor):
                all_enc[key] = torch.cat(all_enc[key], dim=0).to(device)

        # VIS sky coordinates (shared)
        vis_ra, vis_dec = vwcs.wcs_pix2world(vis_xy_base[:, 0], vis_xy_base[:, 1], 0)

        # ---- PER-BAND: only MLP head + raw offset computation ----
        refine_radius = int(getattr(args, 'refine_radius', 3))
        refine_ffs = float(getattr(args, 'refine_flux_floor_sigma', 1.5))
        min_matches = int(getattr(args, 'min_matches', 20))

        rubin_refine_radius = max(1, refine_radius // 3)

        for target_band, bidx in band_idx_map.items():
            # Compute raw offsets for this target band
            if target_band.startswith('rubin_'):
                tidx = RUBIN_BAND_ORDER.index(target_band.split('_', 1)[1])
                if tidx >= rubin_cube.shape[0]:
                    continue
                target_img = np.nan_to_num(_to_float32(rubin_cube[tidx]), nan=0.0)
                target_xy_seed = project_vis_to_band_xy(vis_xy_base, vwcs, rwcs)
                target_keep = signal_mask_in_band(target_img, target_xy_seed,
                                                   radius=rubin_refine_radius, flux_floor_sigma=refine_ffs)
                if int(target_keep.sum()) < min_matches:
                    continue
                vis_xy = vis_xy_base[target_keep]
                target_xy = refine_centroids_in_band(target_img, target_xy_seed[target_keep],
                                                      radius=rubin_refine_radius, flux_floor_sigma=refine_ffs)
                t_ra, t_dec = rwcs.wcs_pix2world(target_xy[:, 0], target_xy[:, 1], 0)
                v_ra, v_dec = vwcs.wcs_pix2world(vis_xy[:, 0], vis_xy[:, 1], 0)
            elif target_band.startswith('nisp_'):
                nb = target_band.split('_', 1)[1]
                if nb not in nisp_data:
                    continue
                nisp_img, nwcs = nisp_data[nb]
                nisp_xy_init = project_vis_to_band_xy(vis_xy_base, vwcs, nwcs)
                nisp_radius = refine_radius  # MER mosaics: NISP at 0.1"/px, same as VIS
                target_keep = signal_mask_in_band(nisp_img, nisp_xy_init,
                                                   radius=nisp_radius, flux_floor_sigma=refine_ffs)
                if int(target_keep.sum()) < min_matches:
                    continue
                vis_xy = vis_xy_base[target_keep]
                nisp_xy = refine_centroids_in_band(nisp_img, nisp_xy_init[target_keep],
                                                    radius=nisp_radius, flux_floor_sigma=refine_ffs)
                t_ra, t_dec = nwcs.wcs_pix2world(nisp_xy[:, 0], nisp_xy[:, 1], 0)
                v_ra, v_dec = vwcs.wcs_pix2world(vis_xy[:, 0], vis_xy[:, 1], 0)
            else:
                continue

            raw_dra = (v_ra - t_ra) * np.cos(np.deg2rad(v_dec)) * 3600.0
            raw_ddec = (v_dec - t_dec) * 3600.0
            raw_offsets = np.stack([raw_dra, raw_ddec], axis=1).astype(np.float32)

            # Subset the cached encoder features to kept sources
            keep_idx = np.where(target_keep)[0]
            enc_sub = {k: v[keep_idx] if isinstance(v, torch.Tensor) else v for k, v in all_enc.items()}

            # Run MLP head with band embedding (cheap)
            N = len(keep_idx)
            preds, sigmas = [], []
            for i in range(0, N, batch_size):
                enc_batch = {k: v[i:i+batch_size].to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in enc_sub.items()}
                bp2s = pix2sky_t[keep_idx[i:i+batch_size]]
                if use_band_idx:
                    bb = torch.full((min(batch_size, N-i),), int(bidx), dtype=torch.long, device=device)
                else:
                    bb = None
                out = model._mlp_head(enc_batch, bp2s, bb, bp2s.shape[0], device)
                preds.append(out['pred_offset_arcsec'].cpu().numpy())
                sigmas.append(torch.exp(out['log_sigma']).cpu().numpy())

            pred_offsets = np.concatenate(preds, axis=0).astype(np.float32)
            sigma = np.concatenate(sigmas, axis=0).astype(np.float32)

            r = results[target_band]
            r['ra'].append(vis_ra[target_keep])
            r['dec'].append(vis_dec[target_keep])
            r['pred'].append(pred_offsets)
            r['raw'].append(raw_offsets)
            r['sigma'].append(sigma)
            r['tiles'].extend([tile_id] * len(keep_idx))

        # Free GPU memory for this tile
        del rubin_t, vis_t, pix2sky_t, all_enc
        torch.cuda.empty_cache()

        if (ti + 1) % 50 == 0 or ti == 0:
            n_src = sum(len(r['ra']) for r in results.values()) // max(1, len(results))
            print(f'  [{ti+1}/{len(pairs)}] {tile_id}  (~{n_src} sources/band so far)')

    # Concatenate per-band results
    out = {}
    for tb, r in results.items():
        if not r['ra']:
            continue
        out[tb] = {
            'ra':           np.concatenate(r['ra']),
            'dec':          np.concatenate(r['dec']),
            'pred_offsets': np.concatenate(r['pred'], axis=0),
            'raw_offsets':  np.concatenate(r['raw'], axis=0),
            'sigma':        np.concatenate(r['sigma']),
            'tile_ids':     r['tiles'],
        }
    return out


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

    # Build WCS: uniform angular pixels, TAN projection handles cos(dec).
    # CDELT1 is negative (RA increases westward in FITS convention).
    mesh_h, mesh_w = mesh['dra'].shape
    dstep_deg = dstep_arcsec / 3600.0

    return {
        'field':               field,
        'mesh':                mesh,
        'ra_ref':              ra_ref,
        'dec_ref':             dec_ref,
        'x_min_arcsec':        x_min,
        'y_min_arcsec':        y_min,
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

def _build_concordance_wcs(result: dict) -> 'fits.Header':
    """Build a shared TAN-projection WCS for the concordance mesh.

    Uses uniform angular pixels (dstep x dstep arcsec) in both axes.
    CDELT1 is negative (RA increases westward per FITS convention).
    CRPIX is set so that mesh pixel (0,0) maps to the SW corner of the footprint.
    """
    ra_ref  = result['ra_ref']
    dec_ref = result['dec_ref']
    x_min   = result['x_min_arcsec']
    y_min   = result['y_min_arcsec']
    dstep   = result['dstep_arcsec']
    mesh_h, mesh_w = result['mesh']['dra'].shape

    dstep_deg = dstep / 3600.0
    # CRPIX: pixel that corresponds to (ra_ref, dec_ref).
    # Mesh pixel 0 corresponds to x_min arcsec offset from ra_ref.
    crpix1 = -x_min / dstep + 1.0   # x_min is negative (west of center)
    crpix2 = -y_min / dstep + 1.0   # y_min is negative (south of center)

    w = WCS(naxis=2)
    w.wcs.crpix = [crpix1, crpix2]
    w.wcs.crval = [ra_ref, dec_ref]
    w.wcs.cdelt = [-dstep_deg, dstep_deg]  # RA negative, Dec positive
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.cunit = ['deg', 'deg']
    return w.to_header()


def _build_band_hdus(
    result: dict,
    target_band: str,
    wcs_header: 'fits.Header' = None,
) -> list:
    """Build FITS HDUs for one band's global concordance field.

    Returns a list of ImageHDUs (DRA, DDE) with a band prefix in the
    extension name so multiple bands coexist in one FITS file.
    """
    mesh    = result['mesh']
    ra_ref  = result['ra_ref']
    dec_ref = result['dec_ref']
    dstep   = result['dstep_arcsec']

    if wcs_header is None:
        wcs_header = _build_concordance_wcs(result)

    # Band key for extension names: e.g. "r", "nisp_Y"
    if target_band.startswith('rubin_'):
        band_key = target_band.split('_', 1)[1]
    else:
        band_key = target_band  # nisp_Y etc.

    def _make_hdu(data, suffix, comment):
        name = f'{band_key}.{suffix}'
        hdu = fits.ImageHDU(data=data.astype(np.float32), name=name)
        hdu.header.update(wcs_header)
        hdu.header['DSTEP']    = (float(dstep),   'Mesh step size in arcsec')
        hdu.header['DUNIT']    = ('arcsec',        'Unit of offset values')
        hdu.header['INTERP']   = ('bilinear',      'Recommended interpolation method')
        hdu.header['CONCRDNC'] = (True,            'Global concordance field')
        hdu.header['TGTBAND']  = (target_band,     'Target band')
        hdu.header['REFFRAME'] = ('euclid_VIS',    'Reference astrometric frame')
        hdu.header['SOLVETYP'] = ('global_sky',    'Global sky-coord field solve')
        hdu.header['RA_REF']   = (float(ra_ref),   'Field centroid RA (deg)')
        hdu.header['DEC_REF']  = (float(dec_ref),  'Field centroid Dec (deg)')
        hdu.header['NSRC']     = (result['n_sources'], 'Total sources used in solve')
        hdu.header['GRIDH']    = (result['grid_shape'][0], 'Control grid height')
        hdu.header['GRIDW']    = (result['grid_shape'][1], 'Control grid width')
        hdu.header['COMMENT']  = comment
        return hdu

    hdus = []
    hdus.append(_make_hdu(mesh['dra'],  'DRA', f'DeltaRA* offset field ({target_band}, arcsec)'))
    hdus.append(_make_hdu(mesh['ddec'], 'DDE', f'DeltaDec offset field ({target_band}, arcsec)'))
    return hdus


def write_global_fits(
    band_hdus: list,
    output_path: str,
    target_bands: list,
    n_sources_total: int = 0,
    coverage_hdu: fits.ImageHDU = None,
) -> None:
    """Write the combined multi-band global concordance FITS.

    Coverage is written once as a shared 'COVERAGE' extension since all
    bands share the same CenterNet-detected anchor positions.
    """
    primary = fits.PrimaryHDU()
    primary.header['CONCRDNC'] = (True, 'JAISP global sky-coord concordance product')
    primary.header['SOLVETYP'] = ('global_sky', 'Single field fitted over full mosaic')
    primary.header['NBANDS']   = (len(target_bands), 'Number of target bands')
    primary.header['NSRCTOT']  = (n_sources_total, 'Total sources across all bands')
    for i, b in enumerate(target_bands):
        primary.header[f'BAND{i}'] = (b, f'Target band {i}')

    all_hdus = [primary] + band_hdus
    if coverage_hdu is not None:
        all_hdus.append(coverage_hdu)

    fits.HDUList(all_hdus).writeto(output_path, overwrite=True)
    n_band_ext = len(band_hdus)
    n_cov = 1 if coverage_hdu is not None else 0
    print(f'\nWrote global concordance: {output_path}')
    print(f'  Bands: {target_bands}')
    print(f'  Extensions: {n_band_ext} band ({n_band_ext//2} x DRA/DDE) + {n_cov} shared COV')


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
            ext_names = {h.name.upper(): h.name for h in hdul if h.name != 'PRIMARY'}

            # Multi-band format: {band}.DRA, {band}.DDE, shared COVERAGE
            # Single-band legacy: GLOBAL.DRA, GLOBAL.DDE, GLOBAL.COV
            self.band_fields = {}
            self.wcs = None
            self.dstep_arcsec = 1.0

            # Try multi-band format first
            for ext_name in ext_names.values():
                if ext_name.endswith('.DRA'):
                    band_key = ext_name[:-4]  # e.g. 'R', 'NISP_Y'
                    dde_name = f'{band_key}.DDE'
                    if dde_name.upper() in ext_names:
                        dra_hdu = hdul[ext_name]
                        dde_hdu = hdul[ext_names[dde_name.upper()]]
                        self.band_fields[band_key] = {
                            'dra': dra_hdu.data.astype(np.float32),
                            'dde': dde_hdu.data.astype(np.float32),
                        }
                        if self.wcs is None:
                            self.wcs = WCS(dra_hdu.header, naxis=2)
                            self.dstep_arcsec = float(dra_hdu.header.get('DSTEP', 1.0))

            # Legacy single-band format fallback
            if not self.band_fields and 'GLOBAL.DRA' in ext_names:
                dra_hdu = hdul[ext_names['GLOBAL.DRA']]
                dde_hdu = hdul[ext_names['GLOBAL.DDE']]
                self.band_fields['GLOBAL'] = {
                    'dra': dra_hdu.data.astype(np.float32),
                    'dde': dde_hdu.data.astype(np.float32),
                }
                self.wcs = WCS(dra_hdu.header, naxis=2)
                self.dstep_arcsec = float(dra_hdu.header.get('DSTEP', 1.0))

            self.cov = hdul['COVERAGE'].data.astype(np.float32) if 'COVERAGE' in ext_names else None
            # Legacy coverage
            if self.cov is None and 'GLOBAL.COV' in ext_names:
                self.cov = hdul[ext_names['GLOBAL.COV']].data.astype(np.float32)

        bands_str = ', '.join(sorted(self.band_fields.keys()))
        first = next(iter(self.band_fields.values()))
        print(f'GlobalConcordanceMap: {first["dra"].shape} mesh at {self.dstep_arcsec}"/px  '
              f'bands=[{bands_str}]')

    def _resolve_band_key(self, band: str) -> str:
        """Map a band name to the FITS extension key."""
        # Try exact match first, then common variants
        for candidate in [band, band.upper(), f'NISP_{band}', f'NISP_{band.upper()}',
                          band.replace('rubin_', '').upper(), 'GLOBAL']:
            if candidate in self.band_fields:
                return candidate
        raise KeyError(f'Band {band} not found in concordance. Available: {list(self.band_fields.keys())}')

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
        band: str = 'r',
    ):
        """Return (dra_arcsec, ddec_arcsec) at sky positions for a specific band."""
        key = self._resolve_band_key(band)
        fields = self.band_fields[key]
        mesh_xy = self._sky_to_mesh_xy(np.atleast_1d(ra), np.atleast_1d(dec))
        return self._interp(fields['dra'], mesh_xy), self._interp(fields['dde'], mesh_xy)

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
        band: str = 'r',
    ):
        """Project Rubin pixel(s) onto the VIS grid with global concordance applied."""
        rubin_x = np.atleast_1d(np.asarray(rubin_x, dtype=np.float64))
        rubin_y = np.atleast_1d(np.asarray(rubin_y, dtype=np.float64))
        ra, dec = rubin_wcs.wcs_pix2world(rubin_x, rubin_y, 0)
        dra, ddec = self.correction_at_sky(ra, dec, band=band)
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
    p.add_argument('--cache-predictions', type=str, default='',
                   help='Path to save/load cached per-band predictions (.npz). '
                        'If the file exists, skip detection+encoding and load from cache. '
                        'If it does not exist, run detection+encoding and save to this path. '
                        'This makes re-running with different grid/solver settings instant.')
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    v6_ckpt = getattr(args, 'v6_checkpoint', '') or ''
    v7_ckpt = getattr(args, 'v7_checkpoint', '') or ''
    model, ckpt = load_model(args.checkpoint, device,
                             v6_checkpoint=v6_ckpt, v7_checkpoint=v7_ckpt)

    # Optional neural detector for source detection
    detr_ckpt = getattr(args, 'detector_checkpoint', '') or ''
    if detr_ckpt and v7_ckpt:
        from train_astro_v7 import _load_detector
        args._detr_detector = _load_detector(detr_ckpt, v7_ckpt, device)
    else:
        args._detr_detector = None

    # Determine target bands to process
    target_band_raw = str(ckpt.get('target_band', ckpt.get('args', {}).get('rubin_band', 'r')))
    is_multiband = target_band_raw == 'multiband'

    if is_multiband and getattr(args, 'all_bands', False):
        ckpt_target_bands = [str(b) for b in ckpt.get('target_bands', []) if str(b).strip()]
        if ckpt_target_bands:
            target_bands_to_process = []
            for b in ckpt_target_bands:
                try:
                    target_bands_to_process.append(_normalize_any_band(b))
                except Exception:
                    pass
        else:
            target_bands_to_process = ['rubin_r']
        print(f'Processing all {len(target_bands_to_process)} bands: {target_bands_to_process}')
    else:
        if is_multiband:
            target_bands_to_process = ['rubin_r']
        else:
            target_bands_to_process = [_normalize_any_band(target_band_raw)]

    input_bands_raw = [str(x) for x in ckpt.get('input_bands', target_bands_to_process[:1])]
    input_bands = []
    for b in input_bands_raw:
        try:
            nb = _normalize_any_band(b)
        except Exception:
            continue
        if nb not in input_bands:
            input_bands.append(nb)
    detect_bands = normalize_rubin_bands(args.detect_bands) or [f'rubin_{b}' for b in ('g','r','i','z')]

    pairs = discover_tile_pairs(args.rubin_dir, args.euclid_dir)
    if args.tile_id:
        pairs = [(t, r, e) for t, r, e in pairs if t == args.tile_id]

    all_summaries = []
    all_band_hdus = []
    completed_bands = []
    n_sources_total = 0
    shared_coverage_hdu = None  # built from first band, shared across all

    # ---- Prediction cache: save/load the expensive detection+encoding step ----
    cache_path = getattr(args, 'cache_predictions', '') or ''
    all_band_sources = None

    if cache_path and os.path.exists(cache_path):
        print(f'\nLoading cached predictions from {cache_path}')
        cache = np.load(cache_path, allow_pickle=True)
        all_band_sources = {}
        for tb in target_bands_to_process:
            key = tb.replace('rubin_', '').replace('nisp_', 'nisp_')
            if f'{key}_ra' in cache:
                all_band_sources[tb] = {
                    'ra':           cache[f'{key}_ra'],
                    'dec':          cache[f'{key}_dec'],
                    'pred_offsets': cache[f'{key}_pred'],
                    'raw_offsets':  cache[f'{key}_raw'],
                    'sigma':        cache[f'{key}_sigma'],
                    'tile_ids':     list(cache[f'{key}_tiles']),
                }
        print(f'Loaded {len(all_band_sources)} bands from cache.')
    else:
        # Use fast multiband path when processing multiple bands:
        # detect once per tile, encode once, then predict each band via MLP head only.
        if len(target_bands_to_process) > 1:
            print(f'\nCollecting predictions for {len(target_bands_to_process)} bands '
                  f'from {len(pairs)} tiles (detect-once mode)...')
            all_band_sources = collect_all_predictions_multiband(
                model, device, pairs, target_bands_to_process,
                input_bands, detect_bands, args,
            )
            print(f'Done. Bands with sources: {list(all_band_sources.keys())}')
        # else: single-band, handled in the loop below

        # Save cache if requested
        if cache_path and all_band_sources is not None:
            cache_dict = {}
            for tb, src in all_band_sources.items():
                key = tb.replace('rubin_', '').replace('nisp_', 'nisp_')
                cache_dict[f'{key}_ra'] = src['ra']
                cache_dict[f'{key}_dec'] = src['dec']
                cache_dict[f'{key}_pred'] = src['pred_offsets']
                cache_dict[f'{key}_raw'] = src['raw_offsets']
                cache_dict[f'{key}_sigma'] = src['sigma']
                cache_dict[f'{key}_tiles'] = np.array(src['tile_ids'], dtype=object)
            np.savez_compressed(cache_path, **cache_dict)
            print(f'Saved prediction cache to {cache_path}')

    for band_i, target_band in enumerate(target_bands_to_process):
        print(f'\n{"="*60}')
        print(f'[{band_i+1}/{len(target_bands_to_process)}] Target band: {target_band}')
        print(f'{"="*60}')

        if all_band_sources is not None:
            if target_band not in all_band_sources:
                print(f'  Skipping {target_band}: no sources collected')
                continue
            sources = all_band_sources[target_band]
        else:
            print(f'Collecting predictions from {len(pairs)} tiles...')
            sources = collect_all_predictions(
                model, device, pairs, target_band, input_bands, detect_bands, args,
            )
        n_src = sources.get('n_sources', len(sources['ra']))
        print(f'Total sources: {n_src}')

        if n_src < 10:
            print(f'  Skipping {target_band}: too few sources ({n_src})')
            continue

        print(f'Fitting global field for {target_band}...')
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

        pred = sources['pred_offsets']
        raw  = sources['raw_offsets']
        pred_mag = np.hypot(pred[:, 0], pred[:, 1]) * 1000.0
        raw_mag  = np.hypot(raw[:,  0], raw[:,  1]) * 1000.0
        print(f'  Raw WCS  median: {np.median(raw_mag):.1f} mas')
        print(f'  NN pred  median: {np.median(pred_mag):.1f} mas')
        print(f'  Field footprint: {result["field_shape_arcsec"][1]:.0f}" x {result["field_shape_arcsec"][0]:.0f}"')

        # Build shared WCS from first band; reuse for all subsequent bands
        if shared_coverage_hdu is None:
            shared_wcs_header = _build_concordance_wcs(result)
            if 'coverage' in result['mesh'] and result['mesh']['coverage'] is not None:
                cov_hdu = fits.ImageHDU(data=result['mesh']['coverage'].astype(np.float32), name='COVERAGE')
                cov_hdu.header.update(shared_wcs_header)
                cov_hdu.header['DSTEP'] = (float(result['dstep_arcsec']), 'Mesh step size in arcsec')
                cov_hdu.header['DUNIT'] = ('arcsec', 'Min distance to nearest anchor source')
                cov_hdu.header['COVTYPE'] = ('min_dist', 'Shared across all bands')
                shared_coverage_hdu = cov_hdu

        # Accumulate HDUs for combined FITS (all bands share the same WCS)
        band_hdus = _build_band_hdus(result, target_band, wcs_header=shared_wcs_header)
        all_band_hdus.extend(band_hdus)
        completed_bands.append(target_band)
        n_sources_total += n_src
        print(f'  Mesh shape: {result["mesh"]["dra"].shape}  ({result["dstep_arcsec"]}" per pixel)')

        if getattr(args, 'plot', ''):
            from viz import plot_global_concordance
            band_suffix = target_band.replace('rubin_', '').replace('nisp_', 'nisp_')
            plot_base, plot_ext = os.path.splitext(args.plot)
            plot_path = f'{plot_base}_{band_suffix}{plot_ext}' if len(target_bands_to_process) > 1 else args.plot
            plot_global_concordance(result, sources, plot_path, target_band=target_band)

        all_summaries.append({
            'target_band':     target_band,
            'n_sources':       int(n_src),
            'solver':          result['solver'],
            'grid_shape':      list(result['grid_shape']),
            'dstep_arcsec':    result['dstep_arcsec'],
            'raw_median_mas':  float(np.median(raw_mag)),
            'pred_median_mas': float(np.median(pred_mag)),
            'ra_ref':          result['ra_ref'],
            'dec_ref':         result['dec_ref'],
            'field_shape_arcsec': list(result['field_shape_arcsec']),
        })

    # Write single combined FITS with all bands + one shared coverage
    if all_band_hdus:
        write_global_fits(all_band_hdus, args.output, completed_bands,
                          n_sources_total, coverage_hdu=shared_coverage_hdu)

    if args.summary_json and all_summaries:
        with open(args.summary_json, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        print(f'Summary: {args.summary_json}')


if __name__ == '__main__':
    main()
