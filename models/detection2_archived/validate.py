"""Fixed patch-25 development monitoring, using the preceding MER protocol."""
from __future__ import annotations

import numpy as np
import torch
from astropy.io import fits
from scipy.spatial import cKDTree

from .common import ROOT, resolve
from .data import artifact_mask
from detection.validation_utils import _wcs_vis
from detection.visnir_eval_experiment import predict_features


@torch.no_grad()
def validate(model, cfg, tiles, device):
    model.eval()
    cat = fits.getdata(ROOT / 'data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits')
    clean = np.asarray(cat['spurious_flag']) != 1
    vis = np.asarray(cat['vis_det']) == 1
    mag = np.asarray(cat['mag_vis'], float)
    groups = {'vis_bright': clean & vis & np.isfinite(mag) & (mag < 24.5),
              'vis_all': clean & vis, 'nir_only': clean & ~vis, 'full_mer': clean}
    acc = {str(c): {'recovered': {g: 0 for g in groups}, 'total': {g: 0 for g in groups},
                    'n_det': 0, 'n_matched': 0} for c in cfg['validation_thresholds']}
    scored = []
    for tid in tiles:
        if not tid.endswith(f"_patch_{cfg['excluded_patch']}"):
            raise ValueError('Validation must use only the excluded patch')
        with np.load(resolve(cfg['euclid_dir']) / f'{tid}_euclid.npz', allow_pickle=True) as ed:
            wcs = _wcs_vis(ed)
            h, w = ed['img_VIS'].shape
        mask = artifact_mask(cfg, tid)
        x, y = wcs.all_world2pix(cat['ra'], cat['dec'], 0)
        inside = np.isfinite(x) & np.isfinite(y) & (x >= 4) & (x < w - 4) & (y >= 4) & (y < h - 4)
        idx = np.where(inside)[0]
        unmasked = ~mask[y[idx].astype(int), x[idx].astype(int)]
        reference = idx[unmasked]
        if (clean[reference] & vis[reference]).sum() < 2:
            continue
        points = np.column_stack([x[reference], y[reference]])
        full = cKDTree(np.column_stack([x[idx], y[idx]]))
        cached = torch.load(resolve(cfg['feature_dir']) / f'{tid}_aug0.pt', map_location='cpu', weights_only=True)
        if tuple(cached['aug_params']) != (0, False, False):
            raise ValueError('Validation requires identity cached features')
        xy, scores = predict_features(model, cached['features'][None].to(device), mask, (h, w),
                                      floor=min(cfg['validation_thresholds']) - 0.01)
        for conf in cfg['validation_thresholds']:
            a = acc[str(conf)]
            det = xy[scores >= conf]
            hit = cKDTree(det).query(points)[0] < 5 if len(det) else np.zeros(len(points), bool)
            for g, group in groups.items():
                sel = group[reference]
                a['recovered'][g] += int(hit[sel].sum())
                a['total'][g] += int(sel.sum())
            a['n_det'] += len(det)
            a['n_matched'] += int((full.query(det)[0] < 5).sum()) if len(det) else 0
        scored.append(tid)
    metrics = {}
    for conf, a in acc.items():
        prefix = f'val/t{float(conf):.2f}'
        for g in groups:
            metrics[f'{prefix}/{g}_completeness'] = a['recovered'][g] / max(a['total'][g], 1)
        metrics[f'{prefix}/mer_match_fraction'] = a['n_matched'] / max(a['n_det'], 1)
        metrics[f'{prefix}/n_detections'] = a['n_det']
    return {'metrics': metrics, 'counts': acc, 'tiles_scored': scored,
            'definition': 'tile-pooled nearest matches <0.5 arcsec; VIS bright <24.5; '
                          'NIR/full have no magnitude cut; match fraction uses full MER including flagged; '
                          'development monitoring, not independent truth or final test'}
