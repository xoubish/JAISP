"""Compare frozen detection heads against full Q1 MER on held-out fields.

Preserves the paper's masks, 0.5 arcsec match radius, four-pixel reference
margin and EDF-S purity footprint. Reports VIS, NIR-only and full-catalogue
completeness separately. A fixed confidence of 0.30 is the primary comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'models'))
from detection.centernet_detector import CenterNetDetector, _prepare_artifact_mask
from detection.detector import JAISPEncoderWrapper
from detection.masks import bright_star_saturation_mask, load_gaia_cache
from detection.train_centernet import _cached_forward
from detection.validation_utils import build_inputs
from load_foundation import load_foundation

FIELDS = {
    'ECDFS_patch25': dict(
        euclid='data/euclid_tiles_all_q1', rubin='data/rubin_tiles_all',
        catalog='data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits',
        gaia='data/gaia_ecdfs_astrometry_cache.npz', glob='*_patch_25_euclid.npz'),
    'EDF-S': dict(
        euclid='data/edf_s_ood/euclid_tiles_edfs_q1', rubin='data/edf_s_ood/rubin_tiles_edfs',
        catalog='data/edf_s_ood/catalogs_compact/mer_FINAL_q1_TILE102021011_footprint.fits',
        gaia='data/gaia_edfs_cache.npz', glob='tile_*_euclid.npz'),
}
ENCODER = ROOT/'models/checkpoints/jaisp_v11_q1_soft/checkpoint_best.pt'
CONFS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


@torch.no_grad()
def predict_features(model, features, mask, shape, floor=0.10):
    out = _cached_forward(model, features)
    hm = out['heatmap'][0, 0]
    h, w = hm.shape
    peaks = (hm == F.max_pool2d(hm[None, None], 7, stride=1, padding=3)[0, 0]) & (hm > floor)
    if mask is not None:
        peaks &= ~_prepare_artifact_mask(mask, (h, w), hm.device)
    yi, xi = torch.where(peaks)
    off = out['offset'][0]
    x = ((xi.float()+off[0, yi, xi])/(w-1)).clamp(0, 1)
    y = ((yi.float()+off[1, yi, xi])/(h-1)).clamp(0, 1)
    xy = torch.stack((x*(shape[1]-1), y*(shape[0]-1)), 1)
    return xy.cpu().numpy(), hm[yi, xi].cpu().numpy()


def run_field(name, models, device, outdir, limit=0):
    cfg = FIELDS[name]
    cat = fits.getdata(ROOT/cfg['catalog'])
    clean = np.asarray(cat['spurious_flag']) != 1
    vis = np.asarray(cat['vis_det']) == 1
    mag = np.asarray(cat['mag_vis'], float)
    groups = dict(vis_bright=clean & vis & np.isfinite(mag) & (mag < 24.5),
                  vis_all=clean & vis, nir_only=clean & ~vis, full_clean=clean,
                  full_including_flagged=np.ones(len(cat), bool))
    with np.load(ROOT/cfg['gaia']) as g:
        gaia = dict(ra=np.asarray(g['ra'], float), dec=np.asarray(g['dec'], float),
                    g=np.asarray(g['g'] if 'g' in g else g['phot_g_mean_mag'], float))
    ra, dec = np.asarray(cat['ra'], float), np.asarray(cat['dec'], float)
    cvis = clean & vis
    bbox = (ra[cvis].min()+0.003, ra[cvis].max()-0.003,
            dec[cvis].min()+0.003, dec[cvis].max()-0.003)
    paths = sorted((ROOT/cfg['euclid']).glob(cfg['glob']))
    if limit:
        paths = paths[:limit]
    encoder = None
    if name == 'EDF-S':
        foundation = load_foundation(str(ENCODER), device=torch.device('cpu'), freeze=True)
        encoder = JAISPEncoderWrapper(foundation, freeze=True).to(device).eval()
    acc = {m: {str(c): dict(recovered={g: 0 for g in groups},
                             totals={g: 0 for g in groups}, n_det=0, n_matched=0,
                             n_det_footprint=0, n_matched_footprint=0) for c in CONFS}
           for m in models}
    (outdir/name).mkdir(parents=True, exist_ok=True)
    for i, ep in enumerate(paths):
        tid = ep.name.removesuffix('_euclid.npz')
        with np.load(ep, allow_pickle=True) as f:
            ed = dict(f)
        shape = ed['img_VIS'].shape
        h, w = shape
        wcs = WCS(fits.Header.fromstring(str(ed['wcs_VIS'])))
        mask = bright_star_saturation_mask(np.nan_to_num(ed['img_VIS']), str(ed['wcs_VIS']), gaia)
        x, y = wcs.all_world2pix(ra, dec, 0)
        inside = (x >= 4) & (x < w-4) & (y >= 4) & (y < h-4)
        unmasked = ~mask[np.clip(y.astype(int), 0, h-1), np.clip(x.astype(int), 0, w-1)]
        reference = inside & unmasked
        if (reference & cvis).sum() < 2:
            continue
        pts = np.column_stack((x[reference], y[reference]))
        full_tree = cKDTree(np.column_stack((x[inside], y[inside])))
        selected = {g: sel[reference] for g, sel in groups.items()}
        if encoder is None:
            cached = torch.load(ROOT/'data/cached_features_v11_q1'/f'{tid}_aug0.pt',
                                map_location='cpu', weights_only=True)
            assert tuple(cached['aug_params']) == (0, False, False)
            features = cached['features'][None].to(device)
        else:
            with np.load(ROOT/cfg['rubin']/f'{tid}.npz', allow_pickle=True) as f:
                rd = dict(f)
            images, rms, _ = build_inputs(ed, rd)
            with torch.no_grad():
                features = encoder({b: torch.from_numpy(v[None, None]).to(device) for b, v in images.items()},
                                   {b: torch.from_numpy(v[None, None]).to(device) for b, v in rms.items()})
        for model_name, model in models.items():
            xy, scores = predict_features(model, features, mask, shape)
            np.savez_compressed(outdir/name/f'{model_name}__{tid}.npz', xy=xy, scores=scores)
            for conf in CONFS:
                row = acc[model_name][str(conf)]
                detection = xy[scores >= conf]
                hit = cKDTree(detection).query(pts)[0] < 5 if len(detection) else np.zeros(len(pts), bool)
                for g, select in selected.items():
                    row['recovered'][g] += int(hit[select].sum())
                    row['totals'][g] += int(select.sum())
                row['n_det'] += len(detection)
                if len(detection):
                    match = full_tree.query(detection)[0] < 5
                    dra, ddec = wcs.all_pix2world(detection[:, 0], detection[:, 1], 0)
                    footprint = ((dra > bbox[0]) & (dra < bbox[1])
                                 & (ddec > bbox[2]) & (ddec < bbox[3])) if name == 'EDF-S' else np.ones(len(detection), bool)
                    row['n_matched'] += int(match.sum())
                    row['n_det_footprint'] += int(footprint.sum())
                    row['n_matched_footprint'] += int(match[footprint].sum())
        if (i+1) % 10 == 0 or i+1 == len(paths):
            print(name, i+1, '/', len(paths), flush=True)
    for rows in acc.values():
        for row in rows.values():
            row['completeness_percent'] = {g: 100*row['recovered'][g]/max(n, 1) for g, n in row['totals'].items()}
            row['purity_percent'] = 100*row['n_matched_footprint']/max(row['n_det_footprint'], 1)
            row['purity_unrestricted_percent'] = 100*row['n_matched']/max(row['n_det'], 1)
    result = dict(field=name, n_tiles=len(paths), rows=acc, primary_threshold=0.30,
                  completeness='tile-pooled; 0.5 arcsec; clean unless stated; NIR has no magnitude cut',
                  purity='full MER including flagged; EDF-S restricted to existing VIS-reference footprint')
    (outdir/f'{name}_metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    for m, rows in acc.items():
        print(name, m, json.dumps(rows['0.3']), flush=True)
    del encoder
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', action='append', required=True, help='name=path')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--field', choices=list(FIELDS)+['both'], default='both')
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    models = {}
    for spec in args.checkpoint:
        name, path = spec.split('=', 1)
        models[name] = CenterNetDetector.load(path, None, device).eval()
    (args.out/'config.json').write_text(json.dumps(dict(vars(args), encoder=str(ENCODER)), default=str, indent=2)+'\n')
    for name in FIELDS if args.field == 'both' else [args.field]:
        run_field(name, models, device, args.out, args.limit)


if __name__ == '__main__':
    main()
