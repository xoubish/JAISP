"""Export RAW-only cross-band anchors (no head, no encoder, CPU-only).

Fig 7 ("before" / raw cross-survey agreement) reads only the `{band}_raw` and
`{band}_snr` columns of the anchor cache -- both are head-independent classical
per-band Gaussian centroids vs the VIS reference. This script reproduces exactly
the raw path of eval_latent_position.py (same centroiding, same clip/SNR cuts,
same key layout) WITHOUT loading the foundation encoder or the (still-training)
latent-position head, so it runs on CPU and does not compete with head training
for the GPUs. When the head finishes, the full eval cache supersedes this one and
its `_raw`/`_snr` columns are identical.

Usage:
    PYTHONPATH=models python models/astrometry2/export_raw_anchors.py \
        --rubin-dir data/rubin_tiles_all \
        --euclid-dir data/euclid_tiles_all_q1 \
        --detector-labels data/detection_labels/centernet_q1_790_vissep_thresh03.pt \
        --out models/checkpoints/latent_position_q1_vissep/anchors_q1_vissep_raw.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
for p in (MODELS_DIR, SCRIPT_DIR):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

from astrometry2.dataset import (
    discover_tile_pairs, signal_mask_in_band, _to_float32, NISP_BAND_ORDER,
)
from astrometry2.eval_latent_position import centroid_in_band_and_project
from source_matching import (
    RUBIN_BAND_ORDER, refine_centroids_psf_fit, safe_header_from_card_string,
)
from astropy.wcs import WCS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rubin-dir', required=True)
    ap.add_argument('--euclid-dir', required=True)
    ap.add_argument('--detector-labels', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--clip-mas', type=float, default=200.0)
    ap.add_argument('--min-snr', type=float, default=5.0)
    ap.add_argument('--max-tiles', type=int, default=0)
    args = ap.parse_args()

    label_payload = torch.load(args.detector_labels, weights_only=False)
    detector_labels = label_payload['labels'] if 'labels' in label_payload else label_payload
    pairs = discover_tile_pairs(args.rubin_dir, args.euclid_dir)
    if args.max_tiles > 0:
        pairs = pairs[:args.max_tiles]
    print(f'{len(pairs)} tiles | {len(detector_labels)} labelled tiles', flush=True)

    rubin_bands = [f'rubin_{b}' for b in RUBIN_BAND_ORDER]
    nisp_bands = [f'nisp_{b}' for b in NISP_BAND_ORDER]
    all_bands = rubin_bands + nisp_bands
    anchors = {b: {'ra': [], 'dec': [], 'raw': [], 'snr': [], 'tiles': []} for b in all_bands}
    n_tiles = 0

    for idx, (tile_id, rubin_path, euclid_path) in enumerate(pairs, start=1):
        if idx == 1 or idx % 50 == 0 or idx == len(pairs):
            print(f'  tile {idx}/{len(pairs)}: {tile_id}', flush=True)
        if tile_id not in detector_labels:
            continue
        try:
            rdata = np.load(rubin_path, allow_pickle=True)
            edata = np.load(euclid_path, allow_pickle=True)
            rubin_cube = rdata['img']
            rubin_var = rdata['var'] if 'var' in rdata.files else None
            vis_img = np.nan_to_num(_to_float32(edata['img_VIS']), nan=0.0)
            vwcs = WCS(safe_header_from_card_string(edata['wcs_VIS'].item()))
            rwcs = WCS(rdata['wcs_hdr'].item())
        except Exception:
            continue
        vis_hw = vis_img.shape

        entry = detector_labels[tile_id]
        xy_norm = entry[0] if isinstance(entry, tuple) else entry
        xy_norm = np.asarray(xy_norm, dtype=np.float32)
        if xy_norm.ndim != 2 or xy_norm.shape[0] < 10:
            continue
        H_vis, W_vis = vis_hw
        vis_seed = np.stack([xy_norm[:, 0] * max(W_vis - 1, 1),
                             xy_norm[:, 1] * max(H_vis - 1, 1)], axis=1).astype(np.float32)
        vis_keep = signal_mask_in_band(vis_img, vis_seed, radius=3)
        if vis_keep.sum() < 10:
            continue
        vis_xy, _, _ = refine_centroids_psf_fit(vis_img, vis_seed[vis_keep], radius=3, fwhm_guess=2.5)
        vis_xy = vis_xy.astype(np.float32)

        for band_name in all_bands:
            if band_name.startswith('rubin_'):
                short = band_name.split('_', 1)[1]
                bidx = RUBIN_BAND_ORDER.index(short)
                if bidx >= rubin_cube.shape[0]:
                    continue
                band_img = np.nan_to_num(_to_float32(rubin_cube[bidx]), nan=0.0)
                band_wcs = rwcs
                fwhm = 3.0
            else:
                short = band_name.split('_', 1)[1]
                img_key, wcs_key = f'img_{short}', f'wcs_{short}'
                if img_key not in edata or wcs_key not in edata:
                    continue
                band_img = np.nan_to_num(_to_float32(edata[img_key]), nan=0.0)
                try:
                    band_wcs = WCS(safe_header_from_card_string(edata[wcs_key].item()))
                except Exception:
                    continue
                fwhm = 2.5

            band_xy_vis, offset_arcsec, valid, band_snr = centroid_in_band_and_project(
                band_img, band_wcs, vis_xy, vwcs,
                refine_radius=3, fwhm_guess=fwhm, band_rms=None,
                epsf_head=None, band_name=band_name,
            )
            if valid.sum() < 3:
                continue
            raw_radial = np.sqrt((offset_arcsec[valid] ** 2).sum(axis=1))
            keep = np.ones(int(valid.sum()), dtype=bool)
            if args.clip_mas > 0:
                keep &= (raw_radial * 1000) < args.clip_mas
            if args.min_snr > 0:
                keep &= band_snr[valid] >= args.min_snr
            if keep.sum() < 3:
                continue
            offset_valid = offset_arcsec[valid][keep]
            band_pos_valid = band_xy_vis[valid][keep]
            snr_valid = band_snr[valid][keep]
            ras, decs = vwcs.wcs_pix2world(band_pos_valid[:, 0], band_pos_valid[:, 1], 0)
            n_keep = int(keep.sum())
            anchors[band_name]['ra'].append(ras.astype(np.float32))
            anchors[band_name]['dec'].append(decs.astype(np.float32))
            anchors[band_name]['raw'].append(offset_valid.astype(np.float32))
            anchors[band_name]['snr'].append(snr_valid.astype(np.float32))
            anchors[band_name]['tiles'].append(np.full(n_keep, tile_id, dtype='U64'))
        n_tiles += 1

    cache = {}
    for band_name in all_bands:
        a = anchors[band_name]
        if not a['ra']:
            continue
        key = band_name.split('_', 1)[1] if band_name.startswith('rubin_') else band_name
        cache[f'{key}_ra'] = np.concatenate(a['ra'])
        cache[f'{key}_dec'] = np.concatenate(a['dec'])
        cache[f'{key}_raw'] = np.concatenate(a['raw'])
        cache[f'{key}_snr'] = np.concatenate(a['snr'])
        cache[f'{key}_tiles'] = np.concatenate(a['tiles'])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **cache)
    total = sum(len(cache[k]) for k in cache if k.endswith('_ra'))
    print(f'\n{n_tiles} tiles | saved {total} raw anchors to {out}')
    print('per-band median raw radial [mas] (all / bright S/N>=30):')
    for band_name in all_bands:
        key = band_name.split('_', 1)[1] if band_name.startswith('rubin_') else band_name
        if f'{key}_raw' not in cache:
            continue
        raw = cache[f'{key}_raw'] * 1000; snr = cache[f'{key}_snr']
        r = np.hypot(raw[:, 0], raw[:, 1]); br = snr >= 30
        rb = np.median(r[br]) if br.any() else np.nan
        print(f'  {key:8s} N={len(r):6d}  all={np.median(r):6.1f}  bright={rb:6.1f}')


if __name__ == '__main__':
    main()
