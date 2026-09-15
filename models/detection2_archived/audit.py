"""Local ten-band candidate gallery; no fitted or supplied PSF model."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree

from .common import resolve, spaced, write_json
from .data import novel_candidates

BANDS = ['rubin_' + b for b in 'ugrizy'] + ['euclid_' + b for b in ('VIS', 'Y', 'J', 'H')]


def read_wcs(value):
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, str) and value.lstrip().startswith('{'):
        value = ast.literal_eval(value)
    if isinstance(value, dict):
        value = fits.Header(value)
    result = WCS(value)
    if not result.has_celestial:
        raise ValueError('Missing celestial WCS in input tile')
    return result


def load_images(cfg, tid):
    out = {}
    with np.load(resolve(cfg['rubin_dir']) / f'{tid}.npz', allow_pickle=True) as rd:
        wcs = read_wcs(rd['wcs_hdr'])
        for i, b in enumerate('ugrizy'):
            out['rubin_' + b] = (rd['img'][i], wcs)
    with np.load(resolve(cfg['euclid_dir']) / f'{tid}_euclid.npz', allow_pickle=True) as ed:
        for b in ('VIS', 'Y', 'J', 'H'):
            out['euclid_' + b] = (ed['img_' + b], read_wcs(ed['wcs_' + b]))
    return out


def gallery(images, point, seeds, title, path):
    vw = images['euclid_VIS'][1]
    world = vw.all_pix2world(point[None], 0)
    seed_world = vw.all_pix2world(seeds, 0)
    fig, axes = plt.subplots(1, 10, figsize=(18, 2.5), constrained_layout=True)
    for ax, band in zip(axes, BANDS):
        data, wcs = images[band]
        x, y = wcs.all_world2pix(world, 0)[0]
        other = wcs.all_world2pix(seed_world, 0)
        radius = 8 if band.startswith('rubin') else 16
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(0, cx-radius), min(data.shape[1], cx+radius+1)
        y0, y1 = max(0, cy-radius), min(data.shape[0], cy+radius+1)
        if x0 >= x1 or y0 >= y1:
            ax.set_axis_off()
            continue
        stamp = data[y0:y1, x0:x1]
        finite = stamp[np.isfinite(stamp)]
        lo, hi = np.percentile(finite, [5, 99]) if finite.size else (0, 1)
        ax.imshow(stamp, origin='lower', cmap='gray', vmin=lo, vmax=hi,
                  extent=[x0-.5, x1-.5, y0-.5, y1-.5])
        inside = (other[:, 0] >= x0) & (other[:, 0] < x1) & (other[:, 1] >= y0) & (other[:, 1] < y1)
        ax.scatter(other[inside, 0], other[inside, 1], s=45, facecolors='none', edgecolors='cyan', linewidths=.7)
        ax.scatter([x], [y], marker='+', color='magenta', s=55, linewidths=.7)
        ax.set_xlim(x0-.5, x1-.5)
        ax.set_ylim(y0-.5, y1-.5)
        ax.set_title(band.replace('rubin_', '').replace('euclid_', ''))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title + '\nMagenta: candidate; cyan: existing seed labels. Contrast scaled separately per band.', fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def audit(prepared, out, count=4, per_group=4):
    payload = torch.load(prepared / 'labels.pt', map_location='cpu', weights_only=False)
    cfg = payload['config']
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', {'prepared': str(prepared.resolve()), 'tiles': count,
                                   'per_group': per_group, 'experiment_config': cfg})
    rng = np.random.default_rng(cfg['seed'])
    records = []
    for tid in spaced(sorted(payload['tiles']), count):
        row = payload['tiles'][tid]
        h, w = row['shape']
        seeds = row['seeds'] * [w-1, h-1]
        xy, scores = row['proposal_xy'], row['proposal_scores']
        novel = novel_candidates(xy, seeds, cfg['label_match_radius_vis_px'])
        groups = {'unlabelled_ge_03': novel & (scores >= .3),
                  'unlabelled_015_03': novel & (scores < .3),
                  'seed_matched_control': ~novel & (scores >= .3)}
        images = load_images(cfg, tid)
        for group, select in groups.items():
            indices = np.where(select)[0]
            for j, index in enumerate(rng.permutation(indices)[:per_group]):
                point = xy[index]
                name = f'{tid}__{group}_{j:02d}.png'
                gallery(images, point, seeds, f'{tid} | {group} | score={scores[index]:.3f}', out / name)
                records.append({'tile_id': tid, 'group': group, 'xy_vis': point.tolist(),
                                'teacher_score': float(scores[index]), 'gallery': name,
                                'nearest_seed_distance_vis_px': float(cKDTree(seeds).query(point)[0])})
        write_json(out / 'records.json', records)
        print(f'{tid}: {len(records)} candidates illustrated', flush=True)
    write_json(out / 'summary.json', {'n_candidates': len(records),
                'groups': {g: sum(r['group'] == g for r in records) for g in groups},
                'interpretation': 'Illustrative teacher proposals; scores are not calibrated probabilities; '
                                  'unlabelled refers to seed labels, not MER matching or source truth.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tiles', type=int, default=4)
    p.add_argument('--per-group', type=int, default=4)
    a = p.parse_args()
    audit(a.prepared, a.out, a.tiles, a.per_group)


if __name__ == '__main__':
    main()
