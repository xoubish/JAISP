"""Choose non-overlapping held-out JAISP tiles for the detection-catalog comparison.

Candidates are patch-25 tiles (held out from the detection and astrometry heads)
and tiles in the photometry prior's RA test partition. Only the stride-2 grid
(x, y multiples of 512 Rubin pixels) is kept so no two tiles share sky, and any
tile overlapping one of the 28 tiles used to train/validate/test the mixture
prior is excluded.
"""
import sys
import numpy as np
import torch
from .common import (ROOT, OUT, LABELS, CACHE, PRIOR_RUN, RA_B2, RA_GUARD,
                     tile_xy, tile_patch, tangent_arcsec, read_json, write_json)
sys.path.insert(0, str(ROOT / 'models'))
from foundation_utils import discover_tile_pairs  # noqa: E402
from ..data import wcs  # noqa: E402


def tile_center(euclid_path):
    with np.load(euclid_path, allow_pickle=True) as e:
        wc = wcs(e['wcs_VIS']); h, w = e['img_VIS'].shape
        ra, dec = wc.pixel_to_world_values([0, w - 1, 0, w - 1], [0, 0, h - 1, h - 1])
        return float(e['ra_center']), float(e['dec_center']), float(min(ra)), float(max(ra))


def main():
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--training', type=int, default=0, help='Select this many stride-2 tiles OUTSIDE the held-out partitions (for training a new head)')
    p.add_argument('--exclude', type=str, default='', help='tiles.json of another selection whose tiles must not overlap this one')
    p.add_argument('--seed', type=int, default=20261003)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = discover_tile_pairs(str(ROOT / 'data/rubin_tiles_all'), str(ROOT / 'data/euclid_tiles_all_q1'))
    labels = torch.load(ROOT / LABELS, map_location='cpu', weights_only=False)['labels']
    selected = read_json(ROOT / PRIOR_RUN / 'metadata.json')['selected_tiles']
    # Scenes from the prior's train/val tiles entered training; its test tiles were held out and may be reused.
    prior_tiles = {t for split in ('train', 'val') for t in selected[split]}
    by_name = {p[0]: p for p in pairs}
    prior_centers = np.array([tile_center(by_name[t][2])[:2] for t in prior_tiles if t in by_name])
    chosen = []
    for name, rubin, euclid in sorted(pairs):
        if name not in labels or not (ROOT / CACHE / (name + '_aug0.pt')).exists(): continue
        x, y = tile_xy(name)
        if x % 512 or y % 512: continue
        ra, dec, ra_min, ra_max = tile_center(euclid)
        patch = tile_patch(name)
        partition = 'patch25' if patch == '25' else ('ra_test' if ra_min > RA_B2 + RA_GUARD else None)
        if args.training:
            if partition is not None: continue
            partition = 'training'
        elif partition is None: continue
        if name in prior_tiles: continue
        if len(prior_centers) and np.abs(tangent_arcsec(prior_centers, (ra, dec))).max(axis=1).min() < 108.4 - 14: continue  # prior scenes stayed 14" inside their tiles
        chosen.append(dict(tile=name, rubin=str(rubin), euclid=str(euclid), ra=ra, dec=dec,
                           patch=patch, partition=partition, detections=int(len(labels[name][0]))))
    if args.exclude:
        other = np.array([[t['ra'], t['dec']] for t in read_json(args.exclude)['tiles']])
        chosen = [t for t in chosen if np.abs(tangent_arcsec(other, (t['ra'], t['dec']))).max(axis=1).min() >= 108.4 - 1]
    if args.training:
        rng = np.random.default_rng(args.seed); rng.shuffle(chosen); chosen = sorted(chosen[:args.training], key=lambda t: t['tile'])
    for i, item in enumerate(chosen): item['region'] = i
    write_json(OUT / 'tiles.json', dict(tiles=chosen, labels=LABELS, feature_cache=CACHE,
                                        rule=('random stride-2 tiles outside patch 25 and the RA test partition' if args.training else
                                              'stride-2 grid of patch-25 and RA-test tiles, excluding tiles overlapping prior train/val tiles')))
    print(f'{len(chosen)} tiles; detections {sum(t["detections"] for t in chosen)}; '
          f'partitions {dict(zip(*np.unique([t["partition"] for t in chosen], return_counts=True)))}')


if __name__ == '__main__': main()
