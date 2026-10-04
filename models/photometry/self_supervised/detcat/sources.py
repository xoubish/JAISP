"""Detection-head sources with the frozen anchored astrometry correction, per tile.

The v11 CenterNet detections (conf 0.3, spike veto off) are read from the
production export; the anchored v11 astrometry head then moves every source to
its VIS-canonical position. All ten bands are later sampled at that single
corrected sky position through each band's own WCS.
"""
import argparse
import sys
import numpy as np
import pandas as pd
import torch
from .common import ROOT, OUT, LABELS, CACHE, FOUNDATION, ASTROMETRY, region_dir, read_json, write_json
from ..data import load_tile
from ..astrometry import FrozenAstrometry


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--regions', type=int, nargs='*'); p.add_argument('--threads', type=int, default=8)
    args = p.parse_args(); torch.set_num_threads(args.threads)
    tiles = read_json(OUT / 'tiles.json')['tiles']
    if args.regions is not None: tiles = [t for t in tiles if t['region'] in set(args.regions)]
    payload = torch.load(ROOT / LABELS, map_location='cpu', weights_only=False)
    if payload['config']['encoder_ckpt'] != FOUNDATION: raise ValueError('Detection export and foundation disagree')
    labels, scores = payload['labels'], payload.get('scores', {})
    astrometry = FrozenAstrometry(ROOT, ASTROMETRY, FOUNDATION)
    for t in tiles:
        folder = region_dir(t['region']); folder.mkdir(parents=True, exist_ok=True)
        tile = load_tile((t['tile'], t['rubin'], t['euclid']), labels)
        raw_xy = tile['xy'].copy(); raw_sky = tile['sky'].copy()
        astrometry.apply(tile, ROOT / CACHE)
        shift = tile['xy'] - raw_xy
        frame = pd.DataFrame(dict(source=np.arange(len(raw_xy)), x_vis_raw=raw_xy[:, 0], y_vis_raw=raw_xy[:, 1],
                                  x_vis=tile['xy'][:, 0], y_vis=tile['xy'][:, 1], ra_raw=raw_sky[:, 0], dec_raw=raw_sky[:, 1],
                                  ra=tile['sky'][:, 0], dec=tile['sky'][:, 1], shift_px=np.linalg.norm(shift, axis=1),
                                  score=np.asarray(scores.get(t['tile'], np.full(len(raw_xy), np.nan)), float)))
        frame.to_csv(folder / 'detections.csv', index=False)
        write_json(folder / 'metadata.json', dict(**t, n_detections=len(frame),
                   astrometry=ASTROMETRY, detection_labels=LABELS, detection_config=payload['config'],
                   astrometry_median_shift_px=tile['astrometry_median_shift_px']))
        print(f"region {t['region']:03d} {t['tile']}: {len(frame)} detections, median shift {tile['astrometry_median_shift_px']:.3f} px", flush=True)


if __name__ == '__main__': main()
