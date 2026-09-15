"""Freeze training-only proposals and label policy before either training arm."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .common import (DEFAULT_CONFIG, read_config, resolve, split_tiles, load_seeds,
                     source_manifest, spaced, write_json)
from .data import artifact_mask, novel_candidates, disk_mask
from detection.centernet_detector import CenterNetDetector
from detection.visnir_eval_experiment import predict_features


def prepare(cfg, out, device, limit_tiles=0):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', cfg)
    write_json(out / 'source_sha256.json', source_manifest(cfg))
    train, val = split_tiles(cfg)
    tiles = spaced(train, limit_tiles) if limit_tiles else train
    seeds = load_seeds(cfg, tiles)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('An accessible GPU is required for proposal inference')
    teacher = CenterNetDetector.load(str(resolve(cfg['initial_checkpoint'])), encoder=None, device=device).eval()
    labels, diagnostics = {}, []
    start = time.monotonic()
    for i, tid in enumerate(tiles):
        mask = artifact_mask(cfg, tid)
        h, w = mask.shape
        cached = torch.load(resolve(cfg['feature_dir']) / f'{tid}_aug0.pt', map_location='cpu', weights_only=True)
        if tuple(cached['aug_params']) != (0, False, False):
            raise ValueError(f'{tid}: aug0 must be the identity view')
        xy, scores = predict_features(teacher, cached['features'][None].to(device), mask, (h, w),
                                      floor=cfg['proposal_floor'])
        novel = novel_candidates(xy, seeds[tid] * [w - 1, h - 1], cfg['label_match_radius_vis_px'])
        unknown = (xy[novel] / [w - 1, h - 1]).astype(np.float32)
        fraction = float(disk_mask(unknown, (h, w), cfg['ignore_radius_vis_px']).mean())
        if fraction > cfg['max_ignore_fraction']:
            raise ValueError(f'{tid}: ignore mask covers {fraction:.1%}; inspect before proceeding')
        labels[tid] = {'seeds': seeds[tid], 'unknown': unknown, 'shape': (h, w),
                       'proposal_xy': xy.astype(np.float32), 'proposal_scores': scores.astype(np.float32)}
        row = {'tile_id': tid, 'seeds': len(seeds[tid]), 'proposals': len(xy),
               'unknown_candidates': int(novel.sum()), 'ignored_fraction': fraction}
        diagnostics.append(row)
        print(f'[{i+1}/{len(tiles)}] {tid}: {row["seeds"]} seeds, '
              f'{row["unknown_candidates"]} unknown, masked {fraction:.2%}', flush=True)
    payload = {'config': cfg, 'tiles': labels, 'complete': tiles == train,
               'validation_tiles': spaced(val, cfg['validation_tiles'])}
    torch.save(payload, out / 'labels.pt')
    summary = {'complete_training_split': tiles == train, 'n_tiles': len(tiles),
               'n_seeds': sum(r['seeds'] for r in diagnostics),
               'n_unknown': sum(r['unknown_candidates'] for r in diagnostics),
               'mean_ignored_fraction': float(np.mean([r['ignored_fraction'] for r in diagnostics])),
               'elapsed_seconds': time.monotonic() - start, 'tiles': diagnostics}
    write_json(out / 'summary.json', summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit-tiles', type=int, default=0, help='Diagnostic subset; cannot be used for full training')
    args = p.parse_args()
    torch.set_num_threads(4)
    prepare(read_config(args.config), args.out, torch.device(args.device), args.limit_tiles)


if __name__ == '__main__':
    main()
