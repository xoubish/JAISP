from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'models'))
DEFAULT_CONFIG = HERE / 'configs/unknown_regions_v1.json'


def read_config(path=DEFAULT_CONFIG):
    return json.loads(Path(path).read_text())


def resolve(path):
    return ROOT / path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def spaced(items, count):
    return [items[i] for i in np.linspace(0, len(items) - 1, min(count, len(items)), dtype=int)]


def split_tiles(cfg):
    tiles = sorted(p.stem.removesuffix('_aug0') for p in resolve(cfg['feature_dir']).glob('*_aug0.pt'))
    suffix = f"_patch_{cfg['excluded_patch']}"
    train = [t for t in tiles if not t.endswith(suffix)]
    val = [t for t in tiles if t.endswith(suffix)]
    if len(train) != cfg['expected_train_tiles'] or len(val) != cfg['expected_validation_tiles']:
        raise ValueError(f'Unexpected feature split: train={len(train)}, validation={len(val)}')
    return train, val


def load_seeds(cfg, tiles):
    """Read existing labels without modifying the shared feature cache."""
    vis = torch.load(resolve(cfg['vis_labels']), map_location='cpu', weights_only=False)
    nir = torch.load(resolve(cfg['nir_labels']), map_location='cpu', weights_only=False)
    if vis.get('label_version') != 3 or vis.get('nsig') != 3:
        raise ValueError('Expected the version-3, threshold-3 VIS label cache')
    seeds = {}
    for tid in tiles:
        xy = np.concatenate([vis['labels'][tid][0], nir['promoted'][tid]], axis=0).astype(np.float32)
        if not np.isfinite(xy).all() or (xy < 0).any() or (xy > 1).any():
            raise ValueError(f'Invalid normalized seed coordinates for {tid}')
        seeds[tid] = xy
    return seeds


def source_manifest(cfg):
    paths = list(HERE.glob('*.py')) + list((HERE / 'configs').glob('*.json'))
    paths += list(HERE.glob('*.sh'))
    paths += list((HERE / 'notebooks').glob('*.ipynb')) + list((HERE / 'tests').glob('*.py'))
    paths += [resolve(cfg[k]) for k in ('initial_checkpoint', 'vis_labels', 'nir_labels')]
    paths += [ROOT / 'models/detection' / name for name in
              ('centernet_detector.py', 'centernet_loss.py', 'train_centernet.py',
               'visnir_eval_experiment.py', 'dataset.py', 'masks.py')]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def online_run(cfg, out, job_type, name, extra=None):
    import wandb
    wc = cfg['wandb']
    if wc['mode'] != 'online':
        raise ValueError('This experiment requires live W&B monitoring')
    run = wandb.init(entity=wc['entity'], project=wc['project'], group=wc['group'],
                     name=name, job_type=job_type, mode='online', dir=str(out),
                     config={**cfg, **(extra or {})},
                     settings=wandb.Settings(init_timeout=45, disable_code=True, disable_git=True))
    write_json(Path(out) / 'wandb_run.json', {'id': run.id, 'url': run.url})
    print(f'W&B: {run.url}', flush=True)
    return run
