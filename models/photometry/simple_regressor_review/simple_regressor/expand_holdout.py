"""Evaluate frozen pilot models on all tiles in the original guarded test region.

Keeps the saved RA boundary fixed. Selects one most-interior stamp per object,
checks training/validation IDs, and streams tile batches without a large cache.
"""
import argparse
import csv
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from .config import Config
from .catalog import load_sources
from .geometry import parse_wcs, world_to_pixel, cut_stamp, border_distance
from .evaluate import _load, predict, _metrics
from .data import StampDataset
from torch.utils.data import DataLoader


def image_shape(path, key):
    with zipfile.ZipFile(path) as z, z.open(key + '.npy') as f:
        version = np.lib.format.read_magic(f)
        reader = { (1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0 }[version]
        return reader(f)[0]


def expand(before_dir, after_dir, output):
    before_dir, after_dir, output = map(Path, (before_dir, after_dir, output))
    meta = json.loads((after_dir / 'metadata.json').read_text())
    info = meta['split_info']
    if info['mode'] != 'ra':
        raise ValueError('This evaluator requires the saved RA split')
    cutoff = info['ra_boundary_2'] + info['buffer_deg']
    cfg = Config(**meta['config'])
    src = load_sources(cfg)
    selected = np.flatnonzero(src.ra > cutoff)
    forbidden = set()
    for folder in (before_dir, after_dir):
        with np.load(folder / 'stamps_cache.npz') as z:
            forbidden.update(z['object_id'][z['split'] != 'test'].tolist())
    assert not forbidden.intersection(src.object_id[selected].tolist())
    assignments = {}
    files = sorted(Path(cfg.tiles_root).rglob(cfg.tile_glob))
    for path in files:
        with np.load(path, allow_pickle=True) as z:
            wcs = parse_wcs(z[cfg.wcs_key])
        shape = image_shape(path, cfg.img_key)
        x, y = world_to_pixel(wcs, src.ra[selected], src.dec[selected])
        half = cfg.stamp // 2
        inside = np.flatnonzero((x >= half) & (x < shape[1]-half) &
                                (y >= half) & (y < shape[0]-half))
        for j in inside:
            i = int(selected[j])
            distance = border_distance(x[j], y[j], shape, cfg.stamp)
            if i not in assignments or distance > assignments[i][0]:
                assignments[i] = (distance, path, float(x[j]), float(y[j]))
    by_tile = {}
    for i, (_, path, x, y) in assignments.items():
        by_tile.setdefault(path, []).append((i, x, y))
    print(f'[holdout] fixed RA > {cutoff:.9f}; {len(assignments)} candidates in {len(by_tile)} tiles', flush=True)
    models = [_load(folder, 'best.pt') for folder in (before_dir, after_dir)]
    torch.set_num_threads(4)
    # One fixed 1.5-arcsec aperture, with only a training-derived scalar calibration.
    model, model_cfg, _ = models[1]
    with np.load(after_dir / 'stamps_cache.npz') as z:
        train_cache = {k: z[k] for k in ('stamps','flux','fluxerr','mag','split')}
    ds = StampDataset(train_cache, train_cache['split'] == 'train', model_cfg.input_scale,
                      model_cfg.bin_factor, model_cfg.centre_sigma_frac, model_version=2)
    ratios = []
    with torch.no_grad():
        for _, linear, flux, _ in DataLoader(ds, batch_size=64):
            aperture, _ = model.aperture_features(linear)
            good = aperture[:, -1] > 0
            ratios.extend((flux[good] / aperture[good, -1]).tolist())
    single_scale = float(np.median(ratios))
    del train_cache, ds
    rows = [[], []]
    S = cfg.stamp
    yy, xx = np.mgrid[:S, :S]
    radius = np.hypot(xx-(S-1)/2, yy-(S-1)/2)
    sky = (radius > .86*S/2) & (radius < .98*S/2)
    rejected = 0
    for number, (path, sources) in enumerate(by_tile.items(), 1):
        with np.load(path, allow_pickle=True) as z:
            image, variance = np.asarray(z[cfg.img_key], 'f4'), np.asarray(z[cfg.var_key], 'f4')
        stamps, indices = [], []
        for i, x, y in sources:
            im, var = cut_stamp(image, x, y, S), cut_stamp(variance, x, y, S)
            if im is None or var is None:
                rejected += 1; continue
            valid = np.isfinite(im) & np.isfinite(var) & (var > 0)
            if valid.mean() < cfg.valid_frac_min or not (valid & sky).any():
                rejected += 1; continue
            stamps.append(np.stack([np.where(valid, im, 0), np.sqrt(np.where(valid, var, 0))]))
            indices.append(i)
        if not indices:
            continue
        indices = np.asarray(indices)
        cache = dict(stamps=np.asarray(stamps, 'f4'), flux=src.flux[indices],
                     fluxerr=src.fluxerr[indices], mag=src.mag[indices])
        for k, (model, model_cfg, scaler) in enumerate(models):
            predictions = predict(model, scaler, cache, np.ones(len(indices), bool), model_cfg, 'cpu')
            if not np.isfinite(predictions).all():
                raise ValueError('Nonfinite predictions')
            if k == 1:
                aperture_predictions = predict(model, scaler, cache, np.ones(len(indices), bool), model_cfg, 'cpu', baseline=True)
                ds = StampDataset(cache, np.ones(len(indices), bool), model_cfg.input_scale,
                                  model_cfg.bin_factor, model_cfg.centre_sigma_frac, model_version=2)
                single_predictions = []
                with torch.no_grad():
                    for _, linear, _, _ in DataLoader(ds, batch_size=64):
                        aperture, _ = model.aperture_features(linear)
                        single_predictions.extend((single_scale * aperture[:, -1]).tolist())
            for j, (i, pred) in enumerate(zip(indices, predictions)):
                rows[k].append(dict(object_id=int(src.object_id[i]), mag=float(src.mag[i]),
                                    flux_true_ujy=float(src.flux[i]), fluxerr_ujy=float(src.fluxerr[i]),
                                    flux_pred_ujy=float(pred), ra=float(src.ra[i]), dec=float(src.dec[i])))
                if k == 1:
                    rows[k][-1].update(aperture_flux_ujy=float(aperture_predictions[j]),
                                       single_aperture_flux_ujy=float(single_predictions[j]))
        if number % 20 == 0 or number == len(by_tile):
            print(f'[holdout] {number}/{len(by_tile)} tiles; {len(rows[0])} sources', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for label, result in zip(('before', 'after'), rows):
        result.sort(key=lambda r: r['object_id'])
        if not result:
            raise ValueError('No usable holdout sources')
        with (output / f'predictions_{label}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(result[0])); writer.writeheader(); writer.writerows(result)
        arrays = [np.array([r[key] for r in result]) for key in ('flux_pred_ujy','flux_true_ujy','fluxerr_ujy','mag')]
        metrics[label] = _metrics(*arrays)
    assert [r['object_id'] for r in rows[0]] == [r['object_id'] for r in rows[1]]
    (output / 'metadata.json').write_text(json.dumps(dict(
        fixed_test_ra_min=cutoff, original_split=info, n_sources=len(rows[0]),
        rejected_stamps=rejected, tiles_considered=len(files), tiles_used=len(by_tile),
        before_checkpoint=str(before_dir/'best.pt'), after_checkpoint=str(after_dir/'best.pt'),
        forbidden_train_val_ids=len(forbidden), metrics=metrics,
        single_aperture_radius_arcsec=model_cfg.aperture_radii_arcsec[-1],
        single_aperture_training_scale=single_scale,
        note='Frozen models; existing test-region boundary and guard retained. No retraining or selection.'), indent=2))
    print(f'[holdout] wrote {len(rows[0])} matched predictions to {output}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before-dir', required=True)
    ap.add_argument('--after-dir', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    expand(args.before_dir, args.after_dir, args.output)
