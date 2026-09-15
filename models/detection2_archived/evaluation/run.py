"""Reproducible final-epoch comparison on all cached patch-25 tiles.

Prepare on CPU while training runs. With --wait, GPU inference starts only
after the paired launcher has exited successfully and both final checkpoints
and final validation records exist.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..common import ROOT, digest, resolve, split_tiles, write_json
from .geometry import Tile, owners, recovered, tangent, regions
from .statistics import (ARMS, LABELS, GROUPS, THRESHOLDS, block_counts,
                         summarize, discordances)

DEFAULT_STUDY = ROOT / 'models/detection2/runs/unknown_regions_20260915_v1_12ep'
CATALOG = ROOT / 'data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits'


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def status(out, stage, **extra):
    record = {'stage': stage, 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
    write_json(out / 'status.json', record)
    print(json.dumps(record), flush=True)


def load_prepared(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    metadata = json.loads((out / 'geometry.json').read_text())
    tiles = []
    for row in metadata:
        with np.load(out / 'geometry' / (row['name']+'.npz')) as z:
            mask = z['mask']
        tiles.append(Tile(row['name'], WCS(fits.Header.fromstring(row['header'], sep='\n')), mask))
    with np.load(out / 'references.npz', allow_pickle=False) as z:
        references = dict(z)
    return protocol, tiles, references


def prepare(study, out):
    cfg = json.loads((study / 'control/config.json').read_text())
    inputs = {str(p): digest(p) for p in [CATALOG, resolve(cfg['gaia']), study/'control/config.json',
                                         study/'unknown/config.json']}
    if (out / 'protocol.json').exists():
        protocol, tiles, refs = load_prepared(out)
        if protocol['input_sha256'] != inputs:
            raise ValueError('Prepared validation inputs changed; use a fresh output directory')
        return protocol, tiles, refs
    from detection.validation_utils import _wcs_vis
    from detection.masks import bright_star_saturation_mask, load_gaia_cache
    _, validation_tiles = split_tiles(cfg)
    cat = fits.getdata(CATALOG)
    if len(np.unique(cat['object_id'])) != len(cat):
        raise ValueError('MER object_id must identify unique catalogue rows')
    ra, dec = np.asarray(cat['ra'], float), np.asarray(cat['dec'], float)
    if not (np.isfinite(ra).all() and np.isfinite(dec).all()):
        raise ValueError('Nonfinite catalogue coordinates')
    clean, vis = np.asarray(cat['spurious_flag']) != 1, np.asarray(cat['vis_det']) == 1
    mag = np.asarray(cat['mag_vis'], float)
    groups = np.column_stack([clean & vis & np.isfinite(mag) & (mag < 24.5),
                              clean & vis, clean & ~vis, clean])
    gaia = load_gaia_cache(str(resolve(cfg['gaia'])))
    tiles, metadata, corners, skipped = [], [], [], []
    owner, best = np.full(len(cat), -1, np.int32), np.full(len(cat), -np.inf)
    pooled = np.zeros(len(GROUPS), dtype=int)
    for i, tid in enumerate(validation_tiles):
        path = resolve(cfg['euclid_dir']) / f'{tid}_euclid.npz'
        with np.load(path, allow_pickle=True) as ed:
            img, header = np.nan_to_num(ed['img_VIS']), str(ed['wcs_VIS'])
            wcs = _wcs_vis(ed)
        tile = Tile(tid, wcs, bright_star_saturation_mask(img, header, gaia))
        distance = tile.edge_distance(ra, dec)
        eligible = np.isfinite(distance)
        if (groups[:, 1] & eligible).sum() < 2:
            skipped.append(tid)
            continue
        index = len(tiles)
        take = distance > best
        owner[take], best[take] = index, distance[take]
        pooled += groups[eligible].sum(0)
        tiles.append(tile)
        metadata.append({'name': tid, 'header': wcs.to_header(relax=True).tostring(sep='\n'),
                         'input_path': str(path), 'input_size': path.stat().st_size,
                         'input_mtime_ns': path.stat().st_mtime_ns})
        save_npz(out / 'geometry' / f'{tid}.npz', mask=tile.mask)
        h, w = tile.mask.shape
        cr, cd = wcs.all_pix2world([4, w-4, w-4, 4], [4, 4, h-4, h-4], 0)
        corners.extend(zip(cr, cd))
        if (i+1) % 12 == 0 or i+1 == len(validation_tiles):
            status(out, 'preparing_geometry', tiles_done=i+1, tiles_total=len(validation_tiles))
    select = owner >= 0
    corners = np.asarray(corners)
    center = np.median(corners, axis=0).tolist()
    plane = tangent(corners[:, 0], corners[:, 1], center)
    frame = {'center_deg': center, 'low_arcmin': plane.min(0).tolist(), 'high_arcmin': plane.max(0).tolist()}
    refs = {'row': np.flatnonzero(select), 'object_id': np.asarray(cat['object_id'][select]),
            'ra': ra[select], 'dec': dec[select], 'groups': groups[select], 'owner': owner[select],
            'full_catalog_ra': ra, 'full_catalog_dec': dec}
    if refs['object_id'].dtype.kind == 'O':
        refs['object_id'] = refs['object_id'].astype(str)
    save_npz(out / 'references.npz', **refs)
    write_json(out / 'geometry.json', metadata)
    protocol = {
        'version': 1, 'study': str(study), 'input_sha256': inputs,
        'requested_tiles': validation_tiles, 'scored_tiles': [t.name for t in tiles], 'skipped_tiles': skipped,
        'thresholds': list(THRESHOLDS), 'primary_threshold': .30, 'checkpoint_selection': 'fixed final epoch',
        'groups': list(GROUPS), 'group_totals_unique': dict(zip(GROUPS, groups[select].sum(0).tolist())),
        'group_totals_tile_pooled': dict(zip(GROUPS, pooled.tolist())), 'frame': frame,
        'bootstrap': {'primary_grid': 4, 'sensitivity_grid': 3, 'replicates': 5000, 'seed': 20260915},
        'ownership': 'Largest distance to the four-pixel edge among unmasked eligible tiles; sorted-order tie break. '
                     'Same sky partition for catalogue references and predicted centroids, independent of model and score.',
        'matching': 'Spherical nearest-neighbour separation strictly below 0.5 arcsec; not one-to-one. '
                    'Completeness uses unique clean MER IDs; agreement uses full MER including flagged.',
        'scope': 'Development evaluation within one patch; approximate pointwise spatial bootstrap intervals. '
                 'No uncertainty over training seeds, other fields or catalogue truth. Grid-size sensitivity is reported. '
                 'These partitioned-sky metrics differ from tile-pooled training monitoring.',
    }
    write_json(out / 'protocol.json', protocol)
    status(out, 'prepared', scored_tiles=len(tiles), unique_reference_counts=protocol['group_totals_unique'])
    return protocol, tiles, refs


def wait_for_final(study, out, wait):
    started = time.monotonic()
    while True:
        cfgs = [json.loads((study / a / 'config.json').read_text()) for a in ('control', 'unknown')]
        if cfgs[0]['epochs'] != cfgs[1]['epochs']:
            raise ValueError('Different epoch budgets')
        epochs = cfgs[0]['epochs']
        exit_path = study / 'exit_status.json'
        if exit_path.exists():
            codes = json.loads(exit_path.read_text())
            if codes != {'control_exit_code': 0, 'unknown_exit_code': 0}:
                raise RuntimeError(f'Training did not finish successfully: {codes}')
            if not all((study/a/'final.pt').exists() and (study/a/f'validation_epoch_{epochs:02d}.json').exists()
                       for a in ('control', 'unknown')):
                raise RuntimeError('Successful launcher exit without complete final checkpoints/validation')
            return cfgs[0]
        if not wait:
            raise RuntimeError('Final checkpoints are not ready; use --wait or --prepare-only')
        if time.monotonic()-started > 6*3600:
            raise TimeoutError('Waited six hours; restart explicitly')
        status(out, 'waiting_for_training', completed_epochs={a: max(
            int(p.stem.rsplit('_', 1)[1]) for p in (study/a).glob('validation_epoch_*.json'))
            for a in ('control', 'unknown')})
        time.sleep(30)


def infer(study, out, protocol, tiles, cfg, device_name):
    import torch
    from detection.centernet_detector import CenterNetDetector
    from detection.visnir_eval_experiment import predict_features
    torch.set_num_threads(4)
    device = torch.device(device_name)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('GPU is required for full inference')
    checkpoints = [resolve(cfg['initial_checkpoint']), study/'control/final.pt', study/'unknown/final.pt']
    hashes = {arm: {'path': str(p), 'sha256': digest(p)} for arm, p in zip(ARMS, checkpoints)}
    manifest_path = out / 'inference_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['checkpoints'] != hashes:
        raise ValueError('Checkpoint changed; use a new output directory')
    code_paths = list(Path(__file__).parent.glob('*.py')) + [ROOT/'models/detection'/p for p in
                 ('centernet_detector.py', 'visnir_eval_experiment.py', 'train_centernet.py')]
    manifest = {'checkpoints': hashes, 'source_sha256': {str(p): digest(p) for p in code_paths},
                'protocol_sha256': digest(out/'protocol.json'), 'device': device_name,
                'prepared_sha256': {str(p.relative_to(out)): digest(p) for p in
                                    [out/'references.npz', out/'geometry.json'] + sorted((out/'geometry').glob('*.npz'))},
                'precision': 'float32 (same as per-epoch validation)', 'features': {}}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ('source_sha256', 'protocol_sha256', 'prepared_sha256'):
            if old[key] != manifest[key]:
                raise ValueError(f'Inference {key} changed; use a new output directory')
        manifest['features'] = old['features']
    write_json(manifest_path, manifest)
    models = [CenterNetDetector.load(str(p), encoder=None, device=device).eval() for p in checkpoints]
    for i, tile in enumerate(tiles):
        paths = [out/'predictions'/f'{arm}__{tile.name}.npz' for arm in ARMS]
        feature_path = resolve(cfg['feature_dir']) / f'{tile.name}_aug0.pt'
        feature_hash = digest(feature_path)
        if tile.name in manifest['features'] and manifest['features'][tile.name] != feature_hash:
            raise ValueError('Cached features changed')
        manifest['features'][tile.name] = feature_hash
        write_json(manifest_path, manifest)
        if all(p.exists() for p in paths):
            continue
        cached = torch.load(feature_path, map_location='cpu', weights_only=True)
        if tuple(cached['aug_params']) != (0, False, False):
            raise ValueError('Expected identity validation features')
        features = cached['features'][None].to(device)
        for model, path in zip(models, paths):
            if not path.exists():
                xy, scores = predict_features(model, features, tile.mask, tile.mask.shape, floor=.10)
                save_npz(path, xy=xy, scores=scores)
        if (i+1) % 12 == 0 or i+1 == len(tiles):
            status(out, 'predicting', tiles_done=i+1, tiles_total=len(tiles))
    del models, features
    torch.cuda.empty_cache()


def merge_predictions(tiles, refs, paths):
    all_ra, all_dec, all_score, origins = [], [], [], []
    for i, (tile, path) in enumerate(zip(tiles, paths)):
        with np.load(path, allow_pickle=False) as z:
            xy, scores = z['xy'], z['scores']
        if not np.isfinite(xy).all() or not np.isfinite(scores).all():
            raise ValueError('Nonfinite prediction')
        ra, dec = tile.wcs.all_pix2world(xy[:, 0], xy[:, 1], 0)
        all_ra.append(ra); all_dec.append(dec); all_score.append(scores)
        origins.append(np.full(len(ra), i, dtype=np.int32))
    ra, dec, scores, origin = map(np.concatenate, (all_ra, all_dec, all_score, origins))
    keep = owners(tiles, ra, dec) == origin
    ra, dec, scores, origin = ra[keep], dec[keep], scores[keep], origin[keep]
    matched = recovered(ra, dec, refs['full_catalog_ra'], refs['full_catalog_dec'])
    return {'ra': ra, 'dec': dec, 'scores': scores, 'origin_tile': origin, 'matched': matched}


def check_monitor_counts(study, tiles, refs, paths):
    """Archived starting predictions must reproduce the saved monitoring counts.

    This uses the original tile-pooled/pixel matching rule as a separate check,
    before applying the new partitioned-sky protocol.
    """
    from scipy.spatial import cKDTree
    saved = json.loads((study/'control/validation_epoch_00.json').read_text())
    totals = {k: {g: 0 for g in GROUPS} for k in saved['counts']}
    hits = {k: {g: 0 for g in GROUPS} for k in saved['counts']}
    n_det = {k: 0 for k in saved['counts']}
    n_match = {k: 0 for k in saved['counts']}
    for tile, path in zip(tiles, paths):
        if tile.name not in saved['tiles_scored']:
            continue
        x, y = tile.wcs.all_world2pix(refs['ra'], refs['dec'], 0)
        select = np.isfinite(tile.edge_distance(refs['ra'], refs['dec']))
        pts = np.column_stack([x[select], y[select]])
        fx, fy = tile.wcs.all_world2pix(refs['full_catalog_ra'], refs['full_catalog_dec'], 0)
        h, w = tile.mask.shape
        inside = np.isfinite(fx) & np.isfinite(fy) & (fx >= 4) & (fx < w-4) & (fy >= 4) & (fy < h-4)
        full = cKDTree(np.column_stack([fx[inside], fy[inside]]))
        with np.load(path, allow_pickle=False) as z:
            xy, scores = z['xy'], z['scores']
        for conf in saved['counts']:
            det = xy[scores >= float(conf)]
            found = cKDTree(det).query(pts)[0] < 5 if len(det) else np.zeros(len(pts), bool)
            for j, group in enumerate(GROUPS):
                gs = refs['groups'][select, j]
                totals[conf][group] += int(gs.sum())
                hits[conf][group] += int(found[gs].sum())
            n_det[conf] += len(det)
            n_match[conf] += int((full.query(det)[0] < 5).sum()) if len(det) else 0
    for conf, saved_counts in saved['counts'].items():
        actual = {'total': totals[conf], 'recovered': hits[conf], 'n_det': n_det[conf], 'n_matched': n_match[conf]}
        if actual != saved_counts:
            raise ValueError(f'Archived baseline does not reproduce monitoring at {conf}: {actual} vs {saved_counts}')
    return {'monitoring_tiles': len(saved['tiles_scored']), 'thresholds': list(saved['counts']),
            'all_reference_recovery_and_detection_counts_match': True}


def analyze(out, protocol, tiles, refs, prediction_paths=None, suffix=''):
    detections, hits = [], []
    for arm in ARMS:
        paths = prediction_paths[arm] if prediction_paths else [out/'predictions'/f'{arm}__{t.name}.npz' for t in tiles]
        det = merge_predictions(tiles, refs, paths)
        detections.append(det)
        save_npz(out/f'{arm}_catalog{suffix}.npz', **det)
        arm_hits = []
        for threshold in THRESHOLDS:
            select = det['scores'] >= threshold
            arm_hits.append(recovered(refs['ra'], refs['dec'], det['ra'][select], det['dec'][select]))
        hits.append(arm_hits)
        status(out, 'matching_unique_sources'+suffix, model=arm, owned_detections=len(det['scores']))
    hits = np.asarray(hits)
    save_npz(out/f'source_recovery{suffix}.npz', hits=hits, object_id=refs['object_id'], groups=refs['groups'],
             arms=np.asarray(ARMS), thresholds=np.asarray(THRESHOLDS))
    result = {'protocol': protocol, 'labels': dict(zip(ARMS, LABELS)), 'grids': {},
              'paired_source_counts': discordances(hits, refs['groups'])}
    settings = protocol['bootstrap']
    for side in (settings['primary_grid'], settings['sensitivity_grid']):
        ref_regions = regions(refs['ra'], refs['dec'], protocol['frame'], side)
        det_regions = [regions(d['ra'], d['dec'], protocol['frame'], side) for d in detections]
        numer, denom = block_counts(hits, refs['groups'], ref_regions, detections, det_regions, side)
        summary, weights = summarize(numer, denom, settings['replicates'], settings['seed'])
        summary['region_reference_counts'] = np.bincount(ref_regions[refs['groups'][:, -1]], minlength=side*side).tolist()
        summary['region_size_arcmin'] = ((np.asarray(protocol['frame']['high_arcmin'])-
                                         protocol['frame']['low_arcmin'])/side).tolist()
        result['grids'][str(side)] = summary
        save_npz(out/f'bootstrap_{side}x{side}{suffix}.npz', numerators=numer, denominators=denom, weights=weights)
    write_json(out/f'results{suffix}.json', result)
    from .present import present
    present(out, result, suffix)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    p.add_argument('--out', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--wait', action='store_true')
    p.add_argument('--publish', action='store_true')
    p.add_argument('--check-saved-baseline', type=Path,
                   help='CPU execution check with archived visnir__tile predictions, no new model result')
    args = p.parse_args()
    study = args.study.resolve(); out = (args.out or study/'full_validation').resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            protocol, tiles, refs = prepare(study, out)
            if args.check_saved_baseline:
                paths = [args.check_saved_baseline/f'visnir__{t.name}.npz' for t in tiles]
                if not all(p.exists() for p in paths):
                    raise ValueError('Missing archived starting-model predictions')
                check = check_monitor_counts(study, tiles, refs, paths)
                result = analyze(out, protocol, tiles, refs, {a: paths for a in ARMS}, suffix='_execution_check')
                for grid in result['grids'].values():
                    for metrics in grid['paired_differences'].values():
                        for row in metrics.values():
                            for value in row.values():
                                if value['difference_pp'] is not None:
                                    assert value['difference_pp'] == 0
                                    if value['ci95_percentile'] is not None:
                                        assert value['ci95_percentile'] == [0, 0]
                check.update({'identical_models_have_zero_paired_difference_and_interval': True,
                              'input_prediction_sha256': {str(p): digest(p) for p in paths},
                              'kind': 'CPU execution check only; final inference uses actual final checkpoints'})
                write_json(out/'execution_check.json', check)
                status(out, 'cpu_execution_check_passed', note='Identical archived baseline in all slots; not final results')
                return
            if args.prepare_only:
                return
            cfg = wait_for_final(study, out, args.wait)
            infer(study, out, protocol, tiles, cfg, args.device)
            result = analyze(out, protocol, tiles, refs)
            if args.publish:
                from .publish import publish
                publish(study, out, result)
            status(out, 'complete', results=str(out/'results.json'))
        except BaseException as exc:
            status(out, 'failed', exception=type(exc).__name__, message=str(exc))
            raise


if __name__ == '__main__':
    main()
