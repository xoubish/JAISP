"""Measure every detection with the calibrated mixture photometer (image and foundation priors).

Each detection is the central source of its own 12-arcsec, ten-band scene;
only that central measurement is kept. Both priors share the rendered profile
banks, so the paired image-prior control costs little extra.
"""
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import torch
from .common import ROOT, OUT, PRIORS, PSF_CALIBRATION, region_dir, read_json, write_json
from .prepare import load_inputs, scene_for_source
from ..scene_features import SceneEncoder, image_features
from ..run_mixture import predict_prior
from ..mixture import fit_multiband, SCALES

STATE = {}


def _init(threads, scarlet_checkpoint=None, device=None):
    torch.set_num_threads(threads)
    STATE['sigmas'] = {b: v['sigma_px'] for b, v in read_json(ROOT / PSF_CALIBRATION).items()}
    STATE['inputs'] = {}
    if scarlet_checkpoint:
        from ..amortised_scarlet import AmortisedScarletPhotometry
        STATE['scarlet'] = AmortisedScarletPhotometry(ROOT / scarlet_checkpoint, device=device)
        return
    cp = torch.load(ROOT / PRIORS, map_location='cpu', weights_only=False)
    if not np.array_equal(cp['metadata']['scales_arcsec'], SCALES): raise ValueError('Checkpoint and renderer size dictionaries differ')
    STATE['cp'] = cp
    STATE['encoder'] = SceneEncoder(ROOT / cp['metadata']['original']['foundation_checkpoint'])


def _fits(scene):
    """{model: per-band results} for the configured photometer(s)."""
    if 'scarlet' in STATE:
        return {'scarlet': STATE['scarlet'](scene)}
    cp = STATE['cp']; meta = cp['metadata']
    item = dict(scene=scene, foundation=STATE['encoder'](scene), image=image_features(scene)); banks = {}; out = {}
    for mode in ('image', 'foundation'):
        prior = predict_prior(item, cp['heads'][mode], cp['population'], mode)
        out[mode] = fit_multiband(scene, prior, banks=banks, prior_precision=cp['heads'][mode].get('precision'),
                                  strength=meta['prior_strength'], band_strength=meta['band_strength'])
    return out


def measure(task):
    region, indices = task
    if region not in STATE['inputs']: STATE['inputs'] = {region: load_inputs(region_dir(region))}
    inputs = STATE['inputs'][region]; rows = []; failures = []
    for i in indices:
        scene, info = scene_for_source(inputs, i, STATE['sigmas'])
        if scene is None: failures.append(dict(region=region, source=int(i), reason=info)); continue
        try:
            for mode, fits in _fits(scene).items():
                for band, r in fits.items():
                    if band.startswith('_'): continue
                    conversion = 10 ** (.4 * (23.9 - float(inputs[band + '__magzero']))) if band.startswith('euclid') else np.nan
                    row = dict(region=region, source=int(i), band=band, model=mode, flux_native=r['flux'][0], error_native=r['error'][0],
                               flux_ujy=r['flux'][0] * conversion, error_ujy=r['error'][0] * conversion, footprint=r['footprint'][0],
                               reduced_chi2=r['reduced_chi2'], condition=r['condition'], n_scene_sources=info['n_sources'],
                               truncated=info['truncated'], valid_fraction=info['valid_fraction'].get(band, np.nan),
                               nearest_detection_arcsec=info['nearest_neighbor_arcsec'],
                               effective_scale_arcsec=float(r['weights'][0] @ SCALES) if 'weights' in r else np.nan)
                    if band == 'euclid_VIS' and 'weights' in r: row.update({f'w{k}': float(r['weights'][0, k]) for k in range(len(SCALES))})
                    rows.append(row)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failures.append(dict(region=region, source=int(i), reason=f'{type(exc).__name__}: {exc}'))
    return region, rows, failures


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--regions', type=int, nargs='*'); p.add_argument('--workers', type=int, default=32)
    p.add_argument('--threads', type=int, default=1); p.add_argument('--limit', type=int, help='sources per region, for smoke tests'); p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--scarlet', type=str, default='', help='Amortised-scarlet checkpoint: measure with it instead of the mixture priors (writes scarlet_fluxes.csv)')
    p.add_argument('--device', type=str, default=None); p.add_argument('--tag', type=str, default='', help='suffix for scarlet outputs, e.g. _epoch1')
    a = p.parse_args()
    output_name = f'scarlet{a.tag}_fluxes.csv' if a.scarlet else 'foundation_fluxes.csv'
    status_name = f'scarlet{a.tag}_photometry_status.json' if a.scarlet else 'photometry_status.json'
    folders = sorted(f.parent for f in OUT.glob('region_*/tile_inputs.npz'))
    if a.regions is not None: folders = [f for f in folders if int(f.name.split('_')[1]) in set(a.regions)]
    tasks = []; counts = {}
    for f in folders:
        region = int(f.name.split('_')[1])
        with np.load(f / 'tile_inputs.npz', allow_pickle=True) as z: usable = np.flatnonzero(z['scene_inside'] & z['central_valid']); n = len(z['sky'])
        if a.limit: usable = usable[:a.limit]
        counts[region] = dict(sources=int(n), usable=int(len(usable)))
        for start in range(0, len(usable), a.chunk): tasks.append((region, usable[start:start + a.chunk].tolist()))
    print(f'{len(folders)} regions, {sum(c["usable"] for c in counts.values())} usable sources, {len(tasks)} tasks', flush=True)
    results = {r: ([], []) for r in counts}
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.threads, a.scarlet, a.device)) as pool:
        for k, (region, rows, failures) in enumerate(pool.map(measure, tasks, chunksize=1)):
            results[region][0].extend(rows); results[region][1].extend(failures)
            if k % 20 == 0: print(f'task {k + 1}/{len(tasks)}', flush=True)
    status = []
    for region, (rows, failures) in results.items():
        pd.DataFrame(rows).to_csv(region_dir(region) / output_name, index=False)
        write_json(region_dir(region) / output_name.replace('fluxes.csv', 'photometry_failures.json'), failures)
        status.append(dict(region=region, **counts[region], measured=len({r['source'] for r in rows}), failed=len(failures)))
        print(status[-1], flush=True)
    write_json(OUT / status_name, status)


if __name__ == '__main__': main()
