"""Truth-scored training and validation scenes from the amortised-scarlet training tiles.

Stages (run in order, from the project root):

    split        whole training tiles -> train / validation regions
    mer          MER query + match per region (needed for star exclusion and neighbour masking)
    library      empirical donors per split (empirical.make_library, regions restricted)
    backgrounds  blank-sky patches per split (empirical.make_backgrounds, regions restricted)
    export       randomised injected scenes with per-source truth flux, truth profile and oracle error
    real         central-source scenes from the validation tiles, with MER reference fluxes
    reference    calibrated mixture photometer on the validation injections and real scenes

Donors and sky come only from the training tiles, which were selected to exclude
the 28 detection-catalog test tiles; the 128-blend benchmark, the empirical pilot
and the detection-catalog comparison stay untouched as frozen tests.
"""
import argparse
import hashlib
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

from ..core import BANDS
from .. import empirical as emp

HERE = Path(__file__).resolve().parent
TILES = HERE.parent / 'runs/amortised_scarlet/training_tiles'
OUT = Path(os.environ.get('JAISP_TRUTH_SCARLET_OUT', HERE.parent / 'runs/truth_scarlet'))
SEED = 20261009
EUCLID = tuple(b for b in BANDS if b.startswith('euclid_'))


def regions_of(split):
    return json.loads((OUT / 'split.json').read_text())[split]


def make_split(n_val=8):
    regions = sorted(int(f.parent.name.split('_')[1]) for f in TILES.glob('region_*/tile_inputs.npz'))
    rng = np.random.default_rng(SEED); val = sorted(int(r) for r in rng.choice(regions, n_val, replace=False))
    split = dict(train=[r for r in regions if r not in val], val=val, tiles=str(TILES), seed=SEED)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'split.json').write_text(json.dumps(split, indent=2))
    print(f"{len(split['train'])} train regions, {len(val)} validation regions {val}")


def require_mer(regions):
    missing = [r for r in regions if not (TILES / f'region_{r:03d}/mer_match.csv').exists()]
    if missing: raise SystemExit(f'MER match missing for regions {missing}: run the `mer` stage until it completes')


def match_mer():
    os.environ['JAISP_DETCAT_OUT'] = str(TILES)
    from ..detcat.mer import match_region
    for folder in sorted(f.parent for f in TILES.glob('region_*/tile_inputs.npz')):
        if (folder / 'mer_match.csv').exists(): continue
        print(match_region(folder), flush=True)


# ----------------------------------------------------------------------------- export
def _design(rng):
    """Random scene design; everything is drawn before any rendering."""
    kind = rng.choice(['isolated', 'pair', 'triple'], p=[.35, .55, .10])
    n = dict(isolated=1, pair=2, triple=3)[kind]
    offsets = [rng.uniform(-.05, .05, 2)]
    for _ in range(n - 1):
        sep = rng.uniform(.3, 2.); angle = rng.uniform(0, 2 * np.pi)
        offsets.append(offsets[0] + sep * np.array([np.cos(angle), np.sin(angle)]))
    offsets = np.array(offsets)
    return dict(kind=str(kind), n=n, offsets=offsets,
                snr=float(np.exp(rng.uniform(np.log(1.5), np.log(80.)))),
                ratios=np.exp(rng.uniform(np.log(.1), np.log(10.), n - 1)),
                angle=float(rng.uniform(0, 2 * np.pi)),
                broadening_euclid=float(rng.uniform(0, .05)), broadening_rubin=float(rng.uniform(0, .2)),
                min_sep=float(np.min(np.linalg.norm(offsets[1:] - offsets[0], axis=1))) if n > 1 else np.inf)


def _export_one(task):
    split_dir, scene_id, donors, backgrounds, split_seed = task
    rng = np.random.default_rng([SEED, split_seed, scene_id])
    design = _design(rng)
    picks = rng.choice(len(donors), design['n'], replace=False)
    ds = [emp.read_npz(donors[k]) for k in picks]
    bg_id = int(rng.integers(len(backgrounds))); bg = emp.read_npz(backgrounds[bg_id])
    offsets = design['offsets']
    sky = bg['sky'] + offsets / np.array([3600 * np.cos(np.deg2rad(bg['sky'][1])), 3600])
    arrays = dict(sky=sky); profiles = {}; kernels = {}; positions = {}
    for b in BANDS:
        matrix = bg[b + '__matrix']; center = bg[b + '__center']; shape = bg[b + '__image'].shape
        positions[b] = center + np.linalg.solve(matrix, offsets.T).T
        broadening = design['broadening_euclid'] if b.startswith('euclid_') else design['broadening_rubin']
        ksize = 65 if b.startswith('euclid_') else 49
        pp, kk = [], []
        for j, d in enumerate(ds):
            pp.append(emp.render(d[b + '__latent'], d[b + '__matrix'], matrix, positions[b][j], shape, kernel=d[b + '__kernel'],
                                 angle=design['angle'], extra_sigma_arcsec=broadening, donor_center=d[b + '__center'],
                                 kernel_matrix=d[b + '__kernel_matrix']))
            kk.append(emp.render(d[b + '__kernel'], d[b + '__kernel_matrix'], matrix, [(ksize - 1) / 2] * 2, (ksize, ksize),
                                 angle=design['angle'], extra_sigma_arcsec=broadening))
        profiles[b] = np.array(pp); kernels[b] = np.array([emp.unit(k) for k in kk])
    _, error = emp.oracle_fit(np.zeros_like(bg['euclid_VIS__image']), bg['euclid_VIS__variance'], bg['euclid_VIS__mask'], profiles['euclid_VIS'][:1])
    fvis = design['snr'] * error[0]
    scales = [fvis / float(ds[0]['euclid_VIS__flux'])] + [r * fvis / float(d['euclid_VIS__flux']) for r, d in zip(design['ratios'], ds[1:])]
    rows = []
    for b in BANDS:
        truth = np.array([s * float(d[b + '__flux']) for s, d in zip(scales, ds)])
        image = bg[b + '__image'].astype(float) + np.einsum('n,nhw->hw', truth, profiles[b])
        variance = bg[b + '__variance'].astype(float); mask = bg[b + '__mask']
        k0 = kernels[b][0]; sigma = np.sqrt(np.sum(k0 * (np.indices(k0.shape)[0] - (k0.shape[0] - 1) / 2) ** 2))
        oflux, oerr = emp.oracle_fit(image, variance, mask, list(profiles[b]))
        arrays.update({b + '__image': image.astype('float32'), b + '__variance': variance.astype('float32'), b + '__mask': mask,
                       b + '__positions': positions[b], b + '__sky_to_pixel': np.linalg.inv(bg[b + '__matrix']),
                       b + '__psf_sigma': float(sigma), b + '__psf_kernels': kernels[b].astype('float32'),
                       'truth__' + b + '__flux': truth, 'truth__' + b + '__error': oerr,
                       'truth__' + b + '__profiles': profiles[b].astype('float32'),
                       'truth__' + b + '__weak': np.array([bool(d[b + '__fallback']) for d in ds])})
        for j in range(len(ds)):
            rows.append(dict(scene=scene_id, source=j, band=b, truth_flux=truth[j], oracle_error=oerr[j], oracle_flux=oflux[j],
                             true_snr=truth[j] / oerr[j], weak_band=bool(ds[j][b + '__fallback'])))
    np.savez_compressed(split_dir / 'scenes' / f'scene_{scene_id:05d}.npz', **arrays)
    manifest = dict(scene=scene_id, kind=design['kind'], n_sources=design['n'], vis_snr=design['snr'], min_sep_arcsec=design['min_sep'],
                    ratios=';'.join(f'{r:.3f}' for r in design['ratios']), rotation_rad=design['angle'],
                    broadening_euclid=design['broadening_euclid'], broadening_rubin=design['broadening_rubin'],
                    donors=';'.join(str(int(d['donor'])) for d in ds), background=bg_id)
    return manifest, rows


def export(split, count, workers):
    split_dir = OUT / split; (split_dir / 'scenes').mkdir(parents=True, exist_ok=True)
    donors = sorted((split_dir / 'donors').glob('donor_*.npz'))
    quality = pd.read_csv(split_dir / 'donor_quality.csv').set_index('donor')
    donors = [p for p in donors if bool(quality.loc[int(p.stem.split('_')[1]), 'suitable_primary'])]
    backgrounds = sorted((split_dir / 'backgrounds').glob('background_*.npz'))
    print(f'{split}: {len(donors)} suitable donors, {len(backgrounds)} backgrounds, {count} scenes', flush=True)
    split_seed = {'train': 1, 'val': 2}[split]
    tasks = [(split_dir, i, donors, backgrounds, split_seed) for i in range(count)
             if not (split_dir / 'scenes' / f'scene_{i:05d}.npz').exists()]
    manifests, rows = [], []; t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, (m, r) in enumerate(pool.map(_export_one, tasks, chunksize=4)):
            manifests.append(m); rows.extend(r)
            if (k + 1) % 500 == 0: print(f'{split}: {k + 1}/{len(tasks)} ({time.time() - t0:.0f}s)', flush=True)
    def merge(name, frame, keys):
        path = split_dir / name
        if path.exists(): frame = pd.concat([pd.read_csv(path), frame]).drop_duplicates(keys)
        frame.sort_values(keys).to_csv(path, index=False)
    if manifests:
        merge('scenes.csv', pd.DataFrame(manifests), ['scene']); merge('truth.csv', pd.DataFrame(rows), ['scene', 'source', 'band'])


# ----------------------------------------------------------------------------- real validation scenes
def real_scenes(limit, seed=SEED + 7):
    """Central-source scenes from the validation tiles with MER reference fluxes (Euclid bands)."""
    os.environ['JAISP_DETCAT_OUT'] = str(TILES)
    from ..detcat.prepare import load_inputs, scene_for_source
    from ..detcat.report import reference
    sigmas = {b: v['sigma_px'] for b, v in json.loads((HERE.parent / 'runs/q1_all_bands/psf_calibration.json').read_text()).items()}
    dest = OUT / 'val_real'; (dest / 'scenes').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed); rows = []; per_region = int(np.ceil(limit / len(regions_of('val'))))
    for region in regions_of('val'):
        folder = TILES / f'region_{region:03d}'; inputs = load_inputs(folder)
        match = pd.read_csv(folder / 'mer_match.csv').set_index('source')
        usable = np.flatnonzero(inputs['scene_inside'] & inputs['central_valid'])
        usable = [i for i in usable if bool(match.loc[int(inputs['source'][i]), 'matched']) and bool(match.loc[int(inputs['source'][i]), 'primary_match'])]
        for i in rng.permutation(usable)[:per_region]:
            scene, info = scene_for_source(inputs, int(i), sigmas)
            if scene is None: continue
            m = match.loc[[int(inputs['source'][i])]]
            arrays = dict(sky=scene['sky'])
            for b, d in scene['bands'].items():
                arrays.update({b + '__' + k: d[k].numpy() for k in ('image', 'variance', 'mask', 'positions', 'sky_to_pixel')})
                arrays[b + '__psf_sigma'] = float(d['psf_sigma'])
                if 'psf_kernels' in d: arrays[b + '__psf_kernels'] = np.asarray(d['psf_kernels'], 'float32')
            name = f'region{region:03d}_source{int(inputs["source"][i])}'
            np.savez_compressed(dest / 'scenes' / f'{name}.npz', **arrays)
            row = dict(name=name, region=region, source=int(inputs['source'][i]), n_sources=info['n_sources'],
                       nearest_arcsec=info['nearest_neighbor_arcsec'], mer_is_star=bool(m.mer_is_star.iloc[0]))
            for b in EUCLID:
                flux, err, column = reference(m, b)
                row.update({f'{b}__ref_ujy': float(flux[0]), f'{b}__ref_err_ujy': float(err[0]),
                            f'{b}__ujy_per_native': 10 ** (.4 * (23.9 - float(inputs[b + '__magzero'])))})
            rows.append(row)
        print(f'region {region}: {len(rows)} real scenes so far', flush=True)
    pd.DataFrame(rows).to_csv(dest / 'manifest.csv', index=False)


# ----------------------------------------------------------------------------- mixture reference
_MIXTURE = None


def _reference_one(path):
    global _MIXTURE
    import torch
    from ..predict import MixturePhotometry
    torch.set_num_threads(1)
    if _MIXTURE is None: _MIXTURE = MixturePhotometry(HERE.parent / 'runs/q1_mixture_calibrated/priors.pt', mode='foundation')
    try:
        result = _MIXTURE(load_scene(path)[0])
        return [dict(name=path.stem, source=j, band=b, flux=float(f), error=float(e))
                for b, r in result.items() if not b.startswith('_') for j, (f, e) in enumerate(zip(r['flux'], r['error']))], None
    except Exception:
        return [], dict(name=path.stem, error=traceback.format_exc())


def reference(val_limit, workers):
    for name, paths in (('val', sorted((OUT / 'val/scenes').glob('scene_*.npz'))[:val_limit]),
                        ('val_real', sorted((OUT / 'val_real/scenes').glob('*.npz')))):
        rows, failures = [], []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, (r, f) in enumerate(pool.map(_reference_one, paths, chunksize=2)):
                rows.extend(r); failures += [f] if f else []
                if (k + 1) % 100 == 0: print(f'mixture reference {name}: {k + 1}/{len(paths)}', flush=True)
        pd.DataFrame(rows).to_csv(OUT / name / 'reference_mixture.csv', index=False)
        (OUT / name / 'reference_failures.json').write_text(json.dumps(failures, indent=2))
        print(f'{name}: {len(failures)} mixture failures', flush=True)


# ----------------------------------------------------------------------------- loading
def load_scene(path):
    """(scene for the photometer, truth dict or None). Truth arrays never enter the scene."""
    import torch
    z = emp.read_npz(path); bands = {}; truth = {}
    for b in BANDS:
        bands[b] = {k: torch.tensor(z[b + '__' + k]) for k in ('image', 'variance', 'mask', 'positions', 'sky_to_pixel')}
        for k in ('positions', 'sky_to_pixel', 'image', 'variance'): bands[b][k] = bands[b][k].float()
        bands[b]['psf_sigma'] = float(z[b + '__psf_sigma'])
        if b + '__psf_kernels' in z: bands[b]['psf_kernels'] = z[b + '__psf_kernels']
        if 'truth__' + b + '__flux' in z:
            truth[b] = {k: z['truth__' + b + '__' + k] for k in ('flux', 'error', 'profiles', 'weak')}
    return dict(sky=z['sky'], bands=bands, central=0, tile=Path(path).stem), (truth or None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('split', 'mer', 'library', 'backgrounds', 'export', 'real', 'reference'))
    p.add_argument('--split', choices=('train', 'val'), default='train')
    p.add_argument('--count', type=int, default=16000); p.add_argument('--workers', type=int, default=40)
    p.add_argument('--donor-snr', type=float, default=25.); p.add_argument('--per-region', type=int, default=30)
    p.add_argument('--real-limit', type=int, default=500); p.add_argument('--val-limit', type=int, default=800)
    a = p.parse_args()
    if a.stage in ('library', 'backgrounds', 'real'): require_mer(regions_of('val' if a.stage == 'real' else a.split))
    if a.stage == 'split': make_split()
    elif a.stage == 'mer': match_mer()
    elif a.stage == 'library':
        out = OUT / a.split; out.mkdir(parents=True, exist_ok=True)
        emp.make_library(out, TILES, a.donor_snr, regions=set(regions_of(a.split))); emp.donor_quality(out, TILES)
    elif a.stage == 'backgrounds':
        emp.make_backgrounds(OUT / a.split, TILES, per_region=a.per_region, regions=regions_of(a.split), minimum_per_region=5)
    elif a.stage == 'export': export(a.split, a.count, a.workers)
    elif a.stage == 'real': real_scenes(a.real_limit)
    elif a.stage == 'reference': reference(a.val_limit, a.workers)


if __name__ == '__main__': main()
