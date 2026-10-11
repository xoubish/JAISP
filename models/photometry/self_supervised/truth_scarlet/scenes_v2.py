"""Version-2 training scenes: wider donor library and injections into real tile scenes.

Changes from scenes.py (v1), motivated by the v1 real-tile diagnosis:
  * donors down to VIS S/N 5, MER point-like sources (stars, compact galaxies) included;
  * 75% of scenes inject donors into real tile cutouts at random positions (30% placed
    0.3-2" from a real detection). Real detections within 14" are fitted as extra
    sources without truth, with local PSFs, real masks and real variance;
  * 25% blank-sky scenes as in v1, with the new donors.
Truth (flux, profile, oracle error) exists only for the injected sources, which are
always the first sources of a scene; the oracle error ignores real neighbours.

    python -m models.photometry.self_supervised.truth_scarlet.scenes_v2 library --split train
    python -m models.photometry.self_supervised.truth_scarlet.scenes_v2 export --split train --count 16000
    (same for --split val --count 800), then: reference

Outputs: runs/truth_scarlet/{train_v2,val_v2}. Uses the v1 split and blank-sky backgrounds.
"""
import argparse
import json
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

from ..core import BANDS
from .. import empirical as emp
from ..data import jacobian
from ..pixel_psf import normalize_kernel
from ..detcat.common import tangent_arcsec
from ..detcat.prepare import load_inputs, _crop
from .scenes import OUT, TILES, SEED, HERE, regions_of, require_mer, _reference_one, load_scene

SIGMAS = {b: v['sigma_px'] for b, v in json.loads((HERE.parent / 'runs/q1_all_bands/psf_calibration.json').read_text()).items()}
NEIGHBOR_ARCSEC, MAX_SOURCES, HALF_ARCSEC = 14., 40, 6.
EUCLID_K, RUBIN_RADIUS = 21, 18          # native GRID stamp size; Rubin Gaussian kernel radius (37 px)


def split_dir(split): return OUT / f'{split}_v2'


# ----------------------------------------------------------------------------- donor library
def _library_region(task):
    out, region, snr = task
    try:
        emp.make_library(out, TILES, snr, regions={region}, exclude_point_like=False)
        return region, None
    except RuntimeError as exc:   # 'No qualified donors'
        return region, str(exc)


def library(split, snr, workers):
    out = split_dir(split); tmp = out / 'by_region'; tmp.mkdir(parents=True, exist_ok=True)
    regions = regions_of(split)
    tasks = [(tmp / f'region_{r:03d}', r, snr) for r in regions if not (tmp / f'region_{r:03d}' / 'donor_bands.csv').exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for region, err in pool.map(_library_region, tasks):
            print(f'region {region}: {err or "done"}', flush=True)
    # Merge into one numbered library.
    dest = out / 'donors'
    if dest.exists(): shutil.rmtree(dest)
    dest.mkdir(); bands = []; k = 0
    for r in regions:
        src = tmp / f'region_{r:03d}'
        if not (src / 'donor_bands.csv').exists(): continue
        b = pd.read_csv(src / 'donor_bands.csv')
        for path in sorted((src / 'donors').glob('donor_*.npz')):
            d = emp.read_npz(path); old = int(d['donor']); d['donor'] = k
            np.savez_compressed(dest / f'donor_{k:05d}.npz', **d)
            bands.append(b[b.donor == old].assign(donor=k)); k += 1
    pd.concat(bands).to_csv(out / 'donor_bands.csv', index=False)
    q = emp.donor_quality(out, TILES)
    # v2 suitability: a unique MER match and a stable centre; stars and weak cross-band donors allowed.
    bad = q.reasons.fillna('').str.contains('no_unique_MER_galaxy_match|large_donor_center_shift')
    q['suitable_v2'] = ~bad; q['point_like'] = q.point_like_prob.fillna(0) > .5
    snr_map = {int(p.stem.split('_')[1]): float(emp.read_npz(p)['selection_snr']) for p in sorted(dest.glob('donor_*.npz'))}
    q['selection_snr'] = q.donor.map(snr_map)
    q.to_csv(out / 'donor_quality.csv', index=False)
    print(f'{split}: {k} donors, {int(q.suitable_v2.sum())} suitable, {int((q.suitable_v2 & q.point_like).sum())} point-like, '
          f'selection S/N quartiles {q.selection_snr.quantile([.25, .5, .75]).round(1).tolist()}', flush=True)


# ----------------------------------------------------------------------------- scene generation
def _pad_kernel(k, size):
    k = np.asarray(k, float); out = np.zeros((size, size)); n = k.shape[0]
    if n >= size:
        c = (n - size) // 2; return emp.unit(k[c:c + size, c:c + size])
    c = (size - n) // 2; out[c:c + n, c:c + n] = k; return emp.unit(out)


def _design(rng, donor_snrs):
    """Primary donor first; target S/N log-uniform, at most 3x the donor's own S/N (keeps smooth faint reconstructions from being shown very bright)."""
    kind = str(rng.choice(['isolated', 'pair', 'triple'], p=[.35, .55, .10])); n = dict(isolated=1, pair=2, triple=3)[kind]
    picks = rng.choice(len(donor_snrs), n, replace=False)
    hi = min(80., 3 * donor_snrs[picks[0]])
    snr = float(np.exp(rng.uniform(np.log(1.5), np.log(max(hi, 2.)))))
    offsets = [rng.uniform(-.05, .05, 2)]
    for _ in range(n - 1):
        sep = rng.uniform(.3, 2.); ang = rng.uniform(0, 2 * np.pi); offsets.append(offsets[0] + sep * np.array([np.cos(ang), np.sin(ang)]))
    ratios = []
    for k in picks[1:]:
        r = np.exp(rng.uniform(np.log(.1), np.log(10.)))
        ratios.append(min(r, 3 * donor_snrs[k] / snr))     # neighbour S/N also capped at 3x its own
    return dict(kind=kind, n=n, picks=picks, snr=snr, offsets=np.array(offsets), ratios=np.array(ratios))


_TILE = {}


def _tile(region):
    if region not in _TILE: _TILE.clear(); _TILE[region] = load_inputs(TILES / f'region_{region:03d}')
    return _TILE[region]


def _real_center(inputs, rng):
    """A position in the tile with a valid VIS core and Rubin coverage; 30% placed next to a real detection."""
    vis_im, _, vis_mask, vwc = emp.band_data(inputs, 'euclid_VIS'); h, w = vis_im.shape
    _, _, rmask, rwc = emp.band_data(inputs, 'rubin_r'); rh, rw = rmask.shape
    for _ in range(200):
        if rng.random() < .3:
            j = rng.integers(len(inputs['sky'])); sep = rng.uniform(.3, 2.); ang = rng.uniform(0, 2 * np.pi)
            sky = inputs['sky'][j] + sep * np.array([np.cos(ang), np.sin(ang)]) / np.array([3600 * np.cos(np.deg2rad(inputs['sky'][j][1])), 3600])
        else:
            sky = np.array(vwc.pixel_to_world_values(*rng.uniform([70, 70], [w - 71, h - 71])))
        x, y = (float(v) for v in vwc.world_to_pixel_values(*sky))
        if not (70 <= x <= w - 71 and 70 <= y <= h - 71): continue
        yy, xx = np.indices((21, 21)) - 10; core = vis_mask[int(round(y)) - 10:int(round(y)) + 11, int(round(x)) - 10:int(round(x)) + 11][np.hypot(xx, yy) <= 10]
        if core.mean() < .97: continue
        rx, ry = (float(v) for v in rwc.world_to_pixel_values(*sky))
        if not (8 <= rx <= rw - 9 and 8 <= ry <= rh - 9) or not rmask[int(round(ry)), int(round(rx))]: continue
        return sky
    raise ValueError('no valid injection position')


def _render_scene(rng, design, ds, frame, real):
    """Render injected donors onto `frame` (per-band dict of image, variance, mask, matrix, origin-relative positions, kernels)."""
    arrays = {}; rows = []
    profiles = {}
    for b in BANDS:
        f = frame[b]; shape = f['image'].shape
        pp = [emp.render(d[b + '__latent'], d[b + '__matrix'], f['matrix'], f['positions'][j], shape, kernel=f['inj_kernel'][j],
                         angle=f['angle'], extra_sigma_arcsec=f['broadening'], donor_center=d[b + '__center'], kernel_matrix=f['kernel_matrix'][j])
              for j, d in enumerate(ds)]
        profiles[b] = np.array(pp)
    vis = frame['euclid_VIS']
    _, error = emp.oracle_fit(np.zeros_like(vis['image']), vis['variance'], vis['mask'] & np.isfinite(vis['image']), profiles['euclid_VIS'][:1])
    fvis = design['snr'] * error[0]
    scales = [fvis / float(ds[0]['euclid_VIS__flux'])] + [r * fvis / float(d['euclid_VIS__flux']) for r, d in zip(design['ratios'], ds[1:])]
    for b in BANDS:
        f = frame[b]; truth = np.array([s * float(d[b + '__flux']) for s, d in zip(scales, ds)])
        image = f['image'].astype(float) + np.einsum('n,nhw->hw', truth, profiles[b])
        valid = f['mask'] & np.isfinite(image) & np.isfinite(f['variance']) & (f['variance'] > 0)
        oflux, oerr = emp.oracle_fit(np.where(valid, image, 0), np.where(valid, f['variance'], 1), valid, list(profiles[b]))
        k0 = f['kernels'][0]; sigma = np.sqrt(np.sum(k0 * (np.indices(k0.shape)[0] - (k0.shape[0] - 1) / 2) ** 2))
        arrays.update({b + '__image': image.astype('float32'), b + '__variance': f['variance'].astype('float32'), b + '__mask': f['mask'],
                       b + '__positions': f['all_positions'], b + '__sky_to_pixel': np.linalg.inv(f['matrix']),
                       b + '__psf_sigma': float(sigma), b + '__psf_kernels': f['kernels'].astype('float32'),
                       'truth__' + b + '__flux': truth, 'truth__' + b + '__error': oerr,
                       'truth__' + b + '__profiles': profiles[b].astype('float32'),
                       'truth__' + b + '__weak': np.array([bool(d[b + '__fallback']) for d in ds])})
        for j in range(len(ds)):
            rows.append(dict(source=j, band=b, truth_flux=truth[j], oracle_error=oerr[j], oracle_flux=oflux[j],
                             true_snr=truth[j] / oerr[j], weak_band=bool(ds[j][b + '__fallback'])))
    return arrays, rows


def _real_frame(inputs, rng, design, ds):
    center = _real_center(inputs, rng)
    cosd = np.cos(np.deg2rad(center[1]))
    inj_sky = center + design['offsets'] / np.array([3600 * cosd, 3600])
    dist = np.linalg.norm(tangent_arcsec(inputs['sky'], center), axis=1)
    members = np.flatnonzero(dist < NEIGHBOR_ARCSEC); members = members[np.argsort(dist[members])][:MAX_SOURCES - len(ds)]
    all_sky = np.r_[inj_sky, inputs['sky'][members]]
    off = tangent_arcsec(all_sky, inj_sky[0]); d_all = np.linalg.norm(off[1:], axis=1)
    frame = {}
    for b in BANDS:
        image, variance, mask, wc = emp.band_data(inputs, b)
        pix = np.array(wc.world_to_pixel_values(*center)); matrix = jacobian(wc, pix); scale = np.sqrt(abs(np.linalg.det(matrix)))
        crop = _crop(image, variance, mask, pix, int(np.ceil(HALF_ARCSEC / scale)), pad=True)
        if crop is None: raise ValueError('edge')
        im, var, m, origin = crop
        inj_pos = np.column_stack(wc.world_to_pixel_values(*inj_sky.T)) - origin
        real_pos = (inputs[b + '__positions'] if b.startswith('euclid_') else inputs['rubin__positions'])[members] - origin
        if b.startswith('euclid_'):
            local = inputs[b + '__psf_stamps'][inputs[b + '__kernel_index']]
            # injected sources take the stamp of the nearest real detection's PSF sample (same local GRID PSF)
            near = inputs[b + '__kernel_index'][members[0]] if len(members) else inputs[b + '__kernel_index'][0]
            inj_k = [normalize_kernel(inputs[b + '__psf_stamps'][near])] * len(ds)
            nb_k = [normalize_kernel(local[m_]) for m_ in members]
            size = EUCLID_K
        else:
            g = emp.gaussian_psf(SIGMAS[b], radius=RUBIN_RADIUS); inj_k = [g] * len(ds); nb_k = [g] * len(members); size = 2 * RUBIN_RADIUS + 1
        frame[b] = dict(image=np.where(m, im, np.where(np.isfinite(im), im, 0.)), variance=np.where(np.isfinite(var), var, 1.), mask=m & np.isfinite(im) & np.isfinite(var),
                        matrix=matrix, positions=inj_pos, all_positions=np.r_[inj_pos, real_pos],
                        inj_kernel=inj_k, kernel_matrix=[matrix] * len(ds), angle=0., broadening=0.,
                        kernels=np.array([_pad_kernel(k, size) for k in inj_k + nb_k]))
    meta = dict(n_sources=len(all_sky), n_real=len(members), min_sep_arcsec=float(d_all.min()) if len(d_all) else np.inf,
                nearest_real_arcsec=float(np.linalg.norm(off[len(ds):], axis=1).min()) if len(members) else np.inf)
    return frame, all_sky, meta


def _blank_frame(bg, rng, design, ds):
    angle = float(rng.uniform(0, 2 * np.pi)); be = float(rng.uniform(0, .05)); br = float(rng.uniform(0, .2))
    inj_sky = bg['sky'] + design['offsets'] / np.array([3600 * np.cos(np.deg2rad(bg['sky'][1])), 3600])
    frame = {}
    for b in BANDS:
        matrix = bg[b + '__matrix']; pos = bg[b + '__center'] + np.linalg.solve(matrix, design['offsets'].T).T
        broadening = be if b.startswith('euclid_') else br; ksize = 65 if b.startswith('euclid_') else 49
        kk = [emp.unit(emp.render(d[b + '__kernel'], d[b + '__kernel_matrix'], matrix, [(ksize - 1) / 2] * 2, (ksize, ksize),
                                  angle=angle, extra_sigma_arcsec=broadening)) for d in ds]
        frame[b] = dict(image=bg[b + '__image'], variance=bg[b + '__variance'], mask=bg[b + '__mask'], matrix=matrix,
                        positions=pos, all_positions=pos, inj_kernel=[d[b + '__kernel'] for d in ds],
                        kernel_matrix=[d[b + '__kernel_matrix'] for d in ds], angle=angle, broadening=broadening, kernels=np.array(kk))
    d_all = np.linalg.norm(design['offsets'][1:] - design['offsets'][0], axis=1)
    return frame, inj_sky, dict(n_sources=design['n'], n_real=0, min_sep_arcsec=float(d_all.min()) if len(d_all) else np.inf, nearest_real_arcsec=np.inf,
                                broadening_euclid=be, broadening_rubin=br, rotation_rad=angle)


def _export_one(task):
    out, scene_id, mode, where, donors, donor_snrs, split_seed = task
    rng = np.random.default_rng([SEED, 20, split_seed, scene_id])
    for attempt in range(5):
        try:
            design = _design(rng, donor_snrs); ds = [emp.read_npz(donors[k]) for k in design['picks']]
            if mode == 'real': frame, sky, meta = _real_frame(_tile(where), rng, design, ds)
            else: frame, sky, meta = _blank_frame(emp.read_npz(where), rng, design, ds)
            arrays, rows = _render_scene(rng, design, ds, frame, mode == 'real')
            break
        except ValueError as exc:
            if attempt == 4: return None, [], f'{scene_id}: {exc}'
    arrays['sky'] = sky; arrays['n_truth'] = design['n']
    np.savez_compressed(out / 'scenes' / f'scene_{scene_id:05d}.npz', **arrays)
    manifest = dict(scene=scene_id, mode=mode, where=str(where if mode == 'real' else Path(where).stem), kind=design['kind'], n_injected=design['n'],
                    vis_snr=design['snr'], donors=';'.join(str(int(d['donor'])) for d in ds),
                    primary_point_like=None, ratios=';'.join(f'{r:.3f}' for r in design['ratios']), **meta)
    return manifest, [dict(scene=scene_id, **r) for r in rows], None


def export(split, count, workers, real_fraction=.75):
    out = split_dir(split); (out / 'scenes').mkdir(parents=True, exist_ok=True)
    q = pd.read_csv(out / 'donor_quality.csv'); q = q[q.suitable_v2].sort_values('donor')
    donors = [out / 'donors' / f'donor_{int(k):05d}.npz' for k in q.donor]; snrs = q.selection_snr.to_numpy(float)
    point_like = dict(zip(q.donor.astype(int), q.point_like))
    backgrounds = sorted((OUT / split / 'backgrounds').glob('background_*.npz'))
    regions = regions_of(split); rng = np.random.default_rng([SEED, 21, {'train': 1, 'val': 2}[split]])
    modes = np.where(rng.random(count) < real_fraction, 'real', 'blank')
    where = [int(regions[rng.integers(len(regions))]) if m == 'real' else str(backgrounds[rng.integers(len(backgrounds))]) for m in modes]
    done = set(pd.read_csv(out / 'scenes.csv').scene) if (out / 'scenes.csv').exists() else set()   # only scenes with a manifest row count as done
    tasks = [(out, i, modes[i], where[i], donors, snrs, {'train': 1, 'val': 2}[split]) for i in range(count) if i not in done]
    tasks.sort(key=lambda t: (t[2], str(t[3])))     # group by tile so workers reuse the loaded tile
    print(f'{split}_v2: {len(donors)} donors ({int(q.point_like.sum())} point-like), {len(regions)} tiles, {len(backgrounds)} blank patches; '
          f'{len(tasks)} scenes to make ({(modes == "real").mean():.0%} real-scene)', flush=True)
    manifests, rows, failures = [], [], []; t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, (m, r, f) in enumerate(pool.map(_export_one, tasks, chunksize=8)):
            if m: manifests.append(m); rows.extend(r)
            if f: failures.append(f)
            if (k + 1) % 2000 == 0: print(f'{split}_v2: {k + 1}/{len(tasks)} ({time.time() - t0:.0f}s, {len(failures)} failed)', flush=True)
    for m in manifests: m['primary_point_like'] = bool(point_like[int(m['donors'].split(';')[0])])
    def merge(name, frame, keys):
        path = out / name
        if path.exists(): frame = pd.concat([pd.read_csv(path), frame]).drop_duplicates(keys)
        frame.sort_values(keys).to_csv(path, index=False)
    if manifests: merge('scenes.csv', pd.DataFrame(manifests), ['scene']); merge('truth.csv', pd.DataFrame(rows), ['scene', 'source', 'band'])
    (out / 'export_failures.json').write_text(json.dumps(failures, indent=2))
    print(f'{split}_v2: done, {len(failures)} failed scenes', flush=True)


def reference(workers, limit=800):
    out = split_dir('val'); paths = sorted((out / 'scenes').glob('scene_*.npz'))[:limit]
    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, (r, f) in enumerate(pool.map(_reference_one, paths, chunksize=2)):
            rows.extend(r); failures += [f] if f else []
            if (k + 1) % 200 == 0: print(f'mixture reference val_v2: {k + 1}/{len(paths)}', flush=True)
    pd.DataFrame(rows).to_csv(out / 'reference_mixture.csv', index=False)
    (out / 'reference_failures.json').write_text(json.dumps(failures, indent=2))
    print(f'val_v2: {len(failures)} mixture failures', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('library', 'export', 'reference'))
    p.add_argument('--split', choices=('train', 'val'), default='train')
    p.add_argument('--count', type=int, default=16000); p.add_argument('--workers', type=int, default=40)
    p.add_argument('--donor-snr', type=float, default=5.); p.add_argument('--real-fraction', type=float, default=.75)
    a = p.parse_args()
    if a.stage == 'library': require_mer(regions_of(a.split)); library(a.split, a.donor_snr, a.workers)
    elif a.stage == 'export': export(a.split, a.count, a.workers, a.real_fraction)
    elif a.stage == 'reference': reference(a.workers)


if __name__ == '__main__': main()
