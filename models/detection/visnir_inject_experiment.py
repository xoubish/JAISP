"""Paired injection/recovery and induced-artifact check for the VIS+NISP pilot.

Use existing isolated, tapered point-source stamps. Identical sources and
positions are presented to every head; the frozen encoder is shared. Positions
avoid full MER and the union of pre-existing detections. Each pass contains
one magnitude and one band combination, following inject_purity_eval.py.
Magnitudes are donor VIS-equivalent values, including for NISP-only injections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'models'))
from detection.centernet_detector import CenterNetDetector
from detection.detector import JAISPEncoderWrapper
from detection.masks import bright_star_saturation_mask, load_gaia_cache
from detection.validation_utils import build_inputs, _add, _mode_bands, RUBIN_BANDS, EUCLID_BANDS
from detection.visnir_eval_experiment import ENCODER, predict_features
from load_foundation import load_foundation


@torch.no_grad()
def encode(encoder, images, rms, device):
    return encoder({b: torch.from_numpy(v[None, None]).to(device) for b, v in images.items()},
                   {b: torch.from_numpy(v[None, None]).to(device) for b, v in rms.items()})


def distances(points, refs):
    return cKDTree(refs).query(points)[0] if len(refs) else np.full(len(points), np.inf)


def one_tile(tid, models, encoder, library, catalog, gaia, mags, modes, n_per_mag, device):
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(tid.encode()).digest()[:4], 'little'))
    with np.load(ROOT/'data/euclid_tiles_all_q1'/f'{tid}_euclid.npz', allow_pickle=True) as d:
        ed = dict(d)
    with np.load(ROOT/'data/rubin_tiles_all'/f'{tid}.npz', allow_pickle=True) as d:
        rd = dict(d)
    images0, rms, shape = build_inputs(ed, rd)
    h, w = shape
    wcs = WCS(fits.Header.fromstring(str(ed['wcs_VIS'])))
    rw = WCS(rd['wcs_hdr'].item())
    ew = {k: WCS(fits.Header.fromstring(str(ed[f'wcs_{k}']))) for k in ('VIS', 'Y', 'J', 'H')}
    mask = bright_star_saturation_mask(images0['euclid_VIS'], str(ed['wcs_VIS']), gaia)
    coverage = ~mask
    for band in ('VIS', 'Y', 'J', 'H'):
        v = ed[f'var_{band}']
        coverage &= np.isfinite(v) & (v > 0) & (v < 1e20)
    features = encode(encoder, images0, rms, device)
    pre = {name: predict_features(model, features, mask, shape, floor=0.02)
           for name, model in models.items()}
    baseline = {name: xy[scores >= 0.30] for name, (xy, scores) in pre.items()}
    union = np.concatenate(list(baseline.values()))
    cx, cy = wcs.all_world2pix(catalog['ra'], catalog['dec'], 0)
    inside = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
    occupied = np.column_stack((cx[inside], cy[inside]))
    occupied_tree = cKDTree(occupied) if len(occupied) else None
    detection_tree = cKDTree(union) if len(union) else None
    out = {name: {mode: {} for mode in modes} for name in models}
    for mag in mags:
        injected, positions = [], []
        for attempt in range(2000):
            if len(injected) == n_per_mag:
                break
            di = int(rng.integers(len(library['mag'])))
            scale = 10**(-0.4*(mag-float(library['mag'][di])))
            if scale > 1:
                continue
            tx, ty = rng.uniform(84, w-84), rng.uniform(84, h-84)
            ix, iy = int(round(tx)), int(round(ty))
            if not coverage[iy-3:iy+4, ix-3:ix+4].all():
                continue
            if occupied_tree is not None and occupied_tree.query([tx, ty])[0] < 10:
                continue
            if detection_tree is not None and detection_tree.query([tx, ty])[0] < 8:
                continue
            if positions and min(np.linalg.norm(np.asarray(positions)-[tx, ty], axis=1)) < 20:
                continue
            ra, dec = wcs.all_pix2world(tx, ty, 0)
            pend = []
            valid = True
            for bi, band in enumerate(EUCLID_BANDS):
                key = band.split('_', 1)[1]
                x, y = ew[key].all_world2pix(ra, dec, 0)
                x, y = int(round(float(x))), int(round(float(y)))
                stamp = library['euclid'][di, bi]*scale
                r = stamp.shape[0]//2
                if x-r < 0 or y-r < 0 or x+r >= w or y+r >= h:
                    valid = False
                pend.append((band, x, y, stamp))
            for bi, band in enumerate(RUBIN_BANDS):
                x, y = rw.all_world2pix(ra, dec, 0)
                x, y = int(round(float(x))), int(round(float(y)))
                stamp = library['rubin'][di, bi]*scale
                r = stamp.shape[0]//2
                rh, rwidth = images0[band].shape
                if x-r < 0 or y-r < 0 or x+r >= rwidth or y+r >= rh:
                    valid = False
                elif not (np.isfinite(rd['var'][bi, y-2:y+3, x-2:x+3]) &
                          (rd['var'][bi, y-2:y+3, x-2:x+3] > 0)).all():
                    valid = False
                pend.append((band, x, y, stamp))
            if valid:
                injected.append(pend)
                positions.append((tx, ty))
        positions = np.asarray(positions).reshape(-1, 2)
        if not len(positions):
            raise ValueError(f'No valid injection positions on {tid}')
        for mode in modes:
            images = {b: x.copy() for b, x in images0.items()}
            active = _mode_bands(mode)
            for pend in injected:
                for band, x, y, stamp in pend:
                    if band in active:
                        assert _add(images[band], x, y, stamp)
            features = encode(encoder, images, rms, device)
            for name, model in models.items():
                xy, scores = predict_features(model, features, mask, shape)
                detections = xy[scores >= 0.30]
                recovery = distances(positions, detections) < 3.0
                new = detections[distances(detections, baseline[name]) > 4.0]
                unrelated = new[distances(new, positions) >= 3.0]
                subthreshold = distances(unrelated, pre[name][0]) < 3.0
                out[name][mode][str(mag)] = dict(recovered=int(recovery.sum()), injected=len(positions),
                    artifacts=int((~subthreshold).sum()), threshold_flips=int(subthreshold.sum()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', action='append', required=True, help='name=path')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--start-index', type=int, default=0)
    p.add_argument('--stop-index', type=int, default=108)
    p.add_argument('--mags', default='23.5,24.5,25,25.5,26,26.5,27,27.5,35')
    p.add_argument('--modes', default='all,vis,nisp')
    p.add_argument('--n-per-mag', type=int, default=15)
    args = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    models = {}
    hashes = {}
    for spec in args.checkpoint:
        name, path = spec.split('=', 1)
        hashes[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        models[name] = CenterNetDetector.load(path, None, device).eval()
    foundation = load_foundation(str(ENCODER), device=torch.device('cpu'), freeze=True)
    encoder = JAISPEncoderWrapper(foundation, freeze=True).to(device).eval()
    with np.load(ROOT/'checkpoints/q1_detection_v11/donor_library_r60.npz') as d:
        library = dict(d)
    catalog = fits.getdata(ROOT/'data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits')
    gaia = load_gaia_cache(str(ROOT/'data/gaia_ecdfs_astrometry_cache.npz'))
    paths = sorted((ROOT/'data/euclid_tiles_all_q1').glob('*_patch_25_euclid.npz'))
    paths = paths[args.start_index:args.stop_index]
    mags = tuple(float(x) for x in args.mags.split(','))
    modes = tuple(args.modes.split(','))
    config = dict(vars(args), checkpoint_sha256=hashes, encoder=str(ENCODER),
                  library='checkpoints/q1_detection_v11/donor_library_r60.npz',
                  confidence=0.30, match_px=3.0, new_detection_px=4.0, pre_floor=0.02,
                  magnitude_definition='donor VIS-equivalent; NOT physical NISP magnitude',
                  placement='shared; excludes full MER, union of pre-detections, masked/invalid centers')
    conf_path = args.out/'config.json'
    normalized = json.loads(json.dumps(config, default=str))
    if conf_path.exists() and json.loads(conf_path.read_text()) != normalized:
        raise ValueError('Existing injection cache belongs to a different configuration')
    conf_path.write_text(json.dumps(normalized, indent=2)+'\n')
    start = time.monotonic()
    for i, path in enumerate(paths):
        tid = path.name.removesuffix('_euclid.npz')
        target = args.out/f'{tid}.json'
        if not target.exists():
            result = one_tile(tid, models, encoder, library, catalog, gaia,
                              mags, modes, args.n_per_mag, device)
            temporary = target.with_suffix('.tmp')
            temporary.write_text(json.dumps(result, indent=2)+'\n')
            temporary.replace(target)
        print(f'{i+1}/{len(paths)} tiles; elapsed {time.monotonic()-start:.1f}s', flush=True)
    print('Completed', args.out, flush=True)


if __name__ == '__main__':
    main()
