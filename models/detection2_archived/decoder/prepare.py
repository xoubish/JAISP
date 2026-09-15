"""Freeze image crops, detector proposals and training-only normalization."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from astropy.wcs.utils import proj_plane_pixel_scales

from ..common import ROOT, resolve, read_config, split_tiles, spaced, digest, write_json
from ..audit import read_wcs
from ..data import disk_mask
from ..evaluation.run import save_npz
from .data import BANDS
from detection.masks import bright_star_saturation_mask, load_gaia_cache
from detection.centernet_detector import CenterNetDetector
from detection.visnir_eval_experiment import predict_features

DEFAULT_CONFIG = ROOT/'models/detection2/configs/decoder_warmup_v1.json'


def load_bands(cfg, tid):
    images, variance, wcs = [], [], []
    with np.load(resolve(cfg['rubin_dir'])/f'{tid}.npz', allow_pickle=True) as z:
        img, var, ww = z['img'], z['var'], read_wcs(z['wcs_hdr'])
        for i in range(6):
            images.append(img[i]); variance.append(var[i]); wcs.append(ww)
    with np.load(resolve(cfg['euclid_dir'])/f'{tid}_euclid.npz', allow_pickle=True) as z:
        for band in ('VIS', 'Y', 'J', 'H'):
            images.append(z['img_'+band]); variance.append(z['var_'+band]); wcs.append(read_wcs(z['wcs_'+band]))
    return images, variance, wcs


def normalization(data_cfg, tiles):
    centers, scales = [], []
    for tid in tiles:
        images, variance, _ = load_bands(data_cfg, tid)
        cc, ss = [], []
        for image, var in zip(images, variance):
            # A deterministic subsample limits preprocessing memory. No
            # validation pixels contribute to fitted centering/scaling.
            image, var = image.ravel()[::8], var.ravel()[::8]
            valid = np.isfinite(image) & np.isfinite(var) & (var > 0)
            values = image[valid]
            if len(values) < 100:
                raise ValueError(f'{tid}: insufficient valid normalization pixels')
            center = np.median(values)
            scale = 1.4826*np.median(np.abs(values-center))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f'{tid}: degenerate image scale')
            cc.append(float(center)); ss.append(float(scale))
        centers.append(cc); scales.append(ss)
    return {'center': np.median(centers, axis=0).tolist(), 'scale': np.median(scales, axis=0).tolist(),
            'training_tiles': tiles, 'definition': 'Median of per-training-tile pixel medians and 1.4826 MADs; '
            'scale is an empirical image scale, not a calibrated independent-pixel noise standard deviation.'}


def cutout(array, origin, shape, fill=0):
    x0, y0 = map(int, origin); h, w = shape
    result = np.full((h, w), fill, dtype=array.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(array.shape[1], x0+w), min(array.shape[0], y0+h)
    if sx1 > sx0 and sy1 > sy0:
        result[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = array[sy0:sy1, sx0:sx1]
    return result


def sample_object_features(features, xy, vis_shape, roi_side):
    if roi_side % 2 != 1:
        raise ValueError('ROI side must be odd')
    if not len(xy):
        return np.zeros((1, features.shape[1]*roi_side**2), np.float16)
    h, w = vis_shape
    centers = torch.as_tensor(xy, device=features.device, dtype=torch.float32)/torch.tensor(
        [w-1, h-1], device=features.device)*2-1
    offset = torch.arange(roi_side, device=features.device)-roi_side//2
    yy, xx = torch.meshgrid(offset, offset, indexing='ij')
    shifts = torch.stack([xx.flatten()*2/(features.shape[-1]-1), yy.flatten()*2/(features.shape[-2]-1)], -1)
    grid = (centers[:, None]+shifts[None]).reshape(1, len(xy)*roi_side**2, 1, 2)
    sampled = F.grid_sample(features, grid, mode='bilinear', padding_mode='border', align_corners=True)
    result = sampled[0, :, :, 0].T.reshape(len(xy), -1).cpu().numpy()
    if not np.isfinite(result).all() or np.abs(result).max() > np.finfo(np.float16).max:
        raise ValueError('Features cannot be stored safely in float16')
    return result.astype(np.float16)


def prepare_tile(tid, images, variance, wcs, mask, xy, scores, features, cfg, norm, count):
    scales = [float(np.mean(proj_plane_pixel_scales(w)*3600)) for w in wcs]
    shapes = [(int(round(cfg['crop_arcsec']/s)),)*2 for s in scales]
    stamps = [2*int(round(cfg['stamp_width_arcsec']/s/2))+1 for s in scales]
    headers = [w.to_header(relax=True).tostring() for w in wcs]
    world = wcs[6].all_pix2world(xy, 0) if len(xy) else np.empty((0, 2))
    positions_full = np.stack([ww.all_world2pix(world, 0) for ww in wcs], 1) if len(xy) else np.empty((0, 10, 2))
    vh, vw = images[6].shape
    rng = np.random.default_rng((cfg['seed']+int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16)) % 2**32)
    crops, rejected = [], {'invalid_pixels': 0, 'too_many_objects': 0}
    for i in range(count):
        for attempt in range(300):
            if i < count//2 and len(xy):
                center = xy[rng.integers(len(xy))]+rng.uniform(-.25, .25, 2)*shapes[6][0]
            else:
                margin = shapes[6][0]/2+4
                center = rng.uniform([margin, margin], [vw-margin, vh-margin])
            sky_center = wcs[6].all_pix2world(center[None], 0)
            origins = np.array([np.rint(ww.all_world2pix(sky_center, 0)[0]-(np.array(shape[::-1])-1)/2).astype(int)
                                for ww, shape in zip(wcs, shapes)])
            local = positions_full-origins[None]
            include = np.zeros(len(xy), bool)
            for band, ((h, w), stamp) in enumerate(zip(shapes, stamps)):
                radius = (stamp-1)/2+1
                include |= ((local[:, band, 0] >= -radius) & (local[:, band, 0] < w+radius)
                            & (local[:, band, 1] >= -radius) & (local[:, band, 1] < h+radius))
            indices = np.flatnonzero(include)
            if len(indices) > cfg['max_objects_per_crop']:
                rejected['too_many_objects'] += 1
                continue
            targets, valid, source = [], [], []
            geometry_cache = {}
            for band, (image, var, ww, shape, origin) in enumerate(zip(images, variance, wcs, shapes, origins)):
                target = cutout(image, origin, shape)
                vv = cutout(var, origin, shape)
                good = np.isfinite(target) & np.isfinite(vv) & (vv > 0)
                key = (headers[band], tuple(shape), tuple(origin))
                if key not in geometry_cache:
                    if headers[band] == headers[6]:
                        artifact = cutout(mask, origin, shape, fill=True)
                    else:
                        py, px = np.mgrid[:shape[0], :shape[1]]
                        sky = ww.all_pix2world(np.column_stack([px.ravel()+origin[0], py.ravel()+origin[1]]), 0)
                        vxy = wcs[6].all_world2pix(sky, 0)
                        ok = np.isfinite(vxy).all(1) & (vxy[:, 0] >= 0) & (vxy[:, 0] < vw) & (vxy[:, 1] >= 0) & (vxy[:, 1] < vh)
                        artifact = np.ones(len(vxy), bool)
                        ii = np.flatnonzero(ok)
                        artifact[ii] = mask[vxy[ii, 1].astype(int), vxy[ii, 0].astype(int)]
                        artifact = artifact.reshape(shape)
                    radius = cfg['source_aperture_arcsec']/scales[band]
                    centers = local[indices, band]
                    near = ((centers[:, 0]+radius >= 0) & (centers[:, 0]-radius < shape[1])
                            & (centers[:, 1]+radius >= 0) & (centers[:, 1]-radius < shape[0]))
                    aperture = disk_mask(centers[near]/[shape[1]-1, shape[0]-1], shape, radius)
                    geometry_cache[key] = artifact, aperture
                artifact, aperture = geometry_cache[key]
                good &= ~artifact
                target = np.where(good, (target-norm['center'][band])/norm['scale'][band], 0).astype(np.float32)
                targets.append(target); valid.append(good); source.append(aperture & good)
            if min(float(v.mean()) for v in valid) < cfg['minimum_valid_fraction']:
                rejected['invalid_pixels'] += 1
                continue
            crops.append({'indices': indices, 'positions': local[indices].astype(np.float32),
                          'targets': targets, 'valid': valid, 'source': source, 'center_vis': center})
            break
        else:
            raise ValueError(f'{tid}: cannot obtain {count} valid crops; rejected={rejected}')
    max_k = max(1, max(len(c['indices']) for c in crops))
    ids = np.full((count, max_k), -1, np.int32)
    pp = np.zeros((count, max_k, len(BANDS), 2), np.float32)
    for i, crop in enumerate(crops):
        k = len(crop['indices']); ids[i, :k] = crop['indices']; pp[i, :k] = crop['positions']
    arrays = {'object_features': features, 'object_indices': ids, 'positions': pp,
              'proposal_xy_vis': xy, 'proposal_scores': scores,
              'crop_centers_vis': np.asarray([c['center_vis'] for c in crops])}
    for band in range(len(BANDS)):
        for key, label in [('targets', 'target'), ('valid', 'valid'), ('source', 'source')]:
            arrays[f'{label}_{band}'] = np.stack([c[key][band] for c in crops])
    return arrays, {'band_shapes': shapes, 'stamp_sizes': stamps, 'pixel_scales_arcsec': scales,
                    'n_proposals': len(xy), 'objects_per_crop': [len(c['indices']) for c in crops], 'rejections': rejected}


def prepare(config_path, out, device_name, train_limit=0, validation_limit=0):
    cfg = read_config(config_path)
    selection = read_config(resolve(cfg['selected_baseline']))
    checkpoint = resolve(selection['checkpoint'])
    if digest(checkpoint) != selection['checkpoint_sha256']:
        raise ValueError('Selected baseline checkpoint hash changed')
    data_cfg = read_config(resolve(cfg['detection_config']))
    if resolve(data_cfg['feature_dir']).resolve() != resolve(selection['feature_dir']).resolve():
        raise ValueError('Feature cache differs from the selected detector baseline')
    train, validation = split_tiles(data_cfg)
    if train_limit: train = spaced(train, train_limit)
    if validation_limit: validation = spaced(validation, validation_limit)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'config.json', cfg)
    norm = normalization(data_cfg, spaced(train, cfg['normalization_tiles']))
    write_json(out/'normalization.json', norm)
    device = torch.device(device_name)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('GPU is required for frozen proposal extraction')
    torch.set_num_threads(4)
    teacher = CenterNetDetector.load(str(checkpoint), encoder=None, device=device).eval()
    gaia = load_gaia_cache(str(resolve(data_cfg['gaia'])))
    metadata = {'config': cfg, 'selected_baseline': selection, 'normalization': norm, 'bands': BANDS,
                'train_tiles': train, 'validation_tiles': validation, 'complete': not train_limit and not validation_limit,
                'catalogue': 'Fixed selected CenterNet proposals above 0.15 in training and validation; '
                             'no MER labels or new pseudo-label promotion.',
                'scope': 'Renderer warm-up on full-image cached features; not held-out-pixel prediction or a new detector.',
                'tiles': {}}
    paths = [Path(config_path), resolve(cfg['selected_baseline']), checkpoint,
             resolve(cfg['detection_config']), resolve(data_cfg['gaia'])]
    inputs = {str(p): digest(p) for p in paths}
    source_paths = list(Path(__file__).parent.glob('*.py')) + [ROOT/'models/detection2'/name
                    for name in ('common.py', 'audit.py', 'data.py', 'evaluation/run.py')]
    source_paths += [ROOT/'models/detection'/name for name in
                     ('centernet_detector.py', 'visnir_eval_experiment.py', 'masks.py')]
    write_json(out/'source_sha256.json', {str(p): digest(p) for p in source_paths})
    start = time.monotonic()
    for i, tid in enumerate(train+validation):
        images, variance, wcs = load_bands(data_cfg, tid)
        mask = bright_star_saturation_mask(np.nan_to_num(images[6]), wcs[6].to_header().tostring(), gaia)
        feature_path = resolve(data_cfg['feature_dir'])/f'{tid}_aug0.pt'
        cached = torch.load(feature_path, map_location='cpu', weights_only=True)
        if tuple(cached['aug_params']) != (0, False, False):
            raise ValueError('Expected identity cached features')
        tensor = cached['features'][None].to(device)
        with torch.no_grad():
            xy, scores = predict_features(teacher, tensor, mask, images[6].shape, floor=cfg['proposal_threshold'])
            object_features = sample_object_features(tensor, xy, images[6].shape, cfg['roi_side'])
        count = cfg['train_crops_per_tile'] if tid in train else cfg['validation_crops_per_tile']
        arrays, row = prepare_tile(tid, images, variance, wcs, mask, xy, scores, object_features, cfg, norm, count)
        path = out/'tiles'/f'{tid}.npz'
        save_npz(path, **arrays)
        if i == 0:
            metadata.update({k: row[k] for k in ('band_shapes', 'stamp_sizes', 'pixel_scales_arcsec')})
            metadata['feature_dim'] = object_features.shape[1]
        if row['band_shapes'] != metadata['band_shapes'] or row['stamp_sizes'] != metadata['stamp_sizes']:
            raise ValueError('Image grid scales changed across tiles')
        row['prepared_sha256'] = digest(path)
        row['input_sha256'] = {str(p): digest(p) for p in [feature_path,
            resolve(data_cfg['euclid_dir'])/f'{tid}_euclid.npz', resolve(data_cfg['rubin_dir'])/f'{tid}.npz']}
        metadata['tiles'][tid] = row
        if (i+1) % 10 == 0 or i+1 == len(train+validation):
            update = {'stage': 'preparing', 'tiles_done': i+1, 'tiles_total': len(train+validation),
                      'elapsed_minutes': (time.monotonic()-start)/60}
            write_json(out/'status.json', update); print(json.dumps(update), flush=True)
    write_json(out/'input_sha256.json', inputs)
    write_json(out/'metadata.json', metadata)
    write_json(out/'status.json', {'stage': 'complete', 'tiles': len(train+validation),
        'training_crops': len(train)*cfg['train_crops_per_tile'],
        'validation_crops': len(validation)*cfg['validation_crops_per_tile'],
        'elapsed_minutes': (time.monotonic()-start)/60})
    print('Prepared:', out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--train-tiles', type=int, default=0, help='Diagnostic subset only')
    p.add_argument('--validation-tiles', type=int, default=0, help='Diagnostic subset only')
    args = p.parse_args()
    prepare(args.config, args.out, args.device, args.train_tiles, args.validation_tiles)


if __name__ == '__main__':
    main()
