"""Frozen leave-one-object-out image evidence; no parameter fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F

from ..common import ROOT, digest, resolve, write_json
from ..decoder.model import ObjectRenderer
from ..decoder.prepare import load_bands, cutout
from ..evaluation.run import load_prepared, save_npz

STUDY = ROOT/'models/detection2/runs/decoder_warmup_20260915_v1'
EVALUATION = ROOT/'models/detection2/runs/unknown_regions_20260915_v1_12ep/full_validation'


def combine_scores(probability, evidence):
    """A fixed ranking rule, not a calibrated posterior probability."""
    p = np.clip(np.asarray(probability, float), 1e-6, 1-1e-6)
    return np.log(p/(1-p)) + np.arcsinh(np.asarray(evidence, float))


def image_evidence(full, contribution, targets, valid, delta=3.):
    """Huber-loss increase on removing the focal object, per template L2 norm.

    Background and all neighbours remain unchanged. Each band receives equal
    weight later; the training-only image normalization is retained.
    """
    bands = []
    for prediction, component, target, mask in zip(full, contribution, targets, valid):
        with_object = F.huber_loss(prediction, target, reduction='none', delta=delta)
        without_object = F.huber_loss(prediction-component, target, reduction='none', delta=delta)
        numerator = ((without_object-with_object)*mask).sum((-2, -1))
        denominator = ((component.square()*mask).sum((-2, -1))).sqrt().clamp_min(1e-8)
        bands.append(numerator/denominator)
    return torch.stack(bands, -1)


def normalized_images(data_cfg, tile, meta):
    images, variance, wcs = load_bands(data_cfg, tile.name)
    headers = [ww.to_header(relax=True).tostring() for ww in wcs]
    mask_cache, targets, valid = {}, [], []
    vh, vw = tile.mask.shape
    for band, (im, var, ww) in enumerate(zip(images, variance, wcs)):
        key = headers[band], im.shape
        if key not in mask_cache:
            if headers[band] == headers[6]:
                artifact = tile.mask
            else:
                y, x = np.mgrid[:im.shape[0], :im.shape[1]]
                sky = ww.all_pix2world(np.column_stack([x.ravel(), y.ravel()]), 0)
                xy = wcs[6].all_world2pix(sky, 0)
                inside = np.isfinite(xy).all(1) & (xy[:, 0] >= 0) & (xy[:, 0] < vw) & (xy[:, 1] >= 0) & (xy[:, 1] < vh)
                artifact = np.ones(len(xy), bool)
                ii = np.flatnonzero(inside)
                artifact[ii] = tile.mask[xy[ii, 1].astype(int), xy[ii, 0].astype(int)]
                artifact = artifact.reshape(im.shape)
            mask_cache[key] = artifact
        good = np.isfinite(im) & np.isfinite(var) & (var > 0) & ~mask_cache[key]
        norm = meta['normalization']
        target = np.where(good, (im-norm['center'][band])/norm['scale'][band], 0).astype(np.float32)
        targets.append(target); valid.append(good)
    return targets, valid, wcs


def make_scene(focal, features, band_positions, targets, valid, meta):
    shapes, stamps = meta['band_shapes'], meta['stamp_sizes']
    origins = np.asarray([np.rint(band_positions[focal, b]-(np.array(shape[::-1])-1)/2).astype(int)
                          for b, shape in enumerate(shapes)])
    local = band_positions-origins[None]
    include = np.zeros(len(features), bool)
    for b, ((h, w), stamp) in enumerate(zip(shapes, stamps)):
        radius = (stamp-1)/2+1
        include |= ((local[:, b, 0] >= -radius) & (local[:, b, 0] < w+radius)
                    & (local[:, b, 1] >= -radius) & (local[:, b, 1] < h+radius))
    ids = np.r_[focal, np.flatnonzero(include & (np.arange(len(features)) != focal))]
    if len(ids) > meta['config']['max_objects_per_crop']:
        return None, 'too_many_context_objects'
    vv = [cutout(v, origin, shape, fill=False) for v, origin, shape in zip(valid, origins, shapes)]
    if min(v.mean() for v in vv) < meta['config']['minimum_valid_fraction']:
        return None, 'insufficient_valid_pixels'
    return {'features': features[ids], 'positions': local[ids].astype(np.float32), 'valid': vv,
            'targets': [cutout(t, origin, shape) for t, origin, shape in zip(targets, origins, shapes)]}, None


@torch.no_grad()
def score_batch(model, scenes, device, cfg):
    n, k, dim = len(scenes), max(len(s['features']) for s in scenes), scenes[0]['features'].shape[1]
    features = np.zeros((n, k, dim), np.float32)
    present = np.zeros((n, k), bool)
    positions = np.zeros((n, k, 10, 2), np.float32)
    for i, s in enumerate(scenes):
        count = len(s['features'])
        features[i, :count] = s['features']; present[i, :count] = True; positions[i, :count] = s['positions']
    features, present, positions = [torch.from_numpy(a).to(device) for a in (features, present, positions)]
    targets = [torch.from_numpy(np.stack([s['targets'][b] for s in scenes])).to(device) for b in range(10)]
    valid = [torch.from_numpy(np.stack([s['valid'][b] for s in scenes])).to(device) for b in range(10)]
    full = model(features, present, positions)
    solo = model(features[:, :1], present[:, :1], positions[:, :1])
    # Discard the solo prediction's different background. Only the focal
    # source's additive contribution is subtracted from the full prediction.
    component = [im-solo['background'][:, b, None, None] for b, im in enumerate(solo['images'])]
    result = image_evidence(full['images'], component, targets, valid, cfg['huber_delta'])
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite image evidence')
    return result.cpu().numpy()


def worker(out, shard, shards, device_name):
    torch.set_num_threads(4)
    device = torch.device(device_name)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    protocol = json.loads((out/'protocol.json').read_text())
    checkpoint = STUDY/'training/final.pt'
    if digest(checkpoint) != protocol['decoder_sha256']: raise ValueError('Decoder changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    meta, cfg = state['metadata'], state['metadata']['config']
    data_cfg = json.loads(resolve(cfg['detection_config']).read_text())
    _, tiles, _ = load_prepared(EVALUATION)
    with np.load(EVALUATION/'control_catalog.npz') as z:
        cat = {k: z[k][z['scores'] > .15] for k in z.files}
    model = ObjectRenderer(meta['feature_dim'], cfg, meta['band_shapes'], meta['stamp_sizes'], meta['pixel_scales_arcsec']).to(device)
    model.load_state_dict(state['model']); model.eval()
    before = digest(checkpoint)
    started, done, count_done = time.monotonic(), 0, 0
    work = list(range(shard, len(tiles), shards))
    for tile_index in work:
        tile = tiles[tile_index]
        query = np.flatnonzero(cat['origin_tile'] == tile_index)
        source = STUDY/'prepared/tiles'/f'{tile.name}.npz'
        if digest(source) != meta['tiles'][tile.name]['prepared_sha256']: raise ValueError('Prepared tile changed')
        with np.load(source) as z:
            features, xy, confidence = z['object_features'], z['proposal_xy_vis'], z['proposal_scores']
        projected = np.column_stack(tile.wcs.all_world2pix(cat['ra'][query], cat['dec'][query], 0))
        distance, ids = cKDTree(xy).query(projected)
        if np.any(distance > 1e-3) or len(np.unique(ids)) != len(ids): raise ValueError('Candidate coordinates do not match frozen pool')
        if not np.allclose(confidence[ids], cat['scores'][query], rtol=0, atol=1e-6): raise ValueError('Candidate confidence changed')
        targets, valid, wcs = normalized_images(data_cfg, tile, meta)
        world = wcs[6].all_pix2world(xy, 0)
        positions = np.stack([ww.all_world2pix(world, 0) for ww in wcs], 1)
        evidence = np.zeros((len(query), 10), np.float32)
        fallback = np.zeros(len(query), bool)
        reasons = {'too_many_context_objects': 0, 'insufficient_valid_pixels': 0}
        pending, pending_ids = [], []
        for i, focal in enumerate(ids):
            scene, reason = make_scene(focal, features, positions, targets, valid, meta)
            if scene is None:
                fallback[i] = True; reasons[reason] += 1
            else:
                pending.append(scene); pending_ids.append(i)
            if len(pending) == 16 or (i+1 == len(ids) and pending):
                evidence[pending_ids] = score_batch(model, pending, device, cfg)
                pending, pending_ids = [], []
        save_npz(out/'tiles'/f'{tile.name}.npz', catalogue_index=query, evidence_by_band=evidence, fallback=fallback)
        done += 1; count_done += len(query)
        row = {'shard': shard, 'tiles_done': done, 'tiles_total': len(work), 'candidates_done': count_done,
               'last_tile': tile.name, 'last_tile_fallback': reasons, 'elapsed_seconds': time.monotonic()-started}
        write_json(out/f'progress_{shard}.json', row)
        print(json.dumps(row), flush=True)
    assert digest(checkpoint) == before
    write_json(out/f'worker_{shard}_complete.json', {'tiles': done, 'candidates': count_done, 'optimizer_updates': 0,
                                                  'checkpoint_unchanged': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    worker(args.out.resolve(), args.shard, args.shards, args.device)
