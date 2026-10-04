"""Assemble per-tile fitting inputs and build 12-arcsec scenes around each detection.

Euclid pixels, variance, flags and GRID PSFs come from the archive cutouts;
Rubin pixels come from the local tile. Every band is sampled at the corrected
VIS-canonical sky position through its own WCS. The same masks are written for
the Tractor step so both photometers see identical pixels.
"""
import argparse
import traceback
import numpy as np
import pandas as pd
import torch
from .common import (OUT, EUCLID, EUCLID_BANDS, RUBIN_BANDS, VIS_BAD, NISP_BAD, STARSIGNAL, RUBIN_BAD,
                     SCENE_HALF_ARCSEC, NEIGHBOR_ARCSEC, MAX_SCENE_SOURCES, region_dir, tangent_arcsec, read_json, write_json)
from ..data import wcs, jacobian
from ..pixel_psf import normalize_kernel


def euclid_mask(short, image, variance, flag, visstar):
    bad = VIS_BAD if short == 'VIS' else NISP_BAD
    mask = np.isfinite(image) & np.isfinite(variance) & (variance > 0) & ((flag.astype('int64') & bad) == 0) & ~visstar
    if mask.any():
        typical = np.median(variance[mask]); mask &= variance < typical * 1e6  # archive "no coverage" RMS
    return mask


def nearest_kernel(sky, psf_sky):
    distance = np.hypot((sky[:, None, 0] - psf_sky[None, :, 0]) * np.cos(np.deg2rad(sky[:, None, 1])), sky[:, None, 1] - psf_sky[None, :, 1])
    return distance.argmin(1)


def pixel_identity(local_image, local_wcs, archive_image, archive_wcs):
    """Compare the local tile with the archive cutout on their common pixels."""
    h, w = local_image.shape
    ra, dec = local_wcs.pixel_to_world_values(0, 0)
    x0, y0 = archive_wcs.world_to_pixel_values(ra, dec)
    offset = np.array([x0, y0]); integer = np.rint(offset)
    if np.abs(offset - integer).max() > 1e-2: return dict(aligned=False, offset=offset.tolist())
    ox, oy = integer.astype(int)
    xs = slice(max(0, ox), min(archive_image.shape[1], ox + w)); ys = slice(max(0, oy), min(archive_image.shape[0], oy + h))
    a = archive_image[ys, xs]; l = local_image[ys.start - oy:ys.stop - oy, xs.start - ox:xs.stop - ox]
    both = np.isfinite(a) & np.isfinite(l)
    diff = np.abs(a[both] - l[both])
    return dict(aligned=True, offset=integer.tolist(), compared=int(both.sum()), max_abs_diff=float(diff.max()) if diff.size else None,
                fraction_equal=float((diff <= 1e-5 * np.maximum(1, np.abs(l[both]))).mean()) if diff.size else None)


def build_inputs(folder):
    meta = read_json(folder / 'metadata.json'); det = pd.read_csv(folder / 'detections.csv')
    sky = det[['ra', 'dec']].to_numpy(float)
    arrays = dict(sky=sky, source=det.source.to_numpy(), tile=meta['tile'])
    raw = {}
    for short in EUCLID:
        with np.load(folder / f'{short}.npz') as z: raw[short] = {k: z[k] for k in z.files}
    vis_shape = raw['VIS']['image'].shape
    visstar = (raw['VIS']['flag'].astype('int64') & STARSIGNAL) != 0
    report = dict(region=meta['region'], tile=meta['tile'], bands={})
    inside = np.ones(len(sky), bool); radius = int(np.ceil(SCENE_HALF_ARCSEC / .1))
    for short, d in raw.items():
        b = 'euclid_' + short; wc = wcs(d['wcs'])
        if d['image'].shape != vis_shape: raise ValueError(f'{short}: cutout grid differs from VIS')
        if not np.allclose(wc.wcs.crval, wcs(raw['VIS']['wcs']).wcs.crval) or not np.allclose(wc.wcs.crpix, wcs(raw['VIS']['wcs']).wcs.crpix, atol=1e-3):
            raise ValueError(f'{short}: WCS differs from VIS; STARSIGNAL mask cannot be shared')
        im = d['image'].astype('float32'); var = d['variance'].astype('float32')
        mask = euclid_mask(short, im, var, d['flag'], visstar)
        xy = np.column_stack(wc.world_to_pixel_values(*sky.T))
        h, w = im.shape
        inside &= (xy[:, 0] >= radius) & (xy[:, 0] <= w - 1 - radius) & (xy[:, 1] >= radius) & (xy[:, 1] <= h - 1 - radius)
        psf_sky = np.column_stack([d['psf_ra'], d['psf_dec']])
        arrays.update({b + '__image': im, b + '__variance': var, b + '__mask': mask, b + '__wcs': d['wcs'], b + '__magzero': float(d['magzero']),
                       b + '__psf_stamps': d['psf_stamps'], b + '__psf_sky': psf_sky, b + '__psf_fwhm': d['psf_fwhm'],
                       b + '__positions': xy.astype('float64'), b + '__kernel_index': nearest_kernel(sky, psf_sky)})
        report['bands'][b] = dict(valid_fraction=float(mask.mean()), starsignal_fraction=float(visstar.mean()), magzero=float(d['magzero']), shape=list(im.shape))
    with np.load(meta['euclid'], allow_pickle=True) as e:
        report['pixel_identity'] = pixel_identity(np.asarray(e['img_VIS'], 'float32'), wcs(e['wcs_VIS']), raw['VIS']['image'], wcs(raw['VIS']['wcs']))
    with np.load(meta['rubin'], allow_pickle=True) as r:
        wc = wcs(r['wcs_hdr']); assert [str(x) for x in r['bands']] == [b[-1] for b in RUBIN_BANDS]
        im = np.asarray(r['img'], 'float32'); var = np.asarray(r['var'], 'float32'); flags = np.asarray(r['mask'])
        mask = np.isfinite(im) & np.isfinite(var) & (var > 0) & ((flags & RUBIN_BAD) == 0)
        xy = np.column_stack(wc.world_to_pixel_values(*sky.T))
        scale = np.sqrt(abs(np.linalg.det(jacobian(wc, np.array(im.shape[1:]) / 2.))))
        rr = int(np.ceil(1.5 / scale)); h, w = im.shape[1:]
        # The Rubin tile is 6 arcsec smaller than the Euclid one: scenes may extend past its edge (padded, masked).
        inside &= (xy[:, 0] >= rr) & (xy[:, 0] <= w - 1 - rr) & (xy[:, 1] >= rr) & (xy[:, 1] <= h - 1 - rr)
        arrays.update({'rubin__image': im, 'rubin__variance': var, 'rubin__mask': mask, 'rubin__wcs': wc.to_header().tostring(),
                       'rubin__positions': xy.astype('float64'), 'rubin__scale_arcsec': scale})
        report['rubin'] = dict(valid_fraction=[float(m.mean()) for m in mask], scale_arcsec=float(scale))
    xx, yy = np.rint(arrays['euclid_VIS__positions']).astype(int).T
    ok = (xx >= 0) & (yy >= 0) & (xx < vis_shape[1]) & (yy < vis_shape[0]); central = np.zeros(len(sky), bool)
    central[ok] = arrays['euclid_VIS__mask'][yy[ok], xx[ok]]
    arrays['scene_inside'] = inside; arrays['central_valid'] = central
    report.update(n_sources=len(sky), n_inside=int(inside.sum()), n_inside_valid=int((inside & central).sum()))
    np.savez_compressed(folder / 'tile_inputs.npz', **arrays)
    write_json(folder / 'prepare.json', report)
    return report


def load_inputs(folder):
    with np.load(folder / 'tile_inputs.npz', allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}
    for k, v in out.items():
        if v.ndim == 0: out[k] = v.item()
    return out


def _crop(image, variance, mask, xy, radius, pad=False):
    origin = np.rint(xy).astype(int) - radius; x, y = origin; n = 2 * radius + 1
    if x < 0 or y < 0 or x + n > image.shape[1] or y + n > image.shape[0]:
        if not pad: return None
        im = np.full((n, n), np.nan, image.dtype); var = np.full((n, n), np.nan, variance.dtype); m = np.zeros((n, n), bool)
        xs = slice(max(x, 0), min(x + n, image.shape[1])); ys = slice(max(y, 0), min(y + n, image.shape[0]))
        if xs.stop <= xs.start or ys.stop <= ys.start: return None
        target = (slice(ys.start - y, ys.stop - y), slice(xs.start - x, xs.stop - x))
        im[target] = image[ys, xs]; var[target] = variance[ys, xs]; m[target] = mask[ys, xs]
        return im, var, m, origin
    return image[y:y + n, x:x + n].copy(), variance[y:y + n, x:x + n].copy(), mask[y:y + n, x:x + n].copy(), origin


def scene_for_source(inputs, i, psf_sigmas, half_arcsec=SCENE_HALF_ARCSEC, neighbor_arcsec=NEIGHBOR_ARCSEC, max_sources=MAX_SCENE_SOURCES):
    """Scene centred on detection i (index 0 of the scene) with its neighbours; returns (scene, info) or (None, reason)."""
    sky = inputs['sky']; offsets = tangent_arcsec(sky, sky[i]); distance = np.linalg.norm(offsets, axis=1)
    members = np.flatnonzero(distance < neighbor_arcsec); members = members[np.argsort(distance[members], kind='stable')]
    if members[0] != i: members = np.r_[i, members[members != i]]
    truncated = len(members) > max_sources; members = members[:max_sources]
    bands = {}; info = dict(n_sources=int(len(members)), truncated=bool(truncated), valid_fraction={},
                            nearest_neighbor_arcsec=float(np.sort(distance)[1]) if len(sky) > 1 else np.inf)
    for short in EUCLID:
        b = 'euclid_' + short; pos = inputs[b + '__positions']; wc = wcs(inputs[b + '__wcs'])
        radius = int(np.ceil(half_arcsec / .1))
        crop = _crop(inputs[b + '__image'], inputs[b + '__variance'], inputs[b + '__mask'], pos[i], radius)
        if crop is None: return None, 'edge_' + short
        im, var, mask, origin = crop
        if short == 'VIS':
            cx, cy = np.rint(pos[i]).astype(int) - origin
            if not mask[cy, cx]: return None, 'masked_center'
            if mask.mean() < .5: return None, 'masked_scene'
        kernels = np.array([normalize_kernel(inputs[b + '__psf_stamps'][k]) for k in inputs[b + '__kernel_index'][members]])
        jac = np.linalg.inv(jacobian(wc, pos[i]))
        bands[b] = dict(image=torch.tensor(im), variance=torch.tensor(var), mask=torch.tensor(mask),
                        positions=torch.tensor(pos[members] - origin, dtype=torch.float32),
                        sky_to_pixel=torch.tensor(jac, dtype=torch.float32), psf_sigma=psf_sigmas[b], psf_kernels=kernels)
        info['valid_fraction'][b] = float(mask.mean())
    wc = wcs(inputs['rubin__wcs']); pos = inputs['rubin__positions']; radius = int(np.ceil(half_arcsec / inputs['rubin__scale_arcsec']))
    jac = np.linalg.inv(jacobian(wc, pos[i]))
    for k, b in enumerate(RUBIN_BANDS):
        crop = _crop(inputs['rubin__image'][k], inputs['rubin__variance'][k], inputs['rubin__mask'][k], pos[i], radius, pad=True)
        if crop is None: return None, 'edge_rubin'
        im, var, mask, origin = crop
        bands[b] = dict(image=torch.tensor(im), variance=torch.tensor(var), mask=torch.tensor(mask),
                        positions=torch.tensor(pos[members] - origin, dtype=torch.float32),
                        sky_to_pixel=torch.tensor(jac, dtype=torch.float32), psf_sigma=psf_sigmas[b])
        info['valid_fraction'][b] = float(mask.mean())
    scene = dict(tile=inputs['tile'], central=0, sky=sky[members], bands=bands, source_indices=members)
    return scene, info


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--regions', type=int, nargs='*'); a = p.parse_args()
    folders = sorted(f.parent for f in OUT.glob('region_*/H.npz'))
    if a.regions is not None: folders = [f for f in folders if int(f.name.split('_')[1]) in set(a.regions)]
    reports = []
    for f in folders:
        try: reports.append(build_inputs(f)); print(f.name, reports[-1]['n_inside_valid'], '/', reports[-1]['n_sources'], 'photometrable;', 'pixel identity', reports[-1]['pixel_identity'], flush=True)
        except Exception: reports.append(dict(region=f.name, error=traceback.format_exc())); print(reports[-1]['error'], flush=True)
    write_json(OUT / 'prepare_status.json', reports)
    if any('error' in r for r in reports): raise RuntimeError('Preparation failures; see prepare_status.json')


if __name__ == '__main__': main()
