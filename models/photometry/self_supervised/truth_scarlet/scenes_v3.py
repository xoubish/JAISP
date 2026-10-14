"""Version-3 training scenes: realistic size-brightness relation for the injected galaxies.

The v2 real-tile diagnosis: heads overestimate compact (faint) sources and underestimate
extended ones, because v2 training galaxies all had intrinsic half-light radii ~0.4"
regardless of brightness (dimmed bright donors, and a fixed 2.3" reconstruction support
that turned noise into a halo). v3 changes only the donors and their brightness:

  * nonparametric donors (aperture VIS S/N >= 8) reconstructed inside their own detection
    footprint (smoothed S/N > 2, grown 0.2", at least 0.25" and at most 2.3" in radius), or in
    the full 3" disk when the footprint holds S/N > 30 (bright galaxies keep their outskirts);
  * fainter galaxies (aperture S/N 3-8) as Sersic profiles with MER's VIS Sersic radius,
    index and axis ratio and MER's position angle; their colours are data-only amplitude
    fits of that profile in each band (MER fluxes are never used); MER point-like sources
    (probability > 0.8) as point sources;
  * point-like donors with no counterpart in another band (S/N < 8) are dropped;
  * every injected source within a factor 2 of its own model S/N (scenes_v2 native=True).
Scene construction (75% real-scene injections, 25% blank sky) is unchanged from v2.

    M=models.photometry.self_supervised.truth_scarlet.scenes_v3
    python -m $M library --split train; python -m $M library --split val
    python -m $M sizecheck --split train
    python -m $M export --split train --count 16000; python -m $M export --split val --count 800
    python -m $M reference
    python -m $M export_b --split train --count 16000     # v3b: augmented, bright extended galaxies oversampled
    python -m $M library4; python -m $M sizecheck4; python -m $M export4 --count 20000   # v4: 105 training tiles
"""
import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label, binary_dilation
from scipy.spatial import cKDTree
import galsim

from ..core import BANDS
from .. import empirical as emp
from ..data import jacobian
from ..pixel_psf import normalize_kernel
from ..detcat.prepare import load_inputs, _crop
from .scenes import TILES, regions_of, require_mer, tile_dir, extra_regions
from . import scenes_v2 as v2

VERSION = 'v3'
CAL = v2.SIGMAS


RADIUS = 3.0   # arcsec: reconstruction disk for bright donors = the head's profile extent (amortised_scarlet.R_MAX)


def footprint_support(signal, variance, valid, center, scale, threshold=2., grow_arcsec=.2, min_arcsec=.25, max_arcsec=RADIUS, bright_snr=30.):
    """Reconstruction support = the source's own detection footprint (faint sources), or the full 2.3\" disk when the
    footprint holds S/N > bright_snr, where the reconstruction is reliable and the outskirts are real light."""
    rms = np.sqrt(np.median(variance[valid])); s = 1.
    smoothed = gaussian_filter(np.where(valid, signal, 0.), s) / (rms / (2 * np.sqrt(np.pi) * s))   # S/N of the Gaussian-smoothed map
    lab, _ = label(smoothed > threshold)
    yy, xx = np.indices(signal.shape); rr = np.hypot(xx - center[0], yy - center[1]) * scale
    ids = np.unique(lab[(rr < .15) & (lab > 0)])                     # components touching the centre
    seg = np.isin(lab, ids) if len(ids) else np.zeros(signal.shape, bool)
    if seg.any(): seg = binary_dilation(seg, iterations=max(1, int(round(grow_arcsec / scale))))
    support = (seg | (rr < min_arcsec)) & (rr < max_arcsec)
    # Bright sources: their faint outskirts are real light beyond the detection footprint; keep the full disk.
    if signal[support & valid].sum() / np.sqrt(variance[support & valid].sum()) > bright_snr: return rr < max_arcsec
    return support


def intrinsic_r50(latent, center, scale=.1):
    yy, xx = np.indices(latent.shape); r = np.hypot(xx - center[0], yy - center[1]) * scale
    o = np.argsort(r.ravel()); c = np.cumsum(latent.ravel()[o]); return float(r.ravel()[o][np.searchsorted(c, .5 * c[-1])])


# ----------------------------------------------------------------------------- Sersic donors
def _band_kernel(inputs, b, i):
    if b.startswith('euclid_'): return normalize_kernel(inputs[b + '__psf_stamps'][inputs[b + '__kernel_index'][i]])
    return emp.gaussian_psf(CAL[b])


def _sersic_latent(m, size=81, oversample=4):
    """Intrinsic VIS-grid image from MER structural parameters (no MER flux): the profile is
    evaluated directly (no FFT) on a 4x finer grid and binned to 0.1" pixels."""
    if np.nan_to_num(m.mer_point_like_prob) > .8:
        img = np.zeros((size, size)); img[size // 2, size // 2] = 1.; return img
    gal = galsim.Sersic(n=float(np.clip(m.mer_sersic_sersic_vis_index, .3, 6.)), half_light_radius=float(max(m.mer_sersic_sersic_vis_radius, .02)))
    gal = gal.shear(q=float(np.clip(m.mer_sersic_sersic_vis_axis_ratio, .1, 1.)), beta=float(np.nan_to_num(m.mer_position_angle)) * galsim.degrees)
    fine = gal.drawImage(nx=size * oversample, ny=size * oversample, scale=.1 / oversample, method='no_pixel').array
    return emp.unit(np.maximum(fine.reshape(size, oversample, size, oversample).sum((1, 3)), 0))


def _sersic_region(task):
    out, region, snr_lo, snr_hi = task
    folder = tile_dir(region); inputs = load_inputs(folder)
    match = pd.read_csv(folder / 'mer_match.csv').set_index('source')
    xy = inputs['euclid_VIS__positions']; nearest = cKDTree(xy).query(xy, k=2)[0][:, 1] * .1
    out.mkdir(parents=True, exist_ok=True); rows = []; k = 0
    for i in np.flatnonzero(inputs['scene_inside'] & inputs['central_valid'] & (nearest > 2.)):
        m = match.loc[int(inputs['source'][i])]
        if not (bool(m.matched) and bool(m.primary_match)) or not np.isfinite(m.mer_sersic_sersic_vis_radius): continue
        # aperture S/N exactly as make_library, to split the faint range from the nonparametric library
        im, var, valid, wc = emp.band_data(inputs, 'euclid_VIS'); crop = _crop(im, var, valid, xy[i], 40)
        if crop is None: continue
        im, var, valid, origin = crop; p = xy[i] - origin
        yy, xx = np.indices(im.shape); radius = np.hypot(xx - p[0], yy - p[1])
        ann = valid & (radius > 30) & (radius < 39); aperture = valid & (radius < 25)
        if valid.mean() < .98 or ann.sum() < 100: continue
        bg = np.median(im[ann]); ap_snr = (im[aperture] - bg).sum() / np.sqrt(var[aperture].sum() + aperture.sum() ** 2 * np.median(var[ann]) / ann.sum())
        if not (snr_lo <= ap_snr < snr_hi): continue
        latent = _sersic_latent(m); vis_center = np.array([(latent.shape[1] - 1) / 2, (latent.shape[0] - 1) / 2])
        arrays = {}; vis_matrix = None; failure = None; band_snr = {}
        for b in ('euclid_VIS',) + tuple(x for x in BANDS if x != 'euclid_VIS'):
            image, variance, mask, bwc = emp.band_data(inputs, b)
            pix = np.array(bwc.world_to_pixel_values(*inputs['sky'][i])); matrix = jacobian(bwc, pix); scale = np.sqrt(abs(np.linalg.det(matrix)))
            if b == 'euclid_VIS': vis_matrix = matrix
            c = _crop(image, variance, mask, pix, int(np.ceil(4 / scale)))
            if c is None: failure = 'edge_' + b; break
            bim, bvar, bval, borig = c; center = pix - borig
            yy, xx = np.indices(bim.shape); rr = np.hypot(xx - center[0], yy - center[1]) * scale
            ann = bval & (rr > 3.) & (rr < 3.9)
            for q in emp.catalog_xy(folder, inputs, bwc) - borig:
                if np.linalg.norm(q - center) * scale > .3: ann &= np.hypot(xx - q[0], yy - q[1]) * scale > 1.
            if ann.sum() < 50 or bval[rr < 2.3].mean() < .98: failure = 'mask_or_annulus_' + b; break
            signal = np.where(bval, bim - np.median(bim[ann]), 0); kernel = _band_kernel(inputs, b, i)
            profile = emp.render(latent, vis_matrix, matrix, center, bim.shape, donor_center=vis_center, kernel=kernel, kernel_matrix=matrix)
            f, e = emp.amplitude(signal, bvar, bval, profile); band_snr[b] = f / e
            sed = emp.positive_amplitude(f, e)
            if not np.isfinite(sed) or sed <= 0: failure = 'bad_SED_' + b; break
            arrays.update({b + '__latent': latent.astype('float32'), b + '__kernel': kernel, b + '__matrix': vis_matrix, b + '__kernel_matrix': matrix,
                           b + '__center': vis_center, b + '__flux': sed, b + '__flux_error': e, b + '__fallback': b != 'euclid_VIS',
                           b + '__sigma': CAL[b]})
        if failure: continue
        arrays.update(donor=k, region=region, source=int(inputs['source'][i]), sky=inputs['sky'][i], nearest_arcsec=nearest[i], selection_snr=ap_snr, kind='sersic')
        np.savez_compressed(out / f'sersic_{k:05d}.npz', **arrays)
        rows.append(dict(file=f'sersic_{k:05d}.npz', region=region, source=int(inputs['source'][i]), aperture_snr=ap_snr,
                         point_like_prob=float(np.nan_to_num(m.mer_point_like_prob)), strongest_non_VIS_snr=max(v for kk, v in band_snr.items() if kk != 'euclid_VIS'),
                         mer_radius=float(m.mer_sersic_sersic_vis_radius))); k += 1
    return region, rows


# ----------------------------------------------------------------------------- library
def library(split, workers, version=VERSION, regions=None):
    out = v2.split_dir(split, version); regions = regions_of(split) if regions is None else list(regions)
    v2.library(split, 8., workers, version=version, support_fn=footprint_support, radius=RADIUS, regions=regions)   # nonparametric, aperture S/N >= 8
    q = pd.read_csv(out / 'donor_quality.csv'); q['kind'] = 'nonparametric'
    tmp = out / 'sersic_by_region'
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_sersic_region, [(tmp / f'region_{r:04d}', r, 3., 8.) for r in regions]))
    k = int(q.donor.max()) + 1; extra = []
    for region, rows in results:
        for row in rows:
            d = emp.read_npz(tmp / f'region_{region:04d}' / row['file']); d['donor'] = k
            np.savez_compressed(out / 'donors' / f'donor_{k:05d}.npz', **d)
            extra.append(dict(donor=k, region=region, source=row['source'], kind='sersic', point_like_prob=row['point_like_prob'],
                              strongest_non_VIS_snr=row['strongest_non_VIS_snr'], suitable_v2=True, reasons='')); k += 1
    q = pd.concat([q, pd.DataFrame(extra)], ignore_index=True)
    # common columns for all donors: model S/N (native brightness), intrinsic r50, VIS magnitude
    snr, r50, mag = {}, {}, {}
    zp = {r: float(load_inputs(tile_dir(r))['euclid_VIS__magzero']) for r in q.region.unique()}
    for d in q.donor:
        z = emp.read_npz(out / 'donors' / f'donor_{int(d):05d}.npz')
        snr[d] = float(z['euclid_VIS__flux'] / z['euclid_VIS__flux_error'])
        r50[d] = intrinsic_r50(z['euclid_VIS__latent'], z['euclid_VIS__center'])
        mag[d] = zp[int(z['region'])] - 2.5 * np.log10(float(z['euclid_VIS__flux']))
    q['selection_snr'] = q.donor.map(snr); q['r50_arcsec'] = q.donor.map(r50); q['vis_mag'] = q.donor.map(mag)
    q['point_like'] = q.point_like_prob.fillna(0) > .5
    no_counterpart = q.point_like & (q.strongest_non_VIS_snr < 8)
    q.loc[no_counterpart, 'suitable_v2'] = False
    q.loc[no_counterpart, 'reasons'] = q.loc[no_counterpart, 'reasons'].fillna('') + ';point_like_without_counterpart'
    q.to_csv(out / 'donor_quality.csv', index=False)
    s = q[q.suitable_v2]
    print(f'{split}_{version}: {len(s)} suitable donors ({int((s.kind == "sersic").sum())} Sersic, {int(s.point_like.sum())} point-like); '
          f'dropped {int(no_counterpart.sum())} point-like without counterpart', flush=True)


def sizecheck(split, version=VERSION, regions=None):
    """Median intrinsic r50 of the donors by VIS magnitude, next to MER's Sersic radius for real detections."""
    q = pd.read_csv(v2.split_dir(split, version) / 'donor_quality.csv'); q = q[q.suitable_v2]
    edges = [18, 22, 23, 24, 25, 26, 28]; bins = list(zip(edges[:-1], edges[1:]))
    real = pd.concat([pd.read_csv(tile_dir(r) / 'mer_match.csv') for r in (regions_of(split) if regions is None else regions)])
    real = real[real.matched & (real.mer_flux_vis_sersic > 0)]; rmag = 23.9 - 2.5 * np.log10(real.mer_flux_vis_sersic)
    print(f"{'VIS mag':>10}" + ''.join(f'{f"{a}-{b}":>10}' for a, b in bins))
    for label, sel, col in (('v3 donors', lambda a, b: (q.vis_mag >= a) & (q.vis_mag < b), q.r50_arcsec),):
        print(f'{label:>10}' + ''.join(f'{np.median(col[sel(a, b)]) if sel(a, b).sum() > 4 else np.nan:10.3f}' for a, b in bins))
        print(f"{'N':>10}" + ''.join(f'{int(sel(a, b).sum()):10d}' for a, b in bins))
    print(f"{'MER real':>10}" + ''.join(f'{np.median(real.mer_sersic_sersic_vis_radius[(rmag >= a) & (rmag < b)]):10.3f}' for a, b in bins))
    print('(donor r50: half-light radius of the intrinsic VIS image; MER: VIS Sersic half-light radius; both arcsec)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('library', 'sizecheck', 'export', 'export_b', 'reference', 'library4', 'sizecheck4', 'export4'))
    p.add_argument('--split', choices=('train', 'val'), default='train')
    p.add_argument('--count', type=int, default=16000); p.add_argument('--workers', type=int, default=40)
    p.add_argument('--real-fraction', type=float, default=.75)
    a = p.parse_args()
    if a.stage == 'library': require_mer(regions_of(a.split)); library(a.split, a.workers)
    elif a.stage == 'sizecheck': sizecheck(a.split)
    elif a.stage == 'export': v2.export(a.split, a.count, a.workers, a.real_fraction, version=VERSION, native=True)
    elif a.stage == 'export_b':   # v3b: same v3 donors, every galaxy randomly rotated/flipped, 30% bright extended primaries
        v2.export(a.split, a.count, a.workers, a.real_fraction, version='v3b', native=True, augment=True, p_big=.3, donors_from=VERSION)
    elif a.stage == 'reference': v2.reference(a.workers, version=VERSION)
    # v4: donors and real injection scenes from the 40 original + 65 extra training tiles; validation stays val_v3.
    elif a.stage == 'library4':
        regions = regions_of('train') + extra_regions(); require_mer(regions); library('train', a.workers, version='v4', regions=regions)
    elif a.stage == 'sizecheck4': sizecheck('train', version='v4', regions=regions_of('train') + extra_regions())
    elif a.stage == 'export4':
        v2.export('train', a.count, a.workers, a.real_fraction, version='v4', native=True, augment=True, p_big=.3, regions=regions_of('train') + extra_regions())


if __name__ == '__main__': main()
