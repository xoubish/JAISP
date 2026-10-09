"""Independent empirical, multiband injection pilot on held-out real sky.

Run from the project root with ``python -m models.photometry.self_supervised.empirical``.
Truth is a finite, positive, nonparametric reconstruction of real galaxies, not
a JAISP/Tractor profile. GalSim renders it independently with one pixel response.
Neither catalog fluxes nor simulated truth are passed to the photometers.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import traceback
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.signal import fftconvolve
from scipy.ndimage import laplace
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.special import ndtr
import galsim

from .core import BANDS
from .data import wcs, jacobian
from .detcat.prepare import load_inputs, _crop

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / 'runs/empirical_injection_pilot_v2'
BACKGROUND_REGIONS = (3, 7, 11, 15, 19, 23, 27)
SEED = 20261004


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def unit(a):
    a = np.asarray(a, float)
    if not np.isfinite(a).all() or a.sum() <= 0:
        raise ValueError('Invalid positive-mass image')
    return a / a.sum()


def band_data(inputs, band):
    if band.startswith('rubin_'):
        k = 'ugrizy'.index(band[-1]); prefix = 'rubin'
        return (inputs[prefix+'__image'][k], inputs[prefix+'__variance'][k],
                inputs[prefix+'__mask'][k], wcs(inputs[prefix+'__wcs']))
    return (inputs[band+'__image'], inputs[band+'__variance'],
            inputs[band+'__mask'], wcs(inputs[band+'__wcs']))


def gaussian_psf(sigma, radius=18):
    """Independent GalSim PSF, including precisely one native pixel response."""
    return unit(galsim.Gaussian(sigma=float(sigma)).drawImage(
        nx=2*radius+1, ny=2*radius+1, scale=1., method='fft').array)


def reconstruct(image, variance, valid, kernel, support, steps=100):
    """Positive pixel reconstruction under a Gaussian likelihood.

    FISTA with fixed smoothness and sparsity regularization. All quantities are
    in local sky-RMS units. It has no parametric galaxy family or learned prior.
    Missing pixels have zero weight; negative PSF lobes remain in the operator.
    This is an approximate empirical truth model, not a claim of exact recovery
    of the donor's unobservable intrinsic light distribution.
    """
    rms = np.sqrt(np.median(variance[valid])); y = np.where(valid, image/rms, 0)
    weight = np.where(valid, rms*rms/np.maximum(variance, 1e-30), 0)
    kernel = unit(kernel); reverse = kernel[::-1, ::-1]
    # ||convolution|| <= sum(abs(kernel)); ||-Laplacian|| <= 8.
    smoothness, sparsity = .04, .15
    lipschitz = weight.max()*np.abs(kernel).sum()**2 + 8*smoothness
    q = np.maximum(y, 0)*support; z = q.copy(); t = 1.
    for _ in range(steps):
        pred = fftconvolve(z, kernel, mode='same')
        grad = fftconvolve(weight*(pred-y), reverse, mode='same') - smoothness*laplace(z, mode='constant')
        new = np.maximum(z-(grad+sparsity)/lipschitz, 0)*support
        nt = (1+np.sqrt(1+4*t*t))/2
        z = new+(t-1)/nt*(new-q); q, t = new, nt
    prediction = fftconvolve(q, kernel, mode='same')*rms
    chi2 = float(np.sum(((prediction-image)[valid])**2/variance[valid])/valid.sum())
    return q*rms, chi2


def amplitude(image, variance, valid, profile):
    """Signed data-only amplitude and constant sky, with conditional error."""
    a = np.column_stack((profile[valid], np.ones(valid.sum())))
    aw = a/np.sqrt(variance[valid, None]); yw = image[valid]/np.sqrt(variance[valid])
    cov = np.linalg.inv(aw.T@aw)
    return (cov@(aw.T@yw))[0], np.sqrt(cov[0, 0])


def positive_amplitude(mean, sigma):
    """Posterior mean with a nonnegative flat amplitude prior; weak SED flagged."""
    x = mean/sigma
    return mean + sigma*np.exp(-.5*x*x)/np.sqrt(2*np.pi)/max(ndtr(x), 1e-200)


def world_wcs(matrix):
    return galsim.JacobianWCS(*np.asarray(matrix, float).ravel())


def render(stamp, donor_matrix, target_matrix, position, shape, kernel=None,
           angle=0., extra_sigma_arcsec=0., donor_center=None, kernel_matrix=None):
    """Render finite unit-mass intrinsic light or a PSF; never renormalize crops.

    Native PSF stamps already contain their pixel response: draw ``no_pixel``.
    Donor/target matrices map native x,y pixels to tangent-plane arcsec.
    """
    stamp = unit(stamp)
    iwcs = world_wcs(donor_matrix); owcs = world_wcs(target_matrix)
    offset = None
    if donor_center is not None:
        offset = galsim.PositionD(*(np.asarray(donor_center)-(np.array(stamp.shape[::-1])-1)/2))
    obj = galsim.InterpolatedImage(galsim.ImageD(stamp, wcs=iwcs),
                                  x_interpolant=galsim.Lanczos(5), offset=offset)
    if kernel is not None:
        psf_wcs = iwcs if kernel_matrix is None else world_wcs(kernel_matrix)
        psf = galsim.InterpolatedImage(galsim.ImageD(unit(kernel), wcs=psf_wcs), x_interpolant=galsim.Lanczos(5))
        obj = galsim.Convolve(obj, psf)
    if angle:
        obj = obj.rotate(angle*galsim.radians)
    if extra_sigma_arcsec:
        obj = galsim.Convolve(obj, galsim.Gaussian(sigma=extra_sigma_arcsec))
    offset = galsim.PositionD(*(np.asarray(position)-(np.array(shape[::-1])-1)/2))
    return obj.drawImage(nx=shape[1], ny=shape[0], wcs=owcs,
                         method='no_pixel', offset=offset, dtype=np.float64).array.copy()


def catalog_xy(folder, inputs, wc):
    """All catalog positions, including MER objects missed by JAISP detection."""
    sky = inputs['sky']
    with fits.open(folder/'mer.fits', memmap=False) as hdus:
        m = hdus[1].data
        sky = np.r_[sky, np.column_stack((m['ra'], m['dec']))]
    return np.column_stack(wc.world_to_pixel_values(*sky.T))


def make_library(out, source_root, minimum_snr=25., regions=None):
    """Donors from every region except the background regions, or from ``regions`` only."""
    folder_out = out/'donors'; folder_out.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((HERE/'runs/q1_all_bands/psf_calibration.json').read_text())
    report = []; rejected = []; donor_id = 0
    for folder in sorted(source_root.glob('region_*')):
        region = int(folder.name.split('_')[-1])
        if (region in BACKGROUND_REGIONS if regions is None else region not in regions) or not (folder/'tile_inputs.npz').exists(): continue
        inputs = load_inputs(folder); xy = inputs['euclid_VIS__positions']
        nearest = cKDTree(xy).query(xy, k=2)[0][:, 1]*.1
        matches = pd.read_csv(folder/'mer_match.csv').set_index('source')
        for i in np.flatnonzero(inputs['scene_inside'] & inputs['central_valid'] & (nearest > 3.)):
            match = matches.loc[int(inputs['source'][i])]
            # Catalog used only for independently excluding likely stars.
            if pd.notna(match.mer_point_like_prob) and match.mer_point_like_prob > .2: continue
            b = 'euclid_VIS'; im, var, valid, wc = band_data(inputs, b)
            crop = _crop(im, var, valid, xy[i], 40)
            if crop is None: continue
            im, var, valid, origin = crop; p = xy[i]-origin
            yy, xx = np.indices(im.shape); radius = np.hypot(xx-p[0], yy-p[1])
            ann = valid & (radius > 30) & (radius < 39); aperture = valid & (radius < 25)
            if valid.mean() < .98 or ann.sum() < 100: continue
            bg = np.median(im[ann]); flux = (im[aperture]-bg).sum()
            err = np.sqrt(var[aperture].sum() + aperture.sum()**2*np.median(var[ann])/ann.sum())
            if flux/err < minimum_snr: continue
            arrays = {}; rows = []; vis_intrinsic = None; vis_matrix = None; vis_center = None
            failure = None
            # VIS first: common fallback for weak bands, explicitly recorded.
            for b in ('euclid_VIS',)+tuple(x for x in BANDS if x != 'euclid_VIS'):
                image, variance, mask, wc = band_data(inputs, b)
                pix = np.array(wc.world_to_pixel_values(*inputs['sky'][i]))
                matrix = jacobian(wc, pix); scale = np.sqrt(abs(np.linalg.det(matrix)))
                crop = _crop(image, variance, mask, pix, int(np.ceil(4/scale)))
                if crop is None: failure = 'edge_'+b; break
                im, var, valid, origin = crop; center = pix-origin
                yy, xx = np.indices(im.shape); rr = np.hypot(xx-center[0], yy-center[1])*scale
                ann = valid & (rr > 3.) & (rr < 3.9)
                # Exclude neighbors in the background annulus (all known catalogs).
                for q in catalog_xy(folder, inputs, wc)-origin:
                    if np.linalg.norm(q-center)*scale > .3:
                        ann &= np.hypot(xx-q[0], yy-q[1])*scale > 1.
                if ann.sum() < 50 or valid[rr < 2.3].mean() < .98:
                    failure = 'mask_or_annulus_'+b; break
                bg = float(np.median(im[ann])); signal = np.where(valid, im-bg, 0)
                if b.startswith('euclid_'):
                    kernel = unit(inputs[b+'__psf_stamps'][inputs[b+'__kernel_index'][i]])
                else: kernel = gaussian_psf(calibration[b]['sigma_px'])
                support = rr < 2.3
                fallback = False
                if b == 'euclid_VIS':
                    latent, chi2 = reconstruct(signal, var, valid, kernel, support)
                    if latent.sum() <= 0: failure = 'empty_VIS'; break
                    vis_intrinsic = unit(latent); vis_matrix = matrix
                    vis_center = np.array([(latent*xx).sum(), (latent*yy).sum()])/latent.sum()
                    # One world center, shared across all bands; retain color offsets.
                    center_world = matrix@(vis_center-center)
                    center = vis_center
                    profile = fftconvolve(vis_intrinsic, kernel, mode='same')
                    f, e = amplitude(signal, var, valid, profile)
                else:
                    center = center+np.linalg.solve(matrix, center_world)
                    # Convolve before sampling: a fine VIS intrinsic image must
                    # not be point-sampled on the coarser Rubin grid first.
                    profile = render(vis_intrinsic, vis_matrix, matrix, center, im.shape,
                                     donor_center=vis_center, kernel=kernel, kernel_matrix=matrix)
                    f, e = amplitude(signal, var, valid, profile)
                    fallback = f/e < 8
                    if fallback:
                        latent = vis_intrinsic; chi2 = float(np.mean((signal[valid]-f*profile[valid])**2/var[valid]))
                    else:
                        latent, chi2 = reconstruct(signal, var, valid, kernel, support)
                        if latent.sum() <= 0: failure = 'empty_'+b; break
                        profile = fftconvolve(unit(latent), kernel, mode='same')
                        f, e = amplitude(signal, var, valid, profile)
                # Known synthetic fluxes are posterior/data-derived colors in
                # native units, then a common dimming factor. Never MER fluxes.
                sed_flux = positive_amplitude(f, e)
                if not np.isfinite(sed_flux) or sed_flux <= 0: failure = 'bad_SED_'+b; break
                arrays.update({b+'__latent': unit(latent).astype('float32'), b+'__kernel': kernel,
                               b+'__matrix': vis_matrix if fallback else matrix,
                               b+'__kernel_matrix': matrix,
                               b+'__center': vis_center if fallback else center,
                               b+'__flux': sed_flux, b+'__flux_error': e,
                               b+'__fallback': fallback, b+'__sigma': calibration[b]['sigma_px']})
                rows.append(dict(donor=donor_id, band=b, snr=f/e, fallback=fallback,
                                 reconstruction_chi2=chi2, flux=sed_flux, flux_error=e))
            if failure:
                rejected.append(dict(region=region, source=int(inputs['source'][i]), reason=failure)); continue
            arrays.update(donor=donor_id, region=region, source=int(inputs['source'][i]),
                          sky=inputs['sky'][i], nearest_arcsec=nearest[i], selection_snr=flux/err)
            np.savez_compressed(folder_out/f'donor_{donor_id:04d}.npz', **arrays)
            report.extend(rows); donor_id += 1
        print(f'Donors after {folder.name}: {donor_id}', flush=True)
    if not donor_id: raise RuntimeError('No qualified donors')
    pd.DataFrame(report).to_csv(out/'donor_bands.csv', index=False)
    write_json(out/'donor_rejections.json', rejected)
    return donor_id


def sky_object_mask(image, valid, scale):
    """Independent all-band blank-sky guard; no injected pixels or truth used."""
    import sep
    data=np.ascontiguousarray(np.where(valid,image,0),dtype=np.float32)
    background=sep.Background(data,mask=~valid)
    residual=np.ascontiguousarray(data-background.back())
    objects,segments=sep.extract(residual,3.,err=background.rms(),mask=~valid,
                                 minarea=10,segmentation_map=True)
    noise=background.rms()[np.clip(np.rint(objects['y']).astype(int),0,image.shape[0]-1),
                           np.clip(np.rint(objects['x']).astype(int),0,image.shape[1]-1)]
    secure=objects['flux']/np.sqrt(np.maximum(objects['npix'],1))/np.maximum(noise,1e-20)>10
    detected=np.isin(segments,np.flatnonzero(secure)+1)
    grown=distance_transform_edt(~detected)<=int(np.ceil(1.5/scale)) if detected.any() else detected
    objects=objects[secure]
    return valid & ~grown,np.column_stack((objects['x'],objects['y']))


def make_backgrounds(out, source_root, per_region=50, regions=BACKGROUND_REGIONS, minimum_per_region=20):
    target = out/'backgrounds'; target.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED+1); rows = []; idx = 0
    for region in regions:
        folder = source_root/f'region_{region:03d}'; inputs = load_inputs(folder)
        wc = wcs(inputs['euclid_VIS__wcs']); xy = catalog_xy(folder, inputs, wc)
        tree = cKDTree(xy); chosen = []; h, w = inputs['euclid_VIS__image'].shape
        # VIS-only catalogs miss very red objects, stars and extended Rubin
        # wings. Detect independently in EACH band before drawing injections.
        # Use empirical local sky RMS, not the diagonal resampled variance.
        band_masks = {}; multi_positions = []; known_positions = {}
        for b in BANDS:
            im, var, valid, bwcs = band_data(inputs, b)
            scale = np.sqrt(abs(np.linalg.det(jacobian(bwcs, np.array(im.shape[::-1])/2))))
            known_positions[b] = catalog_xy(folder, inputs, bwcs)
            band_masks[b], object_xy = sky_object_mask(im, valid, scale)
            if len(object_xy):
                ra, dec = bwcs.pixel_to_world_values(*object_xy.T)
                multi_positions.extend(np.column_stack(wc.world_to_pixel_values(ra, dec)))
        multi_tree = cKDTree(np.array(multi_positions)) if multi_positions else None
        for _ in range(20000):
            p = rng.uniform([90, 90], [w-91, h-91])
            if tree.query(p)[0]*.1 < 4.: continue
            if multi_tree is not None and multi_tree.query(p)[0]*.1 < 4.: continue
            if chosen and np.min(np.linalg.norm(np.array(chosen)-p, axis=1))*.1 < 3.: continue
            sky = np.array(wc.pixel_to_world_values(*p)); arrays = {}; bad = False
            for b in BANDS:
                im, var, mask, bwcs = band_data(inputs, b)
                mask = band_masks[b]
                pos = np.array(bwcs.world_to_pixel_values(*sky)); matrix = jacobian(bwcs, pos)
                scale = np.sqrt(abs(np.linalg.det(matrix)))
                crop = _crop(im, var, mask, pos, int(np.ceil(6/scale)))
                if crop is None: bad = True; break
                image, variance, valid, origin = crop; center = pos-origin
                yy, xx = np.indices(image.shape)
                # Existing catalog sources outside the protected injection area
                # remain in the real sky but their cores are masked identically.
                near = known_positions[b]-origin
                near = near[(near[:,0]>-1.5/scale)&(near[:,0]<image.shape[1]+1.5/scale)&
                            (near[:,1]>-1.5/scale)&(near[:,1]<image.shape[0]+1.5/scale)]
                for q in near:
                    valid &= np.hypot(xx-q[0], yy-q[1])*scale > 1.5
                rr = np.hypot(xx-center[0], yy-center[1])*scale
                if valid[rr < 2.3].mean() < .97 or valid.mean() < .65: bad = True; break
                arrays.update({b+'__image': image, b+'__variance': variance, b+'__mask': valid,
                               b+'__matrix': matrix, b+'__center': center})
            if bad: continue
            arrays.update(sky=sky, region=region, background=idx)
            np.savez_compressed(target/f'background_{idx:04d}.npz', **arrays)
            chosen.append(p); rows.append(dict(background=idx, region=region, ra=sky[0], dec=sky[1])); idx += 1
            if len(chosen) == per_region: break
        print(f'Background {region}: {len(chosen)} patches', flush=True)
        if len(chosen) < minimum_per_region: raise RuntimeError(f'Too few usable backgrounds: {region}')
    pd.DataFrame(rows).to_csv(out/'backgrounds.csv', index=False)
    return idx


def donor_quality(out, source_root):
    """Data-only QA, independent of injection outcomes; retain every trial."""
    bands = pd.read_csv(out/'donor_bands.csv'); cache = {}; rows = []
    for path in sorted((out/'donors').glob('donor_*.npz')):
        d = read_npz(path); region = int(d['region']); donor = int(d['donor'])
        if region not in cache:
            folder = source_root/f'region_{region:03d}'
            matches = pd.read_csv(folder/'mer_match.csv').set_index('source')
            with np.load(folder/'tile_inputs.npz') as z: wc = wcs(z['euclid_VIS__wcs'])
            cache[region] = matches, wc
        matches, wc = cache[region]; m = matches.loc[int(d['source'])]
        pix = np.array(wc.world_to_pixel_values(*d['sky']))
        nominal = pix-np.round(pix)+(np.array(d['euclid_VIS__latent'].shape[::-1])-1)/2
        shift = float(np.linalg.norm(d['euclid_VIS__matrix']@(d['euclid_VIS__center']-nominal)))
        cross_snr = float(bands[(bands.donor==donor)&(bands.band!='euclid_VIS')].snr.max())
        reasons = []
        if not m['matched'] or not m['primary_match']: reasons.append('no_unique_MER_galaxy_match')
        if not np.isfinite(m.mer_point_like_prob) or m.mer_point_like_prob > .2: reasons.append('uncertain_galaxy_class')
        if cross_snr < 8: reasons.append('no_independent_band_counterpart')
        if shift > .5: reasons.append('large_donor_center_shift')
        rows.append(dict(donor=donor, region=region, source=int(d['source']),
                         suitable_primary=not reasons, reasons=';'.join(reasons),
                         center_shift_arcsec=shift, strongest_non_VIS_snr=cross_snr,
                         point_like_prob=float(m.mer_point_like_prob) if np.isfinite(m.mer_point_like_prob) else None))
    pd.DataFrame(rows).to_csv(out/'donor_quality.csv', index=False)
    return pd.DataFrame(rows)


def read_npz(path):
    with np.load(path) as z: return {k: z[k] for k in z.files}


def oracle_fit(image, variance, valid, profiles):
    a = np.column_stack([p.ravel() for p in profiles]+[np.ones(image.size)])
    v = valid.ravel() & np.isfinite(image.ravel()) & (variance.ravel() > 0)
    aw = a[v]/np.sqrt(variance.ravel()[v, None]); yw = image.ravel()[v]/np.sqrt(variance.ravel()[v])
    cov = np.linalg.inv(aw.T@aw); coeff = cov@(aw.T@yw)
    return coeff[:-1], np.sqrt(np.diag(cov)[:-1])


def export_scenes(out, count=1000, gains=None):
    """Freeze identical native pixels, PSFs, masks and known positions for both fits."""
    dest = out/'tractor_inputs'; dest.mkdir(exist_ok=True)
    donors = sorted((out/'donors').glob('donor_*.npz'))
    backgrounds = sorted((out/'backgrounds').glob('background_*.npz'))
    rng = np.random.default_rng(SEED+2); manifests = []; truth_rows = []
    # Balanced reuse of donors; scene design never uses fitted photometry.
    sequence = rng.permutation(np.tile(np.arange(len(donors)), int(np.ceil(count/len(donors)))))[:count]
    for scene_id in range(count):
        donor_id = int(sequence[scene_id]); d = read_npz(donors[donor_id])
        bg_id = scene_id % len(backgrounds); bg = read_npz(backgrounds[bg_id])
        context = ('isolated', 'equal_blend', 'bright_neighbor', 'different_SED')[scene_id % 4]
        snr = (2., 3., 5., 10., 20.)[(scene_id//4) % 5]
        separation = (.35, .7, 1.4)[(scene_id//20) % 3] if context != 'isolated' else 0.
        ratio = 1. if context in ('isolated', 'equal_blend') else (3. if context == 'different_SED' else 10.)
        angle = rng.uniform(0, 2*np.pi); direction = rng.uniform(0, 2*np.pi)
        broadening = (0., .06)[(scene_id//60) % 2]
        other_id = int(rng.integers(len(donors)))
        ds = [d] if context == 'isolated' else [d, read_npz(donors[other_id])]
        # A common scene angle rotates BOTH intrinsic morphology and PSF.
        offsets = [rng.uniform(-.05, .05, 2)]
        if len(ds) == 2: offsets.append(offsets[0]+separation*np.array([np.cos(direction), np.sin(direction)]))
        offsets = np.array(offsets)
        sky = bg['sky'] + offsets/np.array([3600*np.cos(np.deg2rad(bg['sky'][1])), 3600])
        arrays = dict(sky=sky); profiles = {}; kernels = {}; positions = {}
        for b in BANDS:
            matrix = bg[b+'__matrix']; center = bg[b+'__center']; shape = bg[b+'__image'].shape
            positions[b] = center+np.linalg.solve(matrix, offsets.T).T
            pp = []; kk = []
            # Wide PSF stamp to preserve transferred and broadened wings.
            ksize = 65 if b.startswith('euclid_') else 49
            for j, donor in enumerate(ds):
                dm = donor[b+'__matrix']; psf = donor[b+'__kernel']
                pp.append(render(donor[b+'__latent'], dm, matrix, positions[b][j], shape,
                                 kernel=psf, angle=angle, extra_sigma_arcsec=broadening,
                                 donor_center=donor[b+'__center'], kernel_matrix=donor[b+'__kernel_matrix']))
                kk.append(render(psf, donor[b+'__kernel_matrix'], matrix, [(ksize-1)/2]*2, (ksize, ksize),
                                 angle=angle, extra_sigma_arcsec=broadening))
            profiles[b] = pp; kernels[b] = np.array([unit(k) for k in kk])
        b = 'euclid_VIS'; _, error = oracle_fit(np.zeros_like(bg[b+'__image']), bg[b+'__variance'], bg[b+'__mask'], profiles[b][:1])
        fvis = snr*error[0]; scale = fvis/float(d[b+'__flux'])
        for b in BANDS:
            truth = [scale*float(d[b+'__flux'])]
            if len(ds) == 2:
                other_scale = ratio*fvis/float(ds[1]['euclid_VIS__flux'])
                truth.append(other_scale*float(ds[1][b+'__flux']))
            truth = np.array(truth)
            injected = np.einsum('n,nhw->hw', truth, profiles[b])
            variance = bg[b+'__variance'].astype(float).copy()
            shot = np.zeros_like(injected)
            if gains is not None:
                gain = float(gains[b])
                if not np.isfinite(gain) or gain <= 0: raise ValueError('Effective gains must be finite and positive')
                expectation = np.maximum(injected, 0)
                shot = rng.poisson(expectation*gain)/gain-expectation
                variance += expectation/gain
            image = bg[b+'__image'].astype(float)+injected+shot
            matrix = bg[b+'__matrix']; sigma = np.sqrt(np.sum(kernels[b][0]*(np.indices(kernels[b][0].shape)[0]-(kernels[b][0].shape[0]-1)/2)**2))
            arrays.update({b+'__image': image.astype('float32'), b+'__variance': variance.astype('float32'),
                           b+'__mask': bg[b+'__mask'], b+'__positions': positions[b],
                           b+'__sky_to_pixel': np.linalg.inv(matrix), b+'__psf_sigma': float(sigma),
                           b+'__psf_kernels': kernels[b], b+'__truth': truth})
            oflux, oerr = oracle_fit(image, variance, bg[b+'__mask'], profiles[b])
            for j in range(len(ds)):
                fraction = float(profiles[b][j].sum()); usable = float(profiles[b][j][bg[b+'__mask']].sum())
                truth_rows.append(dict(scene=scene_id, source=j, band=b, truth_flux=truth[j], oracle_error=oerr[j],
                                       oracle_flux=oflux[j], true_snr=truth[j]/oerr[j], footprint=fraction,
                                       valid_footprint=usable, donor=int(ds[j]['donor']),
                                       weak_band=bool(ds[j][b+'__fallback'])))
        # Save diagnostic truth profiles separately, never in photometer inputs.
        if scene_id < 20:
            diagnostics = out/'diagnostics'; diagnostics.mkdir(exist_ok=True)
            np.savez_compressed(diagnostics/f'profiles_{scene_id:04d}.npz', **{b:np.array(v) for b,v in profiles.items()})
        np.savez_compressed(dest/f'scene_{scene_id:04d}.npz', **arrays)
        manifests.append(dict(scene=scene_id, donor=donor_id, neighbor_donor=other_id if len(ds)==2 else -1,
                              background=bg_id, background_region=int(bg['region']), context=context,
                              target_snr=snr, separation=separation, flux_ratio=ratio,
                              rotation_rad=angle, broadening_arcsec=broadening,
                              pixel_sha256=hashlib.sha256(np.ascontiguousarray(arrays['euclid_VIS__image']).tobytes()).hexdigest()))
        if (scene_id+1) % 50 == 0: print(f'Exported {scene_id+1}/{count}', flush=True)
    pd.DataFrame(manifests).to_csv(out/'scenes.csv', index=False)
    pd.DataFrame(truth_rows).to_csv(out/'truth.csv', index=False)


def load_scene(path):
    """Truth-blind adapter: retain only data, noise, PSF and known coordinates."""
    import torch
    z = read_npz(path); bands = {}
    for b in BANDS:
        bands[b] = {k: torch.tensor(z[b+'__'+k]) for k in ('image', 'variance', 'mask', 'positions', 'sky_to_pixel')}
        for k in ('positions', 'sky_to_pixel'): bands[b][k] = bands[b][k].float()
        bands[b].update(psf_sigma=float(z[b+'__psf_sigma']), psf_kernels=z[b+'__psf_kernels'])
    return dict(sky=z['sky'], bands=bands, central=0, tile='empirical_injection')


_JAISP_MODELS = None


def _fit_jaisp_path(path):
    global _JAISP_MODELS
    import torch
    from .predict import MixturePhotometry
    torch.set_num_threads(2)
    if _JAISP_MODELS is None:
        checkpoint = HERE/'runs/q1_mixture_calibrated/priors.pt'
        _JAISP_MODELS = {mode: MixturePhotometry(checkpoint, mode=mode) for mode in ('foundation', 'image')}
    scene_id = int(path.stem.split('_')[-1]); start = time.monotonic(); rows = []; errors = []
    scene = load_scene(path)
    for name, model in _JAISP_MODELS.items():
        try:
            result = model(scene)
            for b, band_result in result.items():
                for j, (flux, err) in enumerate(zip(band_result['flux'], band_result['error'])):
                    if not np.isfinite([flux, err]).all(): raise ValueError('Nonfinite measurement')
                    rows.append(dict(scene=scene_id, source=j, band=b, model=name,
                                     flux=float(flux), error=float(err)))
        except Exception: errors.append(dict(model=name, error=traceback.format_exc()))
    return dict(scene=scene_id, rows=rows, failures=errors, seconds=time.monotonic()-start)


def run_jaisp(out, count=None, workers=4):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    results = out/'jaisp_results'; results.mkdir(exist_ok=True)
    paths = sorted((out/'tractor_inputs').glob('scene_*.npz'))
    if count is not None: paths = paths[:count]
    pending = []; digests = {}
    for path in paths:
        dest = results/(path.stem+'.json')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if dest.exists():
            if json.loads(dest.read_text()).get('input_sha256') != digest:
                raise RuntimeError(f'Input changed since fit: {path}; use a fresh run directory')
            continue
        pending.append(path); digests[path] = digest
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        futures = {pool.submit(_fit_jaisp_path, p):p for p in pending}
        for i, future in enumerate(as_completed(futures)):
            path = futures[future]; result = future.result(); result['input_sha256'] = digests[path]
            write_json(results/(path.stem+'.json'), result)
            if i % 20 == 0: print(f"JAISP {result['scene']}: {len(result['failures'])} failures, {result['seconds']:.1f}s ({i+1}/{len(pending)})", flush=True)
    all_results = [json.loads(p.read_text()) for p in sorted(results.glob('*.json'))]
    pd.DataFrame([r for p in all_results for r in p['rows']]).to_csv(out/'jaisp_fluxes.csv', index=False)
    write_json(out/'jaisp_failures.json', [p for p in all_results if p['failures']])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('library', 'backgrounds', 'export', 'jaisp', 'all'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--source-root', type=Path, default=HERE/'runs/detection_catalog')
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--donor-snr', type=float, default=25.)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gain-json', type=Path)
    args = parser.parse_args(); out = args.out; out.mkdir(parents=True, exist_ok=True)
    gains = json.loads(args.gain_json.read_text()) if args.gain_json else None
    if args.stage in ('library', 'all'): make_library(out, args.source_root, args.donor_snr)
    if args.stage in ('backgrounds', 'all'): make_backgrounds(out, args.source_root)
    if args.stage in ('export', 'all'):
        donor_quality(out, args.source_root)
        protocol = dict(version='empirical_pilot_v2', seed=SEED, n_scenes=args.count,
                        background_regions=list(BACKGROUND_REGIONS), donor_snr_min=args.donor_snr,
                        units='Each native band; measured donor colors dimmed by a common scalar',
                        donor_model='Independent positive FISTA pixels, Gaussian likelihood, fixed smoothness .04 / sparsity .15 in sky-RMS units, support 2.3 arcsec',
                        weak_bands='Data-only common VIS-shape amplitude when SNR<8; truncated-positive posterior mean, individually flagged',
                        psf='Archive local Euclid pixel PSF; calibrated approximate Gaussian Rubin PSF; rotation of source+PSF, optional extra Gaussian blur; GalSim no_pixel',
                        noise='Real masked coadd sky, including its existing correlations; no stochastic donor-noise copies',
                        blank_sky='All-band independent SEP detections at 3 local empirical RMS, minimum area 10 and integrated SNR>10; grow isophotes 1.5 arcsec and exclude centers within 4 arcsec; retain faint undetected sky',
                        source_noise=('Post-coadd effective-gain Poisson approximation' if gains else 'Not added: faint, sky-dominated pilot; no calibrated effective gain in cached mosaic headers'),
                        effective_gains=gains, positions='Known fixed positions; forced photometry only',
                        splits='Disjoint donor/background regions; prior train/validation excluded by detection-catalog field selection; foundation pretraining independence not established',
                        MER='No injection rerun available; MER only supplies positions and star exclusions, never flux truth',
                        truth='Finite reconstructed morphology and assigned flux, not exact intrinsic truth of the original observed donor',
                        normalization='Unit intrinsic and PSF mass before rendering; no post-crop profile normalization',
                        galsim_version=galsim.__version__)
        dest = out/'protocol.json'
        if dest.exists() and json.loads(dest.read_text()) != protocol:
            raise RuntimeError('Protocol changed: use a separate output directory')
        write_json(dest, protocol); export_scenes(out, args.count, gains)
    if args.stage in ('jaisp', 'all'): run_jaisp(out, args.count, args.workers)


if __name__ == '__main__': main()
