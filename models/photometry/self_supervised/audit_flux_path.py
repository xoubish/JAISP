"""Audit units, compression, stellar profiles and centering without retraining.

Noiseless diagnostic scenes deliberately retain variance maps: they isolate
deterministic morphology/normalization errors from random noise. These are
engineering checks, not the proposed realistic faint-source benchmark.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import erf

from .core import BANDS, fit_flux, templates
from .data import wcs
from .mixture import dictionary, positive_profile_fit, signed_measurement, fit_multiband
from .pixel_psf import convolved_profile, normalize_kernel
from .run_mixture import predict_prior
from .scene_features import SceneEncoder, image_features

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'models/photometry/self_supervised/runs/flux_path_audit'


def gaussian_pixels(shape, position, sigma):
    """Independent exact integral of a circular Gaussian over native pixels."""
    axes = [.5 * (erf((np.arange(n) - p + .5) / (np.sqrt(2) * sigma))
                 - erf((np.arange(n) - p - .5) / (np.sqrt(2) * sigma)))
            for n, p in zip(shape[::-1], position)]
    return np.outer(axes[1], axes[0])


def pixel_identity(payload):
    """Compare prepared science arrays with their source NPZs, using WCS origins."""
    sys.path.insert(0, str(ROOT / 'models'))
    from foundation_utils import discover_tile_pairs
    pairs = {name: (rp, ep) for name, rp, ep in discover_tile_pairs(
        str(ROOT / 'data/rubin_tiles_all'), str(ROOT / 'data/euclid_tiles_all_q1'))}
    rows = []
    for split, scenes in payload['splits'].items():
        scene = scenes[0]
        rp, ep = pairs[scene['tile']]
        with np.load(rp, allow_pickle=True) as r, np.load(ep, allow_pickle=True) as e:
            for j, band in enumerate(BANDS):
                d = scene['bands'][band]
                if band.startswith('rubin'):
                    raw, wc = r['img'][j], wcs(r['wcs_hdr'])
                else:
                    short = band.split('_')[1]
                    raw, wc = e['img_' + short], wcs(e['wcs_' + short])
                full = np.column_stack(wc.world_to_pixel_values(*scene['sky'].T))
                origins = full - d['positions'].numpy()
                origin = np.rint(origins[scene['central']]).astype(int)
                h, width = d['image'].shape
                expected = raw[origin[1]:origin[1]+h, origin[0]:origin[0]+width].astype('float32')
                same = np.array_equal(expected, d['image'].numpy(), equal_nan=True)
                residual = np.max(np.abs(origins - origin))
                rows.append(dict(split=split, band=band, pixels_identical=same,
                                 max_position_residual_px=float(residual)))
                if not same or residual > 2e-4:
                    raise AssertionError(f'Science pixels or WCS positions changed: {split}/{band}')
    return rows


def encoder_geometry(payload):
    """Quantify the encoder's size-only resize assumption against actual WCS positions.

    This is a feature-alignment diagnostic, not an offset applied to the native
    flux templates. The foundation itself does not consume a WCS.
    """
    rows = []
    for split, scenes in payload['splits'].items():
        for index, scene in enumerate(scenes):
            v = scene['bands']['euclid_VIS']; vh, vw = v['image'].shape
            central = scene['central']; vp = v['positions'].numpy()[central]
            for band, d in scene['bands'].items():
                h, width = d['image'].shape
                assumed = (vp+.5)*[width/vw, h/vh]-.5
                actual = d['positions'].numpy()[central]
                offset = np.linalg.solve(d['sky_to_pixel'].numpy(), actual-assumed)
                rows.append(dict(split=split, scene=index, band=band,
                    encoder_resize_offset_arcsec=float(np.linalg.norm(offset)),
                    east_offset_arcsec=float(offset[0]), north_offset_arcsec=float(offset[1])))
    return rows


def make_star_scene(template, peak_snr, phase, kernels=None, blended=False):
    bands, truth, oracle = {}, {}, {}
    offsets = np.array([[0., 0.], [.4, .13]]) if blended else np.zeros((1, 2))
    for band in BANDS:
        original = template['bands'][band]
        shape = tuple(original['image'].shape)
        center = np.array([(shape[1]-1)/2, (shape[0]-1)/2]) + [phase, -phase]
        pos = center + offsets @ original['sky_to_pixel'].numpy().T
        variance = original['variance'].numpy()
        rms = np.sqrt(np.median(variance[original['mask'].numpy()]))
        profiles = []
        for xy in pos:
            if kernels is not None and band in kernels:
                profile = convolved_profile(shape, xy, np.zeros((2, 2)), kernels[band])
            else:
                profile = gaussian_pixels(shape, xy, original['psf_sigma'])
            profiles.append(profile.ravel())
        bank = np.column_stack(profiles)
        flux = peak_snr * rms / bank.max(0)
        if blended:
            flux[1] *= .15
        image = (bank @ flux).reshape(shape) + .3 * rms
        d = dict(image=torch.tensor(image, dtype=torch.float32),
                 variance=torch.full(shape, rms**2), mask=torch.ones(shape, dtype=torch.bool),
                 positions=torch.tensor(pos, dtype=torch.float32),
                 sky_to_pixel=original['sky_to_pixel'].clone(), psf_sigma=original['psf_sigma'])
        if kernels is not None and band in kernels:
            d['psf_kernels'] = np.repeat(kernels[band][None], len(pos), axis=0)
        bands[band], truth[band], oracle[band] = d, flux, bank
    return dict(tile='flux_audit', central=0, sky=offsets, bands=bands), truth, oracle


def assess(scene, checkpoint, encoder):
    item = dict(scene=scene, image=image_features(scene), foundation=encoder(scene))
    banks, fits = {}, {}
    for mode in ('image', 'foundation'):
        head = checkpoint['heads'][mode]
        prior = predict_prior(item, head, checkpoint['population'], mode)
        fits[mode] = fit_multiband(scene, prior, banks=banks, prior_precision=head['precision'],
            strength=checkpoint['metadata']['prior_strength'],
            band_strength=checkpoint['metadata']['band_strength'])
    for band in BANDS:
        # A data-only morphology fit with the same seven-profile bank isolates
        # the point-source size floor from the learned prior.
        weights, _ = positive_profile_fit(scene['bands'][band], banks[band])
        fits.setdefault('unregularized_mixture', {})[band] = signed_measurement(
            scene['bands'][band], banks[band], weights)
    return fits, item


def scarlet_renderer_diagnostic():
    """Probe the separate experimental renderer against independent Gaussian pixels."""
    from .amortised_scarlet import render_templates, render_unconvolved, band_kernels, fft_convolve, MORPH_SIZE
    rows = []
    g = torch.arange(MORPH_SIZE) - (MORPH_SIZE-1)/2
    yy, xx = torch.meshgrid(g, g, indexing='ij')
    for size in (.10, .30, .60):
        morph = torch.exp(-(xx*xx+yy*yy)/(2*(size/.1)**2))
        morph = (morph/morph.sum())[None, None]
        for scale in (.1, .2):
            d = dict(image=torch.zeros(81, 81), positions=torch.tensor([[40., 40.]]),
                     sky_to_pixel=torch.eye(2)/scale, psf_sigma=.932 if scale == .1 else 2.18)
            scene = dict(bands={'audit': d})
            rendered = render_templates(scene, {'audit': morph}, torch.device('cpu'))['audit']
            exact = gaussian_pixels((81, 81), (40, 40), np.hypot(size/scale, d['psf_sigma']))
            fit = fit_flux(torch.tensor(exact*10000), torch.ones(81, 81), rendered)
            ours = rendered.numpy().reshape(81, 81)
            y, x = np.indices(ours.shape)
            row = dict(intrinsic_sigma_arcsec=size, pixel_scale_arcsec=scale,
                mass=float(ours.sum()), flux_error_percent=float(100*(fit['flux'][0]/10000-1)),
                extra_variance_px2=float((ours*(x-40)**2).sum()/ours.sum()
                    - (size/scale)**2 - d['psf_sigma']**2 - 1/12))
            if scale == .1:
                # Same-grid integer-center control only: omit the intrinsic
                # pixel integration before the delivered pixel-integrated PSF.
                # Oversample=1 is NOT a general fix on rotated/coarser grids.
                canvas = render_unconvolved(morph, d['positions'], d['sky_to_pixel'], (81,81), 40, 1)
                sampled = fft_convolve(canvas, band_kernels(d,1,torch.device('cpu')))[0,0,40:121,40:121]
                once = fit_flux(torch.tensor(exact*10000), torch.ones(81,81), sampled.reshape(-1,1))
                row['same_grid_point_sampled_flux_error_percent'] = float(100*(once['flux'][0]/10000-1))
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, default=RUN)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    checkpoint = torch.load(ROOT/'models/photometry/self_supervised/runs/q1_mixture_calibrated/priors.pt',
                            map_location='cpu', weights_only=False)
    payload = torch.load(Path(checkpoint['metadata']['source'])/'scenes.pt',
                         map_location='cpu', weights_only=False)
    encoder = SceneEncoder(ROOT/checkpoint['metadata']['original']['foundation_checkpoint'])
    sys.path.insert(0, str(ROOT/'models'))
    from jaisp_foundation_v10 import compress_snr, decompress_snr
    snr = torch.tensor([-1e5, -500., -50., 0., 1e-4, 1., 50., 100., 1e3, 1e5])
    reconstructed = decompress_snr(compress_snr(snr, 'asinh50'), 'asinh50')
    roundtrip = float(((reconstructed-snr).abs()/snr.abs().clamp_min(1)).max())
    report = dict(compression_modes={b:s.compression for b,s in encoder.encoder.stems.items()},
                  asinh_roundtrip_max_relative_error=roundtrip, science_pixels=pixel_identity(payload))
    geometry = pd.DataFrame(encoder_geometry(payload))
    geometry.to_csv(a.output/'encoder_alignment.csv', index=False)
    report['encoder_alignment'] = geometry.groupby('band').encoder_resize_offset_arcsec.agg(
        ['median','max']).to_dict('index')
    stamps = {}
    for short in ('VIS', 'Y', 'J', 'H'):
        paths = sorted((ROOT/'models/photometry/self_supervised/runs/psf_study/psfs').glob(
            f'psf_grid_stamps_{short.lower()}_*.npz'))
        if paths:
            with np.load(paths[0]) as z:
                stamps['euclid_'+short] = normalize_kernel(z['stamps'][0])
    report['archive_psf_bands'] = list(stamps)
    rows = []; case = 0; template = payload['splits']['test'][0]
    repeat_checks = []
    for psf in ('gaussian', 'archive'):
        if psf == 'archive' and len(stamps) != 4:
            continue
        for blended, peak, phase in [(False, level, phase) for level in (5., 100., 5000.)
                                     for phase in (0., .37)] + [(True, 500., .37)]:
            scene, truth, oracle = make_star_scene(template, peak, phase,
                stamps if psf == 'archive' else None, blended)
            originals = {b:d['image'].clone() for b,d in scene['bands'].items()}
            fits, item = assess(scene, checkpoint, encoder)
            # This restricted repair is tested only with supplied stellar
            # classifications. It is not an automatic star/galaxy classifier.
            stellar_scene = dict(scene, point_sources=np.ones(len(scene['sky']), dtype=bool))
            head = checkpoint['heads']['foundation']
            prior = predict_prior(item, head, checkpoint['population'], 'foundation')
            fits['foundation_known_stars'] = fit_multiband(stellar_scene, prior,
                prior_precision=head['precision'], strength=checkpoint['metadata']['prior_strength'],
                band_strength=checkpoint['metadata']['band_strength'])
            for band in BANDS:
                d = scene['bands'][band]
                r = fit_flux(d['image'], d['variance'], torch.tensor(oracle[band]), d['mask'])
                fits.setdefault('matched_psf_only', {})[band] = dict(flux=r['flux'].numpy())
                assert torch.equal(d['image'], originals[band]), 'Photometer mutated science pixels'
                for mode, measured in fits.items():
                    for source, actual in enumerate(truth[band]):
                        rows.append(dict(case=case, psf=psf, blended=blended, peak_snr=peak,
                            phase=phase, band=band, source=source, mode=mode, truth_flux=actual,
                            flux=measured[band]['flux'][source],
                            flux_error_percent=100*(measured[band]['flux'][source]/actual-1)))
            if case in (0, 7):
                # A->changed scene->A catches stale state/caches; unit changes
                # must change image and RMS together, preserving the S/N input.
                changed = copy.deepcopy(scene)
                for d in changed['bands'].values():
                    d['image'] *= 7; d['variance'] *= 49
                scaled, _ = assess(changed, checkpoint, encoder)
                repeated, after = assess(scene, checkpoint, encoder)
                repeat_checks.append(dict(psf=psf,
                    max_feature_difference=float(np.abs(item['foundation']-after['foundation']).max()),
                    max_flux_difference=float(max(np.abs(fits['foundation'][b]['flux']-repeated['foundation'][b]['flux']).max() for b in BANDS)),
                    max_unit_scaling_relative_error=float(max(np.abs(scaled['foundation'][b]['flux']/fits['foundation'][b]['flux']/7-1).max() for b in BANDS))))
            case += 1
            pd.DataFrame(rows).to_csv(a.output/'stellar_checks.csv', index=False)
            print('Stellar audit case', case, psf, peak, phase, 'blend', blended, flush=True)
    report.update(repeat_and_units=repeat_checks, scarlet_renderer=scarlet_renderer_diagnostic(),
        protocol='Noiseless point sources; realistic supplied variance; known positions; Gaussian exact-pixel truth or normalized archive finite stamps; no fitting/retraining to truth',
        scope='Engineering audit, not a realistic galaxy/SED faint-end benchmark. Archive PSF truth shares the convolution routine with the pixelized fitter; Gaussian truth is independent.')
    (a.output/'audit.json').write_text(json.dumps(report, indent=2))
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['psf','blended','peak_snr','band','mode']).flux_error_percent.agg(['median','min','max']).reset_index()
    summary.to_csv(a.output/'stellar_summary.csv', index=False)
    print(summary[summary.band.eq('euclid_VIS')].to_string(index=False), flush=True)
    print(json.dumps({k:v for k,v in report.items() if k!='science_pixels'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
