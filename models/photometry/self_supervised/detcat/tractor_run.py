"""Upstream SEP/Tractor VIS model selection and forced photometry on the detection-head source list.

Run in the isolated Tractor environment (Python 3.11, no torch):
    runs/tractor_env/bin/python -m models.photometry.self_supervised.detcat.tractor_run

This follows the linked upstream notebook: SEP blobs on VIS, the chi-squared
profile ladder with GRID PSFs, then flux-only forced photometry in every band
with the VIS profiles held fixed. Positions are frozen at the astrometry-head
values so both photometers measure the same source list at the same centroids.
"""
import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[4]
VENDOR = ROOT / 'models/photometry/self_supervised/vendor/euclid_forced_photometry'
sys.path.insert(0, str(VENDOR / 'src'))
from tractor import Image, Tractor, ConstantSky, LinearPhotoCal, PointSource, RaDecPos, NanoMaggies  # noqa: E402
from tractor.wcs import ConstantFitsWcs  # noqa: E402
from euclid_phot.images import AstropyWCSAdapter  # noqa: E402
from euclid_phot.selection import ModelSelector, run_model_selection  # noqa: E402
from euclid_phot.nisp import _clone_for_band  # noqa: E402
from euclid_phot.spatial_psf import SpatialPixelizedPSF  # noqa: E402

OUT = ROOT / 'models/photometry/self_supervised/runs/detection_catalog'
EUCLID = ('VIS', 'Y', 'J', 'H')
UJY_PER_NMGY = 3.631


class FrozenPositionSelector(ModelSelector):
    """Upstream profile ladder with every position frozen at the astrometry-head value."""
    def _optimize(self, tractor):
        for source in tractor.getCatalog(): source.freezeParam('pos')
        super()._optimize(tractor)


def load_inputs(folder):
    with np.load(folder / 'tile_inputs.npz', allow_pickle=True) as z: out = {k: z[k] for k in z.files}
    return {k: (v.item() if v.ndim == 0 else v) for k, v in out.items()}


def tractor_image(inputs, short):
    b = 'euclid_' + short
    image = inputs[b + '__image'].astype(float); var = inputs[b + '__variance'].astype(float); mask = inputs[b + '__mask'].astype(bool)
    wcs = WCS(fits.Header.fromstring(str(inputs[b + '__wcs']))); twcs = ConstantFitsWcs(AstropyWCSAdapter(wcs))
    scale = 10 ** (.4 * (float(inputs[b + '__magzero']) - 22.5))  # counts per nanomaggie
    psf = SpatialPixelizedPSF(dict(stamps=inputs[b + '__psf_stamps'], ra=inputs[b + '__psf_sky'][:, 0], dec=inputs[b + '__psf_sky'][:, 1]), twcs)
    tim = Image(data=np.where(mask, image, 0.), invvar=np.where(mask, 1 / np.where(mask, var, 1.), 0.), psf=psf, wcs=twcs,
                photocal=LinearPhotoCal(scale, band=short), sky=ConstantSky(0.), name=b)
    tim.freezeAllParams()
    return tim, scale


def seed_fluxes(inputs):
    """Data-only VIS seeds in microJansky: masked 0.5-arcsec aperture sums (seeds only, never truth)."""
    b = 'euclid_VIS'; image = inputs[b + '__image']; mask = inputs[b + '__mask']; pos = inputs[b + '__positions']
    conversion = 10 ** (.4 * (23.9 - float(inputs[b + '__magzero']))); seeds = []
    for x, y in pos:
        xi, yi = int(round(x)), int(round(y)); xs = slice(max(xi - 6, 0), xi + 7); ys = slice(max(yi - 6, 0), yi + 7)
        yy, xx = np.mgrid[ys, xs]; inside = (np.hypot(xx - x, yy - y) <= 5) & mask[ys, xs]
        seeds.append(max(float(np.sum(image[ys, xs][inside])) * conversion, .01))
    return np.array(seeds)


def shape_parameters(source):
    shape = getattr(source, 'shape', None) or getattr(source, 'shapeExp', None)
    out = dict(re_arcsec=np.nan, axis_ratio=np.nan, sersic_n=np.nan)
    if shape is not None:
        try: out['re_arcsec'] = float(shape.re); out['axis_ratio'] = float(shape.ab)
        except Exception: pass
    index = getattr(source, 'sersicindex', None)
    if index is not None:
        try: out['sersic_n'] = float(index.getValue())
        except Exception: pass
    return out


def worker(args):
    folder, blob_workers = args; start = time.monotonic(); region = int(folder.name.split('_')[1])
    try:
        inputs = load_inputs(folder); sky = inputs['sky']; seeds = seed_fluxes(inputs)
        cat = Table(dict(ra=sky[:, 0], dec=sky[:, 1], flux_vis_sersic=seeds))
        tim, scale = tractor_image(inputs, 'VIS')
        selected, counts = run_model_selection(cat, tim, tim.getImage(), tim.getInvvar(), pixscale_arcsec=.1,
                                               selector=FrozenPositionSelector(), n_workers=blob_workers)
        fallback = np.array([s is None for s in selected])
        sources = [s if s is not None else PointSource(RaDecPos(float(ra), float(dec)), NanoMaggies(VIS=float(seed) / UJY_PER_NMGY))
                   for s, (ra, dec), seed in zip(selected, sky, seeds)]
        for s in sources: s.freezeParam('pos')
        moved = np.array([np.hypot((s.getPosition().ra - ra) * np.cos(np.deg2rad(dec)), s.getPosition().dec - dec) * 3600 for s, (ra, dec) in zip(sources, sky)])
        if moved.max() > 1e-3: raise RuntimeError(f'Positions moved by up to {moved.max():.4f} arcsec despite freezing')
        shapes = [shape_parameters(s) for s in sources]
        rows = []; models = {}; selection_seconds = time.monotonic() - start
        for short in EUCLID:
            tim, scale = tractor_image(inputs, short); b = 'euclid_' + short
            clones = [_clone_for_band(s, short) for s in sources]
            for s in clones: s.freezeAllBut('brightness')
            # A source whose rendered patch has no valid pixel has no derivative; Tractor's LSQR
            # solver then mis-indexes its parameter map. Such sources get NaN in this band.
            invvar = tim.getInvvar(); fitted = []
            for s in clones:
                patch = s.getModelPatch(tim)
                if patch is None or patch.patch is None: fitted.append(False); continue
                sl = patch.getSlice(tim)
                fitted.append(bool(np.any(invvar[sl] > 0)) if patch.patch.size else False)
            fitted = np.array(fitted); active = [s for s, ok in zip(clones, fitted) if ok]
            flux = np.full(len(clones), np.nan); error = np.full(len(clones), np.nan)
            if active:
                tr = Tractor([tim], active)
                fit = tr.optimize_forced_photometry(minsb=0., mindlnp=1., sky=False, variance=True)
                flux[fitted] = np.array([s.brightness.getFlux(short) for s in active]) * scale
                iv = np.asarray(fit.IV, float); error[fitted] = np.where(iv > 0, scale / np.sqrt(np.maximum(iv, 1e-300)), np.nan)
            else: tr = Tractor([tim], clones)
            conversion = 10 ** (.4 * (23.9 - float(inputs[b + '__magzero'])))
            models[b] = tr.getModelImage(0).astype('float32')
            for i, s in enumerate(sources):
                rows.append(dict(region=region, source=int(inputs['source'][i]), band=b, model='tractor', flux_native=flux[i],
                                 flux_ujy=flux[i] * conversion, error_ujy=error[i] * conversion, profile=type(s).__name__,
                                 blobbed=not fallback[i], fitted=bool(fitted[i]), **shapes[i]))
        pd.DataFrame(rows).to_csv(folder / 'tractor_fluxes.csv', index=False)
        np.savez_compressed(folder / 'tractor_models.npz', **models)
        (folder / 'selection.json').write_text(json.dumps(dict(counts=counts, selection_seconds=selection_seconds,
                                                              total_seconds=time.monotonic() - start, blob_workers=blob_workers), indent=2))
        print(f'Tractor region {region}: {len(sources)} sources, {counts}, {time.monotonic() - start:.0f}s', flush=True)
        return dict(region=region, sources=len(sources), seconds=time.monotonic() - start, counts=counts)
    except Exception:
        return dict(region=region, error=traceback.format_exc(), seconds=time.monotonic() - start)


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--regions', type=int, nargs='*')
    p.add_argument('--tile-workers', type=int, default=14); p.add_argument('--blob-workers', type=int, default=1)
    a = p.parse_args()
    folders = sorted(f.parent for f in OUT.glob('region_*/tile_inputs.npz'))
    if a.regions is not None: folders = [f for f in folders if int(f.name.split('_')[1]) in set(a.regions)]
    jobs = [(f, a.blob_workers) for f in folders]
    if a.tile_workers > 1:
        with ProcessPoolExecutor(max_workers=a.tile_workers) as pool: results = list(pool.map(worker, jobs))
    else: results = [worker(j) for j in jobs]
    status_path = OUT / 'tractor_status.json'
    previous = {}
    if a.regions is not None and status_path.exists():
        previous = {int(x['region']): x for x in json.loads(status_path.read_text())['regions']}
    previous.update({int(x['region']): x for x in results})
    results = [previous[k] for k in sorted(previous)]
    status_path.write_text(json.dumps(dict(protocol=dict(upstream=json.loads((VENDOR / 'UPSTREAM.json').read_text()),
        models='Upstream full VIS profile ladder (point, simple, exp, dev, composite, guarded Sersic tier); SEP blobs and segments',
        positions='Detection-head sources at anchored-astrometry positions, frozen at every optimization stage',
        psf='Per-source nearest GRID-PSF stamp in every Euclid band', sky='Constant zero sky (background-subtracted MER mosaics), as upstream',
        flux_solver='Tractor optimize_forced_photometry, brightness only, formal errors from inverse variance',
        units='Native counts converted with image MAGZERO to microJansky'), regions=results), indent=2))
    failed = [r for r in results if 'error' in r]
    for r in failed: print(r['region'], r['error'][-600:])
    if failed: raise RuntimeError(f'{len(failed)} regions failed; see tractor_status.json')


if __name__ == '__main__': main()
