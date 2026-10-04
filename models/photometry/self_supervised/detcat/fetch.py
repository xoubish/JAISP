"""Fetch full-tile Euclid Q1 science/RMS/flag cutouts and GRID PSFs for each selected tile."""
import argparse
import copy
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .common import ROOT, OUT, EUCLID, CUTOUT_SIZES_ARCSEC, region_dir, read_json, write_json
sys.path.insert(0, str(ROOT / 'models/photometry/self_supervised/vendor/euclid_forced_photometry/src'))
from euclid_phot.cutouts import discover_mer_mosaics, fetch_cutout, _matching_tile  # noqa: E402
from euclid_phot.psf import extract_grid_psf  # noqa: E402


def products_for(tile_id, ra, dec):
    path = OUT / f'products_{tile_id}.json'
    if path.exists(): return read_json(path)
    products = discover_mer_mosaics(ra, dec, .02)
    write_json(path, products)
    return products


def fetch_region(t):
    folder = region_dir(t['region']); folder.mkdir(parents=True, exist_ok=True)
    try:
        with np.load(t['euclid'], allow_pickle=True) as e: tile_id = str(e['euclid_tile_id']).replace('TILE', '')
        products = products_for(tile_id, t['ra'], t['dec'])
        info = dict(region=t['region'], tile=t['tile'], euclid_tile=tile_id, ra=t['ra'], dec=t['dec'], bands={})
        for band in EUCLID:
            chosen = copy.deepcopy(products); chosen[band]['science'] = _matching_tile(products[band]['science'], tile_id)
            cutout = None; errors = []
            for size in CUTOUT_SIZES_ARCSEC:
                try:
                    cutout = fetch_cutout(band, t['ra'], t['dec'], size, products=chosen, data_dir=folder / 'cutouts', with_flag=True); break
                except Exception as exc: errors.append(f'{size}: {exc}')
            if cutout is None: raise ValueError(f'{band}: no cutout; ' + '; '.join(errors))
            if cutout.header['MERTILE'] != tile_id: raise ValueError(f'{band}: cutout tile {cutout.header["MERTILE"]} != {tile_id}')
            if 'MAGZERO' not in cutout.header: raise ValueError('Missing image zero point')
            psf = extract_grid_psf(band, t['ra'], t['dec'], products=products, tile_id=tile_id,
                                   radius_arcsec=size / np.sqrt(2) + 15, data_dir=folder / 'psfs')
            if not len(psf['stamps']): raise ValueError(f'No PSF coverage: {band}')
            np.savez_compressed(folder / f'{band}.npz', image=cutout.data.astype('float32'), variance=(cutout.rms ** 2).astype('float32'),
                                flag=cutout.flag.astype('int32'), wcs=cutout.wcs.to_header().tostring(), magzero=float(cutout.header['MAGZERO']),
                                psf_stamps=psf['stamps'], psf_ra=psf['ra'], psf_dec=psf['dec'], psf_fwhm=psf['fwhm'])
            info['bands'][band] = dict(size_arcsec=size, shape=list(cutout.shape), magzero=float(cutout.header['MAGZERO']),
                                       psf_samples=int(len(psf['stamps'])), psf_path=str(psf['s3_path']), psf_median_fwhm=float(np.median(psf['fwhm'])))
        write_json(folder / 'fetch.json', info); print('fetched region', t['region'], flush=True)
        return info
    except Exception:
        return dict(region=t['region'], error=traceback.format_exc())


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--regions', type=int, nargs='*'); p.add_argument('--workers', type=int, default=4)
    args = p.parse_args(); tiles = read_json(OUT / 'tiles.json')['tiles']
    if args.regions is not None: tiles = [t for t in tiles if t['region'] in set(args.regions)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool: infos = list(pool.map(fetch_region, tiles))
    failed = [x for x in infos if 'error' in x]
    write_json(OUT / 'fetch_status.json', infos)
    for f in failed: print(f['region'], f['error'][-800:])
    print(f'{len(infos) - len(failed)} fetched, {len(failed)} failed')


if __name__ == '__main__': main()
