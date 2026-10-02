"""Retrieve actual Q1 pixels, masks, GRID-PSFs and MER labels for heldout fields."""
from pathlib import Path
import sys,json,copy
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from astropy.table import Table
from astroquery.ipac.irsa import Irsa
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor/euclid_forced_photometry/src'))
from euclid_phot.catalog import _ADQL
from euclid_phot.cutouts import discover_mer_mosaics,fetch_cutout,_matching_tile
from euclid_phot.psf import extract_grid_psf

OUT=Path('models/photometry/self_supervised/runs/real_mer')

def fetch_region(item):
    i,scene,products=item;ra,dec=map(float,scene['sky'][scene['central']])
    folder=OUT/f'region_{i:03d}';folder.mkdir(exist_ok=True)
    tilefile=next(Path('data/euclid_tiles_all_q1').rglob(scene['tile']+'*'))
    with np.load(tilefile) as d:tile=str(d['euclid_tile_id']).replace('TILE','')
    chosen=copy.deepcopy(products)
    info=dict(region=i,tile=scene['tile'],ra=ra,dec=dec,size_arcsec=20.,bands={})
    for band in ('VIS','Y','J','H'):
        chosen[band]['science']=_matching_tile(products[band]['science'],tile)
        cutout=fetch_cutout(band,ra,dec,20.,products=chosen,data_dir=folder/'cutouts',with_flag=True)
        psf=extract_grid_psf(band,ra,dec,products=products,tile_id=cutout.header['MERTILE'],radius_arcsec=22,data_dir=folder/'psfs')
        if not len(psf['stamps']):raise ValueError(f'No PSF coverage: {i} {band}')
        if 'MAGZERO' not in cutout.header:raise ValueError('Missing image zero point')
        np.savez_compressed(folder/f'{band}.npz',image=cutout.data,variance=cutout.rms**2,flag=cutout.flag,
            wcs=cutout.wcs.to_header().tostring(),magzero=float(cutout.header['MAGZERO']),
            psf_stamps=psf['stamps'],psf_ra=psf['ra'],psf_dec=psf['dec'],psf_fwhm=psf['fwhm'])
        info['bands'][band]=dict(tile=cutout.header['MERTILE'],magzero=float(cutout.header['MAGZERO']),psf_path=str(psf['s3_path']),shape=list(cutout.shape))
    (folder/'metadata.json').write_text(json.dumps(info,indent=2));print('Fetched real region',i,flush=True)
    return info


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    scenes=torch.load('models/photometry/self_supervised/runs/q1_all_bands/scenes.pt',weights_only=False,map_location='cpu')['splits']['test']
    centers=np.array([s['sky'][s['central']] for s in scenes]);lo=centers.min(0)-.007;hi=centers.max(0)+.007
    catalog=OUT/'mer_catalog.fits'
    if not catalog.exists():
        query=_ADQL.format(ra_where=f'm.ra BETWEEN {lo[0]} AND {hi[0]}',dec_min=lo[1],dec_max=hi[1],flux_min=0,extra_where='')
        (OUT/'mer_query.adql').write_text(query)
        cat=Irsa.query_tap(query,maxrec=200000).to_table()
        if len(cat)>=200000:raise ValueError('Truncated MER query')
        cat['is_star']=np.ma.filled(cat['point_like_prob'],0)>.96
        cat.write(catalog)
        print('Retrieved',len(cat),'real MER sources',flush=True)
    products_path=OUT/'products.json'
    if products_path.exists():products=json.loads(products_path.read_text())
    else:
        center=centers.mean(0)
        products=discover_mer_mosaics(*center,.25)
        products_path.write_text(json.dumps(products,indent=2))
    with ThreadPoolExecutor(max_workers=4) as pool:
        infos=list(pool.map(fetch_region,[(i,s,products) for i,s in enumerate(scenes)]))
    (OUT/'regions.json').write_text(json.dumps(infos,indent=2))

if __name__=='__main__':main()
