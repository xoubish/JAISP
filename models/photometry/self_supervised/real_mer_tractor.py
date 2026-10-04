"""Actual-image upstream SEP/Tractor fitting; runs in the isolated environment."""
from pathlib import Path
import argparse,json,sys,traceback,copy
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from astropy.table import Table
from tractor import Tractor,LinearPhotoCal,ConstantSky
from .tractor_compare import image_for_band,joint_flux,VENDOR
from euclid_phot.selection import run_model_selection
from euclid_phot.models import build_sources_from_mer
from euclid_phot.nisp import _clone_for_band
from euclid_phot.spatial_psf import SpatialPixelizedPSF

OUT=Path('models/photometry/self_supervised/runs/real_mer')


def make_image(scene,band):
    tim=image_for_band(scene,band)
    short=band.split('_')[1]
    scale=10**(.4*(float(scene[band+'__magzero'])-22.5))
    tim.photocal=LinearPhotoCal(scale,band=short)
    tim.sky=ConstantSky(0.)  # upstream notebook's background-subtracted mosaic convention
    tim.psf=SpatialPixelizedPSF(dict(stamps=scene[band+'__psf_field_stamps'],
        ra=scene[band+'__psf_field_sky'][:,0],dec=scene[band+'__psf_field_sky'][:,1]),tim.getWcs())
    tim.freezeAllParams()
    return tim,scale


def worker(folder):
    try:
        with np.load(folder/'scene.npz') as z:scene={k:z[k] for k in z.files}
        cat=Table.read(folder/'sources.fits');region=int(folder.name.split('_')[1])
        tim,scale=make_image(scene,'euclid_VIS')
        # Actual upstream detection, segmentation, full model ladder and bounded
        # centroid refinement. No geometric synthetic segmentation is used.
        selected,counts=run_model_selection(cat,tim,tim.getImage(),tim.getInvvar(),pixscale_arcsec=.1,n_workers=1)
        with np.load(folder/'VIS.npz') as vis_archive: psf_fwhm=float(np.median(vis_archive['psf_fwhm']))
        fallback=build_sources_from_mer(cat,band='VIS',psf_fwhm_arcsec=psf_fwhm)
        fallback_used=np.array([s is None for s in selected])
        sources=[s if s is not None else f for s,f in zip(selected,fallback)]
        fitted_sky=np.array([[s.getPosition().ra,s.getPosition().dec] for s in sources])
        np.save(folder/'fitted_sky.npy',fitted_sky)
        rows=[];models={}
        for band in ['euclid_VIS','euclid_Y','euclid_J','euclid_H']:
            tim,scale=make_image(scene,band);short=band.split('_')[1]
            clones=[_clone_for_band(s,short) for s in sources]
            for s in clones:s.freezeAllBut('brightness')
            tr=Tractor([tim],clones)
            fit=tr.optimize_forced_photometry(minsb=0.,mindlnp=1.,sky=False,variance=True)
            flux=np.array([s.brightness.getFlux(short) for s in clones])*scale
            iv=np.asarray(fit.IV);error=np.where(iv>0,scale/np.sqrt(np.maximum(iv,1e-300)),np.nan)
            models[band+'__tractor_upstream']=tr.getModelImage(0)
            try:
                joint=joint_flux(tim,sources,short)
                joint_ok=True
            except RuntimeError as exc:
                if 'Singular flux/sky fit' not in str(exc):raise
                joint=dict(flux=np.full(len(sources),np.nan),error=np.full(len(sources),np.nan))
                joint_ok=False
            if joint_ok:
                joint_model=joint['model']
            else:
                # Preserve the valid upstream fit when the optional joint-sky
                # diagnostic is underconstrained; its model is unavailable.
                joint_model=np.full_like(tim.getImage(),np.nan,dtype=float)
            models[band+'__tractor_jointsky']=joint_model
            for mode,values,errors in [('tractor_upstream',flux,error),('tractor_jointsky',joint['flux']*scale,joint['error']*scale)]:
                conversion=10**(.4*(23.9-float(scene[band+'__magzero'])))
                for i,s in enumerate(sources):
                    rows.append(dict(region=region,source=i,object_id=int(cat['object_id'][i]),band=band,model=mode,
                        flux_native=values[i],flux_ujy=values[i]*conversion,error_ujy=errors[i]*conversion,
                        profile=type(s).__name__,catalog_shape_fallback=bool(fallback_used[i]),
                        jointsky_constrained=bool(joint_ok) if mode=='tractor_jointsky' else True))
        pd.DataFrame(rows).to_csv(folder/'tractor_fluxes.csv',index=False)
        np.savez_compressed(folder/'tractor_models.npz',**models)
        (folder/'selection.json').write_text(json.dumps(counts,indent=2))
        print('Real Tractor region',region,'sources',len(sources),flush=True)
        return dict(region=region,sources=len(sources))
    except Exception:return dict(region=folder.name,error=traceback.format_exc())


def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int);p.add_argument('--regions',type=int,nargs='*');p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    folders=sorted(f.parent for f in OUT.glob('region_*/scene.npz'))
    if a.regions is not None:
        selected=set(a.regions);folders=[f for f in folders if int(f.name.split('_')[1]) in selected]
    elif a.count:folders=folders[:a.count]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(worker,folders))
    status=OUT/'tractor_status.json'
    if a.regions is not None and status.exists():
        previous=json.loads(status.read_text());mapping={int(str(x['region']).split('_')[-1]) if isinstance(x['region'],str) else int(x['region']):x for x in previous}
        mapping.update({int(x['region']):x for x in results});results=[mapping[k] for k in sorted(mapping)]
    status.write_text(json.dumps(results,indent=2))
    if any('error' in r for r in results):raise RuntimeError('See tractor_status.json for failed real regions')

if __name__=='__main__':main()
