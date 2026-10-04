"""Run upstream VIS model selection and Tractor rendering on paired JAISP scenes.

Run with the isolated Python >=3.10 Tractor environment. No torch required.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.wcs import WCS
from astropy.io import fits

VENDOR=Path(__file__).resolve().parent/'vendor/euclid_forced_photometry'
sys.path.insert(0,str(VENDOR/'src'))
from tractor import Image, ConstantSky, LinearPhotoCal, RaDecPos
from tractor.psf import GaussianMixturePSF
from tractor.wcs import ConstantFitsWcs
from euclid_phot.images import AstropyWCSAdapter
from euclid_phot.selection import Blob, ModelSelector
from euclid_phot.nisp import _clone_for_band


class FixedPositionSelector(ModelSelector):
    """Use upstream profile ladder, fixing shared positions in every fit stage.

    Upstream catches optimizer exceptions; raise them instead so failures cannot
    silently masquerade as converged measurements in this benchmark.
    """
    def _optimize(self, tractor):
        for source in tractor.getCatalog():
            source.freezeParam('pos')
        for _ in range(self.max_steps):
            improvement,_,_=tractor.optimize()
            if not np.isfinite(improvement):raise RuntimeError('Nonfinite optimization improvement')
            if improvement<self.dlnp_crit:break


def image_for_band(scene,band):
    image=scene[f'{band}__image'].astype(float)
    var=scene[f'{band}__variance'].astype(float)
    valid=scene[f'{band}__mask']&np.isfinite(image)&np.isfinite(var)&(var>0)
    pos=scene[f'{band}__positions'];jac=scene[f'{band}__sky_to_pixel']
    wcs=WCS(naxis=2);wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    wcs.wcs.crval=scene['sky'][0];wcs.wcs.crpix=pos[0]+1
    wcs.wcs.cd=np.linalg.inv(jac)/3600.
    if f'{band}__wcs' in scene:
        wcs=WCS(fits.Header.fromstring(str(scene[f'{band}__wcs'])))
    twcs=ConstantFitsWcs(AstropyWCSAdapter(wcs))
    for sky,xy in zip(scene['sky'],pos):
        np.testing.assert_allclose(twcs.positionToPixel(RaDecPos(*sky)),xy,atol=2e-3)
    sigma=float(scene[f'{band}__psf_sigma'])
    psf=GaussianMixturePSF(np.ones(1),np.zeros((1,2)),np.eye(2)[None]*sigma**2)
    if f'{band}__psf_kernels' in scene:
        from euclid_phot.spatial_psf import SpatialPixelizedPSF
        psf=SpatialPixelizedPSF(dict(stamps=scene[f'{band}__psf_kernels'],
            ra=scene['sky'][:,0],dec=scene['sky'][:,1]),twcs)
    yy,xx=np.indices(image.shape)
    distance=np.min(np.linalg.norm(np.stack((xx,yy),-1)[:,:,None,:]-pos,axis=-1),axis=-1)
    outer=valid&(distance>4*np.sqrt(abs(np.linalg.det(jac))))
    sky=float(np.median(image[outer])) if outer.any() else float(np.median(image[valid]))
    tim=Image(data=np.where(valid,image,0),invvar=np.where(valid,1/np.where(valid,var,1),0),
        psf=psf,wcs=twcs,photocal=LinearPhotoCal(1.,band='VIS' if band=='euclid_VIS' else band),sky=ConstantSky(sky),name=band)
    tim.freezeAllParams()
    return tim


def joint_flux(tim,sources,band):
    """Signed linear fit of Tractor templates plus sky, full blend covariance.

    This uses the same likelihood as our foundation fitter, avoiding the
    upstream frozen-sky and diagonal-only uncertainty convention differences.
    """
    clones=[_clone_for_band(s,band) for s in sources]
    columns=[]
    for s in clones:
        s.freezeAllBut('brightness')
        patch=s.getModelPatch(tim)
        model=np.zeros(tim.shape)
        if patch is not None:patch.addTo(model)
        columns.append(model.ravel())
    design=np.column_stack(columns+[np.ones(np.prod(tim.shape))])
    weight=tim.getInvvar().ravel();valid=weight>0
    a=design[valid]*np.sqrt(weight[valid,None]);y=tim.getImage().ravel()[valid]*np.sqrt(weight[valid])
    coeff,_,rank,_=np.linalg.lstsq(a,y,rcond=None)
    if rank<a.shape[1]:raise RuntimeError('Singular flux/sky fit')
    covariance=np.linalg.inv(a.T@a)
    return dict(flux=coeff[:-1],error=np.sqrt(np.diag(covariance)[:-1]),
                sky=coeff[-1],condition=np.linalg.cond(a),chi2=float(np.sum((y-a@coeff)**2)),
                model=(design@coeff).reshape(tim.shape))


def fit_scene(path):
    start=time.monotonic();scene_id=int(path.stem.split('_')[-1])
    try:
        with np.load(path) as z:scene={k:z[k] for k in z.files}
        vis=image_for_band(scene,'euclid_VIS');pos=scene['euclid_VIS__positions']
        yy,xx=np.indices(vis.shape);dist=np.linalg.norm(np.stack((xx,yy),-1)[:,:,None,:]-pos,axis=-1)
        nearest=dist.argmin(axis=-1);pixscale=np.sqrt(abs(np.linalg.det(np.linalg.inv(scene['euclid_VIS__sky_to_pixel']))))
        # Entire supplied group is fitted together, including VIS-faint sources.
        # Fixed geometric segments replace detection-dependent SEP membership.
        segments={i:(nearest==i)&(dist[:,:,i]<1.5/pixscale) for i in range(len(pos))}
        blob=Blob(0,vis.getInvvar()>0,list(range(len(pos))),segments)
        # Data-only initial amplitudes; factor 3.631 cancels the upstream
        # microJy-to-NanoMaggies seed conversion. These remain native counts.
        fluxseed=[max(float(np.sum((vis.getImage()-vis.getSky().getValue())[segments[i]])),1e-8) for i in range(len(pos))]
        cat=Table(dict(ra=scene['sky'][:,0],dec=scene['sky'][:,1],flux_vis_sersic=np.array(fluxseed)*3.631))
        selector=FixedPositionSelector()
        chosen,_=selector.fit_blob(blob,vis,cat,pixscale)
        sources=[chosen[i] for i in range(len(pos))]
        rows=[]
        for band in [k[:-7] for k in scene if k.endswith('__image')]:
            tim=image_for_band(scene,band);label='VIS' if band=='euclid_VIS' else band
            result=joint_flux(tim,sources,label)
            for i,source in enumerate(sources):
                true=scene[f'{band}__truth'][i]
                rows.append(dict(scene=scene_id,source=i,model='tractor_vis',band=band,
                    truth_flux=true,flux=result['flux'][i],error=result['error'][i],
                    fractional_error=result['flux'][i]/true-1,profile=type(source).__name__,
                    condition=result['condition'],chi2=result['chi2']))
        return dict(scene=scene_id,rows=rows,seconds=time.monotonic()-start)
    except Exception:
        return dict(scene=scene_id,error=traceback.format_exc(),seconds=time.monotonic()-start)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--count',type=int,default=None);p.add_argument('--workers',type=int,default=1)
    args=p.parse_args();out=args.run/'tractor_comparison';out.mkdir(exist_ok=True)
    paths=sorted((args.run/'tractor_inputs').glob('scene_*.npz'))
    if args.count:paths=paths[:args.count]
    if not paths:raise ValueError('Export scenes using tractor_export first')
    with np.load(paths[0]) as first:
        kernel_bands=[key[:-13] for key in first.files if key.endswith('__psf_kernels')]
    rows=[];failures=[];timings=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(fit_scene,paths):
            timings.append(dict(scene=result['scene'],seconds=result['seconds']))
            if 'error' in result:failures.append(result)
            else:rows.extend(result['rows'])
            print(f"Scene {result['scene']}: {'FAILED' if 'error' in result else 'OK'} ({result['seconds']:.1f}s)",flush=True)
            pd.DataFrame(rows).to_csv(out/'tractor_fluxes.csv',index=False)
            (out/'failures.json').write_text(json.dumps(failures,indent=2))
    pd.DataFrame(timings).to_csv(out/'timings.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(upstream=json.loads((VENDOR/'UPSTREAM.json').read_text()),
        dependencies=dict(tractor_commit='3fd2e80eafb9cc092e203ba50a95557eb8543878',
            astrometry_commit='d20a0503739e74b02418cde2e7d65013dadac579',python=sys.version.split()[0],
            numerical_versions='requirements-tractor-lock.txt'),
        n_scenes=len(paths),models='Upstream full VIS profile ladder, including guarded Sersic tier',
        positions='Known shared source list; fixed at every optimization stage',
        segmentation='Joint whole-scene group; nearest-source segments within 1.5 arcsec; no detection selection',
        psf=('Per-source pixelized kernels in '+', '.join(kernel_bands)+'; shared Gaussian sigma elsewhere' if kernel_bands else 'Shared Gaussian sigma; Tractor analytic Gaussian PSF'),
        background='VIS shape fit: median outside 4 arcsec of all sources; final every-band flux fit: free constant sky',
        flux_solver='Joint signed weighted linear least squares using Tractor-rendered VIS profiles; full blend+sky covariance',
        units='Native image units; no assumed AB calibration; upstream seed conversion canceled',
        reference='Same saved controlled simulations and foundation predictions; no ground-truth morphology used',
        scope='Adapted upstream Tractor VIS-prior comparison, not the unmodified archive notebook'),indent=2))
    if failures:raise RuntimeError(f'{len(failures)} scenes failed; see failures.json')

if __name__=='__main__':main()
