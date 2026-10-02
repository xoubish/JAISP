"""Paired PSF sensitivity test with archive GRID-PSFs and frozen learned priors."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from .injections import simulate_scene
from .scene_features import SceneEncoder,image_features
from .run_mixture import predict_prior
from .mixture import fit_multiband
from .pixel_psf import gaussian_kernel

EUCLID=('VIS','Y','J','H')
CONDITIONS=('gaussian','core_gaussian','grid')


def load_stamps(folder):
    data={}
    for band in EUCLID:
        path=next((folder/'psfs').glob(f'psf_grid_stamps_{band.lower()}_*.npz'))
        with np.load(path) as z:data[band]={k:z[k] for k in z.files}
    return data


def choose_kernels(data,seed):
    # Simulated sky coordinates are a local tangent-plane placeholder. The
    # selected PSFs are physically sampled near the actual heldout field.
    rng=np.random.default_rng(seed+719)
    anchor=seed%len(data['VIS']['ra'])
    center=np.array([data['VIS']['ra'][anchor],data['VIS']['dec'][anchor]])
    # Same deterministic relative blend positions as simulate_scene, without
    # using fluxes, sizes or other truth in the measurement pipeline.
    g=np.random.default_rng(seed);sep=g.uniform(.4,1.4);angle=g.uniform(0,2*np.pi)
    offsets=np.array([-1,1])[:,None]*sep/2*np.array([np.cos(angle),np.sin(angle)])
    center+=rng.uniform(-2,2,2)/[3600*np.cos(np.deg2rad(center[1])),3600]
    sky=center+offsets/[3600*np.cos(np.deg2rad(center[1])),3600]
    kernels={};cores={};records=[]
    for band,d in data.items():
        selected=[];gaussians=[]
        for source,pos in enumerate(sky):
            distances=np.hypot((d['ra']-pos[0])*np.cos(np.deg2rad(pos[1])),d['dec']-pos[1])*3600
            j=distances.argmin();k=d['stamps'][j].astype(float);k/=k.sum()
            selected.append(k)
            gaussians.append(gaussian_kernel(k.shape,float(d['fwhm'][j])/(2.354820045*.1)))
            records.append(dict(source=source,band='euclid_'+band,stamp_index=int(j),
                field_ra=pos[0],field_dec=pos[1],distance_arcsec=float(distances[j]),fwhm_arcsec=float(d['fwhm'][j])))
        kernels['euclid_'+band]=np.array(selected);cores['euclid_'+band]=np.array(gaussians)
    return kernels,cores,records


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,default=Path('models/photometry/self_supervised/runs/q1_mixture_calibrated'))
    p.add_argument('--out',type=Path,default=Path('models/photometry/self_supervised/runs/psf_study'))
    p.add_argument('--count',type=int,default=128);p.add_argument('--seed',type=int,default=20261201)
    args=p.parse_args();torch.set_num_threads(2)
    cp=torch.load(args.run/'priors.pt',weights_only=False,map_location='cpu');meta=cp['metadata']
    template=torch.load(Path(meta['source'])/'scenes.pt',weights_only=False,map_location='cpu')['splits']['test'][0]
    encoder=SceneEncoder(meta['original']['foundation_checkpoint']);data=load_stamps(args.out)
    rows=[];stamp_rows=[];failures=[]
    for i in range(args.count):
        seed=args.seed+i;kernels,cores,selection=choose_kernels(data,seed)
        scene,truth,config=simulate_scene(template,seed,weak_vis=(seed//2)%2==0,psf_kernels=kernels)
        stamp_rows.extend([dict(scene=i,**r) for r in selection])
        # Encode the new empirical-PSF image once: same observed data and same
        # frozen morphology prior for all three assumed-PSF conditions.
        item=dict(scene=scene,foundation=encoder(scene),image=image_features(scene))
        priors={mode:predict_prior(item,cp['heads'][mode],cp['population'],mode) for mode in ('image','foundation')}
        for condition in CONDITIONS:
            candidate=copy.deepcopy(scene)
            for band,d in candidate['bands'].items():
                if band not in kernels:continue
                if condition=='gaussian':d.pop('psf_kernels')
                elif condition=='core_gaussian':d['psf_kernels']=cores[band]
            export={'sky':candidate['sky']}
            for band,d in candidate['bands'].items():
                for key,value in d.items():export[f'{band}__{key}']=value.numpy() if torch.is_tensor(value) else value
                export[f'{band}__truth']=truth[band]['flux']
            target=args.out/condition/'tractor_inputs';target.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target/f'scene_{i:04d}.npz',**export)
            banks={}
            for mode,prior in priors.items():
                try:
                    fits=fit_multiband(candidate,prior,banks=banks,prior_precision=cp['heads'][mode].get('precision'),
                        strength=meta['prior_strength'],band_strength=meta['band_strength'])
                    for band,r in fits.items():
                        for source in range(2):
                            true=truth[band]['flux'][source]
                            rows.append(dict(scene=i,source=source,condition=condition,model=mode,band=band,
                                truth_flux=true,flux=r['flux'][source],error=r['error'][source],fractional_error=r['flux'][source]/true-1,**config))
                except Exception as exc:failures.append(dict(scene=i,condition=condition,model=mode,error=str(exc)))
        pd.DataFrame(rows).to_csv(args.out/'mixture_fluxes.csv',index=False)
        print(f'PSF scene {i+1}/{args.count}',flush=True)
    pd.DataFrame(stamp_rows).to_csv(args.out/'stamp_selection.csv',index=False)
    (args.out/'failures.json').write_text(json.dumps(failures,indent=2))
    (args.out/'protocol.json').write_text(json.dumps(dict(count=args.count,seed=args.seed,conditions=list(CONDITIONS),
        truth='Euclid: archived GRID-PSF samples, Fourier-convolved exponential core+disk galaxies; Rubin: original Gaussian simulation',
        template_tile=template['tile'],field='EDF-S, MER tile 102044185; 19 grid locations around RA 53.25555525 Dec -28.06391707',
        source_sample='128 two-source blends if default count; half weak VIS, half correlated noise; known fixed positions',
        conditions_detail='gaussian: old global sigma; core_gaussian: nearest-stamp FWHM circular Gaussian, pixelized; grid: nearest full archive stamp',
        controls='Identical noisy images, truth fluxes, source lists, priors and model settings across PSF conditions; Gaussian ellipse initializer held fixed',
        training='Existing priors frozen, no retraining or tuning on this PSF experiment',
        units='Native units; no AB zero points assumed',
        limitations='Finite stamps normalized as in upstream notebook; one local field; empirical truth and grid fitter share PSF so this is a matched-PSF sensitivity experiment, not survey validation'),indent=2))
    if failures:raise RuntimeError(f'{len(failures)} failures; audit failures.json')

if __name__=='__main__':main()
