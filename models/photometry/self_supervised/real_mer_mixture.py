"""Evaluate frozen mixture priors on actual archive pixels at shared positions."""
from pathlib import Path
import argparse,json,traceback
import numpy as np
import pandas as pd
import torch
from .scene_features import SceneEncoder,image_features
from .run_mixture import predict_prior
from .mixture import fit_multiband
from .pixel_psf import normalize_kernel
from .data import wcs

OUT=Path('models/photometry/self_supervised/runs/real_mer')


def shared_scene(folder):
    scene=torch.load(folder/'scene.pt',weights_only=False,map_location='cpu')
    sky=np.load(folder/'fitted_sky.npy');old=scene['sky']
    offset=(sky-old)*np.c_[np.cos(np.deg2rad(old[:,1]))*3600,np.full(len(old),3600.)]
    with np.load(folder/'scene.npz') as arrays:
        for band,d in scene['bands'].items():
            delta=offset@d['sky_to_pixel'].numpy().T
            d['positions']=d['positions']+torch.tensor(delta,dtype=torch.float32)
            if not band.startswith('euclid'):continue
            wc=wcs(arrays[band+'__wcs'])
            d['positions']=torch.tensor(np.column_stack(wc.world_to_pixel_values(*sky.T)),dtype=torch.float32)
            psfsky=arrays[band+'__psf_field_sky']
            distance=np.hypot((sky[:,None,0]-psfsky[None,:,0])*np.cos(np.deg2rad(sky[:,None,1])),sky[:,None,1]-psfsky[None,:,1])
            d['psf_kernels']=np.array([normalize_kernel(arrays[band+'__psf_field_stamps'][j]) for j in distance.argmin(1)])
    scene['sky']=sky
    return scene


def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int);p.add_argument('--regions',type=int,nargs='*');a=p.parse_args();torch.set_num_threads(2)
    cp=torch.load('models/photometry/self_supervised/runs/q1_mixture_calibrated/priors.pt',weights_only=False,map_location='cpu');meta=cp['metadata']
    encoder=SceneEncoder(meta['original']['foundation_checkpoint'])
    folders=sorted(f.parent for f in OUT.glob('region_*/fitted_sky.npy'))
    if a.regions is not None:
        wanted=set(a.regions);folders=[f for f in folders if int(f.name.split('_')[1]) in wanted]
    if a.count:folders=folders[:a.count]
    status_path=OUT/'mixture_status.json'
    previous=[]
    if a.regions is not None and status_path.exists():
        previous=json.loads(status_path.read_text())
    statuses=[]
    for folder in folders:
        try:
            scene=shared_scene(folder);refs=pd.read_csv(folder/'references.csv');metadata=json.loads((folder/'metadata.json').read_text())
            item=dict(scene=scene,foundation=encoder(scene),image=image_features(scene));rows=[];models={};banks={}
            for mode in ['image','foundation']:
                prior=predict_prior(item,cp['heads'][mode],cp['population'],mode)
                fits=fit_multiband(scene,prior,banks=banks,prior_precision=cp['heads'][mode].get('precision'),strength=meta['prior_strength'],band_strength=meta['band_strength'])
                for band,r in fits.items():
                    conversion=10**(.4*(23.9-metadata['bands'][band.split('_')[1]]['magzero'])) if band.startswith('euclid') else np.nan
                    models[band+'__'+mode]=r['model']
                    for i,ref in refs.iterrows():
                        rows.append(dict(region=metadata['region'],source=i,object_id=int(ref.object_id),band=band,model=mode,
                            flux_native=r['flux'][i],flux_ujy=r['flux'][i]*conversion,error_ujy=r['error'][i]*conversion,
                            footprint=r['footprint'][i],reduced_chi2=r['reduced_chi2']))
            pd.DataFrame(rows).to_csv(folder/'mixture_fluxes.csv',index=False)
            np.savez_compressed(folder/'mixture_models.npz',**models)
            torch.save(scene,folder/'shared_scene.pt')
            statuses.append(dict(region=metadata['region'],sources=len(refs)));print('Real mixture',statuses[-1],flush=True)
        except Exception:statuses.append(dict(region=folder.name,error=traceback.format_exc()));print(statuses[-1],flush=True)
        if a.regions is None:
            status_path.write_text(json.dumps(statuses,indent=2))
        else:
            merged={int(str(x['region']).split('_')[-1]) if isinstance(x['region'],str) else int(x['region']):x for x in previous}
            merged.update({int(x['region']):x for x in statuses})
            status_path.write_text(json.dumps([merged[k] for k in sorted(merged)],indent=2))
    if a.regions is not None:
        mapping={int(str(x['region']).split('_')[-1]) if isinstance(x['region'],str) else int(x['region']):x for x in previous}
        mapping.update({int(x['region']):x for x in statuses})
        statuses=[mapping[k] for k in sorted(mapping)]
        status_path.write_text(json.dumps(statuses,indent=2))
    if any('error' in s for s in statuses):raise RuntimeError('Audit mixture_status.json')

if __name__=='__main__':main()
