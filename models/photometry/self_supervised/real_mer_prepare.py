"""Prepare measured image scenes and MER references; no simulated pixels."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from astropy.table import Table
from .data import wcs,jacobian,cut
from .core import BANDS
from .pixel_psf import normalize_kernel

OUT=Path('models/photometry/self_supervised/runs/real_mer')


def prepare(folder,fullcat,sigmas):
    meta=json.loads((folder/'metadata.json').read_text());raw={}
    for b in ('VIS','Y','J','H'):
        with np.load(folder/f'{b}.npz') as z:raw[b]={k:z[k] for k in z.files}
    wc=wcs(raw['VIS']['wcs']);positions=np.column_stack(wc.world_to_pixel_values(fullcat['ra'],fullcat['dec']))
    h,width=raw['VIS']['image'].shape
    on=(positions[:,0]>=0)&(positions[:,0]<width)&(positions[:,1]>=0)&(positions[:,1]<h)
    cat=fullcat[on];sky=np.column_stack((cat['ra'],cat['dec'])).astype(float)
    if not len(cat):raise ValueError('Empty source list')
    bands={};arrays={'sky':sky,'object_id':np.array(cat['object_id'])}
    # Real flags: exclude fatal pixels and the VIS bright-star signal mask.
    visstar=(raw['VIS']['flag'].astype('int64')&(1<<18))!=0
    for short,d in raw.items():
        b='euclid_'+short;wc=wcs(d['wcs']);xy=np.column_stack(wc.world_to_pixel_values(*sky.T))
        center=np.array(wc.world_to_pixel_values(meta['ra'],meta['dec']))
        jac=np.linalg.inv(jacobian(wc,center));im=d['image'].astype('float32');var=d['variance'].astype('float32')
        bad=((1<<0)|(1<<3)|(1<<22)) if short=='VIS' else ((1<<0)|(1<<10))
        mask=np.isfinite(im)&np.isfinite(var)&(var>0)&((d['flag'].astype('int64')&bad)==0)&~visstar
        # Huge archive RMS denotes no coverage even when flag bits are absent.
        typical=np.median(var[mask]);mask &= var<typical*1e6
        distance=np.hypot((sky[:,None,0]-d['psf_ra'][None,:])*np.cos(np.deg2rad(sky[:,None,1])),sky[:,None,1]-d['psf_dec'][None,:])
        kernels=np.array([normalize_kernel(d['psf_stamps'][j]) for j in distance.argmin(1)])
        bands[b]=dict(image=torch.tensor(im),variance=torch.tensor(var),mask=torch.tensor(mask),positions=torch.tensor(xy,dtype=torch.float32),
            sky_to_pixel=torch.tensor(jac,dtype=torch.float32),psf_sigma=sigmas[b],psf_kernels=kernels)
        arrays[b+'__wcs']=d['wcs']
        arrays[b+'__magzero']=d['magzero'];arrays[b+'__psf_field_stamps']=d['psf_stamps']
        arrays[b+'__psf_field_sky']=np.column_stack([d['psf_ra'],d['psf_dec']])
    rp=next(Path('data/rubin_tiles_all').rglob(meta['tile']+'.npz'))
    with np.load(rp,allow_pickle=True) as r:
        wc=wcs(r['wcs_hdr']);center=np.array(wc.world_to_pixel_values(meta['ra'],meta['dec']))
        jac=np.linalg.inv(jacobian(wc,center));scale=np.sqrt(abs(np.linalg.det(np.linalg.inv(jac))))
        for i,b in enumerate(BANDS[:6]):
            var=r['var'][i].copy();var[(r['mask'][i]&(1|2|256))!=0]=np.nan
            stamp=cut(r['img'][i],var,center,int(np.ceil(10/scale)))
            if stamp is None:raise ValueError(f'Incomplete Rubin scene: {b}')
            im,var,mask,origin=stamp;xy=np.column_stack(wc.world_to_pixel_values(*sky.T))-origin
            bands[b]=dict(image=torch.tensor(im),variance=torch.tensor(var),mask=torch.tensor(mask),positions=torch.tensor(xy,dtype=torch.float32),
                sky_to_pixel=torch.tensor(jac,dtype=torch.float32),psf_sigma=sigmas[b])
    scene=dict(tile=meta['tile'],central=0,sky=sky,bands=bands)
    xy=bands['euclid_VIS']['positions'].numpy();edge=np.min(np.c_[xy,width-1-xy[:,0],h-1-xy[:,1]],axis=1)*.1
    separation=np.linalg.norm((sky[:,None]-sky[None,:])*[np.cos(np.deg2rad(meta['dec']))*3600,3600],axis=-1)
    np.fill_diagonal(separation,np.inf)
    reference=cat.to_pandas();reference['edge_arcsec']=edge;reference['neighbor_arcsec']=separation.min(1)
    flags=np.ma.filled(cat['det_quality_flag'],0).astype('int64')
    reference['clean_geometry']=(edge>=3)&((flags&((1<<7)|(1<<8)))==0)
    centerok=np.ones(len(cat),bool)
    for b,d in bands.items():
        if not b.startswith('euclid'):continue
        xx,yy=np.rint(d['positions'].numpy()).astype(int).T
        inside=(xx>=0)&(yy>=0)&(xx<d['image'].shape[1])&(yy<d['image'].shape[0]);ok=np.zeros(len(cat),bool)
        ok[inside]=d['mask'].numpy()[yy[inside],xx[inside]];centerok &=ok
    reference['clean_geometry'] &=centerok
    reference['region']=meta['region'];reference.to_csv(folder/'references.csv',index=False)
    # Separate archive reference catalog from fitting arrays. No truth arrays.
    cat.write(folder/'sources.fits',overwrite=True)
    for b,d in bands.items():
        for k,v in d.items():arrays[b+'__'+k]=v.numpy() if torch.is_tensor(v) else v
    np.savez_compressed(folder/'scene.npz',**arrays)
    torch.save(scene,folder/'scene.pt')
    return dict(region=meta['region'],sources=len(cat),interior=int(reference.clean_geometry.sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int);a=p.parse_args()
    fullcat=Table.read(OUT/'mer_catalog.fits')
    template=torch.load('models/photometry/self_supervised/runs/q1_all_bands/scenes.pt',weights_only=False)['splits']['test'][0]
    sigmas={b:d['psf_sigma'] for b,d in template['bands'].items()}
    folders=sorted(p.parent for p in OUT.glob('region_*/metadata.json'))
    if a.count:folders=folders[:a.count]
    reports=[];failed=[]
    for f in folders:
        try:reports.append(prepare(f,fullcat,sigmas));print(reports[-1],flush=True)
        except Exception as exc:failed.append(dict(region=f.name,error=str(exc)));print(f,exc,flush=True)
    (OUT/'preparation.json').write_text(json.dumps(dict(regions=reports,failed=failed),indent=2))
    if failed:raise RuntimeError('Preparation failures require audit')

if __name__=='__main__':main()
