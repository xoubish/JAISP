"""Independent exponential-galaxy blend benchmark with fresh encoder evaluation.

Profiles are rendered on a fine grid and convolved with scipy, independently of
our Gaussian fitting dictionary. Truth is used only by the evaluator, never the
photometry head. This tests known positions and PSFs under specified synthetic
noise; it is not a survey completeness or PSF-calibration test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from .core import BANDS, ConstantMorphologyHead, fit_scene
from .mixture import fit_multiband
from .scene_features import SceneEncoder, image_features
from .run_mixture import predict_prior


def exponential_profile(shape,position,covariance,psf_sigma,oversample=7):
    """Independent pixel integration + Gaussian PSF convolution, unit total flux."""
    yy,xx=np.meshgrid((np.arange(shape[0]*oversample)+.5)/oversample-.5,
                      (np.arange(shape[1]*oversample)+.5)/oversample-.5,indexing='ij')
    delta=np.stack((xx-position[0],yy-position[1]),-1)
    r2=np.einsum('...i,ij,...j->...',delta,np.linalg.inv(covariance),delta)
    fine=np.exp(-np.sqrt(np.maximum(r2,0)))
    fine/=fine.sum()/oversample**2
    fine=gaussian_filter(fine,psf_sigma*oversample,mode='constant',truncate=5.)
    return fine.reshape(shape[0],oversample,shape[1],oversample).mean((1,3))


def legacy_covariance(scene):
    d=scene['bands']['euclid_VIS'];image=d['image'].numpy()
    valid=d['mask'].numpy();result=[];p2s=np.linalg.inv(d['sky_to_pixel'].numpy())
    for xy in d['positions'].numpy():
        x,y=np.rint(xy).astype(int);a=image[y-10:y+11,x-10:x+11];mask=valid[y-10:y+11,x-10:x+11]
        yy,xx=np.indices(a.shape);delta=np.stack((xx+x-10-xy[0],yy+y-10-xy[1]),-1)
        radius=np.linalg.norm(delta,axis=-1);bg=np.median(a[(radius>8)&mask])
        signal=np.maximum(a-bg,0)*np.exp(-radius**2/32)*mask
        moment=np.einsum('hw,hwi,hwj->ij',signal,delta,delta)/max(signal.sum(),1e-12)
        sky=p2s@(moment-np.eye(2)*d['psf_sigma']**2)@p2s.T
        values,vectors=np.linalg.eigh(sky)
        result.append((vectors*np.clip(values,.025**2,.45**2))@vectors.T)
    return torch.tensor(np.array(result),dtype=torch.float32)


def simulate_scene(template,seed,weak_vis=False,psf_kernels=None):
    rng=np.random.default_rng(seed)
    separation=float(rng.uniform(.4,1.4));angle=rng.uniform(0,2*np.pi)
    direction=np.array([np.cos(angle),np.sin(angle)])
    offsets=np.array([-direction*separation/2,direction*separation/2])
    sizes=rng.uniform(.09,.28,2);ratios=rng.uniform(.65,1,2)
    angles=rng.uniform(0,np.pi,2);core_fraction=rng.uniform(.15,.55,2)
    gradient=rng.uniform(-.25,.25,2)
    geometry=[]
    for size,q,theta in zip(sizes,ratios,angles):
        rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        geometry.append(rotation@np.diag([size**2,(q*size)**2])@rotation.T)
    bands={};truth={}
    correlated=bool(seed%2)
    base_snr=np.exp(rng.uniform(np.log(8),np.log(60),2))
    for ib,band in enumerate(BANDS):
        original=template['bands'][band];shape=tuple(original['image'].shape)
        jac=original['sky_to_pixel'].numpy()
        center=np.array([(shape[1]-1)/2,(shape[0]-1)/2])
        positions=center+offsets@jac.T
        sigma=float(np.sqrt(np.median(original['variance'].numpy()[original['mask'].numpy()])))
        signal=np.zeros(shape);fluxes=[];isolated_snr=[]
        for source in range(2):
            covariance=jac@geometry[source]@jac.T
            if psf_kernels is not None and band in psf_kernels:
                from .pixel_psf import convolved_profile
                disk=convolved_profile(shape,positions[source],covariance,psf_kernels[band][source],kind='exponential')
                core=convolved_profile(shape,positions[source],covariance*.35**2,psf_kernels[band][source],kind='exponential')
            else:
                disk=exponential_profile(shape,positions[source],covariance,original['psf_sigma'])
                core=exponential_profile(shape,positions[source],covariance*.35**2,original['psf_sigma'])
            fraction=np.clip(core_fraction[source]+gradient[source]*(ib-6)/6,.05,.9)
            profile=fraction*core+(1-fraction)*disk
            # Broad colors; faint u/y and variable VIS visibility.
            multiplier=[.45,.9,1.3,1.2,1.,.6,1.,.9,1.,1.1][ib]
            snr=base_snr[source]*multiplier*np.exp(rng.normal(0,.3))
            if weak_vis and band=='euclid_VIS':snr/=3
            flux=snr*sigma/np.sqrt(np.sum(profile**2))
            signal+=flux*profile;fluxes.append(flux);isolated_snr.append(snr)
        noise=rng.normal(size=shape)
        if correlated:
            # Gaussian-correlated noise with the same marginal pixel variance.
            noise=gaussian_filter(noise,.65,mode='reflect')
            noise/=noise.std()
        image=signal+sigma*noise+rng.uniform(-.5,.5)*sigma
        bands[band]=dict(image=torch.tensor(image,dtype=torch.float32),
                         variance=torch.full(shape,sigma**2,dtype=torch.float32),
                         mask=torch.ones(shape,dtype=torch.bool),
                         positions=torch.tensor(positions,dtype=torch.float32),
                         sky_to_pixel=original['sky_to_pixel'].clone(),psf_sigma=original['psf_sigma'])
        if psf_kernels is not None and band in psf_kernels:
            bands[band]['psf_kernels']=np.asarray(psf_kernels[band])
        truth[band]=dict(flux=np.array(fluxes),snr=np.array(isolated_snr))
    scene=dict(tile=f'simulation_{seed}',central=0,
               sky=np.array([53.,-28.])+offsets/[3600*np.cos(np.deg2rad(-28)),3600],bands=bands)
    scene['covariance']=legacy_covariance(scene)
    return scene,truth,dict(seed=seed,separation_arcsec=separation,correlated_noise=correlated,weak_vis=weak_vis,
                            sizes_arcsec=sizes.tolist(),axis_ratios=ratios.tolist())


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,default=Path('models/photometry/self_supervised/runs/q1_mixture'))
    p.add_argument('--count',type=int,default=64)
    p.add_argument('--seed',type=int,default=20261001)
    p.add_argument('--weak-vis',action='store_true',help='Reduce VIS S/N by three in half the scenes, balanced across noise types')
    p.add_argument('--scarlet',type=str,default='',help='Amortised-scarlet checkpoint to evaluate alongside the mixture priors (model name "scarlet")')
    p.add_argument('--suffix',type=str,default='',help='Output file suffix, e.g. _scarlet, to keep the original benchmark files untouched')
    p.add_argument('--mixture-fixed-background',action='store_true',help='Also evaluate the mixture priors with a fixed robust background (models *_fixedbg)')
    args=p.parse_args();torch.set_num_threads(4)
    scarlet=None
    if args.scarlet:
        from .amortised_scarlet import AmortisedScarletPhotometry
        scarlet=AmortisedScarletPhotometry(args.scarlet)
    checkpoint=torch.load(args.run/'priors.pt',map_location='cpu',weights_only=False)
    meta=checkpoint['metadata'];source=Path(meta['source'])
    old=torch.load(source.parent/'q1_all_bands_constant/head.pt',map_location='cpu',weights_only=False)
    baseline=ConstantMorphologyHead(old['head_state']['mean_features']);baseline.load_state_dict(old['head_state']);baseline.eval()
    encoder=SceneEncoder(meta['original']['foundation_checkpoint'])
    templates=torch.load(source/'scenes.pt',map_location='cpu',weights_only=False)['splits']['test']
    rows=[];examples=[];failures=[]
    for i in range(args.count):
        scene,truth,config=simulate_scene(templates[i%len(templates)],args.seed+i,weak_vis=args.weak_vis and ((args.seed+i)//2)%2==0)
        # Fresh encoding is mandatory: no cache from real/clean images is used.
        item=dict(scene=scene,foundation=encoder(scene),image=image_features(scene))
        priors={'population':np.tile(checkpoint['population'],(2,1))}
        for kind,head in checkpoint['heads'].items():
            priors[kind]=predict_prior(item,head,checkpoint['population'],kind)
        fits={};banks={}
        for mode,prior in priors.items():
            try:
                precision=checkpoint.get('population_precision') if mode=='population' else checkpoint['heads'][mode].get('precision')
                fits[mode]=fit_multiband(scene,prior,banks=banks,strength=meta['prior_strength'],band_strength=meta['band_strength'],prior_precision=precision)
            except (ValueError,RuntimeError) as exc:failures.append(dict(scene=i,model=mode,error=str(exc)))
            if args.mixture_fixed_background and mode!='population':
                try:fits[mode+'_fixedbg']=fit_multiband(scene,prior,banks=banks,strength=meta['prior_strength'],band_strength=meta['band_strength'],prior_precision=precision,fixed_background=True)
                except (ValueError,RuntimeError) as exc:failures.append(dict(scene=i,model=mode+'_fixedbg',error=str(exc)))
        if scarlet is not None:
            try:
                r=scarlet(scene);fits['scarlet']={b:v for b,v in r.items() if not b.startswith('_')}
            except (ValueError,RuntimeError) as exc:failures.append(dict(scene=i,model='scarlet',error=str(exc)))
        with torch.no_grad():
            old_cov=baseline(torch.zeros(2,1),scene['covariance'])
            old_fit=fit_scene(scene,old_cov)
        fits['v1_global']={b:dict(flux=r['flux'].numpy(),error=r['error'].numpy(),model=r['model'].numpy()) for b,r in old_fit.items()}
        for mode,measurement in fits.items():
            for band,r in measurement.items():
                for j in range(2):
                    actual=truth[band]['flux'][j]
                    rows.append(dict(scene=i,source=j,model=mode,band=band,truth_flux=actual,
                                     flux=r['flux'][j],error=r['error'][j],
                                     fractional_error=(r['flux'][j]-actual)/actual,
                                     true_snr=truth[band]['snr'][j],**config))
        if i<4:examples.append(dict(scene=scene,truth=truth,config=config,fits=fits))
        print('Injection',i+1,'/',args.count,flush=True)
    sfx=args.suffix
    frame=pd.DataFrame(rows);frame.to_csv(args.run/f'injections{sfx}.csv',index=False)
    torch.save(examples,args.run/f'injection_examples{sfx}.pt')
    (args.run/f'injection_failures{sfx}.json').write_text(json.dumps(failures,indent=2))
    summary=[]
    for (mode,band),group in frame.groupby(['model','band']):
        error=group.fractional_error.to_numpy();med=np.median(error)
        summary.append(dict(model=mode,band=band,n=len(group),bias=float(med),
                            nmad=float(1.4826*np.median(np.abs(error-med))),
                            median_absolute_error=float(np.median(np.abs(error))),
                            p16=float(np.percentile(error,16)),p84=float(np.percentile(error,84))))
    pd.DataFrame(summary).to_csv(args.run/f'injection_summary{sfx}.csv',index=False)
    (args.run/f'injection_protocol{sfx}.json').write_text(json.dumps(dict(count=args.count,seed=args.seed,weak_vis_half=args.weak_vis,
      renderer='Independent oversampled exponential core+disk; scipy Gaussian PSF convolution',
      encodings='Fresh native-scene encoder evaluation for every noisy injected scene; same protocol as prior training',
      source_positions='Known, fixed across models',noise='Alternating white and correlated (0.65 native pixel kernel); diagonal supplied variance',
      profile_color='Band-dependent core/disk fractions; independent band fluxes',
      scope='Controlled synthetic scenes at heldout tile PSFs/noise levels, not additive injections into real survey backgrounds',
      selection='No truth-dependent model or hyperparameter selection'),indent=2))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':main()
