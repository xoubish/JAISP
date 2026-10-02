"""Train and compare image-derived multiscale morphology priors on all bands."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .core import BANDS
from .mixture import (SCALES, dictionary, ellipse_from_pixels, positive_profile_fit,
                      signed_measurement, FeaturePrior, fit_multiband)
from .scene_features import SceneEncoder, image_features


def curve_error(prediction,truth):
    radii=np.geomspace(.03,3,30)
    cdf=1-np.exp(-radii[None,:]**2/(2*(SCALES[:,None]**2+.095**2)))
    return float(np.mean(((prediction-truth)@cdf)**2))


def train_prior(train,val,kind,population):
    x=np.concatenate([x[kind][x['teacher_good']] for x in train])
    y=np.concatenate([x['teacher_weights'][x['teacher_good']] for x in train])
    vx=np.concatenate([x[kind][x['teacher_good']] for x in val])
    vy=np.concatenate([x['teacher_weights'][x['teacher_good']] for x in val])
    best=(curve_error(np.tile(population,(len(vy),1)),vy),None,0.)
    trials=[]
    for components in (4,12):
        for alpha in (1.,10.,100.):
            model=FeaturePrior().fit(x,y,alpha=alpha,components=components)
            predicted=model.predict(vx)
            for fraction in (.25,.5,1.):
                score=curve_error(fraction*predicted+(1-fraction)*population,vy)
                trials.append(dict(components=components,alpha=alpha,fraction=fraction,error=score))
                if score<best[0]:best=(score,model,fraction)
    return dict(model=best[1],fraction=best[2],validation_error=best[0],trials=trials,
                n_train=len(y),n_validation=len(vy))


def predict_prior(item,head,population,kind):
    base=np.tile(population,(len(item['scene']['sky']),1))
    if head['model'] is None:return base
    return head['fraction']*head['model'].predict(item if getattr(head['model'],'grouped',False) else item[kind])+(1-head['fraction'])*base


def evaluate(items,heads,population,output,population_precision=None,prior_strength=10.):
    rows=[];details=[];failures=[]
    for sid,item in enumerate(items):
        scene=item['scene'];banks={};entry={}
        priors={'population':np.tile(population,(len(scene['sky']),1))}
        for kind,head in heads.items():priors[kind]=predict_prior(item,head,population,kind)
        for mode,prior in priors.items():
            try:
                precision=population_precision if mode=='population' else heads[mode].get('precision')
                fits=fit_multiband(scene,prior,banks=banks,strength=prior_strength,prior_precision=precision)
            except (ValueError,RuntimeError) as exc:
                failures.append(dict(scene=sid,model=mode,error=str(exc)));continue
            entry[mode]=fits
            for band,r in fits.items():
                for idx,(ra,dec) in enumerate(scene['sky']):
                    rows.append(dict(scene=sid,source=idx,tile=scene['tile'],ra=ra,dec=dec,
                                     central=idx==scene['central'],model=mode,band=band,
                                     flux=r['flux'][idx],error=r['error'][idx],
                                     footprint=r['footprint'][idx],reduced_chi2=r['reduced_chi2'],
                                     chi2=r['chi2'],dof=r['dof'],condition=r['condition'],
                                     flag='OK_CONDITIONAL' if r['footprint'][idx]>.95 and r['condition']<1e5 else 'PARTIAL_OR_UNSTABLE'))
        details.append(entry)
        print('Evaluated',sid+1,'/',len(items),flush=True)
    pd.DataFrame(rows).to_csv(output/'fluxes.csv',index=False)
    (output/'failures.json').write_text(json.dumps(failures,indent=2))
    torch.save(details,output/'fits.pt')
    frame=pd.DataFrame(rows).drop_duplicates(['scene','model','band'])
    summary={}
    for mode in frame.model.unique():
        summary[mode]={b:float(frame[(frame.model==mode)&(frame.band==b)].chi2.sum()/frame[(frame.model==mode)&(frame.band==b)].dof.sum()) for b in BANDS}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source',type=Path,default=Path('models/photometry/self_supervised/runs/q1_all_bands'))
    parser.add_argument('--output',type=Path,default=Path('models/photometry/self_supervised/runs/q1_mixture'))
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads);torch.manual_seed(7301)
    original=torch.load(args.source/'scenes.pt',weights_only=False,map_location='cpu')
    cache=out/'prepared.pt'
    if cache.exists():prepared=torch.load(cache,weights_only=False,map_location='cpu')
    else:
        encoder=SceneEncoder(original['metadata']['foundation_checkpoint'])
        prepared={}
        for split,scenes in original['splits'].items():
            prepared[split]=[]
            for i,scene in enumerate(scenes):
                item=dict(scene=scene,foundation=encoder(scene),image=image_features(scene))
                if split!='test':
                    d=scene['bands']['euclid_VIS'];bank=dictionary(scene,'euclid_VIS')
                    weights,flux=positive_profile_fit(d,bank)
                    measured=signed_measurement(d,bank,weights)
                    good=(measured['flux']/measured['error']>20)&(measured['footprint']>.98)
                    item.update(teacher_weights=weights,teacher_good=good)
                prepared[split].append(item)
                print('Prepared',split,i+1,'/',len(scenes),flush=True)
        torch.save(prepared,cache)
    targets=np.concatenate([x['teacher_weights'][x['teacher_good']] for x in prepared['train']])
    population=targets.mean(0);population/=population.sum()
    heads={k:train_prior(prepared['train'],prepared['val'],k,population) for k in ('image','foundation')}
    report={k:{key:value for key,value in h.items() if key!='model'} for k,h in heads.items()}
    print(json.dumps(report,indent=2),flush=True)
    (out/'prior_validation.json').write_text(json.dumps(report,indent=2))
    metadata=dict(source=str(args.source.resolve()),feature_protocol='Re-encoded 12-arcsec native scene: 256 bottleneck + 64 VIS stem; identical train/injection protocol',
                  bands=list(BANDS),scales_arcsec=SCALES.tolist(),prior_strength=10.,band_strength=100.,
                  training='Bright image-derived VIS profiles, no catalog flux labels',
                  test_status='Previously inspected spatial holdout: diagnostic, not fresh blind validation',
                  original=original['metadata'])
    torch.save(dict(heads=heads,population=population,metadata=metadata),out/'priors.pt')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(evaluate(prepared['test'],heads,population,out),indent=2),flush=True)

if __name__=='__main__':main()
