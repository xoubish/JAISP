"""Image-only and foundation-assisted priors trained with label-free noise degradation."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from .mixture import GroupedPrior
from .scene_features import SceneEncoder,image_features
from .run_mixture import curve_error,evaluate


def train(train,val,population,kind):
    def collect(items):
        x={key:np.concatenate([item[key][item['teacher_good']] for item in items]) for key in ('image','foundation')}
        y=np.concatenate([item['teacher_weights'][item['teacher_good']] for item in items])
        return x,y
    x,y=collect(train);vx,vy=collect(val)
    base=curve_error(np.tile(population,(len(vy),1)),vy)
    best=(base,None,0.);trials=[]
    for components in ((6,12,24,40,64) if kind=='image' else (6,12,24)):
        for alpha in (1.,10.,100.):
            model=GroupedPrior().fit(x,y,kind=kind,alpha=alpha,components=components)
            prediction=model.predict(vx)
            for fraction in (.25,.5,1.):
                error=curve_error(fraction*prediction+(1-fraction)*population,vy)
                trials.append(dict(components=components,alpha=alpha,fraction=fraction,error=error))
                if error<best[0]:best=(error,model,fraction)
    return dict(model=best[1],fraction=best[2],validation_error=best[0],population_error=base,
                n_train_samples=len(y),n_validation_samples=len(vy),trials=trials)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source',type=Path,default=Path('models/photometry/self_supervised/runs/q1_mixture'))
    parser.add_argument('--output',type=Path,default=Path('models/photometry/self_supervised/runs/q1_mixture_denoise'))
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(8121)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    original=torch.load(args.source/'prepared.pt',map_location='cpu',weights_only=False)
    first=torch.load(args.source/'priors.pt',map_location='cpu',weights_only=False)
    if (out/'augmented.pt').exists():data=torch.load(out/'augmented.pt',map_location='cpu',weights_only=False)
    else:
        encoder=SceneEncoder(first['metadata']['original']['foundation_checkpoint'])
        data={};rng=np.random.default_rng(91873)
        for split in ('train','val'):
            data[split]=list(original[split])
            for index,item in enumerate(original[split]):
                for factor in (2.,4.):
                    scene=copy.deepcopy(item['scene'])
                    d=scene['bands']['euclid_VIS'];valid=d['mask']
                    rms=torch.sqrt(torch.where(valid,d['variance'],0))
                    extra=torch.tensor(rng.normal(size=d['image'].shape),dtype=torch.float32)*rms*np.sqrt(factor**2-1)
                    d['image']=d['image']+extra;d['variance']=d['variance']*factor**2
                    data[split].append(dict(scene=scene,foundation=encoder(scene),image=image_features(scene),
                        teacher_weights=item['teacher_weights'],teacher_good=item['teacher_good'],noise_factor=factor))
                print('Noise augmentation',split,index+1,'/',len(original[split]),flush=True)
        torch.save(data,out/'augmented.pt')
    population=first['population']
    heads={kind:train(data['train'],data['val'],population,kind) for kind in ('image','foundation')}
    metadata=copy.deepcopy(first['metadata'])
    metadata.update(prior_training='Bright image-fitted VIS profiles; independent added VIS noise factors 1,2,4; no catalog flux labels',
                    feature_protocol='Separate training-fitted PCAs for raw VIS pixels, VIS foundation stem, and multiband bottleneck',
                    independent_teacher_sources=dict(train=sum(int(x['teacher_good'].sum()) for x in original['train']),
                                                     val=sum(int(x['teacher_good'].sum()) for x in original['val'])),
                    prepared_source=str(args.source.resolve()))
    torch.save(dict(heads=heads,population=population,metadata=metadata),out/'priors.pt')
    report={k:{key:value for key,value in h.items() if key!='model'} for k,h in heads.items()}
    (out/'prior_validation.json').write_text(json.dumps(report,indent=2))
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    print(json.dumps(evaluate(original['test'],heads,population,out),indent=2),flush=True)

if __name__=='__main__':main()
