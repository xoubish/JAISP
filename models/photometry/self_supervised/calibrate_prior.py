"""Calibrate profile precision on validation residuals, without flux labels."""
from pathlib import Path
import argparse
import copy
import json
import numpy as np
import torch
from .mixture import SCALES
from .run_mixture import evaluate


def profile_precision(prediction,truth):
    radius=np.geomspace(.03,3,30)
    cdf=1-np.exp(-radius[None,:]**2/(2*(SCALES[:,None]**2+.095**2)))
    residual=(prediction-truth)@cdf
    # Uncentered: systematic prediction errors count as uncertainty too.
    second=residual.T@residual/len(residual)
    # Shrinkage and a 3% curve-of-growth floor protect the small calibration set.
    covariance=.8*second+.2*np.diag(np.diag(second))+.03**2*np.eye(len(radius))
    precision=cdf@np.linalg.solve(covariance,cdf.T)
    return (precision+precision.T)/2


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',type=Path,default=Path('models/photometry/self_supervised/runs/q1_mixture_denoise'))
    p.add_argument('--output',type=Path,default=Path('models/photometry/self_supervised/runs/q1_mixture_calibrated'))
    args=p.parse_args();torch.set_num_threads(4);out=args.output;out.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(args.source/'priors.pt',weights_only=False,map_location='cpu')
    augmented=torch.load(args.source/'augmented.pt',weights_only=False,map_location='cpu')['val']
    features={key:np.concatenate([a[key][a['teacher_good']] for a in augmented]) for key in ('image','foundation')}
    targets=np.concatenate([a['teacher_weights'][a['teacher_good']] for a in augmented])
    population=checkpoint['population'];base=np.tile(population,(len(targets),1))
    checkpoint['population_precision']=profile_precision(base,targets)
    for kind,head in checkpoint['heads'].items():
        prediction=base if head['model'] is None else head['fraction']*head['model'].predict(features)+(1-head['fraction'])*base
        head['precision']=profile_precision(prediction,targets)
    checkpoint['metadata'].update(prior_strength=1.,
        precision_calibration='Validation curve-of-growth residual second moments; 20% diagonal shrinkage and 3% CDF floor; no injected truth used',
        calibration_samples=len(targets),calibration_unique_sources=checkpoint['metadata']['independent_teacher_sources']['val'],
        prior_source=str(args.source.resolve()))
    torch.save(checkpoint,out/'priors.pt')
    (out/'metadata.json').write_text(json.dumps(checkpoint['metadata'],indent=2))
    prepared=torch.load(Path(checkpoint['metadata']['prepared_source'])/'prepared.pt',weights_only=False,map_location='cpu')
    print(json.dumps(evaluate(prepared['test'],checkpoint['heads'],population,out,
        population_precision=checkpoint['population_precision'],prior_strength=1.),indent=2),flush=True)

if __name__=='__main__':main()
