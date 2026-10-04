"""Export identical known-flux scenes to NumPy for the isolated Tractor runtime."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .injections import simulate_scene


def export(run):
    run=Path(run);out=run/'tractor_inputs';out.mkdir(exist_ok=True)
    protocol=json.loads((run/'injection_protocol.json').read_text())
    checkpoint=torch.load(run/'priors.pt',weights_only=False,map_location='cpu')
    templates=torch.load(Path(checkpoint['metadata']['source'])/'scenes.pt',weights_only=False,map_location='cpu')['splits']['test']
    ref=pd.read_csv(run/'injections.csv')
    for i in range(protocol['count']):
        seed=protocol['seed']+i
        scene,truth,config=simulate_scene(templates[i%len(templates)],seed,
            weak_vis=protocol['weak_vis_half'] and (seed//2)%2==0)
        values={'sky':scene['sky']}
        for band,d in scene['bands'].items():
            for key,value in d.items():
                values[f'{band}__{key}']=value.numpy() if torch.is_tensor(value) else value
            values[f'{band}__truth']=truth[band]['flux']
            expected=ref[(ref.scene==i)&(ref.band==band)&(ref.model=='foundation')].sort_values('source')
            np.testing.assert_allclose(truth[band]['flux'],expected.truth_flux,rtol=1e-10)
        np.savez_compressed(out/f'scene_{i:04d}.npz',**values)
    print(f'Exported {protocol["count"]} scenes; all regenerated truth fluxes match saved comparison.')

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True)
    export(p.parse_args().run)
