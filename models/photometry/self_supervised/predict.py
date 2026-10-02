"""Reusable all-band inference entry point for the fitted mixture head."""
from pathlib import Path
import numpy as np
import torch
from .core import BANDS
from .mixture import SCALES,fit_multiband
from .scene_features import SceneEncoder,image_features
from .run_mixture import predict_prior


class MixturePhotometry:
    def __init__(self,checkpoint,mode='foundation'):
        self.checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if mode not in ('population','image','foundation'):raise ValueError(mode)
        if not np.array_equal(self.checkpoint['metadata']['scales_arcsec'],SCALES):
            raise ValueError('Checkpoint and renderer size dictionaries differ')
        self.mode=mode
        self.encoder=None
        if mode=='foundation':
            path=Path(self.checkpoint['metadata']['original']['foundation_checkpoint'])
            if not path.is_absolute():path=Path(__file__).resolve().parents[3]/path
            self.encoder=SceneEncoder(path)

    def __call__(self,scene):
        if set(scene['bands'])!=set(BANDS):raise ValueError('All ten native bands are required')
        n=len(scene['sky'])
        for band,d in scene['bands'].items():
            if len(d['positions'])!=n:raise ValueError(f'{band}: inconsistent source list')
            if d['image'].shape!=d['variance'].shape or d['image'].shape!=d['mask'].shape:
                raise ValueError(f'{band}: image/variance/mask shapes disagree')
        if self.mode=='population':prior=np.tile(self.checkpoint['population'],(n,1))
        else:
            item=dict(scene=scene,image=image_features(scene))
            if self.encoder is not None:item['foundation']=self.encoder(scene)
            prior=predict_prior(item,self.checkpoint['heads'][self.mode],self.checkpoint['population'],self.mode)
        precision=self.checkpoint.get('population_precision') if self.mode=='population' else self.checkpoint['heads'][self.mode].get('precision')
        return fit_multiband(scene,prior,prior_precision=precision,
            strength=self.checkpoint['metadata']['prior_strength'],
            band_strength=self.checkpoint['metadata']['band_strength'])
