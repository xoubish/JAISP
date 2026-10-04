"""Frozen encoder evaluated on the actual scene, including injected pixels.

Training and inference both use the same native-grid scene cutouts. This avoids
mixing full-tile cached features with a different injection-time context size.
"""
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F


def windows(feature, positions, size=3):
    off=torch.arange(-(size//2),size//2+1,dtype=positions.dtype,device=positions.device)
    yy,xx=torch.meshgrid(off,off,indexing='ij')
    grid=positions[:,None,None]+torch.stack((xx,yy),-1)
    grid=grid/positions.new_tensor([feature.shape[-1]-1,feature.shape[-2]-1])*2-1
    return F.grid_sample(feature.expand(len(positions),-1,-1,-1),grid,align_corners=True)


def image_features(scene):
    d=scene['bands']['euclid_VIS']
    image=d['image'];variance=d['variance'];valid=d['mask']
    background=image[valid].median()
    snr=torch.where(valid,(image-background)/torch.sqrt(variance.clamp_min(1e-20)),0)
    compressed=torch.asinh(snr/5)
    return windows(compressed[None,None],d['positions'],17).numpy()


class SceneEncoder:
    def __init__(self, checkpoint):
        root=Path(__file__).resolve().parents[3]
        sys.path.insert(0,str(root/'models'))
        from load_foundation import load_foundation
        foundation=load_foundation(str(checkpoint),device=torch.device('cpu'),freeze=True)
        self.encoder=foundation.encoder.eval()
        # Fail if even one encoder parameter was absent; no random frozen features.
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        expected={'encoder.'+k for k in self.encoder.state_dict()}
        missing=expected-set(payload['model'])
        if missing:raise ValueError(f'Incomplete encoder checkpoint: {sorted(missing)[:5]}')

    @torch.no_grad()
    def feature_views(self, scene, bottleneck_window=3, stem_window=3):
        """Separate spatial views; larger windows need a newly trained prior."""
        images={};rms={}
        for band,d in scene['bands'].items():
            valid=d['mask'] & torch.isfinite(d['image']) & torch.isfinite(d['variance']) & (d['variance']>0)
            images[band]=torch.where(valid,d['image'],0)[None,None].float()
            rms[band]=torch.sqrt(torch.where(valid,d['variance'],1))[None,None].float()
        encoded=self.encoder(images,rms)
        bn=encoded['bottleneck']
        vis=images['euclid_VIS'];vrms=rms['euclid_VIS']
        stem=self.encoder.stems['euclid_VIS'](vis,vrms)
        pos=scene['bands']['euclid_VIS']['positions']
        # Pixel-center mapping matches interpolate(..., align_corners=False).
        ratio=pos.new_tensor([bn.shape[-1]/vis.shape[-1],bn.shape[-2]/vis.shape[-2]])
        bn_pos=(pos+.5)*ratio-.5
        out=dict(bottleneck=windows(bn,bn_pos,bottleneck_window),vis_stem=windows(stem,pos,stem_window))
        # CPU callers (mixture prior) expect numpy; GPU callers keep tensors on the device.
        return {k:(v.numpy() if v.device.type=='cpu' else v) for k,v in out.items()}

    def __call__(self, scene):
        views = self.feature_views(scene)
        return np.concatenate((views['bottleneck'], views['vis_stem']), axis=1)
