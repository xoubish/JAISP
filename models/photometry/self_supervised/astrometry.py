"""Apply the existing, compatible v11 anchored canonical-position head frozen."""
from pathlib import Path
import numpy as np
import torch
from .data import jacobian


class FrozenAstrometry:
    def __init__(self, root, checkpoint, foundation_checkpoint):
        from jaisp_foundation_v10 import BandStem
        from astrometry2.train_latent_position_v2 import AnchoredHead
        payload=torch.load(root/checkpoint,map_location='cpu',weights_only=False)
        cfg=payload['config']
        if cfg.get('foundation_checkpoint') != foundation_checkpoint or cfg.get('variant') != 'anchored':
            raise ValueError('Expected anchored astrometry trained with the same foundation')
        foundation=torch.load(root/foundation_checkpoint,map_location='cpu',weights_only=False)
        fc=foundation['config']
        self.stem=BandStem(fc['stem_ch'],compression=fc['input_compression'])
        prefix='encoder.stems.euclid_VIS.'
        self.stem.load_state_dict({k[len(prefix):]:v for k,v in foundation['model'].items() if k.startswith(prefix)},strict=True)
        self.head=AnchoredHead(hidden_ch=fc['hidden_ch'],stem_ch=fc['stem_ch'],
                               bottleneck_window=11,stem_window=17,
                               fused_pixel_scale=fc['fused_pixel_scale_arcsec'],vis_pixel_scale=.1)
        self.head.load_state_dict(payload['head_state_dict'],strict=True)
        for module in (self.stem,self.head):
            module.eval()
            for parameter in module.parameters():parameter.requires_grad_(False)

    @torch.no_grad()
    def apply(self,tile,cache_dir):
        payload=torch.load(Path(cache_dir)/(tile['name']+'_aug0.pt'),map_location='cpu',weights_only=False)
        bn=payload['features'].float()
        if bn.ndim==3:bn=bn[None]
        im,var,wc=tile['bands']['euclid_VIS']
        valid=np.isfinite(im)&np.isfinite(var)&(var>0)
        image=torch.tensor(np.where(valid,im,0))[None,None]
        rms=torch.tensor(np.sqrt(np.where(valid,var,1)))[None,None]
        stem=self.stem(image,rms)
        positions=torch.tensor(tile['xy'],dtype=torch.float32)
        matrix=torch.tensor(np.stack([jacobian(wc,xy) for xy in tile['xy']]),dtype=torch.float32)
        result=self.head(bn,stem,positions,matrix,bn.shape[-2:],image.shape[-2:],vis_img=image)
        offset=torch.stack((result['dx_px'],result['dy_px']),1).numpy()
        if not np.isfinite(offset).all():raise ValueError('Nonfinite astrometry output')
        tile['xy']=tile['xy']+offset
        tile['sky']=np.column_stack(wc.pixel_to_world_values(*tile['xy'].T))
        tile['astrometry_median_shift_px']=float(np.median(np.linalg.norm(offset,axis=1)))
