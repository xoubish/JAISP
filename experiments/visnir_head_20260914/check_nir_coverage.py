"""Diagnostic only: NISP variance coverage at the already-scored MER references."""
from pathlib import Path
import json,sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'models'))
from detection.visnir_eval_experiment import FIELDS
from detection.masks import bright_star_saturation_mask
out={}
for field,cfg in FIELDS.items():
 cat=fits.getdata(ROOT/cfg['catalog'])
 g=np.load(ROOT/cfg['gaia'])
 gaia=dict(ra=g['ra'],dec=g['dec'],g=g['g'] if 'g' in g else g['phot_g_mean_mag'])
 total=valid_all=valid_any=0
 for ep in sorted((ROOT/cfg['euclid']).glob(cfg['glob'])):
  with np.load(ep,allow_pickle=True) as d:
   h,w=d['img_VIS'].shape;wc=WCS(fits.Header.fromstring(str(d['wcs_VIS'])))
   x,y=wc.all_world2pix(cat['ra'],cat['dec'],0)
   mask=bright_star_saturation_mask(d['img_VIS'],str(d['wcs_VIS']),gaia)
   ix=np.clip(x.astype(int),0,w-1);iy=np.clip(y.astype(int),0,h-1)
   ref=(x>=4)&(x<w-4)&(y>=4)&(y<h-4)&~mask[iy,ix]&(cat['spurious_flag']!=1)
   if (ref&(cat['vis_det']==1)).sum()<2:continue
   ref &= cat['vis_det']==0
   ok=[]
   for b in ('Y','J','H'):
    a=d['var_'+b][iy[ref],ix[ref]];im=d['img_'+b][iy[ref],ix[ref]]
    ok.append(np.isfinite(a)&(a>0)&(a<1e20)&np.isfinite(im))
   total+=int(ref.sum());valid_all+=int(np.logical_and.reduce(ok).sum());valid_any+=int(np.logical_or.reduce(ok).sum())
 out[field]=dict(nir_only_references=total,valid_all_three=valid_all,valid_at_least_one=valid_any,valid_all_percent=100*valid_all/total)
 print(field,out[field],flush=True)
(HERE/'nir_coverage.json').write_text(json.dumps(out,indent=2)+'\n')
