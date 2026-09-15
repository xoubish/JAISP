"""Training-only diagnosis of the label purity/recovery trade-off; no new head training."""
from pathlib import Path
import sys,json
import numpy as np,torch
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'models'))
from detection.masks import bright_star_saturation_mask,load_gaia_cache
cat=fits.getdata(ROOT/'data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits')
gaia=load_gaia_cache(str(ROOT/'data/gaia_ecdfs_astrometry_cache.npz'))
vis=torch.load(ROOT/'data/cached_features_v11_q1/pseudo_labels_vis_sep.pt',weights_only=False)['labels']
variants={k:torch.load(HERE/folder/'nir_extra_labels.pt',weights_only=False)['promoted'] for k,folder in [('threshold3','labels'),('threshold4','training_label_threshold4')]}
tids=sorted(variants['threshold4']);out={k:dict(added=0,added_mer_matches=0,nir_recovered=0,nir_total=0) for k in variants}
for tid in tids:
 with np.load(ROOT/'data/euclid_tiles_all_q1'/f'{tid}_euclid.npz',allow_pickle=True) as d:
  h,w=d['img_VIS'].shape;wc=WCS(fits.Header.fromstring(str(d['wcs_VIS'])));mask=bright_star_saturation_mask(d['img_VIS'],str(d['wcs_VIS']),gaia)
 x,y=wc.all_world2pix(cat['ra'],cat['dec'],0);inside=(x>10)&(x<w-10)&(y>10)&(y<h-10)
 inside &= ~mask[np.clip(y.astype(int),0,h-1),np.clip(x.astype(int),0,w-1)]
 nir=inside&(cat['vis_det']==0)&(cat['spurious_flag']!=1)
 refs=np.c_[x[nir],y[nir]];tree=cKDTree(np.c_[x[inside],y[inside]])
 for name,labels in variants.items():
  added=labels[tid]*[w-1,h-1];ix=np.clip(np.rint(added[:,0]).astype(int),0,w-1);iy=np.clip(np.rint(added[:,1]).astype(int),0,h-1);added=added[~mask[iy,ix]]
  combined=np.concatenate((vis[tid][0]*[w-1,h-1],added))
  a=out[name];a['added']+=len(added);a['added_mer_matches']+=int((tree.query(added)[0]<5).sum());a['nir_total']+=len(refs);a['nir_recovered']+=int((cKDTree(combined).query(refs)[0]<5).sum())
for a in out.values():
 a['added_label_mer_match_percent']=100*a['added_mer_matches']/a['added'];a['nir_reference_label_coverage_percent']=100*a['nir_recovered']/a['nir_total']
result=dict(scope='four training tiles only; threshold 4 was tested after seeing the primary head purity trade-off; no new head was trained',tiles=tids,results=out)
(HERE/'label_threshold_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(out,indent=2))
