import sys,json
from pathlib import Path
sys.path.insert(0,'models')
import numpy as np,torch
from scipy.spatial import cKDTree
from detection.centernet_detector import CenterNetDetector
from detection.visnir_eval_experiment import predict_features
p=Path('experiments/visnir_head_20260914');device=torch.device('cuda:0');torch.set_num_threads(4)
labels=torch.load(p/'labels/nir_extra_labels.pt',weights_only=False)['promoted']
old=torch.load('data/cached_features_v11_q1/pseudo_labels_vis_sep.pt',weights_only=False)['labels']
paths={'baseline':Path('checkpoints/q1_detection_v11/centernet_vis_sep.pt'),'vis_control':p/'vis_control/epoch_02.pt','visnir':p/'visnir/epoch_02.pt'}
models={k:CenterNetDetector.load(str(v),None,device).eval() for k,v in paths.items()}
acc={k:{'old_vis':[0,0],'added_nisp':[0,0]} for k in models}
for tid in sorted(labels)[::170][:4]:
 features=torch.load(Path('data/cached_features_v11_q1')/f'{tid}_aug0.pt',weights_only=True)['features'][None].to(device)
 for k,m in models.items():
  xy,sc=predict_features(m,features,None,(1084,1084));xy=xy[sc>=.3]
  for label,y in [('old_vis',old[tid][0]),('added_nisp',labels[tid])]:
   hits=cKDTree(xy).query(y*1083)[0]<5 if len(xy) else np.zeros(len(y),bool)
   acc[k][label][0]+=int(hits.sum());acc[k][label][1]+=len(y)
print(json.dumps(acc,indent=2),flush=True)
(p/'training_fit_epoch02.json').write_text(json.dumps({'scope':'four training tiles; diagnostic only, not held-out performance','counts':acc},indent=2)+'\n')
