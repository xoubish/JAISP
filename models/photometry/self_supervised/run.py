"""CPU-friendly all-band pilot. Run as python -m models.photometry.self_supervised.run."""
import argparse
import copy
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
from .core import BANDS, MorphologyHead, ConstantMorphologyHead, fit_scene
from .data import wcs, load_tile, calibrate_psf, make_scenes

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'models'))
from foundation_utils import discover_tile_pairs


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',default='models/photometry/self_supervised/runs/q1_all_bands')
    p.add_argument('--train-tiles',type=int,default=12)
    p.add_argument('--val-tiles',type=int,default=4)
    p.add_argument('--test-tiles',type=int,default=12)
    p.add_argument('--scenes-per-tile',type=int,default=4)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--rebuild',action='store_true')
    p.add_argument('--scenes-from',type=Path,help='Reuse an existing prepared run without recalibrating PSFs or positions')
    p.add_argument('--feature-mode',choices=['foundation','constant'],default='foundation')
    p.add_argument('--astrometry-checkpoint', default='models/checkpoints/latent_position_v11_anchored_v11labels_patchval25/best.pt', help='Compatible anchored v11 checkpoint; empty string keeps detection positions')
    args=p.parse_args()
    torch.set_num_threads(args.threads);torch.manual_seed(42)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cache_dir=ROOT/'data/cached_features_v11_q1'
    labels_path=ROOT/'data/detection_labels/centernet_q1_790_vissep_v11_thresh03.pt'
    labels_payload=torch.load(labels_path,map_location='cpu',weights_only=False)
    labels=labels_payload['labels']
    expected='models/checkpoints/jaisp_v11_q1_soft/checkpoint_best.pt'
    if labels_payload['config']['encoder_ckpt']!=expected:
        raise ValueError('Detection and feature-cache foundation versions disagree')
    scene_file=(args.scenes_from or out)/'scenes.pt'
    if args.scenes_from and (args.rebuild or not scene_file.exists()):
        raise ValueError('--scenes-from requires an existing cache and cannot be combined with --rebuild')
    if scene_file.exists() and not args.rebuild:
        saved=torch.load(scene_file,weights_only=False,map_location='cpu')
        if saved['preparation'] != {k:vars(args)[k] for k in ('train_tiles','val_tiles','test_tiles','scenes_per_tile','astrometry_checkpoint')}:
            raise ValueError('Preparation settings changed; use --rebuild or a new output directory')
        splits=saved['splits'];metadata=saved['metadata']
        if metadata.get('renderer') != 'gauss_legendre3_float64_wls_v1':
            raise ValueError('Renderer/preparation version changed; rebuild the source run')
    else:
        pairs=discover_tile_pairs(str(ROOT/'data/rubin_tiles_all'),str(ROOT/'data/euclid_tiles_all_q1'))
        np.random.default_rng(42).shuffle(pairs)
        selected={s:[] for s in ('train','val','test')}
        desired=dict(train=args.train_tiles,val=args.val_tiles,test=args.test_tiles)
        # Training/test tiles and every scene must be in guarded partitions. Same RA
        # thresholds as the supervised experiment; no source catalog is read.
        b1,b2,guard=53.166483388,53.220111106,.009453496771566816
        for pair in pairs:
            if pair[0] not in labels or not (cache_dir/(pair[0]+'_aug0.pt')).exists():continue
            with np.load(pair[2],allow_pickle=True) as e:
                wc=wcs(e['wcs_VIS'])
                height,width=e['img_VIS'].shape
                ra,_=wc.pixel_to_world_values([0,width-1,0,width-1],[0,0,height-1,height-1])
            split='train' if max(ra)<b1-guard else ('test' if min(ra)>b2+guard else ('val' if b1+guard < float(np.mean(ra)) < b2-guard else None))
            if split and len(selected[split])<desired[split]:selected[split].append(pair)
            if all(len(selected[s])==desired[s] for s in selected):break
        if any(len(selected[s])<desired[s] for s in selected):
            raise ValueError(f'Not enough guarded tiles: { {s:len(v) for s,v in selected.items()} }')
        print('Loading training tiles and calibrating approximate Q1 PSFs',flush=True)
        train_tiles=[load_tile(pair,labels) for pair in selected['train']]
        astrometry=None
        if args.astrometry_checkpoint:
            from .astrometry import FrozenAstrometry
            astrometry=FrozenAstrometry(ROOT,args.astrometry_checkpoint,expected)
            for tile in train_tiles:
                astrometry.apply(tile,cache_dir)
                print('Aligned',tile['name'],flush=True)
        psf=calibrate_psf(train_tiles)
        (out/'psf_calibration.json').write_text(json.dumps(psf,indent=2))
        print('PSF samples:',{b:p['n'] for b,p in psf.items()},flush=True)
        splits={};occupied=[]
        for split in selected:
            splits[split]=[]
            tiles=train_tiles if split=='train' else [load_tile(pair,labels) for pair in selected[split]]
            for tile in tiles:
                if astrometry is not None and split!='train':astrometry.apply(tile,cache_dir)
                bounds={'train':(-np.inf,b1-guard),'val':(b1+guard,b2-guard),'test':(b2+guard,np.inf)}[split]
                scenes=make_scenes(tile,psf,cache_dir,args.scenes_per_tile,occupied,ra_bounds=bounds)
                for scene in scenes:
                    try:
                        fit_scene(scene,scene['covariance'])
                    except ValueError as exc:
                        print('Skip degenerate scene',tile['name'],str(exc),flush=True);continue
                    splits[split].append(scene)
            print(split,len(splits[split]),'scenes',flush=True)
            if not splits[split]:raise ValueError(f'Empty {split} split')
        metadata=dict(bands=BANDS,renderer='gauss_legendre3_float64_wls_v1',foundation_checkpoint=expected,feature_cache=str(cache_dir),
                      detection_cache=str(labels_path),astrometry=args.astrometry_checkpoint or 'Fixed detection centroids',
                      selected_tiles={s:[p[0] for p in v] for s,v in selected.items()},
                      split=dict(ra_boundaries=[b1,b2],guard_deg=guard,whole_scene=True),psf=psf,
                      training_target='Native image Gaussian likelihood; no catalog flux labels',
                      uncertainties='Conditional on fixed shape and approximate PSF; diagonal pixel variance',
                      units='Native image flux units, independent per band; not calibrated microJy',
                      mask='Finite image, positive finite variance; Rubin BAD|SAT|NO_DATA bits excluded per io/ingest_tiles.py',
                      foundation_pretraining='May include downstream heldout sky; downstream head split only',
                      limitations=['Shared Gaussian morphology across bands','Global circular approximate PSF',
                                   'No per-band astrometric correction','No missing-source model',
                                   'Full observed pixels enter frozen features: not blind pixel prediction'])
        torch.save(dict(splits=splits,metadata=metadata,preparation={k:vars(args)[k] for k in ('train_tiles','val_tiles','test_tiles','scenes_per_tile','astrometry_checkpoint')}),out/'scenes.pt')
    metadata=copy.deepcopy(metadata)
    metadata['feature_mode']=args.feature_mode
    metadata['scene_cache']=str(scene_file.resolve())
    mean_features=torch.cat([s['features'] for s in splits['train']]).mean(0,keepdim=True)
    head=MorphologyHead() if args.feature_mode=='foundation' else ConstantMorphologyHead(mean_features)
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01)
    def evaluate(split,model):
        rows=[]
        with torch.no_grad():
            for scene in splits[split]:
                cov=scene['covariance'] if model is None else model(scene['features'],scene['covariance'])
                fits=fit_scene(scene,cov)
                for b,r in fits.items():
                    rows.append((b,float(r['chi2']),r['dof']))
        return {b:sum(c for bb,c,d in rows if bb==b)/sum(d for bb,c,d in rows if bb==b) for b in BANDS}
    before=evaluate('val',None)
    best_score=np.mean(list(before.values()));best=copy.deepcopy(head.state_dict());history=[]
    for epoch in range(args.epochs):
        head.train();losses=[];skipped=0
        for idx in np.random.default_rng(42+epoch).permutation(len(splits['train'])):
            scene=splits['train'][idx]
            optimizer.zero_grad()
            cov=head(scene['features'],scene['covariance'])
            try: fits=fit_scene(scene,cov)
            except ValueError:skipped+=1;continue
            loss=torch.stack([r['loss'] for r in fits.values()]).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            optimizer.step();losses.append(float(loss.detach()))
        head.eval();validation=evaluate('val',head)
        score=float(np.mean(list(validation.values())))
        if score<best_score:best_score=score;best=copy.deepcopy(head.state_dict())
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),val_chi2=validation,skipped=skipped))
        print(f'Epoch {epoch+1}: train {np.mean(losses):.5f}; validation {score:.5f}; best {best_score:.5f}',flush=True)
        (out/'history.json').write_text(json.dumps(history,indent=2))
    head.load_state_dict(best);head.eval()
    torch.save(dict(head_state=best,metadata=metadata,args=vars(args)),out/'head.pt')
    summary={s:dict(scenes=len(splits[s]),before=evaluate(s,None),after=evaluate(s,head)) for s in ('val','test')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    rows=[];covariances={}
    with torch.no_grad():
        for scene_id,scene in enumerate(splits['test']):
            for mode,model in [('image_moments',None),(args.feature_mode,head)]:
                cov=scene['covariance'] if model is None else model(scene['features'],scene['covariance'])
                for b,r in fit_scene(scene,cov).items():
                    covariances[f'{scene_id}/{mode}/{b}']=dict(indices=r['source_indices'],covariance=r['covariance'])
                    for j,idx in enumerate(r['source_indices'].tolist()):
                        ra,dec=scene['sky'][idx]
                        rows.append(dict(scene=scene_id,tile=scene['tile'],source=idx,central=idx==scene['central'],
                                         ra=ra,dec=dec,band=b,model=mode,flux=float(r['flux'][j]),
                                         flux_error=float(r['error'][j]),background=float(r['background']),
                                         reduced_chi2=float(r['chi2']/r['dof']),condition=float(r['condition']),
                                         n_neighbors=len(scene['sky'])-1))
    with (out/'test_fluxes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    torch.save(covariances,out/'test_flux_covariances.pt')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
