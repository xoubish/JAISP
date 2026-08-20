"""Evaluate a retrained head against the clamp-failure testbench.

Usage: python eval_fix.py <head_checkpoint> <tag>

Outputs (scratch/<tag>_evalfix.pkl):
  - flux-sweep curves on the same 22 stars as the nb23 mechanism section
  - bright-star population displacement (input at label) for 214 VIS<19.5 stars
  - faint-end: displacement at label + jitter-recovery (50 mas) for ~300 faint sources
"""
import sys, pickle
import numpy as np, pandas as pd, torch
from pathlib import Path

REPO = Path('/home/shemmati/Work/Projects/JAISP')
sys.path.insert(0, str(REPO/'models'))
sys.path.insert(0, str(REPO/'models/astrometry2'))
SCRATCH = Path('/home/shemmati/Work/Projects/JAISP/io/_nb23_outputs/clampfix_harness')

from foundation_utils import load_tile_data, FrozenEncoder
from load_foundation import load_foundation
from latent_position_head import LatentPositionHead
from latent_position_head_v2 import LatentPositionHeadV2, soft_snr_map
from astrometry2.dataset import local_vis_pixel_to_sky_matrix
from astrometry2.source_matching import safe_header_from_card_string
from astropy.wcs import WCS
from astropy.io.fits import Header

CKPT, TAG = sys.argv[1], sys.argv[2]
DEV_ARG = sys.argv[3] if len(sys.argv) > 3 else 'cuda:0'
device = torch.device(DEV_ARG)

foundation = load_foundation(str(REPO/'models/checkpoints/jaisp_v10_q1_long/checkpoint_best.pt'),
                             device=torch.device('cpu'))
enc = FrozenEncoder(foundation).to(device).eval()
hck = torch.load(CKPT, map_location='cpu', weights_only=False)
cfg = hck['config']; sd = hck['head_state_dict']
is_v2 = any(k.startswith('raw_conv') for k in sd)
cls = LatentPositionHeadV2 if is_v2 else LatentPositionHead
kw = dict(hidden_ch=256, stem_ch=64, bottleneck_out=cfg['bottleneck_out'], stem_out=cfg['stem_out'],
          mlp_hidden=cfg['mlp_hidden'], bottleneck_window=cfg['bottleneck_window'],
          stem_window=cfg['stem_window'], fused_pixel_scale=0.4, vis_pixel_scale=0.1)
if is_v2: kw['raw_out'] = cfg.get('raw_out', 32)
head = cls(**kw).to(device)
head.load_state_dict(sd); head.eval()
print(f'{TAG}: loaded {"V2 (raw side-channel)" if is_v2 else "V1"} head from {CKPT} (epoch {hck["epoch"]})')

st = pickle.load(open(SCRATCH/'state.pkl','rb'))
long, ks, sp = st['long'], st['ks'], st['sp']
cosd, s_r = st['cosd'], st['s_r']
sti = ks.join(sp[['mag_vis','plike_prob','gaia']])
sti['is_star'] = (sti.plike_prob>0.7)|sti.gaia
RUBIN = ['u','g','r','i','z','y']
blr = long[(long.snr>20)&long.band.isin(RUBIN)].copy()
blr['lra'] = blr.ra - s_r*blr.rx/3.6e6/cosd
blr['ldec'] = blr.dec - s_r*blr.ry/3.6e6
lab = blr.groupby('src')[['lra','ldec']].median()
tile1 = long.groupby('src')['tile'].first()
rng = np.random.default_rng(5)

def wmapsf(edata, rdata):
    out={}
    for b,k in zip(['euclid_VIS','euclid_Y','euclid_J','euclid_H'],['VIS','Y','J','H']):
        out[b]=WCS(safe_header_from_card_string(edata[f'wcs_{k}'].item()))
    rw=WCS(Header(rdata['wcs_hdr'].item()))
    for b in RUBIN: out['rubin_'+b]=rw
    return out

def predict(feats, vis_soft, pos, J):
    kw2 = {'vis_soft': vis_soft} if is_v2 else {}
    with torch.no_grad():
        out = head(feats['bottleneck'], feats['vis_stem'], pos, J,
                   feats['fused_hw'], feats['vis_hw'], **kw2)
    return out['pred_offset_arcsec'].cpu().numpy()*1000., np.exp(out['log_sigma'].cpu().numpy())*1000.

# ---------- part 1: flux sweep on the fixed star set ----------
sweep_old = pd.DataFrame({k: np.load(REPO/'io/_nb23_outputs/nb23_flux_sweep.npz', allow_pickle=True)[k]
                          for k in ['src','grp','alpha','peak_snr','pE','pN']})
star_set = sweep_old[['src','grp']].drop_duplicates().set_index('src')['grp'].to_dict()
ALPHAS=[1.0,0.6,0.35,0.2,0.12,0.07,0.04,0.02]
sweep_rows=[]
for sid, grp in star_set.items():
    t=tile1[sid]; ra,dec=lab.loc[sid,'lra'],lab.loc[sid,'ldec']
    rp=REPO/f'data/rubin_tiles_all/{t}.npz'; ep=REPO/f'data/euclid_tiles_all_q1/{t}_euclid.npz'
    edata=np.load(ep,allow_pickle=True); rdata=np.load(rp,allow_pickle=True)
    img_t,rms_t,vis_hw,vis_wcs=load_tile_data(str(rp),str(ep),device)
    wm=wmapsf(edata,rdata)
    mods={}
    for b in img_t:
        w=wm[b]; scale=0.1 if b.startswith('euclid') else 0.2
        x,y=map(float,w.world_to_pixel_values(ra,dec))
        Hh,Ww=img_t[b].shape[-2:]; Rr=int(round(5.0/scale)); xi,yi=int(round(x)),int(round(y))
        if not (Rr<=xi<Ww-Rr and Rr<=yi<Hh-Rr): continue
        yy,xx=np.mgrid[yi-Rr:yi+Rr+1,xi-Rr:xi+Rr+1]
        r=np.hypot(xx-x,yy-y)*scale
        wgt=np.clip((3.0-r)/1.0,0,1); wgt[r<2.0]=1.0
        sl=(slice(yi-Rr,yi+Rr+1),slice(xi-Rr,xi+Rr+1))
        patch=img_t[b][0,0][sl].cpu().numpy()
        bg=float(np.median(patch[(r>3.5)&(r<5.0)]))
        mods[b]=(sl,torch.from_numpy(wgt.astype(np.float32)).to(device),bg,img_t[b][0,0][sl].clone())
    if 'euclid_VIS' not in mods: continue
    vx,vy=map(float,vis_wcs.world_to_pixel_values(ra,dec))
    pos=torch.tensor([[vx,vy]],device=device)
    J=torch.tensor(local_vis_pixel_to_sky_matrix(vis_wcs,np.array([vx,vy])),
                   device=device,dtype=torch.float32).unsqueeze(0)
    for a in ALPHAS:
        for b,(sl,wgt,bg,orig) in mods.items():
            img_t[b][0,0][sl]=orig-(1.0-a)*wgt*(orig-bg)
        with torch.no_grad():
            feats=enc.encode_tile(img_t,rms_t)
            vsoft=soft_snr_map(img_t['euclid_VIS'],rms_t['euclid_VIS']) if is_v2 else None
        po,sg=predict(feats,vsoft,pos,J)
        sl,wgt,bg,orig=mods['euclid_VIS']
        snrp=(img_t['euclid_VIS'][0,0][sl]/(rms_t['euclid_VIS'][0,0][sl]+1e-10)).cpu().numpy()
        sweep_rows.append(dict(src=sid,grp=grp,alpha=a,peak_snr=float(np.nanmax(snrp)),
                               pE=po[0,0],pN=po[0,1],sig=sg[0]))
    for b,(sl,wgt,bg,orig) in mods.items(): img_t[b][0,0][sl]=orig
    del img_t,rms_t; torch.cuda.empty_cache()
sw=pd.DataFrame(sweep_rows)
print(f'sweep done: {sw.src.nunique()} stars')

# ---------- part 2: population — bright stars + faint sources, input at label (+ jitter recovery) ----------
bright_ids=np.intersect1d(sti[sti.is_star&(sti.mag_vis<19.5)].index.values, lab.index)
faint_pool=long[(long.snr>15)&(long.snr<40)&long.band.isin(RUBIN)].src.unique()
faint_ids=np.intersect1d(faint_pool, lab.index)
faint_ids=rng.choice(faint_ids, min(300,len(faint_ids)), replace=False)
targets=pd.concat([pd.Series('bright',index=bright_ids),pd.Series('faint',index=faint_ids)])
targets=targets[~targets.index.duplicated()]
tt=pd.DataFrame({'cls':targets}).join(tile1.rename('tile')).join(lab).dropna()
pop_rows=[]
for t,g in tt.groupby('tile'):
    rp=REPO/f'data/rubin_tiles_all/{t}.npz'; ep=REPO/f'data/euclid_tiles_all_q1/{t}_euclid.npz'
    if not (rp.exists() and ep.exists()): continue
    try: img_t,rms_t,vis_hw,vis_wcs=load_tile_data(str(rp),str(ep),device)
    except Exception: continue
    with torch.no_grad():
        feats=enc.encode_tile(img_t,rms_t)
        vsoft=soft_snr_map(img_t['euclid_VIS'],rms_t['euclid_VIS']) if is_v2 else None
    xs,ys=vis_wcs.world_to_pixel_values(g.lra.values,g.ldec.values)
    ok=(xs>20)&(xs<vis_hw[1]-20)&(ys>20)&(ys<vis_hw[0]-20)
    if ok.sum()==0:
        del img_t,rms_t; continue
    gg=g[ok]; xs,ys=xs[ok],ys[ok]
    pos=torch.tensor(np.stack([xs,ys],1),device=device,dtype=torch.float32)
    Js=torch.tensor(np.stack([local_vis_pixel_to_sky_matrix(vis_wcs,np.array([x_,y_]))
                              for x_,y_ in zip(xs,ys)]),device=device,dtype=torch.float32)
    po,sg=predict(feats,vsoft,pos,Js)
    # jitter recovery: displace input by 50 mas in a random direction; pred should point back
    th=rng.uniform(0,2*np.pi,len(xs))
    jpx=np.stack([0.5*np.cos(th),0.5*np.sin(th)],1)  # 0.5 VIS px = 50 mas
    posj=torch.tensor(np.stack([xs+jpx[:,0],ys+jpx[:,1]],1),device=device,dtype=torch.float32)
    poj,_=predict(feats,vsoft,posj,Js)
    joff=np.stack([(Js.cpu().numpy()[i]@jpx[i])*1000. for i in range(len(xs))])
    rec=np.hypot(poj[:,0]+joff[:,0],poj[:,1]+joff[:,1])   # residual after correcting the jitter
    for i,(sid,row) in enumerate(gg.iterrows()):
        pop_rows.append(dict(src=sid,cls=row.cls,pE=po[i,0],pN=po[i,1],sig=sg[i],jrec=rec[i]))
    del img_t,rms_t; torch.cuda.empty_cache()
pop=pd.DataFrame(pop_rows)
pop['disp']=np.hypot(pop.pE,pop.pN)
print(pop.groupby('cls')[['disp','jrec','sig']].median().round(1))
pickle.dump(dict(sweep=sw,pop=pop,is_v2=is_v2,ckpt=CKPT), open(SCRATCH/f'{TAG}_evalfix.pkl','wb'))
print(f'saved {TAG}_evalfix.pkl')
