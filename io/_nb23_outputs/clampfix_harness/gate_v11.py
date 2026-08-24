"""Step-1 gate: v11 (asinh50) vs v10 (clamp) foundation, inference + linear probes only.

A. Masked-band reconstruction on the 39 held-out val tiles (trainer split, seed 42):
   bright-pixel pearson r per band (each model scored in its own target space; identical
   below S/N 50 where both transforms are linear) + reconstruction error stratified by raw
   per-pixel S/N, scored in raw S/N space via each model's inverse transform.
B. Frozen probes (identical protocol both models, 5-fold ridge):
   - centroid probe: sub-pixel offset of stars from integer-centered 5x5 stem windows,
     split bright (VIS 17.5-19.5) vs faint (VIS 20.5-22.5) -> mas
   - photometry probe: VIS mag from 3x3 bottleneck windows -> pearson r
C. Bright-core feature structure: spatial gradient power of stem features in the core
   vs an annulus, bright stars, v11/v10 ratio (clamped plateau -> flat core features).
"""
import sys, glob, pickle
import numpy as np, pandas as pd, torch
from pathlib import Path

REPO = Path('/home/shemmati/Work/Projects/JAISP')
sys.path.insert(0, str(REPO/'models'))
sys.path.insert(0, str(REPO/'models/astrometry2'))
H = REPO/'io/_nb23_outputs/clampfix_harness'

from foundation_utils import load_tile_data, FrozenEncoder, discover_tile_pairs
from load_foundation import load_foundation
from jaisp_foundation_v10 import decompress_snr

device = torch.device('cuda:0')
MODELS = {'v10': 'models/checkpoints/jaisp_v10_q1_long/checkpoint_best.pt',
          'v11': 'models/checkpoints/jaisp_v11_q1_soft/checkpoint_best.pt'}
BIN_EDGES = (2., 10., 50., 100., 500.)

pairs = discover_tile_pairs(str(REPO/'data/rubin_tiles_all'), str(REPO/'data/euclid_tiles_all_q1'))
rng = np.random.RandomState(42)
val_idx = sorted(rng.permutation(len(pairs)).tolist()[:max(1, int(len(pairs)*0.05))])
val_pairs = [pairs[i] for i in val_idx]
print(f'{len(val_pairs)} held-out val tiles (trainer split reproduced)')

# ---- star sample for probes ----
st = pickle.load(open(H/'state.pkl','rb'))
long, ks, sp = st['long'], st['ks'], st['sp']
cosd, s_r = st['cosd'], st['s_r']
sti = ks.join(sp[['mag_vis','plike_prob','gaia']])
sti['is_star'] = (sti.plike_prob>0.7)|sti.gaia
blr = long[(long.snr>20)&long.band.isin(['u','g','r','i','z','y'])].copy()
blr['lra'] = blr.ra - s_r*blr.rx/3.6e6/cosd
blr['ldec'] = blr.dec - s_r*blr.ry/3.6e6
lab = blr.groupby('src')[['lra','ldec']].median()
tile1 = long.groupby('src')['tile'].first()
stars = sti[sti.is_star].join(lab).join(tile1.rename('tile')).dropna(subset=['lra','tile'])
stars['cls'] = np.where((stars.mag_vis>17.5)&(stars.mag_vis<19.5),'bright',
                np.where((stars.mag_vis>20.5)&(stars.mag_vis<22.5),'faint',''))
stars = stars[stars.cls!='']
tile_counts = stars.groupby('tile').size().sort_values(ascending=False)
probe_tiles = tile_counts.index[:80].tolist()
stars = stars[stars.tile.isin(probe_tiles)]
print(f'probe stars: {(stars.cls=="bright").sum()} bright, {(stars.cls=="faint").sum()} faint '
      f'on {len(probe_tiles)} tiles')

from astrometry2.source_matching import safe_header_from_card_string
from astropy.wcs import WCS

def ridge_cv(X, y, lam=1e-2, k=5):
    X = np.asarray(X, np.float64); y = np.asarray(y, np.float64)
    mu, sd = X.mean(0), X.std(0)+1e-9
    X = (X-mu)/sd
    n = len(y); idx = np.arange(n); np.random.RandomState(0).shuffle(idx)
    pred = np.zeros_like(y)
    for f in range(k):
        te = idx[f::k]; tr = np.setdiff1d(idx, te)
        A = X[tr].T@X[tr] + lam*len(tr)*np.eye(X.shape[1])
        w = np.linalg.solve(A, X[tr].T@(y[tr]-y[tr].mean()))
        pred[te] = X[te]@w + y[tr].mean()
    return pred

results = {}
for tag, ck in MODELS.items():
    model = load_foundation(str(REPO/ck), device=device, freeze=True)
    mode = getattr(model, 'input_compression', 'clamp')
    enc = FrozenEncoder(model).to(device).eval()

    # ---- A. masked-band reconstruction on val tiles ----
    rec_rows = []
    for tile_id, rp, ep in val_pairs:
        try: img_t, rms_t, vis_hw, vis_wcs = load_tile_data(rp, ep, device)
        except Exception: continue
        bands = list(img_t.keys())
        for b in bands:
            ctx = {k: img_t[k] for k in bands if k != b}
            crx = {k: rms_t[k] for k in bands if k != b}
            if not ctx: continue
            with torch.no_grad():
                out = model(ctx, crx, b, img_t[b], rms_t[b])
            snr = (img_t[b]/(rms_t[b]+1e-10)).flatten().float()
            pr = out['pred'].flatten().float(); tn = out['target_norm'].flatten().float()
            ok = torch.isfinite(snr)&torch.isfinite(pr)&torch.isfinite(tn)
            snr, pr, tn = snr[ok], pr[ok], tn[ok]
            m2 = snr > 2
            r = float(np.corrcoef(pr[m2].cpu(), tn[m2].cpu())[0,1]) if m2.sum()>50 else np.nan
            row = dict(tile=tile_id, band=b, r_bright=r)
            err_raw = (decompress_snr(pr, mode)-snr).abs()
            edges = (-np.inf,)+BIN_EDGES+(np.inf,)
            for i in range(len(edges)-1):
                m = (snr>edges[i])&(snr<=edges[i+1])
                row[f'e_{edges[i]:g}_{edges[i+1]:g}'] = float(err_raw[m].mean()) if m.any() else np.nan
                row[f'n_{edges[i]:g}_{edges[i+1]:g}'] = int(m.sum())
            rec_rows.append(row)
        del img_t, rms_t; torch.cuda.empty_cache()
    rec = pd.DataFrame(rec_rows)

    # ---- B/C. probes on stem/bottleneck features ----
    Xs, ys, cls_l, Xb, mags, flat = [], [], [], [], [], []
    for t in probe_tiles:
        rp = REPO/f'data/rubin_tiles_all/{t}.npz'; ep = REPO/f'data/euclid_tiles_all_q1/{t}_euclid.npz'
        if not (rp.exists() and ep.exists()): continue
        try: img_t, rms_t, vis_hw, vis_wcs = load_tile_data(str(rp), str(ep), device)
        except Exception: continue
        with torch.no_grad():
            f = enc.encode_tile(img_t, rms_t)
        stem = f['vis_stem'][0].cpu().numpy(); bn = f['bottleneck'][0].cpu().numpy()
        sx = bn.shape[-1]/stem.shape[-1]*1.0
        for sid, row in stars[stars.tile==t].iterrows():
            x, y = vis_wcs.world_to_pixel_values(row.lra, row.ldec)
            x, y = float(x), float(y)
            xi, yi = int(round(x)), int(round(y))
            if not (10<=xi<stem.shape[-1]-10 and 10<=yi<stem.shape[-2]-10): continue
            w5 = stem[:, yi-2:yi+3, xi-2:xi+3]
            Xs.append(w5.ravel()); ys.append((x-xi, y-yi)); cls_l.append(row.cls)
            bxi, byi = int(round(xi*bn.shape[-1]/stem.shape[-1])), int(round(yi*bn.shape[-2]/stem.shape[-2]))
            if 2<=bxi<bn.shape[-1]-2 and 2<=byi<bn.shape[-2]-2:
                Xb.append(bn[:, byi-1:byi+2, bxi-1:bxi+2].ravel()); mags.append(row.mag_vis)
            else:
                Xb.append(None); mags.append(np.nan)
            if row.cls=='bright':
                w17 = stem[:, yi-8:yi+9, xi-8:xi+9]
                gy, gx = np.gradient(w17, axis=(1,2))
                g = np.sqrt(gy**2+gx**2).mean(0)
                core = g[6:11,6:11].mean(); ann = np.concatenate([g[:3].ravel(), g[-3:].ravel()]).mean()
                flat.append(core/max(ann,1e-9))
        del img_t, rms_t, f; torch.cuda.empty_cache()
    Xs = np.array(Xs); ys = np.array(ys); cls_a = np.array(cls_l)
    pred = np.stack([ridge_cv(Xs, ys[:,0]), ridge_cv(Xs, ys[:,1])], 1)
    res_mas = np.hypot(*(100.*(pred-ys)).T)
    okb = np.array([(Xb[i] is not None) and np.isfinite(mags[i]) for i in range(len(Xb))])
    Xb2 = np.array([Xb[i] for i in range(len(Xb)) if okb[i]])
    magv = np.array(mags)[okb]
    r_mag = float(np.corrcoef(ridge_cv(Xb2, magv), magv)[0,1]) if len(magv)>50 else np.nan

    results[tag] = dict(rec=rec, res_mas=res_mas, cls=cls_a, r_mag=r_mag,
                        flat=np.array(flat), mode=mode)
    print(f'\n===== {tag} ({mode}) =====')
    print(f'A. bright-pixel recon r: mean {rec.r_bright.mean():.4f} (per-band spread '
          f'{rec.groupby("band").r_bright.mean().std():.3f})')
    for k in [c for c in rec.columns if c.startswith('e_')]:
        n = rec[k.replace('e_','n_')].sum()
        print(f'   raw-space |err| {k[2:]:>12s}: {rec[k].mean():8.3f}   (n={n/1e6:.1f}M px)')
    for c in ['bright','faint']:
        print(f'B. centroid probe {c:6s}: median {np.median(res_mas[cls_a==c]):6.1f} mas (N={(cls_a==c).sum()})')
    print(f'B. photometry probe r(mag): {r_mag:.3f}')
    print(f'C. core/annulus feature-gradient ratio (bright stars): median {np.median(np.array(flat)):.3f}')
    del model, enc; torch.cuda.empty_cache()

pickle.dump(results, open(H/'gate_v11_results.pkl','wb'))
print('\nsaved gate_v11_results.pkl')
