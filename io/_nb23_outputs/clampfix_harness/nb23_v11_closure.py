"""nb23 closure test under v11: rerun the bright-worsener census on the v11-plain-head
anchors archive with the EXACT nb23 pipeline, and test the prediction that only the
chance (regression-to-the-mean) population remains: repeat counts at the permutation
null, no k>=3 excess, flat star-brightness ramp, no S/N>~100 upturn.

Writes nb23_v11_closure_results.pkl; the notebook section is spliced from this.
"""
import sys, pickle
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

REPO = Path('/home/shemmati/Work/Projects/JAISP')
H = REPO/'io/_nb23_outputs/clampfix_harness'
ARCH11 = REPO/'models/checkpoints/latent_position_v11_q1_plain/anchors_centernet_v11plain.npz'

def cluster(xy, r):
    t = cKDTree(xy); pairs = t.query_pairs(r, output_type='ndarray')
    parent = np.arange(len(xy))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in pairs:
        ra_, rb_ = find(a), find(b)
        if ra_ != rb_: parent[ra_] = rb_
    lab = np.array([find(i) for i in range(len(xy))])
    _, lab = np.unique(lab, return_inverse=True)
    return lab

def build_tables(arch_path):
    """Exact nb23 pipeline: per-band dedup at 0.3", cross-band clustering at 0.4"."""
    d = np.load(arch_path, allow_pickle=True)
    BANDS = ['u','g','r','i','z','y','nisp_Y','nisp_J','nisp_H']
    cosd = np.cos(np.deg2rad(np.median(np.asarray(d['i_dec'], float))))
    ent = {}
    for b in BANDS:
        ra = np.asarray(d[f'{b}_ra'], float); dec = np.asarray(d[f'{b}_dec'], float)
        raw = np.asarray(d[f'{b}_raw'], float)*1000.; res = np.asarray(d[f'{b}_head_resid'], float)*1000.
        snr = np.asarray(d[f'{b}_snr'], float); tiles = np.asarray(d[f'{b}_tiles'])
        ok = np.isfinite(raw).all(1) & np.isfinite(res).all(1) & np.isfinite(snr) & (snr > 0)
        ra, dec, raw, res, snr, tiles = ra[ok], dec[ok], raw[ok], res[ok], snr[ok], tiles[ok]
        lab = cluster(np.column_stack([ra*cosd*3600., dec*3600.]), 0.3)
        df = pd.DataFrame(dict(g=lab, ra=ra, dec=dec, rx=raw[:,0], ry=raw[:,1],
                               ex=res[:,0], ey=res[:,1], snr=snr, tile=tiles))
        agg = df.groupby('g').agg(ra=('ra','median'), dec=('dec','median'), rx=('rx','median'),
            ry=('ry','median'), ex=('ex','median'), ey=('ey','median'), snr=('snr','median'),
            tile=('tile','first')).reset_index(drop=True)
        agg['raw'] = np.hypot(agg.rx, agg.ry); agg['res'] = np.hypot(agg.ex, agg.ey)
        ent[b] = agg
    parts = []
    for b in BANDS:
        a = ent[b].copy(); a['band'] = b; parts.append(a)
    lng = pd.concat(parts, ignore_index=True)
    lng['src'] = cluster(np.column_stack([lng.ra*cosd*3600., lng.dec*3600.]), 0.4)
    lng = lng.sort_values('snr', ascending=False).drop_duplicates(['src','band']).reset_index(drop=True)
    return lng, cosd

RUBIN = ['u','g','r','i','z','y']

def census(lng, seed_base):
    """Strong-worsening flags + k-repeat permutation null (exact nb23 protocol)."""
    bb = lng[(lng.snr > 20) & (lng.band.isin(RUBIN))].copy()
    bb['strong'] = (bb.res > 1.5*bb.raw) & (bb.res - bb.raw > 10)
    bb['rawbin'] = np.digitize(bb.raw, np.geomspace(2, 250, 10))
    bb['snrbin'] = np.digitize(np.log10(bb.snr), np.linspace(1.3, 2.5, 5))
    ks = bb.groupby('src').agg(n=('strong','size'), k=('strong','sum')); ks = ks[ks.n >= 2]
    obs = {k: int((ks.k >= k).sum()) for k in [2, 3, 4, 5]}
    nullk = {k: [] for k in [2, 3, 4, 5]}
    for it in range(200):
        sh = bb.copy()
        sh['strong'] = sh.groupby(['band','rawbin','snrbin'])['strong'].transform(
            lambda s: s.sample(frac=1, random_state=seed_base + it).values)
        p2 = sh.groupby('src').agg(n=('strong','size'), k=('strong','sum')); p2 = p2[p2.n >= 2]
        for k in [2, 3, 4, 5]: nullk[k].append(int((p2.k >= k).sum()))
    return bb, ks, obs, {k: (float(np.mean(v)), float(np.std(v))) for k, v in nullk.items()}

print('building v11 tables...')
lng11, cosd = build_tables(ARCH11)
print(f'v11 archive: {lng11.src.nunique():,} sources')
bb11, ks11, obs11, null11 = census(lng11, 3000)
print('v11 k-repeat census: observed vs null')
for k in [2,3,4,5]:
    print(f'  k>={k}: {obs11[k]:4d} vs {null11[k][0]:6.1f} +- {null11[k][1]:.1f}')

# star classification (same MER + Gaia attach as nb23)
from astropy.io import fits
mc = fits.open(REPO/'data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits')[1].data
hs = fits.open(REPO/'data/edf_s_ood/catalogs_compact/mer_q1_ECDFS_Hsize.fits')[1].data
gaia = np.load(REPO/'data/gaia_ecdfs_astrometry_cache.npz', allow_pickle=True)
sp11 = lng11.groupby('src')[['ra','dec']].median()
def attach(cra, cdec, vals, rad=0.5):
    t = cKDTree(np.column_stack([np.asarray(cra,float)*cosd, np.asarray(cdec,float)]))
    dist, idx = t.query(np.column_stack([sp11.ra*cosd, sp11.dec]), k=1)
    return np.where(dist*3600 < rad, np.asarray(vals,float)[idx], np.nan)
sp11['mag_vis'] = attach(mc['ra'], mc['dec'], mc['mag_vis'])
sp11['plike_prob'] = attach(hs['ra'], hs['dec'], hs['point_like_prob'])
tg = cKDTree(np.column_stack([np.asarray(gaia['ra'],float)*cosd, np.asarray(gaia['dec'],float)]))
gd, gi = tg.query(np.column_stack([sp11.ra*cosd, sp11.dec]), k=1)
sp11['gaia'] = gd*3600 < 0.5

# star-brightness ramp (k>=3 rate per VIS mag bin, stars vs extended)
kk = ks11.join(sp11[['mag_vis','plike_prob','gaia']])
kk['cls'] = np.where(kk.plike_prob > 0.7, 'star', np.where(kk.plike_prob < 0.3, 'extended', 'ambig'))
kk['off'] = kk.k >= 3
ramp = {}
for cls in ['star','extended']:
    ramp[cls] = []
    for lo, hi in [(17,19),(19,20.5),(20.5,22),(22,24)]:
        m = (kk.cls==cls)&(kk.mag_vis>=lo)&(kk.mag_vis<hi)
        ramp[cls].append((0.5*(lo+hi), float(kk.off[m].mean()) if m.sum() else np.nan, int(m.sum())))
print('\nstar-brightness ramp (k>=3 rate):')
for cls in ['star','extended']:
    print(f'  {cls:9s}:', ['%.1f%%(N=%d)'%(100*r,n) for _,r,n in ramp[cls]])

# offset-vs-S/N curves (the fig-8 upturn check), stars only
sti = kk  # per src
star_ids = set(sp11.index[(sp11.plike_prob>0.7)|(sp11.gaia)])
bl = lng11[lng11.snr>20 & lng11.band.isin(RUBIN)] if False else lng11[(lng11.snr>20)&(lng11.band.isin(RUBIN))]
bl = bl[bl.src.isin(star_ids)]
curves = {}
edges = np.geomspace(20, 500, 8)
bl2 = bl.copy(); bl2['snrb'] = np.digitize(bl2.snr, edges)
g = bl2.groupby('snrb').agg(snr=('snr','median'), raw=('raw','median'), res=('res','median'), n=('res','size'))
print('\nstars offset vs S/N (median raw -> head):')
print(g.round(1).to_string())

pickle.dump(dict(obs=obs11, null=null11, ramp=ramp, snr_curve=g,
                 n_src=int(lng11.src.nunique()), n_bright_multi=int(len(ks11))),
            open(H/'nb23_v11_closure_results.pkl','wb'))
print('\nsaved nb23_v11_closure_results.pkl')
