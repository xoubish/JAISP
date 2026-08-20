"""Rebuild the minimal upstream state of nb23 (no permutation nulls, no plots)."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

REPO = Path('/home/shemmati/Work/Projects/JAISP')
ARCH = REPO/'models/checkpoints/latent_position_q1_vissep/anchors_centernet_q1_vissep.npz'
d = np.load(ARCH, allow_pickle=True)
BANDS = ['u','g','r','i','z','y','nisp_Y','nisp_J','nisp_H']
RUBIN = ['u','g','r','i','z','y']
rng = np.random.default_rng(42)

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

cosd = np.cos(np.deg2rad(np.median(np.asarray(d['i_dec'], float))))

ent = {}
for b in BANDS:
    ra = np.asarray(d[f'{b}_ra'], float); dec = np.asarray(d[f'{b}_dec'], float)
    raw = np.asarray(d[f'{b}_raw'], float)*1000.; res = np.asarray(d[f'{b}_head_resid'], float)*1000.
    snr = np.asarray(d[f'{b}_snr'], float); tiles = np.asarray(d[f'{b}_tiles'])
    ok = np.isfinite(raw).all(1) & np.isfinite(res).all(1) & np.isfinite(snr) & (snr > 0)
    ra, dec, raw, res, snr, tiles = ra[ok], dec[ok], raw[ok], res[ok], snr[ok], tiles[ok]
    xy = np.column_stack([ra*cosd*3600., dec*3600.])
    lab = cluster(xy, 0.3)
    df = pd.DataFrame(dict(g=lab, ra=ra, dec=dec, rx=raw[:,0], ry=raw[:,1],
                           ex=res[:,0], ey=res[:,1], snr=snr, tile=tiles))
    agg = df.groupby('g').agg(ra=('ra','median'), dec=('dec','median'), rx=('rx','median'),
        ry=('ry','median'), ex=('ex','median'), ey=('ey','median'), snr=('snr','median'),
        ndup=('g','size'), tile=('tile','first')).reset_index(drop=True)
    agg['raw'] = np.hypot(agg.rx, agg.ry); agg['res'] = np.hypot(agg.ex, agg.ey)
    agg['cx'] = agg.rx - agg.ex; agg['cy'] = agg.ry - agg.ey
    ent[b] = agg

long = []
for b in BANDS:
    a = ent[b].copy(); a['band'] = b; long.append(a)
long = pd.concat(long, ignore_index=True)
xy = np.column_stack([long.ra*cosd*3600., long.dec*3600.])
long['src'] = cluster(xy, 0.4)
long = long.sort_values('snr', ascending=False).drop_duplicates(['src','band']).reset_index(drop=True)
src_pos = long.groupby('src')[['ra','dec']].median()

# strong-worsening flags and k counts (cells 11/21, ids only)
bb = long[(long.snr > 20) & (long.band.isin(RUBIN))].copy()
bb['strong'] = (bb.res > 1.5*bb.raw) & (bb.res - bb.raw > 10)
ks = bb.groupby('src').agg(n=('strong','size'), k=('strong','sum')); ks = ks[ks.n >= 2]
rep3_ids = ks[ks.k >= 3].index.values

# morphology + Gaia (cell 13)
from astropy.io import fits
mc = fits.open(REPO/'data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits')[1].data
hs = fits.open(REPO/'data/edf_s_ood/catalogs_compact/mer_q1_ECDFS_Hsize.fits')[1].data
gaia = np.load(REPO/'data/gaia_ecdfs_astrometry_cache.npz', allow_pickle=True)
sp = src_pos.copy()
def attach(cra, cdec, vals, rad=0.5):
    t = cKDTree(np.column_stack([np.asarray(cra,float)*cosd, np.asarray(cdec,float)]))
    dist, idx = t.query(np.column_stack([sp.ra*cosd, sp.dec]), k=1)
    return np.where(dist*3600 < rad, np.asarray(vals,float)[idx], np.nan)
sp['mag_vis'] = attach(mc['ra'], mc['dec'], mc['mag_vis'])
sp['plike_prob'] = attach(hs['ra'], hs['dec'], hs['point_like_prob'])
tg = cKDTree(np.column_stack([np.asarray(gaia['ra'],float)*cosd, np.asarray(gaia['dec'],float)]))
gd, gi = tg.query(np.column_stack([sp.ra*cosd, sp.dec]), k=1)
sp['gaia'] = gd*3600 < 0.5

# sign convention (cell 28)
EPOCH = 2024.83; gaia2 = np.load(REPO/'data/gaia_ecdfs_astrometry_cache.npz', allow_pickle=True)
REF = float(np.asarray(gaia2['ref_epoch']).ravel()[0]); dt = EPOCH - REF
gra = np.asarray(gaia2['ra'],float) + np.asarray(gaia2['pmra'],float)*dt/3.6e6/cosd
gde = np.asarray(gaia2['dec'],float) + np.asarray(gaia2['pmdec'],float)*dt/3.6e6
posn2 = long.groupby('src')[['ra','dec']].median()
gd_, gi_ = tg.query(np.column_stack([posn2.ra*cosd, posn2.dec]), k=1)
posn2['gi'] = np.where(gd_*3600 < 0.5, gi_, -1)
blr = long[(long.snr > 20) & long.band.isin(RUBIN)].copy()
gm = blr[blr.src.map(posn2.gi) >= 0].copy(); gm['gi'] = gm.src.map(posn2.gi)
gm = gm[np.isfinite(gra[gm.gi]) & np.isfinite(gde[gm.gi])]
bx = (gm.ra.values - gra[gm.gi])*cosd*3.6e6; by = (gm.dec.values - gde[gm.gi])*3.6e6
best = None
for s_r in [+1,-1]:
    for cdd in [1.0, cosd]:
        vx = bx - s_r*gm.rx.values/cdd; vy = by - s_r*gm.ry.values
        med = np.nanmedian(np.hypot(vx, vy))
        if best is None or med < best[0]: best = (med, s_r, cdd)
med_vis, s_r, cdd = best
print(f'state ready: {long.src.nunique():,} sources, {len(rep3_ids)} k>=3 offenders, '
      f'sign={s_r:+d} cdd={"cosd" if cdd!=1.0 else "1"} (median VIS-Gaia {med_vis:.1f} mas)')
