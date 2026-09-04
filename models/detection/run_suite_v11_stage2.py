"""Stage 2 of the v11 detection suite: deep + subthresh injections and the
mer_recovery product (with a v10 reproduction run to validate the reconstructed
protocol; the original producer did not survive).

Usage: python run_suite_v11_stage2.py {deep|subthresh|merrec_v10|merrec_v11}
"""
import sys, json, glob
from pathlib import Path
import numpy as np

REPO = Path('/home/shemmati/Work/Projects/JAISP')
sys.path.insert(0, str(REPO / 'models'))

V11 = REPO / 'checkpoints/q1_detection_v11'
PROD = REPO / 'checkpoints/q1_detection'
ENC11 = REPO / 'models/checkpoints/jaisp_v11_q1_soft/checkpoint_best.pt'
ENC10 = REPO / 'models/checkpoints/jaisp_v10_q1_long/checkpoint_best.pt'

which = sys.argv[1]

if which in ('deep', 'subthresh', 'deepstem', 'extended'):
    import detection.inject_eval as ie
    ie.ENC = ENC11
    ie.OUTD = V11
    if which == 'deepstem':
        ie.MODELS = [('stem_mer', 'stem', V11 / 'stem_mer.pt')]
        sys.argv = ['inject_eval.py', '--rvis', '30', '--donor-conc', '0.65',
                    '--donor-faint', '22.5', '--mags', '27.0,27.5', '--tag', '_r30_star_deep']
    elif which == 'extended':
        # legacy extended-donor protocol (no concentration cut) -> tag '_r30'
        ie.MODELS = [('cn_vis_sep', 'centernet', V11 / 'centernet_vis_sep.pt'),
                     ('stem_mer', 'stem', V11 / 'stem_mer.pt')]
        sys.argv = ['inject_eval.py', '--rvis', '30', '--donor-faint', '22.5']
    else:
        # BOTH models: inject_eval dumps only the models it runs, so a single-model
        # rerun on the same tag SILENTLY DROPS the other model from the metrics json
        # (cache keeps everything; this bit us 2026-09-02).
        ie.MODELS = [('cn_vis_sep', 'centernet', V11 / 'centernet_vis_sep.pt'),
                     ('stem_mer', 'stem', V11 / 'stem_mer.pt')]
        mags = '27.0,27.5' if which == 'deep' else '28.0,28.5,29.0,35.0'
        sys.argv = ['inject_eval.py', '--rvis', '30', '--donor-conc', '0.65',
                    '--donor-faint', '22.5', '--mags', mags, '--tag', f'_r30_star_{which}']
    ie.main()
    raise SystemExit

# ---- mer_recovery reconstruction ----
import torch
from scipy.spatial import cKDTree
import detection.bakeoff_eval as be
from detection.validation_utils import build_inputs, _wcs_vis, tile_paths, load_mer, PXE, run_detect
from detection.validation_utils import completeness_curve

tag = which.split('_')[1]           # v10 | v11
be.ENC = ENC10 if tag == 'v10' else ENC11
ckpt = (PROD if tag == 'v10' else V11) / 'centernet_vis_sep.pt'
outp = (V11 / ('mer_recovery_v10check.json' if tag == 'v10' else 'mer_recovery.json'))
ref = json.load(open(PROD / 'mer_recovery.json'))
medges = np.array(ref['medges']); snr_x = np.array(ref['snr_x'])
# geometric bin edges around the reference centers
se = np.sqrt(snr_x[1:] * snr_x[:-1])
snr_edges = np.concatenate([[snr_x[0]**2 / se[0]], se, [snr_x[-1]**2 / se[-1]]])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mer = load_mer(str(REPO / 'data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits'))
det = be.load_model('centernet', ckpt, device)
EUCLID = REPO / 'data/euclid_tiles_all_q1'; RUBIN = REPO / 'data/rubin_tiles_all'
stems = sorted(p.name.replace('_euclid.npz', '')
               for p in EUCLID.glob('tile_*_patch_25_euclid.npz'))
print(f'{tag}: {len(stems)} tiles, ckpt={ckpt.name}')

CONF, MARGIN = 0.30, 4
rpx = 0.5 / PXE
mag_all, hit_all, snr_all = [], [], []
for si, stem in enumerate(stems):
    ep, rp = tile_paths(stem, str(EUCLID), str(RUBIN))
    if not (glob.glob(ep) and glob.glob(rp)):
        continue
    ed = dict(np.load(ep, allow_pickle=True)); rd = dict(np.load(rp, allow_pickle=True))
    images, rms, vh = build_inputs(ed, rd); H, W = vh
    vw = _wcs_vis(ed)
    D = run_detect(det, images, rms, vh, device, CONF)
    cx, cy = vw.all_world2pix(mer['cRA'], mer['cDEC'], 0)
    ck = (cx >= MARGIN) & (cx < W - MARGIN) & (cy >= MARGIN) & (cy < H - MARGIN)
    if ck.sum() < 2 or len(D) < 2:
        continue
    d_c, _ = cKDTree(D).query(np.c_[cx[ck], cy[ck]])
    vis = np.nan_to_num(np.asarray(ed['img_VIS'], np.float32))
    vrms = np.sqrt(np.clip(np.asarray(ed['var_VIS'], np.float32), 1e-12, None))
    # peak S/N in a 3x3 box at each clean MER position
    xs = np.clip(np.round(cx[ck]).astype(int), 1, W - 2)
    ys = np.clip(np.round(cy[ck]).astype(int), 1, H - 2)
    # Peak S/N (3x3 max over local rms), the astrometry-side convention. The ORIGINAL
    # mer_recovery producer did not survive and its S/N definition is not recoverable
    # (rec_comp reproduces the stored v10 values to 0.1%; the S/N axis re-derived here
    # reads ~20-50% lower than the lost definition, a monotonic relabeling only).
    snr = np.array([vis[y-1:y+2, x-1:x+2].max() / max(vrms[y, x], 1e-9)
                    for x, y in zip(xs, ys)])
    mag_all.append(mer['cMAG'][ck]); hit_all.append(d_c < rpx); snr_all.append(snr)
    if (si + 1) % 25 == 0:
        print(f'  {si+1}/{len(stems)} tiles', flush=True)

mag = np.concatenate(mag_all); hit = np.concatenate(hit_all); snr = np.concatenate(snr_all)
fin = np.isfinite(mag)
mag, hit, snr = mag[fin], hit[fin], snr[fin]
# recovery on equal-count bins (12) so every point carries the same statistics;
# the histogram stays on the fixed reference medges (quantile-binned hist is flat)
_inr = (mag >= medges[0]) & (mag <= medges[-1])
qedges = np.unique(np.quantile(mag[_inr], np.linspace(0, 1, 13)))
cen, comp, _ = completeness_curve(mag, hit, qedges)
snr_comp = [100 * hit[(snr >= a) & (snr < b)].mean() if ((snr >= a) & (snr < b)).sum() >= 20 else np.nan
            for a, b in zip(snr_edges[:-1], snr_edges[1:])]
hist, _ = np.histogram(mag, bins=medges)
# scale hist to the reference normalization (data-only quantity)
scale = np.array(ref['mer_hist']).max() / max(hist.max(), 1)
out = dict(medges=medges.tolist(), rec_mag=cen.tolist(), rec_comp=comp.tolist(),
           snr_x=snr_x.tolist(), snr_comp=[float(x) for x in snr_comp],
           mer_hist=(hist * scale).tolist())
json.dump(out, open(outp, 'w'), indent=1)
print(f'saved -> {outp}')
print('rec_comp     :', [round(c, 1) for c in comp])
print('ref rec_comp :', [round(c, 1) for c in ref['rec_comp']])
print('snr_comp     :', [round(float(x), 1) for x in snr_comp])
print('ref snr_comp :', [round(c, 1) for c in ref['snr_comp']])
