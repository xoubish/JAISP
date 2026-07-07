"""OOD detection eval on EDF-S: the production CenterNet+SEP head (trained only on
ECDFS) applied to the held-out EDF-S field with NO retraining, scored against the
EDF-S MER catalogue. Reuses bakeoff_eval's MER-match machinery with EDF-S paths.
Fetches EDF-S Gaia for the bright-star mask (same recipe as ECDFS). Writes
checkpoints/q1_detection/edfs_ood_metrics.json and prints an ECDFS-vs-EDF-S table.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import detection.bakeoff_eval as B
from detection.validation_utils import load_mer

REPO = Path(__file__).resolve().parent.parent.parent
EDFS_EUCLID = REPO / 'data/edf_s_ood/euclid_tiles_edfs_q1'
EDFS_RUBIN  = REPO / 'data/edf_s_ood/rubin_tiles_edfs'
EDFS_MER    = REPO / 'data/edf_s_ood/catalogs_compact/mer_FINAL_q1_TILE102021011_footprint.fits'
GAIA_CACHE  = REPO / 'data/gaia_edfs_cache.npz'
CKPT        = REPO / 'checkpoints/q1_detection/centernet_vis_sep.pt'


def fetch_edfs_gaia():
    if GAIA_CACHE.exists():
        d = np.load(GAIA_CACHE, allow_pickle=True)
        return {'ra': np.asarray(d['ra'], np.float64), 'dec': np.asarray(d['dec'], np.float64),
                'g': np.asarray(d['g'], np.float64)}
    # footprint centre/radius from tile centres
    ras, decs = [], []
    for p in EDFS_EUCLID.glob('tile_*_euclid.npz'):
        z = np.load(p, allow_pickle=True)
        ras.append(float(z['ra_center'])); decs.append(float(z['dec_center']))
    ra0, dec0 = float(np.mean(ras)), float(np.mean(decs))
    rad = float(max(np.ptp(ras) * np.cos(np.deg2rad(dec0)), np.ptp(decs)) / 2 + 0.1)
    print(f'Fetching Gaia around EDF-S ({ra0:.3f},{dec0:.3f}) r={rad:.2f} deg ...', flush=True)
    from astroquery.gaia import Gaia
    q = (f"SELECT ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source "
         f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra0},{dec0},{rad})) "
         f"AND phot_g_mean_mag < 20")
    r = Gaia.launch_job_async(q).get_results()
    gaia = {'ra': np.asarray(r['ra'], np.float64),
            'dec': np.asarray(r['dec'], np.float64),
            'g': np.asarray(r['phot_g_mean_mag'], np.float64)}
    np.savez(GAIA_CACHE, **gaia)
    print(f'  {len(gaia["ra"])} Gaia stars cached -> {GAIA_CACHE.name}')
    return gaia


def eval_footprint(det, stems, mer, gaia, dev):
    """Per-conf completeness + purity, with purity computed BOTH unrestricted and
    restricted to the MER footprint (the EDF-S MER catalogue covers only a thin
    Dec strip; detections outside it are not real false positives). Single forward
    per tile via detect_scored, re-cut at each conf."""
    import numpy as np
    from scipy.spatial import cKDTree
    # MER footprint bbox from the clean catalogue, inset to avoid edge effects
    ra_lo, ra_hi = mer['cRA'].min() + 0.003, mer['cRA'].max() - 0.003
    dc_lo, dc_hi = mer['cDEC'].min() + 0.003, mer['cDEC'].max() - 0.003
    acc = {c: dict(mag=[], hit=[], tot=0, hitour=0, tot_fp=0, hitour_fp=0) for c in B.CONFS}
    import glob
    for stem in stems:
        ep, rp = B.tile_paths(stem, str(B.EUCLID), str(B.RUBIN))
        if not (glob.glob(ep) and glob.glob(rp)):
            continue
        ed = dict(np.load(ep, allow_pickle=True)); rd = dict(np.load(rp, allow_pickle=True))
        images, rms, vh = B.build_inputs(ed, rd); H, W = vh
        vw = B._wcs_vis(ed)
        mask = B.bright_star_saturation_mask(images['euclid_VIS'], str(ed['wcs_VIS']), gaia)
        xy, sc = B.detect_scored(det, images, rms, vh, dev, mask)
        cx, cy = vw.all_world2pix(mer['cRA'], mer['cDEC'], 0)
        ck = (cx >= B.MARGIN) & (cx < W - B.MARGIN) & (cy >= B.MARGIN) & (cy < H - B.MARGIN)
        cxi = np.clip(cx.astype(int), 0, W - 1); cyi = np.clip(cy.astype(int), 0, H - 1)
        ck &= ~mask[cyi, cxi]
        fx, fy = vw.all_world2pix(mer['fRA'], mer['fDEC'], 0)
        fk = (fx >= B.MARGIN) & (fx < W - B.MARGIN) & (fy >= B.MARGIN) & (fy < H - B.MARGIN)
        if ck.sum() < 2:
            continue
        mag_c = mer['cMAG'][ck]; ctree = cKDTree(np.c_[cx[ck], cy[ck]])
        ftree = cKDTree(np.c_[fx[fk], fy[fk]]) if fk.sum() else None
        for conf in B.CONFS:
            keep = sc >= conf; D = xy[keep]
            if len(D) < 1:
                acc[conf]['mag'].append(mag_c); acc[conf]['hit'].append(np.zeros(len(mag_c), bool)); continue
            dt = cKDTree(D)
            d_c, _ = dt.query(np.c_[cx[ck], cy[ck]])
            acc[conf]['mag'].append(mag_c); acc[conf]['hit'].append(d_c < B.RAD_PX)
            # in-footprint mask for the detections
            dra, ddec = vw.all_pix2world(D[:, 0], D[:, 1], 0)
            infp = (dra > ra_lo) & (dra < ra_hi) & (ddec > dc_lo) & (ddec < dc_hi)
            acc[conf]['tot'] += len(D); acc[conf]['tot_fp'] += int(infp.sum())
            if ftree is not None:
                d_o, _ = ftree.query(D); m = d_o < B.RAD_PX
                acc[conf]['hitour'] += int(m.sum()); acc[conf]['hitour_fp'] += int(m[infp].sum())
    rows = {}
    for conf, a in acc.items():
        if not a['mag']:
            continue
        mag = np.concatenate(a['mag']); hit = np.concatenate(a['hit'])
        sel = np.isfinite(mag) & (mag < B.MAGLIM)
        comp = 100 * hit[sel].mean() if sel.sum() else 0.0
        pur_all = 100 * a['hitour'] / max(a['tot'], 1)
        pur_fp = 100 * a['hitour_fp'] / max(a['tot_fp'], 1)
        f1 = 2 * comp * pur_fp / max(comp + pur_fp, 1e-9)
        rows[conf] = dict(completeness=round(comp, 1), purity=round(pur_fp, 1),
                          purity_unrestricted=round(pur_all, 1), f1=round(f1, 1),
                          n_det=a['tot'], n_det_footprint=a['tot_fp'])
    return rows


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B.EUCLID = EDFS_EUCLID; B.RUBIN = EDFS_RUBIN
    mer = load_mer(str(EDFS_MER)); gaia = fetch_edfs_gaia()
    stems = sorted(p.name.replace('_euclid.npz', '') for p in EDFS_EUCLID.glob('tile_*_euclid.npz'))
    print(f'EDF-S OOD: {len(stems)} tiles | MER {EDFS_MER.name} | maglim VIS<{B.MAGLIM} | '
          f'purity restricted to MER footprint | Gaia mask ON', flush=True)
    det = B.load_model('centernet', CKPT, dev)
    rows = eval_footprint(det, stems, mer, gaia, dev)
    best = max(rows, key=lambda c: rows[c]['f1'])
    out = {'field': 'EDF-S', 'n_tiles': len(stems), 'rows': rows, 'best_conf': best, 'best': rows[best]}
    json.dump(out, open(REPO / 'checkpoints/q1_detection/edfs_ood_metrics.json', 'w'), indent=1)
    print('\n=== EDF-S OOD (centernet_vis_sep, NO retraining) ===')
    for c in sorted(rows):
        r = rows[c]; star = ' <-- best F1' if c == best else ''
        print(f'  conf {c:.2f}: comp={r["completeness"]:5.1f} pur(fp)={r["purity"]:5.1f} '
              f'pur(all)={r["purity_unrestricted"]:5.1f} F1={r["f1"]:5.1f} '
              f'(Ndet={r["n_det"]}, in-fp={r["n_det_footprint"]}){star}')
    ec = json.load(open(REPO / 'checkpoints/q1_detection/bakeoff_metrics.json'))['cn_vis_sep']['rows']
    print('\n=== ECDFS (in-dist) vs EDF-S (OOD, footprint-fair) @ conf 0.30 ===')
    print(f'{"field":8s} {"comp":>6s} {"pur":>6s} {"F1":>6s}')
    for lbl, rr in [('ECDFS', ec.get('0.3') or ec.get('0.30')), ('EDF-S', rows.get(0.3) or rows.get(0.30))]:
        if rr: print(f'{lbl:8s} {rr["completeness"]:6.1f} {rr["purity"]:6.1f} {rr["f1"]:6.1f}')
    print('\nsaved -> checkpoints/q1_detection/edfs_ood_metrics.json')


if __name__ == '__main__':
    main()
