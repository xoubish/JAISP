"""Deep-truth detection purity/completeness on the HLF GOODS-S overlap strip.

The ECDFS Q1 footprint overlaps the southern edge of the Hubble Legacy Fields
GOODS-S region (Whitaker et al. 2019, v2.1; GEMS-depth ACS there: F606W~28.3,
F850LP~27.1 at 5-sigma), 0.5-1.7 mag below the head's VIS~26.6 point-source
depth. Every detection in the overlap can therefore be classified against a
catalog deeper than the detector, giving TRUE purity rather than the
MER-agreement lower bound, and an in-situ completeness against deep truth.

Per tile (center Dec > -28.03, >=50 in-frame HLF sources):
  - one scored forward pass of the production detector (cn_vis_sep) at a 0.10
    conf floor, Gaia+saturation mask ON (identical to bakeoff_eval).
  - HLF projected to VIS px; per-tile astrometric shift measured from
    score>=0.5 detections matched to HLF within 1" and applied to HLF.
  - per-detection record: score, coverage flag, distance to nearest HLF /
    full-MER / clean-MER source, matched-HLF magnitudes, and 8 dithered-HLF
    distances (+-5-7" offsets) for chance-match correction.
  - per-HLF-source record (covered, in-frame): F606W/F850LP/F160W mags, S/G,
    Use1, distance to nearest detection per conf threshold (completeness).
  - per-band aperture S/N (per_band_snr) for HLF-unmatched detections
    (score>=0.15), a matched control subsample, and empty-sky null positions.
  - covered unmasked area from a 24x24 grid (for FP surface density).

Coverage rule: a position counts as HLF-covered iff its 5th-nearest in-frame
HLF source lies within 10" (catalog density in the strip ~360/arcmin^2, so
real coverage passes easily; mosaic edges fail).

Caches per tile under checkpoints/q1_detection/deep_truth_cache/; writes
deep_truth_records.npz (concatenated arrays) and deep_truth_purity.json
(aggregate summary) for the paper_figures notebook.

Usage: PYTHONPATH=models python models/detection/deep_truth_purity.py
"""
from __future__ import annotations
import glob, json
import numpy as np
import torch
from pathlib import Path
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.wcs import WCS

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detection.validation_utils import (
    build_inputs, _wcs_vis, tile_paths, load_mer, per_band_snr, PXE,
    RUBIN_BANDS, EUCLID_BANDS)
from detection.masks import load_gaia_cache, bright_star_saturation_mask
from detection.bakeoff_eval import load_model, detect_scored

REPO = Path(__file__).resolve().parent.parent.parent
ENC = REPO / 'models/checkpoints/jaisp_v10_q1_long/checkpoint_best.pt'
MER = REPO / 'data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits'
HLF = REPO / 'data/external/hlf_goodss_v21_strip.fits'
EUCLID = REPO / 'data/euclid_tiles_all_q1'
RUBIN = REPO / 'data/rubin_tiles_all'
GAIA = REPO / 'data/gaia_ecdfs_astrometry_cache.npz'
OUTD = REPO / 'checkpoints/q1_detection'
CACHE = OUTD / 'deep_truth_cache'
CONFS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
FLOOR, MARGIN = 0.10, 4
COV_K, COV_AS = 5, 10.0          # coverage: 5th-nn HLF source within 10"
SNR_SCORE_MIN = 0.15             # unmatched dets to S/N-profile
DITHERS_AS = [(7, 0), (-7, 0), (0, 7), (0, -7), (5, 5), (5, -5), (-5, 5), (-5, -5)]
DEC_MIN, MIN_HLF = -28.03, 50
N_NULL, N_MATCHED_CTRL = 60, 60


def load_hlf(path):
    t = fits.open(path)[1].data
    ra = np.asarray(t['RAGdeg'], float); dec = np.asarray(t['DEGdeg'], float)
    bad = ~(np.isfinite(ra) & np.isfinite(dec))
    ra[bad] = np.asarray(t['RAJ2000'], float)[bad]
    dec[bad] = np.asarray(t['DEJ2000'], float)[bad]
    def mag(col):
        f = np.asarray(t[col], float)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(f > 0, 25.0 - 2.5 * np.log10(f), np.nan)
    return dict(ra=ra, dec=dec, m606=mag('F606W'), m850=mag('F850LP'),
                m160=mag('F160W'), sg=np.asarray(t['S/G'], float),
                use1=np.asarray(t['Use1'], float))


def tile_center_dec(ep):
    ed = np.load(ep, allow_pickle=True)
    from detection.validation_utils import safe_header_from_card_string
    w = WCS(safe_header_from_card_string(ed['wcs_VIS'].item()))
    return float(w.all_pix2world(541.5, 541.5, 0)[1])


def process_tile(det, stem, hlf, mer, gaia, device):
    ep, rp = tile_paths(stem, str(EUCLID), str(RUBIN))
    if not (glob.glob(ep) and glob.glob(rp)):
        return None
    ed = dict(np.load(ep, allow_pickle=True)); rd = dict(np.load(rp, allow_pickle=True))
    images, rms, vh = build_inputs(ed, rd); H, W = vh
    vw = _wcs_vis(ed)
    hx, hy = vw.all_world2pix(hlf['ra'], hlf['dec'], 0)
    hin = (hx >= MARGIN) & (hx < W - MARGIN) & (hy >= MARGIN) & (hy < H - MARGIN)
    if hin.sum() < MIN_HLF:
        return {}                                  # outside HLF footprint: cached skip
    mask = bright_star_saturation_mask(images['euclid_VIS'], str(ed['wcs_VIS']), gaia)
    xy, sc = detect_scored(det, images, rms, vh, device, mask)
    if len(xy) < 2:
        return {}
    hxy = np.c_[hx[hin], hy[hin]]
    hmask = mask[np.clip(hy[hin].astype(int), 0, H - 1), np.clip(hx[hin].astype(int), 0, W - 1)]
    # per-tile astrometric shift: bright dets vs nearest HLF < 1"
    t_h = cKDTree(hxy)
    strong = sc >= 0.5
    d_s, j_s = t_h.query(xy[strong])
    pair = d_s < 1.0 / PXE
    shift = (np.median(hxy[j_s[pair]] - xy[strong][pair], axis=0)
             if pair.sum() >= 10 else np.zeros(2))
    hxy_c = hxy - shift                            # HLF moved onto detector frame
    t_h = cKDTree(hxy_c)
    # coverage via 5th-nn distance (evaluated on the shifted catalog)
    cov_px = COV_AS / PXE
    def covered(pts):
        if len(pts) == 0:
            return np.zeros(0, bool)
        d5 = t_h.query(pts, k=COV_K)[0][:, -1]
        return d5 < cov_px
    # detections: in-frame, covered
    din = (xy[:, 0] >= MARGIN) & (xy[:, 0] < W - MARGIN) & (xy[:, 1] >= MARGIN) & (xy[:, 1] < H - MARGIN)
    xy, sc = xy[din], sc[din]
    dcov = covered(xy)
    d_h, j_h = t_h.query(xy)
    dith = np.stack([cKDTree(hxy_c + np.array(o) / PXE).query(xy)[0]
                     for o in DITHERS_AS], axis=1)
    fx, fy = vw.all_world2pix(mer['fRA'], mer['fDEC'], 0)
    fk = (fx >= MARGIN) & (fx < W - MARGIN) & (fy >= MARGIN) & (fy < H - MARGIN)
    d_f = cKDTree(np.c_[fx[fk], fy[fk]]).query(xy)[0] if fk.sum() >= 2 else np.full(len(xy), 1e9)
    cx, cy = vw.all_world2pix(mer['cRA'], mer['cDEC'], 0)
    ck = (cx >= MARGIN) & (cx < W - MARGIN) & (cy >= MARGIN) & (cy < H - MARGIN)
    d_c = cKDTree(np.c_[cx[ck], cy[ck]]).query(xy)[0] if ck.sum() >= 2 else np.full(len(xy), 1e9)
    hidx = np.where(hin)[0][j_h]                   # matched-HLF properties per det
    det_rec = dict(score=sc, x=xy[:, 0], y=xy[:, 1], cov=dcov,
                   d_hlf=d_h * PXE, d_mer_full=d_f * PXE, d_mer_clean=d_c * PXE,
                   dith=dith * PXE, m606=hlf['m606'][hidx], m850=hlf['m850'][hidx],
                   m160=hlf['m160'][hidx], sg=hlf['sg'][hidx])
    # HLF-source completeness records (covered, unmasked)
    scov = covered(hxy_c) & ~hmask
    src = hxy_c[scov]; sidx = np.where(hin)[0][scov]
    d_per_conf = np.full((len(src), len(CONFS)), 1e9)
    for ci, conf in enumerate(CONFS):
        keep = sc >= conf
        if keep.sum() >= 1:
            d_per_conf[:, ci] = cKDTree(xy[keep]).query(src)[0]
    src_rec = dict(m606=hlf['m606'][sidx], m850=hlf['m850'][sidx], m160=hlf['m160'][sidx],
                   sg=hlf['sg'][sidx], use1=hlf['use1'][sidx], d_det=d_per_conf * PXE)
    # covered unmasked area (arcmin^2) from a 24x24 grid
    gy, gx = np.mgrid[MARGIN:H - MARGIN:24j, MARGIN:W - MARGIN:24j]
    gpts = np.c_[gx.ravel(), gy.ravel()]
    gcov = covered(gpts) & ~mask[gpts[:, 1].astype(int), gpts[:, 0].astype(int)]
    area = gcov.mean() * (W - 2 * MARGIN) * (H - 2 * MARGIN) * PXE**2 / 3600.0
    # per-band aperture S/N: unmatched dets, matched controls, empty-sky nulls
    rng = np.random.default_rng(7)
    rad_px = 0.5 / PXE
    un = dcov & (sc >= SNR_SCORE_MIN) & (d_h > rad_px)
    ma = np.where(dcov & (sc >= SNR_SCORE_MIN) & (d_h < rad_px))[0]
    ma = rng.permutation(ma)[:N_MATCHED_CTRL]
    nulls, tries = [], 0
    t_d = cKDTree(xy)
    while len(nulls) < N_NULL and tries < 4000:
        tries += 1
        p = rng.uniform(MARGIN, [W - MARGIN, H - MARGIN])
        if (covered(p[None])[0] and not mask[int(p[1]), int(p[0])]
                and t_h.query(p)[0] > 2.0 / PXE and t_d.query(p)[0] > 2.0 / PXE):
            nulls.append(p)
    nulls = np.array(nulls) if nulls else np.zeros((0, 2))
    snr = {}
    for tag, pts in (('un', xy[un]), ('ma', xy[ma]), ('nu', nulls)):
        s = per_band_snr(ed, rd, vw, pts, ap_arcsec=0.3)
        snr.update({f'snr_{tag}_{b}': s[b] for b in list(RUBIN_BANDS) + list(EUCLID_BANDS)})
    snr['snr_un_score'] = sc[un]; snr['snr_ma_score'] = sc[ma]
    out = dict(area=np.array([area]), shift=shift * PXE, n_hlf=np.array([int(hin.sum())]))
    out.update({f'det_{k}': np.asarray(v, np.float32) for k, v in det_rec.items()})
    out.update({f'src_{k}': np.asarray(v, np.float32) for k, v in src_rec.items()})
    out.update({k: np.asarray(v, np.float32) for k, v in snr.items()})
    return out


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CACHE.mkdir(parents=True, exist_ok=True)
    hlf = load_hlf(HLF); mer = load_mer(str(MER)); gaia = load_gaia_cache(str(GAIA))
    stems = sorted(p.name.replace('_euclid.npz', '') for p in EUCLID.glob('tile_*_euclid.npz'))
    stems = [s for s in stems if tile_center_dec(f'{EUCLID}/{s}_euclid.npz') > DEC_MIN]
    print(f'{len(stems)} candidate strip tiles (Dec > {DEC_MIN})')
    det = None
    for i, stem in enumerate(stems):
        cpath = CACHE / f'{stem}.npz'
        if cpath.exists():
            continue
        if det is None:
            det = load_model('centernet', OUTD / 'centernet_vis_sep.pt', device)
        r = process_tile(det, stem, hlf, mer, gaia, device)
        np.savez_compressed(cpath, **(r or {'empty': np.zeros(1)}))
        print(f'[{i + 1}/{len(stems)}] {stem}' + (' (no HLF coverage)' if not r else
              f'  ndet={len(r["det_score"])} nsrc={len(r["src_m850"])} area={r["area"][0]:.2f}\''
              f' shift=({r["shift"][0]:+.2f}",{r["shift"][1]:+.2f}")'), flush=True)
    # ---- aggregate ----
    recs, keys = [], None
    for stem in stems:
        z = np.load(CACHE / f'{stem}.npz')
        if 'empty' in z or 'det_score' not in z:
            continue
        d = {k: z[k] for k in z.files}
        d['stem'] = stem; d['heldout'] = 'patch_25' in stem
        recs.append(d)
    print(f'{len(recs)} tiles with HLF coverage ({sum(r["heldout"] for r in recs)} held-out patch_25)')
    flat = {}
    for k in recs[0]:
        if k in ('stem', 'heldout'):
            continue
        flat[k] = np.concatenate([np.atleast_1d(r[k]) for r in recs]) if recs[0][k].ndim <= 1 \
            else np.concatenate([r[k] for r in recs], axis=0)
    for tag, arrname in (('det', 'det_score'), ('src', 'src_m850')):
        flat[f'{tag}_tile'] = np.concatenate(
            [np.full(len(np.atleast_1d(r[arrname])), i) for i, r in enumerate(recs)])
        flat[f'{tag}_heldout'] = np.concatenate(
            [np.full(len(np.atleast_1d(r[arrname])), r['heldout'], bool) for r in recs])
    flat['tile_stems'] = np.array([r['stem'] for r in recs])
    flat['tile_heldout'] = np.array([r['heldout'] for r in recs])
    np.savez_compressed(OUTD / 'deep_truth_records.npz', **flat)
    # summary: purity per conf per group, 0.5" radius, chance-corrected
    rad = 0.5
    summ = {}
    for gname, gsel in (('heldout', flat['det_heldout']), ('train', ~flat['det_heldout']),
                        ('all', np.ones(len(flat['det_score']), bool))):
        rows = {}
        for ci, conf in enumerate(CONFS):
            s = gsel & flat['det_cov'].astype(bool) & (flat['det_score'] >= conf)
            n = int(s.sum())
            if n == 0:
                continue
            m_h = float((flat['det_d_hlf'][s] < rad).mean())
            ch = float((flat['det_dith'][s] < rad).mean())
            m_f = float((flat['det_d_mer_full'][s] < rad).mean())
            rows[f'{conf:.2f}'] = dict(
                n_det=n, matched_hlf=round(100 * m_h, 2), chance=round(100 * ch, 2),
                purity_corr=round(100 * (m_h - ch) / max(1 - ch, 1e-9), 2),
                matched_mer_full=round(100 * m_f, 2))
        summ[gname] = rows
    summ['total_area_arcmin2'] = round(float(flat['area'].sum()), 2)
    summ['median_shift_as'] = [round(float(np.median(flat['shift'][0::2])), 3),
                               round(float(np.median(flat['shift'][1::2])), 3)]
    json.dump(summ, open(OUTD / 'deep_truth_purity.json', 'w'), indent=1)
    print(json.dumps({g: summ[g].get('0.30') for g in ('heldout', 'train', 'all')}, indent=1))
    print('saved ->', OUTD / 'deep_truth_purity.json', 'and deep_truth_records.npz')


if __name__ == '__main__':
    main()
