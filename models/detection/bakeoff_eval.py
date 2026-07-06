"""Q1 detection bake-off eval (no injection): pick the label source + head by
MER-match F1 on the PATCH-DISJOINT held-out set (patch 25), with the Gaia +
saturation mask applied to both detections and the MER reference.

For each model: one forward pass per tile at a low conf floor (records peak
scores), then the threshold is swept in numpy. Per conf we compute, against the
clean Q1 MER catalogue (vis_det & !spurious, 0.5" match, masked regions dropped):
  completeness (VIS < maglim), purity (vs full MER), F1.
Winner = max F1 at its own optimal threshold.

Usage:
  PYTHONPATH=models python models/detection/bakeoff_eval.py
"""
from __future__ import annotations
import glob, json
import numpy as np
import torch
from pathlib import Path
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detection.validation_utils import (
    build_inputs, _wcs_vis, tile_paths, load_mer, PXE, RUBIN_BANDS, EUCLID_BANDS)
from detection.masks import load_gaia_cache, bright_star_saturation_mask
from detection.centernet_detector import CenterNetDetector
from detection.stem_centernet_detector import StemCenterNetDetector
from detection.detector import JAISPEncoderWrapper
from load_foundation import load_foundation

REPO = Path(__file__).resolve().parent.parent.parent
ENC = REPO / 'models/checkpoints/jaisp_v10_q1_long/checkpoint_best.pt'
MER = REPO / 'data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits'
EUCLID = REPO / 'data/euclid_tiles_all_q1'
RUBIN = REPO / 'data/rubin_tiles_all'
GAIA = REPO / 'data/gaia_ecdfs_astrometry_cache.npz'
OUTD = REPO / 'checkpoints/q1_detection'
MODELS = [
    ('cn_vis_peak', 'centernet', OUTD / 'centernet_vis_peak.pt'),
    ('cn_vis_sep',  'centernet', OUTD / 'centernet_vis_sep.pt'),
    ('cn_mer',      'centernet', OUTD / 'centernet_mer.pt'),
    ('stem_mer',    'stem',      OUTD / 'stem_mer.pt'),
]
CONFS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MAGLIM = 24.5          # completeness reference depth (matches paper working point)
RAD_PX = 0.5 / PXE     # 0.5" match radius in VIS px
MARGIN = 4


def load_model(kind, ckpt, device):
    foundation = load_foundation(str(ENC), device=torch.device('cpu'), freeze=True)
    if kind == 'centernet':
        enc = JAISPEncoderWrapper(foundation, freeze=True).to(device).eval()
        return CenterNetDetector.load(str(ckpt), encoder=enc, device=device).eval()
    return StemCenterNetDetector.load(str(ckpt), foundation, device=device).eval()


@torch.no_grad()
def detect_scored(det, images, rms, vh, device, mask):
    """One forward pass; return (xy_px [N,2], scores [N]) at a low conf floor."""
    im = {b: torch.from_numpy(images[b][None, None].copy()).to(device) for b in images}
    rm = {b: torch.from_numpy(rms[b][None, None].copy()).to(device) for b in rms}
    res = det.predict(im, rm, conf_threshold=0.10, tile_hw=vh, nms_kernel=7, artifact_mask=mask)
    c = res['centroids'].cpu().numpy(); s = res['scores'].cpu().numpy()
    H, W = vh
    xy = np.c_[c[:, 0] * (W - 1), c[:, 1] * (H - 1)] if len(c) else np.zeros((0, 2))
    return xy, s


def eval_model(det, stems, mer, gaia, device):
    """Return dict conf -> (mag[], hit[], our_tot, our_hit) accumulated over tiles."""
    acc = {c: {'mag': [], 'hit': [], 'tot': 0, 'hit_our': 0} for c in CONFS}
    for stem in stems:
        ep, rp = tile_paths(stem, str(EUCLID), str(RUBIN))
        if not (glob.glob(ep) and glob.glob(rp)):
            continue
        ed = dict(np.load(ep, allow_pickle=True)); rd = dict(np.load(rp, allow_pickle=True))
        images, rms, vh = build_inputs(ed, rd); H, W = vh
        vw = _wcs_vis(ed)
        mask = bright_star_saturation_mask(images['euclid_VIS'], str(ed['wcs_VIS']), gaia)
        xy, sc = detect_scored(det, images, rms, vh, device, mask)
        # clean-MER completeness ref (in-frame, NOT under mask)
        cx, cy = vw.all_world2pix(mer['cRA'], mer['cDEC'], 0)
        ck = (cx >= MARGIN) & (cx < W - MARGIN) & (cy >= MARGIN) & (cy < H - MARGIN)
        cxi = np.clip(cx.astype(int), 0, W - 1); cyi = np.clip(cy.astype(int), 0, H - 1)
        ck &= ~mask[cyi, cxi]
        # full-MER for purity
        fx, fy = vw.all_world2pix(mer['fRA'], mer['fDEC'], 0)
        fk = (fx >= MARGIN) & (fx < W - MARGIN) & (fy >= MARGIN) & (fy < H - MARGIN)
        if ck.sum() < 2:
            continue
        mag_c = mer['cMAG'][ck]; ctree = cKDTree(np.c_[cx[ck], cy[ck]])
        ftree = cKDTree(np.c_[fx[fk], fy[fk]]) if fk.sum() else None
        for conf in CONFS:
            keep = sc >= conf
            D = xy[keep]
            if len(D) < 1:
                acc[conf]['mag'].append(mag_c); acc[conf]['hit'].append(np.zeros(len(mag_c), bool))
                continue
            dt = cKDTree(D)
            d_c, _ = dt.query(np.c_[cx[ck], cy[ck]])
            acc[conf]['mag'].append(mag_c); acc[conf]['hit'].append(d_c < RAD_PX)
            acc[conf]['tot'] += len(D)
            if ftree is not None:
                d_o, _ = ftree.query(D); acc[conf]['hit_our'] += int((d_o < RAD_PX).sum())
    return acc


def summarize(acc):
    rows = {}
    for conf, a in acc.items():
        if not a['mag']:
            continue
        mag = np.concatenate(a['mag']); hit = np.concatenate(a['hit'])
        sel = np.isfinite(mag) & (mag < MAGLIM)
        comp = 100 * hit[sel].mean() if sel.sum() else 0.0
        pur = 100 * a['hit_our'] / max(a['tot'], 1)
        f1 = 2 * comp * pur / max(comp + pur, 1e-9)
        rows[conf] = dict(completeness=round(comp, 1), purity=round(pur, 1), f1=round(f1, 1),
                          n_det=a['tot'])
    return rows


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mer = load_mer(str(MER)); gaia = load_gaia_cache(str(GAIA))
    stems = sorted(p.name.replace('_euclid.npz', '')
                   for p in EUCLID.glob('tile_*_patch_25_euclid.npz'))
    print(f'{len(stems)} held-out patch-25 tiles | maglim VIS<{MAGLIM} | match {RAD_PX:.0f}px | Gaia mask ON')
    results = {}
    for name, kind, ckpt in MODELS:
        if not ckpt.exists():
            print(f'[skip] {name}: no checkpoint'); continue
        print(f'\n=== {name} ({kind}) ===', flush=True)
        det = load_model(kind, ckpt, device)
        rows = summarize(eval_model(det, stems, mer, gaia, device))
        best = max(rows, key=lambda c: rows[c]['f1']) if rows else None
        results[name] = dict(rows=rows, best_conf=best, best=rows.get(best))
        for conf in CONFS:
            if conf in rows:
                r = rows[conf]; star = ' <-- best F1' if conf == best else ''
                print(f'  conf {conf:.2f}: comp={r["completeness"]:5.1f}  pur={r["purity"]:5.1f}  '
                      f'F1={r["f1"]:5.1f}  (Ndet={r["n_det"]}){star}')
        del det
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    out = OUTD / 'bakeoff_metrics.json'
    json.dump(results, open(out, 'w'), indent=1)
    print(f'\nsaved -> {out}')
    print('\n=== WINNER TABLE (at each model\'s F1-optimal threshold) ===')
    print(f'{"model":16s} {"conf":>5s} {"comp":>6s} {"pur":>6s} {"F1":>6s}')
    for name, r in results.items():
        b = r.get('best')
        if b:
            print(f'{name:16s} {r["best_conf"]:5.2f} {b["completeness"]:6.1f} {b["purity"]:6.1f} {b["f1"]:6.1f}')


if __name__ == '__main__':
    main()
