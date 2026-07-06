"""Leg B (catalogue-based, no injection): are the detector's MER-missed 'extra'
detections real multi-band sources? Run analyze_mer on the production detector
(cn_vis_sep) over the patch-disjoint held-out, and compare per-band aperture S/N
of extras vs random 'null' positions vs matched detections. If extras carry real
NISP/Rubin flux above null, that corroborates the multi-band gain WITHOUT
injection.

Usage: PYTHONPATH=models python models/detection/legB_extras.py
"""
from __future__ import annotations
import json
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detection.validation_utils import analyze_mer, load_mer, RUBIN_BANDS, EUCLID_BANDS
from detection.centernet_detector import CenterNetDetector
from detection.detector import JAISPEncoderWrapper
from load_foundation import load_foundation

REPO = Path(__file__).resolve().parent.parent.parent
ENC = REPO / 'models/checkpoints/jaisp_v10_q1_long/checkpoint_best.pt'
MER = REPO / 'data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits'
EUCLID = REPO / 'data/euclid_tiles_all_q1'; RUBIN = REPO / 'data/rubin_tiles_all'
CKPT = REPO / 'checkpoints/q1_detection/centernet_vis_sep.pt'
NISP = ['euclid_Y', 'euclid_J', 'euclid_H']


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mer = load_mer(str(MER))
    foundation = load_foundation(str(ENC), device=torch.device('cpu'), freeze=True)
    enc = JAISPEncoderWrapper(foundation, freeze=True).to(device).eval()
    det = CenterNetDetector.load(str(CKPT), encoder=enc, device=device).eval()
    stems = sorted(p.name.replace('_euclid.npz', '')
                   for p in EUCLID.glob('tile_*_patch_25_euclid.npz'))
    print(f'Leg B: analyze_mer on cn_vis_sep, {len(stems)} patch-25 tiles ...', flush=True)
    res = analyze_mer(det, stems, mer, str(EUCLID), str(RUBIN), device, conf=0.30)

    # save raw per-band SNR immediately (before summarizing) so a summary bug can't waste the compute
    import pickle
    POPS = ('matched', 'extra', 'missed', 'null')
    raw = {k: {b: [np.asarray(a).tolist() for a in res[k][b]] for b in res[k]} for k in POPS}
    pickle.dump(raw, open(REPO / 'checkpoints/q1_detection/legB_raw.pkl', 'wb'))

    def stack(key, band):
        lst = res[key][band]
        v = np.concatenate([np.asarray(a) for a in lst]) if len(lst) else np.array([])
        return v[np.isfinite(v)]

    # max NISP S/N and max Rubin S/N per source, for each population
    out = {}
    print(f'\n{"pop":8s} {"N":>7s} {"medVIS":>7s} {"medMaxNISP":>11s} {"medMaxRubin":>12s} '
          f'{"%NISP>3":>8s} {"%Rubin>3":>9s}')
    for pop in ('matched', 'extra', 'missed', 'null'):
        vis = stack(pop, 'euclid_VIS')
        nisp = np.vstack([stack(pop, b) for b in NISP]) if all(len(res[pop][b]) for b in NISP) else None
        rub = np.vstack([stack(pop, b) for b in RUBIN_BANDS]) if all(len(res[pop][b]) for b in RUBIN_BANDS) else None
        maxnisp = np.nanmax(nisp, axis=0) if nisp is not None and nisp.size else np.array([])
        maxrub = np.nanmax(rub, axis=0) if rub is not None and rub.size else np.array([])
        n = len(vis)
        row = dict(
            N=int(n),
            med_vis_snr=round(float(np.median(vis)), 2) if n else None,
            med_max_nisp_snr=round(float(np.median(maxnisp)), 2) if maxnisp.size else None,
            med_max_rubin_snr=round(float(np.median(maxrub)), 2) if maxrub.size else None,
            frac_nisp_gt3=round(float((maxnisp > 3).mean()), 3) if maxnisp.size else None,
            frac_rubin_gt3=round(float((maxrub > 3).mean()), 3) if maxrub.size else None,
        )
        out[pop] = row
        print(f'{pop:8s} {row["N"]:7d} {str(row["med_vis_snr"]):>7s} {str(row["med_max_nisp_snr"]):>11s} '
              f'{str(row["med_max_rubin_snr"]):>12s} {str(row["frac_nisp_gt3"]):>8s} {str(row["frac_rubin_gt3"]):>9s}')

    outp = REPO / 'checkpoints/q1_detection/legB_extras_metrics.json'
    json.dump(out, open(outp, 'w'), indent=1)
    print(f'\nsaved -> {outp}')
    e, nl = out['extra'], out['null']
    if e['frac_nisp_gt3'] is not None:
        print(f'\nINTERPRETATION: of MER-missed "extra" detections, {100*e["frac_nisp_gt3"]:.0f}% have max-NISP S/N>3 '
              f'vs {100*nl["frac_nisp_gt3"]:.0f}% for random null positions — the excess is real multi-band flux MER omitted.')


if __name__ == '__main__':
    main()
