"""Leg A: source-recycling injection depth on the patch-disjoint held-out set,
for the ready Q1 detectors (cn_vis_sep, cn_mer, stem_mer). Gives the 50%
point-source depth per mode (all-10 / VIS-only / NISP-only), the multi-band
fusion gain (all - vis), and settles CenterNet-vs-Stem. conf=0.30 uniform for
comparability (the all-vs-vis GAP is robust to conf). Curves saved for Fig 6.

Usage: PYTHONPATH=models python models/detection/inject_eval.py
"""
from __future__ import annotations
import json
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detection.validation_utils import eval_injection, rec_to_curve, load_mer
from detection.centernet_detector import CenterNetDetector
from detection.stem_centernet_detector import StemCenterNetDetector
from detection.detector import JAISPEncoderWrapper
from load_foundation import load_foundation

REPO = Path(__file__).resolve().parent.parent.parent
ENC = REPO / 'models/checkpoints/jaisp_v10_q1_long/checkpoint_best.pt'
MER = REPO / 'data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits'
EUCLID = REPO / 'data/euclid_tiles_all_q1'; RUBIN = REPO / 'data/rubin_tiles_all'
OUTD = REPO / 'checkpoints/q1_detection'
MODELS = [('cn_vis_sep', 'centernet', OUTD / 'centernet_vis_sep.pt'),
          ('cn_mer', 'centernet', OUTD / 'centernet_mer.pt'),
          ('stem_mer', 'stem', OUTD / 'stem_mer.pt')]
MAGS = (22.5, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5)
N_TILES = 200         # all patch-25 held-out (~108); firm-up run
CONF = 0.30


def load_model(kind, ckpt, device):
    foundation = load_foundation(str(ENC), device=torch.device('cpu'), freeze=True)
    if kind == 'centernet':
        enc = JAISPEncoderWrapper(foundation, freeze=True).to(device).eval()
        return CenterNetDetector.load(str(ckpt), encoder=enc, device=device).eval()
    return StemCenterNetDetector.load(str(ckpt), foundation, device=device).eval()


def depth50(mags, comp):
    """Linear-interp VIS mag where completeness crosses 50% (going faint)."""
    mags, comp = np.asarray(mags, float), np.asarray(comp, float)
    for i in range(len(mags) - 1):
        if comp[i] >= 50 >= comp[i + 1]:
            f = (50 - comp[i]) / (comp[i + 1] - comp[i] + 1e-9)
            return round(float(mags[i] + f * (mags[i + 1] - mags[i])), 2)
    return None


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mer = load_mer(str(MER))
    stems = sorted(p.name.replace('_euclid.npz', '')
                   for p in EUCLID.glob('tile_*_patch_25_euclid.npz'))[:N_TILES]
    print(f'injection on {len(stems)} held-out patch-25 tiles, conf={CONF}, modes=all/vis/nisp')
    results = {}
    for name, kind, ckpt in MODELS:
        print(f'\n=== {name} ===', flush=True)
        det = load_model(kind, ckpt, device)
        rec = eval_injection(det, stems, mer, str(EUCLID), str(RUBIN), device,
                             modes=('all', 'vis', 'nisp'), target_mags=MAGS, conf=CONF)
        curves, depths = {}, {}
        for mode in ('all', 'vis', 'nisp'):
            m, c = rec_to_curve(rec[mode], MAGS)
            curves[mode] = {'mag': m.tolist(), 'comp': c.tolist()}
            depths[mode] = depth50(m, c)
            print(f'  {mode:5s}: ' + ' '.join(f'{cc:4.0f}' for cc in c) + f'   50%depth={depths[mode]}')
        gain = (depths['all'] - depths['vis']) if (depths['all'] and depths['vis']) else None
        print(f'  --> depth all={depths["all"]} vis={depths["vis"]} nisp={depths["nisp"]} | fusion gain(all-vis)={gain}')
        results[name] = {'curves': curves, 'depth50': depths, 'fusion_gain_mag': gain}
        del det
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    out = OUTD / 'injection_metrics.json'
    json.dump(results, open(out, 'w'), indent=1)
    print(f'\nsaved -> {out}')
    print('\n=== DEPTH SUMMARY (50% point-source) ===')
    print(f'{"model":14s} {"all":>6s} {"vis":>6s} {"nisp":>6s} {"gain":>6s}')
    for n, r in results.items():
        d = r['depth50']
        print(f'{n:14s} {str(d["all"]):>6s} {str(d["vis"]):>6s} {str(d["nisp"]):>6s} {str(r["fusion_gain_mag"]):>6s}')


if __name__ == '__main__':
    main()
