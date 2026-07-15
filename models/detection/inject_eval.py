"""Leg A: source-recycling injection depth on the patch-disjoint held-out set,
for the ready Q1 detectors (cn_vis_sep, cn_mer, stem_mer). Gives the 50%
point-source depth per mode (all-10 / VIS-only / NISP-only), the multi-band
fusion gain (all - vis), and settles CenterNet-vs-Stem. conf=0.30 uniform for
comparability (the all-vs-vis GAP is robust to conf). Curves saved for Fig 6.

Usage: PYTHONPATH=models python models/detection/inject_eval.py [--rvis 30]

--rvis sets the stamp cutout radius in VIS pixels (default 30 = 3''). The
original 2026-06 runs used rvis=8 (0.8''), which truncates the PSF wings of
the bright donors relative to the MER *total* magnitudes that label them:
measured stamp-flux/catalog-flux ratios are ~35% low for the 19.5-21.5 donor
pool (io/22_injection_nodim_validation.ipynb), so dimmed injections landed
~0.4-0.5 mag fainter than labeled and the r8 depths are mislabeled
conservative. At 3'' the VIS wings and the ~1'' Rubin PSF are essentially
fully enclosed. Results are written to injection_metrics_r{rvis}.json for
rvis != 8 (the committed r8 file is left untouched); progress is cached
per (model, tile) in injection_cache_r{rvis}.json so interrupted runs resume.
"""
from __future__ import annotations
import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--rvis', type=int, default=8, help="stamp radius in VIS px (8 = legacy, 30 = 3'')")
    ap.add_argument('--donor-conc', type=float, default=None,
                    help="min donor VIS concentration f(<0.3'')/f(<1.5''); 0.65 = star-like donors")
    ap.add_argument('--donor-faint', type=float, default=21.5, help='faint bound of donor pool')
    ap.add_argument('--mags', default=None, help='comma-separated target mags (default: module MAGS)')
    ap.add_argument('--models', default=None, help='comma-separated model names to run (default: all)')
    ap.add_argument('--tag', default=None, help='output tag override (required with --mags)')
    args = ap.parse_args()
    global MAGS, MODELS
    if args.mags:
        assert args.tag, '--mags changes the cache schema; pass an explicit --tag'
        MAGS = tuple(float(x) for x in args.mags.split(','))
    if args.models:
        keep = set(args.models.split(','))
        MODELS = [m for m in MODELS if m[0] in keep]
    tag = args.tag if args.tag else ((f'_r{args.rvis}' if args.rvis != 8 else '') +
                                     ('_star' if args.donor_conc else ''))
    cache_path = OUTD / f'injection_cache{tag}.json'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mer = load_mer(str(MER))
    stems = sorted(p.name.replace('_euclid.npz', '')
                   for p in EUCLID.glob('tile_*_patch_25_euclid.npz'))[:N_TILES]
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    print(f'injection on {len(stems)} held-out patch-25 tiles, conf={CONF}, rvis={args.rvis}, '
          f'modes=all/vis/nisp ({len(cache)} model-tile pairs cached)')
    results = {}
    for name, kind, ckpt in MODELS:
        print(f'\n=== {name} ===', flush=True)
        det = None
        for stem in stems:
            key = f'{name}|{stem}'
            if key in cache:
                continue
            if det is None:
                det = load_model(kind, ckpt, device)
            r1 = eval_injection(det, [stem], mer, str(EUCLID), str(RUBIN), device,
                                modes=('all', 'vis', 'nisp'), target_mags=MAGS, conf=CONF,
                                rvis=args.rvis, donor_mag=(19.5, args.donor_faint),
                                donor_conc=args.donor_conc)
            cache[key] = {mode: {str(mg): r1[mode][mg] for mg in MAGS} for mode in r1}
            json.dump(cache, open(cache_path, 'w'))
            print(f'  {stem}', flush=True)
        # aggregate this model's tiles from the cache
        rec = {mode: {mg: [0, 0] for mg in MAGS} for mode in ('all', 'vis', 'nisp')}
        for stem in stems:
            tile = cache.get(f'{name}|{stem}')
            if not tile:
                continue
            for mode in rec:
                for mg in MAGS:
                    r, n = tile[mode][str(mg)]
                    rec[mode][mg][0] += r; rec[mode][mg][1] += n
        curves, depths = {}, {}
        for mode in ('all', 'vis', 'nisp'):
            m, c = rec_to_curve(rec[mode], MAGS)
            curves[mode] = {'mag': m.tolist(), 'comp': c.tolist()}
            depths[mode] = depth50(m, c)
            print(f'  {mode:5s}: ' + ' '.join(f'{cc:4.0f}' for cc in c) + f'   50%depth={depths[mode]}')
        gain = (round(depths['all'] - depths['vis'], 2) if (depths['all'] and depths['vis']) else None)
        print(f'  --> depth all={depths["all"]} vis={depths["vis"]} nisp={depths["nisp"]} | fusion gain(all-vis)={gain}')
        results[name] = {'curves': curves, 'depth50': depths, 'fusion_gain_mag': gain, 'rvis': args.rvis,
                         'donor_conc': args.donor_conc, 'donor_mag': [19.5, args.donor_faint]}
        if det is not None:
            del det
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    out = OUTD / f'injection_metrics{tag}.json'
    json.dump(results, open(out, 'w'), indent=1)
    print(f'\nsaved -> {out}')
    print('\n=== DEPTH SUMMARY (50% point-source) ===')
    print(f'{"model":14s} {"all":>6s} {"vis":>6s} {"nisp":>6s} {"gain":>6s}')
    for n, r in results.items():
        d = r['depth50']
        print(f'{n:14s} {str(d["all"]):>6s} {str(d["vis"]):>6s} {str(d["nisp"]):>6s} {str(r["fusion_gain_mag"]):>6s}')


if __name__ == '__main__':
    main()
