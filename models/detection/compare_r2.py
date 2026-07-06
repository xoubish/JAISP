"""Round-1 vs round-2 (self-trained) CenterNet comparison, no figure clobber.
Both metrics on the patch-disjoint held-out: MER-match F1 (Gaia-masked) + injection
depth/fusion-gain. Writes checkpoints/q1_detection_r2/r1_vs_r2.json.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detection import bakeoff_eval as B
from detection.validation_utils import eval_injection, rec_to_curve, load_mer
from detection.masks import load_gaia_cache

REPO = Path(__file__).resolve().parent.parent.parent
MODELS = [
    ('round1 (cn_vis_sep)', REPO / 'checkpoints/q1_detection/centernet_vis_sep.pt'),
    ('round2 (self-trained)', REPO / 'checkpoints/q1_detection_r2/centernet_round2.pt'),
]
MAGS = (22.5, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5)


def depth50(mag, comp):
    mag, comp = np.asarray(mag, float), np.asarray(comp, float)
    for i in range(len(mag) - 1):
        if comp[i] >= 50 >= comp[i + 1]:
            f = (50 - comp[i]) / (comp[i + 1] - comp[i] + 1e-9)
            return round(float(mag[i] + f * (mag[i + 1] - mag[i])), 2)
    return None


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mer = load_mer(str(B.MER)); gaia = load_gaia_cache(str(B.GAIA))
    stems = sorted(p.name.replace('_euclid.npz', '')
                   for p in B.EUCLID.glob('tile_*_patch_25_euclid.npz'))
    print(f'{len(stems)} held-out patch-25 tiles\n')
    out = {}
    for name, ckpt in MODELS:
        print(f'=== {name} ===', flush=True)
        det = B.load_model('centernet', ckpt, dev)
        # (1) MER-match F1 (Gaia mask), best over conf sweep
        rows = B.summarize(B.eval_model(det, stems, mer, gaia, dev))
        best = max(rows, key=lambda c: rows[c]['f1'])
        bf = rows[best]
        # (2) injection depth (all / vis), conf 0.30
        rec = eval_injection(det, stems, mer, str(B.EUCLID), str(B.RUBIN), dev,
                             modes=('all', 'vis'), target_mags=MAGS, conf=0.30)
        cA = rec_to_curve(rec['all'], MAGS)[1]; cV = rec_to_curve(rec['vis'], MAGS)[1]
        dA, dV = depth50(MAGS, cA), depth50(MAGS, cV)
        gain = round(dA - dV, 2) if (dA and dV) else None
        out[name] = dict(best_conf=best, f1=bf['f1'], completeness=bf['completeness'], purity=bf['purity'],
                         depth_all=dA, depth_vis=dV, fusion_gain=gain)
        print(f'  MER-F1 {bf["f1"]}@{best} (comp {bf["completeness"]}/pur {bf["purity"]}) | '
              f'depth all={dA} vis={dV} gain={gain}', flush=True)
        del det
        if dev.type == 'cuda': torch.cuda.empty_cache()
    json.dump(out, open(REPO / 'checkpoints/q1_detection_r2/r1_vs_r2.json', 'w'), indent=1)
    print('\n=== ROUND-1 vs ROUND-2 ===')
    print(f'{"model":24s} {"F1":>6s} {"comp":>6s} {"pur":>6s} {"d_all":>6s} {"d_vis":>6s} {"gain":>6s}')
    for n, r in out.items():
        print(f'{n:24s} {r["f1"]:6.1f} {r["completeness"]:6.1f} {r["purity"]:6.1f} '
              f'{str(r["depth_all"]):>6s} {str(r["depth_vis"]):>6s} {str(r["fusion_gain"]):>6s}')
    print('saved -> checkpoints/q1_detection_r2/r1_vs_r2.json')


if __name__ == '__main__':
    main()
