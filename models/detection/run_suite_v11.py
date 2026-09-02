"""v11 twins of the fig-6/fig-11 detection eval suite (2026-09-02).

Reuses the production scripts wholesale by overriding module constants
(pattern of io/_nb23_outputs/clampfix_harness/eval_inject_v11.py).
All outputs land in checkpoints/q1_detection_v11/ — NEVER the production dir
(ood_eval's hardcoded production path is handled by the shell driver via
backup/move/restore).

Usage: python run_suite_v11.py {bakeoff|purity|magrec|ood}
"""
import sys
from pathlib import Path

REPO = Path('/home/shemmati/Work/Projects/JAISP')
sys.path.insert(0, str(REPO / 'models'))

V11 = REPO / 'checkpoints/q1_detection_v11'
ENC11 = REPO / 'models/checkpoints/jaisp_v11_q1_soft/checkpoint_best.pt'

import detection.bakeoff_eval as be
be.ENC = ENC11

which = sys.argv[1]
if which == 'bakeoff':
    be.OUTD = V11
    be.MODELS = [(n, k, V11 / f) for n, k, f in [
        ('cn_vis_peak', 'centernet', 'centernet_vis_peak.pt'),
        ('cn_vis_sep',  'centernet', 'centernet_vis_sep.pt'),
        ('cn_mer',      'centernet', 'centernet_mer.pt'),
        ('stem_mer',    'stem',      'stem_mer.pt'),
    ]]
    be.main()
elif which == 'purity':
    import detection.inject_purity_eval as ipe
    ipe.OUTD = V11
    ipe.CACHE = V11 / 'inject_purity_cache_iso.json'
    ipe.LIB = V11 / 'donor_library_r60.npz'   # copied from production (detector-independent)
    ipe.main()
elif which == 'magrec':
    import detection.detection_mag_records as dmr
    dmr.OUTD = V11
    dmr.CACHE = V11 / 'mag_records_cache'
    dmr.CACHE.mkdir(exist_ok=True)
    dmr.main()
elif which == 'ood':
    import detection.ood_eval as oe
    oe.CKPT = V11 / 'centernet_vis_sep.pt'
    oe.main()   # writes to the PRODUCTION path: shell driver backs up + moves
else:
    raise SystemExit('which must be bakeoff|purity|magrec|ood')
