"""Injection-recovery eval for the v11 detector twins, star-donor paper protocol
(rvis=30, donor concentration >= 0.65). Reuses detection/inject_eval.py wholesale by
overriding its module registry; results land in checkpoints/q1_detection_v11/ with
the same file schema as the production JSONs.

Usage: python eval_inject_v11.py <which>
  which = v11      -> evaluate whatever v11 detectors exist (vis_peak/vis_sep/stem, v11 encoder)
  which = v10base  -> evaluate the v10 BASE centernet_vis_peak (v10 encoder) into
                      checkpoints/q1_detection/ (production dir, resumable cache, new model key)
"""
import sys
from pathlib import Path

REPO = Path('/home/shemmati/Work/Projects/JAISP')
sys.path.insert(0, str(REPO/'models'))
import detection.inject_eval as ie

which = sys.argv[1]
if which == 'v11':
    OUT11 = REPO/'checkpoints/q1_detection_v11'
    ie.ENC = REPO/'models/checkpoints/jaisp_v11_q1_soft/checkpoint_best.pt'
    ie.OUTD = OUT11
    models = []
    for name, kind, fn in [('cn_vis_peak_v11', 'centernet', 'centernet_vis_peak.pt'),
                           ('cn_vis_sep_v11', 'centernet', 'centernet_vis_sep.pt'),
                           ('cn_mer_v11', 'centernet', 'centernet_mer.pt'),
                           ('stem_mer_v11', 'stem', 'stem_mer.pt')]:
        if (OUT11/fn).exists():
            models.append((name, kind, OUT11/fn))
    ie.MODELS = models
    print('v11 eval models:', [m[0] for m in models])
elif which == 'v10base':
    ie.OUTD = REPO/'checkpoints/q1_detection_v11'
    ie.MODELS = [('cn_vis_peak_v10base', 'centernet', REPO/'checkpoints/q1_detection/centernet_vis_peak.pt')]
    print('v10 base eval: cn_vis_peak_v10base (v10 encoder, v11 comparison OUTD)')
else:
    raise SystemExit('which must be v11 or v10base')

sys.argv = ['inject_eval.py', '--rvis', '30', '--donor-conc', '0.65', '--donor-faint', '22.5']
ie.main()
