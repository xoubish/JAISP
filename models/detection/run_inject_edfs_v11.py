"""EDF-S v11 injection depth (truth-based OOD transfer)."""
import sys
from pathlib import Path
REPO = Path('/home/shemmati/Work/Projects/JAISP')
sys.path.insert(0, str(REPO/'models'))
import detection.inject_eval_edfs as ie
V11 = REPO/'checkpoints/q1_detection_v11'
ie.ENC = REPO/'models/checkpoints/jaisp_v11_q1_soft/checkpoint_best.pt'
ie.MER = REPO/'data/edf_s_ood/catalogs_compact/mer_FINAL_q1_TILE102021011_footprint.fits'
ie.EUCLID = REPO/'data/edf_s_ood/euclid_tiles_edfs_q1'
ie.RUBIN = REPO/'data/edf_s_ood/rubin_tiles_edfs'
ie.OUTD = V11
ie.MODELS = [('cn_vis_sep_edfs', 'centernet', V11/'centernet_vis_sep.pt')]
import sys as _s
_deep = len(_s.argv) > 1 and _s.argv[1] == 'deep'
sys.argv = ['inject_eval_edfs.py', '--rvis', '30', '--donor-conc', '0.65',
            '--donor-faint', '22.5', '--tag', '_r30_star_edfs'] + \
           (['--mags', '27.0,27.5', '--tag', '_r30_star_edfs_deep'] if _deep else [])
if _deep:
    sys.argv = ['inject_eval_edfs.py', '--rvis', '30', '--donor-conc', '0.65',
                '--donor-faint', '22.5', '--mags', '27.0,27.5', '--tag', '_r30_star_edfs_deep']
ie.main()
