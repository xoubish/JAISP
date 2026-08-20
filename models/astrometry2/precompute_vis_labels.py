"""Precompute the per-tile VIS refined labels used by train_latent_position.

Reproduces exactly the per-epoch label derivation in the training loop
(CenterNet seeds -> signal gate -> classical Gaussian refinement), so the
result can be passed as --canonical-labels and the per-epoch refinement
skipped. Output entries carry 'joint_ok' (all False) purely to satisfy the
loader's bookkeeping print.

Usage:
    PYTHONPATH=models:models/astrometry2 python models/astrometry2/precompute_vis_labels.py \
        --rubin-dir data/rubin_tiles_all --euclid-dir data/euclid_tiles_all_q1 \
        --centernet-labels data/detection_labels/centernet_q1_790_vissep_thresh03.pt \
        --out data/detection_labels/vis_refined_labels_q1_vissep.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from foundation_utils import discover_tile_pairs
from astrometry2.dataset import _to_float32
from astrometry2.source_matching import safe_header_from_card_string
from train_latent_position import detect_and_refine_vis
from astropy.wcs import WCS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rubin-dir', default='data/rubin_tiles_all')
    ap.add_argument('--euclid-dir', default='data/euclid_tiles_all_q1')
    ap.add_argument('--centernet-labels', default='data/detection_labels/centernet_q1_790_vissep_thresh03.pt')
    ap.add_argument('--out', default='data/detection_labels/vis_refined_labels_q1_vissep.pt')
    ap.add_argument('--vis-nsig', type=float, default=4.0)
    ap.add_argument('--vis-smooth', type=float, default=1.2)
    ap.add_argument('--vis-min-dist', type=int, default=9)
    ap.add_argument('--max-sources-vis', type=int, default=800)
    ap.add_argument('--refine-radius', type=int, default=3)
    ap.add_argument('--refine-flux-floor-sigma', type=float, default=1.5)
    args = ap.parse_args()

    cn = torch.load(args.centernet_labels, map_location='cpu', weights_only=False)
    cn = cn['labels'] if 'labels' in cn else cn
    pairs = discover_tile_pairs(args.rubin_dir, args.euclid_dir)
    print(f'{len(pairs)} tile pairs, {len(cn)} centernet label tiles')

    out = {}
    for n, (tile_id, rubin_path, euclid_path) in enumerate(pairs):
        try:
            edata = np.load(euclid_path, allow_pickle=True)
            vis_img = np.nan_to_num(_to_float32(edata['img_VIS']), nan=0.0)
            vis_var = _to_float32(edata['var_VIS']) if 'var_VIS' in edata.files else None
            vis_rms = (np.maximum(np.nan_to_num(np.sqrt(np.clip(vis_var, 0, None)), nan=1.0), 1e-10)
                       if vis_var is not None else None)
            vis_wcs = WCS(safe_header_from_card_string(edata['wcs_VIS'].item()))
            det_px = None
            if tile_id in cn:
                entry = cn[tile_id]
                xy_norm = entry[0] if isinstance(entry, tuple) else entry
                H, W = vis_img.shape
                det_px = np.stack([xy_norm[:, 0]*max(W-1, 1), xy_norm[:, 1]*max(H-1, 1)],
                                  axis=1).astype(np.float32)
            vis_xy, vis_snr = detect_and_refine_vis(
                vis_img, vis_wcs, nsig=args.vis_nsig, smooth=args.vis_smooth,
                min_dist=args.vis_min_dist, max_sources=args.max_sources_vis,
                refine_radius=args.refine_radius,
                flux_floor_sigma=args.refine_flux_floor_sigma,
                vis_rms=vis_rms, detections_vis_px=det_px,
            )
            if vis_xy.shape[0] < 5:
                continue
            out[tile_id] = {'xy': vis_xy.astype(np.float32),
                            'snr': vis_snr.astype(np.float32),
                            'joint_ok': np.zeros(len(vis_xy), dtype=bool)}
        except Exception as exc:
            print(f'skip {tile_id}: {exc}')
        if (n+1) % 50 == 0:
            print(f'{n+1}/{len(pairs)} tiles', flush=True)

    torch.save({'labels': out}, args.out)
    print(f'saved {len(out)} tiles -> {args.out}')


if __name__ == '__main__':
    main()
