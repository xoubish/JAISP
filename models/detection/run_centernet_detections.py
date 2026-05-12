"""Run CenterNet inference on Rubin+Euclid tiles and save detection labels.

This is the active detection-package exporter for v8/v9/v10 foundation
checkpoints loaded through ``load_foundation()``.

Usage
-----
    python models/detection/run_centernet_detections.py \
        --encoder_ckpt   models/checkpoints/jaisp_v10_warmstart/checkpoint_best.pt \
        --centernet_ckpt checkpoints/centernet_v10/centernet_best.pt \
        --rubin_dir      data/rubin_tiles_all \
        --euclid_dir     data/euclid_tiles_all \
        --out            data/detection_labels/centernet_v10.pt \
        --conf_threshold 0.3

Output format
-------------
    {
      'labels': { tile_stem: (xy_normalized [N, 2], classes [N]) },
      'scores': { tile_stem: scores [N] },
      'config': { 'encoder_ckpt': ..., 'centernet_ckpt': ..., ... },
    }
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from detection.centernet_detector import CenterNetDetector                 # noqa: E402
from detection.dataset import (                                           # noqa: E402
    _vis_bright_core_and_spike_mask,
    _vis_bright_extended_rescue_labels,
)
from detection.detector import JAISPEncoderWrapper                         # noqa: E402
from jaisp_foundation_v10 import EUCLID_BANDS, RUBIN_BANDS                 # noqa: E402
from load_foundation import load_foundation                                # noqa: E402


def _tile_band_dicts(
    rubin_path: Path,
    euclid_path: Path,
    device: torch.device,
):
    """Load a paired tile into foundation-model image/RMS dicts."""
    rd = np.load(rubin_path, allow_pickle=True, mmap_mode='r')
    ed = np.load(euclid_path, allow_pickle=True, mmap_mode='r')

    images: Dict[str, torch.Tensor] = {}
    rms: Dict[str, torch.Tensor] = {}

    rubin_img = np.asarray(rd['img'], dtype=np.float32)
    rubin_var = np.asarray(rd['var'], dtype=np.float32)
    for bi, band in enumerate(RUBIN_BANDS):
        img = np.nan_to_num(rubin_img[bi], nan=0.0)
        rm = np.maximum(
            np.nan_to_num(np.sqrt(np.clip(rubin_var[bi], 0, None)), nan=1.0),
            1e-10,
        )
        images[band] = torch.from_numpy(img[None, None].copy()).to(device)
        rms[band] = torch.from_numpy(rm[None, None].copy()).to(device)

    vis_hw = None
    vis_img = None
    for band in EUCLID_BANDS:
        euclid_key = band.split('_', 1)[1]
        img_arr = np.asarray(ed[f'img_{euclid_key}'], dtype=np.float32)
        var_arr = np.asarray(ed[f'var_{euclid_key}'], dtype=np.float32)
        img = np.nan_to_num(img_arr, nan=0.0)
        rm = np.maximum(
            np.nan_to_num(np.sqrt(np.clip(var_arr, 0, None)), nan=1.0),
            1e-10,
        )
        images[band] = torch.from_numpy(img[None, None].copy()).to(device)
        rms[band] = torch.from_numpy(rm[None, None].copy()).to(device)
        if euclid_key == 'VIS':
            vis_hw = img.shape
            vis_img = img

    return images, rms, vis_hw, vis_img


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder_ckpt', required=True)
    p.add_argument('--centernet_ckpt', required=True)
    p.add_argument('--rubin_dir', required=True)
    p.add_argument('--euclid_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--conf_threshold', type=float, default=0.3)
    p.add_argument('--nms_kernel', type=int, default=7)
    p.add_argument('--spike_veto_width', type=float, default=3.0,
                   help='Thin bright-star spike veto half-width in VIS pixels; set 0 to disable.')
    p.add_argument('--spike_veto_radius', type=int, default=40,
                   help='Base radial search length for the bright-star spike veto.')
    p.add_argument('--spike_veto_min_star_area', type=int, default=20,
                   help='Minimum saturated VIS blob area for the spike veto anchor.')
    p.add_argument('--bright_rescue', action='store_true',
                   help='Append conservative bright/extended VIS rescue labels to exported teacher labels.')
    p.add_argument('--bright_rescue_nsig', type=float, default=2.5,
                   help='Robust S/N threshold for bright/extended rescue components.')
    p.add_argument('--bright_rescue_min_area', type=int, default=45,
                   help='Minimum connected VIS area in pixels for bright/extended rescue.')
    p.add_argument('--bright_rescue_min_radius', type=float, default=5.0,
                   help='Minimum footprint/moment radius in VIS pixels for bright/extended rescue.')
    p.add_argument('--bright_rescue_min_peak_snr', type=float, default=8.0,
                   help='Minimum peak robust S/N for bright/extended rescue.')
    p.add_argument('--bright_rescue_match_scale', type=float, default=1.5,
                   help='Adaptive existing-detection match radius = scale * footprint radius.')
    p.add_argument('--bright_rescue_match_min', type=float, default=8.0,
                   help='Minimum existing-detection match radius in VIS pixels for rescue coverage.')
    p.add_argument('--bright_rescue_match_max', type=float, default=35.0,
                   help='Maximum existing-detection match radius in VIS pixels for rescue coverage.')
    p.add_argument('--bright_rescue_max_per_tile', type=int, default=32,
                   help='Maximum bright/extended rescue labels to append per tile.')
    p.add_argument('--bright_rescue_score', type=float, default=0.95,
                   help='Score assigned to exported bright/extended rescue labels.')
    p.add_argument('--device', default='')
    p.add_argument('--max_tiles', type=int, default=0, help='0 = all tiles')
    args = p.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f'Loading foundation encoder from {args.encoder_ckpt}')
    foundation = load_foundation(args.encoder_ckpt, device=torch.device('cpu'), freeze=True)
    encoder = JAISPEncoderWrapper(foundation, freeze=True).to(device)
    encoder.eval()

    print(f'Loading CenterNet from {args.centernet_ckpt}')
    detector = CenterNetDetector.load(
        args.centernet_ckpt, encoder=encoder, device=device
    ).eval()

    rubin_dir = Path(args.rubin_dir)
    euclid_dir = Path(args.euclid_dir)
    euclid_map = {
        p.stem.replace('_euclid', ''): p
        for p in euclid_dir.glob('tile_x*_y*_euclid.npz')
    }
    tiles = []
    for rp in sorted(rubin_dir.glob('tile_x*_y*.npz')):
        ep = euclid_map.get(rp.stem)
        if ep is not None:
            tiles.append({'stem': rp.stem, 'rubin_path': rp, 'euclid_path': ep})

    if args.max_tiles > 0:
        tiles = tiles[:args.max_tiles]
    print(f'Found {len(tiles)} paired tiles')

    labels: Dict[str, tuple] = {}
    scores: Dict[str, torch.Tensor] = {}

    t_start = time.time()
    last_print = t_start
    n_errors = 0
    tot_det = 0
    tot_vetoed = 0
    tot_rescue = 0

    for i, tile in enumerate(tiles):
        try:
            images, rms, vis_hw, vis_img = _tile_band_dicts(
                tile['rubin_path'], tile['euclid_path'], device=device
            )

            artifact_mask = None
            if args.spike_veto_width > 0 and vis_img is not None:
                _, _, spike_mask = _vis_bright_core_and_spike_mask(
                    vis_img,
                    spike_radius=args.spike_veto_radius,
                    min_star_area=args.spike_veto_min_star_area,
                    spike_width=args.spike_veto_width,
                    include_core=False,
                )
                artifact_mask = torch.from_numpy(spike_mask).to(device)

            with torch.no_grad():
                if artifact_mask is not None:
                    raw_result = detector.predict(
                        images, rms,
                        conf_threshold=args.conf_threshold,
                        tile_hw=vis_hw,
                        nms_kernel=args.nms_kernel,
                        artifact_mask=None,
                    )
                result = detector.predict(
                    images, rms,
                    conf_threshold=args.conf_threshold,
                    tile_hw=vis_hw,
                    nms_kernel=args.nms_kernel,
                    artifact_mask=artifact_mask,
                )

            if artifact_mask is not None:
                tot_vetoed += int(raw_result['scores'].numel() - result['scores'].numel())

            xy_norm = result['centroids'].detach().cpu().numpy().astype(np.float32)
            score_tensor = result['scores'].detach().cpu()
            if args.bright_rescue and vis_img is not None:
                try:
                    rescue_xy, _ = _vis_bright_extended_rescue_labels(
                        vis_img,
                        existing_norm=xy_norm,
                        thresh_nsig=args.bright_rescue_nsig,
                        min_area=args.bright_rescue_min_area,
                        min_radius=args.bright_rescue_min_radius,
                        min_peak_snr=args.bright_rescue_min_peak_snr,
                        match_radius_scale=args.bright_rescue_match_scale,
                        match_radius_min=args.bright_rescue_match_min,
                        match_radius_max=args.bright_rescue_match_max,
                        spike_radius=args.spike_veto_radius,
                        spike_width=args.spike_veto_width,
                        min_star_area=args.spike_veto_min_star_area,
                        max_rescue_per_tile=args.bright_rescue_max_per_tile,
                    )
                    if rescue_xy.shape[0] > 0:
                        xy_norm = np.concatenate([xy_norm, rescue_xy], axis=0).astype(np.float32)
                        rescue_scores = torch.full(
                            (rescue_xy.shape[0],),
                            float(args.bright_rescue_score),
                            dtype=score_tensor.dtype,
                        )
                        score_tensor = torch.cat([score_tensor, rescue_scores], dim=0)
                        tot_rescue += int(rescue_xy.shape[0])
                except Exception as exc:
                    print(f'  [warn] bright rescue failed for {tile["stem"]}: {exc}')

            cls = np.zeros(xy_norm.shape[0], dtype=np.int64)
            labels[tile['stem']] = (xy_norm, cls)
            scores[tile['stem']] = score_tensor
            tot_det += xy_norm.shape[0]
        except Exception as exc:
            n_errors += 1
            print(f'  [err] {tile["stem"]}: {type(exc).__name__}: {exc}')

        now = time.time()
        if now - last_print >= 10.0 or i == len(tiles) - 1:
            elapsed = now - t_start
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(tiles) - i - 1) / max(rate, 1e-6)
            mean_det = tot_det / max(1, (i + 1 - n_errors))
            print(f'  [{i+1:4d}/{len(tiles)}]  '
                  f'mean_det={mean_det:.0f}  '
                  f'rescue={tot_rescue}  '
                  f'vetoed={tot_vetoed}  '
                  f'rate={rate:.2f} tile/s  '
                  f'eta={eta/60:.1f} min')
            last_print = now

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'labels': labels,
        'scores': scores,
        'config': {
            'encoder_ckpt': str(args.encoder_ckpt),
            'centernet_ckpt': str(args.centernet_ckpt),
            'conf_threshold': args.conf_threshold,
            'nms_kernel': args.nms_kernel,
            'spike_veto_width': args.spike_veto_width,
            'spike_veto_radius': args.spike_veto_radius,
            'spike_veto_min_star_area': args.spike_veto_min_star_area,
            'bright_rescue': args.bright_rescue,
            'bright_rescue_nsig': args.bright_rescue_nsig,
            'bright_rescue_min_area': args.bright_rescue_min_area,
            'bright_rescue_min_radius': args.bright_rescue_min_radius,
            'bright_rescue_min_peak_snr': args.bright_rescue_min_peak_snr,
            'bright_rescue_match_scale': args.bright_rescue_match_scale,
            'bright_rescue_match_min': args.bright_rescue_match_min,
            'bright_rescue_match_max': args.bright_rescue_match_max,
            'bright_rescue_max_per_tile': args.bright_rescue_max_per_tile,
            'bright_rescue_score': args.bright_rescue_score,
            'n_tiles': len(labels),
            'n_errors': n_errors,
            'n_vetoed': tot_vetoed,
            'n_bright_rescue': tot_rescue,
        },
    }, out_path)
    print(
        f'Saved {len(labels)} tiles to {out_path}  '
        f'({n_errors} errors, vetoed={tot_vetoed}, bright_rescue={tot_rescue})'
    )


if __name__ == '__main__':
    main()
