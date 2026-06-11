"""Precompute joint multi-band canonical positions for every tile (one-time).

For each tile: take the exported CenterNet seeds (same gating as training:
VIS signal mask), run the validated joint multi-band fit, fall back to the
classical VIS Gaussian refine where the joint fit fails. Saves a labels dict
consumable by train_latent_position.py --canonical-labels:

    {tile_id: {'xy': [N,2] float32 VIS px, 'snr': [N] float32,
               'joint_ok': [N] bool, 'n_bands': [N] int16}}

Run:  python models/astrometry2/precompute_joint_canonical_labels.py \
          --out data/astrometry_labels/joint_canonical_790.pt --workers 12
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))


def process_tile(task):
    tile_id, rubin_path, euclid_path, det_xy_norm = task
    import numpy as np
    from astropy.wcs import WCS
    from astrometry2.source_matching import safe_header_from_card_string
    from astrometry2.dataset import signal_mask_in_band, refine_centroids_psf_fit
    from astrometry2.joint_centroid import build_band_stack, joint_refine_positions

    try:
        rdata = dict(np.load(rubin_path, allow_pickle=True))
        edata = dict(np.load(euclid_path, allow_pickle=True))
        vis_img = np.nan_to_num(np.asarray(edata["img_VIS"], dtype=np.float32), nan=0.0)
        H, W = vis_img.shape
        vwcs = WCS(safe_header_from_card_string(edata["wcs_VIS"].item()))

        seeds = np.stack([
            det_xy_norm[:, 0] * max(W - 1, 1),
            det_xy_norm[:, 1] * max(H - 1, 1),
        ], axis=1).astype(np.float32)
        keep = signal_mask_in_band(vis_img, seeds, radius=3, flux_floor_sigma=1.5)
        seeds = seeds[keep]
        if seeds.shape[0] < 5:
            return tile_id, None

        # classical VIS refine: fallback positions + the SNR estimate
        vis_xy_cls, vis_snr, _ = refine_centroids_psf_fit(
            vis_img, seeds, radius=3, fwhm_guess=2.5, flux_floor_sigma=1.5)
        if vis_xy_cls.shape[0] != seeds.shape[0]:
            # refine can drop sources; rerun mask bookkeeping by matching counts
            return tile_id, None

        bands = build_band_stack(rdata, edata)
        joint_xy, ok, nbu = joint_refine_positions(bands, vis_xy_cls, vwcs)
        xy = np.where(ok[:, None], joint_xy, vis_xy_cls.astype(np.float32))
        return tile_id, dict(
            xy=xy.astype(np.float32),
            snr=np.asarray(vis_snr, dtype=np.float32),
            joint_ok=ok,
            n_bands=nbu,
        )
    except Exception as exc:
        return tile_id, f"ERROR: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubin-dir", default="data/rubin_tiles_all")
    ap.add_argument("--euclid-dir", default="data/euclid_tiles_all")
    ap.add_argument("--detector-labels", default="data/detection_labels/centernet_v10_790_thresh03.pt")
    ap.add_argument("--out", default="data/astrometry_labels/joint_canonical_790.pt")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-tiles", type=int, default=0)
    args = ap.parse_args()

    import torch
    from astrometry2.dataset import discover_tile_pairs

    labels = torch.load(args.detector_labels, map_location="cpu", weights_only=False)
    labels = labels["labels"] if "labels" in labels else labels

    pairs = discover_tile_pairs(args.rubin_dir, args.euclid_dir)
    if args.max_tiles > 0:
        pairs = pairs[: args.max_tiles]
    tasks = []
    for tile_id, rp, ep in pairs:
        if tile_id not in labels:
            continue
        ent = labels[tile_id]
        xy_norm = np.asarray(ent[0] if isinstance(ent, tuple) else ent, dtype=np.float32)
        tasks.append((tile_id, rp, ep, xy_norm))
    print(f"{len(tasks)} tiles to process with {args.workers} workers")

    out = {}
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_tile, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            tile_id, payload = fut.result()
            if isinstance(payload, dict):
                out[tile_id] = payload
            else:
                errors.append((tile_id, payload))
            if i % 25 == 0 or i == len(tasks):
                n_ok = sum(p["joint_ok"].sum() for p in out.values())
                n_tot = sum(len(p["joint_ok"]) for p in out.values())
                print(f"  {i}/{len(tasks)} tiles | joint-fit success {n_ok}/{n_tot} sources", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"labels": out, "config": vars(args)}, args.out)
    print(f"saved {len(out)} tiles to {args.out}; {len(errors)} errors")
    for t, e in errors[:10]:
        print("  ", t, e)


if __name__ == "__main__":
    main()
