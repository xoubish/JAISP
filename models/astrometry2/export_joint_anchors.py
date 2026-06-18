"""Export field-fit anchors built on JOINT multi-band canonical positions.

Per tile: CenterNet seeds (VIS-gated, same as production) -> classical VIS
refine -> joint multi-band canonical refinement (with per-source sigma from
the chi2 curvature) -> per-band classical centroid + projection (the exact
eval-path function) -> per-band offsets RELATIVE TO THE JOINT CANONICAL.

Output npz schema is a superset of anchors_centernet_v10.npz so the existing
PINN/HGP tools work unchanged:
  {b}_ra, {b}_dec      : canonical sky position of the source (deg)
  {b}_raw [N,2]        : band-minus-canonical offset (arcsec, tangent plane)
  {b}_head_resid [N,2] : copy of raw (no head in this chain; kept for loaders)
  {b}_snr              : band SNR
  {b}_tiles            : tile id strings
  {b}_sigma_band [N]   : per-axis band centroid noise (FWHM/(2.355*SNR) model, arcsec)
  {b}_sigma_can [N,2]  : per-axis joint-canonical sigma (chi2 curvature, arcsec)
  {b}_joint_ok [N]     : whether the canonical position is joint-fit (else VIS fallback)

Run: python models/astrometry2/export_joint_anchors.py \
        --out models/checkpoints/anchors_joint_canonical_790.npz --workers 12
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "astrometry2"))

BANDS = [("rubin_u", 3.0), ("rubin_g", 3.0), ("rubin_r", 3.0), ("rubin_i", 3.0),
         ("rubin_z", 3.0), ("rubin_y", 3.0), ("nisp_Y", 2.5), ("nisp_J", 2.5), ("nisp_H", 2.5)]
RUBIN_ORDER = ["u", "g", "r", "i", "z", "y"]


def process_tile(task):
    tile_id, rubin_path, euclid_path, det_xy_norm = task
    import numpy as np
    from astropy.wcs import WCS
    from astrometry2.source_matching import safe_header_from_card_string
    from astrometry2.dataset import signal_mask_in_band, refine_centroids_psf_fit
    from astrometry2.joint_centroid import build_band_stack, joint_refine_positions
    from eval_latent_position import centroid_in_band_and_project

    try:
        rdata = dict(np.load(rubin_path, allow_pickle=True))
        edata = dict(np.load(euclid_path, allow_pickle=True))
        vis_img = np.nan_to_num(np.asarray(edata["img_VIS"], dtype=np.float32), nan=0.0)
        H, W = vis_img.shape
        vwcs = WCS(safe_header_from_card_string(edata["wcs_VIS"].item()))
        rwcs = WCS(rdata["wcs_hdr"].item())

        seeds = np.stack([
            det_xy_norm[:, 0] * max(W - 1, 1),
            det_xy_norm[:, 1] * max(H - 1, 1),
        ], axis=1).astype(np.float32)
        keep = signal_mask_in_band(vis_img, seeds, radius=3, flux_floor_sigma=1.5)
        seeds = seeds[keep]
        if seeds.shape[0] < 5:
            return tile_id, None
        vis_xy_cls, vis_snr, _ = refine_centroids_psf_fit(
            vis_img, seeds, radius=3, fwhm_guess=2.5, flux_floor_sigma=1.5)
        if vis_xy_cls.shape[0] != seeds.shape[0]:
            return tile_id, None

        bands_stack = build_band_stack(rdata, edata)
        can_xy, ok, nbu, can_sig = joint_refine_positions(
            bands_stack, vis_xy_cls.astype(np.float32), vwcs, return_sigma=True)
        can_xy = np.where(ok[:, None], can_xy, vis_xy_cls.astype(np.float32))
        can_ra, can_dec = vwcs.pixel_to_world_values(can_xy[:, 0], can_xy[:, 1])

        out = {}
        rcube = np.nan_to_num(np.asarray(rdata["img"], dtype=np.float32), nan=0.0)
        rvar = np.asarray(rdata["var"], dtype=np.float32)
        for band_name, fwhm in BANDS:
            if band_name.startswith("rubin_"):
                short = band_name.split("_", 1)[1]
                bi = RUBIN_ORDER.index(short)
                band_img = rcube[bi]
                band_rms = np.sqrt(np.clip(rvar[bi], 1e-12, None))
                band_wcs = WCS(rdata["wcs_hdr"].item())
                px_scale = 0.2
            else:
                short = band_name.split("_", 1)[1]
                if f"img_{short}" not in edata:
                    continue
                band_img = np.nan_to_num(np.asarray(edata[f"img_{short}"], dtype=np.float32), nan=0.0)
                from astrometry2.dataset import _rms_from_var_or_image
                _vk = f"var_{short}"
                band_rms = _rms_from_var_or_image(
                    np.asarray(edata[_vk], dtype=np.float32) if _vk in edata else None, band_img)
                band_wcs = WCS(safe_header_from_card_string(edata[f"wcs_{short}"].item()))
                px_scale = 0.1
            band_xy_vis, offset_arcsec, valid, band_snr = centroid_in_band_and_project(
                band_img, band_wcs, can_xy.astype(np.float32), vwcs,
                refine_radius=3, fwhm_guess=fwhm, band_rms=band_rms,
            )
            if valid.sum() < 3:
                continue
            vi = np.where(valid)[0]
            # same anchor filters as eval_latent_position defaults
            off = np.asarray(offset_arcsec)[vi]
            snr_all = np.asarray(band_snr)[vi]
            keep_b = (np.hypot(off[:, 0], off[:, 1]) * 1000 < 200.0) & (snr_all >= 5.0)
            if keep_b.sum() < 3:
                continue
            vi = vi[keep_b]
            snr_b = np.clip(np.asarray(band_snr)[vi], 1.0, None)
            sigma_band = (fwhm * px_scale) / (2.355 * snr_b)  # per-axis, arcsec
            out[band_name] = dict(
                ra=np.asarray(can_ra)[vi].astype(np.float32),
                dec=np.asarray(can_dec)[vi].astype(np.float32),
                raw=np.asarray(offset_arcsec)[vi].astype(np.float32),
                snr=snr_b.astype(np.float32),
                sigma_band=sigma_band.astype(np.float32),
                sigma_can=can_sig[vi].astype(np.float32),
                joint_ok=ok[vi],
            )
        return tile_id, out
    except Exception as exc:
        import traceback
        return tile_id, f"ERROR: {exc}\n{traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubin-dir", default="data/rubin_tiles_all")
    ap.add_argument("--euclid-dir", default="data/euclid_tiles_all")
    ap.add_argument("--detector-labels", default="data/detection_labels/centernet_v10_790_thresh03.pt")
    ap.add_argument("--out", default="models/checkpoints/anchors_joint_canonical_790.npz")
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
    print(f"{len(tasks)} tiles, {args.workers} workers")

    acc = {}
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_tile, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            tile_id, payload = fut.result()
            if isinstance(payload, dict):
                for b, d in payload.items():
                    acc.setdefault(b, []).append((tile_id, d))
            elif payload is not None:
                errors.append((tile_id, payload))
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} tiles", flush=True)

    out = {}
    for b, items in acc.items():
        key = b.replace("rubin_", "").replace("nisp_", "nisp_")
        for field in ("ra", "dec", "raw", "snr", "sigma_band", "sigma_can", "joint_ok"):
            out[f"{key}_{field if field != 'raw' else 'raw'}"] = np.concatenate(
                [d[field] for _, d in items], axis=0)
        out[f"{key}_head_resid"] = out[f"{key}_raw"].copy()
        out[f"{key}_tiles"] = np.concatenate(
            [np.full(len(d["snr"]), tid, dtype="U64") for tid, d in items])
        # rename to match existing loader expectations
        out[f"{key}_sigma_band"] = out.pop(f"{key}_sigma_band")
        out[f"{key}_sigma_can"] = out.pop(f"{key}_sigma_can")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    n = {k[:-4]: len(v) for k, v in out.items() if k.endswith("_snr")}
    print(f"saved {args.out}; per-band counts: {n}; errors: {len(errors)}")
    for t, e in errors[:8]:
        print("  ", t, e)


if __name__ == "__main__":
    main()
