"""Build the PSF v4 training set from CenterNet v8 detections.

For each high-confidence, well-isolated detection, extracts a small image stamp
in every available band at native resolution and records the source's
sub-pixel position within the stamp. The training set is then consumed by
``train_psf_v4.py`` to fit the NN ePSF head.

Output schema (per band, saved as one .npz file in ``out_dir``):

    stamps    : float32 [N, H, W]  -- native-resolution image stamps centred on each star
    rms       : float32 [N, H, W]  -- pixel-wise RMS for each stamp
    frac_xy   : float32 [N, 2]     -- sub-pixel offset of the source within the stamp
                                     (in native pixels; (0, 0) means at the centre pixel
                                     centre, (0.5, 0.5) means the source is at the
                                     intersection of four pixels)
    pos_norm  : float32 [N, 2]     -- (x, y) source position normalised to [-1, 1] across
                                     the tile (used as input to the NN ePSF model)
    pos_pix   : float32 [N, 2]     -- (x, y) source position in this band's native pixels
    snr       : float32 [N]        -- estimated SNR of the source in this band
    flux      : float32 [N]        -- estimated total flux (for normalisation)
    tile_id   : <U64   [N]         -- which tile each star came from
    score     : float32 [N]        -- CenterNet detection score (in VIS frame)

Run::

    PYTHONPATH=models python models/psf/build_psf_v4_training_set.py \\
        --rubin-dir   data/rubin_tiles_all \\
        --euclid-dir  data/euclid_tiles_all \\
        --detections  data/detection_labels/centernet_v8_r2_790.pt \\
        --out-dir     data/psf_training_v4 \\
        --score-thr   0.5 \\
        --isolation-arcsec 1.5 \\
        --stamp-size  32

The defaults select roughly the top ~30 % of CenterNet detections (the well-isolated
brighter half) and use a 32-pixel native stamp — 6.4 arcsec for Rubin, 3.2 arcsec
for Euclid, both well above the relevant PSF FWHMs.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from jaisp_dataset_v6 import JAISPDatasetV6  # noqa: E402
from jaisp_foundation_v7 import (  # noqa: E402
    ALL_BANDS,
    EUCLID_BANDS,
    RUBIN_BANDS,
    band_group,
    STREAM_PIXEL_SCALES,
)


# ============================================================
# Helpers
# ============================================================

def _load_detections(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load CenterNet v8 detection labels.

    Returns ``{tile_id: (positions[N, 2] in [0,1] VIS-frame, scores[N])}``.
    """
    blob = torch.load(str(path), weights_only=False, map_location="cpu")
    labels = blob["labels"]
    scores = blob["scores"]
    out = {}
    for tid, (xy_norm, _cls) in labels.items():
        sc = scores.get(tid)
        if sc is None:
            continue
        out[tid] = (np.asarray(xy_norm, dtype=np.float32),
                    sc.cpu().numpy().astype(np.float32) if hasattr(sc, "cpu") else np.asarray(sc, dtype=np.float32))
    return out


def _isolation_mask(xy_pix: np.ndarray, radius_pix: float) -> np.ndarray:
    """Boolean mask of detections that have *no* other detection within ``radius_pix``.

    Naive O(N^2) — fine for ~200 detections per tile.
    """
    n = len(xy_pix)
    if n == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        d = np.hypot(xy_pix[:, 0] - xy_pix[i, 0], xy_pix[:, 1] - xy_pix[i, 1])
        d[i] = np.inf
        if (d < radius_pix).any():
            keep[i] = False
    return keep


def _vis_norm_to_band_pixel(xy_norm: np.ndarray, vis_hw: Tuple[int, int],
                            band: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Convert normalised VIS-frame (x, y) in [0, 1] to native-band pixel coords.

    Rubin pixel scale is 0.2"/px, Euclid is 0.1"/px. The VIS image is at 0.1"/px
    natively. So::

        rubin_pix = vis_pix / 2
        euclid_pix = vis_pix
    """
    H_vis, W_vis = vis_hw
    x_vis = xy_norm[:, 0] * W_vis
    y_vis = xy_norm[:, 1] * H_vis
    if band in RUBIN_BANDS:
        x_b = x_vis / 2.0
        y_b = y_vis / 2.0
    else:
        x_b = x_vis
        y_b = y_vis
    return np.stack([x_b, y_b], axis=-1), (H_vis, W_vis)


def _cut_stamp(img: np.ndarray, rms: np.ndarray,
               x_pix: float, y_pix: float, stamp: int) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
    """Cut a stamp×stamp window from img/rms centred on (x_pix, y_pix).

    Returns ``(stamp_img, stamp_rms, frac_x, frac_y, ok)`` where ``frac_*`` is
    the source's sub-pixel offset from the centre of the central pixel
    (range (-0.5, +0.5]), and ``ok`` is False if the stamp would fall off the
    image edge.
    """
    H, W = img.shape[-2:]
    half = stamp // 2

    # Integer pixel that the source falls in (round-half-to-even is fine here)
    ix = int(round(x_pix))
    iy = int(round(y_pix))
    frac_x = float(x_pix - ix)   # in (-0.5, +0.5]
    frac_y = float(y_pix - iy)

    x0 = ix - half
    y0 = iy - half
    x1 = x0 + stamp
    y1 = y0 + stamp

    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None, None, 0.0, 0.0, False

    return img[..., y0:y1, x0:x1].copy(), rms[..., y0:y1, x0:x1].copy(), frac_x, frac_y, True


def _estimate_snr_flux(stamp_img: np.ndarray, stamp_rms: np.ndarray,
                       frac_x: float, frac_y: float, ap_radius: float = 3.0
                       ) -> Tuple[float, float]:
    """Aperture SNR + flux estimate inside ``ap_radius`` pixels of the centroid."""
    H, W = stamp_img.shape[-2:]
    yy, xx = np.indices(stamp_img.shape[-2:], dtype=np.float32)
    cx = (W - 1) * 0.5 + frac_x
    cy = (H - 1) * 0.5 + frac_y
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    in_ap = r2 < ap_radius ** 2
    if not in_ap.any():
        return 0.0, 0.0

    img = stamp_img[..., :, :]
    rms = stamp_rms[..., :, :]
    if img.ndim == 3:
        img = img[0]; rms = rms[0]

    # Local background from an annulus
    bg_mask = (r2 > (ap_radius + 1.0) ** 2) & (r2 < (ap_radius + 4.0) ** 2)
    bg = float(np.median(img[bg_mask])) if bg_mask.any() else 0.0
    flux = float((img[in_ap] - bg).sum())
    var = float((rms[in_ap] ** 2).sum())
    snr = flux / max(np.sqrt(var), 1e-9)
    return snr, flux


# ============================================================
# Main
# ============================================================

def build_training_set(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.rubin_dir} + {args.euclid_dir} …")
    ds = JAISPDatasetV6(
        rubin_dir=str(args.rubin_dir),
        euclid_dir=str(args.euclid_dir),
        load_euclid=True,
        augment=False,
    )
    tile_id_to_idx = {ds.tiles[i]["tile_id"]: i for i in range(len(ds))}
    print(f"  {len(ds)} tiles available.")

    print(f"Loading CenterNet detections from {args.detections} …")
    dets = _load_detections(Path(args.detections))
    print(f"  {len(dets)} tiles with detections, "
          f"{sum(len(v[0]) for v in dets.values())} total detections.")

    # Per-band accumulators
    out: Dict[str, Dict[str, list]] = {b: {
        "stamps": [], "rms": [], "frac_xy": [],
        "pos_norm": [], "pos_pix": [],
        "snr": [], "flux": [],
        "tile_id": [], "score": [],
    } for b in ALL_BANDS}

    n_kept_total = 0
    n_seen = 0
    score_thr = float(args.score_thr)
    iso_arcsec = float(args.isolation_arcsec)
    snr_thr = float(args.snr_thr)
    stamp = int(args.stamp_size)

    for tile_id, (xy_norm_all, sc_all) in dets.items():
        if tile_id not in tile_id_to_idx:
            continue
        sample = ds[tile_id_to_idx[tile_id]]

        # Score cut + sort by score (descending) for stable behaviour
        ok = sc_all >= score_thr
        if not ok.any():
            continue
        xy_norm = xy_norm_all[ok]
        sc = sc_all[ok]

        # Need at least one Euclid band to define the VIS frame
        euclid = sample.get("euclid", {})
        if not euclid:
            continue
        ref_b = "euclid_VIS" if "euclid_VIS" in euclid else next(iter(euclid))
        H_vis, W_vis = euclid[ref_b]["image"].shape[-2:]

        # Convert to VIS pixels and apply isolation cut (in VIS-pixel units)
        xy_vis_pix = np.stack([xy_norm[:, 0] * W_vis, xy_norm[:, 1] * H_vis], axis=-1)
        # 1 VIS pixel = 0.1 arcsec → isolation radius in VIS pixels
        iso_pix = iso_arcsec / STREAM_PIXEL_SCALES["euclid"]
        keep = _isolation_mask(xy_vis_pix, iso_pix)
        xy_norm = xy_norm[keep]
        sc = sc[keep]
        if len(xy_norm) == 0:
            continue
        n_seen += len(xy_norm)

        # Per band, cut stamps + filter by SNR
        for band in ALL_BANDS:
            if band in RUBIN_BANDS:
                if band not in sample.get("rubin", {}):
                    continue
                img = sample["rubin"][band]["image"][0].numpy()  # [H, W]
                rms = sample["rubin"][band]["rms"][0].numpy()
            else:
                if band not in sample.get("euclid", {}):
                    continue
                img = sample["euclid"][band]["image"][0].numpy()
                rms = sample["euclid"][band]["rms"][0].numpy()

            xy_pix, _ = _vis_norm_to_band_pixel(xy_norm, (H_vis, W_vis), band)

            for k in range(len(xy_norm)):
                xs, ys = float(xy_pix[k, 0]), float(xy_pix[k, 1])
                stamp_img, stamp_rms, fx, fy, ok = _cut_stamp(img, rms, xs, ys, stamp)
                if not ok:
                    continue
                snr, flux = _estimate_snr_flux(stamp_img, stamp_rms, fx, fy)
                if snr < snr_thr:
                    continue
                # Position normalised to [-1, 1] across the tile (band's native frame)
                # We also store the un-normalised pixel coords for convenience.
                Hb, Wb = img.shape[-2:]
                pos_norm = np.array([
                    2.0 * xs / max(Wb - 1, 1) - 1.0,
                    2.0 * ys / max(Hb - 1, 1) - 1.0,
                ], dtype=np.float32)
                out[band]["stamps"].append(stamp_img.astype(np.float32))
                out[band]["rms"].append(stamp_rms.astype(np.float32))
                out[band]["frac_xy"].append([fx, fy])
                out[band]["pos_norm"].append(pos_norm)
                out[band]["pos_pix"].append([xs, ys])
                out[band]["snr"].append(snr)
                out[band]["flux"].append(flux)
                out[band]["tile_id"].append(tile_id)
                out[band]["score"].append(float(sc[k]))
                n_kept_total += 1

        if (n_seen and n_seen % 5000 < 50):
            print(f"  ... seen {n_seen} candidate detections, kept {n_kept_total} (band, star) stamps")

    # Save per-band npz
    print()
    print(f"Total kept: {n_kept_total} (band, star) stamps from {n_seen} candidates "
          f"(score>={score_thr}, isolation>={iso_arcsec}\" SNR>={snr_thr}).")
    for band in ALL_BANDS:
        rec = out[band]
        n = len(rec["stamps"])
        if n == 0:
            print(f"  {band}: 0 stamps — skipping.")
            continue
        path = out_dir / f"{band}.npz"
        np.savez_compressed(
            path,
            stamps=np.stack(rec["stamps"]).astype(np.float32),
            rms=np.stack(rec["rms"]).astype(np.float32),
            frac_xy=np.array(rec["frac_xy"], dtype=np.float32),
            pos_norm=np.array(rec["pos_norm"], dtype=np.float32),
            pos_pix=np.array(rec["pos_pix"], dtype=np.float32),
            snr=np.array(rec["snr"], dtype=np.float32),
            flux=np.array(rec["flux"], dtype=np.float32),
            tile_id=np.array(rec["tile_id"], dtype=np.str_),
            score=np.array(rec["score"], dtype=np.float32),
        )
        print(f"  {band:14s}  {n:6d} stamps  →  {path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rubin-dir", required=True, type=Path)
    p.add_argument("--euclid-dir", required=True, type=Path)
    p.add_argument("--detections", required=True, type=Path,
                   help="Path to CenterNet detection labels (centernet_v8_r2_790.pt).")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--score-thr", type=float, default=0.5,
                   help="CenterNet detection-score threshold (default 0.5).")
    p.add_argument("--isolation-arcsec", type=float, default=1.5,
                   help="Reject detections with another detection within this radius (default 1.5\").")
    p.add_argument("--snr-thr", type=float, default=10.0,
                   help="Per-band SNR cut for keeping a (band, star) stamp (default 10).")
    p.add_argument("--stamp-size", type=int, default=32,
                   help="Stamp side in native band pixels (default 32). "
                        "32 px → 6.4\" Rubin, 3.2\" Euclid.")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    build_training_set(args)
