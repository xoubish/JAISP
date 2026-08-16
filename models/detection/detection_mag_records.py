"""Per-detection records for purity/completeness vs magnitude (MER + injection only).

For every detection on the 108 patch-disjoint held-out tiles (production detector
cn_vis_sep, one scored pass at a 0.10 conf floor, Gaia+saturation mask as in
bakeoff_eval): measured VIS aperture flux and S/N (0.3" radius, local background
annulus), distance to the nearest full-MER and clean-MER source, and the MER VIS
magnitude of the match. A global VIS zeropoint is calibrated from MER-matched
detections at 20 < mag_MER < 24, so unmatched detections, which have no catalog
magnitude by definition, still land on the same magnitude axis. This is what a
purity-vs-magnitude curve needs.

Writes checkpoints/q1_detection/detection_mag_records.npz; per-tile cache in
checkpoints/q1_detection/mag_records_cache/.

Usage: PYTHONPATH=models python models/detection/detection_mag_records.py
"""
from __future__ import annotations
import glob
import numpy as np
import torch
from pathlib import Path
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detection.validation_utils import build_inputs, _wcs_vis, tile_paths, load_mer, PXE
from detection.masks import load_gaia_cache, bright_star_saturation_mask
from detection.bakeoff_eval import load_model, detect_scored

REPO = Path(__file__).resolve().parent.parent.parent
MER = REPO / 'data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits'
EUCLID = REPO / 'data/euclid_tiles_all_q1'
RUBIN = REPO / 'data/rubin_tiles_all'
GAIA = REPO / 'data/gaia_ecdfs_astrometry_cache.npz'
OUTD = REPO / 'checkpoints/q1_detection'
CACHE = OUTD / 'mag_records_cache'
MARGIN, AP_PX = 8, 0.3 / PXE          # 0.3" aperture radius in VIS px


def ap_flux(img, var, x, y, r):
    """Background-subtracted aperture flux and noise at (x, y); (nan, nan) off-image."""
    H, W = img.shape
    xi, yi = int(round(x)), int(round(y))
    R = int(r) + 3
    if xi - R < 0 or yi - R < 0 or xi + R + 1 > W or yi + R + 1 > H:
        return np.nan, np.nan
    yy, xx = np.mgrid[yi - R:yi + R + 1, xi - R:xi + R + 1]
    rr = np.hypot(xx - x, yy - y)
    sub = np.nan_to_num(img[yi - R:yi + R + 1, xi - R:xi + R + 1].astype(np.float32))
    vsub = np.nan_to_num(var[yi - R:yi + R + 1, xi - R:xi + R + 1].astype(np.float32), nan=0.0)
    disk, ann = rr <= r, (rr > r) & (rr <= R)
    if disk.sum() < 1 or ann.sum() < 3:
        return np.nan, np.nan
    flux = float((sub[disk] - np.median(sub[ann])).sum())
    return flux, float(np.sqrt(max(vsub[disk].sum(), 1e-12)))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CACHE.mkdir(parents=True, exist_ok=True)
    mer = load_mer(str(MER))
    gaia = load_gaia_cache(str(GAIA))
    stems = sorted(p.name.replace('_euclid.npz', '')
                   for p in EUCLID.glob('tile_*_patch_25_euclid.npz'))
    print(f'{len(stems)} held-out patch-25 tiles')
    det = None
    for i, stem in enumerate(stems):
        cpath = CACHE / f'{stem}.npz'
        if cpath.exists():
            continue
        if det is None:
            det = load_model('centernet', OUTD / 'centernet_vis_sep.pt', device)
        ep, rp = tile_paths(stem, str(EUCLID), str(RUBIN))
        if not (glob.glob(ep) and glob.glob(rp)):
            np.savez_compressed(cpath, empty=np.zeros(1))
            continue
        ed = dict(np.load(ep, allow_pickle=True)); rd = dict(np.load(rp, allow_pickle=True))
        images, rms, vh = build_inputs(ed, rd); H, W = vh
        vw = _wcs_vis(ed)
        mask = bright_star_saturation_mask(images['euclid_VIS'], str(ed['wcs_VIS']), gaia)
        xy, sc = detect_scored(det, images, rms, vh, device, mask)
        din = (xy[:, 0] >= MARGIN) & (xy[:, 0] < W - MARGIN) & (xy[:, 1] >= MARGIN) & (xy[:, 1] < H - MARGIN)
        xy, sc = xy[din], sc[din]
        if len(xy) < 2:
            np.savez_compressed(cpath, empty=np.zeros(1))
            continue
        fx, fy = vw.all_world2pix(mer['fRA'], mer['fDEC'], 0)
        fk = (fx >= MARGIN) & (fx < W - MARGIN) & (fy >= MARGIN) & (fy < H - MARGIN)
        cx, cy = vw.all_world2pix(mer['cRA'], mer['cDEC'], 0)
        ck = (cx >= MARGIN) & (cx < W - MARGIN) & (cy >= MARGIN) & (cy < H - MARGIN)
        d_f, j_f = cKDTree(np.c_[fx[fk], fy[fk]]).query(xy)
        d_c = cKDTree(np.c_[cx[ck], cy[ck]]).query(xy)[0] if ck.sum() >= 2 else np.full(len(xy), 1e9)
        # MER mag of the full-catalog match (mag_vis exists only for VIS-detected rows;
        # load_mer keeps clean mags — map full-index magnitudes via nearest clean source instead)
        d2c, j2c = cKDTree(np.c_[cx[ck], cy[ck]]).query(xy) if ck.sum() >= 2 else (np.full(len(xy), 1e9), None)
        mer_mag = np.where(d2c < 0.5 / PXE, mer['cMAG'][ck][j2c] if j2c is not None else np.nan, np.nan)
        vis = np.asarray(ed['img_VIS'], np.float32)
        var = np.asarray(ed['var_VIS'], np.float32)
        fl = np.array([ap_flux(vis, var, x, y, AP_PX) for x, y in xy], np.float32)
        np.savez_compressed(cpath, score=sc.astype(np.float32),
                            flux=fl[:, 0], fnoise=fl[:, 1],
                            d_mer_full=(d_f * PXE).astype(np.float32),
                            d_mer_clean=(d_c * PXE).astype(np.float32),
                            mer_mag=mer_mag.astype(np.float32))
        print(f'[{i + 1}/{len(stems)}] {stem}  ndet={len(xy)}', flush=True)
    # aggregate
    recs = {k: [] for k in ('score', 'flux', 'fnoise', 'd_mer_full', 'd_mer_clean', 'mer_mag', 'tile')}
    for ti, stem in enumerate(stems):
        z = np.load(CACHE / f'{stem}.npz')
        if 'empty' in z:
            continue
        for k in recs:
            recs[k].append(z[k] if k != 'tile' else np.full(len(z['score']), ti))
    flat = {k: np.concatenate(v) for k, v in recs.items()}
    # global zeropoint from MER-matched detections at 20 < mag < 24
    m = (np.isfinite(flat['mer_mag']) & (flat['mer_mag'] > 20) & (flat['mer_mag'] < 24)
         & (flat['flux'] > 0) & (flat['d_mer_clean'] < 0.5))
    zp = float(np.median(flat['mer_mag'][m] + 2.5 * np.log10(flat['flux'][m])))
    flat['zp'] = np.array([zp])
    np.savez_compressed(OUTD / 'detection_mag_records.npz', **flat)
    print(f'{len(flat["score"])} detections | ZP={zp:.3f} from {int(m.sum())} matched 20<mag<24')
    print('saved ->', OUTD / 'detection_mag_records.npz')


if __name__ == '__main__':
    main()
