"""Purity against injection truth, per injected magnitude, one magnitude per pass.

Inside an injected scene the truth is fully known: every detection appearing
after injection that was not in the pre-injection output is either an injected
source (true positive, within 0.3"), a pre-existing sub-threshold peak nudged
over the working point (a threshold flip, counted separately: the pre-pass runs
at a 0.02 score floor), or an injection-induced false positive. Injecting ONE
magnitude per detector pass makes the accounting attribution-free: whatever
extra appears anywhere in the tile belongs to that magnitude by construction.

    purity_vs_injection(m) = recovered(m) / (recovered(m) + artifacts(m))
    artifact_rate(m)       = artifacts(m) / injected(m)

Protocol hygiene (each item traced to an io/29 diagnostic):
  - 6" tapered stamps (rvis=60, cosine apodization of the outer 15%): the 3"
    truncation edge of the depth-measurement protocol itself fires the detector
    around bright injections.
  - donor isolation 7" (vs 3" for depth runs): a 6" cutout must not carry the
    donor's real cataloged neighbors along as stowaways. Per tile that cut
    leaves almost no donors, so donors come from a GLOBAL library harvested
    once from all ECDFS tiles (donor_library_r60.npz, ~sky-deduplicated).
  - placement margin = stamp radius: _add silently skips stamps crossing the
    image edge, which would count as injected-but-absent.
  - 2" minimum separation between the injections of a pass.

Modes all/vis; conf 0.30; 108 held-out tiles; per-tile cache
inject_purity_cache_iso.json; writes inject_purity_metrics.json.

Usage: PYTHONPATH=models python models/detection/inject_purity_eval.py
"""
from __future__ import annotations
import glob, json
import numpy as np
import torch
from pathlib import Path
from scipy.spatial import cKDTree
from astropy.wcs import WCS

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detection.validation_utils import (
    build_inputs, run_detect, _wcs_vis, tile_paths, load_mer, _stamp, _add,
    _mode_bands, safe_header_from_card_string, PXE, RUBIN_BANDS, EUCLID_BANDS)
from detection.bakeoff_eval import load_model

REPO = Path(__file__).resolve().parent.parent.parent
MER = REPO / 'data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits'
EUCLID = REPO / 'data/euclid_tiles_all_q1'
RUBIN = REPO / 'data/rubin_tiles_all'
OUTD = REPO / 'checkpoints/q1_detection'
CACHE = OUTD / 'inject_purity_cache_iso.json'
LIB = OUTD / 'donor_library_r60.npz'
MAGS = (22.5, 23.5, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0)
MODES = ('all', 'vis')
CONF, RVIS, EDGE = 0.30, 60, 24
N_PER_MAG = 15
MATCH_PX = 3.0            # 0.3": counts as the injected source / same pre-peak
NEW_PX = 4.0              # farther than this from any pre-detection = new
SEP_PX = 20.0             # 2" minimum separation between injections of a pass
PRE_FLOOR = 0.02          # pre-pass floor: peaks above this that flip are not artifacts
DONOR_MAG, DONOR_CONC, DONOR_ISO_AS = (19.5, 22.5), 0.65, 7.0


def _taper(st, frac=0.85):
    """Cosine-apodize the outer (1-frac) of a square stamp so the cut edge is smooth."""
    if st is None:
        return None
    r = st.shape[0] // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    rr = np.hypot(xx, yy) / max(r, 1)
    w = np.ones_like(st)
    m = rr > frac
    w[m] = 0.5 * (1 + np.cos(np.pi * np.clip((rr[m] - frac) / (1 - frac), 0, 1)))
    w[rr > 1] = 0
    return st * w


@torch.no_grad()
def detect_floor(det, images, rms, vh, device, floor):
    im = {b: torch.from_numpy(images[b][None, None].copy()).to(device) for b in images}
    rm = {b: torch.from_numpy(rms[b][None, None].copy()).to(device) for b in rms}
    res = det.predict(im, rm, conf_threshold=floor, tile_hw=vh, nms_kernel=7, artifact_mask=None)
    c = res['centroids'].cpu().numpy(); s = res['scores'].cpu().numpy(); H, W = vh
    xy = np.c_[c[:, 0] * (W - 1), c[:, 1] * (H - 1)] if len(c) else np.zeros((0, 2))
    return xy, s


def build_donor_library(mer):
    """Harvest ultra-isolated star-like donors from ALL ECDFS tiles into one library.

    Cuts: 19.5<mag<22.5, nearest cataloged neighbor >7", VIS concentration >=0.65,
    full 6" stamp inside every band's frame. Stamps stored tapered, per band.
    Sky-deduplicated at 1" (tiles overlap heavily)."""
    sky = cKDTree(np.c_[mer['cRA'], mer['cDEC']])
    nn2 = sky.query(np.c_[mer['cRA'], mer['cDEC']], k=2)[0][:, 1]
    don = (mer['cMAG'] > DONOR_MAG[0]) & (mer['cMAG'] < DONOR_MAG[1]) & (nn2 > DONOR_ISO_AS / 3600.)
    DRA, DDEC, DMAG = mer['cRA'][don], mer['cDEC'][don], mer['cMAG'][don]
    yy_, xx_ = np.mgrid[-30:31, -30:31]; RR_ = np.hypot(xx_, yy_)
    seen, e_st, r_st, mags = [], [], [], []
    stems = sorted(p.name.replace('_euclid.npz', '') for p in EUCLID.glob('tile_*_euclid.npz'))
    for stem in stems:
        ep, rp = tile_paths(stem, str(EUCLID), str(RUBIN))
        if not (glob.glob(ep) and glob.glob(rp)):
            continue
        ed = dict(np.load(ep, allow_pickle=True)); rd = dict(np.load(rp, allow_pickle=True))
        vw = _wcs_vis(ed); H, W = np.asarray(ed['img_VIS']).shape
        dx, dy = vw.all_world2pix(DRA, DDEC, 0)
        din = np.where((dx > EDGE + RVIS) & (dx < W - EDGE - RVIS) & (dy > EDGE + RVIS) & (dy < H - EDGE - RVIS))[0]
        if not len(din):
            continue
        vis = np.nan_to_num(np.asarray(ed['img_VIS'], np.float32))
        ew = {k: WCS(safe_header_from_card_string(ed[f'wcs_{k}'].item())) for k in ('VIS', 'Y', 'J', 'H')}
        rw = WCS(rd['wcs_hdr'].item())
        for j in din:
            if any(abs(DRA[j] - a) < 1 / 3600. and abs(DDEC[j] - b) < 1 / 3600. for a, b in seen):
                continue
            a, b = int(round(float(dx[j]))), int(round(float(dy[j])))
            s = vis[b - 5:b + 6, a - 5:a + 6] - np.median(vis[b - 5:b + 6, a - 5:a + 6])
            tot = s[s > 0].sum()
            if not (tot > 0 and s[3:8, 3:8][s[3:8, 3:8] > 0].sum() / tot > 0.6):
                continue
            st = vis[b - 30:b + 31, a - 30:a + 31]
            if st.shape != (61, 61):
                continue
            st = st - np.median(np.concatenate([st[0], st[-1], st[:, 0], st[:, -1]]))
            f15 = st[RR_ <= 15].sum()
            if not (f15 > 0 and st[RR_ <= 3].sum() / f15 >= DONOR_CONC):
                continue
            es, rs, ok = [], [], True
            for bnd in EUCLID_BANDS:
                k = bnd.split('_', 1)[1]
                stp = _taper(_stamp(np.nan_to_num(np.asarray(ed[f'img_{k}'], np.float32)), ew[k], DRA[j], DDEC[j], RVIS))
                if stp is None or stp.shape != (2 * RVIS + 1, 2 * RVIS + 1):
                    ok = False; break
                es.append(stp)
            if not ok:
                continue
            for bi, bnd in enumerate(RUBIN_BANDS):
                stp = _taper(_stamp(np.asarray(rd['img'], np.float32)[bi], rw, DRA[j], DDEC[j], RVIS // 2))
                if stp is None or stp.shape != (RVIS + 1, RVIS + 1):
                    ok = False; break
                rs.append(stp)
            if not ok:
                continue
            seen.append((DRA[j], DDEC[j]))
            e_st.append(np.stack(es)); r_st.append(np.stack(rs)); mags.append(DMAG[j])
    lib = dict(euclid=np.stack(e_st).astype(np.float32), rubin=np.stack(r_st).astype(np.float32),
               mag=np.array(mags, np.float32))
    np.savez_compressed(LIB, **lib)
    print(f'donor library: {len(mags)} donors, mags {lib["mag"].min():.1f}-{lib["mag"].max():.1f} -> {LIB}')
    return lib


def process_tile(det, stem, mer, device, rng, lib):
    ep, rp = tile_paths(stem, str(EUCLID), str(RUBIN))
    if not (glob.glob(ep) and glob.glob(rp)):
        return None
    ed = dict(np.load(ep, allow_pickle=True)); rd = dict(np.load(rp, allow_pickle=True))
    images0, rms, vh = build_inputs(ed, rd); H, W = vh
    vw = _wcs_vis(ed); rw = WCS(rd['wcs_hdr'].item())
    ew = {k: WCS(safe_header_from_card_string(ed[f'wcs_{k}'].item())) for k in ('VIS', 'Y', 'J', 'H')}
    cover = np.isfinite(np.asarray(ed['var_VIS'], np.float32)) & (np.asarray(ed['var_VIS'], np.float32) > 0)
    xy0, sc0 = detect_floor(det, images0, rms, vh, device, PRE_FLOOR)
    D0 = xy0[sc0 >= CONF]
    t0 = cKDTree(D0) if len(D0) else None
    t_pre = cKDTree(xy0) if len(xy0) else None
    cx, cy = vw.all_world2pix(mer['cRA'], mer['cDEC'], 0)
    ckm = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    occ = cKDTree(np.c_[cx[ckm], cy[ckm]]) if ckm.sum() else None
    LMAG = lib['mag']
    out = {m: {} for m in MODES}
    for mg in MAGS:                                     # one magnitude per pass
        injected, placed_xy = [], []
        placed = tries = 0
        while placed < N_PER_MAG and tries < 1200:
            tries += 1
            di = int(rng.integers(len(LMAG))); f = 10 ** (-0.4 * (mg - LMAG[di]))
            if f > 1:
                continue
            # margin = stamp radius: _add silently skips stamps crossing the image edge
            tx = rng.uniform(EDGE + RVIS, W - EDGE - RVIS)
            ty = rng.uniform(EDGE + RVIS, H - EDGE - RVIS)
            ix, iy = int(tx), int(ty)
            if not cover[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4].all():
                continue
            if occ is not None and occ.query([tx, ty])[0] < 10:
                continue
            if t0 is not None and t0.query([tx, ty])[0] < 8:
                continue
            if placed_xy and cKDTree(np.array(placed_xy)).query([tx, ty])[0] < SEP_PX:
                continue
            tra, tdc = vw.all_pix2world(tx, ty, 0); pend = []
            for bi, b in enumerate(EUCLID_BANDS):
                k = b.split('_', 1)[1]
                px, py = ew[k].all_world2pix(tra, tdc, 0)
                pend.append((b, int(round(float(px))), int(round(float(py))), lib['euclid'][di, bi] * f))
            for bi, b in enumerate(RUBIN_BANDS):
                px, py = rw.all_world2pix(tra, tdc, 0)
                pend.append((b, int(round(float(px))), int(round(float(py))), lib['rubin'][di, bi] * f))
            injected.append((tx, ty, pend)); placed_xy.append((tx, ty)); placed += 1
        if not injected:
            for m in MODES:
                out[m][f'{mg}'] = [0, 0, 0, 0]
            continue
        t_inj = cKDTree(np.array([(tx, ty) for tx, ty, _ in injected]))
        for mode in MODES:
            bset = _mode_bands(mode)
            imgs = {b: images0[b].copy() for b in images0}
            for (tx, ty, pend) in injected:
                for (b, a, c2, s) in pend:
                    if b in bset:
                        _add(imgs[b], a, c2, s)
            D1 = run_detect(det, imgs, rms, vh, device, CONF)
            t1 = cKDTree(D1) if len(D1) else None
            rec = art = flip = 0
            for (tx, ty, _) in injected:
                if t1 is not None and (t0 is None or t0.query([tx, ty])[0] > MATCH_PX) \
                        and t1.query([tx, ty])[0] < MATCH_PX:
                    rec += 1
            if t1 is not None:
                d_old = t0.query(D1)[0] if t0 is not None else np.full(len(D1), 1e9)
                for k in np.where(d_old > NEW_PX)[0]:
                    if t_inj.query(D1[k])[0] < MATCH_PX:
                        continue                        # the injected source itself
                    if t_pre is not None and t_pre.query(D1[k])[0] < MATCH_PX:
                        flip += 1                       # pre-existing sub-threshold peak
                    else:
                        art += 1                        # injection-induced false positive
            out[mode][f'{mg}'] = [rec, len(injected), art, flip]
    return out


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mer = load_mer(str(MER))
    stems = sorted(p.name.replace('_euclid.npz', '')
                   for p in EUCLID.glob('tile_*_patch_25_euclid.npz'))
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    if LIB.exists():
        lib = {k: v for k, v in np.load(LIB).items()}
    else:
        lib = build_donor_library(mer)
    print(f'{len(stems)} held-out tiles | {len(lib["mag"])} library donors | mags {MAGS} x '
          f'{N_PER_MAG}/tile, ONE MAG PER PASS | modes {MODES} | conf {CONF} | pre-floor {PRE_FLOOR}')
    det = None
    for i, stem in enumerate(stems):
        if stem in cache:
            continue
        if det is None:
            det = load_model('centernet', OUTD / 'centernet_vis_sep.pt', device)
        cache[stem] = process_tile(det, stem, mer, device, np.random.default_rng(11), lib)
        json.dump(cache, open(CACHE, 'w'))
        print(f'[{i + 1}/{len(stems)}] {stem}', flush=True)
    agg = {m: {f'{mg}': [0, 0, 0, 0] for mg in MAGS} for m in MODES}
    for r in cache.values():
        if not r:
            continue
        for m in MODES:
            for mg, v in r[m].items():
                for k in range(4):
                    agg[m][mg][k] += v[k]
    print(f'\n{"mode":5s} {"mag":>5s} {"rec/inj":>10s} {"artifacts":>9s} {"flips":>6s} '
          f'{"art/inj%":>9s} {"purity_inj%":>11s}')
    results = {}
    for m in MODES:
        results[m] = {}
        for mg in MAGS:
            rec, n, art, flip = agg[m][f'{mg}']
            pur = 100 * rec / (rec + art) if (rec + art) else None
            results[m][f'{mg}'] = dict(recovered=rec, injected=n, artifacts=art, flips=flip,
                                       artifact_rate=round(100 * art / max(n, 1), 2),
                                       purity_vs_injection=round(pur, 2) if pur is not None else None)
            print(f'{m:5s} {mg:>5} {rec:>5}/{n:<5} {art:>8} {flip:>6} {100*art/max(n,1):>8.2f} '
                  f'{(pur if pur is not None else float("nan")):>10.2f}')
    json.dump(results, open(OUTD / 'inject_purity_metrics.json', 'w'), indent=1)
    print('\nsaved ->', OUTD / 'inject_purity_metrics.json')


if __name__ == '__main__':
    main()
