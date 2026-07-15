"""Reusable detection-validation utilities (MER catalogue match, injection-recovery).

These back the plots in `io/17_detection_validation.ipynb`. Everything here operates on
the FROZEN detector (no training) so the notebook cells stay short and the logic is
committed/reviewable rather than living in throwaway scripts.

Conventions
-----------
- Detector detections and injected positions are in VIS pixels.
- MER VIS magnitude = 23.9 - 2.5*log10(flux_detection_total[uJy]).
- MER catalogue is Euclid Q1; imaging is likely DR1 (same sky, different release).
"""
from __future__ import annotations
import glob
import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree

from jaisp_foundation_v10 import RUBIN_BANDS, EUCLID_BANDS
from detection.centernet_detector import CenterNetDetector
from detection.detector import JAISPEncoderWrapper
from load_foundation import load_foundation
from astrometry2.source_matching import safe_header_from_card_string

PXE = 0.1  # VIS arcsec/px

# ----------------------------------------------------------------------------- detector
def load_detector(centernet_ckpt: str, encoder_ckpt: str, device=None):
    """Load the frozen v10 foundation + a CenterNet head checkpoint."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    foundation = load_foundation(encoder_ckpt, device=torch.device('cpu'), freeze=True)
    enc = JAISPEncoderWrapper(foundation, freeze=True).to(device).eval()
    det = CenterNetDetector.load(centernet_ckpt, encoder=enc, device=device).eval()
    return det, device


def build_inputs(ed, rd):
    """10-band image+rms dicts (var-based rms, matching production) from tile NPZs."""
    images, rms = {}, {}
    rimg = np.asarray(rd['img'], np.float32); rvar = np.asarray(rd['var'], np.float32)
    for bi, b in enumerate(RUBIN_BANDS):
        images[b] = np.nan_to_num(rimg[bi])
        rms[b] = np.maximum(np.sqrt(np.clip(rvar[bi], 1e-12, None)), 1e-10)
    vis_hw = None
    for b in EUCLID_BANDS:
        k = b.split('_', 1)[1]
        images[b] = np.nan_to_num(np.asarray(ed[f'img_{k}'], np.float32))
        var = np.nan_to_num(np.asarray(ed[f'var_{k}'], np.float32), nan=1.0)
        rms[b] = np.maximum(np.sqrt(np.clip(var, 1e-12, None)), 1e-10)
        if k == 'VIS':
            vis_hw = images[b].shape
    return images, rms, vis_hw


def run_detect(det, images, rms, vis_hw, device, conf=0.30):
    """Run the detector; return detection centroids in VIS pixels [N,2]."""
    im = {b: torch.from_numpy(images[b][None, None].copy()).to(device) for b in images}
    rm = {b: torch.from_numpy(rms[b][None, None].copy()).to(device) for b in rms}
    with torch.no_grad():
        res = det.predict(im, rm, conf_threshold=conf, tile_hw=vis_hw, nms_kernel=7, artifact_mask=None)
    c = res['centroids'].cpu().numpy(); H, W = vis_hw
    return np.c_[c[:, 0] * (W - 1), c[:, 1] * (H - 1)]


# ----------------------------------------------------------------------------- MER catalogue
def load_mer(fits_path):
    """Return dict of clean (vis_det & !spurious) and full RA/Dec + mag from a MER footprint FITS."""
    cat = fits.open(fits_path)[1].data
    clean = (np.asarray(cat['vis_det']) == 1) & (np.asarray(cat['spurious_flag']) != 1)
    return dict(
        cRA=np.asarray(cat['ra'], float)[clean], cDEC=np.asarray(cat['dec'], float)[clean],
        cMAG=np.asarray(cat['mag_vis'], float)[clean],
        fRA=np.asarray(cat['ra'], float), fDEC=np.asarray(cat['dec'], float),
    )


def _wcs_vis(ed):
    return WCS(safe_header_from_card_string(ed['wcs_VIS'].item()))


def tile_paths(stem, euclid_dir, rubin_dir):
    return f'{euclid_dir}/{stem}_euclid.npz', f'{rubin_dir}/{stem}.npz'


def eval_mer(det, stems, mer, euclid_dir, rubin_dir, device, conf=0.30, rad_as=0.5, margin=4):
    """Per-tile match of detections to the MER catalogue.

    Returns (mag, hit, purity, offsets_mas): completeness ref = clean MER; purity vs full MER.
    """
    rpx = rad_as / PXE
    mag_all, hit_all, offs = [], [], []
    our_tot = our_hit = 0
    for stem in stems:
        ep, rp = tile_paths(stem, euclid_dir, rubin_dir)
        if not (glob.glob(ep) and glob.glob(rp)):
            continue
        ed = dict(np.load(ep, allow_pickle=True)); rd = dict(np.load(rp, allow_pickle=True))
        images, rms, vh = build_inputs(ed, rd); H, W = vh
        vw = _wcs_vis(ed)
        D = run_detect(det, images, rms, vh, device, conf)
        cx, cy = vw.all_world2pix(mer['cRA'], mer['cDEC'], 0)
        ck = (cx >= margin) & (cx < W - margin) & (cy >= margin) & (cy < H - margin)
        fx, fy = vw.all_world2pix(mer['fRA'], mer['fDEC'], 0)
        fk = (fx >= margin) & (fx < W - margin) & (fy >= margin) & (fy < H - margin)
        if ck.sum() < 2 or len(D) < 2:
            continue
        d_c, _ = cKDTree(D).query(np.c_[cx[ck], cy[ck]])
        mag_all.append(mer['cMAG'][ck]); hit_all.append(d_c < rpx)
        d_o, _ = cKDTree(np.c_[fx[fk], fy[fk]]).query(D)
        our_tot += len(D); our_hit += int((d_o < rpx).sum())
        offs.extend((d_o[d_o < rpx] * PXE * 1000).tolist())
    mag = np.concatenate(mag_all); hit = np.concatenate(hit_all)
    fin = np.isfinite(mag)
    return mag[fin], hit[fin], (our_hit / max(our_tot, 1)), np.asarray(offs)


def completeness_curve(mag, hit, edges):
    cen, comp, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mag >= lo) & (mag < hi)
        if m.sum() < 20:
            continue
        cen.append((lo + hi) / 2); comp.append(100 * hit[m].mean()); n.append(int(m.sum()))
    return np.array(cen), np.array(comp), np.array(n)


# ----------------------------------------------------------------------------- injection-recovery
def compact_donors(ed, vw, mer, edge, rvis, conc_min=0.6):
    """Indices into mer['cRA'] for compact (point-source-like) donors inside this tile."""
    H, W = np.asarray(ed['img_VIS']).shape
    dx, dy = vw.all_world2pix(mer['cRA'], mer['cDEC'], 0)
    cand = np.where((dx > edge + rvis) & (dx < W - edge - rvis) & (dy > edge + rvis) & (dy < H - edge - rvis))[0]
    vis = np.nan_to_num(np.asarray(ed['img_VIS'], np.float32)); out = []
    for j in cand:
        a, b = int(round(float(dx[j]))), int(round(float(dy[j])))
        s = vis[b - 5:b + 6, a - 5:a + 6] - np.median(vis[b - 5:b + 6, a - 5:a + 6])
        tot = s[s > 0].sum()
        if tot > 0 and s[3:8, 3:8][s[3:8, 3:8] > 0].sum() / tot > conc_min:
            out.append(j)
    return np.array(out)


def _stamp(arr, wcs, ra, dc, rpx):
    px, py = wcs.all_world2pix(ra, dc, 0); px, py = int(round(float(px))), int(round(float(py)))
    H, W = arr.shape
    if px - rpx < 0 or px + rpx + 1 > W or py - rpx < 0 or py + rpx + 1 > H:
        return None
    s = arr[py - rpx:py + rpx + 1, px - rpx:px + rpx + 1].astype(np.float32).copy()
    bg = np.median(np.concatenate([s[0], s[-1], s[:, 0], s[:, -1]]))
    return s - bg


def _add(arr, cx, cy, stamp):
    r = stamp.shape[0] // 2; H, W = arr.shape
    if cx - r < 0 or cy - r < 0 or cx + r + 1 > W or cy + r + 1 > H:
        return False
    arr[cy - r:cy + r + 1, cx - r:cx + r + 1] += stamp; return True


def _mode_bands(mode):
    """Which bands an injection mode places the source into.

    'all'   -> all 10 bands (full multi-instrument source)
    'vis'   -> Euclid VIS only
    'nisp'  -> Euclid NIR Y/J/H only (high-z dropout: gone from optical, present in NIR)
    'rubin' -> Rubin u..y only (optical-only)
    """
    allb = set(RUBIN_BANDS) | set(EUCLID_BANDS)
    return {
        'all': allb,
        'vis': {'euclid_VIS'},
        'nisp': {'euclid_Y', 'euclid_J', 'euclid_H'},
        'rubin': set(RUBIN_BANDS),
    }.get(mode, allb)


def eval_injection(det, stems, mer, euclid_dir, rubin_dir, device, modes=('all',),
                   target_mags=(22.5, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5),
                   n_per_mag=10, conf=0.30, rad_as=0.5, match_px=3.0, edge=24, rvis=8, seed=3,
                   donor_mag=(19.5, 21.5), donor_conc=None):
    """Source-recycling injection-recovery completeness vs VIS mag.

    For each mode ('all' = inject in all 10 bands; 'vis' = VIS only), inject real
    isolated donors scaled to target mags at empty good-coverage positions and measure
    recovery by the frozen detector. Returns {mode: {mag: (recovered, injected)}}.

    donor_mag bounds the donor pool in MER VIS magnitude. donor_conc, if set,
    additionally requires the donor's VIS concentration f(<0.3'')/f(<1.5'') to
    exceed it, restricting to unresolved (star-like) donors: the default pool at
    19.5-21.5 is dominated by compact GALAXIES (median concentration ~0.3 vs
    ~0.6 for the real faint population; io/22), so dimming it measures
    extended-morphology completeness, not point-source depth.
    """
    rng = np.random.default_rng(seed)
    # donor pool: bright, isolated
    sky = cKDTree(np.c_[mer['cRA'], mer['cDEC']]); nn2 = sky.query(np.c_[mer['cRA'], mer['cDEC']], k=2)[0][:, 1]
    don = (mer['cMAG'] > donor_mag[0]) & (mer['cMAG'] < donor_mag[1]) & (nn2 > 3 / 3600.)
    yy_, xx_ = np.mgrid[-30:31, -30:31]; RR_ = np.hypot(xx_, yy_)
    DRA, DDEC, DMAG = mer['cRA'][don], mer['cDEC'][don], mer['cMAG'][don]
    rec = {m: {mg: [0, 0] for mg in target_mags} for m in modes}
    for stem in stems:
        ep, rp = tile_paths(stem, euclid_dir, rubin_dir)
        if not (glob.glob(ep) and glob.glob(rp)):
            continue
        ed = dict(np.load(ep, allow_pickle=True)); rd = dict(np.load(rp, allow_pickle=True))
        images0, rms, vh = build_inputs(ed, rd); H, W = vh
        vw = _wcs_vis(ed); rw = WCS(rd['wcs_hdr'].item())
        ew = {k: WCS(safe_header_from_card_string(ed[f'wcs_{k}'].item())) for k in ('VIS', 'Y', 'J', 'H')}
        cover = np.isfinite(np.asarray(ed['var_VIS'], np.float32)) & (np.asarray(ed['var_VIS'], np.float32) > 0)
        D0 = run_detect(det, images0, rms, vh, device, conf); t0 = cKDTree(D0) if len(D0) else None
        # donors inside this tile, and occupancy for empty-region rejection
        dx, dy = vw.all_world2pix(DRA, DDEC, 0)
        din = np.where((dx > edge + rvis) & (dx < W - edge - rvis) & (dy > edge + rvis) & (dy < H - edge - rvis))[0]
        cx, cy = vw.all_world2pix(mer['cRA'], mer['cDEC'], 0); ckm = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        occ = cKDTree(np.c_[cx[ckm], cy[ckm]]) if ckm.sum() else None
        # filter donors to compact ones
        vis = np.nan_to_num(np.asarray(ed['img_VIS'], np.float32)); cdon = []
        for j in din:
            a, b = int(round(float(dx[j]))), int(round(float(dy[j])))
            s = vis[b - 5:b + 6, a - 5:a + 6] - np.median(vis[b - 5:b + 6, a - 5:a + 6]); tot = s[s > 0].sum()
            if not (tot > 0 and s[3:8, 3:8][s[3:8, 3:8] > 0].sum() / tot > 0.6):
                continue
            if donor_conc is not None:
                st = vis[b - 30:b + 31, a - 30:a + 31]
                if st.shape != (61, 61):
                    continue
                st = st - np.median(np.concatenate([st[0], st[-1], st[:, 0], st[:, -1]]))
                f15 = st[RR_ <= 15].sum()
                if f15 <= 0 or st[RR_ <= 3].sum() / f15 < donor_conc:
                    continue
            cdon.append(j)
        if len(cdon) < 2:
            continue
        cdon = np.array(cdon)
        # build injection list (shared across modes)
        injected = []
        for mg in target_mags:
            placed = tries = 0
            while placed < n_per_mag and tries < 800:
                tries += 1
                di = int(rng.choice(cdon)); f = 10 ** (-0.4 * (mg - DMAG[di]))
                if f > 1:
                    continue
                tx = rng.uniform(edge, W - edge); ty = rng.uniform(edge, H - edge); ix, iy = int(tx), int(ty)
                if not cover[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4].all():
                    continue
                if occ is not None and occ.query([tx, ty])[0] < 10:
                    continue
                if t0 is not None and t0.query([tx, ty])[0] < 8:
                    continue
                tra, tdc = vw.all_pix2world(tx, ty, 0); ok = True; pend = []
                for b in EUCLID_BANDS:
                    k = b.split('_', 1)[1]
                    st = _stamp(np.nan_to_num(np.asarray(ed[f'img_{k}'], np.float32)), ew[k], DRA[di], DDEC[di], rvis)
                    if st is None:
                        ok = False; break
                    px, py = ew[k].all_world2pix(tra, tdc, 0)
                    pend.append((b, int(round(float(px))), int(round(float(py))), st * f))
                if not ok:
                    continue
                for bi, b in enumerate(RUBIN_BANDS):
                    st = _stamp(np.asarray(rd['img'], np.float32)[bi], rw, DRA[di], DDEC[di], rvis // 2)
                    if st is None:
                        ok = False; break
                    px, py = rw.all_world2pix(tra, tdc, 0)
                    pend.append((b, int(round(float(px))), int(round(float(py))), st * f))
                if not ok:
                    continue
                injected.append((tx, ty, mg, pend)); placed += 1
        for mode in modes:
            bset = _mode_bands(mode)
            imgs = {b: images0[b].copy() for b in images0}
            for (tx, ty, mg, pend) in injected:
                for (b, a, c2, s) in pend:
                    if b in bset:
                        _add(imgs[b], a, c2, s)
            D1 = run_detect(det, imgs, rms, vh, device, conf); t1 = cKDTree(D1) if len(D1) else None
            for (tx, ty, mg, pend) in injected:
                rec[mode][mg][1] += 1
                new = (t0 is None or t0.query([tx, ty])[0] > match_px)
                if new and t1 is not None and t1.query([tx, ty])[0] < match_px:
                    rec[mode][mg][0] += 1
    return rec


def rec_to_curve(rec_mode, target_mags):
    return np.array(target_mags), np.array([100 * rec_mode[m][0] / max(rec_mode[m][1], 1) for m in target_mags])


# ----------------------------------------------------------------------------- miss / extra anatomy
def _ap_snr(img, var, x, y, r):
    """Background-subtracted aperture flux / noise at pixel (x, y); r in px. NaN if off-image."""
    H, W = img.shape; xi, yi = int(round(x)), int(round(y)); R = int(r) + 2
    if xi - R < 0 or yi - R < 0 or xi + R + 1 > W or yi + R + 1 > H:
        return np.nan
    yy, xx = np.mgrid[yi - R:yi + R + 1, xi - R:xi + R + 1]
    rr = np.hypot(xx - x, yy - y)
    sub = np.nan_to_num(img[yi - R:yi + R + 1, xi - R:xi + R + 1].astype(np.float32))
    vsub = np.nan_to_num(var[yi - R:yi + R + 1, xi - R:xi + R + 1].astype(np.float32), nan=0.0)
    disk = rr <= r; ann = (rr > r) & (rr <= R)
    if disk.sum() < 1 or ann.sum() < 3:
        return np.nan
    bg = np.median(sub[ann])
    flux = float((sub[disk] - bg).sum())
    noise = float(np.sqrt(max(vsub[disk].sum(), 1e-12)))
    return flux / noise


def per_band_snr(ed, rd, wcs_vis, xy_vis, ap_arcsec=0.3):
    """For positions in VIS px, aperture SNR in every band (project via per-band WCS).

    Returns dict band -> array[N] of aperture SNR. Bands: rubin u..y, euclid VIS/Y/J/H.
    """
    if len(xy_vis) == 0:
        return {b: np.zeros(0) for b in list(RUBIN_BANDS) + list(EUCLID_BANDS)}
    ra, dec = wcs_vis.all_pix2world(xy_vis[:, 0], xy_vis[:, 1], 0)
    out = {}
    rcube = np.asarray(rd['img'], np.float32); rvar = np.asarray(rd['var'], np.float32)
    rw = WCS(rd['wcs_hdr'].item())
    rx, ry = rw.all_world2pix(ra, dec, 0)
    for bi, b in enumerate(RUBIN_BANDS):
        out[b] = np.array([_ap_snr(rcube[bi], rvar[bi], rx[i], ry[i], ap_arcsec / 0.2) for i in range(len(xy_vis))])
    for b in EUCLID_BANDS:
        k = b.split('_', 1)[1]
        img = np.asarray(ed[f'img_{k}'], np.float32); var = np.asarray(ed[f'var_{k}'], np.float32)
        ew = WCS(safe_header_from_card_string(ed[f'wcs_{k}'].item()))
        ex, ey = ew.all_world2pix(ra, dec, 0)
        out[b] = np.array([_ap_snr(img, var, ex[i], ey[i], ap_arcsec / PXE) for i in range(len(xy_vis))])
    return out


def analyze_mer(det, stems, mer, euclid_dir, rubin_dir, device, conf=0.30, rad_as=0.5,
                margin=8, n_null=60, n_stamps=12, seed=1):
    """Anatomy of detection vs MER: per-band SNR of matched detections, unmatched ('extra')
    detections, missed MER sources, and random 'null' positions; plus example stamps.

    Returns a dict of stacked arrays across the tile list. SNR keys are per band; 'visdet'
    is each MER source's VIS-detection flag (1 = MER detected it in VIS, 0 = NIR-only).
    """
    rng = np.random.default_rng(seed); rpx = rad_as / PXE
    BANDS = list(RUBIN_BANDS) + list(EUCLID_BANDS)
    acc = {k: {b: [] for b in BANDS} for k in ('matched', 'extra', 'missed', 'null')}
    extra_mag_proxy = []  # not available without phot; placeholder
    stamps = {'extra': [], 'missed': []}  # (vis_cutout, label)
    fullRA, fullDEC = mer['fRA'], mer['fDEC']
    for stem in stems:
        ep, rp = tile_paths(stem, euclid_dir, rubin_dir)
        if not (glob.glob(ep) and glob.glob(rp)):
            continue
        ed = dict(np.load(ep, allow_pickle=True)); rd = dict(np.load(rp, allow_pickle=True))
        images, rms, vh = build_inputs(ed, rd); H, W = vh
        vw = _wcs_vis(ed)
        D = run_detect(det, images, rms, vh, device, conf)
        if len(D) < 2:
            continue
        # MER (full + clean) in this tile, in VIS px
        fx, fy = vw.all_world2pix(fullRA, fullDEC, 0)
        fk = (fx >= margin) & (fx < W - margin) & (fy >= margin) & (fy < H - margin)
        fmer = np.c_[fx[fk], fy[fk]]
        cx, cy = vw.all_world2pix(mer['cRA'], mer['cDEC'], 0)
        ck = (cx >= margin) & (cx < W - margin) & (cy >= margin) & (cy < H - margin)
        cmer = np.c_[cx[ck], cy[ck]]
        din = (D[:, 0] >= margin) & (D[:, 0] < W - margin) & (D[:, 1] >= margin) & (D[:, 1] < H - margin)
        D = D[din]
        if len(fmer) < 2 or len(D) < 2:
            continue
        # detection -> nearest full-MER: matched vs extra
        d_d, _ = cKDTree(fmer).query(D); matched = d_d < rpx
        # clean-MER -> nearest detection: recovered vs missed
        d_m, _ = cKDTree(D).query(cmer); missed = d_m >= rpx
        # null: random empty positions (far from any MER and any detection)
        occ = cKDTree(np.vstack([fmer, D])); nulls = []
        tries = 0
        while len(nulls) < n_null and tries < n_null * 40:
            tries += 1
            p = [rng.uniform(margin, W - margin), rng.uniform(margin, H - margin)]
            if occ.query(p)[0] > 12:
                nulls.append(p)
        nulls = np.array(nulls) if nulls else np.zeros((0, 2))
        # per-band SNR
        for key, pts in (('matched', D[matched]), ('extra', D[~matched]),
                         ('missed', cmer[missed]), ('null', nulls)):
            snr = per_band_snr(ed, rd, vw, pts)
            for b in BANDS:
                acc[key][b].append(snr[b])
        # stamps of a few extras and missed (VIS cutout)
        vis = np.nan_to_num(np.asarray(ed['img_VIS'], np.float32))
        for key, pts in (('extra', D[~matched]), ('missed', cmer[missed])):
            for p in pts[:max(0, n_stamps - len(stamps[key]))]:
                xi, yi = int(p[0]), int(p[1])
                if xi - 15 >= 0 and yi - 15 >= 0 and xi + 16 <= W and yi + 16 <= H:
                    stamps[key].append(vis[yi - 15:yi + 16, xi - 15:xi + 16].copy())
    res = {k: {b: (np.concatenate(v) if v else np.zeros(0)) for b, v in acc[k].items()} for k in acc}
    res['stamps'] = stamps
    return res

