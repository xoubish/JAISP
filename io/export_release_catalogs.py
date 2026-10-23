#!/usr/bin/env python3
"""Export the JAISP public products (tile table, detection lists, per-band
positions) from the internal NPZ/PT files into FITS and CSV tables.

Usage (from the repo root):
    python io/export_release_catalogs.py tiles      --field ecdfs --out release/catalogs/jaisp_tiles_ecdfs.csv
    python io/export_release_catalogs.py detections --field ecdfs --out release/catalogs/jaisp_detections_ecdfs.fits
    python io/export_release_catalogs.py positions  --anchors <anchors.npz> --out release/catalogs/jaisp_positions_ecdfs.fits
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'models'))
from astrometry2.source_matching import safe_header_from_card_string  # noqa: E402

FIELDS = {
    'ecdfs': dict(rubin=ROOT / 'data/rubin_tiles_all', euclid=ROOT / 'data/euclid_tiles_all_q1',
                  labels=ROOT / 'data/detection_labels/centernet_q1_790_vissep_v11_thresh03.pt'),
    'edfs': dict(rubin=ROOT / 'data/edf_s_ood/rubin_tiles_edfs', euclid=ROOT / 'data/edf_s_ood/euclid_tiles_edfs_q1',
                 labels=ROOT / 'data/detection_labels/centernet_q1_edfs_vissep_v11_thresh03.pt'),
}
BANDS = ['u', 'g', 'r', 'i', 'z', 'y', 'nisp_Y', 'nisp_J', 'nisp_H']


def euclid_path(f: dict, stem: str) -> Path:
    return f['euclid'] / f'{stem}_euclid.npz'


def export_tiles(field: str, out: Path):
    f = FIELDS[field]
    rows = []
    for rp in sorted(glob.glob(str(f['rubin'] / 'tile_x*.npz'))):
        stem = Path(rp).stem
        r = np.load(rp, allow_pickle=True)
        e = np.load(euclid_path(f, stem), allow_pickle=True)
        rows.append(dict(tile_id=stem, ra_center=float(e['ra_center']), dec_center=float(e['dec_center']),
                         rubin_x0=int(r['x0']), rubin_y0=int(r['y0']), rubin_tile_size=int(r['tile_size']),
                         euclid_mer_tile=str(e['euclid_tile_id']),
                         euclid_cutout_px=int(e['img_VIS'].shape[0])))
    t = Table(rows=rows)
    t.meta['comments'] = ['JAISP tile grid. ra_center/dec_center are the tile centres (deg, ICRS) used to cut',
                          'the Euclid Q1 MER mosaics; rubin_x0/y0 are the pixel origins of the 512-px Rubin',
                          'DP1 deep_coadd cutouts (tract 5063 for ECDFS).']
    t.write(out, format='ascii.csv', overwrite=True, comment='# ')
    print(f'{field}: {len(t)} tiles -> {out}')


def export_detections(field: str, out: Path):
    import torch
    f = FIELDS[field]
    d = torch.load(f['labels'], weights_only=False)
    cols = {k: [] for k in ('tile_id', 'x_vis', 'y_vis', 'ra', 'dec', 'score')}
    for stem, (xy, _cls) in sorted(d['labels'].items()):
        e = np.load(euclid_path(f, stem), allow_pickle=True)
        H, W = e['img_VIS'].shape
        wcs = WCS(safe_header_from_card_string(e['wcs_VIS'].item()))
        x = xy[:, 0].astype(np.float64) * (W - 1); y = xy[:, 1].astype(np.float64) * (H - 1)
        ra, dec = wcs.wcs_pix2world(x, y, 0)
        sc = np.asarray(d['scores'][stem]).astype(np.float32)
        n = len(x)
        cols['tile_id'] += [stem] * n; cols['x_vis'] += list(x); cols['y_vis'] += list(y)
        cols['ra'] += list(ra); cols['dec'] += list(dec); cols['score'] += list(sc)
    t = Table(cols)
    t['ra'].unit = 'deg'; t['dec'].unit = 'deg'; t['x_vis'].unit = 'pix'; t['y_vis'].unit = 'pix'
    t['ra'].description = 'ICRS right ascension of the detection peak (Euclid Q1 MER VIS WCS)'
    t['dec'].description = 'ICRS declination of the detection peak'
    t['x_vis'].description = '0-based x in the tile VIS cutout (0.1 arcsec/px)'
    t['y_vis'].description = '0-based y in the tile VIS cutout'
    t['score'].description = 'CenterNet confidence (working threshold 0.30)'
    t['tile_id'].description = 'JAISP tile; sources in tile overlaps appear once per tile'
    cfg = d.get('config', {})
    hdr = fits.Header()
    hdr['PRODUCT'] = 'JAISP detection list'
    hdr['FIELD'] = field.upper()
    hdr['DETHEAD'] = 'jaisp_detection_head.pt (fused CenterNet)'
    hdr['ENCODER'] = 'jaisp_foundation.pt'
    hdr['THRESH'] = (float(cfg.get('conf_threshold', 0.3)), 'confidence threshold')
    hdr['NMSKERN'] = (int(cfg.get('nms_kernel', 7)), 'peak NMS kernel (fused px)')
    hdr['LICENSE'] = 'CC BY 4.0'
    hdr['REFERENC'] = 'Hemmati et al. 2026, ApJ, JAISP'
    t.meta.update({k: hdr[k] for k in hdr})
    t.write(out, format='fits', overwrite=True)
    print(f'{field}: {len(t)} detections on {len(d["labels"])} tiles -> {out}')


def export_positions(anchors: Path, out: Path, field: str):
    d = np.load(anchors, allow_pickle=True)
    parts = []
    for b in BANDS:
        if f'{b}_ra' not in d.files:
            continue
        ra = np.asarray(d[f'{b}_ra'], dtype=np.float64); dec = np.asarray(d[f'{b}_dec'], dtype=np.float64)
        raw = np.asarray(d[f'{b}_raw'], dtype=np.float64)          # VIS - band, arcsec (dRA*cos(dec), dDec)
        res = np.asarray(d[f'{b}_head_resid'], dtype=np.float64)   # VIS - head-corrected, arcsec
        corr = raw - res                                            # head correction applied to the band position
        cosd = np.cos(np.deg2rad(dec))
        t = Table()
        t['band'] = np.full(len(ra), b.replace('nisp_', 'NISP_'))
        t['tile_id'] = np.asarray(d[f'{b}_tiles']).astype(str)
        t['ra_band'] = ra; t['dec_band'] = dec
        t['ra_head'] = ra + corr[:, 0] / 3600.0 / cosd; t['dec_head'] = dec + corr[:, 1] / 3600.0
        t['ra_vis'] = ra + raw[:, 0] / 3600.0 / cosd;  t['dec_vis'] = dec + raw[:, 1] / 3600.0
        t['dra_raw'] = raw[:, 0] * 1000; t['ddec_raw'] = raw[:, 1] * 1000
        t['dra_head'] = res[:, 0] * 1000; t['ddec_head'] = res[:, 1] * 1000
        t['sigma_head'] = np.asarray(d[f'{b}_sigma'], dtype=np.float32) * 1000
        t['snr_band'] = np.asarray(d[f'{b}_snr'], dtype=np.float32)
        parts.append(t)
    from astropy.table import vstack
    t = vstack(parts)
    for c in ('ra_band', 'dec_band', 'ra_head', 'dec_head', 'ra_vis', 'dec_vis'):
        t[c].unit = 'deg'
    for c in ('dra_raw', 'ddec_raw', 'dra_head', 'ddec_head', 'sigma_head'):
        t[c].unit = 'mas'
    t['ra_band'].description = 'classical Gaussian centroid in this band, projected to the VIS frame'
    t['ra_head'].description = 'band position after the JAISP astrometry-head correction'
    t['ra_vis'].description = 'VIS Gaussian centroid of the matched source (training label convention)'
    t['dra_raw'].description = '(VIS - band) * cos(dec), before correction'
    t['dra_head'].description = '(VIS - head-corrected band) * cos(dec)'
    t['sigma_head'].description = 'head-predicted per-axis uncertainty (uncalibrated; see paper)'
    t['snr_band'].description = 'aperture S/N of the source in this band'
    t.meta.update(dict(PRODUCT='JAISP per-band positions', FIELD=field.upper(), ASTHEAD='jaisp_astrometry_head.pt',
                       ENCODER='jaisp_foundation.pt', SIGNCONV='offsets are VIS minus band (Eq. 2 of the paper)',
                       LICENSE='CC BY 4.0', REFERENC='Hemmati et al. 2026, ApJ, JAISP'))
    t.write(out, format='fits', overwrite=True)
    print(f'{field}: {len(t)} band-positions -> {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('what', choices=['tiles', 'detections', 'positions'])
    p.add_argument('--field', choices=list(FIELDS), default='ecdfs')
    p.add_argument('--anchors', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.what == 'tiles': export_tiles(a.field, a.out)
    elif a.what == 'detections': export_detections(a.field, a.out)
    else: export_positions(a.anchors, a.out, a.field)
