"""Query the MER catalogue around each tile and position-match it to the detection-head sources."""
import argparse
import sys
import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.spatial import cKDTree
from .common import ROOT, OUT, EUCLID, MATCH_RADIUS_ARCSEC, SCENE_HALF_ARCSEC, region_dir, tangent_arcsec, read_json, write_json
from .prepare import load_inputs
from ..data import wcs
sys.path.insert(0, str(ROOT / 'models/photometry/self_supervised/vendor/euclid_forced_photometry/src'))
from euclid_phot.catalog import _ADQL  # noqa: E402

MER_COLUMNS = ['object_id', 'ra', 'dec', 'flux_vis_psf', 'fluxerr_vis_psf', 'flux_vis_sersic', 'fluxerr_vis_sersic',
               'flux_y_templfit', 'fluxerr_y_templfit', 'flux_j_templfit', 'fluxerr_j_templfit', 'flux_h_templfit', 'fluxerr_h_templfit',
               'flux_y_sersic', 'fluxerr_y_sersic', 'flux_j_sersic', 'fluxerr_j_sersic', 'flux_h_sersic', 'fluxerr_h_sersic',
               'flux_detection_total', 'semimajor_axis', 'ellipticity', 'position_angle', 'point_like_prob', 'det_quality_flag',
               'sersic_sersic_vis_radius', 'sersic_sersic_vis_index', 'sersic_sersic_vis_axis_ratio', 'is_star']


def query_region(folder, margin_arcsec=20.):
    path = folder / 'mer.fits'
    if path.exists(): return Table.read(path)
    from astroquery.ipac.irsa import Irsa
    inputs = load_inputs(folder); wc = wcs(inputs['euclid_VIS__wcs']); h, w = inputs['euclid_VIS__image'].shape
    ra, dec = wc.pixel_to_world_values([0, w - 1, 0, w - 1], [0, 0, h - 1, h - 1])
    m = margin_arcsec / 3600; mra = m / np.cos(np.deg2rad(np.mean(dec)))
    query = _ADQL.format(ra_where=f'm.ra BETWEEN {ra.min() - mra} AND {ra.max() + mra}', dec_min=dec.min() - m, dec_max=dec.max() + m, flux_min=0, extra_where='')
    cat = Irsa.query_tap(query, maxrec=100000).to_table()
    if len(cat) >= 100000: raise ValueError('Truncated MER query')
    cat['is_star'] = np.ma.filled(cat['point_like_prob'], 0) > .96
    cat.write(path); (folder / 'mer_query.adql').write_text(query)
    return cat


def match_sky(det_sky, ref_sky, center, radius=MATCH_RADIUS_ARCSEC):
    """Nearest reference within radius for every detection, in the tangent plane about center.

    Returns (index or -1, separation, second-nearest separation, count within 1 arcsec)."""
    d = tangent_arcsec(det_sky, center); r = tangent_arcsec(ref_sky, center)
    if not len(r): n = len(d); return np.full(n, -1), np.full(n, np.inf), np.full(n, np.inf), np.zeros(n, int)
    tree = cKDTree(r); k = min(2, len(r))
    distance, index = tree.query(d, k=k); distance = np.atleast_2d(distance.T).T if k == 1 else distance; index = np.atleast_2d(index.T).T if k == 1 else index
    nearest = np.where(distance[:, 0] <= radius, index[:, 0], -1)
    second = distance[:, 1] if k > 1 else np.full(len(d), np.inf)
    within = np.array([len(x) for x in tree.query_ball_point(d, 1.)])
    return nearest, distance[:, 0], second, within


def match_region(folder):
    inputs = load_inputs(folder); cat = query_region(folder); meta = read_json(folder / 'metadata.json')
    det_sky = inputs['sky']; ref_sky = np.column_stack((np.asarray(cat['ra'], float), np.asarray(cat['dec'], float)))
    center = (meta['ra'], meta['dec'])
    nearest, sep, second, within = match_sky(det_sky, ref_sky, center)
    frame = pd.DataFrame(dict(region=meta['region'], source=inputs['source'], ra=det_sky[:, 0], dec=det_sky[:, 1],
                              matched=nearest >= 0, match_sep_arcsec=sep, second_mer_sep_arcsec=second, n_mer_within_1arcsec=within))
    table = cat[MER_COLUMNS].to_pandas()
    for column in MER_COLUMNS:
        values = table[column].to_numpy(); fill = -1 if column == 'object_id' else (False if column == 'is_star' else np.nan)
        out = np.full(len(frame), fill, dtype=object if column == 'is_star' else float)
        out[nearest >= 0] = values[nearest[nearest >= 0]]
        frame['mer_' + column] = out.astype(bool) if column == 'is_star' else out.astype('int64' if column == 'object_id' else float)
    # Duplicate matches: several detections claiming one MER object (keep the nearest as primary).
    frame['primary_match'] = False
    for oid, group in frame[frame.matched].groupby('mer_object_id'):
        frame.loc[group.match_sep_arcsec.idxmin(), 'primary_match'] = True
    frame.to_csv(folder / 'mer_match.csv', index=False)
    # Completeness: MER sources inside the photometrable footprint (same edge rule as the scenes).
    wc = wcs(inputs['euclid_VIS__wcs']); h, w = inputs['euclid_VIS__image'].shape; radius = int(np.ceil(SCENE_HALF_ARCSEC / .1))
    xy = np.column_stack(wc.world_to_pixel_values(*ref_sky.T))
    inside = (xy[:, 0] >= radius) & (xy[:, 0] <= w - 1 - radius) & (xy[:, 1] >= radius) & (xy[:, 1] <= h - 1 - radius)
    rw = wcs(inputs['rubin__wcs']); rh, rwid = inputs['rubin__image'].shape[1:]; rr = int(np.ceil(1.5 / inputs['rubin__scale_arcsec']))
    rxy = np.column_stack(rw.world_to_pixel_values(*ref_sky.T))
    inside &= (rxy[:, 0] >= rr) & (rxy[:, 0] <= rwid - 1 - rr) & (rxy[:, 1] >= rr) & (rxy[:, 1] <= rh - 1 - rr)
    back, back_sep, _, _ = match_sky(ref_sky, det_sky, center)
    completeness = table.copy(); completeness.insert(0, 'region', meta['region'])
    completeness['in_footprint'] = inside; completeness['detected'] = back >= 0; completeness['detection_sep_arcsec'] = back_sep
    completeness['detection_source'] = np.where(back >= 0, inputs['source'][np.maximum(back, 0)], -1)
    completeness.to_csv(folder / 'mer_completeness.csv', index=False)
    return dict(region=meta['region'], detections=len(frame), matched=int(frame.matched.sum()), mer_in_footprint=int(inside.sum()),
                mer_detected=int((inside & (back >= 0)).sum()))


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--regions', type=int, nargs='*'); a = p.parse_args()
    folders = sorted(f.parent for f in OUT.glob('region_*/tile_inputs.npz'))
    if a.regions is not None: folders = [f for f in folders if int(f.name.split('_')[1]) in set(a.regions)]
    status = []
    for f in folders:
        status.append(match_region(f)); print(status[-1], flush=True)
    write_json(OUT / 'mer_status.json', status)


if __name__ == '__main__': main()
