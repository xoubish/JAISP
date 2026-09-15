"""Equal-agreement detection curves and paired spatial resampling."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..evaluation.geometry import vectors, regions

GROUPS = ('vis_bright', 'vis_all', 'nir_only', 'full_mer')


def reference_neighbours(refs, cat):
    a, b = vectors(refs['ra'], refs['dec']), vectors(cat['ra'], cat['dec'])
    radius = 2*np.sin(np.deg2rad(.5/3600)/2)
    neighbours = cKDTree(a).query_ball_tree(cKDTree(b), radius)
    # Match the existing protocol's strict distance inequality exactly.
    return [np.asarray([i for i in nn if np.linalg.norm(b[i]-a[j]) < radius], dtype=int)
            for j, nn in enumerate(neighbours)]


def ranked(scores, matched, neighbours, groups):
    order = np.argsort(-scores, kind='stable')
    sorted_scores = scores[order]
    ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])+1
    inverse = np.empty(len(order), int); inverse[order] = np.arange(len(order))+1
    first = np.asarray([inverse[nn].min() if len(nn) else len(order)+1 for nn in neighbours])
    agreement = 100*np.cumsum(matched[order])[ends-1]/ends
    completeness = np.stack([100*np.searchsorted(np.sort(first[groups[:, g]]), ends, side='right')/groups[:, g].sum()
                             for g in range(4)], -1)
    return {'order': order, 'ends': ends, 'first': first, 'agreement': agreement,
            'completeness': completeness, 'thresholds': sorted_scores[ends-1]}


def operating_point(curve, target, matched, groups):
    eligible = np.flatnonzero((curve['agreement'] >= target-1e-10) & (curve['ends'] >= 100))
    if not len(eligible): return None
    i = eligible[-1]; n = int(curve['ends'][i]); keep = curve['order'][:n]
    hits = curve['first'] <= n
    return {'threshold': float(curve['thresholds'][i]), 'detections': n, 'matched': int(matched[keep].sum()),
            'agreement_percent': float(curve['agreement'][i]),
            'completeness_percent': dict(zip(GROUPS, curve['completeness'][i].tolist())),
            'recovered': dict(zip(GROUPS, (groups & hits[:, None]).sum(0).tolist()))}


def resampled_completeness(curve, matched, groups, det_region, ref_region, weights, target):
    """Reselect the equal-agreement threshold inside every spatial replicate."""
    output = []
    order, ends, first = curve['order'], curve['ends'], curve['first']
    for start in range(0, len(weights), 100):
        w = weights[start:start+100]
        dw = w[:, det_region[order]]
        den = np.cumsum(dw, axis=1)[:, ends-1]
        num = np.cumsum(dw*matched[order][None], axis=1)[:, ends-1]
        agreement = np.divide(100*num, den, out=np.zeros(num.shape, float), where=den > 0)
        eligible = (agreement >= target-1e-10) & (den >= 100)
        cut = np.max(np.where(eligible, ends[None], 0), axis=1)
        rw = w[:, ref_region]
        found = first[None] <= cut[:, None]
        recovered = (rw*found) @ groups.astype(float)
        totals = rw @ groups.astype(float)
        values = np.divide(100*recovered, totals, out=np.full(totals.shape, np.nan), where=totals > 0)
        values[cut == 0] = np.nan
        output.append(values)
    return np.concatenate(output)


def compare(cat, refs, scores, protocol, replicates=2000):
    neighbours = reference_neighbours(refs, cat)
    curves = {name: ranked(value, cat['matched'], neighbours, refs['groups']) for name, value in scores.items()}
    anchor = cat['scores'] >= .30
    target = float(100*cat['matched'][anchor].mean())
    result = {'candidate_count': len(cat['scores']), 'reference_counts': dict(zip(GROUPS, refs['groups'].sum(0).tolist())),
              'primary_target_agreement_percent': target, 'operating_points': {}, 'paired_intervals': {}}
    targets = {'primary': target, 'secondary_94': 94.}
    for label, value in targets.items():
        points = {name: operating_point(curve, value, cat['matched'], refs['groups']) for name, curve in curves.items()}
        result['operating_points'][label] = {'target_agreement_percent': value, 'models': points}
        if all(p is not None for p in points.values()):
            result['operating_points'][label]['difference_pp'] = {g: points['rescored']['completeness_percent'][g]-points['baseline']['completeness_percent'][g] for g in GROUPS}
    for side in (4, 3):
        dr = regions(cat['ra'], cat['dec'], protocol['frame'], side)
        rr = regions(refs['ra'], refs['dec'], protocol['frame'], side)
        rng = np.random.default_rng(20260915)
        weights = rng.multinomial(side*side, np.full(side*side, 1/(side*side)), size=replicates)
        grid = {}
        for label, value in targets.items():
            draws = {name: resampled_completeness(curve, cat['matched'], refs['groups'], dr, rr, weights, value)
                     for name, curve in curves.items()}
            delta = draws['rescored']-draws['baseline']
            grid[label] = {g: {'ci95_percentile_pp': np.nanquantile(delta[:, i], [.025, .975]).tolist()
                                   if np.isfinite(delta[:, i]).mean() >= .99 else None,
                              'valid_replicates': int(np.isfinite(delta[:, i]).sum())} for i, g in enumerate(GROUPS)}
        result['paired_intervals'][f'{side}x{side}'] = grid
    primary = result['operating_points']['primary'].get('difference_pp')
    secondary = result['operating_points']['secondary_94'].get('difference_pp')
    interval = result['paired_intervals']['4x4']['primary']['nir_only']['ci95_percentile_pp']
    go = bool(primary and secondary and interval and primary['nir_only'] >= 1. and interval[0] > 0
              and primary['full_mer'] >= 0 and secondary['nir_only'] >= 0 and secondary['full_mer'] >= 0)
    result['decision'] = 'meets_predeclared_development_gate' if go else 'close_branch_keep_selected_detector'
    return result, curves, neighbours
