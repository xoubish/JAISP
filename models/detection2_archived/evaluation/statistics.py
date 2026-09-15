"""Paired percentile intervals from resampled contiguous sky blocks.

These are approximate, within-patch intervals, conditional on the trained
checkpoints and catalogue. They do not quantify variation between training
seeds, sky fields, catalogue errors, or multiple threshold selection.
"""
from __future__ import annotations

import numpy as np

GROUPS = ('vis_bright', 'vis_all', 'nir_only', 'full_mer')
METRICS = tuple(g+'_completeness' for g in GROUPS) + ('mer_match_fraction',)
ARMS = ('starting', 'control', 'masked')
LABELS = ('Starting model', 'Standard loss', 'Masked background loss')
PAIRS = ((1, 0), (2, 0), (2, 1))
THRESHOLDS = (.15, .20, .25, .30, .35, .40, .50, .60, .70, .80, .90)


def block_counts(hits, groups, reference_regions, detections, detection_regions, side):
    """Numerators and denominators: model × threshold × metric × sky region."""
    nblocks = side*side
    shape = (len(ARMS), len(THRESHOLDS), len(METRICS), nblocks)
    numer, denom = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
    for m in range(len(ARMS)):
        for t, threshold in enumerate(THRESHOLDS):
            for g in range(len(GROUPS)):
                select = groups[:, g]
                denom[m, t, g] = np.bincount(reference_regions[select], minlength=nblocks)
                numer[m, t, g] = np.bincount(reference_regions[select & hits[m, t]], minlength=nblocks)
            det = detections[m]
            select = det['scores'] >= threshold
            denom[m, t, -1] = np.bincount(detection_regions[m][select], minlength=nblocks)
            numer[m, t, -1] = np.bincount(detection_regions[m][select & det['matched']], minlength=nblocks)
    return numer, denom


def bootstrap(numer, denom, replicates=5000, seed=20260915):
    if numer.shape != denom.shape or np.any(numer < 0) or np.any(numer > denom):
        raise ValueError('Invalid binomial counts')
    nblocks = numer.shape[-1]
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(nblocks, np.full(nblocks, 1/nblocks), size=replicates)
    n = np.einsum('mtkb,rb->mtkr', numer, weights, optimize=True)
    d = np.einsum('mtkb,rb->mtkr', denom, weights, optimize=True)
    draws = np.divide(100*n, d, out=np.full(n.shape, np.nan), where=d > 0)
    total_n, total_d = numer.sum(-1), denom.sum(-1)
    point = np.divide(100*total_n, total_d, out=np.full(total_n.shape, np.nan), where=total_d > 0)
    return point, draws, weights


def interval(values):
    valid = np.isfinite(values)
    if valid.mean() < .99:
        return {'ci95_percentile': None, 'valid_replicates': int(valid.sum())}
    return {'ci95_percentile': np.quantile(values[valid], [.025, .975]).tolist(),
            'valid_replicates': int(valid.sum())}


def summarize(numer, denom, replicates=5000, seed=20260915):
    point, draws, weights = bootstrap(numer, denom, replicates, seed)
    metrics, differences = {}, {}
    for m, arm in enumerate(ARMS):
        metrics[arm] = {}
        for t, threshold in enumerate(THRESHOLDS):
            metrics[arm][str(threshold)] = {
                key: {'percent': float(point[m, t, k]) if np.isfinite(point[m, t, k]) else None,
                      'numerator': int(numer[m, t, k].sum()), 'denominator': int(denom[m, t, k].sum()),
                      **interval(draws[m, t, k])} for k, key in enumerate(METRICS)}
    for a, b in PAIRS:
        pair = ARMS[a]+'_minus_'+ARMS[b]
        differences[pair] = {}
        for t, threshold in enumerate(THRESHOLDS):
            differences[pair][str(threshold)] = {
                key: {'difference_pp': float(point[a, t, k]-point[b, t, k])
                      if np.isfinite(point[a, t, k]-point[b, t, k]) else None,
                      **interval(draws[a, t, k]-draws[b, t, k])}
                for k, key in enumerate(METRICS)}
    return {'metrics': metrics, 'paired_differences': differences}, weights


def discordances(hits, groups):
    result = {}
    for a, b in PAIRS:
        result[ARMS[a]+'_minus_'+ARMS[b]] = {}
        for t, threshold in enumerate(THRESHOLDS):
            result[ARMS[a]+'_minus_'+ARMS[b]][str(threshold)] = {}
            for g, group in enumerate(GROUPS):
                aa, bb = hits[a, t, groups[:, g]], hits[b, t, groups[:, g]]
                result[ARMS[a]+'_minus_'+ARMS[b]][str(threshold)][group] = {
                    'both': int((aa & bb).sum()), 'only_first': int((aa & ~bb).sum()),
                    'only_second': int((~aa & bb).sum()), 'neither': int((~aa & ~bb).sum())}
    return result
