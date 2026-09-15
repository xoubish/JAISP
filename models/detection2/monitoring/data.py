from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..common import HERE, write_json

DEFAULT_STUDY = HERE / 'runs/unknown_regions_20260915_v1_12ep'
PRIMARY = {
    'vis_bright_completeness': 'VIS completeness (mag < 24.5)',
    'nir_only_completeness': 'NIR-only completeness',
    'full_mer_completeness': 'Full-MER completeness',
    'mer_match_fraction': 'MER match fraction',
}
LABELS = ['Starting model', 'Control', 'Unknown regions']
COLORS = ['#667085', '#2563eb', '#ea7c19']


def read_snapshot(study):
    """Only consume complete epoch JSON files, written atomically by training."""
    histories = {}
    for arm in ('control', 'unknown'):
        histories[arm] = {int(p.stem.rsplit('_', 1)[1]): json.loads(p.read_text())
                          for p in (study / arm).glob('validation_epoch_*.json')}
    shared = sorted(set(histories['control']) & set(histories['unknown']))
    if not shared or shared[0] != 0:
        raise ValueError('Both runs must have initial validation before monitoring')
    if shared != list(range(shared[-1] + 1)):
        raise ValueError('Missing shared validation epoch')
    baseline = histories['control'][0]
    if baseline != histories['unknown'][0]:
        raise ValueError('The two runs have different starting validation results')
    for arm, epochs in histories.items():
        for epoch in shared:
            row = epochs[epoch]
            if row['tiles_scored'] != baseline['tiles_scored']:
                raise ValueError('Validation footprint changed')
            if set(row['counts']) != set(baseline['counts']):
                raise ValueError('Validation threshold grid changed')
            for conf, counts in row['counts'].items():
                if counts['total'] != baseline['counts'][conf]['total']:
                    raise ValueError('Validation reference counts changed')
                checks = {f'{key}_completeness': counts['recovered'][key] / max(n, 1)
                          for key, n in counts['total'].items()}
                checks['mer_match_fraction'] = counts['n_matched'] / max(counts['n_det'], 1)
                for key, expected in checks.items():
                    actual = row['metrics'][f'val/t{float(conf):.2f}/{key}']
                    if not np.isfinite(actual) or not np.isclose(actual, expected, atol=1e-12, rtol=0):
                        raise ValueError(f'Metric/count disagreement: {arm}, epoch {epoch}, {key}')
    return {'histories': histories, 'shared_epochs': shared, 'baseline': baseline,
            'latest_shared_epoch': shared[-1]}


def tradeoff(snapshot, epoch, metric):
    rows = [snapshot['baseline'], snapshot['histories']['control'][epoch],
            snapshot['histories']['unknown'][epoch]]
    thresholds = sorted(map(float, rows[0]['counts']))
    xs = [[100 * r['metrics'][f'val/t{c:.2f}/mer_match_fraction'] for c in thresholds] for r in rows]
    ys = [[100 * r['metrics'][f'val/t{c:.2f}/{metric}'] for c in thresholds] for r in rows]
    return xs, ys, thresholds


def preview(study, snapshot):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = study / 'monitoring'
    out.mkdir(exist_ok=True)
    epochs = snapshot['shared_epochs']
    latest = epochs[-1]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, PRIMARY.items()):
        metric = f'val/t0.30/{key}'
        values = [[100 * snapshot['baseline']['metrics'][metric]] * len(epochs)]
        values += [[100 * snapshot['histories'][arm][e]['metrics'][metric] for e in epochs]
                   for arm in ('control', 'unknown')]
        for label, color, y in zip(LABELS, COLORS, values):
            ax.plot(epochs, y, '--' if label == 'Starting model' else 'o-', color=color, label=label)
        ax.set(title=title, xlabel='Completed validation epoch', ylabel='Percent')
        ax.set_xticks(epochs)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for ax, key in zip(axes[2], ('full_mer_completeness', 'nir_only_completeness')):
        xs, ys, thresholds = tradeoff(snapshot, latest, key)
        for label, color, x, y in zip(LABELS, COLORS, xs, ys):
            ax.plot(x, y, 'o-', color=color, label=label)
            for px, py, threshold in zip(x, y, thresholds):
                ax.annotate(f'{threshold:.2f}', (px, py), xytext=(3, 3), textcoords='offset points', fontsize=7)
        ax.set(title=f'{PRIMARY[key]} — shared epoch {latest}',
               xlabel='MER match fraction (%)', ylabel='Completeness (%)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Detection2: completeness and catalogue agreement\n'
                 'Top four: score threshold 0.30. Bottom: four threshold operating points.', fontsize=14)
    fig.savefig(out / 'dashboard_preview.png', dpi=150)
    plt.close(fig)
    write_json(out / 'snapshot.json', snapshot)
