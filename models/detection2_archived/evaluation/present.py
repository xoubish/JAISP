"""Standalone scientific plots and a human-readable paired comparison."""
from __future__ import annotations

import numpy as np

from .statistics import ARMS, LABELS

PRIMARY = {
    'vis_bright_completeness': 'VIS completeness (mag < 24.5)',
    'nir_only_completeness': 'NIR-only completeness',
    'full_mer_completeness': 'Full-MER completeness',
    'mer_match_fraction': 'MER match fraction',
}
COLORS = ('#667085', '#2563eb', '#ea7c19')


def format_estimate(row, difference=False):
    value = row['difference_pp' if difference else 'percent']
    if value is None:
        return 'undefined'
    bounds = row['ci95_percentile']
    ci = f'[{bounds[0]:.2f}, {bounds[1]:.2f}]' if bounds is not None else '[CI unavailable]'
    return f'{value:+.2f} {ci}' if difference else f'{value:.2f}% {ci}'


def present(out, result, suffix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protocol = result['protocol']
    primary = str(protocol['bootstrap']['primary_grid'])
    grid = result['grids'][primary]
    title = 'Detection2: all patch-25 tiles, unique-source evaluation'
    if suffix:
        title = 'CPU execution check: same archived starting model in all three slots'
    text = [f'# {title}', '',
            f"Scored {len(protocol['scored_tiles'])}/{len(protocol['requested_tiles'])} tiles. "
            f"Unique reference counts: {protocol['group_totals_unique']}.", '',
            'Fixed final checkpoints; threshold 0.30 is the primary comparison. Values in brackets '
            'are approximate 95% paired spatial-bootstrap percentile intervals. Percentages for '
            'model estimates; percentage points for differences. These are pointwise intervals, '
            'not simultaneous guarantees across thresholds or metrics.', '',
            f'Primary spatial grid: {primary}×{primary}, with 5,000 shared region resamples. '
            'A 3×3 grid provides a block-size sensitivity check. These intervals describe variation '
            'within this patch, conditional on these models and MER; they do not include training-seed '
            'variation, other fields, or catalogue truth errors.', '',
            '| Metric | ' + ' | '.join(LABELS) + ' |', '|---|---:|---:|---:|']
    for metric, label in PRIMARY.items():
        text.append('| '+label+' | '+' | '.join(format_estimate(grid['metrics'][a]['0.3'][metric]) for a in ARMS)+' |')
    text += ['', '## Paired differences at threshold 0.30', '',
             'Positive means the first model has a larger metric. Completeness and MER agreement '
             'must be assessed together. MER agreement is not absolute source purity.', '']
    for side, summary in result['grids'].items():
        text += [f'### {side}×{side} regions', '',
                 '| Metric | Standard − starting | Masked − starting | Masked − standard |',
                 '|---|---:|---:|---:|']
        for metric, label in PRIMARY.items():
            text.append('| '+label+' | '+' | '.join(format_estimate(summary['paired_differences'][pair]['0.3'][metric], True)
                         for pair in ('control_minus_starting', 'masked_minus_starting', 'masked_minus_control'))+' |')
        text += ['']
    text += ['The geometry and ownership rule are fixed before final predictions. Each MER ID is counted '
             'once; predictions are retained only in their originating tile’s assigned sky area. '
             'No spatial merging radius is used to collapse nearby physical sources. Matching uses '
             'nearest neighbours within 0.5 arcsec and is not one-to-one. Small centroid changes across '
             'tile-ownership boundaries can still affect seam detections. These are partitioned-sky '
             'metrics, so they need not equal the tile-pooled training curves.', '',
             'Raw predictions, unique catalogues, per-source recovery flags, block counts, bootstrap '
             'weights, input/code hashes, and the protocol remain local beside this file.', '']
    (out/f'comparison{suffix}.md').write_text('\n'.join(text))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, PRIMARY.items()):
        for i, (arm, color) in enumerate(zip(ARMS, COLORS)):
            row = grid['metrics'][arm]['0.3'][metric]
            if row['percent'] is None:
                continue
            lohi = row['ci95_percentile']
            if lohi:
                ax.vlines(i, *lohi, color=color, linewidth=2)
                ax.hlines(lohi, i-.07, i+.07, color=color)
            ax.scatter(i, row['percent'], color=color, s=55, zorder=3)
        ax.set(title=label, ylabel='Percent', xticks=range(3), xticklabels=LABELS, xlim=(-.5, 2.5))
        ax.tick_params(axis='x', labelsize=9, rotation=10)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(title+'\nThreshold 0.30; approximate 95% intervals from sky regions', fontsize=13)
    fig.savefig(out/f'quality_intervals{suffix}.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, metric in zip(axes, ('full_mer_completeness', 'nir_only_completeness')):
        for arm, label, color in zip(ARMS, LABELS, COLORS):
            rows = grid['metrics'][arm]
            xx = [rows[str(t)]['mer_match_fraction']['percent'] for t in protocol['thresholds']]
            yy = [rows[str(t)][metric]['percent'] for t in protocol['thresholds']]
            ax.plot(xx, yy, 'o-', label=label, color=color, markersize=4)
            i = protocol['thresholds'].index(.30)
            ax.scatter([xx[i]], [yy[i]], marker='s', color=color, s=70, zorder=4)
        ax.set(title=PRIMARY[metric], xlabel='MER match fraction (%)', ylabel='Completeness (%)')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(title+'\nMeasured threshold tradeoffs; squares mark 0.30', fontsize=12)
    fig.savefig(out/f'tradeoffs{suffix}.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    pairs = ('control_minus_starting', 'masked_minus_starting', 'masked_minus_control')
    for ax, (metric, label) in zip(axes.flat, PRIMARY.items()):
        for j, (side, summary) in enumerate(result['grids'].items()):
            offset, color = (-.10, '#2563eb') if j == 0 else (.10, '#ea7c19')
            for i, pair in enumerate(pairs):
                row = summary['paired_differences'][pair]['0.3'][metric]
                if row['ci95_percentile']:
                    ax.hlines(i+offset, *row['ci95_percentile'], color=color, linewidth=2)
                ax.scatter(row['difference_pp'], i+offset, color=color,
                           label=f'{side}×{side} regions' if i == 0 else None, s=35)
        ax.axvline(0, color='grey', linestyle='--', linewidth=1)
        ax.set(title=label, xlabel='Difference (percentage points)', yticks=range(3),
               yticklabels=['Standard − starting', 'Masked − starting', 'Masked − standard'])
        ax.legend(fontsize=8); ax.grid(axis='x', alpha=.2)
    fig.suptitle(title+'\nPaired differences and sensitivity to region size', fontsize=12)
    fig.savefig(out/f'paired_differences{suffix}.png', dpi=160)
    plt.close(fig)
