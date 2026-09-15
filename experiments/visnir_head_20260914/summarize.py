"""Assemble the experiment's checked catalogue comparison and injection pilot."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIELDS = ('ECDFS_patch25', 'EDF-S')
MODELS = ('baseline', 'vis_control', 'visnir')
LABELS = ('Production', 'VIS control', 'VIS + NISP')


def catalogue_results():
    configs = [json.loads((HERE/m/'config.json').read_text()) for m in MODELS[1:]]
    for key in ('seed', 'baseline_sha256', 'epochs', 'batch_size', 'lr', 'n_parameters',
                'n_train_samples', 'train_tiles', 'precision', 'checkpoint_selection'):
        assert configs[0][key] == configs[1][key], f'Unpaired training setting: {key}'
    out = {}
    for field in FIELDS:
        baseline = json.loads((HERE/'baseline_evaluation'/f'{field}_metrics.json').read_text())
        trained = json.loads((HERE/f'evaluation_{field}'/f'{field}_metrics.json').read_text())
        out[field] = dict(baseline=baseline['rows']['baseline'], **trained['rows'])
        for model in MODELS[1:]:
            assert out[field][model]['0.3']['totals'] == out[field]['baseline']['0.3']['totals']
    return out


def injections():
    paths = sorted(HERE.glob('injections_*/tile*.json'))
    if len(paths) != 27:
        return None
    assert len({p.name for p in paths}) == 27
    agg = {m: {mode: defaultdict(lambda: defaultdict(int)) for mode in ('all', 'vis', 'nisp')}
           for m in MODELS}
    for path in paths:
        tile = json.loads(path.read_text())
        for mode, values in tile['baseline'].items():
            for mag, row in values.items():
                assert all(tile[m][mode][mag]['injected'] == row['injected'] for m in MODELS)
        for model, modes in tile.items():
            for mode, mags in modes.items():
                for mag, row in mags.items():
                    for key, value in row.items():
                        agg[model][mode][mag][key] += value
    return agg


def depth50(rows):
    points = [(float(m), 100*r['recovered']/r['injected']) for m, r in rows.items()
              if float(m) < 30 and r['injected']]
    points.sort()
    for (m0, c0), (m1, c1) in zip(points[:-1], points[1:]):
        if c0 >= 50 and c1 < 50:
            return m0+(50-c0)*(m1-m0)/(c1-c0)
    return None


def main():
    result = catalogue_results()
    inj = injections()
    metrics = [('vis_bright', 'Clean VIS <24.5 completeness'),
               ('vis_all', 'All clean VIS completeness'),
               ('nir_only', 'NIR-only completeness'),
               ('full_clean', 'Full clean MER completeness'),
               ('purity', 'Full-MER match purity')]
    lines = ['# VIS + NISP detection-head experiment', '',
             'Four-epoch paired fine-tuning with the encoder frozen. Primary confidence threshold: 0.30.', '',
             'The production checkpoint, manuscript and paper figures are unchanged.', '']
    for field in FIELDS:
        lines.extend([f'## {field}', '', '| Metric (%) | Production | VIS control | VIS + NISP |',
                      '|---|---:|---:|---:|'])
        for key, label in metrics:
            vals = []
            for model in MODELS:
                row = result[field][model]['0.3']
                vals.append(row['purity_percent'] if key == 'purity' else row['completeness_percent'][key])
            lines.append('| '+label+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
        row = result[field]['visnir']['0.3']
        old = result[field]['vis_control']['0.3']
        lines.extend(['', f"NIR-only change relative to the matched VIS-only control: "
                      f"{row['completeness_percent']['nir_only']-old['completeness_percent']['nir_only']:+.2f} percentage points.", ''])
    lines.extend(['## Counts and interpretation', '',
                  'Completeness uses the same masked, tile-pooled reference sample for all heads. '
                  'The matching radius is 0.5 arcsec. NIR-only and full-MER completeness have no magnitude cut. '
                  'Objects in overlapping tiles occur more than once. EDF-S purity uses the existing catalogue footprint restriction.', '',
                  'The control isolates the effect of the added labels from additional training. '
                  'This is a short fine-tune, so it does not establish fully converged performance. '
                  'Catalogue matching measures agreement with MER, not absolute source truth.', ''])
    lines.extend(['## Diagnostic threshold 0.40', '',
                  'This is an exploratory operating-point comparison, not a replacement for the fixed 0.30 primary result.', '',
                  '| Field | VIS <24.5 completeness | NIR-only completeness | Full clean MER completeness | MER match purity |',
                  '|---|---:|---:|---:|---:|'])
    for field in FIELDS:
        row = result[field]['visnir']['0.4']
        values = [row['completeness_percent'][k] for k in ('vis_bright', 'nir_only', 'full_clean')]+[row['purity_percent']]
        lines.append('| '+field+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines.append('')
    if inj is not None:
        lines.extend(['## Paired injection pilot', '',
                      '27 tiles distributed across patch 25; identical injections for all heads. '
                      'The recovery radius is 0.3 arcsec. Magnitudes are donor VIS-equivalent values, '
                      'including for NISP-only injections. This pilot does not replace the paper’s existing full injection analysis.', '',
                      '| Injection mode | Production d50 | VIS control d50 | VIS + NISP d50 |', '|---|---:|---:|---:|'])
        for mode in ('all', 'vis', 'nisp'):
            depths = [depth50(inj[m][mode]) for m in MODELS]
            lines.append('| '+mode+' | '+' | '.join('not bracketed' if d is None else f'{d:.2f}' for d in depths)+' |')
        lines.extend(['', 'The d50 values use linear interpolation only where the sampled curve crosses 50%; no extrapolation.', '',
                      '| Model | Induced artifacts / recovered sources, all modes (mag <30) | Faint-control recoveries (mag 35) |',
                      '|---|---:|---:|'])
        for model in MODELS:
            rows = [r for mode in inj[model].values() for mag, r in mode.items() if float(mag) < 30]
            faint = [mode['35.0'] for mode in inj[model].values()]
            lines.append(f"| {model} | {sum(r['artifacts'] for r in rows)} / {sum(r['recovered'] for r in rows)} | "
                         f"{sum(r['recovered'] for r in faint)} / {sum(r['injected'] for r in faint)} |")
        lines.append('')
    else:
        lines.extend(['The paired injection pilot is still running.', ''])
    (HERE/'comparison.json').write_text(json.dumps(dict(catalogue=result, injections=inj), indent=2)+'\n')
    (HERE/'RESULTS.md').write_text('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    colors = ('#566777', '#D09332', '#167D8D')
    for ax, field in zip(axes, FIELDS):
        for i, (model, label, color) in enumerate(zip(MODELS, LABELS, colors)):
            row = result[field][model]['0.3']
            values = [row['completeness_percent'][k] for k in ('vis_bright', 'nir_only', 'full_clean')]+[row['purity_percent']]
            ax.bar(np.arange(4)+(i-1)*0.25, values, 0.24, color=color, label=label)
        ax.set_xticks(np.arange(4), ['VIS <24.5\nrecovery', 'NIR-only\nrecovery', 'Full clean MER\nrecovery', 'MER match\npurity'])
        ax.set_title(field.replace('ECDFS_patch25', 'ECDFS, patch 25'))
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Percent')
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False,
               loc='lower center', ncol=3, fontsize=9)
    fig.suptitle('Fixed threshold 0.30 · frozen encoder · four-epoch paired comparison')
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(HERE/'catalogue_comparison.png', dpi=180)
    plt.close(fig)
    if inj is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, mode in zip(axes, ('all', 'vis', 'nisp')):
            for model, label, color in zip(MODELS, LABELS, colors):
                points = sorted((float(m), 100*r['recovered']/r['injected'])
                                for m, r in inj[model][mode].items() if float(m) < 30)
                ax.plot(*np.asarray(points).T, 'o-', color=color, label=label, markersize=4)
            ax.set_title(dict(all='All ten bands', vis='VIS only', nisp='NISP only')[mode])
            ax.set_xlabel('Donor VIS-equivalent magnitude')
            ax.axhline(50, color='gray', ls=':', lw=1)
            ax.set_ylim(0, 103)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel('Injected sources recovered (%)')
        fig.legend(*axes[0].get_legend_handles_labels(), frameon=False,
                   loc='lower center', ncol=3, fontsize=9)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(HERE/'injection_comparison.png', dpi=180)
        plt.close(fig)
    print(HERE/'RESULTS.md')


if __name__ == '__main__':
    main()
