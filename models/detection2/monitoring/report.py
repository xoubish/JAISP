"""Create/update a focused W&B report from existing scalar training metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / '_vendor'))
import wandb_workspaces.reports.v2 as wr

from ..common import write_json
from .data import DEFAULT_STUDY, PRIMARY, COLORS


def build(study):
    config = json.loads((study / 'control/config.json').read_text())
    wc = config['wandb']
    monitor = json.loads((study / 'monitoring/monitor_state.json').read_text())
    ids = [monitor['id']] + [json.loads((study / a / 'wandb_run.json').read_text())['id']
                             for a in ('control', 'unknown')]
    colors = dict(zip(ids, COLORS))

    def runset(selected):
        return wr.Runset(entity=wc['entity'], project=wc['project'], name='Detection2 comparison',
                         filters=f'ID in {selected!r}')

    report = wr.Report(entity=wc['entity'], project=wc['project'], width='fluid',
                       title='Detection2 detection progress',
                       description='12-epoch control versus unknown regions. Fixed validation tiles, '
                                   'starting-model reference and threshold tradeoffs.')
    record = study / 'monitoring/report.json'
    if record.exists():
        report.id = json.loads(record.read_text())['id']

    panels = []
    for i, (key, title) in enumerate(PRIMARY.items()):
        metric = f'val/t0.30/{key}'
        panels.append(wr.LinePlot(
            title=title, x='epoch', y=[metric],
            title_x='Completed validation epoch', title_y='Percent',
            custom_expressions=['${' + metric + '} * 100'],
            range_y=(None, 100), range_x=(0, None),
            smoothing_type='none', ignore_outliers=False, aggregate=False,
            max_runs_to_show=3, legend_position='south', font_size='medium',
            layout=wr.Layout(x=12*(i % 2), y=8*(i // 2), w=12, h=8)))

    tradeoff_panels = []
    for i, (key, title) in enumerate([('full_mer', 'Full-MER completeness (%)'),
                                      ('nir_only', 'NIR-only completeness (%)')]):
        tradeoff_panels.append(wr.CustomChart(
            query={'summaryTable': {'tableKey': f'tradeoff/{key}_table'}},
            chart_name='wandb/lineseries/v0',
            chart_fields={'step': 'step', 'lineKey': 'lineKey', 'lineVal': 'lineVal'},
            chart_strings={'title': title + ' — latest shared epoch', 'xname': 'MER match fraction (%)'},
            layout=wr.Layout(x=i*12, y=0, w=12, h=9)))
    report.blocks = [
        wr.H1('Are we recovering more objects without losing catalogue agreement?'),
        wr.P('In the epoch plots, blue is control, orange is unknown regions, and grey is the unchanged starting model. '
             'Compare the validation curves, not the raw losses. Each quality point is one completed epoch '
             'on the same 12 patch-25 tiles; all four panels use score threshold 0.30.'),
        wr.PanelGrid(runsets=[runset(ids)], panels=panels, hide_run_sets=True, custom_run_colors=colors),
        wr.H2('Does the tradeoff improve when the threshold changes?'),
        wr.P('Higher and farther right is better. Each curve connects the four measured thresholds '
             '0.20, 0.30, 0.40 and 0.50; the lines are visual guides, not extra measurements. '
             'Control and unknown always use their latest shared completed epoch. '
             'The starting model remains the epoch-0 reference. Legend labels identify each curve.'),
        wr.PanelGrid(runsets=[runset([monitor['id']])], hide_run_sets=True,
                     panels=[wr.ScalarChart(title='Latest shared validation epoch', metric='monitor/latest_shared_epoch',
                                            layout=wr.Layout(x=0, y=0, w=8, h=4))]),
        wr.PanelGrid(runsets=[runset([monitor['id']])], hide_run_sets=True, panels=tradeoff_panels),
        wr.H2('How to interpret the result'),
        wr.P('An improvement should retain or increase completeness at comparable MER agreement. '
             'A shift in score calibration can change the fixed-threshold plots without improving the tradeoff. '
             'MER agreement is not absolute object purity: real sources absent from MER count as unmatched. '
             'Counts are tile-pooled and matching is nearest-neighbour within 0.5 arcsec. '
             'VIS here means clean VIS sources with magnitude <24.5; NIR-only and full-MER completeness have no magnitude cut. '
             'These are development metrics, not a blind final test.'),
        wr.H2('Optimization diagnostics'),
        wr.P('The unknown arm omits some background penalties, so its loss can be lower by construction. '
             'Use these secondary plots to spot instability, not to decide which detector is better.'),
        wr.PanelGrid(runsets=[runset(ids[1:])], hide_run_sets=True,
                     custom_run_colors={k: colors[k] for k in ids[1:]}, panels=[
            wr.LinePlot(title='Mean training loss per epoch', x='epoch', y=['train/epoch_mean_loss'],
                        smoothing_type='none', aggregate=False, layout=wr.Layout(x=0,y=0,w=12,h=7)),
            wr.LinePlot(title='Learning rate', x='train/step', y=['train/lr'],
                        smoothing_type='none', aggregate=False, layout=wr.Layout(x=12,y=0,w=12,h=7))]),
        wr.P('The reference/comparison monitor reads saved validation metrics only. It uses no GPU and '
             'does not alter either training process. Tradeoff tables refresh within about 30 seconds '
             'of both validation files arriving. No candidate images or model checkpoints are uploaded.')
    ]
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    p.add_argument('--publish', action='store_true')
    args = p.parse_args()
    report = build(args.study)
    # Save the exact serialized panel specification locally for review/reuse.
    write_json(args.study / 'monitoring/report_spec.json',
               report._to_model().model_dump(by_alias=True, exclude_none=True))
    if args.publish:
        report.save()
        write_json(args.study / 'monitoring/report.json', {'id': report.id, 'url': report.url})
        print(report.url, flush=True)
    else:
        print('Report specification saved locally; use --publish to save to W&B.')


if __name__ == '__main__':
    main()
