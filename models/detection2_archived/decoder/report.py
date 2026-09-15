"""Create a focused W&B report from the renderer's numeric metrics only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..common import write_json
from .data import BANDS


def build_report(out):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'monitoring/_vendor'))
    import wandb_workspaces.reports.v2 as wr
    cfg = json.loads((out/'config.json').read_text())['wandb']
    record = json.loads((out/'wandb_run.json').read_text())
    report = wr.Report(entity=cfg['entity'], project=cfg['project'],
                       title='Detection2 — learned object reconstruction',
                       description='Fixed-catalogue renderer warm-up; ten-band reconstruction diagnostics.')
    runset = wr.Runset(entity=cfg['entity'], project=cfg['project'], name='Learned renderer',
                      filters=f"ID in {[record['id']]!r}")
    panels = []
    specs = [
        ('Improvement over training-median background', 'reconstruction_gain_percent', 'MSE reduction (%)'),
        ('Contribution of learned objects beyond the background', 'gain_over_learned_background_percent', 'MSE reduction (%)'),
        ('Residual near fixed source positions', 'source_rmse', 'RMSE (training MAD units)'),
        ('Residual away from fixed source positions', 'background_rmse', 'RMSE (training MAD units)'),
    ]
    for i, (title, metric, ylabel) in enumerate(specs):
        panels.append(wr.LinePlot(title=title, x='epoch', y=[f'val/{b}/{metric}' for b in BANDS],
                                  title_x='Epoch', title_y=ylabel, smoothing_type='none',
                                  aggregate=False, layout=wr.Layout(x=12*(i % 2), y=9*(i//2), w=12, h=9)))
    panels.append(wr.LinePlot(title='Training and validation reconstruction loss', x='epoch',
                              y=['train/epoch_mean_reconstruction_huber', 'val/reconstruction_huber'],
                              title_x='Epoch', title_y='Huber loss', smoothing_type='none',
                              aggregate=False, layout=wr.Layout(x=0, y=18, w=24, h=8)))
    report.blocks = [wr.MarkdownBlock(text=(
        'This run learns image appearance at **fixed detector positions**. It does not yet add or remove objects. '
        'The encoder and detector stay frozen; all ten bands contribute. No supplied PSF is used.\n\n'
        '**Higher is better** for MSE reduction; **lower is better** for residuals and loss. '
        'The first panel compares against the training-median image. The second removes the rendered '
        'objects while keeping the same predicted constant background, measuring the objects\' contribution. '
        'Zero means no improvement over the corresponding background; negative values mean worse.\n\n'
        'Validation uses 864 fixed, potentially overlapping crops from 108 patch-25 tiles. '
        'Source apertures are within 1.5 arcsec of fixed proposals; these are not truth labels. '
        'Cached features see the full input image, so these curves describe ordinary reconstruction, '
        'not hidden-pixel prediction or detection completeness. Ten-band observed/reconstructed/residual '
        'panels are saved locally each epoch and displayed in `02_learned_decoder.ipynb`.'
    )), wr.PanelGrid(runsets=[runset], hide_run_sets=True, panels=panels)]
    return report


def publish_report(out):
    path = out/'report.json'
    if path.exists():
        record = json.loads(path.read_text())
    else:
        report = build_report(out)
        write_json(out/'report_spec.json', report._to_model().model_dump(mode='json', by_alias=True, exclude_none=True))
        report.save()
        record = {'id': report.id, 'url': report.url}
        write_json(path, record)
    print('Reconstruction report:', record['url'], flush=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True, help='Training output directory')
    publish_report(p.parse_args().out.resolve())


if __name__ == '__main__':
    main()
