"""Publish only derived numeric evaluation metrics to the existing W&B project."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import wandb

from ..common import write_json
from .statistics import ARMS, LABELS
from .present import PRIMARY


def publish(study, out, result):
    cfg = json.loads((study/'control/config.json').read_text())['wandb']
    path = out/'wandb_run.json'
    record = json.loads(path.read_text()) if path.exists() else {'id': wandb.util.generate_id()}
    write_json(path, record)
    primary = result['grids'][str(result['protocol']['bootstrap']['primary_grid'])]
    run = wandb.init(entity=cfg['entity'], project=cfg['project'], group=cfg['group'],
                     id=record['id'], resume='allow', mode='online', dir=str(out),
                     name='Final comparison — unique sources and spatial uncertainty', job_type='final-validation',
                     config={'kind': 'full patch-25 final-epoch evaluation',
                             'tiles': len(result['protocol']['scored_tiles']), 'primary_threshold': .30,
                             'unique_reference_counts': result['protocol']['group_totals_unique'],
                             'bootstrap': result['protocol']['bootstrap']},
                     settings=wandb.Settings(init_timeout=45, disable_code=True, disable_git=True, x_disable_stats=True))
    record['url'] = run.url
    write_json(path, record)
    try:
        log = {}
        for arm in ARMS:
            for key in PRIMARY:
                row = primary['metrics'][arm]['0.3'][key]
                log[f'final/{arm}/{key}_percent'] = row['percent']
                if row['ci95_percentile']:
                    log[f'final/{arm}/{key}_ci95_low'] = row['ci95_percentile'][0]
                    log[f'final/{arm}/{key}_ci95_high'] = row['ci95_percentile'][1]
        rows, diffs = [], []
        for side, grid in result['grids'].items():
            for arm in ARMS:
                for threshold, metrics in grid['metrics'][arm].items():
                    for key, row in metrics.items():
                        lo, hi = row['ci95_percentile'] or (None, None)
                        rows.append([side+'x'+side, result['labels'][arm], float(threshold), key,
                                     row['percent'], lo, hi, row['numerator'], row['denominator']])
            for pair, thresholds in grid['paired_differences'].items():
                for threshold, metrics in thresholds.items():
                    for key, row in metrics.items():
                        lo, hi = row['ci95_percentile'] or (None, None)
                        diffs.append([side+'x'+side, pair, float(threshold), key, row['difference_pp'], lo, hi])
        log['final/estimates'] = wandb.Table(columns=['grid', 'model', 'threshold', 'metric', 'percent',
                                                     'ci95_low', 'ci95_high', 'numerator', 'denominator'], data=rows)
        log['final/paired_differences'] = wandb.Table(columns=['grid', 'comparison', 'threshold', 'metric',
                                                             'difference_pp', 'ci95_low', 'ci95_high'], data=diffs)
        for name, metric in [('full_mer', 'full_mer_completeness'), ('nir_only', 'nir_only_completeness')]:
            xs, ys = [], []
            for arm in ARMS:
                metrics = primary['metrics'][arm]
                xs.append([metrics[str(t)]['mer_match_fraction']['percent'] for t in result['protocol']['thresholds']])
                ys.append([metrics[str(t)][metric]['percent'] for t in result['protocol']['thresholds']])
            log[f'final/tradeoff/{name}'] = wandb.plot.line_series(xs=xs, ys=ys, keys=list(LABELS),
                                           title=PRIMARY[metric]+' — unique sources', xname='MER match fraction (%)')
        run.log(log)
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'monitoring/_vendor'))
    import wandb_workspaces.reports.v2 as wr
    report_record = json.loads((study/'monitoring/report.json').read_text())
    report = wr.Report.from_url(report_record['url'])
    marker = 'Full patch-25 validation'
    blocks = []
    for block in report.blocks:
        if isinstance(block, wr.H1) and block.text == marker:
            break
        blocks.append(block)
    panels = [wr.WeavePanelSummaryTable(table_name='final/estimates', layout=wr.Layout(x=0,y=0,w=24,h=10)),
              wr.WeavePanelSummaryTable(table_name='final/paired_differences', layout=wr.Layout(x=0,y=10,w=24,h=10))]
    tradeoffs = []
    for i, key in enumerate(('full_mer', 'nir_only')):
        tradeoffs.append(wr.CustomChart(
            query={'summaryTable': {'tableKey': f'final/tradeoff/{key}_table'}},
            chart_name='wandb/lineseries/v0', chart_fields={'step':'step', 'lineKey':'lineKey', 'lineVal':'lineVal'},
            chart_strings={'title': key.replace('_', ' ')+' completeness (%) — full patch',
                           'xname': 'MER match fraction (%)'}, layout=wr.Layout(x=12*i,y=0,w=12,h=9)))
    runset = wr.Runset(entity=cfg['entity'], project=cfg['project'], name='Final unique-source comparison',
                       filters=f"ID in {[record['id']]!r}")
    report.blocks = blocks + [wr.H1(marker), wr.MarkdownBlock(text=(out/'comparison.md').read_text()),
                              wr.PanelGrid(runsets=[runset], hide_run_sets=True, panels=tradeoffs),
                              wr.PanelGrid(runsets=[runset], hide_run_sets=True, panels=panels)]
    write_json(out/'report_spec.json', report._to_model().model_dump(mode='json', by_alias=True, exclude_none=True))
    report.save()
    write_json(out/'report.json', {'id': report.id, 'url': report.url})
    api_run = wandb.Api(timeout=30).run(f"{cfg['entity']}/{cfg['project']}/{record['id']}")
    for key, value in log.items():
        if isinstance(value, (int, float)):
            actual = api_run.summary.get(key)
            if actual is None or abs(actual-value) > 1e-8:
                raise ValueError(f'W&B scalar readback mismatch: {key}')
    saved = wr.Report.from_url(report.url)
    serialized = json.dumps(saved._to_model().model_dump(mode='json', by_alias=True, exclude_none=True))
    if 'Paired differences at threshold 0.30' not in serialized:
        raise ValueError('Final comparison missing from saved W&B report')
    write_json(out/'publication_verification.json', {'scalar_values_verified': True,
                                                   'report_readback_verified': True, 'url': report.url})
    print('Full validation report:', report.url, flush=True)


def main():
    """Retry publication of saved results without rerunning model inference."""
    import argparse
    import fcntl
    from .run import DEFAULT_STUDY, status
    from ..common import digest
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    args = parser.parse_args()
    study = args.study.resolve()
    out = study/'full_validation'
    with (out/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = out/'results.json'
        result = json.loads(result_path.read_text())
        if result['protocol'] != json.loads((out/'protocol.json').read_text()):
            raise ValueError('Saved result and protocol disagree')
        before = digest(result_path)
        status(out, 'publishing_saved_results')
        try:
            publish(study, out, result)
            assert digest(result_path) == before, 'Publication changed scientific results'
            write_json(out/'publication_retry.json', {'results_sha256': before,
                'publication_code_sha256': digest(Path(__file__)), 'inference_repeated': False,
                'reason': 'Serialize report timestamps using Pydantic JSON mode'})
            status(out, 'complete', results=str(result_path))
        except BaseException as exc:
            status(out, 'failed', exception=type(exc).__name__, message=str(exc))
            raise


if __name__ == '__main__':
    main()
