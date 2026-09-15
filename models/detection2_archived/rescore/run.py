"""Run one fixed rescoring rule, evaluate, publish numeric results, and stop."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ..common import ROOT, digest, write_json
from ..evaluation.run import load_prepared, save_npz
from .score import STUDY, EVALUATION, combine_scores
from .metrics import GROUPS, compare

DEFAULT_OUT = ROOT/'models/detection2/runs/decoder_rescore_20260915_v1'


def protocol():
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'One frozen-decoder scoring rule on the existing patch-25 development footprint. No training, score-weight tuning, candidate births, centroid changes, or automatic follow-up experiment.',
        'decoder_sha256': digest(STUDY/'training/final.pt'),
        'baseline_catalogue_sha256': digest(EVALUATION/'control_catalog.npz'),
        'reference_sha256': digest(EVALUATION/'references.npz'),
        'geometry_sha256': digest(EVALUATION/'geometry.json'),
        'source_sha256': {str(p): digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'candidate_floor': .15, 'candidate_rule': 'Existing standard-loss predictions strictly above 0.15, using their previously frozen sky ownership.',
        'image_evidence': 'E_ib = sum_valid[Huber(yhat_ib - component_ib - y_b) - Huber(yhat_ib - y_b)] / sqrt(sum_valid(component_ib^2)), with Huber delta 3 and denominator floor 1e-8. The full prediction background and all neighbours stay fixed.',
        'scoring_rule': 'score_i = log(p_i/(1-p_i)) + asinh(mean_b(E_ib)); p clipped to [1e-6,1-1e-6]. Equal weight for all ten bands; coefficient fixed at one before results.',
        'fallback': 'Evidence zero if a crop exceeds 128 context objects or has less than 80% valid pixels in any band; retain original detector ranking contribution and report all fallbacks.',
        'calibration': 'A ranking heuristic, not calibrated probability or independent-pixel likelihood. Full-image cached features see the image being scored. MER information does not enter candidate scoring.',
        'comparison': 'For each score ordering, select the largest whole-score prefix with at least 100 detections and agreement at or above the common target. Primary target equals baseline agreement at p>=0.30; secondary target 94%. These thresholds describe development PR curves; they are not deployable thresholds validated on another field.',
        'uncertainty': '2000 paired spatial bootstrap replicates, 4x4 primary and 3x3 sensitivity. Reselect both equal-agreement thresholds in each replicate. Pointwise within-patch intervals, not uncertainty across seeds or fields.',
        'decision_gate': 'Pass only if primary NIR gain >=1 percentage point with 4x4 paired 95% lower bound >0 and no loss in full-MER completeness; secondary 94% agreement must also have no loss in NIR or full-MER completeness. Otherwise close this branch and keep the selected detector. No further score search.',
        'inference_wall_limit_seconds': 900,
    }


def present(out, result, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, group in zip(axes, ('full_mer', 'nir_only')):
        g = GROUPS.index(group)
        for name, curve in curves.items():
            valid = curve['ends'] >= 100
            ax.plot(curve['agreement'][valid], curve['completeness'][valid, g], label=name, linewidth=1.5)
        ax.set(xlim=(65, 100), ylim=(0, 100), xlabel='MER match fraction (%)',
               ylabel='Completeness (%)', title='Full clean MER' if group == 'full_mer' else 'NIR-only clean MER')
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(out/'detection_tradeoff.png', dpi=150); plt.close(fig)
    lines = ['# Single frozen-decoder rescoring test', '', f"Decision: **{result['decision']}**.", '',
             f"Scored {result['candidate_count']:,} candidates; {result['fallback_count']} used neutral image evidence because their crops were unsuitable.", '',
             'No model weights or positions changed. The rescored outputs form a different selected detection catalogue at each threshold.', '',
             '| Target agreement | Model | Actual agreement | Full-MER completeness | NIR completeness | Detections |',
             '|---|---|---:|---:|---:|---:|']
    for label, entry in result['operating_points'].items():
        for name, row in entry['models'].items():
            if row is None:
                lines.append(f'| {label} | {name} | unattainable | — | — | — |')
            else:
                lines.append(f"| {entry['target_agreement_percent']:.3f}% | {name} | {row['agreement_percent']:.3f}% | {row['completeness_percent']['full_mer']:.3f}% | {row['completeness_percent']['nir_only']:.3f}% | {row['detections']} |")
    lines += ['', '## Paired differences (rescored minus baseline)', '',
              '| Target | Metric | Difference (pp) | 4x4 paired 95% interval | 3x3 sensitivity |', '|---|---|---:|---|---|']
    for label, entry in result['operating_points'].items():
        for group, value in entry.get('difference_pp', {}).items():
            intervals = [result['paired_intervals'][grid][label][group]['ci95_percentile_pp'] for grid in ('4x4', '3x3')]
            shown = [f'[{v[0]:+.3f}, {v[1]:+.3f}]' if v else 'unavailable' for v in intervals]
            lines.append(f'| {label} | {group} | {value:+.3f} | {shown[0]} | {shown[1]} |')
    lines += ['', 'Both rankings use the same candidates, geometry, unique MER references and nearest 0.5-arcsec matching. Matching is not one-to-one. MER agreement is not absolute physical purity.', '',
              'Cutoffs are read from development curves at matched agreement and are reselected within each bootstrap replicate. Intervals do not include other fields or training seeds. The one scoring rule was fixed before this evaluation; it will not be adjusted after seeing this result.', '',
              'The paper has not been edited and its existing detector results have not been replaced.']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')


def evaluate(out):
    frozen, _, refs = load_prepared(EVALUATION)
    with np.load(EVALUATION/'control_catalog.npz') as z:
        cat = {k: z[k][z['scores'] > .15] for k in z.files}
    evidence = np.full((len(cat['scores']), 10), np.nan)
    fallback = np.zeros(len(cat['scores']), bool)
    seen = np.zeros(len(cat['scores']), int)
    for path in sorted((out/'tiles').glob('*.npz')):
        with np.load(path) as z:
            ii = z['catalogue_index']; evidence[ii] = z['evidence_by_band']; fallback[ii] = z['fallback']; seen[ii] += 1
    if not np.all(seen == 1) or not np.isfinite(evidence).all(): raise ValueError('Incomplete or duplicate candidate scoring')
    scores = {'baseline': cat['scores'].astype(float), 'rescored': combine_scores(cat['scores'], evidence.mean(1))}
    result, curves, neighbours = compare(cat, refs, scores, frozen)
    # Verify the unchanged detector reproduces the previously published counts.
    existing = json.loads((EVALUATION/'results.json').read_text())['grids']['4']['metrics']['control']['0.3']
    n = int((cat['scores'] >= .30).sum())
    hits = curves['baseline']['first'] <= n
    for g, name in enumerate(GROUPS):
        if int((hits & refs['groups'][:, g]).sum()) != existing[name+'_completeness']['numerator']:
            raise ValueError('Baseline recovery failed the archived-count cross-check')
    if int(cat['matched'][cat['scores'] >= .30].sum()) != existing['mer_match_fraction']['numerator']:
        raise ValueError('Baseline agreement failed the archived-count cross-check')
    result.update({'fallback_count': int(fallback.sum()), 'baseline_archived_counts_reproduced': True,
                   'optimizer_updates': 0, 'protocol_sha256': digest(out/'protocol.json')})
    save_npz(out/'candidate_scores.npz', **cat, rescored=scores['rescored'], evidence_by_band=evidence, fallback=fallback)
    for label, entry in result['operating_points'].items():
        for name, row in entry['models'].items():
            if row is not None:
                keep = scores[name] >= row['threshold']
                save_npz(out/f'{label}_{name}_catalogue.npz', **{k: v[keep] for k, v in cat.items()}, ranking_score=scores[name][keep])
    save_npz(out/'curves.npz', **{name+'_'+key: val for name, curve in curves.items() for key, val in curve.items()})
    write_json(out/'results.json', result)
    present(out, result, curves)
    return result, curves


def publish(run, out, result, curves):
    import wandb
    table = []
    for label, entry in result['operating_points'].items():
        for name, row in entry['models'].items():
            if row is None: continue
            prefix = f'result/{label}/{name}/'
            run.log({prefix+'agreement_percent': row['agreement_percent'],
                     **{prefix+g+'_completeness_percent': v for g, v in row['completeness_percent'].items()}})
            table.append([label, name, entry['target_agreement_percent'], row['agreement_percent'],
                          row['completeness_percent']['full_mer'], row['completeness_percent']['nir_only'], row['detections']])
    log = {'result/operating_points': wandb.Table(columns=['target', 'model', 'target_agreement_percent', 'actual_agreement_percent',
                'full_mer_completeness_percent', 'nir_only_completeness_percent', 'detections'], data=table)}
    intervals = []
    for grid, entries in result['paired_intervals'].items():
        for label, entry in entries.items():
            for group, item in entry.items():
                ci = item['ci95_percentile_pp'] or [None, None]
                intervals.append([grid, label, group, result['operating_points'][label].get('difference_pp', {}).get(group), *ci])
    log['result/paired_differences'] = wandb.Table(columns=['grid', 'target', 'group', 'difference_pp', 'ci95_low_pp', 'ci95_high_pp'], data=intervals)
    for group in ('full_mer', 'nir_only'):
        xs, ys = [], []
        for curve in curves.values():
            valid = np.flatnonzero(curve['ends'] >= 100)
            ii = valid[np.unique(np.linspace(0, len(valid)-1, min(len(valid), 500), dtype=int))]
            xs.append(curve['agreement'][ii].tolist()); ys.append(curve['completeness'][ii, GROUPS.index(group)].tolist())
        log['result/tradeoff/'+group] = wandb.plot.line_series(xs=xs, ys=ys, keys=list(curves),
                    title=group+' completeness (%)', xname='MER match fraction (%)')
    run.log(log)
    run.summary['decision'] = result['decision']
    run.summary['candidate_count'] = result['candidate_count']
    run.summary['optimizer_updates'] = 0
    run.summary['fallback_count'] = result['fallback_count']
    sys.path.insert(0, str(ROOT/'models/detection2/monitoring/_vendor'))
    import wandb_workspaces.reports.v2 as wr
    report = wr.Report(entity='AI-Astro', project='JAISP-Detection-Q1', title='Detection2 — final bounded rescoring test',
                       description='Detection completeness at matched catalogue agreement; one frozen scoring rule, no new training.')
    panels = []
    for i, group in enumerate(('full_mer', 'nir_only')):
        panels.append(wr.CustomChart(query={'summaryTable': {'tableKey': 'result/tradeoff/'+group+'_table'}},
                       chart_name='wandb/lineseries/v0', chart_fields={'step': 'step', 'lineKey': 'lineKey', 'lineVal': 'lineVal'},
                       chart_strings={'title': group+' completeness (%)', 'xname': 'MER match fraction (%)'},
                       layout=wr.Layout(x=12*i, y=0, w=12, h=9)))
    report.blocks = [wr.MarkdownBlock(text=(out/'comparison.md').read_text()),
                     wr.PanelGrid(runsets=[wr.Runset(entity='AI-Astro', project='JAISP-Detection-Q1', filters=f"ID in {[run.id]!r}")],
                                  hide_run_sets=True, panels=panels)]
    write_json(out/'report_spec.json', report._to_model().model_dump(mode='json', by_alias=True, exclude_none=True))
    report.save()
    write_json(out/'report.json', {'id': report.id, 'url': report.url})
    print('Detection comparison report:', report.url, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(); out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'logs').mkdir(); (out/'tiles').mkdir()
    spec = protocol(); write_json(out/'protocol.json', spec)
    import wandb
    run, processes, streams = None, [], []
    try:
        run = wandb.init(entity='AI-Astro', project='JAISP-Detection-Q1', group='detection2-final-rescore-v1',
                         job_type='detection-rescoring', name='One fixed decoder rescoring test', mode='online', dir=str(out),
                         config=spec, settings=wandb.Settings(init_timeout=45, disable_code=True, disable_git=True))
        write_json(out/'wandb_run.json', {'id': run.id, 'url': run.url})
        print('W&B:', run.url, flush=True)
        started = time.monotonic()
        for shard in range(2):
            stream = (out/'logs'/f'gpu{shard}.log').open('w'); streams.append(stream)
            command = [sys.executable, '-u', '-m', 'models.detection2.rescore.score', '--out', str(out),
                       '--shard', str(shard), '--shards', '2', '--device', f'cuda:{shard}']
            processes.append(subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes): raise RuntimeError('A scoring worker failed; see logs')
            if time.monotonic()-started > spec['inference_wall_limit_seconds']: raise TimeoutError('Bounded inference time exceeded')
            counts = [json.loads(p.read_text()) for p in out.glob('progress_*.json')]
            row = {'progress/tiles': sum(r['tiles_done'] for r in counts), 'progress/candidates': sum(r['candidates_done'] for r in counts),
                   'progress/elapsed_minutes': (time.monotonic()-started)/60}
            run.log(row); write_json(out/'status.json', {'stage': 'scoring', **row}); print(json.dumps(row), flush=True)
            time.sleep(10)
        if any(p.returncode != 0 for p in processes): raise RuntimeError('Scoring worker failed')
        write_json(out/'status.json', {'stage': 'evaluating_detection'})
        result, curves = evaluate(out)
        try:
            publish(run, out, result, curves)
        except Exception as exc:
            write_json(out/'publication_error.json', {'message': str(exc), 'type': type(exc).__name__})
            print('Publication error; complete scientific results are saved locally:', exc, flush=True)
        run.finish()
        write_json(out/'status.json', {'stage': 'complete', 'decision': result['decision'], 'elapsed_minutes': (time.monotonic()-started)/60})
        print((out/'comparison.md').read_text(), flush=True)
    except BaseException as exc:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill()
        if run: run.finish(exit_code=1)
        write_json(out/'status.json', {'stage': 'failed', 'message': str(exc), 'type': type(exc).__name__})
        raise
    finally:
        for stream in streams: stream.close()


if __name__ == '__main__':
    main()
