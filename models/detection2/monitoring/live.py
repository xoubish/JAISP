"""CPU-only scalar monitor. Never imports a detector or reads source images.

Mirrors the starting model as a constant reference, and logs threshold tradeoff
tables from completed validation JSON files. Original training runs are untouched.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time

import wandb

from ..common import write_json
from .data import DEFAULT_STUDY, PRIMARY, LABELS, read_snapshot, tradeoff, preview


def monitor(study, watch=False, poll_seconds=30):
    out = study / 'monitoring'
    out.mkdir(exist_ok=True)
    with (out / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = json.loads((study / 'control/config.json').read_text())
        wc = cfg['wandb']
        state_path = out / 'monitor_state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            'id': wandb.util.generate_id(), 'last_shared_epoch': -1}
        source_runs = {arm: json.loads((study / arm / 'wandb_run.json').read_text())['id']
                       for arm in ('control', 'unknown')}
        if 'source_runs' in state and state['source_runs'] != source_runs:
            raise ValueError('Monitor belongs to different training runs')
        state['source_runs'] = source_runs
        write_json(state_path, state)
        run = wandb.init(entity=wc['entity'], project=wc['project'], group=wc['group'],
                         id=state['id'], resume='allow', mode='online',
                         name='Starting model (reference)', job_type='comparison-monitor',
                         dir=str(out), config={'kind': 'constant baseline and derived scalar comparison',
                                              'source_runs': source_runs, 'epochs': cfg['epochs']},
                         settings=wandb.Settings(init_timeout=45, disable_code=True,
                                                 disable_git=True, x_disable_stats=True))
        state['url'] = run.url
        write_json(state_path, state)
        print('Comparison/reference run:', run.url, flush=True)
        run.define_metric('val/*', step_metric='epoch')
        started = time.monotonic()
        try:
            while True:
                snapshot = read_snapshot(study)
                latest = snapshot['latest_shared_epoch']
                if latest > state['last_shared_epoch']:
                    for epoch in range(state['last_shared_epoch'] + 1, latest + 1):
                        row = {'epoch': epoch, 'monitor/latest_shared_epoch': epoch}
                        # A fixed reference, not another optimization run.
                        row.update({f'val/t0.30/{k}': snapshot['baseline']['metrics'][f'val/t0.30/{k}'] for k in PRIMARY})
                        for name, metric in [('full_mer', 'full_mer_completeness'),
                                             ('nir_only', 'nir_only_completeness')]:
                            xs, ys, thresholds = tradeoff(snapshot, epoch, metric)
                            row[f'tradeoff/{name}'] = wandb.plot.line_series(
                                xs=xs, ys=ys, keys=LABELS,
                                title=f'{PRIMARY[metric]} (%) — shared epoch {epoch}',
                                xname='MER match fraction (%)')
                        run.log(row)
                        print(f'Updated comparison through shared epoch {epoch}; '
                              f'thresholds {thresholds}', flush=True)
                    state['last_shared_epoch'] = latest
                    write_json(state_path, state)
                    preview(study, snapshot)
                if not watch:
                    break
                if all((study / arm / 'final.pt').exists() for arm in ('control', 'unknown')) and latest >= cfg['epochs']:
                    run.summary['monitor/status'] = 'complete'
                    break
                exit_path = study / 'exit_status.json'
                if exit_path.exists():
                    run.summary['monitor/status'] = 'training processes exited'
                    break
                if time.monotonic() - started > 6 * 3600:
                    raise TimeoutError('Monitor stopped after six hours; restart explicitly if needed')
                time.sleep(poll_seconds)
            run.finish()
        except BaseException:
            run.finish(exit_code=1)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--preview-only', action='store_true')
    a = p.parse_args()
    if a.preview_only:
        preview(a.study, read_snapshot(a.study))
    else:
        monitor(a.study, a.watch)


if __name__ == '__main__':
    main()
