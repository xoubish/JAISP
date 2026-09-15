"""Run the authorized evaluation and paired injection pilot after both trains finish."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--field', required=True, choices=['ECDFS_patch25', 'EDF-S'])
    p.add_argument('--device', required=True)
    p.add_argument('--start-index', type=int, required=True)
    p.add_argument('--stop-index', type=int, required=True)
    args = p.parse_args()
    state = HERE/f'posttrain_{args.field}_state.json'

    def status(stage):
        state.write_text(json.dumps(dict(stage=stage, field=args.field, device=args.device), indent=2)+'\n')
        print(stage, flush=True)

    status('waiting_for_both_four_epoch_runs')
    while True:
        finished = []
        for name in ('vis_control', 'visnir'):
            log = HERE/f'{name}.log'
            finished.append((HERE/name/'final.pt').exists() and log.exists()
                            and 'Finished in ' in log.read_text()[-1000:])
        if all(finished):
            break
        time.sleep(5)
    checkpoints = [f'vis_control={HERE}/vis_control/final.pt', f'visnir={HERE}/visnir/final.pt']
    status('evaluating_held_out_catalogue')
    cmd = [sys.executable, '-u', str(ROOT/'models/detection/visnir_eval_experiment.py'),
           '--out', str(HERE/f'evaluation_{args.field}'), '--field', args.field, '--device', args.device]
    for item in checkpoints:
        cmd.extend(['--checkpoint', item])
    subprocess.run(cmd, cwd=ROOT, check=True)
    status('evaluating_paired_injections')
    cmd = [sys.executable, '-u', str(ROOT/'models/detection/visnir_inject_experiment.py'),
           '--out', str(HERE/f'injections_{args.start_index:02d}_{args.stop_index:02d}'),
           '--device', args.device, '--tile-stride', '4',
           '--start-index', str(args.start_index), '--stop-index', str(args.stop_index)]
    for item in [f'baseline={ROOT}/checkpoints/q1_detection_v11/centernet_vis_sep.pt']+checkpoints:
        cmd.extend(['--checkpoint', item])
    subprocess.run(cmd, cwd=ROOT, check=True)
    status('complete')


if __name__ == '__main__':
    main()
