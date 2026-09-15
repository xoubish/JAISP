#!/usr/bin/env bash
# Foreground launcher: run inside screen; detach only after starting this script.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

resume=false
if [[ "${1:-}" == "--help" ]]; then
    echo 'Usage: bash models/detection2/decoder/run.sh [--resume] [OUTPUT_DIRECTORY]'
    exit 0
fi
if [[ "${1:-}" == "--resume" ]]; then resume=true; shift; fi
if (( $# > 1 )); then echo 'Too many arguments; use --help.' >&2; exit 2; fi
study="${1:-models/detection2/runs/decoder_warmup_20260915_v1}"
python_bin="${DETECTION2_PYTHON:-/usr/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/jaisp-decoder-matplotlib}"

"$python_bin" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise SystemExit('Expose exactly two GPUs with CUDA_VISIBLE_DEVICES=0,1.')
print('Using both GPUs:', ', '.join(torch.cuda.get_device_name(i) for i in range(2)), flush=True)
PY

if "$resume"; then
    [[ -d "$study" ]] || { echo "No existing study: $study" >&2; exit 2; }
else
    mkdir "$study" || { echo 'Use a fresh output directory, or --resume for an interrupted run.' >&2; exit 2; }
fi
exec 9>"$study/launch.lock"
flock -n 9 || { echo 'This study already has an active launcher.' >&2; exit 2; }
mkdir -p "$study/launch_logs"
finish() { local code=$?; printf '%s\n' "$code" > "$study/exit_code.txt"; }
trap finish EXIT

if "$resume"; then
    "$python_bin" - "$study/prepared" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not (p/'metadata.json').exists() or not json.loads((p/'metadata.json').read_text())['complete']:
    raise SystemExit('Preparation was interrupted or partial. Preserve it and start with a fresh output directory.')
PY
else
    echo "Preparing 682 training tiles and 108 validation tiles. Output: $study"
    "$python_bin" -u -m models.detection2.decoder.prepare --out "$study/prepared" \
        2>&1 | tee "$study/launch_logs/preparation.log"
fi

args=()
if [[ -d "$study/training" ]]; then
    "$resume" || { echo 'Training output already exists.' >&2; exit 2; }
    [[ -f "$study/training/training_state.pt" ]] || { echo 'No recovery checkpoint; use a fresh output directory.' >&2; exit 2; }
    args+=(--resume)
fi
echo 'Starting one 20-epoch renderer run across both GPUs. W&B links will print below.'
"$python_bin" -u -m torch.distributed.run --standalone --nproc-per-node=2 \
    -m models.detection2.decoder.train --prepared "$study/prepared" --out "$study/training" "${args[@]}" \
    2>&1 | tee -a "$study/launch_logs/training.log"
echo "Finished successfully. Results: $study/training"
