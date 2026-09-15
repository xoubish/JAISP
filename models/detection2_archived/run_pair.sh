#!/usr/bin/env bash
# Run inside screen: prepare/reuse proposals, then train on separate GPUs.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
DETECTION2_PYTHON="${DETECTION2_PYTHON:-/usr/bin/python}"
DETECTION2_RUN_DIR="${1:-models/detection2/runs/unknown_regions_20260915_v1_12ep}"

if [[ "$DETECTION2_RUN_DIR" == --help ]]; then
    printf 'Usage: bash models/detection2/run_pair.sh [fresh_output_directory] [existing_prepared_directory]\n'
    printf 'Prepares or reuses labels; trains control on cuda:0 and unknown on cuda:1.\n'
    printf 'Both arms use the epoch budget in configs/unknown_regions_v1.json.\n'
    exit 0
fi

if [[ -e "$DETECTION2_RUN_DIR" ]]; then
    printf 'Output already exists: %s\nChoose a fresh directory; existing runs are never overwritten.\n' "$DETECTION2_RUN_DIR" >&2
    exit 2
fi

DETECTION2_PREPARED="${2:-$DETECTION2_RUN_DIR/prepared}"
if [[ $# -ge 2 && ! -f "$DETECTION2_PREPARED/labels.pt" ]]; then
    printf 'Prepared labels not found: %s/labels.pt\n' "$DETECTION2_PREPARED" >&2
    exit 2
fi
DETECTION2_EPOCHS=$("$DETECTION2_PYTHON" -c 'from models.detection2.common import read_config; print(read_config()["epochs"])')
"$DETECTION2_PYTHON" -c 'import torch; assert torch.cuda.device_count() >= 2, "Two visible CUDA GPUs are required"'
mkdir -p "$DETECTION2_RUN_DIR/launch_logs"
export MPLCONFIGDIR="$PWD/$DETECTION2_RUN_DIR/matplotlib_cache"
if [[ "$DETECTION2_RUN_DIR" == /* ]]; then
    export MPLCONFIGDIR="$DETECTION2_RUN_DIR/matplotlib_cache"
fi

printf 'Output: %s\nEpochs per arm: %s\n' "$DETECTION2_RUN_DIR" "$DETECTION2_EPOCHS"
if [[ $# -ge 2 ]]; then
    printf 'Reusing prepared labels: %s\n' "$DETECTION2_PREPARED" | tee "$DETECTION2_RUN_DIR/launch_logs/prepare.log"
else
    printf 'Preparing the shared labels on GPU 0 ...\n'
    "$DETECTION2_PYTHON" -u -m models.detection2.prepare \
        --config models/detection2/configs/unknown_regions_v1.json \
        --out "$DETECTION2_PREPARED" --device cuda:0 \
        2>&1 | tee "$DETECTION2_RUN_DIR/launch_logs/prepare.log"
fi

printf 'Starting control on GPU 0 and unknown on GPU 1. W&B URLs will appear below.\n'
"$DETECTION2_PYTHON" -u -m models.detection2.train \
    --prepared "$DETECTION2_PREPARED" --epochs "$DETECTION2_EPOCHS" \
    --out "$DETECTION2_RUN_DIR/control" --arm control --device cuda:0 \
    > "$DETECTION2_RUN_DIR/launch_logs/control.log" 2>&1 &
control_pid=$!

"$DETECTION2_PYTHON" -u -m models.detection2.train \
    --prepared "$DETECTION2_PREPARED" --epochs "$DETECTION2_EPOCHS" \
    --out "$DETECTION2_RUN_DIR/unknown" --arm unknown --device cuda:1 \
    > "$DETECTION2_RUN_DIR/launch_logs/unknown.log" 2>&1 &
unknown_pid=$!

# Stream both durable logs to the screen terminal. The Python processes write
# directly to their files, so stopping the log viewer does not interrupt them.
tail -n +1 --follow=name --retry --pid="$$" \
    "$DETECTION2_RUN_DIR/launch_logs/control.log" \
    "$DETECTION2_RUN_DIR/launch_logs/unknown.log" &
viewer_pid=$!

stop_children() {
    # Stop only still-running children of this launcher on an explicit signal.
    local pid
    while read -r pid; do
        kill -TERM "$pid" 2>/dev/null || true
    done < <(jobs -pr)
    wait || true
    exit 130
}
trap stop_children INT TERM HUP

control_status=0
unknown_status=0
wait "$control_pid" || control_status=$?
wait "$unknown_pid" || unknown_status=$?
kill "$viewer_pid" 2>/dev/null || true
wait "$viewer_pid" 2>/dev/null || true
printf '{"control_exit_code":%d,"unknown_exit_code":%d}\n' \
    "$control_status" "$unknown_status" > "$DETECTION2_RUN_DIR/exit_status.json"
printf '\nFinished: control exit=%d, unknown exit=%d.\n' "$control_status" "$unknown_status"
(( control_status == 0 && unknown_status == 0 ))
