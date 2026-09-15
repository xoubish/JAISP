#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/jaisp-decoder-matplotlib}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
log=models/detection2/runs/decoder_rescore_20260915_v1_console.log
/usr/bin/python -u -m models.detection2.rescore.run "$@" 2>&1 | tee -a "$log"
