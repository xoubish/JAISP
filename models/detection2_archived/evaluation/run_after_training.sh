#!/usr/bin/env bash
# Explicit screen worker; waits for successful paired training before GPU use.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
DETECTION2_STUDY="${1:-models/detection2/runs/unknown_regions_20260915_v1_12ep}"
mkdir -p "$DETECTION2_STUDY/full_validation"
export MPLCONFIGDIR="$PWD/$DETECTION2_STUDY/matplotlib_cache"
if [[ "$DETECTION2_STUDY" == /* ]]; then
    export MPLCONFIGDIR="$DETECTION2_STUDY/matplotlib_cache"
fi
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/usr/bin/python -u -m models.detection2.evaluation.run --study "$DETECTION2_STUDY" --wait --publish \
    2>&1 | tee -a "$DETECTION2_STUDY/full_validation/console.log"
