#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
DETECTION2_STUDY="${1:-models/detection2/runs/unknown_regions_20260915_v1_12ep}"
mkdir -p "$DETECTION2_STUDY/monitoring"
export MPLCONFIGDIR="$PWD/$DETECTION2_STUDY/matplotlib_cache"
if [[ "$DETECTION2_STUDY" == /* ]]; then
    export MPLCONFIGDIR="$DETECTION2_STUDY/matplotlib_cache"
fi
/usr/bin/python -u -m models.detection2.monitoring.live --study "$DETECTION2_STUDY" --watch \
    2>&1 | tee -a "$DETECTION2_STUDY/monitoring/console.log"
