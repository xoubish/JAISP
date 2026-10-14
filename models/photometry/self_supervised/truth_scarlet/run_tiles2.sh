#!/bin/bash
# 150 more training tiles (outside the held-out partitions and the first 48 training tiles) for a larger donor library.
set -uo pipefail
cd "$(dirname "$0")/../../../.."
export JAISP_DETCAT_OUT=models/photometry/self_supervised/runs/amortised_scarlet/training_tiles_2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=models.photometry.self_supervised.detcat
mkdir -p models/photometry/self_supervised/runs/amortised_scarlet/training_tiles_2
for step in "select_tiles --training 150 --allow-prior-tiles --exclude models/photometry/self_supervised/runs/truth_scarlet/heldout_tiles.json --exclude-same models/photometry/self_supervised/runs/amortised_scarlet/training_tiles/tiles.json" sources "fetch" prepare mer; do
  echo "=== $(date +%F_%T) $step"; python3 -m $P.$step || echo "=== step $step FAILED"
done
echo "=== $(date +%F_%T) tiles2 done"
