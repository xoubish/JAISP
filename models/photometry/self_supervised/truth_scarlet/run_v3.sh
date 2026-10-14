#!/bin/bash
# Version-3 data (scenes_v3.py) and fine-tuning. Start inside tmux:
#   tmux new-session -d -s ts_v3 "bash models/photometry/self_supervised/truth_scarlet/run_v3.sh data"
#   tmux new-session -d -s ts_v3 "bash models/photometry/self_supervised/truth_scarlet/run_v3.sh train"   # after checking sizecheck
set -euo pipefail
cd "$(dirname "$0")/../../../.."
LOG=models/photometry/self_supervised/runs/truth_scarlet/run_v3.log
exec > >(tee -a "$LOG") 2>&1
M=models.photometry.self_supervised.truth_scarlet.scenes_v3
T=models.photometry.self_supervised.truth_scarlet.train
R=models/photometry/self_supervised/runs/truth_scarlet/runs
stage() { echo "=== $(date +%F_%T) $*"; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m $M "$@"; }
if [ "$1" = data ]; then
  stage library --split train --workers 40
  stage library --split val --workers 8
  stage sizecheck --split train
  stage export --split train --count 16000 --workers 40
  stage export --split val --count 800 --workers 40
  stage reference --workers 40
  echo "=== $(date +%F_%T) data done"
fi
if [ "$1" = train ]; then
  launch() {   # variant init-run device
    mkdir -p $R/$1_v3
    tmux new-session -d -s ts_$1_v3 "cd $PWD && OMP_NUM_THREADS=8 python3 -m $T --variant $1 --name $1_v3 --device $3 \
      --init $R/$2/best.pt --train-dirs train_v3 --val-dir val_v3 --extra-val val_v2 val --epochs 6 --lr 1e-4 >> $R/$1_v3/log.txt 2>&1"
    echo "launched $1_v3 from $2 on $3 in tmux session ts_$1_v3"
  }
  launch A2 A2_v2 "${DEV_A2:-cuda:0}"
  launch A3 A3_v2 "${DEV_A3:-cuda:1}"
fi
