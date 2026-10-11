#!/bin/bash
# Version-2 data and fine-tuning (see scenes_v2.py). Run from anywhere; meant to be started inside tmux:
#   tmux new-session -d -s ts_v2 "bash models/photometry/self_supervised/truth_scarlet/run_v2.sh"
# Each fine-tuning run gets its own tmux session (ts_A2_v2, ts_A3_v2, ts_A4_v2) and can be resumed with --resume auto.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
LOG=models/photometry/self_supervised/runs/truth_scarlet/run_v2.log
exec > >(tee -a "$LOG") 2>&1
M=models.photometry.self_supervised.truth_scarlet.scenes_v2
T=models.photometry.self_supervised.truth_scarlet.train
stage() { echo "=== $(date +%F_%T) $*"; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m $M "$@"; }
stage library --split train --workers 40
stage library --split val --workers 8
stage export --split train --count 16000 --workers 40
stage export --split val --count 800 --workers 40
stage reference --workers 40
echo "=== $(date +%F_%T) data done; starting fine-tuning"
R=models/photometry/self_supervised/runs/truth_scarlet/runs
launch() {   # variant device
  mkdir -p $R/$1_v2
  tmux new-session -d -s ts_$1_v2 "cd $PWD && OMP_NUM_THREADS=8 python3 -m $T --variant $1 --name $1_v2 --device $2 \
    --init $R/$1/best.pt --train-dirs train_v2 --val-dir val_v2 --extra-val val --epochs 6 --lr 1e-4 >> $R/$1_v2/log.txt 2>&1"
  echo "launched $1_v2 on $2 in tmux session ts_$1_v2"
}
launch A2 cuda:1
launch A3 cuda:1
launch A4 cuda:0
