#!/bin/bash
# Version-4 fine-tuning (data: scenes_v3.py library4/export4 -> runs/truth_scarlet/train_v4; validation stays val_v3).
# Four runs, each in its own tmux session (resume with --resume auto):
#   A2_v4          main: A2 head (foundation, monotone x bounded shapes), new loss, 3 refinement steps
#   A5_v4          free-shape track (7x modulation), same loss and steps
#   A3_v4          foundation control (no foundation inputs), same loss and steps
#   A2_v4_oldloss  loss control: A2 on v4 data with the v3 loss and 2 steps
#   bash models/photometry/self_supervised/truth_scarlet/run_v4.sh
set -euo pipefail
cd "$(dirname "$0")/../../../.."
T=models.photometry.self_supervised.truth_scarlet.train
R=models/photometry/self_supervised/runs/truth_scarlet/runs
NEW="--lambda-frac 1 --frac-snr 10 --frac-scale 20 --bias-mode ema --lambda-bias 1 --steps 3"
OLD="--bias-mode batch --lambda-bias .5 --steps 2"
COMMON="--train-dirs train_v4 --val-dir val_v3 --extra-val val_v2 --epochs 5 --lr 1e-4"
launch() {   # name variant init-run device extra-args
  mkdir -p $R/$1
  tmux new-session -d -s ts_$1 "cd $PWD && OMP_NUM_THREADS=8 python3 -m $T --variant $2 --name $1 --device $4 --init $R/$3/best.pt $COMMON $5 >> $R/$1/log.txt 2>&1"
  echo "launched $1 ($2 from $3) on $4 in tmux session ts_$1"
}
launch A2_v4         A2 A2_v3  "${DEV0:-cuda:0}" "$NEW"
launch A5_v4         A5 A5_v3b "${DEV0:-cuda:0}" "$NEW"
launch A3_v4         A3 A3_v3  "${DEV1:-cuda:1}" "$NEW"
launch A2_v4_oldloss A2 A2_v3  "${DEV1:-cuda:1}" "$OLD"
