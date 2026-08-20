#!/bin/bash
# Two head retrains for the clamp-fix test (nb23 mechanism):
#   GPU0: H2 = asinh raw side-channel + brightness-balanced sampling  (the fix)
#   GPU1: H1 = balanced sampling only                                  (control)
cd /home/shemmati/Work/Projects/JAISP
COMMON="--rubin-dir data/rubin_tiles_all --euclid-dir data/euclid_tiles_all_q1 \
 --foundation-checkpoint models/checkpoints/jaisp_v10_q1_long/checkpoint_best.pt \
 --centernet-labels data/detection_labels/centernet_q1_790_vissep_thresh03.pt \
 --canonical-labels data/detection_labels/vis_refined_labels_q1_vissep.pt \
 --features-cache-dir data/cached_features_v10_q1 \
 --epochs 30 --seed 42 --wandb-mode disabled --balanced 1"

PYTHONPATH=models:models/astrometry2 nohup python3 models/astrometry2/train_latent_position_v2.py \
 $COMMON --raw-channel 1 --device cuda:0 \
 --output-dir models/checkpoints/latent_position_q1_clampfix_raw \
 > /tmp/claude-6100/-home-shemmati-Work-Projects-JAISP/258b79a7-d1d6-4edc-aa5c-c28e9dbcd8d6/scratchpad/train_raw.log 2>&1 &
echo "H2 (raw side-channel) pid $!"

PYTHONPATH=models:models/astrometry2 nohup python3 models/astrometry2/train_latent_position_v2.py \
 $COMMON --raw-channel 0 --device cuda:1 \
 --output-dir models/checkpoints/latent_position_q1_balanced_ctl \
 > /tmp/claude-6100/-home-shemmati-Work-Projects-JAISP/258b79a7-d1d6-4edc-aa5c-c28e9dbcd8d6/scratchpad/train_ctl.log 2>&1 &
echo "H1 (balanced control) pid $!"
