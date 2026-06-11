#!/bin/bash
# Run after latent_position_v10_patchval25 training completes.
cd /home/shemmati/Work/Projects/JAISP
CUDA_VISIBLE_DEVICES=1 python models/astrometry2/eval_latent_position.py \
  --rubin-dir data/rubin_tiles_patch25 --euclid-dir data/euclid_tiles_patch25 \
  --foundation-checkpoint models/checkpoints/jaisp_v10_warmstart/checkpoint_best.pt \
  --head-checkpoint models/checkpoints/latent_position_v10_patchval25/best.pt \
  --features-cache-dir data/cached_features_v10_warmstart \
  --detector-labels data/detection_labels/centernet_v10_790_thresh03.pt \
  --save-anchors models/checkpoints/latent_position_v10_patchval25/anchors_patch25.npz \
  --output-dir models/checkpoints/latent_position_v10_patchval25/eval_patch25
