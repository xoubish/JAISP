#!/bin/bash
# Queued: CenterNet[mer] v11 on GPU0 after the chained injection evals finish.
cd /home/shemmati/Work/Projects/JAISP
until grep -q "ALL INJECTION EVALS DONE" logs/q1_detection_v11/chain.log 2>/dev/null; do sleep 300; done
[ -f checkpoints/q1_detection_v11/centernet_mer.pt ] && exit 0
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=models WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 models/detection/train_centernet.py \
  --feature_dir data/cached_features_v11_q1 --rubin_dir data/rubin_tiles_all --euclid_dir data/euclid_tiles_all_q1 \
  --labels_mode mer --mer_fits data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits \
  --val_patches 25 --epochs 60 --batch_size 2 --num_workers 4 \
  --wandb_project JAISP-Detection-Q1 --wandb_run cn_mer_v11 \
  --out checkpoints/q1_detection_v11/centernet_mer.pt \
  > logs/q1_detection_v11/centernet_mer_v11.log 2>&1
echo "[queue] cn_mer_v11 finished" >> logs/q1_detection_v11/chain.log
