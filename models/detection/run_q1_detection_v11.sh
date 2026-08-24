#!/usr/bin/env bash
# Overnight v11 detection retrains (foundation-gate step: user-approved 2026-08-22).
# Mirrors the production v10 recipes exactly, foundation swapped to jaisp_v11_q1_soft:
#   Lane A (GPU0): precompute v11 feature cache (4 augs, on data00, symlinked)
#                  -> CenterNet[vis_peak] from cache   (60 ep, bs 2 — overnight recipe)
#   Lane B (GPU1): StemCenterNet[mer] live encoder     (35 ep, bs 1 — parallel recipe)
# Guarded (skip if checkpoint exists), per-model logs, W&B offline. Launch detached:
#   setsid bash models/detection/run_q1_detection_v11.sh </dev/null \
#       > logs/q1_detection_v11_driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

RUBIN=data/rubin_tiles_all
EUCLID=data/euclid_tiles_all_q1
ENC=models/checkpoints/jaisp_v11_q1_soft/checkpoint_best.pt
MER=data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits
FEAT_REAL=/stage/irsa-jointproc-data00/JAISP/cached_features_v11_q1
FEAT=data/cached_features_v11_q1
OUT=checkpoints/q1_detection_v11
LOGD=logs/q1_detection_v11
VAL_PATCHES=25
mkdir -p "$OUT" "$LOGD" "$FEAT_REAL"
[ -e "$FEAT" ] || ln -s "$FEAT_REAL" "$FEAT"

log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

lane_b(){
  local ckpt="$OUT/stem_mer.pt"
  if [ -f "$ckpt" ]; then log "skip Stem[mer] (exists)"; return 0; fi
  log "Lane B (GPU1): StemCenterNet[mer] v11 (35 ep, bs 1, patch-disjoint val=$VAL_PATCHES)"
  CUDA_VISIBLE_DEVICES=1 python3 models/detection/train_stem_centernet.py \
    --rubin_dir "$RUBIN" --euclid_dir "$EUCLID" --encoder_ckpt "$ENC" \
    --labels_mode mer --mer_fits "$MER" --val_patches "$VAL_PATCHES" \
    --epochs 35 --batch_size 1 --num_workers 4 \
    --wandb_project JAISP-Detection-Q1 --wandb_run stem_mer_v11 \
    --out "$ckpt" > "$LOGD/stem_mer_v11.log" 2>&1 \
    && log "Stem[mer] v11: DONE -> $ckpt" \
    || log "Stem[mer] v11: FAILED (see $LOGD/stem_mer_v11.log)"
}

lane_a(){
  NF=$(ls "$FEAT_REAL" 2>/dev/null | grep -c aug || true)
  if [ "$NF" -lt 3000 ]; then
    log "Lane A (GPU0): precomputing v11 feature cache ($NF present) ..."
    CUDA_VISIBLE_DEVICES=0 python3 models/precompute_features.py \
      --rubin_dir "$RUBIN" --euclid_dir "$EUCLID" --encoder_ckpt "$ENC" \
      --out_dir "$FEAT_REAL" --n_augments 4 > "$LOGD/precompute_v11.log" 2>&1 \
      || { log "precompute FAILED (see $LOGD/precompute_v11.log)"; return 1; }
    log "v11 cache done: $(ls "$FEAT_REAL" | grep -c aug) aug files."
  else
    log "Lane A: v11 cache already present ($NF aug files)."
  fi
  local ckpt="$OUT/centernet_vis_peak.pt"
  if [ -f "$ckpt" ]; then log "skip CenterNet[vis_peak] (exists)"; return 0; fi
  log "Lane A (GPU0): CenterNet[vis_peak] v11 (60 ep, bs 2, patch-disjoint val=$VAL_PATCHES)"
  CUDA_VISIBLE_DEVICES=0 python3 models/detection/train_centernet.py \
    --feature_dir "$FEAT" --rubin_dir "$RUBIN" --euclid_dir "$EUCLID" \
    --labels_mode vis_peak --val_patches "$VAL_PATCHES" \
    --epochs 60 --batch_size 2 --num_workers 4 \
    --wandb_project JAISP-Detection-Q1 --wandb_run cn_vis_peak_v11 \
    --out "$ckpt" > "$LOGD/centernet_vis_peak_v11.log" 2>&1 \
    && log "CenterNet[vis_peak] v11: DONE -> $ckpt" \
    || log "CenterNet[vis_peak] v11: FAILED (see $LOGD/centernet_vis_peak_v11.log)"
}

lane_b & LB=$!
lane_a & LA=$!
wait "$LA" "$LB"
log "ALL DONE. Models in $OUT/. Morning step: bake-off eval (injection recovery + depth) vs the v10 production pair."
