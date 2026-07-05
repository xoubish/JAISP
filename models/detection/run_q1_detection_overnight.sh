#!/usr/bin/env bash
# Overnight Q1 detection driver: waits for the feature cache, then trains the
# 3-way CenterNet label bake-off (vis_peak / vis_sep / MER-Q1) + StemCenterNet
# on MER, all on a PATCH-DISJOINT split (hold out patch 25). Injection-recovery
# eval / label pick is a separate morning step (kept off the critical path).
#
# Robust by design: each training is guarded (skips if its checkpoint exists),
# logs to its own file, and a failure in one stage does not delete another's
# output. Launch detached so it survives disconnect:
#   setsid bash models/detection/run_q1_detection_overnight.sh </dev/null \
#       > logs/q1_detection_driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
export PYTHONPATH=models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FEAT=data/cached_features_v10_q1
RUBIN=data/rubin_tiles_all
EUCLID=data/euclid_tiles_all_q1
ENC=models/checkpoints/jaisp_v10_q1_long/checkpoint_best.pt
MER=data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits
OUT=checkpoints/q1_detection
LOGD=logs/q1_detection
WANDB_PROJECT=JAISP-Detection-Q1
VAL_PATCHES=25
CN_EPOCHS=60
STEM_EPOCHS=60
mkdir -p "$OUT" "$LOGD"

log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# ---- Stage 0: wait for the feature cache to finish ----
log "Stage 0: waiting for feature cache ($FEAT) ..."
while pgrep -f precompute_features.py >/dev/null 2>&1; do sleep 120; done
NF=$(ls "$FEAT" 2>/dev/null | grep -c aug)
log "feature cache: precompute process gone; $NF aug files present."
if [ "$NF" -lt 3000 ]; then
  log "WARNING: only $NF/3160 aug files — cache may be incomplete. Proceeding anyway (training uses whatever tiles are present)."
fi

run_centernet(){  # $1 = labels_mode ; $2 = extra flags
  local mode="$1"; shift
  local ckpt="$OUT/centernet_${mode}.pt"
  if [ -f "$ckpt" ]; then log "skip CenterNet[$mode] (exists)"; return 0; fi
  log "CenterNet[$mode]: training ($CN_EPOCHS ep, patch-disjoint val=$VAL_PATCHES) ..."
  python3 models/detection/train_centernet.py \
    --feature_dir "$FEAT" --rubin_dir "$RUBIN" --euclid_dir "$EUCLID" \
    --labels_mode "$mode" "$@" --val_patches "$VAL_PATCHES" \
    --epochs "$CN_EPOCHS" --batch_size 2 --num_workers 4 \
    --wandb_project "$WANDB_PROJECT" --wandb_run "cn_${mode}" \
    --out "$ckpt" > "$LOGD/centernet_${mode}.log" 2>&1 \
    && log "CenterNet[$mode]: DONE -> $ckpt" \
    || log "CenterNet[$mode]: FAILED (see $LOGD/centernet_${mode}.log)"
}

run_stem(){  # $1 = labels_mode ; $2 = extra flags
  local mode="$1"; shift
  local ckpt="$OUT/stem_${mode}.pt"
  if [ -f "$ckpt" ]; then log "skip Stem[$mode] (exists)"; return 0; fi
  log "StemCenterNet[$mode]: training ($STEM_EPOCHS ep, patch-disjoint val=$VAL_PATCHES) ..."
  python3 models/detection/train_stem_centernet.py \
    --rubin_dir "$RUBIN" --euclid_dir "$EUCLID" --encoder_ckpt "$ENC" \
    --labels_mode "$mode" "$@" --val_patches "$VAL_PATCHES" \
    --epochs "$STEM_EPOCHS" --batch_size 1 --num_workers 4 \
    --wandb_project "$WANDB_PROJECT" --wandb_run "stem_${mode}" \
    --out "$ckpt" > "$LOGD/stem_${mode}.log" 2>&1 \
    && log "Stem[$mode]: DONE -> $ckpt" \
    || log "Stem[$mode]: FAILED (see $LOGD/stem_${mode}.log)"
}

# ---- Stage 1: CenterNet 3-way label bake-off (fast; cached features) ----
log "Stage 1: CenterNet bake-off (vis_peak / vis_sep / mer)"
run_centernet vis_peak
run_centernet vis_sep
run_centernet mer --mer_fits "$MER"

# ---- Stage 2: StemCenterNet on MER (strong prior; raw tiles + encoder, slow) ----
log "Stage 2: StemCenterNet on MER"
run_stem mer --mer_fits "$MER"

log "ALL DONE. Models in $OUT/. Next (morning): injection-recovery bake-off eval to confirm the label winner."
