#!/usr/bin/env bash
# Parallel Q1 detection driver: two GPU lanes, 35 epochs, batch sizes proven
# to fit (CenterNet 2, Stem 1). vis_peak is already trained (kept).
#   Lane A (GPU0): CenterNet mer  -> CenterNet vis_sep
#   Lane B (GPU1): StemCenterNet mer
# MER models (both heads) finish first; vis_sep bake-off comparison after.
# Guarded (skip if checkpoint exists), per-model logs, W&B on. Launch detached:
#   setsid bash models/detection/run_q1_detection_parallel.sh </dev/null \
#       > logs/q1_detection_parallel.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
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
CN_EPOCHS=35
STEM_EPOCHS=35
mkdir -p "$OUT" "$LOGD"
log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

cn(){  # $1=gpu $2=mode $3+=extra
  local gpu="$1" mode="$2"; shift 2
  local ckpt="$OUT/centernet_${mode}.pt"
  if [ -f "$ckpt" ]; then log "skip CenterNet[$mode] (exists)"; return 0; fi
  log "CenterNet[$mode] on GPU$gpu ..."
  CUDA_VISIBLE_DEVICES="$gpu" python3 models/detection/train_centernet.py \
    --feature_dir "$FEAT" --rubin_dir "$RUBIN" --euclid_dir "$EUCLID" \
    --labels_mode "$mode" "$@" --val_patches "$VAL_PATCHES" \
    --epochs "$CN_EPOCHS" --batch_size 2 --num_workers 6 \
    --wandb_project "$WANDB_PROJECT" --wandb_run "cn_${mode}" \
    --out "$ckpt" > "$LOGD/centernet_${mode}.log" 2>&1 \
    && log "CenterNet[$mode]: DONE" || log "CenterNet[$mode]: FAILED ($LOGD/centernet_${mode}.log)"
}
stem(){  # $1=gpu $2=mode $3+=extra
  local gpu="$1" mode="$2"; shift 2
  local ckpt="$OUT/stem_${mode}.pt"
  if [ -f "$ckpt" ]; then log "skip Stem[$mode] (exists)"; return 0; fi
  log "StemCenterNet[$mode] on GPU$gpu ..."
  CUDA_VISIBLE_DEVICES="$gpu" python3 models/detection/train_stem_centernet.py \
    --rubin_dir "$RUBIN" --euclid_dir "$EUCLID" --encoder_ckpt "$ENC" \
    --labels_mode "$mode" "$@" --val_patches "$VAL_PATCHES" \
    --epochs "$STEM_EPOCHS" --batch_size 1 --num_workers 6 \
    --wandb_project "$WANDB_PROJECT" --wandb_run "stem_${mode}" \
    --out "$ckpt" > "$LOGD/stem_${mode}.log" 2>&1 \
    && log "Stem[$mode]: DONE" || log "Stem[$mode]: FAILED ($LOGD/stem_${mode}.log)"
}

log "START parallel driver (2 GPUs, ${CN_EPOCHS}/${STEM_EPOCHS} ep)"
( cn 0 mer --mer_fits "$MER" ) &                     # Lane A: GPU0  (cn_mer ~8h)
LA=$!
( stem 1 mer --mer_fits "$MER"; cn 1 vis_sep ) &     # Lane B: GPU1  (stem_mer ~4h, then cn_vis_sep ~8h)
LB=$!
wait $LA; wait $LB
log "ALL DONE. Models in $OUT/ (centernet_{vis_peak,vis_sep,mer}, stem_mer). Next: injection bake-off eval."
