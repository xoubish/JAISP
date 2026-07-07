#!/usr/bin/env bash
# Q1 astrometry downstream: waits for the two latent-position heads to finish,
# then runs eval -> dedup -> PINN raw -> HGP raw/headresid (super-tight) and
# rebuilds Figs 8 & 9 on Q1. Launch detached:
#   nohup bash models/astrometry2/run_q1_astrometry_downstream.sh > logs/q1_astrometry_downstream.log 2>&1 &
set -eo pipefail
cd /home/shemmati/Work/Projects/JAISP

FOUND=models/checkpoints/jaisp_v10_q1_long/checkpoint_best.pt
RUBIN=data/rubin_tiles_all
EUCLID=data/euclid_tiles_all_q1
FEAT=data/cached_features_v10_q1
SEED=data/detection_labels/centernet_q1_790_vissep_thresh03.pt
DIR=models/checkpoints/latent_position_q1_vissep
HEAD=$DIR/best.pt
ANCH=$DIR/anchors_centernet_q1_vissep.npz
DEDUP=$DIR/anchors_centernet_q1_vissep_dedup.npz
BANDS=u,g,r,i,z,y,nisp_Y,nisp_J,nisp_H
HGP_PRIORS="--length-scales 300,900 --max-centers-per-scale 120 \
  --prior-common-mas 4.0 --prior-group-mas 2.0 --prior-band-mas 1.0 \
  --robust-iters 3 --huber-k 3.0 --dstep-arcsec 5.0 \
  --holdout-mode spatial --save-components --write-coverage --seed 42"

echo "=== $(date) waiting for latent_position training to finish ==="
# wait until no training process remains (this script does not match the pattern)
while pgrep -f "train_latent_position.py" >/dev/null 2>&1; do sleep 60; done
echo "=== $(date) training done; head=$HEAD ==="

# 1. eval prod head -> per-source anchors (raw + head_resid), gaussian centroids
echo "=== $(date) STEP 1/6: eval_latent_position ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=models python -u models/astrometry2/eval_latent_position.py \
  --rubin-dir $RUBIN --euclid-dir $EUCLID --foundation-checkpoint $FOUND \
  --head-checkpoint $HEAD --features-cache-dir $FEAT --detector-labels $SEED \
  --centroid-engine gaussian --save-anchors $ANCH \
  --output-dir $DIR/eval_centernet

# 2. dedup overlap-region duplicates
echo "=== $(date) STEP 2/6: dedup_anchors ==="
PYTHONPATH=models python -u models/astrometry2/dedup_anchors.py \
  --anchors $ANCH --output $DEDUP --radius-arcsec 0.05

# 3. PINN raw QA field
echo "=== $(date) STEP 3/6: fit_direct_pinn (raw) ==="
PYTHONPATH=models python -u models/astrometry2/fit_direct_pinn.py \
  --cache $DEDUP --output $DIR/concordance_pinn_q1_vissep_raw.fits

# 4. HGP raw, super-tight priors (the production-quality field view)
echo "=== $(date) STEP 4/6: HGP raw super-tight ==="
PYTHONPATH=models python -u models/astrometry2/fit_hierarchical_gp_concordance.py \
  --anchors $DEDUP --output $DIR/hgp_q1_vissep_raw_supertight.fits \
  --offset-kind raw --bands $BANDS $HGP_PRIORS

# 5. HGP head-residual, super-tight priors (after-head field)
echo "=== $(date) STEP 5/6: HGP head_resid super-tight ==="
PYTHONPATH=models python -u models/astrometry2/fit_hierarchical_gp_concordance.py \
  --anchors $DEDUP --output $DIR/hgp_q1_vissep_headresid_supertight.fits \
  --offset-kind head_resid --bands $BANDS $HGP_PRIORS

# 6. rebuild Figs 8 & 9 on Q1 (auto-upgrades Fig 7 too via its cache candidates)
echo "=== $(date) STEP 6/6: render Figs 7, 8, 9 ==="
for nbk in figure_7 figure_8 figure_9_concordance; do
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 \
    paper/paper_figures/$nbk.ipynb
done

echo "=== $(date) DONE. Q1 astrometry downstream complete; Figs 7/8/9 rendered ==="
echo "  NOTE: Fig 9 Gaia annotation still reads DR1 io/_nb09_outputs/gaia_selfcal_curve.json"
echo "        -- Gaia recheck on Q1 positions is a separate follow-up."