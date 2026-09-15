# VIS + NISP detection-head experiment

The production detection head was supervised with VIS-derived SEP centroids.
An unlabelled NIR-only source therefore contributes negative heatmap loss even
though the frozen representation includes all ten bands. This experiment tests
whether adding NISP image detections to the supervision improves NIR recovery.

## Fixed protocol

- Preserve production checkpoint `checkpoints/q1_detection_v11/centernet_vis_sep.pt`.
- Use the existing frozen v11 features and unchanged 554,052-parameter head.
- Exclude patch 25 and all EDF-S data from head training. There are 682 training
  tiles with four cached augmentations each (2,728 training samples).
- Preserve every existing VIS label and add NISP Y/J/H detections. Inverse-variance
  stacking, Gaussian smoothing (sigma 2.1 VIS pixels), empirical local RMS,
  SEP threshold 3, minimum area 10, deblend contrast 0.005, five-pixel duplicate
  removal, and the existing Gaia/VIS saturation mask for added NISP labels.
- No MER positions or flags construct the labels. A four-tile training-only
  smoke check was compared to MER to check label coverage; no held-out metrics
  were used to set extraction parameters.
- Run two fine-tunes from the same production checkpoint: VIS-only control and
  VIS+NISP. Same sample order, AdamW, seed 42, batch size 2, four epochs, learning
  rate 3e-5 with cosine decay, weight decay 1e-4, clip norm 1, target sigma 2.
  Convolutions use bfloat16; losses use float32.
- Compare the final fourth-epoch checkpoints. No held-out checkpoint selection.
- Evaluate all 108 patch-25 tiles and 72 EDF-S tiles. The primary threshold is
  the existing 0.30, NMS kernel 7, match radius 0.5 arcsec. Preserve existing
  masks, four-pixel catalogue margin, and EDF-S purity footprint.
- Report clean VIS <24.5 completeness, all clean VIS completeness, NIR-only
  completeness, full clean MER completeness, and full-MER match purity.

## Interpretation

This is a short, paired fine-tuning experiment, not a claim of fully converged
training. MER completeness is agreement with a reference catalogue and is not
absolute truth. The NIR-only denominator has no magnitude cut: the compact FITS
column `mag_vis` contains detection-band magnitude for NIR-only objects.
Tile-pooled catalogue counts repeat objects in overlapping tiles, matching the
paper's existing measurement convention.

The manuscript, paper figures and production checkpoint are unchanged.
Training histories, per-run configuration, checkpoints and evaluation detections
are saved under this directory. Baseline measurements preceding this experiment
are in `baseline_full_mer.json` and `baseline_nir_diagnosis.json`.

## Commands

```bash
python models/detection/visnir_labels_experiment.py \
  --out experiments/visnir_head_20260914/labels --workers 6

python models/detection/visnir_train_experiment.py \
  --out experiments/visnir_head_20260914/vis_control --device cuda:0

python models/detection/visnir_train_experiment.py \
  --out experiments/visnir_head_20260914/visnir --device cuda:1 \
  --extra-labels experiments/visnir_head_20260914/labels/nir_extra_labels.pt

python models/detection/visnir_eval_experiment.py \
  --checkpoint baseline=checkpoints/q1_detection_v11/centernet_vis_sep.pt \
  --checkpoint vis_control=experiments/visnir_head_20260914/vis_control/final.pt \
  --checkpoint visnir=experiments/visnir_head_20260914/visnir/final.pt \
  --out experiments/visnir_head_20260914/evaluation
```
