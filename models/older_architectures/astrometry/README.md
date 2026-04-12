# JAISP Astrometry Concordance (ARCHIVED -- v5/v6 era)

> **Note**: This is the original astrometry module using the v5/v6 foundation backbone. The current astrometry pipeline is in `models/astrometry2/` using the V7 matcher. Kept for historical reference.

This folder now has two explicit paths:

1. `fit_concordance_no_neural_baseline.py`
   - direct, non-neural baseline
   - measures Rubin->VIS offsets from matched sources
   - fits a smooth field (`affine + RBF residual`)
   - useful as a sanity check and as a teacher-label generator

2. `train_astrometry_multiband_teacher.py`
   - the current neural path
   - frozen JAISP foundation backbone
   - `AstrometryConcordanceHead` on top
   - supervised by dense smooth teacher fields from the direct matcher
   - uses multiple Rubin bands as latent-space context while still targeting one Rubin band at a time

Older experiments that are not the recommended path anymore are under `models/astrometry/older/`.
Thin wrappers are left at the old entrypoints so existing commands still run.

## Current Layout

- `head.py`: concordance heads (`AstrometryConcordanceHead`, `NonParametricConcordanceHead`)
- `source_matching.py`: shared source detection + WCS matching utilities
- `teacher_fields.py`: shared smooth-field fitting utilities used by both baseline and neural trainer
- `fit_concordance_no_neural_baseline.py`: explicit no-neural baseline
- `train_astrometry_multiband_teacher.py`: current recommended trainer
- `export_fits.py`: export neural predictions to FITS
- `inspect_concordance_fits.ipynb`: visual inspection notebook
- `older/`: legacy synthetic and sparse-point trainers

## How The Current Neural Path Works

For a target band (for example `rubin_g -> euclid_VIS`):

1. Detect and match Rubin/VIS sources.
2. Fit a smooth teacher field from those matches.
3. Load multiple Rubin bands through the frozen foundation.
4. Fuse Rubin latent features by blending:
   - the target-band latent
   - the mean latent of the other context bands
5. Feed the fused Rubin latent plus VIS latent into `AstrometryConcordanceHead`.
6. Train against the dense teacher field.

This keeps the head on top of the foundation, but avoids the old failure mode of learning from tiny synthetic shifts or from sparse point labels alone.

## Recommended Commands

### Non-Neural Baseline (all Rubin bands -> VIS)

```bash
python3 models/astrometry/fit_concordance_no_neural_baseline.py \
  --rubin-dir data/rubin_tiles_ecdfs \
  --euclid-dir data/euclid_tiles_ecdfs \
  --rubin-bands all \
  --detect-bands g r i z \
  --output models/checkpoints/concordance_no_neural_allbands.fits \
  --summary-json models/checkpoints/concordance_no_neural_allbands.json
```

### Neural Trainer (all target bands, multiband context)

```bash
python3 models/astrometry/train_astrometry_multiband_teacher.py \
  --rubin-dir data/rubin_tiles_ecdfs \
  --euclid-dir data/euclid_tiles_ecdfs \
  --output-dir models/checkpoints/jaisp_astrometry_multiband_teacher \
  --target-bands all \
  --context-bands all \
  --detect-bands g r i z \
  --use-stem-refine \
  --stem-stride 2 \
  --epochs 50 \
  --batch-size 2 \
  --wandb-mode online
```

### Legacy Trainers

These still work, but they are compatibility wrappers now:

- `train_astrometry.py` -> old synthetic trainer
- `train_astrometry_pseudolabel.py` -> old sparse-point pseudo-label trainer

## W&B Visuals

`train_astrometry_multiband_teacher.py` logs a fixed preview tile each epoch:

- VIS image with matched source offsets
- teacher quiver field on VIS
- predicted quiver field on VIS
- residual field on the mesh
- teacher vs prediction scatter on the mesh

That preview is intentionally fixed to a held-out validation tile when `--val-frac > 0`, so the images are comparable across epochs.
