# Astrometry2: Standalone Native-Resolution Matcher

This module is a standalone astrometry path that does not depend on the JAISP foundation model.

It is built around the structure the problem actually supports:

1. Detect and WCS-match candidate sources with classical methods.
2. Reproject Rubin locally into the VIS pixel frame using WCS.
3. Use a small native-resolution CNN patch matcher to predict subpixel local residual offsets.
4. Fit a low-order smooth control-grid field from those local offsets.

The learned part is local matching. The global field is constrained explicitly.

## Files

- `dataset.py`: builds matched local patch samples with WCS-based Rubin->VIS reprojection
- `matcher.py`: lightweight native-resolution local patch matcher
- `field_solver.py`: explicit weighted bilinear control-grid solver
- `train_local_matcher.py`: train the local matcher on matched patch supervision
- `infer_concordance.py`: run the matcher on a tile set and export FITS concordance fields

## Train

```bash
/home/shemmati/venvs/superres/bin/python models/astrometry2/train_local_matcher.py \
  --rubin-dir data/rubin_tiles_ecdfs \
  --euclid-dir data/euclid_tiles_ecdfs \
  --rubin-band r \
  --output-dir models/checkpoints/astrometry2_r \
  --epochs 20 \
  --batch-size 64 \
  --wandb-mode online
```

## Export Concordance

```bash
/home/shemmati/venvs/superres/bin/python models/astrometry2/infer_concordance.py \
  --rubin-dir data/rubin_tiles_ecdfs \
  --euclid-dir data/euclid_tiles_ecdfs \
  --checkpoint models/checkpoints/astrometry2_r/best_matcher.pt \
  --output models/checkpoints/astrometry2_r/concordance_r.fits
```

## Current Scope

This first version is intentionally conservative:

- one target Rubin band at a time
- optional extra Rubin context bands for the local patch matcher
- explicit smooth control grid instead of free dense field regression
- local WCS prealignment before any learned matching

That keeps the learning problem focused on the part DL should actually solve: local subpixel cross-modal alignment.
