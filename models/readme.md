# JAISP Foundation v5 (Strict Position Encoding)

`JAISPFoundationV5` is a self-supervised cross-view foundation model for Rubin + Euclid image tiles.

It learns token embeddings that are:
- aligned across instruments/bands, and
- spatially consistent (same sky location should map to the same token position across views).

## What Changed in v5

Compared with earlier versions, v5 enforces **strict token-position matching**:
- Alignment loss uses `shift_px=0` (no spatial shift tolerance).
- Token `(i, j)` in one view is trained to match token `(i, j)` in the paired view after grid interpolation.

This is implemented in `models/jaisp_foundation_v5.py` via `AlignmentLoss`.

## Model Summary

Main components:
- Band-specific stems (`BandStem`) for per-band low-level differences.
- Signal-based information weighting (`InformationMap`) from SNR + gradients.
- Shared ViT-like encoder (`SharedEncoder`) with interpolated positional embeddings.
- Student projector + predictor.
- EMA teacher (teacher stems/encoder/projector).
- VICReg regularization (variance + covariance terms).

Core training objective:
- Bidirectional strict alignment (student predictor to teacher target across views).
- Information-map-weighted token loss.
- VICReg regularization to avoid collapse.

## Supported Bands

- Rubin: `rubin_u`, `rubin_g`, `rubin_r`, `rubin_i`, `rubin_z`, `rubin_y`
- Euclid: `euclid_VIS`, `euclid_Y`, `euclid_J`, `euclid_H`

Total: 10 bands.

## Data Assumptions (JAISPDatasetV4)

Implemented in `models/jaisp_dataset_v4.py`.

### Native resolutions
- Rubin tiles: typically `512x512`
- Euclid tiles: typically `1050x1050`
- Both represent the same sky area; model aligns at token-grid level.

### Expected filenames
- Rubin: `tile_x*_y*.npz`
- Euclid: `tile_x*_y*_euclid.npz`

### Expected NPZ keys
- Rubin NPZ:
  - `img`: shape `[6, H, W]`
  - `var`: shape `[6, H, W]` (converted to RMS inside dataset)
- Euclid NPZ:
  - `img_VIS`, `img_Y`, `img_J`, `img_H`
  - optional `var_VIS`, `var_Y`, `var_J`, `var_H`

## Training (v5)

Training script: `models/train_jaisp_foundation_v5.py`

Default behavior in script:
- Uses `JAISPFoundationV5` with strict alignment.
- Uses `make_loader(...)` from `jaisp_dataset_v4.py` (variable-size collate).
- Logs training and visual diagnostics to Weights & Biases.
- Saves checkpoints to `./checkpoints/jaisp_v5` (relative to run directory).

Run (from `models/` directory):

```bash
python3 train_jaisp_foundation_v5.py
```

If running from repo root, update the data paths in `main()` accordingly.

## Checkpoints

The training script writes:
- `best.pt` when loss improves
- periodic `ckpt_XXX.pt` based on `save_freq`

Default output directory in `main()`:
- `models/checkpoints/jaisp_v5` (if launched from `models/`)

## Initial Testing Notebook

Notebook: `models/testing_model.ipynb`

It provides three diagnostics for v5:
- Experiment 1: masked prediction similarity (full vs masked vs cross-view)
- Experiment 2: token-level cross-view correspondence maps
- Experiment 3: position-encoding test on source-containing masked regions

Default notebook paths assume running from `models/`:
- checkpoint: `checkpoints/jaisp_v5/best.pt`
- Rubin data: `../data/rubin_tiles_ecdfs/`
- Euclid data: `../data/euclid_tiles_ecdfs/`

## Related Files

- `models/jaisp_foundation_v5.py` - model + losses
- `models/jaisp_dataset_v4.py` - paired dataset + variable-size loader
- `models/train_jaisp_foundation_v5.py` - training loop + W&B visualization
- `models/testing_model.ipynb` - evaluation/diagnostics notebook
- `models/reconstruction/` - downstream masked-region reconstruction head pipeline
