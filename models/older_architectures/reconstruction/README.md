# JAISP Masked Reconstruction Head

This folder adds a reconstruction pipeline to predict masked pixels in a target band from:
- the masked target image itself, and
- a sampled subset of context bands (`k->1`, including `9->1`).

## Files
- `head.py`: token-level reconstruction head + token-grid interpolation helper.
- `dataset.py`: multi-band dataset that samples one target band and random context bands per tile.
- `masking.py`: mixed mask generator (`random`, `object`, `hard`).
- `train_masked_reconstruction.py`: end-to-end training script.

## Quick Start
Run from repo root:

```bash
python3 models/reconstruction/train_masked_reconstruction.py \
  --rubin-dir data/rubin_tiles_ecdfs \
  --euclid-dir data/euclid_tiles_ecdfs \
  --backbone-ckpt checkpoints/jaisp_v5/best.pt \
  --output-dir checkpoints/jaisp_reconstruction \
  --epochs 30 \
  --batch-size 2 \
  --min-context 1 \
  --max-context 9
```

Notes:
- Default mode freezes backbone (`--freeze-backbone`).
- To fine-tune backbone too, add `--train-backbone`.
- Default mask mix is `random/object/hard = 0.5/0.4/0.1`.
- Reconstruction loss emphasizes masked area, with a small unmasked regularizer.

## Recommended First Run
Start with frozen backbone and short run:

```bash
python3 models/reconstruction/train_masked_reconstruction.py \
  --rubin-dir data/rubin_tiles_ecdfs \
  --euclid-dir data/euclid_tiles_ecdfs \
  --backbone-ckpt checkpoints/jaisp_v5/best.pt \
  --output-dir checkpoints/jaisp_reconstruction_smoke \
  --epochs 3 \
  --batch-size 1 \
  --num-workers 0
```

Then evaluate predicted masked patches against ground truth with PSNR/SSIM and source-level metrics.
