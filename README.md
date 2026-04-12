# JAISP (Joint AI Survey Processing)

Self-supervised multi-instrument foundation model for precision cosmology with Rubin Observatory and Euclid, trained on overlapping ECDFS imaging.

**Pipeline**: Foundation MAE (10 bands, native resolution per instrument) -> frozen encoder -> lightweight downstream heads for detection, astrometry, and photometry.

## Key Components

| Component | Directory | Description |
|-----------|-----------|-------------|
| Foundation (V7) | `models/jaisp_foundation_v7.py` | Mixed-resolution masked autoencoder |
| Detection | `models/detection/` | CenterNet heatmap source detector |
| Astrometry | `models/astrometry2/` | Rubin<->Euclid concordance field |
| Photometry | `models/photometry/` | PSF modeling + forced photometry |

## Documentation

- **[DOCUMENTATION.md](DOCUMENTATION.md)** -- Full project documentation (architecture, data, training, inference)
- `models/readme.md` -- Foundation model technical details
- `models/detection/README.md` -- Detection head architecture and training
- `models/astrometry2/README.md` -- Astrometry concordance pipeline

## Quick Start

```bash
# Foundation training (790 tiles, multi-GPU)
cd models && torchrun --nproc_per_node=2 train_jaisp_foundation_v7.py \
    --rubin_dir ../data/rubin_tiles_all --euclid_dir ../data/euclid_tiles_all \
    --output_dir ./checkpoints/jaisp_v7_concat --epochs 100 --lr 3e-4 \
    --hidden_ch 256 --cross_instrument_prob 1.0

# Detection (CenterNet self-training, 200-tile subset)
python models/detection/self_train.py \
    --feature_dir  data/cached_features_v7_rms_aware \
    --rubin_dir    data/rubin_tiles_200 \
    --euclid_dir   data/euclid_tiles_200 \
    --out_dir      checkpoints/centernet_v7_rms_aware \
    --rounds 2 --epochs 100 --batch_size 4

# Astrometry (V7 matcher + CenterNet sources)
cd models && python astrometry2/train_astro_v7.py \
    --v7-checkpoint       checkpoints/jaisp_v7_concat/checkpoint_best.pt \
    --detector-checkpoint ../checkpoints/centernet_v7_rms_aware/centernet_best.pt \
    --rubin-dir ../data/rubin_tiles_200 --euclid-dir ../data/euclid_tiles_200 \
    --multiband --epochs 120 --output-dir checkpoints/astro_v7_psffit

# Latent position head (per-object multi-band alignment)
python models/astrometry2/train_latent_position.py \
    --rubin-dir data/rubin_tiles_200 --euclid-dir data/euclid_tiles_200 \
    --v7-checkpoint models/checkpoints/jaisp_v7_concat/checkpoint_best.pt \
    --epochs 30 --lr 3e-4
```
