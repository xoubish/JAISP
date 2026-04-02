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
# Foundation training
python models/train_jaisp_foundation_v7.py \
    --rubin_dir data/rubin_tiles_ecdfs --euclid_dir data/euclid_tiles_ecdfs \
    --output_dir checkpoints/jaisp_v7_baseline --epochs 100

# Detection (CenterNet)
python models/detection/train_centernet.py \
    --rubin_dir data/rubin_tiles_ecdfs --euclid_dir data/euclid_tiles_ecdfs \
    --encoder_ckpt checkpoints/jaisp_v7_baseline/checkpoint_best.pt \
    --out models/checkpoints/centernet_v7.pt --epochs 100

# Astrometry
python models/astrometry2/train_astro_v7.py \
    --v7-checkpoint checkpoints/jaisp_v7_baseline/checkpoint_best.pt \
    --rubin-dir data/rubin_tiles_ecdfs --euclid-dir data/euclid_tiles_ecdfs \
    --multiband --output-dir models/checkpoints/astro_v7
```
