# JAISP (Joint AI Survey Processing)

Self-supervised multi-instrument foundation model for precision cosmology with Rubin Observatory and Euclid, trained on overlapping ECDFS imaging.

**Pipeline**: Foundation MAE (10 bands, native resolution per instrument) -> frozen encoder -> lightweight downstream heads for detection, per-object astrometry, concordance QA, PSF photometry, and scarlet-like residual photometry.

## Key Components

| Component | Directory | Description |
|-----------|-----------|-------------|
| Foundation (V7) | `models/jaisp_foundation_v7.py` | Mixed-resolution masked autoencoder |
| Detection | `models/detection/` | CenterNet heatmap source detector |
| Astrometry | `models/astrometry2/` | Per-object Rubin/Euclid alignment head + concordance QA fields |
| Photometry | `models/photometry/` | PSFField forced photometry + scarlet-like residual scene fitting |

## Documentation

- **[DOCUMENTATION.md](DOCUMENTATION.md)** -- Full project documentation (architecture, data, training, inference)
- `models/readme.md` -- Foundation model technical details
- `models/detection/README.md` -- Detection head architecture and training
- `models/astrometry2/README.md` -- Astrometry head, centering diagnostics, and concordance QA

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

# Astrometry (current v8 latent head, per-object alignment)
PYTHONPATH=models python models/astrometry2/eval_latent_position.py \
    --rubin-dir data/rubin_tiles_all --euclid-dir data/euclid_tiles_all \
    --foundation-checkpoint models/checkpoints/jaisp_v8_fine/checkpoint_best.pt \
    --head-checkpoint models/checkpoints/latent_position_v8_no_psf/best.pt \
    --save-anchors models/checkpoints/latent_position_v8_no_psf/anchors.npz \
    --output-dir models/checkpoints/latent_position_v8_no_psf/eval

# Concordance QA/fallback field from exported anchors
PYTHONPATH=models python models/astrometry2/fit_direct_pinn.py \
    --cache models/checkpoints/latent_position_v8_no_psf/anchors.npz \
    --use-head-resid \
    --output models/checkpoints/latent_position_v8_no_psf/concordance_pinn_head_resid_fixed.fits \
    --bands r i g z --include-nisp
```

Current astrometry finding: the large raw 40-120 mas offsets are dominated by source centering, not by a smooth WCS field. The v8 latent head reduces most bands to ~9-15 mas median residuals (Rubin u: ~30 mas); residual concordance fields after the head are ~1 mas and mainly serve as QA.

Current photometry direction: use PSFField matched-filter photometry as the compact-source baseline, then train `models/photometry/foundation_head.py` on frozen V8 features. The training path is CenterNet detections -> latent astrometry correction -> V8 morphology head -> PSFField renderer -> residual chi-square. `models/photometry/scarlet_like.py` remains the per-scene optimizer baseline/refinement reference.
