# JAISP (Joint AI Survey Processing)

Self-supervised multi-instrument foundation model for precision cosmology with Rubin Observatory and Euclid, trained on overlapping ECDFS imaging.

**Pipeline**: Foundation MAE (10 bands, native resolution per instrument) -> frozen encoder -> lightweight downstream heads for detection, per-object astrometry, concordance QA, PSF photometry, and scarlet-like residual photometry.

## Key Components

| Component | Directory | Description |
|-----------|-----------|-------------|
| Foundation (V10) | `models/jaisp_foundation_v10.py` | Current production multi-instrument MAE backbone |
| Detection | `models/detection/` | CenterNet and StemCenterNet source detectors |
| Astrometry | `models/astrometry2/` | Per-object Rubin/Euclid alignment head + concordance QA fields |
| PSF | `models/psf/` | Foundation-conditioned ePSF head and centroid refiner |
| Photometry | `models/photometry/` | Foundation photometry, rendered-stamp, and residual-scene experiments |

## Documentation

- **[DOCUMENTATION.md](DOCUMENTATION.md)** -- Full project documentation (architecture, data, training, inference)
- `models/readme.md` -- Foundation model technical details
- `models/detection/README.md` -- Detection head architecture and training
- `models/astrometry2/README.md` -- Astrometry head, centering diagnostics, and concordance QA

## Quick Start

```bash
# Export v10 CenterNet labels for astrometry anchors
PYTHONPATH=models python models/detection/run_centernet_detections.py \
    --encoder_ckpt      models/checkpoints/jaisp_v10_warmstart/checkpoint_best.pt \
    --centernet_ckpt    checkpoints/centernet_v10_uncertain_synth_r2/centernet_best.pt \
    --rubin_dir         data/rubin_tiles_all \
    --euclid_dir        data/euclid_tiles_all \
    --out               data/detection_labels/centernet_v10_790_thresh03.pt \
    --conf_threshold    0.3 \
    --spike_veto_radius 0 \
    --spike_veto_width  0

# Train the v10 Gaussian-centroid astrometry control
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=models python models/astrometry2/train_latent_position.py \
    --rubin-dir data/rubin_tiles_all --euclid-dir data/euclid_tiles_all \
    --foundation-checkpoint models/checkpoints/jaisp_v10_warmstart/checkpoint_best.pt \
    --centernet-labels data/detection_labels/centernet_v10_790_thresh03.pt \
    --output-dir models/checkpoints/latent_position_v10_no_psf \
    --epochs 30 --bottleneck-window 5 --dual-gpu

# Evaluate and export anchors for the v10 control
PYTHONPATH=models python models/astrometry2/eval_latent_position.py \
    --rubin-dir data/rubin_tiles_all --euclid-dir data/euclid_tiles_all \
    --foundation-checkpoint models/checkpoints/jaisp_v10_warmstart/checkpoint_best.pt \
    --head-checkpoint models/checkpoints/latent_position_v10_no_psf/best.pt \
    --detector-labels data/detection_labels/centernet_v10_790_thresh03.pt \
    --save-anchors models/checkpoints/latent_position_v10_no_psf/anchors_centernet_v10.npz \
    --output-dir models/checkpoints/latent_position_v10_no_psf/eval_centernet

# Fit the residual-field QA product with overlap-anchor deduplication
PYTHONPATH=models python models/astrometry2/fit_hierarchical_gp_concordance.py \
    --anchors models/checkpoints/latent_position_v10_no_psf/anchors_centernet_v10.npz \
    --output models/checkpoints/latent_position_v10_no_psf/concordance_hgp_head_resid_dedup.fits \
    --offset-kind head_resid --pool all --length-scales 60,180,600 \
    --dstep-arcsec 5 --dedup-radius-arcsec 0.05 --write-coverage
```

Current astrometry finding: the completed v8 baseline showed that the large raw 40-120 mas offsets are dominated by source centering, not by a smooth WCS field. The current v10 rerun keeps the Gaussian-centroid control, then tests `FoundationEPSFHead` centroiding as a separate PSF ablation.

Current photometry direction: port compact-source baselines from archived PSFField templates to the active `FoundationEPSFHead`; `models/photometry/scarlet_like.py` remains the per-scene optimizer baseline/refinement reference.
