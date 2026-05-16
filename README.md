# JAISP

**Joint AI Survey Processing** for overlapping Rubin Observatory and Euclid
imaging.

JAISP trains a multi-instrument foundation model on 10-band Rubin+Euclid image
tiles, then reuses the frozen representation for source detection, per-object
astrometry, PSF modelling, photometry, and residual-field QA. The project is a
research codebase: the documentation records both the current runnable path and
the negative results that shaped it.

## Current Direction

- **Foundation**: v10 warm-start MAE, `models/checkpoints/jaisp_v10_warmstart/checkpoint_best.pt`.
- **Detection**: v10 CenterNet/StemCenterNet label export, with CenterNet threshold `0.3` for the current astrometry rerun.
- **Astrometry**: v10 latent-position rerun using cached v10 bottlenecks, with a Gaussian-centroid control and a FoundationEPSF centroid ablation.
- **PSF**: foundation-conditioned ePSF head trained on Gaia-selected stars with proper-motion correction.
- **Photometry**: active migration away from archived PSFField dependencies toward the current ePSF head.

## Documentation

`DOCUMENTATION.md` is the single source of truth for architecture, data layout,
checkpoints, commands, current results, and historical context:

**[Read the full documentation](DOCUMENTATION.md)**

The short version of the current astrometry experiment is:

1. Export v10 CenterNet labels at threshold `0.3`.
2. Train `latent_position_v10_no_psf` with Gaussian centroids.
3. Train `latent_position_v10_epsf_centroid` with the improved ePSF centroid engine.
4. Evaluate both, export anchors, and run HGP QA with overlap-anchor deduplication.

See `DOCUMENTATION.md` for the exact commands.
