# JAISP

**Joint AI Survey Processing** for overlapping Rubin Observatory and Euclid
imaging.

JAISP trains a multi-instrument foundation model on 10-band Rubin+Euclid image
tiles, then reuses the frozen representation for source detection, per-object
astrometry, PSF modelling, photometry, and residual-field QA. The project is a
research codebase: the documentation records both the current runnable path and
the negative results that shaped it.

## Paper and public products

The paper (Hemmati et al. 2026, ApJ, submitted) is built from `paper/`. The
trained weights, detection lists, per-band positions, and concordance-field
maps it reports are released with the tagged `v1.0` GitHub release and
archived on Zenodo; `release/README.md` describes every file. Code is
BSD 3-Clause (`LICENSE`); released weights and catalogs are CC BY 4.0
(`LICENSE-DATA`). Please cite the paper (`CITATION.cff`).

Production checkpoints behind the paper:

- **Foundation**: `models/checkpoints/jaisp_v11_q1_soft/checkpoint_best.pt`
  (released as `jaisp_foundation.pt`).
- **Detection**: `checkpoints/q1_detection_v11/centernet_vis_sep.pt`, fused
  CenterNet at threshold `0.30` (released as `jaisp_detection_head.pt`).
- **Astrometry**: `models/checkpoints/latent_position_v11_anchored_v11labels/best.pt`,
  the VIS-anchored head (released as `jaisp_astrometry_head.pt`).

## Running inference

Inputs are the JAISP tile NPZs. Regenerate them from the public releases with
`io/getdata_batch.py` (Rubin DP1, on the Rubin Science Platform) and
`io/refetch_euclid_q1.py --tile-table io/tile_tables/jaisp_tiles_<field>.csv`
(Euclid Q1 MER from IRSA). Then run the two scripts that produced the paper's
catalogs; the exact commands are in `release/README.md`:

- `models/detection/run_centernet_detections.py` — tiles to detection list.
- `models/astrometry2/eval_latent_position.py --save-anchors` — detections to
  per-band classical and head-corrected positions.
- `io/export_release_catalogs.py` — both to FITS tables.

No GPU is required: the frozen encoder takes about a minute per tile on 8 CPU
cores and ~0.2 s per tile on one GPU.

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
