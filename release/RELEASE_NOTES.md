# JAISP v1.0 — paper release

Weights, catalogs, and field maps for Hemmati et al. (2026), *JAISP: A Joint
Rubin–Euclid Foundation Model for Cross-Instrument Source Detection and
Astrometry* (ApJ, submitted). Code is BSD 3-Clause; the attached assets are
CC BY 4.0 (`LICENSE-DATA`). See `release/README.md` in the repository for a
description of every file and the commands that reproduce the catalogs.
Archived on Zenodo: https://doi.org/10.5281/zenodo.23598660

## Assets

Weights (PyTorch, weights-only):
- `jaisp_foundation.pt` — ten-band masked-band autoencoder, 9.2M parameters
- `jaisp_detection_head.pt` — fused CenterNet detection head
- `jaisp_stem_centernet_control.pt` — StemCenterNet control variant (Table 2)
- `jaisp_astrometry_head.pt` — VIS-anchored astrometry head

Catalogs (FITS/CSV):
- `jaisp_detections_ecdfs.fits`, `jaisp_detections_edfs.fits` — detection lists at the 0.30 working threshold
- `jaisp_positions_ecdfs.fits`, `jaisp_positions_edfs.fits` — per-band classical and head-corrected positions
- `jaisp_tiles_ecdfs.csv`, `jaisp_tiles_edfs.csv` — tile grid to regenerate the inputs from Rubin DP1 and Euclid Q1

Concordance-field maps (FITS, ECDFS):
- `jaisp_concordance_field_pinn_raw_ecdfs.fits`
- `jaisp_concordance_field_hgp_raw_ecdfs.fits`
- `jaisp_concordance_field_hgp_headcorrected_ecdfs.fits`

`SHA256SUMS` lists the checksum of every asset.

## Inputs

Rubin LSST Data Preview 1 `deep_coadd` tiles (tract 5063, ECDFS, and the EDF-S
tract) and Euclid Quick Data Release 1 MER mosaics (VIS, NISP Y/J/H), both public.
