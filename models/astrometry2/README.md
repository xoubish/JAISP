# Astrometry2: Rubin → Euclid VIS Concordance

## What this is for

Rubin and Euclid observe the same sky but their astrometric solutions are not perfectly aligned — there are small residual offsets between the two instruments, varying slowly across the field. For JAISP to combine Rubin and VIS data at the pixel level, we need to know, at every position in the VIS frame: "if I project a Rubin sky coordinate through the VIS WCS, by how many arcseconds is it off?"

The answer is a smooth 2D field of corrections (ΔRA*, ΔDec) stored as a coarse mesh over the VIS tile. This is the **astrometry concordance product**.

Key design principle (from the JAISP data product spec):
- The correction is stored in **sky coordinates** (arcsec), not pixel offsets.
- At every VIS pixel position (x, y): apply the correction to Rubin's sky coord before projecting onto the VIS grid.
- Perfect alignment means ΔRA* = ΔDec = 0 everywhere.
- WCS handles geometry; concordance handles the residual astrometric error.
- The correction mesh is sampled every ~8 VIS pixels (DSTEP=8, i.e. 0.8"), keeping files ~100 MB rather than multi-GB. Downstream code bilinearly interpolates to native resolution at runtime.

---

## Why a neural network?

Classical source matching gives you a discrete set of (Rubin source position, VIS source position, offset) triples — typically a few hundred per tile. To turn those sparse noisy measurements into a smooth continuous field you need to:

1. Suppress outliers (blended sources, bad detections).
2. Interpolate coherently across the tile.

A small CNN matcher operating at native VIS resolution does (1) better than sigma-clipping alone: it looks at the actual image morphology of each matched source pair and predicts how confident that measurement is. The learned uncertainty (log_sigma) then weights the control-grid field solver, so reliable point sources dominate the field and bad matches are down-weighted automatically.

---

## Pipeline overview

```
Tiles on disk
    │
    ▼
[source_matching.py]
    detect_sources()       — local-peak detection + subpixel centroiding in Rubin & VIS
    match_sources_wcs()    — mutual-nearest-neighbor match in sky coords + sigma-clip
    refine_centroids_in_band() — flux-weighted centroid refinement in the target Rubin band
    │
    ▼  per matched source pair:
[dataset.py]
    reproject_rubin_patch_to_vis()  — warp a 33×33 Rubin stamp onto the VIS pixel grid via WCS
    extract_vis_patch()             — cut the corresponding VIS stamp
    local_vis_pixel_to_sky_matrix() — compute the 2×2 pixel→sky Jacobian at that position
    target_offset_arcsec            — (VIS_ra - Rubin_ra)*cos(dec)*3600, (VIS_dec - Rubin_dec)*3600
    │
    ▼  for each (rubin_patch, vis_patch, pixel_to_sky, target_offset):
[matcher.py  LocalAstrometryMatcher]
    PatchEncoder          — lightweight CNN [C,H,W] → [hidden,H,W], per modality
    _weighted_cost_volume — spatially-weighted cross-correlation over a (2r+1)² search window
                            weights = rubin_feature_energy × Gaussian_center_prior
                            → soft-argmax → coarse (dx_px, dy_px)
    center-biased pooling — Gaussian-weighted spatial pooling for MLP features
    residual MLP          — refine coarse estimate, predict log_sigma
    pixel_to_sky Jacobian — convert pixel shift to (DRA*, DDec) in arcsec
    │
    ▼  per tile at inference:
[field_solver.py  solve_control_grid_field]
    bilinear control grid — sparse (anchor_xy, pred_offset, weight=1/sigma²) → grid coefficients
    smoothness prior      — Tikhonov regularization on finite differences of grid nodes
    evaluate_control_grid_mesh() — interpolate to a regular mesh at DSTEP spacing
    │
    ▼
[infer_concordance.py]
    FITS output           — one DRA + DDE image HDU pair per tile, with WCS and DSTEP metadata
```

---

## Files

| File | What it does |
|------|-------------|
| `matcher.py` | `LocalAstrometryMatcher` — the neural network |
| `dataset.py` | `build_patch_samples` builds all training samples from tile pairs; `MatchedPatchDataset` serves them with optional augmentation |
| `field_solver.py` | Pure-numpy bilinear control-grid solver + evaluator — no learning |
| `train_local_matcher.py` | Training loop with W&B logging and tile-level diagnostic previews |
| `infer_concordance.py` | Run a trained checkpoint over all tiles and write a FITS concordance file |
| `viz.py` | Diagnostic figures: raw offsets on VIS, solved field, component maps, residual vectors |

`source_matching.py` lives in `../astrometry/` and is shared with the older pipeline.

---

## Architecture details

### Spatially-weighted cost volume

The original approach computed the global mean correlation across all patch pixels equally. The problem: most pixels in a 33×33 patch are sky background, not source. Background pixels are noise — they dilute the astrometric signal coming from the actual star/galaxy at the center.

The fix is to weight each pixel's contribution by:

```
spatial_w[i,j] = rubin_feature_energy[i,j] × gaussian_center_prior[i,j]
```

- `rubin_feature_energy`: L2 norm of the Rubin feature vector at pixel (i,j). High where the source is bright/detected.
- `gaussian_center_prior`: Gaussian peaked at the patch center (sigma = 0.5 in normalized [-1,1] coords). Suppresses edge pixels which are often noisier and less relevant.

The result: the bright point source at the anchor position dominates the offset vote.

### Center-biased pooling

The same Gaussian weights are used for the MLP's feature pooling. Instead of global mean-pool (which equally weights source and background), the pooled representation is dominated by the source center.

### Uncertainty output

`log_sigma` is the predicted log standard deviation of the offset error (in arcsec). The training loss is:

```
L = radial_error / sigma + log_sigma
```

This is the negative log-likelihood of a Rayleigh distribution. It forces the model to be honest: if sigma is large (uncertain), the penalty is smaller, but you pay `log_sigma`. If sigma is small but the error is large, you pay `radial_error / sigma`. At inference, `weight = 1 / sigma²` is used to weight each anchor's contribution to the control-grid field solver.

Initial sigma is set to 50 mas (log(0.05) ≈ -3.0) so the loss is informative from epoch 1.

### Patch normalization

Raw flux patches are not cross-tile comparable — a source with flux=100 in a shallow tile may have the same S/N as a source with flux=10 in a deep tile. Feeding raw flux to the encoder means it must learn different features for the same physical object at different depths.

Each patch is background-subtracted and MAD-normalized before reaching the encoder:
```
patch_norm = (patch - median(patch)) / (1.4826 * MAD(patch))
```
Using the median background and MAD noise estimate (rather than mean/std) ensures the bright source itself does not bias the normalization. After this, background pixels cluster around 0 with std ≈ 1, and the source is a clean high-amplitude S/N peak. This also makes the `rubin_feature_energy` weighting in the cost volume physically meaningful: high energy = high S/N, not just high flux.

Applied per-channel for Rubin (each band independently) and to the VIS channel. Applied in `__getitem__` after augmentation so stored patches remain as raw flux for visualization.

### Augmentation

Both Rubin and VIS patches are flipped (horizontal and/or vertical) synchronously with equal probability. Since the target sky-coord offset is a physical measurement independent of display orientation, it is unchanged. The pixel→sky Jacobian is updated consistently: a horizontal flip negates the x-column of the Jacobian (a +x pixel shift now maps to the opposite sky direction). This up to 4× multiplies the effective training set size with no extra data.

---

## Precision target

Residuals should be below the native astrometric uncertainties of each survey:
- Rubin (LSST coadds): ~10–20 mas RMS
- Euclid VIS: ~5 mas RMS

Aim for `p68_total < 20 mas` on the validation set.

---

## Training

```bash
/home/shemmati/venvs/superres/bin/python models/astrometry2/train_local_matcher.py \
  --rubin-dir data/rubin_tiles_ecdfs \
  --euclid-dir data/euclid_tiles_ecdfs \
  --rubin-band r \
  --output-dir models/checkpoints/astrometry2_r \
  --epochs 30 \
  --batch-size 64 \
  --wandb-mode online
```

Key arguments:
- `--rubin-band`: which Rubin band to use as the alignment target (e.g. `r`, `i`, `z`)
- `--context-bands`: optional additional Rubin bands fed to the encoder alongside the target band
- `--detect-bands`: bands used to build the multi-band detection image for source finding (default: g r i z)
- `--max-patches-per-tile`: max matched sources to use per tile (default 64). With small data, raise this.
- `--search-radius`: half-width of the pixel-shift search window (default 3, meaning ±3 VIS pixels = ±0.3")
- `--patch-size`: must be odd (default 33 = 3.3" at VIS resolution)

Training prints per-epoch metrics in mas: `train_MAE`, `train_p68`, `val_MAE`, `val_p68`, plus pixel-space error and loss components.

---

## Export concordance FITS

```bash
/home/shemmati/venvs/superres/bin/python models/astrometry2/infer_concordance.py \
  --rubin-dir data/rubin_tiles_ecdfs \
  --euclid-dir data/euclid_tiles_ecdfs \
  --checkpoint models/checkpoints/astrometry2_r/best_matcher.pt \
  --output models/checkpoints/astrometry2_r/concordance_r.fits \
  --plot-dir models/checkpoints/astrometry2_r/plots
```

The output FITS file has one pair of HDUs per tile:
- `{tile_id}.r.DRA` — ΔRA* field in arcsec, sampled every `dstep` VIS pixels
- `{tile_id}.r.DDE` — ΔDec field in arcsec, sampled every `dstep` VIS pixels

Each HDU carries `DSTEP`, `DUNIT=arcsec`, `INTERP=bilinear`, `CONCRDNC=True`, `RBNBAND`, and a scaled WCS so the pixel coordinates map correctly to the VIS frame.

Downstream usage:
```
Rubin pixel (x_r, y_r)
  → Rubin WCS → (RA_r, Dec_r)
  → interpolate concordance at (RA_r, Dec_r) → (dra, ddec)
  → corrected sky: (RA_r + dra/cos(dec)/3600, Dec_r + ddec/3600)
  → VIS WCS inverse → VIS pixel (x_v, y_v)
```

---

## Design decisions and open questions

**Mesh size (DSTEP)**: Default is 8 VIS pixels (0.8"). Rubin astrometric distortions from DCR vary on arcminute scales; 0.8" sampling is likely overkill. The `smooth_lambda` regularizer keeps the field smooth regardless. Empirically, DSTEP=16 or 32 may give equally good results with fewer grid nodes.

**One band at a time**: Each checkpoint targets one Rubin band. DCR causes wavelength-dependent offsets, so each band should have its own concordance field. Use `--context-bands` to pass additional bands as encoder input context without making them the alignment target.

**No epoch correction**: The surveys are assumed to handle proper motion and parallax internally. The concordance corrects for systematic WCS residuals only.

**Scaling to Roman/NISP**: The reference frame encoder (`vis_encoder`) is hardcoded to 1-channel VIS input. To extend to NISP or Roman, instantiate a new matcher with the appropriate number of input channels for the reference instrument.
