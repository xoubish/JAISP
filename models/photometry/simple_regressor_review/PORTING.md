# Porting guide: running simple_regressor on your own tiles and catalogue

This guide is for someone who has tiles in the JAISP `tiles_product` npz format and a MER-like
catalogue. They may cover another field, other coordinates, or another catalogue release.
Everything data-specific is a key in `configs/local.json`, so **no code edits should be
needed** for the cases below.

**Workflow:** edit `configs/local.json` → run `python -m simple_regressor.check_inputs --config
configs/local.json` until it says `ALL CHECKS PASSED` → run the notebook.

---

## 1. The data contract (what the code assumes)

### Tiles

- **Discovery.** `prepare` searches `tiles_root` **recursively** for files matching
  `tile_glob` (default `*_euclid.npz`).
- **Patch name.** The name of each tile's **parent folder** is its *patch*, e.g.
  `tract_5063/patch_14/tile_x00000_y00000_euclid.npz` is in `patch_14`. Patches drive the
  train/val/test split (§3).
- **Required npz keys**, defaults shown:

| config key | default | content |
|---|---|---|
| `img_key` | `img_VIS` | image, float, linear flux units (any unit; see §5) |
| `var_key` | `var_VIS` | variance map, same shape. Required; missing variance causes the tile to be rejected. |
| `wcs_key` | `wcs_VIS` | WCS, either a FITS header card string (Euclid tiles) or a dict of header keywords (Rubin tiles) |
| `img_plane` | `-1` | `-1` means img/var are 2-D. `k ≥ 0` means take plane `k` of a `[band, H, W]` cube. |

- **Pixel scale.** `native_pixscale` (default 0.1″/px) must match the WCS pixel scale;
  `check_inputs` verifies this. It sets the physical size of `stamp` (128 px = 12.8″), the
  centre-cut aperture, and the §3b baseline apertures.
- **Bad pixels.** Non-finite image pixels and nonfinite/nonpositive variance pixels are jointly
  masked: image and rms are both set to 0. Stamps with less than `valid_frac_min` (50%) valid image/variance pixels are rejected. **A
  separate integer mask array, such as the Rubin `mask`, is ignored.**
- **Unreadable tiles** are skipped and listed in `metadata.json` (`skipped_tiles`).
- **Overlapping tiles.** A source in several tiles keeps only its **most-centred** stamp.

### Catalogue

FITS table in HDU `catalog_hdu` (default 1). Column lookup is **case-insensitive**, so
`FLUX_DETECTION_TOTAL` and `flux_detection_total` both work.

| config key | default (MER Q1) | role |
|---|---|---|
| `id_column` | `object_id` | integer ID (cast to int64) |
| `ra_column`, `dec_column` | `ra`, `dec` | degrees, same frame as the tile WCS |
| `mag_column` | `mag_detection_total` | **selection** magnitude cut `mag_min < mag < mag_max` |
| `sel_flux_column`, `sel_fluxerr_column` | `flux_detection_total`, `fluxerr_detection_total` | **selection** SNR cut |
| `flux_column`, `fluxerr_column` | Kron, as above | **training label** and its error (the notebook's `LABEL` sets these) |
| `flux_zeropoint` | `23.9` | label mag = ZP − 2.5 log10(flux). 23.9 assumes **µJy**; set it for other units. |

Optional quality columns are used **only if present**; otherwise the cut is silently
skipped (`check_inputs` warns):

- `vis_det == 1` (`require_vis_det`);
- `spurious_flag == 0` (`require_spurious_zero`);
- `det_quality_flag & det_quality_mask == 0` (`det_quality_mask`, 0 = off).

### Selection actually applied (in `catalog.py`)

A source is kept if:

1. ra, dec, mag and the selection flux/err are finite, and flux and err > 0;
2. it passes the flags above;
3. `mag_min < mag_column < mag_max`;
4. selection SNR > `snr_min`;
5. the label flux and err are finite and > 0;
6. **the label SNR > `snr_min`** (see §4).

The printout says how many sources each label cut removed. After stamps are cut, the
**empty-centre cut** removes objects whose central r = `centre_aper_arcsec` aperture has
S/N < `centre_snr_min`. It is image-only and applies to train only by default.

---

## 2. Common porting scenarios

### A. Same format, different field or coordinates

Usually only paths and the split need changing:

```json
"tiles_root": "/data/tiles_product",
"catalog":    "/data/mer_catalogue_<field>.fits",
"val_patches":  ["patch_XX"],
"test_patches": ["patch_YY"]
```

Run `check_inputs` and confirm:

- tiles are found;
- the catalogue RA/Dec range overlaps the tiles ("N selected sources fully inside a stamp" > 0);
- the val/test patches exist.

### B. A different MER catalogue release (e.g. Q1 → DR1)

- **Column names.** If they changed, map them in the JSON (`*_column` keys). Case does not
  matter. In the notebook, `LABEL` must be a column whose error column is the same name with
  `flux_` → `fluxerr_`. Otherwise set `cfg.fluxerr_column` by hand after that line.
- **Flux units.** If they are not µJy, set `flux_zeropoint`.
- **Table HDU.** If the table is not in HDU 1, set `catalog_hdu`.
- **Flags.** If flag columns were renamed, the corresponding cut is **silently skipped**.
  Check the `check_inputs` warnings.
- **Inspect the error columns** before trusting a label (§4).
- **Image–catalogue consistency.** If the images and the catalogue come from different
  processing versions, expect zeropoint and centroid offsets. The §3b aperture baseline is
  the quickest way to measure this: compare its median Δmag and scatter with a known-good run.

### C. Another band from the same Euclid npz (Y/J/H)

The NISP images in the Euclid tiles are already resampled onto the VIS grid (0.1″/px):

```json
"img_key": "img_H", "var_key": "var_H", "wcs_key": "wcs_H"
```

Then set the label to the matching MER flux column. Also set `require_vis_det: false` if
NIR-only sources should be included. **This configuration has not been tested.**

### D. Rubin tiles (`tile_x*_y*.npz`, 6-band cube)

```json
"tile_glob": "tile_x?????_y?????.npz",
"img_key": "img", "var_key": "var", "wcs_key": "wcs_hdr",
"img_plane": 2,
"native_pixscale": 0.2,
"stamp": 64
```

- **Bands.** `img_plane` indexes the band list stored in the npz; `bands = [u,g,r,i,z,y]`,
  so 2 = r.
- **Stamp size.** `stamp: 64` keeps the same 12.8″ field of view at 0.2″/px.
- **What was tested.** `check_inputs` passes with these settings on tract 5063, but the
  pipeline has **not been trained** on Rubin.
- **Labels.** You need a catalogue with Rubin-band fluxes; a MER VIS label on a Rubin image
  makes no sense.
- **Masks.** The Rubin `mask` plane is not used.

---

## 3. Train/val/test split

- **By patch (default, `split_by="patch"`).** Whole patch folders are held out.
  - `val_patches` / `test_patches` set them explicitly. **Always set these for new data.**
  - If they are left empty, the code takes the last two patch names in **string** order.
    For tract 5063 that happens to give `patch_4` / `patch_5`, but on other data it may be
    arbitrary or very unbalanced.
  - The split counts are printed by `prepare`.
- **By RA stripe (`split_by="ra"`).** The alternative: `train_frac` / `val_frac` stripes in
  RA, with a `buffer_arcsec` guard band dropped. Use it if patch folders don't exist or are
  few.

---

## 4. Pitfalls found with MER Q1 (check these for any new release)

1. **Broken aperture errors.**
   - In Q1 ECDFS, 6–13% of `fluxerr_vis_{1-4}fwhm_aper` are ~10²–10⁴ µJy on ~1 µJy fluxes.
     The fraction grows with aperture size. **Kron has none.**
   - The fluxes themselves look normal.
   - These rows are removed by the label SNR cut. If you set `snr_min = 0`, they come back
     and contaminate the q-statistics and the error bars in the diagnostic plot.
   - With `mag_weighted=True` they get near-zero weight.
2. **The label set changes the sample.** Since the label SNR cut was added, each label keeps
   a slightly different set of objects. For Q1 ECDFS at SNR > 3 and mag < 25:

   | label | objects kept |
   |---|---|
   | Kron | 28.0k |
   | 1fwhm | 25.7k |
   | 2fwhm | 25.1k |
   | 3fwhm | 24.3k |
   | 4fwhm | 23.3k |

   Keep this in mind when comparing labels.
3. **Kron is not a fixed function of the pixels.** It depends on MER's adaptive aperture and
   deblending. The fixed-aperture labels are measured on **PSF-matched** images (VIS
   degraded to the worst NISP band), so the net must also learn that smoothing.
4. **Empty centres.** About 1–2% of catalogue positions have no source at the stamp centre in
   these images. The centre cut handles this for training.
5. **Integer centring.** Stamps are cut at the nearest pixel (no interpolation), so the source
   can be up to 0.5 px off-centre.

---

## 5. Things that do NOT transfer between datasets

- **Checkpoints are dataset-specific.** `best.pt` stores two things fitted on the training
  split:
  - `input_scale`, the global asinh scale of the image channel;
  - the aperture calibration coefficients and aperture feature scales;
  - the target scaler (mean and std of ln F).

  A checkpoint trained on one image unit, zeropoint or band will give wrong fluxes on
  another. **Retrain on new data.** Don't reuse a checkpoint unless the images are on the
  identical photometric system.
- **The image unit is never converted.** The network learns the image-unit → µJy mapping
  from the labels. This is fine within one dataset, but it is another reason a checkpoint
  doesn't transfer.
- **Stamp geometry is tied to pixel scale.** `stamp`, `bin_factor`, `centre_aper_arcsec` and
  `centre_sigma_frac` all assume the pixel scale above. Rescale `stamp` if you want the same
  angular size.

---

## 6. Outputs (one folder, overwritten each run)

The notebook and CLI both use the configured `output_dir` (the repository pilot uses
`runs/q1_v2_pilot/`):

| file | content |
|---|---|
| `stamps_cache.npz` | stamps `[N, 2, S, S]` (image, rms) as float32, plus `flux`, `fluxerr`, `mag`, `ra`, `dec`, `patch`, `object_id`, `split`, `centre_snr` |
| `metadata.json` | full resolved config, split info, skipped tiles, centre-cut counts. **Keep this with any result you share.** |
| `best.pt`, `last.pt`, `history.json` | checkpoints and the loss curve |
| `metrics_{split}.json`, `predictions_{split}.csv`, `diagnostic_{val,test}.png` | evaluation outputs |

---

## 7. Checklist for a new dataset

- [ ] `configs/local.json`: set `tiles_root`, `catalog`, `val_patches` and `test_patches`.
- [ ] Format keys: `tile_glob`, `img_key`, `var_key`, `wcs_key`, `img_plane` and
      `native_pixscale` if the tiles differ; the `*_column` keys,
      `catalog_hdu` and `flux_zeropoint` if the catalogue differs.
- [ ] `python -m simple_regressor.check_inputs --config configs/local.json` passes, and the
      overlap count is > 0 per tile.
- [ ] Look at the label error distribution: `log10(fluxerr)` should be one peak, not two (§4).
- [ ] Notebook setup cell: set `LABEL`, the cuts, `n_tiles` (start small, e.g. 24, to test
      the plumbing) and `epochs`.
- [ ] Run prepare. Check the split counts, the label-cut and centre-cut printouts, and the
      example stamps.
- [ ] Train, evaluate, then run the **§3b aperture baseline**. A CNN that doesn't beat the
      baseline at the bright end is not yet extracting the available information.
