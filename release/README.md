# JAISP v1.0 public products

Trained weights, catalogs, and field maps accompanying
Hemmati et al. (2026), *JAISP: A Joint Rubin–Euclid Foundation Model for
Cross-Instrument Source Detection and Astrometry* (ApJ, submitted).
Code: https://github.com/xoubish/JAISP (BSD 3-Clause). These products: CC BY 4.0
(see `LICENSE-DATA`). Zenodo DOI: *to be added at the tagged release*.

All products were derived from public imaging only: Rubin LSST Data Preview 1
`deep_coadd` tiles (tract 5063, ECDFS; EDF-S tract) and Euclid Quick Data
Release 1 MER mosaics (VIS, NISP Y/J/H) served by IRSA.

## Weights (`weights/`)

| file | what | size |
|---|---|---|
| `jaisp_foundation.pt` | Masked-band autoencoder, 9.2M parameters (paper Sec. 3; internal run `jaisp_v11_q1_soft`). Weights + config only; optimizer state stripped. | 37 MB |
| `jaisp_detection_head.pt` | Fused-bottleneck CenterNet detection head, 0.55M parameters (Sec. 4; `centernet_vis_sep`). | 2 MB |
| `jaisp_stem_centernet_control.pt` | StemCenterNet control variant used in Table 2. | 4 MB |
| `jaisp_astrometry_head.pt` | VIS-anchored astrometry head (`AnchoredHead`, 0.7M parameters; Sec. 5.2). | 3 MB |

Load the foundation with `models/load_foundation.py:load_foundation(path)`; the
heads are loaded by the scripts below. Inference does not need a GPU: the frozen
encoder takes ~1 min per tile (102" on a side) on 8 CPU cores, ~0.2 s on one GPU (bf16,
<5 GB). CPU (fp32) and GPU (bf16) inference agree to numerical precision; detections
scoring within ~0.01 of the 0.30 threshold can differ between the two.

`SHA256SUMS` lists the checksum of every released file.

## Catalogs (`catalogs/`)

Coordinates are ICRS degrees from the delivered Euclid Q1 MER VIS WCS; offsets
follow the paper's sign convention, VIS minus band (Eq. 2).

* `jaisp_detections_{ecdfs,edfs}.fits` — detection lists of the fused CenterNet
  head at the 0.30 working threshold, one row per detection per tile
  (`tile_id, x_vis, y_vis, ra, dec, score`). Tiles overlap by 50%, so a source in
  an overlap region appears once per tile; deduplicate on sky position if you need
  unique sources. ECDFS: 180,968 rows on 790 tiles; EDF-S: 20,639 rows on 72 tiles.
* `jaisp_positions_{ecdfs,edfs}.fits` — per-band positions for every detection
  matched between a band and VIS (Figs. 7–8): the classical Gaussian centroid in
  that band (`ra_band, dec_band`), the same source after the astrometry-head
  correction (`ra_head, dec_head`), the VIS centroid it is tied to
  (`ra_vis, dec_vis`), the raw and head-corrected offsets in mas
  (`dra_raw, ddec_raw, dra_head, ddec_head`), the head's predicted per-axis
  uncertainty (`sigma_head`, mas; uniformly conservative, see Sec. 5.3), and the
  band S/N. Both tables are deduplicated across tile overlaps (50 mas
  clustering of each band's entries, keeping the highest-S/N copy). ECDFS:
  154,451 rows; EDF-S: 17,073 rows. Sky positions are stored in float64; the
  paper's internal anchor files kept them in float32, so the clustering here
  differs from the paper's by 32 entries out of 154k.
* `jaisp_tiles_{ecdfs,edfs}.csv` — the tile grid (centres, Rubin pixel origins,
  MER tile ids) needed to regenerate the input tiles with the fetch scripts
  (`io/refetch_euclid_q1.py --tile-table ...` for Euclid from IRSA;
  `io/getdata_batch.py` for Rubin, which runs against the DP1 Butler on the
  Rubin Science Platform).

## Concordance-field maps (`fields/`)

Maps of the coherent cross-survey offset field over the ECDFS footprint
(Sec. 5.4, Fig. 10). Each file holds one image extension per band and axis
(`<BAND>.DRA`, `<BAND>.DDE`, in arcsec, VIS minus band) with a celestial WCS,
plus a `COVERAGE` map.

* `jaisp_concordance_field_pinn_raw_ecdfs.fits` — physics-informed network fit
  to the raw offsets (1" mesh).
* `jaisp_concordance_field_hgp_raw_ecdfs.fits` — hierarchical Gaussian-process
  fit to the raw offsets (5" mesh), with posterior std maps (`*_STD`) and the
  hierarchical decomposition into common, per-instrument, and per-band
  components (`COMP_*`).
* `jaisp_concordance_field_hgp_headcorrected_ecdfs.fits` — the same HGP fit to
  the head-corrected offsets (the ~1 mas residual field).

## Scope and caveats

The foundation was trained on one Rubin tract at DP1 and Q1 depth and has been
tested only on a second field observed with the same instruments and
processing. The detection head is VIS-anchored: the other bands add depth in
combination with VIS but rarely drive detections alone. The astrometry head
delivers positions in the VIS Gaussian-centroid convention; it does not improve
either survey's tie to Gaia. The weights are tied to the data releases they were
trained on and should be treated as a baseline to fine-tune or retrain as
deeper Rubin and Euclid releases arrive.

## Reproducing the catalogs

```bash
# detections (fused CenterNet, threshold 0.30, spike veto off)
python models/detection/run_centernet_detections.py \
    --encoder_ckpt release/weights/jaisp_foundation.pt \
    --centernet_ckpt release/weights/jaisp_detection_head.pt \
    --rubin_dir data/rubin_tiles_all --euclid_dir data/euclid_tiles_all_q1 \
    --out data/detection_labels/centernet_release.pt \
    --conf_threshold 0.3 --spike_veto_radius 0 --spike_veto_width 0

# per-band positions (classical centroids + astrometry-head correction)
PYTHONPATH=models python models/astrometry2/eval_latent_position.py \
    --rubin-dir data/rubin_tiles_all --euclid-dir data/euclid_tiles_all_q1 \
    --foundation-checkpoint release/weights/jaisp_foundation.pt \
    --head-checkpoint release/weights/jaisp_astrometry_head.pt \
    --detector-labels data/detection_labels/centernet_release.pt \
    --save-anchors anchors.npz --output-dir eval_release
python models/astrometry2/dedup_anchors.py --anchors anchors.npz --output anchors_dedup.npz --radius-arcsec 0.05

# FITS/CSV export of the above
python io/export_release_catalogs.py detections --field ecdfs --out jaisp_detections_ecdfs.fits
python io/export_release_catalogs.py positions  --anchors anchors_dedup.npz --out jaisp_positions_ecdfs.fits
```
