# Learned object renderer: fixed-catalogue warm-up

Launch this stage after the completed detection-loss comparison. The selected
standard-loss CenterNet checkpoint is recorded in [`../selected_baseline.json`](../selected_baseline.json).
The new experiment learns to reconstruct ten-band images using compact object
vectors and learned source templates. **No supplied PSF or analytic source
profile is used.** Positions and object count stay fixed during this warm-up.

## Start in the existing screen session

From your terminal, reconnect with `screen -r detection2`. Inside screen,
press **Ctrl+A, then C** for a new window, then run:

```bash
cd /home/shemmati/Work/Projects/JAISP
bash models/detection2/decoder/run.sh
```

The foreground script first prepares the data on GPU 0, then launches **one
20-epoch run across GPUs 0 and 1** using PyTorch DDP. Detach with **Ctrl+A, then
D**. The old comparison stays in its own output directory. Do not start two
copies of this script on the two GPUs; the launcher coordinates both workers.

Default output: `models/detection2/runs/decoder_warmup_20260915_v1/`.
An optional path argument starts a separately named run. A fresh launch refuses
to overwrite an existing directory.

To recover interrupted training after preparation completed:

```bash
bash models/detection2/decoder/run.sh --resume
```

Recovery restores the last completed epoch's weights, optimizer, schedule and
RNG state, using the same W&B run ID. An interrupted partial epoch is replayed;
its already logged intermediate points may remain in W&B. An initial epoch-0
state is also saved before any update. Changed code, prepared metadata or GPU
count prevents recovery into the same run. Interrupted preparation is preserved;
start with a fresh output path if preparation did not complete.

## What is learned

1. Freeze proposals from the selected CenterNet head at score **>0.15**, using
   the existing NMS and Gaia/saturation mask. Apply the same detector rule to
   training and validation tiles. No MER labels are used for this stage.
2. At each proposal, sample a 3×3 feature neighbourhood from frozen v11
   full-image features: 2,304 values, projected through a 128-unit layer to a
   **32-dimensional object vector**.
3. Decode that vector plus a learned band embedding into a positive normalized
   template, and the vector into ten positive fluxes. Flux scaling by 100 is
   numerical scaling in normalized image units, not supplied flux supervision.
4. Place templates at fixed subpixel positions using differentiable bilinear
   placement and sum them. A scene can also predict one constant background
   per band from its mean object vector. There is no dense image bypass.

The shared template network has a 49×49 output grid. Templates have a fixed
4.8-arcsec center-to-center span: 49×49 on the existing Euclid 0.1-arcsec grids,
and downsampled/renormalized 25×25 on Rubin's 0.2-arcsec grid. The Euclid NIR
inputs are already resampled onto the stored 0.1-arcsec grid; these are not
native 0.3-arcsec NISP pixels. The stored WCS maps the same source into each band.
Flux outside a crop is discarded without renormalizing a boundary source.

Loss: valid-pixel Huber loss (delta 3), averaged within each scene and then
equally over scenes and bands, plus 0.01 times the mean squared template-centroid
offset in arcsec. A per-band training median and robust scale normalize images:
the median of pixel medians and of 1.4826×MAD across 32 evenly spaced training
tiles. Variance maps define valid pixels (finite, positive); they are not used
as calibrated independent-pixel noise weights. Artifact masks are projected
from the existing VIS mask to the stored band grids. No validation pixels fit
the normalization. Diagnostics labeled “training MAD units” use this robust scale.

Optimization: FP32, AdamW, learning rate 3e-4, cosine decay over 20 epochs to
3e-6, weight decay 1e-4, gradient clipping at norm 2. Select the final fixed
epoch; all epoch checkpoints are retained.

## Data and monitoring

- **682 training tiles**, excluding patch 25; 16 fixed 12.8-arcsec crops per
  tile, or **10,912 training scenes**. Half are sampled around proposals with
  random jitter; half are uniform. Crop choices are deterministic and saved.
- **108 patch-25 validation tiles**, eight crops each, or **864 validation
  scenes**. Crops and overlapping tiles can repeat sky objects; these are
  development reconstruction metrics, not unique-source completeness estimates.
- One tile per GPU per step gives **32 scenes per distributed step** and
  **341 training steps per epoch**. Each tile file is loaded once per epoch.
- All proposals whose templates can intersect a crop are included. A crop with
  over 128 objects or less than 80% valid pixels in any band is resampled,
  never silently truncated. Rejections and counts are saved per tile.
- Validation runs before training and after every epoch. Online W&B is required
  before the first update; initialization failure stops training.

The launcher prints the W&B run and focused reconstruction-report links and
saves them in `training/wandb_run.json` and `training/report.json`. A report
creation failure is recorded in `report_error.json`; numeric logging continues.
Publication alone can be retried with:

```bash
python -m models.detection2.decoder.report \
  --out models/detection2/runs/decoder_warmup_20260915_v1/training
```

The report shows per-band MSE reduction relative to the training-median image
and relative to the same model with object contributions removed. Higher is
better; zero means no improvement. It also shows RMSE within 1.5 arcsec of fixed
proposals and away from them, plus training/validation reconstruction loss.
Lower residuals are better. The learned-background comparison is an ablation
of the same prediction, not a separately trained optimal background model.

Numeric training/validation metrics are sent to W&B. Observed/reconstructed/
residual panels stay local and appear in
[`../notebooks/02_learned_decoder.ipynb`](../notebooks/02_learned_decoder.ipynb).
Opening or running the inspection notebook launches no training job.

Artifacts under the new run directory:

- `prepared/`: frozen crops, proposal features/coordinates, normalization,
  input/source hashes, tile statistics and progress.
- `launch_logs/`: preparation and training console logs; `exit_code.txt` is
  the launcher's final exit code.
- `training/`: W&B links, numeric history, per-band validation sums and counts,
  reconstruction curves and fixed-scene galleries for every epoch.
- `training/epoch_XX.pt`, `final.pt`, `training_state.pt`: local renderer
  checkpoints and the latest recovery state. Encoder/detector weights are frozen.

## Scope and checks

This stage checks whether the object bottleneck can learn useful image
appearance. Cached features see the full input image, so it is **ordinary
reconstruction**, not hidden-pixel prediction. It cannot yet discover omitted
objects or change detection completeness. A later stage must mask before
feature encoding and introduce position/presence learning and object addition/
removal. Better reconstruction alone does not establish source purity.

The local diagnostic uses four training tiles and two validation tiles at
`../runs/decoder_warmup_execution_check_20260915/`. The two-GPU execution check
performed three real forward/backward passes with **zero optimizer updates**,
identical synchronized gradient norms and unchanged parameter hashes. The tested
batches used approximately 0.47/0.44 GB peak allocated memory; these are sample
measurements, not a full-run memory bound. Partial preparation is accepted only
for `--check-only`, never full training. Five structural tests cover padding and
order invariance, flux/background constraints, training gradients, invalid-pixel
masking, and image/feature registration.
