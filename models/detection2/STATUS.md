# Setup status — 2026-09-15

**Active: control and unknown runs are training for 12 epochs each.**

The user approved extending the initial four-epoch budget. The original jobs
had already started, so both were stopped and restarted together from the same
initial checkpoint. The new cosine schedules have `T_max=12`. Label preparation
was reused: 682 tiles, 248,410 seed positions and 92,525 unknown candidate
positions (tile-pooled, not verified new sources).

- Control / GPU 0: https://wandb.ai/AI-Astro/JAISP-Detection-Q1/runs/mfviwjw1
- Unknown / GPU 1: https://wandb.ai/AI-Astro/JAISP-Detection-Q1/runs/rj1onedv
- Screen session: `detection2`, window `detection2-12ep`.
- Active output: `runs/unknown_regions_20260915_v1_12ep/`.
- Reused input: `runs/unknown_regions_20260915_v1/prepared/`.
- Comparison: [live report](https://wandb.ai/AI-Astro/JAISP-Detection-Q1/reports/Detection2-detection-progress--VmlldzoxNzkzOTM5NQ==).
- CPU scalar monitor: screen window `detection2-monitor`; see `monitoring/README.md`.

Both active configs record `epochs=12`, `epoch_override=12`, and
`cosine_T_max=12`. Their prepared input retains `epochs=4` as provenance; that
value is explicitly overridden for training. The original runs and logs remain
under `runs/unknown_regions_20260915_v1/`, marked superseded. The learned object
decoder remains untrained; the active pair is the detection-loss comparison.

The setup checks below describe the earlier, zero-update diagnostic phase.

The active setup follows the user's learned-model direction, with no precomputed
PSF dependency. Begin with `notebooks/01_unknown_regions.ipynb`.

## Completed setup checks

- Fixed configuration and paired training code: identical VIS+NISP seeds,
  starting checkpoint, data order and budget; unknown-region negative-loss
  masking is the only difference between control and experimental arms.
- Four-training-tile proposal pilot: **1,588 seed positions and 504 unlabelled
  candidate positions**; ignore disks cover **0.834%** of image area on average.
  Tile counts can repeat overlapping sky sources. These are proposals, not
  verified new objects. Full preparation subsequently completed for all 682 tiles.
- **24 ten-band candidate galleries** with seed overlays, saved locally.
- Notebook inspection cells executed successfully with rendered outputs saved;
  all four preparation/training/final-evaluation launch cells were skipped.
- **Seven tests passed**: negative-loss semantics; all 16 rotation/flip
  combinations; matching boundary; zero-object decoder output; catalogue-order
  invariance; position/appearance gradients; subpixel flux/centroid conservation.
- Real GPU forward/backward check on two augmented training samples: finite
  loss **1.07297**, finite gradients, **zero optimizer updates**. BatchNorm
  running buffers are restored before validation.
- The unchanged head's validation detection counts match the previous saved
  evaluation on one patch-25 tile at all four thresholds: 392 at 0.20, 310 at
  0.30, 252 at 0.40, and 205 at 0.50. This is an execution/protocol check,
  not a new performance measurement.
- Existing W&B account connection verified: **AI-Astro**. Training targets
  **JAISP-Detection-Q1**, group **detection2-unknown-regions-v1**, online mode.
  Run links are listed above. The W&B settings
  were instantiated successfully with code/git uploads disabled. Both active runs have now logged finite losses and completed optimizer updates.

## Entry points and artifacts

- `README.md`: experiment, comparison and metric definitions.
- `DESIGN.md`: proposed learned object-reconstruction architecture and stages.
- `configs/unknown_regions_v1.json`: explicit first-experiment settings.
- `notebooks/01_unknown_regions.ipynb`: inspection and foreground launches.
- `runs/pilot_dl_20260915/prepared/`: frozen four-tile seeds/proposals and summary.
- `runs/pilot_dl_20260915/gallery/`: candidate cutouts and coordinates/scores.
- `runs/pilot_dl_20260915/execution_check/check.json`: final GPU execution check.
- `runs/pilot_dl_20260915/validation_crosscheck.json`: saved-prediction cross-check.

The learned object decoder is an untrained architectural prototype. Its
structural tests do not establish detection accuracy or successful reconstruction.
The current training loop uses the seeded detection/unknown-background loss;
the learned reconstruction objective is a separate proposed next experiment.

## Superseded diagnostics

An initial PSF-fitting audit was superseded when the user clarified the desired
DL direction. Its files are under `runs/archive/superseded_psf_setup_20260915/`;
they are not inputs to the active experiment. They are not validated evidence
and must not be used as results. An earlier execution check that did not restore
BatchNorm buffers is retained as `execution_check_before_bn_restore/`; only
`execution_check/check.json` is the final check.

Automatic approval review rejected uploading diagnostic payloads to W&B.
Preparation and inspection were changed to stay entirely local. Training is
configured to send the authorized training/validation metrics, while candidate
images, audit records and model checkpoints remain local.
