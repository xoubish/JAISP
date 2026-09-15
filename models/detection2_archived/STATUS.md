# Setup status — 2026-09-15

**Archived; not used in the current paper.** See the
[retained experiment records](records/README.md). The completed results below
are followed by historical setup notes; references to planned or active work
in those notes describe the earlier stages, not current jobs.

## Experimental branch closed for paper finalization

The user requested one final bounded rescoring test and an end to further
detection experimentation for this paper. That test completed successfully in
`runs/decoder_rescore_20260915_v1/`, with **no optimizer updates**. The fixed
renderer-based score performed worse than the selected standard-loss detector.
At matched 87.23% MER agreement, full-MER completeness fell from 91.70% to
85.34%, and NIR completeness from 83.92% to 74.10%. At 94% agreement, both
completeness measures also decreased. Paired 4×4 and 3×3 spatial intervals
exclude zero for these losses. All 13,650 candidates were accounted for;
657 unsuitable crops retained neutral image evidence. The original detector's
reference/recovery/match counts at threshold 0.30 were reproduced exactly.

**Decision: reject this rescoring rule; no further score search or training.**
The selected standard-loss head remains the experimental baseline. The paper's
existing production detector and its injection/transfer results have not been
replaced by these development experiments. Manuscript changes require the
user's review of the proposed wording before application.

Result: [`comparison.md`](runs/decoder_rescore_20260915_v1/comparison.md).
Inspection: [`03_detection_rescoring.ipynb`](notebooks/03_detection_rescoring.ipynb).
W&B: [final rescoring report](https://wandb.ai/AI-Astro/JAISP-Detection-Q1/reports/Detection2-%E2%80%94-final-bounded-rescoring-test--VmlldzoxNzk0MDc3NQ==).
Four additional tests passed for image evidence and equal-agreement evaluation.

The renderer warm-up also completed all 20 epochs successfully before this
test. It reduced validation reconstruction loss by 37.45%, but froze the
detector and could not change detections. Its image gains were preparation,
not detection results. Earlier setup notes below are retained as history.

**Complete: both 12-epoch training runs and the full 108-tile evaluation finished successfully.**

The user accepted the **standard-loss final checkpoint** as the working
baseline for the next learned-object-decoder experiment. Its path, verified
SHA-256, training configuration and comparison are recorded in
[`selected_baseline.json`](selected_baseline.json). The learned decoder remains
untrained; recording this selection did not launch another training job.

**Learned renderer warm-up is ready for the user's screen launch.** The code,
configuration and commands are in [`decoder/README.md`](decoder/README.md), and
local monitoring is in [`notebooks/02_learned_decoder.ipynb`](notebooks/02_learned_decoder.ipynb).
It uses fixed proposals from the selected head, 32-dimensional object vectors,
learned templates/fluxes and constant backgrounds, without supplied PSFs. The
planned run is 20 epochs across both GPUs, with 10,912 training crops and 864
validation crops. Full preparation/training has not been started by the agent.

The four-training-tile/two-validation-tile preparation check produced 64/16
crops. Five renderer structural tests passed. Three real forward/backward
passes on both GPUs completed with zero updates and matching gradient norms
(0.02283186). Sample peak allocated memory was 0.467/0.439 GB. These are
execution checks of an untrained renderer, not performance results. Artifacts
are in `runs/decoder_warmup_execution_check_20260915/`.
All 17 detection2 tests passed, the five W&B report panels serialized locally,
the launch script passed its syntax check, and the inspection notebook executed
successfully with saved outputs. Final code hashes are in `setup_verification.json`
inside the execution-check directory. W&B run/report creation occurs when the
user launches full training; no decoder training run has been created yet.

Both training exit codes were zero. The unique-source comparison and paired
spatial intervals are saved in
`runs/unknown_regions_20260915_v1_12ep/full_validation/comparison.md` and the
[W&B report](https://wandb.ai/AI-Astro/JAISP-Detection-Q1/reports/Detection2-detection-progress--VmlldzoxNzkzOTM5NQ==).
At threshold 0.30, starting / standard / masked NIR completeness is
80.98% / 83.88% / 90.37%, while MER match fraction is
87.76% / 87.23% / 76.24%. The masking gain comes with a substantial decrease
in catalogue agreement. This is a development result on one patch; EDF-S has
not been run for these final heads.

Report publication initially failed on timestamp serialization. It was repaired
and retried from saved results, with no repeated inference or changed metrics.
`publication_verification.json` records successful scalar and report readback;
`publication_retry.json` preserves the results hash and publication-code hash.

The user approved extending the initial four-epoch budget. The original jobs
had already started, so both were stopped and restarted together from the same
initial checkpoint. The new cosine schedules have `T_max=12`. Label preparation
was reused: 682 tiles, 248,410 seed positions and 92,525 unknown candidate
positions (tile-pooled, not verified new sources).

- Control / GPU 0: https://wandb.ai/AI-Astro/JAISP-Detection-Q1/runs/mfviwjw1
- Unknown / GPU 1: https://wandb.ai/AI-Astro/JAISP-Detection-Q1/runs/rj1onedv
- Training ran in screen session `detection2`, window `detection2-12ep`.
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

## Full validation setup

The user approved expanding the comparison to all 108 patch-25 tiles and
estimating paired uncertainty across sky regions. Code and the fixed protocol
are in `evaluation/`; outputs are in the active study's `full_validation/`.
The footprint contains 9,391 distinct clean MER sources, including 2,513 NIR-only
and 2,670 VIS sources brighter than 24.5. Each source is counted once using a
fixed sky partition. Approximate 95% intervals use 5,000 paired region resamples,
with 4×4 primary regions and a 3×3 sensitivity check.

Five structural tests passed. The CPU execution check reproduced all original
12-tile baseline reference/recovery/detection counts at four thresholds and
verified zero paired differences and intervals for identical predictions.
Execution-check outputs are labelled explicitly and are not final model results.
The GPU worker ran in screen `detection2`, window `detection2-eval` (3), after
both training jobs completed. All 324 model/tile predictions are saved, and the
final results and paired intervals have been appended to the W&B report.

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

The initial object decoder prototype and the runnable renderer warm-up remain
untrained. Their structural tests do not establish detection accuracy or
successful reconstruction. The completed first pair used the seeded
detection/background losses; the next loop in `decoder/` learns image
reconstruction while keeping the detector fixed.

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
