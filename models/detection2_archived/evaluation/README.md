# Full patch-25 validation

This is the final-epoch comparison of the **starting model**, **standard loss**
(`control`) and **masked background loss** (`unknown`). All three use the same
existing CenterNet architecture. The frozen, predeclared settings are saved in
`runs/unknown_regions_20260915_v1_12ep/full_validation/protocol.json`.

The 108 eligible tiles contain **9,391 distinct clean MER objects**: 6,878 VIS
and 2,513 NIR-only; the bright VIS subset contains 2,670 objects. Pooling tiles
would count 34,175 entries, so it substantially repeats sky sources.

**Completed on 2026-09-15.** Both training runs exited successfully, all 324
model/tile predictions were evaluated, and the results with paired intervals
are in the existing W&B report. See `full_validation/comparison.md` for the
local result. EDF-S evaluation of these final heads remains separate.

## Overlap and matching

For each sky position, select the unmasked tile in which it lies farthest from
the four-pixel tile edge. Resolve ties in sorted tile order. This rule is fixed
without model scores or outcomes. Each MER object ID appears once. Keep a
prediction only if its originating tile owns its predicted sky position.
Apply this same sky partition to all models. Close physical sources are not
merged by an arbitrary global deduplication radius. Centroid jitter across an
ownership boundary can still affect seam detections.

Completeness uses clean references (`spurious_flag != 1`), split by `vis_det`.
The bright VIS subset also requires finite `mag_vis < 24.5`. NIR/full have no
magnitude cut. Use spherical nearest matching strictly below 0.5 arcsec.
Matching is not one-to-one. MER match fraction uses the full catalogue,
including flagged entries, and is **catalogue agreement, not absolute purity**.
Four-pixel edge and artifact exclusions apply to the owned sky footprint.
The final metrics therefore differ from the tile-pooled monitoring protocol.

## Paired uncertainty

Divide the full footprint's tangent-plane bounding rectangle into 4×4 contiguous
sky regions (about 2.16×2.80 arcmin each). Resample 16 whole regions with
replacement 5,000 times, using seed 20260915. Use exactly the same region weights
for all three models, all metrics and all thresholds. Recompute each fraction
as the ratio of summed numerators to summed denominators, then compute paired
differences within each resample. Report the 2.5th and 97.5th percentiles.

Repeat with a 3×3 grid (about 2.88×3.73 arcmin) as a block-size sensitivity check.
Save source-level both/only-first/only-second/neither recovery counts too.
Reference groups, regions and the threshold grid are fixed before final results.
Threshold 0.30 is primary; the eleven thresholds from 0.15 through 0.90 are a
secondary tradeoff diagnostic, not a search for the most flattering result.

These are **approximate, pointwise, within-patch intervals**. Nine or sixteen
regions cannot establish generalization to other fields, and block size can
matter. The intervals condition on the trained checkpoints and MER catalogue;
they do not include variation over training seeds, unknown truth, or simultaneous
selection across metrics/thresholds. A separate-field check remains necessary.
Resampling regions preserves local dependence better than treating every object
as an independent trial, but does not guarantee independence between regions.
See [Lahiri and Zhu (2006)](https://arxiv.org/abs/math/0611261) for why spatial
resampling requires care with dependence and sampling design.

## Execution

From the project root, prepare the geometry on CPU:

```bash
python -m models.detection2.evaluation.run --prepare-only
```

The full CPU check also reproduces the saved 12-tile baseline counts at all four
monitoring thresholds, then feeds identical archived predictions into all three
model slots and verifies zero paired differences and intervals:

```bash
python -m unittest models.detection2.tests.test_full_validation -v
python -m models.detection2.evaluation.run --check-saved-baseline experiments/visnir_head_20260914/evaluation_ECDFS_patch25/ECDFS_patch25
```

Execution-check outputs have `_execution_check` suffixes. They are not final
model results and are never published by that command.

To run after the existing paired training jobs finish:

```bash
bash models/detection2/evaluation/run_after_training.sh
```

The worker waits for successful launcher exit, both final checkpoints and both
final validation records before using GPU 0. It runs all three heads on the
same cached features, with no optimizer updates. A file lock prevents duplicate
workers. There is a six-hour waiting limit. Resume reruns this same command;
completed per-tile inference is reused only with unchanged checkpoints, code,
prepared geometry and cached features. No checkpoint is selected by validation.

The completed worker ran in screen session `detection2`, window
`detection2-eval`; its logs and status are in `full_validation/`.
Do not launch it again while that window is active. The notebook links to these
commands and provides inspection cells. Starting the worker does not alter
the training launcher or the two training processes.

## Outputs and W&B

All outputs are under `runs/unknown_regions_20260915_v1_12ep/full_validation/`:

- `protocol.json`, `geometry.json`, `geometry/`, `references.npz`: exact footprint and references.
- `inference_manifest.json`, `predictions/`: checkpoint/code/input hashes and raw per-tile predictions.
- `*_catalog.npz`, `source_recovery.npz`: partitioned prediction catalogues and paired per-source outcomes.
- `bootstrap_4x4.npz`, `bootstrap_3x3.npz`: block counts and shared resampling weights.
- `results.json`, `comparison.md`: estimates, intervals, paired differences and discordant source counts.
- `quality_intervals.png`, `paired_differences.png`, `tradeoffs.png`: standalone scientific plots.
- `console.log`, `status.json`, `execution_check.json`: progress and verification.
- `wandb_run.json`, `report.json`, `publication_verification.json`: online output and readback checks.

The worker publishes only derived numeric metrics/tables to **AI-Astro /
JAISP-Detection-Q1**, in the existing experiment group. It appends full-patch
results and paired intervals to the existing comparison report. Source IDs,
coordinates, images, checkpoints and code remain local. W&B credentials are
handled by the existing SDK account connection.

If only report publication needs to be retried, use the saved results without
rerunning inference:

```bash
python -m models.detection2.evaluation.publish
```

The initial report update failed because loaded W&B reports contain datetime
values. Publication now uses Pydantic JSON serialization for those values.
The retry records its code hash and verifies that `results.json` is unchanged.

EDF-S evaluation remains a separate experiment; this worker does not use it for
training, checkpoint selection or threshold tuning.
