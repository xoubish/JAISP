# Archived detection experiments

**Not used in the current paper. This experimental branch is closed.**

Start with the [retained results and configurations](records/README.md).
The paper's production detector remains in [`../detection/`](../detection/),
using `checkpoints/q1_detection_v11/centernet_vis_sep.pt` from the project root.

This folder was renamed from `models/detection2/` on 2026-09-15. The old path
is a compatibility symlink so existing imports, notebooks and saved checkpoint
paths continue to work. All original code, checkpoints and run outputs are
preserved locally. Generated files under `runs/` are excluded from Git;
`records/` contains exact copies of the retained summaries, results and
configurations, with their source paths and checksums.

The remaining sections and launch commands document the completed experiments.
For their original launch and inspection workflow, see
[`01_unknown_regions.ipynb`](notebooks/01_unknown_regions.ipynb).

**Paper-finalization decision:** the final frozen-decoder rescoring test is
complete and worsens the detection tradeoff. This experimental branch is
closed; no further training or score search is planned. See
[`rescore/README.md`](rescore/README.md), the
[saved comparison](records/rescoring/comparison.md), and
[`STATUS.md`](STATUS.md). The paper's existing validated production-detector
results have not been replaced by the development runs below.

The completed comparison selected **CenterNet with standard loss** as the
experimental baseline. See [`selected_baseline.json`](selected_baseline.json)
for the exact checkpoint and verified hash, and [`STATUS.md`](STATUS.md) for results.

**Completed stage: learned object reconstruction.** The runnable fixed-catalogue
warm-up is in [`decoder/README.md`](decoder/README.md), with the inspection
notebook [`02_learned_decoder.ipynb`](notebooks/02_learned_decoder.ipynb).
The 20-epoch run completed across both GPUs. It learned source appearance at
fixed positions; the subsequent bounded scoring test did not improve detection.
Object addition/removal remains outside the scope of this paper-finalization pass.

## First question

Does treating plausible unlabelled objects as **unknown**, rather than training
them as background, improve the completeness–catalogue-agreement tradeoff?

The first comparison isolates this loss change. Both arms start from the
September 14 VIS+NISP head and retain exactly the same VIS+NISP seed labels.
The foundation encoder stays frozen, with all ten bands represented in the
existing v11 features. Both arms use the same data order, augmentations, seed,
twelve epochs, optimizer, learning rate and final-epoch checkpoint selection.

| Arm | Seed positions | Other image locations |
|---|---|---|
| `control` | Existing positive loss | Existing background loss |
| `unknown` | Same positive loss | Ignore small regions around unlabelled teacher proposals; retain background loss elsewhere |

The fixed teacher is the initial VIS+NISP checkpoint. Its proposals use score
≥0.15 (implementation uses `>0.15`), NMS 7, and the existing Gaia/saturation
mask. A proposal farther than 5 VIS pixels from every seed gets an ignore disk
of radius 5 VIS pixels (0.5 arcsec). Positive centres remain supervised even
where a disk overlaps. Coordinates and masks follow the actual cached
augmentation metadata. The proposal set is frozen before either arm trains.

These candidates are not automatically promoted to positive labels. Existing
NISP seed errors also remain in both arms. This test may improve completeness
without resolving contamination; the paired control makes that result useful.

## Learned direction and local inspection

The user prefers learning the image formation and source appearance from the
existing ten-band data. **No precomputed PSF or external source-profile model
is used by the active experiment.** Existing image registration metadata is
used to locate the same candidate in each band's native pixels.

`audit.py` makes ten-band cutouts for unlabelled high-score and low-score
proposals and seed-matched controls on training tiles. A magenta cross marks
the candidate; cyan circles show existing seeds. These are inspection examples,
not real/fake classifications or calibrated confidence estimates. Images and
candidate records stay local. "Unlabelled" refers to the seed labels, not MER
matching or source truth.

`object_decoder.py` contains an **untrained, tested architectural prototype**:
compact per-object appearance vectors and band embeddings feed a shared neural
decoder, which learns positive, normalized band-dependent templates and fluxes.
Those templates are placed at the predicted subpixel positions and added into
each native-band image. There is no dense feature-to-image skip connection; with
zero object gates, only a constant background remains. No supplied PSF enters.

The first paired ablation does not train this decoder. The new `decoder/`
warm-up connects the selected head's frozen proposals to a trainable renderer.
Learning object count, masked reconstruction/consistency losses, and validating
that a learned component corresponds to one source remain later stages.
Soft gates alone have a gate/flux scaling
degeneracy, so a source-count penalty must not be presented as solved here.
The fixed-support templates and compact appearance vectors are constraints to
test, not guarantees of correct separation. See `DESIGN.md` for the proposal.
Rotated views share noise; reconstruction on the same input is not independent
verification of a source. No independent exposure split is assumed.

## Data split and metrics

- Training: 682 tiles, four cached augmentations each; all patch-25 tiles excluded.
- Monitoring: 12 fixed, evenly spaced patch-25 tiles at epoch 0 and after every epoch.
  Save their IDs and all raw recovered/reference counts with each run.
- Report VIS bright, VIS all, NIR-only and full clean MER completeness, plus
  full-MER match fraction, at thresholds 0.20, 0.30, 0.40 and 0.50. Primary: 0.30.
- Same nearest-match radius as the previous experiment: 0.5 arcsec. Counts are
  tile-pooled and can repeat sky objects in overlapping tiles. Nearest matching
  is not one-to-one. MER agreement does not establish true object purity.
- Patch 25 has already been examined during previous experiments: these are
  development metrics. EDF-S is reserved for the final comparison in this new
  iteration, although it was also measured previously; it is not a new blind test.
- Full final evaluation reuses `detection.visnir_eval_experiment` with explicit
  checkpoint paths. HLF/deeper references and paired injections are subsequent
  diagnostics; a deeper catalogue still is not complete ground truth.

## Reproducible execution

To run both GPUs in a disconnect-safe screen session, from the project root:

```bash
screen -S detection2
bash models/detection2/run_pair.sh
```

The launcher prepares all training tiles once, then runs the control on GPU 0
and the unknown arm on GPU 1. Both train for twelve epochs with live W&B monitoring.
Detach with **Ctrl+A, then D**; reconnect with `screen -r detection2`.
Logs are saved under `runs/unknown_regions_20260915_v1_12ep/launch_logs/`; final exit
codes are saved in `exit_status.json`. Pass a fresh output directory as the
script's first argument for another experiment. Supply an existing prepared-label
directory as the second argument to reuse it. The launcher passes the current
configuration's epoch budget explicitly to both trainers, extending their cosine
schedule through that final epoch. The prepared label file retains its original
configuration as provenance; run configs record the override. This launcher runs the first
paired detection-loss comparison, not the untrained object-decoder prototype.

Run cells individually in the notebook; commands run in the foreground with
streamed output and a saved `console.log`. There is no background scheduler.
The notebook has separate local audit, preparation, control training, unknown
training, comparison, and final-evaluation cells.

Configuration: [`configs/unknown_regions_v1.json`](configs/unknown_regions_v1.json).
Outputs: `runs/<experiment>/<prepared|audit|control|unknown>/`.
Use a new output directory for a new execution; scripts refuse to overwrite runs.
The shared feature and label caches are only read. Previous models remain intact.

Training requires **online W&B** at
[AI-Astro / JAISP-Detection-Q1](https://wandb.ai/AI-Astro/JAISP-Detection-Q1),
group `detection2-unknown-regions-v1`. Training logs losses, source counts,
ignored area, learning rate, gradient norm and per-epoch validation metrics.
Each run prints its URL and saves it in `wandb_run.json`. Failed initialization
stops training; there is no silent offline fallback. Candidate images, audit
records, checkpoints and code stay local; no image or model artifact uploads
are requested by these scripts. Preparation and auditing make no W&B calls.

For the focused comparison, open the
[live progress report](https://wandb.ai/AI-Astro/JAISP-Detection-Q1/reports/Detection2-detection-progress--VmlldzoxNzkzOTM5NQ==).
It overlays the starting model on four percentage validation plots and compares
completeness against MER match fraction across thresholds. The small CPU
monitor runs in screen window `detection2-monitor`; its code, definitions and
reproduction commands are in [`monitoring/README.md`](monitoring/README.md).

The approved full-patch comparison is in [`evaluation/README.md`](evaluation/README.md).
It evaluates the starting model and both final heads on all 108 patch-25 tiles,
counts each reference object once, and computes paired uncertainty by resampling
sky regions. Its worker waits for both training jobs to finish successfully;
outputs go to `runs/unknown_regions_20260915_v1_12ep/full_validation/`.

Each training run saves its configuration, source/input SHA-256 hashes, label
hash, local scalar history, validation counts, per-epoch head checkpoints,
`final.pt`, and the latest optimizer/scheduler/RNG state. Automatic resume is
not implemented; the state is preserved for an explicit recovery if needed.

Tests, from the project root:

```bash
python -m unittest discover -s models/detection2/tests -v
```

`train --check-only` performs one real-data backward pass and one validation
tile with **zero optimizer updates**, no checkpoint and no W&B upload. Partial
four-tile proposal preparation is accepted for this execution check only; full
training refuses partial preparation.

## Setup status

The initial setup includes a four-training-tile local pilot under
`runs/pilot_dl_20260915/`. See `STATUS.md` for completed checks and diagnostic results.
The initial four-epoch jobs were superseded after the user approved twelve
epochs. The active pair uses `runs/unknown_regions_20260915_v1_12ep/` and reuses
the original run's complete prepared labels. See `STATUS.md` for the change.
