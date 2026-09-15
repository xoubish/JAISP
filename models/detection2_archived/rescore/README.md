# Final bounded detection-rescoring test

**Complete; rule rejected.** At matched 87.23% MER agreement, full-MER
completeness decreased from 91.70% to 85.34% and NIR completeness from 83.92%
to 74.10%. The secondary 94% comparison also worsened. This branch is closed,
with no further score tuning or training. See
[`comparison.md`](../runs/decoder_rescore_20260915_v1/comparison.md).

This is one test of whether the already-trained object renderer can improve
**detection completeness at matched catalogue agreement**. It performs no new
training, changes no source positions, and does not search score weights or
architectures. If it fails the fixed decision rule below, close this branch and
return to the paper with the selected detector.

## Fixed scoring rule

Use the selected standard-loss head's candidates with score `p > 0.15` in the
existing, geometrically partitioned 108-tile patch-25 footprint: 13,650 unique
owned detections. Neighbours from the same frozen proposal pool provide context.
The final epoch-20 renderer and its training-only image normalization are frozen.

For every candidate, construct a centered 12.8-arcsec crop, including every
neighbour whose template can intersect it. Let `m_b` be the full reconstructed
image and `c_ib` the focal candidate's additive image contribution in band `b`.
The contribution is decoded with the same object features as in the full scene.
Remove it while keeping the **same full-scene background and neighbours**:

```
E_ib = sum_valid[Huber(m_b - c_ib - y_b) - Huber(m_b - y_b)]
       / max(sqrt(sum_valid(c_ib**2)), 1e-8)

score_i = log(p_i / (1-p_i)) + asinh(mean_b(E_ib))
```

Huber delta is 3 in the stored training-normalized image units. The template L2
norm prevents the loss difference from scaling quadratically with predicted
flux. The signed `asinh` compresses large evidence values. All ten bands have
equal weight. The combination coefficient is fixed at one before results; it is
not fitted on MER. This is a ranking heuristic, **not** a calibrated probability
or independent-pixel likelihood. Cached features see the images being scored.

Crops with fewer than 80% valid pixels in any band, or over 128 context objects,
receive zero image evidence and retain their original detector contribution.
All such cases are reported. No candidates are silently removed. The two GPU
workers divide the 108 tiles; inference has a 15-minute wall-time limit.

## Detection comparison and stopping rule

Use the exact saved candidate positions, sky ownership, reference catalogue and
nearest 0.5-arcsec matching from `../evaluation/`. Completeness counts each clean
MER reference ID once; catalogue agreement includes counterparts in full MER.
Matching is not one-to-one, and catalogue agreement is not absolute source purity.

For each ranking, compare the largest whole-score prefix whose agreement reaches
the common target. Primary agreement is the selected detector's agreement at
`p >= 0.30` (87.2257%); secondary agreement is 94%. Both models use the same
operating-point selection rule, with at least 100 detections. These are
development precision/recall-curve comparisons, not independently validated
deployment thresholds.

Use 2,000 paired spatial bootstrap replicates, 4×4 primary regions and 3×3
sensitivity. Both thresholds are reselected within every replicate at the
common target agreement. Intervals are pointwise within-patch estimates and do
not include other fields or training seeds.

The predeclared gate requires all of:

- At least +1 percentage point in primary NIR completeness, with the 4×4 paired
  95% interval entirely above zero.
- No decrease in primary full-MER completeness.
- No decrease in NIR or full-MER completeness at the secondary 94% agreement.

Otherwise the result is `close_branch_keep_selected_detector`. There is no
automatic follow-up training or score search. The complete rule, input hashes
and code hashes are written to `protocol.json` before inference. The original
detector's counts at 0.30 must reproduce the archived comparison exactly.

## Execution and outputs

The authorized run uses the existing screen session `detection2`, window
`detection2-rescore`, with the visible foreground launcher:

```bash
bash models/detection2/rescore/run.sh
```

It creates `runs/decoder_rescore_20260915_v1/` and refuses to overwrite it.
Opening the inspection notebook launches no work. After this test, read
[`../notebooks/03_detection_rescoring.ipynb`](../notebooks/03_detection_rescoring.ipynb).

Saved outputs include per-tile evidence and fallback flags, complete candidate
scores, actual selected catalogues at both agreement targets, exact curves,
paired intervals, `comparison.md`, and `detection_tradeoff.png`. The console log
is beside the run directory; worker logs and progress records are inside it.
Numeric progress, comparisons and curve tables go to the existing W&B project.
Images, sky coordinates and checkpoints stay local. W&B and report URLs are
saved in `wandb_run.json` and `report.json`.

The paper is not edited by any script. A successful development result would
still need to be assessed against the paper's existing injection and
cross-field evidence before replacing its production detector.
