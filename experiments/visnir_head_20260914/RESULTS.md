# VIS + NISP detection-head experiment

Four-epoch paired fine-tuning with the encoder frozen. Primary confidence threshold: 0.30.

The production checkpoint, manuscript and paper figures are unchanged.

## ECDFS_patch25

| Metric (%) | Production | VIS control | VIS + NISP |
|---|---:|---:|---:|
| Clean VIS <24.5 completeness | 93.93 | 93.39 | 95.62 |
| All clean VIS completeness | 90.28 | 89.21 | 94.85 |
| NIR-only completeness | 15.78 | 15.22 | 81.48 |
| Full clean MER completeness | 70.26 | 69.33 | 91.26 |
| Full-MER match purity | 93.72 | 94.34 | 86.28 |

NIR-only change relative to the matched VIS-only control: +66.25 percentage points.

## EDF-S

| Metric (%) | Production | VIS control | VIS + NISP |
|---|---:|---:|---:|
| Clean VIS <24.5 completeness | 94.47 | 93.96 | 95.68 |
| All clean VIS completeness | 89.23 | 88.20 | 94.49 |
| NIR-only completeness | 11.77 | 11.40 | 73.40 |
| Full clean MER completeness | 70.81 | 69.94 | 89.47 |
| Full-MER match purity | 91.96 | 92.87 | 85.54 |

NIR-only change relative to the matched VIS-only control: +62.01 percentage points.

## Counts and interpretation

Completeness uses the same masked, tile-pooled reference sample for all heads. The matching radius is 0.5 arcsec. NIR-only and full-MER completeness have no magnitude cut. Objects in overlapping tiles occur more than once. EDF-S purity uses the existing catalogue footprint restriction.

The control isolates the effect of the added labels from additional training. This is a short fine-tune, so it does not establish fully converged performance. Catalogue matching measures agreement with MER, not absolute source truth.

## Diagnostic threshold 0.40

This is an exploratory operating-point comparison, not a replacement for the fixed 0.30 primary result.

| Field | VIS <24.5 completeness | NIR-only completeness | Full clean MER completeness | MER match purity |
|---|---:|---:|---:|---:|
| ECDFS_patch25 | 90.79 | 69.68 | 84.55 | 92.67 |
| EDF-S | 91.59 | 57.95 | 82.28 | 92.17 |

## Follow-up label diagnostic

After the primary result, a stricter NISP label threshold was checked on four training tiles only. The added-label MER match fraction rises from 68.1% to 82.3% when the extraction threshold rises from 3 to 4. Combined-label coverage of clean NIR-only references falls from 77.2% to 51.8%. These are label diagnostics, not trained-head results or definitive false-positive rates. No head was trained with threshold-4 labels.

## Paired injection pilot

27 tiles distributed across patch 25; identical injections for all heads. The recovery radius is 0.3 arcsec. Magnitudes are donor VIS-equivalent values, including for NISP-only injections. This pilot does not replace the paper’s existing full injection analysis.

| Injection mode | Production d50 | VIS control d50 | VIS + NISP d50 |
|---|---:|---:|---:|
| all | 26.71 | 26.71 | 26.97 |
| vis | 26.59 | 26.51 | 26.48 |
| nisp | not bracketed | not bracketed | 26.63 |

The d50 values use linear interpolation only where the sampled curve crosses 50%; no extrapolation.

| Model | Induced artifacts / recovered sources, all modes (mag <30) | Faint-control recoveries (mag 35) |
|---|---:|---:|
| baseline | 13 / 4694 | 0 / 1215 |
| vis_control | 11 / 4641 | 0 / 1215 |
| visnir | 101 / 6860 | 0 / 1215 |
