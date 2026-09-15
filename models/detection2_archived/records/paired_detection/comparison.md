# Detection2: all patch-25 tiles, unique-source evaluation

Scored 108/108 tiles. Unique reference counts: {'vis_bright': 2670, 'vis_all': 6878, 'nir_only': 2513, 'full_mer': 9391}.

Fixed final checkpoints; threshold 0.30 is the primary comparison. Values in brackets are approximate 95% paired spatial-bootstrap percentile intervals. Percentages for model estimates; percentage points for differences. These are pointwise intervals, not simultaneous guarantees across thresholds or metrics.

Primary spatial grid: 4×4, with 5,000 shared region resamples. A 3×3 grid provides a block-size sensitivity check. These intervals describe variation within this patch, conditional on these models and MER; they do not include training-seed variation, other fields, or catalogue truth errors.

| Metric | Starting model | Standard loss | Masked background loss |
|---|---:|---:|---:|
| VIS completeness (mag < 24.5) | 95.06% [93.13, 96.58] | 95.13% [93.19, 96.63] | 96.40% [94.93, 97.59] |
| NIR-only completeness | 80.98% [78.15, 83.60] | 83.88% [81.43, 86.14] | 90.37% [88.51, 92.08] |
| Full-MER completeness | 90.87% [89.56, 91.99] | 91.68% [90.29, 92.83] | 95.25% [94.25, 96.04] |
| MER match fraction | 87.76% [86.87, 88.54] | 87.23% [86.29, 88.05] | 76.24% [75.16, 77.26] |

## Paired differences at threshold 0.30

Positive means the first model has a larger metric. Completeness and MER agreement must be assessed together. MER agreement is not absolute source purity.

### 4×4 regions

| Metric | Standard − starting | Masked − starting | Masked − standard |
|---|---:|---:|---:|
| VIS completeness (mag < 24.5) | +0.07 [-0.26, 0.47] | +1.35 [0.79, 2.00] | +1.27 [0.76, 1.90] |
| NIR-only completeness | +2.90 [2.07, 3.76] | +9.39 [7.76, 11.12] | +6.49 [5.30, 7.75] |
| Full-MER completeness | +0.81 [0.59, 1.01] | +4.38 [3.78, 4.99] | +3.57 [2.96, 4.20] |
| MER match fraction | -0.54 [-0.87, -0.21] | -11.52 [-12.11, -10.80] | -10.98 [-11.50, -10.43] |

### 3×3 regions

| Metric | Standard − starting | Masked − starting | Masked − standard |
|---|---:|---:|---:|
| VIS completeness (mag < 24.5) | +0.07 [-0.26, 0.36] | +1.35 [0.85, 1.91] | +1.27 [0.67, 1.94] |
| NIR-only completeness | +2.90 [2.30, 3.57] | +9.39 [7.63, 11.18] | +6.49 [4.96, 7.95] |
| Full-MER completeness | +0.81 [0.59, 1.02] | +4.38 [3.76, 4.96] | +3.57 [2.89, 4.14] |
| MER match fraction | -0.54 [-0.85, -0.24] | -11.52 [-12.09, -10.99] | -10.98 [-11.55, -10.54] |

The geometry and ownership rule are fixed before final predictions. Each MER ID is counted once; predictions are retained only in their originating tile’s assigned sky area. No spatial merging radius is used to collapse nearby physical sources. Matching uses nearest neighbours within 0.5 arcsec and is not one-to-one. Small centroid changes across tile-ownership boundaries can still affect seam detections. These are partitioned-sky metrics, so they need not equal the tile-pooled training curves.

Raw predictions, unique catalogues, per-source recovery flags, block counts, bootstrap weights, input/code hashes, and the protocol remain local beside this file.
