# Single frozen-decoder rescoring test

Decision: **close_branch_keep_selected_detector**.

Scored 13,650 candidates; 657 used neutral image evidence because their crops were unsuitable.

No model weights or positions changed. The rescored outputs form a different selected detection catalogue at each threshold.

| Target agreement | Model | Actual agreement | Full-MER completeness | NIR completeness | Detections |
|---|---|---:|---:|---:|---:|
| 87.226% | baseline | 87.230% | 91.705% | 83.924% | 9984 |
| 87.226% | rescored | 87.226% | 85.337% | 74.095% | 9230 |
| 94.000% | baseline | 94.004% | 84.592% | 71.827% | 8473 |
| 94.000% | rescored | 94.010% | 78.735% | 62.356% | 7846 |

## Paired differences (rescored minus baseline)

| Target | Metric | Difference (pp) | 4x4 paired 95% interval | 3x3 sensitivity |
|---|---|---:|---|---|
| primary | vis_bright | -6.629 | [-8.072, -5.425] | [-7.687, -5.736] |
| primary | vis_all | -5.103 | [-6.013, -4.301] | [-6.073, -4.282] |
| primary | nir_only | -9.829 | [-12.026, -7.863] | [-12.058, -7.816] |
| primary | full_mer | -6.368 | [-7.311, -5.646] | [-7.289, -5.704] |
| secondary_94 | vis_bright | -4.532 | [-5.640, -3.065] | [-5.723, -3.159] |
| secondary_94 | vis_all | -4.536 | [-5.411, -3.560] | [-5.598, -3.608] |
| secondary_94 | nir_only | -9.471 | [-12.571, -6.953] | [-12.175, -7.315] |
| secondary_94 | full_mer | -5.857 | [-6.982, -4.757] | [-7.155, -4.863] |

Both rankings use the same candidates, geometry, unique MER references and nearest 0.5-arcsec matching. Matching is not one-to-one. MER agreement is not absolute physical purity.

Cutoffs are read from development curves at matched agreement and are reselected within each bootstrap replicate. Intervals do not include other fields or training seeds. The one scoring rule was fixed before this evaluation; it will not be adjusted after seeing this result.

The paper has not been edited and its existing detector results have not been replaced.
