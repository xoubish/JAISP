# Proposed learned detection and reconstruction experiment

User direction: use the information in the ten-band dataset and foundation
representation; learn source appearance instead of importing precomputed PSFs.

```
Ten-band images → foundation features → object proposals
                                      ↓
                    positions + object gates + compact appearance vectors
                                      ↓
                    shared neural decoder + learned band embeddings
                                      ↓
                    objects placed and summed on native image grids
                                      ↓
                    image reconstruction and residuals
```

Labels initialize and anchor detection; they do not define all possible objects.
Unlabelled plausible positions can remain unknown. Eventually, useful additions
should improve reconstruction/consistency enough to justify another object.

The first executable training experiment changes only the unknown-background
loss. It establishes the paired control for treating labels as incomplete.
The learned decoder prototype is implemented separately and has not been trained.

For the next experiment, the concrete development steps are:

1. Extract compact object vectors from foundation features at candidate positions.
   Start with seed positions plus sensitive teacher proposals, keeping separate
   seed, unlabelled and background statuses.
2. Train the neural renderer on small ten-band crops with fixed object positions
   first. Use per-band normalization fitted on training data; inspect whether
   templates learn compact and extended appearances and represent NIR-only sources.
   This stage tests learning image appearance without trying to optimize object
   count simultaneously.
3. Jointly optimize positions and detection with seeded supervision, masked image
   reconstruction, and consistency under valid spatial/band perturbations. To
   claim held-out-pixel prediction, mask before feature encoding: features cached
   from the full image would leak the target pixels. Recomputing those features
   is necessary for that experiment. Full cached features remain valid for the
   present supervised/unknown-loss ablation and renderer warm-up.
4. Introduce discrete object presence and candidate addition/removal. Penalizing
   soft gates while allowing arbitrary flux is degenerate (small gate × large
   flux can leave the image unchanged). Test hard stochastic gates or explicit
   proposal comparisons, with a specified training estimator and complexity cost.
5. Evaluate both detection and reconstruction: source recovery by VIS/NIR group,
   MER agreement, duplicate/split rates, residual structure, and paired injections
   across source types. Residuals and model confidence alone are not independent
   truth. Keep final comparisons distinct from development monitoring.

Spatial equivariance and band perturbation consistency constrain the model but
do not create new observations. A source visible only in NIR must not be forced
to retain the same confidence when its supporting bands are removed. Use known
band-availability masks and only apply consistency where its meaning is valid.

The small renderer enforces an object bottleneck, not a guarantee of identifiable
objects. One template could absorb neighbours, multiple templates could split an
extended source, and a decoder could fit noise. These behaviours are measurable
design questions for the next experiment, not reasons to import a PSF by default.

Monitoring and artifacts follow the same visible notebook/configuration/W&B
conventions as the first paired test. No automatic pseudo-label promotion or
long training starts as a side effect of opening the notebook.
