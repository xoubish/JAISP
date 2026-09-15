# Proposed learned detection and reconstruction experiment

**Paper-finalization status:** the renderer warm-up and one fixed rescoring
test are complete. Rescoring worsened completeness at matched MER agreement.
The user requested closing this branch after that test; the later stages below
remain research ideas, not scheduled work or requirements for this paper.

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

The completed first experiment compared standard and masked background loss.
The selected standard-loss head supplies fixed proposals to the now executable
renderer warm-up in [`decoder/README.md`](decoder/README.md). The warm-up has
passed execution checks and awaits the user's full training launch.

For the next experiment, the concrete development steps are:

1. Extract compact object vectors from foundation features at candidate positions.
   The warm-up uses the selected head's proposals above 0.15 in both training
   and validation. This keeps proposal selection consistent without requiring
   seed labels on validation tiles. No MER labels enter the warm-up. Seed,
   unlabelled and background statuses become relevant again when detection
   supervision and object-presence learning are introduced.
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
