# Retained experiment records

These development experiments are complete and archived. Their models and
results are not used in the current paper. The paper continues to use
`checkpoints/q1_detection_v11/centernet_vis_sep.pt` with `models/detection/`.

| Experiment | Outcome | Saved records |
|---|---|---|
| Standard versus masked background loss | Masking increased completeness but reduced MER agreement. The standard-loss head was selected as the experimental baseline. | [Comparison](paired_detection/comparison.md), [results](paired_detection/results.json), [protocol](paired_detection/protocol.json) |
| Learned renderer warm-up | Completed 20 epochs and improved reconstruction. Fixed detector positions and counts prevented any detection change. | [Configuration](decoder_warmup/config.json), [initial validation](decoder_warmup/initial_validation.json), [final validation](decoder_warmup/final_validation.json) |
| Frozen-renderer rescoring | Reduced completeness at matched MER agreement. Rejected; branch closed. | [Comparison](rescoring/comparison.md), [results](rescoring/results.json), [closure](rescoring/closure.json) |

The experiment configurations, source hashes and W&B run/report links are saved
in each subdirectory. The selected experimental checkpoint and its checksum
remain in [`../selected_baseline.json`](../selected_baseline.json).

Every copied record is byte-for-byte identical to its original run file.
[`archive_manifest.json`](archive_manifest.json) lists its original location,
size and SHA-256 checksum. Statements inside these records describe the time
they were written; later manuscript wording changes did not replace the
paper's detector results.

All original outputs remain locally under `../runs/`, including checkpoints,
prepared crops, predictions, validation histories, logs, plots and superseded
pilots. That directory is excluded from Git. The retained records and the
archive's source code, configurations and inspection notebooks are versioned;
reproducing the experiments also requires the local data and checkpoints.

The main local run directories are:

- Paired detection: `../runs/unknown_regions_20260915_v1_12ep/`.
- Renderer warm-up: `../runs/decoder_warmup_20260915_v1/`.
- Rescoring: `../runs/decoder_rescore_20260915_v1/`.

`models/detection2` remains a compatibility symlink to this archive, preserving
the paths recorded by the experiments. No local experiment files were deleted
during this cleanup.
