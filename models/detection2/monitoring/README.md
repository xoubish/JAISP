# Detection2 monitoring

[Live comparison report](https://wandb.ai/AI-Astro/JAISP-Detection-Q1/reports/Detection2-detection-progress--VmlldzoxNzkzOTM5NQ==)

The four primary panels show VIS, NIR-only and full-MER completeness and MER
match fraction as percentages at score threshold 0.30. Both training runs are
compared with the unchanged starting model. Two further panels compare
completeness against MER match fraction across thresholds 0.20–0.50, using the
latest epoch completed by both runs. MER match fraction measures catalogue
agreement; unmatched detections are not necessarily false sources.

`live.py` reads completed validation JSON files every 30 seconds. It checks
that both arms share the starting model, validation tiles, reference counts and
threshold grid, and verifies each metric against the saved counts. It logs a
constant baseline and derived numeric tradeoff tables to the auxiliary W&B run
`wlymkj1t` (`Starting model (reference)`). This is not another trained model.
Training processes and GPU work are unaffected. Images and checkpoints are not
read or uploaded by this monitor.

The monitor is already running in screen session `detection2`, window
`detection2-monitor`. It stops after both final validations or launcher exit;
a six-hour limit guards against abandoned jobs. A file lock prevents duplicates.
Local outputs live in `runs/unknown_regions_20260915_v1_12ep/monitoring/`:
`console.log`, `monitor_state.json`, `snapshot.json`, `dashboard_preview.png`,
`report_spec.json`, `report.json`, and the verification record.

From the project root, generate a local snapshot without W&B:

```bash
python -m models.detection2.monitoring.live --preview-only
```

To restart the monitor after it has stopped (reuses its recorded W&B run ID):

```bash
bash models/detection2/monitoring/run_monitor.sh
```

`report.py` builds the report from the existing run IDs. The isolated report
helper is pinned in `requirements.txt`, installed under ignored `_vendor/`;
the training environment is unchanged. To reproduce or update the report:

```bash
python -m pip install --no-deps --target models/detection2/monitoring/_vendor -r models/detection2/monitoring/requirements.txt
python -m models.detection2.monitoring.report --publish
```

The report ID is reused when present. Without `--publish`, only its local
specification is written. Original W&B training histories remain untouched.
