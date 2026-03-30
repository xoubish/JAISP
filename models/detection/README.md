# JAISP Detection Head (v1)

`JaispDetector` is a DETR-style source detector built on JAISP v6 encoder features.

This module is intended to turn the shared Rubin+Euclid foundation representation into per-tile object sets (centroid + class + confidence), while staying compatible with set prediction and Hungarian matching.

## Architecture

`models/detection/detector.py`

Pipeline:

1. `JAISPEncoderV6` bottleneck features: `[B, 512, H/8, W/8]`
2. `1x1` projection + GroupNorm to `d_model` (default `256`)
3. 2D sinusoidal positional encoding on flattened memory tokens
4. Transformer decoder with learned object queries (`num_queries`, default `300`)
5. Query heads:
   - centroid head: normalized `(x, y)` in `[0, 1]`
   - class head: logits over `['star', 'galaxy', 'artifact']`
   - confidence head: objectness logit
   - flux head: `log_flux` proxy

The encoder is frozen by default (`JAISPEncoderWrapper(freeze=True)`), with optional end-to-end fine-tuning via `--finetune_encoder`.

## Matching And Loss

`models/detection/matcher.py`

Training uses Hungarian assignment from predictions to ground-truth source sets.

Matching cost:

`cost = cost_pos * L1(xy) + cost_cls * CE(class) + cost_conf * (-sigmoid(conf))`

Default matching weights:

- `cost_pos = 5.0`
- `cost_cls = 1.0`
- `cost_conf = 1.0`

Final loss combines:

- matched centroid L1 (`lambda_pos = 5.0`)
- matched class CE (`lambda_cls = 1.0`)
- matched confidence BCE to 1 (`lambda_conf = 2.0`)
- unmatched confidence BCE to 0 (`lambda_noobj = 0.1`)

## Training Data Path

`models/detection/dataset.py`

`TileDetectionDataset` wraps `JAISPDatasetV6` and generates pseudo labels from Rubin imagery.

Pseudo-label flow:

1. Build Rubin detection image from `(g, r, i)`
2. Detect peaks (`detect_sources`)
3. Compute concentration index in Rubin r-band
4. Assign class:
   - `0` = star (`concentration >= 0.5`)
   - `1` = galaxy (`concentration < 0.5`)

Notes:

- Labels are pseudo labels, not a survey catalog.
- `artifact` is currently predicted but not separately pseudo-labeled.
- Euclid-band detection conditioning is scaffolded in the module, but pseudo-label generation itself is Rubin-driven.

## Quick Start

From repo root:

```bash
# Recommended: train on top of a Phase B encoder checkpoint
python models/detection/train_detection.py \
  --rubin_dir data/rubin_tiles_ecdfs \
  --euclid_dir data/euclid_tiles_ecdfs \
  --encoder_ckpt models/checkpoints/jaisp_v6_phaseB/checkpoint_best.pt \
  --out models/checkpoints/detector_v1.pt \
  --epochs 50

# Quick smoke test: no encoder checkpoint (uses stub CNN encoder)
python models/detection/train_detection.py \
  --rubin_dir data/rubin_tiles_ecdfs \
  --out models/checkpoints/detector_stub.pt \
  --epochs 5
```

## Inference Example

```python
import torch
from models.jaisp_foundation_v6 import JAISPFoundationV6
from models.detection.detector import JaispDetector, JAISPEncoderWrapper

ckpt = torch.load('models/checkpoints/jaisp_v6_phaseB/checkpoint_best.pt', map_location='cpu')
foundation = JAISPFoundationV6()
foundation.load_state_dict(ckpt['model'])

encoder = JAISPEncoderWrapper(foundation.encoder, freeze=True)
detector = JaispDetector(encoder=encoder)

# images/rms are dict[str, Tensor] in JAISP format
# out = detector(images, rms)
# pred = detector.predict(images, rms, conf_threshold=0.5, tile_hw=(H, W))
```

## File Map

- `detector.py`: model, wrapper, inference helpers, save/load
- `matcher.py`: Hungarian matcher + DETR-style loss
- `dataset.py`: pseudo-label dataset + collate
- `train_detection.py`: train/val loop, W&B logging
