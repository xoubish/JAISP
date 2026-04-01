# JAISP Detection Head

`JaispDetector` is a DETR-style source detector built on frozen `JAISPFoundationV7` encoder features.

## Architecture

`models/detection/detector.py`

Pipeline:

1. `JAISPFoundationV7.encode()` bottleneck: `[B, encoder_dim, h, w]`  
   `encoder_dim = hidden_ch` (default 256), spatial size from fused physical scale (~0.8"/px)
2. `1×1` projection + GroupNorm to `d_model` (default 256)
3. 2D sinusoidal positional encoding on flattened memory tokens
4. Transformer decoder with learned object queries (`num_queries`, default 300)
5. Query heads:
   - centroid: normalised `(x, y)` in `[0, 1]`
   - class logits: `['star', 'galaxy', 'artifact']`
   - confidence: objectness logit
   - flux: `log_flux` proxy

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

`TileDetectionDataset` loads Rubin + Euclid tiles and generates pseudo-labels from Rubin imagery.

Pseudo-label flow:

1. Build Rubin detection image from `(g, r, i)`
2. Detect peaks (`detect_sources`, default 5σ threshold)
3. Compute concentration index in Rubin r-band
4. Assign class:
   - `0` = star (`concentration >= 0.5`)
   - `1` = galaxy (`concentration < 0.5`)

Notes:

- Labels are pseudo-labels from classical peak-finding, not a survey catalog.
- `artifact` is predicted by the model but not separately pseudo-labeled.
- Euclid bands enrich the encoder features but pseudo-label generation is Rubin-driven.

## Quick Start

From repo root:

```bash
python models/detection/train_detection.py \
  --rubin_dir  data/rubin_tiles_ecdfs \
  --euclid_dir data/euclid_tiles_ecdfs \
  --encoder_ckpt models/checkpoints/jaisp_v7_baseline/checkpoint_best.pt \
  --out models/checkpoints/detector_v7.pt \
  --epochs 50 \
  --wandb_project jaisp-detection

# Smoke test — no encoder checkpoint (stub CNN)
python models/detection/train_detection.py \
  --rubin_dir data/rubin_tiles_ecdfs \
  --out models/checkpoints/detector_stub.pt \
  --epochs 5
```

## Inference Example

```python
import torch
from jaisp_foundation_v7 import JAISPFoundationV7
from detection.detector import JaispDetector, JAISPEncoderWrapper

ckpt = torch.load('checkpoints/jaisp_v7_baseline/checkpoint_best.pt', map_location='cpu')
cfg  = ckpt.get('config', {})
foundation = JAISPFoundationV7(
    band_names=cfg.get('band_names'),
    hidden_ch=cfg.get('hidden_ch', 256),
)
foundation.load_state_dict(ckpt['model'], strict=False)

encoder  = JAISPEncoderWrapper(foundation, freeze=True)
detector = JaispDetector.load('checkpoints/detector_v7.pt', encoder=encoder)

# images/rms are dict[str, Tensor[B,1,H,W]]
pred = detector.predict(images, rms, conf_threshold=0.5, tile_hw=(H, W))
# pred['centroids']    [N, 2]
# pred['classes']      [N]
# pred['scores']       [N]
# pred['positions_px'] [N, 2]  (if tile_hw given)
```

## File Map

- `detector.py`: model, encoder wrapper, inference helpers, save/load
- `matcher.py`: Hungarian matcher + DETR-style loss
- `dataset.py`: pseudo-label dataset + collate
- `train_detection.py`: train/val loop, W&B logging
