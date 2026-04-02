# JAISP Detection Head

DETR-style source detector built on the frozen V7 foundation encoder.

## Architecture

```
Input: multi-band tile dicts {band: [B, 1, H, W]} images + rms
    |
Frozen JAISPFoundationV7.encode()
    -> bottleneck [B, 256, ~130, ~130]
    |
1x1 Conv + GroupNorm -> [B, 256, ~130, ~130]
    |
Flatten + 2D sinusoidal positional encoding
    -> memory tokens [~17000, B, 256]
    |
Transformer decoder (6 layers, 8 heads, FFN=1024)
    with 500 learned object queries
    |
    +-- Centroid head:    3-layer MLP -> sigmoid -> (x, y) in [0, 1]
    +-- Confidence head:  Linear -> objectness logit
    +-- Log-flux head:    2-layer MLP -> log10(flux) proxy
    +-- Class head:       Linear -> 1 class ('source')
```

The encoder is frozen by default. Trainable parameters: ~6.7M.

## Training

### Hungarian Matching

Each training step matches 500 predicted queries to ground-truth pseudo-labels
via Hungarian assignment. The cost matrix uses:

- L1 position distance (weight 5.0)
- Negative sigmoid confidence (weight 1.0)

### Loss

| Term | Weight | Description |
|------|--------|-------------|
| `loss_pos` | 5.0 | L1 on matched centroid positions |
| `loss_conf_obj` | 2.0 | BCE(matched confidence, 1) |
| `loss_conf_noobj` | 0.5 | BCE(unmatched confidence, 0) |

Classification loss is not used -- with a single 'source' class, the
confidence head handles object-vs-background.

### Pseudo-Labels

Ground-truth labels are generated from classical peak-finding (not a curated catalog):

1. Build detection image from Rubin g+r+i coadd
2. Detect peaks above 3-sigma with Gaussian smoothing
3. Subpixel centroiding via flux-weighted center of mass
4. Cap at `num_queries` sources per tile (default 500)

The pseudo-labels are computed on the **augmented** image (after random 90-degree
rotations and flips) so GT coordinates match the image the encoder sees.

### Data

- 130 training tiles, 14 validation tiles (90/10 split, seed=42)
- Augmentation: random 90-degree rotations + horizontal/vertical flips
- Euclid bands are loaded when `--euclid_dir` is provided, enriching encoder features
- Batch size: 4, AdamW optimizer, cosine LR schedule

## Quick Start

```bash
# From repo root
python models/detection/train_detection.py \
    --rubin_dir    data/rubin_tiles_ecdfs \
    --euclid_dir   data/euclid_tiles_ecdfs \
    --encoder_ckpt checkpoints/jaisp_v7_baseline/checkpoint_best.pt \
    --out          models/checkpoints/detector_v7.pt \
    --num_queries 500 \
    --epochs 100 \
    --wandb_project jaisp-detection
```

## Inference

```python
import torch
from jaisp_foundation_v7 import JAISPFoundationV7
from detection.detector import JaispDetector, JAISPEncoderWrapper

# Load foundation model
ckpt = torch.load('checkpoints/jaisp_v7_baseline/checkpoint_best.pt', map_location='cpu')
cfg = ckpt.get('config', {})
foundation = JAISPFoundationV7(
    band_names=cfg.get('band_names'),
    hidden_ch=cfg.get('hidden_ch', 256),
)
foundation.load_state_dict(ckpt['model'], strict=False)

# Load detector
encoder = JAISPEncoderWrapper(foundation, freeze=True)
detector = JaispDetector.load('models/checkpoints/detector_v7.pt',
                               encoder=encoder, device='cuda')
detector.eval()

# Run prediction
# images, rms: dict[str, Tensor[1, 1, H, W]]
pred = detector.predict(images, rms, conf_threshold=0.5, tile_hw=(512, 512))
# pred['centroids']    [N, 2]    normalized (x, y)
# pred['scores']       [N]       confidence
# pred['positions_px'] [N, 2]    pixel coordinates
# pred['log_flux']     [N]       flux proxy
```

## Files

| File | Description |
|------|-------------|
| `detector.py` | `JaispDetector` model, encoder wrapper, save/load, inference |
| `matcher.py` | Hungarian matcher + position/confidence loss |
| `dataset.py` | Pseudo-label generation, tile dataset, DETR-compatible collation |
| `train_detection.py` | Training loop with W&B logging and visualization |
