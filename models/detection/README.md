# JAISP Detection Head

CenterNet-style heatmap source detector built on the frozen V7 foundation encoder.

## Architecture

The detector predicts dense per-pixel outputs on the V7 encoder's bottleneck feature map:

```
Input: multi-band tile dicts {band: [B, 1, H, W]} images + rms (10 bands)
    |
Frozen JAISPFoundationV7.encode()
    -> bottleneck [B, 256, ~130, ~130]
    |
Refinement neck: 3x (Conv 3x3 + BN + ReLU)
    -> [B, 256, ~130, ~130]
    |
    +-- Heatmap head:   Conv 3x3 -> ReLU -> Conv 1x1 -> sigmoid -> [B, 1, H, W]
    +-- Offset head:    Conv 3x3 -> ReLU -> Conv 1x1 -> [B, 2, H, W]
    +-- Log-flux head:  Conv 3x3 -> ReLU -> Conv 1x1 -> [B, 1, H, W]
    +-- Profile head:   Conv 3x3 -> ReLU -> Conv 1x1 -> [B, 4, H, W]  (optional)
```

At inference, sources are detected by finding local maxima in the heatmap via
max-pooling NMS, thresholding on confidence, and reading off the offset, flux,
and optional profile values at each peak.

The optional profile head outputs (e1, e2, half-light radius, Sersic index) per
pixel for future integration with shape-fitting tools like Tractor.

The encoder is frozen by default. Trainable parameters: ~3.5M.

## Why CenterNet, Not DETR

An earlier version used a DETR-style transformer decoder with learned object
queries and Hungarian matching. This was abandoned because:

- **DETR needs large datasets** -- the original trained for 500 epochs on 118k images.
  With 130 tiles, query collapse (all predictions at one location) persisted for many epochs.
- **Overkill for point sources** -- DETR solves duplicate/NMS problems that don't exist
  for astronomical sources at the bottleneck resolution.
- **Slow convergence** -- DETR took 25+ epochs to reach val loss 1.02; CenterNet
  reaches lower loss within the first 5 epochs.

The DETR code is preserved in `detector.py`, `matcher.py`, and `train_detection.py`
for reference.

## Training

### Pseudo-Labels

Ground-truth labels are generated from classical peak-finding (not a curated catalog):

1. Build detection image from Rubin g+r+i coadd
2. Detect peaks above 3-sigma with Gaussian smoothing
3. Subpixel centroiding via flux-weighted center of mass

Pseudo-labels are cached at dataset initialization and coordinate-transformed
to match each tile's random augmentation (90-degree rotations + flips).

### Loss

Each GT source is rendered as a 2D Gaussian (sigma=2 bottleneck pixels) on
the heatmap target. The loss combines:

| Term | Type | Applied | Weight | Purpose |
|------|------|---------|--------|---------|
| `loss_hm` | Focal loss | All pixels | 1.0 | Source vs background |
| `loss_off` | L1 | GT positions only | 1.0 | Sub-pixel offset refinement |
| `loss_flux` | L1 | GT positions only | 0.1 | Flux estimation |

### Data

- 130 training tiles, 14 validation tiles (90/10 split, seed=42)
- Augmentation: random 90-degree rotations + horizontal/vertical flips
- All 10 bands loaded when `--euclid_dir` is provided
- Default batch size: 2 (Euclid VIS at 1050x1050 is memory-heavy)

## Quick Start

```bash
# From repo root
python models/detection/train_centernet.py \
    --rubin_dir    data/rubin_tiles_ecdfs \
    --euclid_dir   data/euclid_tiles_ecdfs \
    --encoder_ckpt checkpoints/jaisp_v7_baseline/checkpoint_best.pt \
    --out          models/checkpoints/centernet_v7.pt \
    --epochs 100 \
    --wandb_project jaisp-detection

# With profile prediction head (needs shape labels to train):
    --predict_profile
```

## Inference

```python
import torch
from jaisp_foundation_v7 import JAISPFoundationV7
from detection.centernet_detector import CenterNetDetector
from detection.detector import JAISPEncoderWrapper

# Load foundation model
ckpt = torch.load('checkpoints/jaisp_v7_baseline/checkpoint_best.pt', map_location='cpu')
cfg = ckpt.get('config', {})
foundation = JAISPFoundationV7(
    band_names=cfg.get('band_names'),
    hidden_ch=cfg.get('hidden_ch', 256),
)
foundation.load_state_dict(ckpt['model'], strict=False)

# Load CenterNet detector
encoder = JAISPEncoderWrapper(foundation, freeze=True)
detector = CenterNetDetector.load('models/checkpoints/centernet_v7.pt',
                                   encoder=encoder, device='cuda')
detector.eval()

# Run prediction
# images, rms: dict[str, Tensor[1, 1, H, W]]
pred = detector.predict(images, rms, conf_threshold=0.3, tile_hw=(512, 512))
# pred['centroids']    [N, 2]    normalized (x, y)
# pred['scores']       [N]       heatmap confidence
# pred['positions_px'] [N, 2]    pixel coordinates
# pred['log_flux']     [N]       flux proxy
```

The `predict()` API is identical to the DETR detector, so CenterNet drops into
the astrometry pipeline without code changes.

## Files

| File | Description |
|------|-------------|
| `centernet_detector.py` | `CenterNetDetector` model, heatmap/offset/flux/profile heads |
| `centernet_loss.py` | Focal loss, Gaussian heatmap rendering, masked L1 |
| `train_centernet.py` | CenterNet training loop with W&B heatmap visualization |
| `detector.py` | `JaispDetector` DETR model (archived, kept for reference) |
| `matcher.py` | Hungarian matcher + DETR loss (archived) |
| `train_detection.py` | DETR training loop (archived) |
| `dataset.py` | Pseudo-label dataset with caching, shared by both architectures |
