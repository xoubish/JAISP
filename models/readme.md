# JAISP Foundation v3: Per-Band Multi-View Representation Learning

A self-supervised foundation model for learning unified representations across astronomical imaging surveys, specifically designed for **Rubin Observatory (LSST)** and **Euclid** cross-matching.

## Key Insight

Traditional approaches stack all bands into a multi-channel image and process them together. This is problematic for astronomical data because:

1. **Sources appear/disappear across wavelengths** — A faint red galaxy visible in H-band may be completely absent in u-band
2. **Per-band astrometric offsets** — Real data has small registration errors between bands
3. **Vastly different PSFs** — Rubin u-band PSF ≠ Euclid H-band PSF
4. **Different noise characteristics** — Each band has its own depth and systematics

**Our approach:** Treat each band as a separate "view" of the same sky patch. The model learns that *the same celestial object should have similar representations regardless of which wavelength we observe it at*.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         JAISP Foundation v3                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   Band Image (1, H, W)                                                    │
│        │                                                                  │
│        ▼                                                                  │
│   ┌─────────────────┐     ┌─────────────────┐                            │
│   │  Band-Specific  │     │  Information    │                            │
│   │     Stem        │     │     Map         │                            │
│   │  (per band)     │     │  (S/N weights)  │                            │
│   └────────┬────────┘     └────────┬────────┘                            │
│            │                       │                                      │
│            ▼                       │                                      │
│   ┌─────────────────┐              │                                      │
│   │  Shared ViT     │              │                                      │
│   │    Trunk        │◄─────────────┘  (weights used in loss)             │
│   │ (band-agnostic) │                                                     │
│   └────────┬────────┘                                                     │
│            │                                                              │
│            ▼                                                              │
│   Token Grid (N, D)  ──────►  Token-wise JEPA Loss                       │
│            │                  (weighted by info map)                      │
│            ▼                         +                                    │
│   Global Embedding   ──────►  VICReg Regularization                      │
│                               (prevent collapse)                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| **Band Stems** | 10 small CNNs (one per band) that handle band-specific PSF/noise. Input: single-channel image + RMS. Output: 64-channel features. |
| **Information Map** | Computes per-pixel weights from S/N and gradient magnitude. High weight = astronomical source, low weight = background sky. |
| **Shared Trunk** | Vision Transformer that processes features from *any* band identically. Forces the representation to be band-agnostic. |
| **Predictor** | Asymmetric MLP applied to online branch (JEPA-style). Helps prevent representation collapse. |
| **EMA Target** | Exponential moving average of trunk weights. Provides stable targets for self-supervised learning. |

## Supported Bands

| Instrument | Bands | Wavelength Range |
|------------|-------|------------------|
| Rubin/LSST | u, g, r, i, z, y | 320–1060 nm |
| Euclid | VIS, Y, J, H | 550–2000 nm |

Total: **10 separate views** of each sky patch.

## Training Objective

We use **JEPA (Joint Embedding Predictive Architecture)** with information weighting:

```
Loss = Σ_tokens [ w_i × (1 - cos_sim(pred_i, target_i)) ] + VICReg
```

Where:
- `w_i` = information weight for token i (high for sources, ~0 for sky)
- `pred_i` = online encoder + predictor output
- `target_i` = EMA target encoder output (stop-gradient)
- `VICReg` = variance + covariance regularization to prevent collapse

### Why Information Weighting?

Astronomical images are ~90% empty sky. Without weighting, the model would "win" by perfectly matching the background (easy, but useless). The information map ensures loss is dominated by **anchors** (stars, galaxies) not sky.

### Pairing Strategy

Not all band pairs are equally informative. We use smart sampling:

| Strategy | Probability | Purpose |
|----------|-------------|---------|
| Cross-instrument | 50% | Rubin ↔ Euclid alignment |
| Hard pairs (large λ gap) | 30% | u ↔ H forces robust features |
| Random | 20% | Coverage of all combinations |

We also track per-band usage to ensure no band is neglected.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/jaisp-foundation.git
cd jaisp-foundation

# Create environment
conda create -n jaisp python=3.10
conda activate jaisp

# Install dependencies
pip install torch torchvision
pip install numpy scipy matplotlib
pip install wandb tqdm
pip install umap-learn  # optional, for visualization
```

## Data Format

Expected directory structure:

```
data/
├── rubin_tiles_ecdfs/
│   ├── tile_001.npz
│   ├── tile_002.npz
│   └── ...
└── euclid_tiles_ecdfs/
    ├── tile_001.npz
    ├── tile_002.npz
    └── ...
```

Each `.npz` file should contain:
- `image`: `(n_bands, H, W)` — pixel values (6 bands for Rubin, 4 for Euclid)
- `rms`: `(n_bands, H, W)` — per-pixel noise (RMS) for each band

Tiles with matching names (e.g., `tile_001.npz` in both directories) are treated as the same sky patch observed by both instruments.

## Usage

### Training

```python
from train_jaisp_foundation_v3 import JAISPTrainerV3

trainer = JAISPTrainerV3(
    rubin_dir="./data/rubin_tiles_ecdfs",
    euclid_dir="./data/euclid_tiles_ecdfs",
    output_dir="./checkpoints",
    embed_dim=256,
    trunk_depth=6,
    patch_size=16,  # ViT patch size
    batch_size=4,
    wandb_project="JAISP-Foundation"
)

trainer.train(
    epochs=100,
    lr=3e-4,
    warmup_epochs=10
)
```

### Inference

```python
import torch
from jaisp_foundation_v3 import JAISPFoundationV3

# Load model
checkpoint = torch.load("checkpoints/best_model.pt")
model = JAISPFoundationV3(
    band_names=checkpoint['band_names'],
    embed_dim=256,
    trunk_depth=6
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get representation for a single band
image = torch.randn(1, 1, 512, 512)  # (B, 1, H, W)
rms = torch.ones(1, 1, 512, 512) * 0.1

with torch.no_grad():
    out = model.get_representation(image, rms, band='rubin_r')
    
    tokens = out['tokens']      # (1, N_tokens, 256) - spatial features
    global_emb = out['global']  # (1, 256) - pooled embedding
    weights = out['weights']    # (1, 1, H, W) - information map
```

### Cross-Band Matching

```python
# Encode same patch in two different bands
out_rubin_r = model.get_representation(img_rubin_r, rms_rubin_r, 'rubin_r')
out_euclid_h = model.get_representation(img_euclid_h, rms_euclid_h, 'euclid_h')

# Compare global embeddings
similarity = F.cosine_similarity(out_rubin_r['global'], out_euclid_h['global'])
print(f"Cross-band similarity: {similarity.item():.3f}")

# Or compare token grids for spatial matching
# (tokens are aligned spatially if images are registered)
```

## Training Diagnostics

The training script tracks several metrics to ensure healthy learning:

### Must Watch

| Metric | Good Value | Bad Sign |
|--------|------------|----------|
| `token_similarity` | 0.3–0.8, rising slowly | Jumps to >0.95 quickly → collapse |
| `var_loss` | Drops to ~0, stays there | Stays high → dimensions not used |
| `var_min` | >0.1 | <0.01 → some dimensions dead |
| `effective_dim` | Close to embed_dim | <<embed_dim → using subspace only |

### Diagnostic Plots

1. **Cross-Band Similarity Matrix** — 10×10 heatmap showing avg similarity between all band pairs. Should show structure (nearby wavelengths more similar) but no band should be isolated.

2. **Information Map Overlay** — Verifies that high-weight regions correspond to real sources, not noise or artifacts.

3. **Token Activation vs Weight Correlation** — Should be positive (tokens activate where sources are).

4. **Per-Band Collapse Check** — Ensures no individual band has collapsed to constant output.

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `embed_dim` | 256 | Token/embedding dimension |
| `trunk_depth` | 6 | Number of transformer layers |
| `patch_size` | 16 | ViT patch size (smaller = finer spatial resolution) |
| `ema_decay` | 0.996 | Target encoder momentum (higher = more stable) |
| `var_weight` | 1.0 | VICReg variance term weight |
| `cov_weight` | 0.04 | VICReg covariance term weight |
| `cross_instrument_prob` | 0.5 | Fraction of pairs that are Rubin↔Euclid |
| `hard_pair_prob` | 0.3 | Fraction of pairs with large wavelength gap |

## Design Decisions

### Why per-band stems instead of one multi-channel encoder?

1. **Flexibility** — Can handle missing bands (just don't use that stem)
2. **Specialization** — Each stem learns band-specific denoising/PSF handling
3. **Interpretability** — Can diagnose issues per-band

### Why token-level loss instead of global?

1. **Spatial grounding** — Forces local correspondence, not just "same patch"
2. **More signal** — Thousands of token pairs vs one global pair per image
3. **Information weighting** — Can downweight background tokens

### Why JEPA instead of contrastive (SimCLR, MoCo)?

1. **No negative sampling** — Avoids tricky batch construction
2. **Predictive** — Learns to predict, not just discriminate
3. **Works well for dense tasks** — Token-level predictions natural for spatial data

## Extending to More Instruments

To add a new instrument (e.g., HST, JWST):

```python
# 1. Define bands
HST_BANDS = ['hst_f606w', 'hst_f814w']

# 2. Update model
model = JAISPFoundationV3(
    band_names=RUBIN_BANDS + EUCLID_BANDS + HST_BANDS,
    ...
)

# 3. Add data directory to dataset
dataset = JAISPPerBandDataset(
    data_dirs={
        'rubin': './data/rubin',
        'euclid': './data/euclid',
        'hst': './data/hst'
    },
    band_configs={
        'hst_f606w': {'instrument': 'hst', 'index': 0},
        'hst_f814w': {'instrument': 'hst', 'index': 1},
    }
)
```

## Citation

```bibtex
@software{jaisp_foundation,
  title={JAISP Foundation: Per-Band Multi-View Representation Learning for Astronomical Surveys},
  author={Your Name},
  year={2026},
  url={https://github.com/your-org/jaisp-foundation}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

This work is based on concepts from:
- [JEPA (Joint Embedding Predictive Architecture)](https://arxiv.org/abs/2301.08243)
- [VICReg (Variance-Invariance-Covariance Regularization)](https://arxiv.org/abs/2105.04906)
- [DETR (Detection Transformer)](https://arxiv.org/abs/2005.12872) — for the original object query concept
- [SuperPoint](https://arxiv.org/abs/1712.07629) — for learned keypoint detection ideas

Developed for joint analysis of Rubin Observatory and Euclid survey data.
