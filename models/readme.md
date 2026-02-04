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
