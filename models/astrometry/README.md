# JAISP Astrometry Concordance

Predicts smooth offset fields (ΔRA\*, ΔDec) in arcseconds that map Rubin astrometry onto the Euclid VIS reference frame. Follows the concordance data product specification: physically meaningful offsets in sky coordinates where Δ=0 means perfectly aligned, VIS is the reference grid, and geometry is handled by WCS.

## Architecture

The head operates on frozen backbone token embeddings from a (Rubin band, Euclid VIS) pair:

```
Rubin band tokens [B, N_r, D]     VIS tokens [B, N_v, D]
         │                                  │
         └──────── interpolate to ──────────┘
                   common grid
                       │
              cross-correlation
              (local, ±r tokens)
                       │
              correlation volume
              [B, S², H_tok, W_tok]
                       │
                  soft-argmax
              (differentiable peak)
                       │
               raw offsets (tokens)
                       │
              refinement CNN (k=5)
              (enforces smoothness)
                       │
            token → arcsec conversion
                       │
          ΔRA*, ΔDec [B, 1, H, W]
                       │
         optional mesh upsample (DSTEP)
```

Key design choices:
- **Cross-correlation** rather than regression — the foundation model's position encoding already encodes spatial correspondences, so correlation at the token level naturally extracts offsets.
- **Soft-argmax** provides differentiable sub-pixel precision from the correlation peak.
- **Learnable temperature** controls the sharpness of the soft-argmax peak, adapting to the correlation landscape.
- **Large-kernel refinement CNN** (5×5 convolutions) enforces spatial smoothness — astrometric distortions are low-frequency by nature.
- **Smoothness loss** explicitly penalizes high-frequency structure in the predicted field.

## Self-Supervised Training

No external astrometric catalogs needed. The training loop:

1. Takes a tile with both Rubin and VIS data
2. Generates a synthetic smooth offset field (ΔRA\*, ΔDec)
3. Warps the Rubin image by that offset using `grid_sample`
4. Encodes both (warped Rubin, original VIS) through the frozen backbone
5. Head predicts the offset → loss against the known synthetic ground truth

A curriculum ramps offset complexity: constant → affine → smooth sinusoidal fields over the first ~10 epochs.

## Files

| File | Role |
|------|------|
| `head.py` | `AstrometryConcordanceHead` — correlation + soft-argmax + refinement |
| `offsets.py` | Synthetic offset generators (constant/affine/smooth) + image warping |
| `dataset.py` | `AstrometryDataset` — always pairs Rubin band with VIS |
| `train_astrometry.py` | Training loop with curriculum + wandb logging |
| `export_fits.py` | Inference + FITS export following concordance spec |

## Quick Start

### Train

```bash
python3 models/astrometry/train_astrometry.py \
    --rubin-dir data/rubin_tiles_ecdfs \
    --euclid-dir data/euclid_tiles_ecdfs \
    --backbone-ckpt checkpoints/jaisp_v5/best.pt \
    --output-dir checkpoints/jaisp_astrometry \
    --epochs 50 \
    --batch-size 2 \
    --max-offset 0.5
```

### Smoke test

```bash
python3 models/astrometry/train_astrometry.py \
    --rubin-dir data/rubin_tiles_ecdfs \
    --euclid-dir data/euclid_tiles_ecdfs \
    --backbone-ckpt checkpoints/jaisp_v5/best.pt \
    --output-dir checkpoints/jaisp_astrometry_smoke \
    --epochs 3 \
    --batch-size 1 \
    --num-workers 0 \
    --wandb-mode disabled
```

### Export FITS concordance product

```bash
python3 models/astrometry/export_fits.py \
    --backbone-ckpt checkpoints/jaisp_v5/best.pt \
    --head-ckpt checkpoints/jaisp_astrometry/best_astrometry.pt \
    --rubin-dir data/rubin_tiles_ecdfs \
    --euclid-dir data/euclid_tiles_ecdfs \
    --output concordance_ecdfs.fits \
    --dstep 8
```

## Key Arguments

| Arg | Default | Notes |
|-----|---------|-------|
| `--max-offset` | 0.5 | Max synthetic offset amplitude in arcseconds |
| `--search-radius` | 3 | ±3 tokens = 7×7 correlation window |
| `--softmax-temp` | 0.1 | Initial temperature (learnable) |
| `--smooth-weight` | 0.1 | Gradient penalty weight on predicted field |
| `--curriculum-epochs` | 10 | Ramp from constant → smooth offsets |
| `--dstep` | 8 | Mesh sampling in VIS pixels for FITS export |

## Metrics

Training logs (wandb) include:
- `MAE_total` — mean absolute positional error in milliarcseconds
- `MAE_ra`, `MAE_dec` — per-component
- `frac_01arcsec` — fraction of positions with error < 100 mas
- `frac_02arcsec` — fraction of positions with error < 200 mas
- `temp` — learned soft-argmax temperature

## FITS Product Structure

Following the concordance spec, the output FITS file contains:

```
HDU 0: Primary (keywords: CONCRDNC, DSTEP, DUNIT, REFFRAME, INTERP)
HDU 1: tile_x00_y00.r.DRA  — ΔRA* for rubin_r
HDU 2: tile_x00_y00.r.DDE  — ΔDec for rubin_r
HDU 3: tile_x00_y00.g.DRA  — ΔRA* for rubin_g
...
```

Each HDU has keywords: `DSTEP`, `DUNIT=arcsec`, `INTERP=bilinear`, `CONCRDNC=True`, `RBNBAND`, `REFFRAME=euclid_VIS`, `TILEID`.

## Open Questions (from spec)

- **Mesh size**: DSTEP=8 is the baseline (~0.8″ sampling). The model can predict at any resolution; the question is what's needed for the science. The `--dstep` flag controls this at export time.
- **Extension to Roman/NISP**: The architecture is band-agnostic — just swap the reference band and adjust pixel scales. The head itself doesn't know which instruments are involved.
- **Epoch corrections**: Currently assumes surveys handle proper motion. If needed, a time-dependent term could be added as an additional output channel.
