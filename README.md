## Overview

JAISP (Joint AI Survey Processing) is a **self-supervised, multi-instrument foundation model for astronomical imaging**.
It learns a shared latent representation of the same sky region across different telescopes, bands, resolutions, and noise regimes.

The core objective is representation learning, not supervised prediction.
The learned embeddings are intended to support downstream tasks such as:

- astrometric alignment
- cross-band/instrument matching
- deblending and morphology analysis
- photometric consistency checks
- reconstruction or generative heads on top of the latent space

---

## Current Model (v5)

The active foundation model is **JAISP Foundation v5** (`models/jaisp_foundation_v5.py`).

Key characteristics:

- **Band-specific stems + shared encoder**
  Each band has its own shallow stem, but all bands share a common transformer encoder and projection space.

- **Strict spatial alignment objective**
  v5 enforces token-to-token matching at corresponding spatial positions (`shift_px=0`).
  This is designed to preserve precise position correspondence across views.

- **Student/Teacher self-supervision**
  Training uses a student branch with predictor and an EMA teacher target branch.

- **Information-weighted alignment**
  Loss is weighted using signal-based information maps (SNR + gradients) so training emphasizes informative regions over empty sky.

- **VICReg regularization**
  Variance and covariance terms are used to reduce representation collapse and redundancy.

---

## Core Training Setup

For each sample, the model sees two views of the same sky tile (often cross-instrument) and learns aligned token embeddings.

Training behavior is implemented in:
- `models/train_jaisp_foundation_v5.py`
- `models/jaisp_dataset_v4.py`

Dataset behavior (current):
- Supports Rubin + Euclid band pairs sampled from available data
- Preserves native image sizes (e.g., Rubin ~512x512, Euclid ~1050x1050)
- Handles alignment at token-grid level via interpolation inside the model/loss

---

## Diagnostics and Validation

Initial model checks are documented in:
- `models/testing_model.ipynb`

Notebook diagnostics include:
- masked prediction similarity checks
- token-level cross-view correspondence maps
- source-aware position-encoding tests

These are used to verify that embeddings are cross-view consistent and spatially meaningful.

---

## Reconstruction Extension

On top of the v5 foundation model, a downstream reconstruction pipeline is now included in:
- `models/reconstruction/`

It adds:
- multi-band `k->1` masked reconstruction (including `9->1` scenarios)
- mixed masking strategies (random/object-aware/hard)
- a reconstruction head with training + W&B monitoring

This is a downstream head, separate from foundation pretraining.

---

## Project Scope

This repository focuses on building and validating the **foundation representation** and related downstream prototypes.
It is an active research codebase, so architecture and training procedures are still evolving.
