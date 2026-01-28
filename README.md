## Overview

JAISP (Joint AI Survey Processing) is a **self-supervised, multi-instrument foundation model for astronomical imaging**.  
Its goal is to learn a shared latent representation of the sky that is consistent across
different telescopes, resolutions, noise properties, and wavelength bands.

Rather than predicting pixels directly, the model learns to **align and structure
representations** of the same sky region observed under different conditions.  
Once trained, this latent space can be reused for downstream tasks such as:

- sub-pixel astrometric alignment  
- deblending in crowded fields  
- cross-band and cross-instrument photometry  
- shape and morphology measurements  
- image reconstruction or enhancement  

The model is intentionally trained **without labels**, relying only on the fact that
different views correspond to the same underlying scene.

---

## Core Idea

For each sky patch, the model is given **two different views** of the same region
(e.g. different bands, instruments, or resolutions).  
Each view is encoded independently, and the model is trained so that:

- corresponding spatial regions produce **similar embeddings**
- different regions remain distinguishable
- information is distributed across embedding dimensions (no collapse)

This follows the philosophy of modern **joint-embedding predictive architectures (JEPA)**:
learn *what is shared* between views, not how to reconstruct one from the other.

---

## Architecture Summary

The model consists of:

- **Two view-specific encoders**  
  Each encoder processes an image from a given instrument/band and produces a grid of
  spatial tokens (patch-level embeddings).

- **Shared embedding space**  
  Although the encoders are separate, their outputs are trained to live in a common
  latent space, allowing direct comparison across instruments.

- **Spatial token alignment**  
  Tokens correspond to fixed spatial locations, enabling the model to learn fine-grained
  geometric consistency rather than only global similarity.

- **Optional teacher network (EMA)**  
  A momentum-updated teacher provides a slowly evolving target representation, improving
  stability and preventing representational drift.

---

## Training Objective

Training uses a combination of self-supervised losses inspired by VICReg-style methods:

### 1. Alignment Loss
Encourages corresponding tokens from two views of the same sky patch to be close in
embedding space.

This is what drives cross-band and cross-instrument consistency.

### 2. Variance Loss
Ensures that each embedding dimension has sufficient variance across samples, preventing
collapse to a constant representation.

### 3. Covariance Loss
Penalizes correlations between embedding dimensions, encouraging information to be
spread across the latent space.

Together, these losses produce embeddings that are:
- aligned but not identical
- expressive but stable
- structured rather than noisy

---

## What the Model Learns

Without any explicit labels or wavelength ordering, the model learns:

- **Instrument-agnostic morphology**
- **Implicit spectral energy distribution (SED) relationships**
- **Robust features insensitive to noise and resolution**
- **Salient structures (sources, blends, diffraction patterns)**
- **Fine-grained spatial correspondences suitable for astrometry**

Wavelengths are *not* explicitly encoded or ordered; instead, spectral relationships are
learned implicitly through consistent co-occurrence across views.

---

## Why This Is a Foundation Model

The model is trained **once**, without task-specific supervision, and can then be reused
or adapted for many downstream problems.

Key properties that make it a foundation model:

- multi-band, multi-instrument training
- self-supervised objectives
- spatially resolved embeddings
- transferable latent space
- no reliance on catalogs or labels

Downstream models can either:
- operate directly in the learned latent space, or
- fine-tune the encoders for specific scientific tasks

---

## Intended Use

This repository focuses on **learning the representation**, not on final science products.
The foundation model is meant to be a building block for later stages such as:

- precision astrometry
- joint deblending across bands
- photometric consistency checks
- shape measurements for weak lensing
- generative or predictive models built on top of the latent space

---

## Status

This is an active research project.
Architecture, losses, and training strategies may evolve as the model scales to larger
datasets and higher-resolution inputs.
