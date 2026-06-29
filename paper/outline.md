# JAISP Paper — Outline

> Fig. 7 (completeness) merged into Fig. 6; later figures renumbered. Captions stay
> version-agnostic.

## 1. Introduction
- Big surveys, need for joint processing (cite JSP)
- State of AI in astronomy — replace or augment traditional?
- Foundation models: growing interest, not an answer to everything (self-cite)
- Prior work positioning: AstroCLIP, Multimodal Universe, AstroPT, classical forced phot/astrometry — novelty boundary (pixel-level, multi-instrument, mixed-resolution fusion)
- Start from processed mosaics; joint representation learning; can be pushed earlier
- This paper

## 2. Data
- Multi-band, multi-survey imaging
- Rubin *ugrizy* — DP1 ComCam coadds, patches/tracts
- Euclid VIS + NISP *YJH* — imaging from DR1 MER mosaics, catalogs from Q1 MER (don't conflate)
- MER vs NIR mosaics
- Gaia DR3 — PSF training + absolute astrometric reference
- Fields, area, depth
- Tiling and overlap; dedup; no leakage (patch-disjoint audit)
- Variance maps vs MAD fallback
- **Fig. 1**: 10-band patch cutouts + RMS

## 3. AI Architecture
- Foundation + downstream heads
- Detection & astrometry heads here; photometry & shape in prep
- Foundation evolution: where we started, where we landed (JEPA abandoned)
- Version-insensitivity (v8 ≈ v9 ≈ v10): judge by OOD, not in-distribution probes
- **Fig. 2**: architecture
- **Fig. 3**: masked-band reconstruction
- **Fig. 4**: information-routing / probes
- GPU, W&B, training time

## 4. Detection
- Question: does the latent carry multi-band source evidence, and can a light head beat classical?
- Heads: CenterNet (fused bottleneck) vs StemCenterNet (native stems); DETR head tried, abandoned
- Labels: internal VIS peak-finder + SEP assist + spike/dup cleaning; self-training cleans labels
- MER not for training — external cross-check; MER fine-tuning gave no gain
- Qualitative overlay vs classical & MER (**Fig. 5**)
- Head choice from injection: fused fuses, stems don't (**Table 1**)
- Injection depth + multi-band gain (**Fig. 6** top)
- Band-subset: NISP-only = high-*z* dropout proxy; Rubin-only ≈ 0 (Euclid-anchored)
- Completeness/purity vs MER, MER is not truth (**Fig. 6** bottom)
- What we gain, what we miss (large resolved galaxies), how to improve

## 5. Astrometry
- Idea: transfer sharp VIS localization to faint bands; map the concordance field — not "beat the solution"
- Rubin–Euclid agreement before correction (both Gaia-aligned)
- Raw offset is S/N-dependent, dominated by per-source centering not WCS distortion (**Fig. 7**)
- Do faint sources help constrain the field? (**Fig. 8**)
- Head corrects faint-source centering, re-measures the field (**Fig. 9**)
- Head has a faint-end ceiling; classical joint 10-band fitter is the best position instrument
- Injection-truth test: real low-SNR gain (VIS-localization transfer), bounded by VIS floor
- Centroid-emulation caveat: faint-end residual metric = emulation, not truth → need injection + Gaia
- Absolute validation vs Gaia: frame tie + bright-end accuracy
- Concordance field is a measurement, not a correction (doesn't improve held-out positions)
- PINN vs HGP (agree at degree scale; sub-degree is solver noise)
- Ian's centering dependence on morphology

## 6. Out-of-Distribution
- Do results persist OOD (EDF-S, no retraining)?
- Detection transfers nearly perfectly (**Fig. 10**)
- Astrometry: modest penalty (earlier scary factor was a MAD-variance artifact) (**Fig. 11**)
- Bright-star suppression reproduces; origin is input normalization, not the field
- Extensible all-sky

## 7. Discussion
- Native rather than processed mosaics
- Better PSF
- Classical to inform training, faster inference once learned
- Label noise limits in-distribution metrics; next head needs a finer bottleneck
- Next paper

---
### Figures / table
- Fig. 1 — 10-band patch + RMS
- Fig. 2 — architecture
- Fig. 3 — reconstruction
- Fig. 4 — probes
- Fig. 5 — detection overlay
- Table 1 — detection-head comparison
- Fig. 6 — detection performance
- Fig. 7 — raw offset vs S/N
- Fig. 8 — faint sources & field
- Fig. 9 — head correction & field
- Fig. 10 — OOD detection
- Fig. 11 — OOD astrometry
