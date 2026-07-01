# Foundation field-measurement head — design

## Goal
Test whether the foundation multi-band representation can **measure the concordance
field better than the classical joint fit**, specifically by letting **faint sources**
carry the field measurement at lower per-source noise.

## Why the production head can't do this
The production latent-position head regresses every source to the **VIS centroid**.
The concordance field IS the coherent band−VIS offset, so regressing to VIS removes
it: head-residual field ≈ 1 mas ≈ shuffled-null. A corrector, not a measurer.

## The tension to resolve
The multi-band information that reduces centroid noise is largely the **sharp VIS
localization** — but using VIS's *position* to place band X biases band X toward VIS
and collapses the field. So a field-preserving denoiser must:
  (a) reduce band X's centroid **noise**, but
  (b) predict band X's position in **band X's own frame** (preserve F_X = band−VIS).

## Approach (this head)
Per-band position head, differing from the production head in TWO ways:
  1. **Target = the band's OWN centroid** (not VIS). Trained by jitter self-supervision
     against each band's own high-S/N Gaussian centroid, so F_X is preserved by
     construction (the head denoises toward band X's frame, not VIS's).
  2. **Input stem = the band's OWN stem** (`encoder.stems[band]`), plus the fused
     bottleneck used ONLY for shape/deblending context (which flux is the source,
     its extent/SED) — not for copying VIS's absolute position.
The head outputs an offset in the band's pixel frame → sky via the band's local WCS
Jacobian. Reuses the LatentPositionHead architecture (bottleneck ConvNeXt + stem
conv + MLP), instantiated per band pixel scale.

## The decisive tests (eval, after training)
1. **Field-preservation control (must pass first):** fit the field on the head-refined
   band−VIS offsets and check it CORRELATES with the classical raw field (not collapsed
   to ~0 like the VIS-target head). If it collapses, the fused bottleneck is leaking VIS
   position → drop the bottleneck or down-weight it. Also: head band-averaged offset vs
   classical ≈ 0 (denoise, not shift).
2. **Beats-classical test:** on FAINT sources (S/N<10), split-half reproducibility of the
   field from (a) classical raw band centroids, (b) classical joint-fit residuals, (c) this
   head. Metric: split-half vector correlation / diff at 150" smoothing, and agreement with
   the era-stable v8 reference. Head wins if it gives higher split-half r / lower diff at
   equal faint-source count — i.e. lower per-source noise while preserving F_X.
3. **Noise check:** per-source band-centroid scatter (head vs classical) at matched S/N,
   using injection truth (known positions) — confirms the noise reduction is real, not
   label emulation.

## Files
- `train_field_head.py`  — trainer (this experiment).
- eval: extend `eval_latent_position.py` / a notebook cell to export band−VIS offsets from
  the field head and run the split-half field comparison vs classical + joint.

## Status
v0 — training scaffold. Open risks: (i) fused-bottleneck VIS leakage collapsing the field
(mitigation: band-stem-only ablation); (ii) Rubin 0.2"/px vs Euclid 0.1"/px frame handling
(head parameterized by per-band pixel scale); (iii) whether denoising beats classical at
all given the coadd sub-pixel-phase floor.
