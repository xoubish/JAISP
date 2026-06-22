# Design Doc: Cross-Instrument Predictive Sufficiency & the Less-Lossy Catalog

**Status:** draft v0 (2026-06-21). A spec to start building, on the frozen JAISP v10 representation.
**Thesis (one sentence):** *The per-source information a classical catalog discards is measurable in bits, it is concentrated on exactly the objects the catalog gets wrong, and both quantities are read off a single engine — leave-one-band-out reconstruction inside one jointly-calibrated 10-band Rubin+Euclid model.*

---

## 0. Gating literature check — verdict and how it reshaped this doc

Full notes from four parallel probes live in the session; the load-bearing conclusions:

| Component | Status | Nearest prior art | Surviving differentiator |
|---|---|---|---|
| **1. Information budget / sufficiency** | **Open in astronomy** | V-information/MDL/conditional probing (Xu 2020 `2002.10689`; Hewitt 2021 `2109.09234`; Voita-Titov `2003.12298`; Pimentel `2004.03061`) — all NLP. Astronomy: DeepDISC MI(features;z) appendix (Merz 2025 `2510.27032`); Sui 2023/25 sufficiency-via-MI but *summary→parameter* (`2307.04994`,`2511.08716`). | First **capacity-controlled, information-theoretic** test that a cross-instrument representation is a *predictively sufficient statistic for the catalog*, quantified **in usable bits**, with the **catalog as the conditioning baseline**. |
| **2. Anomaly / residual** | **Largely scooped (2026)** | **AS-Bridge** `2603.11928` (LSST↔Euclid translation, pixel-residual anomaly, unified artifact+discovery score, flux-norm aggregation). **Mercader-Perez** `2604.09787` (cross-instrument signal/instrument disentanglement, anomaly in both latents). **AION-1** `2510.17960` (has masked-multiband engine, never scores anomaly). | **Masked-band leave-one-out (LOO) residual inside ONE jointly-calibrated model** — not A→B translation, not latent disentanglement. Plus the within-/cross-instrument residual *decomposition* and the photo-z-outlier linkage neither paper makes. |
| **3. Fusion (paper target)** | **Open** | none | Tying "discarded information" (sufficiency gap) to "catalog error" (anomaly residual) as **two readouts of one LOO engine**. This is the contribution. |

**Decision:** lead with sufficiency + fusion as the thesis; the anomaly map is *the bridge that makes the bits physical*, not a standalone claim. AS-Bridge and Mercader-Perez are cited in the intro as the works we differentiate from on **mechanism** (single jointly-calibrated model, masked-band LOO) and on **framing** (information budget, not just a score). Do **not** sell "unified anomaly score" as novel — it is now prior art.

**The unification that nobody else has.** Both deliverables are computed from the *same* object: with band $b$ masked, the frozen foundation predicts $\hat b$ from the other 9 jointly-calibrated bands. The residual $r_b = (b-\hat b)/\sigma_b$ is simultaneously
- the **local failure of cross-instrument predictive sufficiency** (the other 9 bands + the representation cannot predict $b$ here), and
- the **anomaly score**.

The catalog enters as a *lossy summary baseline* in the same prediction task. This gives a 2-axis decomposition per source — *catalog prediction error* vs *representation prediction error* — whose quadrants are the whole story (§5, §6).

---

## 1. What we predict, and why it is simulator-free

The target $Y$ is **the actual pixels (or per-source flux/profile) of a held-out band** — not a simulated label, not a fitted parameter. Reality is the held-out band itself. Three predictors of $Y$:

- **C (catalog baseline):** the classical catalog row for the source — VIS-px position, per-band fluxes of the *other 9 bands*, Gaussian shape (e1,e2,r_half), SNR. The honest "what a downstream user actually has."
- **R (representation):** frozen v10 features at the source — bottleneck window + VIS-stem window (the astrometry head's extraction pattern).
- **C⊕R:** both, same probe family.

Everything is data-space and parameter-free up to the final readout. The MAE objective JAISP already trained *is* the held-out-prediction engine; notebook 05's probes are ~80% of the sufficiency apparatus.

---

## 2. (a) The sufficiency metric, and the confound controls

### 2.1 Metric: conditional usable information (V-information)

Report **conditional predictive V-information** (Hewitt et al. 2021; Xu et al. 2020):

$$I_{\mathcal V}(R \to Y \mid C) \;=\; H_{\mathcal V}(Y\mid C) \;-\; H_{\mathcal V}(Y\mid C,R)$$

estimated as the **held-out cross-entropy / negative-log-likelihood gap** between a probe given $C$ and an identical-capacity probe given $(C,R)$. Units: bits per source (classification readouts) or nats / held-out $R^2$ / $\Delta\chi^2$ (continuous). This is, literally, **the information the catalog discards but the representation keeps**.

Why conditional V-information and not Shannon MI: classical MI is invariant under invertible maps, so "R has more MI than C" is meaningless to an information-theory referee (data-processing inequality). V-information is restricted to a bounded model family $\mathcal V$, *violates* DPI, and makes "computation creates usable information" a coherent, defensible claim. **This sentence must be in the paper** — it is the single most likely referee attack and Xu 2020 is the pre-emption.

### 2.2 The four confound controls (the known weakness)

The measure is sensitive to probe capacity, architecture, and what C is allowed to contain. Control all four:

1. **Matched probe family.** Probe$(C)$, Probe$(R)$, Probe$(C,R)$ are the *same* architecture, capacity, regularization, optimizer, and HPO budget. The gap is then attributable to inputs, not probe expressivity.
2. **Capacity sweep + MDL cross-check.** Sweep capacity (ridge → small MLP → larger MLP). A real gap is *stable or grows* with capacity (Pimentel: stronger probe ⇒ tighter bound). Add **online/prequential MDL codelength** (Voita-Titov) as a capacity-internalizing second number: if R wins on codelength too, capacity is not the explanation.
3. **Fair, non-strawman baseline.** C gets engineered nonlinear features (colors, flux ratios, SNR interactions, shape moments) so it is *representation-vs-best-catalog*, not vs linear-catalog. Optionally a parameter/FLOP-matched control.
4. **Selectivity + seed floor.** Report selectivity against a **control task** (shuffled-embedding / random-label, Hewitt-Liang 2019) to bound memorization. Because JAISP's own audits show v8≈v9≈v10 foundation equivalence and a ~5 mas run-to-run PINN floor, **the gap must exceed encoder-seed and probe-seed noise** — report bootstrap + seed-ensemble error bars, and state the lower-bound honesty ("the representation has *at least* X bits beyond the catalog").

---

## 3. (b) The anomaly score and the masking/inference scheme

### 3.1 No new training for the raw map

The raw anomaly map is **pure inference** on the frozen v10 foundation + decoder. For each band $b$: mask $b$, forward-pass, get $\hat b$, compute the noise-normalized residual using the **RMS maps that already exist** in the NPZ `var` arrays:

$$r_b(x) = \frac{b(x)-\hat b(x)}{\sigma_b(x)}, \qquad \sigma_b = \sqrt{\mathrm{var}_b}\ \text{(or the EDF-S MAD fallback)}.$$

This is $\chi$-like and matches the foundation's own SNR-normalized loss space. On blank sky $r_b \sim \mathcal N(0,1)$ by construction — a **built-in null** (§5.4).

### 3.2 Masking variants (the decomposition that's still novel)

Run three masking regimes and keep them separate — the decomposition is a differentiator vs AS-Bridge/Mercader-Perez:

- **Single-band LOO** (mask one of 10): per-band surprise.
- **Leave-one-instrument-out:** mask *all* Euclid, predict from the 6 Rubin bands (and vice versa). This isolates **cross-instrument** inconsistency specifically — reality departing from what the other, independently-calibrated instrument predicts. This is the cleanest "cross-instrument" signal and the one tied to resolution/morphology/lensing.
- **Cross-band-within-instrument:** Rubin bands disagree among themselves ⇒ DCR, variability, single-band artifact.

### 3.3 Aggregation and calibration

- **Per-source score:** flux-normalized sum of $|r_b|$ over the source footprint (adopt AS-Bridge's flux-normalized aggregation explicitly — cite, don't re-derive — it is the published fix for low-SNR noise domination).
- **Excess over expectation:** anomaly $=$ measured $\sum r_b^2$ minus the $\chi^2$ expectation given the footprint pixel count and DOF. Calibrate the DOF on blank sky so a quiet galaxy scores ~0.
- **Two channels per source:** within-instrument surprise and cross-instrument surprise, reported separately. Artifacts (persistence, spikes, trails) tend to be *within-instrument, single-band*; lenses/mergers tend to be *cross-instrument, morphology-driven*. Showing this separation is the useful, publishable result.

---

## 4. (c) The fusion analysis — does discarded info predict catalog failure?

The headline test. For each source compute two scalars on the **same held-out-band prediction task**:
- $e_C$ = catalog prediction error (NLL or $\chi^2$ of Probe$(C)$),
- $e_R$ = representation prediction error (NLL/$\chi^2$ of Probe$(C,R)$, i.e. with the representation added).

The **discarded information** at that source is $\Delta_s = e_C - e_R \ge 0$ (per-source conditional V-information). The **catalog-failure** axis is the anomaly residual $A_s$ from §3 (which is itself $e_C$-like when C is the predictor — they are consistent by construction, which is the elegance). Plot/decompose into quadrants:

| | catalog predicts well ($e_C$ low) | catalog fails ($e_C$ high) |
|---|---|---|
| **rep adds little ($\Delta_s$ low)** | catalog sufficient (boring sky, point sources) | **irreducible anomaly** → noise, OR genuine surprise: systematics & discoveries (both fail) |
| **rep adds a lot ($\Delta_s$ high)** | minor refinement | **the less-lossy regime** → reality *is* predictable, but only the representation captures it: blends, color-gradient galaxies, resolution-limited structure, photo-z-relevant morphology |

**The claim to test:** $\Delta_s$ (discarded info) and $A_s$ (catalog error) are **positively correlated and co-located** — the catalog throws away the most exactly where it is most wrong. Quantify with rank correlation + the fraction of known interesting objects (lenses/mergers/variables, §5) falling in the bottom-right two cells vs a random-source baseline. A clean separation here *is* the paper.

**Photo-z linkage (nobody has this):** for sources with external photo-z PDFs available, test whether high-$A_s$ / high-$\Delta_s$ sources are enriched in **catastrophic photo-z outliers** (objects that collapse to the same broadband colors but not the same 10-band reconstruction). This connects an image-space residual to a catalog-failure mode that is currently only flagged in P(z) space.

---

## 5. (d) Validation plan that survives a referee

**Coordinate caveat, load-bearing:** CDFS/ECDFS/GOODS-S (RA~03:32, Dec −27.8°) and EDF-S (RA 61.24°, Dec −48.4°) are **21° apart with disjoint ground truth**. Match each truth set to the right field.

**5.1 Recover the known objects.**
- *EDF-S (production OOD field):* Euclid Q1 Strong-Lens catalog (Walmsley/Euclid 2025, `2503.15324`, Zenodo `10.5281/zenodo.15003116`) filtered to the EDF-S footprint — *same imaging as your tiles*; DES Bright Arcs / DES-CNN / Space Warps-DES for Dec −48.
- *ECDFS/GOODS-S:* mergers — Kartaltepe 2015 (`1401.2455`, VizieR J/ApJS/221/11), Galaxy Zoo CANDELS (Simmons 2017); variable AGN — Villforth 2010 (`1008.3384`), Poulain 2020 (`2001.02560`, ready-made AGN/star/SN labels); X-ray AGN — CDF-S 7Ms (Luo 2017, `1611.03501`). ECDFS is also an LSST DDF and DES-SN field C3.

**5.2 Injection-recovery (does most of the work, given few real objects).** Inject simulated lensed arcs / synthetic anomalies into **real cutouts of the actual field**; report completeness vs arc SNR, Einstein radius, lens-light contrast, magnitude. **Reuse the existing JAISP injection harness** (`io/_nb09_outputs/injection_truth_test.py`) — it already injects sub-pixel sources into all 10 bands of held-out tiles; extend it from point sources to arcs/blends.

**5.3 Metrics referees now expect.** Lead with **TPR₀** (TPR before first false positive) and **precision@top-N / recall-vs-rank** at a realistic inspection budget (few hundred–2000), anchored against three baselines: (i) random floor, (ii) **isolation-forest / embedding-distance on the same frozen features** (proves the LOO residual adds value over plain latent outlierness — this is the Astronomaly-style baseline and the thing a referee will demand), (iii) supervised baseline (proves signal exists). Comparable scales from the literature: Euclid Q1 SLDE-E reached purity 52% @ 50% completeness vs 0.05% random; ERO-Perseus single-field saw real-data purity ~11% despite great sim scores.

**5.4 Null / negative controls.** (a) Blank-sky $\chi^2$ calibration — quiet sky must score ~0. (b) Shuffle test / inject into empty fields for false-positive rate. (c) Held-out-class novelty: hide an object class from any supervised comparator and show recovery (k-fold given small counts).

**5.5 False-positive audit against artifacts.** Cross-match top candidates against Euclid MER `spurious_flag`/`det_quality_flag`, Rubin DP1 mask planes (STREAK/CR/edge/ghost), and the labeled artifact set (Sreejith 2024, `2504.08053`, Zenodo). Report the dominant FP categories explicitly — and use the within-instrument anomaly channel (§3.2) to *predict* them, turning the audit into a positive result.

**5.6 Honest sim-to-real gap.** Report sim and real-field numbers both; Q1 referees penalize sim-only (`2512.05899` documents a 92%→24% purity collapse).

---

## 6. (e) The one figure that carries the paper

**A 2D map: catalog prediction error $e_C$ (x) vs representation-added information $\Delta_s$ (y), one point per source**, colored by anomaly residual $A_s$, with known lenses (★), mergers (▲), variable AGN (◆), and flagged artifacts (×) overplotted, and the four quadrants of §4 labeled. The figure must *show*, in one panel, that (1) most sky sits in "catalog sufficient," (2) discarded information concentrates off-diagonal, and (3) the known interesting objects and the known artifacts populate the two high-error cells — and separate from each other along the within-vs-cross-instrument axis (inset or marker style). If that separation is visible, the thesis is carried; if it isn't, the project's central claim has failed and we learn that cheaply.

---

## 7. (f) Head architecture against the JAISP frozen-feature pattern

Follows the existing pattern (small head, bottleneck + BandStem windows, cached features), mirroring the latent-position head's extraction.

### 7.1 Anomaly engine — **no head, inference only**
- Input: frozen v10 foundation (`models/jaisp_foundation_v10.py`, ckpt `jaisp_v10_warmstart/checkpoint_best.pt`) run in LOO masking mode via its existing `JAISPDatasetV10` masking path.
- Output: 10 single-band residual maps + 2 instrument-LOO maps per tile, noise-normalized by the NPZ `var` (MAD fallback on EDF-S, reusing the patches already added in `run_centernet_detections.py` / `astrometry2`).
- Per-source aggregation reuses the CenterNet detection catalog (`data/detection_labels/centernet_v10_790_thresh03.pt`) for footprints.

### 7.2 Sufficiency probes — **small, matched-capacity**
- **Inputs:** per source, R = bottleneck 5×5 window (`[256,5,5]`) + VIS-stem 17×17 window, pooled to a fixed vector (reuse notebook 05 / `precompute_features.py` extraction). C = catalog row (+ engineered nonlinear features).
- **Probe family:** ridge and a 2-layer MLP (the notebook-05 probe), held fixed across {C}, {R}, {C,R}. Add the online-MDL probe wrapper.
- **Targets Y:** held-out per-source band flux and a small profile readout (the classical Gaussian fit gives the comparison label *only* for C; the *truth* for Y is the held-out band pixels). Per band, per SNR bin.
- **Output:** $I_{\mathcal V}(R\to Y\mid C)$ table (band × SNR), with capacity sweep, MDL, selectivity, bootstrap bars.

### 7.3 Optional learned anomaly head (phase 2, only if needed)
A tiny per-source MLP mapping (R, footprint residual statistics) → calibrated anomaly probability, trained on injection labels + recovered known objects. Keep it last — the inference-only score must be shown to work first.

### 7.4 First-experiment scope (smallest convincing slice)
1. **Week 1 — anomaly engine sanity (inference only).** Run LOO masking on ~30 ECDFS tiles. Verify (a) blank-sky $r_b\sim\mathcal N(0,1)$; (b) the existing diffraction-spike masks and known chip edges light up in the *within-instrument* channel; (c) a hand-picked merger/blend lights up in the *cross-instrument* channel. Deliverable: one calibrated residual-map figure. No training, no new data.
2. **Week 2 — sufficiency on g/r/i/z/VIS.** Reuse notebook-05 probes; produce the $I_{\mathcal V}(R\to Y\mid C)$ table with the four controls. Deliverable: "the catalog discards ≥ X bits/source, stable across capacity."
3. **Week 3 — fusion + the figure.** Compute $e_C, \Delta_s, A_s$ on the same sources; make the §6 figure on ECDFS; overplot Kartaltepe mergers + CDF-S variable AGN; report rank correlation + enrichment vs random and vs iForest baseline.
4. **Gate decision:** if the §6 separation holds on ECDFS, scale to EDF-S (Euclid Q1 lenses as truth) and write. If not, the cheap negative result still answers "is the discarded information physical?"

---

## 8. Positioning sentence for the intro (so the contribution is unambiguous)

> Unlike cross-survey image translation (AS-Bridge, `2603.11928`) or signal/instrument latent disentanglement (Mercader-Perez et al., `2604.09787`), we use leave-one-band-out reconstruction *within a single jointly-calibrated 10-band Rubin+Euclid model* and, for the first time in astronomy, cast its residual as a *capacity-controlled measurement of the predictive information the classical catalog discards* — then show that this discarded information is concentrated on the objects the catalog gets wrong.

---

## 9. Open risks to decide before building
- **Novelty margin is now narrow** (two 2026 scoops on the anomaly half). The defensible core is sufficiency + fusion + the LOO-within-one-model mechanism. If a referee reads "less-lossy catalog" as merely relabeling "pixels beat catalog features" (DeepDISC etc.), the *information-budget measurement* and *catalog-as-baseline* must be the rebuttal — they are what no one reports as a headline.
- **Single field limits discovery statistics** — injection-recovery (§5.2) is the load-bearing validation, not the handful of real lenses.
- **The held-out-band-as-truth target couples "anomaly" and "noise"** — the $\chi^2$/blank-sky calibration (§3.3, §5.4) is what separates them and must be airtight.
