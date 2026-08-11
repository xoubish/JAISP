# Fisher information in JAISP — summary of the nb25–nb28 investigation (2026-08-11)

**Question.** How much information about a source's position survives at each stage of the
pipeline (pixels → stems → bottleneck → head), and is the unused part worth collecting?

## The three rules (learned the hard way; full treatment in nb25)

1. **Fisher is conditional on the data model — walk the ladder.** The same question gave
   ×5.5 (idealized), ×2.2 (real SEDs), and ~nothing-bright / ~10%-faint (real sky). None
   of these is wrong; each answers a progressively more honest question. Never treat a
   rung's number as a target until the noise model includes the systematics.
2. **Use the analytic pixel CRB as ceiling and falsifier, not Monte Carlo.** It is exact
   (a pixel sum over the injected profile and the variance maps) and bounds everything
   downstream. Do **not** estimate Fisher of learned representations from realizations
   (`J^T C^-1 J`): at any realistic budget it fabricates information (Jacobian noise ×
   effective rank of the window covariance) — our version reported 8–17 mas of
   "information" from a *zero-signal* null. No latent-Fisher number is admissible without
   that null.
3. **Measure representations by decoding.** Train a simple reader on injection
   realizations and score held-out error: a valid, assumption-free lower bound.
   Slope-calibrate first — regularized readers shrink toward the training prior, and the
   bias is shared across bands, poisoning any combination.

## The ladder (per-axis, peak S/N 10 unless noted)

| rung | data model | multi-band gain over VIS | source |
|---|---|---|---|
| 1 idealized | equal peak S/N, WCS = truth | 7.98 → 3.40 mas (×5.5 info); reachable by a joint 10-band fit (3.42) | nb25 |
| 2 real SEDs | measured per-band amplitudes (1,710 sources) | ×1.47 in σ; carried by VIS (46%) + Rubin gri (32%) | nb26 |
| 3 real sky | + inter-band error correlations | bright: ~0 (ρ̄ = 0.4–0.9; the concordance field, paper §5; six Rubin bands share one coadd WCS → ~one vote); faint: ~10% via IV pooling | nb25 §5, nb27 |

## Key measured facts

- The production head **exactly saturates the VIS pixel CRB** (7.7 ≈ 8.0 mas) — optimal
  for its inputs; its gap to the multi-band bound is architectural (it reads only the
  fused bottleneck + VIS stem). Its predicted σ is overconfident ×1.7 at the bright end.
- Per-band **stems are near-lossless** (decoder bound: 80–85% of each band's pixel
  information); ten calibrated linear readers combine to 3.96 mas vs the 3.40 ceiling.
- On injections, per-band errors are independent (|ρ| ≈ 0.01) and informations add
  exactly; on real sky the concordance field appears as large inter-band correlations
  that cap pooling at the bright end.
- At S/N 5 even the matched joint fit misses the CRB by ~2× (ML is asymptotic): CRBs are
  not faint-end targets.

## Decisions (nb28: 6 tiles × 20 sources, realistic SEDs, truth-based, bootstrap CIs)

- **Router v2 — adopt.** Calibrated inverse-variance pooling of per-band positions beats
  the median pool by +8–12% at every S/N (all CIs exclude zero) and beats the current
  head everywhere (9.7 vs 13.9 mas at S/N 10). Catalog-level arithmetic, no retraining.
- **Head v2 — proceed only under the revised pitch.** The head-to-joint-fit gap is
  +38–50%, but classical IV pooling collects ~half of it with no network, and the
  classical baseline is flattered by a matched PSF. A learned multi-band head is
  justified as an *amortized joint fit* (no per-source optimizer, no explicit per-band
  ePSF) that stays robust on blends and real morphology — benchmarked against classical
  multi-band alternatives, not against the current head. Design: shared per-band stem
  reader (VIS + gri priority) → per-band (dx, dy, log σ) votes → calibrated IV
  combination; trained on injection truth (real labels carry a 5 mas floor).
- Bright-end improvement is calibration territory, not head territory: the paper's
  concordance field already maps and (within-footprint) removes the shared systematic;
  the longer-term remedy is native-sampling / single-exposure inputs (paper §7).

## Notebook map

- `io/25_fisher_information_astrometry.ipynb` — the field guide: rules, ceilings,
  estimators-vs-bounds, the null-test demo, the decoder bound, the real-sky matrix.
- `io/26_multiband_head_feasibility.ipynb` — head-v2 gates: real-SED census + ten-stem
  decoder bound.
- `io/27_router_v2_crossband_pooling.ipynb` — real-sky pooling on the anchors archive vs
  Gaia; the independence matrix; faint-end consistency.
- `io/28_v2_decision_plots.ipynb` — the truth-based decision test (records cached in
  `_nb28_outputs/v2_decision_records.json`; re-runs load the cache).

Same machinery applies to the next heads by changing only the injection delta:
photometry (θ = flux), shape (θ = e₁, e₂).
