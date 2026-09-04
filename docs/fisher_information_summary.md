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

- The production head **sits on the VIS pixel CRB** — optimal for its inputs; its gap to
  the multi-band bound is architectural (it is tied to the VIS convention and does not
  read the other bands' stems). v11+anchored: 8.37 vs ceiling 7.98 mas, pulled up from the
  bare anchor's 9.02 by the bounded network residual (v10-era head: 7.69, mildly
  shrinkage-aided). σ_pred is overconfident ×3.1 at S/N 10 on injections (photon-noise
  regime; nb32's flat ×0.79 applies on real sky where systematics dominate).
- Per-band **stems are near-lossless** (decoder bound: 80–85% of each band's pixel
  information); ten calibrated linear readers combine to 3.96 mas vs the 3.40 ceiling.
- On injections, per-band errors are independent (|ρ| ≈ 0.01) and informations add
  exactly; on real sky the concordance field appears as large inter-band correlations
  that cap pooling at the bright end.
- ~~At S/N 5 even the matched joint fit misses the CRB by ~2×~~ — **did not reproduce**
  in the v11 sweep (2026-09-02, `fisher_bench_vs_snr.json`): the joint fit stays on the
  all-band bound even at S/N 5 (6.85 vs 6.80 mas); classical lags its bound ~7% there.
  The measured S/N sweep (5/10/30/100, injection truth): the anchored head tracks the VIS
  CRB within ±10% across the full range (15.50/8.37/2.91/0.88 vs 15.96/7.98/2.66/0.80),
  the joint fit tracks the all-band curve throughout — figure
  `_nb25_outputs/fisher_bench_vs_snr.png`.

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

## v11 re-base (2026-09-02) — nb25 is now v11-native

The production stack changed after the investigation above: foundation v11
(`jaisp_v11_q1_soft`, soft compression instead of the bright-core clamp) and the
ANCHORED head (band-aware head + in-model VIS windowed-centroid anchor + residual
tanh-bounded at 3× the anchor's noise), adopted 2026-09-01. nb25 was first extended
with a §7 side-by-side (v10 vs v11, table below), then **cleared to v11 end-to-end**:
the whole field guide now loads `jaisp_v11_q1_soft` + the patch-disjoint anchored head,
§2 carries an anchor-alone arm, §3/§4 re-verify on the v11 stems (null test still
fabricates: 7.9–15.3 mas from zero shift; decoder bound VIS 9.46 / rubin_r 18.66 vs
ceilings 7.98 / 15.96 — stems equally near-lossless in v11), and rung 3 now uses the
anchored dedup anchors archive with the nb32 Q1 epoch 2024.34 (ρ̄ = 0.43, Rubin block
0.92 — reproduces, since the classical arm is head-independent). The v10-era notebook
and numbers live in git history; the transition bench (identical noise streams):

| estimator | per-axis σ [mas] | \|mean bias\| [mas] |
|---|---|---|
| all-band pixel CRB | 3.40 | — |
| joint 10-band fit | 3.42 | — |
| v11 plain head (caveat: in-train tile) | 7.57 | 8.1 |
| **v11 anchored head (adopted)** | **8.37** | 11.1 |
| anchor alone (no network) | 9.02 | 14.7 |
| v10 head | 7.69 | 8.8 |
| classical VIS centroid | 8.06 | 9.4 |
| VIS pixel CRB | 7.98 | — |

- **The conclusion above survives unchanged**: the adopted head sits at the VIS
  pixel CRB, not below it. v11+anchored moved the head *along* the VIS rung (bias,
  robustness, worsener population) rather than *up* the ladder (information). The
  ×1.47 realistic multi-band headroom is still uncollected and the head-v2 /
  amortized-joint-fit pitch remains open.
- The anchor alone costs scatter and scene-pull bias (9.0 / 14.7 mas; a 17 px window
  eats undetected background flux); the bounded network residual buys part of both
  back (8.4 / 11.1). The few-percent scatter concession vs the plain head is the
  price of convention anchoring; the real-sky ledger (nb31: anchored faint 8.5 vs
  plain 19.7 mas) shows what it buys. Plain's bench advantage corroborates nb31:
  its real-sky faint failure is mis-centering on real morphology, not statistical
  inefficiency.
- The bias column is common-mode scene contamination (all estimators, classical
  included, 8–15 mas): read the differences, not the level.
- σ_pred: anchored is ×3.1 overconfident on S/N 10 injections (photon-noise regime)
  even though nb32's real-sky factor is a flat ×0.79 (systematics-floor regime).
  Keep the nb32 global rescale for catalogs; σ_pred does not track photon noise at
  the faint end.

Numbers: `_nb25_outputs/fisher_field_guide_summary.json`; transition figure:
`_nb25_outputs/fisher_v11_anchored_bench.png`. The talk/paper-ready summary is
`_nb25_outputs/fisher_ladder_story.png` (nb25 §6 "story" cell): top panel = the real-sky
~×7 win over per-band classical (nb18), bottom panel = the S/N-10 bench against the VIS
and all-band ceilings with the forbidden region shaded — improvement and limit in one
frame, per-axis σ converted to median total offset (×√(2 ln 2)) so the panels share units.
Companion figure `_nb25_outputs/fisher_crb_vs_snr_realsky.png`: the CRB as a 1/S/N curve
over the real-sky per-band hexbins (nb31/nb18 archive) — the head runs flat at ~8 mas and
crosses *below* each query band's own pixel CRB at S/N ≲ 20 (legitimate: it imports VIS
information), while in-band classical sits ~4–5× above its bound; bright-end head–label
agreement reaches 3–4 mas but absolute accuracy floors at ~7 mas (Gaia tie, nb32).
