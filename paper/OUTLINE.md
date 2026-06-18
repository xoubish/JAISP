# JAISP paper — working outline (target: ApJ)

**Working title:** Cross-Instrument Source Detection and Astrometry with a Joint Rubin–Euclid Foundation Model

**Framing (from the venue discussion):** the defensible novel claim is *cross-instrument localization transfer* — measuring positions in bands where a source is too faint to localize alone, by fusing the sharp Euclid VIS constraint. The foundation enables the multi-instrument detections; the astrometry gain is real but VIS-floor-bounded. We lead with the capability and the honest ablations, not an inflated "beats-SOTA" claim.

**Class:** AASTeX v6.3.1 (`aastex631.cls`). Not vendored — download from AAS and drop into `paper/`, or `tlmgr install aastex`. Build: `latexmk -pdf main.tex`.

## File map
- `main.tex` — preamble, author block, `\input`s
- `sections/abstract.tex`, `intro.tex`, `data.tex`, `methods.tex`, `results.tex`, `ood.tex`, `discussion.tex`, `summary.tex`
- `refs.bib` — bibliography (stub)
- `figures/` — empty; populate from `docs/figures/` and `io/_nb_ood_outputs/`

## Status of each section (✅ have it / 🟡 partial / ⛔ blocked on data)

| Section | In hand | Needs |
|---|---|---|
| Abstract | 🟡 numbers drafted | final OOD Rubin result; one-line takeaway |
| Intro | 🟡 structure | **prior-work positioning** (AstroCLIP, Multimodal Universe, AstroPT) — defines novelty boundary |
| Data | 🟡 | exact ECDFS area/depth/epoch; Rubin pixel/tile spec; ⛔ real-var EDF-S state; Gaia DR; MER catalog |
| Methods | 🟡 | foundation architecture detail; MAE training config; detection head + self-training; (all derivable from `DOCUMENTATION.md` + code) |
| Results — detection | 🟡 | ✅ counts, bright-star corr; ⛔ completeness/purity vs MER |
| Results — astrometry | ✅ mostly | final per-band table + 11 mas pooled; figure |
| Results — injection | ✅ | confirm numbers, table, caveats (already in `io/_nb09_outputs/`) |
| Results — Gaia | ✅ | DR, N, mag-binned table (in `gaia_absolute_check.txt`) |
| OOD | 🟡 | ⛔ **real-var rerun** settles Rubin g/r/i/z; ⛔ MER completeness |
| Discussion | ✅ structure | quantify foundation-equivalence; centroid-emulation summary |
| Summary | 🟡 | final OOD statement; closing |

## Critical path (what unblocks the paper)
1. ⛔ **Real-var EDF-S tiles** → rerun `io/15`; resolves the single biggest open result (Rubin optical transfer). Either "fully transfers" (strong paper) or "real field difference" (still publishable, more nuanced).
2. ⛔ **MER Euclid catalog** → detection completeness/purity (ECDFS + EDF-S), and whether bright-star suppression loses real sources. Turns Results-detection from a count into a benchmark.
3. ✅ Already-solid pieces needing only write-up: injection truth test, Gaia absolute, astrometry SNR scaling, foundation-equivalence ablation, bright-star mechanism.

## Sources to pull text/numbers from (no new compute)
- `DOCUMENTATION.md` "Current Status" — checkpoints, configs, all headline numbers
- `io/_nb09_outputs/` — injection (`injection_truth_results.json`), joint-centroid, Gaia (`gaia_absolute_check.txt`), gate verdict
- `io/15_ood_evaluation_edfs.ipynb` — OOD figures + findings cell
- `docs/figures/astrometry_truth_vs_snr.png`

## Figures to produce
- F1: astrometry residual vs SNR (per-band + pooled), injection-truth overlay
- F2: OOD per-band + SNR-binned residual, EDF-S vs ECDFS (from `io/15`)
- F3 (maybe): detection overlay + bright-star completeness vs halo-fraction
- F4 (maybe): architecture schematic

## Open decisions
- Author list / collaboration policy (Rubin/Euclid data rights, builder lists).
- How much architecture detail vs. citing a separate methods/instrument note.
- Whether the learned position head appears as an ablation or is omitted.
