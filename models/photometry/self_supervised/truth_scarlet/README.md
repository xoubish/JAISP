# Truth-supervised amortised scarlet (A0–A5)

This implements the plan in [../../architecture_brainstorm/README.md](../../architecture_brainstorm/README.md). Each head is trained on injected scenes where every injected source's true flux and profile are known. Fluxes are never predicted directly; every variant ends in the same signed linear solve over all sources in the scene, with a fixed robust background.

| Variant | Head | Question |
|---|---|---|
| A0 | existing `fixedbg/epoch1.pt`, χ²-trained (evaluated only) | baseline |
| A1 | same head, truth loss | is the training target the bottleneck? |
| A2 | A1 + per-band branch (band pixels, residual and PSF on the morphology grid) | does band-specific input help? |
| A3 | A2 with foundation inputs zeroed (encoder not run) | any foundation-specific gain? |
| A4 | A2 with no PSF anywhere: observed-space templates, learned band widths and wing fraction | can the head infer the PSF? |
| A5 | A2 with freer shapes: the CNN modulation of the monotone profile may reach ~7× (A2: 1.65×) | do arms/clumps need more freedom? |

## Data versions

All versions use the same tile split (`split.json`: 40 training tiles, 8 validation tiles, chosen from the 48 `amortised_scarlet/training_tiles`) and never touch the frozen tests. Validation sets are kept across versions so runs stay comparable.

| Version | Module / stage | Donors | Scenes | What it fixed |
|---|---|---|---|---|
| v1 (`train`, `val`) | `scenes.py` | bright isolated galaxies, VIS S/N > 25, no stars | 1–3 injected sources on blank sky | — |
| v2 (`train_v2`, `val_v2`) | `scenes_v2.py` | S/N > 5, point-like included | 75% injected into real tile scenes (real detections fitted alongside, no truth), 25% blank sky | crowding, undetected light, compact sources |
| v3 (`train_v3`, `val_v3`) | `scenes_v3.py` | nonparametric (aperture S/N ≥ 8, footprint-limited reconstruction, 3″ disk for S/N > 30) + Sérsic from MER structure (S/N 3–8); every source near its own brightness | as v2 | the size–brightness relation (v2 training galaxies were ~0.4″ at every brightness; real faint sources are compact) |
| v3b (`train_v3b`) | `scenes_v3.py export_b` | v3 donors | v3 + random intrinsic rotation/flip (PSF unchanged), 30% bright extended primaries | morphological variety |
| v4 (`train_v4`) | `scenes_v3.py library4/export4` | v3 recipe over 105 tiles (40 + 65 in `training_tiles_2`, regions numbered 1000+) | v3b recipe, real scenes from all 105 tiles, 20k scenes | 2.6× more galaxies |

`plot_scenes.py --split <folder>` writes galleries (one page per scene type) and a distributions page under `<folder>/gallery/`. `scenes_v3.py sizecheck[4]` prints the donors' half-light radius by magnitude next to MER's Sérsic radius for real detections; it should follow the real trend before any training.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
M=models.photometry.self_supervised.truth_scarlet
python -m $M.scenes split; python -m $M.scenes mer                       # once
python -m $M.scenes library --split train; ...; python -m $M.scenes real; python -m $M.scenes reference   # v1 + val_real (see scenes.py)
python -m $M.scenes_v2 library --split train; python -m $M.scenes_v2 export --split train --count 16000; (val likewise); python -m $M.scenes_v2 reference
python -m $M.scenes_v3 library --split train; python -m $M.scenes_v3 sizecheck --split train; python -m $M.scenes_v3 export --split train --count 16000; (val likewise); python -m $M.scenes_v3 reference
python -m $M.scenes_v3 export_b --split train --count 16000               # v3b
bash $M/run_tiles2.sh                                                     # 65 extra tiles: select, fetch, prepare, MER (hours)
python -m $M.scenes_v3 library4; python -m $M.scenes_v3 sizecheck4; python -m $M.scenes_v3 export4 --count 20000   # v4
```

`run_v2.sh`, `run_v3.sh`, `run_v4.sh` chain the data stages and launch the fine-tuning runs in tmux.

## Training

```bash
T=models.photometry.self_supervised.truth_scarlet.train
python -m $T --variant A2 --device cuda:0                                                  # from scratch on v1 (20 epochs)
python -m $T --variant A2 --name A2_v4 --init runs/truth_scarlet/runs/A2_v3/best.pt \
    --train-dirs train_v4 --val-dir val_v3 --extra-val val_v2 --epochs 5 --lr 1e-4 \
    --lambda-frac 1 --bias-mode ema --lambda-bias 1 --steps 3                             # fine-tune with the v4 loss
python -m $T ... --resume auto                                                             # continue after an interruption (same W&B run)
```

Loss per batch: Huber((f − f*)/σ_oracle) + per-source model-image error + 0.1 · mean log χ²/dof + bias term, plus optional:
- `--lambda-frac`: Huber(frac_scale · (f − f*)/f*) for sources with true S/N > `--frac-snr`. The χ term saturates for bright sources (a 10% error at S/N 100 is deep in the linear regime); this term makes bright-end fractional error count as it does in the real-tile metric.
- `--bias-mode ema`: a running mean of the clipped signed error per (S/N bin × band), used as a dual-ascent push (penalty = Σ ema · batch mean) toward zero bias. The clipped mean tracks the median; the plain mean is dominated by the positive tail from unmodelled neighbours at low S/N. The tracker is saved in checkpoints.
- `--steps`: unrolled refinement steps (the refinement net is shared, so a checkpoint trained with 2 can be fine-tuned with 3).

Defaults from scratch: 20 epochs, batch 8, lr 3e-4 with warm-up and cosine decay to 5%. Validation every 8000 training scenes on the `--val-dir` set (selection metric `val/median_abs_chi`), on each `--extra-val` set (`extra_<name>/*`) and on the 504 real validation-tile scenes against MER (`real/*`). Checkpoints, plots and W&B files go to `runs/truth_scarlet/runs/<name>/`; `best.pt` is chosen on `val/median_abs_chi` only. Speed on an L40S: 4–5 scenes/s for v1 scenes, 2.3–3.9 scenes/s for real-scene injections (two runs per GPU).

## What W&B shows (project `jaisp-photometry`, group `truth_scarlet`)

- `train/*`: loss terms (flux, template, χ², bias, frac), gradient norm, learning rate, throughput, `bias_ema_*`.
- `val/median_abs_chi`: mean over bands of the median |f − f*|/σ_oracle; `mixture_val/median_abs_chi` is the flat reference. `val_snr/*`, `val_context/*`, `val_band/*` split it by true S/N, separation (isolated > 2″, blend, close < 0.7″) and band.
- `real/*`: NMAD and median offset (mag) against MER for VIS/Y/J/H on the real validation tiles; `mixture_real/mean_nmad_mag` is the reference (0.133).
- `plots/*`: flux scatter, fractional error vs S/N, real vs MER, and a residual gallery; PNGs also under `plots/`.

## Results so far (best checkpoints; injections = `val_v3`, real = 504 validation-tile sources vs MER)

| | val_v3 median \|Δ\|/σ | close blends | real NMAD (mag) | real VIS NMAD |
|---|---|---|---|---|
| mixture (`q1_mixture_calibrated`) | 2.16 | 2.18 | 0.133 | 0.121 |
| A0 (χ² only) | — | — | 0.186 | 0.219 |
| A2 v1 / v2 / v3 | — / 2.37 / 2.06 | — / — / 1.99 | 0.207 / 0.183 / 0.156 | 0.255 / — / 0.153 |
| A3 v3 (no foundation) | 2.12 | 2.04 | 0.154 | 0.167 |
| A5 v3b (free shapes) | 2.09 | 2.08 | 0.151 | 0.142 |
| A2 v4 (v4 loss: frac term, EMA bias, 3 steps) | 2.05 | 1.93 | 0.152 | 0.141 |
| A2 v4 old loss (data-only control) | 2.05 | 1.96 | 0.158 | 0.150 |
| A3 v4 (no foundation) | 2.11 | 2.03 | 0.154 | 0.169 |
| A5 v4 | 2.02 | 1.93 | 0.146 | 0.133 |
| A5 v4j (A5_v4 + position jitter in training) | 2.05 | 2.01 | 0.145 | 0.124 |
| **A6 v4** (A5_v4 + jitter + learned centroid correction) | **1.98** | **1.88** | **0.140** | **0.116** |
| A7 v4 (A6_v4 + learned point-source template; last.pt, best never improved on init) | 1.98 | 1.96 | 0.139 | 0.113 |
| A8 v4 (A6_v4 + S/N-weighted image term, `--template-snr 10 --template-wmax 10`) | 1.97 | 1.90 | 0.144 | 0.120 |

**Four-way scatter decomposition on the validation tiles** (one head + mixture + Tractor + MER; six pairwise NMADs solved for each method's own scatter, all Euclid bands). With A5_v4: mixture 0.077, head 0.089, Tractor 0.149, MER 0.105. With A6_v4: mixture 0.081, head 0.081, Tractor 0.149, MER 0.102 (per band A6 vs mixture: VIS 0.091/0.092, Y 0.076/0.090, J 0.078/0.065, H 0.075/0.077; MER-clean bright sources 0.057/0.073). MER's own noise is the largest term in any head-vs-MER number, so "NMAD vs MER" understates how close the heads are. Do not put two heads in one decomposition (they share errors and both come out too low). On real tiles A6 is the best deblender of the three (close pairs 0.173 vs 0.177 vs 0.266 vs MER) and trails the mixture only on the largest galaxies (0.132 vs 0.097) and in J. Position jitter by the real centroid-error distribution (`val_v3_jit`) degrades A5 and the mixture equally overall (+8%) but A5 twice as much for offsets > 0.15"; A6's centroid correction removes that (0.154 vs mixture 0.145 vs A5 0.242 at > 0.15") and A6 beats the mixture on the jittered set by 9%. Scripts: scratch `hat_analysis.py`, `jitter_scenes.py`, `val_diag.py`, `real_diag.py` (session notes). Note `best.pt` is selected on exact-position injections; for jitter-trained heads whose best is the initial checkpoint (A5_v4j) use `last.pt`.

Diagnoses behind each version are in the session notes and `architecture_brainstorm/`. In short: the truth loss fixed light trading in blends (A1 vs A0); real-scene injections fixed crowding (v2); the size–brightness relation fixed the compact/extended flux bias (v3). Remaining gap to the mixture on real data is mostly bright extended galaxies (an amortisation limit, visible on injections too) and a faint-end underestimate in Euclid bands that the χ loss is blind to; the v4 loss targets both.

## Per-scene optimisation (`tto.py`)

`tto.optimise(model, scene, steps=60, lr=.02, snr_min=10, prior=1)` refines the head's shape tensors for one scene (profile/ellipse/mix vector, modulation map, centroid, per-band terms; point weight for A7) by Adam against the scene objective mean_b log(chi2_b/dof_b), flux solve inside the loop, same bounds as the head. It removes the core residuals of bright galaxies (first report example: VIS core chi2/dof 74 -> 1.5, r 552 -> 4; integrated flux unchanged). **Ungated it destroys faint/blended photometry** on known truth (300 val_v3 scenes: overall 2.02 -> 5.12 after 60 steps, close blends 2.11 -> 6.79, faint bias -12% -> +28%): the free modulation absorbs noise and neighbours' light, the same failure as chi-square-only training. Gating (`snr_min=10`, `prior=1`) does not rescue it: close blends 2.11 -> 3.04, faint neighbours -12% -> -28% (bright sources fitted to their pixels absorb their neighbours' light). Do not use it for fluxes; it remains a diagnostic of what the amortised shapes miss. The training-side answer was `--template-snr/--template-wmax` (per-source image term up-weighted for bright sources) with more trained passes (run A8_v4). **Outcome (CPU evaluation 2026-10-19, section 6b of the report notebook):** A8_v4 halves the bright-galaxy core residuals (VIS core chi2/dof 74 -> 38, r 552 -> 152 on the brightest example) and on injections improves the bright end (S/N > 30: 4.44 -> 4.31) and the faint bias (-13% -> -10%) with blends unchanged, but on the real tiles it is not better than A6 (4-band NMAD vs MER 0.139 vs 0.136, own scatter 0.081 = 0.081, close pairs 0.193 vs 0.174). A7_v4 (point template) leaves the star residuals unchanged (VIS core ~110 for every head): the limit for point sources is the PSF model. A6_v4 remains the reference head. Per-head CSVs `diag_{val_v3,real,core}_<tag>.csv` and `core_examples_<tag>.npz` come from the scratch `eval_head.py` (session notes).

## Report notebook

`io/30_a6_v4_photometry_head.ipynb` documents A6_v4 end to end (architecture, data, training curves, injections with known truth, real held-out tiles vs MER / mixture / Tractor / aperture, example cutouts, open issues). Its figures and tables come from `report_a6.py`; `python -m models.photometry.self_supervised.truth_scarlet.report_a6 precompute` regenerates the aperture photometry (`diag_real_aperture.csv`) and example cutouts (`report_examples.*`) it needs, and the per-source measurement CSVs (`diag_real_<run>.csv`, `diag_val_v3[_jit]_<run>.csv`) come from the scratch `real_diag.py` / `val_diag.py` scripts (session notes). Headline: bright isolated galaxies agree with MER to 0.049 mag NMAD over VIS/Y/J/H (mixture 0.060, Tractor 0.059). Shape gallery (`report_shapes.npz`, scratch `shape_gallery.py`): data, PSF-convolved model and intrinsic shape for 100 held-out sources. Refinement passes at inference (A6_v4, 400 val_v3 scenes): 1/2/3/4/5/6 passes give 2.57/2.45/2.03/2.28/2.12/2.38 overall; only trained-for passes help, so train with more passes rather than running more.

## Rules

- Do not train or tune on the frozen tests: the 128-blend benchmark, `empirical_injection_pilot_v2`, and the 28 detection-catalog tiles. Run those once per selected checkpoint at the end.
- σ_oracle is the error of a fit with the true templates and a free constant background. In real scenes even that fit is biased at S/N < 3 (undetected light couples into the background); compare heads with the oracle and the mixture on the same scenes rather than against zero.
- MER is a reference, not truth; Tractor and the mixture agree with it at 0.12–0.13 mag, so a head that disagrees by more has a real problem.
- No shot noise is added. Weak bands (donor band S/N < 8) use VIS-shape truth and are flagged in `truth.csv`.
