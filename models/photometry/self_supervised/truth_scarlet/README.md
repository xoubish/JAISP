# Truth-supervised amortised scarlet (A0–A4)

This implements the plan in [../../architecture_brainstorm/README.md](../../architecture_brainstorm/README.md). Each head is trained on injected scenes where every source's true flux is known: real galaxies from the training tiles (donors), dimmed and blended onto real blank sky from the same tiles. Fluxes are never predicted directly; every variant ends in the same signed linear solve with a fixed robust background.

| Variant | Head | Question |
|---|---|---|
| A0 | existing `fixedbg/epoch1.pt`, χ²-trained (evaluated only) | baseline |
| A1 | same head, truth loss | is the training target the bottleneck? |
| A2 | A1 + per-band branch (band pixels, residual and PSF on the morphology grid) | does band-specific input help? |
| A3 | A2 with foundation inputs zeroed (encoder not run) | any foundation-specific gain? |
| A4 | A2 with no PSF anywhere: observed-space templates, learned band widths and wing fraction | can the head infer the PSF? |

## Data (CPU; run once, in order, from the project root)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
M=models.photometry.self_supervised.truth_scarlet.scenes
python -m $M split                                  # 40 train / 8 validation tiles (whole tiles)
python -m $M mer                                    # MER query + match for the 48 training tiles (IRSA, a few minutes)
python -m $M library --split train                  # donors, VIS S/N > 25, isolated (serial, ~15 min)
python -m $M library --split val
python -m $M backgrounds --split train --per-region 30
python -m $M backgrounds --split val --per-region 30
python -m $M export --split train --count 16000 --workers 40   # ~30 min, ~20 GB
python -m $M export --split val --count 800 --workers 40
python -m $M real --real-limit 500                  # real validation-tile scenes + MER references
python -m $M reference --val-limit 800 --workers 40 # mixture photometer on both validation sets (~10 min)
```

Outputs are written to `runs/truth_scarlet/{split.json,train,val,val_real}`. `export` resumes if interrupted. Each scene draws its configuration independently:
- 35% isolated, 55% pairs, 10% triples;
- VIS S/N log-uniform from 1.5 to 80;
- separations 0.3–2″, neighbour flux ratios 0.1–10, random rotation;
- extra PSF blur of 0–0.05″ in Euclid and 0–0.2″ in Rubin. This gives A4 a range of PSFs to learn from. A1–A3 receive the true blurred PSF.

## Training (GPU; one command per run)

```bash
T=models.photometry.self_supervised.truth_scarlet.train
python -m $T --variant A0 --eval-only --init models/photometry/self_supervised/runs/amortised_scarlet/fixedbg/epoch1.pt --device cuda:0
python -m $T --variant A1 --device cuda:0
python -m $T --variant A2 --device cuda:0
python -m $T --variant A3 --device cuda:1
python -m $T --variant A4 --device cuda:1
```

Defaults: 20 epochs over 16k scenes, batch 8, lr 3e-4 with 100-step warm-up and cosine decay to 5% of the peak. Loss weights: λ_flux 1, λ_template 1, λ_χ² 0.1, λ_bias 0.5. Validation runs every 8000 training scenes (40 points per run). Measured on an idle L40S: about 4–5 training scenes/s and 6.5 GB peak per run, so one epoch is roughly an hour and a 20-epoch run about a day; several runs fit on one GPU but slow each other. Checkpoints, plots and the W&B directory go to `runs/truth_scarlet/runs/<variant>/`. Use `--name` for repeat runs with different settings.

## What W&B shows (project `jaisp-photometry`, group `truth_scarlet`)

- `train/*`: total loss and each term (flux, template, χ², bias), gradient norm, learning rate, skipped scenes, throughput.
- `val/median_abs_chi`: the selection metric. It is the mean over bands of the median |f − f*| / σ_oracle on the injected validation scenes. `mixture_val/median_abs_chi` is logged on the same steps as a flat reference line.
- `val_snr/*`, `val_context/*`, `val_band/*`: the same metric and the median fractional bias split by true S/N, by isolated vs. blend separation, and by band.
- `real/*`: NMAD and median offset in magnitudes against MER, on central sources in the 8 validation tiles (VIS, Y, J, H). `mixture_real/mean_nmad_mag` is the reference.
- `plots/flux_scatter`: measured vs. true flux (both divided by σ_oracle) per band, coloured by context, with mixture points in grey.
- `plots/frac_vs_snr`: running 16/50/84% fractional error vs. S/N, this model vs. the mixture.
- `plots/real_vs_mer`: this − MER magnitude vs. MER magnitude, with the mixture in grey.
- `plots/residuals`: a fixed gallery of four validation scenes (faint isolated, bright isolated, close pair, triple). It shows VIS data and model, χ residuals in VIS, r and H, and f/f* per source.

The same PNGs are saved under `plots/`. `best.pt` is chosen on `val/median_abs_chi` only. χ² is logged but never used for selection.

## Rules

- Do not train or tune on the frozen tests: the 128-blend benchmark, `empirical_injection_pilot_v2`, and the 28 detection-catalog tiles. Run those once per selected checkpoint at the end.
- σ_oracle is the error of a fit with the true templates. The loss uses it instead of the model's own conditional error, so a head cannot lower its loss by widening its templates.
- No shot noise is added, the same as in the pilot. Weak bands (donor band S/N < 8) use VIS-shape truth and are flagged in `truth.csv`.
