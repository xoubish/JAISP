# Architecture brainstorm: truth-supervised amortised-scarlet head

These are design figures to discuss before any run. Nothing here is trained or benchmarked yet. To regenerate them:

```bash
python -m models.photometry.architecture_brainstorm.make_figures
```

## Motivation

The amortised-scarlet head ([../self_supervised/amortised_scarlet.py](../self_supervised/amortised_scarlet.py)) already fits every detection in a scene jointly. It uses frozen foundation features and is trained only on residual χ². Its χ² kept improving while its injection flux accuracy got worse, and on real tiles it lost to the mixture photometer. The likely cause is the training target, not the network: a residual of zero does not determine how flux is split between blended sources or how much faint flux there is ([Fig. 2](figures/fig2_chi2_degeneracy.png)). The plan is to keep the scene-level renderer and linear flux solve, and to train on scenes where each source's true flux is known.

## Figures

| | |
|---|---|
| [fig1_architecture](figures/fig1_architecture.png) | Existing shared-shape trunk on the frozen foundation features, plus a **new per-band branch**. It reads band-b native pixels, the PSF and the current residual, and outputs bounded per-band shape corrections. Fluxes still come from the signed linear solve with a fixed background. |
| [fig2_chi2_degeneracy](figures/fig2_chi2_degeneracy.png) | 1-D toy example. (b) Free-form shapes reproduce a 0.7″ pair exactly with fluxes 1.37 / 0.63 instead of 1 / 1. (c) Even with monotone profiles and small centroid errors, Δχ² < 1 allows source-A flux errors of about ±25% at scene S/N 10, ±7% at S/N 30 and ±2% at S/N 100. |
| [fig3_training_and_loss](figures/fig3_training_and_loss.png) | Training scenes: (A) bright donors dimmed onto real sky; (B) pairs of real galaxies blended on real sky. Loss terms: flux error in noise units, per-source model image, the existing log χ² term, and an optional bias penalty. The images are schematic, not real cutouts. |
| [fig4_protocol](figures/fig4_protocol.png) | Donors and sky come only from the 43 training tiles; whole tiles are held out for validation. Checkpoints are selected on validation injection metrics, never on χ². The three existing benchmarks stay frozen tests. Ablations A0–A3. |

## Decisions to make before a run

1. **Flux reference F_ref for donors.** Options are the per-band reconstruction amplitude (`empirical.reconstruct`), the mixture photometer, or a curve of growth. Any error in F_ref is common to the bright and dimmed copies, so these scenes test faint-end bias, not absolute calibration.
2. **Donor rendering.** Use (a) the noise-free reconstruction re-rendered with the PSF, which gives exact templates but approximate morphology, or (b) the raw bright stamp scaled by α, which keeps true morphology but carries scaled noise and possibly neighbours. I'd start with (a), since `empirical.py` already does it.
3. **Per-band branch capacity.** How large δε_b may be (its bound), and whether the branch sees only its own band or also neighbouring bands. Too much freedom brings back light trading within a band.
4. **Loss weights λ.** Start with flux and template terms dominant and χ² small. Decide whether the batch bias term λ_β is needed or whether Huber is enough.
5. **Scene mix.** Fraction of isolated vs blended scenes; the range of α, separation and flux ratio; whether to include triples and stars (PSF-only flag).
6. **Order of ablations.** Run A1 first (current architecture + new loss). If it does not beat A0 and approach the mixture on the validation injections, the per-band branch (A2) is unlikely to rescue it.

## Constraints carried over

- Keep monotone, centred profiles and the fixed robust background. The amortised-scarlet runs showed free-form shapes and a free background both bias fluxes.
- Never train on or select with `empirical_injection_pilot_v2`, the 128-blend benchmark, or the 28 detcat tiles.
- Report the foundation-vs-raw-pixel difference (A3) with paired intervals. Past tests found no clear foundation-specific gain.

## Implementation

The runs are implemented in [../self_supervised/truth_scarlet/](../self_supervised/truth_scarlet/README.md): scene generation, the A0–A4 heads, and the W&B trainer.
