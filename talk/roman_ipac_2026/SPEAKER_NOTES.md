# JAISP talk — Roman Science Conference, Caltech/IPAC, July 13–16 2026

**Slot:** 15 minutes. Plan for **~12:30 of speaking + 2:30 questions.**
Build: `pdflatex jaisp_roman_talk.tex` (×2) in this directory; figures come from `../../paper/figures/`.

Per-slide talk track below. Times are cumulative targets — if you're past 8:00 when you
finish slide 9 (detection), compress astrometry to its two headline numbers and protect
the Roman slide: it's the slide this audience came for.

---

## 1. Title — 0:00–0:30
JAISP is a joint Rubin–Euclid foundation model we built at IPAC: one self-supervised
representation of both surveys' pixels, reused for detection and astrometry. Everything
shown is from the earliest public data of each survey, and the point of the last part of
the talk is that the design extends naturally to Roman as a third instrument.

## 2. Why joint pixel-level processing — 0:30–2:00
Frame the stakes: cosmology is systematics-limited; the value of Rubin + Euclid + Roman
observing the same sky from 0.3–2 μm is mutual calibration, and that only fully pays off
at the pixel level. The obstacle is heterogeneity — two (soon three) PSFs, samplings,
depths, noise models at once. Position the work carefully for this audience: not "replace
the classical pipelines" but "move the *joint, cross-instrument* part of the problem into
a learned representation, validated against classical controls."

## 3. The recipe — 2:00–3:00
The foundation-model paradigm in one slide. Pretraining is masked-band autoencoding —
hide one of ten bands, predict it from the other nine — so supervision comes from pixels,
no labels or simulations. Land the cost box hard: **2.5 GPU-hours**, ~9M parameters, and
every downstream capability is a few-million-parameter head on cached frozen features.
This is deliberately a *small* foundation model; the paradigm, not scale, does the work.

## 4. Data — 3:00–4:00
Rubin DP1 ComCam commissioning imaging (PSF ~1", so Euclid VIS is the sharp reference
throughout) plus Euclid Q1 MER mosaics; 790 matched ten-band tiles over ~0.2 deg² of
ECDFS. Stress two things: (1) all public data, first release of each survey — a floor,
not a ceiling; (2) EDF-S is held out *entirely* — never seen by foundation or heads —
for the transfer test at the end.

## 5. Architecture — 4:00–5:15
One branch per instrument at native sampling (0.2" and 0.1"), fused on a common 0.4"
grid, transformer bottleneck mixes all ten bands, decoders reconstruct the withheld band.
Give 20 seconds to the negative result — we tried contrastive/JEPA alignment first and it
failed structurally: similarity objectives never require *where*, and sub-pixel position
is the whole game for astrometry. Pixel reconstruction forces spatial layout to survive.
(Likely Q&A seed: "why not one big shared grid?" — resampling throws away the sharper
instrument's information.)

## 6. Validation: reconstruction — 5:15–6:00
Held-out tiles, one band withheld: Rubin r, VIS, and NISP H all come back faithfully.
The key sentence: a band from one instrument is predicted from bands of the *other* —
that's genuine cross-instrument correspondence, not within-instrument copying.

## 7. Validation: probe + routing — 6:00–6:45
Two quick checks that the frozen features are usable and physical: every band's
brightness is *linearly* decodable (R² 0.7–0.95), and attribution shows the model routes
information the way SEDs do — VIS is the universal spatial scaffold, Rubin bands lean on
spectral neighbors, NISP YJH form a near-closed block. If pressed for time, this slide
can be 20 seconds: "we checked the representation is accessible and physically organized."

## 8. Detection: overlay — 6:45–7:30
First measurement head. Trained on lightweight SEP labels — deliberately *not* on the MER
catalog (training on MER gives essentially the same detector anyway). On held-out sky the
four source lists agree on the vast majority of sources; the learned head returns one
detection per resolved galaxy where classical extraction fragments spiral arms into
pieces.

## 9. Detection: depth — 7:30–8:45
The quantitative slide. Against the MER catalog: 93% completeness, 95% purity (purity is
a conservative floor — a VIS-selected catalog can't vouch for real multi-band sources it
doesn't contain). Depth by source-recycling injection, which is catalog-independent:
50% depth VIS ≈ 25.0, **+0.45 mag deeper than the same detector on VIS alone** — past the
turnover of the VIS-selected catalog. The control matters: an identical head reading the
native per-band stems gains only +0.03, so the multi-band gain lives in the *fused
representation*, not the head.

## 10. Astrometry: position head — 8:45–10:00
Second head. Raw cross-survey scatter of matched sources is ~50 mas, dominated by
photon-noise centroiding of faint galaxies. The 0.7M-parameter position head collapses it
to 14–17 mas in every band and re-centers the clouds on the origin. The honesty check:
against *injected truth* (not the training labels), 19 mas at S/N = 5 vs ~50 mas
classical — within ~2 mas of the 17 mas VIS anchor floor. Say the mechanism plainly: the
head doesn't beat VIS, it *carries VIS-quality localization* into bands where photon
noise says single-band centroiding can't reach it.

## 11. Concordance field — 10:00–11:00
The unexpected measurement. With per-source scatter under control, a coherent 9–10 mas
pattern emerges between the two surveys' astrometric solutions — both independently tied
to Gaia, yet offset from each other, with ~7 mas common to all ten bands. Two solvers of
different character (PINN, hierarchical GP) recover the same degree-scale pattern, and
the head absorbs it source-by-source down to ~1.5 mas. Pitch it as a product: a few-mas
map of how well two independently calibrated surveys agree on the sky — QA that no single
pipeline can produce, and exactly what a three-survey era needs.

## 12. Zero-shot transfer — 11:00–11:45
The foundation-model claim, tested. Same frozen foundation, same frozen heads, applied to
EDF-S with **nothing retrained**: detection 93.4%/93.1% vs 93.3%/94.5% in-distribution;
astrometry lands on the same 12–18 mas floor within 1–3 mas. Honest caveat in one clause:
this is transfer across sky, not across instruments — which is the segue.

## 13. Roman — 11:45–13:15 (protect this slide)
The instrument-transfer test is the harder and more valuable one, and Roman is the
natural third stream. Three beats:
1. **Architecturally trivial to add**: one encoder branch per instrument is the design;
   Roman is a third branch, and pretraining needs only overlapping public pixels — no
   labels, no simulations, hours of GPU time once WFI imaging overlaps Rubin/Euclid sky.
2. **Roman upgrades the stack**: sharp, deep NIR at 0.11"/px is the near-infrared analog
   of the VIS spatial anchor — today the NIR side is resampled NISP, and we can *see* the
   ~3 mas registration residual that resampling left behind. Roman's wide-area survey
   sits inside the LSST footprint at survey scale.
3. **The stack upgrades Roman**: multi-band detection depth beyond any single-band
   catalog, WFI-anchored positions carried into Rubin's seeing-limited bands, and
   three-way Rubin–Euclid–Roman concordance fields as a few-mas cross-mission
   astrometric QA product.
Close the slide on the launch window: the recipe is trained in an afternoon on public
data — it will be ready when the first overlapping WFI mosaics are.

## 14. Takeaways — 13:15–13:45
Read the five bullets almost verbatim — they're the summary. Final line: one
representation, trained once, reused across tasks, fields — and, next, instruments.

---

## Likely questions
- **"Why not train on simulations / MER labels?"** — MER-trained detector is essentially
  identical (same injection depth); MER is VIS-selected so it adds nothing beyond our
  VIS-based labels. Simulations would import their own model assumptions.
- **"Is 14–17 mas good?"** — It's the label floor, not the method floor: the head sits
  within ~2 mas of the VIS Gaussian-centroid anchor it's trained toward. Better labels
  (ePSF centroids, Gaia-anchored positions, deeper VIS) lower the floor with no
  architecture change. Classical state of the art (Libralato+24, 0.7 mas) needs
  unresampled exposures and bright cluster stars — different regime from faint
  extragalactic survey sources.
- **"Does the concordance field extrapolate?"** — No, and we say so: it's coherent but
  not constant, valid within its 300"–900" correlation scales. It's a measurement/QA map,
  not a correction you can transport.
- **"What about photometry and shapes?"** — In development, same frozen features; the
  encoder was forced to retain per-band morphology and PSF structure, which becomes the
  measured quantity.
- **"Roman data won't overlap Euclid deep fields immediately."** — Any Rubin/Euclid
  overlap works for pretraining; the wide survey guarantees Rubin overlap at scale, and
  the approach needs only ~0.2 deg² of overlap to train, as shown here.
- **"9M parameters — is that really a foundation model?"** — It's the paradigm
  (pretrain once, freeze, many heads), not the scale. Small is a feature: 2.5 GPU-hours
  means anyone can retrain it as new data arrives.
