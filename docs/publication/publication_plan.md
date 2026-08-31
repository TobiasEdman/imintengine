# Publication plan — *Breaking the label ceiling*

**Status:** plan · **Created:** 2026-08-13 · **Owner:** Tobias Edman
**Working title:** *Breaking the label ceiling: field-calibrated land-cover
mapping that outperforms its national-map teacher*

---

## 1. The contribution (one sentence)

A segmentation model distilled from Sweden's national land-cover map
(NMD2023 — itself a *modelled* product, not field truth) **surpasses that
map's own accuracy** when both are scored against independent field data
(National Forest Inventory plots and LUCAS points) — showing that
cartographic labels impose an accuracy *ceiling* that field calibration
breaks, and that on this task a **cheap frozen-embedding backbone matches a
600 M-parameter geospatial foundation model**.

## 2. Why it is publishable (the novelty, ranked)

1. **The label-ceiling result (flagship).** National/continental land-cover
   maps (NMD, CORINE, ESA WorldCover, Dynamic World) are used everywhere as
   training labels *and* as de-facto ground truth. We show a model trained
   on such a map beats it on independent field truth (**forest type ≈ 0.39
   → ≈ 0.59**, +50 % relative). The mechanism — random per-pixel label
   noise averaging out over thousands of tiles, and field truth exposing the
   teacher's own errors that self-scoring hides — is general, not
   Sweden-specific. This reframes how these maps should be used.
2. **A systematic foundation-model comparison on one field-calibrated
   task.** Prithvi-600M, Tessera, (CROMA, Clay, TerraMind — in progress) on
   the *same* target and truth. Finding: **bigger/more complex is not
   better** — Tessera's frozen 128-d embeddings tie Prithvi-600M (p≈0.88),
   gated aux-fusion ties simple concat, a 6-member ensemble is within noise;
   **patch size (spatial resolution) matters more than parameter count**.
3. **A forest-type fraction head** (per-pixel crown-cover of tree species)
   that beats the hard 28-class head on forest type and yields threshold-free,
   calibratable outputs.
4. **A reusable field-calibrated benchmark**: a 28-class Sentinel-2 dataset
   over Sweden with an evaluation protocol grounded in NFI/LUCAS field data
   rather than in the map — reusable by others to test the same claim
   elsewhere.

## 3. Target venue

**Primary recommendation: *Remote Sensing of Environment* (RSE).** Highest
impact for a result that changes how national maps are used as labels;
strongly values independent field validation and methodological rigor — both
our strengths.

| Venue | Fit | Note |
|---|---|---|
| **Remote Sensing of Environment** | ★ flagship finding | field-validation culture; high bar, high impact |
| ISPRS J. Photogrammetry & RS | methods-forward | strong second choice |
| IEEE TGRS / JSTARS | broad EO+ML | faster, solid, lower ceiling |
| Remote Sensing (MDPI) | applied/dataset | fast, open access |
| NeurIPS Datasets & Benchmarks | the FM comparison + dataset | possible **companion** paper |

**Decision to make (Tobias):** one flagship paper (label ceiling + race +
dataset), or a flagship (RSE) **plus** a dataset/benchmark companion (MDPI
*Remote Sensing* or NeurIPS D&B). Recommendation: single flagship first;
spin the dataset out only if reviewers ask to separate it.

## 4. Related work to position against

- **Learning with noisy labels / knowledge distillation** — the
  "student can exceed a noisy teacher" thread (our result is a concrete EO
  instance).
- **Geospatial foundation models** — Prithvi-EO, Clay, CROMA, TerraMind,
  Tessera, SatMAE, Scale-MAE (fair single-task comparison is rare).
- **National/continental land-cover products & their validation** — NMD
  (Naturvårdsverket), CORINE, ESA WorldCover, Google Dynamic World.
- **Forest-type / species mapping from Sentinel-2**, and **NFI/LUCAS-based
  accuracy assessment** (area-frame vs plot-scale caveats).

## 5. Claims → evidence map (what's DONE vs TODO)

| # | Claim | Evidence (metric / figure) | Status | Provenance |
|---|---|---|---|---|
| C1 | Distilled model beats NMD2023 on NFI field truth | forest type 0.39→0.59; held-out 43.1→50.2→57.9 %, κ 0.30→0.42 | **done** | `nfi_validation_findings.md`, `distill_finding.md`, `benchmark-nmd-vs-nfi.json` |
| C2 | The ceiling is the *label*, not the representation | same features + NFI supervision → 63.7 % OOF | **done** | `hybrid_nfi_head_finding.md` |
| C3 | Result holds on a 2nd independent truth (LUCAS) | L2b 0.499, L2a 0.809 (10,108 pts) | **done** | `lucas_l2_finding.md` |
| C4 | Cheap backbone ties the 600M FM | Tessera 0.589 vs Prithvi 0.579, p≈0.88 | **INCONCLUSIVE** — underpowered null; needs TOST on 944+LUCAS | `model_race_standings.json` |
| C5 | Architectural complexity doesn't pay | gated≈concat; ensemble within noise (p=0.37) | **INCONCLUSIVE** — same; equivalence test required | this session; `ensemble_g1g2_finding.md` |
| C6 | ~~Resolution > size~~ → **REFORMULATED (08-17): encoder quality gates; cheap-and-strong (Tessera p1) beats big (600M)** | measured axis: p1 0.589 · p8 (Clay) **0.483** · p14 0.579 · p16 0.558 — Clay breaks the monotone curve | **reframed** | CROMA = second p8 point (pending S1 v2); keep as a *finding* (interior point refutes the naive claim), not a casualty |
| C7 | Fraction head > hard head on forest type | 57.9 vs 50.2 % | **done** | `tradslag_fraction_finding.md` |
| C8 | Labels are internally well-founded | LUCAS×LPIS crop agreement 0.81 (excl. pasture) | **done** | `lucas_lpis_crosscheck_finding.md` |
| C9 | Full model race across 5 FMs | CROMA/Clay/TerraMind vs Prithvi/Tessera | **IN PROGRESS** | training campaign (2026-08-13) |

## 6. Experiments still needed for a rigorous submission

**Blocking (must have):**
- **Finish the model race** — CROMA + Clay + TerraMind + **Prithvi-300M**
  on the same target/truth. TerraMind and 300M are the two patch-16 **coarse
  anchors** for the resolution figure (F6). Closes C9 and completes C6.
  *Note:* patch-16 backbones (300M, TerraMind) cannot use the 504 crop
  (504 ÷ 16 = 31.5) — train/eval them at a crop divisible by 16 (496 or 512);
  this is almost certainly the "schema/tile mismatch" 300M was parked for.
- **Equivalence testing, not just null p-values** (adversarial-review mandate,
  2026-08-14) — every "tie" in the paper (Tessera≈Prithvi, gated≈concat,
  ensemble-within-noise) is currently an **underpowered non-difference**, not a
  proven equivalence: McNemar p=0.88 on n=209 hides a paired-difference CI of
  ~±8–10 pts. Fix: pre-register a SESOI (smallest accuracy difference that
  matters), run **TOST / Bayesian ROPE** on every head-to-head, and only call a
  pair "equivalent" if its difference-CI falls inside the SESOI band. Report
  McNemar + bootstrap CIs uniformly; Holm/BH-correct the family of comparisons.
  Until then, every current "tie" is relabelled **INCONCLUSIVE**.
- **Spatial cross-validation** — replace/augment the single 209-plot holdout
  with spatial-block CV over all 944 NFI plots; use LUCAS (10 k) as the
  higher-power independent test. Pre-empts the "thin test set" objection.
- **A resolution ablation** — the patch-size claim (C6) needs a controlled
  figure: accuracy vs patch size / boundary-F1 vs patch size.

**Strengthening (should have):**
- **Baselines beyond NMD**: a from-scratch U-Net and/or Random Forest on the
  same features, to bracket the FMs.
- **Ablations**: aux channels (forestry/DEM/VPP), temporal frames, fraction
  vs hard head, distillation vs direct NFI supervision.
- **Calibration/uncertainty** of the fraction head (reliability diagrams).
- **Error analysis**: spruce weakness (shared across all backbones — a
  data/task limit), confusion structure, and spatial error maps.

## 7. Planned figures & tables

- **F1** Study area + data: Sweden, NFI/LUCAS spatial + class distribution.
- **F2** Method: dual-head architecture + label construction (NMD→LPIS→SKS)
  + the 35-channel tile.
- **F3** The label ceiling: NMD generations vs NFI-supervised (63.7 %).
- **F4** Held-out benchmark: NMD2023 vs distilled vs fraction (with CIs).
- **F5** Qualitative maps: the 5-view crispness comparison.
- **F6** Model race: backbones × {NFI, LUCAS} with compute + patch-size axes.
- **F7** Error analysis: per-species, confusion, spatial error map.
- **T1** Full metrics with CIs. **T2** Ablations. **T3** Dataset spec.

## 8. Anticipated reviewer objections → pre-emptions

| Objection | Pre-emption |
|---|---|
| "209-plot test set is too small" | Spatial-block CV over 944 + LUCAS (10 k) as primary independent test; report CIs |
| "NMD2023 wasn't built for plot-scale grading" | Frame as *operational baseline as used*; grade like-for-like on identical plots |
| "NFI plots ≈ 1.5 S2 pixels, GPS error" | Use RTK (2024–25) plots for the tight validation; area-frame caveats explicit |
| "~59 % absolute isn't high" | 28-class @ 10 m is hard; the *relative* gain and *ceiling-breaking* is the claim, not absolute SOTA |
| "Sweden-only" | Scope the empirical claim to Sweden; argue the *mechanism* generalizes; invite replication via released benchmark |
| "Distillation beating teacher is known" | Position precisely: novelty is the *field-truth* demonstration on an operational cartographic label, quantified |

## 9. Data, code & reproducibility

- **Dataset**: `unified_v2_512` (28-class, 4 temporal frames + 11 aux, EPSG:3006,
  ~7 k tiles). Release with a data card + license notes (S2/NMD/LPIS/SKS/SLU).
- **Field truth**: NFI (SLU Riksskogstaxeringen calibration subset) + LUCAS —
  document access + the year-matching protocol.
- **Code**: training + the two-stage cached eval pipeline (`infer_tiles.py`
  + `score_against_truth.py`) + provenance MANIFESTs.
- **Weights**: release the distilled model + fraction head.

## 10. Roadmap

- **Phase 0 (now):** finish the model race; lock all numbers; re-run the
  cached-eval bit-parity gate. *(In flight.)*
- **Phase 1:** rigor pass — spatial CV, uniform significance, resolution
  ablation, error analysis.
- **Phase 2:** draft (structure = §7 figures + §5 claims).
- **Phase 3:** internal review (adversarial: a domain/stats reviewer on the
  significance + spatial-leakage arguments) → revise.
- **Phase 4:** submit to RSE.

## 11. Open decisions for Tobias

1. **Venue**: RSE flagship (recommended) vs faster IEEE/MDPI.
2. **Scope**: single flagship vs flagship + dataset companion.
3. **Test protocol**: adopt spatial-block CV + LUCAS-primary (recommended)
   vs keep the single 209 holdout.
4. **Model race breadth**: RESOLVED (2026-08-13) — race all of CROMA, Clay,
   TerraMind, **Prithvi-300M**. TerraMind + 300M are kept deliberately as the
   two patch-16 coarse anchors for the resolution figure (they confirm C6
   rather than needing to win).
5. **Authorship & timeline**.
