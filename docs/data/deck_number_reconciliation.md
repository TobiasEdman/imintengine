# Deck number reconciliation — clean-holdout audit (2026-08-18)

Task: recompute every quantitative claim in `docs/decks/v8b_nfi_conference.pptx`
from current dumps, and ensure **no validation figure uses training data** (no
training-tile leakage into any accuracy number).

## Contamination model

The models are trained on 7,882 Sentinel-2 tiles with NMD labels. A field point
(NFI or LUCAS) is *contaminated* if it falls on a **training tile** — the model
saw that tile's imagery + NMD supervision, even if the field label itself was
never a training target. "Never trained on the label" ≠ "spatially held out".
Clean validation = points on the 53 held-out tiles from `distill_split.json`
(`test_tiles`), which no training saw.

## KEY FINDING — LUCAS numbers in the deck are contaminated

`scripts/lucas_tile_coverage.py:115` assigns `split="test"` iff the point's tile
is in `test_tiles`, else `"train"`. `colocate_plots` keeps only points that fall
on *some* tile, so every LUCAS point is on a tile:

| split | n | meaning |
|---|---|---|
| test | 71 | on the 53 held-out tiles — **clean** |
| train | 10,037 | on training tiles — **contaminated (memorization)** |

The deck cites "10,108 points" and Tessera L2b **0.499** — that is the **`all`**
split (train+test), 99.3% training-tile points, essentially a train-set number.
`validate_against_lucas.py` itself warns the train split is "on tiles the
backbone saw; memorization-[inflated]".

**Deck LUCAS numbers = `all` split (CONTAMINATED). Clean = `test` split, n=71.**

L2b 28-class overall, per split (recomputed from `data/lucas/per_point/`):

| model | all (deck) | test (clean, n=71) | train |
|---|---|---|---|
| tessera | 0.4987 → deck 0.499 | 0.4789 | 0.4989 |
| distill | 0.4842 → deck 0.484 | 0.5070 | 0.4840 |
| tradslag (Prithvi-600M) | 0.4770 → deck 0.477 | 0.4085 | 0.4775 |
| prithvi300m | 0.4383 | 0.4789 | 0.4380 |
| clay | 0.4042 | 0.4507 | 0.4039 |

At n=71 the clean numbers are noisy (±~12 pp) and the ordering scrambles
(distill tops it; tradslag drops to 0.41). **The deck's claim "Tessera ≥ Prithvi
on every independent truth" does not survive on the clean LUCAS split** — but
n=71 is too thin to make any LUCAS ordering claim either way. LUCAS can no
longer serve as the *high-power* cross-check the deck presents it as.

## NFI numbers — already test-tiles-only (clean)

The NFI held-out family uses the grouped-tile `test_tiles` split (canonical
5-class scoring: fraction models → calibrated collapse floor0.05/dom0.6, hard →
28→5 argmax, treeless truth → non-forest). Anchors reproduce exactly:
tessera_frac 0.5885, Prithvi-600M/tradslag 0.5789. See
`data/distill/model_race_standings.json` (regenerated) and
`data/distill/race_rigor_stats.{json,md}`.

### LUCAS L2a forest-fraction (dominant-species agreement)

Deck cites Tessera 0.809 / Prithvi-600M 0.784. Recomputed:

| model | all (deck) | test (clean) |
|---|---|---|
| tessera | 0.8086 → deck 0.809 | 0.9474 (n=19) |
| tradslag | 0.7837 → deck 0.784 | 0.9474 (n=19) |
| prithvi300m | 0.7463 | 0.7895 (n=19) |
| clay | 0.6976 | 0.7368 (n=19) |

Same story: deck = contaminated `all` split. Clean L2a rests on **19 dominated
forest points** — `validate_against_lucas.py` docstring: "an accuracy on a
handful of points is noise, not a metric." tessera=tradslag=0.947 is a tie at
n=19 (meaningless separation).

### Implication for the deck's "three independent truths"

- **NFI** (209 test-tile plots): clean, adequately powered — the real backbone.
- **LUCAS**: clean set is n=71 (L2b) / n=19 (L2a) — too thin to be a
  high-power cross-check. The "10,108 points" framing is the contaminated
  all-split. Must be reframed or demoted.
- **LUCAS×LPIS crop cross-check** (79.8 %, 0.81, per-crop): label-vs-label, no
  model prediction involved → contamination N/A, stays valid as a *label*
  sanity check (not a model-validation metric).

### NFI held-out benchmark + per-class F1 — recomputed (test tiles, canonical scoring)

Deduped test set n=202 (vs deck's 209 rows — the 7 tile-overlap duplicates).
Reproduces the deck within the dedup delta:

| metric | Distilled (hard) | Fractions/600M (frac) | deck |
|---|---|---|---|
| OA | 0.4950 | 0.5792 | 50.2 / 57.9 |
| kappa | 0.361 | 0.416 | 0.371 / 0.420 |
| F1 Pine | 0.614 | 0.744 | 0.61 / 0.74 |
| F1 Spruce | 0.624 | 0.538 | 0.63 / 0.55 |
| F1 Deciduous | 0.390 | 0.576 | 0.43 / 0.59 |
| F1 Mixed | 0.302 | 0.300 | 0.30 / 0.29 |
| F1 Non-forest | 0.297 | 0.000 | 0.32 / 0.00 |

**NFI benchmark + F1 are CLEAN** (grouped-tile test split) and reproduce.

### Provenance resolved (finding docs are the deck's cited source)

The benchmark/ceiling numbers come from `docs/data/*_finding.md`, NOT the
`nfi-validation-*.json` (those carry only `forest_type_accuracy`, a different
metric). Verified against the finding docs:

**Benchmark slide — fully CLEAN (held-out 209):**
- NMD2023 43.1 = `tradslag_fraction_finding.md:15` "(NMD2023: 0.431)" held-out.
  (An earlier audit mis-flagged this as contaminated by reading
  `benchmark-nmd-vs-nfi.json` = 0.390, which is the all-982 NMD2018 map — the
  WRONG artifact. The deck's 43.1 is the held-out NMD2023 value.)
- Distilled 50.2 = `distill_finding.md:15` 0.502 held-out.
- Fractions 57.9 = 0.579 held-out.

**Ceiling slide (8) — all-944 BY DESIGN (labeled "944 plots"), but the
headline survives an honest split:**
- v8b 46.5 / NMD2023-map 49.3 / Distilled 52.7 = all-944 framings
  (`hybrid_nfi_head_finding.md:16`, `tradslag_fraction_finding.md:16`,
  `distill_finding.md:20`).
- **NFI head OOF 63.7** (stratified, all-944) has a spatially-honest twin:
  **0.632 (OOF grouped-by-tile)** — `hybrid_nfi_head_finding.md:13,21`:
  "grouped by tile (honest train/test) confirms the result is NOT spatial
  leakage: 0.632 (+0.139)." Δ = 0.5 pp → ceiling-break claim is robust.
  *Fix:* cite 0.632 (grouped-by-tile) to pre-empt the leakage objection.

**Ensemble slide (13) — fully CLEAN** (209 held-out, config selected on train
OOF only): 0.579 / 0.617 / McNemar 0.3742 / CI[-0.038,+0.11] / G2 <0.03. All
reproduce (`ensemble_results.json`).

## VERDICT

The ONLY genuine train-data-in-validation problem is **LUCAS** (slides 14, 16):
the deck presents the all-split (99.3 % training-tile points) as "independent
validation." Clean LUCAS = n=71 / n=19 — too thin to headline. Everything NFI
is clean or has a clean confirmation. The ceiling slide is all-944 by design
and its headline (63.7) holds under the honest grouped-by-tile split (63.2).

## The proper fix (chosen direction) — off-footprint independent test set

Retraining is the wrong tool: the training set is fine; only *validation*
reused training tiles. The clean, powered validation already exists as field
plots OUTSIDE the training footprint — score them by **inference only** (fetch
year-matched tiles at those locations, run the existing trained models). No
retraining.

Available clean pools (never in training):
- **NFI:** 43,051 of 43,892 plots off-footprint; 38,465 forest; **13,459
  forest & year-matched 2018–2024**, in only **587 tracts**. Forest-type truth
  derivable from per-plot species volumes (VolPine/VolSpruce/VolBirch/…). vs the
  current 209 held-out.
- **LUCAS:** ~24,372 SE-2022 points off-footprint (of 34,480 total; 10,108 on
  footprint). vs the current 71 clean.

Payoff: (1) fixes LUCAS contamination; (2) turns the underpowered 209-plot NFI
test (±7 pp CIs — all top-4 race comparisons INCONCLUSIVE) into a powered
independent benchmark (±1–2 pp) that can actually RESOLVE the equivalence/
difference claims the rigor pass exists for; (3) pre-empts the publication_plan
"thin 209-plot test set" reviewer risk; (4) reuses every trained model.

Cost: a bounded, year-matched fetch at off-footprint plot tiles + inference.
Per CLAUDE.md, scope against current DES allotment; do NOT launch bulk fetch
unasked. NEXT: produce precise scoping (target N → tiles → fetch cost/time).




