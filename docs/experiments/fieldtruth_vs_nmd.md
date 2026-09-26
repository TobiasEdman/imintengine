# Field truth — the measure that can beat the map

**Status:** RESULT, held-out numbers · **Created:** 2026-09-26
**Parent:** [`ladder_distill_stage.md`](ladder_distill_stage.md) ·
**Related:** [`lucas_validation_plan.md`](lucas_validation_plan.md)

Every earlier comparison in this project scored the ladder against NMD — a map
against a map. That can only ever say which model reproduces the reference
best, never whether any of them is *right*. Field truth changes the question:
NFI and LUCAS are ground observations, so the map and the model are both
candidates and the plot is the judge.

This is the write-up of that measurement. It has been rebuilt twice, because
the first two versions measured something other than what they claimed.

## The result

Scored on **held-out plots** — the distillation's own `test_tiles` — with
NMD2023 scored on **exactly the same plots**:

| cell | n | model | NMD2023 | margin |
|---|---|---|---|---|
| tessera_r3 | 184 | 0.5598 | 0.4891 | **+7.1** |
| tessera_r4 | 184 | 0.5489 | 0.4891 | +6.0 |
| prithvi600m_r3 | 184 | 0.5326 | 0.4891 | +4.4 |
| prithvi300m4f_r3 | 169 | 0.5385 | 0.4970 | +4.2 |
| **tessera_r2** | 184 | **0.5109** | 0.4891 | **+2.2** |

**The cleanest evidence is `tessera_r2`, not the largest margin.** Rung 2
never touched NFI in training, so it is structurally immune to the
contamination described below. It beats the map anyway. Every other row in
this table is an upper bound on its own honesty; `tessera_r2` is a floor.

The headline is therefore narrow and defensible: **a model trained without any
NFI supervision scores higher against forest field observations than the
national land-cover map does, on the same plots.** Two points of overall
accuracy, on 184 plots. Not a large claim. A real one.

## What was wrong the first two times

The first version reported +10 points and called it independent field truth.
It was neither.

The NFI index is a **full cohort**, and the same index feeds the distillation
teachers at rungs 3 and 4. `clay_r2_split.json` puts 208 training tiles / 735
plots against 53 test tiles / 209 plots — so roughly **three quarters of the
plots being scored were plots whose NFI labels had shaped the training targets
of the very rungs showing the gain.**

The recomputation proves the contamination by asymmetry rather than by
assertion. Restricted to held-out plots:

| | change vs full cohort |
|---|---|
| tessera_r3 | −4.0 |
| clay_r3 | −4.9 |
| terramind_r4 | −8.3 |
| tessera_r1 | **+3.9** |

**Every rung-3/4 cell falls. Every rung-1/2 cell rises.** Contaminated cells
lose their advantage on plots they have not seen; clean cells gain slightly,
as a smaller and differently distributed sample will. That pattern is what a
leak looks like from the outside, and no other explanation fits both signs.

The distillation effect survives the correction, halved: rung 2 → 3 gives
**+2.9 to +10.3 across five backbones, mean ≈ +5.3**, against the +9.8 the
contaminated measurement reported.

## What these numbers still cannot carry

**n is small — 169 to 190 plots per cell — and that is a property of the
cohort, not of the index.** Measured 2026-09-26: rebuilding the plot→tile
index against the full 18,075-tile cohort moved it from 982 rows / 270 tiles
to 1,105 / 306. A 12.5% gain. The index was mildly stale; it was never the
constraint.

The constraint is that **tiles were sampled first and plots counted
afterwards**. Of 18,661 NFI plots in the Sentinel-2 era, 1,105 land on a tile.
Drawn the other way — tiles placed at the plots — the whole inventory
(2007–2025, 43,892 plots) fits on about 7,939 tiles, and a single Sentinel-2-era
inventory year needs only **518–617**. The cohort holds 18,075 tiles and
captures 306 of them.

Two consequences follow directly:

- **Only 2018, 2021 and 2022 exist in the cohort.** NFI's 2019, 2020, 2023,
  2024 and 2025 plots cannot land on a tile that was never fetched.
- **Zero RTK plots.** Emlid RTK navigation starts in 2024 (2–5 cm against the
  earlier handheld metre-level). The most positionally accurate plots in the
  entire inventory contribute nothing, because no 2024+ tiles exist.

**Point identity overlaps the protected holdout.** The scored set carries
2,491 rows at 1,521 unique point IDs, 404 of which also appear in the
crop-holdout — 386 in the same year, touching 616 of the holdout's 1,064 rows.
Exact `(tile_name, point_id)` keys are disjoint, which is why the split froze
cleanly; point identity is not. The hash-bound E01 index audit records this as
`protected_role_overlap_detected` with `cleared_for_scoring: false`.

**One run per cell.** No seeds, no intervals. A margin of +2.2 on 184 plots is
a result, not an estimate with a bound on it.

## How to reproduce, and how to tell the two apart

Full-cohort and held-out runs are now distinguishable from their output alone.
`compare_nmd2023_nfi.py --split-json <split>` restricts scoring to the split's
`test_tiles` and writes the split's path, SHA-256 and before/after plot counts
into the result under `holdout`. Without the flag the comparison is
full-cohort and `holdout` is `null`.

Any figure quoted as held-out must come from a result whose `holdout` block is
populated. The two versions of this document that had to be withdrawn were
both cases of a full-cohort number wearing a held-out label, and the flag
exists so that mistake leaves a trace instead of a claim.

## What would make this stronger

In the order that buys the most per unit of cost:

1. **Draw tiles at the plots for one inventory year.** **518 tiles** covers
   all 2,389 plots of 2024 — and 2024 is the year to pick: it is the first RTK
   year, and LPIS crop labels still reach it. That single fetch would more than
   double the entire field-truth set using under 3% as many tiles as the
   current cohort, and it is PU-free over DES.
2. **Re-score the ladder on the held-out flag** so every published cell is
   generated by the restricted path rather than corrected after the fact.
3. **Seeds.** Until then, no margin here is separable from another.
