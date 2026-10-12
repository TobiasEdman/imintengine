# Historical NFI comparison — provenance pending

**Status:** PROVISIONAL historical numbers; held-out provenance not verified here
**Created:** 2026-09-26 · **Review update:** 2026-10-12
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

The September run was reported as restricted to the distillation's
`test_tiles`, with NMD2023 scored on the same plots. The table is retained
as a historical record, not as verified held-out evidence: an immutable
result linked to the split and process revision has not been provided here.
The reproduction requirements below must be met before using it for a
held-out superiority claim.

| cell | n | model | NMD2023 | margin |
|---|---|---|---|---|
| tessera_r3 | 184 | 0.5598 | 0.4891 | **+7.1** |
| tessera_r4 | 184 | 0.5489 | 0.4891 | +6.0 |
| prithvi600m_r3 | 184 | 0.5326 | 0.4891 | +4.4 |
| prithvi300m4f_r3 | 169 | 0.5385 | 0.4970 | +4.2 |
| **tessera_r2** | 184 | **0.5109** | 0.4891 | **+2.2** |

`tessera_r2` is the relevant comparison for a model without NFI supervision:
rung 2 did not use the NFI distillation targets used at rungs 3 and 4. That
removes this particular route for target leakage; it does not establish
spatial independence, rule out selection effects, or verify the scored split.

The printed `tessera_r2` margin is +2.2 percentage points on 184 plots.
It is a descriptive difference, not a lower bound on performance or evidence
of statistically established superiority. The other rows are not upper bounds.

## What was wrong the first two times

The first version reported +10 points and called it independent field truth.
It was neither.

The NFI index is a **full cohort**, and the same index feeds the distillation
teachers at rungs 3 and 4. `clay_r2_split.json` puts 208 training tiles / 735
plots against 53 test tiles / 209 plots — so roughly **three quarters of the
plots being scored were plots whose NFI labels had shaped the training targets
of the very rungs showing the gain.**

The reported recomputation on the restricted subset changed the scores as
follows; its held-out provenance remains pending:

| | change vs full cohort |
|---|---|
| tessera_r3 | −4.0 |
| clay_r3 | −4.9 |
| terramind_r4 | −8.3 |
| tessera_r1 | **+3.9** |

The report states that every rung-3/4 cell fell and every rung-1/2 cell rose.
Those directions alone do not prove contamination: a change in sample
composition can improve one fixed classifier and worsen another without
label leakage. Overlap between scored identities and teacher targets is the
relevant evidence; it must be checked against the actual splits and inputs.

The reported rung 2 → 3 contrasts were **+2.9 to +10.3 across five backbones,
mean ≈ +5.3**, compared with +9.8 on the full cohort. These are descriptive
contrasts on different populations, not a causal distillation effect or a
measurement of the amount of leakage removed.

## Historical coverage and remaining limits

The coverage figures below describe the September 2026 cohort, before the
2024 campaign and the legacy-holdout census. They are not current availability.

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

**The crop overlap audit is a separate population.** The 2,491 distill rows
and 1,064 protected-holdout rows belong to the LUCAS crop experiment, indexed
in [`crop-distill/README.md`](../evidence/crop-distill/README.md). Its point-ID
overlap and `cleared_for_scoring: false` finding do not describe the NFI rows
in this table. NFI independence requires its own identity and split audit.

**One run per cell.** No repeated training seeds or uncertainty intervals
are recorded here. The +2.2-point margin on 184 plots has no reported bound.

## Reproduction requirements

On main (checked 2026-10-12, `d4d24f9`), `compare_nmd2023_nfi.py` has no
`--split-json` argument and emits no `holdout` block. Its existing
`--model-per-plot` option compares the supplied model rows with NMD; it does
not establish that those rows were held out. The split-aware flag proposed
in #65 was closed unmerged on 2026-10-08, together with #62 and #63. Their
replacement is Codex's #68, a reproducible held-out NFI evaluation that
excludes every teacher's feature plot-years by `(TractID, PlotID, Year)` and
binds inputs, runtime, checkpoints and outputs by content hash. Use #68's
evaluation, once integrated, rather than a flag on this script.

Before promoting these numbers to verified held-out results, retain for each
cell an immutable result and receipt that binds the process commit, checkpoint,
input hashes, split path and SHA-256, held-out identities, and before/after
plot counts. For a split-aware rerun, require a populated `holdout` block
and validate it against the split; the presence of the block alone is not
proof of independence. Historical filtered results may be used only with
equivalent auditable provenance. No such rerun or receipt is supplied by this
write-up, and arithmetic checks on the margins cannot substitute for it.

## Next evidence

The 2024 plot-driven campaign is recorded in PR #67. The September table
is being superseded by a multi-year evaluation, about 1,000 per
inventory year 2018–2024 (plan: #69). Merged #71 versioned the 2024
leftovers and ran the
[2026-10-12 legacy-holdout census](https://github.com/TobiasEdman/imintengine/blob/91902bf2cf3b8f9bef7aa0579da52d877245fbb8/docs/experiments/nfi_multiyear_holdout_census_2026-10-12.md),
which excludes 212 tiles that overlap same-year training imagery; #75
enriches the remaining holdout copy and tops up 2018–2022; #68 freezes and
scores the result. These later populations do not retroactively validate the
September table.

The next scoring pass needs audited held-out provenance, a fixed eligible
population and paired uncertainty estimates that account for tract clustering.
Repeated training seeds address training variability separately. Any new fetch,
evaluation run or freeze needs its own authorisation.
