# Multi-year NFI control tiles — plan

2026-10-08 · Claude · for Tobias (decisions) and Codex (task 0103, PR #68)

**Status:** plan only. No job, fetch, GPU time or freeze is authorised by
this document. Each step below needs its own go.

## Goal

PR #68 should freeze a held-out NFI evaluation whose observations are spread
evenly across inventory years. At head `a67aaec`, PR #68 already enforces
explicit per-year quotas. What is missing is imagery for most years.

**Scope (decided by Tobias, 2026-10-08):**
- Evaluate every year that has training data, evenly.
- Codex computes the tile counts per year.
- This plan covers the runbook, prerequisites and decisions.

## Where we stand

Plot-years are NFI identities `(TractID, PlotID, Year)`.

| Year | NFI plot-years | In a teacher feature pool | Eligible on imagery we have | Imagery |
|---|---:|---:|---:|---|
| 2018 | 2,348 | ~446 | 91 | cohort (training) tiles |
| 2019 | 2,319 | 0 | 0 in the #68 audit | holdout_val_512, not indexed |
| 2020 | 2,325 | 0 | 0 in the #68 audit | holdout_val_512, not indexed |
| 2021 | 2,238 | ~53 | 15 | cohort tiles |
| 2022 | 2,318 | ~342 | 36 | cohort tiles |
| 2023 | 2,329 | 0 | 0 in the #68 audit | holdout_val_512, not indexed |
| 2024 | 2,389 | 0 | 2,212 | nfi2024 campaign, 475 tiles |

**Sources:**
- NFI plot table: `data/nfi/nfi_plots.parquet` (local only, untracked by
  design because of NFI coordinates).
- Eligible counts: the #68 cluster exposure audit (job
  `nfi0103-exposure-audit-v2`, 841 feature plot-years excluded).
- Planned holdout points: `data/nfi/holdout_val_phaseA.json` (local only,
  untracked).
- The pool split per year is inferred, but it sums to the audited 841.
- 2017 and 2025 are left out: neither has LPIS, DES has no L2A for 2017,
  and the tessera enrichment maps them to 2018 and 2024.

### An existing asset: `/data/holdout_val_512`

- **What it is:** 1,463 tiles, built in August 2026 outside the training
  footprint (plan `docs/plans/independent_holdout_validation.md`). 1,650
  lattice-cell tiles were planned across 2018–2024.
- **What it covers:** about half of each 2019, 2020 and 2023 NFI inventory
  falls on these tiles (planned 1,119 / 1,143 / 1,366 plots), and none of
  those plots is in a teacher pool. The 2018, 2021 and 2022 holdout tiles
  (136 / 131 / 659 planned) may add more.
- **Why the audit misses it:** no NFI index has been built over this
  directory.
- **Open points:**
  - Was it used for model selection or early stopping by any evaluated
    model? `scripts/ladder_inference_matrix.py` calls it "never trained
    on", but use for selection has not been checked.
  - Its tessera came from the old source, before promotion. Evaluation
    input must match what each model was trained with.
  - Are frame_2016, S1-2016 and vpp_year complete? Which 187 planned tiles
    were never built?

## Steps

0. **Prerequisites.**
   - tessera-promote-v2 reaches a terminal state with a clean report.
   - PR #67, the 2024 runbook, is reviewed and merged.
   - The four 2024 steps that ran unversioned are versioned (vpp-retry5,
     zero-probe, ref-inventory, tessera-refetch-all).
1. **Census of holdout_val_512.** One read-only CPU job of about 1 h. It:
   - checks keys and flags against their arrays, the same way as the
     nfi2024 census;
   - reads each tile's spectral year, tessera source, and whether
     frame_2016, S1-2016 and vpp_year are complete;
   - builds an NFI index with the same identity and pool exclusion as index
     v4;
   - checks every teacher and student split file for training or selection
     use;
   - measures how many eligible plots of every year lie under the footprint
     of any training tile.
2. **Bring the holdout tiles up to the 2024 standard,** in an
   evaluation-only copy, never in place:
   - re-source tessera to the promoted source;
   - run sen2cor for frame_2016, and S1-2016, where missing;
   - stamp vpp_year.
3. **Top-up fetch** for the years still short of N.
   - **Placement:**
     - Make the year a parameter; the nfi2024 manifest hardcodes 2024.
     - Exclude plot-years that are in a pool, and plots already on
       holdout tiles.
     - Fix the cell selection with a seed before any scores exist.
   - **Pipeline:** the same as 2024:
     1. WEkEO VPP prefetch (`VPP_SOURCE=wekeo`).
     2. DES spectral fetch.
     3. Retry pass.
     4. S1, SKG, tessera (promoted source) and the vpp_year stamp.
     5. sen2cor for frame_2016.
     6. S1-2016.
     7. Full key census.
   - **Storage:** one staging root per year,
     `/data/unified_v2_512_nfi{Y}_staging`. These tiles stay out of
     training.
4. **Index v5 and re-audit.**
   - Build the index over cohort, nfi2024, the holdout copy and the new
     years.
   - Rerun the #68 exposure audit with per-year support after the modality
     and crop checks.
5. **Freeze.** PR #68 freezes with explicit year quotas. Then the go/no-go
   for inference and NMD scoring.

**What PR #68 needs first** (from Claude's review at `a67aaec`):
- support for several evaluation roots, with per-root tile roles and
  per-year strata;
- NMD-coverage filtering before quota selection;
- the same masked-frame filling at inference as training uses.

## Cost and time (scaled from 2024; re-measure after step 1)

| Item | 2024 measured | Driver |
|---|---|---|
| DES spectral fetch | 31.6 h for 507 tiles (16–24/h), plus 4.6 h retry | new tiles in step 3 |
| sen2cor frame_2016 | ~8.3 GPU-h for 475 tiles (≈1.75 GPU-h per 100) | new tiles plus holdout gaps |
| WEkEO VPP requests | ~208 for one year; quota 500; cache wiped 2026-08-24 | new years in step 3 |
| CDSE PU | 0 | stays 0 |

## Decisions and recommendations

**D1. Years.** Decided: every year that has training data, evenly. Codex
sizes it.

**D2. Equal count per year (N).** Recommendation: N = 1,000 per year,
lowered to the smallest year's eligible count if the census shows any year
cannot reach it. Never drop a year.
- 2018 caps what is possible: 1,884 candidates before image losses. N ≈
  2,000 cannot be reached in every year.
- At ~90 % yield, 1,000 leaves a reserve in every year, and the holdout
  tiles likely supply most of 2019, 2020 and 2023.
- Statistical precision (z-approximation, 95 %; tract clustering widens
  these by perhaps ×1.5):
  - one year, N = 1,000: about ±3 percentage points of accuracy;
  - seven years pooled (~7,000): about ±1.2 points.
- That is enough to rank models and to test "beats NMD". Going to 1,500
  would cost several hundred more tiles for little gain.
- N is a count, not a score, so fixing it after the census is still
  prespecified.

**D3. Reuse holdout_val_512.** Recommendation: yes, on three conditions
checked in step 1:
1. No evaluated model used it for training or for selection.
2. Tessera is re-sourced to the promoted source in the evaluation copy.
3. frame_2016, S1-2016 and vpp_year are topped up.

Why:
- It is the strongest held-out imagery we have, since it was built outside
  the training footprint.
- It is already VPP-filled, which protects the WEkEO quota.
- It is the cheapest route by far.

If condition 1 fails for any model, fetch new tiles for the affected years
instead. Report the holdout as its own tile role.

**D4. GPU time for sen2cor.** Recommendation: approve a cap of 25 GPU-h on
one RTX 2080 Ti-class GPU. Stop and ask if it is reached.
- That covers frame_2016 for about 1,000 new tiles plus about 400 holdout
  gaps. The 400 is an assumption until step 1 measures it.
- frame_2016 stays, per the 2026-10-05 rule.
- The existing 8 GPU-h evaluation allowance does not cover a campaign.

**D5. DES workers.** Recommendation: ask the DES team for the current
allotment before step 3. Then run dynamically up to the allotment, capped
at 6, per the 2026-07-21 rule:
- watch `DES: permits=N` and the rate per worker;
- lower the count on [408] or 429 errors;
- count the SCL-screen stream in the total.

Never copy a number from 2024.

**D6. Evaluation tiles overlapping training footprints.** Recommendation:
keep plot-year identity as the exclusion key. Prespecify "on / off any
training-tile footprint" as a reported stratum, and require the primary
conclusion to hold in the off-footprint stratum as a sensitivity check.
- **Effect on model ranking:** on-footprint pixels were seen in training
  (other years, NMD labels). That can favour models that memorise.
- **Effect on "beats NMD":** a model that memorises NMD agrees with NMD
  there, which makes "beats NMD" harder to show.
- **Why not forbid the overlap:** banning it outright would shrink every
  year, most of all 2018 and 2022 where training tiles are dense. Step 1
  measures by how much.
- If the off-footprint stratum turns out large enough, it can become the
  primary population instead. Decide that before the freeze, not after.

## Rules carried

- `VPP_SOURCE=wekeo` plus WEkEO prefetch before spectral; never pay CDSE PU
  for VPP.
- Ask for the DES allotment; never copy an old number.
- Fetch and labels stay separate.
- Declare complete only after a full key census.
- Retry loops consume options only after `rc == 0`.
- Evaluation tiles never enter training.
