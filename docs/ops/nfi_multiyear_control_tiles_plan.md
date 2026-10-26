# Multi-year NFI control tiles — plan

2026-10-08 · Claude · for Tobias (decisions) and Codex (task 0103, PR #68)

**Status (2026-10-26):** steps 0–4 are done. Step 3's top-up tiles are
aux-complete where the source allows ([3k record](../experiments/nfi_multiyear_topup_aux_census_2026-10-24.md)).
Step 4 measured per-year support under the freeze's own rules, and Tobias
set the quota at 742 plot-years per year, 2018–2024
([step 4 record](../experiments/nfi_multiyear_support_2026-10-26.md)). The
freeze (step 5) still needs its own go, after the prerequisites listed
there.

**Status (2026-10-12):** steps 0–1 are done (#71, merged as `91902bf`).
Tobias decided D2–D6 on 2026-10-11 and the same-year exclusion and the
84-file chmod on 2026-10-12 (see Decisions). Steps 2–3 have his go after
#71; their Jobs are versioned in #75, which is under review, and step 2 is
running from it. The freeze (step 5) still needs its own go. This document
records the plan and the decisions; it authorises no Job by itself.

## Goal

PR #68 should freeze a held-out NFI evaluation whose observations are spread
evenly across inventory years. At head `a67aaec`, PR #68 already enforces
explicit per-year quotas. The census now measures existing imagery across
2018–2024; modality readiness and support after exclusions remain the gates.

**Scope (decided by Tobias, 2026-10-08):**
- Evaluate every year that has training data, evenly.
- Codex computes the tile counts per year.
- This plan covers the runbook, prerequisites and decisions.

## Historical sizing before #71 (2026-10-08)

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

## Current evidence: #71 census (2026-10-12)

The authoritative dated source is the
[census report at main 91902bf](https://github.com/TobiasEdman/imintengine/blob/91902bf2cf3b8f9bef7aa0579da52d877245fbb8/docs/experiments/nfi_multiyear_holdout_census_2026-10-12.md)
and its
[JSON record](https://github.com/TobiasEdman/imintengine/blob/91902bf2cf3b8f9bef7aa0579da52d877245fbb8/docs/experiments/nfi_multiyear_holdout_census_20261012.json).

- 1,463 of 1,650 planned files exist. The census identifies the 187 not built.
- Named-tile scans completed with no training/selection use detected. This
  is evidence about names, not proof of spatial independence.
- 212 tiles overlap same-year training imagery. Tobias decided on 2026-10-12
  to exclude those tiles; the JSON records their names. The original
  “off-footprint” description was plot-based, not a spatial guarantee.
- All 1,463 TESSERA arrays lack the promoted-source stamp; 255 tiles lack
  `frame_2016`, four more have an empty frame behind a positive flag, and
  261 lack S1-2016. One TESSERA array is empty behind a positive flag.
- Twelve tiles have zero forestry channels and are excluded in step 2, not
  re-fetched.
  All tiles carry `vpp_year` matching their year; 306 lack the autumn frame.
- 84 root-owned mode-0600 files were not world-readable. Tobias decided on
  2026-10-12 to count them; Job `nfi-holdout-chmod` (versioned in #75) added
  the read bits at 19:01Z that day, mode bits only, with no content or mtime
  change. The "all files" column below is therefore the operative one.

Support after excluding the 212 same-year overlapping tiles:

| Year | Eligible (all files) | Eligible (world-readable) | Gap to 1,000 (all files) | Gap to 1,000 (world-readable) |
|---|---:|---:|---:|---:|
| 2018 | 489 | 462 | 511 | 538 |
| 2019 | 967 | 938 | 33 | 62 |
| 2020 | 1,030 | 1,003 | 0 | 0 |
| 2021 | 848 | 826 | 152 | 174 |
| 2022 | 840 | 769 | 160 | 231 |
| 2023 | 1,245 | 1,181 | 0 | 0 |
| 2024 | 1,402 | 1,282 | 0 | 0 |

These are eligible plot-year counts in the census, not final modality-qualified
freeze counts or a union with campaign support. The top-up candidates now
include **2018, 2019, 2021 and 2022**. Recompute support after step 2 exclusions,
accessibility, modality and NMD-coverage checks before fixing fetch counts.

## Steps

0. **Prerequisites.** The four formerly unversioned steps are recorded in
   merged #71 (vpp-retry5, zero-probe, ref-inventory, tessera-refetch-all).
   They are historical records, not instructions to rerun; tessera-refetch-all
   explicitly records a rejected in-place overwrite. Review #67 before using
   any campaign job as a template. A terminal promotion Job alone is not a
   readiness check: verify actual source, year, array validity and expected
   membership, with every exclusion recorded.
1. **Census complete in #71.** Use its input hashes, named-tile scan results,
   modality counts and same-year exclusion inventory. Repeat only if changed
   inputs or a revised check justify it.
2. **Bring admitted holdout tiles up to the 2024 standard,** in an
   evaluation-only copy, never in place:
   - exclude the 212 same-year overlapping tiles before copying, then exclude
     the zero-forestry tiles; apply exclusions by identity, not by adding counts;
   - ensure admitted files are readable by the evaluation identity;
   - re-source tessera to the promoted source, validate arrays and provenance,
     and declare any train/evaluation source difference;
   - fill missing/empty frame_2016 and missing S1-2016 on admitted tiles;
   - verify existing vpp_year stamps rather than blindly restamping;
   - audit the autumn-frame gaps and record which tiles meet the prespecified
     modality policy. Tobias gave the go on 2026-10-12 (after #71); the
     Jobs are in #75.
3. **Top-up fetch** for the years still short of N.
   - **Placement:**
     - Make the year a parameter; the nfi2024 manifest hardcodes 2024.
     - Exclude teacher-pool plot-years and plots already on admitted evaluation
       tiles. Reject candidate tiles overlapping same-year training imagery;
       apply D6 before selection, not after scoring.
     - Fix the cell selection with a seed before any scores exist.
   - **Pipeline:** the same as 2024:
     1. WEkEO VPP prefetch (`VPP_SOURCE=wekeo`).
     2. DES spectral fetch.
     3. Retry pass.
     4. S1, forestry, tessera (promoted source) and verified vpp_year.
        Forestry comes from the open-data mirror `/data/skg_open` (IM-022,
        #74, merged as `2e34ff9`) once its staging and parity report are
        done; tree height is not mirrored. Until then new tiles are fetched
        spectral-only and carry an aux-incomplete flag, never zero-filled
        (decided 2026-10-12; flag convention in #75).
     5. sen2cor for frame_2016.
     6. S1-2016.
     7. Full key census.
   - **Storage:** one staging root per year,
     `/data/unified_v2_512_nfi{Y}_staging`. These tiles stay out of
     training.
4. **Index v5 and re-audit.**
   - Build an exposure/reference index over cohort, nfi2024, the holdout copy
     and new years with explicit root roles. Training tiles are not eligible
     evaluation roots; apply same-year footprint exclusions before quotas.
   - Rerun the #68 exposure audit with per-year support after the modality
     and crop checks.
5. **Freeze.** PR #68 freezes with explicit year quotas. Then the go/no-go
   for inference and NMD scoring.

**What PR #68 needs first** (from Claude's review at `a67aaec`):
- support for several evaluation roots, with per-root tile roles and
  per-year strata;
- NMD-coverage filtering before quota selection;
- the same masked-frame filling at inference as training uses.

## Cost and time (historical rates; re-size after admission checks)

| Item | 2024 measured | Driver |
|---|---|---|
| DES spectral fetch | 31.6 h for 507 tiles (16–24/h), plus 4.6 h retry | new tiles in step 3 |
| sen2cor frame_2016 | ~8.3 GPU-h for 475 tiles (≈1.75 GPU-h per 100) | new tiles plus holdout gaps |
| WEkEO VPP requests | ~208 for one year; quota 500; cache wiped 2026-08-24 | new years in step 3 |
| CDSE PU | 0 | stays 0 |

## Decisions

**D1. Years.** Decided: every year that has training data, evenly. Codex
sizes it.

**D2. Equal count per year (N).** Decided by Tobias, 2026-10-11: about
1,000 per year, 2018–2024 (Codex's 2026-10-08 "new1000" sizing: about
1,858 new tiles). The census measures support in eligible plot-years. Never drop a year to meet
the target; fix the final quotas before any score exists.
- 2018 caps what is possible: 1,884 candidates before image losses. N ≈
  2,000 cannot be reached in every year.
- At ~90 % yield, 1,000 leaves a reserve in every year, and the holdout
  tiles likely supply most of 2019, 2020 and 2023.
- Under independent Bernoulli sampling, the worst-case 95 % normal
  half-width is about ±3.1 points at N = 1,000 and ±1.2 at N = 7,000.
  These are planning approximations for a single accuracy, not power for a
  paired model-minus-NMD comparison. Tract clustering can widen uncertainty;
  no design effect has been estimated here.
- Ranking or superiority requires a target effect, paired disagreement and
  tract structure in a prespecified uncertainty/power analysis. N = 1,000
  alone does not guarantee either result.
- Finalise quotas and any top-up/shortfall policy before inspecting scores.

**D3. Reuse holdout_val_512.** Decided by Tobias, 2026-10-11: reuse it
after a read-only readiness check. That check is the #71 census: the
named-tile scans found no training or selection use. It did not establish
spatial independence. Reuse stays conditional on D6, accessibility and
modality checks in the evaluation-only copy. Report the holdout as its own tile role. If later evidence shows
training or selection use for an evaluated model, re-establish an independent
population before scoring it.

**D4. GPU time for sen2cor.** Decided by Tobias, 2026-10-11: about
30 GPU-h for sen2cor frame_2016, scaling with D2. Stop and ask if it is
reached.
- The census measured 255 missing plus four empty frame_2016 arrays across
  the full holdout (the earlier 400-gap assumption is superseded). At the
  2024 rate that is about 4–5 GPU-h; the rest is for the top-up tiles.
  Recount on admitted tiles before step 2's sen2cor Job.
- frame_2016 stays, per the 2026-10-05 rule.
- The existing 8 GPU-h evaluation allowance does not cover a campaign.

**D5. DES workers.** Decided by Tobias, 2026-10-11: `--workers 6` as a
cap, run dynamically, per the 2026-07-21 rule:
- watch `DES: permits=N` and the rate per worker;
- lower the count on [408] or 429 errors;
- count the SCL-screen stream in the total.

Never copy a number from 2024.

**D6. Evaluation tiles overlapping training footprints.** Decided by
Tobias: plot-year `(TractID, PlotID, Year)` is the exclusion key and an
evaluation tile may overlap another year's training footprint (2026-10-11);
**same-year training-footprint overlap is excluded** (2026-10-12, after the
#71 census measured it). Apply
the 212-tile exclusion inventory before making the evaluation copy, enforce
the same rule for top-up tiles, and re-check before index/quota selection.
Teacher-pool plot-year exclusion remains required independently.

Other-year footprint overlap remains allowed under D6. Report it as a
separate stratum; it does not imply spatial independence or a predictable
direction of bias relative to NMD. Prespecify the off-footprint sensitivity
analysis and its support requirement before the freeze. If support is too
small for that analysis, report the limitation rather than treating the
sensitivity check as passed.

## Rules carried

- `VPP_SOURCE=wekeo` plus WEkEO prefetch before spectral; never pay CDSE PU
  for VPP.
- DES workers per D5 (cap 6, dynamic on `DES: permits=N`); never copy an
  old number.
- Fetch and labels stay separate.
- Declare complete only after a full key census.
- Retry loops consume options only after `rc == 0`.
- Evaluation tiles never enter training.
