# Task 0103: measured multi-year tile requirement

Measured 2026-10-08 on the data volume. This is a planning result, not a
frozen holdout, acquisition manifest or GPU authorization.

## Recommended target

Plan for the complete independent candidate pool for **2018–2024**, then
freeze the same number of fully eligible plot-years in each year before
any model scores are produced. The current upper limit is **1,902 per
year: 13,314 observations in total**. Readiness can lower this limit; no
year may be dropped or allowed a larger final quota.

The deterministic cover for all 15,425 candidates needs **2,562 additional
512 × 512 tiles**, conditional on reuse described below. That is only
97 more tiles than fetching solely for the 1,902-per-year identity sample
(2,465 tiles), while covering 2,111 further candidate observations across
the other years. This provides alternatives when some candidates fail
quality checks; 2018 itself has no spare candidates above the upper quota.
The smaller 1,000-per-year scenario needs 1,858 additional tiles.

“All years” here means years with field references and usable same-year
satellite/model sources. It does not mean only the three years consumed
by the current teacher feature tables. NFI candidates and existing satellite
tiles span all seven years; the repository's common Tessera source range
is 2018–2024. The 2025 fallback uses 2024 Tessera and is not a same-year
input. The missing 2020 LPIS cache is a crop-label concern, not a reason to
remove 2020 from this NFI forest-truth comparison.

## Measured counts

Counts below are plot-years after the conservative exclusion of all seven
teacher feature pools. “Reuse” counts observations inside verified existing
geometry, not ready-to-score observations. New-tile columns use that reuse.

| Year | Independent candidates | Geometric reuse | New tiles: 1,000/year | New tiles: 1,902/year | New tiles: all candidates |
|---|---:|---:|---:|---:|---:|
| 2018 | 1,902 | 779 | 306 | 390 | 390 |
| 2019 | 2,319 | 938 | 342 | 449 | 475 |
| 2020 | 2,325 | 1,003 | 333 | 458 | 483 |
| 2021 | 2,185 | 885 | 351 | 458 | 477 |
| 2022 | 1,976 | 1,167 | 224 | 314 | 320 |
| 2023 | 2,329 | 1,181 | 284 | 369 | 387 |
| 2024 | 2,389 | 2,287 | 18 | 27 | 30 |
| **Total** | **15,425** | **8,240** | **1,858** | **2,465** | **2,562** |

The existing v4 index provides cohort/campaign coverage. The older
`holdout_val_512` root contains 1,463 files: 1,379 passed the planner's
spectral-header, year and geometry checks; 84 were unreadable to the
non-root audit process and received no reuse credit. All 1,379 readable
legacy files lack a `tessera_source` stamp. Their spectral coverage can
potentially be reused, but their modalities and promoted Tessera source
are **not verified ready**. Previous use for model selection also requires
review. No source files were changed or permissions relaxed.

The 2,562 count assumes credited legacy and indexed coverage can be made
usable. Failed source/quality/independence checks may require additional
replacement tiles. This calculation therefore sizes a concrete geometric
plan; it is neither a guaranteed fetch-success count nor a global minimum.
The previous eight GPU-hour allowance must be re-estimated for the expanded
population at go/no-go, using measured throughput.

## Reproduction and verification

- Process commit: `ba31235c6bb5f4c2fa9d5c71929926a0023d062b`.
- Driver: `scripts/plan_nfi_tile_counts.py`; job: `k8s/nfi0103-tile-counts-job.yaml`.
- Job `nfi0103-tile-counts-v1` completed successfully, exit 0, on
  2026-10-08 at 17:21:55 UTC. Actual container time was 129 seconds.
- Data mounted read-only; CPU only; raw field identities and coordinates
  remained in the data environment. No input freeze, fetch, model forward
  pass, NMD scoring or GPU job ran.
- Exclusion union: 841 `(TractID, PlotID, Year)` observations. No negative
  or nonfinite field-volume/coordinate rows occurred in the selected years.
- Common crop: 496 × 496 pixels, including the 80 m border exclusion on
  each side of a 512-pixel tile. Placement reuses the 2024 centred lattice
  strategy, recursively splitting groups that do not fit the common crop.
- Equal-count scenarios use the same fixed identity SHA256 ranking as
  the preparer (`20261008:TractID:PlotID:Year`). Counts are independent of
  field class and model predictions.
- Seven planner tests passed, including pixel-boundary equivalence,
  crop splits, cross-year rejection, shuffle invariance, actual NPZ/parquet
  integration, malformed geometry and embedded-job source identity.
- Annual counts, exclusion totals, geometric support and zero exit status
  were checked before publishing this table.

The [aggregate execution record](nfi_fieldtruth_0103_tile_sizing_20261008.json)
contains all input hashes, planner SHA256, runtime image digest, Kubernetes
UIDs, timestamps, annual scenario details and the complete log hash.

Input freeze still waits for terminal successful `tessera-promote-v2`, its
clean promotion report, complete source/quality/exposure checks and the
final shared NMD coverage check. The verified frozen set must be presented
to Tobias before inference or joint scoring.
