# SPEC — Meteorology-matched forest S2 POC

**Created:** 2026-08-20
**Target repos:** `/Users/tobiasedman/Developer/ImintEngine` and
`/Users/tobiasedman/Developer/space-datalab-metafilter`
**Status:** approved for fresh-session execution

## Context

Build a proof of concept that finds comparable Sentinel-2 observations from
different years for forest-dominated locations across Sweden. Candidate scenes
must be cloud-free and fall between 15 May and 15 June; among valid S2 passes,
meteorological similarity determines the match. The comparison covers NDVI,
VPP phenological phase, and multispectral difference.

ImintEngine owns tile selection, Sentinel-2/VPP/NMD fetching, coregistration,
analysis, and reporting. `space-datalab-metafilter` owns meteorological feature
generation and analog ranking and must be imported as a library, not copied.

## In scope

- Make `space-datalab-metafilter` installable and expose a small public API.
- Add meteorological analog ranking to that public API.
- Pin ImintEngine to the exact metafilter git commit used by the POC.
- Select ten 512×512 tiles, each at least 80% NMD forest, geographically
  stratified from southern to northern Sweden.
- Exclude NMD temporary non-forest / clearcut classes from forest dominance.
- Use 2019 as the reference year and match independently against 2020–2024.
- Restrict all acquisition searches to 15 May–15 June of the relevant year.
- Use ImintEngine's existing cloud-aware S2 selection and fetch pipelines.
- Compare forest-mask-only NDVI, VPP phase, and multispectral differences.
- Persist a complete run manifest, tabular summaries, arrays, and figures.

## Out of scope

- Training or evaluating an ML model.
- Changing the existing 13-PR DES metafilter wave.
- Copying metafilter ERA5 or analog code into ImintEngine.
- New openEO, CDSE, NMD, or VPP fetch implementations.
- Mutating existing national 512 datasets or caches.
- Automatically launching a national/bulk fetch.
- Treating meteorological similarity as evidence of unchanged forest state.

## Architecture

### 1. Metafilter library boundary

Add an installable `metafilter` package with a narrow public API. Production
logic may initially delegate to existing modules, but ImintEngine must never
import from `scripts.*` or `utils.*`.

Proposed API:

```python
from metafilter import (
    AnalogMatch,
    AnalogModel,
    DailyMeteorology,
    calculate_daily_metrics,
    fetch_daily_meteorology,
)

daily: DailyMeteorology = fetch_daily_meteorology(
    bbox_wgs84=bbox,
    date_start="2019-04-15",
    date_end="2019-06-15",
    backend="open-meteo",
)

model = AnalogModel(
    features=(
        "gdd_prev30d_c",
        "precip_prev30d_mm",
        "precip_prev7d_mm",
        "swvl1_prev30d_mean",
        "ssrd_prev30d_mj_m2",
    ),
    metric="mahalanobis",
)
model.fit({year: frame, ...})
matches: list[AnalogMatch] = model.query(
    reference_year=2019,
    reference_date="2019-06-01",
    candidate_dates=[...],
)
```

Requirements:

- Use pooled feature standardisation fitted only from the supplied run data.
- Handle correlated variables with a regularised Mahalanobis covariance.
- Reject rows with missing required features; report reasons and counts.
- Return component feature values and normalized deltas for auditability.
- Use Open-Meteo for the POC so no CDS credentials are required.
- Fetch at least 30 days before 15 May so strict 30-day windows are populated.
- Packaging must use `pyproject.toml`; dependencies must be minimal and split
  from visualization / Sentinel / openEO extras where practical.
- Preserve legacy CLI behavior and legacy filter JSON compatibility.

### 2. ImintEngine orchestration

Add a standalone experiment command rather than a production analyzer:

```bash
python scripts/run_meteo_analog_forest_poc.py \
  --reference-year 2019 \
  --candidate-years 2020 2021 2022 2023 2024 \
  --window 05-15 06-15 \
  --tiles 10 \
  --size-px 512 \
  --min-forest-fraction 0.80 \
  --fetch-source des \
  --output-dir outputs/meteo_analog_forest_poc
```

The command must be resumable and idempotent. Each expensive result is cached
under the run directory and recorded in `manifest.json`.

## Selection pipeline

### Tile selection

1. Read centers from an existing national 512 tile inventory or ledger.
2. Read/fetch NMD through existing ImintEngine helpers.
3. Define forest as NMD 111–117 and 121–127.
4. Require forest fraction >= 0.80 after nodata exclusion.
5. Divide Sweden into ten south-to-north latitude strata and select one valid
   tile per stratum. Prefer spatial separation and stable NMD coverage.
6. Persist the selected centers, bbox, forest fraction, NMD histogram, and
   source inventory in `tiles.csv`; never silently resample on rerun.

### Reference and candidate dates

For each tile and year:

1. Call `optimal_fetch_dates(..., mode="era5_then_scl")` for 15 May–15 June.
2. Preserve the SCL hard gates for cloud, snow, and AOI coverage.
3. For 2019, choose the cloud-valid date closest to 1 June as the reference.
4. For each candidate year, rank its cloud-valid S2 dates by metafilter analog
   distance to the 2019 reference-day meteorology.
5. Use absolute calendar-day displacement from 1 June only as a tie-breaker.
6. Record rejected dates, stage counts, cloud fractions, meteorological
   vectors, normalized feature deltas, and final scores.
7. If no valid date exists, mark the tile/year missing; do not relax cloud,
   snow, coverage, forest, or date-window constraints automatically.

### Sentinel-2 fetch

- Use existing ImintEngine fetch APIs only; no custom openEO graph.
- Use DES for the POC unless unavailable and an explicitly approved fallback
  is configured.
- Fetch all bands needed for B02/B03/B04/B08/B11/B12 comparisons.
- Use 512×512 final tiles on the national NMD 10 m lattice.
- Apply existing M1 grid snap and M2 inter-frame coregistration. Estimate the
  shift on B04 and apply one shift to the full band stack; never per-band.
- Preserve cloud/SCL and valid-coverage masks for both dates.

### VPP

- Use the existing WEkEO VPP path and cache only.
- Set `VPP_SOURCE=wekeo` and use `VPP_WEKEO_DIR`; never fall back to CDSE PU.
- Compare annual start/peak/end-of-season milestones and express each selected
  S2 acquisition as days relative to those milestones.
- Missing VPP is a visible per-tile/year failure, not a paid fallback.

## Comparison outputs

All pixel summaries are computed only where both acquisitions are valid,
cloud-free, and within the stable NMD forest mask.

Per matched pair:

- NDVI reference, candidate, signed difference, absolute difference, median,
  interquartile range, and valid-pixel fraction.
- Per-band signed and relative differences for B02/B03/B04/B08/B11/B12.
- Spectral angle mapper distance per pixel and aggregate distribution.
- VPP start/peak/end-of-season shifts and each acquisition's relative phase.
- Coregistration offset and post-coregistration residual.
- Meteorological distance and per-feature normalized delta.

Artifacts:

```text
outputs/meteo_analog_forest_poc/<run-id>/
  manifest.json
  tiles.csv
  matches.csv
  summary.csv
  arrays/<tile>/<year>.npz
  figures/<tile>_<year>.png
  report.html
  logs/
```

The report shows, per tile: Sweden location, forest/NMD mask, true-color image
pair, NDVI maps and difference, spectral-angle map, meteorology feature table,
VPP phase, acquisition dates, cloud/coverage metrics, and validity warnings.

## Data and state

- Read existing national 512 inventory/ledger, NMD sources/cache, VPP WEkEO
  cache, S2/STAC/SCL caches, and metafilter's Open-Meteo cache.
- Write only below the explicit output directory and normal cache locations.
- Never mutate source tiles, national datasets, or existing experiment runs.
- Use a unique run ID containing UTC timestamp and git SHAs for both repos.
- Manifest records both repo SHAs, package version, command, environment,
  thresholds, selected tiles/dates, input hashes, fetch source, and failures.

## Dependencies

- ImintEngine must depend on metafilter by an exact immutable git SHA during
  the POC; no branch-name dependency.
- Do not add duplicate scientific dependencies already provided transitively.
- Metafilter must remain independently testable and usable outside ImintEngine.
- Existing ImintEngine fetch, coregistration, NMD, and VPP APIs must not break.

## Failure modes and verification

| Scenario | Expected behavior | Verification |
|---|---|---|
| No cloud-free S2 date | Mark missing; no threshold relaxation | Unit test with empty FetchPlan |
| Missing meteorology feature | Candidate rejected with named features | Unit test with NaN column |
| Singular covariance | Regularize and return finite distance | Collinear-feature unit test |
| Missing VPP cache | Fail visibly; no CDSE fallback | Routing test with cache miss |
| Fetch interruption | Resume from persisted stage artifacts | Kill/restart smoke test |
| S2 grid drift | M1+M2; residual below configured guard | Dot/COM and real two-date check |
| Low forest content | Tile rejected before any S2 fetch | Synthetic NMD test |
| Cloud pixels in comparison | Excluded from all pixel metrics | Masked-array unit test |
| Empty common valid mask | Pair marked invalid | Unit test |
| Repeated run | Same tile/date manifest; cached stages reused | Idempotence smoke test |

Verification sequence:

1. Metafilter unit tests and import smoke test in a clean environment.
2. ImintEngine unit tests with all external calls mocked.
3. One-tile, two-year live smoke run; inspect imagery/coregistration visually.
4. Report exact fetch duration, failures, and backend concurrency.
5. Ask for explicit approval before running all ten tiles and six years.
6. Ten-tile run must yield a non-empty manifest, exactly ten fixed tile rows,
   and one explicit success/missing result per tile and candidate year.

## Constraints

- Work in isolated worktrees; the active ImintEngine benchmark branch is dirty
  and off-limits.
- Reuse repo-owned fetch pipelines; no new openEO calls.
- Do not exceed the currently approved DES server worker allotment. Ask the
  user/team immediately before the live fetch and set workers accordingly.
- VPP is WEkEO cache-only.
- Persist every expensive artifact as it is produced.
- Tests and report must distinguish cloud validity, meteorological similarity,
  date proximity, VPP phase, and observed spectral difference.
- A meteorologically similar pair is not automatically a no-change ground-truth
  pair; results are descriptive and retain all caveats.

## Tradeoffs accepted

- Reference date is the cloud-valid 2019 pass closest to 1 June; meteorology
  ranks candidate-year dates and calendar proximity only breaks ties.
- Mahalanobis is preferred over Euclidean because precip and soil moisture are
  correlated; regularization handles small candidate populations.
- Ten geographically stratified tiles provide a national POC without claiming
  statistical representativeness.
- VPP is compared as phenological timing/phase, not as an S2-like raster.
- The first implementation is an experiment script, not a registered analyzer.

## Execution hints

Likely metafilter changes:

- `pyproject.toml`
- `metafilter/__init__.py`
- `metafilter/meteorology.py`
- `metafilter/analog.py`
- focused packaging/analog tests

Likely ImintEngine changes:

- dependency metadata / lockfile
- `scripts/run_meteo_analog_forest_poc.py`
- a small experiment module under `imint/experiments/`
- mocked unit tests and report renderer tests
- optional K8s one-tile and approved ten-tile job manifests

Use separate logical commits. Review fixes are amended into their introducing
commit rather than stacked. Commit messages stay terse, with one-line
`Verified-by:` and model-accurate `Co-Authored-By:` trailers.

## Rollback

- Delete the isolated worktrees/branches and POC output directory.
- Metafilter packaging and ImintEngine integration remain separate commits and
  can be reverted independently.
- No existing dataset requires rollback because the POC never mutates it.

## Open questions deferred to execution

- Exact existing national 512 inventory/ledger path in the execution
  environment.
- Current DES server worker allotment immediately before live fetching.
- Whether all ten selected tile/year VPP products are already present in the
  WEkEO cache; fill gaps only through the approved WEkEO prefetch path.
- Final immutable metafilter git SHA after its packaging commit is reviewed.
