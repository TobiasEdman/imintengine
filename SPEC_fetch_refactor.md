# SPEC — Fetch-layer simplification (imint/fetch.py)

**Created:** 2026-04-27
**Target repo:** `/Users/tobiasedman/Developer/ImintEngine`
**Status:** draft, pending fresh-session execution
**Estimated diff:** ~−400 LOC net in `imint/fetch.py`, ~5 import-list updates in callers, no behaviour changes

## Context

`imint/fetch.py` is 5290 LOC. A walk-through of the Sentinel-2 fetch surface
(this session, 2026-04-27) found three duplicated code clusters that have drifted
independently and now require parallel maintenance:

1. `fetch_des_data` and `fetch_copernicus_data` — ~190 near-identical lines each.
   The two functions diverge only in (a) which `_connect*` they call, (b) which
   set of band-name constants they pass to `load_collection()`, (c) the
   `dn_to_reflectance(source=…)` argument, and (d) one log-prefix string.
2. `fetch_seasonal_dates` and `fetch_seasonal_dates_doy` — same outer loop,
   different (year, window) → (start, end) derivation.
3. The openEO load-cube skeleton (`load_collection` 10 m → resample 20 m bilinear
   → resample 60 m bilinear → optional SCL nearest → merge → download gtiff →
   parse) is repeated in `fetch_des_data`, `fetch_copernicus_data`,
   `_fetch_tci_bands`, `_fetch_ai2_bands`, `_fetch_tci_scl_batch`, and `_fetch_scl`.

Per CLAUDE.md *Kodgranskningsstandard* §1, redundant code is a nolltolerans-issue.
Per global rule §6, this refactor must ship with a verification artefact (bit-likhet
mot cachad referens-tile + full pytest pass on `tests/test_fetch.py` and
`tests/test_baseline_fetch.py`).

## In scope

Three independent merges, each landable as its own commit so they can be reviewed
and reverted in isolation.

### Merge A — Unify `fetch_des_data` + `fetch_copernicus_data`

Collapse both into a single private worker `_fetch_s2_via_openeo(*, source, …)` and
re-expose `fetch_des_data` and `fetch_copernicus_data` as **thin one-line wrappers**
that pin `source="des"` / `source="copernicus"` respectively. The dispatcher
`fetch_sentinel2_data` is unchanged.

Reason for keeping the public names alive: 9 production call sites import them
directly (executors, run_*.py drivers, training pipeline). Renaming them would
churn a lot of unrelated code for no behaviour gain. The names become trivial
shims; the body lives once.

The unified worker selects:

| Aspect | source="des" | source="copernicus" |
|---|---|---|
| connection | `_connect(token=token)` | `_connect_cdse()` |
| collection id | `COLLECTION` | `CDSE_COLLECTION` |
| bands_10m | `BANDS_10M` (lowercase) | `CDSE_BANDS_10M` (uppercase) |
| bands_20m_spectral | `BANDS_20M_SPECTRAL` | `CDSE_BANDS_20M_SPECTRAL` |
| bands_60m | `BANDS_60M` | `CDSE_BANDS_60M` |
| bands_scl | `BANDS_20M_CATEGORICAL` | `CDSE_BANDS_20M_CATEGORICAL` |
| reflectance offset | `dn_to_reflectance(…, source="des")` | `dn_to_reflectance(…, source="copernicus")` |
| key-case mapping | `des_to_imint_bands(…)` (lowercase → IMINT) | identity (already uppercase IMINT) |
| log prefix | `"    Cloud fraction: …"` | `"    [CDSE] Cloud fraction: …"` |

These collapse to a small per-source config dict at the top of the worker.

### Merge B — Unify `fetch_seasonal_dates` + `fetch_seasonal_dates_doy`

Extract a tiny helper:

```python
def _seasonal_window_to_date_range(year: int, window: tuple[int, int],
                                    mode: str) -> tuple[str, str]:
    """mode='month' → (start_month, end_month); mode='doy' → (doy_start, doy_end)."""
```

Both public functions become 8-line wrappers that call the helper inside their
existing year/window loop. The month-end date logic at
[imint/fetch.py:1391-1396](imint/fetch.py:1391) is replaced by `calendar.monthrange()`
to remove the hardcoded leap-year-ignoring February-28 case.

Public function names and signatures unchanged — 8 call sites stay as-is.

### Merge C — Extract `_load_s2_cube` openEO skeleton helper

A single private helper with a band-selection contract:

```python
def _load_s2_cube(
    conn,
    *,
    projected_coords: dict,
    temporal: list,
    collection_id: str,
    bands_10m: list[str],
    bands_20m: list[str] | None = None,
    bands_60m: list[str] | None = None,
    scl_band: str | None = None,
    reduce_last: bool = False,
) -> tuple[np.ndarray, "rasterio.crs.CRS", "rasterio.Affine"]:
    """Load + merge + download + parse. Returns (raw, crs, transform)."""
```

Consumers updated:

- `_fetch_s2_via_openeo` (the unified worker from Merge A) — full 10/20/60/SCL
- `_fetch_tci_bands` — 10 m TCI only + SCL
- `_fetch_ai2_bands` — 10 m + 20 m + SCL, no 60 m
- `_fetch_scl` — SCL only (10 m reference grid is the SCL collection itself; needs care)
- `_fetch_tci_scl_batch` — TCI + SCL with `reduce_last=False` and a temporal range.
  *Note:* this one parses a tar.gz vs gtiff and extracts per-date — keep the
  per-date split outside the helper, only reuse the cube assembly.
- `_fetch_scl_batch` — same caveat as `_fetch_tci_scl_batch`. Possibly out of scope
  for Merge C if the tar.gz handling makes the helper signature ugly. Decide
  during execution.

Each consumer keeps its own DN→reflectance / RGB-composite / FetchResult-build
logic — the helper only abstracts the openEO + parse-gtiff stage, not the
post-processing.

## Out of scope

- **Public API names** (`fetch_des_data`, `fetch_copernicus_data`,
  `fetch_sentinel2_data`, `fetch_seasonal_dates`, `fetch_seasonal_dates_doy`,
  `fetch_seasonal_image`, `_fetch_scl`, `_fetch_scl_batch`) — preserved.
  Nine call sites import these directly; renaming is pure churn.
- **`tile_fetch.py`** — orchestration (semaphores, racing CDSE/DES/openEO
  providers) is genuinely different. Already correctly delegates to
  `fetch_seasonal_image`. No change.
- **`scripts/fetch_unified_tiles.py`** — generators + `fetch_tile`/`refetch_tile`
  workers are scenario-specific orchestration, not fetch primitives. No change.
- **Reference-layer fetchers** — `fetch_nmd_data`, `fetch_lpis_polygons`,
  `fetch_grazing_lpis`, `fetch_grazing_timeseries`, `fetch_sjokort_data`,
  `fetch_vessel_heatmap`. Different APIs/protocols, no overlap.
- **`fetch_cloud_free_baseline` + the CoT machinery**
  ([imint/fetch.py:2929](imint/fetch.py:2929) onwards). Internal `fetch_des_data`
  call ([imint/fetch.py:3101](imint/fetch.py:3101)) keeps working via the wrapper.
- **`scripts/fetch_lucas_tiles.py`, `scripts/fetch_hormuz_timeseries.py`** —
  separate executable scripts that consume `fetch_seasonal_image`; no change.
- **No backward-compat shims for removed names.** Per CLAUDE.md
  *Kodgranskningsstandard* §Regler. Since this spec keeps the public names alive
  as wrappers, this rule is honoured by design — no public name is removed.
- **No simultaneous behaviour changes.** Logging strings, retry logic, cloud
  thresholds, grid snapping all unchanged. Pure structural refactor.

## Interface

No public-API change. After the refactor:

```python
# imint/fetch.py — unchanged exports
def fetch_des_data(date, coords, cloud_threshold=0.1, token=None,
                    include_scl=True, date_window=0) -> FetchResult: ...
def fetch_copernicus_data(date, coords, cloud_threshold=0.1,
                           include_scl=True, date_window=0) -> FetchResult: ...
def fetch_sentinel2_data(source="des", **kwargs) -> FetchResult: ...
def fetch_seasonal_dates(coords, seasonal_windows, years,
                          scene_cloud_max=50.0) -> list: ...
def fetch_seasonal_dates_doy(coords, doy_windows, years,
                              scene_cloud_max=50.0) -> list: ...
```

Internal additions (private):

```python
def _fetch_s2_via_openeo(*, source, date, coords, cloud_threshold,
                          token=None, include_scl, date_window) -> FetchResult: ...
def _seasonal_window_to_date_range(year, window, mode) -> tuple[str, str]: ...
def _load_s2_cube(conn, *, projected_coords, temporal, collection_id,
                   bands_10m, bands_20m=None, bands_60m=None,
                   scl_band=None, reduce_last=False) -> tuple[np.ndarray, CRS, Affine]: ...
```

## Dependencies — call-site map (verified 2026-04-27)

### `fetch_des_data` — 9 call sites (8 prod + tests)

- [run_full_analysis.py:81](run_full_analysis.py:81)
- [run_des_segmentation.py:113](run_des_segmentation.py:113)
- [run_des_pipeline.py:99](run_des_pipeline.py:99)
- [executors/colonyos.py:76](executors/colonyos.py:76)
- [executors/local.py:87](executors/local.py:87)
- [imint/fetch.py:3101](imint/fetch.py:3101) (internal — `fetch_cloud_free_baseline`)
- [imint/training/prepare_data.py:840](imint/training/prepare_data.py:840)
- [tests/test_fetch.py](tests/test_fetch.py) (heavy mock-based suite)
- [tests/test_baseline_fetch.py](tests/test_baseline_fetch.py) (3 `@patch` decorators)

### `fetch_copernicus_data` — 2 prod + tests

- [scripts/generate_marine_commercial_showcase.py:91](scripts/generate_marine_commercial_showcase.py:91)
- [imint/training/prepare_data.py:210](imint/training/prepare_data.py:210) (import)
- [tests/test_fetch.py](tests/test_fetch.py)

### `fetch_sentinel2_data` (dispatcher) — 3 prod + tests

- [scripts/generate_vegetationskant_showcase.py:189](scripts/generate_vegetationskant_showcase.py:189)
- [imint/training/cdse_s2.py:433](imint/training/cdse_s2.py:433)
- [imint/training/prepare_data.py:215](imint/training/prepare_data.py:215)
- [tests/test_fetch.py](tests/test_fetch.py) (dispatcher tests)

### `fetch_seasonal_dates` — 5 call sites

- [executors/seasonal_fetch.py:363](executors/seasonal_fetch.py:363)
- [scripts/test_seasonal_fetch.py:54](scripts/test_seasonal_fetch.py:54)
- [scripts/test_batch_fetch.py:387](scripts/test_batch_fetch.py:387)
- [imint/training/cdse_s2.py:278](imint/training/cdse_s2.py:278)
- [imint/training/prepare_data.py:1162](imint/training/prepare_data.py:1162)

### `fetch_seasonal_dates_doy` — 3 call sites

- [executors/seasonal_fetch.py:357](executors/seasonal_fetch.py:357)
- [scripts/gapfill_tiles.py:168](scripts/gapfill_tiles.py:168)
- [imint/training/prepare_data.py:1156](imint/training/prepare_data.py:1156)

### `_fetch_scl` — externally imported (treat as semi-public)

- [imint/training/prepare_data.py:794](imint/training/prepare_data.py:794)
- [executors/seasonal_fetch.py:418](executors/seasonal_fetch.py:418)
- [scripts/test_seasonal_fetch.py:98](scripts/test_seasonal_fetch.py:98)
- [tests/test_baseline_fetch.py](tests/test_baseline_fetch.py) (multiple `@patch`)

### `_fetch_scl_batch` — externally imported

- [imint/training/prepare_data.py:747](imint/training/prepare_data.py:747)
- [executors/seasonal_fetch.py:397](executors/seasonal_fetch.py:397)
- [scripts/test_batch_fetch.py:406](scripts/test_batch_fetch.py:406)

### `_fetch_tci_bands`, `_fetch_ai2_bands`, `_fetch_tci_scl_batch`

Internal to `imint/fetch.py` only — used by `fetch_vessel_heatmap` at lines
1983, 2007, 2009. Safe to refactor without external coordination.

### Active-jobs / running-code check

- `~/.claude/active-jobs/` contains only `README.md` — no locked paths.
- `kubectl get pods` shows no active fetch or enrich pods.
- `git status` shows no uncommitted edits to `imint/fetch.py`.

→ Per global rule §3, no fetch code is currently executing. Refactor permitted.

## Data & state

**Read:** existing test fixtures, recent reference tile (cached `.npz` from
`/data/unified_v2/` — pick one tile with both DES and CDSE coverage for
bit-likhet check).
**Write:** `imint/fetch.py` only.
**Mutate:** the one internal call site at line 3101 inside the same file
(unchanged behaviour through the wrapper).

## Failure modes & verification

| Scenario | Expected | How verified |
|---|---|---|
| All `tests/test_fetch.py` mocks (DES, CDSE, dispatcher, STAC, projected coords, cloud-too-high, missing date) | All pass | `pytest tests/test_fetch.py -v` |
| `tests/test_baseline_fetch.py` `@patch("imint.fetch._fetch_scl")` keeps working | All pass | `pytest tests/test_baseline_fetch.py -v` |
| Bit-likhet på cachad referens-tile (DES) | Identisk array `np.array_equal()` på `bands["B02"]`–`bands["B12"]` + `scl` mot pre-refaktor `.npz` | `scripts/verify_fetch_refactor.py <tile.npz>` (one-shot, gitignored) |
| Bit-likhet på cachad referens-tile (CDSE) | Same | Same |
| All call sites importable | `python -c "from imint.fetch import fetch_des_data, fetch_copernicus_data, fetch_sentinel2_data, fetch_seasonal_dates, fetch_seasonal_dates_doy, fetch_seasonal_image, _fetch_scl, _fetch_scl_batch"` | one-liner |
| `fetch_vessel_heatmap` end-to-end smoke | Runs without import error on a known tile | `python -m imint.fetch_vessel_heatmap_smoke` (or existing showcase script) |
| Static type / lint | No new pyflakes/mypy regressions | `ruff check imint/fetch.py` (or whatever lint is configured) |
| `_seasonal_window_to_date_range(2024, (1, 2), mode='month')` | Returns `("2024-01-01", "2024-02-29")` (leap-year correct via `calendar.monthrange`) | unit test `tests/test_fetch.py` (new, ~10 lines) |

## Constraints

**Hard:**

- **No public-API rename.** All eight currently-exported names keep their
  current signature.
- **No behaviour change.** Same logs (modulo de-duplication), same retry logic,
  same grid snapping, same error types and messages.
- **No simultaneous merge.** Three commits, in order A → B → C, each with its
  own verification.
- **No new dependencies.** `calendar` is stdlib; everything else is already imported.
- **No backward-compat shims for code removed in this refactor.** None is removed
  externally — only the internal duplicate bodies disappear, replaced by the
  unified worker / helpers.
- **Co-Authored-By trailer** on every commit per global rule §7.

**Soft:**

- Keep import order in callers stable (avoid re-formatting their import lists).
- Match existing code style in `imint/fetch.py` — 4-space indent, type hints on
  public functions, `from __future__ import annotations`.

## Tradeoffs accepted

- **Two thin wrappers (`fetch_des_data`, `fetch_copernicus_data`) that simply
  pin `source=…` and forward** introduce one extra function-call frame in the
  hot path. Cost: nanoseconds per fetch, fetch-bound is on the network anyway.
  Benefit: no public-API churn, 9 call sites untouched.
- **Keeping `_fetch_scl_batch` and `_fetch_tci_scl_batch` partly separate from
  `_load_s2_cube`.** The tar.gz vs gtiff dispatch + per-date split makes a fully
  unified helper signature ugly. Better to share the cube-assembly stage and
  keep the parse-stage as-is.
- **Keeping `fetch_seasonal_image` as-is.** It already correctly delegates to
  `fetch_sentinel2_data`. No simplification available without churning callers.
- **Not touching `tile_fetch.py`'s `_fetch_single_scene`.** Looks like overlap
  but is genuinely orchestration over the unified fetch primitives.

## Execution hints

**Files to modify:**

- [imint/fetch.py](imint/fetch.py) — only file touched
- [tests/test_fetch.py](tests/test_fetch.py) — add ~10 lines testing
  `_seasonal_window_to_date_range`; verify all existing tests still pass
  unmodified
- (one-shot, gitignored) `scripts/verify_fetch_refactor.py` — diff cached
  reference-tile arrays before/after refactor, then delete

**Files NOT to modify:**

- Any caller listed in §Dependencies — public API is preserved
- `imint/training/tile_fetch.py`
- `scripts/fetch_unified_tiles.py`, `scripts/fetch_lucas_tiles.py`, `scripts/fetch_hormuz_timeseries.py`

**Suggested commit sequence:**

1. **Commit A** — Merge A: unify `fetch_des_data` + `fetch_copernicus_data` via
   `_fetch_s2_via_openeo`. Verify: full `tests/test_fetch.py` + bit-likhet on
   one DES tile + one CDSE tile.
2. **Commit B** — Merge B: extract `_seasonal_window_to_date_range` and
   `calendar.monthrange()` fix. Verify: full pytest run + 4 new unit-test cases
   covering month-31, month-30, Feb-leap, Feb-non-leap, doy.
3. **Commit C** — Merge C: extract `_load_s2_cube` and re-route consumers.
   Verify: full pytest run + bit-likhet on one tile via every code path
   (`fetch_des_data`, `fetch_vessel_heatmap` TCI mode, `fetch_vessel_heatmap`
   AI2 mode, `_fetch_scl` standalone).
4. (Optional) **Commit D** — drop the verification helper script if it lives
   in-tree.

Each commit ends with:

```
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

**Rollback strategy:**

- Each commit is structurally independent. `git revert <sha>` returns to the
  previous state without touching the others.
- Public-API preservation means no caller needs to roll back.

## Open questions

1. **Reference-tile choice for bit-likhet check.** Need one `.npz` that has been
   fetched both via DES and via CDSE, ideally from a recent unified_v2 sample.
   Executing session should pick from `/data/unified_v2/` and verify it has
   both `source="des"` and `source="copernicus"` historical fetches available.
2. **Should `_fetch_scl` and `_fetch_scl_batch` be promoted to public** (drop
   underscore) given they're imported in 4 places outside `imint/fetch.py`?
   Defer — orthogonal to this refactor.
3. **Should `_fetch_scl_batch` switch from tar.gz to gtiff** to match
   `_fetch_tci_scl_batch`? Out of scope; behaviour-preserving refactor only.
4. **Type-narrowing on `source` parameter.** Use `Literal["des", "copernicus"]`
   or accept any `str` and raise on unknown? Match the existing convention in
   `fetch_sentinel2_data` (currently any-`str`, raises in the `else` branch).

---

**Reference paths cited in this spec:**

- Subject of refactor: [imint/fetch.py](imint/fetch.py)
- Test suite: [tests/test_fetch.py](tests/test_fetch.py),
  [tests/test_baseline_fetch.py](tests/test_baseline_fetch.py)
- Repo conventions (Kodgranskningsstandard, verifiering): [CLAUDE.md](CLAUDE.md)
- Global rules (running-code, attribution): `~/.claude/CLAUDE.md`
