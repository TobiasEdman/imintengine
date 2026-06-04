# Tile Fetch Architecture

> **2026-06-04 rewrite.** The per-scene race-pool that previously lived
> in `_fetch_single_scene` has been retired. The new architecture is a
> **unified per-slot dispatcher** (`imint.training.fetch_spectral.fetch_spectral`)
> that the orchestrator calls once per (slot, backend) — no backend race,
> no cross-backend fallback inside a slot, no separate "tile-graph vs
> race-pool" path. Silent failures (the
> `except Exception: pass` family) have been replaced with visibility
> markers so every per-slot rejection is grep-able from the pod log.
>
> Earlier versions of this doc described the race-pool as the primary
> path and the lazy STAC→ERA5→SCL selector as a layered addition. Both
> were folded into the unified flow; the doc below reflects the
> post-refactor reality.

## Data Sources

Three Sentinel-2 access paths, plus an AWS-hosted STAC for the pre-2018
gap, plus an off-line Sen2Cor backfill for the data DES openEO cannot
serve:

| Backend | Endpoint | Role | Concurrency | Billing |
|---|---|---|---|---|
| **CDSE SH Process** | `services.sentinel-hub.com/api/v1/process` | Spectral fetch. Two-stage (SCL prescreen + spectral) in one HTTP cycle via `cdse_s2.fetch_s2_scene`. | 10 (`_CDSE_SEMAPHORE`) | Processing Units (30 k / month) |
| **CDSE openEO** | `openeo.dataspace.copernicus.eu` | Spectral fetch via `fetch_tile_at_specific_dates`. | **1** (`_CDSE_OPENEO_SEMAPHORE`) — HARD per-account ceiling | Credit pool (`402 PaymentRequired` when drained) |
| **DES openEO** | `openeo.digitalearth.se` (RISE) | Spectral fetch + SCL screening (`verify_aoi_scl`). | 6 (`_DES_SEMAPHORE`) | Free, rate-limited |
| **DES STAC** | `explorer.digitalearth.se/stac` (`s2_msi_l2a`) | Cheap anonymous catalogue of real acquisition dates. Catalogue starts 2018. | (HTTP, no semaphore) | Anonymous |
| **earth-search STAC** | `earth-search.aws.element84.com/v1` (`sentinel-2-l2a`) | AWS-hosted L2A catalogue, full S2 history including pre-2018. Pluggable via `stac_backend="earth-search"`. | (HTTP, no semaphore) | Anonymous |
| **Sen2Cor L1C→L2A** | offline, `scripts/sen2cor_pipeline/` | Backfill for slots DES openEO cannot reach (year=2018 slot 0 / autumn 2017). Writes directly into the temporal stack via `_write_temporal_slot`. | k8s job (GPU pod, 4 workers) | Compute only |

## The Unified Per-Slot Dispatcher

`imint/training/fetch_spectral.py` exports the single entry point:

```python
def fetch_spectral(
    bbox_3006, coords_wgs84, date_str,
    *, backend, size_px, cloud_threshold,
) -> np.ndarray | None
```

Returns `(6, size_px, size_px)` float32 reflectance or `None`. `None`
means **"this date is not usable for this slot via this backend"** —
cloud-rejected, no scene available, backend marked dead, or the call
errored (errors are logged visibly, not swallowed). The orchestrator
treats every `None` the same: advance to the next ranked candidate.

Three adapters dispatch on `backend`:

```
fetch_spectral
├── backend="cdse"        → _fetch_via_cdse_sh_process
│                            (cdse_s2.fetch_s2_scene — two-stage)
├── backend="cdse-openeo" → _fetch_via_openeo
│                            (verify_aoi_scl → fetch_tile_at_specific_dates)
└── backend="des"         → _fetch_via_openeo
                             (verify_aoi_scl → fetch_tile_at_specific_dates)
```

Every adapter fast-fails on `is_source_dead(backend)` at entry — a
credit guard that prevents the first per-process credit/PU-exhaustion
error from being followed by hundreds of doomed calls.

### Why verify + fetch lives in the adapter, not the orchestrator

Per-backend verify nuance differs: SH Process pairs SCL prescreen +
spectral fetch in one HTTP cycle (no point splitting them); openEO
backends benefit from a cheap explicit `verify_aoi_scl` call before the
expensive spectral fetch (skip the doom cycle on cloudy candidates).
Folding both shapes into one adapter contract keeps the orchestrator
backend-agnostic and the design contract enforceable in one file.

## Orchestrator Flow — `repair_to_canonical_layout`

`scripts/fetch_unified_tiles.py::repair_to_canonical_layout` is the
single per-tile entry. The flow:

```
Per tile (once):
  1. Pick primary_backend = first --sources token that is supported
     AND not is_source_dead(). No primary → fail tile with
     "no_healthy_backend".

Per slot (cheap pre-rank, no openEO):
  2. rank_stac_era5_candidates(coords, slot_window,
                               stac_backend="des")
     ├── DES STAC      → real S2 passes in window
     └── ERA5 overpass → 10–11 h local cloud cover (Open-Meteo)
     → drop overcast-at-overpass (> 50 %)
     → sort ascending by overpass cloud (best-first)

  2b. (Pre-2018 fallback, primary=cdse only)
      candidates empty AND _de < "2018-01-01" AND primary == "cdse"
      → re-rank via stac_backend="earth-search"
      (For primary=des/cdse-openeo, pre-2018 is structurally
       unreachable on the openEO catalogue — suppress fallback,
       Sen2Cor backfill is the canonical path. Logged as
       [slot-no-candidates] with reason
       "pre-2018 fallback suppressed on this backend".)

Per candidate (lazy expensive verify):
  3. Mid-tile health re-check (cdse-openeo can flip dead mid-tile
     on 402 PaymentRequired) — bail slot cleanly if so.
  4. ceiling = era5_to_scl_gate(era5_overpass_pct, is_autumn)
  5. fetch_spectral(..., backend=primary_backend,
                          cloud_threshold=ceiling)
     ├── None → continue to next candidate
     └── (6, H, W) → write slot, break out of candidate loop
  6. for-else: all candidates iterated without break
     → [slot-exhausted] log, failed_slots.append(sidx)
```

**No backend race within a slot.** **No cross-backend fallback within a
slot.** The chosen backend gets the slot's full candidate budget; if it
exhausts, the slot fails. Single failure mode, single log line per
slot.

### ERA5-adaptive SCL gate

`era5_to_scl_gate(era5_overpass_pct, *, is_autumn)` in `optimal_fetch.py`
maps ERA5 overpass cloud (0–100 %) to the per-candidate SCL acceptance
ceiling that `fetch_spectral(cloud_threshold=...)` enforces:

| ERA5 overpass | Growing-season ceiling | Autumn ceiling |
|---|---|---|
| 0 %  | 0.20 | 0.30 |
| 25 % | 0.25 | 0.425 |
| 50 % | 0.30 | 0.55 |

Calibration: the floor (ERA5 = 0 %) is anchored at the pre-adaptive
static defaults — a 0.05 / 0.10 floor rejected ~96 % of candidates the
lazy chain ranked because ERA5 systematically under-reports pixel-level
SCL cloud fraction at the 5×5 km reanalysis grid. The slope opens the
gate further when ERA5 reports cloud (where ERA5↔SCL variance is
largest). Growing-season floor raised 0.15 → 0.20 on 2026-06-04 after
the visibility patches measured 11 % of failures as `verify-cloud`
rejections at SCL = 0.50–0.66 when ERA5 was reporting 0 %.

## Concurrency Semaphores

Unchanged from the previous architecture; what changed is where they
are entered. Race-pool removed → semaphores are now acquired inside the
`fetch_spectral` adapters, one acquisition per call, no cross-backend
race.

```python
_DES_SEMAPHORE        = AdaptiveSemaphore(initial=6, max_permits=6)
_CDSE_SEMAPHORE       = AdaptiveSemaphore(initial=10, max_permits=20)
_CDSE_OPENEO_SEMAPHORE = AdaptiveSemaphore(initial=1, max_permits=1)
```

CDSE openEO is locked single-flight: synchronous fetches over the
1-connection limit return `[429] max connections reached: 1` at
preflight, before any process graph runs, so adaptive ramp-up would
just bounce into 429-spam. Throughput tradeoff: ~60–120 frames/h via
this source alone — acceptable because DES carries the bulk and
CDSE-openEO is opportunistic.

DES concurrency was raised 3→6 on 2026-05-26 after CDSE openEO became
primary (single-flight) and DES needed to absorb the parallel-worker
load. The race-bug fix (`shutdown(wait=False, cancel_futures=True)` +
180 s timeout) means a DES hang no longer blocks tile completion.

## Credit Guard — `is_source_dead` / `mark_source_dead`

`imint/training/openeo_tile_graph.py` exposes a per-process source
health map. Three trigger paths mark a source dead for the rest of the
process:

| Source | Trigger | Detector |
|---|---|---|
| `cdse` (SH Process) | `403 ACCESS_INSUFFICIENT_PROCESSING_UNITS` or `HTTP 403 + processing units` | `cdse_s2._is_pu_exhausted_error` |
| `cdse-openeo`       | `402 PaymentRequired`                                                       | `openeo_tile_graph._is_payment_required_error` |
| `des`               | (none — not subject to credit/PU billing)                                   | — |

Every adapter in `fetch_spectral` checks `is_source_dead(backend)` at
entry and fast-fails (returns `None`) without an HTTP call. The
orchestrator also re-checks mid-tile so cdse-openeo flipping dead on
slot 2 doesn't fire doomed calls on slots 3+.

A pod restart clears the health map, so a monthly credit reset
auto-recovers. CronJob `refetch-credit-reset-restart` kicks the pod on
the 1st of each month.

**Current state (2026-06-04):** CDSE openEO credits exhausted; SH
Process PUs exhausted (30 k / 30 k). The active refetch job runs
`--sources des` only. Monthly reset is Jul 1.

## Silent-Path Visibility Markers

Every previously-silent rejection point now emits a grep-able marker.
The full inventory:

| Marker | Emitted by | Meaning |
|---|---|---|
| `[verify-none]`        | `fetch_spectral._fetch_via_openeo`   | `verify_aoi_scl` returned `None` (no scene available on backend for date). |
| `[verify-cloud]`       | `fetch_spectral._fetch_via_openeo`   | SCL says AOI cloud fraction > `cloud_threshold`. Logged with both fraction + ceiling. |
| `[fetch-none]`         | `fetch_spectral._fetch_via_openeo`   | `fetch_tile_at_specific_dates` returned `None` (post-verify path). |
| `[fetch-zero]`         | `fetch_spectral._fetch_via_openeo`   | Returned an array but all-zero (degenerate). |
| `[result-none]`        | `fetch_spectral._fetch_via_cdse_sh_process` | `cdse_s2.fetch_s2_scene` returned `None`. |
| `[result-zero]`        | `fetch_spectral._fetch_via_cdse_sh_process` | Returned an array but all-zero. |
| `[slot-no-candidates]` | `repair_to_canonical_layout`         | Rank returned empty candidate list. Reason: either rank-empty-in-window or pre-2018 fallback suppressed on this backend. |
| `[slot-exhausted]`     | `repair_to_canonical_layout`         | All ranked candidates iterated without success — every per-candidate reject is upstream in `fetch_spectral`. |
| `[rank:earth-search]`  | `repair_to_canonical_layout`         | Pre-2018 earth-search fallback rank failed (exception). |
| `[pu-exhausted]`       | `cdse_s2`                            | SH Process PU pool drained; this is the call that flipped `is_source_dead("cdse")`. |
| `[skip-index]`         | `scripts/refetch_affected_tiles.SkipIndex` | Load / flush / mismatch events. |

Rule: any new code path that can reject a candidate MUST emit a marker.
The orchestrator's job is to show *why* a slot failed, not just *that*
it failed.

## SkipIndex — Persistent Known-OK Set

`scripts/refetch_affected_tiles.SkipIndex` is a thread-safe persistent
set of audit tiles that prior runs confirmed are already in canonical
layout. Without it, every restart re-loads every audit tile's `.npz`
just to discover it's already OK — ~3 min wasted per restart on the
6786-tile audit list.

- **Keyed by audit-JSON basename** — different audit lists don't
  cross-contaminate. Mismatch on load → ignored (with log line), not
  silently merged.
- **Atomic flush** via tempfile + `os.replace` — pod kill mid-write
  never corrupts the file. Default flush every 100 adds; explicit
  `flush_final()` at process end.
- **Path:** `--skip-index-path <path>` CLI flag, defaults to
  `<data_dir>/.skip_index.json`. Disable with `--no-skip-index`.
- **Invalidation:** delete the JSON. The only thing that modifies a
  tile's `.npz` is a successful refetch, which leaves it canonical, so
  the index never needs invalidation in normal operation.

## Sen2Cor 2017 Backfill — the Off-Line Path

`k8s/sen2cor-slot0-2017-512-job.yaml` runs in parallel with the DES
refetch job. It exists because DES openEO's catalogue starts 2018 — the
~1900 year=2018 audit tiles whose slot 0 (autumn 2017) is empty are
structurally unreachable on DES. Sen2Cor V2.12.04 turns L1C SAFE
archives into L2A reflectance and writes them directly into the tile's
`.npz` via `_write_temporal_slot`:

```
For each year=2018 audit tile with empty slot 0:
  1. select_scenes.py --target slot:0 --year 2017
                      --audit-json $AUDIT
     → ERA5+STAC L1C lookup over autumn 2017 (Aug 15–Oct 31)
     → plan JSON {scene_id, tile_assignments[]}

  2. run_sen2cor_per_scene.py --target slot:0
     → download L1C SAFE → Sen2Cor L2A → crop to tile bbox
     → _write_temporal_slot writes:
        spectral[0:6]      = the cropped 6-band L2A frame
        dates[0]           = scene acquisition date
        doy[0]             = day-of-year
        temporal_mask[0]   = 1
        slot_0_scene       = scene_id
        slot_0_source      = "sen2cor_l1c_l2a"
        slot_0_bands       = ["B02","B03","B04","B8A","B11","B12"]
```

Both jobs write to the same `.npz` files but to *different slots*.
Atomic `_write_temporal_slot` (load → modify slice → tempfile +
`os.replace`) means concurrent writes are safe. DES handles year ≥ 2019
tiles (all 4 slots); Sen2Cor handles year = 2018 slot 0 only. Slots
1–3 of year=2018 tiles still go through DES.

The Sen2Cor pipeline supports `--target {frame_2016|slot:N}` (N ∈
0..3); the 2016 background-frame backfill is the original use, the
slot:N variant is the 2026-06-04 addition.

## Tile Pipeline (Outer)

```
workers=N:
  Tile A: rank slot0 → rank slot1 → rank slot2 → rank slot3
            ↓             ↓             ↓             ↓
          fetch_spectral × ranked-candidates per slot
  Tile B: (same, in parallel)
  ...
  └─ All share: 6 DES slots OR 10 CDSE-SH slots OR 1 CDSE-openEO slot
                (only the primary backend's semaphore is contended)
```

Each tile has up to 4 temporal frames + 1 background frame (2016
summer) + auxiliary channels (VPP, DEM — separate APIs, no semaphore
needed). `--workers N` controls outer parallelism; with global
semaphores N can be set higher than the backend's hard ceiling without
risk — excess requests queue cleanly.

## Why Not Adaptive Backoff (historical)

Earlier versions used adaptive concurrency that reduced workers on
rate-limit detection (tile > 60 s → reduce workers). This caused:

1. 512 px tiles naturally take longer (4× pixels) — falsely triggered backoff.
2. Once reduced to 1 worker, recovery was too slow (needed 20 consecutive OK).
3. The first 3 tiles always triggered backoff, making the whole run single-threaded.

Global semaphores are simpler and correct: API limits are fixed, so
enforce them directly instead of guessing from response times.

## Configuration

| Parameter | Location | Default | Notes |
|---|---|---|---|
| DES concurrency | `tile_fetch.py` `_DES_SEMAPHORE` | 6 | Raised 3→6 (2026-06) |
| CDSE openEO concurrency | `tile_fetch.py` `_CDSE_OPENEO_SEMAPHORE` | 1 | HARD per-account limit; single-flight |
| CDSE SH-Process concurrency | `tile_fetch.py` `_CDSE_SEMAPHORE` | 10 | PU pool (separate from openEO credits) |
| Tile workers | `--workers` CLI arg | 6 | Semaphores queue excess |
| Backend priority list | `--sources` CLI arg | `cdse-openeo,des` | First healthy = primary; current deploy uses `des` only |
| ERA5 overpass cloud ceiling | `DEFAULT_ATMOSPHERE_RULES["overpass_cloud_max_pct"]` | 50 % | Open-Meteo 10–11 h prefilter |
| SCL gate floor (growing) | `era5_to_scl_gate` | 0.20 | Raised 0.15→0.20 on 2026-06-04 |
| SCL gate floor (autumn)  | `era5_to_scl_gate` | 0.30 | Unchanged |
| STAC backend (per call) | `rank_stac_era5_candidates(stac_backend=...)` | `"des"` | `"earth-search"` for pre-2018 windows |
| SkipIndex path | `--skip-index-path` CLI arg | `<data_dir>/.skip_index.json` | Disable with `--no-skip-index` |

## Tile Size

Tile size is configurable via `--tile-size-px`:
- 256 (default): 2560 m × 2560 m at 10 m resolution
- 512: 5120 m × 5120 m at 10 m resolution (for 448 px training)

The module-level constants `TILE_SIZE_PX` and `TILE_SIZE_M` are
overridden at startup when `--tile-size-px` is specified. All
downstream code (bbox computation, NMD rasterization, spectral fetch)
reads these constants.

## Output

Tiles are saved as `.npz` files with keys:

- `spectral` — `(T×6, H, W)` float32 reflectance [0, 1]; per-slot via
  `_write_temporal_slot` (atomic).
- `frame_2016` — `(6, H, W)` float32 background frame.
- `temporal_mask`, `doy`, `dates` — temporal metadata.
- `bbox_3006` — `(4,)` bounding box in SWEREF99 TM.
- `slot_N_scene`, `slot_N_source`, `slot_N_bands` — provenance per slot
  (only written by Sen2Cor backfill currently; DES/CDSE fetches don't
  populate these).
- Auxiliary — `height`, `volume`, `dem`, `vpp_*`, etc.

## Diagnostic + Operational Tooling

- `scripts/diagnostics/` — read-only audit tools (audit_strategy,
  audit_funnel, probe_scl) for asking "what's broken and why" against
  the live cluster.
- `scripts/dashboards/refetch_progress/` — live HTML dashboard with ETA
  countdown for the refetch K8s job.
