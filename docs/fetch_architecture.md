# Tile Fetch Architecture — API Concurrency Control

> **2026-06 update.** The refetch/repair path now uses a lazy
> STAC→ERA5→SCL date selector + a per-tile CDSE openEO tile-graph for
> the spectral fetch. See the new **"Refetch date selection"** and
> **"CDSE tile-graph spectral"** sections below. The per-scene race-pool
> described further down is retained as the fallback path. DES concurrency
> was raised 3→6.

## Data Sources

Three Sentinel-2 access paths are used:

- **CDSE openEO** (`openeo.dataspace.copernicus.eu`, API 1.2): the primary
  spectral backend for the tile-graph. Hard per-account limit of **1
  concurrent connection** — over it returns `[429] Too Many Requests`.
  Monthly credit pool; `[402] PaymentRequired` when drained (see credit
  guard below).
- **DES openEO** (`openeo.digitalearth.se`, API 1.1, via RISE): SCL
  screening + spectral fallback. Higher concurrency. Note: its
  `aggregate_spatial` has a geopandas-dtype server bug, so AOI cloud
  screening uses the pixel `scl_stack_screen` path, not aggregate.
- **DES STAC** (`explorer.digitalearth.se/stac`, `s2_msi_l2a`): cheap
  anonymous catalogue of real acquisition dates + granule cloud cover.

## Concurrency Model

Global `AdaptiveSemaphore`s in `imint/training/tile_fetch.py` enforce hard
per-backend limits across all tiles, frames, and threads in the process:

```python
_DES_SEMAPHORE        = AdaptiveSemaphore(initial=6, max_permits=6)  # DES openEO
_CDSE_SEMAPHORE       = AdaptiveSemaphore(initial=10, max_permits=20) # CDSE SH Process (PU)
_CDSE_OPENEO_SEMAPHORE = AdaptiveSemaphore(initial=1, max_permits=1)  # CDSE openEO: HARD 1-conn
```

CDSE openEO is locked single-flight (the backend allows only 1 concurrent
connection per account). Both the season-SCL-era code and the tile-graph
spectral route their CDSE calls through `_CDSE_OPENEO_SEMAPHORE` (via the
`_cdse_single_flight` contextmanager in `openeo_tile_graph.py`) so 6
workers queue cleanly instead of triggering a 429 storm.

## Per-Scene Fetch Strategy (fallback path)

> This race-pool is the FALLBACK, used when the tile-graph path above is
> off or fails for a slot. The primary refetch path is the lazy
> STAC→ERA5→SCL selector + CDSE tile-graph documented above.

Each temporal frame for each tile calls `_fetch_single_scene()` which:

1. Queries STAC catalog for candidate dates (sorted by cloud cover)
2. Submits top 3 candidates to DES (3 threads, blocked by `_DES_SEMAPHORE`)
3. Submits best candidate to CDSE (1 thread, blocked by `_CDSE_SEMAPHORE`)
4. First successful result wins — remaining futures are cancelled

```
Per scene:
  ┌─ DES candidate 1 ──┐
  ├─ DES candidate 2 ──┤── first success wins
  ├─ DES candidate 3 ──┤
  └─ CDSE candidate 1 ─┘
  (blocked by global semaphores if other scenes are in flight)
```

## Refetch date selection — lazy STAC→ERA5→SCL (primary)

`repair_to_canonical_layout` (refetch path) selects ONE cloud-clean date
per temporal frame with the **minimum number of expensive calls**, by
ordering the filters cheap-first and the expensive SCL verification lazily:

```
Per tile (cheap pre-rank — no openEO):
  1. DES STAC       (1 anonymous HTTP)  → real S2 passes + granule eo:cloud_cover
  2. ERA5 overpass  (1 Open-Meteo HTTP) → cloud at 10-11h local (S2 pass ~10:30)
     → drop overcast-at-overpass, rank ascending by (overpass, granule)
     → rank_stac_era5_candidates()  in optimal_fetch.py

Per slot (lazy expensive verify):
  for date in ranked_candidates (best-first):
      verify_aoi_scl(date, backend="des")   # 1 openEO SCL call, [date,date+1)
      if AOI cloud ≤ threshold:  pick it, STOP
  → usually 1 SCL call/slot (top STAC+ERA5 candidate is AOI-clean too)
```

Why this order: STAC + ERA5 are free (HTTP, no openEO/credits); SCL is the
expensive openEO call. Ranking with the free signals and verifying SCL
best-first means we pay for SCL only on the most-promising date(s) — not
the whole window. SCL runs on DES so CDSE's single connection stays free
for the spectral fetch.

ERA5 cloud is the Open-Meteo **overpass-time** mean (10-11h local), not a
daily mean — a date can be clear in the afternoon yet overcast at 10:30
when Sentinel-2 passes. `DEFAULT_ATMOSPHERE_RULES["overpass_cloud_max_pct"]`
(50%) drops the obviously-overcast; the AOI-SCL step is the precise cut.

## CDSE tile-graph spectral (primary)

Once a date is chosen per slot, all 4 slots' spectral are fetched in ONE
openEO process graph (`fetch_tile_at_specific_dates` →
`fetch_tile_all_slots_cdse_openeo` in `openeo_tile_graph.py`): per-slot
`load_collection` (crs=3006) + reduce("t","first") + `rename_labels`
("s{slot}_{band}") merged via `merge_cubes` → one `download`. The
(4×6, H, W) cube is parsed back per slot. DES races as opportunistic
secondary; the race-bug fix (break + 180 s timeout +
`shutdown(wait=False)`) prevents a DES hang from blocking tile completion.

**Credit guard:** the first `[402] PaymentRequired` marks CDSE openEO dead
for the process (`is_source_dead`), routing everything to DES with no
wasted HTTP. A pod restart clears the mark, so a monthly credit reset
auto-recovers (CronJob `refetch-credit-reset-restart` kicks the pod on the
1st of each month).

Enabled via `IMINT_USE_TILE_GRAPH=1`. On any tile-graph failure the per-slot
race-pool (below) covers the tile.

## Tile Pipeline

Each tile has multiple scenes to fetch:
- 4 temporal frames (autumn, spring, summer, late summer)
- 1 background frame (2016 summer)
- Auxiliary channels (VPP, DEM) — separate APIs, no semaphore needed

The outer `--workers N` parameter controls how many tiles are processed
in parallel. With global semaphores, N can be set higher than 4 without
risk — the semaphores will naturally queue excess requests.

```
workers=4:
  Tile A: frame 0 ──→ frame 1 ──→ frame 2 ──→ frame 3 ──→ bg frame
  Tile B: frame 0 ──→ frame 1 ──→ ...
  Tile C: frame 0 ──→ ...
  Tile D: frame 0 ──→ ...
  │
  └─ All share: 3 DES slots + 1 CDSE slot (global semaphores)
```

## Why Not Adaptive Backoff?

Earlier versions used adaptive concurrency that reduced workers on
rate-limit detection (tile >60s → reduce workers). This caused problems:

1. 512px tiles naturally take longer (4x pixels) — falsely triggered backoff
2. Once reduced to 1 worker, recovery was too slow (needed 20 consecutive OK)
3. The first 3 tiles always triggered backoff, making the whole run single-threaded

Global semaphores are simpler and correct: the API limits are fixed (3 DES,
1 CDSE), so enforce them directly instead of guessing from response times.

## Configuration

| Parameter | Location | Default | Notes |
|-----------|----------|---------|-------|
| DES concurrency | `tile_fetch.py` `_DES_SEMAPHORE` | 6 | Raised 3→6 (2026-06) |
| CDSE openEO concurrency | `tile_fetch.py` `_CDSE_OPENEO_SEMAPHORE` | 1 | HARD per-account limit; single-flight |
| CDSE SH-Process concurrency | `tile_fetch.py` `_CDSE_SEMAPHORE` | 10 | PU pool (separate from openEO credits) |
| Tile workers | `--workers` CLI arg | 6 | Semaphores queue excess |
| ERA5 overpass cloud ceiling | `DEFAULT_ATMOSPHERE_RULES` | 50% | Open-Meteo 10-11h prefilter |
| AOI-SCL threshold | `--max-aoi-cloud` | 0.10 | Precise per-tile cut (lazy verify) |
| Tile-graph spectral | `IMINT_USE_TILE_GRAPH` env | off | `=1` → CDSE one-call-per-tile fetch |

## Tile Size

Tile size is configurable via `--tile-size-px`:
- 256 (default): 2560m × 2560m at 10m resolution
- 512: 5120m × 5120m at 10m resolution (for 448px training)

The module-level constants `TILE_SIZE_PX` and `TILE_SIZE_M` are overridden
at startup when `--tile-size-px` is specified. All downstream code
(bbox computation, NMD rasterization, spectral fetch) reads these constants.

## Output

Tiles are saved as `.npz` files with keys:
- `spectral`: (T×6, H, W) float32 reflectance [0, 1]
- `frame_2016`: (6, H, W) float32 background frame
- `temporal_mask`, `doy`, `dates`: temporal metadata
- `bbox_3006`: (4,) bounding box in SWEREF99 TM
- Auxiliary: `height`, `volume`, `dem`, `vpp_*`, etc.
