# Tile Fetch Architecture — API Concurrency Control

## Data Sources

Two independent Sentinel-2 data sources are used in parallel:

- **CDSE** (Copernicus Data Space Ecosystem): Sentinel Hub Process API.
  Rate-limits aggressively — max 1 concurrent request to avoid throttling.
- **DES** (Digital Earth Sweden): openEO backend via RISE.
  Higher capacity, supports 3 concurrent requests without throttling.

## Concurrency Model

Global semaphores in `imint/training/tile_fetch.py` enforce hard limits
across all tiles, frames, and threads in the process:

```python
_DES_SEMAPHORE = threading.Semaphore(3)   # max 3 DES requests at once
_CDSE_SEMAPHORE = threading.Semaphore(1)  # max 1 CDSE request at once
```

This means: **max 3 DES + 1 CDSE = 4 API calls simultaneously**, regardless
of how many tiles or frames are being fetched in parallel.

## Per-Scene Fetch Strategy

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
| DES concurrency | `tile_fetch.py` semaphore | 3 | Hard limit, don't increase |
| CDSE concurrency | `tile_fetch.py` semaphore | 1 | CDSE throttles at >1 |
| Tile workers | `--workers` CLI arg | 4 | Can be higher, semaphores will queue |
| Candidates per scene | `max_candidates` | 3 | Top 3 lowest-cloud dates tried |
| Cloud threshold (scene) | `--cloud-max` | 40 | STAC catalog filter |
| Cloud threshold (tile) | `cloud_threshold` | 0.15 | Per-tile quality gate |

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
