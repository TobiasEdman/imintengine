# Pipeline Fundamentals: Grid, Data Fetching, Quality Gates & Dashboard

> Core LULC training pipeline architecture — grid system, NMD ground truth, DES data fetching, quality gates, resumability, and monitoring dashboard.

---

## Overview

**Purpose:** Train a Land Use / Land Cover (LULC) classifier for Sweden using Prithvi-EO-2.0 as frozen backbone, UPerNet decoder, NMD Level 2 as ground truth, and Sentinel-2 data from Digital Earth Sweden (DES).

**Repository:** `https://github.com/TobiasEdman/imintengine` (private)

---

## File Structure

```
imint/training/
  config.py              # TrainingConfig dataclass
  class_schema.py        # NMD-to-LULC mapping (19-class + 10-class), class weights
  sampler.py             # Geographic grid, generate_grid(), filter_land_cells(), latitude split
  dataset.py             # PyTorch Dataset with Prithvi normalization + augmentation
  prepare_data.py        # Resumable data fetching, NMD pre-filter, adaptive workers, quality gates
  evaluate.py            # mIoU, per-class IoU, confusion matrix, overall accuracy
  trainer.py             # Training loop: frozen Prithvi + UPerNet, JSON logging
  losses.py              # FocalLoss class
  dashboard.py           # Self-contained HTML dashboard + HTTP server

imint/fm/
  terratorch_loader.py   # TASK_HEAD_REGISTRY (incl. "nmd_lulc"), load_segmentation_model()
  upernet.py             # UPerNet decoder with MPS-compatible pool sizes
  prithvi_mae.py         # PrithviMAE model (bundled from HuggingFace)

imint/fetch.py           # DES/openEO fetching, STAC, SCL, NMD fetch, _to_nmd_grid()

scripts/
  prepare_lulc_data.py   # CLI entry for data preparation
  train_lulc.py          # CLI entry for training + evaluation
  run_lulc_pipeline.py   # Combined prepare + train: --background, --dashboard

data/
  sweden_land_epsg3006.json  # Sweden land polygon (EPSG:3006, from Lantmateriet sve5milj)
  lulc_full/                 # Full training data directory
    tiles/                   # Tile .npz files
    progress.json            # Completed/failed cell tracking
    prepare_log.json         # Live preparation progress (dashboard reads this)
    des_api_stats.jsonl      # DES API call statistics
    class_stats.json         # Class distribution and tile weights
    system_metrics.json      # CPU/RAM/GPU/network per-process metrics
    training_log.json        # Training progress (loss, mIoU, per-class IoU)
    nmd_prefilter_log.json   # NMD pre-filter progress
```

---

## Grid System

- Coordinate system: EPSG:3006 (SWEREF 99 TM)
- Default spacing: 10,000m (10 km) for full runs; 50-100 km for tests
- Full grid: ~4,381 cells over Sweden at 10 km
- `_to_nmd_grid()` snaps WGS84 bbox to 10m NMD grid via `floor()/ceil()`, giving tiles ~257x262 pixels

### Fetch Parameters

- `fetch_pixels`: 256 (from DES at 10m = 2,560m x 2,560m)
- `patch_pixels`: 224 (training crop — random crop for augmentation)
- The 256-224 difference = free augmentation (32-pixel offset variation per epoch)

### Geographic Split

- Latitude-based in `sampler.py`: train / val / test bands
- At 100km grid: ~27 train, 8 val, 8 test (of ~43 land cells)

### Temporal Strategy

- Years: `["2019", "2018"]` (DES has no data for 2017; 2019 prioritized)
- Growing season: `(6, 8)` — June to August (May/Sept can be snowy in north)
- Search order: August first (peak green), then July, then June
- Within each month: lowest STAC cloud cover first
- Temporal alternation: deterministic `hashlib.md5` of cell_key decides year-order per cell

### Tile Format

`.npz` files containing:
- `image`: (6, H, W) float32 — 6 Prithvi bands
- `label`: (H, W) uint8 — NMD L2 class indices
- Scalar metadata: date, easting, northing, lat, lon

---

## Sweden Land Mask

- Source: Lantmateriet `sve5milj` shapefile (`land5y.shp`, KKOD 901 = Sweden)
- Converted to: `data/sweden_land_epsg3006.json` (GeoJSON, 138 KB, 56 polygons, 2909 vertices)
- `filter_land_cells()` in `sampler.py` — Shapely point-in-polygon on cell center
- At 100km grid: 48 of 112 cells on land (64 ocean filtered instantly)

---

## NMD Ground Truth

### Collection

`NMD_2018_Basskikt_v1_1` on DES. Resolution: 10m.

**Critical:** Temporal extent filter must be removed — DES stores NMD with ingest timestamp (2024-08-28), not data year.

### Level 2 Classes (19 classes, indexed 0-18)

| Index | Name |
|-------|------|
| 0 | Hav/vatten (sea) |
| 1 | Tallskog (pine) |
| 2 | Granskog (spruce) |
| 3 | Lovskog (deciduous) |
| 4 | Blandskog (mixed) |
| 5 | Exploaterad mark (developed) |
| 6 | Exploaterad mark (buildings) |
| 7 | Exploaterad mark (infrastructure) |
| 8 | Jordbruksmark (cropland) |
| 9 | Oppen mark (open land) |
| 10 | Fjall (mountain/alpine) |
| 11 | Vatmark tallskog (wetland pine) |
| 12 | Vatmark granskog (wetland spruce) |
| 13 | Vatmark lovskog (wetland deciduous) |
| 14 | Oppen vatmark (open wetland) |
| 15 | Vatten sjoar (lakes) |
| 16+ | Additional alpine/fjall classes |

Full schema in `imint/training/class_schema.py` (also supports 10-class grouped mapping).

### NMD Cache

- `.nmd_cache/` directory, key = snapped EPSG:3006 coordinates
- Hit rate: 71-85% (returns in <10ms vs 15-31s for network)
- `nmd_raster_to_l2()` maps raw NMD codes to 0-18 L2 indices
- NMD pre-filter: cells with `land_frac < 0.05` originally skipped (later replaced with nodata-only filter `valid_frac < 0.01`)

---

## DES Data Fetching

### Authentication

- **Preferred:** Basic Auth via `DES_USER` / `DES_PASSWORD` env vars (EGI/OIDC removed)
- Endpoint: `openeo.digitalearth.se`
- STAC: `explorer.digitalearth.se/stac/search`

### Bands

- 6 Prithvi bands stored per tile: B02, B03, B04, B8A, B11, B12
- DES uses lowercase band names; conversion via `DES_TO_IMINT` dict

### DN-to-Reflectance

`reflectance = (DN - 1000) / 10000` (DES applies offset=1000)

### Batch SCL via tar.gz

- DES returns multi-date downloads as `.tar.gz` archives
- `_fetch_scl_batch()` decompresses, extracts TIFs, parses dates via regex
- ~3x fewer API calls than per-date fetching

### Adaptive Worker Pool

`_AdaptiveWorkerPool` in `prepare_data.py`:
- Initial: 3 workers, range 1-3 (matched to DES backend's 3 workers)
- Semaphore-based: scales down if p90 > 60s or error rate > 30%, up if p90 < 30s and 0 errors

### DES API Latency (observed)

| Call Type | Avg | P90 |
|-----------|-----|-----|
| stac_search | 0.3s | 0.3s |
| scl_batch | 26.1s | 31.8s |
| full_spectral | 28.0s | 36.6s |
| nmd_fetch | 9.9s | 26.3s |

---

## Quality Gates

### 1. SCL Cloud Screening
- `SCL_CLOUD_CLASSES = {3, 8, 9, 10}` (shadow, cloud medium/high, cirrus)
- Per-tile: `cloud_threshold = 0.05` (5% max cloud+shadow)

### 2. B02 Haze Gate
- `b02_haze_threshold: float = 0.06`
- Clean tiles: B02 mean ~0.015-0.035; hazy: 0.06+

### 3. Nodata Gate
- Rejects tiles with >10% B02 zeros (orbit edge issues at high latitudes)

---

## Pipeline Architecture

### Parallelization

```
NMD-producer ──filter──▶ approved_q ──▶ spectral-workers (2-3)
                           ▲
                     skips water
```

- NMD pre-filter runs in background thread, feeds approved cells to queue
- Worker threads consume: STAC search -> SCL batch -> spectral fetch
- Thread-safe shared state with locks

### Resumability

- `progress.json` tracks completed/failed cells (atomic writes: tmp + rename)
- Auto-recovery: if corrupt, rebuilds from tile files on disk
- Class distribution restored from existing tiles on restart

---

## Model Architecture

### Prithvi-EO-2.0 (Frozen Backbone)

- 300M parameters, frozen (no gradients in initial training)
- Input: `(B, C=6, T=1, H=224, W=224)` 5D tensor
- 6 bands: B02, B03, B04, B8A, B11, B12
- Normalization: reflectance * 10000, then `(x - mean) / std` with HLS statistics

### HLS to Sentinel-2 Band Mapping

| HLS | S2 | Note |
|-----|-----|------|
| B02 (Blue) | B02 | |
| B03 (Green) | B03 | |
| B04 (Red) | B04 | |
| B05 (Narrow NIR) | **B8A** | HLS B05 != S2 B05! |
| B06 (SWIR1) | **B11** | HLS B06 != S2 B06! |
| B07 (SWIR2) | **B12** | HLS B07 != S2 B07! |

### UPerNet Decoder (Trainable)

- ~15-20M trainable parameters
- PSP pool sizes: (1, 2, 7, 14) — MPS-compatible (14x14 feature maps)
- `.contiguous()` after all `F.interpolate()` and in `ConvBnRelu.forward()`

---

## Training Dashboard

Self-contained HTML + Chart.js (inlined, no CDN). File: `imint/training/dashboard.py`.

### Features
- Auto-refresh every 5s via `fetch()` polling JSON files
- Status badge: Running / Completed / Stopped
- ETA: rolling 5-minute window (session rate, not cumulative average)
- Disconnect detection: 3 failed fetches -> retains last data

### Charts
1. Loss curve (per epoch)
2. mIoU curve (best marked gold)
3. Per-class IoU bar chart
4. Worst-class trend line
5. Class distribution: doughnut (7 grouped) with toggle to pie (19 detailed)
6. System metrics gauges (SVG half-circle: CPU, RAM, GPU, network)

### Dashboard Server
- HTTP daemon thread, default port 8000 (`--dashboard-port`)
- Keep-alive loop after pipeline completion

---

## MPS / GPU Issues (Apple Silicon)

| Problem | Fix |
|---------|-----|
| `adaptive_avg_pool2d` crash | Pool sizes (1, 2, 7, 14) for 14x14 maps |
| Non-contiguous after interpolate | `.contiguous()` everywhere |
| `torch.gather` stride issues | `.contiguous()` on gather index |
| ReLU inplace issues | `inplace=False` |
| Training hangs after 2-8 epochs | **Use CPU on Mac** (PyTorch 2.8.0 MPS unstable) |

**Recommendation:** CPU on Mac (~41-61s/epoch), CUDA for production.

---

## Data Format

```
Copernicus original:  uint16 DN (0-10000+)
DES delivers:         float64 via openEO
Pipeline converts:    float32 reflectance [0, 1]
Saved:                float32 in .npz files
```

---

## Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `prepare_training_data()` | `prepare_data.py` | Main data collection entry point |
| `generate_grid()` | `sampler.py` | EPSG:3006 grid with land filtering |
| `_fetch_worker()` | `prepare_data.py` | Thread worker: STAC + SCL + spectral |
| `_fetch_scl_batch()` | `fetch.py` | Batch SCL via tar.gz |
| `fetch_des_data()` | `fetch.py` | Full spectral + SCL fetch |
| `_to_nmd_grid()` | `fetch.py` | WGS84 -> EPSG:3006 + NMD grid snap |
| `_AdaptiveWorkerPool` | `prepare_data.py` | Adaptive DES concurrency (1-3) |
| `start_dashboard_server()` | `dashboard.py` | HTTP server for dashboard |
| `LULCTrainer.train()` | `trainer.py` | Training loop with early stopping |
| `FocalLoss` | `losses.py` | Configurable focal loss |

---

## Key Commits

| Commit | Description |
|--------|-------------|
| `9e1c5ce` | LULC training module + rare-class handling |
| `6b0a558` | Dashboard, Sweden land mask, NMD pre-filter |
| `abee379` | Parallel NMD + STAC fetch + system metrics |
| `920ccef` | Adaptive worker pool (semaphore-based) |
| `fd30ed2` | Working SCL batch via tar.gz + rolling ETA |
| `0747ee3` | Nodata quality gate + class distribution restore |
| `fcb37e7` | SCB tatort, coastal, sumpskog densification; simplified DES auth |

---

## Error Solutions

| Problem | Solution |
|---------|----------|
| JWT auth failures | Remove stale `.des_token`; use Basic Auth |
| No 2017 data on DES | Default years = `["2019", "2018"]` |
| Pipeline hangs on DES | `socket.setdefaulttimeout(90)` + timeout wrapper |
| Multi-date GeoTIFF fails | DES returns tar.gz; parse archive |
| MPS training hangs | Fallback to CPU (`--device cpu`) |
| Dashboard flickering | Two servers on different ports; kill old one |
| Progress counter reset | Auto-recovery: cross-check progress.json vs tiles |
| Wrong data directory | Use `--data-dir data/lulc_full` to resume |
| `reduce_spatial` unsupported | tar.gz approach with local computation |
| Snow in training data | Narrowed growing season to June-August |
