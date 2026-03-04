# CDSE Integration & Dual-Source Fetch

> Copernicus Data Space Ecosystem (CDSE) backend, DN offset discovery, dual-source architecture, batch fetch benchmarks, seasonal fetch implementation.

---

## CDSE Backend

### Constants

```python
CDSE_OPENEO_URL = "https://openeo.dataspace.copernicus.eu/"
CDSE_COLLECTION = "SENTINEL2_L2A"
```

CDSE uses UPPERCASE band names (B02, B03, ...) vs DES lowercase (b02, b03, ...).

### Authentication (3-tier)

1. `CDSE_CLIENT_ID` + `CDSE_CLIENT_SECRET` env vars -> client_credentials
2. `.cdse_credentials` file (4 lines: email, password, client_id, client_secret)
3. Fallback: `authenticate_oidc()` (interactive browser)

### DN-to-Reflectance Offset

**Critical discovery:** CDSE via openEO applies `RADIO_ADD_OFFSET` internally — output DN values are already corrected.

| Source | Offset | Scale | Formula |
|--------|--------|-------|---------|
| DES | 1000 | 10000 | `(DN - 1000) / 10000` |
| CDSE (via openEO) | 0 | 10000 | `DN / 10000` |
| CDSE (raw S3/OData) | -1000 | 10000 | `(DN + 1000) / 10000` |

Three profiles in `imint/utils.py`:
```python
DATA_SOURCES = {
    "des":              {"offset": 1000,  "scale": 10000},
    "copernicus":       {"offset": 0,     "scale": 10000},  # openEO applies offset
    "copernicus_raw":   {"offset": -1000, "scale": 10000},  # raw L2A files
}
```

Cross-validation after fix: DES and CDSE give identical reflectance (Diff=0.0000 for all 10m bands, SCL 100% agreement).

---

## Dual-Source Architecture

### Dispatcher

`fetch_sentinel2_data(source="des"|"copernicus")` routes to the correct backend. Both backends use identical approach:
1. Load 10m bands (B02, B03, B04, B08)
2. Load 20m spectral bands, resample to 10m
3. Load 60m bands, resample to 10m
4. Merge all cubes
5. Download as GeoTIFF

Only differences: collection names and band name casing.

### STAC Discovery

All STAC queries go to DES catalog regardless of download backend:
```
STAC discovery -> always explorer.digitalearth.se
Pixel download -> DES or CDSE depending on source
```

### Grid Snapping

CDSE returns EPSG:32633 (UTM33N), reprojected to EPSG:3006 (SWEREF99 TM) via `_snap_to_target_grid()`.

---

## Performance Comparison

| Metric | DES | CDSE |
|--------|-----|------|
| Per tile (224x239 px) | 23.3s | 15.3s |
| Seasonal (4 frames) | ~77s | ~60s |
| Speed | Baseline | ~30% faster |
| Stability | Occasional hangs | Intermittent 502s |
| Backend workers | 3 | ~10-20 concurrent |

---

## Batch Fetch Benchmarks

Benchmarked 6 approaches across both backends:

| Approach | DES | CDSE |
|----------|-----|------|
| Sequential (current) | 77.6s | ~60s |
| Merged narrow-date cubes | xarray crash | overlap resolver error |
| Wide temporal sync | 75.1s (87MB) | 187.9s (collapsed to 1 date) |
| Batch job + NetCDF | xarray crash | 341s (19.6 MB) |

**Key findings:**
- `merge_cubes` across different temporal extents fails on both backends
- DES wide temporal downloads 10x more data (all dates, not just best)
- Current two-step approach (SCL pre-screen -> single-date spectral) is fastest
- The smart pre-filter design makes sequential calls optimal in practice

---

## SCL Batch GeoTIFF Fix

`_fetch_scl_batch()` only handled tar.gz (DES format), but CDSE returns plain GeoTIFF. Added detection by magic bytes:
- `\x1f\x8b` = gzip/tar.gz (DES)
- `4d4d002a` = big-endian TIFF (CDSE)

Including multi-date stacked GeoTIFF support (each band = one date's SCL).

---

## Dual-Source Fetch Pipeline

### Config
```python
fetch_sources: list[str] = ["copernicus", "des"]
```

### Connection Setup
```python
connections = {}
for src in sources:
    if src == "copernicus":
        connections["copernicus"] = _connect_cdse()
    else:
        connections["des"] = _connect()
```

### Worker Distribution (2:1 CDSE:DES)
```python
workers_per_source["copernicus"] = max(2, _MAX_WORKERS // 2 + 1)
```

### Dynamic Source Selection

Replaced static hash-based assignment with adaptive selection:

```python
def _pick_source(cell_key: str, preferred: str) -> tuple[str, str]:
    if preferred in ("copernicus", "des"):
        return preferred, other
    h = int(hashlib.md5(cell_key.encode()).hexdigest(), 16)
    if h % 3 != 2:
        return "copernicus", "des"  # 2:1 ratio
    return "des", "copernicus"
```

Fallback loop: try primary, on failure try secondary source.

### Adaptive Source Selection (Shared Stats)

Each parallel container reads/writes `source_stats.json` on CFS mount. If no history, defaults to DES (faster for seasonal). Compares avg response times and routes to faster backend.

---

## Seasonal Fetch Implementation

### File: `executors/seasonal_fetch.py` (~280 lines)

Standalone ColonyOS executor:
- Reads env vars (EASTING, NORTHING, FETCH_SOURCE, YEARS, etc.)
- Connects to DES or CDSE based on source
- Full pipeline: STAC -> SCL batch -> per-date fallback -> spectral fetch -> quality gates -> stack -> NMD labels -> atomic save

### Seasonal Tile Format

```
image:          (T*6, H, W)  float32 reflectance
label:          (H, W)       uint8 LULC classes
dates:          (T,)         ISO date strings
doy:            (T,)         int32 day-of-year
temporal_mask:  (T,)         uint8: 1=valid, 0=padded
```

### Year Rotation

Deterministic per cell:
```python
cell_hash = int(hashlib.md5(cell_key.encode()).hexdigest(), 16)
years_order = years[cell_hash % len(years):] + years[:cell_hash % len(years)]
```

~50/50 split between years, goal is best cloud-free scene per season.

---

## WGS84 Coordinate Bug

Grid generator produces SWEREF99 TM (EPSG:3006) but WGS84 fields default to 0.0. STAC queries with (0, 0) query the Gulf of Guinea.

Fix: `grid_to_wgs84()` in `sampler.py` using `_sweref99_to_wgs84()`.

---

## Tests (27 total, all passing)

### `tests/test_utils.py` — TestCopernicusReflectanceOffset (6)
DN range, cross-source equivalence, nodata, high reflectance, raw offset, raw vs openEO

### `tests/test_fetch.py` — 19 tests across 4 classes
- TestCDSEBandConstants (6): uppercase bands, collection ID, URL
- TestConnectCDSE (3): auth priority, OIDC fallback, failure
- TestFetchCopernicusData (6): FetchResult, reflectance, cloud rejection
- TestFetchSentinel2Data (4): dispatcher routing, kwarg forwarding

---

## DES Credential Security

Removed hardcoded defaults from `_connect()`:
```python
des_user = os.environ.get("DES_USER")
des_password = os.environ.get("DES_PASSWORD")
if not des_user or not des_password:
    raise FetchError("DES credentials not configured...")
```

`.env` and `.cdse_credentials` confirmed gitignored.

---

## Actual Download Test Results

| Source | Time | Frames | Notes |
|--------|------|--------|-------|
| DES | 228s (3.8 min) | 3/4 valid | Winter correctly skipped (27% nodata) |
| CDSE | 1600-1862s | 2-3/4 valid | Intermittent 502 Bad Gateway |

DES ~8x faster than CDSE for seasonal. CDSE has dedicated Swedish infrastructure disadvantage.

---

## Key Files

| File | Changes |
|------|---------|
| `imint/fetch.py` | `_connect_cdse()`, `fetch_copernicus_data()`, `fetch_sentinel2_data()`, source params |
| `imint/utils.py` | `DATA_SOURCES` dict with 3 DN profiles |
| `imint/training/config.py` | `fetch_sources` list |
| `imint/training/prepare_data.py` | Dual connections, dual worker pools |
| `executors/seasonal_fetch.py` | **New** — ColonyOS executor |
| `scripts/submit_seasonal_jobs.py` | **New** — coordinator with 2:1 balancing |
| `scripts/test_seasonal_local.py` | **New** — local simulation |
| `scripts/test_batch_fetch.py` | **New** — batch benchmark |
| `tests/test_utils.py` | 6 new offset tests |
| `tests/test_fetch.py` | 19 new CDSE tests |

---

## Key Commits

| Commit | Description |
|--------|-------------|
| `c5dcee2` | CDSE backend + dual-source Sentinel-2 fetch |
| `1be4cb6` | ColonyOS seasonal fetch executor, Docker, benchmarks |
| — | Adaptive source selection + SCL batch GeoTIFF fix |
