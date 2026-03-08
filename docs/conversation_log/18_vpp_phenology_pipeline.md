# HR-VPP Phenology Pipeline

> High-Resolution Vegetation Phenology and Productivity (HR-VPP) data from Copernicus via Sentinel Hub Process API, PPI vegetation index, phenology metrics for LULC classification.

---

## What is HR-VPP?

HR-VPP (High-Resolution Vegetation Phenology and Productivity) is derived from Sentinel-2 time series and provides 10 m resolution phenology metrics per growing season. It captures the temporal dynamics of vegetation that spectral-only approaches miss.

### Why Phenology Helps LULC

- **Deciduous vs evergreen** -- deciduous forests have later SOSD and earlier EOSD
- **Cropland vs grassland** -- crops have sharper, shorter growing seasons
- **Wetland vegetation** -- distinct phenological timing due to water table
- **Urban green** -- managed vegetation has different seasonal patterns

---

## Parameters Fetched

| Band | Name | Unit | Scale | Description |
|------|------|------|-------|-------------|
| SOSD | Start Of Season Day | Day of year | Integer | When vegetation growth begins |
| EOSD | End Of Season Day | Day of year | Integer | When vegetation growth ends |
| LENGTH | Season Length | Days | Integer | EOSD - SOSD |
| MAXV | Maximum PPI | Unitless (0-2) | INT16 x 0.0001 | Peak Plant Phenology Index |
| MINV | Minimum PPI | Unitless (0-2) | INT16 x 0.0001 | Base PPI value |

PPI (Plant Phenology Index) is NOT part of the S2 spectral fetch -- it comes from HR-VPP as a derived product.

---

## Access

| Parameter | Value |
|-----------|-------|
| **API** | Sentinel Hub Process API |
| **Endpoint** | `https://sh.dataspace.copernicus.eu/api/v1/process` |
| **Collection** | BYOC (Bring Your Own COG), Season 1 |
| **Collection ID** | `67c73156-095d-4f53-8a09-9ddf3848fbb6` |
| **Auth** | CDSE OAuth2 `client_credentials` grant |
| **License** | Copernicus Open Access |
| **Resolution** | 10 m (native) |
| **Coverage** | Pan-European, from ~2017 |
| **Default year** | 2021 (recent, complete for Sweden) |

---

## Fetch Implementation

### File: `imint/training/cdse_vpp.py` (474 lines)

Single HTTP POST fetches all 5 bands as multi-band GeoTIFF via evalscript:

```python
_VPP_BANDS = ["SOSD", "EOSD", "LENGTH", "MAXV", "MINV"]
_VPP_COLLECTION_ID = "67c73156-095d-4f53-8a09-9ddf3848fbb6"

def fetch_vpp_tiles(west, south, east, north, *, size_px=256, cache_dir=None, year=2021):
    """Fetch HR-VPP Season 1 phenology for a tile.
    Returns dict: {sosd, eosd, length, maxv, minv} -> (H, W) float32 arrays."""
```

### Band Scaling

```python
# PPI bands stored as INT16, scale to real values
_PPI_BANDS = {"MAXV", "MINV"}  # Multiply by 0.0001
_DOY_BANDS = {"SOSD", "EOSD", "LENGTH"}  # Keep as integers -> float32
```

### Caching

Results cached as `.npz` files keyed by bbox coordinates. Cache hit skips the HTTP request entirely.

### Rate Limiting

Same pattern as S2 fetch: HTTP 429 with `Retry-After` header backoff, 3 retries, 60s timeout.

---

## ColonyOS Executor

### File: `executors/vpp_fetch.py`

Standalone executor for distributed VPP fetching via ColonyOS:
- Reads env vars (EASTING, NORTHING, bbox coordinates)
- Calls `fetch_vpp_tiles()` per cell
- Saves to CFS mount for collection

### Job Submission

```bash
python scripts/submit_vpp_jobs.py --tiles-dir seasonal-tiles
```

### Config

```json
// config/vpp_fetch_job.json
{
    "conditions": {
        "executortype": "container-executor",
        "walltime": 300
    },
    "kwargs": {
        "cmd": "python executors/vpp_fetch.py"
    }
}
```

---

## Integration with Training

### Config Flags

```python
# imint/training/config.py
enable_vpp_channels: bool = False  # Enable 5 VPP bands
```

When enabled, VPP adds 5 channels to the AuxEncoder input:

| Channel | Normalization Mean | Normalization Std |
|---------|-------------------|-------------------|
| vpp_sosd | 120.0 (day) | 30.0 |
| vpp_eosd | 280.0 (day) | 30.0 |
| vpp_length | 160.0 (days) | 40.0 |
| vpp_maxv | 0.50 (PPI) | 0.25 |
| vpp_minv | 0.05 (PPI) | 0.05 |

**Note:** These are initial estimates. Recompute with `scripts/compute_aux_stats.py` after prefetching VPP data.

### Training Command

```bash
python scripts/train_lulc.py \
  --enable-all-aux \           # Includes VPP + height/volume/basal/diameter/DEM
  --checkpoint-dir checkpoints/lulc_aux_vpp
```

Or just VPP:

```bash
python scripts/train_lulc.py --enable-vpp
```

### Channel Stacking Order

When all aux channels are enabled, the AuxEncoder receives 10 channels:

```
[height, volume, basal_area, diameter, dem,
 vpp_sosd, vpp_eosd, vpp_length, vpp_maxv, vpp_minv]
```

---

## Status

- **Fetch code**: Complete (`imint/training/cdse_vpp.py`)
- **ColonyOS executor**: Complete (`executors/vpp_fetch.py`)
- **Job submission script**: Complete (`scripts/submit_vpp_jobs.py`)
- **VPP data fetch**: Not yet submitted (S2 fetch running first)
- **Training with VPP**: Not yet attempted (need VPP data first)

---

## Key Files

| File | Purpose |
|------|---------|
| `imint/training/cdse_vpp.py` | HR-VPP fetch module (474 lines) |
| `executors/vpp_fetch.py` | ColonyOS executor for VPP fetch |
| `scripts/submit_vpp_jobs.py` | VPP job submission |
| `config/vpp_fetch_job.json` | ColonyOS job spec template |
| `imint/training/config.py` | VPP toggle and normalization stats |

---

## Key Commits

| Commit | Description |
|--------|-------------|
| `4af3121` | Add HR-VPP phenology enrichment pipeline via Sentinel Hub Process API |
