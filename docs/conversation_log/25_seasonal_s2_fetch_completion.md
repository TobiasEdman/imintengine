# Seasonal S2 Fetch Completion & Data Status

> Multitemporal S2 L2A data fetch via CDSE Sentinel Hub Process API completed on M1 Max: 4,305/4,381 tiles (98.3%) successfully fetched. Each tile contains 4 seasonal frames x 6 bands = 24 channels. VPP phenology data not yet fetched.

---

## Fetch Results

| Metric | Value |
|--------|-------|
| **Method** | `submit_s2_jobs.py --local --workers 4` (CDSE Sentinel Hub Process API) |
| **Machine** | M1 Max (192.168.50.100) |
| **Success** | 4,305 / 4,381 tiles (98.3%) |
| **Failed** | 76 tiles (far-north Lapland, persistent cloud cover) |
| **Duration** | ~30 minutes for final batch; ~12 seconds per tile |
| **Storage** | CFS (MinIO) on M1 Max |

### Failure Analysis

All 76 failures are `no valid frames` errors for far-north tiles in the 600-700 km northing range. These are tiles in Lapland where cloud-free imagery does not exist in the Sentinel-2 archive for one or more seasonal windows. This is an inherent data availability limitation, not a pipeline bug.

---

## Tile Format

Each multitemporal tile is stored as `.npz` with:

| Key | Shape / Type | Description |
|-----|-------------|-------------|
| `image` | `(24, 224, 224)` | 4 frames x 6 bands (B02, B03, B04, B8A, B11, B12) |
| `label` | `(224, 224)` | NMD ground truth class map |
| `easting` | scalar | SWEREF99 TM easting (meters) |
| `northing` | scalar | SWEREF99 TM northing (meters) |
| `dates` | `(4,)` string | ISO date for each seasonal frame |
| `doy` | `(4,)` int | Day-of-year for each frame |
| `temporal_mask` | `(4,)` bool | Which frames have valid data |
| `num_frames` | scalar | Number of temporal frames (4) |
| `num_bands` | scalar | Bands per frame (6) |
| `multitemporal` | `True` | Flag for dataset loader |
| `source` | string | `"sentinel_hub"` |

### Seasonal Windows

| Frame | Months | Purpose |
|-------|--------|---------|
| Spring | April - May | Leaf-out phenology, deciduous/conifer discrimination |
| Summer | June - July | Peak growing season (primary classification signal) |
| Autumn | August - September | Senescence, crop harvest |
| Winter | January - February | Snow cover, evergreen detection |

### Cloud Filtering

Each seasonal window applies quality gates:
- **SCL cloud fraction** < threshold (0.10 for seasonal, slightly relaxed vs 0.05 for single-date)
- **Nodata** < 10% zero pixels in B02
- **B02 haze** < 0.06 mean reflectance

If no clear date exists for a seasonal window, the frame is zero-filled and `temporal_mask` is set to `False` for that frame. The model masks these during training.

---

## Data Inventory

### On M1 Max (192.168.50.100)

| Location | Tile Count | Type |
|----------|-----------|------|
| `data/lulc_full/tiles/` | ~5,801 | Mix of single-date (DES/openEO) and multitemporal (CDSE) |
| `~/cfs/tiles/` | ~531 | Multitemporal (CFS/MinIO storage) |

**Note**: The 4,305 CDSE multitemporal tiles were fetched to CFS (MinIO object storage). Their current distribution across `data/lulc_full/tiles/` and `~/cfs/tiles/` needs verification when M1 Max is accessible. Some may have been merged into `lulc_full/tiles/`, overwriting the original single-date DES tiles.

### On MacBook (development machine)

| Location | Tile Count | Type |
|----------|-----------|------|
| `data/lulc_full/tiles/` | 5,801 | Single-date only (original DES/openEO fetch) |
| `data/seasonal_tiles/` | 20 | Multitemporal (test tiles) |
| `data/test_tiles/` | 2 | Multitemporal (development) |
| `data/test_tile/` | 1 | Multitemporal (development) |
| `data/test_tile_v2/` | 1 | Multitemporal (development) |

The MacBook has only test/development multitemporal tiles. The full 4,305-tile dataset is on M1 Max.

---

## VPP / PPI Phenology Status

**Not yet fetched.** The code is ready but has never been executed:

| Component | Status |
|-----------|--------|
| `imint/training/cdse_vpp.py` | Implemented (Sentinel Hub BYOC fetch) |
| `executors/vpp_fetch.py` | Implemented (ColonyOS executor) |
| `scripts/submit_vpp_jobs.py` | Implemented (job submission) |
| VPP data on disk | **0 tiles** |

### VPP Bands (planned)

| Band | Description | Normalization (mean, std) |
|------|-------------|--------------------------|
| SOSD | Start of season (day-of-year) | (120.0, 30.0) |
| EOSD | End of season (day-of-year) | (280.0, 30.0) |
| LENGTH | Season length (days) | (160.0, 40.0) |
| MAXV | Max vegetation index (PPI) | (0.50, 0.25) |
| MINV | Min vegetation index (PPI) | (0.05, 0.05) |

VPP normalization stats in `config.py` are initial estimates. Should be recomputed with `compute_aux_stats.py` after fetching.

### VPP Fetch Command (when ready)

```bash
# On M1 Max
cd ~/ImintEngine
source .venv/bin/activate
set -a; source .env; set +a
python scripts/submit_vpp_jobs.py --local --workers 4
```

---

## Next Steps for Seasonal Training

1. **Verify M1 Max data** — Confirm 4,305 multitemporal tiles are intact and accessible
2. **Fetch VPP data** — Run `submit_vpp_jobs.py --local` on M1 Max to add phenology channels
3. **Transfer to H100** — Use `setup_h100_vm.sh --transfer-data` to rsync data to ICE Connect
4. **Enable multitemporal training** — Pass `--enable-multitemporal` to `train_lulc.py`
5. **Enable VPP channels** — Pass `--enable-vpp` once VPP data is fetched

### Training Command (H100 at ICE Connect)

```bash
./scripts/ssh_train.sh user@ice-h100 \
    --enable-multitemporal \
    --enable-all-aux \
    --enable-vpp \
    --batch-size 16 \
    --epochs 50
```

---

## Key Files

| File | Purpose |
|------|---------|
| `imint/training/cdse_s2.py` | Sentinel Hub Process API fetch (seasonal S2 L2A) |
| `scripts/submit_s2_jobs.py` | Job submission (--local for direct fetch) |
| `imint/training/cdse_vpp.py` | HR-VPP BYOC fetch (not yet executed) |
| `scripts/submit_vpp_jobs.py` | VPP job submission (not yet executed) |
| `imint/training/config.py` | TrainingConfig with multitemporal + VPP settings |
| `scripts/setup_h100_vm.sh` | H100 VM bootstrap (ICE Connect) |
| `scripts/ssh_train.sh` | Remote training launcher |
