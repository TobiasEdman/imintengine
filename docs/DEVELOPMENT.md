# IMINT Engine — Development Guide

> Environments, showcase workflow, training pipeline, and deployment.

---

## Environments

The project uses three environments controlled by `IMINT_ENV`:

| Environment | Device | Batch | Epochs | Purpose |
|-------------|--------|-------|--------|---------|
| **dev** | MPS (M1 Max) | 8 | 50 | Local development, full training |
| **test** | CPU | 2 | 2 | Fast iteration, CI, smoke tests |
| **prod** | CUDA (H100) | 16 | 50 | Production training at ICE Connect |

### Setup

```bash
# 1. Copy the template and fill in credentials
cp config/environments/template.env config/environments/dev.env
# Edit dev.env with your DES_USER, DES_PASSWORD, CDSE_CLIENT_ID, etc.

# 2. Same for test and prod
cp config/environments/template.env config/environments/test.env
cp config/environments/template.env config/environments/prod.env
```

The `.env` files are gitignored (secrets). Only `template.env` is committed.

### Usage

```bash
# Makefile shortcuts
make train-dev                        # M1 Max, MPS, batch 8
make train-test                       # CPU, batch 2, 2 epochs
make train-prod                       # H100, CUDA, batch 16

# Direct script usage
IMINT_ENV=dev .venv/bin/python scripts/train_lulc.py
IMINT_ENV=test .venv/bin/python scripts/train_lulc.py --epochs 1

# Extra args forwarded
make train-dev ARGS="--epochs 10 --batch-size 4"
```

### Priority

Configuration values are resolved in this order (highest wins):

1. **CLI arguments** (`--batch-size 4`)
2. **Shell environment variables** (`IMINT_BATCH_SIZE=4`)
3. **Environment file** (`config/environments/dev.env`)
4. **TrainingConfig defaults** (dataclass in `imint/training/config.py`)

### Docker

Docker Compose reads the environment file automatically:

```bash
# Default: dev.env
docker compose run --rm seasonal-fetch-cdse

# Override to use prod credentials
IMINT_ENV_FILE=config/environments/prod.env docker compose run --rm seasonal-fetch-cdse
```

---

## Showcase

The showcase is an interactive HTML dashboard served via GitHub Pages at:

**https://tobiasedman.github.io/imintengine/**

It's embedded as an iframe on **digitalearth.se**.

### Current Tabs

| Tab | Date | Area | Key analysis |
|-----|------|------|--------------|
| Brand | 2018-07-24 | Ljusdal, Gavleborg | Wildfire detection (dNBR, Prithvi segmentation) |
| Marin | 2025-07-10 | Hunnebostrand, Bohuslan | Vessel detection (YOLO, heatmap) |
| Betesmark | 2025-06-14 | Oland | Grazing classification (CNN-biLSTM, LPIS) |
| Kustlinje | 2025-06-14 | Ystad, Skane | Shoreline change 2018-2025 (CoastSat, NDWI) |

### Showcase Files

```
outputs/
  imint_showcase.html          ← Main dashboard (committed, served by Pages)
  showcase/
    fire/                      ← 9 PNGs + metadata
    marine/                    ← 7 PNGs + vessel GeoJSON + sjokort
    grazing/                   ← 7 PNGs + LPIS GeoJSON
    kustlinje/                 ← 7 PNGs + coastline vectors GeoJSON
```

### Generating Showcase Images

Each tab has a generation script that fetches Sentinel-2 data, runs analyzers, and saves PNGs:

```bash
# Grazing showcase (Oland)
.venv/bin/python scripts/generate_grazing_showcase.py

# Kustlinje showcase (Ystad coast, 2018-2025)
.venv/bin/python scripts/generate_kustlinje_showcase.py
```

Fire and marine showcases were generated via the standard analysis pipeline (`executors/local.py`).

### Adding a New Showcase Tab

1. **Generate images** — Create a script `scripts/generate_<name>_showcase.py` following the pattern in `generate_kustlinje_showcase.py`. Save PNGs to `outputs/showcase/<name>/`.

2. **Add HTML tab** — In `outputs/imint_showcase.html`:
   - Add tab button in `<div class="header-nav">` with `data-tab="<name>"`
   - Add `<div class="tab-content" id="tab-<name>">` section with summary cards, intro text, panels
   - Use prefix `<x>-` for panel IDs (e.g., `k-rgb`, `k-ndvi`)

3. **Add JavaScript** — At the end of the `<script>` block:
   - Define `<NAME>_VIEWERS` array (set `"vector": true` for GeoJSON overlays)
   - Define `<NAME>_IMAGES` dict mapping panel IDs to image paths
   - Optionally define `<NAME>_GEOJSON` for vector overlays
   - Call `initMaps(<NAME>_VIEWERS, <NAME>_IMAGES, imgH, imgW, hasBgToggle, geojsonData)`

4. **Commit and push** — Images and HTML go to `main` branch. GitHub Pages auto-deploys.

### Vector Overlays

The showcase supports interactive GeoJSON overlays on Leaflet maps. Set `"vector": true` in the viewer config and pass GeoJSON as the 6th argument to `initMaps()`.

Supported geometry types:
- **Polygon** — vessel detections, LPIS parcels (colored by `predicted_class`)
- **LineString** — coastline shorelines (colored by `year`)

The `initMaps()` style function auto-detects the feature type based on properties.

---

## Training Pipeline

### Data Flow

```
1. Grid generation        sampler.py → 5,801 tiles across Sweden
2. Fetch Sentinel-2       fetch.py → seasonal S2 L2A (CDSE or DES)
3. Fetch NMD labels        nmd.py → 19-class land cover labels
4. Fetch aux channels      skg_height.py, skg_grunddata.py, copernicus_dem.py
5. Train model             trainer.py → Prithvi + UPerNet + AuxEncoder
6. Evaluate                evaluate.py → per-class IoU, confusion matrix
```

### Quick Commands

```bash
# Prepare training data (fetches S2 + NMD + aux for all tiles)
.venv/bin/python scripts/prepare_lulc_data.py --data-dir ~/training_data

# Train (dev environment)
make train-dev

# Train with specific overrides
make train-dev ARGS="--epochs 100 --enable-all-aux --unfreeze-backbone-layers 12"

# Evaluate only (no training)
.venv/bin/python scripts/train_lulc.py --evaluate-only --checkpoint checkpoints/lulc/best_model.pt

# Prefetch auxiliary data
.venv/bin/python scripts/prefetch_aux.py --data-dir ~/training_data
```

### Auxiliary Channels

5 optional channels fused via late fusion (AuxEncoder → 64ch → concat with UPerNet):

| Channel | Source | Unit | Enable flag |
|---------|--------|------|-------------|
| Tree height | Skogsstyrelsen | meters | `--enable-height-channel` |
| Volume | Skogsstyrelsen | m3sk/ha | `--enable-volume-channel` |
| Basal area | Skogsstyrelsen | m2/ha | `--enable-basal-area-channel` |
| Diameter | Skogsstyrelsen | cm | `--enable-diameter-channel` |
| DEM | Copernicus GLO-30 | meters | `--enable-dem-channel` |

Enable all: `--enable-all-aux`

Skogsstyrelsen endpoints require `.skg_endpoints` config file or `SKG_HEIGHT_URL` / `SKG_GRUNDDATA_URL` env vars (see `config/environments/template.env`).

---

## ColonyOS (Distributed Fetch)

### Local Development Stack

```bash
make colony-up       # Start TimescaleDB + MinIO + Colonies server + executor
make colony-status   # Check services are running
make colony-logs     # Follow logs
make colony-down     # Stop everything
make colony-reset    # Stop + destroy volumes
```

### Job Submission

```bash
# Dry run — show what would be submitted
make submit-dry

# Submit 100 tiles
make submit-live

# Submit all tiles
make submit-all

# Check progress
make status
```

### S2 Process API (Alternative Fetch)

```bash
# Dry run
make s2-submit-dry

# Local mode (no ColonyOS needed)
make s2-local
```

---

## Docker

### Build

```bash
# ARM64 (M1 Max, local dev + ColonyOS)
make build

# x86_64 CUDA (H100 VM training)
make build-cuda
```

### Images

| Image | Platform | Use |
|-------|----------|-----|
| `imint-engine:latest` | ARM64 | Fetch executors on M1 Max |
| `imint-engine:cuda` | x86_64 | Training on H100 GPU |

---

## Project Structure

```
config/
  environments/
    template.env             ← Committed (no secrets)
    dev.env                  ← Gitignored (M1 Max local)
    test.env                 ← Gitignored (fast CI)
    prod.env                 ← Gitignored (H100 production)
  analyzers.yaml             ← Analyzer config
  seasonal_fetch_job.json    ← ColonyOS job specs
  s2_seasonal_fetch_job.json
  vpp_fetch_job.json

imint/
  config/
    env.py                   ← Environment loader
  training/
    config.py                ← TrainingConfig dataclass
    trainer.py               ← Training loop
    dataset.py               ← Dataset with aux channels
    evaluate.py              ← Evaluation metrics
    sampler.py               ← Grid generation
    prepare_data.py          ← Data preparation pipeline
    skg_height.py            ← Skogsstyrelsen tree height
    skg_grunddata.py         ← Skogsstyrelsen volume/basal/diameter
    copernicus_dem.py        ← Copernicus DEM
  fm/
    upernet.py               ← UPerNet + AuxEncoder + Prithvi backbone
  analyzers/                 ← Analysis modules
  exporters/                 ← Output formatters

scripts/
  train_lulc.py              ← Main training entry point
  prepare_lulc_data.py       ← Data preparation
  prefetch_aux.py            ← Batch auxiliary data prefetch
  submit_seasonal_jobs.py    ← ColonyOS job submission
  generate_*_showcase.py     ← Showcase image generators

executors/
  seasonal_fetch.py          ← ColonyOS S2 fetch executor
  s2_seasonal_fetch.py       ← S2 Process API executor
  vpp_fetch.py               ← VPP phenology executor
```

---

## Tests

```bash
make test                    # All tests (332 pass, ~7 min)
make test-utils              # Quick unit tests
make test-cdse               # CDSE-specific tests
make test-seasonal           # Local seasonal fetch test (CDSE)
make test-seasonal-des       # Local seasonal fetch test (DES)
```

---

## Credentials

| Service | Variables | Source |
|---------|-----------|--------|
| Digital Earth Sweden | `DES_USER`, `DES_PASSWORD` | DES account |
| Copernicus (CDSE) | `CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET` | Sentinel Hub dashboard |
| Skogsstyrelsen | `SKG_HEIGHT_URL`, `SKG_GRUNDDATA_URL` | Licensed endpoints |

All credentials go in `config/environments/*.env` (gitignored) or `.skg_endpoints` (gitignored).

Never commit credentials. The `*.env` and `*credentials*` patterns in `.gitignore` protect against accidental commits.
