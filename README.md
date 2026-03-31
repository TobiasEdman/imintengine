# IMINT Engine

Modular satellite image intelligence engine built as part of the Swedish Space Data Lab and Digital Earth Sweden.

Analyzes cloud-free Sentinel-2 imagery for change detection, spectral classification, shoreline monitoring, vessel detection, and land-use classification.

---

## Architecture

```
imint/                      Core engine (executor-agnostic)
  engine.py                 run_job() — single entry point
  job.py                    IMINTJob / IMINTResult data models
  fetch.py                  Sentinel-2, NMD, Sjokort, LPIS data fetching (DES + CDSE)
  coregistration.py         Sub-pixel image alignment
  utils.py                  Shared helpers

  analyzers/                One file per analyzer
    base.py                 Abstract BaseAnalyzer + AnalysisResult
    spectral.py             NDVI, NDWI, EVI, MNDWI spectral indices
    change_detection.py     Multispectral change detection (baseline comparison)
    object_detection.py     YOLO-based region-of-interest detection
    shoreline.py            CoastSat-method shoreline extraction (NDWI/MNDWI + Otsu)
    cot.py                  Cloud optical thickness (MLP5 model)
    nmd.py                  NMD (Nationellt Marktackedata) land-cover overlay
    grazing.py              LPIS grazing activity classification
    marine_vessels.py       Marine vessel detection (fine-tuned YOLO)
    ai2_vessels.py          AI2 satellite vessel detection
    prithvi.py              Prithvi-EO foundation model segmentation

  fm/                       Foundation models & weights
    prithvi_mae/            Prithvi MAE encoder (IBM/NASA)
    coastseg/               CoastSeg SegFormer weights (4-class, 512x512)
    cot_models/             Cloud optical thickness MLP5
    marine_vessels/          Fine-tuned YOLO for vessel detection
    ai2_vessels/            AI2 vessel detection model
    pib_grazing/            PIB grazing classification model
    terratorch_loader.py    TerraTorch model loading
    upernet.py              UPerNet segmentation head

  training/                 Training pipeline
    trainer.py              Training loop orchestrator
    unified_dataset.py      Multitemporal unified dataset (4-frame)
    unified_schema.py       20-class schema (NMD + LPIS crops + SKS harvest)
    tile_fetch.py           Shared fetch primitives (STAC → CDSE → DES)
    dataset.py              Legacy tile dataset with augmentation
    config.py               Training configuration
    class_schema.py         LULC class hierarchy (10-class legacy)
    prepare_data.py         Data preparation from NMD/DEM/SCB
    sampler.py              Balanced sampling strategies
    evaluate.py             Model evaluation & metrics
    losses.py               Custom loss functions (Dice, Focal)
    dashboard.py            Training progress dashboard

  exporters/
    export.py               PNG, GeoTIFF, GeoJSON export helpers
    html_report.py          Interactive HTML showcase generator

  config/
    env.py                  Environment configuration loader

executors/                  How jobs are submitted and run
  base.py                   Abstract BaseExecutor interface
  local.py                  Run locally from CLI or notebook
  colonyos.py               Run inside a ColonyOS container job
  seasonal_fetch.py         Multi-year seasonal data fetching

config/
  analyzers.yaml            Enable/disable analyzers and tune params
  analyzers_full.yaml       Full configuration variant
  colonyos_job.json         ColonyOS job spec
  seasonal_fetch_job.json   Seasonal fetch job spec
  environments/
    dev.env                 Development settings
    test.env                Test settings
    prod.env                Production settings

scripts/                    Standalone utility scripts
  generate_kustlinje_showcase.py   Generate coastline showcase images
  generate_grazing_showcase.py     Generate grazing showcase images
  generate_lulc_showcase.py        Generate LULC tile gallery + chart data
  fetch_unified_tiles.py           Unified 4-frame tile fetcher (LULC + crop + urban)
  train_unified.py                 Train with unified 20-class schema
  train_lulc.py                    Train LULC segmentation model (legacy 10-class)
  predict_lulc.py                  Run LULC inference, save per-tile predictions
  run_lulc_pipeline.py             Run LULC classification pipeline
  run_grazing_model.py             Run grazing model inference
  run_evaluation.py                Run model evaluation suite
  des_login.py                     DES authentication
  prefetch_aux.py                  Prefetch NMD/DEM/SCB auxiliary data
  batch_local_fetch.py             Batch Sentinel-2 fetching

tests/                      Pytest test suite
  test_spectral.py          Spectral analyzer tests
  test_change_detection.py  Change detection tests
  test_object_detection.py  Object detection tests
  test_nmd.py               NMD overlay tests
  test_prithvi.py           Prithvi segmentation tests
  test_fetch.py             Data fetching tests
  test_integration.py       End-to-end integration tests
  ...

k8s/                        Kubernetes job specs (ICE Connect H100)
  unified-training-job.yaml Training on H100 (GPU)
  fetch-lulc-job.yaml       Tile fetching (CPU-only)

data/                       Training data & caches
  unified_v2/               Unified tiles (LULC + crop + urban, 4-frame)
  lulc_full/                Full LULC training dataset (legacy)
  seasonal_tiles/           Multi-year seasonal tiles
  symbols/                  Map symbol library

docs/                       GitHub Pages showcase
  index.html                Dashboard shell (tabs, descriptions, chart canvases)
  css/
    leaflet.css             Leaflet 1.9.4 styles
    styles.css              Custom dashboard styles
  js/
    vendor/                 Third-party libraries (Leaflet, Chart.js)
    tab-data.js             Shared legends, GeoJSON paths, tab configs
    app.js                  Reusable components, map init, event handlers
  data/
    vessels.geojson         YOLO vessel detections
    lpis.geojson            LPIS grazing block polygons
    erosion.geojson         Coastline erosion vectors
    segformer-shorelines.geojson  SegFormer shoreline vectors
    coastline-shorelines.geojson  Index-based shoreline vectors
    chart-data.json         NMD cross-reference chart data
  showcase/
    fire/                   Wildfire analysis images (Ljusdal)
    marine/                 Marine vessel detection images (Hunnebostrand)
    grazing/                Grazing land monitoring images (Lund)
    kustlinje/              Coastline erosion images (Ystad)

outputs/                    Generated files (gitignored except showcase)
checkpoints/                Model training checkpoints
```

The executor resolves job context (coordinates, dates, data fetching, cloud detection) and hands a populated `IMINTJob` to `run_job()`. The engine runs analyzers and writes outputs. Neither side knows about the other's internals.

---

## Showcase

Live dashboard: **[digitalearth.se](https://digitalearth.se)** (GitHub Pages)

Four analysis tabs with interactive Leaflet maps, vector overlays, and background toggles:

| Tab | Area | Analyses |
|---|---|---|
| **Brand** | Ljusdal, Gavleborg | dNBR burn severity, Prithvi segmentation, change gradient |
| **Marin** | Lysekil, Bohuslan | YOLO vessel detection, AI2 vessels, heatmap, sjokort toggle |
| **Bete** | Vastervik, Kalmar | LPIS grazing classification, NMD overlay |
| **Kustlinje** | Ystad, Skane | 8-year shoreline vectors, erosion analysis, 2018/2025 toggle |

A separate **training dashboard** (`imint/training/dashboard.py`) monitors the full pipeline in real-time: NMD pre-filter, seasonal fetch, data preparation, training, evaluation, and LULC inference with a per-tile gallery showing S2 pseudocolor, NMD labels, model predictions, and quality overlays.

---

## Quickstart (local, no DES account needed)

```bash
git clone https://github.com/TobiasEdman/imintengine
cd imintengine

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run with synthetic data (for analyzer development)
python executors/local.py \
  --date 2022-06-15 \
  --west 14.5 --south 56.0 --east 15.5 --north 57.0
```

Outputs land in `outputs/2022-06-15/`.

---

## Run with real DES data

```bash
# Authenticate with Digital Earth Sweden
python scripts/des_login.py

# Run analysis
python executors/local.py \
  --date 2024-06-15 \
  --west 14.5 --south 56.0 --east 15.5 --north 57.0
```

---

## Run in ColonyOS

```bash
# Sync code to ColonyOS filesystem
colonies fs sync -l imint -d . --yes

# Submit a job
colonies function submit --spec config/colonyos_job.json --follow
```

The ColonyOS executor reads `DATE`, `WEST`, `SOUTH`, `EAST`, `NORTH` from environment variables set by the job spec.

---

## Training (LULC segmentation)

### Current: Unified 20-class multitemporal (v3)

```bash
# Fetch tiles with 4-frame temporal pattern (1 autumn + 3 VPP-guided growing season)
python scripts/fetch_unified_tiles.py --mode all --output-dir data/unified_v2

# Train on H100 (ICE Connect k8s)
kubectl apply -f k8s/unified-training-job.yaml

# Or train locally (M1 Max / MPS)
python scripts/train_unified.py --data-dirs data/unified_v2 \
  --enable-multitemporal --num-temporal-frames 4 --num-classes 20
```

The unified schema merges NMD (forest, water, wetland, urban) + LPIS crops (vete, korn, havre, oljeväxter, vall, potatis, trindsäd) + SKS harvest (hygge) into **20 classes**. Each tile has 4 temporal frames: autumn (Sep–Oct from year-1) + 3 VPP-guided growing season frames adapted per-tile to local phenology. Training uses Prithvi-EO-2.0 backbone with 11 auxiliary channels (tree metrics, DEM, VPP phenology, harvest probability).

All tile types (LULC grid, crop, urban) are stored in a single flat directory and handled identically by the dataset.

**Results:** Best **37.5% mIoU** at epoch 49 (single-frame, 19-class). Multitemporal 4-frame training pending.

### Legacy: 10-class single-frame

```bash
python scripts/train_lulc.py
```

Best: **44.14% mIoU** at epoch 42 (10-class schema, single summer frame, 5 aux channels).

The training dashboard shows live progress and, after inference, a per-tile gallery comparing S2 pseudocolor (B8/B3/B4), NMD ground truth, model predictions, and a quality overlay highlighting correct/wrong/high-confidence-wrong pixels.

---

## Add a new analyzer

1. Create `imint/analyzers/my_analyzer.py`, subclass `BaseAnalyzer`, implement `analyze()`
2. Register it in `imint/engine.py`:
   ```python
   from .analyzers.my_analyzer import MyAnalyzer
   ANALYZER_REGISTRY = {
       ...
       "my_analyzer": MyAnalyzer,
   }
   ```
3. Add a config block to `config/analyzers.yaml`

That's it — no other files need to change.

---

## Swap the executor

To run on a different scheduler (Airflow, cron, AWS Batch):

1. Subclass `BaseExecutor` in `executors/`
2. Implement `build_job()` and `handle_result()`
3. Call `executor.execute()`

The engine code is untouched.

---

## Foundation models

| Model | Source | License | Use |
|---|---|---|---|
| **Prithvi-EO-2.0** | IBM/NASA | Apache 2.0 | LULC segmentation backbone (300M params) |
| **CoastSeg SegFormer** | Vos et al. | GPL-3.0 | Shoreline classification (weights only) |
| **YOLO11s** | Ultralytics | AGPL-3.0 | Object & vessel detection |
| **COT MLP5** | Pirinen / RISE | TBD | Cloud optical thickness |
| **PIB Grazing** | RISE | TBD | Grazing activity classification |

---

## License

Copyright (c) 2024-2025 RISE Research Institutes of Sweden AB

The original source code and documentation in this repository are dedicated
to the public domain under **CC0 1.0 Universal**. See [LICENSE](LICENSE)
for the full text.

Third-party components (models, data, libraries) retain their original
licenses. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for
details. Notable obligations:

- **YOLO11s** (Ultralytics): AGPL-3.0 — commercial closed-source use
  requires an Enterprise license
- **Prithvi-EO** (IBM/NASA): Apache 2.0
- **COT MLP5** (Pirinen et al. / RISE): license TBD
- **Sjokort S-57** (Sjofartsverket): academic use only via SLU GET
