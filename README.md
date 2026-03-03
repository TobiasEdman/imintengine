# IMINT Engine

Modular satellite image intelligence engine built on as part of the Swedish Space Data Lab and Digital Earth Sweden.

Analyzes cloud-free Sentinel-2 imagery for change detection, spectral classification, and object detection.

---

## Architecture

```
executors/          ← How jobs are submitted and run
  local.py          ← Run locally from CLI or notebook
  colonyos.py       ← Run inside a ColonyOS container job
  base.py           ← Interface — add new executors here

imint/              ← Core engine (executor-agnostic)
  job.py            ← IMINTJob and IMINTResult data models
  engine.py         ← run_job() — the single entry point
  analyzers/        ← One file per analyzer
    base.py         ← Abstract BaseAnalyzer + AnalysisResult
    change_detection.py
    spectral.py
    object_detection.py
  exporters/
    export.py       ← PNG, GeoTIFF, GeoJSON output helpers

config/
  analyzers.yaml    ← Enable/disable analyzers and tune params

outputs/            ← Generated files (gitignored)
```

The executor resolves job context (coordinates, dates, data fetching, cloud detection) and hands a populated `IMINTJob` to `run_job()`. The engine runs analyzers and writes outputs. Neither side knows about the other's internals.

---

## Quickstart (local, no DES account needed)

```bash
git clone https://github.com/YOUR_ORG/imint-engine
cd imint-engine

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

Add `my_cloud_filtering/` from [erikkallman/ai-pipelines-poc](https://github.com/erikkallman/ai-pipelines-poc) to your `PYTHONPATH` or clone it alongside this repo. The local executor will pick it up automatically.

```bash
PYTHONPATH=../ai-pipelines-poc python executors/local.py \
  --date 2022-06-15 \
  --west 14.5 --south 56.0 --east 15.5 --north 57.0
```

---

## Run in ColonyOS

```bash
# Sync code to ColonyOS filesystem
colonies fs sync -l imint -d . --yes

# Submit a job
colonies function submit --spec config/get_cloud_free.json --follow
```

The ColonyOS executor reads `DATE`, `WEST`, `SOUTH`, `EAST`, `NORTH` from environment variables set by the job spec.

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

## Outputs

| File | Description |
|---|---|
| `{date}_rgb.png` | Cloud-free RGB composite |
| `{date}_change_overlay.png` | Change regions highlighted in red |
| `{date}_change_regions.geojson` | Change region bounding boxes (WGS84) |
| `{date}_ndvi.png` | NDVI colormap |
| `{date}_land_cover.tif` | GeoTIFF land cover classification |
| `{date}_detections.geojson` | Detected regions of interest |
| `{date}_imint_summary.json` | Full run summary |

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
