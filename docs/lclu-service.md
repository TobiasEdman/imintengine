# Annual LCLU service — local operator prototype

The existing v1 API serves `/lclu`: select a Swedish location and an available
historical year, submit a job, view its overlay and download GeoTIFF/provenance.
The annual path uses a resident strict-loaded TESSERA R4 head and canonical
inference inputs. It never calls the exact-day DES path. `/v1/analyze` remains.

Click a completed map to read the class at the native 10 m cell, without
starting another inference. `GET /v1/lclu/jobs/{job_id}/pixel?lon=…&lat=…`
returns the saved result's class ID, display name, colour and aggregation
note; nodata and outside-footprint points are distinct. Choose a new place
outside the result or with the coordinate fields. The [NMD legend check](verification/lclu-service/nmd-legend.md)
documents official colours and the model's aggregated/extended classes.

**Real R4 execution is verified for Uppsala, Stockholm and Kiruna tätort, year 2024.**
The demo supports independently qualified regions, with an area selector in the map.
The Stockholm municipality demo reads lossless compact COGs and prepares
auxiliary windows on demand. New maps took 18–19 seconds; repeated HTTP
retrieval took under one second. A 504×504 Stockholm map matches canonical
inference pixel for pixel, including preparation across the native UTM-zone
boundary. These checks establish execution equivalence, not accuracy against
reference labels. Only qualified area/year combinations are enabled; the
Stockholm demo offers 2024 because its complete 2025 VPP sources are absent.
The local `--demo` launcher uses clearly labelled synthetic inputs and random
weights. See the [Stockholm run recipe](../docker/lclu/README.md) for the
bounded private deployment and retained cache.
See [verification](verification/lclu-service/verification.md) and
[public source checks](verification/lclu-service/source-checks.md).
The [Sweden TESSERA GeoTIFF storage estimate](tessera-geotiff-storage.md)
records measured compression, national capacity projections and planning
budgets for 2018–2024.

## Run locally

For the interactive prototype with clearly labelled synthetic data:

```sh
make lclu-demo
```

Open `http://127.0.0.1:8784/lclu`, click the Uppsala marker, choose 2023 or 2024
and create a map. Both years use the same synthetic test pattern; this does not
demonstrate historical change. The banner remains visible while scrolling,
and downloaded provenance explicitly identifies random weights and synthetic
inputs. Stop with Ctrl+C. Data/cache stay in ignored `outputs/lclu-demo/`.
For a different existing environment, set `PYTHON=/path/to/python` on the make
command. The equivalent launcher is `python scripts/serve_lclu.py --demo`.

For actual R4 inputs, run
`python scripts/serve_lclu.py --manifest /absolute/operator/serving.json`.
The two modes are explicit and mutually exclusive; missing real configuration
never starts a synthetic demo automatically.

Run from a repository checkout with the existing `api` dependencies installed.
This prototype reuses repository scripts and web assets; a standalone wheel
without those files is not a supported deployment artifact.

```sh
export IMINT_API_MAX_PIXELS=262144
export IMINT_API_MAX_JOBS=4
export IMINT_API_OUTPUT_DIR=/absolute/operator/cache
export IMINT_LCLU_MANIFEST=/absolute/operator/serving.json
python -m uvicorn imint.api.v1:app --host 127.0.0.1 --port 8784 --workers 1 --lifespan on
```

Open `http://127.0.0.1:8784/lclu`. Without a manifest, the page and geometry work
but classification is disabled. Invalid checkpoints or prepared catalogue
inputs fail startup; preparation-source failures fail the individual job.
There is no random-weight production fallback. Use one process/host
and the existing output-directory lock. This is an unauthenticated local API;
no public deployment is included. The basemap uses OpenStreetMap and fonts use
Google Fonts.

## Manifest

An identity is `{"path":"...","sha256":"<64 lowercase hex>","size":123}`.
Generate it with `scripts.cache_predict.file_identity`; do not invent values.
Paths resolve against the manifest directory. Strict loading reuses
`scripts/inference_comparison.py`. The head must have 28 classes, 128 embedding
channels, gated fusion, all 11 ordered auxiliaries and the fraction head. Only
approved R4 weights belong here; architecture compatibility alone does not
establish model lineage.

Use one of `inputs` or `preparation`:

```json
{
  "version": 1,
  "device": "cpu",
  "checkpoint": {"path": "r4.pt", "sha256": "<verified hash>", "size": 123},
  "preparation": {
    "config": {"path": "sources.json", "sha256": "<verified hash>", "size": 456},
    "python": "/absolute/preparation-env/bin/python",
    "timeout_seconds": 600
  }
}
```

For prepared inputs, replace `preparation` with `inputs`, an array of entry
objects produced by `scripts/prepare_lclu.py`. Update each `input.path` to its
final location. Each entry records `center_3006`, exact `year`, `public_area:
true`, pinned input, embedding contract and source provenance. The catalogue
verifies entries at startup and before inference. Only exact snapped centres
and years are available; the page marks these prepared locations.

Never point the catalogue at a training/held-out directory. A public-area flag
is operator qualification, not an automated privacy review. The service does
not require private field identifiers or labels.

## Source preparation

GeoTessera 0.10.2 requires Python >=3.12. The explicit preparation interpreter
needs the repository's `fetch` and `inference` dependencies plus
**`geotessera==0.10.2`**. It can differ from the API/model interpreter. The
server does not install anything. Preparation runs in a child with a hard
timeout (1–900 seconds); failures retain a private operator log.

`sources.json` contains:

| Field | Contract |
|---|---|
| `version` | `1` |
| `embedding` | `{"producer":"geotessera","producer_version":"0.10.2","dataset":"v1","variant":"vultr"}` |
| `registry`, `landmasks` | Pinned `manifest.parquet` and `landmasks.parquet` in one directory |
| `max_blocks` | Integer 1–4 |
| `max_download_bytes` | Positive cap on missing embedding/scales/landmask bytes |
| `embedding_cache` | Dedicated writable cache; metadata must match v1/vultr |
| `identity_cache` | Private writable directory for local-source verification receipts |
| `vpp_index` | Pinned canonical WEkEO `index.json` |
| `vpp_files` | Map of indexed COG filename to pinned SHA256 and size |
| `skg_manifest` | Pinned canonical SKG `MANIFEST.json` |
| `skg_vrts` | Identities for all staged grunddata omdrev VRTs, `markfukt/markfukt.vrt`, `tradhojd/tradhojd_10m.vrt`; paths relative to mirror root |
| `dem_sources` | Pinned local Copernicus GLO-30 COGs covering the complete window |
| `qualified_regions` | Operator-verified Swedish coverage regions, below |

Paths resolve against the source config directory, except VPP COG filenames
relative to the index and SKG VRTs relative to the mirror manifest. Canonical
SKG readers use the fixed mirror and their normal latest-omdrev mosaic. County
validity-mask sources are also pinned, even when they do not contribute model
auxiliary values. Metadata/source changes fail preparation. Large unchanged
COGs reuse verified hashes only while inode, device, size, mtime and ctime match.
The directories are trusted operator storage, not multi-user uploads.

A region has `country: "SE"`, a nonempty `verification` reference, an exact
`years` list (subset of 2018–2025) and ordered `bounds_3006: [west,south,east,north]`.
The entire 512-pixel input must lie inside a qualified region. Qualify actual
embedding and auxiliary coverage before adding it; registry presence alone is
insufficient. Empty regions advertise no years. New preparations check source
validity again and fail if sources are unavailable.

Prepare an exact centre explicitly with the preparation interpreter:

```sh
python scripts/prepare_lclu.py --config /absolute/operator/sources.json \
  --east 647850 --north 6638770 --year 2024 --output /absolute/operator/uppsala-2024
```

This is a public Uppsala example, not a verified classification. The command
does not bulk-stage auxiliaries or substitute another year. Individual
embedding fetch failures fail the request. It hashes source files and the final
input and applies the canonical Swedish county mask to cross-border pixels.

## Numerical and temporal contract

- Clicks snap to the canonical EPSG:3006 10m centre grid. Preparation makes a
  512×512 input; inference takes exactly `[4:508,4:508]`. Output is 504×504 and
  5,040m square, with affine shifted 40m east/40m south.
- Embeddings are dequantized and reprojected by the existing TESSERA preparer,
  then stored as float16 `(128,512,512)`. Bilinear validity propagation excludes
  pixels supported by missing embeddings. Valid pixels must be finite/nonzero.
- The canonical order is height, volume, basal area, diameter, DEM, VPP
  SOSD/EOSD/length/max/min, markfukt. The existing inference builder performs
  cropping, YYDDD conversion, log transforms and normalization.
- TESSERA and HR-VPP season 1 use the **exact requested year**. Forestry uses
  the latest scan per pixel in a fixed release. Height, markfukt and DEM are
  fixed supporting products, not observations from the selected year. The open
  height/markfukt products differ from the retired proxies; qualify their
  compatibility against actual R4 inputs before interpreting results.
- Missing DEM coverage is an error; real zero elevation remains valid.
  Canonical auxiliary nodata conventions are preserved and counted. A valid
  embedding does not imply that every auxiliary has an observed value.
- Reopening the GeoTIFF verifies CRS, affine, shape, extent, nodata and data.
  Class 0 is opaque background; nodata 255 is distinct. Class IDs are warped
  to EPSG:3857 using nearest-neighbor **before** applying the display palette.
  The downloadable TIFF remains EPSG:3006 at 10m.

## API, queue and cache

| Route | Purpose |
|---|---|
| `GET /v1/lclu` | Readiness, candidate years, legend and prepared-location markers |
| `GET /v1/lclu/availability?lon=17.64&lat=59.86` | Snapped geometry and qualified years |
| `POST /v1/lclu` | Submit `{"lon":17.64,"lat":59.86,"year":2024}`; receive 202 and `poll_url` |
| `GET /v1/jobs/{id}` | Shared lifecycle, phase and annual result |
| `GET /v1/lclu/jobs/{id}/files/{name}` | Completed TIFF, PNG overlay, legend or provenance only |

One shared execution slot serializes legacy and annual jobs. Admission counts
queued/running work across both routes. `IMINT_API_MAX_JOBS` defaults to 4 and
accepts 1–64; excess work returns 429 with Retry-After. Identical grid/year/model/
source requests reuse pending or completed jobs; failures can be retried.
Shutdown stops admission and drains accepted work before releasing the lock.

Prepared inputs persist in `lclu-inputs/`; outputs in `lclu/<cache-key>/`.
Identity includes model hash, source/config identity, actual input hash, geometry,
year, numerical implementation and runtime versions. A staging directory is
renamed atomically only after export/verification. Warm results verify all four
artifact hashes; corruption fails closed. Code edits require API restart. Job
IDs are in memory; after restart resubmit to reuse a verified persistent result.

Public provenance contains source hashes, approved numerical metadata, geometry,
legend, runtime/implementation identity, timings and temporal policy. Arbitrary
operator metadata, filenames and local paths are not published. There is no
automatic disk retention; manage storage while stopped. Multi-host execution,
public authentication, distributed queues and national batches are outside scope.

## Synthetic QA

```sh
python -m pytest tests/lclu tests/api_services tests/test_inference_aux_parity.py \
  tests/test_tessera_wiring.py tests/test_tessera_year.py -q
python -m tests.lclu.browser_fixture --output /tmp/lclu-synthetic-qa --port 8785
```

The separate browser fixture prominently states **SYNTHETIC TEST · RANDOM
WEIGHTS · NOT AN R4 CLASSIFICATION**. It exercises strict loading, the real
architecture and forward pass, map overlay and downloads. Production startup
does not use it. It provides no scientific validation.
