# National annual land-cover service: own copy for our areas, original tiles for the rest of Sweden

Status: decided 2026-11-01 (Tobias), implementation started. Each cluster
step needs Tobias' approval and `kubectl create --dry-run=server` first.

## Summary

The service already chooses a source per region (`RegionalSources`), so a
national service is two kinds of region:
- **Our areas** — the ten municipalities and Kiruna tätort — keep the
  staged copy.
- **The rest of Sweden** reads the same TESSERA tiles on demand from the
  public source.

Both hold the exact bytes and georeferencing R4 was trained on, so they share
one embedding identity (`EMBEDDING`, geotessera 0.10.2 v1/vultr). Only the
transport differs.

The published Zarr store is not used for serving. It rounds every tile's
position to its zone lattice, and a shift of up to 5 m is not acceptable
(D1).

TESSERA is not the only national input. HR-VPP, DEM and the forestry layers
must also cover the whole country, and HR-VPP does not yet.

## Decisions (Tobias, 2026-11-01)

| | Decision |
|---|---|
| D1 | No 5 m shift: the Zarr store's lattice-rounded tiles are not used for maps. The Zarr comparison still runs, to record the size of the shift. |
| Source | Rest of Sweden: the original NPY tiles from the public source, read by HTTP range (only the rows a window needs, one request per tile) |
| D2 | A window uses the own copy only when it lies entirely inside one of our areas; otherwise the remote tiles. Never mixed inside one map. |
| D3 | The own copy stays on the PVC. |
| D4 | Audit national coverage first; HR-VPP storage is decided from the measured size. |
| D5 | 200 GB disk cache for remote TESSERA, least recently used evicted first. |
| D6 | The map shows its source ("Rikstäckning, hämtas vid förfrågan") and the year's verification status; provenance says the same; TESSERA attribution is always shown. |

## Where we are

| Fact | Where |
|---|---|
| One served model, TESSERA R4 (28 classes, 11 aux), years 2017–2025, only in operator-qualified regions | `imint/inference/lclu.py`, `imint/inference/lclu_sources.py` |
| A new Stockholm map took 18.1 s: 12.9 s preparation, 4.6 s model, 0.6 s export; a cached one 1.0 s | `docs/verification/lclu-service/verification.md` |
| The viewer warms a place's inputs while the year is chosen, and a map request reuses them (one preparation per key) | `POST /v1/lclu/prepare`, `af1f209a`, `ce8cc8f5` |
| geotessera 0.10.2 downloads the same tiles from `data.source.coop/tessera/tessera/npy/v1/` and landmasks from `…/landmasks/v1/`; `geotessera.remote.read_npy_window` reads a row band with one ranged GET | geotessera 0.10.2 `registry.py`, `remote.py` |
| Windowed reads with a 2-pixel halo already reproduce the full-tile path bit for bit (the compact COG path) | `imint/inference/lclu_cog.py`, `tests/lclu/test_cog.py` |
| The Zarr store rounds tile origins to the 10 m lattice (2–9 m off for sampled Swedish tiles) | `imint/inference/lclu_zarr.py` (docstring) |
| HR-VPP (WEkEO cache) covers only the demo areas' MGRS tiles; 2025 is unpublished everywhere | `docs/verification/lclu-service/coverage-matrix.md` |
| DEM (GLO-30) is staged per area; forestry and markfukt are per-county mosaics on the PVC; county masks limit windows to Sweden | `scripts/stage_lclu_demo.py`, `imint/training/skg_open.py` |
| From this cloud session `data.source.coop` is blocked; from ICE it is unverified | session network policy |

## Target

| Region | TESSERA source | First new map (estimate) | Storage |
|---|---|---|---|
| Our 11 areas | staged COGs on the PVC | about 5 s after a warm-up (model + export) | about 300 GB, exists |
| Rest of Sweden | original tiles by HTTP range, 256-row band cache, 8 parallel requests | 40–150 MB of rows: 10–30 s at the one-stream 4.7 MB/s measured from the public source (`tessera-storage.json`), less in parallel — to be measured from ICE; the warm-up runs it while the year is chosen | ≤ 200 GB cache |

## Work packages

1. **Click warm-up — done.** Measure the live effect after the next
   deployment: preparation should leave the new-map time.
2. **Range reading of the original tiles.**
   - A `remote_tiles` branch in `scripts/prepare_lclu.py`. It uses the same
     pinned registry and landmask parquet files as the tile path.
   - Per tile, rows are read in fixed 256-row bands, cached on disk.
   - Resampling and merging are identical to the COG path.
   - Tested bit for bit against `assemble_embedding` across a seam,
     through a local HTTP server.
3. **National coverage audit.** Per year and per source, list what the
   country has:
   - TESSERA registry tiles;
   - HR-VPP (WEkEO index against the MGRS tiles covering Sweden);
   - forestry and markfukt (counties in the mosaics);
   - DEM.

   It is a cluster job, and its output is the national counterpart of
   `coverage-matrix.md`.
4. **HR-VPP national gap-fill** after the audit (D4). Same WEkEO path as the
   demo gap-fill, no CDSE units. 2025 stays unavailable until it is
   published.
5. **DEM on demand.** Read GLO-30 windows from the public COGs per request,
   with a disk cache.
6. **National region in the service.**
   - `RegionalSources` gets the window rule (D2) and a national region
     backed by the remote tiles.
   - The viewer and provenance get the source label (D6).
   - The embedding identity is unchanged.
7. **View and municipality prefetch.**
   - The 256-row bands are the unit.
   - Prefetch the view on pan/zoom, and the municipality on area choice
     from the view outward.
   - Click work always comes first; 200 GB in total, least recently used
     evicted (D5).
8. **Qualification of the national region.**
   - Take a deterministic sample of windows per year, stratified by county.
   - Check canonical pixel parity against `scripts/inference_comparison.py`,
     pixel inspection and validity.
   - Put the receipts in `docs/verification/lclu-service/`.
   - A year is offered nationally only with receipts.
9. **Deployment (separate approval).**
   - Egress from ICE to `data.source.coop`, verified.
   - Cache volume sizing.
   - A new serving manifest.
10. **Zarr measurement (information only).** Run
    `k8s/tessera-zarr-compare-job.yaml` and record the size of the shift.

## Order

1. Packages 2, 3 and 10 in parallel.
2. Then 4 and 5 (after the audit).
3. Then 6 and 7.
4. Then 8.
5. Package 9 last.

Each package ships with its own tests and receipts; nothing national is
offered before package 8.

## Risks

- **External host.** National maps depend on Source Cooperative's
  availability and policy. The band cache softens outages but cannot replace
  the source.
- **Licence.** Check the TESSERA terms and the required attribution before a
  public national service.
- **HR-VPP gaps.** A year without national HR-VPP cannot be offered
  nationally (no year substitution).
- **Abuse and cost.** Prefetch can be driven by anyone with the viewer.
  Newest-wins, the 200 GB budget and the job queue bound it.
