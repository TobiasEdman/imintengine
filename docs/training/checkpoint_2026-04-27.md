# ImintEngine training pipeline — checkpoint 2026-04-27

State after a 4-day investigation that uncovered and fixed five
interlocking bugs in the LPIS labeling path, plus the start of dataset
symmetrisation between the 256-px and 512-px pipelines.

## Where we are

### ✅ Verified working

- **Labels**: build-labels v5 (Apr 27) finished cleanly. 7076/7084 tiles
  rebuilt; 4716 of them carry LPIS crop classes (was 0 before fix).
  Visual inspection confirmed cyan label boundaries follow field edges
  in the RGB.
- **Rasterizer**: `scripts/enrich_tiles_lpis_mask.py:rasterize_parcels`
  now uses standard (W, S, E, N) bbox order and returns the rasterio
  output as-is (the vestigial `rot90+T` post-transform was removed).
  Regression test: agent A's reverse-fit shows `IoU=1.0` between the
  saved mask and a fresh rasterize (commits `f0306db`, `394121e`,
  `ff0d51f`).
- **LPIS GeoParquets** at `/data/lpis/jordbruksskiften_*_spatial.parquet`
  rewritten in standard (X, Y) order (Apr 25, commit `ae5fe17`); the
  original (Y, X)-stored versions are kept as `.bak` for reproducibility.
- **Thread safety**: `imint/training/spatial_parquet.py` and
  `imint/training/tile_fetch.py` hold rasterio + pyarrow handles in
  `threading.local()` (commit `3f9df62`). Regression tests at
  `tests/test_thread_safety.py` pass on a 16-thread workload.
- **256-px dataset** at `/data/unified_v2` (8290 tiles, 10 m GSD,
  2560 m extent) confirmed to use the standard (W, S, E, N) bbox
  convention — verified by `probe-256-convention-job.yaml` returning
  IoU=1.0 identity orientation against fresh NMD reads.

### 🟡 In progress

- **Stage A** (free enrichment): `enrich-512-skg-dem-job` running on
  `p02r09srv07`, adding `dem`, `height`, `volume`, `basal_area`,
  `diameter` to the 512 dataset via Skogsstyrelsen ImageServer +
  Copernicus DEM S3. ETA ~75 min after first launch + path-bug fix
  in `prefetch_aux.py` (commit `7682ba5`).
- **Watcher** `blmsyqjn1` chains alignment-viz + v7d-prithvi300 baseline
  after stage A finishes.

### 📅 Scheduled

- **Stage B** on **2026-05-01 08:17** (persistent task
  `enrich-256-stage-b` in `~/.claude/scheduled-tasks/`):
  - 256 ← rededge, s1_vv_vh, tessera (`enrich-256-rededge-s1-tessera-job`)
  - 512 ← VPP phenology (`enrich-512-vpp-job`)
  Both depend on the monthly CDSE PU quota reset.

## Bug history (the "quagmire")

The LPIS misalignment was actually **five distinct bugs**, several of
which compensated for each other and only became visible when one was
fixed in isolation. In chronological order of discovery:

| # | Bug | Where | Fix commit |
|---|---|---|---|
| 1 | LPIS GeoParquets stored geometry as (Y, X) instead of (X, Y) | upstream data preprocessing | `ae5fe17` (rewrote all 5 parquets in standard order) |
| 2 | `rasterize_parcels()` queried bbox as (S, W, N, E) — coded to compensate for #1 | `scripts/enrich_tiles_lpis_mask.py:285,302` | `ff0d51f` |
| 3 | `ThreadPoolExecutor` shared not-thread-safe rasterio + pyarrow handles across 32 workers | `tile_fetch._NMD_SRC` + `SpatialParquet._parq` | `3f9df62` (per-thread `threading.local()`) |
| 4 | `rasterize_parcels()` looked up `crop_class` column that doesn't exist in raw LPIS parquets | `enrich_tiles_lpis_mask.py:314` | `394121e` (added `grdkod_mar` to candidate column list) |
| 5 | Vestigial `np.rot90(mask, 2).T` post-transform — also compensated for #1 | `enrich_tiles_lpis_mask.py:374-376` | `f0306db` |

The key lesson: **fixing #1 + #2 in isolation made #5 the sole-uncompensated
bug, which was the user-visible misalignment**. The investigation that
nailed #5 used five parallel agents each testing a different hypothesis;
agent A found it via reverse-fit (apply candidate-inverse to saved mask,
check IoU=1.0 vs fresh — bulletproof regardless of which dihedral
elements you enumerate up front).

## Training run history (all on contaminated labels — not directly comparable)

| Run | Backbone | Hyperparams | Best mIoU | Status |
|---|---|---|---|---|
| v6a (reference) | prithvi_300m | bs=4, lr=1e-4, FP32, no rewind | **0.3663** | reference |
| v7-prithvi300 | prithvi_300m | bs=32, lr=3e-4, BF16 | 0.3082 | rare-class collapse (4 classes) |
| v7-prithvi600 | prithvi_600m | bs=32, lr=3e-4, BF16 | partial | OOM on 2080 Ti |
| v7b-prithvi300 | prithvi_300m | bs=16, lr=2e-4, BF16 | 0.3407 | rare-class collapse (2 classes) |
| v7c-2080-prithvi300 | prithvi_300m | bs=4, lr=1e-4, FP32 | 0.2959 | broken-dataset cohort |
| v7c-2080-prithvi600 | prithvi_600m | bs=4, lr=1e-4, FP32 | 0.2457 | broken + OOM at epoch 7 |
| v7d-prithvi300 | prithvi_300m | bs=4, lr=1e-4, FP32, **rewind** | 0.2676 | broken-dataset; rewind fired once at epoch 3 |
| v7d-prithvi600 | prithvi_600m | bs=4, lr=1e-4, FP32, **rewind** | 0.2457 | partial (7/10 epochs) |

**No run yet on the post-fix v5 dataset.** The v7d baseline that the
master watcher chains after stage A will be the first one with clean
labels. Expected mIoU > v6a's 0.3663 if the cleaned LPIS overlay is the
dominant lever.

## Code architecture invariants (post-fix)

- **One** `rasterize_parcels(gdf, bbox_3006, tile_size)` in
  `scripts/enrich_tiles_lpis_mask.py`. Called by `build_labels.py`,
  `enrich_lulc_tiles_lpis.py`, and `inspect_tile.py`. Tile-size
  agnostic via the `tile_size` parameter — no per-pipeline rasterizers.
- **All** SpatialParquet handles + rasterio handles are now per-thread
  via `threading.local()`. Regression test in `tests/test_thread_safety.py`
  asserts that 16-thread reads match a single-thread reference.
- **Atomic writes** in `build_labels.py` (`tmp + os.replace`) and
  `enrich_tiles_s1.py`. No more half-written `.npz` corruption when a
  worker is killed mid-write.
- **Invariant assertions** inside `build_tile_label`:
  shape match, NMD value range ≤ 19, unified label range ≤ 22,
  `crop_*`-named tiles must produce `n_parcels > 0`.
- **`launch_train_auto.sh`** prefers H100 (default), falls back to
  2080 Ti within 90 s if quota is denied. Per-accelerator defaults
  right-size pod resources + batch + LR + BF16.
- **`launch_train.sh`** explicit-accelerator wrapper.
  `k8s/unified-train-template.yaml` is parametric via envsubst.
- **Collapse-rewind** in `imint/training/trainer.py` reloads best
  checkpoint + halves LR per group + re-anneals cosine when
  `≥ collapse_threshold` new classes hit 0.0 IoU.

## Outstanding work

- **Stage 3**: extend `imint/training/unified_dataset.py` to read from
  both `/data/unified_v2` (256, native key `image`) and
  `/data/unified_v2_512` (512, native key `spectral`) in the same
  training run. The class already accepts both keys; what's missing is
  graceful zero-fill of any aux channels missing on either side, and
  random-crop-to-256 on 512 inputs so batches can stack.
- **Stage 4**: `v7e-prithvi300` training run on the mixed 256+512
  dataset, post-stage-A and post-stage-B. Target: beat v7d-prithvi300
  on the now-clean 512.
- **Stage 5**: full ensemble — T2 prithvi_600m, T3 terramind_v1_base,
  T4 clay_v1_5 (clay weights cache still needs HF download).
- **CROMA / THOR** ensemble members deferred until S1 enrichment
  lands on 512 (stage B).
- **frame_2016** intentionally skipped on 512; Prithvi-EO-v2 has
  built-in temporal baseline encoding.

## Key infrastructure pointers

- **Repo**: `/Users/tobiasedman/Developer/ImintEngine`, branch `main`
- **Cluster**: `icekube`, namespace `prithvi-training-default`
- **Datasets**:
  - 512 tiles: `/data/unified_v2_512` on PVC `training-data-cephfs` (RWX)
  - 256 tiles: `/data/unified_v2` on PVC `training-data` (RWO)
  - LPIS parquets: `/data/lpis/`
  - NMD raster: `/data/nmd/nmd2018bas_ogeneraliserad_v1_1.tif`
  - SKS parquets: `/data/sks/`
  - Tessera cache: `/data/tessera_cache`
  - Model checkpoints: PVC `training-checkpoints` mounted at `/checkpoints`
- **Secrets**: `cdse-credentials`, `des-credentials`, `hf-token`,
  `skg-endpoints` (added today)
- **Quotas**:
  - H100: 8 cluster-wide, often saturated; auto-fallback to 2080 Ti
    via `launch_train_auto.sh`
  - 2080 Ti: ~88 cluster-wide, never contended
  - CDSE PU (Sentinel Hub Process API): 10k/month, currently exhausted
    until 2026-05-01 reset
  - CDSE STAC + COG: 50k requests/month (S1 fetch via STAC backend
    bypasses PU)
  - CDSE openEO: 10k credits/month (parallel pool, used as fallback)
  - DES openEO: separate quota (used during the PU-blackout days for
    refetching the 188 quarantined tiles via `--fetch-sources=des`)

## File-tree highlights (committed today, 2026-04-26..27)

```
imint/training/
  spatial_parquet.py          (per-thread TLS handles)
  tile_fetch.py               (per-thread NMD rasterio handle)
  trainer.py                  (collapse-rewind, BF16 toggle)
scripts/
  build_labels.py             (atomic writes, invariants, --executor flag)
  enrich_tiles_lpis_mask.py   (rasterizer fixes #2, #4, #5)
  prefetch_aux.py             (unified_v2 layout support)
  launch_train.sh             (explicit accelerator)
  launch_train_auto.sh        (H100→2080Ti auto-fallback)
k8s/
  build-labels-job.yaml       (32 workers / 128 Gi, idempotent)
  enrich-512-skg-dem-job.yaml (stage A: forestry + DEM, free)
  enrich-512-vpp-job.yaml     (stage B 1-maj: VPP phenology)
  enrich-256-rededge-s1-tessera-job.yaml (stage B 1-maj)
  fix-lpis-axes-job.yaml      (one-shot parquet rewriter, done)
  quarantine-corrupt-tiles-job.yaml (one-shot, done)
  tile-audit-v2-job.yaml      (NMD freshness + seam detection audit)
  alignment-viz-job.yaml      (RGB + label overlay scoring)
  unified-train-template.yaml (parametric training job)
tests/
  test_thread_safety.py       (regression: per-thread handles)
docs/training/
  hyperparameters.md          (v6a→v7→v7b→v7c→v7d ladder)
  s1_fetch.md                 (STAC vs SH Process tradeoffs)
  checkpoint_2026-04-27.md    (this file)
```
