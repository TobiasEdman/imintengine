# Plan — faster, reusable validation/inference architecture

**Status:** plan, ready to execute · **Created:** 2026-08-13
**Motivation:** `lucas-validate-tessera-gated` took ~2 h wall for one
checkpoint on one truth source (4,303 tiles). CPU is not the bottleneck —
the pod used 1.5 of 4 allocated cores. The cost is structural: serial,
batch-1, double-forward GPU inference with no caching.

## Current architecture (what's slow, with evidence)

The validation scripts fuse **GPU inference** and **truth scoring** into one
per-tile loop:

- **Batch = 1, one tile at a time.** `score_against_*` iterates
  `for tile_name, grp in index_df.groupby("tile_name")` and calls
  `predict_fn(tile_path)` per tile — [`validate_against_lucas.py:330`](../../scripts/validate_against_lucas.py).
  Each forward is a single tile, `unsqueeze(0)` + `.to(device)` per call
  ([`inference_comparison.py:268,296,303`](../../scripts/inference_comparison.py)).
  The GPU is idle most of the wall-clock.
- **Two forward passes per tile.** L2b/L2a run `run_inference` *and*
  `run_fraction_inference` separately
  ([`validate_against_lucas.py:428-433`](../../scripts/validate_against_lucas.py)),
  but the model carries both `classifier` and `frac_head` — one forward
  already produces both. That doubles GPU calls (≈8,600 for LUCAS).
- **Synchronous I/O, no prefetch.** `np.load(tile_path)` runs inline on the
  main thread ([`inference_comparison.py:257`](../../scripts/inference_comparison.py));
  no `DataLoader`/`num_workers`, so the GPU stalls on disk between tiles.
- **No prediction cache.** NFI and LUCAS re-infer the *same* tiles
  independently; re-runs (e.g. the 4 loader-fix iterations on 2026-08-13)
  re-inferred everything from zero. Violates the repo rule "cacha alltid
  mellanresultat till disk".

Net: wall-time scales as `n_tiles × 2 forwards × (load + transfer + compute)`,
serialized. For LUCAS that is ~2 h; it should be minutes.

## Target architecture — decouple inference from scoring

Split the fused loop into two stages with a **cached prediction layer**
between them.

### Stage A — `infer_tiles.py` (GPU, batched, cached)
Produce per-tile prediction rasters once and persist them.

- **Batched `DataLoader`** over the tile list: `num_workers=8` for parallel
  `np.load` + pinned-memory prefetch, `batch_size=16–32` tiles per GPU
  forward. Saturates the GPU; collapses ~8,600 serial calls to a few
  hundred batches.
- **Single forward, dual-head.** One `model(x)` per batch → read
  `classifier` *and* `frac_head` outputs together. Halves GPU work.
- **Write a compact per-tile cache** keyed by
  `(checkpoint_sha256, tile_name)`: argmax class map (`uint8`) + the 4
  fraction channels (`float16`), as compressed `.npz` or a partitioned
  parquet under `/cephfs/pred_cache/<ckpt_sha>/`. A `MANIFEST.json`
  records `{checkpoint, git_sha, model_config, produced_at, n_tiles}`
  (mirrors the repo's Docker-artifact provenance rule).
- **Idempotent skip** — a tile already in the cache for this checkpoint is
  not re-inferred (same pattern as `_valid_existing_tile` in the fetchers).

### Stage B — `score_against_truth.py` (CPU only, seconds)
Read the cache + a truth index → metrics. No GPU.

- One scorer, parameterised by truth source: `--truth {nfi,lucas,lpis}`
  with a `truth_index.parquet` giving `(tile_name, row, col, label, year)`.
- Samples the cached class map / fractions at each plot/point and computes
  the existing metrics (overall, kappa, per-class F1, L2a agreement).
- Because it is pure CPU on cached arrays, **all three truth sources score
  from the same cache** — inference is paid once, not once per truth.

### Optional Stage A-shard — map/reduce across GPUs
For large tile lists, shard the list across `K` GPU pods
(`--shard i/K`), each writing its slice of the cache; Stage B reduces over
the whole cache. Embarrassingly parallel; `K=4` ≈ 4× on top of batching.

## Expected speedup (LUCAS, 4,303 tiles)

| Lever | Effect |
|---|---|
| Single dual-head forward (drop double pass) | ~2× |
| Batch 16–32 + `num_workers` prefetch | ~8–15× (GPU saturation) |
| Cache reuse across NFI/LUCAS/LPIS + re-runs | inference paid **once** |
| 4-way GPU shard (optional) | ~4× on top |

Realistically ~2 h → **~5–10 min** for the first checkpoint, and **seconds**
for every subsequent truth source or re-score.

## Migration (non-breaking)

1. Extract the existing per-tile predict wiring into `infer_tiles.py`
   unchanged (verbatim first — preserve the full inference chain), add the
   `DataLoader`/batch/dual-head/cache around it.
2. Add `score_against_truth.py` that reproduces the *current* NFI/LUCAS
   numbers from the cache — **gate on bit-parity**: the refactor must
   reproduce today's 0.588 NFI and the concat 0.499/0.809 LUCAS numbers
   before it replaces the fused scripts.
3. Keep `validate_against_{nfi,lucas}.py` as thin wrappers (infer → score)
   for one release, then retire.
4. k8s: one `infer-tiles` GPU job (or `K` sharded) + a tiny `score` CPU job
   per truth source, replacing the monolithic `*-validate-*` GPU jobs.

## Non-goals

- Not changing the model, the metrics, or the truth definitions — this is
  purely an inference/scoring **plumbing** refactor.
- Not touching training.

## Verification

- Bit-parity gate (step 2) is the acceptance test: cached-path metrics must
  equal the current fused-path metrics on NFI-209 and LUCAS before switch.
- Then a wall-clock before/after on the LUCAS tile set.
