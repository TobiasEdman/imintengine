# Heltäckande markslagskarta 2024

Status: draft for Codex (2026-11-03). The batch producer is written and tested
on synthetic windows; nothing has run on the cluster. Each cluster step needs
Tobias' approval and `kubectl apply --dry-run=server` first.

Asked by Tobias (2026-11-03): the page shows two things at once.

- **A wall-to-wall land-cover map can be produced.** The label is always
  "Heltäckande markslagskarta 2024".
- **The analysis runs on demand for the earlier years (2017–2023).**

This is the next step after the current publication (r9, the public API and
#125), and it must not delay that.

## Scope and size

Measured from `config/lclu/municipalities.json` by
`scripts/produce_lclu_coverage.py --dry-run`:

| | |
|---|---|
| Areas | 11 (ten municipalities and Kiruna tätort), 12,601 km² |
| Lattice | the national 5,040 m lattice in EPSG:3006 (origin 0,0); one cell is one 504 × 504 px output window at 10 m |
| Lattice windows per area | Uppsala 119, Norrköping 120, Jönköping 109, Örebro 95, Linköping 86, Västerås 67, Göteborg 62, Helsingborg 32, Malmö 24, Stockholm 21, Kiruna tätort 4: 739 in total |
| Distinct lattice windows | 726: 13 windows straddle Linköping and Norrköping and run once |
| Admitted by the service | 490 |
| Refused by the service | 236 (centre outside every municipality polygon; see below) |
| Fill windows | 250 |
| Windows to run | 740 (93 or 92 per pod at 8 pods) |
| Municipality pixels left uncovered | 0 in every area |

- **Compute:** about 22 s per window on 4 CPU (r9 Göteborg 2017 took 22.4 s
  and r8 Västerås 2019 took 23.3 s). That is about 4.5 h for one pod, or
  about 35 min for 8 pods at the same CPU-hours. The batch also builds a
  fresh source catalogue per window, which adds some overhead per window;
  measure that on the Stockholm run.
- **Storage:** about 0.3 MB of output per window, plus one COG per area.
  Inputs are about 70 MiB each and are deleted after their window.
- **Year:** 2024, the newest year with every input. HR-VPP 2025 is absent
  from the staged cache, and the VITO UTM catalogue returned no 2025
  products (Codex 2026-11-02).

## The service admits fewer lattice windows than the lattice has

The service admits a grid only when its centre lies inside exactly one
qualified municipality polygon and its 512 px input lies inside that region's
`bounds_3006` (`RepositorySources.years`). A lattice cell can meet the
municipality while its centre lies outside. The live service refuses such a
cell, and so does any batch on the same sources.

Without fills, the map would cover these shares of each municipality:

| Area | Lattice-only cover |
|---|---|
| Stockholm | 58.9 % (6 of 21 cells admitted) |
| Helsingborg | 82.7 % |
| Malmö | 86.1 % |
| Göteborg, Örebro, Jönköping | 89–90 % |
| Västerås, Linköping, Uppsala, Norrköping | 92–94 % |
| Kiruna tätort | 0 % (0 of 4 cells admitted) |

The producer therefore adds **fill windows**:

- A fill is centred on a 10 m corner inside the municipality polygon. That is
  exactly the map the live service makes for a click at that point, so it has
  the same admission, sources and provenance as any on-demand map.
- Fills are placed greedily per refused cell. Each fill covers the most
  uncovered municipality pixels it can, and ties go to the north-west-most
  corner, so the plan is deterministic.
- Pixels no admissible fill can reach would be counted as `unreachable` and
  reported, never dropped. The current config has none.
- A geometric check confirms the result: what remains uncovered is a few
  hundred m² of slivers per area, and none of it contains a pixel centre.

**Ownership is fixed by the plan:**

- A lattice window owns its whole cell.
- A fill owns the pixels of refused cells that no earlier fill covers.

Fills overlap admitted lattice windows but never replace them in the mosaic.
The alternative is to requalify the regions with a buffered polygon. That
changes the source configuration, so it means a new serving release, and it
would also widen what the live service offers. It is open question 1 below.

## Design

### 1. Batch producer: `scripts/produce_lclu_coverage.py`

**Same path as on-demand.**

- The producer opens the live release's `serving.json`, with the same R4
  checkpoint and qualified sources.
- Each window goes through the server's own sequence: a fresh
  `StorageBudget`, `RegionalSources` and `LcluService`, then
  `LcluService.run(grid, 2024)`.
- The model is loaded once (`TesseraR4`, as in `LcluService.from_manifest`).
- A window's request key is the one the live service computes for that grid
  and year (tested against an independent service). It holds as long as the
  numerical code is identical to r9's.

**Checks before any window.** A shard refuses to start unless all of these
hold:

- **Inference identity.** `contract_pin` of the service's inference identity
  (model contract, numerical code, runtime versions, embedding, legend,
  crop) equals `--expect-inference-pin`. The Job's code comes from a checkout
  at a pinned commit, while r9 runs from a source bundle. Equal pins prove
  the two compute the same thing. Take the pin from the `inference` field of
  any r9 result's `provenance.json` (on the volume under
  `/work/releases/lclu-stockholm-0124-r9/results/lclu/<key>/`, or downloaded
  from the public API):

  ```sh
  python -c "import json, sys; from scripts.cache_predict import contract_pin; \
  print(contract_pin(json.load(open(sys.argv[1]))['inference']))" provenance.json
  ```
- **Source type.** Every source configuration is a staged `cog_index` without
  `remote_tiles`. A remote-tile source trims its band cache during
  preparation (`enforce_band_budget`) and could delete the live service's
  bands.
- **Admission.** For every planned window, the manifest's own sources admit
  2024 (`sources.years(grid)`) exactly when the plan says the window is
  runnable. The plan derives admission from the campaign config, so this
  catches any drift from the qualified regions. Mismatches are listed.
  `--dry-run --manifest serving.json` runs the same two checks without
  producing.
- **Free space.** The volume must have the batch reserve plus one input and
  one map free (see storage below).
- **Exclusive shard.** The shard holds `flock` on `scratch/shard-i-of-N/.lock`
  for its whole lifetime and fails fast if it is taken. Staging cleanup only
  happens under that lock.

**Own storage.**

- Output goes to `/work/coverage/2024/<release>/`.
- Each shard uses `scratch/shard-i-of-N/` as the service root for its budget,
  prepared inputs and pending results.
- The producer refuses an output root that overlaps any `--live-root`, the
  manifest directory or `IMINT_API_OUTPUT_DIR`.
- It takes no API lock. In the synthetic run, every file and directory in
  the live tree keeps its size and modification time.
- **Free-space reserve.** The batch shares the volume with r9, whose reserve
  is 2 GiB (`docker/lclu/render_job.py`). The batch's reserve
  (`IMINT_LCLU_STORAGE_MIN_FREE_BYTES`) is 6 GiB: r9's 2 GiB, plus
  8 × (256 + 64) MiB for one input and one map per pod, plus mosaic headroom.
  The batch therefore stops writing before the live API starts refusing maps.
  A test ties the Job's value to r9's.

**One shared write: receipts.** Preparation records source-verification
receipts in the release's `identity_cache`
(`/work/releases/lclu-stockholm-0124-r9/shared/receipts`).

- Each receipt stores the file's stat snapshot, including `st_dev` and
  `st_ino` (`imint/sealed_io.py::_file_snapshot`), and is reused
  only while the snapshot matches.
- On CephFS, `st_dev` can differ per node. A pod on another node would then
  miss r9's receipt, re-hash the source and rewrite the receipt with its own
  snapshot, and r9 would do the same on its next use. The two could
  alternate, each re-hashing large sources.
- The writes are atomic and record the same identity, but they are **not**
  content-identical across nodes. Moving the receipts elsewhere would change
  the source configuration and with it the on-demand identity.

**Receipt pre-gate (required before the full run).**

1. On a node other than r9's, run `stat -c '%d %i'` on one pinned source
   (for example a VPP COG from r9's sources config).
2. Run the same command in r9's pod and compare.
3. If they are equal, receipts are stable across nodes.
4. If they differ, either co-locate the batch with r9 using the podAffinity
   option in the Job (all 8 × 4 CPU pods must then fit on r9's node), or
   measure the re-hash cost on the Stockholm run and accept it explicitly.

**Inputs are deleted.** After every attempt, successful or not, the shard
deletes all prepared inputs and staging leftovers in its scratch root.
Preparation logs move to `operator/preparation-logs/`. The tests check that at
most one input exists at any time.

**Outputs per window:**

- `windows/<id>/` holds the five service artifacts
  (`classification.tif`, `overlay.png`, `legend.json`, `provenance.json`,
  `result.json`), moved atomically from the service root.
- `windows/<id>.json` is the sidecar. A produced window records the window,
  request and cache keys, the source identity, the hashes of all five
  artifacts and the timings.

**Unavailable windows.** Some windows can never be prepared. The common case
is open water: Vättern at Jönköping, or the sea at Malmö and Göteborg.

- Preparation masks validity to the counties. When no valid TESSERA pixel
  remains, `validate_bundle` raises `InputUnavailable`. The child exits, and
  `RepositorySources._prepare` wraps the exit in a `RuntimeError`.
- The producer treats one cause as deterministic: the child's operator log
  ends with exactly
  `imint.inference.lclu.InputUnavailable: TESSERA saknar giltiga pixlar för platsen och året.`
  The same exception in process also counts.
- It prepares the window a second time. If the second attempt fails with the
  same cause, it writes a terminal `unavailable` sidecar. That records the
  reason, the cause line, the request key, `deterministic: true` and both
  attempts' log hashes.
- No other error counts as unavailable: a missing COG block, an I/O error, a
  timeout or any other `RuntimeError` stays a failure.
- The mosaic accepts unavailable windows and leaves their pixels as nodata,
  counted as `unavailable_pixels`.

**Resumable.**

- A window is skipped when its sidecar (and, if produced, all five artifacts)
  verifies, the release identity matches, and the request key equals the
  current one. A verified unavailable window is final.
- A corrupt, missing or stale window runs again.
- Failures are recorded in `failed/<id>.json` for the operator, and the shard
  continues.
- After 3 consecutive failures of the same type, the shard stops, because the
  error is systemic. The abort is recorded in its marker.
- The shard exits non-zero if any window failed or it aborted.

**Release identity.**

- `plan.json` and `release.json` are created atomically by the first shard,
  and every later shard must match them.
- `release.json` records the plan hash, the serving-manifest hash, the full
  inference identity and its pin.
- Any change (areas, config, code, model, runtime or manifest) needs a new
  release name. Run one Job per release at a time.

**Sharding.** `--shard i --num-shards N` takes every N-th runnable window in
plan order. The N shards partition the plan.

**Shard markers.**

- Each shard deletes its marker `done/shard-i-of-N.json` when it starts.
- When it finishes it writes the marker atomically: the run id
  (`--run-id`), counts, failed and unavailable windows, any abort, and the
  plan and inference identity.
- The mosaic waits for all N markers of its own run id. Markers from an
  earlier run, or a marker set of another size, do not count, and neither
  does sidecar existence.
- If any shard recorded a failure or an abort, the mosaic refuses.

**Dry run.** `--dry-run --num-shards 8` prints the per-area counts above and
the shard split, and writes nothing. `--list` prints one line per window.
`--manifest` adds the source-type and admission checks.

### 2. Seam gate (measure first, on Stockholm)

- Run with `--areas 0180 --release seam-stockholm`: 6 lattice and 16 fill
  windows.
- The metric is computed on the area mosaic. Over 4-neighbour pairs of
  classified pixels it reports the class-disagreement rate in three groups:
  - **seam** pairs, whose pixels come from different windows;
  - **near-edge** pairs, from the same window, with a pixel within
    `--seam-band` (default 3) px of that window's edge;
  - **interior** pairs.
- It also reports `seam_to_interior`, the seam rate divided by the interior
  rate. A seamless map gives about 1. In the synthetic tests a seamless field
  gives 0.5–2, and a relabelled neighbour gives more than 5.
- The statistics describe model behaviour, so they are **operator-only**.
  They go to `operator/<area>-2024-seams.json` and never into the coverage
  manifest that will be served publicly. Results stay private until the
  paper.
- The pass criterion must be fixed before the Stockholm numbers are seen. The
  code reports the metric and does not judge it (open question 3).
  - The synthetic range above is a test observation, not an acceptance
    threshold.
  - With no interior disagreement, `seam_to_interior` is `null`. The
    criterion must say what that means.
  - The criterion must also account for the compared pair counts and the
    land-cover composition along the seams.
- If seams show, the fallback is overlapping windows with a centre crop
  (stride 252 px, about 4× the compute), used only where needed.

### 3. Mosaic and products per area (`--mosaic`)

The mosaic step holds `flock` on `mosaic/.lock`. It writes partial files
under `mkstemp` names, removes leftover partials under the lock, and checks
free space against the reserve before each area. It runs only when all N
shard markers of its run are present and clean and every runnable window has
a verified sidecar (produced or unavailable); otherwise it lists what is
missing. For each area it writes:

- **`mosaic/<area>-2024.tif`:**
  - a COG at native 10 m in EPSG:3006, uint8, nodata 255, with the
    NMD-aligned palette from `lclu_legend.MAP_COLORS`;
  - overviews resampled with nearest neighbour, so no class is invented;
  - clipped to the municipality by the pixel-centre rule;
  - the downloadable GeoTIFF.

  The mosaic is assembled in 512-row strips and never held whole. Each
  window's raster must sit on its planned grid. The COG is reopened and
  compared pixel for pixel with the assembled strips before it is published.
- **`mosaic/<area>-2024.json`, the coverage manifest:**
  - label, release, year and area;
  - CRS, transform and shape, and the COG hash;
  - the checkpoint, embedding, inference pin and serving-manifest hash;
  - the distinct source configurations;
  - the campaign config and plan hashes, and the legend;
  - coverage pixel counts (inside, classified, model nodata, unavailable,
    uncovered) and window counts;
  - the window list: each window's status (produced, unavailable or
    refused), keys, and the hashes of its classification, provenance and
    input;
  - `produced_at`.

  It contains no model scores, evaluation results or seam statistics.
- **TODO, not implemented: XYZ PNG tiles (z8–z14) clipped to the
  municipality.** This depends on the serving decision (open question 4).
  The rendering must follow `export_map`: warp class IDs to EPSG:3857 with
  nearest neighbour before applying the palette, and keep class 0 opaque and
  nodata transparent. If FastAPI serves the COG directly, tiles can be cut on
  request instead.

### 4. Serving (Codex: backend route plus review)

- Read-only routes under the existing public prefix, so the ingress needs no
  change: `/v1/lclu/coverage` (the catalogue),
  `/v1/lclu/coverage/{area}/2024/{z}/{x}/{y}.png` and
  `/v1/lclu/coverage/{area}/2024.tif`.
- The files sit outside the runtime storage budget, and no request writes
  anything.
- They are served from the same r9 release, or the next one that carries the
  route.

### 5. Page (Claude)

- The page opens with "Heltäckande markslagskarta 2024" as the default layer.
  It has an opacity slider, the same legend, and pixel inspection from the
  COG's own value.
- Clicking a place still offers "Skapa egen årskarta" for 2017–2023. For
  2024 it points to the wall-to-wall layer.
- The text says clearly which is which.

## Jobs: `k8s/lclu-coverage-2024-job.yaml`

The file holds two Jobs. `kubectl apply --dry-run=client` and
`--dry-run=server` accept both.

**`lclu-coverage-2024`** is an Indexed Job of N = 8 pods, each with
4 CPU / 8 GiB (requests equal limits). Each pod runs its shard.

**`lclu-coverage-2024-mosaic`** is one pod with 1 CPU / 4 GiB. It waits for
the N markers of the run, then builds the mosaics. Create it after the
producer Job completes (the commands are in the yaml header), or alongside
it, since it waits cheaply.

Both Jobs share:

- **Source.** An `alpine/git` initContainer, pinned by digest, fetches the
  pinned commit with `--depth 1 --filter=blob:none` and a sparse checkout of
  `imint scripts config docker/lclu`, and checks `git rev-parse HEAD`. The
  Python base image (3.12.15 slim) has no git, and nothing is installed with
  `apt-get`.
- **Runtime.** r9's hash-locked runtime `/work/runtime/<lock>` is reused
  read-only; the Job fails if that runtime has no READY marker. The libexpat
  system package is extracted pod-locally.
- **Hardening.** All capabilities are dropped and privilege escalation is
  disabled, as for r9.
- **Placeholders.** `__SOURCE_COMMIT__` and `__INFERENCE_PIN__` must be filled
  in, with the same values in both Jobs; each script refuses to run with a
  placeholder. `RELEASE`, `RUN` and `SHARDS` must also match across the two
  Jobs (tested).
- **Resume.** With `backoffLimit: 0`, a failed shard fails the Job. To resume,
  bump `RUN` in both Jobs and recreate them; verified windows are skipped.
- **No `VPP_SOURCE`.** It is unset because nothing reaches the CDSE VPP
  router:
  - `cdse_vpp` is not imported anywhere in the producer's or the preparation
    child's import closure on the staged-COG path;
  - preparation reads HR-VPP only through `wekeo_vpp.fetch_vpp_tiles_local`;
  - `tile_fetch` is imported, through `enrich_tiles_tessera`, but only for
    `clean_stale_tile_tmps`, `infer_tile_year` and `list_tile_paths`;
  - nothing on the path calls `fetch_aux_channels`, `_get_vpp_doy_windows`
    or `fetch_vpp_tiles`.

  `test_cdse_vpp_is_unreachable_from_the_producer_and_preparation` keeps this
  true.
- **podAffinity.** The co-location option from the receipt pre-gate is in the
  producer Job as a comment.

## Order

1. Codex agrees the design and ownership, including the fill windows and open
   question 1.
2. The producer is cross-reviewed. Pin the Jobs' source commit and r9's
   inference pin. Run the Stockholm seam gate (its own release) and look at
   it.
3. Pass the receipt pre-gate (above).
4. The full run: 11 areas, N parallel pods. A time-bounded batch is not a
   long-running service, so the two-worker rule does not apply (Tobias
   2026-11-03). Tobias gives the go.
5. The serving route is built, reviewed and deployed in the next release; the
   page change goes in a PR.
6. Live check, then publication.

## Decisions and open questions

Codex answered the design questions in its review of PR 130 (2026-11-03).
Its answers do not authorise a deployment.

1. **Fill windows.** Keep the admitted lattice plus fill windows for the first
   release; the pixel ownership passes a full-area check over all 11 areas.
   Requalifying the regions with a buffered polygon would change the public
   qualification contract and needs its own decision.
2. **Result identity outside the server.** Rebuilding the server's objects per
   window around one loaded model is acceptable. Keep the request-key parity
   tests. A batch entry point in `lclu.py` waits for a concrete caller.
3. **Seam threshold: open.** It is fixed before the Stockholm results are seen;
   see the seam gate for what the criterion must cover.
4. **Serving.** Prefer a read-only coverage route in the existing FastAPI
   service, with immutable release URLs and nearest-neighbour categorical
   rendering. The tile delivery is decided there; the producer need not
   precompute every XYZ tile.
5. **Receipts and free space.** The cross-node receipt check and the
   concurrent free-space check stay deployment gates.
6. **Inference identity: open, Tobias decides.**
   - The pin stays fail-closed.
   - `imint/data/wekeo_vpp.py` (from PR 123) is part of
     `preparation_identity`, and main's version is not r9's, so a batch from
     main cannot carry r9's request keys.
   - The next serving release and the batch must run the same frozen
     inference code and runtime. Claude proposes r10 from main.
