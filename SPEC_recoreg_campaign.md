# SPEC — Decoupled 2-phase re-fetch campaign (des-fast + sen2cor-separate)

**Created:** 2026-06-15
**Target repo:** ImintEngine — execute on `main` (PR #26 "Fetch Phase 4" is merged)
**Status:** draft, pending fresh-session execution
**Note:** filename is `SPEC_recoreg_campaign.md` because `SPEC.md` holds an unrelated draft
(WaterQualityAnalyzer). Don't touch that one.

> **CORRECTION (2026-06-16) — Phase 2 AUGMENTS the existing `scripts/sen2cor_pipeline/`; it does NOT build a new `fetch_pre2018_sen2cor.py`.**
> The granule selection + per-scene sen2cor + frame write-back the text below proposes building from scratch
> ALREADY EXISTS and is production-proven (it wrote the current `frame_2016` + 2017-slot0 data via the
> `sen2cor-frame2016-512` / `sen2cor-slot0-2017-512` jobs): `select_scenes.py` (ERA5/STAC + greedy set-cover)
> + `run_sen2cor_per_scene.py` (SAFE → COT-gate → sen2cor → `_write_frame_2016` / `_write_temporal_slot`).
> Phase 2 is therefore **three changes only**:
> 1. `select_scenes._resolve_tile`: reuse each tile's **STORED** date (`frame_2016_date` / slot-0) → ±7 d window →
>    **UTM-zone filter** (drops CDSE-STAC antimeridian false-positives, MGRS 01/60) → closest scene; ERA5 fallback
>    only when no stored date. Set-cover then just groups tiles by the resolved granule.
> 2. `run_sen2cor_per_scene`: insert `imint.coregistration.coregister_to_reference` (M2, B04 MI vs the tile's
>    Phase-1 anchor) before the write-back — the existing pipeline does **not** coregister; this is the only real gap.
> 3. Persistent keep-all granule cache (optional; the runner currently purges SAFE/L2A per scene).
> The from-scratch Stage-0 `build_pre2018_granule_map.py` (commit `b367aaf`) is **SUPERSEDED** — its stored-date +
> zone logic ports into `select_scenes._resolve_tile`. Decisions (user, 2026-06-16): augment existing; reuse stored dates.

## Context

PR #26 routed all S2 fetch through the canonical M1+M2 entry
`imint/training/fetch_spectral.py::fetch_tile_spectral`: per-slot des downloads with an era-split
(≥2018 → des openEO tile-graph; pre-2018 → `l1c_sen2cor` = GCS L1C → sen2cor → L2A). A 10-tile
validation batch confirmed the path works but is **sen2cor-bound (~15 min/tile)** — re-fetching the
full **6921-tile** `unified_v2_512` in one coupled pass at `--workers 2` ≈ ~weeks. des (≥2018) is
~3 min/tile; sen2cor (pre-2018) is the long pole. This spec splits the campaign into a fast des-only
**Phase 1** and a separate, granule-efficient sen2cor **Phase 2**, then promotes.

Two pre-2018 frames need sen2cor:
- **2016 summer background** (`frame_2016`) — on ~all tiles (change-detection overlay).
- **2017 autumn** — only slot 0 of **2018-labelled** tiles (autumn = year-1 = 2017). 2019/2022/2023
  tiles have slot 0 ≥2018 (des-able).

## In scope

### Phase 1 — des-only re-coreg + orphan fresh-fetch (fast, `--workers 2`)
1. **Re-coreg existing 512 tiles:** for every tile in `unified_v2_512`, re-fetch ONLY its ≥2018 slots
   through the entry (per-frame des, M2-coregister), reusing the tile's STORED dates (`refetch_tile`
   semantics — never re-select). Pre-2018 temporal slot(s) left empty in Phase 1. Write to a **new
   dir** `unified_v2_512_recoreg` (live dataset untouched).
2. **Orphan tiles** = **the 256 tiles with no 512 cooccurrence** (locations in the 256 dataset that
   have no tile in `unified_v2_512`). FRESH 512 fetch (`fetch_tile` path) at those locations for the
   ≥2018 slots, into `unified_v2_512_recoreg`. (No existing 512 npz → fresh path, not refetch.)
3. Phase-1 output per tile: 4-frame temporal cube with ≥2018 slots filled + M2-coregistered;
   pre-2018 slot(s) zero-filled with `temporal_mask` recording the gap; `coreg_*` provenance from the
   ≥2018 M2 pass; NO `frame_2016` yet.

### Phase 2 — sen2cor pre-2018 frames, coreg-to-reference (separate process)
4. For each Phase-1 tile, fetch its pre-2018 frames (2016 background always; 2017 autumn iff a 2018
   tile with an empty slot 0) via sen2cor, then **coregister each back to that tile's Phase-1 anchor**
   with `imint.coregistration.coregister_to_reference` (MI offset on B04 vs the anchor frame, applied
   POSITIVELY — keeps 4b M2 alignment; argordning/tecken are sign-bearing, see CLAUDE.md M2 section).
   Write `frame_2016*` + the (re-coregistered) slot-0 back into the `_recoreg` tile.
5. **sen2cor efficiency (core of Phase 2):**
   - **Granule dedup / mapping:** build a map of which S2 granules (granule_id + date) are needed
     across ALL Phase-2 tiles, download + run sen2cor on each granule **exactly once**, then
     window/crop per tile from the processed granule. Many tiles share a granule → avoids redundant
     per-tile sen2cor (the ~15 min cost). This is the main throughput win.
   - **Persistent processed-granule cache (KEEP-ALL):** save every sen2cor-processed L2A granule
     permanently on the PVC at `/data/l2a_granule_cache`, keyed by `granule_id + date +
     sen2cor_version`. Cache is **first choice**; on miss, **GCS L1C → sen2cor** is the fallback (and
     populates the cache). Reused across this campaign AND future runs.

### Promotion
6. After both phases + verification on a sample, **promote** `unified_v2_512_recoreg` →
   `unified_v2_512` (swap/rename). Live dataset stays intact until then.

## Out of scope

- The **matplotlib CI fix** (`imint/exporters/export.py` `get_cmap` → `matplotlib.colormaps` + pin
  matplotlib). Tracked separately; main CI is red on it (orthogonal, pre-existing dep drift).
- Re-architecting the entry's per-frame des fetch (done in PR #26).
- The 256 dataset itself (read-only source for orphan locations).
- Model/training changes.

## Interface

- **Phase 1:** `scripts/fetch_unified_tiles.py` (existing) —
  `--mode refetch --from-existing /data/unified_v2_512 --output-dir /data/unified_v2_512_recoreg
  --tile-size-px 512 --workers 2 --fetch-sources des --force`.
  ⚠️ **des ONLY** (not `des,l1c_sen2cor`) — Phase 1 must SKIP pre-2018 slots. Today the era-split
  still *requests* pre-2018 dates and lets `l1c_sen2cor` fill them; Phase 1 wants those slots
  **omitted entirely**. **Code change needed:** a "skip pre-2018" mode, e.g.
  `fetch_tile_spectral(..., skip_pre2018=True)` / a `--no-pre2018` CLI flag, so the entry drops
  pre-`DES_L2A_FLOOR` slots from `slot_dates` before fetching (no l1c fallthrough).
  Orphans: `--mode refetch --from-json <orphan_locations.json>` (or the fresh path) into the same out
  dir, also des-only.
- **Phase 2:** NEW driver `scripts/fetch_pre2018_sen2cor.py` + k8s job on the `imint-sen2cor` image:
  enumerate Phase-2 tiles → build granule map → process granules (cache-first) → window +
  coreg-to-reference → write back into `_recoreg`.
- **k8s:** `k8s/campaign-phase1-des-recoreg-job.yaml` (python:3.11-slim ok; des-only) and
  `k8s/campaign-phase2-sen2cor-job.yaml` (imint-sen2cor image; mounts the granule cache). Both clone
  `main`, `export PYTHONPATH=<repo>`, pin image by digest, print `CLONED HEAD`.

## Data & state

- **Reads:** `/data/unified_v2_512` (existing 6921 tiles — STORED dates); the 256 dataset for orphan
  locations (**path TBD — open question**).
- **Writes:** `/data/unified_v2_512_recoreg` (Phase 1 + Phase 2), then promoted to
  `/data/unified_v2_512`.
- **Granule cache:** `/data/l2a_granule_cache/` (keep-all, keyed `granule_id+date+sen2cor_version`).
- PVC: `training-data-cephfs` at `/data`.

## Dependencies

- New deps: none expected (numpy/scipy/rasterio/pyproj/openeo/shapely already used; sen2cor via the
  imint-sen2cor image).
- **Reuse — do NOT rebuild:** `fetch_tile_spectral` (entry, per-frame des), `refetch_tile`/`fetch_tile`
  (`scripts/fetch_unified_tiles.py`), `imint.coregistration.coregister_to_reference` +
  `estimate_mi_offset`, `imint/training/sen2cor_l2a.py` (`stac_l1c_scenes`, `run_sen2cor`,
  `read_l2a_allband`, GCS `fetch_l1c_safe_by_name`), `fetch_spectral._l1c_sen2cor_allband_cube`,
  `add_background_frame.py` (reference for the CDSE background path), `DES_L2A_FLOOR`.
- **Must not break:** the merged PR #26 fetch path; `add_background_frame.py` + its 4 k8s jobs;
  `fetch_lucas_tiles.py` (`stack_frames`).
- Off-limits: other repos.

## Failure modes & verification

| Scenario | Expected behavior | Verification |
|---|---|---|
| Phase-1 des slot fails | per-slot skip (partial success); tile kept if ≥3/4 ≥2018 frames | layout + `temporal_mask` |
| Phase-2 sen2cor fails for a granule | GCS L1C→sen2cor fallback; if that fails, mark granule dead, skip dependent tiles' pre-2018 frame (gap) | log + cache miss path |
| Phase-2 anchor missing (Phase-1 left <1 valid ≥2018 frame) | skip coreg-to-reference; do NOT write a mis-aligned pre-2018 frame | per-tile guard + log |
| Granule cache hit | reuse processed L2A; no re-download/re-sen2cor | cache-hit count in summary |
| Tiles share a granule | granule processed once; all window from it | dedup-map unit test |
| Coreg correctness | pre-2018 frame aligned to anchor | **reverse-fit dot/center-of-mass test** on a sample + `coreg_*` quality fields |
| Restart / partial campaign | idempotent — skip tiles already in `_recoreg` (P1) / granules in cache (P2) | re-run no-ops on done work |

**Verification before promotion (rigorous, per user):** per-phase assert npz layout (shapes/keys);
check coreg-quality fields (`coreg_m2`, `coreg_max_shift`, `coreg_anchor_valid_frac`); AND a
**reverse-fit dot/center-of-mass coreg test** on a sample of Phase-2 frames (inject a known shift,
confirm coreg removes it — per `tests/test_coregistration.py`). Do NOT promote on red verification.

## Constraints

- **Hard:** Phase-1 `--workers 2`. New-dir output ⇒ ~2× dataset storage during the run. sen2cor is
  compute/memory-heavy (L2A_Process ~1h internal timeout per granule). Image pinned by digest.
- **Soft:** keep-all granule cache (bounded by Sweden's finite S2 granule set × window dates; monitor
  PVC). Phase-2 worker/pod resources TBD (sen2cor-bound).

## Tradeoffs accepted

- **New-dir → promote** over in-place: safer (live dataset intact until swap), ~2× storage during run.
- **M2-coregistered pre-2018 (Phase-2 coreg-to-reference)** over the fast non-M2
  `add_background_frame.py`: keeps 4b alignment, at Phase-2 complexity cost.
- **Keep-all granule cache** over LRU: simpler + cross-run reuse, at unbounded-ish PVC growth
  (mitigated by finite granule set).
- **Decouple sen2cor** over the coupled 5-slot entry fetch: entry stays the per-tile des authority;
  Phase 2 is a separate granule-oriented process. Per-tile one-pass elegance traded for throughput.

## Execution hints

- **Files likely to change / add:**
  - `imint/training/fetch_spectral.py` — add "des-only / skip pre-2018" mode to `fetch_tile_spectral`.
  - `scripts/fetch_pre2018_sen2cor.py` (new) — Phase-2 driver (granule map + cache + coreg-to-ref).
  - `imint/training/sen2cor_l2a.py` — granule-cache get/put helpers (key incl. sen2cor_version).
  - `scripts/enumerate_orphan_256_tiles.py` (new) — 256-without-512 → locations JSON.
  - `k8s/campaign-phase1-des-recoreg-job.yaml`, `k8s/campaign-phase2-sen2cor-job.yaml` (new).
  - A promote script (rename/rsync `_recoreg` → `unified_v2_512`).
- **Tests to add:** granule-dedup mapping (N tiles → 1 granule); cache hit/miss + key incl.
  sen2cor_version; coreg-to-reference reverse-fit (known shift removed); Phase-1 des-only omits
  pre-2018; orphan enumeration; idempotent restart.
- **Rollback:** new-dir ⇒ rollback = don't promote (delete `_recoreg`). Granule cache is additive
  (safe to keep). No live-dataset mutation until the explicit promote.
- **Templates:** `k8s/smoke-through-entry-5frame-job.yaml` (sen2cor image, creds, PVC, PYTHONPATH) and
  `k8s/campaign-refetch-validation-job.yaml` (refetch invocation).
- **Session lessons:** always `export PYTHONPATH=<repo>` in k8s bash (a smoke failed on `import imint`
  without it); pin the sen2cor image by digest; verify the pod's `CLONED HEAD` matches intent.

## Open questions

- **Exact path + identifier for the 256 dataset** (to enumerate "256 without 512"). Match by tile
  center/name (`tile_<easting>_<northing>`)?
- **Phase-2 worker/pod resources** — concurrent sen2cor runs per pod / number of pods (compute-bound).
- **Promote mechanism** — atomic rename, rsync, or symlink swap? Any live consumers of
  `unified_v2_512` during promotion?
- **Do orphan tiles also get Phase-2 pre-2018 frames?** (Fresh 512 fetches — presumably same 2-phase
  treatment; confirm.)
- **2017 autumn for 2018 tiles** — count of 2018-labelled tiles (sizes the 2017 sen2cor burden);
  confirm slot-0 re-coreg-to-reference is wanted vs leaving existing slot-0.
- **Granule cache monitoring** — keep-all chosen; add a PVC size alarm?
- **Validation-batch timing** — `campaign-refetch-validation` (this session) gives real per-tile P1
  des timing to size Phase 1; check its summary at `/data/debug/campaign_validation/`.
