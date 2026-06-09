# SPEC — National 512 superset re-grid (S2 spectral fetch harmonisation)

**Created:** 2026-06-08
**Target repo:** `/Users/tobiasedman/Developer/ImintEngine`
**Status:** design approved (store-fork A chosen), pending implementation — no code until this spec is executed
**Estimated diff:** ~3 lines in `tile_config.py` + 1 new orchestrator script (~300 LOC) + test updates; retires the `fill_tiles_l2a.py` keep-clean retrofit for this campaign

## Context

Overarching goal: fold the **full Sentinel-2 L2A spectral (12 bands)** into one fetch and put all S2 spectral on a single grid, so spectral + labels + aux are co-registered and one fetch yields complete training + sample material.

Harmonisation directive (verbatim, prior session): *"discard all the code that are not snapping the grid and coregister based on subpixel shifts. Those dead ends must die and the fetch strategy harmonised throughout the repo"* — scoped **Sentinel-2 only**.

Two distinct alignment mechanisms exist and are **both required** (this is the load-bearing correction):

- **M1 — grid snap (deterministic, transform-based):** `imint.fetch._snap_to_target_grid`. Reads the offset straight from each scene's source transform vs the target grid (`dx_m = src_x0 - target_w`); integer slice + Fourier sinc `subpixel_shift` for the fractional residual. Aligns each frame's **transform** to the grid. Per-scene, exact, not estimated.
- **M2 — inter-frame coregistration (estimated, image-based):** `imint.coregistration` phase correlation. Aligns frame **content** to a reference frame. **Mandatory** because S2 L2A orthorectification is *relative* → real per-date geometric drift (~0.3 px, up to ~1 px on older baselines) that M1 (transform-only) cannot remove. (Standing memory: `feedback_s2_coreg_mandatory`.)

The fill tool's old ~28 px coreg failure was M2 applied **across a grid gap** (fresh-on-bbox vs existing-on-S2-native, `transforms=None`, decorrelated content → the `estimate_subpixel_offset` >1 px guard fired and returned 0,0). Re-gridding both frames onto the shared national grid is the precondition that makes M2's phase correlation valid. M2 is **not** broken.

**Snap target = the Swedish national NMD 10 m lattice.** Probed from `data/nmd/nmd2018bas_ogeneraliserad_v1_1.tif`: EPSG:3006, origin (208450.0, 7671060.0), pixel 10/−10, lattice phase 0.0/0.0 (exact 10 m multiples). The current `bbox_from_center` integer-metre grid is 0–0.5 px off this lattice.

**Store fork = A (decided 2026-06-08).** Store the **cropped 512 as canonical**; the 520 halo is an in-orchestrator scratch buffer, cropped away before write. Rationale: crop happens once at write time (not at every reader, forever); on-disk size == model input size == 512; matches the existing dataset convention (tiles are model-ready). The halo's only job — absorbing M2's sinc wrap-around at the edges — happens entirely during fetch+coreg.

## In scope

### Change 1 — `imint/training/tile_config.py::bbox_from_center` (the one cascading change)

Snap the centre to the 10 m national lattice before building the bbox:

```python
gsd = int(self.gsd_m)                 # 10
cx = round(int(east) / gsd) * gsd
cy = round(int(north) / gsd) * gsd
h = self.half_m
return {"west": cx-h, "east": cx+h, "south": cy-h, "north": cy+h}
```

With even `size_px` (520/512/256), `half_m` (`size_m // 2` = 2600/2560/1280) is a 10 m multiple, so all four edges land on the national lattice. Add an assert: `bbox["west"] % gsd == 0 and (bbox["east"]-bbox["west"]) % gsd == 0`. Odd `size_px` breaks edge alignment → out of scope, the assert guards it.

Cascade: this single change makes 520/512/256 — when built from the same centre — all lattice-aligned and co-centred, so the 512 is a clean centred crop of the 520, and `fetch_nmd_label_local` (`from_bounds` → integer window → `Resampling.nearest`) becomes a pixel-exact no-op, and aux (`_to_nmd_grid_bounds`) aligns automatically.

### Change 2 — new orchestrator script (e.g. `scripts/regrid_national_512.py`)

Per tile:
1. Read source (existing `unified_v2_512` npz | orphan 256 npz / ledger): centre `C`, **stored dates** `D[4]`, `year`.
2. Snap `C → C'` on the 10 m lattice (via the new `bbox_from_center`).
3. **Spectral (needs coreg → halo):** fetch **520** all-band @ `D[4]` through the tested all-band path; per-scene **M1** snaps each frame's transform to `TileConfig(520).bbox_from_center(C')`. Integer offset between frames is now ≡ 0 by construction.
4. **M2** coreg on the shared 520 grid — explicit reference-loop (see §"Coregistration decision"), pure subpixel.
5. Crop 520 → 512 centred `[4:516, 4:516]` (discards the sinc wrap-around ring).
6. **Labels + aux (no coreg → no halo):** `fetch_nmd_label_local` + `fetch_aux_channels` at the **512** national bbox directly. All final layers 512.
7. Write tile to a **NEW dir** (e.g. `/data/unified_national_512`), `national_grid=1` sentinel + `year`/`dates`/`center` carried.

Drive the all-band fetch by **stored dates**, not a fresh `optimal_fetch_dates` selection — preserves temporal matching (CLAUDE.md *Dataregler*) and avoids re-billing date selection.

### Change 3 — `build_labels` national (no logic change)

Labels become national automatically once the bbox is lattice-snapped. Two-step pipeline preserved (fetch national → build_labels national); never blend fetch and label logic.

### Change 4 — retire the fill keep-clean retrofit for this campaign

`scripts/fill_tiles_l2a.py` keep-if-clean is superseded — the re-grid re-fetches **all** frames fresh (existing spectral is on the S2-native/bbox grid, no stored per-frame transform → cannot shift in place → re-fetch is mandatory). Update `tests/test_spectral_harmonisation.py::test_clean_frame_kept_byte_identical` → fresh-for-all + national-snap + coreg-residual invariants. The `_snap_to_target_grid` and coreg-residual tests stay.

## Coregistration decision (a real finding — read before implementing)

`imint/coregistration.py::coregister_timeseries` (line 360) **must not be used blind**: `coregister_to_reference` (line 249) shifts the **reference** arg toward the target (lines 335–341, `reference[...] = subpixel_shift(reference, -dy, -dx)`) and returns the **unshifted** target; `coregister_timeseries` then stores that unshifted target (line 413). It records per-frame offsets in metadata but does not apply subpixel correction to the returned frames.

The orchestrator uses an **explicit reference-loop** on the co-gridded stack:

```python
ref = clearest_frame_idx(frames)                     # reuse the fetch path's cloud/variance metric
for i != ref:
    dy, dx = estimate_subpixel_offset(frames[ref][B04], frames[i][B04])  # offset of i rel. to ref
    for b in bands:
        frames[i][b] = subpixel_shift(frames[i][b], -dy, -dx)            # align i → ref
```

Integer offset is 0 by construction (M1), so this is **pure subpixel** — the regime the >1 px guard (`estimate_subpixel_offset`, line 235) expects, so the guard won't fire and the 28 px failure cannot recur. B04 (index 2) is the correlation band, matching `reference_band=2`.

(Optional follow-up, out of scope here: fix or document `coregister_timeseries` so its returned frames are actually aligned.)

## Out of scope

- **The live 256 dataset (`/data/unified_v2`)** — superseded by the national 512 superset, left **untouched** (not deleted, not mutated).
- **Existing `/data/unified_v2_512`** — retained as-is until the national set is verified at scale; no in-place mutation.
- **Aux / labels / training-crops / super-res / inference / viz logic** — they inherit the national grid automatically; their logic is not changed.
- **Non-S2 fetchers** — NMD/LPIS/SKS reference layers, marine, etc.
- **The `imint/fetch.py` LOC de-duplication** — that is `SPEC_fetch_refactor.md`, a separate behaviour-preserving structural refactor.
- **Re-selecting scene dates** — stored dates are reused verbatim.

## Data & state

- **Read:** `unified_v2_512/*.npz` (142 centres) for stored dates/centre; orphan list from `scripts/build_orphan_fetch_list.py` / `/cephfs/audits/tile_ledger.jsonl` (~1,147 centres, carries `year`); `data/nmd/…tif` for the grid; aux source rasters.
- **Write:** new dir `/data/unified_national_512` (name TBD) only. Dry-run → `/data/_national_dryrun`.
- **Mutate:** nothing existing.

## Failure modes & verification

Dry-run **5 tiles → `/data/_national_dryrun`**, report all five checks; **scale only on explicit go**:

| Check | Expected | How |
|---|---|---|
| Lattice alignment (by construction) | every output bbox edge `% 10 == 0`; NMD window offset integer | assert in orchestrator + dryrun verify |
| Inter-frame residual (measured) | `estimate_subpixel_offset(ref, frame_i)` **after** coreg `< 0.1 px` ∀ i | dryrun verify |
| Spectral↔label overlay | no visible shift (spectral edge band over NMD class boundary) | screenshot |
| L2A floor | minpos `< 0.095` per frame | existing dryrun check |
| Cost | PU/tile, wall-time, **aux cold-cache rebuild** measured → extrapolate ~1,289 tiles | dryrun timing |

Unit: `pytest tests/test_spectral_harmonisation.py` (rewritten keep-clean → fresh-for-all; snap + coreg-residual cases retained).

## Scale (after dry-run approval only)

~1,289 tiles (142 existing 512 centres + ~1,147 orphan centres) × 4 frames all-band, K8s CPU job → national dir; then `build_labels` over the national dir. Existing 512 set retired only after the national set passes verification at scale.

## Constraints

**Hard:**
- Snap target is the **NMD national 10 m lattice** (origin 208450/7671060, phase 0/0), **not** the integer-metre bbox grid.
- **M2 coreg is mandatory** and runs **after** M1 on the shared grid — never replaced by the snap.
- Even `size_px` only (assert-guarded).
- **Stored dates reused** (temporal matching — spectral year must match label year; no year fallback for crop tiles).
- Write to a **NEW dir**; never mutate `unified_v2` or `unified_v2_512` in place.
- **Ask before scaling** beyond the 5-tile dry-run (CLAUDE.md: never restart/discard work without asking).
- Persist all fetched/computed data to disk (never in-memory only).
- Commit trailers: `Verified-by:` + `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.

**Soft:** match repo style (`from __future__ import annotations`, type hints, 4-space); reuse existing fetch/aux primitives — no new openEO calls.

## Tradeoffs accepted

- **Re-fetching the 142 existing 512s is mandatory** (re-grid cannot be done in place — no stored per-frame transform). PU cost accepted; superseded spectral discarded only after the national set verifies.
- **Halo discarded after crop** (store-fork A). Loses free re-cropping/re-coreg of the deprecated 256; if "never re-pay PU for coreg" later becomes a hard requirement, cache raw pre-snap frames separately (option C) — heavy disk, deferred.
- **Aux cold-cache rebuild** is the one unmeasured cost — priced in the dry-run before scale.

## Execution hints

**Files to modify / add:**
- [imint/training/tile_config.py](imint/training/tile_config.py:54) — `bbox_from_center` snap + assert
- `scripts/regrid_national_512.py` — new orchestrator (name TBD)
- [tests/test_spectral_harmonisation.py](tests/test_spectral_harmonisation.py) — keep-clean → fresh-for-all
- [k8s/](k8s/) — a national dry-run + scale job (clone the fix branch; `kubectl apply --dry-run=server` first)

**Files NOT to modify:** `imint/fetch.py` (M1 already correct), `imint/coregistration.py` (use primitives as-is), `imint/training/tile_fetch.py`, the live `unified_v2`/`unified_v2_512` data.

**Suggested sequence:**
1. `bbox_from_center` snap + test (smallest, most-cascading) — verify `pytest tests/test_spectral_harmonisation.py` + a lattice-alignment assertion test.
2. Orchestrator (M1 already in fetch; add M2 reference-loop + crop + labels/aux + write).
3. Dry-run 5 tiles → report 5 checks.
4. Scale on explicit go → build_labels national.

## Open questions (implementation-time confirmations — flagged, not invented)

1. Exact all-band-at-stored-dates entry point (`fetch_4frame_scenes(..., collect_extra=…)` vs an at-specific-dates wrapper in the fetch stack).
2. "Clearest frame" metric for the coreg reference — reuse whatever cloud-frac/variance the fetch path already computes.
3. Aux cold-cache rebuild cost (measured in dry-run, not assumed).
4. Final national dir name.

---

**Reference paths cited:**
- M1: [imint/fetch.py](imint/fetch.py) `_snap_to_target_grid`
- M2: [imint/coregistration.py](imint/coregistration.py:360) (`coregister_timeseries` trap), `estimate_subpixel_offset` (guard line 235), `subpixel_shift`, `coregister_to_reference`
- bbox: [imint/training/tile_config.py](imint/training/tile_config.py:54)
- all-band fetch + persist: [scripts/fetch_unified_tiles.py](scripts/fetch_unified_tiles.py)
- orphan ledger: [scripts/build_orphan_fetch_list.py](scripts/build_orphan_fetch_list.py)
- retired retrofit: [scripts/fill_tiles_l2a.py](scripts/fill_tiles_l2a.py)
- grid source: `data/nmd/nmd2018bas_ogeneraliserad_v1_1.tif`
- standing memory: `feedback_s2_coreg_mandatory`
- repo conventions: [CLAUDE.md](CLAUDE.md); global rules `~/.claude/CLAUDE.md`
