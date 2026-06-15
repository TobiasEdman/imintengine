# SPEC — National 512 superset re-grid (S2 spectral fetch harmonisation)

**Created:** 2026-06-08
**Target repo:** `/Users/tobiasedman/Developer/ImintEngine`
**Status:** implemented, dry-run in progress — NMD native-window reads (`de335ac`); M1 grid-snap + M2 reference-anchored MI coreg orchestrator (`scripts/regrid_national_512.py`); 5-tile dry-run on the cluster. Scale step gated on dry-run + explicit go.
**Estimated diff:** ~3 lines in `tile_config.py` + 1 new orchestrator script (~300 LOC) + test updates; retires the `fill_tiles_l2a.py` keep-clean retrofit for this campaign

## Context

Overarching goal: fold the **full Sentinel-2 L2A spectral (12 bands)** into one fetch and put all S2 spectral on a single grid, so spectral + labels + aux are co-registered and one fetch yields complete training + sample material.

Harmonisation directive (verbatim, prior session): *"discard all the code that are not snapping the grid and coregister based on subpixel shifts. Those dead ends must die and the fetch strategy harmonised throughout the repo"* — scoped **Sentinel-2 only**.

Two distinct alignment mechanisms exist and are **both required** (this is the load-bearing correction):

- **M1 — grid snap (deterministic, transform-based):** `imint.fetch._snap_to_target_grid`. Reads the offset straight from each scene's source transform vs the target grid (`dx_m = src_x0 - target_w`); integer slice + Fourier sinc `subpixel_shift` for the fractional residual. Aligns each frame's **transform** to the grid. Per-scene, exact, not estimated.
- **M2 — inter-frame coregistration (estimated, image-based):** `imint.coregistration.estimate_mi_offset` (mutual information). Registers each frame's **content** onto the clearest (reference) frame, left untouched as the anchor. MI (not phase correlation) because the frames span seasons — same ground point, different radiometry — so intensity correlation chases phenology, not geometry. **Mandatory** because S2 L2A orthorectification is *relative* → real per-date geometric drift that M1 (transform-only) cannot remove. The `regrid-nmd-offset-probe` pod measured **up to ~2 px** relative inter-frame drift on the campaign tiles — multi-pixel, well above the ~0.3–1 px first assumed here (see §"Coregistration decision"). (Standing memory: `feedback_s2_coreg_mandatory`.)

The fill tool's old ~28 px coreg failure was M2 applied **across a grid gap** (fresh-on-bbox vs existing-on-S2-native, `transforms=None`, decorrelated content → the sub-pixel estimator's reject guard fired and returned 0,0). Re-gridding both frames onto the shared national grid is the precondition that makes M2 valid. M2 is **not** broken.

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
4. **M2** coreg on the shared 520 grid — reference-anchored MI, the clearest frame is the fixed anchor (see §"Coregistration decision"); up to ~2 px measured drift, absorbed by the 4 px halo.
5. Crop 520 → 512 centred `[4:516, 4:516]` (discards the sinc wrap-around ring).
6. **Labels + aux (no coreg → no halo):** `fetch_nmd_label_local` + `fetch_aux_channels` at the **512** national bbox directly. All final layers 512.
7. Write tile to a **NEW dir** (e.g. `/data/unified_national_512`), `national_grid=1` sentinel + `year`/`dates`/`center` carried.

Drive the all-band fetch by **stored dates**, not a fresh `optimal_fetch_dates` selection — preserves temporal matching (CLAUDE.md *Dataregler*) and avoids re-billing date selection.

### Change 3 — `build_labels` national (no logic change)

Labels become national automatically once the bbox is lattice-snapped. Two-step pipeline preserved (fetch national → build_labels national); never blend fetch and label logic.

### Change 4 — retire the fill keep-clean retrofit for this campaign

`scripts/fill_tiles_l2a.py` keep-if-clean is superseded — the re-grid re-fetches **all** frames fresh (existing spectral is on the S2-native/bbox grid, no stored per-frame transform → cannot shift in place → re-fetch is mandatory). Update `tests/test_spectral_harmonisation.py::test_clean_frame_kept_byte_identical` → fresh-for-all + national-snap + coreg-residual invariants. The `_snap_to_target_grid` and coreg-residual tests stay.

## Coregistration decision (a real finding — read before implementing)

**Three corrections landed here (all 2026-06-09). Read (0) first.**

> **UPDATE (2026-06-11): M2 is now reference-anchored mutual information.** The
> production inter-frame coreg (`imint.coregistration.coregister_interframe`,
> promoted out of this orchestrator into the library) registers each mover onto
> the **clearest reference frame** — the anchor is left untouched — using
> `estimate_mi_offset`, NOT phase correlation onto a composite centroid. That
> estimator swap + reference-anchor are the current truth: see CLAUDE.md
> §Koregistrering and the dot/COM sign guard
> `tests/test_coregistration.py::test_coregister_to_reference_removes_shift_dot_com`.
> Sections (0)–(2) below are retained as the historical record of the superseded
> phase-correlation/centroid design; the **sign-convention lesson in (0) still
> holds**, now applied to `estimate_mi_offset(moving, reference)` (moving = the
> frame being aligned; the returned shift is applied to it with a positive sign).

**(0) M2 had a sign error — it was AMPLIFYING inter-frame drift, not removing it.** `coregister_interframe` called `estimate_subpixel_offset(anchor, frame_i)` with the args reversed: that returns the anchor's position *relative to the frame* — the **negative** of the frame's drift — so the Pass-2 centroid correction `c − o_i` was applied backwards, pushing every frame the wrong way. A dot test made it unambiguous: a known **+3,−2 px** inter-frame drift came out **+7,−4 px** after coreg. Fix: pass the moving frame as `current` and the anchor as `reference` (`estimate_subpixel_offset(frame_i, anchor)`), so `offsets[i]` is the frame's true position vs the anchor; verified with a 4-frame centre-of-mass dot test (all frames collapse to the shared centroid). The old smooth-field residual unit test passed *even on the inverted sign* and was replaced by dot/centre-of-mass tests that fail loudly on any sign or axis error. **Any tile produced before this fix (including anything from `54b30a3`) is mis-registered — discard, do not trust.**

**(1) The "pure sub-pixel after M1" assumption was wrong.** M1 aligns each frame's *transform* to the lattice, but S2 L2A ortho is *relative*, so the frame **content** still drifts ~2 px (`regrid-nmd-offset-probe`). The old `>1 px` guard in `estimate_subpixel_offset` dropped that real drift to `0,0`. Fix (Part A): the guard is parameterised `max_peak_px` (default `1.0` = old behaviour, preserved for existing callers); the orchestrator passes `max_peak_px=_INTERFRAME_MAX_DRIFT_PX` (= 2·CROP = 8 px) — a non-central reference sees ~2× the per-frame drift *pairwise*, so capping at CROP would drop a real measurement. The *applied* shift is still halo-bounded (`> CROP` → skip).

**(2) M2 = relative inter-frame coreg onto the composite centroid.** Each frame's position `o_i` is measured vs `frames[ref_idx]`; the centroid `c = mean(o_i)`; every frame (anchor included) is shifted by `c − o_i`. Removes inter-frame drift **without** baking any single frame's absolute error into the stack, and is independent of `ref_idx` (which only sets the measurement origin).

```python
ref = clearest_frame_idx(frames)                              # measurement origin only
off = {ref: (0, 0)} | {                                       # i's position vs the anchor
    i: estimate_subpixel_offset(frames[i][B04], frames[ref][B04],   # frame=current, anchor=reference
                                max_peak_px=2 * CROP)
    for i in frames if i != ref
}
cy, cx = mean(dy in off), mean(dx in off)                     # composite centroid
for i in frames:                                              # land every frame on the centroid
    sy, sx = cy - off[i][0], cx - off[i][1]
    frames[i] = subpixel_shift(frames[i], sy, sx)             # skipped if > CROP or < 0.05 px
```

B04 (index 2) is the correlation band. Any M2 shift the 4 px halo cannot absorb (`> CROP`) is skipped; shifts `< 0.05 px` are a no-op (frame byte-identical).

**(3) Absolute alignment to NMD (M3) — UNDER MEASUREMENT, not yet built.** An earlier probe suggested the centroid was "zero-mean vs NMD (|mean| ≈ 0.05 px)", but that ran on the *broken* M2, so it is **not trusted**. With M2 now correct, the open question is whether M1 + fixed-M2 already land the stack on NMD or a residual absolute offset remains. Being measured directly: a clean 5-tile dry-run persists the pre-coreg frames (`--debug-save-precoreg`) and a viz pod renders a **before/after** coreg GIF plus an **S2-edge-vs-NMD-class-boundary** overlay/correlation. **Decision rule:** if the post-coreg S2 composite already sits on the NMD boundaries → no M3; if a consistent per-tile offset remains → add **M3 = align the whole co-registered stack to NMD class-boundary edges** (gradient-magnitude phase correlation — NMD is class-codes, so *edges* are the only shared signal; no co-registered optical reference exists), one full-stack shift after M2 and before the 520→512 crop, confidence-guarded (skip boundary-poor tiles and shifts the halo can't absorb). If built, M3 makes the orchestrator read NMD **as a geometric ruler only** (not label generation) and degrades to a no-op when the raster is absent — preserving fetch/label independence.

(`coregister_timeseries` (line 360) is still **not used**: `coregister_to_reference` (line 249) shifts its *reference* arg toward the target and returns the *unshifted* target — `coregister_timeseries` then stores that unshifted frame (line 413), recording offsets in metadata but never applying them. The orchestrator calls the `estimate_subpixel_offset` + `subpixel_shift` primitives directly. Optional out-of-scope follow-up: fix or document that trap.)

## Out of scope

- **The live 256 dataset (`/data/unified_v2`)** — superseded by the national 512 superset, left **untouched** (not deleted, not mutated).
- **Existing `/data/unified_v2_512`** — retained as-is until the national set is verified at scale; no in-place mutation.
- **Aux / labels / training-crops / super-res / inference / viz logic** — they inherit the national grid automatically; their logic is not changed.
- **Non-S2 fetchers** — NMD/LPIS/SKS reference layers, marine, etc.
- **The `imint/fetch.py` LOC de-duplication** — that is `SPEC_fetch_refactor.md`, a separate behaviour-preserving structural refactor.
- **Re-selecting scene dates** — stored dates are reused verbatim.

## Data & state

- **Read:** `unified_v2_512/*.npz` (**6,916 unique centres** — re-counted 2026-06-09; the earlier "142" was stale) for stored dates/centre; orphan list from `scripts/build_orphan_fetch_list.py` / `/cephfs/audits/tile_ledger.jsonl` (~1,147 centres, carries `year`); `data/nmd/…tif` for the grid; aux source rasters.
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
| Cost | PU/tile, wall-time, **aux cold-cache rebuild** measured → extrapolate ~8,063 tiles | dryrun timing |

Unit: `pytest tests/test_spectral_harmonisation.py` (rewritten keep-clean → fresh-for-all; snap + coreg-residual cases retained).

## Scale (after dry-run approval only)

~8,063 tiles (6,916 existing 512 centres + ~1,147 orphan centres) × 4 frames all-band, K8s CPU job → national dir; then `build_labels` over the national dir. Existing 512 set retired only after the national set passes verification at scale. (Scope re-counted 2026-06-09 — ~6× the original "~1,289" estimate; affects the scale-step PU budget, not the 5-tile dry-run.)

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
