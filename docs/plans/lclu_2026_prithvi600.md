# 2026 land-cover maps with Prithvi-EO-2.0 600M and a 2016 composite per municipality

Status: proposal, revised 2026-10-30 after Tobias' correction. Nothing has
been run. Scope: the IM-024 demo areas — the ten municipalities plus Kiruna
tätort in [`config/lclu/municipalities.json`](../../config/lclu/municipalities.json).
Each cluster step needs Tobias' approval and `kubectl create --dry-run=server`
first. Decisions that are Tobias' to make are marked **Decision**.

## Summary

2026 cannot be served by the current demo model. TESSERA R4 needs TESSERA
embeddings and HR-VPP, and neither exists for 2026. Prithvi-EO-2.0 600M reads
Sentinel-2 directly, so it can cover 2026.

**The 2016 reference is an aux channel.** 2016 enters the models as the
clearcut reference: ΔSAR, two aux channels `delta_vv` and `delta_vh`, each the
Sentinel-1 growing-season γ⁰ minus the 2016 γ⁰ (`0091835e`, 2026-08-17). A
fresh clearcut gives a sharp negative step. The optical `frame_2016` is not a
model input anywhere.

**No existing Prithvi-600M checkpoint can produce 2026 maps without a
retrain:**
- **No 2016 reference.** CROMA and TerraMind are trained with ΔSAR;
  Prithvi-600M and TESSERA R4 are not.
- **VPP.** Every Prithvi-600M checkpoint needs five HR-VPP channels for the
  map year. HR-VPP 2026 is not published (2025 is not either), and the service
  never substitutes another year.

The plan therefore has seven parts:
1. Store a Sentinel-1 2016 growing-season composite per area, built exactly
   as the training enrichment builds it per tile.
2. Train Prithvi-600M without VPP channels, with and without ΔSAR.
3. Pick the 2026 frame dates without VPP.
4. Fetch 2026 Sentinel-2 frames and the Sentinel-1 2026 season composite for
   the eleven areas.
5. Add a second, clearly labelled model to the service.
6. Measure, before any result is shown, how much of the 2025→2026 difference
   is the change of model rather than of land cover.
7. Settle one open question: how 2025 and 2026 frames are loaded (below).

## What the repository says today

| Fact | Where |
|---|---|
| ΔSAR = 10·log10(season γ⁰) − 10·log10(2016 γ⁰), computed at read time from the tile's `s1_vv_vh` and `s1_vv_vh_2016`; normalised with mean 0 and std 4 dB; a tile without the 2016 composite gets zeros ("no change") | `imint/training/unified_dataset.py` `compute_delta_sar`, `imint/training/config.py` `enable_delta_sar_channels` |
| `--enable-delta-sar` is on for CROMA and TerraMind (all four ladder rungs); off for every Prithvi-600M and TESSERA job | `k8s/ladder/ladder-r*-{croma,terramind}-job.yaml`, `k8s/ladder/ladder-r*-prithvi600m-job.yaml` |
| The S1 composites are per-orbit medians of ≤ 3 scenes of Planetary Computer `sentinel-1-rtc` (RTC γ⁰, linear, windowed reads, free and anonymous). The label year and 2016 share the tile's dominant orbit and the same day-of-year window: VPP SOSD→EOSD, or a latitude-scaled May–September fallback. `s1_vv_vh_2016` is written only where `has_frame_2016 == 1`. Version marker `s1_enrich_v == 4` | `scripts/enrich_tiles_s1.py`, `imint/training/pc_s1_rtc.py`, `imint/training/s1_enrichment.py` |
| The TerraMind ΔSAR jobs refuse to train unless ≥ 95 % of tiles pass the shared S1 rule | `k8s/ladder/ladder-r4-terramind-job.yaml` |
| All Prithvi-600M runs pass `--enable-multitemporal --num-temporal-frames 4` (autumn of year−1 plus three VPP-guided frames); `frame_2016` is prepended only when more frames are requested than the tile holds, so it is never read | `scripts/train_unified.py:294`, `imint/training/unified_dataset.py` `_extract_all_frames` |
| Prithvi-600M input: bands B02 B03 B04 B8A B11 B12, 504 px, aux = height, volume, basal area, diameter, DEM, five HR-VPP metrics, markfukt (ladder) | `imint/training/config.py` `enabled_aux_names` |
| Ladder checkpoints `/cephfs/checkpoints/ladder/prithvi600m_r{1..4}`; R2–R4 have 28 classes, the same schema as the served TESSERA R4 | `scripts/gen_ladder_manifests.py` |
| Validation mIoU (v2 metric), R4: Prithvi-600M 0.554, TESSERA 0.582; Prithvi-600M is second at every rung. LUCAS crop distillation, out-of-fold accuracy: 0.797 vs 0.829. The multi-year NFI holdout has been inferred but not scored (waiting for Tobias' go) | `docs/experiments/ladder_miou_v2_2026-10-13.md`, `docs/experiments/ladder_distill_stage.md`, `docs/experiments/nfi_inference_run_2026-10-27.md` |
| Shared inference (`scripts/inference_comparison.py`, `scripts/infer_tiles.py`) already runs Prithvi-600M checkpoints and builds the aux stack the way training does | `load_model`, `_build_inference_inputs` |
| The demo serves one model for all years: `TesseraR4` (28 classes, `tessera_v1`, 11 aux), years 2017–2025, no Sentinel-1 or Sentinel-2 path; the API rejects 2026 | `imint/inference/lclu.py`, `imint/api/v1/schemas.py`, `tests/api_services/test_lclu.py` |
| A cell is source-ready only with HR-VPP for its exact year; the service never substitutes another year (IM-024 rule) | `scripts/stage_lclu_campaign.py`, `tests/lclu/test_campaign.py` `never_substitutes_vpp_year` |

## Open question for Tobias

**"2025 and 2026 frames cannot be preloaded."** Two readings:
1. **On demand.** The service fetches Sentinel-2 (and the Sentinel-1 season)
   for the year when a map is requested; nothing is staged in advance.
   - This changes Phase 5 and Phase 6.
   - A DES fetch is about 3 min per tile, so a whole municipality cannot be
     made on request. A bounded window around the clicked point can.
2. **Fetch new, stage as usual.** The frames exist in no current tiles and
   must be fetched for the demo areas, but may be prepared before serving
   like the other years.

The plan below is written for reading 2 and marks where reading 1 differs.

## Decisions needed before any compute

1. **A retrain.** No checkpoint can run 2026 as is.
2. **HR-VPP for 2026.**
   - **(a) Recommended:** drop the five VPP channels (`--disable-vpp`). This
     also frees 2025, which is blocked by the same gap.
   - **(b)** Wait for HR-VPP 2026 to be published. That is likely more than a
     year away.
   - **(c)** A multi-year VPP climatology. This is still another year's data
     standing in for 2026, which the IM-024 rule forbids, so it needs an
     explicit exception.
3. **Arms.** One change per comparison, as for IM-023:

   | Arm | VPP aux | ΔSAR (2016) | Purpose |
   |---|---|---|---|
   | C0 | yes | no | The ladder R4 recipe with the FractionLoss fix. Control; optional if GPU time is short. |
   | A | no | no | Cost of dropping VPP (A − C0) |
   | B | no | yes | Value of the 2016 clearcut reference (B − A) |

   Without C0, the difference A − R4 mixes the VPP removal with the
   FractionLoss fix.
4. **Class schema.** Default 28 classes, so the existing legend serves 2026.
   Follow IM-023 to 29 classes only if TESSERA b28 is accepted. The Prithvi
   distillation labels would then need the same relabel, verified for
   `/cephfs/distill/prithvi600m_r2`.
5. **Storage location** of the 2016 composites and the 2026 inputs on
   `training-data-cephfs`.
6. **DES worker allotment** for the 2026 Sentinel-2 fetch. Ask the DES team;
   6 is a ceiling, not a default (CLAUDE.md).

## Phase 1 — Sentinel-1 2016 composite per area (independent; can start first)

The training method builds the 2016 composite per tile, with the tile's own
orbit and day-of-year window. The stored per-area composite must hold the same
numbers, or ΔSAR at serving drifts from what the model learned.

- **Method.** For every 512 px tile on the NMD grid over the area polygon, run
  the enrichment logic of `scripts/enrich_tiles_s1.py`:
  - dominant orbit;
  - window;
  - median of ≤ 3 scenes;
  - linear γ⁰.

  Write the 2016 result into the area mosaic. Factor the per-tile function
  out of the enrichment script so both paths call the same code. Do not copy
  it.
- **Window.**
  - For 2026 there is no VPP, so both years use the latitude-scaled fallback,
    which the enrichment already defines.
  - The training tiles mostly used VPP windows. Phase 4 measures the
    difference.
- **Orbit.**
  - The dominant orbit is chosen per tile across both windows (label year and
    2016), so it depends on the 2026 passes too.
  - Store both directions per area (ASCENDING and DESCENDING), so the serving
    side can pick the orbit the 2026 composite uses.
- **Storage.**
  - Per area and orbit direction, one COG with bands VV, VH (linear γ⁰,
    float32) plus a scene-count band, written with
    `imint/inference/lclu_cog.py` (ZSTD, verified reopen, atomic swap):
    `…/lclu_s1_2016/<area_id>/{ascending,descending}.tif`.
  - `MANIFEST.json` beside it with git SHA, source collection, scene IDs and
    dates, window rule and method.
- **Cost.** The windowed reads move about a megabyte per tile (enrichment
  docstring), free and anonymous. The bounding boxes hold 1 509 model windows
  (35 450 km²); 2 × float32 × 2 orbits is roughly 11 GB uncompressed.
  Measure the real size on the first area.
- **Verification before running all areas.**
  - Build Stockholm first.
  - Run the enrichment on Stockholm cohort tiles. Their `s1_vv_vh_2016` must
    equal the mosaic crop bit for bit, for tiles whose window and orbit
    match.
  - Make a quicklook of the composite.
  - Check that every tile has at least one 2016 scene in its window, and
    list where it has none.

## Phase 2 — Sentinel-2 frame dates for 2026 without VPP

The training frames were chosen inside VPP-derived windows. To keep 2026 close
to what the model saw, derive the windows from the training tiles themselves:
the `doy` of each frame slot in `unified_v2_512`, by latitude band. Use those
instead of the generic fallback in `imint/training/vpp_windows.py`.

- Store the measured windows as a versioned config file, with the tile count
  behind each window.
- Add an explicit fetch option to select them (no auto-detection when VPP is
  missing, per CLAUDE.md).
- Dates within each window: `optimal_fetch_dates(mode="era5_then_scl")`, as
  the repository requires.

## Phase 3 — training (1× H100, ≤ 12 h per arm)

The recipe is `k8s/ladder/ladder-r4-prithvi600m-job.yaml` (four frames,
30 epochs, 504 px, NFI-distilled labels `/cephfs/distill/prithvi600m_r2`,
fraction head with `/cephfs/tradslag_fracs`) with the FractionLoss fix. Each
arm changes only what is in its table row.

- One recipe script with a flag-parity test against the ladder manifest, as
  `scripts/im023_retrain.py` does for TESSERA (on branch
  `agent/te/claude/im023-barrblandskog`).
- A one-epoch smoke of every arm, then the full runs.
- **Arm B (ΔSAR).**
  - Copy the TerraMind preflight: refuse unless ≥ 95 % of tiles pass
    `s1_usable`.
  - Report how many tiles lack `s1_vv_vh_2016` and therefore train on
    zero Δ.
  - The S1 composites are already on the cohort tiles; nothing new is
    fetched for training.
- ΔSAR channels are appended last in the aux list (after markfukt), so arms A
  and B differ only in those two trailing channels.

## Phase 4 — evaluation (pre-registered before results are seen)

1. **Holdout.**
   - Per-class IoU on the held-out tiles (`ladder_holdout_eval.py`) with
     bootstrap intervals: A − C0, B − A.
   - Report hygge (22) and the forest classes separately, since ΔSAR is a
     clearcut signal.
2. **NFI.** The NFI holdout after Tobias' go, with the same rule for every
   column.
3. **LUCAS.** Crop accuracy.
4. **ΔSAR window sensitivity (arm B).** On 2024 cohort tiles, compute ΔSAR
   with the tile's VPP window and with the latitude fallback window. Report
   the change in predictions per class.
5. **Sentinel-2 window sensitivity.** On 2024 tiles, predict with VPP-window
   frames and with measured-window frames. Report agreement.
6. **Year-embedding sensitivity.** 2026 is outside the training years.
   Predict 2024 tiles with the year coordinate set to 2024 and to 2026, and
   report agreement.
7. **Bridge year.** Run the chosen arm on 2024 for all eleven areas and compare
   with TESSERA R4's 2024 maps. The per-class disagreement is how much a 2025
   (TESSERA) → 2026 (Prithvi) difference can be the model rather than the
   land. It is shown next to the 2026 map.

**Acceptance:**
- **For a 2026 map:** the chosen arm is not worse than C0 (or R4) beyond the
  bootstrap interval on the forest and crop classes.
- **For ΔSAR:** it is used only if B − A is positive on hygge or a forest
  class, with nothing worse.

## Phase 5 — 2026 inputs for the eleven areas

- **Tile grid.** 512 px tiles on the NMD grid over the area polygons. The
  bounding boxes hold 1 509 model windows; the polygon cover is smaller.
- **Sentinel-2.**
  - Autumn 2025 plus three 2026 frames, through the existing fetch path
    (`fetch_unified_tiles.py` / `tile_fetch`, DES, M1 grid snap, M2
    co-registration).
  - About 3 min per tile per DES worker
    ([`SPEC_recoreg_campaign.md`](../../SPEC_recoreg_campaign.md)), so about
    75 h for one worker and about 13 h for six, if the allotment allows it.
- **Sentinel-1 2026 (arm B).** The season composite per tile, with the same
  shared per-tile function as Phase 1, the latitude window and the orbit
  stored for 2016.
- **Storage and audit.**
  - A separate root on `training-data-cephfs`, e.g. `/cephfs/lclu_2026/`.
  - Size it from a 10-tile pilot before the full run.
  - A key audit afterwards (the `nfi2024-aux-audit` pattern): the campaign is
    not done until every tile has every key.
- **Reading 1 (on demand) instead:** no Phase 5 bulk fetch. Phase 6 fetches
  a bounded window per request, and the Sentinel-1 2016 mosaic from Phase 1
  is what makes that fast.
- **No processing units.** No CDSE processing units are used anywhere in this
  plan: DES for Sentinel-2, Planetary Computer for Sentinel-1, and no VPP.

## Phase 6 — serving (separate deployment and approval)

The current demo ends on 6 November with R4 only. 2026 belongs in a successor
deployment.

- **Model contract.** A Prithvi model contract next to `TesseraR4` in
  `imint/inference/lclu.py`:
  - backbone `prithvi_600m`, 28 classes, the arm's aux list (ΔSAR last if
    arm B);
  - bundle validation for `spectral`, `doy`, `temporal_mask`, the year and,
    for B, `s1_vv_vh` and the 2016 crop;
  - a spectral validity mask instead of `valid_embedding`.
- **ΔSAR.** Computed with the same `compute_delta_sar` as training, never a
  second implementation.
- **Manifest.** It maps each year to a model. One model per year; never two
  models for the same year without the user choosing.
- **Year limits.** Raise them to 2026:
  - `YEARS` in `lclu.py`, the API schema, `render_job.py`,
    `stage_lclu_demo.py`, `prepare_lclu.py`;
  - and the tests that pin 2025 as the last year.
- **Labelling.** The API, UI and provenance name the model per year
  ("Prithvi-EO-2.0 600M, 2026"). The bridge-year disagreement is shown with
  the 2026 map.
- **Receipts.** As for R4: canonical pixel parity against
  `scripts/inference_comparison.py`, pixel inspection, and qualification per
  cell.

## Rough order and time

1. Decisions (including the open question).
2. Phase 1 (Stockholm, then all areas) runs in parallel with Phase 2.
3. Phase 3: smoke, then three arms in parallel, about one day with queueing.
4. Phase 4: about one day, plus the NFI go.
5. Phase 5: can start after Phase 2. The duration depends on the DES
   allotment.
6. Phase 6: development and receipts, a few days.

About two weeks of calendar time if decisions and GPUs come promptly. These
are estimates, not measurements.

## Risks

- **Prithvi-600M is weaker than TESSERA.** It scores below TESSERA on every
  recorded metric. A 2026 Prithvi map next to 2017–2025 TESSERA maps invites
  false change readings. The bridge year measures this, and the UI must state
  it.
- **Dropping VPP may cost accuracy.** Arm A against C0 shows how much.
- **ΔSAR windows shift.** Training used mostly VPP windows; 2026 uses the
  latitude window. Phase 4 item 4 measures this.
- **2026 cloud cover in the north.** Kiruna may lack three clear Sentinel-2
  frames. Report it per tile; do not substitute another year.
- **Static aux layers are older than 2026.** Harvests since they were made are
  not in them; ΔSAR is the channel that sees them. This is the same policy as
  the current demo.

## Not in scope

An optical 2016 composite: `frame_2016` is not a model input. National
composites, TESSERA 2026 (not published), the IM-023 class schema decision,
and the sumpskog question.
