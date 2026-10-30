# 2026 land-cover maps with Prithvi-EO-2.0 600M and a 2016 composite per municipality

Status: proposal, 2026-10-30. Nothing has been run. Scope: the IM-024 demo
areas — the ten municipalities plus Kiruna tätort in
[`config/lclu/municipalities.json`](../../config/lclu/municipalities.json).
Each cluster step needs Tobias' approval and `kubectl create --dry-run=server`
first. Decisions that are Tobias' to make are marked **Decision**.

## Summary

2026 cannot be served by the current demo model. TESSERA R4 needs TESSERA
embeddings and HR-VPP, and neither exists for 2026. Prithvi-EO-2.0 600M reads
Sentinel-2 directly, so it can cover 2026.

**No existing Prithvi-600M checkpoint can produce 2026 maps without a
retrain:**
- **Frames.** Every Prithvi-600M checkpoint reads four frames and never the
  2016 frame, so a 2016 composite does nothing for them.
- **VPP.** Every Prithvi-600M checkpoint needs five HR-VPP channels for the
  map year. HR-VPP 2026 is not published (2025 is not either), and the service
  never substitutes another year.

The plan therefore has six parts:
1. Build the 2016 growing-season composite once per area.
2. Train a Prithvi-600M without VPP channels, with and without the 2016
   frame as frame 0.
3. Pick the frame dates for 2026 without VPP.
4. Fetch 2026 for the eleven areas.
5. Add a second, clearly labelled model to the service.
6. Measure, before any result is shown, how much of the 2025→2026 difference
   is the change of model rather than of land cover.

## What the repository says today

| Fact | Where |
|---|---|
| All Prithvi-600M runs pass `--enable-multitemporal --num-temporal-frames 4`; the CLI default is 4 too, and no job in `k8s/` trains five frames | `k8s/ladder/ladder-r{1..4}-prithvi600m-job.yaml`, `k8s/train-v8b-nmd2023-long-job.yaml`, `scripts/train_unified.py:294` |
| `frame_2016` is prepended only when more frames are requested than the tile holds (`use_bg = num_frames > tile_frames`), so with 4 it is never read | `imint/training/unified_dataset.py` `_extract_all_frames` |
| Prithvi-600M input: autumn of year−1 plus three VPP-guided growing-season frames, bands B02 B03 B04 B8A B11 B12, 504 px, aux = height, volume, basal area, diameter, DEM, five HR-VPP metrics, markfukt (ladder) | `imint/training/config.py`, `scripts/train_unified.py` |
| Ladder checkpoints `/cephfs/checkpoints/ladder/prithvi600m_r{1..4}`; R2–R4 have 28 classes, the same schema as the served TESSERA R4 | `scripts/gen_ladder_manifests.py` |
| Validation mIoU (v2 metric), R4: Prithvi-600M 0.554, TESSERA 0.582; Prithvi-600M is second at every rung. LUCAS crop distillation, out-of-fold accuracy: 0.797 vs 0.829. The multi-year NFI holdout has been inferred but not scored (waiting for Tobias' go) | `docs/experiments/ladder_miou_v2_2026-10-13.md`, `docs/experiments/ladder_distill_stage.md`, `docs/experiments/nfi_inference_run_2026-10-27.md` |
| Shared inference (`scripts/inference_comparison.py`, `scripts/infer_tiles.py`) already runs Prithvi-600M checkpoints | `load_model`, `_build_inference_inputs` |
| The demo serves one model for all years: `TesseraR4` (28 classes, `tessera_v1`, 11 aux), years 2017–2025, no Sentinel-2 path; the API rejects 2026 | `imint/inference/lclu.py`, `imint/api/v1/schemas.py`, `tests/api_services/test_lclu.py` |
| A cell is source-ready only with HR-VPP for its exact year; the service never substitutes another year (IM-024 rule) | `scripts/stage_lclu_campaign.py`, `tests/lclu/test_campaign.py` `never_substitutes_vpp_year` |
| `frame_2016` today is one whole L1C scene per tile through sen2cor (2015 fallback), gated on tile-level cloud optical thickness, with no pixel cloud mask and no co-registration in the 512 job. No multi-scene Sentinel-2 composite exists in the repository | `k8s/sen2cor-frame2016-512-job.yaml`, `scripts/sen2cor_pipeline/` |
| 2016 L2A is not on DES; the 2016 frame comes from GCS L1C through sen2cor, never from CDSE SH-Process | CLAUDE.md |
| Without VPP, frame windows fall back to fixed day-of-year windows | `imint/training/vpp_windows.py` `_FALLBACK_BY_NUM_FRAMES` |

## Decisions needed before any compute

1. **A retrain.** No checkpoint can run 2026 as is. Proposed: train new
   Prithvi-600M variants (Phase 3).
2. **HR-VPP for 2026.**
   - **(a) Recommended:** drop the five VPP channels (`--disable-vpp`). This
     also frees 2025, which is blocked by the same gap.
   - **(b)** Wait for HR-VPP 2026 to be published. That is likely more than a
     year away.
   - **(c)** A multi-year VPP climatology, explicitly labelled. This is still
     another year's data standing in for 2026, which the IM-024 rule forbids,
     so it needs an explicit exception.
3. **Arms.** One change per comparison, as for IM-023:

   | Arm | Frames | VPP aux | Purpose |
   |---|---|---|---|
   | C0 | 4 | yes | The ladder R4 recipe with the FractionLoss fix. Control; optional if GPU time is short. |
   | A | 4 | no | Cost of dropping VPP (A − C0) |
   | B | 5, frame 0 = 2016 | no | Value of the 2016 frame (B − A) |

   Without C0, the difference A − R4 mixes the VPP removal with the
   FractionLoss fix.
4. **Class schema.** Default 28 classes, so the existing legend serves 2026.
   Follow IM-023 to 29 classes only if TESSERA b28 is accepted. The Prithvi
   distillation labels would then need the same relabel, verified for
   `/cephfs/distill/prithvi600m_r2`.
5. **2016 window.** The repository uses two:
   - 06-01..08-16 (CLAUDE.md and the inline fetch);
   - 06-01..08-31 (`select_scenes.py`).

   Proposed: 06-01..08-31 for the composite, with the median day of year
   recorded per pixel.
6. **Storage location** of the composites and the 2026 tiles on
   `training-data-cephfs` (Phase 1, Phase 5).
7. **DES worker allotment** for the 2026 fetch. Ask the DES team; 6 is a
   ceiling, not a default (CLAUDE.md).

## Phase 1 — 2016 growing-season composite per area (independent; can start first)

One composite per area, shared by every map year, like the static aux layers
(`temporal_policy: shared across all years` in `stage_lclu_campaign.py`).

- **Source.** Sentinel-2 L1C from GCS through the pinned sen2cor image
  (`ghcr.io/tobiasedman/imint-sen2cor@sha256:528adef1…`). No CDSE
  processing units.
- **Scene selection.**
  - Reuse `scripts/sen2cor_pipeline/select_scenes.py` (ERA5 prefilter,
    existence check). Do not keep only one scene per tile: keep every scene
    in the window that covers the area.
  - Keep the tile-level cloud-thickness gate (`--cot-max 8.0`) as a first
    filter.
- **Pixel mask.** Use the SCL layer that sen2cor already writes.
  - Clear pixels: SCL 4 vegetation, 5 bare soil, 6 water.
  - Drop cloud, shadow, cirrus and snow.
- **Grid.** Reproject every scene properly to EPSG:3006 on the NMD 10 m
  lattice (`imint/training/nmd_grid.py`). This replaces the bilinear UTM read
  of the 512 job: Göteborg straddles the 32/33 zone edge at 12° E, and Kiruna
  is in zone 34.
- **Composite.**
  - Per-pixel median of clear observations per band, 6 Prithvi bands.
  - Plus two bands: the clear-observation count and the median day of year.
- **Storage.**
  - One COG per area, written with the existing COG writer
    (`imint/inference/lclu_cog.py`: ZSTD, 256 blocks, verified reopen, atomic
    swap): `…/lclu_2016/<area_id>/composite_2016.tif`.
  - `MANIFEST.json` beside it with image digest, git SHA, the L1C product IDs,
    sen2cor version, window and method (repo checklist item 7).
  - Estimated size: 35 450 km² of bounding boxes, 8 bands × uint16, about
    5.7 GB uncompressed. Measure the real size on the first area.
- **Code.** A new `scripts/build_2016_composite.py` plus a k8s job modelled on
  `k8s/sen2cor-frame2016-512-job.yaml`, pinned to a commit and image digest.
  Fill the sen2cor image's `MANIFEST.json` `image_digest`, today
  `TBD-after-first-build`, before the first run.
- **Verification before running all areas** (CLAUDE.md: one change at a time,
  check visually):
  - Build Stockholm first.
  - Check its clear-count map: no holes, and say where the count is below 3.
  - Make an RGB quicklook.
  - Run a centre-of-mass position test against a co-registered 2024 frame
    for the same window, as
    `tests/test_coregistration.py::test_coregister_to_reference_removes_shift_dot_com`
    does.
  - Then run the other ten areas.
- **Estimate.** Roughly 1–4 MGRS granules per area and 5–10 acquisitions each
  in the window, i.e. on the order of 100–200 sen2cor runs. The holdout run
  processed 29 scenes in 56 min on one 2080 Ti, so this is hours, not days.
  Measure on Stockholm before scheduling the rest.

## Phase 2 — frame dates for 2026 without VPP

The training frames were chosen inside VPP-derived windows. To keep 2026 close
to what the model saw, derive the windows from the training tiles themselves:
the `doy` of each frame slot in `unified_v2_512`, by latitude band. Use those
instead of the generic fallback in `vpp_windows.py`.

- Store the measured windows as a versioned config file, with the tile count
  behind each window.
- Add an explicit fetch option to select them (no auto-detection when VPP is
  missing, per CLAUDE.md).
- Dates within each window: `optimal_fetch_dates(mode="era5_then_scl")`,
  as the repository requires.
- Arms A and B are trained on frames from VPP windows. Phase 4 checks whether
  measured windows give the same predictions on tiles where both are possible.

## Phase 3 — training (1× H100, ≤ 12 h per arm)

The recipe is `k8s/ladder/ladder-r4-prithvi600m-job.yaml` (30 epochs, 504 px,
NFI-distilled labels `/cephfs/distill/prithvi600m_r2`, fraction head with
`/cephfs/tradslag_fracs`) with the FractionLoss fix. Each arm changes only what is in
its table row.

- One recipe script with a flag-parity test against the ladder manifest, as
  `scripts/im023_retrain.py` does for TESSERA.
- A one-epoch smoke of every arm, then the full runs.
- **Arm B needs `frame_2016` on its training tiles.** CLAUDE.md requires the
  frame for training tiles.
  - Audit `has_frame_2016` over the cohort first.
  - Fill the gaps with `k8s/sen2cor-frame2016-512-job.yaml`.
- **Training/serving difference in arm B.**
  - Training frame 0 is a single scene per tile, with no pixel mask and no
    co-registration.
  - Serving frame 0 is the composite crop with M2 co-registration.
  - Phase 4 measures the effect. Rebuilding training frame 0 from composites
    would need a national composite, which is out of scope unless the measured
    effect demands it.

## Phase 4 — evaluation (pre-registered before results are seen)

1. **Holdout.**
   - Per-class IoU on the held-out tiles (`ladder_holdout_eval.py`) with
     bootstrap intervals: A − C0, B − A.
   - This needs years with HR-VPP, so it uses 2018–2024 tiles.
2. **NFI.** The NFI holdout after Tobias' go, with the same rule for every
   column.
3. **LUCAS.** Crop accuracy.
4. **Frame-0 sensitivity (arm B).** On cohort tiles inside the eleven areas,
   predict once with the tile's single-scene `frame_2016` and once with the
   composite crop. Report per-class agreement.
5. **Window sensitivity.** On 2024 tiles, predict with VPP-window frames and
   with measured-window frames. Report agreement.
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
- **For the 2016 frame:** it is used only if B − A is positive on at least one
  of those, with nothing worse.

## Phase 5 — fetch 2026 for the eleven areas

- **Tile grid.** 512 px tiles on the NMD grid over the area polygons. The
  bounding boxes hold 1 509 model windows; the polygon cover is smaller.
- **Frames.**
  - Autumn 2025 plus three 2026 frames, through the existing fetch path
    (`fetch_unified_tiles.py` / `tile_fetch`, DES, M1 grid snap, M2
    co-registration).
  - Spectral only; no labels.
  - Arm B: frame 0 cropped from the composite, plus an M2 shift to the
    tile's anchor.
- **Throughput.** About 3 min per tile per DES worker
  ([`SPEC_recoreg_campaign.md`](../../SPEC_recoreg_campaign.md)),
  so about 75 h for one worker and about 13 h for six, if the allotment
  allows it.
- **Storage.**
  - A separate root on `training-data-cephfs`, e.g. `/cephfs/lclu_2026/`.
  - Size it from a 10-tile pilot before the full run.
  - A key audit after the fetch (the `nfi2024-aux-audit` pattern): the
    campaign is not done until every tile has every key.
- **No processing units.** No CDSE processing units are used anywhere in this
  plan: DES for 2026, GCS + sen2cor for 2016, and no VPP.

## Phase 6 — serving (separate deployment and approval)

The current demo ends on 6 November with R4 only. 2026 belongs in a successor
deployment.

- **Model contract.** A Prithvi model contract next to `TesseraR4` in
  `imint/inference/lclu.py`:
  - backbone `prithvi_600m`, 28 classes, the arm's aux list;
  - bundle validation for `spectral`, `doy`, `temporal_mask` and the year,
    with a spectral validity mask instead of `valid_embedding`.
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

1. Decisions (Phase 0).
2. Phase 1 (Stockholm, then all areas) runs in parallel with Phase 2 and the
   arm-B cohort audit.
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
- **Composite and single scene differ.** The composite is cleaner than the
  single-scene frame 0 that arm B trains on. Phase 4 item 4 measures this.
- **2026 cloud cover in the north.** Kiruna may lack three clear frames.
  Report it per tile; do not substitute another year.
- **Static aux layers are older than 2026.** Harvests since they were made are
  not in them. This is the same policy as the current demo.

## Not in scope

National 2016 composites, TESSERA 2026 (not published), the IM-023 class
schema decision, and the sumpskog question.
