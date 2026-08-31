# Plan — Independent (off-footprint) validation fetch: NFI + LUCAS

**Status:** Phase A LAUNCHED (2026-08-18) · **Owner:** race/publication track
**Supersedes:** the "patch the deck" task — the deck is finalized on the clean
powered numbers this produces, not the contaminated LUCAS / thin-209 ones.

### Execution log
- **2026-08-18** — Phase A manifest generated: `data/nfi/holdout_val_phaseA.json`
  (1,650 tiles, 11,280 plots+pts) → 80 MGRS tiles. Staged to PVC
  `/data/audits/holdout_val_phaseA.json`.
  - **`prefetch-vpp-holdout-phasea`** (PU-free WEkEO VPP) — **COMPLETE** (~2,840
    COGs, READBACK OK 5/5 bands).
  - **`fetch-holdout-val-phasea`** (DES spectral+aux → `/data/holdout_val_512`)
    — **LAUNCHED** (clones agent branch; --workers 6, read live `DES: permits=N`).
  - **`enrich-holdout-s1`** + **`enrich-holdout-tessera`** — staged, submit after
    the fetch lands tiles (independent, may run in parallel).

## 1. Objective

Build a large, genuinely held-out validation set by **inference only** on field
plots *outside the training-tile footprint*. No retraining — reuse every trained
race model. This fixes the LUCAS contamination (deck LUCAS numbers are the
all-split, 99.3 % training-tile points) and turns the underpowered 209-plot NFI
test (±7 pp CIs — every top-4 race comparison currently INCONCLUSIVE) into a
powered independent benchmark that can resolve the equivalence/difference
claims and pre-empt the "thin test set" reviewer objection.

Rationale in full: `docs/data/deck_number_reconciliation.md`.

## 2. Why not retrain

The training set is fine; only *validation* reused training tiles. Retraining
would discard the trained models, cost GPU-days, and still need a held-out
fetch. Inference on off-footprint plots gives the same clean validation for a
bounded fetch cost and zero training.

## 3. Scope — maximal coverage

Clean pool = field plots never on any of the 7,882 training tiles, year-matched
to the S2 era. Truth is independent of NMD (NFI forest-type from species
volumes; LUCAS from its own survey).

| Source | Off-footprint plots | Unique (tile,year) |
|---|---|---|
| NFI (2018–2024, all land-use; 13,459 forest) | 15,425 | 3,873 |
| LUCAS (SE 2022 survey) | 27,692 | 13,017 |
| **Combined (388 shared units deduped)** | **43,117** | **16,502** |

Manifest artifact: `data/nfi/offfootprint_val_manifest.parquet` (plot list +
tile-grid key). Optional extension: LUCAS 2018 survey (adds more units).

## 4. Phasing — density-first (front-load statistical power)

Fetch densest tiles first so a powered validation lands early, then continue to
full coverage.

| Phase | Tiles | Plots+points covered | ETA @6w (~84 t/h) |
|---|---|---|---|
| **A** — top 10 % | 1,650 | 11,280 (26 %) | ~1 day |
| **B** — to 25 % | 4,125 | 21,580 (50 %) | ~2 days |
| **C** — full | 16,502 | 43,117 (100 %) | ~8 days |

Phase A alone (~11 k plots) is >50× the current clean NFI set — enough to
resolve the top-4 race and give an honest LUCAS number. B/C add coverage and
tighten rare-class CIs (spruce, mixed, non-forest).

Full-fetch ETA scales with DES allotment: ~8 d @6 workers, ~25 d @2, ~49 d @1.

## 4b. Full tile-prep pipeline (4 stages — reuse repo scripts, no new code)

Each model needs different inputs, so the fetched tile must be enriched to the
full training-equivalent stack before inference. All stages run on
`/data/holdout_val_512`, in-place, `--skip-existing`, reusing the same scripts
the training campaign used:

1. **VPP prefetch** (`prefetch_vpp_wekeo.py`) → `/data/vpp_wekeo` — PU-free WEkEO
   cache for the aux VPP bands. **[RUNNING]**
2. **Spectral + aux fetch** (`fetch_unified_tiles.py`, `--fetch-sources des`,
   `VPP_SOURCE=wekeo`) → 24-ch DES spectral + 11 aux (VPP/DEM/forestry/markfukt,
   via `fetch_aux_channels`) → `/data/holdout_val_512`. **[STAGED]** Includes an
   aux-completeness audit (all 11 channels present, markfukt NaN-gaps allowed).
3. **S1 enrichment** (`enrich_tiles_s1.py --s1-backend pc-rtc`) → adds ΔVV/ΔVH
   γ⁰ SAR-change channels (`s1_enrich_v==3`). **REQUIRED for CROMA/TerraMind v3**
   (commit 0091835). PU-free (PC-RTC windowed COG reads). **[STAGED]**
4. **Tessera embeddings** (`enrich_tiles_tessera.py`) → adds 128-d per-pixel
   TESSERA embeddings via geotessera (global 10 m; new locations download
   fresh, LRU-guarded cache on `/tessera_cache`). **REQUIRED for the Tessera
   model.** **[STAGED]**

Stages 3 & 4 are independent adds — run after stage 2, either order/parallel.
Model → required inputs: Prithvi-600M/300M, Clay = 35-ch (stages 1-2);
CROMA/TerraMind = 35-ch + S1 (stages 1-3); Tessera = TESSERA embeddings
(stage 4). Spectral fetch reuses `imint.fetch`/`tile_fetch` (the canonical DES
path) — never custom openEO.

## 5. Fetch design — reuse the repo pipeline (no new fetch code)

- **Driver:** `scripts/fetch_unified_tiles.py` over the off-footprint tile
  manifest, `--mode refetch`-style year-locked reads. New staging dir
  `/data/holdout_val_512` — **never** touch `unified_v2_512` training tiles.
- **Spectral:** DES bulk (free) via `fetch_des_data` / `fetch_seasonal_image`;
  `optimal_fetch_dates(mode="era5_then_scl")` for clean scenes. `--workers` =
  **current DES allotment** (ASK/verify `DES: permits=N`; ≤6 per 2026-07-21;
  count the SCL-screen stream).
- **VPP:** `VPP_SOURCE=wekeo` + `VPP_WEKEO_DIR` + `envFrom wekeo-creds`
  (MANDATORY — cache-miss must fail loud, never drain CDSE PU → silent exit-0).
  Pre-fill WEkEO cache gaps with `prefetch_vpp_wekeo.py` (PU-free) first.
- **Aux (11 ch):** forestry (`skg_height` pattern), DEM, VPP phenology,
  markfukt — DEM/forestry static → cached once; VPP per-year.
- **Year-matching (non-negotiable):** spectral year = plot inventory year;
  autumn frame from year-1. Forest/non-forest plots may year-fallback; any
  LPIS/crop-adjacent plot may NOT (crops rotate) — but this set is
  forest/land-cover truth, so fallback is generally safe for NFI forest plots.
- **CDSE PU:** only as a parallel wall-time boost if DES is the blocker, with PU
  balance verified and scope measured (per `feedback_cdse_pu_budget_priorities`).
  Never as primary.
- **Resumable + idempotent:** `_valid_existing_tile` skip; `postprocess_qc`
  drops empty-frame tiles — count tiles before/after each step.

## 6. Truth labeling (reuse existing derivations)

- **NFI forest-type:** dominant species volume → {1 pine (VolPine+VolContorta),
  2 spruce (VolSpruce), 3 deciduous (VolBirch+VolOtherDec), 4 mixed (no ≥
  dominant share)}; non-forest from `LandUseClass`. Use the SAME rule that
  produced `nfi_forest` for the 944 (locate & reuse — do not re-derive ad hoc).
- **LUCAS:** unified_class via the existing LUCAS→unified mapping in the
  `validate_against_lucas` prep path.

## 7. Inference + scoring

- Run every race model (Tessera frac/gated, Prithvi-600M/300M, Clay,
  CROMA/TerraMind once trained) on the fetched tiles via
  `validate_against_nfi.py --dump-per-plot` / `validate_against_lucas.py`.
  Reuse the cached fast-path (bit-parity gate PASSED 08-17).
- Score with the **canonical** path (`model_race_standings.py`) + the rigor
  harness (`race_rigor_stats.py` — McNemar/TOST/Holm + spatial-block CV) on the
  powered set. Regenerate `model_race_standings.json` and finalize the deck.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| DES allotment dynamic → throttle/[408] | ASK allotment; `--workers`=allotment ≤6; watch `permits=N` + 429 |
| CDSE PU drain (silent exit-0) | `VPP_SOURCE=wekeo`; PU only as verified boost |
| Year mismatch inflates/deflates truth | strict year-lock; year-fallback only for pure forest/water |
| Mutating training tiles | separate `/data/holdout_val_512` staging dir |
| Empty-frame tiles silently deleted | count before/after `postprocess_qc` |
| Coreg drift on new tiles | M1 grid-snap + M2 inter-frame per repo default |

## 8b. Future work — Lantmäteriet 1 m DEM (NOT for this run)

Candidate aux improvement for the **next training generation** — do NOT swap it
into this validation (would break train-test DEM consistency; the models were
trained on Copernicus GLO-30).

Current DEM aux = Copernicus GLO-30: 30 m native (TanDEM-X, canopy-contaminated
over forest), bilinear-resampled to 10 m, single raw-elevation channel
(`imint/training/copernicus_dem.py`).

Lantmäteriet *Nationella höjdmodellen* (`Markhöjdmodell`, 1 m LiDAR, open data):
~0.1 m vertical accuracy vs GLO-30's ~2–4 m, and a true **bare-earth DTM**.

Assessment:
- A 1:1 raw-elevation swap would gain little — forest type zones by elevation at
  10s–100s m scale, where cm accuracy is irrelevant.
- The real leverage is **derived terrain channels** (slope, aspect, TWI/TPI):
  GLO-30 at 30 m is too coarse for trustworthy 10 m derivatives; a 1 m→10 m DTM
  makes them reliable. Aspect→soil-moisture→spruce-vs-pine is a genuine
  forest-type signal the model currently cannot see.
- Bonus: canopy height is already a separate SKG aux channel, so a bare-earth
  DTM avoids the canopy double-count baked into GLO-30 over forest.
- Cost: open data, tileable; a fetcher follows the `skg_height` aux pattern.

Decision if pursued: pair the 1 m DTM with slope/aspect/TWI channels in a
retrain — not a raw-elevation swap. (Recorded 2026-08-18 per user.)

## 9. Success criteria

- ≥2,000 clean NFI plots scored → top-4 race CIs ±1–2 pp (resolves INCONCLUSIVE).
- ≥5,000 clean LUCAS points scored → honest independent LUCAS ordering.
- All numbers via canonical scoring + rigor harness; deck finalized on them.

## 10. First executable step (no network)

Generate the manifest (`offfootprint_val_manifest.parquet`) + the phased tile
lists; then check DES allotment and submit Phase A. Nothing hits DES until the
allotment is confirmed and the user approves.
