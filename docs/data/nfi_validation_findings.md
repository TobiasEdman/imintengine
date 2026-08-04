# NFI validation — findings (2026-06-11)

Short status note for the `feat/nfi-validation` work-stream. Companion to the
[data card](nfi_plotdata_DATA_CARD.md).

## Coverage gate (the feasibility number)

Ran `scripts/nfi_tile_coverage.py` on the full **`/data/unified_v2_512`** tile
set on the ICE PVC (`k8s/nfi-coverage-gate-job.yaml`, HEAD `47e75d2`):

| | |
|---|---|
| NFI plots (≥2018) | 18,661 |
| tiles scanned | 10,835 (5 corrupt, skipped) |
| **plots co-located** | **982** on **270** tiles (max 10/tile, mean 3.6) |
| by year | 2018: 479 · 2021: 57 · 2022: 446 |
| by GPS tier | GPS (≤2023) 982 · RTK (≥2024) **0** |

Index persisted to `/data/nfi/nfi_index_unified_v2_512.parquet` (982 rows).

## Recommendation: validate, don't (yet) train

- **982 plots is a usable independent validation set** for forest-type accuracy.
- It is **too sparse to train on**: 982 point-targets over 270 tiles is ~3.6
  supervised pixels per 512² tile (~1.4e-5 of the pixels), and **all 982 are
  GPS-tier** (pre-2024, metre-level position — a real fraction of a 10 m pixel).
  A masked sparse-supervision head on this signal is not worth the model
  surgery. **Defer Track T** (the regression / maturity heads) unless coverage
  grows materially (more tile-years, or 2024–25 RTK plots).

## Validation harness — built, unit-tested, blocked on a checkpoint

`scripts/validate_against_nfi.py` (forest-type accuracy + confusion + per-class
AUROC at plot pixels) and `imint/eval/metrics.py:auroc_aupr` are implemented and
unit-tested (24 tests green). The **real run is blocked on model availability**,
not the harness:

- The available checkpoints (`unified_v6a`, `unified_v5c`) are **11-aux** — the
  retired generation that included the leaky `harvest_probability` channel.
- The current `unified_v2_512` tiles + `AUX_CHANNEL_NAMES` are **10-aux**
  (height, volume, basal_area, diameter, dem, vpp×5 — no `harvest_probability`).
- 11 ≠ 10 → channel mismatch. The 11th channel was `harvest_probability`,
  removed in `3447518` ("drop synthetic harvest_probability aux channel") for
  leaking the harvest target. It's gone from the tiles and `AUX_CHANNEL_NAMES`,
  and synthesizing it back would corrupt the validation. There is no
  aux-compatible trained checkpoint to validate against.

**Can't we reuse existing model outputs?** No — the only saved predictions are
`data/viz_tiles/predictions/v5*/v6*_predictions.json`, which cover **5 showcase
tiles** (the `inference_comparison.py` model-comparison set), none of which are
among the 270 NFI-co-located tiles. And re-running v5/v6 hits the
`harvest_probability` wall above.

**To run validation:** point the harness at a checkpoint trained on the current
10-aux dataset (e.g. the next `unified_v2_512` training run), stage it to the
PVC, and submit a validation job mirroring `k8s/nfi-coverage-gate-job.yaml`.
`make_model_predict_fn` will also need adapting from `LULCDataset` to
`UnifiedDataset` (the 512 tiles are `spectral`/`multitemporal` format).

## Result — v8 (2026-08-04)

Unblocked: `unified_v8_full7882` is the first 10-aux checkpoint (val mIoU
0.5148) on the merged 7 882-tile set. Harness rewired to the unified path
(`make_model_predict_fn` → `inference_comparison.{load_model, run_inference}`,
504-crop coord remap), run via `k8s/nfi-validate-v8-job.yaml`. Full numbers in
[`nfi-validation-v8.json`](nfi-validation-v8.json).

| | |
|---|---|
| plots scored (in-crop) | **944** (of 973 in the index; 29 dropped in the 4 px crop border, tiles removed since the June index also excluded) |
| plots with a derivable forest type | 884 |
| **forest-type accuracy (4-way, argmax)** | **41.6 %** |
| per-class AUROC | tallskog 0.80 · granskog 0.80 · lövskog 0.73 · blandskog 0.70 |
| per-class AUPR | tallskog 0.73 · granskog 0.62 · lövskog 0.53 · blandskog 0.28 |

**Reading it.** AUROC 0.70–0.80 says the softmax *ranks* the true forest type
well — the model has real discriminative signal against independent field truth.
The modest 41.6 % exact-argmax accuracy is dragged down by two structural,
largely-expected effects, not a broken model:

1. **Blandskog is definitionally fuzzy** (AUPR 0.28): "mixed" plots scatter
   across tall/gran/löv — 12+19+6 of 105 blandskog plots go to a pure conifer/
   deciduous class, which a strict 4-way match penalises but isn't wrong in kind.
2. **Forest → non-forest-forest confusion**: a large share of true-forest plots
   predict as *sumpskog* (class 5), *tillfälligt ej skog* (6) or *hygge* (22) —
   e.g. 60/151-region of tallskog → sumpskog. Some are real (an NFI plot in a
   wet or recently-thinned stand), some are the model over-reaching on those
   classes. The harness derives only the 4 pure forest types from NFI species
   volume (sumpskog is a *site* condition, not derivable — see
   `derive_nfi_forest_class`), so these land as misses by construction.

**Not** a head-to-head vs the 256 model — that comparison
(`inference-compare-v8-vs-256`) is tracked separately.

## Artifacts

- Loader + co-location: `imint/training/{slu_nfi,nfi_colocate}.py`
- Coverage gate: `scripts/nfi_tile_coverage.py` + `k8s/nfi-coverage-gate-job.yaml`
- Validation harness: `scripts/validate_against_nfi.py` + `imint/eval/metrics.py`
- PVC index: `/data/nfi/nfi_index_unified_v2_512.parquet`
