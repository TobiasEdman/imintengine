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
- 11 ≠ 10 → channel mismatch. There is no aux-compatible trained checkpoint to
  validate against.

**To run validation:** point the harness at a checkpoint trained on the current
10-aux dataset (e.g. the next `unified_v2_512` training run), stage it to the
PVC, and submit a validation job mirroring `k8s/nfi-coverage-gate-job.yaml`.
`make_model_predict_fn` will also need adapting from `LULCDataset` to
`UnifiedDataset` (the 512 tiles are `spectral`/`multitemporal` format).

## Artifacts

- Loader + co-location: `imint/training/{slu_nfi,nfi_colocate}.py`
- Coverage gate: `scripts/nfi_tile_coverage.py` + `k8s/nfi-coverage-gate-job.yaml`
- Validation harness: `scripts/validate_against_nfi.py` + `imint/eval/metrics.py`
- PVC index: `/data/nfi/nfi_index_unified_v2_512.parquet`
