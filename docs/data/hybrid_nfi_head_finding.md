# Finding — hybrid NFI-head BEATS NMD2023 on field truth (+0.14)

Experiment (2026-08-09): extract the 256-dim pre-classifier features of the
saturated NMD2023 seg model (v8b_nmd2023_long) at the 944 NFI plots, then
k-fold cross-validate a small head trained DIRECTLY on the NFI field truth.
Every plot scored out-of-fold (trained on the other folds — leakage-free).

## Result (944 plots, same accuracy_suite as all seg-model validations)

| Method | overall | kappa |
|---|---|---|
| **MLP head (OOF, stratified)** | **0.6367** | **0.4996** |
| MLP head (OOF, grouped by tile) | 0.6324 | 0.4911 |
| logreg head (OOF, stratified) | 0.5763 | 0.4293 |
| NMD2023 v2.1 (baseline) | 0.493 | 0.366 |
| v8b (baseline) | 0.465 | 0.339 |

MLP per-class F1: tallskog 0.76, granskog 0.62, lövskog 0.58, blandskog 0.30,
icke-skog 0.68 — every class far above the seg models except blandskog (~equal).
The grouped-by-tile CV (StratifiedGroupKFold — no same-tile plots across
train/test) confirms the result is not spatial leakage: 0.632 (+0.139).

## Interpretation

The information to beat NMD2023 was IN the model's features all along — what
was wrong was the SUPERVISION TARGET. Every dense training run (NMD2018,
NMD2023 hard labels) optimized agreement with a label proxy and hit its
ceiling; a 33k-parameter head trained on 755 field plots beats them all by
re-aiming the same representation at the real objective. This also retroactively
explains the two negative results: the wetness aux and the NMD2023 labels both
enriched the FEATURES/proxy side, but the binding constraint was the target.

## Caveats
- n=944 (support: tall 372, gran 247, löv 133, bland 132, icke-skog 60) —
  per-class numbers carry real variance; blandskog remains hard (F1 0.30).
- The head is point-wise: this validates plot-level forest-type accuracy, not a
  full dense map. Productionizing = distill the NFI-head into the seg model
  (e.g. finetune the seg head on NFI-pseudo-labels + dense NMD2023) or attach
  the calibrated head as the forest-type decision layer.
- Features: /data/nfi_eval/nfi_plot_features_nmd2023.parquet (256-d @ 944 plots);
  CV: docs/data/nfi-head-cv-nmd2023.json; scripts: extract_plot_features.py,
  nfi_head_cv.py.
