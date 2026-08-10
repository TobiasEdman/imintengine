# Finding — Trädslag fraction supervision beats NMD2023 (dense model, +0.08…+0.15)

Experiment (2026-08-10): multi-task fraction head (4× continuous crown-cover,
supervised by NMD2023's Trädslag tilläggsskikt) on the saturated NMD2023 seg
model; at validation the predicted fractions are collapsed with the NFI
dominance rule instead of NMD's crown-cover thresholds (collapse-rule
alignment). Training: 15 ep warm-start, hard mIoU flat (0.4858) — the gain is
entirely in the fraction path.

## Results (NFI field truth)

| Eval | overall | kappa | vs NMD2023 same plots |
|---|---|---|---|
| Fraction collapse, uncalibrated (944) | 0.520 | 0.350 | +0.027 |
| **Kappa-calibrated*, held-out 209 plots** | **0.579** | **0.420** | **+0.148** (NMD2023: 0.431) |
| Kappa-calibrated*, all 944 | 0.570 | 0.399 | +0.077 (NMD2023: 0.493) |

*floor=0.05, dominant_frac=0.6, tuned ONLY on the 735 train-split plots
(distill_split.json), evaluated on the 209 held-out test-tile plots.

Per-class F1 (held-out): tallskog 0.74, lövskog 0.59, granskog 0.55,
blandskog 0.29, icke-skog 0.00.

## Interpretation

Dense fraction supervision + NFI-rule collapse is the first DENSE model to beat
NMD2023 on field truth. The pre-threshold crown-cover signal carries more
information than the hard classes, and aligning the collapse rule with how the
field truth is defined (volume-fraction dominance) removes the rule mismatch
that capped the hard-label models. Confirms the hybrid-head diagnosis
(supervision target > representation) with a fully dense, deployable model.

## Caveats
- **Non-forest is degenerate in the fraction path** (F1 0.00): predicted
  fractions never sum below the floor, so the fraction head cannot say
  "non-forest". Production composition: hard head (or NMD) decides the forest
  MASK — which SLU rates "mycket bra" — and the fraction head decides forest
  TYPE within it. The 944-plot suite numbers above under-state that composed
  system (all 60 non-forest plots count as errors).
- Accuracy-objective calibration collapses to 3 classes (bland also dropped,
  same overall 0.579) — kappa-objective calibration is the reported, defensible
  variant.
- blandskog remains the hard class everywhere (F1 0.29).
- Checkpoint: /cephfs/checkpoints/v8b_nmd2023_tradslag/best_model.pt; per-plot
  fractions: /data/nfi_eval/tradslag_per_plot.parquet; suite JSON:
  docs/data/nfi-validation-tradslag.json.
