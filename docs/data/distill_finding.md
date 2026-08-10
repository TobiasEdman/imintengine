# Finding — distilled NFI-head model beats NMD2023, honestly held out (+0.07)

Experiment (2026-08-10): the NFI-calibrated head (trained on the 735 train-split
plots only) relabelled the forest pixels of all 7,882 tiles (37.3% of forest px
changed; löv 14.3→7.0%, correcting NMD2023's löv over-prediction and the legacy
113→löv mapping). The seg model was finetuned 15 ep on these distilled labels
(warm-start v8b_nmd2023_long; best val 0.5014 vs distilled target).

## Result — NFI field truth on the 209 HELD-OUT test-tile plots
(no stage — head training, distillation, finetune — ever saw these plots)

| Model | overall | kappa |
|---|---|---|
| Trädslag fraction-collapse (calibrated) | 0.579 | 0.420 |
| **Distilled dense model** | **0.502** | **0.371** |
| NMD2023 v2.1 (same plots) | 0.431 | 0.298 |

Distilled per-class F1 (held-out): gran 0.63, tall 0.61, löv 0.43, icke-skog
0.32, bland 0.30 — all classes functional, unlike the fraction path.
All-944 (secondary; 735 plots trained the head): 0.527 / 0.397.

## The arc's conclusion

Three supervision strategies, one representation:
1. Hard NMD labels (2018 or 2023): capped at the label proxy's ceiling (~0.46).
2. NFI-supervised head (points): 0.617 on held-out plots — proves the target,
   not the representation, was the constraint.
3. Getting it DENSE: fraction supervision + NFI collapse (0.579, forest-type
   layer) or distillation (0.502, full LULC product). Both beat NMD2023.

Production recommendation: distilled model as the 28-class base + the fraction
head (with hard-head forest mask) as the forest-type refinement layer.

Checkpoint: /cephfs/checkpoints/v8b_nmd2023_distill/best_model.pt.
Suite JSON: docs/data/nfi-validation-distill.json; per-plot dump on the PVC.
