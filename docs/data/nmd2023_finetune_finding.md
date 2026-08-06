# Finding — NMD2023-label finetune did NOT reproduce student-beats-teacher

Experiment (2026-08-06): warm-start v8b → finetune on NMD2023 sidecar labels,
28-class head (23 warm + 5 cold: torvtäkt/busk/ris/gräs/bar), identical 600M/504
regime, 15 epochs. NFI field-truth validate on the model's 944 plots.

## Result (NFI, 944 plots)
| Model / source | overall | kappa | forest-type acc |
|---|---|---|---|
| NMD2023 v2.1 (label source) | 0.493 | 0.366 | — |
| v8b+markfukt | 0.466 | 0.339 | — |
| v8b (NMD2018-trained) | 0.465 | 0.339 | 0.431 |
| **v8b-NMD2023** | **0.459** | 0.338 | 0.424 |

Per-class F1 (v8b → v8b-NMD2023): tall 0.572→0.524 (−0.048), gran 0.550→0.595
(+0.045), löv 0.486→0.430 (−0.056), bland 0.331→0.339, non-forest 0.301→0.311.

## Conclusion
The NMD2023-trained model did NOT beat NMD2023 (0.459 vs 0.493) and is marginally
below v8b (0.459 vs 0.465). The effect is a **redistribution** — granskog up,
tallskog/lövskog down — not a uniform gain. Student-beats-teacher is **not
confirmed** for NMD2023.

Caveats (why this is not a clean refutation):
1. **Undertrained** — 15-epoch finetune, val_mIoU still rising at 0.4777 (best
   epoch 15, no plateau). v8b had more training.
2. **Capacity split** — the 28-class head gives 5 new cold-started open-land
   classes that compete with the forest classes for capacity.

Next test to settle it: a longer / full retrain, or a 23-class forest-only
NMD2023 retrain (no new open-land classes) to isolate the label-source effect.
Checkpoint: /cephfs/checkpoints/v8b_nmd2023/best_model.pt (best val_mIoU 0.4777).
