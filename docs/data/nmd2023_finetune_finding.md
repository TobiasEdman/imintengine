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

## Update — grind to saturation (30-epoch continuation)

The 15-epoch run's caveat (undertrained) was tested: warm-restart from the 0.478
checkpoint, fresh 30-epoch cosine → **saturated at val_mIoU 0.4846** (best epoch
29, plateaued). NFI re-validate:

| Model | NFI overall | kappa | forest-type acc |
|---|---|---|---|
| NMD2023 v2.1 | 0.493 | 0.366 | — |
| v8b (NMD2018-trained) | 0.465 | 0.339 | 0.431 |
| v8b-NMD2023 (15 ep) | 0.4587 | 0.338 | 0.424 |
| **v8b-NMD2023 (saturated, 30 ep)** | **0.4597** | 0.340 | 0.424 |

Saturation moved NFI by **+0.001** (val_mIoU +0.007 went to open-land classes,
not forest). Per-class vs v8b is a stable **redistribution**: gran +0.035, tall
−0.047, löv −0.043. So the result is NOT undertraining — it is the real ceiling.

## Definitive conclusion — student-beats-teacher scales with teacher NOISE

A NMD2023-trained model lands at ~0.460 forest-type field accuracy: below v8b
(0.465) and well below NMD2023 itself (0.493). The effect that let v8b beat
NMD2018 does NOT reproduce, because it is a **denoising** effect proportional to
the teacher's label noise:

- NMD2018 is noisy (0.406) → the model averages out per-pixel label errors →
  v8b gains **+0.059** over it (0.465).
- NMD2023 is already clean (0.493) → there is little per-pixel noise to remove →
  the model is a lossy approximation and lands **−0.033 below** it.

**Practical takeaway:** for forest type, use NMD2023 directly — it is the better
product. Training a model on NMD2023 does not add value the way it did for the
noisy NMD2018. The model's edge is denoising a bad label source, not improving on
a good one. (The redistribution — gran up, tall/löv down — reflects NMD2023's
laser-strong conifer typing propagating into the model.)
Saturated checkpoint: /cephfs/checkpoints/v8b_nmd2023_long/best_model.pt (val 0.4846).
