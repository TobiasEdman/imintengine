# Segmentation metric migration: ignored targets

The shared NumPy segmentation metric now identifies its semantics as
`miou-ignore-target-v2` (`imint.metrics.SEGMENTATION_METRIC_VERSION`).
Historical unversioned training mIoU and new mIoU are **not comparable**.
This is a scoring-definition change, not evidence of model improvement.

Previously, predictions of a foreground class on ignored background targets
could contribute false positives to that class's IoU. Now targets equal to
`ignore_index` are removed from the entire confusion matrix. A background
prediction on a valid foreground target still counts as an error. For example,
`pred=[1,1,1,1]`, `target=[1,0,0,0]`, `ignore_index=0` changes mIoU from
0.25 to 1.0 without any change to the predictions. There is no fixed offset
that converts historical scores to the new definition.

On scoreable targets, out-of-range, nonfinite or fractional predicted class
IDs raise `ValueError`; they are neither dropped nor clipped into a real class.
Thus `pred=[1,255]`, `target=[1,1]` cannot silently become perfect accuracy.
Targets outside the class range remain excluded, as before. Predictions on
excluded targets do not affect metrics. Shared training/per-tile evaluation
and global in-distribution evaluation use this same confusion-matrix contract.

## Checkpoints and continuation

Both full-state resume checkpoints and exported best/epoch checkpoints from
`LULCTrainer` contain `metric_version`. New metric dictionaries, training logs
and best-checkpoint metadata also carry this version. Explicit and automatic
full-state resume reject a missing or different version at trainer construction,
before RNG seeding, device/model setup or output-directory/log writes. The
training-entry fallback also checks before setup; neither path restores model,
optimizer, scheduler, patience, logs or historical best scores on rejection.
The direct resume loader repeats the guard, and collapse rewind checks the
best checkpoint's version before loading it.

Do not relabel an old checkpoint as v2: its `best_metric`, `best_miou`, chosen
`best_model.pt`, patience and selection history were computed differently.
To continue that exact training history, use its original implementation.
An explicitly requested weight-only warm start (including the spectral-only
stage transition) may load old weights into a **new output directory**, with
fresh optimizer/schedule, epoch, best score and patience; it does not make the
old score comparable. Existing inference use of weights is not blocked by this
resume guard. No automatic historical rescoring or checkpoint rewriting occurs.

The checkpoint-selection definitions `miou`, `worst_class_iou` and `combined`
all depend on these IoUs and share the migration requirement. The fraction
regression head has no separate validation-selection metric in this trainer.

## Historical results retained unchanged

These named artifacts contain historical training/validation mIoU, or
comparisons informed by it. Preserve their numbers and original context;
do not interpret a larger v2 number as improvement over them. No results,
checkpoints or datasets have been recomputed by this migration.
One explicit rescore has been made since, from stored best-epoch confusion
matrices and without modifying any log or checkpoint: the 28 distillation-ladder
cells, in [ladder_miou_v2_2026-10-13.md](../experiments/ladder_miou_v2_2026-10-13.md).

| Artifact | Historical mIoU / affected context |
|---|---|
| [README](../../README.md), [development notes](../DEVELOPMENT.md) | 37.5%, 44.14% |
| [Prithvi 300M model card](../data/prithvi_300m_256_MODEL_CARD.md) | 0.4716 |
| [NMD2023 finetuning](../data/nmd2023_finetune_finding.md) | 0.4777, 0.4846 |
| [Wetness auxiliary experiment](../data/wetness_aux_finding.md) | 0.535, 0.5527 |
| [Tree-species fractions](../data/tradslag_fraction_finding.md) | 0.4858 |
| [G1/G2 ensemble](../data/ensemble_g1g2_finding.md) | 0.4650 |
| [NFI findings](../data/nfi_validation_findings.md), [v8 JSON](../data/nfi-validation-v8.json) | checkpoint validation mIoU 0.5352, 0.5148 |
| [LULC dashboard data](../data/lulc-data.json) | displayed mIoU 43.27 |
| [2026-04-27 checkpoint notes](checkpoint_2026-04-27.md), [hyperparameters](hyperparameters.md) | 0.3663, 0.3082, 0.2675 |
| [VPP repair](../experiments/vpp_repair_effect.md), [label-source ladder](../experiments/label_source_ladder.md), [model race](../experiments/model_race_plan.md), [ladder distillation](../experiments/ladder_distill_stage.md) | historical training/validation mIoU tables and selection comparisons |
| [Conversation logs](../conversation_log/) | historical training tables |
| Existing training checkpoints | stored best scores and the historical `best_model.pt` choice; unversioned full-state resume is refused |

The global `eval_in_distribution` mIoU already excluded ignored targets, so
its in-range historical values are not changed by the ignored-target fix.
`train_pixel.py` uses a separate metric/training path; NFI/LUCAS accuracy and
kappa, and overall accuracy for in-range predictions, are not redefined here.
The NFI documents above are listed for their **checkpoint validation mIoU**,
not to invalidate their field-observation accuracy/kappa measurements.
