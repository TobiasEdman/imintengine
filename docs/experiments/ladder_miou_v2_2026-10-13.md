# Ladder mIoU under `miou-ignore-target-v2`: a stored-matrix rescore

Run 2026-10-13 20:35:19 to 20:43:06 UTC as `ladder-miou-v2-rescore-xv575`
(uid `ebb11184-3f04-4ff5-90bd-1e8fbfa3b867`), CPU only, from `52a0c90`
(manifest `659abcb`). It rescored all 28 cells of the distillation ladder
(7 backbones × rungs R1–R4, `/cephfs/checkpoints/ladder/<backbone>_r<N>/`)
under the metric PR #76 introduced, and wrote one new `miou_v2_rescore.json`
per cell plus `/cephfs/checkpoints/ladder/_miou_v2_rescore_20261013.json`.
No existing file was modified. Inputs, hashes, image, Job and the full
per-cell results are in the [JSON record](ladder_miou_v2_20261013.json).

## Verdict

**Every within-rung backbone ordering is unchanged.** At each rung, v2 raises
every cell by +0.032 to +0.053 and keeps the ranking. The rung-to-rung signs
are also unchanged: R3 − R2 is positive for all seven backbones, and R4 − R3 is
negative for six, with `prithvi600m` the exception, under both metrics.
The levels move; nothing that was read off the ladder's mIoU changes.

The offset is not constant. `tessera` gains the most at every rung and `clay`
the least, so the top-to-bottom spread widens (R1: 0.1113 → 0.1298). The
historical numbers cannot be converted with a fixed shift, as
[metric_migration.md](../training/metric_migration.md) already states.

## Method

Each `training_log.json` stores the full validation confusion matrix of every
new-best epoch (rows = target, cols = prediction; 23 × 23 at R1, 28 × 28 at
R2–R4). Both metrics are pure functions of that matrix:

- **old** (`miou-unversioned-v1`, `compute_miou` at `905c656^1`): mean IoU
  over classes 1..n−1 on the full matrix, skipping zero denominators. Target-0
  rows stay in, so a foreground prediction on a background target is a false
  positive for that class. This is frozen in the script as
  `legacy_miou_from_confusion`.
- **v2** (`miou-ignore-target-v2`): the same after dropping row 0, computed
  by `imint.metrics.miou_from_confusion`. `compute_miou` now calls this same
  function, so the rescore uses the shipped code path, not a copy.

The best epoch's matrix is the evidence the stored score was computed from.
Scoring it under v2 is therefore exact, not an approximation of re-inference.
Cost: 7 min 47 s of 2 CPUs, **0 GPU-h**, against the ~13 GPU-h estimated
to re-run inference on the 28 checkpoints. Randomised tests
(`tests/test_ladder_miou_v2_rescore.py`) check that the matrix route equals
`compute_miou` on pixels for both definitions, with background targets and
classes predicted but absent from the target.

## Reverse check: 28 / 28

A cell was refused unless two checks held:

- The old metric recomputed from the matrix equals the stored `best_metric`
  to 4 dp.
- `best_model.pt`, loaded `weights_only` on CPU from the exact bytes whose
  sha256 is recorded, carries `epoch == best_epoch` and
  `metrics.miou == best_metric`.

Every cell selected on mIoU (`early_stop_metric = "miou"`), so `best_metric`
is the mIoU. All 28 passed both checks. All 28 checkpoints have
`metric_version = null`: they predate v2, and the trainer's resume guard
refuses them, as designed.

## Results

Best-epoch val mIoU, old and v2, side by side:

| backbone | R1 old | R1 v2 | R2 old | R2 v2 | R3 old | R3 v2 | R4 old | R4 v2 |
|---|---|---|---|---|---|---|---|---|
| clay | 0.4541 | 0.4865 | 0.3902 | 0.4258 | 0.4190 | 0.4580 | 0.4171 | 0.4558 |
| croma | 0.5056 | 0.5461 | 0.4420 | 0.4830 | 0.4720 | 0.5164 | 0.4716 | 0.5156 |
| prithvi300m | 0.5161 | 0.5557 | 0.4498 | 0.4911 | 0.4699 | 0.5141 | 0.4680 | 0.5115 |
| prithvi300m4f | 0.5430 | 0.5843 | 0.4708 | 0.5154 | 0.4929 | 0.5401 | 0.4912 | 0.5376 |
| prithvi600m | 0.5561 | 0.5985 | 0.4800 | 0.5231 | 0.5044 | 0.5505 | 0.5067 | 0.5536 |
| terramind | 0.5137 | 0.5545 | 0.4469 | 0.4876 | 0.4691 | 0.5138 | 0.4684 | 0.5120 |
| tessera | 0.5654 | 0.6163 | 0.4987 | 0.5478 | 0.5307 | 0.5840 | 0.5292 | 0.5823 |

Delta (v2 − old):

| backbone | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| clay | +0.0324 | +0.0356 | +0.0390 | +0.0387 |
| croma | +0.0406 | +0.0410 | +0.0444 | +0.0441 |
| prithvi300m | +0.0396 | +0.0414 | +0.0442 | +0.0435 |
| prithvi300m4f | +0.0413 | +0.0446 | +0.0472 | +0.0463 |
| prithvi600m | +0.0424 | +0.0430 | +0.0462 | +0.0469 |
| terramind | +0.0407 | +0.0407 | +0.0448 | +0.0436 |
| tessera | +0.0509 | +0.0490 | +0.0533 | +0.0531 |
| **mean** | **+0.0411** | **+0.0422** | **+0.0456** | **+0.0452** |

At R1 the largest per-class gains, averaged over backbones, are in the small
crop classes: potatis (17) +0.17, sockerbetor (18) +0.10, slåttervall (15)
+0.09. These are the classes where predictions on background pixels made up
most of the old false positives.

### Ordering per rung

| rung | ordering (old = v2) | changed |
|---|---|---|
| R1 | tessera > prithvi600m > prithvi300m4f > prithvi300m > terramind > croma > clay | no |
| R2 | tessera > prithvi600m > prithvi300m4f > prithvi300m > terramind > croma > clay | no |
| R3 | tessera > prithvi600m > prithvi300m4f > croma > prithvi300m > terramind > clay | no |
| R4 | tessera > prithvi600m > prithvi300m4f > croma > terramind > prithvi300m > clay | no |

The only fragile position is `prithvi300m` against `terramind`. Their v2 gap
is 0.0012 at R1, 0.0002 at R3 and 0.0006 at R4, and their order flips between
R3 and R4 under both metrics. One run per cell cannot separate them.

### Rung comparisons

| backbone | R3 − R2 old | R3 − R2 v2 | R4 − R3 old | R4 − R3 v2 |
|---|---|---|---|---|
| clay | +0.0288 | +0.0322 | −0.0020 | −0.0022 |
| croma | +0.0300 | +0.0333 | −0.0004 | −0.0007 |
| prithvi300m | +0.0201 | +0.0229 | −0.0018 | −0.0026 |
| prithvi300m4f | +0.0222 | +0.0248 | −0.0017 | −0.0026 |
| prithvi600m | +0.0243 | +0.0275 | +0.0023 | +0.0030 |
| terramind | +0.0222 | +0.0263 | −0.0007 | −0.0018 |
| tessera | +0.0320 | +0.0362 | −0.0015 | −0.0016 |

R3 and R4 are scored on **identical val targets**: the target row-sum hash is
equal in all seven columns, as expected, since R4 only adds the Trädslag
fraction head on R3's labels. So R4 − R3 compares the same pixels and labels.
R3 − R2 does not: the targets change from the NMD2023 sidecar to the distilled
labels, under either metric. **R1 (23 classes) is not comparable with R2–R4
(28 classes)** ([label_source_ladder.md](label_source_ladder.md), "The
measurement problem"). R2 − R1 is therefore not reported as a result.

The R1 contrasts that [ladder_distill_stage.md](ladder_distill_stage.md)
cites as "+2.69 and +1.31" (temporality `prithvi300m4f − prithvi300m`, scale
`prithvi600m − prithvi300m4f`) are +2.86 and +1.42 under v2. Same signs, same
order.

## What this does and does not change for R5

R5 is defined as R4 + LUCAS crop type [user-stated 2026-08-31]
([ladder_distill_stage.md](ladder_distill_stage.md), "The LUCAS crop-distill
stage"). Its evidence is the crop-distillability table: OOF accuracy and κ on
2,491 LUCAS plots. That table does not use segmentation mIoU and **is not
touched by this rescore**.

What the new numbers change: the level of every ladder mIoU, by +0.03 to
+0.05. A future R5 run will score under v2 natively. Compare it against the v2
R4 column above, never the old one, and only after checking that its class
vocabulary and val target fingerprint match R4's.

What they do not change:

- the backbone ranking at any rung;
- the sign of R3 − R2 and R4 − R3 for any backbone;
- hence any argument for or against R5 built on the ladder's mIoU. R4 still
  does not improve on R3 for six of seven backbones (|Δ| ≤ 0.003, single runs),
  and `tessera` still leads every rung.

## Caveats

- **Checkpoint selection was made with the old metric.** Only new-best epochs
  stored a matrix (16–24 of 30 per cell). Epochs that never became the old
  best cannot be scored, so whether another epoch would win under v2 is
  unknown. Among the epochs that can be scored, v2 would have picked an
  earlier one in 3 of 28 cells:
  - `clay_r2`: epoch 26 over 28, +0.0001 v2;
  - `prithvi300m_r1`: epoch 28 over 30, +0.0004;
  - `prithvi300m_r4`: epoch 28 over 30, +0.0006.

  None of those epochs has a saved checkpoint. Old-metric margins to the
  runner-up epoch are small: median 0.0003 and maximum 0.0014; 21 of 28 cells
  are within 0.0005. In 5 cells a later epoch tied at 4 dp, and the strict `>`
  kept the earlier one (`croma_r3`, `croma_r4`, `prithvi300m4f_r3`,
  `prithvi300m_r2`, `terramind_r3`). Within a cell, the v2 − old offset varies
  by a median 0.0013 (max 0.0035) over the last five scorable epochs. That
  exceeds those margins, so a v2 selection may differ in more cells than the
  three measured, by amounts of that order.
- **No git sha is recorded in the checkpoints or logs.** The training code
  version behind each cell is not bound by this record.
- **The val set is per backbone.** It is the trainer's hash split,
  `md5(file name incl. .npz) % 100 < 10`, because
  `/cephfs/unified_v2_512/split_val.txt` does not exist. That rule is then
  filtered by each backbone's tile requirements:
  - 825 tiles: clay, prithvi300m, prithvi300m4f, prithvi600m, and tessera at
    R1–R2;
  - 819 tiles: croma and terramind (SAR subset);
  - 822 tiles: tessera at R3–R4.

  Pixel totals also differ with image size (496 or 504). The per-cell
  fingerprint (pixel total, target row-sum sha256, target-0 share) is in the
  JSON. The target-0 share is 3.1 % at R1 and 10.4–10.6 % at R2–R4, yet the
  mean deltas are similar, so the delta does not scale with background share.

## Anomalies checked

- **`terramind_r3` and `terramind_r4` are not duplicated logs.** Their
  `updated_at` values are 0.34 s apart and their last-epoch `elapsed_s` is
  identical (195.8), but they started 0.9 s apart: a parallel launch on
  identical hardware. Train losses differ in 30 of 30 epochs and val mIoU in
  30 of 30. Their matrices, `training_log.json` and `best_model.pt` sha256s all
  differ. Only 3 of 30 per-epoch `elapsed_s` values coincide. `croma_r3/r4`
  (0.7 s) and `prithvi300m4f_r1/r2` (1.9 s) were launched the same way.
- **`tessera`'s val set changes between R2 and R3:** 3 tiles fewer
  (209,563,200 → 208,801,152 px, exactly 3 × 504²). The other six columns keep
  one tile count across all four rungs. The cause was not investigated. It
  does not affect R4 − R3, which shares targets.
- **`status: "stopped"`** in `prithvi300m4f_r1`, `terramind_r1` and
  `tessera_r1`, although all three logged 30 epochs and their checkpoints
  pass the reverse check. Not investigated.
- **`tessera` checkpoints are 1.6 MB**, against 0.40–2.74 GB for the other
  columns. They load, and their epoch and mIoU match. Not investigated
  further.
- **The script's per-cell log line double-rounds.** It prints the 6 dp value
  at 4 dp, so `tessera_r4` shows 0.5824 where the exact value (0.582350 to 6 dp,
  0.5823 to 4 dp) is used above. The JSON values are unaffected.

## Provenance and write safety

The first submission (`ladder-miou-v2-rescore-72rnl`, manifest `d50ba13`)
never created a pod. The cluster's admission webhook requires resource limits
to equal requests. That Job is suspended, not deleted. A Job-level
`--dry-run=server` cannot catch this, because the webhook sees only Pods; the
fix (`659abcb`) was validated by a server dry run of the pod template.

The run cloned `52a0c90` and passed the sha256 guards on the script,
`imint/metrics.py` and the reused checkpoint loader. It used image
`python:3.11-slim@sha256:5be45dba…`, with numpy 2.4.6 and torch 2.10.0+cpu
as in the ladder inference freeze record of 2026-09-24. It wrote only with
exclusive create, and only after all 28 cells had passed.

The `mtime_ns` and size of all 132 pre-existing files in the 28 cell
directories were read through the read-only dashboard pod before and after
the run: none changed, and exactly 28 files were added. The on-PVC summary
equals the block printed in the pod log, and each per-cell file equals its
summary record. Log sha256:
`b215c97401b4f8cd17131d9d99a24009e3f43a8a74e82b9a045917f05d0740f1`.
