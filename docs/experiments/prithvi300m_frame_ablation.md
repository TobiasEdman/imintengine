# Prithvi-300M temporal-frame ablation (1-frame vs 4-frame)

**Status:** base manifest committed; generator/queue/dashboard wiring + cluster
run PENDING (gated behind PR #36 — see Sequencing). **Created:** 2026-09-03.

## Why

The label-source ladder trained prithvi300m **single-frame** while prithvi600m
(its only Prithvi sibling) trained **4-frame** — an overlooked consequence of
the two using different base manifests. `prithvi_300m` is
`native_num_frames=(1,2,3,4)` (`imint/fm/registry.py`), so 4 frames is
admissible; the tiles carry 4 (autumn + 3 VPP season). Rather than *replace*
the 1-frame column (losing the data point), we **add** a 4-frame column so the
pair becomes a controlled ablation: identical backbone, identical
hyperparameters, ONLY `--enable-multitemporal --num-temporal-frames 4` differs
→ a delta attributes purely to temporal frames. [user-stated 2026-09-03]

The other four ladder backbones (tessera/clay/croma/terramind) are
`native_num_frames=(1,)` — architecturally single-date/annual, so their
1-frame training is correct, not a defect. This ablation is Prithvi-only.

## The new column: `prithvi300m4f`

Base: `k8s/train-prithvi300m-4f-job.yaml` (committed) — byte-identical to
`train-prithvi300m-job.yaml` except the two multitemporal flags. batch-size 8
kept unchanged (300m 4-frame = 3844 tokens < 600m 4-frame = 5184, which runs
batch 8 on 80Gi) so the ablation stays clean.

## Wiring (apply AFTER #36 merges — rebase onto main first)

`gen_ladder_manifests.py` is HOT in #36; editing it in parallel conflicts on
the crux file. Do all of the below on a branch rebased onto post-#36 main:

1. `BASES["prithvi300m4f"] = "k8s/train-prithvi300m-4f-job.yaml"`
2. `DISTILL["prithvi300m4f"] = {"img_size": 496, "backbone": "prithvi_300m"}`
   (features come from the 4-frame forward via run_inference threading
   `model.num_frames`; same as any Prithvi distill).
3. Gate crop-distill generation to SKIP `prithvi300m4f` for now — the crop
   stage's per-model UID map (2001-2006, from #36) has no 7th slot; the frame
   ablation needs only ladder r1-r4 + NFI distill, not crop/LUCAS. Extend the
   UID map to 2007 only if a crop column is later justified.
4. `ladder_queue.py` MODEL_ORDER: append `prithvi300m4f` (48Gi-class, place
   after the 80Gi backbones).
5. `dashboards/ladder_dashboard.html` MODELS + LABELS: add `prithvi300m4f`
   ("Prithvi-300M ·4f") — fold into #37 or a follow-up.
6. Tests: `test_ladder_manifests` matrix count updates 24→28 (+4 rungs);
   `test_every_cell_of_the_matrix_exists` accordingly.

## Sequencing / constraints

- **Cluster run gated behind #36 apply-window**: prithvi300m4f r1-r4 are
  full-PVC-RW H100 jobs — exactly the class barred during Codex's apply window.
  Launch only after #36 apply + split-3 + restore.
- **Cost:** 4 rungs × ~a few h H100 + 1 distill (2080ti). Confirm scope with
  user — full r1-r4 vs. a minimal r1+r2+distill first cut.
- **eval:** the frame-ablation checkpoints, once trained, get NFI+LUCAS eval via
  the same per-cell jobs (the `num_frames`-from-checkpoint eval fix, PR #38,
  handles 4-frame correctly).

## Read-out

Compare `prithvi300m` vs `prithvi300m4f` at each rung on: val mIoU, NFI
held-out 5-class accuracy, and distillability OOF. The 4f−1f delta at matched
rung/label-source is the temporal-frame effect, isolated.
