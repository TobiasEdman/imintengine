# Prithvi-300M temporal-frame ablation (1-frame vs 4-frame)

**Status:** wiring in PR #42; #36 merged 2026-09-07 so both sequencing gates
are CLEARED — training starts automatically via ladder-queue when #42 merges.
**Created:** 2026-09-03. **Claim revised 2026-09-07:** see "What this measures".

## Why

The label-source ladder trained prithvi300m **single-frame** while prithvi600m
(its only Prithvi sibling) trained **4-frame** — an overlooked consequence of
the two using different base manifests. `prithvi_300m` is
`native_num_frames=(1,2,3,4)` (`imint/fm/registry.py`), so 4 frames is
admissible; the tiles carry 4 (autumn + 3 VPP season). Rather than *replace*
the 1-frame column (losing the data point), we **add** a 4-frame column so the
pair shares backbone, hyperparameters and training flags, with ONLY
`--enable-multitemporal --num-temporal-frames 4` differing at the flag level.
[user-stated 2026-09-03]

## What this measures (revised 2026-09-07, PR #42 review)

The archived 1-frame cells ran an earlier runtime identity (source
`ceb821f`, image digest `6d85378d`); the 4f jobs pin the current one
(digest-pinned image, baked source SHA, dep-freeze record). The 4f−1f
delta is therefore an **exploratory historical comparison** — flags are
matched, runtime identity is not — NOT a runtime-identical controlled
ablation. Upgrading it requires a paired pinned rerun of BOTH arms
(4 + 4 H100 jobs, separate cost approval via Codex).

The other four ladder backbones (tessera/clay/croma/terramind) are
`native_num_frames=(1,)` — architecturally single-date/annual, so their
1-frame training is correct, not a defect. This ablation is Prithvi-only.

## The new column: `prithvi300m4f`

Base: `k8s/train-prithvi300m-4f-job.yaml` — same training flags as
`train-prithvi300m-job.yaml` except the two multitemporal flags; runtime
identity is pinned and differs from the archived 1f arm (see "What this
measures"). batch-size 8 kept unchanged (300m 4-frame = 3844 tokens < 600m 4-frame = 5184, which runs
batch 8 on 80Gi) so batch size matches the 1f arm.

## Wiring (landed in PR #42)

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
4. `ladder_queue.py` MODEL_ORDER: `prithvi300m4f` after its 1f sibling.
   (The base manifest requests **80Gi**, same as 1f — the 48Gi guess in
   earlier drafts was wrong; the manifest is ground truth.)
5. `dashboards/ladder_dashboard.html` MODELS + LABELS: add `prithvi300m4f`
   ("Prithvi-300M ·4f") — fold into #37 or a follow-up.
6. Tests: `test_ladder_manifests` matrix count updates 24→28 (+4 rungs);
   `test_every_cell_of_the_matrix_exists` accordingly.

## Sequencing / constraints

- ~~Cluster run gated behind #36 apply-window~~ — CLEARED 2026-09-07:
  #36 merged, split verified (attempt-17), restore complete. H100 cost
  approved by Tobias 2026-09-07 ("H100 är OK", via Codex).
- **Cost:** 4 rungs × ~a few h H100 + 1 distill (2080ti). APPROVED by
  Tobias 2026-09-07 ("H100 är OK", via Codex) for the full r1-r4 scope —
  do not re-ask unless the scope changes.
- **eval:** the frame-ablation checkpoints, once trained, get NFI+LUCAS eval via
  the same per-cell jobs (the `num_frames`-from-checkpoint eval fix, PR #38,
  handles 4-frame correctly).

## Read-out

Compare `prithvi300m` vs `prithvi300m4f` at each rung on: val mIoU, NFI
held-out 5-class accuracy, and distillability OOF. The 4f−1f delta at matched
rung/label-source ESTIMATES the temporal-frame effect. It is an
exploratory historical comparison (runtime identities differ between the
arms — see "What this measures"), not an isolated causal measurement.
