# Plan — the label-source ladder (6 backbones × 4 rungs)

**Status:** plan, no GPU spent · **Created:** 2026-08-29 · **Owner:** Tobias
**Branch:** `agent/te/opus/nfi-nmd2023-benchmark`
**Generalises:** [`nmd2023_label_source_retrain.md`](nmd2023_label_source_retrain.md)
(same question, one backbone, two rungs) across the whole
[model race](model_race_plan.md) field.

## Why this replaces the ad-hoc retrain

Until now each backbone was trained at whatever label configuration it happened
to be launched with: `v8b-nmd2023-long` on raw NMD2023, the other five on
NFI-distilled labels **plus** the Trädslag fraction head. Two problems make that
unusable as evidence:

1. **The models are not comparable to each other** — they differ in label
   source *and* in how many heads they carry, so a win cannot be attributed.
2. **No contribution is isolated** — the distilled labels and the fraction head
   arrive together, so neither can be credited or blamed alone.

The requirement (user, 2026-08-29): *"As we are doing science, they should all
train on the same, starting with NMD 2023 and 2018 as teachers, after that
adding the heads. We must be able to follow all of them."* Every backbone walks
the same ladder; each rung adds exactly one thing.

## The ladder

Every rung is the same trainer, the same tiles (`/cephfs/unified_v2_512`), the
same regime per backbone. **Only the label flags change.**

| Rung | Name | Supervision | Flag delta |
|---|---|---|---|
| **1** | `nmd2018` | in-tile 23-class label (NMD2018 v1.1) | no `--label-dir`, `--num-classes 23` |
| **2** | `nmd2023` | 28-class NMD2023 v2.1 sidecar | `--label-dir /cephfs/nmd2023_labels --num-classes 28` |
| **3** | `nfi` | rung 2 + NFI-distilled forest type | `--label-dir /cephfs/nmd2023_distill_labels` |
| **4** | `tradslag` | rung 3 + Trädslag fraction head | rung 3 + `--frac-dir /cephfs/tradslag_fracs` |

Rungs 1 and 2 are the two **teachers** — the published national products, learned
straight. Rungs 3 and 4 are the two **refinements** we add on top. Reading rung
*n* against rung *n−1* isolates one contribution:

- **2 − 1** — is NMD2023 a better teacher than NMD2018?
- **3 − 2** — does distilling NFI field truth into forest type beat the teacher?
- **4 − 3** — does the continuous Trädslag crown-cover head add anything over
  the hard-label refinement?

### Controls

- **Cold start everywhere.** `--warm-start-from` is dropped on every rung,
  including Prithvi-600M. The existing `v8b-nmd2023-long` result (0.4892) is
  *not* a rung-2 cell: it warm-starts from an earlier NMD2023 checkpoint, so it
  measures continuation, not label source.
- **30 epochs everywhere**, so early-stopping differences don't masquerade as
  label-source effects.
- **Per-backbone regime is preserved** (crop 496 vs 504, gated aux-fusion for
  Tessera, ΔSAR for CROMA, …). Those differences are held constant *down* each
  column; they are never compared *across* columns except through the field
  metric below.

## The measurement problem, and the fix

Rung 1 is 23-class and rungs 2–4 are 28-class, so **mIoU cannot be compared
across that boundary** — different denominators, different class vocabularies.
Ranking the ladder on training-label mIoU would be a category error.

The ladder is therefore read on **field truth**, which is independent of the
label vocabulary a model was trained on:

- **Primary — NFI (Riksskogstaxeringen).** `model_race_standings.py` scores every
  member by held-out forest-type accuracy on the grouped-by-tile split, via a
  28→5-class forest collapse. Rung 1's 23-class output collapses to the same 5.
- **Cross-check — LUCAS.** ~160× the NFI sample across ~20 of the 28 classes,
  never trained on by any rung (`validate_against_lucas.py`).

Training-label mIoU is still recorded per run, but only for *within-rung*
comparison between backbones.

### Reference points (912 shared NFI plots, 5-class overall)

From [`nmd2023_label_source_retrain.md`](nmd2023_label_source_retrain.md):

| Source | Overall | Kappa |
|---|---|---|
| NMD2023 v2.1 (the rung-2 teacher) | 0.493 | 0.366 |
| v8b, NMD2018-trained (a rung-1 student) | 0.463 | 0.335 |
| NMD2018 v1.1 (the rung-1 teacher) | 0.406 | 0.257 |

The known result is that a rung-1 student (0.463) beats its own teacher
(0.406) — multitemporal S2 + LiDAR aux denoise the per-pixel label errors. The
ladder asks whether that survives a better teacher, and whether the NFI head
pushes past NMD2023's own 0.493.

## Ordering dependency — rung 3 cannot start early

The rung-3/4 labels in `/cephfs/nmd2023_distill_labels` are produced by
`distill_forest_labels.py` **from a rung-2 checkpoint** — it hooks that model's
256-dim pre-classifier features. The sidecars currently on disk came from the
*old, pre-phenology-repair* `v8b_nmd2023_long`, so they distil a teacher that no
longer exists.

**Hard order:** rung 2 completes → regenerate the distill head
(`train_distill_head.py`) and sidecars (`distill-forest-labels-job.yaml`) from
the new rung-2 checkpoint → rungs 3 and 4 may start.

Open question to settle before rung 3: **which** rung-2 checkpoint is the
teacher — Prithvi-600M (highest capacity, the historical choice) or each
backbone distilling from itself? Self-distillation would make rung 3 a
per-model result rather than a shared-teacher one. Recommendation: one shared
600M teacher, so rung 3 measures *the same* refinement for every backbone.

## Cost

24 runs at ~9 h. The namespace memory quota (250 Gi hard, ~39 Gi standing) fits
3 large (80 Gi: Prithvi-300M/600M, Tessera) or 4 small (48 Gi: CROMA, TerraMind,
Clay) concurrently. At ~3.5 average concurrency: **≈ 62 h wall-clock**, plus the
distillation regeneration between rungs 2 and 3.

GPUs are not the constraint — 6 of 8 H100 would suffice; memory quota is.

## Execution order

1. **Rungs 1 and 2 interleaved** (12 runs). No data dependencies; both teachers
   already exist on the PVC. Pack by memory, largest first.
2. **Regenerate the distill head + sidecars** from the chosen rung-2 checkpoint.
   Verify sidecar count matches the tile count before proceeding.
3. **Rungs 3 and 4 interleaved** (12 runs).
4. **Score every checkpoint** through the cached two-stage eval
   (`infer_tiles.py` → `score_against_truth.py`), then
   `model_race_standings.py` for the NFI ranking and the LUCAS cross-check.

## Manifests

Generated, not hand-written — 24 near-identical files invite drift:

    python scripts/gen_ladder_manifests.py            # writes k8s/ladder/
    python scripts/gen_ladder_manifests.py --check    # CI: regenerate & diff

Each job is named `ladder-r<N>-<model>`, labelled
`{purpose: ladder, rung: r<N>, model: <model>}` so a sweep can be selected by
rung or by model, and checkpoints land in
`/cephfs/checkpoints/ladder/<model>_r<N>/`. Nothing overwrites the existing
non-ladder checkpoint dirs.

## Risks

- **Rung 1 tile coverage.** The in-tile 23-class labels exist for every tile by
  construction; the NMD2023 sidecars cover ~94.5% (basskikt v2.1 coverage). Rung
  1 therefore trains on a *superset* of rungs 2–4. Either accept it and report
  the counts, or restrict all rungs to the sidecar-covered intersection. **This
  needs a decision before launch** — recommendation: restrict, so the ladder
  varies one thing.
- **CROMA/TerraMind SAR subset.** Both require `s1_vv_vh` (~6011/7882 tiles),
  so their columns are a smaller cohort than the other four at every rung.
  Comparable down a column, not across.
- **`test_tile_year_priority` is failing** (pre-existing, year-0 semantics) —
  unrelated to the ladder but should be resolved before results are published.
