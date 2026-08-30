# The ladder's distillation stage — design notes

**Status:** design, nothing built yet · **Created:** 2026-08-30
**Parent:** [`label_source_ladder.md`](label_source_ladder.md) — this is the
per-column step between rungs 2 and 3.

The ladder's rung-3/4 manifests already point at `/cephfs/distill/{model}_r2`
(`gen_ladder_manifests.py:62-63`). The *consumers* are wired; the *producer*
does not exist. These are the notes for building it.

## The pipeline is three steps, not two

The parent plan describes it as "train the head, then the sidecars". It is
actually three, because the head trains on features, not tiles:

    r2(X) ─→ extract_plot_features ─→ train_distill_head ─→ distill_forest_labels ─→ r3(X), r4(X)
             GPU, ~small              CPU, seconds          GPU, ~1 h

| step | script | key args |
|---|---|---|
| 1 | `extract_plot_features.py` | `--checkpoint --plot-index --img-size --out --enable-markfukt --device` |
| 2 | `train_distill_head.py` | `--features --test-frac --seed --out-head --out-split` |
| 3 | `distill_forest_labels.py` | `--checkpoint --head --data-dir --label-dir --out-dir --img-size` |

`--enable-markfukt` on step 1 is **load-bearing**. Every rung-2 model is an
11-aux model (markfukt on, decided 2026-08-29); extracting features without the
flag builds a differently-shaped input and the 256-dim features would not be
the ones the model actually learned. It is a `store_true` — silent when absent.

## Verified inputs (checked on the PVC 2026-08-30)

`training-data-cephfs` (1.6 T, 266 G free, 84 % used) is ONE PVC mounted at
`/data` by the NFI jobs and `/cephfs` by the ladder jobs — the paths are
interchangeable modulo mount point.

    OK   /cephfs/nfi/nfi_index_unified_v2_512.parquet   plot index
    OK   /cephfs/nmd2023_labels                         cohort gate + label dir
    OK   /cephfs/unified_v2_512                         tiles
    OK   /cephfs/checkpoints/ladder/<model>_r1/best_model.pt
    MISS /cephfs/distill                                to be created

Six sidecar sets land under `/cephfs/distill/`. Check free space before rung 3
— 266 G is not obviously enough for six dense label sets over 7 882 tiles.

## Per-backbone regime (from the base manifests, must be preserved)

| model | backbone-name | img-size | multitemporal |
|---|---|---|---|
| prithvi300m | `prithvi_300m` | 496 | no |
| prithvi600m | `prithvi_600m` | 504 | yes |
| croma | `croma_base` | 504 | no |
| terramind | `terramind_v1_base` | 496 | no |
| tessera | `tessera_v1` | 504 | yes |
| clay | `clay_v1_5` | 504 | no |

`img-size` differs per column and must follow the column — it is part of the
backbone's regime, not a protocol knob.

## Four problems

### 0. Clay and CROMA were built at the wrong resolution — FIXED 2026-08-30

Both distill-stage scripts called
``infcmp.load_model(args.checkpoint, device)`` with no ``img_size``. Clay and
CROMA carry no ``pos_embed`` and omit ``img_size`` from their minimal registry
config, so the backbone was BUILT at 224 — wrong ``grid_size``, wrong PSP pool
count — and then handed 504 px tiles by ``run_inference``. Prithvi recovers its
size from ``pos_embed``, which is why this survived: both scripts predate the
six-backbone ladder. ``infer_tiles.py:243`` already passed
``backbone_name=`` and ``img_size=`` and was the correct reference.

The failure is **silent**, not a crash: step 1 would capture 256-dim features
off a wrongly-shaped head and step 3 would write dense sidecars from one, so
2 of the 6 columns would distil garbage while looking healthy.
``distill_forest_labels.py`` even carried a "native != --img-size" warning and
proceeded anyway.

Fixed by forwarding ``img_size=args.img_size`` in both. Pinned by
``tests/test_distill_stage_wiring.py`` (AST-level — exercising the real path
needs GPU weights; what regresses is the keyword going missing).

## Three problems still open

### 1. The distillability metric does not exist yet

The parent plan defines distillability as the head's held-out **OOF**
forest-type accuracy and cites 0.637 as the reference. That number comes from
`nfi_head_cv.py` (`oof_predict`, 5-fold). But `train_distill_head.py` emits a
*single* grouped train/test split which it explicitly labels
`"TEST-SPLIT PREVIEW (head only, not the distilled dense model)"`. The plan's
claim that it "produces it as a by-product" is not true today.

**Fix:** run `nfi_head_cv.py` per column as well (CPU, seconds) and report its
OOF as distillability. `train_distill_head.py` keeps its job: producing the
deployable head. Do not silently redefine distillability as the single-split
preview — the plan's cross-backbone claim rides on the OOF number.

### 2. The split is not pinned across columns — silent

`grouped_split()` (`train_distill_head.py:51`) retries up to 20 seeds and keeps
whichever gives the best test-side class coverage. It prints a note and
continues. So two columns can end up on **different test tiles**.

The plan requires "same NFI plots, same grouped-by-tile split (same seed and
test-frac), same head architecture and hyperparameters" and calls protocol
pinning mandatory — because distillability is the ladder's only *cross-backbone*
claim. Nothing enforces it.

The risk is concrete: CROMA and TerraMind train on the ~6011/7882 SAR subset,
so their plot sets may differ from the other four columns, which can push them
onto a different trial seed.

**Fix — as a runnable check, not prose.** Compute the intersection of plots
available to all six columns, pin the split on that set, and **fail loudly** if
a column's plot set diverges from the pinned one. If the guard fires we decide
with real numbers rather than guessing at the trade-off now (smaller shared
plot set = noisier but controlled; per-column sets = larger but incomparable).

### 3. Rung 1 vs rung 2 as the distillation source

The chain distils **rung 2** features (`/cephfs/distill/<model>_r2`). Rung 2 has
not landed for any column yet, so nothing can be built against a real checkpoint
until it does — but the manifests, the guard and the tests can all be written
and unit-tested first.

## Build plan

1. Extend `scripts/gen_ladder_manifests.py` to emit the per-column distill jobs
   from the same `BASES` dict, so `img-size`/`backbone-name` follow the column
   and the head protocol is identical by construction. Generated, not
   hand-written — same anti-drift argument as the 24 ladder jobs.
2. Add the pinned-plot-set guard (problem 2) and the OOF report (problem 1).
3. Sidecar-count check before r3/r4 submit: the parent plan requires "verify the
   sidecar count matches the cohort count before that column proceeds".
4. Tests in `tests/` pinning: markfukt flag present on every extract job, one
   head hyperparameter set across all six, `--check` regeneration clean.

Template to copy: [`k8s/nfi-extract-plot-features-job.yaml`](../../k8s/nfi-extract-plot-features-job.yaml)
— note it runs on `accelerator: nvidia-gtx-2080ti` with 24 Gi and 1 GPU, so the
extract step does **not** compete with the H100 ladder quota. Keep that.
