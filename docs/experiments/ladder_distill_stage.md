# The ladder's distillation stage — design notes

**Status:** BUILT 2026-08-30 (see "How it resolved" at the end) · **Created:** 2026-08-30
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

## How it resolved (built 2026-08-30)

The stage is generated by `gen_ladder_manifests.py` into `k8s/ladder/`:
one `distill-<model>-job.yaml` per column (four steps in one 2080ti pod:
extract → pinned-OOF distillability → deployable head → dense sidecars +
cohort gate) plus a shared one-shot `distill-pinned-plots-job.yaml`.
Everything is pinned by `tests/test_distill_stage_wiring.py` (40 tests).

**Problem 1 (no distillability metric)** — resolved by running
`nfi_head_cv.py --heads mlp --folds 5` per column as step 2. The head
config is byte-identical between `nfi_head_cv.py` and
`train_distill_head.py` (`MLP((128,)), max_iter=500, early_stopping,
seed 42`), so the historical 0.637 reference stays comparable. Output:
`/cephfs/distill/heads/<model>_r2_distillability.json`.

**Problem 2 (split not pinned)** — resolved WITHOUT a cross-column
barrier by deriving the shared plot set from tile-file properties, not
extraction outcomes: `build_pinned_plot_set.py` reads the npz zip
directories once (no GPU, no model) and pins every plot whose tile
carries `s1_vv_vh` — the intersection requirement imposed by
CROMA/TerraMind. StratifiedKFold's folds depend only on (n, y), so
same plots + canonical (tile, Tract, Plot) order = identical folds in
every column. `nfi_head_cv.py --pinned-plots` subsets, sorts, and
EXITS NON-ZERO if extraction dropped any pinned plot. The deployable
head still trains on each column's full plot set — two numbers, two
purposes: distillability is the controlled cross-backbone comparison,
the head is the best sidecar-writer for that column.

**Problem 0bis (found during build):** `distill_forest_labels.py`'s
tile loop had no status for "this family cannot build an input from
this tile" — a CROMA dense pass would have crashed ~4 h in on the first
optical-only tile. Now: `--require-npz-key s1_vv_vh` pre-filters the
cohort for the SAR columns (zip-directory probe, no arrays loaded),
per-tile exceptions are contained and counted, and the run ends with
the **sidecar gate**: `ok + exists == cohort` or exit non-zero, which
is the plan's "verify the sidecar count matches the cohort count
before that column proceeds" made mechanical.

**Protocol clarification (2026-08-31, after tessera's first submission):**
the feature WIDTH is a backbone property, not a protocol constant —
tessera's classifier reads 128-dim (`final_in = hidden // 2`, verified in
the r2 checkpoint: `model.classifier.weight (28, 128, 1, 1)`), the UPerNet
families 256. The pinned protocol therefore fixes the head RECIPE
(MLP(128,), max_iter, seed, folds, plots, y) on each column's NATIVE
width, and the width is recorded via the parquet's column count and the
head npz's `n_features`. Distillability compares "whose representation is
the better substrate", and dimensionality is part of the representation.
`find_classifier` locates the class-projection conv per family and
refuses ambiguity; per-column loader deps (terratorch, CROMA, Clay) ride
the generated manifests.

**Not resolved here:** the manifests clone the benchmark branch, so
nothing can be submitted until this work is pushed and merged there.
Submission order: `distill-pinned-plots` once, then `distill-<model>`
as each column's rung 2 lands. The ladder queue does NOT submit these
(by design, it only feeds rungs 1–2).

## The LUCAS crop-distill stage (added 2026-09-01) — the R5 evidence pass

**Operational status, 2026-09-03:** split attempt 2 ran as UID/GID 2000 and
failed closed before freezing a split because 1,966 of the 2,074 post-window
candidate tiles were unreadable. No crop consumer has run and no successful
split exists. The retained failed record is evidence of failure, not success.

**R5 = R4 + LUCAS crop type** [user-stated 2026-08-31], and distillability
comes before any retraining. Before a rung 5 exists, every column therefore
gets a crop-distillability number using LUCAS crop points (unified classes
11–21). The experiment deliberately creates neither a gate nor a rung-5
manifest; the six crop-OOF results are evidence for the later decision.

### Immutable multi-phase bootstrap

The Jobs must not fetch a mutable branch or install dependencies at startup.
The stage is bootstrapped in this fixed order:

1. **Payload commit A** contains every byte a storage-prep, source-access,
   split, or crop Pod
   executes: loaders, split/extraction/scoring code, provenance validation,
   the image recipe, exact upstream source identities, and both dependency
   locks. It removes the old runnable crop/split YAMLs; no Job can start from
   an unpinned bootstrap state.
2. CI builds that exact commit into
   `ghcr.io/tobiasedman/imint-ladder-crop-distill`, exercises the offline
   runtime and least-privilege storage-prep smoke tests on the exact pushed
   digest, signs and exactly verifies the image, and only then publishes its
   immutable `@sha256:` digest evidence.
3. **Runtime-pin commit B** sets `CROP_DISTILL_SOURCE_GIT_SHA` to commit A and
   `CROP_DISTILL_IMAGE` to that signed digest. With all later hashes at their
   zero sentinels, `--crop-bootstrap-only` generates only deny-egress,
   storage-prep, and the read-only source-access PLAN. APPLY, split, and all
   crop consumers remain absent.
4. Run PLAN only after the external controller/mount freeze below. Its
   immutable canonical record is stored in the PLAN Pod UID leaf and printed
   to stdout. Independently review its exact 2,074-file inventory, 1,966
   repairs, 108 no-ops, Pod UID, and content SHA-256.
5. **Plan-authority commit C** pins the reviewed PLAN Pod UID and SHA-256.
   `--crop-apply-only` then generates the metadata-only APPLY Job; split and
   consumers remain absent.
6. Run APPLY under the same uninterrupted external freeze. Independently
   verify its completion record proves unchanged SHA-256, size, device/inode,
   and mtime for every candidate and records expected ctime changes.
7. **Completion-authority commit D** pins the APPLY Pod UID and completion
   SHA-256. `--crop-split-only` can now generate split attempt 3. The split
   remains non-root UID/GID 2000, drops all capabilities, and has no token.
8. Independently verify the split completion and frozen manifest. A later
   **consumer-authorization commit E** pins that externally observed digest
   as `CROP_DISTILL_SPLIT_MANIFEST_SHA256` and generates the six crop Jobs.
   Full generation rejects a zero, malformed, or abbreviated source, image,
   PLAN, completion, or split identity. Review E before any crop Job runs.

`CROP_DISTILL_SOURCE_GIT_SHA` is owned by this crop stage. In particular,
PR #37's inference-matrix payload has its own source constant and must never
bump the crop anchor. Ordinary generator or documentation edits also leave
the crop anchor untouched.

This is a one-way freeze. If any Pod-executed crop payload needs a bug fix
after commit A, the recovery procedure restarts all reviewed phases:
review a new complete payload A, build and verify its image, authorize and run
one new PLAN and APPLY, run one new split, then pin that new manifest. Do
not silently move an existing source/image/split tuple, and do not treat an
unrelated manifest/docs change as a reason to move the anchor.

### Least-privilege PVC preparation

The live CephFS `distill/` and `ops/` roots were verified as `0755 root:root`;
recursive `fsGroup` ownership changes would be both unreliable on RWX CephFS
and far too broad. A separate, no-argument storage-prep Job therefore mounts
only those two parents. Before its first mutation it verifies the baked
runtime against commit A and the reviewed image digest. It performs no PVC
data parsing or network access, runs as UID 0:GID 2000 with a read-only root
filesystem, no service-account token, no privilege escalation, RuntimeDefault
seccomp, and only `CHOWN` plus `FOWNER` after dropping every capability. It
prepares exactly 20 directories with baked ownership and modes:

- `/cephfs/distill/crop_split` is UID/GID `2000:2000`, mode `03770` until
  the completed freeze locks it to `0550`;
- `/cephfs/distill/crop_heads` and `/cephfs/ops/crop-distill` are root-owned,
  GID 2000, mode `0750` parents, so an unprivileged workload cannot create or
  replace another model's leaf;
- `/cephfs/ops/crop-distill/split` is the UID/GID `2000:2000`, mode `0750`
  split-evidence leaf; and
- `/cephfs/ops/crop-distill/source-access` plus its `plan`, `apply`, and
  `locks` leaves are root-owned, GID 2000, mode `0750`; every PLAN/APPLY run
  writes one create-only child named by its Downward-API Pod UID; and
- each model owns one mode-`0750` head leaf at
  `/cephfs/distill/crop_heads/<model>_r2_crop_runs` and one mode-`0750`
  evidence leaf at `/cephfs/ops/crop-distill/<model>`. The six owners are
  `clay:2001`, `croma:2002`, `prithvi300m:2003`, `prithvi600m:2004`,
  `terramind:2005`, and `tessera:2006`, all with GID 2000.

The split Job runs non-root as UID/GID 2000. Each crop model can write only
its pre-owned leaves; the shared GID permits read/traverse but not sibling
creation, replacement, or deletion. A seventh model requires a new UID and
two new leaves in a reviewed protocol change. Replaying routine storage prep
preserves the completed split's `0550` lock rather than reopening it. Broad
PVC roots are never changed. A future non-root reader must join GID 2000
explicitly; root-operated dashboard and reaper readers continue to work.

The model→UID map is part of commit A, not merely rendered YAML. Crop and split
entrypoints compare their effective UID/GID with that baked map before any PVC
work, and terminal records include the observed process identity. This catches
an applied-manifest drift or accidental UID collision even if Kubernetes
accepted the YAML.

All selected crop protocol Pods are covered by a namespace `NetworkPolicy`
with an empty egress list. Image pulling remains a kubelet operation; the
running root-prep, PLAN, APPLY, split, and crop containers have no network
path.

The Jobs reference the existing `ghcr-push` secret only through
`imagePullSecrets`; it is not mounted in a container. The credential's actual
registry scope has not been independently proven pull-only. Replacing it with
a dedicated `ghcr-pull` credential is recommended cluster hardening, but
renaming it in a manifest before that secret exists would make the reviewed
rollout unpullable.

### Source-access incident and repair authority

The failed split found a producer-side permission recurrence, not corrupt
NPZs: root-run atomic rewrites used `mkstemp(0600)` and replaced the original
inode without restoring its group/mode. `backfill_vpp.py` (also used by the
unstamped-VPP strip path) and the in-place `fill_tiles_l2a.py` path now retain
an existing destination's UID, GID, and permission bits by applying `fchown`
before `fchmod` to the temporary inode. All three in-place transactions use
the same dataset lock from initial descriptor read through fetch/transform and
replace. They pass the initial descriptor identity plus SHA-256 into one common
drift-detecting writer, which reopens and rehashes the live destination before
publication. POSIX has no portable inode-conditional replacement; exclusion
through the shared lock closes the final check-to-rename interval for these
cooperating writers, while the external watchdog/freeze remains mandatory for
arbitrary Kubernetes writers. The temporary payload is flushed and fsynced
before descriptor-relative rename, its metadata is fsynced after
`fchown`/`fchmod`, and the held parent directory descriptor is fsynced after
rename. For this producer path, `fchown` restores the captured owner and group;
it does not assign the repair Job's UID. New destinations retain private
`0600` and use atomic hard-link
no-replace publication, so a concurrent create is never overwritten. This
claim is deliberately limited to those writers.
`fetch_era5_aux.py` has the same primitive but tracked callers write separate
`/era5` checkpoint sidecars rather than this source population; it is follow-up
hardening, not authority to delay or broaden this repair.

PLAN derives scope only from the producer-shaped LUCAS parquet at exactly
468,614 bytes and SHA-256
`e3bb505c4de469d0436e0e91de27327f063083199c8ddc781ec6eaf7d42e9a41`.
It rejects a source `tile_path` column. Crop rows are normalized first, then
the `[8,504)` row/column crop window is applied, and only then are tile names
deduplicated and sorted: 3,587 crop rows yield exactly 2,074 candidate tiles.
The 2,120 pre-window tiles are not repair authority. Every candidate is opened
relative to a verified dataset directory descriptor with `O_NOFOLLOW`, must
be a regular nonempty file with link count one, and has SHA-256, device,
inode, size, mtime, ctime, UID, GID, mode, and action frozen in canonical JSON.

APPLY consumes only the Git-pinned PLAN Pod UID and SHA-256. Kubernetes must
mount the complete `unified_v2_512` dataset subPath RW (never the PVC root),
because it cannot project a dynamic 2,074-file mount set. The payload narrows
that broader mount: it preflights all 2,074 records before the first mutation,
opens targets `O_RDONLY`, and uses only descriptor metadata syscalls. It
changes only `root:root 0600` to `root:2000 0640`, using
`fchown(fd, -1, 2000)` to preserve UID 0 before `fchmod(fd, 0640)`. Existing
readable `0644` and already-correct `root:2000 0640` states
are metadata-for-metadata no-ops; an exact `root:2000 0600` interrupted state
may resume only for a PLAN repair. The Pod runs UID 0:GID 2000 with drop ALL
plus exactly `CHOWN,FOWNER`, no token, read-only root filesystem, and no
egress. Completion records before/after SHA-256, size, device/inode, mtime,
and observed ctime; content/size/device+inode/nlink/mtime must be unchanged,
while ctime must change for an applied repair. Every repaired fd is fsynced.
After the per-file loop, APPLY separately reopens and rehashes the entire
2,074-file cohort and compares each live identity exactly with its observed
completion `after` record before publishing completion. Ctime is never
predeclared: it is observed after repair, recorded, and rederived by later
full-cohort verification.

The `DAC_READ_SEARCH` canary was rejected by cluster admission before a Pod
ran, so it supplied no data-access evidence. `DAC_OVERRIDE` was considered
and rejected: it would bypass more filesystem authorization than the exact
metadata repair requires. Neither capability is part of PLAN, APPLY, split,
or crop workloads.

### Frozen runtime and upstream sources

The image carries exact CROMA and Clay source trees; no `git clone`, package
index lookup, or model-hub access is allowed in a Job:

| source | exact Git commit | source-archive SHA-256 |
|---|---|---|
| CROMA | `59505a6bcadbf36ba20767270154bf9f3067c5e7` | `939d0918991ad7604bbb0a782df2674b8e30ade6edc061bcad6ab486e6f94001` |
| Clay 1.5 | `f14e698f3c237cabf8d28dec669a362d66625381` | `0b908ea11d5348736c26512f695221a304883ec88bac68f66822ca07bf435d64` |

One environment cannot reproduce both model loading and the historical
scoring stack. The image therefore contains two isolated, hash-locked Python
3.11 environments:

- the **model environment** runs feature extraction with TerraTorch 1.2.11,
  TorchGeo 0.8.1, and NumPy 2.2.6 (satisfying TerraTorch's NumPy >=2.2
  requirement), plus the pinned Torch/CUDA, CROMA, Clay, Prithvi,
  TerraMind, and Tessera dependencies;
- the **scoring environment** builds/verifies the split and runs
  `nfi_head_cv.py` with NumPy 1.26.4, pandas 2.2.2, pyarrow 17.0.0,
  scikit-learn 1.5.1, and its pinned transitive dependencies.

The 2026-08-31 TerraMind training run did **not** log its installed
TerraTorch version directly. Its Python 3.11 base, successful dependency
installation at 07:47:18Z, and the package metadata available at that cutoff
determine the compatible resolution: TerraTorch 1.2.12 and newer require
TorchGeo 0.9, which requires Python 3.12, while TerraTorch 1.2.11 can resolve
with TorchGeo 0.8.1 on Python 3.11. This is a resolver-derived reconstruction,
not a directly measured package inventory, and must not later be described as
one. TerraTorch 1.2.11 itself requires NumPy >=2.2, which is why scoring stays
in a separate NumPy 1.26 environment.

Both environments install from `--require-hashes` locks during the image
build. Their lock digests and sorted installed-package snapshots are sealed
into the runtime manifest. Jobs run with package indexes and Hugging Face
offline, then verify the sealed source trees and dependency identities before
reading cluster data.

### Exact rung-2 checkpoints

Each crop Job opens its checkpoint with `O_NOFOLLOW`, rejects non-regular or
multiply linked files, and copies and hashes the exact open-descriptor bytes
into an anonymous, mode-`0600` private snapshot. It checks the shared inode,
link count, size, mtime, and ctime again after the copy, then calls
`torch.load(..., weights_only=True)` only on the private snapshot. Terminal
evidence binds this extractor-authenticated size/SHA identity without a second
pathname reopen or another 2.7 GB hash pass:

| model | size (bytes) | SHA-256 |
|---|---:|---|
| Clay | 2,601,012,332 | `0a37ebdbbae8ac61145424350ae8f2990225d2cb15a3e1c178c9d42134c226e2` |
| CROMA | 834,654,805 | `dbfc04cf9475ca6b604dd5133191854736e961deebeb992be855f211d152bd80` |
| Prithvi 300M | 1,285,893,675 | `a27dadd9caf1c9ccfba6ecbd76ac7815fcb7236978e9df807e1d1bf7a498cda0` |
| Prithvi 600M | 2,741,619,081 | `89d544c06fd353772722dec5600a4ba8696fd8971250f471b47f6b53828d1d46` |
| TerraMind | 401,358,843 | `97316cf22612288072f0278f5c90e1a987a845a35acb1dcb431cc13432b4fc8f` |
| Tessera | 1,596,322 | `9dd7cfcad09b26576d23c846c29c3fd540d463b97a72df0b7557f6558dbced04` |

A missing, truncated, replaced, linked, or merely similar checkpoint fails
before inference. There is no opt-in to unrestricted pickle. PyTorch 2.5.1 is
retained to reproduce the trained model ABI, but its `weights_only` unpickler
is not a sandbox: it is covered by known code-execution advisories. This run
therefore admits only the six first-party training artifacts whose exact size
and SHA-256 were anchored at review time on a trust-on-first-use basis. The
digests prove that bytes have not changed since that observation; they do not
prove the checkpoint was safe before it was hashed, and no separate static
pickle scan has been claimed. Digest authentication proves identity, not
safety of an initially malicious pickle. An untrusted or newly sourced
checkpoint requires a reviewed conversion to a non-Pickle tensor format, or a
separately validated stack on PyTorch >=2.10 before admission. The
non-root, read-only-root, no-egress and minimal-mount boundary limits impact but
does not replace that trust decision. Learned-key compatibility is checked
separately by the model loader, and the native-image acceptance must load all
six real checkpoints before deployment.

### Self-contained split freeze

The shared `k8s/ladder/lucas-crop-split-job.yaml` runs once, before any model
Job. `build_lucas_crop_split.py` groups by tile, freezes 70% for distillation
and 30% for validation, and preserves the historical holdout contract. That
contract is Git-anchored at tile level: the tracked prior split contains 53
test tiles, while the current L1 source intersection has 71 points on 24 of
those tiles. Point IDs are accepted only as canonical decimal strings or
integers, normalized once, and checked for uniqueness after normalization.
The exact 71 observed keys are then frozen and hashed for every later
source-independent check. It publishes four content-verified artifacts below
`/cephfs/distill/crop_split/`:

| artifact | permitted consumer |
|---|---|
| `lucas_crop_distill_index.parquet` | crop feature extraction only |
| `lucas_crop_validator_holdout_index.parquet` | later independent validator only |
| `lucas_crop_split.json` | split identity check |
| `lucas_crop_split.MANIFEST.json` | artifact/provenance verification |

The two parquet files have the same exact extraction schema. Together they
must equal the complete qualified source partition exactly once, remain
tile-disjoint, preserve required class support, and match all counts and
logical-key digests in the JSON and manifest.

The source LUCAS parquet path and SHA-256 record what the freeze was built
from. They are historical build provenance only: later verification is
self-contained and must remain valid if that mutable source is changed,
moved, or removed. The split-freeze verifier uses the two frozen parquets
plus their JSON/manifest identities; it does not silently reconstruct a new
partition from the live source.

Crop-distill commands are structurally prohibited from receiving or naming
`lucas_crop_validator_holdout_index.parquet`. They read only the distill
parquet and split JSON from the dedicated read-only `crop_consumer/`
projection. Commit C supplies the expected manifest SHA-256 from Git; the crop
runner never derives its own expectation from PVC bytes. It snapshots only
manifest, index, and split into Pod-private `/work`, authenticates and parses
the same descriptor bytes, and gives only that snapshot to extraction,
scoring, and provenance. A crop Pod does not stat, open, copy, or hash the
holdout. The holdout remains untouched until a separate, later validation
protocol is reviewed.

### Job outputs and terminal evidence

The generated per-column Job remains two steps:

| step | environment | script | crop-specific inputs |
|---|---|---|---|
| 1 | model | `extract_plot_features.py` | frozen distill parquet, exact rung-2 checkpoint, native image size/backbone |
| 2 | scoring | `nfi_head_cv.py` | extracted features, `unified_class`, pinned distill identities |

Every Pod creates intermediates only under private mode-`0700` `/work`, which
is a disk-backed rather than memory-backed `emptyDir`. It
first copies the shared checkpoint into an anonymous mode-`0600` file there
while checking the reviewed size and SHA-256, and only deserializes that
private snapshot. Thus a concurrent CephFS writer cannot change bytes after
authentication but before `torch.load`. Crop Pods request 4 GiB and cap their
`emptyDir` at 8 GiB, above the largest pinned 2.7 GB checkpoint. It publishes
features and OOF directly into the model-owned mode-`0750`
`/cephfs/distill/crop_heads/<model>_r2_crop_runs/` directory with the Pod UID in
each filename and final mode `0444`; published bytes never remain trapped in a
private work leaf. It then publishes one mode-`0444` write-once record below a
mode-`0750` `/cephfs/ops/crop-distill/<model>/<pod-uid>/` directory. The split
record uses `/cephfs/ops/crop-distill/split/<pod-uid>/`. The crop record binds
the Pod and Job identity to commit A, the signed image digest, runtime/source
trees, both dependency locks and installed snapshots, the split manifest and
distill-index identities, the exact checkpoint, and hashes/sizes of every
output. Output size and SHA-256 are computed while copying from the private
source descriptor, then rechecked during finalization; a pathname replacement
between publication and evidence cannot be blessed as the original output.
The split Job's own record binds all four frozen artifacts. Publication is
atomic; an identical retry is idempotent and different content at the same
identity is rejected.

The caught Python-exception/`KeyboardInterrupt` path also attempts a terminal
failure record with bounded raw source/image/runtime-manifest claims, a non-
zero exit code, and the failure stage. It deliberately does not dereference or
verify those claims and omits split, checkpoint, and output identities:
whichever input failed must not become trusted evidence merely because it
existed. Storage prep, a pre-`POD_UID` failure, `SIGKILL`, or a failed evidence
write can leave no record. Absence is fail-closed; a successful Kubernetes
status without the immutable completed record is not accepted as experimental
evidence.

The complete canonical record and its digest are also printed to Pod stdout.
While the Pod exists, the Kubernetes API log is an audit surface outside the
writable PVC. It is not durable: the Job TTL removes Pods and the log reaper
archives onto the same RWX trust domain. Before acceptance or deletion, the
canonical bytes/digest must therefore be compared with the PVC record and
committed under `docs/evidence/crop-distill/<pod-uid>/` as described by that
directory's README. A missing capture or disagreement invalidates the run.
Unix isolation still does not protect against a cluster administrator, the
storage-prep identity, or a hostile workload deliberately given the same UID.

### Deployment and verification order

Use this order. PLAN, APPLY, and split are one externally frozen window; every
stop condition is fail-closed.

1. Review payload A, build/sign its exact digest, and verify the offline
   runtime smoke. Pin that full source SHA and digest in B. Until this happens,
   all crop source/image constants stay at zero and the old a487 image is not
   runnable authority for the new payload. Generate/check only the non-crop
   files, then bootstrap prep/PLAN:

   ```bash
   python scripts/gen_ladder_manifests.py --non-crop-only --check
   python scripts/gen_ladder_manifests.py --crop-bootstrap-only
   python scripts/gen_ladder_manifests.py --crop-bootstrap-only --check
   ```

2. Before storage prep or PLAN, server-side dry-run and apply the expanded
   deny-egress policy, then re-read the live object:

   ```bash
   kubectl --context icekube -n prithvi-training-default apply \
     --server-side --dry-run=server \
     -f k8s/ladder/crop-distill-deny-egress.yaml
   kubectl --context icekube -n prithvi-training-default apply \
     --server-side -f k8s/ladder/crop-distill-deny-egress.yaml
   kubectl --context icekube -n prithvi-training-default get networkpolicy \
     ladder-crop-distill-deny-egress -o yaml
   ```

   Require the live selector to contain exactly the four purposes
   `ladder-crop-distill`, `ladder-crop-distill-storage`,
   `ladder-crop-source-access-plan`, and
   `ladder-crop-source-access-apply`, with `policyTypes: [Egress]` and an
   empty `egress` list. Stop before every Job if that re-read differs.

   Start the executable freeze protocol from a restricted operator directory;
   do not reproduce these checks with ad-hoc `kubectl` pipelines:

   ```bash
   install -d -m 0700 /secure/operator/crop-source-freeze
   python scripts/crop_source_freeze.py --context icekube \
     --namespace prithvi-training-default hold \
     --state-dir /secure/operator/crop-source-freeze \
     --run-id lucas-crop-attempt-3
   python scripts/crop_source_freeze.py --context icekube \
     --namespace prithvi-training-default watch \
     --run-dir /secure/operator/crop-source-freeze/lucas-crop-attempt-3
   ```

   The second command remains alive in a dedicated terminal until exact
   restoration. `hold` captures full CronJob JSON, UID, resourceVersion,
   presence/value of prior `spec.suspend`, and CAS-suspends `ladder-queue` and
   `gpu-reaper`; `campaign-orchestrator` must already be suspended and its
   exact state is recorded. Every re-read must retain UID, change
   resourceVersion for the two patched objects, and differ only by the
   suspend field. Before the first CronJob mutation, `hold` CAS-claims the
   singleton ConfigMap with an expired `initializing` lease that no data Job
   accepts. If `hold` crashes or a later controller CAS fails, the immutable
   before record plus that lease make `restore` a resumable exact rollback;
   do not start PLAN from a partially acquired hold.

   Hold retains a dedicated local ownership lock from run-directory creation
   through `hold-complete` or its fail-closed record. Incomplete-hold restore
   must acquire that same lock, so it can recover only after the hold process
   has exited; it cannot race a live controller-suspension sequence.

   Exactly one watchdog may own a run directory: a local nonblocking `flock`
   remains held for its lifetime. Restore takes that same lock before it
   publishes an immutable `restore-in-progress` marker, excluding a live
   watchdog and every later watchdog restart. A separate nonblocking phase
   lock serializes all gates and is held across restore's idle gate, stop,
   marker, and exact controller restoration; a gate refuses as soon as stop or
   restore state exists. A short publication lock makes the watchdog's final
   phase reread plus lease/heartbeat update atomic against gate publication and
   timeout rollback. `--once` publishes only a closed smoke-scan lease, never
   a held heartbeat; unilateral signals also close the live lease. Neither
   path creates restore authority. Only a restore-requested
   fresh zero-overlap idle scan (or the authenticated expiry path below) can
   create that terminal marker.

3. The watchdog continuously inventories nonterminal Pods, active or latent
   Jobs, and standard Pod-producing controller templates. It inspects init,
   normal, and ephemeral containers plus volume devices, resolves the
   `training-data-cephfs` claim and effective read-only state, and treats a
   full mount, exact `unified_v2_512`, an ancestor/descendant, unsafe path, or
   unresolved `subPathExpr` as RW overlap. Only the exact phase Job and Pods
   owned by that Job UID are considered for an exception. PLAN and split have
   no source-RW exception at all. APPLY is exempt only for one exact overlap:
   `containers/source-access-apply`, PVC volume `training-data-cephfs`, RW
   mount, and subPath `unified_v2_512`; a full mount, descendant, different
   container/volume, device, or additional overlap fails closed. Each full
   object inventory and controller
   re-read is canonical, create-only, mode-`0600`, and SHA-256-bound under the
   run directory. These files contain paths, UIDs, inode-adjacent operational
   identities, environment references, and workload topology: they are
   **restricted operations evidence** and must not be committed or copied into
   ordinary CI logs.

   ICE currently exposes no `ValidatingAdmissionPolicy`, and this identity
   cannot create a Gatekeeper `ConstraintTemplate` or validating webhook, so
   this is not an admission fence. A namespace admin can still race a create
   or mutation between 15-second polls. The residual is explicit and cannot be
   claimed away. The watchdog refreshes a phase-bound ConfigMap lease with a
   maximum 180-second lifetime (long enough for kubelet projected-ConfigMap
   lag). Every non-idle phase request has a random 256-bit request ID and an
   absolute 21,900-second expiry, covering the data-derived 21,600-second
   split deadline plus projection lag (PLAN and APPLY remain at 7,200 seconds).
   A heartbeat must echo that exact request ID
   after a fresh scan. Thus a gate process that dies after changing phase
   cannot leave APPLY or split armed indefinitely; expiry makes the watchdog
   rescan with idle policy and fail the lease closed. Exact restore may consume
   that terminal state only when the create-only stopped record, zero-overlap
   snapshot, expired request, and live failed lease all agree. The directory,
   never an individual key via
   `subPath`, is mounted
   read-only in PLAN/APPLY/split. If the watchdog dies, detects drift, loses
   its lease CAS, or sees an overlap, it stops issuing a held lease; phase
   authorization fails and payload checks abort before terminal publication.

4. Server-side dry-run and create storage prep. Verify exactly 20 baked
   targets. Authorize PLAN only after a fresh watchdog scan, then dry-run and
   create `crop-source-access-plan-job.yaml`:

   ```bash
   python scripts/crop_source_freeze.py --context icekube \
     --namespace prithvi-training-default gate \
     --run-dir /secure/operator/crop-source-freeze/lucas-crop-attempt-3 \
     --phase plan
   kubectl --context icekube -n prithvi-training-default apply \
     --dry-run=server -f k8s/ladder/crop-source-access-plan-job.yaml
   kubectl --context icekube -n prithvi-training-default create \
     -f k8s/ladder/crop-source-access-plan-job.yaml
   ```

   PLAN runs
   root with drop ALL, but tile/index mounts are read-only. Capture its exact
   Pod UID, stdout marker, and immutable `plan/<pod-uid>/plan.json`. Verify
   canonical bytes/hash, the pinned index, 3,587 rows, 2,074 sorted candidates,
   1,966 repairs, and 108 no-ops. The marker contains the complete inventory
   and is restricted; there is deliberately no second pretty-JSON copy on
   stdout. Stop and keep APPLY absent on disagreement.

5. Pin the reviewed PLAN Pod UID and SHA-256 in Git. Generate/check only
   `--crop-apply-only`; independently review it. Run the same `gate` command
   with `--phase apply`, then server-side dry-run/create APPLY. Verify it ran
   UID 0:GID 2000, drop ALL plus only `CHOWN,FOWNER`, with no token or egress,
   and that its completion proves content/hash/size/device+inode/nlink and
   mtime unchanged, with observed ctime changed for actual repairs. The
   watchdog lease is checked before and after preflight, for every tile, after
   the final full-cohort rescan, and before publication. Preserve the exact
   restricted stdout marker and `apply/<pod-uid>/completion.json`; stop on any
   missing/mismatch.

6. Pin the reviewed completion Pod UID and SHA-256 in Git. Generate/check only
   `--crop-split-only`; independently review attempt 3. Run `gate --phase
   split`, then dry-run/create the split. It must remain UID/GID 2000, non-root,
   drop ALL with no added capabilities, no token, no egress, and must consume
   the exact completion file/hash. It checks the fresh split-phase lease and
   re-runs the same complete live-cohort verifier immediately before freeze
   and after generation but before terminal completion. Verify all four frozen
   artifacts and the immutable split record. Also compare normalized
   observed-71-key digest to
   `7808d10432ae8ddfc40623c03e33e4eeb1b9cc8cbc94afc5b11184e9632ace86`.
   The Pod-scoped active deadline is 21,600 seconds. The original
   3,600-second deadline terminated a healthy, progressing full-cohort
   verification after about 523 GB of cumulative reads. The verified PLAN
   totals 225,658,470,857 source bytes; the code path has six full-cohort byte
   passes, for a conservative 1,353,950,825,142-byte workload. At the observed
   throughput that projects to about 9,320 seconds. Twice that projection is
   about 18,639 seconds, so the rounded six-hour deadline provides more than the
   required 2x margin. The 21,900-second phase request leaves 300 seconds over
   the Pod deadline, exceeding the 180-second lease allowance enforced by the
   manifest tests. The reviewed deterministic live subset smoke then exercised
   1,771,912,469 bytes through the production capture/hash/NPZ-decode path in
   8,628,106,202 ns, or 205,365,166 bytes/s when rounded down. Because that is
   faster than the full attempt's observed rate, the slower full-attempt rate
   remains the conservative deadline input. The smoke also verified the
   interpreter, mounts, identity, pins, lease, crop-window counts, and 16 of 16
   qualifying samples without another full-cohort pass.

7. Only after split evidence is captured off PVC may the freeze end. Use the
   protocol's restore, never blind patches:

   ```bash
   python scripts/crop_source_freeze.py --context icekube \
     --namespace prithvi-training-default restore \
     --run-dir /secure/operator/crop-source-freeze/lucas-crop-attempt-3
   ```

   After a completed hold, restore first gates an idle/zero-overlap scan,
   fail-closes the lease and waits for watchdog shutdown. For an interrupted
   hold it instead consumes only the expired `initializing`/`failed` lease and
   immutable before record. Its full preflight accepts each CronJob only in
   the exact held or exact prior state, so a crash or later CAS failure can be
   retried without overwriting drift. Each CAS restore puts back the recorded
   prior suspend field, including absence versus explicit false, and records
   returned resourceVersions. Any UID/spec/resourceVersion race stops for
   review; the tool never assumes both prior states were false.

8. Pin the verified split-manifest SHA-256, run the full generator/check, and
   review the consumer-authorization commit. Run and capture each model as
   described in the evidence README. Assemble six crop OOF results for the R5
   decision; this stage creates no gate, sidecar, queue entry, or rung-5 Job.

The advisory dataset lock is separate from the external Kubernetes freeze.
Storage prep creates the empty, unaliased lock inode as `root:2000` mode
`0660` inside a root-owned `0750` directory and records its exact identity.
PLAN, APPLY, split, `backfill_vpp.py`, `strip_unstamped_vpp.py`, and in-place
`fill_tiles_l2a.py` share that inode from initial read through terminal
publication. Rollout Jobs require the file to exist and match UID/GID/mode;
they never create or silently settle it. The lock cannot detect arbitrary Pods
or armed controllers and never replaces the watchdog.

A deliberate re-freeze is not a prep replay. First confirm no consumer used
the old split, then use an explicitly reviewed UID-2000 recovery operation to
restore owner write on `crop_split/`, remove the four root artifacts and the
consumer projection, and restart at a new payload A/image/B. Routine prep
will preserve a `0550` root and cannot silently reopen the old freeze.

Two generalization holes found while building the stage remain regression
tested in `tests/test_crop_distill_wiring.py`:

- `extract_plot_features.py` previously handed `from_records` a hard-coded
  NFI column list, dropping `point_id` and generic truth columns;
- `nfi_head_cv.py` previously routed crop classes through the NFI-only
  remapping, which could turn crop accuracy into a meaningless 1.0.

Generic truth now remains in its own label space, and NFI-specific behavior
remains NFI-only.
