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

**Operational status, 2026-09-02:** the reproducible protocol is implemented
for review, but no live LUCAS split or crop-distill Job has been submitted.
Nothing below is evidence that a cluster run succeeded.

**R5 = R4 + LUCAS crop type** [user-stated 2026-08-31], and distillability
comes before any retraining. Before a rung 5 exists, every column therefore
gets a crop-distillability number using LUCAS crop points (unified classes
11–21). The experiment deliberately creates neither a gate nor a rung-5
manifest; the six crop-OOF results are evidence for the later decision.

### Immutable three-commit bootstrap

The Jobs must not fetch a mutable branch or install dependencies at startup.
The stage is bootstrapped in this fixed order:

1. **Payload commit A** contains every byte a storage-prep, split, or crop Pod
   executes: loaders, split/extraction/scoring code, provenance validation,
   the image recipe, exact upstream source identities, and both dependency
   locks. It removes the old runnable crop/split YAMLs; no Job can start from
   an unpinned bootstrap state.
2. CI builds that exact commit into
   `ghcr.io/tobiasedman/imint-ladder-crop-distill`, exercises the offline
   runtime and least-privilege storage-prep smoke tests on the exact pushed
   digest, signs and exactly verifies the image, and only then publishes its
   immutable `@sha256:` digest evidence.
3. **Producer commit B** sets `CROP_DISTILL_SOURCE_GIT_SHA` to commit A and
   `CROP_DISTILL_IMAGE` to that signed digest. With the split-manifest digest
   still at its zero sentinel, `--crop-bootstrap-only` generates the deny-
   egress policy and exactly the storage-prep and split-producer Jobs. The six
   crop-consumer YAMLs must remain absent; bootstrap refuses if any stale
   consumer manifest exists. Review commit B before either producer Job is
   submitted.
4. Run storage prep and exactly one split freeze. Independently verify the
   completed record and read the frozen manifest's SHA-256 from that evidence,
   never from a value supplied by a crop consumer.
5. **Consumer-authorization commit C** pins that externally observed digest
   as `CROP_DISTILL_SPLIT_MANIFEST_SHA256` and generates the six crop Jobs.
   Full generation rejects a zero, malformed, or abbreviated source, image,
   or split identity. Review commit C before any crop Job is submitted.

`CROP_DISTILL_SOURCE_GIT_SHA` is owned by this crop stage. In particular,
PR #37's inference-matrix payload has its own source constant and must never
bump the crop anchor. Ordinary generator or documentation edits also leave
the crop anchor untouched.

This is a one-way freeze. If any Pod-executed crop payload needs a bug fix
after commit A, the recovery procedure restarts all three reviewed phases:
review a new complete payload A, build and verify its image, authorize and run
one new split from producer B, then pin that new manifest in consumer C. Do
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
prepares exactly 16 directories with baked ownership and modes:

- `/cephfs/distill/crop_split` is UID/GID `2000:2000`, mode `03770` until
  the completed freeze locks it to `0550`;
- `/cephfs/distill/crop_heads` and `/cephfs/ops/crop-distill` are root-owned,
  GID 2000, mode `0750` parents, so an unprivileged workload cannot create or
  replace another model's leaf;
- `/cephfs/ops/crop-distill/split` is the UID/GID `2000:2000`, mode `0750`
  split-evidence leaf; and
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
running root-prep, split, and crop containers have no network path.

The Jobs reference the existing `ghcr-push` secret only through
`imagePullSecrets`; it is not mounted in a container. The credential's actual
registry scope has not been independently proven pull-only. Replacing it with
a dedicated `ghcr-pull` credential is recommended cluster hardening, but
renaming it in a manifest before that secret exists would make the reviewed
rollout unpullable.

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

Use this order; every stop condition is fail-closed and crop Jobs must never
run in parallel with the freeze:

1. Independently review payload commit A. Build/sign it, verify the image
   digest and runtime smoke, then put exactly A's full SHA and that image
   digest into producer commit B. Cosign verification must bind the exact
   workflow path plus the actual build ref from `crop-distill-image.txt`, the
   full payload-A SHA, the repository claim, and GitHub's OIDC issuer. A
   feature-branch build and a later `main` rebuild intentionally have different
   identities; never loosen this to a repository-prefix match. Generate/check
   the anchor-independent manifests while A is gated, then only the producer
   phase after A's source and image identities are pinned:

   ```bash
   python scripts/gen_ladder_manifests.py --non-crop-only --check
   python scripts/gen_ladder_manifests.py --crop-bootstrap-only
   python scripts/gen_ladder_manifests.py --crop-bootstrap-only --check
   ```

2. Review B. Server-side dry-run and create the storage-prep Job once. Wait
   for completion and verify stdout reports exactly all 16 baked targets: one
   UID/GID 2000 mode-`03770` split root; two root:GID-2000 mode-`0750`
   parents; one UID/GID 2000 mode-`0750` split-record leaf; and six pairs of
   UID-specific mode-`0750` head/record leaves. Stop on any extra or missing
   path, owner, group, or mode.

   ```bash
   kubectl -n prithvi-training-default apply --dry-run=server \
     -f k8s/ladder/crop-distill-deny-egress.yaml
   kubectl -n prithvi-training-default apply \
     -f k8s/ladder/crop-distill-deny-egress.yaml
   kubectl -n prithvi-training-default apply --dry-run=server \
     -f k8s/ladder/crop-distill-storage-prep-job.yaml
   kubectl -n prithvi-training-default create \
     -f k8s/ladder/crop-distill-storage-prep-job.yaml
   ```

3. Server-side dry-run the split manifest, then create it once:

   ```bash
   kubectl -n prithvi-training-default apply --dry-run=server \
     -f k8s/ladder/lucas-crop-split-job.yaml
   kubectl -n prithvi-training-default create \
     -f k8s/ladder/lucas-crop-split-job.yaml
   ```

4. Wait for `job/ladder-lucas-crop-split` to complete. Verify all four
   artifacts, their self-contained cross-check, stdout, and the write-once
   completed record against commit A and the image digest. On the first real
   freeze, also compare its normalized observed-71-key digest with the
   independently observed pre-freeze value
   `7808d10432ae8ddfc40623c03e33e4eeb1b9cc8cbc94afc5b11184e9632ace86`
   (sorted compact-JSONL `(tile_name, int(point_id))` records); investigate
   any change. Before the Pod expires, compare the stdout record with the PVC
   bytes and include the verified off-cluster bundle in commit C. Stop on any
   mismatch or missing capture.
5. Put the verified manifest SHA-256 into
   `CROP_DISTILL_SPLIT_MANIFEST_SHA256`, run the full generator and `--check`,
   and independently review consumer-authorization commit C. Before C, crop
   rendering is expected to refuse; that is the authorization boundary.
6. For each model whose exact rung-2 checkpoint is present, server-side
   dry-run and create its generated
   `k8s/ladder/crop-distill-<model>-job.yaml`. Do not replace a prior Job or
   reuse its Pod/output identity.
7. Accept a column only after the Job completed, the immutable record says
   `completed`, and independent verification agrees across the live Pod log
   and PVC record for image, runtime, split, checkpoint, features, and OOF
   identities. Commit the canonical record/digest and capture metadata under
   `docs/evidence/crop-distill/<pod-uid>/` before the Job TTL or any deletion;
   the six durable bundles are required before PR acceptance.
8. Assemble the six crop-OOF results for the R5 decision. This stage creates
   no `_GATE_OK`, dense crop sidecar, queue entry, or rung-5 training Job.

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
