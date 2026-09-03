# LUCAS crop-distill runtime

This image is the immutable runtime for the frozen LUCAS split and the six
crop-distillation columns. It contains no mutable source checkout and performs
no package or model download at runtime.

## Reconstructed dependency identity

The 2026-08-31 TerraMind training environment did not log its installed
TerraTorch version. It did record Python 3.11 and an installation completing at
`2026-08-31T07:47:18Z`. Resolving against package metadata available at that
instant selects `terratorch==1.2.11` and `torchgeo==0.8.1`: TerraTorch 1.2.12
and newer require TorchGeo 0.9.0, which requires Python 3.12. Version 1.2.11 is
therefore resolver-determined from timestamp, ABI, and official metadata; it is
**not** a directly observed package log from the run.

TerraTorch 1.2.11 requires NumPy >=2.2, so it cannot share the NumPy 1.26
scoring ABI. The image uses a digest-pinned Python 3.11 base and two
independently hash-locked environments:

| Purpose | Interpreter | Key ABI |
| --- | --- | --- |
| model/extraction | `/opt/venvs/model/bin/python` | TerraTorch 1.2.11, TorchGeo 0.8.1, NumPy 2.2.6, Torch 2.5.1+cu121 |
| split/scoring | `/opt/venvs/scoring/bin/python` | NumPy 1.26.4, pandas 2.2.2, PyArrow 17.0.0, scikit-learn 1.5.1 |
| provenance only | `/usr/local/bin/python` | Python standard library |

`requirements-model.lock` and `requirements-scoring.lock` include exact
versions and hashes for every transitive dependency. Torch and TorchVision use
direct, hash-pinned official CUDA 12.1 CPython 3.11 wheels. The model lock was
resolved with `--exclude-newer 2026-08-31T07:47:18Z`; that resolver selected
NumPy 2.2.6 under all transitive constraints. Clay is installed
with `--no-deps` from its verified source tree; CROMA is not pip-installable and
is exposed only through `PYTHONPATH`.

Torch 2.5.1 preserves the observed training ABI but its `weights_only`
unpickler is affected by the upstream
[2025](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)
and
[2026](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
code-execution advisories; the current patched line starts at 2.10. This
evidence run therefore admits only the six
first-party checkpoints whose size/SHA-256 were anchored during review as
trust-on-first-use. The digests prove no change after anchoring, not that bytes
were safe beforehand, and no separate pickle-scan result is claimed. Unknown
checkpoints require reviewed non-Pickle conversion or a separately validated
PyTorch >=2.10 stack. The private snapshot and Pod isolation reduce race and
impact; they are not a malicious-pickle sandbox.

The slim base omits two shared libraries needed by those wheels. The
Dockerfile therefore installs the exact amd64 Debian artifacts
`libexpat1=2.8.3-1~deb13u1` and `libgomp1=14.2.0-19` from fixed package URLs
after verifying their repository SHA-256 values. It never resolves them from a
mutable apt index.

## Source identity

`SOURCE_GIT_SHA` is mandatory and must be one full lowercase 40-hex commit.
The Dockerfile fetches ImintEngine from that exact GitHub codeload URL and
copies only `config/`, tracked `data/distill/distill_split.json`, `imint/`, and
`scripts/`. The tracked split is the historical 53-tile holdout authority; no
mutable LUCAS parquet is baked. The build also verifies and seals:

- CROMA `59505a6bcadbf36ba20767270154bf9f3067c5e7`, archive SHA-256
  `939d0918991ad7604bbb0a782df2674b8e30ade6edc061bcad6ab486e6f94001`.
- Clay `f14e698f3c237cabf8d28dec669a362d66625381`, archive SHA-256
  `0b908ea11d5348736c26512f695221a304883ec88bac68f66822ca07bf435d64`.

Every selected source file, both dependency locks, both installed-package
freezes, all three Python versions, and both upstream source trees are bound
into `/opt/provenance/runtime.json`.

## Build and publish

Build a committed, clean, and remotely addressable source SHA locally:

```sh
docker/ladder-crop-distill/build.sh "$(git rev-parse HEAD)"
```

The script emits only `:sha-<full-source-SHA>` and never `:latest`. Set
`CROP_DISTILL_IMAGE_REPOSITORY` to change the repository. It refuses a SHA that
is not the checked-out `HEAD` and refuses all tracked or untracked worktree
changes, preventing a source/archive identity from being paired with a
different local build context. The canonical publish wrapper is
`.github/workflows/build-pipeline-images.yml`: it builds linux/amd64, publishes
to GHCR, records the manifest digest, emits an SBOM and SLSA provenance, and
signs the digest through GitHub OIDC. Verify a published image:

```sh
PAYLOAD_A_SHA=<40-character-payload-A-sha>
WORKFLOW_REF='TobiasEdman/ImintEngine/.github/workflows/build-pipeline-images.yml@refs/heads/<payload-A-branch>'
cosign verify ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:<digest> \
  --certificate-identity "https://github.com/${WORKFLOW_REF}" \
  --certificate-github-workflow-sha "$PAYLOAD_A_SHA" \
  --certificate-github-workflow-repository TobiasEdman/ImintEngine \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

`WORKFLOW_REF` must match the successful build's
`crop-distill-image.txt` evidence and intentionally includes the actual branch
ref. A later rebuild on `main` therefore has a different expected identity and
source SHA; never replace this exact check with a repository-prefix regexp.

Deployment has five reviewed commits. Payload A builds the image. Runtime-pin
B pins A plus its signed image digest; `--crop-bootstrap-only` then generates
only deny-egress, storage-prep, and the read-only source-access PLAN. Plan
authority C pins the reviewed PLAN Pod UID and SHA-256 and
`--crop-apply-only` generates only the metadata repair. Completion authority D
pins the reviewed APPLY Pod UID and completion SHA-256, after which
`--crop-split-only` generates split attempt 3. Consumer authorization E pins
the verified split-manifest SHA-256 and generates the six crop Jobs. Each
downstream renderer fails before its upstream authority is nonzero and pinned.

## Runtime verification

The Docker build runs with Hugging Face and Transformers offline. It imports
the exact CROMA, Clay, TerraTorch, TorchGeo, Torch, and scoring packages;
constructs CROMA, TerraMind, and Clay through each `pretrained=False`
checkpoint-reconstruction path; checks both environments with `pip check`; and
compiles/imports the scripts used by the Jobs.

Each Job repeats the fail-closed preflight before work:

```sh
/usr/local/bin/python /opt/imintengine/scripts/crop_distill_provenance.py \
  verify-runtime \
  --source-git-sha "$CROP_DISTILL_SOURCE_GIT_SHA" \
  --image-ref "$CROP_DISTILL_IMAGE" \
  --runtime-manifest /opt/provenance/runtime.json
```

The generated Kubernetes Jobs from `scripts/gen_ladder_manifests.py` are the
run wrappers; duplicating their protocol in a `run.sh` would create two sources
of truth. Model extraction must invoke the model interpreter, while split and
head scoring must invoke the scoring interpreter listed above.

Before PLAN, a no-argument storage-prep entrypoint authenticates the baked
runtime against A and the reviewed image digest, then prepares exactly 20
baked targets. `/cephfs/distill/crop_split` is UID/GID 2000 mode `03770` until
frozen to `0550`; `/cephfs/distill/crop_heads` and
`/cephfs/ops/crop-distill` are root:GID-2000 mode-`0750` parents; and
`/cephfs/ops/crop-distill/split` is the UID/GID-2000 mode-`0750` split-record
leaf. The root:GID-2000 mode-`0750` source-access root and its `plan`, `apply`,
and `locks` leaves hold immutable per-Pod evidence and the cooperative lock.
Each of the six model UIDs 2001–2006 owns a mode-`0750` head leaf at
`/cephfs/distill/crop_heads/<model>_r2_crop_runs` and evidence leaf at
`/cephfs/ops/crop-distill/<model>`, all with GID 2000. The root-owned parents
prevent models from creating or replacing sibling leaves.

Storage prep runs UID 0:GID 2000 with exactly `CHOWN,FOWNER`; it parses no
data and preserves a completed split's `0550` lock. PLAN runs UID 0 read-only
with all capabilities dropped. APPLY runs UID 0:GID 2000 with exactly
`CHOWN,FOWNER`, opens dataset files without a data-write handle, and may
change only the PLAN-authorized metadata. Split and crop consumers remain
non-root and drop all capabilities. Every role has a read-only rootfs, no
service-account token, and no runtime egress. The empty-egress
`NetworkPolicy` selects storage-prep, PLAN, APPLY, split, and crop Pods.

The model→UID map is baked into the image. Each split/crop entrypoint validates
its effective UID/GID before PVC work, and terminal evidence records the
observed process identity. The Jobs reference the existing `ghcr-push` secret
only through `imagePullSecrets`; it is not container-mounted. Its actual
registry scope is not independently proven pull-only, so a dedicated
`ghcr-pull` secret remains recommended cluster hardening.

## Required terminal evidence

Every caught split/crop Job failure with a validated `POD_UID`, and every
completed split/crop run, publishes exactly one immutable terminal record at
its dedicated backing path:

```text
split: /cephfs/ops/crop-distill/split/<POD_UID>/completion.json
crop:  /cephfs/ops/crop-distill/<model>/<POD_UID>/completion.json
```

The storage-prep Job does not publish this record. An abrupt `SIGKILL`, a
pre-`POD_UID` failure, or a provenance-publication failure can also leave no
record; absence is therefore fail-closed and never completion evidence.
`crop_distill_provenance.py finalize` re-runs runtime verification for a
completed run and atomically creates the record. A retry is accepted only when
the complete JSON bytes are identical; a mismatched record is never
overwritten. Completed crop
records require a verified runtime, the Git-pinned frozen-split digest, exact
checkpoint identity, and `features` plus `oof` outputs. The checkpoint is
copied and hashed from one shared descriptor into an anonymous private
snapshot, and only that snapshot is deserialized; finalization binds the
expected identity without reopening the shared path. Output hashes are
computed during descriptor-based publication and verified again before the
record is accepted. Private intermediates stay under mode-`0700` `/work` on a
disk-backed (not memory-backed) `emptyDir`; the anonymous checkpoint snapshot
is mode `0600`. The Pod requests 4 GiB of ephemeral storage and caps `/work` at
8 GiB for the largest 2.7 GB checkpoint. Final
output files move to a model-owned mode-`0750` publication directory,
include the Pod UID in their names, and are mode `0444`. Per-Pod terminal-
record directories are likewise mode `0750`, with a mode-`0444` record.
Completed split records require and hash the distill index, validator-only
holdout index, split JSON, and manifest. Crop workers bind the holdout hash
declared in the manifest but deliberately never stat, open, or hash the
validator-only file.

A caught failed split/crop run never dereferences split, checkpoint, output, or
holdout paths while publishing its record. `completion.json` instead contains
`runtime.verification: not-dereferenced`, bounded raw source/image/runtime-
manifest claims, and the terminal failure stage plus exit code. Such a record
is diagnostic failure evidence and can never satisfy completed-state checks.

The canonical completed record and its digest are also printed to Pod stdout,
providing a live audit surface outside the writable PVC. Before the Job TTL or
deletion, those canonical bytes and digest must match the PVC record and be
committed to `docs/evidence/crop-distill/<pod-uid>/`. A reaper archive on the
same RWX PVC is not independent evidence. Unix UID separation blocks ordinary
cross-column mutation, but not a cluster administrator or a hostile workload
explicitly given the same model UID. A missing durable capture or mismatch is
fail-closed.
