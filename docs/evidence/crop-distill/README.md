# Crop-distill evidence archive

This directory is the status index and runbook for the LUCAS crop-distill
rollout. It is not evidence that a live Job succeeded. Successful
storage-prep and source-access PLAN/APPLY bundles have been captured for the
reviewed image and strictly verified against their current Git pins. Split
retry and the six crop-consumer bundles remain pending.

Kubernetes Pod objects disappear after the Job TTL. Workload records on the
RWX PVC share the workload trust domain. Acceptance therefore requires an
external, create-only bundle captured while the exact Pod UID still exists,
followed by offline verification against the reviewed Git pins.

## Rollout status

| event | result | identity |
|---|---|---|
| split attempt 2 | failed closed: UID/GID 2000 could not read 1,966 of 2,074 post-window candidates; no split frozen | Pod UID `b9fbbdae-1a11-4160-9b5b-3762dee95ddf`; failure-record SHA-256 `a13b74a8df5ed5600f9d2b7961c9944c44886cb3c094dbefb4dc22fce4bf6aac` |
| `DAC_READ_SEARCH` canary | rejected by PodSecurity admission; no Pod or data mutation | admission result only |
| `DAC_OVERRIDE` alternative | rejected in review as an unnecessarily broad bypass | design decision only |
| reviewed-image storage prep | succeeded; strict external bundle verified | Pod UID `341b211f-9ff1-4a33-83ea-b01eeae1ec50`; completion SHA-256 `40aba25cbf58c38c190700515c26562462227e387260e9a53c4b8ace1706689a` |
| source-access PLAN | succeeded; 2,074 candidates, 1,966 repairs, 108 no-ops; strict external bundle verified | Pod UID `fe949aaa-8191-4c03-8989-2b3516a2e2a7`; PLAN SHA-256 `d8cdd10b8e9e2668aadafec1aef1a87f5067c2f414e9453e4ce33c196bc04e98` |
| source-access APPLY | succeeded; 1,966 repaired and 108 unchanged; SHA-256, size, device/inode, link count, and mtime unchanged for all 2,074 files; strict external bundle verified | Pod UID `2d193bf7-2217-4878-b2e5-f1367e7137b3`; completion SHA-256 `4cbce71f3224c4e6170b2113c774104ab96a6ef14b4e67d51ab0ddf03354aa97` |
| split attempt 3 | failed closed before source read/write: the manifest selected provenance-only base Python, which has no NumPy | Job UID `366cb6ef-1bce-4bc0-8896-4dba26312f7a`; Pod UID `6feb4ccd-04df-47f4-a785-496a0f1e076a`; no completion record |
| corrected split retry | failed closed at the reviewed 3,600-second Pod deadline while repeated full-cohort verification was still making progress; no completion record | Job UID `3bdf0294-7264-4900-a0e5-bddc01e3e68d`; Pod UID `447d167f-85b6-4b0d-8e5a-bc66857760ba`; `DeadlineExceeded`, exit 137, zero restarts |
| split retry 2 and crop consumers | not run | blocked on review of the 7,200-second split deadline and explicit cleanup of the uncommitted partial freeze |

Attempt 2 used source
`ba6a99aa8662d6af9ba6fb3d08398d505ef8c483` and an older image whose digest
ended `a487ead664aa...`. That image does not contain the reviewed PLAN/APPLY and
capture controls and must not be reused. The table above is a ledger entry,
not a substitute for its external failure evidence.

## Canonical bundle

`scripts/capture_crop_distill_evidence.py` supports five strict evidence kinds:

| kind | marker | immutable workload record |
|---|---|---|
| `storage-prep` | `CROP_DISTILL_STORAGE_PREP_COMPLETION_V1` | `/cephfs/ops/crop-distill/storage-prep/<pod-uid>/completion.json` |
| `source-access-plan` | `CROP_SOURCE_ACCESS_PLAN_V1` | `/cephfs/ops/crop-distill/source-access/plan/<pod-uid>/plan.json` |
| `source-access-apply` | `CROP_SOURCE_ACCESS_COMPLETION_V1` | `/cephfs/ops/crop-distill/source-access/apply/<pod-uid>/completion.json` |
| `split` | `CROP_DISTILL_TERMINAL_EVIDENCE_V1` | marker payload written under the split record root |
| `crop` | `CROP_DISTILL_TERMINAL_EVIDENCE_V1` | marker payload written under the model record root |

Each new bundle contains exactly six files:

```text
<absolute-off-PVC-bundle>/
  completion.json | plan.json  # exact canonical workload-record bytes
  completion.sha256 | plan.sha256
  marker.txt                    # exact accepted stdout marker line
  pod.json                      # exact Kubernetes API Pod observation
  pod.sha256
  capture.json                  # canonical external binding and normalized Pod
```

`capture.json` hashes both raw `pod.json` and the canonical, per-kind normalized
Pod observation. It also hashes the marker and record, records the exact Pod
UID, and records the capture process's kernel effective UID/GID. The capture
process must be outside the workload and must not have the workload UID/GID.
This identity model distinguishes the observer from the workload; it is not a
cryptographic operator signature or proof of a human identity.

The bundle deliberately has no capture timestamp or local output path, so the
same inputs produce the same canonical `capture.json`. Bundle creation is
create-only and refuses an existing output directory.

## Two-phase PLAN/APPLY authority

The output of a workload cannot contain a hash of Pod JSON fetched after that
workload terminated. The external bundle supplies the non-circular binding:
it contains and hashes both the exact workload marker/record bytes and the API
Pod observation, including the Pod UID and runtime `imageID`.

PLAN and APPLY each require two distinct phases:

1. Live capture validates the current, pre-existing Git authority and derives
   the new output identity from the marker, PVC record, and API Pod. PLAN uses
   the Git-pinned source/image/index. APPLY additionally requires the
   Git-pinned PLAN SHA-256 and PLAN Pod UID.
2. Review pins the newly captured record SHA-256 and Pod UID in
   `scripts/gen_ladder_manifests.py`. The normal `verify` subcommand then
   requires the restricted bundle to equal those current Git pins. Only this
   post-pin offline verification authorizes a downstream phase.

Consequently, the `capture` subcommand performs a complete semantic check but
does not pretend that its newly observed output was already Git-authorized.
Commit C must pin the exact verified PLAN record SHA-256 and PLAN Pod UID before
APPLY is rendered. Commit D must pin the exact verified APPLY completion
SHA-256 and completion Pod UID before split is rendered. The split receives
and checks both Git-pinned SHA-256/Pod UID pairs before it reads the source
cohort. Its terminal record must then contain exactly this immutable object:

```json
{
  "source_access": {
    "plan": {"sha256": "<git-pinned-plan-sha256>", "pod_uid": "<plan-pod-uid>"},
    "completion": {"sha256": "<git-pinned-apply-sha256>", "pod_uid": "<apply-pod-uid>"}
  }
}
```

Capture metadata never authorizes an alternate anchor. Offline verification
re-derives expectations from the current Git constants and the immutable
workload record.

## Live capture procedure

Run capture before the Pod TTL removes the object. Select exactly one Pod for
the Job and retain the API response and log in a mode-0700 scratch directory.
Do not continue if the selector returns zero or multiple Pods.

```bash
NAMESPACE=prithvi-training-default
KIND=source-access-plan
JOB=ladder-crop-source-access-plan
CONTAINER=source-access-plan
POD=REPLACE_WITH_EXACT_POD_NAME
BUNDLE_ROOT=/absolute/path/to/access-controlled/crop-distill-evidence

SCRATCH=$(mktemp -d)
chmod 700 "$SCRATCH"
kubectl -n "$NAMESPACE" get pod "$POD" -o json >"$SCRATCH/pod.json"
kubectl -n "$NAMESPACE" logs "$POD" -c "$CONTAINER" >"$SCRATCH/pod.log"
POD_UID=$(jq -er '.metadata.uid' "$SCRATCH/pod.json")
```

For `storage-prep`, `source-access-plan`, and `source-access-apply`, copy the
exact per-Pod record through the approved read-only operator path to
`$SCRATCH/record.json`. Do not reconstruct it from stdout. Then capture:

```bash
python scripts/capture_crop_distill_evidence.py capture \
  --evidence-kind "$KIND" \
  --pod-json "$SCRATCH/pod.json" \
  --pod-log "$SCRATCH/pod.log" \
  --record-file "$SCRATCH/record.json" \
  --container "$CONTAINER" \
  --expected-namespace "$NAMESPACE" \
  --expected-pod "$POD" \
  --expected-job "$JOB" \
  --out-dir "$BUNDLE_ROOT/$POD_UID"
```

For `split` and `crop`, omit `--record-file`; their accepted terminal marker
already carries the exact bytes written to the immutable per-Pod record. Use
`KIND=split`, `JOB=ladder-lucas-crop-split`, `CONTAINER=split` for the split.
For a crop consumer, use `KIND=crop`, the exact
`JOB=ladder-crop-distill-<model>`, and `CONTAINER=crop-distill`.

After the new PLAN or APPLY identity has been reviewed and pinned in Git, run
strict offline verification from that reviewed checkout:

```bash
python scripts/capture_crop_distill_evidence.py verify \
  --bundle-dir "$BUNDLE_ROOT/$POD_UID"
```

Keep the scratch inputs until the restricted bundle has been copied, verified,
backed up, and reviewed. Never delete the only copy of a workload record or
bundle.

## Machine-verifiable acceptance boundary

Capture and offline verification both reject a missing field, extra reviewed
surface, or conflicting identity. They validate:

- exactly one Pod container and one matching container status, with no init or
  ephemeral containers/statuses;
- the exact Job owner, full reviewed label set, namespace, Pod name, Pod UID,
  non-empty `nodeName`, successful Pod phase, zero restarts, exit code zero,
  immutable spec/status image, and runtime `imageID` digest;
- exact command, arguments, complete literal environment with no extra entry,
  the sole `POD_UID` Downward-API reference, and no `envFrom`;
- `automountServiceAccountToken: false`, the exact default service account,
  no host namespaces, no interactive input, and only the reviewed kubelet-only
  image-pull secret;
- exact PVC/emptyDir/ConfigMap volumes, volume mounts, subPaths, read-only
  flags, resources, and no volume devices; `hostPath`, Secret, projected
  service-account-token, and extra PVC sources are refused;
- exact Pod/container UID/GID, `RuntimeDefault` seccomp, read-only root
  filesystem, no privilege escalation, and `drop: [ALL]`;
- the exact Git-pinned spec image, an immutable runtime-reported status image,
  and an `imageID` digest equal to the signed manifest digest; the status image
  may be a CRI-local config digest and is recorded without conflating the two;
- storage prep as root with only `CHOWN,FOWNER` and only its exact RW
  `/cephfs/distill` and `/cephfs/ops` subPaths; its completion also binds the
  precreated, empty, unaliased backing lock
  `/cephfs/ops/crop-distill/source-access/locks/dataset.lock` as
  `root:2000 0660` while the lock directory remains `root:2000 0750`;
- source-access PLAN as root with no added capabilities and the dataset subPath
  read-only;
- source-access APPLY as root with only `CHOWN,FOWNER`, the exact
  `unified_v2_512` dataset subPath RW, and its reviewed PLAN record read-only;
- split/crop as their protocol UID/GID with no added capabilities, including
  the split's verified source-access record and freeze/lock mounts. PLAN,
  APPLY, and split see that same backing inode only through the narrow runtime
  projection `/cephfs/source-access-lock/dataset.lock`; and
- marker SHA-256/base64/canonical JSON, exact marker-to-PVC byte equality for
  prep/PLAN/APPLY, the canonical stdout-marker line ending, record
  schema/cardinality/runtime identity, raw Pod SHA-256, normalized Pod
  SHA-256, and current Git authority.

PLAN requires the pinned 468,614-byte source index with SHA-256
`e3bb505c4de469d0436e0e91de27327f063083199c8ddc781ec6eaf7d42e9a41`,
3,587 crop rows, 2,074 sorted unique candidates, 1,966 reviewed repairs, and
108 no-ops. APPLY verifies all 2,074 before/after records: content hash, size,
device/inode, and mtime remain unchanged; repaired files finish as
`root:2000 0640`; and the action counts match the PLAN.

## Restricted evidence and receipts

The complete PLAN and APPLY records enumerate all 2,074 source files and their
filesystem identities. The immutable workload records remain on their narrow
PVC record subPaths; external copies, full bundles, and raw scratch inputs are
restricted operational evidence. Store those external artifacts outside Git
and outside `/cephfs` in the access-controlled evidence location, with an
independent backup. Do not commit the inventory to this directory, paste it
into a PR, or publish it in CI logs.

After restricted review, a small receipt may be committed here. The receipt
may contain the canonical `capture.json` (subject to node-name disclosure
review) or a separately reviewed projection containing only the schema,
evidence kind, record SHA-256, Pod UID, raw/normalized Pod hashes, and applicable
Git pins. A receipt is an index to the restricted bundle, not independently
verifiable evidence; reviewers with access must run the strict verifier on the
full bundle.

## Security limitation

This control records and verifies what the Kubernetes API, container runtime,
workload marker, and PVC record reported. It is not an admission webhook, does
not sign the evidence, and does not itself enforce the deny-egress
NetworkPolicy or external freeze watchdog. Digest-pinned manifests, reviewed
Git constants, cluster admission, the shared cooperating-producer lock, the
external watchdog lease, and restricted evidence custody remain separate
parts of the authorization boundary. Any missing bundle, pin, or verification
mismatch blocks rollout acceptance.
