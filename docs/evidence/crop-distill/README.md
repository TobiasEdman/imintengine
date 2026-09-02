# Crop-distill evidence archive

This directory is the durable, off-PVC evidence target for the LUCAS
crop-distill rollout. Kubernetes Pods are removed by the Job TTL, and the log
reaper writes back to the same RWX trust domain as the workload. Neither is a
durable review artifact by itself.

No live-run evidence exists yet. This README is an operating procedure, not
proof that a Job ran.

## Bundle contract

Capture one create-only directory per Pod UID:

```text
docs/evidence/crop-distill/<pod-uid>/
  completion.json       # exact canonical bytes decoded from the Pod marker
  completion.sha256     # lowercase SHA-256 of completion.json plus newline
  capture.json          # deterministic Pod, Job, container, and image binding
```

The capture helper accepts a Kubernetes Pod JSON document and the target
container's complete Pod log. It requires exactly one terminal line with this
contract:

```text
CROP_DISTILL_TERMINAL_EVIDENCE_V1 <lowercase-sha256> <strict-base64-canonical-json>
```

It then fails closed unless all of the following agree:

- the marker hash, decoded canonical bytes, and completed terminal record;
- the requested namespace, Pod name, Job name, Pod UID, controller reference,
  and Kubernetes Job label;
- the record kind and its one permitted container name (`crop-distill` or
  `split`);
- the Pod's digest-pinned `.spec.containers[].image`, the runtime-observed
  `.status.containerStatuses[].imageID`, and the exact image digest in the
  completion record; and
- Pod phase `Succeeded` and target-container exit code `0`.

Common ICE/containerd forms such as `containerd://sha256:<digest>` and
`docker-pullable://<repository>@sha256:<digest>` are normalized before the
digest comparison. Mutable tags, missing entries, duplicate target entries,
and conflicting identities are refused. The output has no local path or wall
clock field, so identical evidence produces byte-identical `capture.json`.

## Exact live capture

Run this after the Job has succeeded and before its Pod expires. First choose
the single Pod returned for the Job; do not capture if the selector returns
zero or multiple Pods.

For a crop consumer:

```bash
NAMESPACE=prithvi-training-default
JOB=ladder-crop-distill-croma
CONTAINER=crop-distill

kubectl -n "$NAMESPACE" get pods \
  -l "batch.kubernetes.io/job-name=$JOB" \
  -o custom-columns='NAME:.metadata.name,UID:.metadata.uid,PHASE:.status.phase'

POD=ladder-crop-distill-croma-REPLACE_WITH_ACTUAL_SUFFIX
SCRATCH=$(mktemp -d)
chmod 700 "$SCRATCH"
kubectl -n "$NAMESPACE" get pod "$POD" -o json >"$SCRATCH/pod.json"
kubectl -n "$NAMESPACE" logs "$POD" -c "$CONTAINER" >"$SCRATCH/pod.log"
POD_UID=$(jq -r '.metadata.uid' "$SCRATCH/pod.json")

python scripts/capture_crop_distill_evidence.py capture \
  --pod-json "$SCRATCH/pod.json" \
  --pod-log "$SCRATCH/pod.log" \
  --container "$CONTAINER" \
  --expected-namespace "$NAMESPACE" \
  --expected-pod "$POD" \
  --expected-job "$JOB" \
  --out-dir "docs/evidence/crop-distill/$POD_UID"

python scripts/capture_crop_distill_evidence.py verify \
  --bundle-dir "docs/evidence/crop-distill/$POD_UID"
```

Repeat that procedure for each of the six crop Jobs. Change `JOB` to the exact
model Job name and keep `CONTAINER=crop-distill`.

For the split producer, use:

```bash
NAMESPACE=prithvi-training-default
JOB=ladder-lucas-crop-split
CONTAINER=split

kubectl -n "$NAMESPACE" get pods \
  -l "batch.kubernetes.io/job-name=$JOB" \
  -o custom-columns='NAME:.metadata.name,UID:.metadata.uid,PHASE:.status.phase'

POD=ladder-lucas-crop-split-REPLACE_WITH_ACTUAL_SUFFIX
SCRATCH=$(mktemp -d)
chmod 700 "$SCRATCH"
kubectl -n "$NAMESPACE" get pod "$POD" -o json >"$SCRATCH/pod.json"
kubectl -n "$NAMESPACE" logs "$POD" -c "$CONTAINER" >"$SCRATCH/pod.log"
POD_UID=$(jq -r '.metadata.uid' "$SCRATCH/pod.json")

python scripts/capture_crop_distill_evidence.py capture \
  --pod-json "$SCRATCH/pod.json" \
  --pod-log "$SCRATCH/pod.log" \
  --container "$CONTAINER" \
  --expected-namespace "$NAMESPACE" \
  --expected-pod "$POD" \
  --expected-job "$JOB" \
  --out-dir "docs/evidence/crop-distill/$POD_UID"

python scripts/capture_crop_distill_evidence.py verify \
  --bundle-dir "docs/evidence/crop-distill/$POD_UID"
```

Keep the scratch directory until the bundle has passed offline verification
and been reviewed. Delete it only after the committed bundle preserves the
evidence.

The split producer's bundle belongs in consumer-authorization commit C. Each
crop consumer bundle belongs in the evidence-only commit after all six runs
finish and before the PR is accepted.

## Security boundary

This archive records what the Kubernetes API and container runtime reported;
it does **not** enforce image admission. In particular, the helper is not an
admission webhook and does not prevent a workload from being submitted with a
different image. Digest-pinned manifests, signature verification, source SHA,
split-manifest SHA, and checkpoint identities remain the authorization
boundary. A missing bundle or any capture/verification mismatch blocks rollout
acceptance.
