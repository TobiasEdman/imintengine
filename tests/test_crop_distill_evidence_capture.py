"""Durable Kubernetes evidence capture for crop-distill Jobs."""

from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import capture_crop_distill_evidence as evidence

DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
IMAGE = f"ghcr.io/tobiasedman/imintengine/crop-distill@sha256:{DIGEST}"
POD_UID = "782f18e4-0197-48c0-b8af-70461d50b7d8"
POD_NAME = "ladder-crop-distill-croma-k4j7p"
JOB = "ladder-crop-distill-croma"
JOB_UID = "572df214-fe9d-4b81-8d5f-5ca6a5c54190"
NAMESPACE = "prithvi-training-default"


def _record(*, kind: str = "crop") -> dict[str, object]:
    if kind == "crop":
        model = "croma"
        job = JOB
    else:
        model = None
        job = "ladder-lucas-crop-split"
    return {
        "artifacts": {},
        "checkpoint": None,
        "job": job,
        "kind": kind,
        "model": model,
        "pod_uid": POD_UID,
        "process_identity": {"effective_gid": 2000, "effective_uid": 2002},
        "run_id": POD_UID,
        "runtime": {
            "image": {"digest": DIGEST, "ref": IMAGE},
            "verification": "verified",
        },
        "schema": evidence.COMPLETION_SCHEMA,
        "split_manifest": {},
        "terminal": {
            "exit_code": 0,
            "failure_stage": None,
            "status": "completed",
        },
    }


def _marker(record: dict[str, object] | None = None) -> tuple[bytes, bytes]:
    payload = evidence.canonical_json_bytes(record or _record())
    digest = hashlib.sha256(payload).hexdigest()
    marker = (
        f"preflight complete\n{evidence.TERMINAL_EVIDENCE_PREFIX} {digest} "
        f"{base64.b64encode(payload).decode('ascii')}\n"
    ).encode()
    return marker, payload


def _pod(
    *,
    kind: str = "crop",
    spec_image: str = IMAGE,
    image_id: str = f"containerd://sha256:{DIGEST}",
) -> dict[str, object]:
    if kind == "crop":
        container = "crop-distill"
        job = JOB
        pod_name = POD_NAME
    else:
        container = "split"
        job = "ladder-lucas-crop-split"
        pod_name = "ladder-lucas-crop-split-h8v2c"
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "labels": {
                "batch.kubernetes.io/controller-uid": JOB_UID,
                "batch.kubernetes.io/job-name": job,
                "controller-uid": JOB_UID,
                "job-name": job,
            },
            "name": pod_name,
            "namespace": NAMESPACE,
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "blockOwnerDeletion": True,
                    "controller": True,
                    "kind": "Job",
                    "name": job,
                    "uid": JOB_UID,
                }
            ],
            "uid": POD_UID,
        },
        "spec": {
            "containers": [{"image": spec_image, "name": container}],
            "initContainers": [],
        },
        "status": {
            "containerStatuses": [
                {
                    "image": spec_image,
                    "imageID": image_id,
                    "name": container,
                    "ready": False,
                    "restartCount": 0,
                    "state": {
                        "terminated": {
                            "exitCode": 0,
                            "reason": "Completed",
                        }
                    },
                }
            ],
            "phase": "Succeeded",
        },
    }


def _capture_args(
    tmp_path: Path,
    *,
    pod: dict[str, object] | None = None,
    log: bytes | None = None,
    kind: str = "crop",
) -> list[str]:
    pod = pod or _pod(kind=kind)
    if log is None:
        log, _ = _marker(_record(kind=kind))
    pod_json = tmp_path / "pod.json"
    pod_log = tmp_path / "pod.log"
    pod_json.write_text(json.dumps(pod), encoding="utf-8")
    pod_log.write_bytes(log)
    job = JOB if kind == "crop" else "ladder-lucas-crop-split"
    pod_name = POD_NAME if kind == "crop" else "ladder-lucas-crop-split-h8v2c"
    container = "crop-distill" if kind == "crop" else "split"
    return [
        "capture",
        "--pod-json",
        str(pod_json),
        "--pod-log",
        str(pod_log),
        "--container",
        container,
        "--expected-namespace",
        NAMESPACE,
        "--expected-pod",
        pod_name,
        "--expected-job",
        job,
        "--out-dir",
        str(tmp_path / "bundle"),
    ]


@pytest.mark.parametrize(
    "value",
    [
        f"containerd://sha256:{DIGEST}",
        f"docker-pullable://ghcr.io/tobiasedman/imintengine/crop-distill@sha256:{DIGEST}",
        IMAGE,
        f"sha256:{DIGEST}",
    ],
)
def test_normalize_image_digest_accepts_kubernetes_variants(value: str) -> None:
    assert evidence.normalize_image_digest(value, "image") == DIGEST


@pytest.mark.parametrize(
    "value",
    [
        "ghcr.io/tobiasedman/imintengine/crop-distill:latest",
        f"docker://sha256:{DIGEST}",
        f"containerd://sha256:{DIGEST.upper()}",
        f"containerd://sha512:{DIGEST}",
        f"ghcr.io/example/a@sha256:{DIGEST}@sha256:{OTHER_DIGEST}",
        "",
    ],
)
def test_normalize_image_digest_rejects_mutable_or_ambiguous_values(
    value: str,
) -> None:
    with pytest.raises(evidence.EvidenceCaptureError):
        evidence.normalize_image_digest(value, "image")


def test_capture_writes_deterministic_verified_bundle_for_containerd(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _capture_args(tmp_path)

    evidence.main(args)

    bundle = tmp_path / "bundle"
    _, expected_completion = _marker()
    assert {path.name for path in bundle.iterdir()} == evidence._CAPTURE_FILES
    assert (bundle / "completion.json").read_bytes() == expected_completion
    completion_sha = hashlib.sha256(expected_completion).hexdigest()
    assert (bundle / "completion.sha256").read_text() == f"{completion_sha}\n"

    capture_payload = (bundle / "capture.json").read_bytes()
    capture = json.loads(capture_payload)
    assert capture_payload == evidence.canonical_json_bytes(capture)
    assert capture["completion"]["image_digest"] == DIGEST
    assert capture["completion"]["record_sha256"] == completion_sha
    assert capture["kubernetes"]["pod"] == {
        "name": POD_NAME,
        "phase": "Succeeded",
        "uid": POD_UID,
    }
    assert capture["kubernetes"]["job"]["name"] == JOB
    assert capture["kubernetes"]["container"] == {
        "name": "crop-distill",
        "spec_image": IMAGE,
        "spec_image_digest": DIGEST,
        "status_image_id": f"containerd://sha256:{DIGEST}",
        "status_image_digest": DIGEST,
        "terminated_exit_code": 0,
        "terminated_reason": "Completed",
    }
    assert evidence.verify_bundle(bundle) == capture
    assert json.loads(capsys.readouterr().out) == capture


def test_capture_accepts_docker_pullable_status_image_id(tmp_path: Path) -> None:
    image_id = f"docker-pullable://ghcr.io/example/crop@sha256:{DIGEST}"
    pod = _pod(image_id=image_id)

    evidence.main(_capture_args(tmp_path, pod=pod))

    capture = evidence.verify_bundle(tmp_path / "bundle")
    assert capture["kubernetes"]["container"]["status_image_id"] == image_id


def test_capture_binds_split_job_and_container(tmp_path: Path) -> None:
    evidence.main(_capture_args(tmp_path, kind="split"))

    capture = evidence.verify_bundle(tmp_path / "bundle")
    assert capture["completion"]["kind"] == "split"
    assert capture["kubernetes"]["container"]["name"] == "split"


@pytest.mark.parametrize("source", ["spec", "status"])
def test_capture_rejects_digest_mismatch(tmp_path: Path, source: str) -> None:
    pod = _pod()
    if source == "spec":
        pod["spec"]["containers"][0]["image"] = (
            f"ghcr.io/example/crop@sha256:{OTHER_DIGEST}"
        )
    else:
        pod["status"]["containerStatuses"][0]["imageID"] = (
            f"containerd://sha256:{OTHER_DIGEST}"
        )

    with pytest.raises(SystemExit, match=f"Pod {source} image"):
        evidence.main(_capture_args(tmp_path, pod=pod))
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("location", ["spec", "status", "init"])
def test_capture_rejects_missing_or_ambiguous_target_container(
    tmp_path: Path, location: str
) -> None:
    pod = _pod()
    if location == "spec":
        pod["spec"]["containers"].append({"name": "crop-distill", "image": IMAGE})
    elif location == "status":
        pod["status"]["containerStatuses"].append(
            deepcopy(pod["status"]["containerStatuses"][0])
        )
    else:
        pod["spec"]["initContainers"].append({"name": "crop-distill", "image": IMAGE})

    with pytest.raises(SystemExit, match="ambiguous|exactly once"):
        evidence.main(_capture_args(tmp_path, pod=pod))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda pod: pod["metadata"].update(uid="wrong-uid"), "Pod UID"),
        (
            lambda pod: pod["metadata"]["labels"].update(
                {"batch.kubernetes.io/job-name": "wrong-job"}
            ),
            "Pod label",
        ),
        (
            lambda pod: pod["metadata"]["ownerReferences"][0].update(name="wrong-job"),
            "controller ownerReference",
        ),
        (lambda pod: pod["status"].update(phase="Running"), "phase"),
        (
            lambda pod: pod["status"]["containerStatuses"][0]["state"][
                "terminated"
            ].update(exitCode=9),
            "exitCode",
        ),
    ],
)
def test_capture_rejects_pod_identity_or_success_mismatch(
    tmp_path: Path, mutation: object, message: str
) -> None:
    pod = _pod()
    mutation(pod)

    with pytest.raises(SystemExit, match=message):
        evidence.main(_capture_args(tmp_path, pod=pod))


def test_capture_rejects_multiple_controller_owners(tmp_path: Path) -> None:
    pod = _pod()
    pod["metadata"]["ownerReferences"].append(
        {
            "apiVersion": "batch/v1",
            "controller": True,
            "kind": "Job",
            "name": JOB,
            "uid": "second-controller-uid",
        }
    )

    with pytest.raises(SystemExit, match="exactly one controller"):
        evidence.main(_capture_args(tmp_path, pod=pod))


def test_capture_rejects_operator_selection_mismatch(tmp_path: Path) -> None:
    args = _capture_args(tmp_path)
    args[args.index("--expected-pod") + 1] = "some-other-pod"

    with pytest.raises(SystemExit, match="--expected-pod"):
        evidence.main(args)


@pytest.mark.parametrize("marker_count", [0, 2])
def test_capture_rejects_missing_or_duplicate_terminal_marker(
    tmp_path: Path, marker_count: int
) -> None:
    marker, _ = _marker()
    log = b"ordinary output\n" if marker_count == 0 else marker + marker

    with pytest.raises(SystemExit, match="exactly one terminal evidence marker"):
        evidence.main(_capture_args(tmp_path, log=log))


@pytest.mark.parametrize("corruption", ["sha", "base64", "canonical"])
def test_capture_rejects_corrupt_terminal_marker(
    tmp_path: Path, corruption: str
) -> None:
    record = _record()
    payload = evidence.canonical_json_bytes(record)
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    if corruption == "sha":
        digest = OTHER_DIGEST
    elif corruption == "base64":
        encoded += "!"
    else:
        payload = json.dumps(record, sort_keys=True).encode()
        digest = hashlib.sha256(payload).hexdigest()
        encoded = base64.b64encode(payload).decode("ascii")
    log = f"{evidence.TERMINAL_EVIDENCE_PREFIX} {digest} {encoded}\n".encode()

    with pytest.raises(SystemExit, match="terminal"):
        evidence.main(_capture_args(tmp_path, log=log))


def test_capture_rejects_failed_completion_record(tmp_path: Path) -> None:
    record = _record()
    record["terminal"]["status"] = "failed"
    log, _ = _marker(record)

    with pytest.raises(SystemExit, match="only completed"):
        evidence.main(_capture_args(tmp_path, log=log))


def test_capture_is_create_only(tmp_path: Path) -> None:
    args = _capture_args(tmp_path)
    evidence.main(args)

    with pytest.raises(SystemExit, match="must be new"):
        evidence.main(args)


@pytest.mark.parametrize("target", ["completion", "sha", "capture", "extra"])
def test_offline_verify_rejects_tampered_or_ambiguous_bundle(
    tmp_path: Path, target: str
) -> None:
    evidence.main(_capture_args(tmp_path))
    bundle = tmp_path / "bundle"
    if target == "completion":
        (bundle / "completion.json").write_bytes(b"{}\n")
    elif target == "sha":
        (bundle / "completion.sha256").write_text(f"{OTHER_DIGEST}\n")
    elif target == "capture":
        capture = json.loads((bundle / "capture.json").read_text())
        capture["kubernetes"]["container"]["status_image_id"] = (
            f"containerd://sha256:{OTHER_DIGEST}"
        )
        (bundle / "capture.json").write_bytes(evidence.canonical_json_bytes(capture))
    else:
        (bundle / "unexpected.txt").write_text("ambiguous evidence\n")

    with pytest.raises(evidence.EvidenceCaptureError):
        evidence.verify_bundle(bundle)
