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
IMAGE = f"ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:{DIGEST}"
SOURCE_GIT_SHA = "1" * 40
SPLIT_MANIFEST_SHA256 = "2" * 64
POD_UID = "782f18e4-0197-48c0-b8af-70461d50b7d8"
POD_NAME = "ladder-crop-distill-croma-k4j7p"
JOB = "ladder-crop-distill-croma"
JOB_UID = "572df214-fe9d-4b81-8d5f-5ca6a5c54190"
NAMESPACE = "prithvi-training-default"


def _file_identity(
    path: Path | str, marker: str, *, size: int = 17
) -> dict[str, object]:
    return {"path": str(path), "size_bytes": size, "sha256": marker * 64}


def _python_identity(path: Path) -> dict[str, object]:
    return {
        "implementation": "CPython",
        "path": str(path),
        "version": "3.11.13",
        "version_info": [3, 11, 13],
    }


def _tree_identity(
    *,
    git_sha: str,
    archive_sha256: str,
    marker: str,
) -> dict[str, str]:
    return {
        "archive_sha256": archive_sha256,
        "files_manifest_sha256": marker * 64,
        "git_sha": git_sha,
        "payload_sha256": marker * 64,
    }


def _runtime() -> dict[str, object]:
    return {
        "base_image": evidence.BASE_IMAGE,
        "base_python": _python_identity(evidence.BASE_PYTHON),
        "environments": {
            "model": {
                "pip_freeze": _file_identity(
                    "/opt/provenance/model-pip-freeze.txt", "3"
                ),
                "python": _python_identity(evidence.MODEL_PYTHON),
                "requirements_lock": _file_identity(
                    "/opt/provenance/model-requirements.lock",
                    "4",
                ),
            },
            "scoring": {
                "pip_freeze": _file_identity(
                    "/opt/provenance/scoring-pip-freeze.txt", "5"
                ),
                "python": _python_identity(evidence.SCORING_PYTHON),
                "requirements_lock": _file_identity(
                    "/opt/provenance/scoring-requirements.lock",
                    "6",
                ),
            },
        },
        "external_sources": {
            "clay": _tree_identity(
                git_sha=evidence.CLAY_GIT_SHA,
                archive_sha256=evidence.CLAY_ARCHIVE_SHA256,
                marker="7",
            ),
            "croma": _tree_identity(
                git_sha=evidence.CROMA_GIT_SHA,
                archive_sha256=evidence.CROMA_ARCHIVE_SHA256,
                marker="8",
            ),
        },
        "image": {"digest": DIGEST, "ref": IMAGE},
        "model_resolution": evidence.MODEL_RESOLUTION,
        "runtime_manifest": _file_identity(evidence.RUNTIME_MANIFEST, "9"),
        "source": _tree_identity(
            git_sha=SOURCE_GIT_SHA,
            archive_sha256="c" * 64,
            marker="d",
        ),
        "verification": "verified",
    }


def _split_manifest(kind: str) -> dict[str, object]:
    root = (
        evidence.DISTILL_DIR
        if kind == "split"
        else evidence.WORK_ROOT / POD_UID / "split"
    )
    names = {
        "index": evidence.CROP_INDEX.name,
        "validator_holdout": "lucas_crop_validator_holdout_index.parquet",
        "split": evidence.CROP_SPLIT.name,
    }
    declared: dict[str, object] = {}
    markers = {"index": "3", "validator_holdout": "4", "split": "5"}
    for logical_name, filename in names.items():
        identity = _file_identity(root / filename, markers[logical_name], size=23)
        identity["verification"] = "content"
        if kind == "crop" and logical_name == "validator_holdout":
            identity.pop("size_bytes")
            identity["verification"] = "declaration-only"
        declared[filename] = identity
    immutable_digests = {
        name: f"{index:x}" * 64
        for index, name in enumerate(sorted(evidence._SPLIT_DIGEST_FIELDS), start=1)
    }
    return {
        "counts": {"n_distill": 80, "n_holdout": 20, "n_qualified": 100},
        "declared_artifacts": declared,
        "git_sha": SOURCE_GIT_SHA,
        "immutable_digests": immutable_digests,
        "path": str(root / evidence.CROP_SPLIT_MANIFEST.name),
        "sha256": SPLIT_MANIFEST_SHA256,
        "size_bytes": 4096,
    }


def _record(*, kind: str = "crop") -> dict[str, object]:
    if kind == "crop":
        model = "croma"
        job = JOB
        protocol = evidence.model_protocol(model)
        checkpoint: dict[str, object] | None = {
            "path": str(protocol.checkpoint_path),
            "sha256": protocol.checkpoint_sha256,
            "size_bytes": protocol.checkpoint_size,
            "verification": "extractor-authenticated-private-snapshot",
        }
        artifacts = {
            "features": _file_identity(
                evidence.CROP_HEADS_DIR
                / f"{POD_UID}--{model}_r2_crop_features.parquet",
                "e",
                size=8192,
            ),
            "oof": _file_identity(
                evidence.CROP_HEADS_DIR
                / f"{POD_UID}--{model}_r2_crop_distillability.json",
                "f",
                size=512,
            ),
        }
        effective_uid = evidence.model_process_uid(model)
    else:
        model = None
        job = "ladder-lucas-crop-split"
        checkpoint = None
        split_manifest = _split_manifest(kind)
        artifacts = {
            "index": deepcopy(
                split_manifest["declared_artifacts"][evidence.CROP_INDEX.name]
            ),
            "validator_holdout": deepcopy(
                split_manifest["declared_artifacts"][
                    "lucas_crop_validator_holdout_index.parquet"
                ]
            ),
            "split": deepcopy(
                split_manifest["declared_artifacts"][evidence.CROP_SPLIT.name]
            ),
            "manifest": {
                key: split_manifest[key] for key in ("path", "size_bytes", "sha256")
            },
        }
        for artifact in artifacts.values():
            artifact.pop("verification", None)
        effective_uid = evidence.STORAGE_UID
    return {
        "artifacts": artifacts,
        "checkpoint": checkpoint,
        "job": job,
        "kind": kind,
        "model": model,
        "pod_uid": POD_UID,
        "process_identity": {
            "effective_gid": evidence.STORAGE_GID,
            "effective_uid": effective_uid,
        },
        "run_id": POD_UID,
        "runtime": _runtime(),
        "schema": evidence.COMPLETION_SCHEMA,
        "split_manifest": (_split_manifest(kind) if kind == "crop" else split_manifest),
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
    effective_uid = (
        evidence.model_process_uid("croma") if kind == "crop" else evidence.STORAGE_UID
    )
    env = [
        {"name": "CROP_DISTILL_SOURCE_GIT_SHA", "value": SOURCE_GIT_SHA},
        {"name": "CROP_DISTILL_IMAGE", "value": IMAGE},
    ]
    if kind == "crop":
        env.append(
            {
                "name": "CROP_DISTILL_SPLIT_MANIFEST_SHA256",
                "value": SPLIT_MANIFEST_SHA256,
            }
        )
    env.append(
        {
            "name": "POD_UID",
            "valueFrom": {
                "fieldRef": {"apiVersion": "v1", "fieldPath": "metadata.uid"}
            },
        }
    )
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
            "containers": [
                {
                    "env": env,
                    "image": spec_image,
                    "name": container,
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                        "readOnlyRootFilesystem": True,
                        "runAsGroup": evidence.STORAGE_GID,
                        "runAsNonRoot": True,
                        "runAsUser": effective_uid,
                    },
                }
            ],
            "automountServiceAccountToken": False,
            "initContainers": [],
            "restartPolicy": "Never",
            "securityContext": {
                "runAsGroup": evidence.STORAGE_GID,
                "runAsNonRoot": True,
                "runAsUser": effective_uid,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
        },
        "status": {
            "containerStatuses": [
                {
                    "image": spec_image,
                    "imageID": image_id,
                    "name": container,
                    "ready": False,
                    "restartCount": 0,
                    "lastState": {},
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
    assert capture["kubernetes"]["pod"]["name"] == POD_NAME
    assert capture["kubernetes"]["pod"]["phase"] == "Succeeded"
    assert capture["kubernetes"]["pod"]["uid"] == POD_UID
    assert capture["kubernetes"]["pod"]["isolation"] == {
        "automount_service_account_token": False,
        "host_ipc": False,
        "host_network": False,
        "host_pid": False,
        "restart_policy": "Never",
    }
    assert capture["kubernetes"]["job"]["name"] == JOB
    assert capture["kubernetes"]["container"] == {
        "environment": {
            "literal": {
                "CROP_DISTILL_IMAGE": IMAGE,
                "CROP_DISTILL_SOURCE_GIT_SHA": SOURCE_GIT_SHA,
                "CROP_DISTILL_SPLIT_MANIFEST_SHA256": SPLIT_MANIFEST_SHA256,
            },
            "pod_uid_field_ref": {
                "api_version": "v1",
                "field_path": "metadata.uid",
            },
        },
        "last_state": {},
        "name": "crop-distill",
        "restart_count": 0,
        "security_context": {
            "allow_privilege_escalation": False,
            "capabilities": {"add": [], "drop": ["ALL"]},
            "privileged": False,
            "read_only_root_filesystem": True,
            "run_as_group": evidence.STORAGE_GID,
            "run_as_non_root": True,
            "run_as_user": evidence.model_process_uid("croma"),
            "seccomp_profile": {"source": "pod", "type": "RuntimeDefault"},
        },
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

    with pytest.raises(SystemExit, match="exactly the target container|must be empty"):
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
        (
            lambda pod: pod["status"]["containerStatuses"][0]["state"][
                "terminated"
            ].update(exitCode=False),
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown-model", "allowed crop model"),
        ("wrong-process-identity", "process_identity"),
        ("missing-runtime-source-field", "runtime source must contain exactly"),
        ("wrong-runtime-image-repository", "immutable crop-distill image"),
        ("missing-checkpoint", "checkpoint must be a JSON object"),
        ("wrong-checkpoint-digest", "checkpoint identity sha256"),
        ("missing-artifact", "exactly features and oof"),
        ("wrong-artifact-path", "artifact features path must be exactly"),
        ("missing-split-digest", "immutable_digests must contain exactly"),
        ("boolean-terminal-exit", "exit_code 0"),
        ("negative-python-patch", "exact CPython 3.11"),
        ("zero-split-sha", "nonzero 64 lowercase hex"),
    ],
)
def test_capture_rejects_semantically_incomplete_crop_record(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    record = _record()
    if mutation == "unknown-model":
        record["model"] = "attacker"
        record["job"] = "ladder-crop-distill-attacker"
    elif mutation == "wrong-process-identity":
        record["process_identity"]["effective_uid"] = evidence.STORAGE_UID
    elif mutation == "missing-runtime-source-field":
        del record["runtime"]["source"]["payload_sha256"]
    elif mutation == "wrong-runtime-image-repository":
        record["runtime"]["image"]["ref"] = (
            f"ghcr.io/attacker/crop-distill@sha256:{DIGEST}"
        )
    elif mutation == "missing-checkpoint":
        record["checkpoint"] = None
    elif mutation == "wrong-checkpoint-digest":
        record["checkpoint"]["sha256"] = OTHER_DIGEST
    elif mutation == "missing-artifact":
        del record["artifacts"]["oof"]
    elif mutation == "wrong-artifact-path":
        record["artifacts"]["features"]["path"] = "/tmp/substituted.parquet"
    elif mutation == "missing-split-digest":
        del record["split_manifest"]["immutable_digests"]["distill_input_data_sha256"]
    elif mutation == "boolean-terminal-exit":
        record["terminal"]["exit_code"] = False
    elif mutation == "negative-python-patch":
        record["runtime"]["base_python"]["version"] = "3.11.-1"
        record["runtime"]["base_python"]["version_info"] = [3, 11, -1]
    else:
        record["split_manifest"]["sha256"] = "0" * 64
    log, _ = _marker(record)

    with pytest.raises(SystemExit, match=message):
        evidence.main(_capture_args(tmp_path, log=log))
    assert not (tmp_path / "bundle").exists()


def test_capture_rejects_self_consistent_zero_image_authority(tmp_path: Path) -> None:
    zero_image = "ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:" + "0" * 64
    record = _record()
    record["runtime"]["image"] = {"digest": "0" * 64, "ref": zero_image}
    log, _ = _marker(record)
    pod = _pod(spec_image=zero_image, image_id=f"containerd://sha256:{'0' * 64}")
    pod["spec"]["containers"][0]["env"][1]["value"] = zero_image

    with pytest.raises(SystemExit, match="nonzero 64 lowercase hex"):
        evidence.main(_capture_args(tmp_path, pod=pod, log=log))
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("checkpoint", "split checkpoint must be null"),
        ("missing-artifact", "split artifacts must be exactly"),
        ("artifact-mismatch", "disagrees with split_manifest"),
        ("wrong-manifest-path", "split_manifest identity path must be exactly"),
    ],
)
def test_capture_rejects_semantically_invalid_split_record(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    record = _record(kind="split")
    if mutation == "checkpoint":
        record["checkpoint"] = _file_identity("/tmp/checkpoint.pt", "f")
    elif mutation == "missing-artifact":
        del record["artifacts"]["validator_holdout"]
    elif mutation == "artifact-mismatch":
        record["artifacts"]["index"]["sha256"] = OTHER_DIGEST
    else:
        record["split_manifest"]["path"] = "/tmp/lucas_crop_split.MANIFEST.json"
    log, _ = _marker(record)

    with pytest.raises(SystemExit, match=message):
        evidence.main(_capture_args(tmp_path, kind="split", log=log))
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-source", "CROP_DISTILL_SOURCE_GIT_SHA must exactly once"),
        ("duplicate-source", "CROP_DISTILL_SOURCE_GIT_SHA must exactly once"),
        ("wrong-source", "does not match completion runtime source"),
        ("same-digest-other-image", "does not match completion runtime image"),
        ("missing-split", "CROP_DISTILL_SPLIT_MANIFEST_SHA256 must exactly once"),
        ("zero-split", "nonzero 64 lowercase hex"),
        ("wrong-split", "does not match completion split_manifest"),
        ("split-value-from", "must contain exactly"),
        ("wrong-pod-uid-field", "must reference metadata.uid"),
    ],
)
def test_capture_rejects_missing_or_wrong_pod_authority_environment(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    pod = _pod()
    container = pod["spec"]["containers"][0]
    env = container["env"]
    by_name = {entry["name"]: entry for entry in env}
    if mutation == "missing-source":
        env.remove(by_name["CROP_DISTILL_SOURCE_GIT_SHA"])
    elif mutation == "duplicate-source":
        env.append(deepcopy(by_name["CROP_DISTILL_SOURCE_GIT_SHA"]))
    elif mutation == "wrong-source":
        by_name["CROP_DISTILL_SOURCE_GIT_SHA"]["value"] = "f" * 40
    elif mutation == "same-digest-other-image":
        attacker_ref = f"ghcr.io/attacker/crop-distill@sha256:{DIGEST}"
        by_name["CROP_DISTILL_IMAGE"]["value"] = attacker_ref
        container["image"] = attacker_ref
    elif mutation == "missing-split":
        env.remove(by_name["CROP_DISTILL_SPLIT_MANIFEST_SHA256"])
    elif mutation == "zero-split":
        by_name["CROP_DISTILL_SPLIT_MANIFEST_SHA256"]["value"] = "0" * 64
    elif mutation == "wrong-split":
        by_name["CROP_DISTILL_SPLIT_MANIFEST_SHA256"]["value"] = OTHER_DIGEST
    elif mutation == "split-value-from":
        split_entry = by_name["CROP_DISTILL_SPLIT_MANIFEST_SHA256"]
        del split_entry["value"]
        split_entry["valueFrom"] = {"secretKeyRef": {"name": "x", "key": "y"}}
    else:
        by_name["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] = "metadata.name"

    with pytest.raises(SystemExit, match=message):
        evidence.main(_capture_args(tmp_path, pod=pod))
    assert not (tmp_path / "bundle").exists()


def test_split_capture_rejects_prior_split_anchor_in_pod(tmp_path: Path) -> None:
    pod = _pod(kind="split")
    pod["spec"]["containers"][0]["env"].append(
        {
            "name": "CROP_DISTILL_SPLIT_MANIFEST_SHA256",
            "value": SPLIT_MANIFEST_SHA256,
        }
    )

    with pytest.raises(SystemExit, match="must not be present"):
        evidence.main(_capture_args(tmp_path, kind="split", pod=pod))


@pytest.mark.parametrize("scope", ["pod", "container"])
def test_capture_rejects_pod_process_identity_mismatch(
    tmp_path: Path,
    scope: str,
) -> None:
    pod = _pod()
    if scope == "pod":
        security_context = pod["spec"]["securityContext"]
    else:
        security_context = pod["spec"]["containers"][0]["securityContext"]
    security_context["runAsUser"] = evidence.STORAGE_UID

    with pytest.raises(SystemExit, match="securityContext UID:GID"):
        evidence.main(_capture_args(tmp_path, pod=pod))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("privileged", "securityContext must contain exactly"),
        ("allow-privilege-escalation", "allowPrivilegeEscalation"),
        ("writable-root", "readOnlyRootFilesystem"),
        ("capability-add", "capabilities must contain exactly"),
        ("missing-drop-all", "drop exactly ALL"),
        ("container-seccomp-override", "securityContext must contain exactly"),
        ("pod-seccomp-unconfined", "seccompProfile must be RuntimeDefault"),
        ("supplemental-group", "securityContext must contain exactly"),
        ("host-network", "hostNetwork must be false"),
        ("service-account-token", "automountServiceAccountToken"),
        ("restart-policy", "restartPolicy must be Never"),
        ("sidecar", "no sidecars"),
        ("init-container", "initContainers must be empty"),
        ("restart-count", "restartCount must be zero"),
        ("last-state", "lastState must be empty"),
    ],
)
def test_capture_rejects_weakened_pod_isolation(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    pod = _pod()
    spec = pod["spec"]
    container = spec["containers"][0]
    security = container["securityContext"]
    status = pod["status"]["containerStatuses"][0]
    if mutation == "privileged":
        security["privileged"] = True
    elif mutation == "allow-privilege-escalation":
        security["allowPrivilegeEscalation"] = True
    elif mutation == "writable-root":
        security["readOnlyRootFilesystem"] = False
    elif mutation == "capability-add":
        security["capabilities"]["add"] = ["SYS_ADMIN"]
    elif mutation == "missing-drop-all":
        security["capabilities"]["drop"] = []
    elif mutation == "container-seccomp-override":
        security["seccompProfile"] = {"type": "Unconfined"}
    elif mutation == "pod-seccomp-unconfined":
        spec["securityContext"]["seccompProfile"]["type"] = "Unconfined"
    elif mutation == "supplemental-group":
        spec["securityContext"]["supplementalGroups"] = [0]
    elif mutation == "host-network":
        spec["hostNetwork"] = True
    elif mutation == "service-account-token":
        spec["automountServiceAccountToken"] = True
    elif mutation == "restart-policy":
        spec["restartPolicy"] = "OnFailure"
    elif mutation == "sidecar":
        spec["containers"].append({"name": "sidecar", "image": IMAGE})
    elif mutation == "init-container":
        spec["initContainers"].append({"name": "init", "image": IMAGE})
    elif mutation == "restart-count":
        status["restartCount"] = 1
    else:
        status["lastState"] = {"terminated": {"exitCode": 1}}

    with pytest.raises(SystemExit, match=message):
        evidence.main(_capture_args(tmp_path, pod=pod))
    assert not (tmp_path / "bundle").exists()


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


@pytest.mark.parametrize("removed", ["checkpoint", "artifact"])
def test_offline_verify_rejects_rehashed_semantically_incomplete_completion(
    tmp_path: Path,
    removed: str,
) -> None:
    evidence.main(_capture_args(tmp_path))
    bundle = tmp_path / "bundle"
    completion = json.loads((bundle / "completion.json").read_text())
    if removed == "checkpoint":
        completion["checkpoint"] = None
    else:
        del completion["artifacts"]["features"]
    completion_payload = evidence.canonical_json_bytes(completion)
    completion_sha256 = hashlib.sha256(completion_payload).hexdigest()
    (bundle / "completion.json").write_bytes(completion_payload)
    (bundle / "completion.sha256").write_text(f"{completion_sha256}\n")

    capture = json.loads((bundle / "capture.json").read_text())
    capture["completion"]["record_sha256"] = completion_sha256
    (bundle / "capture.json").write_bytes(evidence.canonical_json_bytes(capture))

    with pytest.raises(
        evidence.EvidenceCaptureError,
        match="checkpoint must be a JSON object|exactly features and oof",
    ):
        evidence.verify_bundle(bundle)


@pytest.mark.parametrize(
    "target",
    [
        "source",
        "image",
        "split",
        "security",
        "hardening",
        "isolation",
        "restart",
    ],
)
def test_offline_verify_rejects_rewritten_pod_authority_binding(
    tmp_path: Path,
    target: str,
) -> None:
    evidence.main(_capture_args(tmp_path))
    bundle = tmp_path / "bundle"
    capture = json.loads((bundle / "capture.json").read_text())
    if target == "security":
        capture["kubernetes"]["container"]["security_context"]["run_as_user"] = (
            evidence.STORAGE_UID
        )
    elif target == "hardening":
        capture["kubernetes"]["container"]["security_context"]["privileged"] = True
    elif target == "isolation":
        capture["kubernetes"]["pod"]["isolation"]["host_network"] = True
    elif target == "restart":
        capture["kubernetes"]["container"]["restart_count"] = False
    else:
        name = {
            "source": "CROP_DISTILL_SOURCE_GIT_SHA",
            "image": "CROP_DISTILL_IMAGE",
            "split": "CROP_DISTILL_SPLIT_MANIFEST_SHA256",
        }[target]
        capture["kubernetes"]["container"]["environment"]["literal"][name] = (
            "f" * 40 if target == "source" else f"sha256:{OTHER_DIGEST}"
        )
    (bundle / "capture.json").write_bytes(evidence.canonical_json_bytes(capture))

    with pytest.raises(
        evidence.EvidenceCaptureError,
        match="inconsistent|invalid",
    ):
        evidence.verify_bundle(bundle)


def test_offline_verify_rejects_boolean_success_exit_code(tmp_path: Path) -> None:
    evidence.main(_capture_args(tmp_path))
    bundle = tmp_path / "bundle"
    capture = json.loads((bundle / "capture.json").read_text())
    capture["kubernetes"]["container"]["terminated_exit_code"] = False
    (bundle / "capture.json").write_bytes(evidence.canonical_json_bytes(capture))

    with pytest.raises(
        evidence.EvidenceCaptureError, match="did not exit successfully"
    ):
        evidence.verify_bundle(bundle)
