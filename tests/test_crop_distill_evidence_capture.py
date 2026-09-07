"""Durable Kubernetes evidence capture for crop-distill Jobs."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import capture_crop_distill_evidence as evidence
from scripts import crop_source_access as source_access

DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
IMAGE = f"ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:{DIGEST}"
SOURCE_GIT_SHA = "1" * 40
SOURCE_ACCESS_DIGEST = "7" * 64
SOURCE_ACCESS_IMAGE = (
    "ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:"
    f"{SOURCE_ACCESS_DIGEST}"
)
SOURCE_ACCESS_SOURCE_GIT_SHA = "5" * 40
SPLIT_SOURCE_GIT_SHA = "6" * 40
SPLIT_MANIFEST_SHA256 = "2" * 64
SOURCE_ACCESS_PLAN_SHA256 = "3" * 64
SOURCE_ACCESS_PLAN_POD_UID = "source-access-plan-pod-uid"
SOURCE_ACCESS_COMPLETION_SHA256 = "4" * 64
SOURCE_ACCESS_COMPLETION_POD_UID = "source-access-apply-pod-uid"
POD_UID = "782f18e4-0197-48c0-b8af-70461d50b7d8"
POD_NAME = "ladder-crop-distill-croma-k4j7p"
JOB = "ladder-crop-distill-croma"
JOB_UID = "572df214-fe9d-4b81-8d5f-5ca6a5c54190"
NAMESPACE = "prithvi-training-default"


@pytest.fixture(autouse=True)
def git_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evidence,
        "_git_authority",
        lambda: {
            "source_git_sha": SOURCE_GIT_SHA,
            "image_ref": IMAGE,
            "source_access_source_git_sha": SOURCE_ACCESS_SOURCE_GIT_SHA,
            "source_access_image_ref": SOURCE_ACCESS_IMAGE,
            "split_source_git_sha": SPLIT_SOURCE_GIT_SHA,
            "source_index_sha256": evidence.SOURCE_ACCESS_INDEX_SHA256,
            "plan_sha256": SOURCE_ACCESS_PLAN_SHA256,
            "plan_pod_uid": SOURCE_ACCESS_PLAN_POD_UID,
            "completion_sha256": SOURCE_ACCESS_COMPLETION_SHA256,
            "completion_pod_uid": SOURCE_ACCESS_COMPLETION_POD_UID,
            "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        },
    )


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


def _runtime(
    *,
    source_git_sha: str = SOURCE_GIT_SHA,
    image_ref: str = IMAGE,
    image_digest: str = DIGEST,
) -> dict[str, object]:
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
        "image": {"digest": image_digest, "ref": image_ref},
        "model_resolution": evidence.MODEL_RESOLUTION,
        "runtime_manifest": _file_identity(evidence.RUNTIME_MANIFEST, "9"),
        "source": _tree_identity(
            git_sha=source_git_sha,
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
        "git_sha": SPLIT_SOURCE_GIT_SHA,
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
        "source_access": (
            None
            if kind == "crop"
            else {
                "plan": {
                    "pod_uid": SOURCE_ACCESS_PLAN_POD_UID,
                    "sha256": SOURCE_ACCESS_PLAN_SHA256,
                },
                "completion": {
                    "pod_uid": SOURCE_ACCESS_COMPLETION_POD_UID,
                    "sha256": SOURCE_ACCESS_COMPLETION_SHA256,
                },
            }
        ),
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
    spec_image: str | None = None,
    image_id: str | None = None,
    subject: Mapping[str, object] | None = None,
    pod_name: str | None = None,
) -> dict[str, object]:
    if subject is None:
        subject = evidence._validate_completion_record(
            _record(kind=kind),
            split_source_git_sha=SPLIT_SOURCE_GIT_SHA,
        )
    else:
        kind = str(subject["kind"])
    contract = evidence._workload_contract(subject)
    if spec_image is None:
        spec_image = str(contract["image_ref"])
    if image_id is None:
        image_id = f"containerd://sha256:{subject['digest']}"
    container = str(contract["container"])
    job = str(contract["job"])
    if pod_name is None:
        pod_name = (
            POD_NAME if kind == "crop" else "ladder-lucas-crop-split-h8v2c"
        )
    env = [
        {"name": name, "value": value}
        for name, value in contract["literal_env"].items()
    ]
    env.append(
        {
            "name": "POD_UID",
            "valueFrom": {
                "fieldRef": {"apiVersion": "v1", "fieldPath": "metadata.uid"}
            },
        }
    )
    volume_mounts = []
    for mount in contract["mounts"]:
        raw_mount = {
            "name": mount["name"],
            "mountPath": mount["mount_path"],
        }
        if mount["sub_path"] is not None:
            raw_mount["subPath"] = mount["sub_path"]
        if mount["read_only"]:
            raw_mount["readOnly"] = True
        volume_mounts.append(raw_mount)
    volumes = []
    for volume in contract["volumes"]:
        if volume["type"] == "persistentVolumeClaim":
            volumes.append(
                {
                    "name": volume["name"],
                    "persistentVolumeClaim": {"claimName": volume["claim_name"]},
                }
            )
        elif volume["type"] == "emptyDir":
            empty_dir = {}
            if volume["size_limit"] is not None:
                empty_dir["sizeLimit"] = volume["size_limit"]
            volumes.append({"name": volume["name"], "emptyDir": empty_dir})
        else:
            volumes.append(
                {
                    "name": volume["name"],
                    "configMap": {
                        "name": volume["config_map_name"],
                        "defaultMode": volume["default_mode"],
                        "optional": volume["optional"],
                        "items": deepcopy(volume["items"]),
                    },
                }
            )
    pod_security = {
        "runAsGroup": contract["effective_gid"],
        "runAsUser": contract["effective_uid"],
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    container_security = {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
        "readOnlyRootFilesystem": True,
        "runAsGroup": contract["effective_gid"],
        "runAsUser": contract["effective_uid"],
    }
    if contract["capabilities_add"]:
        container_security["capabilities"]["add"] = contract["capabilities_add"]
    if contract["run_as_non_root"]:
        pod_security["runAsNonRoot"] = True
        container_security["runAsNonRoot"] = True
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "labels": evidence._expected_labels(contract, JOB_UID),
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
            "uid": subject["pod_uid"],
        },
        "spec": {
            "containers": [
                {
                    "env": env,
                    "image": spec_image,
                    "imagePullPolicy": "IfNotPresent",
                    "command": contract["command"],
                    "args": contract["args"],
                    "name": container,
                    "resources": deepcopy(contract["resources"]),
                    "securityContext": container_security,
                    "terminationMessagePath": "/dev/termination-log",
                    "terminationMessagePolicy": "File",
                    "volumeMounts": volume_mounts,
                }
            ],
            "activeDeadlineSeconds": contract["active_deadline_seconds"],
            "automountServiceAccountToken": False,
            "ephemeralContainers": [],
            "imagePullSecrets": [{"name": "ghcr-push"}],
            "initContainers": [],
            "nodeName": "worker-node-1",
            "nodeSelector": deepcopy(contract["node_selector"]),
            "restartPolicy": "Never",
            "securityContext": pod_security,
            "serviceAccount": "default",
            "serviceAccountName": "default",
            "volumes": volumes,
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


def _source_runtime() -> dict[str, str]:
    return {
        "source_git_sha": SOURCE_ACCESS_SOURCE_GIT_SHA,
        "image_ref": SOURCE_ACCESS_IMAGE,
        "runtime_manifest_sha256": "5" * 64,
        "source_payload_sha256": "6" * 64,
    }


def _source_identity(
    index: int,
    *,
    uid: int = 0,
    gid: int = 0,
    mode: str = "0600",
    ctime_ns: int | None = None,
) -> dict[str, object]:
    return {
        "dev": 11,
        "inode": index + 100,
        "size": 4096 + index,
        "mtime_ns": 1_000_000 + index,
        "ctime_ns": 2_000_000 + index if ctime_ns is None else ctime_ns,
        "uid": uid,
        "gid": gid,
        "mode": mode,
        "nlink": 1,
        "sha256": hashlib.sha256(f"source-tile-{index}".encode()).hexdigest(),
    }


def _source_index_identity() -> dict[str, object]:
    return {
        "path": str(evidence.SOURCE_ACCESS_INDEX_INPUT),
        "dev": 7,
        "inode": 41,
        "size": evidence.SOURCE_ACCESS_INDEX_SIZE,
        "mtime_ns": 800,
        "ctime_ns": 900,
        "uid": 0,
        "gid": 0,
        "mode": "0644",
        "nlink": 1,
        "sha256": evidence.SOURCE_ACCESS_INDEX_SHA256,
    }


def _source_plan_record() -> dict[str, object]:
    files: list[dict[str, object]] = []
    for index in range(source_access.SOURCE_ACCESS_EXPECTED_CANDIDATES):
        tile_name = f"tile-{index:04d}"
        repair = index < source_access.SOURCE_ACCESS_EXPECTED_REPAIRS
        identity = _source_identity(
            index,
            mode="0600" if repair else "0644",
        )
        files.append(
            {
                "tile_name": tile_name,
                "file_name": f"{tile_name}.npz",
                "path": str(evidence.DATA_DIR / f"{tile_name}.npz"),
                **identity,
                "action": (
                    source_access.ACTION_REPAIR
                    if repair
                    else source_access.ACTION_ACCEPT_0644
                ),
            }
        )
    return {
        "schema": evidence.SOURCE_ACCESS_PLAN_SCHEMA,
        "pod_uid": SOURCE_ACCESS_PLAN_POD_UID,
        "runtime": _source_runtime(),
        "source_index": _source_index_identity(),
        "data_dir": str(evidence.DATA_DIR),
        "crop_window": [8, 504],
        "crop_rows": source_access.SOURCE_ACCESS_EXPECTED_CROP_ROWS,
        "target": {
            "uid": source_access.SOURCE_ACCESS_TARGET_UID,
            "gid": source_access.SOURCE_ACCESS_TARGET_GID,
            "mode": format(source_access.SOURCE_ACCESS_TARGET_MODE, "04o"),
        },
        "summary": {
            "candidates": source_access.SOURCE_ACCESS_EXPECTED_CANDIDATES,
            "repairs": source_access.SOURCE_ACCESS_EXPECTED_REPAIRS,
            "accepted_0644": source_access.SOURCE_ACCESS_EXPECTED_NOOPS,
            "already_correct": 0,
        },
        "files": files,
    }


def _source_completion_record() -> dict[str, object]:
    files: list[dict[str, object]] = []
    for index in range(source_access.SOURCE_ACCESS_EXPECTED_CANDIDATES):
        tile_name = f"tile-{index:04d}"
        repair = index < source_access.SOURCE_ACCESS_EXPECTED_REPAIRS
        before = _source_identity(
            index,
            mode="0600" if repair else "0644",
        )
        after = deepcopy(before)
        if repair:
            after.update(
                {
                    "uid": source_access.SOURCE_ACCESS_TARGET_UID,
                    "gid": source_access.SOURCE_ACCESS_TARGET_GID,
                    "mode": format(source_access.SOURCE_ACCESS_TARGET_MODE, "04o"),
                    "ctime_ns": int(before["ctime_ns"]) + 1,
                }
            )
        files.append(
            {
                "tile_name": tile_name,
                "planned_action": (
                    source_access.ACTION_REPAIR
                    if repair
                    else source_access.ACTION_ACCEPT_0644
                ),
                "applied_action": "repaired" if repair else "no-op",
                "before": before,
                "after": after,
                "sha256_unchanged": True,
                "size_unchanged": True,
                "mtime_unchanged": True,
                "inode_unchanged": True,
                "ctime_changed": repair,
            }
        )
    return {
        "schema": evidence.SOURCE_ACCESS_COMPLETION_SCHEMA,
        "pod_uid": SOURCE_ACCESS_COMPLETION_POD_UID,
        "status": "completed",
        "runtime": _source_runtime(),
        "process_identity": {"effective_uid": 0, "effective_gid": 2000},
        "plan": {
            "pod_uid": SOURCE_ACCESS_PLAN_POD_UID,
            "sha256": SOURCE_ACCESS_PLAN_SHA256,
        },
        "source_index": _source_index_identity(),
        "summary": {
            "files": source_access.SOURCE_ACCESS_EXPECTED_CANDIDATES,
            "repaired": source_access.SOURCE_ACCESS_EXPECTED_REPAIRS,
            "already_repaired": 0,
            "no_op": source_access.SOURCE_ACCESS_EXPECTED_NOOPS,
            "content_unchanged": True,
            "ctime_policy": (
                "changed-on-repair; unchanged permitted on idempotent no-op"
            ),
        },
        "files": files,
    }


def _storage_prep_record() -> dict[str, object]:
    targets = []
    for index, target in enumerate(evidence.STORAGE_TARGETS, start=1):
        targets.append(
            {
                "path": str(target.path),
                "uid": target.uid,
                "gid": target.gid,
                "mode": format(target.mode, "04o"),
                "device": 21,
                "inode": index,
                "state": "writable",
            }
        )
    return {
        "schema": evidence.STORAGE_PREP_COMPLETION_SCHEMA,
        "pod_uid": "storage-prep-pod-uid",
        "status": "completed",
        "process_identity": {
            "effective_uid": 0,
            "effective_gid": evidence.STORAGE_GID,
        },
        "preserved_frozen_mode": format(evidence.FROZEN_SPLIT_MODE, "04o"),
        "runtime": _runtime(
            source_git_sha=SOURCE_ACCESS_SOURCE_GIT_SHA,
            image_ref=SOURCE_ACCESS_IMAGE,
            image_digest=SOURCE_ACCESS_DIGEST,
        ),
        "targets": targets,
        "dataset_lock": {
            "path": str(evidence.SOURCE_ACCESS_LOCK_BACKING_FILE),
            "uid": 0,
            "gid": evidence.STORAGE_GID,
            "mode": format(evidence.SOURCE_ACCESS_LOCK_MODE, "04o"),
            "device": 21,
            "inode": 999,
            "size_bytes": 0,
            "nlink": 1,
            "state": "ready",
        },
    }


def _source_subject(
    kind: str,
    record: Mapping[str, object],
) -> dict[str, object]:
    is_plan = kind == "source-access-plan"
    return {
        "container": "source-access-plan" if is_plan else "source-access-apply",
        "digest": SOURCE_ACCESS_DIGEST,
        "effective_gid": evidence.STORAGE_GID,
        "effective_uid": 0,
        "image_ref": SOURCE_ACCESS_IMAGE,
        "job": (
            "ladder-crop-source-access-plan"
            if is_plan
            else "ladder-crop-source-access-apply"
        ),
        "kind": kind,
        "model": None,
        "plan": None if is_plan else record["plan"],
        "pod_uid": record["pod_uid"],
        "record_schema": record["schema"],
        "source_git_sha": SOURCE_ACCESS_SOURCE_GIT_SHA,
    }


def _generic_marker(prefix: str, record: Mapping[str, object]) -> tuple[bytes, bytes]:
    payload = evidence.canonical_json_bytes(dict(record))
    digest = hashlib.sha256(payload).hexdigest()
    marker = (
        f"{prefix} {digest} {base64.b64encode(payload).decode('ascii')}\n"
    ).encode()
    return marker, payload


def _source_capture_args(
    tmp_path: Path,
    *,
    kind: str,
    record: dict[str, object] | None = None,
    pod: dict[str, object] | None = None,
    marker: bytes | None = None,
) -> tuple[list[str], dict[str, object], Path]:
    is_plan = kind == "source-access-plan"
    if record is None:
        record = _source_plan_record() if is_plan else _source_completion_record()
    subject = _source_subject(kind, record)
    pod_name = (
        "ladder-crop-source-access-plan-pod"
        if is_plan
        else "ladder-crop-source-access-apply-pod"
    )
    if pod is None:
        pod = _pod(kind=kind, subject=subject, pod_name=pod_name)
    if marker is None:
        marker, _ = _generic_marker(evidence._MARKER_PREFIX[kind], record)
    pod_json = tmp_path / "pod.json"
    pod_log = tmp_path / "pod.log"
    record_file = tmp_path / ("plan.json" if is_plan else "completion.json")
    pod_json.write_text(json.dumps(pod), encoding="utf-8")
    pod_log.write_bytes(marker)
    record_file.write_bytes(evidence.canonical_json_bytes(record))
    return (
        [
            "capture",
            "--evidence-kind",
            kind,
            "--pod-json",
            str(pod_json),
            "--pod-log",
            str(pod_log),
            "--record-file",
            str(record_file),
            "--container",
            str(subject["container"]),
            "--expected-namespace",
            NAMESPACE,
            "--expected-pod",
            pod_name,
            "--expected-job",
            str(subject["job"]),
            "--out-dir",
            str(tmp_path / "bundle"),
        ],
        record,
        record_file,
    )


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
        "--evidence-kind",
        kind,
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
    assert {path.name for path in bundle.iterdir()} == evidence._bundle_files("crop")
    assert (bundle / "completion.json").read_bytes() == expected_completion
    completion_sha = hashlib.sha256(expected_completion).hexdigest()
    assert (bundle / "completion.sha256").read_text() == f"{completion_sha}\n"

    capture_payload = (bundle / "capture.json").read_bytes()
    capture = json.loads(capture_payload)
    assert capture_payload == evidence.canonical_json_bytes(capture)
    assert capture["workload_record"]["record_sha256"] == completion_sha
    normalized = capture["observed_pod"]["normalized"]
    assert normalized["metadata"]["name"] == POD_NAME
    assert normalized["metadata"]["uid"] == POD_UID
    assert normalized["status"]["phase"] == "Succeeded"
    assert normalized["spec"]["isolation"] == {
        "automount_service_account_token": False,
        "host_ipc": False,
        "host_network": False,
        "host_pid": False,
        "restart_policy": "Never",
    }
    container = normalized["spec"]["containers"][0]
    assert container["command"] == ["/usr/local/bin/python"]
    assert container["args"] == [
        "/opt/imintengine/scripts/run_crop_distill_job.py",
        "--model",
        "croma",
    ]
    assert container["environment"]["literal"] == {
        "CROP_DISTILL_IMAGE": IMAGE,
        "CROP_DISTILL_SOURCE_GIT_SHA": SOURCE_GIT_SHA,
        "CROP_DISTILL_SPLIT_MANIFEST_SHA256": SPLIT_MANIFEST_SHA256,
        "CROP_DISTILL_SPLIT_SOURCE_GIT_SHA": SPLIT_SOURCE_GIT_SHA,
        "HOME": "/work/home",
        "TMPDIR": "/work/tmp",
    }
    split_mount = next(
        mount
        for mount in container["volume_mounts"]
        if mount["mount_path"] == "/cephfs/distill/crop_split"
    )
    assert split_mount["read_only"] is True
    assert normalized["status"]["container_statuses"][0]["image_digest"] == DIGEST
    assert capture["observed_pod"]["raw"]["sha256"] == hashlib.sha256(
        (bundle / "pod.json").read_bytes()
    ).hexdigest()
    assert evidence.verify_bundle(bundle) == capture
    assert json.loads(capsys.readouterr().out) == capture


def test_capture_accepts_docker_pullable_status_image_id(tmp_path: Path) -> None:
    image_id = f"docker-pullable://ghcr.io/example/crop@sha256:{DIGEST}"
    pod = _pod(image_id=image_id)

    evidence.main(_capture_args(tmp_path, pod=pod))

    capture = evidence.verify_bundle(tmp_path / "bundle")
    assert (
        capture["observed_pod"]["normalized"]["status"]["container_statuses"][0][
            "image_id"
        ]
        == image_id
    )


def test_capture_accepts_immutable_runtime_status_image_digest(
    tmp_path: Path,
) -> None:
    runtime_config_digest = "7" * 64
    pod = _pod()
    pod["status"]["containerStatuses"][0]["image"] = (
        f"sha256:{runtime_config_digest}"
    )

    evidence.main(_capture_args(tmp_path, pod=pod))

    capture = evidence.verify_bundle(tmp_path / "bundle")
    status = capture["observed_pod"]["normalized"]["status"][
        "container_statuses"
    ][0]
    assert status["image"] == f"sha256:{runtime_config_digest}"
    assert status["image_digest"] == DIGEST


def test_capture_rejects_mutable_runtime_status_image(tmp_path: Path) -> None:
    pod = _pod()
    pod["status"]["containerStatuses"][0]["image"] = (
        "ghcr.io/example/crop:latest"
    )

    with pytest.raises(SystemExit, match="Pod status image must be an immutable"):
        evidence.main(_capture_args(tmp_path, pod=pod))

    assert not (tmp_path / "bundle").exists()


def test_capture_binds_split_job_and_container(tmp_path: Path) -> None:
    evidence.main(_capture_args(tmp_path, kind="split"))

    capture = evidence.verify_bundle(tmp_path / "bundle")
    assert capture["workload_record"]["kind"] == "split"
    container = capture["observed_pod"]["normalized"]["spec"]["containers"][0]
    assert container["name"] == "split"
    assert container["command"] == [str(evidence.SCORING_PYTHON)]
    assert container["environment"]["literal"] == {
        "CROP_DISTILL_IMAGE": IMAGE,
        "CROP_DISTILL_SOURCE_GIT_SHA": SOURCE_GIT_SHA,
        "CROP_DISTILL_SPLIT_SOURCE_GIT_SHA": SPLIT_SOURCE_GIT_SHA,
        "CROP_SOURCE_ACCESS_IMAGE": SOURCE_ACCESS_IMAGE,
        "CROP_SOURCE_ACCESS_SOURCE_GIT_SHA": SOURCE_ACCESS_SOURCE_GIT_SHA,
        "CROP_SOURCE_FREEZE_LEASE_PATH": "/var/run/crop-source-freeze/lease.json",
        "CROP_SOURCE_ACCESS_COMPLETION_POD_UID": SOURCE_ACCESS_COMPLETION_POD_UID,
        "CROP_SOURCE_ACCESS_COMPLETION_SHA256": SOURCE_ACCESS_COMPLETION_SHA256,
        "CROP_SOURCE_ACCESS_PLAN_POD_UID": SOURCE_ACCESS_PLAN_POD_UID,
        "CROP_SOURCE_ACCESS_PLAN_SHA256": SOURCE_ACCESS_PLAN_SHA256,
        "HOME": "/work/home",
        "TMPDIR": "/work/tmp",
    }


@pytest.mark.parametrize(
    "kind",
    ["source-access-plan", "source-access-apply"],
)
def test_source_access_capture_binds_marker_pvc_pod_then_requires_git_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    args, record, record_file = _source_capture_args(tmp_path, kind=kind)
    record_payload = record_file.read_bytes()
    record_sha256 = hashlib.sha256(record_payload).hexdigest()

    # Initial live capture validates only pre-existing authority: PLAN uses
    # source/image/index; APPLY additionally uses the reviewed PLAN pin.
    evidence.main(args)
    bundle = tmp_path / "bundle"
    capture = json.loads((bundle / "capture.json").read_text())
    normalized = capture["observed_pod"]["normalized"]
    container = normalized["spec"]["containers"][0]
    assert normalized["metadata"]["uid"] == record["pod_uid"]
    assert capture["workload_record"]["record_sha256"] == record_sha256
    assert (bundle / evidence._RECORD_FILE[kind]).read_bytes() == record_payload
    assert capture["observed_pod"]["raw"]["sha256"] == hashlib.sha256(
        (bundle / "pod.json").read_bytes()
    ).hexdigest()
    assert capture["observed_pod"]["normalized_sha256"] == hashlib.sha256(
        evidence.canonical_json_bytes(normalized)
    ).hexdigest()
    data_mount = container["volume_mounts"][0]
    assert data_mount["sub_path"] == "unified_v2_512"
    assert data_mount["read_only"] is (kind == "source-access-plan")
    assert container["security_context"]["run_as_user"] == 0
    assert container["security_context"]["capabilities"]["drop"] == ["ALL"]
    assert container["security_context"]["capabilities"]["add"] == (
        [] if kind == "source-access-plan" else ["CHOWN", "FOWNER"]
    )

    # A live-captured output is not yet downstream authority. Offline verify
    # becomes valid only after review pins this exact record SHA and Pod UID.
    with pytest.raises(evidence.EvidenceCaptureError, match="Git authority|mismatch"):
        evidence.verify_bundle(bundle)
    pinned = dict(evidence._git_authority())
    if kind == "source-access-plan":
        pinned.update(
            plan_sha256=record_sha256,
            plan_pod_uid=str(record["pod_uid"]),
        )
    else:
        pinned.update(
            completion_sha256=record_sha256,
            completion_pod_uid=str(record["pod_uid"]),
        )
    monkeypatch.setattr(evidence, "_git_authority", lambda: pinned)
    assert evidence.verify_bundle(bundle)["workload_record"]["record_sha256"] == (
        record_sha256
    )


def test_source_access_capture_rejects_marker_pvc_byte_mismatch(
    tmp_path: Path,
) -> None:
    record = _source_plan_record()
    marker_record = deepcopy(record)
    marker_record["crop_rows"] = int(marker_record["crop_rows"]) + 1
    marker, _ = _generic_marker(
        evidence.SOURCE_ACCESS_PLAN_MARKER,
        marker_record,
    )
    args, _, _ = _source_capture_args(
        tmp_path,
        kind="source-access-plan",
        record=record,
        marker=marker,
    )

    with pytest.raises(SystemExit, match="marker bytes do not equal the PVC record"):
        evidence.main(args)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("kind", "mutation", "match"),
    [
        ("source-access-plan", "schema", "PLAN schema"),
        ("source-access-plan", "runtime", "runtime differs from Git"),
        ("source-access-plan", "cardinality", "candidate count"),
        ("source-access-apply", "schema", "APPLY schema"),
        ("source-access-apply", "runtime", "runtime differs from Git"),
        ("source-access-apply", "cardinality", "candidate count"),
    ],
)
def test_source_access_capture_rejects_schema_runtime_or_cardinality_drift(
    tmp_path: Path,
    kind: str,
    mutation: str,
    match: str,
) -> None:
    record = (
        _source_plan_record()
        if kind == "source-access-plan"
        else _source_completion_record()
    )
    if mutation == "schema":
        record["schema"] = "alternate-valid-looking-schema"
    elif mutation == "runtime":
        record["runtime"]["source_git_sha"] = "f" * 40
    else:
        record["files"].pop()
    args, _, _ = _source_capture_args(tmp_path, kind=kind, record=record)

    with pytest.raises(SystemExit, match=match):
        evidence.main(args)
    assert not (tmp_path / "bundle").exists()


def test_storage_prep_capture_binds_exact_marker_pvc_and_root_pod(
    tmp_path: Path,
) -> None:
    record = _storage_prep_record()
    subject = evidence._validate_storage_prep_record(
        record,
        evidence._validated_git_authority("storage-prep"),
    )
    pod_name = "ladder-crop-distill-storage-prep-pod"
    pod = _pod(kind="storage-prep", subject=subject, pod_name=pod_name)
    marker, record_payload = _generic_marker(
        evidence.STORAGE_PREP_COMPLETION_MARKER,
        record,
    )
    pod_json = tmp_path / "pod.json"
    pod_log = tmp_path / "pod.log"
    record_file = tmp_path / "completion-record.json"
    pod_json.write_text(json.dumps(pod), encoding="utf-8")
    pod_log.write_bytes(marker)
    record_file.write_bytes(record_payload)

    evidence.main(
        [
            "capture",
            "--evidence-kind",
            "storage-prep",
            "--pod-json",
            str(pod_json),
            "--pod-log",
            str(pod_log),
            "--record-file",
            str(record_file),
            "--container",
            "storage-prep",
            "--expected-namespace",
            NAMESPACE,
            "--expected-pod",
            pod_name,
            "--expected-job",
            "ladder-crop-distill-storage-prep",
            "--out-dir",
            str(tmp_path / "bundle"),
        ]
    )

    capture = evidence.verify_bundle(tmp_path / "bundle")
    normalized = capture["observed_pod"]["normalized"]
    container = normalized["spec"]["containers"][0]
    assert normalized["schema"] == evidence._POD_OBSERVATION_SCHEMAS[
        "storage-prep"
    ]
    assert normalized["metadata"]["uid"] == record["pod_uid"]
    assert container["command"] == [str(evidence.BASE_PYTHON)]
    assert container["security_context"]["capabilities"] == {
        "add": ["CHOWN", "FOWNER"],
        "drop": ["ALL"],
    }
    assert container["volume_mounts"] == [
        evidence._mount("training-data-cephfs", "/cephfs/distill", "distill", False),
        evidence._mount("training-data-cephfs", "/cephfs/ops", "ops", False),
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "/cephfs/other/dataset.lock"),
        ("uid", 2000),
        ("gid", 0),
        ("mode", "0640"),
        ("size_bytes", 1),
        ("nlink", 2),
        ("state", "created"),
    ],
)
def test_storage_prep_record_rejects_dataset_lock_drift(
    field: str,
    value: object,
) -> None:
    record = _storage_prep_record()
    record["dataset_lock"][field] = value

    with pytest.raises(
        evidence.EvidenceCaptureError,
        match="dataset lock",
    ):
        evidence._validate_storage_prep_record(
            record,
            evidence._validated_git_authority("storage-prep"),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("command", "command/args"),
        ("args", "command/args"),
        ("extra-env", "environment"),
        ("env-from", "envFrom"),
        ("mount", "volumeMounts"),
        ("volume-device", "volumeDevices"),
        ("host-path", "volumes"),
        ("secret", "volumes"),
        ("projected-token", "volumes"),
        ("extra-pvc", "volumes"),
        ("service-account", "service-account"),
        ("ephemeral", "ephemeralContainers"),
        ("ephemeral-status", "ephemeralContainerStatuses"),
        ("node-name", "nodeName"),
    ],
)
def test_capture_rejects_unreviewed_pod_authority_surface(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    pod = _pod()
    spec = pod["spec"]
    container = spec["containers"][0]
    if mutation == "command":
        container["command"] = ["/bin/sh"]
    elif mutation == "args":
        container["args"].append("--unreviewed")
    elif mutation == "extra-env":
        container["env"].append({"name": "UNREVIEWED", "value": "1"})
    elif mutation == "env-from":
        container["envFrom"] = [{"secretRef": {"name": "credentials"}}]
    elif mutation == "mount":
        container["volumeMounts"][0]["readOnly"] = False
    elif mutation == "volume-device":
        container["volumeDevices"] = [
            {"name": "training-data-cephfs", "devicePath": "/dev/x"}
        ]
    elif mutation == "host-path":
        spec["volumes"].append(
            {"name": "host", "hostPath": {"path": "/", "type": "Directory"}}
        )
    elif mutation == "secret":
        spec["volumes"].append(
            {"name": "secret", "secret": {"secretName": "credentials"}}
        )
    elif mutation == "projected-token":
        spec["volumes"].append(
            {
                "name": "token",
                "projected": {
                    "sources": [
                        {
                            "serviceAccountToken": {
                                "path": "token",
                                "expirationSeconds": 3600,
                            }
                        }
                    ]
                },
            }
        )
    elif mutation == "extra-pvc":
        spec["volumes"].append(
            {
                "name": "other-pvc",
                "persistentVolumeClaim": {"claimName": "other-pvc"},
            }
        )
    elif mutation == "service-account":
        spec["serviceAccount"] = spec["serviceAccountName"] = "privileged"
    elif mutation == "ephemeral":
        spec["ephemeralContainers"] = [{"name": "debug", "image": IMAGE}]
    elif mutation == "ephemeral-status":
        pod["status"]["ephemeralContainerStatuses"] = [{"name": "debug"}]
    else:
        del spec["nodeName"]

    with pytest.raises(SystemExit, match=match):
        evidence.main(_capture_args(tmp_path, pod=pod))
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("kind", "mutation", "match"),
    [
        ("source-access-plan", "source-rw", "volumeMounts"),
        ("source-access-plan", "add-capability", "capabilities"),
        ("source-access-apply", "source-ro", "volumeMounts"),
        ("source-access-apply", "extra-capability", "capabilities"),
        ("source-access-apply", "non-root", "UID:GID"),
    ],
)
def test_source_access_pod_contract_rejects_privilege_or_mount_drift(
    kind: str,
    mutation: str,
    match: str,
) -> None:
    record: dict[str, object] = {
        "schema": (
            evidence.SOURCE_ACCESS_PLAN_SCHEMA
            if kind == "source-access-plan"
            else evidence.SOURCE_ACCESS_COMPLETION_SCHEMA
        ),
        "pod_uid": (
            SOURCE_ACCESS_PLAN_POD_UID
            if kind == "source-access-plan"
            else SOURCE_ACCESS_COMPLETION_POD_UID
        ),
        "plan": {
            "pod_uid": SOURCE_ACCESS_PLAN_POD_UID,
            "sha256": SOURCE_ACCESS_PLAN_SHA256,
        },
    }
    subject = _source_subject(kind, record)
    pod_name = f"{kind}-pod"
    pod = _pod(kind=kind, subject=subject, pod_name=pod_name)
    container = pod["spec"]["containers"][0]
    if mutation in {"source-rw", "source-ro"}:
        container["volumeMounts"][0]["readOnly"] = mutation == "source-ro"
    elif mutation == "add-capability":
        container["securityContext"]["capabilities"]["add"] = ["CHOWN"]
    elif mutation == "extra-capability":
        container["securityContext"]["capabilities"]["add"].append(
            "DAC_OVERRIDE"
        )
    else:
        pod["spec"]["securityContext"]["runAsUser"] = 1000

    with pytest.raises(evidence.EvidenceCaptureError, match=match):
        evidence.normalize_observed_pod(
            pod,
            contract=evidence._workload_contract(subject),
            expected_namespace=NAMESPACE,
            expected_pod=pod_name,
        )


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

    with pytest.raises(SystemExit, match="Pod spec image|Pod status imageID"):
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

    with pytest.raises(
        SystemExit, match="exactly one container and status|must be empty"
    ):
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
            "ownerReference",
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

    with pytest.raises(SystemExit, match="exactly one Job ownerReference"):
        evidence.main(_capture_args(tmp_path, pod=pod))


def test_capture_rejects_operator_selection_mismatch(tmp_path: Path) -> None:
    args = _capture_args(tmp_path)
    args[args.index("--expected-pod") + 1] = "some-other-pod"

    with pytest.raises(SystemExit, match="live selection"):
        evidence.main(args)


@pytest.mark.parametrize("marker_count", [0, 2])
def test_capture_rejects_missing_or_duplicate_terminal_marker(
    tmp_path: Path, marker_count: int
) -> None:
    marker, _ = _marker()
    log = b"ordinary output\n" if marker_count == 0 else marker + marker

    with pytest.raises(SystemExit, match="exactly one recognized terminal marker"):
        evidence.main(_capture_args(tmp_path, log=log))


@pytest.mark.parametrize("ending", [b"", b"\r\n"])
def test_capture_requires_exact_marker_line_ending(
    tmp_path: Path, ending: bytes
) -> None:
    marker, _ = _marker()
    log = marker.removesuffix(b"\n") + ending

    with pytest.raises(SystemExit, match="canonical LF line ending"):
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
        ("missing-source", "environment"),
        ("duplicate-source", "environment"),
        ("wrong-source", "environment"),
        ("same-digest-other-image", "environment|Git authority"),
        ("missing-split", "environment"),
        ("zero-split", "environment"),
        ("wrong-split", "environment"),
        ("split-value-from", "must contain exactly"),
        ("wrong-pod-uid-field", "POD_UID"),
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

    with pytest.raises(SystemExit, match="environment"):
        evidence.main(_capture_args(tmp_path, kind="split", pod=pod))


@pytest.mark.parametrize(
    ("name", "mutation", "message"),
    [
        (
            "CROP_SOURCE_ACCESS_PLAN_SHA256",
            "remove",
            "environment",
        ),
        (
            "CROP_SOURCE_ACCESS_PLAN_POD_UID",
            "remove",
            "environment",
        ),
        (
            "CROP_SOURCE_ACCESS_PLAN_SHA256",
            "zero",
            "environment",
        ),
        (
            "CROP_SOURCE_ACCESS_COMPLETION_SHA256",
            "remove",
            "environment",
        ),
        (
            "CROP_SOURCE_ACCESS_COMPLETION_SHA256",
            "zero",
            "environment",
        ),
        (
            "CROP_SOURCE_ACCESS_COMPLETION_POD_UID",
            "unsafe",
            "environment",
        ),
    ],
)
def test_split_capture_rejects_invalid_source_access_authority(
    tmp_path: Path,
    name: str,
    mutation: str,
    message: str,
) -> None:
    pod = _pod(kind="split")
    env = pod["spec"]["containers"][0]["env"]
    entry = next(item for item in env if item["name"] == name)
    if mutation == "remove":
        env.remove(entry)
    elif mutation == "zero":
        entry["value"] = "0" * 64
    else:
        entry["value"] = "../unsafe"

    with pytest.raises(SystemExit, match=message):
        evidence.main(_capture_args(tmp_path, kind="split", pod=pod))
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (
            "CROP_SOURCE_ACCESS_PLAN_SHA256",
            "0" * 64,
            "nonzero 64 lowercase hex",
        ),
        (
            "CROP_SOURCE_ACCESS_COMPLETION_SHA256",
            "not-a-digest",
            "nonzero 64 lowercase hex",
        ),
        (
            "CROP_SOURCE_ACCESS_COMPLETION_POD_UID",
            "../unsafe",
            "not a safe identifier",
        ),
        (
            "CROP_SOURCE_ACCESS_PLAN_POD_UID",
            "../unsafe",
            "not a safe identifier",
        ),
    ],
)
def test_offline_verify_rejects_rewritten_split_source_access_authority(
    tmp_path: Path,
    name: str,
    value: str,
    message: str,
) -> None:
    evidence.main(_capture_args(tmp_path, kind="split"))
    bundle = tmp_path / "bundle"
    capture = json.loads((bundle / "capture.json").read_text())
    capture["observed_pod"]["normalized"]["spec"]["containers"][0][
        "environment"
    ]["literal"][name] = value
    (bundle / "capture.json").write_bytes(evidence.canonical_json_bytes(capture))

    with pytest.raises(evidence.EvidenceCaptureError, match="inconsistent"):
        evidence.verify_bundle(bundle)


def test_split_bundle_rejects_internally_consistent_alternate_valid_anchors(
    tmp_path: Path,
) -> None:
    evidence.main(_capture_args(tmp_path, kind="split"))
    bundle = tmp_path / "bundle"
    original_capture = json.loads((bundle / "capture.json").read_text())
    record = _record(kind="split")
    record["source_access"] = {
        "plan": {"sha256": "7" * 64, "pod_uid": "alternate-plan-pod"},
        "completion": {
            "sha256": "8" * 64,
            "pod_uid": "alternate-apply-pod",
        },
    }
    subject = {
        **evidence._validate_completion_record(
            record,
            split_source_git_sha=SPLIT_SOURCE_GIT_SHA,
        ),
        "record_schema": evidence.COMPLETION_SCHEMA,
    }
    pod = _pod(kind="split", subject=subject)
    record_payload = evidence.canonical_json_bytes(record)
    record_sha256 = hashlib.sha256(record_payload).hexdigest()
    marker_payload, _ = _generic_marker(
        evidence.TERMINAL_EVIDENCE_PREFIX,
        record,
    )
    pod_payload = json.dumps(pod).encode()
    normalized = evidence.normalize_observed_pod(
        pod,
        contract=evidence._workload_contract(subject),
        expected_namespace=NAMESPACE,
        expected_pod="ladder-lucas-crop-split-h8v2c",
    )
    rewritten_capture = evidence.build_capture_document(
        evidence_kind="split",
        subject=subject,
        record_sha256=record_sha256,
        marker_payload=marker_payload,
        pod_payload=pod_payload,
        normalized_pod=normalized,
        authority=evidence._git_authority(),
        operator=original_capture["capture_operator"],
    )
    (bundle / "completion.json").write_bytes(record_payload)
    (bundle / "completion.sha256").write_text(f"{record_sha256}\n")
    (bundle / "marker.txt").write_bytes(marker_payload)
    (bundle / "pod.json").write_bytes(pod_payload)
    (bundle / "pod.sha256").write_text(
        f"{hashlib.sha256(pod_payload).hexdigest()}\n"
    )
    (bundle / "capture.json").write_bytes(
        evidence.canonical_json_bytes(rewritten_capture)
    )

    with pytest.raises(
        evidence.EvidenceCaptureError,
        match="source_access authority differs from Git pins",
    ):
        evidence.verify_bundle(bundle)


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
        ("allow-privilege-escalation", "privilege/root-filesystem policy"),
        ("writable-root", "privilege/root-filesystem policy"),
        ("capability-add", "capabilities must contain exactly"),
        ("missing-drop-all", "capabilities differ"),
        ("container-seccomp-override", "securityContext must contain exactly"),
        ("pod-seccomp-unconfined", "seccompProfile must be RuntimeDefault"),
        ("supplemental-group", "securityContext must contain exactly"),
        ("host-network", "hostNetwork must be false"),
        ("service-account-token", "automountServiceAccountToken"),
        ("restart-policy", "restartPolicy must be Never"),
        ("sidecar", "exactly one container"),
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
        capture["observed_pod"]["normalized"]["status"]["container_statuses"][0][
            "image_id"
        ] = (
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

    with pytest.raises(evidence.EvidenceCaptureError):
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
    normalized = capture["observed_pod"]["normalized"]
    container = normalized["spec"]["containers"][0]
    if target == "security":
        container["security_context"]["run_as_user"] = evidence.STORAGE_UID
    elif target == "hardening":
        container["security_context"]["privileged"] = True
    elif target == "isolation":
        normalized["spec"]["isolation"]["host_network"] = True
    elif target == "restart":
        normalized["status"]["container_statuses"][0]["restart_count"] = False
    else:
        name = {
            "source": "CROP_DISTILL_SOURCE_GIT_SHA",
            "image": "CROP_DISTILL_IMAGE",
            "split": "CROP_DISTILL_SPLIT_MANIFEST_SHA256",
        }[target]
        container["environment"]["literal"][name] = (
            "f" * 40
            if target == "source"
            else (
                f"ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:{OTHER_DIGEST}"
                if target == "image"
                else OTHER_DIGEST
            )
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
    capture["observed_pod"]["normalized"]["status"]["container_statuses"][0][
        "terminated_exit_code"
    ] = False
    (bundle / "capture.json").write_bytes(evidence.canonical_json_bytes(capture))

    with pytest.raises(
        evidence.EvidenceCaptureError, match="inconsistent"
    ):
        evidence.verify_bundle(bundle)
