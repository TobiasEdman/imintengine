#!/usr/bin/env python3
"""Capture and verify durable crop-distill Kubernetes evidence.

The terminal completion record is emitted by the workload as a machine-readable
marker.  This helper extracts exactly one such marker from a captured Pod log,
binds it to the corresponding Kubernetes Pod, and creates a small immutable
evidence bundle outside the workload PVC.

This is an evidence-capture control, not an admission control.  It proves what
the Kubernetes API and workload log returned at capture time; it does not make
the scheduler or container runtime choose an image.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

if __package__:
    from .crop_distill_protocol import (
        BASE_PYTHON,
        CROP_HEADS_DIR,
        CROP_INDEX,
        CROP_SPLIT,
        CROP_SPLIT_MANIFEST,
        DISTILL_DIR,
        MODEL_KEYS,
        MODEL_PYTHON,
        RUNTIME_MANIFEST,
        SCORING_PYTHON,
        STORAGE_GID,
        STORAGE_UID,
        WORK_ROOT,
        model_process_uid,
        model_protocol,
    )
    from .crop_distill_provenance import (
        BASE_IMAGE,
        CLAY_ARCHIVE_SHA256,
        CLAY_GIT_SHA,
        COMPLETION_SCHEMA,
        CROMA_ARCHIVE_SHA256,
        CROMA_GIT_SHA,
        MODEL_RESOLUTION,
        TERMINAL_EVIDENCE_PREFIX,
        ProvenanceError,
        canonical_json_bytes,
        parse_terminal_evidence_line,
    )
else:
    from crop_distill_protocol import (  # type: ignore[no-redef]
        BASE_PYTHON,
        CROP_HEADS_DIR,
        CROP_INDEX,
        CROP_SPLIT,
        CROP_SPLIT_MANIFEST,
        DISTILL_DIR,
        MODEL_KEYS,
        MODEL_PYTHON,
        RUNTIME_MANIFEST,
        SCORING_PYTHON,
        STORAGE_GID,
        STORAGE_UID,
        WORK_ROOT,
        model_process_uid,
        model_protocol,
    )
    from crop_distill_provenance import (  # type: ignore[no-redef]
        BASE_IMAGE,
        CLAY_ARCHIVE_SHA256,
        CLAY_GIT_SHA,
        COMPLETION_SCHEMA,
        CROMA_ARCHIVE_SHA256,
        CROMA_GIT_SHA,
        MODEL_RESOLUTION,
        TERMINAL_EVIDENCE_PREFIX,
        ProvenanceError,
        canonical_json_bytes,
        parse_terminal_evidence_line,
    )

CAPTURE_SCHEMA = "imint-crop-distill-evidence-capture-v2"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")
_IMAGE_VALUE = re.compile(
    r"^(?:(?:containerd|docker-pullable)://)?"
    r"(?:[^\s@]+@)?sha256:([0-9a-f]{64})$"
)
_CROP_IMAGE_REF = re.compile(
    r"^ghcr\.io/tobiasedman/imint-ladder-crop-distill@sha256:"
    r"([0-9a-f]{64})$"
)
_CAPTURE_FILES = frozenset({"completion.json", "completion.sha256", "capture.json"})
_COMPLETION_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "model",
        "run_id",
        "job",
        "pod_uid",
        "process_identity",
        "terminal",
        "runtime",
        "split_manifest",
        "checkpoint",
        "artifacts",
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        "verification",
        "image",
        "base_image",
        "model_resolution",
        "base_python",
        "runtime_manifest",
        "source",
        "environments",
        "external_sources",
    }
)
_SPLIT_FILES = {
    "index": CROP_INDEX.name,
    "validator_holdout": "lucas_crop_validator_holdout_index.parquet",
    "split": CROP_SPLIT.name,
    "manifest": CROP_SPLIT_MANIFEST.name,
}
_SPLIT_DIGEST_FIELDS = frozenset(
    {
        "qualified_keys_sha256",
        "distill_keys_sha256",
        "holdout_keys_sha256",
        "partition_sha256",
        "prior_test_tiles_sha256",
        "prior_test_keys_sha256",
        "source_index_sha256",
        "forced_holdout_tiles_sha256",
        "forced_holdout_keys_sha256",
        "distill_input_data_sha256",
        "validator_holdout_input_data_sha256",
    }
)
_SOURCE_ENV = "CROP_DISTILL_SOURCE_GIT_SHA"
_IMAGE_ENV = "CROP_DISTILL_IMAGE"
_SPLIT_ENV = "CROP_DISTILL_SPLIT_MANIFEST_SHA256"
_POD_UID_ENV = "POD_UID"
_MAX_POD_JSON_BYTES = 8 * 1024 * 1024
_MAX_POD_LOG_BYTES = 64 * 1024 * 1024
_MAX_BUNDLE_FILE_BYTES = 16 * 1024 * 1024


class EvidenceCaptureError(ValueError):
    """Raised when live or archived evidence cannot be trusted."""


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceCaptureError(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvidenceCaptureError(f"{label} must be a JSON array")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceCaptureError(f"{label} must be a non-empty string")
    return value


def _require_safe_id(value: Any, label: str) -> str:
    value = _require_string(value, label)
    if _SAFE_ID.fullmatch(value) is None:
        raise EvidenceCaptureError(f"{label} is not a safe identifier")
    return value


def _require_exact_keys(
    value: Any,
    expected: set[str] | frozenset[str],
    label: str,
) -> dict[str, Any]:
    value = _require_object(value, label)
    if set(value) != set(expected):
        raise EvidenceCaptureError(f"{label} must contain exactly {sorted(expected)}")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise EvidenceCaptureError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise EvidenceCaptureError(f"{label} must be a non-negative integer")
    return value


def _require_hex40(value: Any, label: str, *, nonzero: bool = False) -> str:
    value = _require_string(value, label)
    if _HEX40.fullmatch(value) is None or (nonzero and value == "0" * 40):
        qualifier = "nonzero " if nonzero else ""
        raise EvidenceCaptureError(
            f"{label} must be {qualifier}40 lowercase hex digits"
        )
    return value


def _require_hex64(value: Any, label: str, *, nonzero: bool = False) -> str:
    value = _require_string(value, label)
    if _HEX64.fullmatch(value) is None or (nonzero and value == "0" * 64):
        qualifier = "nonzero " if nonzero else ""
        raise EvidenceCaptureError(
            f"{label} must be {qualifier}64 lowercase hex digits"
        )
    return value


def _require_absolute_path(value: Any, label: str) -> str:
    value = _require_string(value, label)
    path = Path(value)
    if not path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts[1:]
    ):
        raise EvidenceCaptureError(f"{label} must be a normalized absolute path")
    return value


def _validate_file_identity(
    value: Any,
    label: str,
    *,
    expected_path: Path | None = None,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    value = _require_exact_keys(value, {"path", "size_bytes", "sha256"}, label)
    path = _require_absolute_path(value.get("path"), f"{label} path")
    size = _require_positive_int(value.get("size_bytes"), f"{label} size_bytes")
    digest = _require_hex64(value.get("sha256"), f"{label} sha256")
    if expected_path is not None and path != str(expected_path):
        raise EvidenceCaptureError(f"{label} path must be exactly {expected_path}")
    if expected_size is not None and size != expected_size:
        raise EvidenceCaptureError(f"{label} size_bytes does not match the protocol")
    if expected_sha256 is not None and digest != expected_sha256:
        raise EvidenceCaptureError(f"{label} sha256 does not match the protocol")
    return value


def _validate_python_identity(
    value: Any,
    label: str,
    *,
    expected_path: Path,
) -> None:
    value = _require_exact_keys(
        value,
        {"implementation", "path", "version", "version_info"},
        label,
    )
    if value.get("implementation") != "CPython":
        raise EvidenceCaptureError(f"{label} implementation must be CPython")
    if value.get("path") != str(expected_path):
        raise EvidenceCaptureError(f"{label} path must be exactly {expected_path}")
    version = _require_string(value.get("version"), f"{label} version")
    version_info = _require_list(value.get("version_info"), f"{label} version_info")
    if (
        len(version_info) != 3
        or any(type(part) is not int for part in version_info)
        or version_info[:2] != [3, 11]
        or version_info[2] < 0
        or version != ".".join(str(part) for part in version_info)
    ):
        raise EvidenceCaptureError(f"{label} must describe one exact CPython 3.11")


def _validate_tree_identity(
    value: Any,
    label: str,
    *,
    expected_git_sha: str | None = None,
    expected_archive_sha256: str | None = None,
) -> dict[str, Any]:
    value = _require_exact_keys(
        value,
        {
            "archive_sha256",
            "payload_sha256",
            "files_manifest_sha256",
            "git_sha",
        },
        label,
    )
    git_sha = _require_hex40(value.get("git_sha"), f"{label} git_sha", nonzero=True)
    archive_sha256 = _require_hex64(
        value.get("archive_sha256"), f"{label} archive_sha256"
    )
    _require_hex64(value.get("payload_sha256"), f"{label} payload_sha256")
    _require_hex64(value.get("files_manifest_sha256"), f"{label} files_manifest_sha256")
    if expected_git_sha is not None and git_sha != expected_git_sha:
        raise EvidenceCaptureError(f"{label} git_sha does not match the protocol")
    if (
        expected_archive_sha256 is not None
        and archive_sha256 != expected_archive_sha256
    ):
        raise EvidenceCaptureError(
            f"{label} archive_sha256 does not match the protocol"
        )
    return value


def _validate_runtime(record: Any) -> dict[str, str]:
    runtime = _require_exact_keys(record, _RUNTIME_FIELDS, "completion runtime")
    if runtime.get("verification") != "verified":
        raise EvidenceCaptureError("completion runtime must be verified")

    image = _require_exact_keys(
        runtime.get("image"), {"digest", "ref"}, "completion runtime image"
    )
    image_ref = _require_string(image.get("ref"), "completion runtime image ref")
    match = _CROP_IMAGE_REF.fullmatch(image_ref)
    if match is None:
        raise EvidenceCaptureError(
            "completion runtime image ref must be the immutable crop-distill image"
        )
    image_digest = _require_hex64(
        image.get("digest"), "completion runtime image digest", nonzero=True
    )
    if match.group(1) != image_digest:
        raise EvidenceCaptureError("completion runtime image ref and digest disagree")

    if runtime.get("base_image") != BASE_IMAGE:
        raise EvidenceCaptureError("completion runtime base image is unexpected")
    if runtime.get("model_resolution") != MODEL_RESOLUTION:
        raise EvidenceCaptureError("completion runtime model resolution is unexpected")
    _validate_python_identity(
        runtime.get("base_python"),
        "completion runtime base Python",
        expected_path=BASE_PYTHON,
    )
    _validate_file_identity(
        runtime.get("runtime_manifest"),
        "completion runtime manifest",
        expected_path=RUNTIME_MANIFEST,
    )
    source = _validate_tree_identity(runtime.get("source"), "completion runtime source")
    source_git_sha = source["git_sha"]

    environments = _require_exact_keys(
        runtime.get("environments"),
        {"model", "scoring"},
        "completion runtime environments",
    )
    for name, python_path in (("model", MODEL_PYTHON), ("scoring", SCORING_PYTHON)):
        environment = _require_exact_keys(
            environments.get(name),
            {"python", "requirements_lock", "pip_freeze"},
            f"completion runtime {name} environment",
        )
        _validate_python_identity(
            environment.get("python"),
            f"completion runtime {name} Python",
            expected_path=python_path,
        )
        _validate_file_identity(
            environment.get("requirements_lock"),
            f"completion runtime {name} requirements lock",
        )
        _validate_file_identity(
            environment.get("pip_freeze"),
            f"completion runtime {name} pip freeze",
        )

    external = _require_exact_keys(
        runtime.get("external_sources"),
        {"clay", "croma"},
        "completion runtime external sources",
    )
    for name, git_sha, archive_sha256 in (
        ("clay", CLAY_GIT_SHA, CLAY_ARCHIVE_SHA256),
        ("croma", CROMA_GIT_SHA, CROMA_ARCHIVE_SHA256),
    ):
        _validate_tree_identity(
            external.get(name),
            f"completion runtime {name} source",
            expected_git_sha=git_sha,
            expected_archive_sha256=archive_sha256,
        )
    return {
        "digest": image_digest,
        "image_ref": image_ref,
        "source_git_sha": source_git_sha,
    }


def _validate_split_manifest(
    value: Any,
    *,
    kind: str,
    pod_uid: str,
    source_git_sha: str,
) -> dict[str, Any]:
    split = _require_exact_keys(
        value,
        {
            "path",
            "size_bytes",
            "sha256",
            "git_sha",
            "counts",
            "immutable_digests",
            "declared_artifacts",
        },
        "completion split_manifest",
    )
    expected_parent = DISTILL_DIR if kind == "split" else WORK_ROOT / pod_uid / "split"
    expected_manifest = expected_parent / CROP_SPLIT_MANIFEST.name
    identity = {name: split[name] for name in ("path", "size_bytes", "sha256")}
    _validate_file_identity(
        identity,
        "completion split_manifest identity",
        expected_path=expected_manifest,
    )
    split_sha256 = _require_hex64(
        identity["sha256"], "completion split_manifest sha256", nonzero=True
    )
    if split.get("git_sha") != source_git_sha:
        raise EvidenceCaptureError(
            "completion split_manifest git_sha must match runtime source"
        )

    counts = _require_exact_keys(
        split.get("counts"),
        {"n_qualified", "n_distill", "n_holdout"},
        "completion split_manifest counts",
    )
    qualified = _require_positive_int(
        counts.get("n_qualified"), "completion split_manifest n_qualified"
    )
    distill = _require_positive_int(
        counts.get("n_distill"), "completion split_manifest n_distill"
    )
    holdout = _require_positive_int(
        counts.get("n_holdout"), "completion split_manifest n_holdout"
    )
    if qualified != distill + holdout:
        raise EvidenceCaptureError("completion split_manifest counts do not add up")

    digests = _require_exact_keys(
        split.get("immutable_digests"),
        _SPLIT_DIGEST_FIELDS,
        "completion split_manifest immutable_digests",
    )
    for name in sorted(_SPLIT_DIGEST_FIELDS):
        _require_hex64(digests.get(name), f"completion split_manifest {name}")

    declared = _require_exact_keys(
        split.get("declared_artifacts"),
        {filename for name, filename in _SPLIT_FILES.items() if name != "manifest"},
        "completion split_manifest declared_artifacts",
    )
    for logical_name in ("index", "validator_holdout", "split"):
        filename = _SPLIT_FILES[logical_name]
        artifact = _require_object(
            declared.get(filename),
            f"completion split_manifest declared artifact {filename}",
        )
        declaration_only = kind == "crop" and logical_name == "validator_holdout"
        expected_keys = {"path", "sha256", "verification"}
        if not declaration_only:
            expected_keys.add("size_bytes")
        artifact = _require_exact_keys(
            artifact,
            expected_keys,
            f"completion split_manifest declared artifact {filename}",
        )
        if artifact.get("path") != str(expected_parent / filename):
            raise EvidenceCaptureError(
                f"completion split_manifest declared artifact {filename} has "
                "an unexpected protocol path"
            )
        _require_hex64(
            artifact.get("sha256"),
            f"completion split_manifest declared artifact {filename} sha256",
        )
        expected_verification = "declaration-only" if declaration_only else "content"
        if artifact.get("verification") != expected_verification:
            raise EvidenceCaptureError(
                f"completion split_manifest declared artifact {filename} must use "
                f"{expected_verification} verification"
            )
        if not declaration_only:
            _require_positive_int(
                artifact.get("size_bytes"),
                f"completion split_manifest declared artifact {filename} size_bytes",
            )
    return {
        "declared_artifacts": declared,
        "identity": identity,
        "sha256": split_sha256,
    }


def _read_regular_file(path: Path, label: str, *, limit: int) -> bytes:
    """Read one bounded regular file without following its final symlink."""
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise EvidenceCaptureError(f"cannot open {label} {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceCaptureError(f"{label} must be a regular file: {path}")
        if before.st_size > limit:
            raise EvidenceCaptureError(
                f"{label} exceeds the {limit}-byte capture limit: {path}"
            )
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > limit:
            raise EvidenceCaptureError(
                f"{label} exceeds the {limit}-byte capture limit: {path}"
            )
        after = os.fstat(fd)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity or len(payload) != after.st_size:
            raise EvidenceCaptureError(f"{label} changed while it was read: {path}")
        return payload
    except OSError as exc:
        raise EvidenceCaptureError(f"cannot read {label} {path}: {exc}") from exc
    finally:
        os.close(fd)


def _load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceCaptureError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    return _require_object(value, label)


def normalize_image_digest(value: Any, label: str) -> str:
    """Normalize accepted Kubernetes image forms to one lowercase hex digest."""
    value = _require_string(value, label)
    match = _IMAGE_VALUE.fullmatch(value)
    if match is None:
        raise EvidenceCaptureError(f"{label} must be an immutable sha256 image value")
    return match.group(1)


def extract_terminal_record(pod_log: bytes) -> tuple[bytes, str, dict[str, Any]]:
    """Extract and authenticate exactly one terminal marker from a Pod log."""
    try:
        text = pod_log.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceCaptureError("Pod log is not valid UTF-8") from exc

    marker_lines = [
        line for line in text.splitlines() if TERMINAL_EVIDENCE_PREFIX in line
    ]
    if len(marker_lines) != 1:
        raise EvidenceCaptureError(
            "Pod log must contain exactly one terminal evidence marker; "
            f"found {len(marker_lines)}"
        )
    line = marker_lines[0]
    try:
        actual_sha256, payload, record = parse_terminal_evidence_line(line)
    except ProvenanceError as exc:
        raise EvidenceCaptureError(
            f"terminal evidence marker is invalid: {exc}"
        ) from exc
    encoded = line.split(" ")[2]
    if base64.b64encode(payload).decode("ascii") != encoded:
        raise EvidenceCaptureError(
            "terminal evidence marker payload is not canonical base64"
        )
    return payload, actual_sha256, record


def _validate_completion_record(record: dict[str, Any]) -> dict[str, Any]:
    _require_exact_keys(record, _COMPLETION_FIELDS, "completion record")
    if record.get("schema") != COMPLETION_SCHEMA:
        raise EvidenceCaptureError("unexpected terminal completion schema")
    kind = record.get("kind")
    if kind not in {"crop", "split"}:
        raise EvidenceCaptureError("completion kind must be crop or split")
    pod_uid = _require_safe_id(record.get("pod_uid"), "completion pod_uid")
    run_id = _require_safe_id(record.get("run_id"), "completion run_id")
    if run_id != pod_uid:
        raise EvidenceCaptureError("completion run_id must equal pod_uid")
    job = _require_safe_id(record.get("job"), "completion job")
    model = record.get("model")
    if kind == "crop":
        model = _require_safe_id(model, "completion model")
        if model not in MODEL_KEYS:
            raise EvidenceCaptureError("completion model is not an allowed crop model")
        if job != f"ladder-crop-distill-{model}":
            raise EvidenceCaptureError("crop completion job does not match model")
        container = "crop-distill"
        expected_uid = model_process_uid(model)
    else:
        if model is not None:
            raise EvidenceCaptureError("split completion model must be null")
        if job != "ladder-lucas-crop-split":
            raise EvidenceCaptureError("split completion has an unexpected job")
        container = "split"
        expected_uid = STORAGE_UID

    process_identity = _require_exact_keys(
        record.get("process_identity"),
        {"effective_uid", "effective_gid"},
        "completion process_identity",
    )
    effective_uid = _require_nonnegative_int(
        process_identity.get("effective_uid"), "completion effective_uid"
    )
    effective_gid = _require_nonnegative_int(
        process_identity.get("effective_gid"), "completion effective_gid"
    )
    if (effective_uid, effective_gid) != (expected_uid, STORAGE_GID):
        raise EvidenceCaptureError(
            "completion process_identity does not match the model protocol"
        )

    terminal = _require_exact_keys(
        record.get("terminal"),
        {"status", "exit_code", "failure_stage"},
        "completion terminal",
    )
    if terminal.get("status") != "completed":
        raise EvidenceCaptureError("only completed terminal records can be archived")
    if type(terminal.get("exit_code")) is not int or terminal.get("exit_code") != 0:
        raise EvidenceCaptureError("completed terminal record must have exit_code 0")
    if terminal.get("failure_stage") is not None:
        raise EvidenceCaptureError(
            "completed terminal record must not have a failure_stage"
        )

    runtime = _validate_runtime(record.get("runtime"))
    split = _validate_split_manifest(
        record.get("split_manifest"),
        kind=kind,
        pod_uid=pod_uid,
        source_git_sha=runtime["source_git_sha"],
    )

    checkpoint = record.get("checkpoint")
    artifacts = _require_object(record.get("artifacts"), "completion artifacts")
    if kind == "crop":
        assert isinstance(model, str)
        protocol = model_protocol(model)
        checkpoint = _require_exact_keys(
            checkpoint,
            {"path", "size_bytes", "sha256", "verification"},
            "completion checkpoint",
        )
        if checkpoint.get("verification") != (
            "extractor-authenticated-private-snapshot"
        ):
            raise EvidenceCaptureError("completion checkpoint verification is invalid")
        _validate_file_identity(
            {name: checkpoint[name] for name in ("path", "size_bytes", "sha256")},
            "completion checkpoint identity",
            expected_path=protocol.checkpoint_path,
            expected_size=protocol.checkpoint_size,
            expected_sha256=protocol.checkpoint_sha256,
        )
        expected_artifacts = {
            "features": CROP_HEADS_DIR / f"{pod_uid}--{model}_r2_crop_features.parquet",
            "oof": CROP_HEADS_DIR / f"{pod_uid}--{model}_r2_crop_distillability.json",
        }
        if set(artifacts) != set(expected_artifacts):
            raise EvidenceCaptureError(
                "completion crop artifacts must be exactly features and oof"
            )
        for name, path in expected_artifacts.items():
            _validate_file_identity(
                artifacts.get(name),
                f"completion crop artifact {name}",
                expected_path=path,
            )
    else:
        if checkpoint is not None:
            raise EvidenceCaptureError("completion split checkpoint must be null")
        if set(artifacts) != set(_SPLIT_FILES):
            raise EvidenceCaptureError(
                "completion split artifacts must be exactly index, "
                "validator_holdout, split, and manifest"
            )
        declared = split["declared_artifacts"]
        expected_paths = {
            "index": CROP_INDEX,
            "validator_holdout": DISTILL_DIR / _SPLIT_FILES["validator_holdout"],
            "split": CROP_SPLIT,
            "manifest": CROP_SPLIT_MANIFEST,
        }
        for name, path in expected_paths.items():
            artifact = _validate_file_identity(
                artifacts.get(name),
                f"completion split artifact {name}",
                expected_path=path,
            )
            if name == "manifest":
                expected_identity = split["identity"]
            else:
                declared_identity = declared[_SPLIT_FILES[name]]
                expected_identity = {
                    key: declared_identity[key]
                    for key in ("path", "size_bytes", "sha256")
                }
            if artifact != expected_identity:
                raise EvidenceCaptureError(
                    f"completion split artifact {name} disagrees with split_manifest"
                )
    return {
        "container": container,
        "digest": runtime["digest"],
        "effective_gid": effective_gid,
        "effective_uid": effective_uid,
        "image_ref": runtime["image_ref"],
        "job": job,
        "kind": kind,
        "model": model,
        "pod_uid": pod_uid,
        "run_id": run_id,
        "source_git_sha": runtime["source_git_sha"],
        "split_manifest_sha256": split["sha256"],
    }


def _named_entry(entries: Any, target: str, label: str) -> dict[str, Any]:
    entries = _require_list(entries, label)
    objects = [_require_object(entry, f"{label} entry") for entry in entries]
    matches = [entry for entry in objects if entry.get("name") == target]
    if len(matches) != 1:
        raise EvidenceCaptureError(
            f"{label} must contain target {target!r} exactly once; found {len(matches)}"
        )
    return matches[0]


def _require_single_target_container(pod: dict[str, Any], target: str) -> None:
    """Reject sidecars, init containers, and ambiguous status histories."""
    spec = _require_object(pod.get("spec"), "Pod spec")
    status = _require_object(pod.get("status"), "Pod status")
    containers = _require_list(spec.get("containers"), "Pod containers")
    if (
        len(containers) != 1
        or _require_object(containers[0], "Pod container").get("name") != target
    ):
        raise EvidenceCaptureError(
            "Pod must contain exactly the target container and no sidecars"
        )
    statuses = _require_list(status.get("containerStatuses"), "Pod containerStatuses")
    if (
        len(statuses) != 1
        or _require_object(statuses[0], "Pod containerStatus").get("name") != target
    ):
        raise EvidenceCaptureError(
            "Pod status must contain exactly the target container"
        )
    for owner, field in (
        (spec, "initContainers"),
        (spec, "ephemeralContainers"),
        (status, "initContainerStatuses"),
        (status, "ephemeralContainerStatuses"),
    ):
        entries = owner.get(field, [])
        entries = _require_list(entries, f"Pod {field}")
        if entries:
            raise EvidenceCaptureError(
                f"Pod {field} must be empty for evidence capture"
            )


def _env_entry(
    entries: list[dict[str, Any]], name: str, *, required: bool
) -> dict[str, Any] | None:
    matches = [entry for entry in entries if entry.get("name") == name]
    expected = 1 if required else 0
    if len(matches) != expected:
        requirement = "exactly once" if required else "not be present"
        raise EvidenceCaptureError(
            f"target container environment {name} must {requirement}; "
            f"found {len(matches)}"
        )
    return matches[0] if matches else None


def _literal_env_value(entries: list[dict[str, Any]], name: str) -> str:
    entry = _env_entry(entries, name, required=True)
    assert entry is not None
    entry = _require_exact_keys(
        entry, {"name", "value"}, f"target container environment {name}"
    )
    return _require_string(entry.get("value"), f"target container environment {name}")


def _container_authority_environment(
    spec_container: dict[str, Any],
    *,
    kind: str,
) -> tuple[dict[str, str], dict[str, str]]:
    raw_entries = _require_list(
        spec_container.get("env"), "target container environment"
    )
    entries = [
        _require_object(entry, "target container environment entry")
        for entry in raw_entries
    ]
    literal = {
        _SOURCE_ENV: _literal_env_value(entries, _SOURCE_ENV),
        _IMAGE_ENV: _literal_env_value(entries, _IMAGE_ENV),
    }
    if kind == "crop":
        literal[_SPLIT_ENV] = _literal_env_value(entries, _SPLIT_ENV)
    else:
        _env_entry(entries, _SPLIT_ENV, required=False)

    pod_uid = _env_entry(entries, _POD_UID_ENV, required=True)
    assert pod_uid is not None
    pod_uid = _require_exact_keys(
        pod_uid,
        {"name", "valueFrom"},
        f"target container environment {_POD_UID_ENV}",
    )
    value_from = _require_exact_keys(
        pod_uid.get("valueFrom"),
        {"fieldRef"},
        f"target container environment {_POD_UID_ENV} valueFrom",
    )
    field_ref = _require_object(
        value_from.get("fieldRef"),
        f"target container environment {_POD_UID_ENV} fieldRef",
    )
    if set(field_ref) not in ({"fieldPath"}, {"apiVersion", "fieldPath"}):
        raise EvidenceCaptureError(
            f"target container environment {_POD_UID_ENV} fieldRef is unexpected"
        )
    if field_ref.get("fieldPath") != "metadata.uid":
        raise EvidenceCaptureError(
            f"target container environment {_POD_UID_ENV} must reference metadata.uid"
        )
    api_version = field_ref.get("apiVersion", "v1")
    if api_version != "v1":
        raise EvidenceCaptureError(
            f"target container environment {_POD_UID_ENV} fieldRef must use v1"
        )
    return literal, {"api_version": "v1", "field_path": "metadata.uid"}


def _validate_pod_security_context(
    value: Any,
    label: str,
    *,
    expected_uid: int,
    expected_gid: int,
) -> dict[str, Any]:
    value = _require_exact_keys(
        value,
        {"runAsUser", "runAsGroup", "runAsNonRoot", "seccompProfile"},
        label,
    )
    uid = _require_nonnegative_int(value.get("runAsUser"), f"{label} runAsUser")
    gid = _require_nonnegative_int(value.get("runAsGroup"), f"{label} runAsGroup")
    if (uid, gid) != (expected_uid, expected_gid):
        raise EvidenceCaptureError(f"{label} UID:GID does not match completion record")
    if value.get("runAsNonRoot") is not True:
        raise EvidenceCaptureError(f"{label} runAsNonRoot must be true")
    seccomp = _require_exact_keys(
        value.get("seccompProfile"), {"type"}, f"{label} seccompProfile"
    )
    if seccomp.get("type") != "RuntimeDefault":
        raise EvidenceCaptureError(f"{label} seccompProfile must be RuntimeDefault")
    return {
        "run_as_group": gid,
        "run_as_non_root": True,
        "run_as_user": uid,
        "seccomp_profile": {"type": "RuntimeDefault"},
    }


def _validate_container_security_context(
    value: Any,
    label: str,
    *,
    expected_uid: int,
    expected_gid: int,
) -> dict[str, Any]:
    value = _require_exact_keys(
        value,
        {
            "allowPrivilegeEscalation",
            "capabilities",
            "readOnlyRootFilesystem",
            "runAsGroup",
            "runAsNonRoot",
            "runAsUser",
        },
        label,
    )
    uid = _require_nonnegative_int(value.get("runAsUser"), f"{label} runAsUser")
    gid = _require_nonnegative_int(value.get("runAsGroup"), f"{label} runAsGroup")
    if (uid, gid) != (expected_uid, expected_gid):
        raise EvidenceCaptureError(f"{label} UID:GID does not match completion record")
    if value.get("runAsNonRoot") is not True:
        raise EvidenceCaptureError(f"{label} runAsNonRoot must be true")
    if value.get("allowPrivilegeEscalation") is not False:
        raise EvidenceCaptureError(f"{label} allowPrivilegeEscalation must be false")
    if value.get("readOnlyRootFilesystem") is not True:
        raise EvidenceCaptureError(f"{label} readOnlyRootFilesystem must be true")
    capabilities = _require_exact_keys(
        value.get("capabilities"), {"drop"}, f"{label} capabilities"
    )
    if capabilities.get("drop") != ["ALL"]:
        raise EvidenceCaptureError(f"{label} must drop exactly ALL capabilities")
    return {
        "allow_privilege_escalation": False,
        "capabilities": {"add": [], "drop": ["ALL"]},
        "privileged": False,
        "read_only_root_filesystem": True,
        "run_as_group": gid,
        "run_as_non_root": True,
        "run_as_user": uid,
        "seccomp_profile": {"source": "pod", "type": "RuntimeDefault"},
    }


def _validate_pod_isolation(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("automountServiceAccountToken") is not False:
        raise EvidenceCaptureError("Pod automountServiceAccountToken must be false")
    if spec.get("restartPolicy") != "Never":
        raise EvidenceCaptureError("Pod restartPolicy must be Never")
    isolation: dict[str, Any] = {
        "automount_service_account_token": False,
        "restart_policy": "Never",
    }
    for field, archived in (
        ("hostNetwork", "host_network"),
        ("hostPID", "host_pid"),
        ("hostIPC", "host_ipc"),
    ):
        value = spec.get(field, False)
        if type(value) is not bool or value:
            raise EvidenceCaptureError(f"Pod {field} must be false")
        isolation[archived] = False
    return isolation


def build_capture(
    pod: dict[str, Any],
    record: dict[str, Any],
    *,
    record_sha256: str,
    container: str,
    expected_namespace: str,
    expected_pod: str,
    expected_job: str,
) -> dict[str, Any]:
    """Bind a completed record to the exact observed Pod and runtime image."""
    completion = _validate_completion_record(record)
    if container != completion["container"]:
        raise EvidenceCaptureError(
            f"container must be {completion['container']!r} for a "
            f"{completion['kind']} record"
        )
    if expected_job != completion["job"]:
        raise EvidenceCaptureError("expected job disagrees with completion record")
    if pod.get("apiVersion") != "v1" or pod.get("kind") != "Pod":
        raise EvidenceCaptureError("captured Kubernetes object must be a v1 Pod")

    metadata = _require_object(pod.get("metadata"), "Pod metadata")
    namespace = _require_safe_id(metadata.get("namespace"), "Pod namespace")
    pod_name = _require_safe_id(metadata.get("name"), "Pod name")
    pod_uid = _require_safe_id(metadata.get("uid"), "Pod UID")
    if namespace != expected_namespace:
        raise EvidenceCaptureError("Pod namespace does not match --expected-namespace")
    if pod_name != expected_pod:
        raise EvidenceCaptureError("Pod name does not match --expected-pod")
    if pod_uid != completion["pod_uid"]:
        raise EvidenceCaptureError("Pod UID does not match completion record")

    labels = _require_object(metadata.get("labels"), "Pod labels")
    job_labels: dict[str, str] = {}
    for key in ("batch.kubernetes.io/job-name", "job-name"):
        if key in labels:
            value = _require_string(labels[key], f"Pod label {key}")
            if value != completion["job"]:
                raise EvidenceCaptureError(
                    f"Pod label {key} does not match completion job"
                )
            job_labels[key] = value
    if "batch.kubernetes.io/job-name" not in job_labels:
        raise EvidenceCaptureError(
            "Pod lacks the batch.kubernetes.io/job-name binding label"
        )

    owner_references = _require_list(
        metadata.get("ownerReferences"), "Pod ownerReferences"
    )
    owner_references = [
        _require_object(owner, "Pod ownerReference") for owner in owner_references
    ]
    controllers = [
        owner for owner in owner_references if owner.get("controller") is True
    ]
    if len(controllers) != 1:
        raise EvidenceCaptureError(
            "Pod must have exactly one controller ownerReference"
        )
    controller = controllers[0]
    if (
        controller.get("apiVersion") != "batch/v1"
        or controller.get("kind") != "Job"
        or controller.get("name") != completion["job"]
    ):
        raise EvidenceCaptureError(
            "Pod controller ownerReference does not match completion Job"
        )
    job_uid = _require_safe_id(controller.get("uid"), "Job controller UID")

    _require_single_target_container(pod, container)
    spec = _require_object(pod.get("spec"), "Pod spec")
    status = _require_object(pod.get("status"), "Pod status")
    pod_isolation = _validate_pod_isolation(spec)
    spec_container = _named_entry(spec.get("containers"), container, "Pod containers")
    status_container = _named_entry(
        status.get("containerStatuses"), container, "Pod containerStatuses"
    )
    literal_env, pod_uid_field_ref = _container_authority_environment(
        spec_container,
        kind=completion["kind"],
    )
    if literal_env[_SOURCE_ENV] != completion["source_git_sha"]:
        raise EvidenceCaptureError(
            f"Pod literal {_SOURCE_ENV} does not match completion runtime source"
        )
    if literal_env[_IMAGE_ENV] != completion["image_ref"]:
        raise EvidenceCaptureError(
            f"Pod literal {_IMAGE_ENV} does not match completion runtime image"
        )
    if completion["kind"] == "crop":
        split_anchor = _require_hex64(
            literal_env[_SPLIT_ENV], f"Pod literal {_SPLIT_ENV}", nonzero=True
        )
        if split_anchor != completion["split_manifest_sha256"]:
            raise EvidenceCaptureError(
                f"Pod literal {_SPLIT_ENV} does not match completion split_manifest"
            )

    pod_security_context = _validate_pod_security_context(
        spec.get("securityContext"),
        "Pod securityContext",
        expected_uid=completion["effective_uid"],
        expected_gid=completion["effective_gid"],
    )
    container_security_context = _validate_container_security_context(
        spec_container.get("securityContext"),
        "target container securityContext",
        expected_uid=completion["effective_uid"],
        expected_gid=completion["effective_gid"],
    )
    spec_image = _require_string(spec_container.get("image"), "Pod spec image")
    status_image_id = _require_string(
        status_container.get("imageID"), "Pod status imageID"
    )
    spec_digest = normalize_image_digest(spec_image, "Pod spec image")
    status_digest = normalize_image_digest(status_image_id, "Pod status imageID")
    expected_digest = completion["digest"]
    if spec_image != completion["image_ref"]:
        raise EvidenceCaptureError(
            "Pod spec image must exactly match completion runtime image ref"
        )
    if spec_digest != expected_digest:
        raise EvidenceCaptureError(
            "Pod spec image digest does not match completion image digest"
        )
    if status_digest != expected_digest:
        raise EvidenceCaptureError(
            "Pod status imageID digest does not match completion image digest"
        )

    if status.get("phase") != "Succeeded":
        raise EvidenceCaptureError("Pod phase must be Succeeded")
    restart_count = status_container.get("restartCount")
    if type(restart_count) is not int or restart_count != 0:
        raise EvidenceCaptureError("target container restartCount must be zero")
    last_state = _require_object(
        status_container.get("lastState", {}), "target container lastState"
    )
    if last_state:
        raise EvidenceCaptureError("target container lastState must be empty")
    state = _require_object(status_container.get("state"), "container state")
    if set(state) != {"terminated"}:
        raise EvidenceCaptureError(
            "target container must have exactly one terminated state"
        )
    terminated = _require_object(state["terminated"], "terminated state")
    if type(terminated.get("exitCode")) is not int or terminated.get("exitCode") != 0:
        raise EvidenceCaptureError("target container exitCode must be 0")
    terminated_reason = terminated.get("reason")
    if terminated_reason is not None:
        terminated_reason = _require_string(
            terminated_reason, "terminated state reason"
        )

    return {
        "schema": CAPTURE_SCHEMA,
        "completion": {
            "image_digest": expected_digest,
            "job": completion["job"],
            "kind": completion["kind"],
            "model": completion["model"],
            "pod_uid": completion["pod_uid"],
            "process_identity": {
                "effective_gid": completion["effective_gid"],
                "effective_uid": completion["effective_uid"],
            },
            "record_sha256": record_sha256,
            "run_id": completion["run_id"],
            "schema": COMPLETION_SCHEMA,
            "source_git_sha": completion["source_git_sha"],
            "split_manifest_sha256": completion["split_manifest_sha256"],
        },
        "kubernetes": {
            "api_version": "v1",
            "container": {
                "environment": {
                    "literal": dict(sorted(literal_env.items())),
                    "pod_uid_field_ref": pod_uid_field_ref,
                },
                "name": container,
                "security_context": container_security_context,
                "spec_image": spec_image,
                "spec_image_digest": spec_digest,
                "status_image_id": status_image_id,
                "status_image_digest": status_digest,
                "terminated_exit_code": 0,
                "terminated_reason": terminated_reason,
                "restart_count": 0,
                "last_state": {},
            },
            "job": {
                "labels": dict(sorted(job_labels.items())),
                "name": completion["job"],
                "uid": job_uid,
            },
            "namespace": namespace,
            "pod": {
                "name": pod_name,
                "isolation": pod_isolation,
                "phase": "Succeeded",
                "security_context": pod_security_context,
                "uid": pod_uid,
            },
        },
        "source": {
            "completion": "pod-log-terminal-marker-v1",
            "pod": "kubernetes-api-v1",
        },
    }


def _write_new_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd: int | None = None
    try:
        fd = os.open(path, flags, 0o644)
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    except OSError as exc:
        raise EvidenceCaptureError(
            f"cannot create evidence file {path}: {exc}"
        ) from exc
    finally:
        if fd is not None:
            os.close(fd)


def write_bundle(
    out_dir: Path,
    completion_payload: bytes,
    completion_sha256: str,
    capture: dict[str, Any],
) -> None:
    """Create one evidence directory without replacing any existing evidence."""
    try:
        out_dir.mkdir(mode=0o755)
    except OSError as exc:
        raise EvidenceCaptureError(
            f"evidence output directory must be new: {out_dir}: {exc}"
        ) from exc
    created: list[Path] = []
    try:
        for name, payload in (
            ("completion.json", completion_payload),
            ("completion.sha256", f"{completion_sha256}\n".encode("ascii")),
            ("capture.json", canonical_json_bytes(capture)),
        ):
            path = out_dir / name
            _write_new_file(path, payload)
            created.append(path)
        directory_fd = os.open(out_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                pass
        try:
            out_dir.rmdir()
        except OSError:
            pass
        raise


def _verify_capture_document(
    capture: dict[str, Any],
    record: dict[str, Any],
    record_sha256: str,
) -> None:
    _require_exact_keys(
        capture,
        {"schema", "completion", "kubernetes", "source"},
        "capture document",
    )
    if capture.get("schema") != CAPTURE_SCHEMA:
        raise EvidenceCaptureError("unexpected capture.json schema")
    completion = _validate_completion_record(record)
    archived_completion = _require_object(
        capture.get("completion"), "capture completion"
    )
    expected_completion = {
        "image_digest": completion["digest"],
        "job": completion["job"],
        "kind": completion["kind"],
        "model": completion["model"],
        "pod_uid": completion["pod_uid"],
        "process_identity": {
            "effective_gid": completion["effective_gid"],
            "effective_uid": completion["effective_uid"],
        },
        "record_sha256": record_sha256,
        "run_id": completion["run_id"],
        "schema": COMPLETION_SCHEMA,
        "source_git_sha": completion["source_git_sha"],
        "split_manifest_sha256": completion["split_manifest_sha256"],
    }
    if archived_completion != expected_completion:
        raise EvidenceCaptureError(
            "capture completion binding disagrees with completion.json"
        )
    if capture.get("source") != {
        "completion": "pod-log-terminal-marker-v1",
        "pod": "kubernetes-api-v1",
    }:
        raise EvidenceCaptureError("capture source description is unexpected")

    kubernetes = _require_exact_keys(
        capture.get("kubernetes"),
        {"api_version", "container", "job", "namespace", "pod"},
        "capture kubernetes",
    )
    if kubernetes.get("api_version") != "v1":
        raise EvidenceCaptureError("capture Kubernetes API version is unexpected")
    _require_safe_id(kubernetes.get("namespace"), "capture namespace")
    pod = _require_exact_keys(
        kubernetes.get("pod"),
        {"isolation", "name", "phase", "security_context", "uid"},
        "capture pod",
    )
    _require_safe_id(pod.get("name"), "capture Pod name")
    if pod.get("uid") != completion["pod_uid"] or pod.get("phase") != "Succeeded":
        raise EvidenceCaptureError("capture Pod identity or phase is inconsistent")
    expected_pod_security_context = {
        "run_as_group": completion["effective_gid"],
        "run_as_non_root": True,
        "run_as_user": completion["effective_uid"],
        "seccomp_profile": {"type": "RuntimeDefault"},
    }
    pod_security_context = _require_object(
        pod.get("security_context"), "capture Pod security context"
    )
    if (
        type(pod_security_context.get("run_as_group")) is not int
        or type(pod_security_context.get("run_as_user")) is not int
        or pod_security_context.get("run_as_non_root") is not True
        or pod_security_context != expected_pod_security_context
    ):
        raise EvidenceCaptureError("capture Pod security context is inconsistent")
    isolation = _require_exact_keys(
        pod.get("isolation"),
        {
            "automount_service_account_token",
            "host_ipc",
            "host_network",
            "host_pid",
            "restart_policy",
        },
        "capture Pod isolation",
    )
    if (
        isolation.get("automount_service_account_token") is not False
        or isolation.get("host_ipc") is not False
        or isolation.get("host_network") is not False
        or isolation.get("host_pid") is not False
        or isolation.get("restart_policy") != "Never"
    ):
        raise EvidenceCaptureError("capture Pod isolation is inconsistent")
    job = _require_exact_keys(
        kubernetes.get("job"),
        {"labels", "name", "uid"},
        "capture job",
    )
    if job.get("name") != completion["job"]:
        raise EvidenceCaptureError("capture Job name is inconsistent")
    _require_safe_id(job.get("uid"), "capture Job UID")
    labels = _require_object(job.get("labels"), "capture Job labels")
    if labels.get("batch.kubernetes.io/job-name") != completion["job"]:
        raise EvidenceCaptureError("capture Job binding label is inconsistent")
    if any(value != completion["job"] for value in labels.values()):
        raise EvidenceCaptureError("capture Job labels disagree")

    container = _require_exact_keys(
        kubernetes.get("container"),
        {
            "environment",
            "name",
            "last_state",
            "restart_count",
            "security_context",
            "spec_image",
            "spec_image_digest",
            "status_image_id",
            "status_image_digest",
            "terminated_exit_code",
            "terminated_reason",
        },
        "capture container",
    )
    if container.get("name") != completion["container"]:
        raise EvidenceCaptureError("capture container name is inconsistent")
    expected_container_security_context = {
        "allow_privilege_escalation": False,
        "capabilities": {"add": [], "drop": ["ALL"]},
        "privileged": False,
        "read_only_root_filesystem": True,
        "run_as_group": completion["effective_gid"],
        "run_as_non_root": True,
        "run_as_user": completion["effective_uid"],
        "seccomp_profile": {"source": "pod", "type": "RuntimeDefault"},
    }
    container_security_context = _require_object(
        container.get("security_context"), "capture container security context"
    )
    if (
        type(container_security_context.get("run_as_group")) is not int
        or type(container_security_context.get("run_as_user")) is not int
        or container_security_context.get("run_as_non_root") is not True
        or container_security_context.get("allow_privilege_escalation") is not False
        or container_security_context.get("privileged") is not False
        or container_security_context.get("read_only_root_filesystem") is not True
        or container_security_context != expected_container_security_context
    ):
        raise EvidenceCaptureError("capture container security context is inconsistent")
    environment = _require_exact_keys(
        container.get("environment"),
        {"literal", "pod_uid_field_ref"},
        "capture container environment",
    )
    expected_literal = {
        _SOURCE_ENV: completion["source_git_sha"],
        _IMAGE_ENV: completion["image_ref"],
    }
    if completion["kind"] == "crop":
        expected_literal[_SPLIT_ENV] = completion["split_manifest_sha256"]
    if environment.get("literal") != dict(sorted(expected_literal.items())):
        raise EvidenceCaptureError("capture literal Pod environment is inconsistent")
    if environment.get("pod_uid_field_ref") != {
        "api_version": "v1",
        "field_path": "metadata.uid",
    }:
        raise EvidenceCaptureError("capture POD_UID fieldRef is inconsistent")
    if container.get("spec_image") != completion["image_ref"]:
        raise EvidenceCaptureError("capture Pod spec image ref is inconsistent")
    spec_digest = normalize_image_digest(
        container.get("spec_image"), "capture Pod spec image"
    )
    status_digest = normalize_image_digest(
        container.get("status_image_id"), "capture Pod status imageID"
    )
    if (
        spec_digest != completion["digest"]
        or status_digest != completion["digest"]
        or container.get("spec_image_digest") != spec_digest
        or container.get("status_image_digest") != status_digest
    ):
        raise EvidenceCaptureError("capture image digest binding is inconsistent")
    if type(container.get("restart_count")) is not int or (
        container.get("restart_count") != 0
    ):
        raise EvidenceCaptureError("capture target container restart count is invalid")
    if container.get("last_state") != {}:
        raise EvidenceCaptureError("capture target container last state is invalid")
    if type(container.get("terminated_exit_code")) is not int or (
        container.get("terminated_exit_code") != 0
    ):
        raise EvidenceCaptureError("capture target container did not exit successfully")
    reason = container.get("terminated_reason")
    if reason is not None:
        _require_string(reason, "capture terminated reason")


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Verify an archived bundle without needing the live cluster or PVC."""
    try:
        entries = {entry.name for entry in bundle_dir.iterdir()}
    except OSError as exc:
        raise EvidenceCaptureError(
            f"cannot inspect evidence bundle {bundle_dir}: {exc}"
        ) from exc
    if entries != _CAPTURE_FILES:
        raise EvidenceCaptureError(
            "evidence bundle must contain exactly completion.json, "
            "completion.sha256, and capture.json"
        )
    completion_payload = _read_regular_file(
        bundle_dir / "completion.json",
        "archived completion record",
        limit=_MAX_BUNDLE_FILE_BYTES,
    )
    record = _load_json_bytes(completion_payload, "archived completion record")
    if canonical_json_bytes(record) != completion_payload:
        raise EvidenceCaptureError(
            "archived completion record does not use canonical JSON bytes"
        )
    record_sha256 = hashlib.sha256(completion_payload).hexdigest()
    sha_payload = _read_regular_file(
        bundle_dir / "completion.sha256",
        "archived completion digest",
        limit=1024,
    )
    if sha_payload != f"{record_sha256}\n".encode("ascii"):
        raise EvidenceCaptureError("completion.sha256 does not match completion.json")
    capture_payload = _read_regular_file(
        bundle_dir / "capture.json",
        "archived capture document",
        limit=_MAX_BUNDLE_FILE_BYTES,
    )
    capture = _load_json_bytes(capture_payload, "archived capture document")
    if canonical_json_bytes(capture) != capture_payload:
        raise EvidenceCaptureError("capture.json does not use canonical JSON bytes")
    _verify_capture_document(capture, record, record_sha256)
    return capture


def capture_from_files(args: argparse.Namespace) -> dict[str, Any]:
    pod_payload = _read_regular_file(
        args.pod_json, "captured Pod JSON", limit=_MAX_POD_JSON_BYTES
    )
    log_payload = _read_regular_file(
        args.pod_log, "captured Pod log", limit=_MAX_POD_LOG_BYTES
    )
    pod = _load_json_bytes(pod_payload, "captured Pod JSON")
    completion_payload, completion_sha256, record = extract_terminal_record(log_payload)
    capture = build_capture(
        pod,
        record,
        record_sha256=completion_sha256,
        container=args.container,
        expected_namespace=args.expected_namespace,
        expected_pod=args.expected_pod,
        expected_job=args.expected_job,
    )
    write_bundle(args.out_dir, completion_payload, completion_sha256, capture)
    return verify_bundle(args.out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture", help="create and immediately verify a new evidence bundle"
    )
    capture.add_argument("--pod-json", type=Path, required=True)
    capture.add_argument("--pod-log", type=Path, required=True)
    capture.add_argument("--container", required=True)
    capture.add_argument("--expected-namespace", required=True)
    capture.add_argument("--expected-pod", required=True)
    capture.add_argument("--expected-job", required=True)
    capture.add_argument("--out-dir", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify", help="verify one archived evidence bundle offline"
    )
    verify.add_argument("--bundle-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = (
            capture_from_files(args)
            if args.command == "capture"
            else verify_bundle(args.bundle_dir)
        )
    except EvidenceCaptureError as exc:
        raise SystemExit(f"crop-distill evidence capture refused: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
