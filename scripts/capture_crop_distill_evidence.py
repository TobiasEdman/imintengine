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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__:
    from .crop_distill_protocol import (
        BASE_PYTHON,
        CROP_HEADS_DIR,
        CROP_INDEX,
        CROP_SPLIT,
        CROP_SPLIT_MANIFEST,
        DATA_DIR,
        DISTILL_DIR,
        FROZEN_SPLIT_MODE,
        MODEL_KEYS,
        MODEL_PYTHON,
        RUNTIME_MANIFEST,
        SCORING_PYTHON,
        SOURCE_ACCESS_EXPECTED_CANDIDATES,
        SOURCE_ACCESS_INDEX_INPUT,
        SOURCE_ACCESS_INDEX_SHA256,
        SOURCE_ACCESS_INDEX_SIZE,
        SOURCE_ACCESS_LOCK_BACKING_FILE,
        STORAGE_GID,
        STORAGE_TARGETS,
        STORAGE_UID,
        SOURCE_ACCESS_LOCK_MODE,
        WORK_ROOT,
        RuntimeIdentity,
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
    from .crop_source_access import (
        COMPLETION_MARKER as SOURCE_ACCESS_COMPLETION_MARKER,
        COMPLETION_SCHEMA as SOURCE_ACCESS_COMPLETION_SCHEMA,
        PLAN_MARKER as SOURCE_ACCESS_PLAN_MARKER,
        PLAN_SCHEMA as SOURCE_ACCESS_PLAN_SCHEMA,
        load_plan as validate_source_access_plan_file,
        verify_completion as validate_source_access_completion_file,
    )
    from .prepare_crop_distill_storage import (
        STORAGE_PREP_COMPLETION_MARKER,
        STORAGE_PREP_COMPLETION_SCHEMA,
    )
else:
    from crop_distill_protocol import (  # type: ignore[no-redef]
        BASE_PYTHON,
        CROP_HEADS_DIR,
        CROP_INDEX,
        CROP_SPLIT,
        CROP_SPLIT_MANIFEST,
        DATA_DIR,
        DISTILL_DIR,
        FROZEN_SPLIT_MODE,
        MODEL_KEYS,
        MODEL_PYTHON,
        RUNTIME_MANIFEST,
        SCORING_PYTHON,
        SOURCE_ACCESS_EXPECTED_CANDIDATES,
        SOURCE_ACCESS_INDEX_INPUT,
        SOURCE_ACCESS_INDEX_SHA256,
        SOURCE_ACCESS_INDEX_SIZE,
        SOURCE_ACCESS_LOCK_BACKING_FILE,
        STORAGE_GID,
        STORAGE_TARGETS,
        STORAGE_UID,
        SOURCE_ACCESS_LOCK_MODE,
        WORK_ROOT,
        RuntimeIdentity,
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
    from crop_source_access import (  # type: ignore[no-redef]
        COMPLETION_MARKER as SOURCE_ACCESS_COMPLETION_MARKER,
        COMPLETION_SCHEMA as SOURCE_ACCESS_COMPLETION_SCHEMA,
        PLAN_MARKER as SOURCE_ACCESS_PLAN_MARKER,
        PLAN_SCHEMA as SOURCE_ACCESS_PLAN_SCHEMA,
        load_plan as validate_source_access_plan_file,
        verify_completion as validate_source_access_completion_file,
    )
    from prepare_crop_distill_storage import (  # type: ignore[no-redef]
        STORAGE_PREP_COMPLETION_MARKER,
        STORAGE_PREP_COMPLETION_SCHEMA,
    )

CAPTURE_SCHEMA = "imint-crop-distill-evidence-capture-v3"

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
_POD_OBSERVATION_SCHEMAS = {
    "crop": "imint-observed-crop-distill-pod-v1",
    "split": "imint-observed-lucas-crop-split-pod-v1",
    "storage-prep": "imint-observed-crop-storage-prep-pod-v1",
    "source-access-plan": "imint-observed-crop-source-access-plan-pod-v1",
    "source-access-apply": "imint-observed-crop-source-access-apply-pod-v1",
}
_RECORD_FILE = {
    "crop": "completion.json",
    "split": "completion.json",
    "storage-prep": "completion.json",
    "source-access-plan": "plan.json",
    "source-access-apply": "completion.json",
}
_RECORD_DIGEST_FILE = {
    kind: filename.removesuffix(".json") + ".sha256"
    for kind, filename in _RECORD_FILE.items()
}
_MARKER_PREFIX = {
    "crop": TERMINAL_EVIDENCE_PREFIX,
    "split": TERMINAL_EVIDENCE_PREFIX,
    "storage-prep": STORAGE_PREP_COMPLETION_MARKER,
    "source-access-plan": SOURCE_ACCESS_PLAN_MARKER,
    "source-access-apply": SOURCE_ACCESS_COMPLETION_MARKER,
}
_COMMON_BUNDLE_FILES = frozenset(
    {"capture.json", "marker.txt", "pod.json", "pod.sha256"}
)
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
        "source_access",
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
_SOURCE_ACCESS_PLAN_ENV = "CROP_SOURCE_ACCESS_PLAN_SHA256"
_SOURCE_ACCESS_PLAN_POD_ENV = "CROP_SOURCE_ACCESS_PLAN_POD_UID"
_SOURCE_ACCESS_COMPLETION_ENV = "CROP_SOURCE_ACCESS_COMPLETION_SHA256"
_SOURCE_ACCESS_COMPLETION_POD_ENV = "CROP_SOURCE_ACCESS_COMPLETION_POD_UID"
_FREEZE_LEASE_ENV = "CROP_SOURCE_FREEZE_LEASE_PATH"
_POD_UID_ENV = "POD_UID"
_MAX_POD_JSON_BYTES = 8 * 1024 * 1024
_MAX_POD_LOG_BYTES = 64 * 1024 * 1024
_MAX_BUNDLE_FILE_BYTES = 16 * 1024 * 1024


class EvidenceCaptureError(ValueError):
    """Raised when live or archived evidence cannot be trusted."""


def _git_authority() -> dict[str, str]:
    """Load the review pins from Git, never from the captured bundle."""
    if __package__:
        from . import gen_ladder_manifests as manifests
    else:
        import gen_ladder_manifests as manifests  # type: ignore[no-redef]

    return {
        "source_git_sha": manifests.CROP_DISTILL_SOURCE_GIT_SHA,
        "image_ref": manifests.CROP_DISTILL_IMAGE,
        "source_index_sha256": manifests.CROP_SOURCE_ACCESS_INDEX_SHA256,
        "plan_sha256": manifests.CROP_SOURCE_ACCESS_PLAN_SHA256,
        "plan_pod_uid": manifests.CROP_SOURCE_ACCESS_PLAN_POD_UID,
        "completion_sha256": manifests.CROP_SOURCE_ACCESS_COMPLETION_SHA256,
        "completion_pod_uid": manifests.CROP_SOURCE_ACCESS_COMPLETION_POD_UID,
        "split_manifest_sha256": manifests.CROP_DISTILL_SPLIT_MANIFEST_SHA256,
    }


def _validated_git_authority(
    evidence_kind: str,
    *,
    require_current_output_anchor: bool = True,
) -> dict[str, str]:
    authority = _git_authority()
    _require_hex40(
        authority.get("source_git_sha"),
        "Git-pinned crop-distill source SHA",
        nonzero=True,
    )
    image_ref = _require_string(
        authority.get("image_ref"), "Git-pinned crop-distill image"
    )
    match = _CROP_IMAGE_REF.fullmatch(image_ref)
    if match is None or match.group(1) == "0" * 64:
        raise EvidenceCaptureError(
            "Git-pinned crop-distill image must be one nonzero immutable digest"
        )
    if authority.get("source_index_sha256") != SOURCE_ACCESS_INDEX_SHA256:
        raise EvidenceCaptureError(
            "Git-pinned source-access index SHA256 differs from the protocol"
        )

    required_anchors: tuple[tuple[str, str], ...] = ()
    if evidence_kind == "source-access-plan" and require_current_output_anchor:
        required_anchors = (("plan_sha256", "plan_pod_uid"),)
    elif evidence_kind == "source-access-apply":
        required_anchors = (("plan_sha256", "plan_pod_uid"),)
        if require_current_output_anchor:
            required_anchors += (("completion_sha256", "completion_pod_uid"),)
    elif evidence_kind == "split":
        required_anchors = (
            ("plan_sha256", "plan_pod_uid"),
            ("completion_sha256", "completion_pod_uid"),
        )
    elif evidence_kind == "crop":
        _require_hex64(
            authority.get("split_manifest_sha256"),
            "Git-pinned crop split SHA256",
            nonzero=True,
        )
    for sha_name, uid_name in required_anchors:
        _require_hex64(
            authority.get(sha_name),
            f"Git-pinned {sha_name}",
            nonzero=True,
        )
        _require_safe_id(authority.get(uid_name), f"Git-pinned {uid_name}")
    return authority


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
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceCaptureError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8")
        value = json.loads(text, object_pairs_hook=reject_duplicates)
    except EvidenceCaptureError:
        raise
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

    source_access = record.get("source_access")
    if kind == "crop":
        if source_access is not None:
            raise EvidenceCaptureError(
                "crop completion source_access authority must be null"
            )
    else:
        source_access = _require_exact_keys(
            source_access,
            {"plan", "completion"},
            "split completion source_access",
        )
        for phase in ("plan", "completion"):
            anchor = _require_exact_keys(
                source_access.get(phase),
                {"sha256", "pod_uid"},
                f"split completion source_access {phase}",
            )
            _require_hex64(
                anchor.get("sha256"),
                f"split completion source_access {phase} SHA256",
                nonzero=True,
            )
            _require_safe_id(
                anchor.get("pod_uid"),
                f"split completion source_access {phase} Pod UID",
            )

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
        "source_access": source_access,
        "source_git_sha": runtime["source_git_sha"],
        "split_manifest_sha256": split["sha256"],
    }


def _source_runtime_binding(value: Any, authority: Mapping[str, str]) -> dict[str, str]:
    runtime = _require_exact_keys(
        value,
        {
            "source_git_sha",
            "image_ref",
            "runtime_manifest_sha256",
            "source_payload_sha256",
        },
        "source-access runtime",
    )
    if runtime.get("source_git_sha") != authority["source_git_sha"]:
        raise EvidenceCaptureError("source-access runtime differs from Git source")
    if runtime.get("image_ref") != authority["image_ref"]:
        raise EvidenceCaptureError("source-access runtime differs from Git image")
    for name in ("runtime_manifest_sha256", "source_payload_sha256"):
        _require_hex64(runtime.get(name), f"source-access runtime {name}", nonzero=True)
    return {name: str(runtime[name]) for name in sorted(runtime)}


def _validate_storage_prep_record(
    record: dict[str, Any], authority: Mapping[str, str]
) -> dict[str, Any]:
    _require_exact_keys(
        record,
        {
            "schema",
            "pod_uid",
            "status",
            "process_identity",
            "preserved_frozen_mode",
            "runtime",
            "targets",
            "dataset_lock",
        },
        "storage-prep completion",
    )
    if (
        record.get("schema") != STORAGE_PREP_COMPLETION_SCHEMA
        or record.get("status") != "completed"
    ):
        raise EvidenceCaptureError("storage-prep completion is not successful")
    pod_uid = _require_safe_id(record.get("pod_uid"), "storage-prep Pod UID")
    if record.get("process_identity") != {
        "effective_uid": 0,
        "effective_gid": STORAGE_GID,
    }:
        raise EvidenceCaptureError("storage-prep process identity is unexpected")
    if record.get("preserved_frozen_mode") != format(FROZEN_SPLIT_MODE, "04o"):
        raise EvidenceCaptureError("storage-prep frozen mode is unexpected")
    runtime = _validate_runtime(record.get("runtime"))
    if (
        runtime["source_git_sha"] != authority["source_git_sha"]
        or runtime["image_ref"] != authority["image_ref"]
    ):
        raise EvidenceCaptureError("storage-prep runtime differs from Git authority")
    dataset_lock = _require_exact_keys(
        record.get("dataset_lock"),
        {
            "path",
            "uid",
            "gid",
            "mode",
            "device",
            "inode",
            "size_bytes",
            "nlink",
            "state",
        },
        "storage-prep dataset lock",
    )
    if dataset_lock.get("path") != str(SOURCE_ACCESS_LOCK_BACKING_FILE):
        raise EvidenceCaptureError("storage-prep dataset lock path is unexpected")
    if (
        dataset_lock.get("uid") != 0
        or dataset_lock.get("gid") != STORAGE_GID
        or dataset_lock.get("mode") != format(SOURCE_ACCESS_LOCK_MODE, "04o")
        or dataset_lock.get("size_bytes") != 0
        or dataset_lock.get("nlink") != 1
        or dataset_lock.get("state") != "ready"
    ):
        raise EvidenceCaptureError("storage-prep dataset lock identity is unexpected")
    _require_nonnegative_int(
        dataset_lock.get("device"), "storage-prep dataset lock device"
    )
    _require_positive_int(
        dataset_lock.get("inode"), "storage-prep dataset lock inode"
    )
    targets = _require_list(record.get("targets"), "storage-prep targets")
    if len(targets) != len(STORAGE_TARGETS):
        raise EvidenceCaptureError("storage-prep target cardinality is unexpected")
    normalized_targets: list[dict[str, Any]] = []
    for actual, expected in zip(targets, STORAGE_TARGETS, strict=True):
        actual = _require_exact_keys(
            actual,
            {"path", "uid", "gid", "mode", "device", "inode", "state"},
            "storage-prep target",
        )
        state = actual.get("state")
        expected_mode = expected.mode
        if state == "preserved-frozen" and expected.preserve_mode is not None:
            expected_mode = expected.preserve_mode
        elif state != "writable":
            raise EvidenceCaptureError("storage-prep target state is unexpected")
        expected_identity = {
            "path": str(expected.path),
            "uid": expected.uid,
            "gid": expected.gid,
            "mode": format(expected_mode, "04o"),
        }
        if any(actual.get(key) != value for key, value in expected_identity.items()):
            raise EvidenceCaptureError("storage-prep target identity is unexpected")
        device = _require_nonnegative_int(actual.get("device"), "target device")
        inode = _require_positive_int(actual.get("inode"), "target inode")
        normalized_targets.append({**expected_identity, "device": device, "inode": inode, "state": state})
    return {
        "container": "storage-prep",
        "digest": normalize_image_digest(authority["image_ref"], "Git image"),
        "effective_gid": STORAGE_GID,
        "effective_uid": 0,
        "image_ref": authority["image_ref"],
        "job": "ladder-crop-distill-storage-prep",
        "kind": "storage-prep",
        "model": None,
        "pod_uid": pod_uid,
        "record_schema": STORAGE_PREP_COMPLETION_SCHEMA,
        "source_git_sha": authority["source_git_sha"],
        "targets": normalized_targets,
        "dataset_lock": dict(dataset_lock),
    }


def _validate_source_access_record(
    record_path: Path,
    record: dict[str, Any],
    *,
    evidence_kind: str,
    authority: Mapping[str, str],
    require_current_output_anchor: bool,
) -> dict[str, Any]:
    pod_uid = _require_safe_id(record.get("pod_uid"), "source-access Pod UID")
    runtime = _source_runtime_binding(record.get("runtime"), authority)
    identity = RuntimeIdentity(
        authority["source_git_sha"], authority["image_ref"], pod_uid
    )
    if evidence_kind == "source-access-plan":
        if record.get("schema") != SOURCE_ACCESS_PLAN_SCHEMA:
            raise EvidenceCaptureError("source-access PLAN schema is unexpected")
        try:
            validated, _ = validate_source_access_plan_file(
                record_path,
                expected_sha256=(
                    authority["plan_sha256"]
                    if require_current_output_anchor
                    else hashlib.sha256(canonical_json_bytes(record)).hexdigest()
                ),
                identity=identity,
                expected_plan_pod_uid=(
                    authority["plan_pod_uid"]
                    if require_current_output_anchor
                    else pod_uid
                ),
                expected_index_sha256=SOURCE_ACCESS_INDEX_SHA256,
                expected_index_size=SOURCE_ACCESS_INDEX_SIZE,
                expected_index_path=SOURCE_ACCESS_INDEX_INPUT,
                expected_data_dir=DATA_DIR,
                expected_runtime_binding=runtime,
            )
        except Exception as exc:  # crop_source_access has its own error type
            raise EvidenceCaptureError(f"source-access PLAN is invalid: {exc}") from exc
        if validated != record:
            raise EvidenceCaptureError("source-access PLAN validation changed the record")
        job = "ladder-crop-source-access-plan"
        container = "source-access-plan"
        record_schema = SOURCE_ACCESS_PLAN_SCHEMA
        plan = None
    else:
        if record.get("schema") != SOURCE_ACCESS_COMPLETION_SCHEMA:
            raise EvidenceCaptureError("source-access APPLY schema is unexpected")
        try:
            validated = validate_source_access_completion_file(
                record_path,
                expected_sha256=(
                    authority["completion_sha256"]
                    if require_current_output_anchor
                    else hashlib.sha256(canonical_json_bytes(record)).hexdigest()
                ),
                expected_source_git_sha=authority["source_git_sha"],
                expected_image_ref=authority["image_ref"],
                expected_completion_pod_uid=(
                    authority["completion_pod_uid"]
                    if require_current_output_anchor
                    else pod_uid
                ),
                expected_plan_sha256=authority["plan_sha256"],
                expected_runtime_binding=runtime,
            )
        except Exception as exc:  # crop_source_access has its own error type
            raise EvidenceCaptureError(f"source-access APPLY is invalid: {exc}") from exc
        if validated != record:
            raise EvidenceCaptureError("source-access APPLY validation changed the record")
        plan = _require_exact_keys(
            record.get("plan"), {"pod_uid", "sha256"}, "source-access APPLY plan"
        )
        if plan != {
            "pod_uid": authority["plan_pod_uid"],
            "sha256": authority["plan_sha256"],
        }:
            raise EvidenceCaptureError("source-access APPLY plan differs from Git pins")
        job = "ladder-crop-source-access-apply"
        container = "source-access-apply"
        record_schema = SOURCE_ACCESS_COMPLETION_SCHEMA
    return {
        "container": container,
        "digest": normalize_image_digest(authority["image_ref"], "Git image"),
        "effective_gid": STORAGE_GID,
        "effective_uid": 0,
        "image_ref": authority["image_ref"],
        "job": job,
        "kind": evidence_kind,
        "model": None,
        "plan": plan,
        "pod_uid": pod_uid,
        "record_schema": record_schema,
        "source_git_sha": authority["source_git_sha"],
    }


def _validate_workload_record(
    record_path: Path,
    record: dict[str, Any],
    *,
    evidence_kind: str,
    authority: Mapping[str, str],
    require_current_output_anchor: bool = True,
) -> dict[str, Any]:
    if evidence_kind in {"crop", "split"}:
        completion = _validate_completion_record(record)
        if completion["kind"] != evidence_kind:
            raise EvidenceCaptureError("record kind disagrees with capture kind")
        if (
            completion["source_git_sha"] != authority["source_git_sha"]
            or completion["image_ref"] != authority["image_ref"]
        ):
            raise EvidenceCaptureError("completion runtime differs from Git authority")
        if evidence_kind == "crop":
            if completion["split_manifest_sha256"] != authority["split_manifest_sha256"]:
                raise EvidenceCaptureError("crop split anchor differs from Git authority")
        else:
            expected_source_access = {
                "plan": {
                    "sha256": authority["plan_sha256"],
                    "pod_uid": authority["plan_pod_uid"],
                },
                "completion": {
                    "sha256": authority["completion_sha256"],
                    "pod_uid": authority["completion_pod_uid"],
                },
            }
            if completion["source_access"] != expected_source_access:
                raise EvidenceCaptureError(
                    "split source_access authority differs from Git pins"
                )
        return {**completion, "record_schema": COMPLETION_SCHEMA}
    if evidence_kind == "storage-prep":
        return _validate_storage_prep_record(record, authority)
    return _validate_source_access_record(
        record_path,
        record,
        evidence_kind=evidence_kind,
        authority=authority,
        require_current_output_anchor=require_current_output_anchor,
    )


def _mount(name: str, path: str, sub_path: str | None, read_only: bool) -> dict[str, Any]:
    return {
        "name": name,
        "mount_path": path,
        "sub_path": sub_path,
        "read_only": read_only,
    }


def _pvc_volume() -> dict[str, Any]:
    return {
        "name": "training-data-cephfs",
        "type": "persistentVolumeClaim",
        "claim_name": "training-data-cephfs",
        "read_only": False,
    }


def _freeze_lease_volume() -> dict[str, Any]:
    return {
        "name": "crop-source-freeze-lease",
        "type": "configMap",
        "config_map_name": "crop-source-freeze-lease",
        "default_mode": 0o644,
        "optional": False,
        "items": [{"key": "lease.json", "path": "lease.json"}],
    }


def _workload_contract(subject: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(subject["kind"])
    source = str(subject["source_git_sha"])
    image = str(subject["image_ref"])
    pod_uid = str(subject["pod_uid"])
    literal_env: dict[str, str] = {_SOURCE_ENV: source, _IMAGE_ENV: image}
    volumes = [_pvc_volume()]
    node_selector: dict[str, str] = {}
    if kind == "crop":
        model = str(subject["model"])
        literal_env.update(
            {
                _SPLIT_ENV: str(subject["split_manifest_sha256"]),
                "HOME": "/work/home",
                "TMPDIR": "/work/tmp",
            }
        )
        command = [str(BASE_PYTHON)]
        args = ["/opt/imintengine/scripts/run_crop_distill_job.py", "--model", model]
        mounts = [
            _mount("training-data-cephfs", str(DATA_DIR), "unified_v2_512", True),
            _mount("training-data-cephfs", f"/cephfs/checkpoints/ladder/{model}_r2", f"checkpoints/ladder/{model}_r2", True),
            _mount("training-data-cephfs", "/cephfs/distill/crop_split", "distill/crop_split/crop_consumer", True),
            _mount("training-data-cephfs", "/cephfs/crop-heads", f"distill/crop_heads/{model}_r2_crop_runs", False),
            _mount("training-data-cephfs", "/cephfs/crop-records", f"ops/crop-distill/{model}", False),
            _mount("work", "/work", None, False),
        ]
        volumes.append({"name": "work", "type": "emptyDir", "size_limit": "8Gi"})
        resources = {
            "requests": {"cpu": "4", "memory": "24Gi", "ephemeral-storage": "8Gi", "nvidia.com/gpu": "1"},
            "limits": {"cpu": "4", "memory": "24Gi", "ephemeral-storage": "8Gi", "nvidia.com/gpu": "1"},
        }
        node_selector = {"accelerator": "nvidia-gtx-2080ti"}
        deadline = 43200
    elif kind == "split":
        source_access = subject["source_access"]
        literal_env.update(
            {
                _SOURCE_ACCESS_PLAN_ENV: source_access["plan"]["sha256"],
                _SOURCE_ACCESS_PLAN_POD_ENV: source_access["plan"]["pod_uid"],
                _SOURCE_ACCESS_COMPLETION_ENV: source_access["completion"]["sha256"],
                _SOURCE_ACCESS_COMPLETION_POD_ENV: source_access["completion"]["pod_uid"],
                _FREEZE_LEASE_ENV: "/var/run/crop-source-freeze/lease.json",
                "HOME": "/work/home",
                "TMPDIR": "/work/tmp",
            }
        )
        command = [str(SCORING_PYTHON)]
        args = ["/opt/imintengine/scripts/run_lucas_crop_split_job.py"]
        mounts = [
            _mount("training-data-cephfs", str(DATA_DIR), "unified_v2_512", True),
            _mount("training-data-cephfs", "/cephfs/lucas", "lucas", True),
            _mount("training-data-cephfs", str(DISTILL_DIR), "distill/crop_split", False),
            _mount("training-data-cephfs", "/cephfs/ops/crop-distill", "ops/crop-distill/split", False),
            _mount("training-data-cephfs", "/cephfs/source-access-completion/completion.json", f"ops/crop-distill/source-access/apply/{source_access['completion']['pod_uid']}/completion.json", True),
            _mount("training-data-cephfs", "/cephfs/source-access-lock", "ops/crop-distill/source-access/locks", False),
            _mount("work", "/work", None, False),
            _mount("crop-source-freeze-lease", "/var/run/crop-source-freeze", None, True),
        ]
        volumes.append({"name": "work", "type": "emptyDir", "size_limit": None})
        volumes.append(_freeze_lease_volume())
        resources = {"requests": {"cpu": "2", "memory": "8Gi"}, "limits": {"cpu": "2", "memory": "8Gi"}}
        deadline = 3600
    elif kind == "storage-prep":
        command = [str(BASE_PYTHON)]
        args = ["/opt/imintengine/scripts/prepare_crop_distill_storage.py"]
        mounts = [
            _mount("training-data-cephfs", "/cephfs/distill", "distill", False),
            _mount("training-data-cephfs", "/cephfs/ops", "ops", False),
        ]
        resources = {"requests": {"cpu": "500m", "memory": "256Mi"}, "limits": {"cpu": "500m", "memory": "256Mi"}}
        deadline = 600
    elif kind == "source-access-plan":
        literal_env["CROP_SOURCE_ACCESS_INDEX_SHA256"] = SOURCE_ACCESS_INDEX_SHA256
        literal_env[_FREEZE_LEASE_ENV] = "/var/run/crop-source-freeze/lease.json"
        command = [str(SCORING_PYTHON)]
        args = ["/opt/imintengine/scripts/crop_source_access.py", "plan"]
        mounts = [
            _mount("training-data-cephfs", str(DATA_DIR), "unified_v2_512", True),
            _mount("training-data-cephfs", str(SOURCE_ACCESS_INDEX_INPUT), "lucas/lucas_tile_index.parquet", True),
            _mount("training-data-cephfs", "/cephfs/source-access-plan-records", "ops/crop-distill/source-access/plan", False),
            _mount("training-data-cephfs", "/cephfs/source-access-lock", "ops/crop-distill/source-access/locks", False),
            _mount("crop-source-freeze-lease", "/var/run/crop-source-freeze", None, True),
        ]
        volumes.append(_freeze_lease_volume())
        resources = {"requests": {"cpu": "2", "memory": "4Gi"}, "limits": {"cpu": "2", "memory": "4Gi"}}
        deadline = 7200
    else:
        plan = subject["plan"]
        literal_env.update(
            {
                "CROP_SOURCE_ACCESS_INDEX_SHA256": SOURCE_ACCESS_INDEX_SHA256,
                "CROP_SOURCE_ACCESS_PLAN_SHA256": plan["sha256"],
                "CROP_SOURCE_ACCESS_PLAN_POD_UID": plan["pod_uid"],
                _FREEZE_LEASE_ENV: "/var/run/crop-source-freeze/lease.json",
            }
        )
        command = [str(SCORING_PYTHON)]
        args = ["/opt/imintengine/scripts/crop_source_access.py", "apply"]
        mounts = [
            _mount("training-data-cephfs", str(DATA_DIR), "unified_v2_512", False),
            _mount("training-data-cephfs", str(SOURCE_ACCESS_INDEX_INPUT), "lucas/lucas_tile_index.parquet", True),
            _mount("training-data-cephfs", "/cephfs/source-access-plan/plan.json", f"ops/crop-distill/source-access/plan/{plan['pod_uid']}/plan.json", True),
            _mount("training-data-cephfs", "/cephfs/source-access-apply-records", "ops/crop-distill/source-access/apply", False),
            _mount("training-data-cephfs", "/cephfs/source-access-lock", "ops/crop-distill/source-access/locks", False),
            _mount("crop-source-freeze-lease", "/var/run/crop-source-freeze", None, True),
        ]
        volumes.append(_freeze_lease_volume())
        resources = {"requests": {"cpu": "2", "memory": "4Gi"}, "limits": {"cpu": "2", "memory": "4Gi"}}
        deadline = 7200
    caps_add = ["CHOWN", "FOWNER"] if kind in {"storage-prep", "source-access-apply"} else []
    return {
        "active_deadline_seconds": deadline,
        "args": args,
        "capabilities_add": caps_add,
        "command": command,
        "container": subject["container"],
        "effective_gid": subject["effective_gid"],
        "effective_uid": subject["effective_uid"],
        "image_ref": image,
        "job": subject["job"],
        "kind": kind,
        "literal_env": literal_env,
        "mounts": mounts,
        "model": subject.get("model"),
        "node_selector": node_selector,
        "pod_uid": pod_uid,
        "resources": resources,
        "run_as_non_root": kind in {"crop", "split"},
        "volumes": volumes,
    }


def _normalize_environment(container: Mapping[str, Any], expected: Mapping[str, str]) -> dict[str, Any]:
    env_from = container.get("envFrom", [])
    if _require_list(env_from, "target container envFrom"):
        raise EvidenceCaptureError("target container envFrom must be empty")
    entries = [_require_object(item, "target container env entry") for item in _require_list(container.get("env"), "target container env")]
    if len(entries) != len(expected) + 1:
        raise EvidenceCaptureError("target container environment must contain exactly the reviewed entries")
    literal: dict[str, str] = {}
    field_ref: dict[str, str] | None = None
    seen: set[str] = set()
    for entry in entries:
        name = _require_string(entry.get("name"), "target container env name")
        if name in seen:
            raise EvidenceCaptureError(f"duplicate target container environment {name}")
        seen.add(name)
        if name == _POD_UID_ENV:
            entry = _require_exact_keys(entry, {"name", "valueFrom"}, "POD_UID environment")
            value_from = _require_exact_keys(entry["valueFrom"], {"fieldRef"}, "POD_UID valueFrom")
            raw_ref = _require_object(value_from["fieldRef"], "POD_UID fieldRef")
            if set(raw_ref) not in ({"fieldPath"}, {"apiVersion", "fieldPath"}):
                raise EvidenceCaptureError("POD_UID fieldRef has unexpected fields")
            if raw_ref.get("fieldPath") != "metadata.uid" or raw_ref.get("apiVersion", "v1") != "v1":
                raise EvidenceCaptureError("POD_UID must reference v1 metadata.uid")
            field_ref = {"api_version": "v1", "field_path": "metadata.uid"}
        else:
            entry = _require_exact_keys(entry, {"name", "value"}, f"environment {name}")
            literal[name] = _require_string(entry.get("value"), f"environment {name}")
    if field_ref is None or literal != dict(expected):
        raise EvidenceCaptureError("target container environment differs from reviewed values")
    return {"env_from": [], "literal": dict(sorted(literal.items())), "pod_uid_field_ref": field_ref}


def _normalize_mounts(value: Any) -> list[dict[str, Any]]:
    mounts: list[dict[str, Any]] = []
    for raw in _require_list(value, "target container volumeMounts"):
        raw = _require_object(raw, "target container volumeMount")
        allowed = {"name", "mountPath", "subPath", "readOnly"}
        if not set(raw) <= allowed or not {"name", "mountPath"} <= set(raw):
            raise EvidenceCaptureError("target container volumeMount has unexpected fields")
        if "subPathExpr" in raw:
            raise EvidenceCaptureError("target container volumeMount must not use subPathExpr")
        read_only = raw.get("readOnly", False)
        if type(read_only) is not bool:
            raise EvidenceCaptureError("target container volumeMount readOnly must be boolean")
        mounts.append(
            _mount(
                _require_string(raw.get("name"), "volumeMount name"),
                _require_absolute_path(raw.get("mountPath"), "volumeMount mountPath"),
                raw.get("subPath"),
                read_only,
            )
        )
        if mounts[-1]["sub_path"] is not None:
            mounts[-1]["sub_path"] = _require_string(mounts[-1]["sub_path"], "volumeMount subPath")
    return mounts


def _normalize_volumes(value: Any) -> list[dict[str, Any]]:
    volumes: list[dict[str, Any]] = []
    for raw in _require_list(value, "Pod volumes"):
        raw = _require_object(raw, "Pod volume")
        name = _require_string(raw.get("name"), "Pod volume name")
        if set(raw) == {"name", "persistentVolumeClaim"}:
            pvc = _require_object(raw["persistentVolumeClaim"], "PVC volume source")
            if not set(pvc) <= {"claimName", "readOnly"} or "claimName" not in pvc:
                raise EvidenceCaptureError("PVC volume has unexpected fields")
            read_only = pvc.get("readOnly", False)
            if type(read_only) is not bool:
                raise EvidenceCaptureError("PVC volume readOnly must be boolean")
            volumes.append({"name": name, "type": "persistentVolumeClaim", "claim_name": _require_string(pvc.get("claimName"), "PVC claimName"), "read_only": read_only})
        elif set(raw) == {"name", "emptyDir"}:
            empty = _require_object(raw["emptyDir"], "emptyDir volume source")
            if not set(empty) <= {"sizeLimit"}:
                raise EvidenceCaptureError("emptyDir volume has unexpected fields")
            size_limit = empty.get("sizeLimit")
            if size_limit is not None:
                size_limit = _require_string(size_limit, "emptyDir sizeLimit")
            volumes.append({"name": name, "type": "emptyDir", "size_limit": size_limit})
        elif set(raw) == {"name", "configMap"}:
            config_map = _require_exact_keys(
                raw["configMap"],
                {"name", "defaultMode", "optional", "items"},
                "configMap volume source",
            )
            items = _require_list(config_map.get("items"), "configMap items")
            normalized_items = [
                _require_exact_keys(item, {"key", "path"}, "configMap item")
                for item in items
            ]
            if type(config_map.get("optional")) is not bool:
                raise EvidenceCaptureError("configMap optional must be boolean")
            if type(config_map.get("defaultMode")) is not int or config_map.get(
                "defaultMode"
            ) != 0o644:
                raise EvidenceCaptureError("configMap defaultMode must be 0644")
            volumes.append(
                {
                    "name": name,
                    "type": "configMap",
                    "config_map_name": _require_string(
                        config_map.get("name"), "configMap name"
                    ),
                    "default_mode": 0o644,
                    "optional": config_map["optional"],
                    "items": normalized_items,
                }
            )
        else:
            raise EvidenceCaptureError(
                "Pod volumes must be exactly reviewed PVC/emptyDir/configMap sources; "
                "hostPath, secret, projected, and extra PVC sources are forbidden"
            )
    if len({item["name"] for item in volumes}) != len(volumes):
        raise EvidenceCaptureError("Pod volume names must be unique")
    return volumes


def _normalize_security_contexts(
    spec: Mapping[str, Any], container: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    non_root = bool(contract["run_as_non_root"])
    pod_keys = {"runAsUser", "runAsGroup", "seccompProfile"}
    container_keys = {"allowPrivilegeEscalation", "capabilities", "readOnlyRootFilesystem", "runAsUser", "runAsGroup"}
    if non_root:
        pod_keys.add("runAsNonRoot")
        container_keys.add("runAsNonRoot")
    pod_security = _require_exact_keys(spec.get("securityContext"), pod_keys, "Pod securityContext")
    container_security = _require_exact_keys(container.get("securityContext"), container_keys, "container securityContext")
    for value, label in ((pod_security, "Pod"), (container_security, "container")):
        if value.get("runAsUser") != contract["effective_uid"] or value.get("runAsGroup") != contract["effective_gid"]:
            raise EvidenceCaptureError(f"{label} securityContext UID:GID is unexpected")
        if non_root and value.get("runAsNonRoot") is not True:
            raise EvidenceCaptureError(f"{label} runAsNonRoot must be true")
    seccomp = _require_exact_keys(pod_security.get("seccompProfile"), {"type"}, "Pod seccompProfile")
    if seccomp.get("type") != "RuntimeDefault":
        raise EvidenceCaptureError("Pod seccompProfile must be RuntimeDefault")
    if container_security.get("allowPrivilegeEscalation") is not False or container_security.get("readOnlyRootFilesystem") is not True:
        raise EvidenceCaptureError("container privilege/root-filesystem policy is unexpected")
    expected_capability_keys = {"drop"} | ({"add"} if contract["capabilities_add"] else set())
    capabilities = _require_exact_keys(container_security.get("capabilities"), expected_capability_keys, "container capabilities")
    if capabilities.get("drop") != ["ALL"] or capabilities.get("add", []) != contract["capabilities_add"]:
        raise EvidenceCaptureError("container capabilities differ from the reviewed set")
    normalized_pod = {
        "run_as_group": contract["effective_gid"],
        "run_as_non_root": non_root,
        "run_as_user": contract["effective_uid"],
        "seccomp_profile": {"type": "RuntimeDefault"},
    }
    normalized_container = {
        "allow_privilege_escalation": False,
        "capabilities": {"add": list(contract["capabilities_add"]), "drop": ["ALL"]},
        "privileged": False,
        "read_only_root_filesystem": True,
        "run_as_group": contract["effective_gid"],
        "run_as_non_root": non_root,
        "run_as_user": contract["effective_uid"],
        "seccomp_profile": {"source": "pod", "type": "RuntimeDefault"},
    }
    return normalized_pod, normalized_container


def _expected_labels(contract: Mapping[str, Any], job_uid: str) -> dict[str, str]:
    kind = str(contract["kind"])
    labels = {
        "app": "unified-training",
        "batch.kubernetes.io/controller-uid": job_uid,
        "batch.kubernetes.io/job-name": str(contract["job"]),
        "controller-uid": job_uid,
        "job-name": str(contract["job"]),
    }
    if kind == "crop":
        labels.update({"purpose": "ladder-crop-distill", "model": str(contract.get("model") or "")})
    elif kind == "split":
        labels.update({"purpose": "ladder-crop-distill", "model": "shared"})
    elif kind == "storage-prep":
        labels["purpose"] = "ladder-crop-distill-storage"
    elif kind == "source-access-plan":
        labels["purpose"] = "ladder-crop-source-access-plan"
    else:
        labels["purpose"] = "ladder-crop-source-access-apply"
    return labels


def normalize_observed_pod(
    pod: dict[str, Any],
    *,
    contract: Mapping[str, Any],
    expected_namespace: str,
    expected_pod: str,
) -> dict[str, Any]:
    """Validate the complete reviewed Pod authority surface and normalize it."""
    if pod.get("apiVersion") != "v1" or pod.get("kind") != "Pod":
        raise EvidenceCaptureError("captured Kubernetes object must be a v1 Pod")
    metadata = _require_object(pod.get("metadata"), "Pod metadata")
    namespace = _require_safe_id(metadata.get("namespace"), "Pod namespace")
    pod_name = _require_safe_id(metadata.get("name"), "Pod name")
    pod_uid = _require_safe_id(metadata.get("uid"), "Pod UID")
    if namespace != expected_namespace or pod_name != expected_pod:
        raise EvidenceCaptureError("Pod namespace/name differs from the live selection")
    if pod_uid != contract["pod_uid"]:
        raise EvidenceCaptureError("Pod UID differs from the immutable workload record")
    owners = _require_list(metadata.get("ownerReferences"), "Pod ownerReferences")
    if len(owners) != 1:
        raise EvidenceCaptureError("Pod must have exactly one Job ownerReference")
    owner = _require_exact_keys(
        owners[0],
        {"apiVersion", "blockOwnerDeletion", "controller", "kind", "name", "uid"},
        "Pod ownerReference",
    )
    job_uid = _require_safe_id(owner.get("uid"), "Job UID")
    if owner != {
        "apiVersion": "batch/v1",
        "blockOwnerDeletion": True,
        "controller": True,
        "kind": "Job",
        "name": contract["job"],
        "uid": job_uid,
    }:
        raise EvidenceCaptureError("Pod ownerReference differs from the reviewed Job")
    labels = _require_object(metadata.get("labels"), "Pod labels")
    expected_labels = _expected_labels(contract, job_uid)
    if labels != expected_labels:
        raise EvidenceCaptureError("Pod labels differ from the exact reviewed Job labels")

    spec = _require_object(pod.get("spec"), "Pod spec")
    status = _require_object(pod.get("status"), "Pod status")
    containers = _require_list(spec.get("containers"), "Pod containers")
    statuses = _require_list(status.get("containerStatuses"), "Pod containerStatuses")
    if len(containers) != 1 or len(statuses) != 1:
        raise EvidenceCaptureError("Pod must contain exactly one container and status")
    container = _require_object(containers[0], "Pod container")
    container_status = _require_object(statuses[0], "Pod container status")
    if container.get("name") != contract["container"] or container_status.get("name") != contract["container"]:
        raise EvidenceCaptureError("Pod target container identity is unexpected")
    for owner_obj, field in (
        (spec, "initContainers"),
        (spec, "ephemeralContainers"),
        (status, "initContainerStatuses"),
        (status, "ephemeralContainerStatuses"),
    ):
        if _require_list(owner_obj.get(field, []), f"Pod {field}"):
            raise EvidenceCaptureError(f"Pod {field} must be empty")

    if spec.get("activeDeadlineSeconds") != contract["active_deadline_seconds"]:
        raise EvidenceCaptureError("Pod activeDeadlineSeconds is unexpected")
    if spec.get("automountServiceAccountToken") is not False:
        raise EvidenceCaptureError("Pod automountServiceAccountToken must be false")
    if spec.get("serviceAccountName") != "default" or spec.get("serviceAccount") != "default":
        raise EvidenceCaptureError("Pod service-account boundary must be exactly default")
    if spec.get("imagePullSecrets") != [{"name": "ghcr-push"}]:
        raise EvidenceCaptureError("Pod imagePullSecrets differ from the kubelet-only pull secret")
    if spec.get("restartPolicy") != "Never":
        raise EvidenceCaptureError("Pod restartPolicy must be Never")
    isolation = {"automount_service_account_token": False, "host_ipc": False, "host_network": False, "host_pid": False, "restart_policy": "Never"}
    for raw, normalized in (("hostIPC", "host_ipc"), ("hostNetwork", "host_network"), ("hostPID", "host_pid")):
        value = spec.get(raw, False)
        if type(value) is not bool or value:
            raise EvidenceCaptureError(f"Pod {raw} must be false")
        isolation[normalized] = False
    node_name = _require_safe_id(spec.get("nodeName"), "Pod nodeName")
    node_selector = spec.get("nodeSelector", {})
    if _require_object(node_selector, "Pod nodeSelector") != contract["node_selector"]:
        raise EvidenceCaptureError("Pod nodeSelector differs from the reviewed contract")

    if container.get("image") != contract["image_ref"]:
        raise EvidenceCaptureError("Pod spec image differs from Git authority")
    status_image = _require_string(
        container_status.get("image"),
        "Pod status image",
    )
    # CRI implementations may report an immutable local config digest here
    # rather than repeating the manifest-digest ref requested in Pod spec.
    # imageID below remains the authoritative runtime-to-manifest binding.
    normalize_image_digest(status_image, "Pod status image")
    if container.get("imagePullPolicy") != "IfNotPresent":
        raise EvidenceCaptureError("target container imagePullPolicy is unexpected")
    if container.get("command") != contract["command"] or container.get("args") != contract["args"]:
        raise EvidenceCaptureError("target container command/args differ from reviewed values")
    if container.get("terminationMessagePath") != "/dev/termination-log" or container.get("terminationMessagePolicy") != "File":
        raise EvidenceCaptureError("target container termination-message policy is unexpected")
    for forbidden in ("lifecycle", "livenessProbe", "readinessProbe", "startupProbe", "ports"):
        if container.get(forbidden) not in (None, [], {}):
            raise EvidenceCaptureError(f"target container {forbidden} is not reviewed")
    if container.get("stdin", False) is not False or container.get("stdinOnce", False) is not False or container.get("tty", False) is not False:
        raise EvidenceCaptureError("target container interactive input must be disabled")
    environment = _normalize_environment(container, contract["literal_env"])
    mounts = _normalize_mounts(container.get("volumeMounts"))
    if mounts != contract["mounts"]:
        raise EvidenceCaptureError("target container volumeMounts differ from reviewed values")
    devices = _require_list(container.get("volumeDevices", []), "target container volumeDevices")
    if devices:
        raise EvidenceCaptureError("target container volumeDevices must be empty")
    volumes = _normalize_volumes(spec.get("volumes"))
    if volumes != contract["volumes"]:
        raise EvidenceCaptureError("Pod volumes differ from reviewed values")
    resources = _require_exact_keys(container.get("resources"), {"requests", "limits"}, "container resources")
    if resources != contract["resources"]:
        raise EvidenceCaptureError("target container resources differ from reviewed values")
    pod_security, container_security = _normalize_security_contexts(spec, container, contract)

    if status.get("phase") != "Succeeded":
        raise EvidenceCaptureError("Pod phase must be Succeeded")
    if type(container_status.get("restartCount")) is not int or container_status.get("restartCount") != 0:
        raise EvidenceCaptureError("target container restartCount must be zero")
    if _require_object(container_status.get("lastState", {}), "container lastState"):
        raise EvidenceCaptureError("target container lastState must be empty")
    state = _require_exact_keys(container_status.get("state"), {"terminated"}, "container state")
    terminated = _require_object(state["terminated"], "terminated state")
    if type(terminated.get("exitCode")) is not int or terminated.get("exitCode") != 0:
        raise EvidenceCaptureError("target container exitCode must be zero")
    reason = terminated.get("reason")
    if reason is not None:
        reason = _require_string(reason, "terminated reason")
    image_id = _require_string(container_status.get("imageID"), "Pod status imageID")
    image_digest = normalize_image_digest(image_id, "Pod status imageID")
    if image_digest != normalize_image_digest(contract["image_ref"], "Git image"):
        raise EvidenceCaptureError("Pod status imageID differs from Git image digest")

    return {
        "schema": _POD_OBSERVATION_SCHEMAS[str(contract["kind"])],
        "api_version": "v1",
        "kind": "Pod",
        "metadata": {
            "labels": dict(sorted(expected_labels.items())),
            "name": pod_name,
            "namespace": namespace,
            "owner": {"api_version": "batch/v1", "kind": "Job", "name": contract["job"], "uid": job_uid},
            "uid": pod_uid,
        },
        "spec": {
            "active_deadline_seconds": contract["active_deadline_seconds"],
            "automount_service_account_token": False,
            "containers": [{
                "args": list(contract["args"]),
                "command": list(contract["command"]),
                "environment": environment,
                "image": contract["image_ref"],
                "image_pull_policy": "IfNotPresent",
                "name": contract["container"],
                "resources": resources,
                "security_context": container_security,
                "termination_message_path": "/dev/termination-log",
                "termination_message_policy": "File",
                "volume_devices": [],
                "volume_mounts": mounts,
            }],
            "ephemeral_containers": [],
            "host_ipc": False,
            "host_network": False,
            "host_pid": False,
            "image_pull_secrets": [{"name": "ghcr-push", "workload_visible": False}],
            "init_containers": [],
            "isolation": isolation,
            "node_name": node_name,
            "node_selector": dict(sorted(node_selector.items())),
            "restart_policy": "Never",
            "security_context": pod_security,
            "service_account": {"automount_token": False, "name": "default"},
            "volumes": volumes,
        },
        "status": {
            "container_statuses": [{
                "image": status_image,
                "image_digest": image_digest,
                "image_id": image_id,
                "last_state": {},
                "name": contract["container"],
                "restart_count": 0,
                "terminated_exit_code": 0,
                "terminated_reason": reason,
            }],
            "ephemeral_container_statuses": [],
            "init_container_statuses": [],
            "phase": "Succeeded",
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


def _parse_generic_marker_line(
    line: str, *, expected_prefix: str
) -> tuple[str, bytes, dict[str, Any]]:
    line = line.removesuffix("\n")
    if not line or "\n" in line or "\r" in line:
        raise EvidenceCaptureError("terminal marker must contain exactly one line")
    parts = line.split(" ")
    if len(parts) != 3 or parts[0] != expected_prefix:
        raise EvidenceCaptureError("terminal marker prefix/field count is unexpected")
    digest = _require_hex64(parts[1], "terminal marker SHA256")
    try:
        payload = base64.b64decode(parts[2].encode("ascii"), validate=True)
    except Exception as exc:
        raise EvidenceCaptureError("terminal marker payload is not strict base64") from exc
    if base64.b64encode(payload).decode("ascii") != parts[2]:
        raise EvidenceCaptureError("terminal marker payload is not canonical base64")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise EvidenceCaptureError("terminal marker payload SHA256 mismatch")
    record = _load_json_bytes(payload, "terminal marker record")
    if canonical_json_bytes(record) != payload:
        raise EvidenceCaptureError("terminal marker record is not canonical JSON")
    return digest, payload, record


def extract_workload_record(
    pod_log: bytes, *, evidence_kind: str
) -> tuple[bytes, bytes, str, dict[str, Any]]:
    try:
        text = pod_log.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceCaptureError("Pod log is not valid UTF-8") from exc
    known = tuple(_MARKER_PREFIX.values())
    lines = [
        line
        for line in text.splitlines(keepends=True)
        if any(prefix in line for prefix in known)
    ]
    if len(lines) != 1:
        raise EvidenceCaptureError(
            f"Pod log must contain exactly one recognized terminal marker; found {len(lines)}"
        )
    prefix = _MARKER_PREFIX[evidence_kind]
    raw_line = lines[0]
    if not raw_line.endswith("\n") or raw_line.endswith("\r\n"):
        raise EvidenceCaptureError(
            "terminal marker must have its exact canonical LF line ending"
        )
    line = raw_line[:-1]
    if evidence_kind in {"crop", "split"}:
        try:
            digest, payload, record = parse_terminal_evidence_line(line)
        except ProvenanceError as exc:
            raise EvidenceCaptureError(f"terminal marker is invalid: {exc}") from exc
    else:
        digest, payload, record = _parse_generic_marker_line(
            line, expected_prefix=prefix
        )
    return raw_line.encode("utf-8"), payload, digest, record


def _authority_projection(kind: str, authority: Mapping[str, str]) -> dict[str, Any]:
    projection: dict[str, Any] = {
        "image_ref": authority["image_ref"],
        "source_git_sha": authority["source_git_sha"],
    }
    if kind in {"source-access-plan", "source-access-apply"}:
        projection["source_index_sha256"] = authority["source_index_sha256"]
    if kind in {"source-access-apply", "split"}:
        projection["plan"] = {
            "pod_uid": authority["plan_pod_uid"],
            "sha256": authority["plan_sha256"],
        }
    if kind == "split":
        projection["completion"] = {
            "pod_uid": authority["completion_pod_uid"],
            "sha256": authority["completion_sha256"],
        }
    if kind == "crop":
        projection["split_manifest_sha256"] = authority["split_manifest_sha256"]
    return projection


def _record_summary(
    subject: Mapping[str, Any], *, filename: str, record_sha256: str
) -> dict[str, Any]:
    kind = str(subject["kind"])
    if kind == "split":
        upstream = subject["source_access"]
    elif kind == "crop":
        upstream = {"split_manifest_sha256": subject["split_manifest_sha256"]}
    elif kind == "source-access-apply":
        upstream = {"plan": subject["plan"]}
    elif kind == "source-access-plan":
        upstream = {"source_index_sha256": SOURCE_ACCESS_INDEX_SHA256}
    else:
        upstream = None
    return {
        "filename": filename,
        "image_ref": subject["image_ref"],
        "job": subject["job"],
        "kind": kind,
        "model": subject.get("model"),
        "pod_uid": subject["pod_uid"],
        "process_identity": {
            "effective_gid": subject["effective_gid"],
            "effective_uid": subject["effective_uid"],
        },
        "record_sha256": record_sha256,
        "schema": subject["record_schema"],
        "source_git_sha": subject["source_git_sha"],
        "upstream_authority": upstream,
    }


def _capture_operator(subject: Mapping[str, Any]) -> dict[str, Any]:
    uid = os.geteuid()
    gid = os.getegid()
    workload_pair = (subject["effective_uid"], subject["effective_gid"])
    if (uid, gid) == workload_pair:
        raise EvidenceCaptureError(
            "external bundle producer must not use the workload UID:GID"
        )
    return {
        "effective_gid": gid,
        "effective_uid": uid,
        "identity_model": "local-kernel-effective-uid-gid-v1",
        "relationship": "external-observer-distinct-from-workload",
        "workload_effective_gid": subject["effective_gid"],
        "workload_effective_uid": subject["effective_uid"],
    }


def _validate_capture_operator(
    value: Any, subject: Mapping[str, Any]
) -> dict[str, Any]:
    value = _require_exact_keys(
        value,
        {
            "effective_gid",
            "effective_uid",
            "identity_model",
            "relationship",
            "workload_effective_gid",
            "workload_effective_uid",
        },
        "capture operator",
    )
    uid = _require_nonnegative_int(value.get("effective_uid"), "capture operator UID")
    gid = _require_nonnegative_int(value.get("effective_gid"), "capture operator GID")
    expected = {
        "identity_model": "local-kernel-effective-uid-gid-v1",
        "relationship": "external-observer-distinct-from-workload",
        "workload_effective_gid": subject["effective_gid"],
        "workload_effective_uid": subject["effective_uid"],
    }
    if any(value.get(name) != item for name, item in expected.items()):
        raise EvidenceCaptureError("capture operator/workload role binding is invalid")
    if (uid, gid) == (subject["effective_uid"], subject["effective_gid"]):
        raise EvidenceCaptureError("capture operator is not distinct from workload")
    return dict(value)


def build_capture_document(
    *,
    evidence_kind: str,
    subject: Mapping[str, Any],
    record_sha256: str,
    marker_payload: bytes,
    pod_payload: bytes,
    normalized_pod: dict[str, Any],
    authority: Mapping[str, str],
    operator: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_payload = canonical_json_bytes(normalized_pod)
    marker_sha256 = hashlib.sha256(marker_payload).hexdigest()
    return {
        "schema": CAPTURE_SCHEMA,
        "evidence_kind": evidence_kind,
        "authority": {
            "model": "git-pinned-generator-and-runtime-protocol-v1",
            "values": _authority_projection(evidence_kind, authority),
        },
        "capture_operator": dict(operator),
        "marker": {
            "filename": "marker.txt",
            "marker_sha256": marker_sha256,
            "payload_sha256": record_sha256,
            "prefix": _MARKER_PREFIX[evidence_kind],
        },
        "observed_pod": {
            "normalized": normalized_pod,
            "normalized_sha256": hashlib.sha256(normalized_payload).hexdigest(),
            "raw": {
                "filename": "pod.json",
                "sha256": hashlib.sha256(pod_payload).hexdigest(),
            },
        },
        "record_source": (
            "stdout-marker-and-operator-supplied-pvc-record-byte-equality"
            if evidence_kind in {"storage-prep", "source-access-plan", "source-access-apply"}
            else "stdout-marker-payload-from-write-once-workload-record"
        ),
        "workload_record": _record_summary(
            subject,
            filename=_RECORD_FILE[evidence_kind],
            record_sha256=record_sha256,
        ),
    }


def _bundle_files(kind: str) -> frozenset[str]:
    return _COMMON_BUNDLE_FILES | {
        _RECORD_FILE[kind],
        _RECORD_DIGEST_FILE[kind],
    }


def write_bundle(
    out_dir: Path,
    *,
    evidence_kind: str,
    record_payload: bytes,
    record_sha256: str,
    marker_payload: bytes,
    pod_payload: bytes,
    capture: dict[str, Any],
) -> None:
    """Create one off-PVC evidence directory without replacing prior evidence."""
    absolute = Path(os.path.abspath(out_dir))
    if out_dir != absolute or absolute == Path("/cephfs") or Path("/cephfs") in absolute.parents:
        raise EvidenceCaptureError("evidence output directory must be absolute and outside /cephfs")
    try:
        out_dir.mkdir(mode=0o700)
    except OSError as exc:
        raise EvidenceCaptureError(f"evidence output directory must be new: {out_dir}: {exc}") from exc
    created: list[Path] = []
    items = (
        (_RECORD_FILE[evidence_kind], record_payload),
        (_RECORD_DIGEST_FILE[evidence_kind], f"{record_sha256}\n".encode("ascii")),
        ("marker.txt", marker_payload),
        ("pod.json", pod_payload),
        ("pod.sha256", f"{hashlib.sha256(pod_payload).hexdigest()}\n".encode("ascii")),
        ("capture.json", canonical_json_bytes(capture)),
    )
    try:
        for name, payload in items:
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


def _read_capture_document(bundle_dir: Path) -> tuple[bytes, dict[str, Any], str]:
    capture_payload = _read_regular_file(
        bundle_dir / "capture.json", "archived capture document", limit=_MAX_BUNDLE_FILE_BYTES
    )
    capture = _load_json_bytes(capture_payload, "archived capture document")
    if canonical_json_bytes(capture) != capture_payload:
        raise EvidenceCaptureError("capture.json does not use canonical JSON bytes")
    _require_exact_keys(
        capture,
        {"schema", "evidence_kind", "authority", "capture_operator", "marker", "observed_pod", "record_source", "workload_record"},
        "capture document",
    )
    if capture.get("schema") != CAPTURE_SCHEMA:
        raise EvidenceCaptureError("unexpected capture.json schema")
    kind = _require_string(capture.get("evidence_kind"), "capture evidence_kind")
    if kind not in _POD_OBSERVATION_SCHEMAS:
        raise EvidenceCaptureError("capture evidence_kind is unsupported")
    return capture_payload, capture, kind


def verify_bundle(
    bundle_dir: Path,
    *,
    require_current_git_output_anchor: bool = True,
) -> dict[str, Any]:
    """Re-derive every record, marker, raw Pod, normalized Pod, and Git pin."""
    _, capture, kind = _read_capture_document(bundle_dir)
    try:
        entries = {entry.name for entry in bundle_dir.iterdir()}
    except OSError as exc:
        raise EvidenceCaptureError(f"cannot inspect evidence bundle {bundle_dir}: {exc}") from exc
    if entries != _bundle_files(kind):
        raise EvidenceCaptureError(f"evidence bundle files differ from the exact {kind} schema")

    record_path = bundle_dir / _RECORD_FILE[kind]
    record_payload = _read_regular_file(record_path, "archived workload record", limit=_MAX_BUNDLE_FILE_BYTES)
    record = _load_json_bytes(record_payload, "archived workload record")
    if canonical_json_bytes(record) != record_payload:
        raise EvidenceCaptureError("archived workload record is not canonical JSON")
    record_sha256 = hashlib.sha256(record_payload).hexdigest()
    digest_payload = _read_regular_file(bundle_dir / _RECORD_DIGEST_FILE[kind], "archived workload digest", limit=1024)
    if digest_payload != f"{record_sha256}\n".encode("ascii"):
        raise EvidenceCaptureError("archived workload digest does not match record bytes")

    marker_payload = _read_regular_file(bundle_dir / "marker.txt", "archived terminal marker", limit=_MAX_BUNDLE_FILE_BYTES)
    parsed_marker, marker_record_payload, marker_digest, marker_record = extract_workload_record(marker_payload, evidence_kind=kind)
    if parsed_marker != marker_payload or marker_digest != record_sha256 or marker_record_payload != record_payload or marker_record != record:
        raise EvidenceCaptureError("terminal marker bytes/hash do not equal archived workload record")

    pod_payload = _read_regular_file(bundle_dir / "pod.json", "archived raw Pod", limit=_MAX_POD_JSON_BYTES)
    pod_sha256 = hashlib.sha256(pod_payload).hexdigest()
    pod_digest_payload = _read_regular_file(bundle_dir / "pod.sha256", "archived raw Pod digest", limit=1024)
    if pod_digest_payload != f"{pod_sha256}\n".encode("ascii"):
        raise EvidenceCaptureError("pod.sha256 does not match raw pod.json bytes")
    pod = _load_json_bytes(pod_payload, "archived raw Pod")

    authority = _validated_git_authority(
        kind,
        require_current_output_anchor=require_current_git_output_anchor,
    )
    subject = _validate_workload_record(
        record_path,
        record,
        evidence_kind=kind,
        authority=authority,
        require_current_output_anchor=require_current_git_output_anchor,
    )
    contract = _workload_contract(subject)
    archived_observation = _require_exact_keys(
        capture.get("observed_pod"), {"normalized", "normalized_sha256", "raw"}, "capture observed_pod"
    )
    archived_normalized = _require_object(archived_observation.get("normalized"), "capture normalized Pod")
    archived_normalized_sha256 = hashlib.sha256(
        canonical_json_bytes(archived_normalized)
    ).hexdigest()
    if archived_observation.get("normalized_sha256") != archived_normalized_sha256:
        raise EvidenceCaptureError(
            "archived normalized Pod bytes/hash are inconsistent"
        )
    metadata = _require_object(archived_normalized.get("metadata"), "capture normalized Pod metadata")
    normalized = normalize_observed_pod(
        pod,
        contract=contract,
        expected_namespace="prithvi-training-default",
        expected_pod=_require_safe_id(metadata.get("name"), "capture normalized Pod name"),
    )
    normalized_sha256 = hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()
    if archived_normalized != normalized or archived_normalized_sha256 != normalized_sha256:
        raise EvidenceCaptureError("normalized Pod observation/hash is inconsistent")
    if archived_observation.get("raw") != {"filename": "pod.json", "sha256": pod_sha256}:
        raise EvidenceCaptureError("raw Pod observation/hash is inconsistent")

    operator = _validate_capture_operator(capture.get("capture_operator"), subject)
    expected_capture = build_capture_document(
        evidence_kind=kind,
        subject=subject,
        record_sha256=record_sha256,
        marker_payload=marker_payload,
        pod_payload=pod_payload,
        normalized_pod=normalized,
        authority=authority,
        operator=operator,
    )
    if capture != expected_capture:
        raise EvidenceCaptureError("capture.json differs from fully re-derived evidence")
    return capture


def capture_from_files(args: argparse.Namespace) -> dict[str, Any]:
    kind = args.evidence_kind
    pod_payload = _read_regular_file(args.pod_json, "captured Pod JSON", limit=_MAX_POD_JSON_BYTES)
    log_payload = _read_regular_file(args.pod_log, "captured Pod log", limit=_MAX_POD_LOG_BYTES)
    pod = _load_json_bytes(pod_payload, "captured Pod JSON")
    marker_payload, record_payload, record_sha256, record = extract_workload_record(log_payload, evidence_kind=kind)
    if canonical_json_bytes(record) != record_payload:
        raise EvidenceCaptureError("workload marker record is not canonical JSON")

    requires_pvc_record = kind in {"storage-prep", "source-access-plan", "source-access-apply"}
    if requires_pvc_record and args.record_file is None:
        raise EvidenceCaptureError(f"{kind} capture requires --record-file from the PVC")
    record_path = args.record_file or args.pod_log
    if args.record_file is not None:
        pvc_payload = _read_regular_file(args.record_file, "operator-supplied PVC record", limit=_MAX_BUNDLE_FILE_BYTES)
        if pvc_payload != record_payload:
            raise EvidenceCaptureError("stdout marker bytes do not equal the PVC record")

    authority = _validated_git_authority(
        kind,
        require_current_output_anchor=False,
    )
    subject = _validate_workload_record(
        record_path,
        record,
        evidence_kind=kind,
        authority=authority,
        require_current_output_anchor=False,
    )
    if args.container != subject["container"] or args.expected_job != subject["job"]:
        raise EvidenceCaptureError("CLI container/job selection differs from record authority")
    if args.expected_namespace != "prithvi-training-default":
        raise EvidenceCaptureError("capture namespace differs from the reviewed namespace")
    contract = _workload_contract(subject)
    normalized = normalize_observed_pod(
        pod,
        contract=contract,
        expected_namespace=args.expected_namespace,
        expected_pod=args.expected_pod,
    )
    operator = _capture_operator(subject)
    capture = build_capture_document(
        evidence_kind=kind,
        subject=subject,
        record_sha256=record_sha256,
        marker_payload=marker_payload,
        pod_payload=pod_payload,
        normalized_pod=normalized,
        authority=authority,
        operator=operator,
    )
    write_bundle(
        args.out_dir,
        evidence_kind=kind,
        record_payload=record_payload,
        record_sha256=record_sha256,
        marker_payload=marker_payload,
        pod_payload=pod_payload,
        capture=capture,
    )
    return verify_bundle(
        args.out_dir,
        require_current_git_output_anchor=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="create and immediately verify a new evidence bundle")
    capture.add_argument("--evidence-kind", choices=tuple(_POD_OBSERVATION_SCHEMAS), required=True)
    capture.add_argument("--pod-json", type=Path, required=True)
    capture.add_argument("--pod-log", type=Path, required=True)
    capture.add_argument("--record-file", type=Path)
    capture.add_argument("--container", required=True)
    capture.add_argument("--expected-namespace", required=True)
    capture.add_argument("--expected-pod", required=True)
    capture.add_argument("--expected-job", required=True)
    capture.add_argument("--out-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="verify one archived evidence bundle offline")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = capture_from_files(args) if args.command == "capture" else verify_bundle(args.bundle_dir)
    except EvidenceCaptureError as exc:
        raise SystemExit(f"crop-distill evidence capture refused: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
