#!/usr/bin/env python3
"""Verify and publish fail-closed LUCAS crop-distill provenance.

The crop-distill Jobs use a single immutable runtime image, but their model
checkpoint, frozen LUCAS split, and output artifacts live on the shared PVC.
This module binds those pieces into one per-Pod terminal record.  A completed
record is impossible unless every input is present and content-verified.

``verify-runtime`` is intentionally cheap enough to run before requesting a
GPU.  ``finalize`` repeats the same checks immediately before publishing the
terminal record, closing the gap between preflight and completion.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

if __package__:
    from .crop_distill_protocol import (
        CROP_HEADS_DIR,
        CROP_INDEX,
        CROP_RECORD_DIR,
        CROP_SPLIT,
        CROP_SPLIT_MANIFEST,
        DISTILL_DIR,
        MODEL_KEYS,
        SPLIT_RECORD_DIR,
        STORAGE_GID,
        STORAGE_UID,
        WORK_ROOT,
        model_process_uid,
        model_protocol,
    )
else:
    from crop_distill_protocol import (
        CROP_HEADS_DIR,
        CROP_INDEX,
        CROP_RECORD_DIR,
        CROP_SPLIT,
        CROP_SPLIT_MANIFEST,
        DISTILL_DIR,
        MODEL_KEYS,
        SPLIT_RECORD_DIR,
        STORAGE_GID,
        STORAGE_UID,
        WORK_ROOT,
        model_process_uid,
        model_protocol,
    )

RUNTIME_SCHEMA = "imint-crop-distill-runtime-v1"
TREE_SCHEMA = "imint-content-tree-v1"
COMPLETION_SCHEMA = "imint-crop-distill-completion-v1"
TERMINAL_EVIDENCE_PREFIX = "CROP_DISTILL_TERMINAL_EVIDENCE_V1"

_COMPLETION_RECORD_FIELDS = {
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

BASE_IMAGE = (
    "python:3.11-slim@sha256:"
    "d1e9ca7c4e78d1e8ecadb5d44bfc8e956e7a65b659a9950f569f243d72b326d0"
)
MODEL_RESOLUTION = {
    "cutoff_utc": "2026-08-31T07:47:18Z",
    "evidence": "timestamp-python-abi-pypi-metadata",
    "observed_package_log": False,
    "terratorch": "1.2.11",
    "torchgeo": "0.8.1",
}

CROMA_GIT_SHA = "59505a6bcadbf36ba20767270154bf9f3067c5e7"
CROMA_ARCHIVE_SHA256 = (
    "939d0918991ad7604bbb0a782df2674b8e30ade6edc061bcad6ab486e6f94001"
)
CLAY_GIT_SHA = "f14e698f3c237cabf8d28dec669a362d66625381"
CLAY_ARCHIVE_SHA256 = (
    "0b908ea11d5348736c26512f695221a304883ec88bac68f66822ca07bf435d64"
)
SOURCE_ARCHIVE_SELECTION = [
    "config",
    "data/distill/distill_split.json",
    "imint",
    "scripts",
]

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_REF = re.compile(r"^[^\s@]+@sha256:([0-9a-f]{64})$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")
_ARTIFACT_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_PYTHON_VERSION = re.compile(r"^3\.11\.\d+$")

_SPLIT_ARTIFACTS = {
    "index": "lucas_crop_distill_index.parquet",
    "validator_holdout": "lucas_crop_validator_holdout_index.parquet",
    "split": "lucas_crop_split.json",
    "manifest": "lucas_crop_split.MANIFEST.json",
}
_CROP_ARTIFACTS = {"features", "oof"}
_SPLIT_DIGEST_FIELDS = {
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
_MAX_DIAGNOSTIC_LENGTH = 1024
_SPLIT_AUTHORITY_ENV = "CROP_DISTILL_SPLIT_MANIFEST_SHA256"
_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
_DIR_FLAGS = _READ_FLAGS | os.O_DIRECTORY
_CREATE_FLAGS = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
)
_COPY_BLOCK_SIZE = 1024 * 1024


class ProvenanceError(ValueError):
    """Raised when evidence is missing, malformed, or mismatched."""


def _path_parts(path: Path) -> tuple[str, ...]:
    if not path.is_absolute():
        raise ProvenanceError(f"path must be absolute: {path}")
    parts = path.parts[1:]
    if any(part in ("", ".", "..") for part in parts):
        raise ProvenanceError(f"path is not a normalized absolute path: {path}")
    return parts


def _open_directory(
    path: Path,
    *,
    create: bool = False,
    create_mode: int = 0o700,
) -> int:
    """Open a directory without following any component symlinks."""
    parts = _path_parts(path)
    current_fd = os.open("/", _DIR_FLAGS)
    try:
        for part in parts:
            try:
                next_fd = os.open(part, _DIR_FLAGS, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise ProvenanceError(f"required directory is missing: {path}")
                try:
                    os.mkdir(part, mode=create_mode, dir_fd=current_fd)
                except FileExistsError:
                    pass
                try:
                    next_fd = os.open(part, _DIR_FLAGS, dir_fd=current_fd)
                except OSError as exc:
                    raise ProvenanceError(
                        f"cannot securely open created directory {path}: {exc}"
                    ) from exc
            except OSError as exc:
                raise ProvenanceError(
                    "directory path contains a symlink or non-directory: "
                    f"{path}: {exc}"
                ) from exc
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _same_file(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _read_regular_at(
    parent_fd: int,
    name: str,
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    file_fd: int | None = None
    try:
        try:
            file_fd = os.open(name, _READ_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise ProvenanceError(f"cannot securely read {label}: {exc}") from exc
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ProvenanceError(f"{label} is not a regular file")
        if before.st_nlink != 1:
            raise ProvenanceError(f"{label} must have exactly one hard link")
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(file_fd, _COPY_BLOCK_SIZE)
            if not block:
                break
            blocks.append(block)
            total += len(block)
        after = os.fstat(file_fd)
        if not _same_file(before, after) or total != before.st_size:
            raise ProvenanceError(f"{label} changed while it was being read")
        return b"".join(blocks), before
    finally:
        if file_fd is not None:
            os.close(file_fd)


def sha256_file(path: Path) -> str:
    """Hash *path* without loading a checkpoint into memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ProvenanceError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Stable bytes used for hashes and immutable terminal records."""
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _require_hex40(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise ProvenanceError(f"{label} must be 40 lowercase hex characters")
    return value


def _require_hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ProvenanceError(f"{label} must be 64 lowercase hex characters")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ProvenanceError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProvenanceError(f"{label} must be a non-negative integer")
    return value


def _require_safe_id(value: str, label: str) -> str:
    if _SAFE_ID.fullmatch(value) is None:
        raise ProvenanceError(
            f"{label} must contain only letters, digits, '.', '_', ':', or '-'"
        )
    return value


def _bounded_claim(value: str, label: str) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        return f"<empty {label}>"
    return normalized[:_MAX_DIAGNOSTIC_LENGTH]


def _load_object_bytes(payload: bytes, path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"invalid {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProvenanceError(f"{label} at {path} must be a JSON object")
    return value


def _regular_file_payload_identity(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Hash one nonempty, single-link file through an O_NOFOLLOW descriptor."""
    parent_fd: int | None = None
    try:
        parent_fd = _open_directory(path.parent)
        payload, file_stat = _read_regular_at(
            parent_fd,
            path.name,
            label=f"{label}: {path}",
        )
    except ProvenanceError as exc:
        raise ProvenanceError(
            f"missing {label} or unsafe path: {path}: {exc}"
        ) from exc
    finally:
        if parent_fd is not None:
            os.close(parent_fd)
    if file_stat.st_size <= 0:
        raise ProvenanceError(f"{label} is empty: {path}")

    if expected_size is not None:
        expected_size = _require_positive_int(expected_size, f"{label} size")
        if file_stat.st_size != expected_size:
            raise ProvenanceError(
                f"{label} size mismatch: expected {expected_size}, "
                f"got {file_stat.st_size}"
            )

    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        expected_sha256 = _require_hex64(
            expected_sha256, f"{label} SHA256"
        )
        if actual_sha256 != expected_sha256:
            raise ProvenanceError(
                f"{label} SHA256 mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
    return {
        "path": str(path),
        "size_bytes": file_stat.st_size,
        "sha256": actual_sha256,
    }, payload


def _regular_file_identity(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> dict[str, Any]:
    identity, _ = _regular_file_payload_identity(
        path,
        label,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )
    return identity


def _tree_entry(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    file_stat = path.lstat()
    if stat.S_ISREG(file_stat.st_mode):
        return {
            "path": relative,
            "type": "file",
            "size_bytes": file_stat.st_size,
            "sha256": sha256_file(path),
        }
    if stat.S_ISLNK(file_stat.st_mode):
        target = os.readlink(path)
        return {
            "path": relative,
            "type": "symlink",
            "target": target,
            "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
        }
    raise ProvenanceError(f"unsupported file type in sealed tree: {path}")


def snapshot_tree(root: Path) -> list[dict[str, Any]]:
    """Return a canonical snapshot of every file and symlink below *root*."""
    if not root.is_dir():
        raise ProvenanceError(f"sealed tree root is not a directory: {root}")
    paths = sorted(
        (path for path in root.rglob("*") if not path.is_dir()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths:
        raise ProvenanceError(f"sealed tree is empty: {root}")
    return [_tree_entry(root, path) for path in paths]


def tree_payload_sha256(entries: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        entries, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_manifest_entries(entries: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(entries, list) or not entries:
        raise ProvenanceError(f"{label} must contain a non-empty entries list")
    paths: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ProvenanceError(f"{label} entry {index} is not an object")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise ProvenanceError(f"{label} entry {index} has no path")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
            raise ProvenanceError(f"{label} has unsafe relative path: {relative}")
        paths.append(relative)
        entry_type = entry.get("type")
        if entry_type == "file":
            _require_nonnegative_int(
                entry.get("size_bytes"), f"{label} {relative} size"
            )
            _require_hex64(entry.get("sha256"), f"{label} {relative} SHA256")
        elif entry_type == "symlink":
            if not isinstance(entry.get("target"), str):
                raise ProvenanceError(
                    f"{label} {relative} has an invalid symlink target"
                )
            _require_hex64(
                entry.get("target_sha256"),
                f"{label} {relative} target SHA256",
            )
        else:
            raise ProvenanceError(
                f"{label} {relative} has unsupported type {entry_type!r}"
            )
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ProvenanceError(f"{label} entries must be sorted and unique")
    return entries


def _verify_tree_identity(record: Any, label: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ProvenanceError(f"runtime manifest {label} must be an object")
    root_raw = record.get("root")
    manifest_raw = record.get("files_manifest")
    if not isinstance(root_raw, str) or not Path(root_raw).is_absolute():
        raise ProvenanceError(f"runtime manifest {label}.root must be absolute")
    if not isinstance(manifest_raw, str) or not Path(manifest_raw).is_absolute():
        raise ProvenanceError(
            f"runtime manifest {label}.files_manifest must be absolute"
        )
    root = Path(root_raw)
    manifest_path = Path(manifest_raw)
    manifest_identity, manifest_payload = _regular_file_payload_identity(
        manifest_path,
        f"{label} file manifest",
        expected_sha256=_require_hex64(
            record.get("files_manifest_sha256"),
            f"runtime manifest {label}.files_manifest_sha256",
        ),
    )
    file_manifest = _load_object_bytes(
        manifest_payload, manifest_path, f"{label} file manifest"
    )
    if file_manifest.get("schema") != TREE_SCHEMA:
        raise ProvenanceError(f"unexpected {label} file-manifest schema")
    entries = _validate_manifest_entries(file_manifest.get("entries"), label)
    declared_payload = _require_hex64(
        file_manifest.get("payload_sha256"), f"{label} payload SHA256"
    )
    record_payload = _require_hex64(
        record.get("payload_sha256"),
        f"runtime manifest {label}.payload_sha256",
    )
    if declared_payload != record_payload:
        raise ProvenanceError(f"{label} payload SHA256 disagrees across manifests")
    if tree_payload_sha256(entries) != declared_payload:
        raise ProvenanceError(f"{label} file manifest has an invalid payload hash")

    actual_entries = snapshot_tree(root)
    if actual_entries != entries:
        raise ProvenanceError(f"{label} sealed tree differs from its manifest")
    return {
        "archive_sha256": _require_hex64(
            record.get("archive_sha256"), f"{label} archive SHA256"
        ),
        "payload_sha256": declared_payload,
        "files_manifest_sha256": manifest_identity["sha256"],
    }


def _verify_dependency(record: Any, label: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ProvenanceError(f"runtime manifest dependency {label} is not an object")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ProvenanceError(f"dependency {label} path must be absolute")
    return _regular_file_identity(
        Path(raw_path),
        f"dependency {label}",
        expected_sha256=_require_hex64(
            record.get("sha256"), f"dependency {label} SHA256"
        ),
        expected_size=_require_positive_int(
            record.get("size_bytes"), f"dependency {label} size"
        ),
    )


def _verify_python(record: Any, label: str) -> dict[str, Any]:
    """Verify one declared CPython 3.11 interpreter in the running image."""
    expected_keys = {"implementation", "path", "version", "version_info"}
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ProvenanceError(
            f"runtime manifest {label} Python identity is incomplete or unexpected"
        )
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ProvenanceError(f"runtime manifest {label} Python path must be absolute")
    path = Path(raw_path)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ProvenanceError(f"runtime manifest {label} Python is not executable")
    if record.get("implementation") != "CPython":
        raise ProvenanceError(f"runtime manifest {label} must use CPython")
    version = record.get("version")
    version_info = record.get("version_info")
    if not isinstance(version, str) or _PYTHON_VERSION.fullmatch(version) is None:
        raise ProvenanceError(
            f"runtime manifest {label} must use an exact Python 3.11 patch version"
        )
    if (
        not isinstance(version_info, list)
        or len(version_info) != 3
        or any(not isinstance(value, int) for value in version_info)
        or version_info[:2] != [3, 11]
        or version != ".".join(str(value) for value in version_info)
    ):
        raise ProvenanceError(
            f"runtime manifest {label} Python version fields disagree"
        )

    probe = (
        "import json, platform, sys; "
        "print(json.dumps({'implementation': platform.python_implementation(), "
        "'path': sys.executable, 'version': platform.python_version(), "
        "'version_info': list(sys.version_info[:3])}, sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [str(path), "-I", "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )
        actual = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ProvenanceError(
            f"cannot verify runtime manifest {label} Python: {exc}"
        ) from exc
    if actual != record:
        raise ProvenanceError(
            f"runtime manifest {label} Python differs from the running interpreter"
        )
    return record


def validate_image_ref(image_ref: str) -> str:
    """Return the digest from an immutable ``repository@sha256:...`` ref."""
    match = _IMAGE_REF.fullmatch(image_ref)
    if match is None:
        raise ProvenanceError(
            "image-ref must be an immutable repository@sha256:<64 lowercase hex>"
        )
    return match.group(1)


def verify_runtime(
    runtime_manifest: Path,
    *,
    source_git_sha: str,
    image_ref: str,
) -> dict[str, Any]:
    """Verify the baked source, dependencies, and exact external sources."""
    source_git_sha = _require_hex40(source_git_sha, "source-git-sha")
    image_digest = validate_image_ref(image_ref)
    manifest_identity, manifest_payload = _regular_file_payload_identity(
        runtime_manifest, "runtime manifest"
    )
    manifest = _load_object_bytes(
        manifest_payload, runtime_manifest, "runtime manifest"
    )
    if manifest.get("schema") != RUNTIME_SCHEMA:
        raise ProvenanceError("unexpected crop-distill runtime-manifest schema")
    if manifest.get("base_image") != BASE_IMAGE:
        raise ProvenanceError("runtime manifest has an unexpected base-image digest")
    if manifest.get("model_resolution") != MODEL_RESOLUTION:
        raise ProvenanceError(
            "runtime manifest has unexpected model dependency resolution evidence"
        )
    base_python = _verify_python(manifest.get("base_python"), "base")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ProvenanceError("runtime manifest source must be an object")
    baked_git_sha = _require_hex40(source.get("git_sha"), "baked source git SHA")
    if baked_git_sha != source_git_sha:
        raise ProvenanceError(
            f"source git SHA mismatch: expected {source_git_sha}, got {baked_git_sha}"
        )
    if source.get("selection") != SOURCE_ARCHIVE_SELECTION:
        raise ProvenanceError(
            "runtime manifest source selection is incomplete or unexpected"
        )
    source_identity = _verify_tree_identity(source, "source")
    source_identity["git_sha"] = baked_git_sha

    environments = manifest.get("environments")
    if not isinstance(environments, dict) or set(environments) != {
        "model",
        "scoring",
    }:
        raise ProvenanceError(
            "runtime manifest must contain exactly model and scoring environments"
        )
    environment_identity: dict[str, Any] = {}
    expected_environment_keys = {"python", "requirements_lock", "pip_freeze"}
    for name in sorted(environments):
        environment = environments[name]
        if (
            not isinstance(environment, dict)
            or set(environment) != expected_environment_keys
        ):
            raise ProvenanceError(
                f"runtime manifest {name} environment is incomplete or unexpected"
            )
        environment_identity[name] = {
            "python": _verify_python(environment["python"], name),
            "requirements_lock": _verify_dependency(
                environment["requirements_lock"], f"{name} requirements lock"
            ),
            "pip_freeze": _verify_dependency(
                environment["pip_freeze"], f"{name} pip freeze"
            ),
        }

    external = manifest.get("external_sources")
    if not isinstance(external, dict) or set(external) != {"croma", "clay"}:
        raise ProvenanceError(
            "runtime manifest must contain exactly croma and clay sources"
        )
    expected_external = {
        "croma": (CROMA_GIT_SHA, CROMA_ARCHIVE_SHA256),
        "clay": (CLAY_GIT_SHA, CLAY_ARCHIVE_SHA256),
    }
    external_identity: dict[str, Any] = {}
    for name in sorted(external):
        record = external[name]
        if not isinstance(record, dict):
            raise ProvenanceError(f"external source {name} must be an object")
        expected_git_sha, expected_archive_sha = expected_external[name]
        git_sha = _require_hex40(record.get("git_sha"), f"{name} git SHA")
        if git_sha != expected_git_sha:
            raise ProvenanceError(
                f"{name} git SHA mismatch: expected {expected_git_sha}, got {git_sha}"
            )
        identity = _verify_tree_identity(record, name)
        if identity["archive_sha256"] != expected_archive_sha:
            raise ProvenanceError(
                f"{name} archive SHA256 mismatch: expected {expected_archive_sha}, "
                f"got {identity['archive_sha256']}"
            )
        identity["git_sha"] = git_sha
        external_identity[name] = identity

    return {
        "image": {"ref": image_ref, "digest": image_digest},
        "base_image": BASE_IMAGE,
        "model_resolution": MODEL_RESOLUTION,
        "base_python": base_python,
        "runtime_manifest": manifest_identity,
        "source": source_identity,
        "environments": environment_identity,
        "external_sources": external_identity,
    }


def _verify_split_manifest(
    path: Path,
    *,
    expected_sha256: str,
    split_source_git_sha: str,
    kind: str,
) -> dict[str, Any]:
    identity, manifest_payload = _regular_file_payload_identity(
        path,
        "frozen split manifest",
        expected_sha256=expected_sha256,
    )
    manifest = _load_object_bytes(
        manifest_payload, path, "frozen split manifest"
    )
    manifest_git_sha = _require_hex40(
        manifest.get("git_sha"), "frozen split manifest git_sha"
    )
    if manifest_git_sha != split_source_git_sha:
        raise ProvenanceError(
            "frozen split manifest git_sha does not match the split source"
        )

    artifacts = manifest.get("artifacts")
    expected_files = {
        "lucas_crop_distill_index.parquet",
        "lucas_crop_validator_holdout_index.parquet",
        "lucas_crop_split.json",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_files:
        raise ProvenanceError(
            "frozen split manifest must declare exactly both indexes and the split artifact"
        )
    declared_artifacts: dict[str, Any] = {}
    split_payload: bytes | None = None
    for name in sorted(artifacts):
        declared_sha256 = _require_hex64(
            artifacts[name], f"frozen split artifact {name} SHA256"
        )
        if kind == "crop" and name == _SPLIT_ARTIFACTS["validator_holdout"]:
            # Validator holdout is intentionally invisible to crop workers.
            # Bind its manifest declaration into evidence without touching
            # the sibling path: no stat, open, size read, or content hash.
            declared_artifacts[name] = {
                "path": str(path.parent / name),
                "sha256": declared_sha256,
                "verification": "declaration-only",
            }
        else:
            artifact_path = path.parent / name
            artifact_identity, artifact_payload = (
                _regular_file_payload_identity(
                    artifact_path,
                    f"frozen split artifact {name}",
                    expected_sha256=declared_sha256,
                )
            )
            declared_artifacts[name] = {
                **artifact_identity,
                "verification": "content",
            }
            if name == _SPLIT_ARTIFACTS["split"]:
                split_payload = artifact_payload

    counts: dict[str, int] = {}
    for name in ("n_qualified", "n_distill", "n_holdout"):
        counts[name] = _require_positive_int(
            manifest.get(name), f"frozen split manifest {name}"
        )
    if counts["n_qualified"] != counts["n_distill"] + counts["n_holdout"]:
        raise ProvenanceError("frozen split manifest partition counts do not add up")
    if split_payload is None:
        raise ProvenanceError("frozen split document was not content-verified")
    split_path = path.parent / _SPLIT_ARTIFACTS["split"]
    split_document = _load_object_bytes(
        split_payload, split_path, "frozen split document"
    )
    digests: dict[str, str] = {}
    for name in sorted(_SPLIT_DIGEST_FIELDS):
        manifest_digest = _require_hex64(
            manifest.get(name), f"frozen split manifest {name}"
        )
        split_digest = _require_hex64(
            split_document.get(name), f"frozen split document {name}"
        )
        if manifest_digest != split_digest:
            raise ProvenanceError(
                f"frozen split digest {name} disagrees between manifest and split"
            )
        digests[name] = manifest_digest
    return {
        **identity,
        "git_sha": manifest_git_sha,
        "counts": counts,
        "immutable_digests": digests,
        "declared_artifacts": declared_artifacts,
    }


def _parse_artifacts(values: Iterable[str]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ProvenanceError(
                f"artifact must use name=/absolute/path syntax: {value!r}"
            )
        name, raw_path = value.split("=", 1)
        if _ARTIFACT_NAME.fullmatch(name) is None:
            raise ProvenanceError(f"invalid artifact name: {name!r}")
        path = Path(raw_path)
        if not path.is_absolute():
            raise ProvenanceError(f"artifact {name} path must be absolute")
        if name in artifacts:
            raise ProvenanceError(f"duplicate artifact name: {name}")
        artifacts[name] = path
    return artifacts


def _parse_artifact_sizes(values: Iterable[str]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ProvenanceError(
                f"artifact-size must use name=bytes syntax: {value!r}"
            )
        name, raw_size = value.split("=", 1)
        if _ARTIFACT_NAME.fullmatch(name) is None:
            raise ProvenanceError(f"invalid artifact-size name: {name!r}")
        if name in sizes:
            raise ProvenanceError(f"duplicate artifact-size name: {name}")
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise ProvenanceError(
                f"artifact-size {name} must be a positive integer"
            ) from exc
        sizes[name] = _require_positive_int(size, f"artifact-size {name}")
    return sizes


def _parse_artifact_hashes(values: Iterable[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ProvenanceError(
                f"artifact-sha256 must use name=digest syntax: {value!r}"
            )
        name, digest = value.split("=", 1)
        if _ARTIFACT_NAME.fullmatch(name) is None:
            raise ProvenanceError(f"invalid artifact-sha256 name: {name!r}")
        if name in hashes:
            raise ProvenanceError(f"duplicate artifact-sha256 name: {name}")
        hashes[name] = _require_hex64(digest, f"artifact-sha256 {name}")
    return hashes


def _optional_split(
    args: argparse.Namespace,
    split_source_git_sha: str,
) -> dict[str, Any] | None:
    if (args.split_manifest is None) != (args.split_sha256 is None):
        raise ProvenanceError(
            "split-manifest and split-sha256 must be supplied together"
        )
    if args.split_manifest is None:
        return None
    return _verify_split_manifest(
        args.split_manifest,
        expected_sha256=args.split_sha256,
        split_source_git_sha=split_source_git_sha,
        kind=args.kind,
    )


def _optional_checkpoint(args: argparse.Namespace) -> dict[str, Any] | None:
    supplied = (
        args.checkpoint is not None,
        args.checkpoint_sha256 is not None,
        args.checkpoint_size is not None,
    )
    if any(supplied) and not all(supplied):
        raise ProvenanceError(
            "checkpoint, checkpoint-sha256, and checkpoint-size are one unit"
        )
    if args.checkpoint is None:
        return None
    if not args.checkpoint.is_absolute():
        raise ProvenanceError("model checkpoint path must be absolute")
    # The extractor has already opened this exact path O_NOFOLLOW and copied
    # and hashed those descriptor bytes into an anonymous private snapshot.
    # PyTorch receives only that snapshot. Re-hashing the multi-gigabyte shared
    # checkpoint here would add cost but would not close a new race.
    return {
        "path": str(args.checkpoint),
        "size_bytes": _require_positive_int(
            args.checkpoint_size, "model checkpoint size"
        ),
        "sha256": _require_hex64(
            args.checkpoint_sha256, "model checkpoint SHA256"
        ),
        "verification": "extractor-authenticated-private-snapshot",
    }


def _require_exact_path(actual: Path, expected: Path, label: str) -> None:
    if actual != expected:
        raise ProvenanceError(
            f"{label} must be exactly {expected}, got {actual}"
        )


def _validate_completion_authority(
    args: argparse.Namespace,
    artifacts: dict[str, Path],
) -> None:
    """Bind a completed record to the reviewed Pod authority envelope."""
    if args.run_id != args.pod_uid:
        raise ProvenanceError("completed record requires run-id equal to pod-uid")

    actual_uid = os.geteuid()
    actual_gid = os.getegid()
    if args.kind == "crop":
        if args.model is None:
            raise ProvenanceError("completed crop record requires --model")
        expected_uid = model_process_uid(args.model)
        expected_job = f"ladder-crop-distill-{args.model}"
        if args.job != expected_job:
            raise ProvenanceError(
                f"completed crop job must be exactly {expected_job}"
            )
        if (actual_uid, actual_gid) != (expected_uid, STORAGE_GID):
            raise ProvenanceError(
                "completed crop process identity must be exactly "
                f"{expected_uid}:{STORAGE_GID}, got {actual_uid}:{actual_gid}"
            )
        _require_exact_path(
            args.record_dir,
            CROP_RECORD_DIR,
            "completed crop record-dir",
        )
        expected_split_manifest = (
            WORK_ROOT
            / args.pod_uid
            / "split"
            / CROP_SPLIT_MANIFEST.name
        )
        _require_exact_path(
            args.split_manifest,
            expected_split_manifest,
            "completed crop consumed split-manifest",
        )
        expected_split_sha256 = _require_hex64(
            os.environ.get(_SPLIT_AUTHORITY_ENV),
            f"{_SPLIT_AUTHORITY_ENV} trust anchor",
        )
        if expected_split_sha256 == "0" * 64:
            raise ProvenanceError(
                f"{_SPLIT_AUTHORITY_ENV} trust anchor must be nonzero"
            )
        claimed_split_sha256 = _require_hex64(
            args.split_sha256,
            "completed crop split-manifest SHA256",
        )
        if claimed_split_sha256 != expected_split_sha256:
            raise ProvenanceError(
                "completed crop split-manifest SHA256 mismatch; must equal the "
                f"environment trust anchor {expected_split_sha256}"
            )

        protocol = model_protocol(args.model)
        _require_exact_path(
            args.checkpoint,
            protocol.checkpoint_path,
            "completed crop checkpoint",
        )
        checkpoint_size = _require_positive_int(
            args.checkpoint_size,
            "model checkpoint size",
        )
        checkpoint_sha256 = _require_hex64(
            args.checkpoint_sha256,
            "model checkpoint SHA256",
        )
        if checkpoint_size != protocol.checkpoint_size:
            raise ProvenanceError(
                "completed crop checkpoint size must equal the baked model "
                f"protocol value {protocol.checkpoint_size}"
            )
        if checkpoint_sha256 != protocol.checkpoint_sha256:
            raise ProvenanceError(
                "completed crop checkpoint SHA256 must equal the baked model "
                f"protocol value {protocol.checkpoint_sha256}"
            )
        expected_artifacts = {
            "features": CROP_HEADS_DIR
            / f"{args.pod_uid}--{args.model}_r2_crop_features.parquet",
            "oof": CROP_HEADS_DIR
            / f"{args.pod_uid}--{args.model}_r2_crop_distillability.json",
        }
        if artifacts != expected_artifacts:
            raise ProvenanceError(
                "completed crop artifacts must use the exact Pod/model-owned "
                f"paths: {expected_artifacts}"
            )
        return

    if args.model is not None:
        raise ProvenanceError("completed split record must not specify --model")
    if args.job != "ladder-lucas-crop-split":
        raise ProvenanceError(
            "completed split job must be exactly ladder-lucas-crop-split"
        )
    if (actual_uid, actual_gid) != (STORAGE_UID, STORAGE_GID):
        raise ProvenanceError(
            "completed split process identity must be exactly "
            f"{STORAGE_UID}:{STORAGE_GID}, got {actual_uid}:{actual_gid}"
        )
    _require_exact_path(
        args.record_dir,
        SPLIT_RECORD_DIR,
        "completed split record-dir",
    )
    _require_exact_path(
        args.split_manifest,
        CROP_SPLIT_MANIFEST,
        "completed split split-manifest",
    )
    expected_artifacts = {
        "index": CROP_INDEX,
        "validator_holdout": DISTILL_DIR
        / _SPLIT_ARTIFACTS["validator_holdout"],
        "split": CROP_SPLIT,
        "manifest": CROP_SPLIT_MANIFEST,
    }
    if artifacts != expected_artifacts:
        raise ProvenanceError(
            "completed split artifacts must use the exact protocol paths: "
            f"{expected_artifacts}"
        )


def _source_access_authority(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return the exact upstream repair authority for a completed split.

    Crop workers are authorized by the frozen split manifest instead.  Failed
    records remain diagnostic-only and deliberately do not turn partially
    parsed environment claims into trusted source-access evidence.
    """
    values = (
        args.source_access_plan_sha256,
        args.source_access_plan_pod_uid,
        args.source_access_completion_sha256,
        args.source_access_completion_pod_uid,
    )
    if args.status != "completed":
        return None
    if args.kind == "crop":
        if any(value is not None for value in values):
            raise ProvenanceError(
                "completed crop record cannot claim source-access authority"
            )
        return None
    if any(value is None for value in values):
        raise ProvenanceError(
            "completed split record requires the PLAN and APPLY SHA256/Pod UID "
            "source-access authority"
        )
    plan_sha256 = _require_hex64(
        args.source_access_plan_sha256,
        "source-access PLAN SHA256",
    )
    completion_sha256 = _require_hex64(
        args.source_access_completion_sha256,
        "source-access APPLY completion SHA256",
    )
    if plan_sha256 == "0" * 64 or completion_sha256 == "0" * 64:
        raise ProvenanceError("source-access authority digests must be nonzero")
    return {
        "plan": {
            "sha256": plan_sha256,
            "pod_uid": _require_safe_id(
                args.source_access_plan_pod_uid,
                "source-access PLAN Pod UID",
            ),
        },
        "completion": {
            "sha256": completion_sha256,
            "pod_uid": _require_safe_id(
                args.source_access_completion_pod_uid,
                "source-access APPLY completion Pod UID",
            ),
        },
    }


def _validate_terminal_args(
    args: argparse.Namespace,
    artifacts: dict[str, Path],
    artifact_sizes: dict[str, int],
    artifact_hashes: dict[str, str],
) -> None:
    _source_access_authority(args)
    if args.status == "completed":
        if args.exit_code != 0:
            raise ProvenanceError("completed status requires exit-code 0")
        if args.failure_stage is not None:
            raise ProvenanceError("completed status cannot have a failure-stage")
        if args.split_manifest is None:
            raise ProvenanceError("completed status requires a frozen split manifest")
        if args.split_source_git_sha is None:
            raise ProvenanceError("completed status requires --split-source-git-sha")
        if args.kind == "crop":
            if args.checkpoint is None:
                raise ProvenanceError("completed crop record requires a checkpoint")
            if set(artifacts) != _CROP_ARTIFACTS:
                raise ProvenanceError(
                    "completed crop record requires exactly features and oof artifacts"
                )
            if set(artifact_sizes) != set(artifacts):
                raise ProvenanceError(
                    "completed crop record requires a published size for every artifact"
                )
            if set(artifact_hashes) != set(artifacts):
                raise ProvenanceError(
                    "completed crop record requires a published SHA256 for every artifact"
                )
        else:
            if artifact_sizes or artifact_hashes:
                raise ProvenanceError(
                    "completed split record cannot claim crop publication identities"
                )
            if args.checkpoint is not None:
                raise ProvenanceError("completed split record must not have a checkpoint")
            if set(artifacts) != set(_SPLIT_ARTIFACTS):
                raise ProvenanceError(
                    "completed split record requires index, validator_holdout, "
                    "split, and manifest artifacts"
                )
            for name, filename in _SPLIT_ARTIFACTS.items():
                if artifacts[name].name != filename:
                    raise ProvenanceError(
                        f"split artifact {name} must be named {filename}"
                    )
            if artifacts["manifest"] != args.split_manifest:
                raise ProvenanceError(
                    "split manifest artifact path must equal --split-manifest"
                )
        _validate_completion_authority(args, artifacts)
    else:
        if args.exit_code <= 0:
            raise ProvenanceError("failed status requires a positive exit-code")
        if args.failure_stage is not None and (
            len(args.failure_stage) > 191
            or any(ord(character) < 32 for character in args.failure_stage)
        ):
            raise ProvenanceError("failure-stage contains invalid characters")


def _recover_linked_temporary(
    parent_fd: int,
    *,
    target_name: str,
    payload: bytes,
) -> bool:
    """Remove only stale publication links to an already-complete target.

    ``write_once_bytes`` publishes with ``link(temp, target)``.  A process
    death after that atomic link but before ``unlink(temp)`` leaves the target
    valid but with link count two.  The next identical publication may safely
    finish only that interrupted cleanup.  Extra links that are not matching
    private publication temporaries remain a hard failure.
    """
    target_fd: int | None = None
    try:
        try:
            target_fd = os.open(target_name, _READ_FLAGS, dir_fd=parent_fd)
        except OSError:
            return False
        before = os.fstat(target_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 2:
            return False
        blocks: list[bytes] = []
        while True:
            block = os.read(target_fd, _COPY_BLOCK_SIZE)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(target_fd)
        if not _same_file(before, after) or b"".join(blocks) != payload:
            return False

        prefix = f".{target_name}."
        suffix = ".create"
        linked_temporaries: list[str] = []
        for name in os.listdir(parent_fd):
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            try:
                identity = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except OSError:
                return False
            if (
                stat.S_ISREG(identity.st_mode)
                and identity.st_dev == before.st_dev
                and identity.st_ino == before.st_ino
            ):
                linked_temporaries.append(name)
        if before.st_nlink != 1 + len(linked_temporaries):
            return False
        for name in sorted(linked_temporaries):
            os.unlink(name, dir_fd=parent_fd)
        return True
    finally:
        if target_fd is not None:
            os.close(target_fd)


def write_once_bytes(path: Path, payload: bytes) -> None:
    """Create a group-readable record through a safe per-run directory."""
    parent_fd = _open_directory(path.parent, create=True, create_mode=0o750)
    parent_identity = os.fstat(parent_fd)
    if parent_identity.st_uid != os.geteuid():
        os.close(parent_fd)
        raise ProvenanceError(
            f"terminal record directory is not owned by runtime UID: {path.parent}"
        )
    if stat.S_IMODE(parent_identity.st_mode) & 0o022:
        os.close(parent_fd)
        raise ProvenanceError(
            "terminal record directory grants group/other write access: "
            f"{path.parent}"
        )
    try:
        os.fchmod(parent_fd, 0o750)
    except OSError as exc:
        os.close(parent_fd)
        raise ProvenanceError(
            f"cannot set terminal record directory mode 0750: {path.parent}: {exc}"
        ) from exc
    temporary_name = (
        f".{path.name}.{os.getpid()}.{secrets.token_hex(16)}.create"
    )
    temporary_fd: int | None = None
    temporary_created = False
    try:
        temporary_fd = os.open(
            temporary_name,
            _CREATE_FLAGS,
            0o600,
            dir_fd=parent_fd,
        )
        temporary_created = True
        view = memoryview(payload)
        while view:
            written = os.write(temporary_fd, view)
            view = view[written:]
        os.fsync(temporary_fd)
        os.fchmod(temporary_fd, 0o444)
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            _recover_linked_temporary(
                parent_fd,
                target_name=path.name,
                payload=payload,
            )
            try:
                existing, _ = _read_regular_at(
                    parent_fd,
                    path.name,
                    label=f"existing terminal record {path}",
                )
            except ProvenanceError as exc:
                raise ProvenanceError(
                    f"cannot verify raced terminal record {path}: {exc}"
                ) from exc
            if existing != payload:
                raise ProvenanceError(
                    f"refusing to overwrite mismatched terminal record: {path}"
                )
        os.unlink(temporary_name, dir_fd=parent_fd)
        temporary_created = False
        try:
            os.fsync(parent_fd)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise ProvenanceError(
                    f"cannot sync terminal record directory {path.parent}: {exc}"
                ) from exc

        verified, _ = _read_regular_at(
            parent_fd,
            path.name,
            label=f"terminal record {path}",
        )
        if verified != payload:
            raise ProvenanceError(f"terminal record verification failed: {path}")
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except OSError:
                pass
        if temporary_fd is not None:
            os.close(temporary_fd)
        os.close(parent_fd)


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    """Validate all available evidence and publish one terminal record."""
    run_id = _require_safe_id(args.run_id, "run-id")
    job = _require_safe_id(args.job, "job")
    pod_uid = _require_safe_id(args.pod_uid, "pod-uid")
    # Failure traps may inherit half-populated or malformed pipeline flags.
    # They are diagnostic-only and must not let those flags suppress the one
    # terminal record. Completed records continue to validate every artifact.
    artifacts = (
        _parse_artifacts(args.artifact)
        if args.status == "completed"
        else {}
    )
    artifact_sizes = (
        _parse_artifact_sizes(args.artifact_size)
        if args.status == "completed"
        else {}
    )
    artifact_hashes = (
        _parse_artifact_hashes(args.artifact_sha256)
        if args.status == "completed"
        else {}
    )
    _validate_terminal_args(
        args, artifacts, artifact_sizes, artifact_hashes
    )
    source_access = _source_access_authority(args)

    if args.status == "failed":
        # A terminal failure record is diagnostic evidence, not completion
        # evidence. Never dereference runtime/split/checkpoint/output paths,
        # even if an EXIT trap inherited those flags.
        runtime: dict[str, Any] = {
            "verification": "not-dereferenced",
            "claimed": {
                "image_ref": _bounded_claim(args.image_ref, "image-ref"),
                "source_git_sha": _bounded_claim(
                    args.source_git_sha, "source-git-sha"
                ),
                "runtime_manifest": _bounded_claim(
                    str(args.runtime_manifest), "runtime-manifest"
                ),
            },
        }
        raw_split_sha256 = (
            args.split_sha256
            if isinstance(args.split_sha256, str)
            else "<missing>"
        )
        split = {
            "verification": "not-dereferenced",
            "claimed_sha256": _bounded_claim(
                raw_split_sha256, "split-manifest-sha256"
            ),
        }
        checkpoint = None
        output_identities: dict[str, Any] = {}
    else:
        runtime_identity = verify_runtime(
            args.runtime_manifest,
            source_git_sha=args.source_git_sha,
            image_ref=args.image_ref,
        )
        runtime = {"verification": "verified", **runtime_identity}
        split_source_git_sha = _require_hex40(
            args.split_source_git_sha,
            "split-source-git-sha",
        )
        split = _optional_split(args, split_source_git_sha)
        checkpoint = _optional_checkpoint(args)
        output_identities = {
            name: _regular_file_identity(
                path,
                f"output artifact {name}",
                expected_size=artifact_sizes.get(name),
                expected_sha256=artifact_hashes.get(name),
            )
            for name, path in sorted(artifacts.items())
        }

    record = {
        "schema": COMPLETION_SCHEMA,
        "kind": args.kind,
        "model": args.model,
        "run_id": run_id,
        "job": job,
        "pod_uid": pod_uid,
        "process_identity": {
            "effective_uid": os.geteuid(),
            "effective_gid": os.getegid(),
        },
        "terminal": {
            "status": args.status,
            "exit_code": args.exit_code,
            "failure_stage": args.failure_stage,
        },
        "runtime": runtime,
        "source_access": source_access,
        "split_manifest": split,
        "checkpoint": checkpoint,
        "artifacts": output_identities,
    }
    target = args.record_dir / pod_uid / "completion.json"
    payload = canonical_json_bytes(record)
    write_once_bytes(target, payload)
    return {
        "record": str(target),
        "record_sha256": hashlib.sha256(payload).hexdigest(),
        **record,
    }


def terminal_evidence_line(result: dict[str, Any]) -> str:
    """Encode the exact canonical terminal-record bytes for log capture."""
    try:
        record = {name: result[name] for name in _COMPLETION_RECORD_FIELDS}
        claimed_sha256 = result["record_sha256"]
    except KeyError as exc:
        raise ProvenanceError(
            f"terminal evidence result is missing {exc.args[0]}"
        ) from exc
    payload = canonical_json_bytes(record)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if claimed_sha256 != actual_sha256:
        raise ProvenanceError(
            "terminal evidence result SHA256 does not match canonical record bytes"
        )
    encoded = base64.b64encode(payload).decode("ascii")
    return f"{TERMINAL_EVIDENCE_PREFIX} {actual_sha256} {encoded}"


def parse_terminal_evidence_line(
    line: str,
) -> tuple[str, bytes, dict[str, Any]]:
    """Strictly validate one machine-delimited terminal-evidence line."""
    if not isinstance(line, str):
        raise ProvenanceError("terminal evidence line must be text")
    line = line.removesuffix("\n")
    if not line or "\n" in line or "\r" in line:
        raise ProvenanceError("terminal evidence must contain exactly one line")
    parts = line.split(" ")
    if len(parts) != 3 or parts[0] != TERMINAL_EVIDENCE_PREFIX:
        raise ProvenanceError(
            "terminal evidence must use prefix, SHA256, and base64 fields"
        )
    digest = _require_hex64(parts[1], "terminal evidence SHA256")
    try:
        payload = base64.b64decode(
            parts[2].encode("ascii"),
            validate=True,
        )
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise ProvenanceError("terminal evidence payload is not strict base64") from exc
    if base64.b64encode(payload).decode("ascii") != parts[2]:
        raise ProvenanceError("terminal evidence payload is not canonical base64")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ProvenanceError("terminal evidence payload SHA256 mismatch")
    try:
        record = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError("terminal evidence payload is not valid JSON") from exc
    if not isinstance(record, dict):
        raise ProvenanceError("terminal evidence payload must be a JSON object")
    if set(record) != _COMPLETION_RECORD_FIELDS:
        raise ProvenanceError(
            "terminal evidence payload has unexpected completion-record fields"
        )
    if record.get("schema") != COMPLETION_SCHEMA:
        raise ProvenanceError("terminal evidence payload has an unexpected schema")
    if canonical_json_bytes(record) != payload:
        raise ProvenanceError("terminal evidence payload is not canonical JSON")
    return digest, payload, record


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify-runtime", help="verify baked runtime identity before GPU work"
    )
    _add_runtime_arguments(verify)

    complete = subparsers.add_parser(
        "finalize", help="publish one immutable per-Pod terminal record"
    )
    _add_runtime_arguments(complete)
    complete.add_argument("--record-dir", type=Path, required=True)
    complete.add_argument("--kind", choices=("crop", "split"), default="crop")
    complete.add_argument("--model", choices=MODEL_KEYS)
    complete.add_argument("--run-id", required=True)
    complete.add_argument("--job", required=True)
    complete.add_argument("--pod-uid", required=True)
    complete.add_argument("--status", choices=("completed", "failed"), required=True)
    complete.add_argument("--exit-code", type=int, default=0)
    complete.add_argument("--failure-stage")
    complete.add_argument("--split-manifest", type=Path)
    complete.add_argument("--split-sha256")
    complete.add_argument("--split-source-git-sha")
    complete.add_argument("--source-access-plan-sha256")
    complete.add_argument("--source-access-plan-pod-uid")
    complete.add_argument("--source-access-completion-sha256")
    complete.add_argument("--source-access-completion-pod-uid")
    complete.add_argument("--checkpoint", type=Path)
    complete.add_argument("--checkpoint-sha256")
    complete.add_argument("--checkpoint-size", type=int)
    complete.add_argument(
        "--artifact", action="append", default=[], metavar="NAME=/ABSOLUTE/PATH"
    )
    complete.add_argument(
        "--artifact-size", action="append", default=[], metavar="NAME=BYTES"
    )
    complete.add_argument(
        "--artifact-sha256", action="append", default=[], metavar="NAME=SHA256"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify-runtime":
            result = verify_runtime(
                args.runtime_manifest,
                source_git_sha=args.source_git_sha,
                image_ref=args.image_ref,
            )
        else:
            result = finalize(args)
    except ProvenanceError as exc:
        raise SystemExit(f"crop-distill provenance refused: {exc}") from exc
    if args.command == "finalize":
        print(terminal_evidence_line(result))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
