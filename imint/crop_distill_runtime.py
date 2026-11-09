"""Identity checks for the sealed crop-distill runtime image.

``verify_runtime`` authenticates what ``docker/ladder-crop-distill`` bakes:
the source tree, both locked environments and the pinned CROMA and Clay
sources. ``snapshot_tree`` and ``tree_payload_sha256`` give the content
identity of a sealed tree. The crop-distill scripts and the field-truth
evaluations (``imint.eval.fieldtruth``, ``imint.eval.nfi_amendment``) share
this implementation; ``scripts/crop_distill_provenance.py`` imports it from
here.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

RUNTIME_SCHEMA = "imint-crop-distill-runtime-v1"
TREE_SCHEMA = "imint-content-tree-v1"

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
_PYTHON_VERSION = re.compile(r"^3\.11\.\d+$")

_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
_DIR_FLAGS = _READ_FLAGS | os.O_DIRECTORY
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
