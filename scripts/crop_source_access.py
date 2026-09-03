#!/usr/bin/env python3
"""Plan and apply the narrowly-scoped LUCAS crop source-access repair.

``plan`` is a read-only dataset pass.  It derives the exact crop-window tile
set from the pinned producer index, captures every candidate through
descriptor-relative no-follow opens, and publishes a write-once canonical
plan.  ``apply`` consumes only a reviewed plan Pod UID and SHA-256. Kubernetes
projects the whole ``unified_v2_512`` dataset subPath RW, but the payload opens
only those exact planned files, with O_RDONLY descriptors used solely for
metadata syscalls. It publishes a write-once completion record proving that
bytes, size, inode, and mtime did not change.
"""

from __future__ import annotations

import argparse
import base64
import errno
import hashlib
import io
import json
import os
import re
import stat
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__:
    from .atomic_npz import exclusive_dataset_lock
    from .crop_distill_protocol import (
        DATA_DIR,
        RUNTIME_MANIFEST,
        SOURCE_ACCESS_APPLY_RECORD_DIR,
        SOURCE_ACCESS_EXPECTED_CANDIDATES,
        SOURCE_ACCESS_EXPECTED_CROP_ROWS,
        SOURCE_ACCESS_EXPECTED_NOOPS,
        SOURCE_ACCESS_EXPECTED_REPAIRS,
        SOURCE_ACCESS_INDEX_INPUT,
        SOURCE_ACCESS_INDEX_SHA256,
        SOURCE_ACCESS_INDEX_SIZE,
        SOURCE_ACCESS_LOCK_FILE,
        SOURCE_ACCESS_LOCK_MODE,
        SOURCE_ACCESS_PLAN_INPUT,
        SOURCE_ACCESS_PLAN_RECORD_DIR,
        SOURCE_ACCESS_TARGET_GID,
        SOURCE_ACCESS_TARGET_MODE,
        SOURCE_ACCESS_TARGET_UID,
        RuntimeIdentity,
        require_process_identity,
        require_source_access_run_id,
        require_source_access_sha256,
        runtime_identity,
    )
    from .crop_distill_provenance import (
        canonical_json_bytes,
        verify_runtime,
        write_once_bytes,
    )
else:
    from atomic_npz import exclusive_dataset_lock
    from crop_distill_protocol import (
        DATA_DIR,
        RUNTIME_MANIFEST,
        SOURCE_ACCESS_APPLY_RECORD_DIR,
        SOURCE_ACCESS_EXPECTED_CANDIDATES,
        SOURCE_ACCESS_EXPECTED_CROP_ROWS,
        SOURCE_ACCESS_EXPECTED_NOOPS,
        SOURCE_ACCESS_EXPECTED_REPAIRS,
        SOURCE_ACCESS_INDEX_INPUT,
        SOURCE_ACCESS_INDEX_SHA256,
        SOURCE_ACCESS_INDEX_SIZE,
        SOURCE_ACCESS_LOCK_FILE,
        SOURCE_ACCESS_LOCK_MODE,
        SOURCE_ACCESS_PLAN_INPUT,
        SOURCE_ACCESS_PLAN_RECORD_DIR,
        SOURCE_ACCESS_TARGET_GID,
        SOURCE_ACCESS_TARGET_MODE,
        SOURCE_ACCESS_TARGET_UID,
        RuntimeIdentity,
        require_process_identity,
        require_source_access_run_id,
        require_source_access_sha256,
        runtime_identity,
    )
    from crop_distill_provenance import (
        canonical_json_bytes,
        verify_runtime,
        write_once_bytes,
    )

PLAN_SCHEMA = "imint-crop-source-access-plan-v1"
COMPLETION_SCHEMA = "imint-crop-source-access-completion-v1"
PLAN_MARKER = "CROP_SOURCE_ACCESS_PLAN_V1"
COMPLETION_MARKER = "CROP_SOURCE_ACCESS_COMPLETION_V1"
FREEZE_LEASE_SCHEMA = "imint-crop-source-freeze-lease-v1"
FREEZE_LEASE_PATH = Path("/var/run/crop-source-freeze/lease.json")
FREEZE_LEASE_MAX_LIFETIME_SECONDS = 180

ACTION_REPAIR = "repair-root-root-0600"
ACTION_ACCEPT_0644 = "accept-root-readable-0644"
ACTION_ALREADY_CORRECT = "accept-root-2000-0640"
_ACTION_PARTIAL_REPAIR = "resume-root-2000-0600"
PLAN_ACTIONS = frozenset(
    {ACTION_REPAIR, ACTION_ACCEPT_0644, ACTION_ALREADY_CORRECT}
)
_IDENTITY_FIELDS = frozenset(
    {"dev", "inode", "size", "mtime_ns", "ctime_ns", "uid", "gid", "mode", "nlink", "sha256"}
)

_READ_SIZE = 1 << 20
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


class SourceAccessError(RuntimeError):
    """The source-access authority or filesystem state is unsafe."""


def require_fresh_freeze_lease(
    path: Path,
    *,
    expected_phase: str,
    now_ns: int | None = None,
) -> dict[str, object]:
    """Require the watchdog's short-lived, clean lease for this exact phase."""
    try:
        payload = path.read_bytes()
        lease = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceAccessError("external freeze lease is missing or invalid") from exc
    expected_fields = {
        "schema",
        "run_id",
        "phase",
        "status",
        "sequence",
        "heartbeat_unix_ns",
        "valid_until_unix_ns",
        "snapshot_sha256",
        "controller_snapshot_sha256",
    }
    if not isinstance(lease, dict) or set(lease) != expected_fields:
        raise SourceAccessError("external freeze lease schema is unexpected")
    if canonical_json_bytes(lease) != payload:
        raise SourceAccessError("external freeze lease is not canonical JSON")
    if lease.get("schema") != FREEZE_LEASE_SCHEMA:
        raise SourceAccessError("external freeze lease version is unexpected")
    if lease.get("status") != "held" or lease.get("phase") != expected_phase:
        raise SourceAccessError(
            f"external freeze lease is not held for phase {expected_phase}"
        )
    if (
        not isinstance(lease.get("run_id"), str)
        or not lease["run_id"]
        or type(lease.get("sequence")) is not int
        or int(lease["sequence"]) < 0
        or type(lease.get("heartbeat_unix_ns")) is not int
        or type(lease.get("valid_until_unix_ns")) is not int
        or re.fullmatch(r"[0-9a-f]{64}", str(lease.get("snapshot_sha256"))) is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(lease.get("controller_snapshot_sha256"))
        )
        is None
    ):
        raise SourceAccessError("external freeze lease values are malformed")
    heartbeat = int(lease["heartbeat_unix_ns"])
    valid_until = int(lease["valid_until_unix_ns"])
    current = time.time_ns() if now_ns is None else now_ns
    max_lifetime = FREEZE_LEASE_MAX_LIFETIME_SECONDS * 1_000_000_000
    if (
        heartbeat > current
        or valid_until <= current
        or valid_until <= heartbeat
        or valid_until - heartbeat > max_lifetime
    ):
        raise SourceAccessError("external freeze watchdog lease is stale")
    return lease


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _open_directory_tree(path: Path, *, create: bool = False) -> int:
    """Open an absolute directory without following any path component."""
    absolute = _absolute(path)
    flags = os.O_RDONLY | _O_CLOEXEC | _O_DIRECTORY | _O_NOFOLLOW
    current_fd = os.open(os.sep, flags)
    try:
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise SourceAccessError(
                    f"unsafe directory component {component!r}: {absolute}"
                )
            if create:
                try:
                    os.mkdir(component, 0o750, dir_fd=current_fd)
                except FileExistsError:
                    pass
            child_fd = os.open(component, flags, dir_fd=current_fd)
            identity = os.fstat(child_fd)
            if not stat.S_ISDIR(identity.st_mode):
                os.close(child_fd)
                raise SourceAccessError(f"not a directory: {absolute}")
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _safe_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not value.isspace()
        and "\x00" not in value
        and value not in {".", ".."}
        and Path(value).name == value
    )


def _open_regular_at(directory_fd: int, name: str) -> int:
    if not _safe_name(name):
        raise SourceAccessError(f"unsafe source tile name: {name!r}")
    # Metadata changes do not require a data-write descriptor.  Keeping this
    # O_RDONLY ensures even APPLY cannot modify NPZ bytes through its fd.
    flags = os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise SourceAccessError(f"source tile must not be a symlink: {name}") from exc
        raise
    identity = os.fstat(fd)
    if not stat.S_ISREG(identity.st_mode):
        os.close(fd)
        raise SourceAccessError(f"source tile is not a regular file: {name}")
    if identity.st_nlink != 1:
        os.close(fd)
        raise SourceAccessError(
            f"source tile must have link count 1: {name}: {identity.st_nlink}"
        )
    if identity.st_size <= 0:
        os.close(fd)
        raise SourceAccessError(f"source tile is empty: {name}")
    return fd


def _path_matches_fd(directory_fd: int, name: str, identity: os.stat_result) -> None:
    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or current.st_dev != identity.st_dev
        or current.st_ino != identity.st_ino
        or current.st_size != identity.st_size
        or current.st_mtime_ns != identity.st_mtime_ns
    ):
        raise SourceAccessError(f"source tile path/fd identity changed: {name}")


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, _READ_SIZE)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _identity(fd: int, *, include_sha256: bool = True) -> dict[str, object]:
    before = os.fstat(fd)
    digest = _sha256_fd(fd) if include_sha256 else None
    after = os.fstat(fd)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or before.st_nlink != after.st_nlink
    ):
        raise SourceAccessError("file changed while its identity was captured")
    result: dict[str, object] = {
        "dev": int(after.st_dev),
        "inode": int(after.st_ino),
        "size": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
        "uid": int(after.st_uid),
        "gid": int(after.st_gid),
        "mode": format(stat.S_IMODE(after.st_mode), "04o"),
        "nlink": int(after.st_nlink),
    }
    if include_sha256:
        result["sha256"] = digest
    return result


def _capture_path(path: Path) -> tuple[bytes, dict[str, object]]:
    parent_fd = _open_directory_tree(path.parent)
    fd: int | None = None
    try:
        fd = _open_regular_at(parent_fd, path.name)
        identity = _identity(fd)
        _path_matches_fd(parent_fd, path.name, os.fstat(fd))
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, _READ_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        final = os.fstat(fd)
        _path_matches_fd(parent_fd, path.name, final)
        if len(payload) != identity["size"]:
            raise SourceAccessError(f"file changed while captured: {path}")
        if hashlib.sha256(payload).hexdigest() != identity["sha256"]:
            raise SourceAccessError(f"file digest changed while captured: {path}")
        if any(
            identity[key] != value
            for key, value in {
                "dev": int(final.st_dev),
                "inode": int(final.st_ino),
                "size": int(final.st_size),
                "mtime_ns": int(final.st_mtime_ns),
                "ctime_ns": int(final.st_ctime_ns),
                "uid": int(final.st_uid),
                "gid": int(final.st_gid),
                "mode": format(stat.S_IMODE(final.st_mode), "04o"),
                "nlink": int(final.st_nlink),
            }.items()
        ):
            raise SourceAccessError(f"file metadata changed while captured: {path}")
        return payload, identity
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


def derive_crop_window_candidates(index_payload: bytes) -> tuple[list[str], int, list[int]]:
    """Derive candidates in the same crop-then-window order as split build."""
    if __package__:
        from .build_lucas_crop_split import (
            _derived_crop_window,
            _prepare_source_crop_rows,
        )
    else:
        from build_lucas_crop_split import (
            _derived_crop_window,
            _prepare_source_crop_rows,
        )
    import pandas as pd

    frame = pd.read_parquet(io.BytesIO(index_payload))
    if "tile_path" in frame.columns:
        raise SourceAccessError(
            "producer LUCAS index must not supply tile_path authority"
        )
    crops, problem = _prepare_source_crop_rows(frame)
    if problem:
        raise SourceAccessError(problem)
    assert crops is not None
    start, stop = _derived_crop_window()
    in_window = (
        (crops["row"] >= start)
        & (crops["row"] < stop)
        & (crops["col"] >= start)
        & (crops["col"] < stop)
    )
    windowed = crops[in_window]
    names = sorted(set(windowed["tile_name"].tolist()))
    if any(not _safe_name(name) for name in names):
        raise SourceAccessError("source index produced an unsafe tile name")
    return names, len(windowed), [start, stop]


def _classify(identity: Mapping[str, object]) -> str:
    uid = identity["uid"]
    gid = identity["gid"]
    mode = identity["mode"]
    if (uid, gid, mode) == (0, 0, "0600"):
        return ACTION_REPAIR
    if uid == 0 and gid in {0, SOURCE_ACCESS_TARGET_GID} and mode == "0644":
        return ACTION_ACCEPT_0644
    if (uid, gid, mode) == (
        SOURCE_ACCESS_TARGET_UID,
        SOURCE_ACCESS_TARGET_GID,
        format(SOURCE_ACCESS_TARGET_MODE, "04o"),
    ):
        return ACTION_ALREADY_CORRECT
    raise SourceAccessError(
        f"unreviewed source metadata uid={uid} gid={gid} mode={mode}"
    )


def runtime_binding(identity: RuntimeIdentity, verified: Mapping[str, Any]) -> dict[str, str]:
    runtime_manifest = verified.get("runtime_manifest")
    source = verified.get("source")
    if not isinstance(runtime_manifest, Mapping) or not isinstance(source, Mapping):
        raise SourceAccessError("runtime verification lacks source identities")
    return {
        "source_git_sha": identity.source_git_sha,
        "image_ref": identity.image_ref,
        "runtime_manifest_sha256": str(runtime_manifest.get("sha256")),
        "source_payload_sha256": str(source.get("payload_sha256")),
    }


def build_plan_record(
    *,
    identity: RuntimeIdentity,
    verified_runtime: Mapping[str, Any],
    source_index: Path,
    data_dir: Path,
    expected_index_sha256: str,
    expected_index_size: int,
    enforce_production_counts: bool,
) -> dict[str, object]:
    index_payload, index_identity = _capture_path(source_index)
    if index_identity["sha256"] != expected_index_sha256:
        raise SourceAccessError("LUCAS source-index SHA256 differs from Git authority")
    if index_identity["size"] != expected_index_size:
        raise SourceAccessError("LUCAS source-index size differs from Git authority")

    names, crop_rows, crop_window = derive_crop_window_candidates(index_payload)
    data_fd = _open_directory_tree(data_dir)
    files: list[dict[str, object]] = []
    try:
        for tile_name in names:
            file_name = f"{tile_name}.npz"
            fd = _open_regular_at(data_fd, file_name)
            try:
                file_identity = _identity(fd)
                _path_matches_fd(data_fd, file_name, os.fstat(fd))
            finally:
                os.close(fd)
            files.append(
                {
                    "tile_name": tile_name,
                    "file_name": file_name,
                    "path": str(data_dir / file_name),
                    **file_identity,
                    "action": _classify(file_identity),
                }
            )
    finally:
        os.close(data_fd)

    actions = Counter(str(record["action"]) for record in files)
    noops = actions[ACTION_ACCEPT_0644] + actions[ACTION_ALREADY_CORRECT]
    if enforce_production_counts and (
        crop_rows != SOURCE_ACCESS_EXPECTED_CROP_ROWS
        or len(files) != SOURCE_ACCESS_EXPECTED_CANDIDATES
        or actions[ACTION_REPAIR] != SOURCE_ACCESS_EXPECTED_REPAIRS
        or noops != SOURCE_ACCESS_EXPECTED_NOOPS
    ):
        raise SourceAccessError(
            "source-access cardinality differs from reviewed production authority: "
            f"rows={crop_rows}, candidates={len(files)}, "
            f"repairs={actions[ACTION_REPAIR]}, noops={noops}"
        )
    return {
        "schema": PLAN_SCHEMA,
        "pod_uid": identity.pod_uid,
        "runtime": runtime_binding(identity, verified_runtime),
        "source_index": {
            "path": str(source_index),
            **index_identity,
        },
        "data_dir": str(data_dir),
        "crop_window": crop_window,
        "crop_rows": crop_rows,
        "target": {
            "uid": SOURCE_ACCESS_TARGET_UID,
            "gid": SOURCE_ACCESS_TARGET_GID,
            "mode": format(SOURCE_ACCESS_TARGET_MODE, "04o"),
        },
        "summary": {
            "candidates": len(files),
            "repairs": actions[ACTION_REPAIR],
            "accepted_0644": actions[ACTION_ACCEPT_0644],
            "already_correct": actions[ACTION_ALREADY_CORRECT],
        },
        "files": files,
    }


def _marker(prefix: str, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"{prefix} {digest} {encoded}"


def publish_plan(record: Mapping[str, object], path: Path) -> tuple[str, bytes]:
    payload = canonical_json_bytes(dict(record))
    write_once_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest(), payload


def _valid_identity_record(record: object) -> bool:
    if not isinstance(record, dict) or set(record) != _IDENTITY_FIELDS:
        return False
    return (
        all(
            type(record.get(key)) is int
            for key in (
                "dev", "inode", "size", "mtime_ns", "ctime_ns",
                "uid", "gid", "nlink",
            )
        )
        and record.get("nlink") == 1
        and isinstance(record.get("mode"), str)
        and re.fullmatch(r"0[0-7]{3}", str(record.get("mode"))) is not None
        and re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))) is not None
    )


def load_plan(
    path: Path,
    *,
    expected_sha256: str,
    identity: RuntimeIdentity,
    expected_plan_pod_uid: str,
    expected_index_sha256: str,
    expected_index_size: int,
    expected_index_path: Path,
    expected_data_dir: Path,
    expected_runtime_binding: Mapping[str, str],
) -> tuple[dict[str, object], bytes]:
    payload, _ = _capture_path(path)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise SourceAccessError("source-access plan SHA256 differs from Git authority")
    try:
        plan = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SourceAccessError("source-access plan is not valid JSON") from exc
    if not isinstance(plan, dict) or canonical_json_bytes(plan) != payload:
        raise SourceAccessError("source-access plan is not canonical JSON")
    expected_fields = {
        "schema", "pod_uid", "runtime", "source_index", "data_dir", "crop_window",
        "crop_rows", "target", "summary", "files",
    }
    if set(plan) != expected_fields or plan["schema"] != PLAN_SCHEMA:
        raise SourceAccessError("source-access plan schema is unexpected")
    if plan.get("pod_uid") != expected_plan_pod_uid:
        raise SourceAccessError("source-access plan Pod UID mismatch")
    runtime = plan.get("runtime")
    if not isinstance(runtime, dict) or runtime != dict(expected_runtime_binding):
        raise SourceAccessError("source-access plan runtime identity mismatch")
    index = plan.get("source_index")
    if not isinstance(index, dict) or (
        set(index) != _IDENTITY_FIELDS | {"path"}
        or
        index.get("path") != str(expected_index_path)
        or
        index.get("sha256") != expected_index_sha256
        or index.get("size") != expected_index_size
    ):
        raise SourceAccessError("source-access plan index identity mismatch")
    if not _valid_identity_record({key: index[key] for key in _IDENTITY_FIELDS}):
        raise SourceAccessError("source-access plan index metadata is malformed")
    if plan.get("data_dir") != str(expected_data_dir):
        raise SourceAccessError("source-access plan data directory mismatch")
    if plan.get("crop_window") != [8, 504]:
        raise SourceAccessError("source-access plan crop window mismatch")
    if plan.get("crop_rows") != SOURCE_ACCESS_EXPECTED_CROP_ROWS:
        raise SourceAccessError("source-access plan crop-row count mismatch")
    if plan.get("target") != {
        "uid": SOURCE_ACCESS_TARGET_UID,
        "gid": SOURCE_ACCESS_TARGET_GID,
        "mode": format(SOURCE_ACCESS_TARGET_MODE, "04o"),
    }:
        raise SourceAccessError("source-access plan target metadata mismatch")
    files = plan.get("files")
    if (
        not isinstance(files, list)
        or len(files) != SOURCE_ACCESS_EXPECTED_CANDIDATES
    ):
        raise SourceAccessError("source-access plan candidate count mismatch")
    names: list[str] = []
    for item in files:
        if (
            not isinstance(item, dict)
            or set(item) != _IDENTITY_FIELDS | {
                "tile_name", "file_name", "path", "action",
            }
            or item.get("action") not in PLAN_ACTIONS
        ):
            raise SourceAccessError("source-access plan has a malformed file record")
        tile_name = item.get("tile_name")
        file_name = item.get("file_name")
        if not _safe_name(tile_name) or file_name != f"{tile_name}.npz":
            raise SourceAccessError("source-access plan has an unsafe tile identity")
        if item.get("path") != str(expected_data_dir / str(file_name)):
            raise SourceAccessError("source-access plan has a mismatched file path")
        identity_record = {key: item[key] for key in _IDENTITY_FIELDS}
        if not _valid_identity_record(identity_record):
            raise SourceAccessError("source-access plan has malformed file metadata")
        if _classify(item) != item["action"]:
            raise SourceAccessError("source-access plan action contradicts metadata")
        names.append(str(tile_name))
    if names != sorted(set(names)):
        raise SourceAccessError("source-access plan files are not canonical and unique")
    summary = plan.get("summary")
    action_counts = Counter(str(item["action"]) for item in files)
    expected_summary = {
        "candidates": SOURCE_ACCESS_EXPECTED_CANDIDATES,
        "repairs": SOURCE_ACCESS_EXPECTED_REPAIRS,
        "accepted_0644": action_counts[ACTION_ACCEPT_0644],
        "already_correct": action_counts[ACTION_ALREADY_CORRECT],
    }
    if (
        summary != expected_summary
        or expected_summary["accepted_0644"]
        + expected_summary["already_correct"]
        != SOURCE_ACCESS_EXPECTED_NOOPS
    ):
        raise SourceAccessError("source-access plan summary mismatch")
    return plan, payload


def _matches_plan_bytes(current: Mapping[str, object], planned: Mapping[str, object]) -> bool:
    return all(
        current.get(key) == planned.get(key)
        for key in ("dev", "inode", "size", "mtime_ns", "sha256", "nlink")
    )


def _compatible_action(planned_action: str, current_action: str) -> bool:
    return (
        planned_action == current_action
        or (
            planned_action == ACTION_REPAIR
            and current_action in {ACTION_ALREADY_CORRECT, _ACTION_PARTIAL_REPAIR}
        )
    )


def _classify_apply(identity: Mapping[str, object]) -> str:
    if (identity["uid"], identity["gid"], identity["mode"]) == (
        SOURCE_ACCESS_TARGET_UID,
        SOURCE_ACCESS_TARGET_GID,
        "0600",
    ):
        # A retry after fchown succeeded but fchmod failed must converge; this
        # state is accepted only while consuming a PLAN repair record.
        return _ACTION_PARTIAL_REPAIR
    return _classify(identity)


def _preflight_plan_files(
    files: Sequence[Mapping[str, object]], *, data_dir: Path
) -> None:
    """Validate every target before the first metadata mutation."""
    data_fd = _open_directory_tree(data_dir)
    try:
        for planned in files:
            tile_name = str(planned["tile_name"])
            file_name = str(planned["file_name"])
            if planned.get("path") != str(data_dir / file_name):
                raise SourceAccessError(
                    f"source tile path differs from plan root: {tile_name}"
                )
            fd = _open_regular_at(data_fd, file_name)
            try:
                current = _identity(fd)
                _path_matches_fd(data_fd, file_name, os.fstat(fd))
            finally:
                os.close(fd)
            if not _matches_plan_bytes(current, planned):
                raise SourceAccessError(
                    f"source tile identity differs from plan: {tile_name}"
                )
            if not _compatible_action(
                str(planned["action"]), _classify_apply(current)
            ):
                raise SourceAccessError(
                    f"source tile metadata differs from plan: {tile_name}"
                )
    finally:
        os.close(data_fd)


def apply_plan_record(
    plan: Mapping[str, object],
    *,
    data_dir: Path,
    lease_check: Callable[[], object] | None = None,
) -> list[dict[str, object]]:
    """Apply only approved metadata transitions and return before/after proof."""
    files = plan["files"]
    assert isinstance(files, list)
    if lease_check is not None:
        lease_check()
    _preflight_plan_files(files, data_dir=data_dir)
    if lease_check is not None:
        lease_check()
    data_fd = _open_directory_tree(data_dir)
    results: list[dict[str, object]] = []
    try:
        for planned in files:
            assert isinstance(planned, dict)
            if lease_check is not None:
                lease_check()
            tile_name = str(planned["tile_name"])
            file_name = str(planned["file_name"])
            fd = _open_regular_at(data_fd, file_name)
            try:
                before = _identity(fd)
                _path_matches_fd(data_fd, file_name, os.fstat(fd))
                if not _matches_plan_bytes(before, planned):
                    raise SourceAccessError(
                        f"source tile identity differs from plan: {tile_name}"
                    )
                planned_action = str(planned["action"])
                current_action = _classify_apply(before)
                if not _compatible_action(planned_action, current_action):
                    raise SourceAccessError(
                        f"source tile metadata differs from plan: {tile_name}"
                    )
                if planned_action == ACTION_REPAIR and current_action in {
                    ACTION_REPAIR,
                    _ACTION_PARTIAL_REPAIR,
                }:
                    # The approved repair preserves root ownership and changes
                    # only the group plus mode: root:root 0600 -> root:2000
                    # 0640.  Passing uid=-1 prevents an accidental owner
                    # mutation even if protocol constants drift later.
                    os.fchown(fd, -1, SOURCE_ACCESS_TARGET_GID)
                    os.fchmod(fd, SOURCE_ACCESS_TARGET_MODE)
                    # Completion is not publishable until the repaired inode's
                    # metadata has crossed the durability boundary.  The fd is
                    # intentionally O_RDONLY: it authorizes metadata syscalls,
                    # never byte writes.
                    os.fsync(fd)
                    applied_action = "repaired"
                elif planned_action == ACTION_REPAIR and current_action == ACTION_ALREADY_CORRECT:
                    # A retry cannot know whether an earlier attempt crashed
                    # after metadata syscalls but before their fsync. Re-cross
                    # the durability boundary before treating target metadata
                    # as an idempotent completed repair.
                    os.fsync(fd)
                    applied_action = "already-repaired"
                elif planned_action == current_action and current_action in {
                    ACTION_ACCEPT_0644,
                    ACTION_ALREADY_CORRECT,
                }:
                    applied_action = "no-op"
                else:  # guarded by _compatible_action above
                    raise AssertionError("unhandled compatible source metadata state")
                after = _identity(fd)
                _path_matches_fd(data_fd, file_name, os.fstat(fd))
                if not _matches_plan_bytes(after, planned):
                    raise SourceAccessError(
                        f"source tile bytes/inode/mtime changed: {tile_name}"
                    )
                if any(
                    before[key] != after[key]
                    for key in (
                        "dev",
                        "inode",
                        "size",
                        "mtime_ns",
                        "nlink",
                        "sha256",
                    )
                ):
                    raise SourceAccessError(
                        f"source tile metadata-only proof failed: {tile_name}"
                    )
                if planned_action == ACTION_REPAIR and (
                    after["uid"], after["gid"], after["mode"]
                ) != (
                    SOURCE_ACCESS_TARGET_UID,
                    SOURCE_ACCESS_TARGET_GID,
                    format(SOURCE_ACCESS_TARGET_MODE, "04o"),
                ):
                    raise SourceAccessError(
                        f"source tile repair did not settle: {tile_name}"
                    )
                ctime_changed = before["ctime_ns"] != after["ctime_ns"]
                if applied_action == "repaired" and not ctime_changed:
                    raise SourceAccessError(
                        f"source tile repair did not change ctime: {tile_name}"
                    )
                if applied_action in {"no-op", "already-repaired"} and (
                    before != after or _classify(after) != current_action
                ):
                    raise SourceAccessError(
                        f"source tile no-op changed metadata: {tile_name}"
                    )
                results.append(
                    {
                        "tile_name": tile_name,
                        "planned_action": planned_action,
                        "applied_action": applied_action,
                        "before": before,
                        "after": after,
                        "sha256_unchanged": before["sha256"] == after["sha256"],
                        "size_unchanged": before["size"] == after["size"],
                        "mtime_unchanged": before["mtime_ns"] == after["mtime_ns"],
                        "inode_unchanged": (
                            before["dev"], before["inode"]
                        ) == (after["dev"], after["inode"]),
                        "ctime_changed": ctime_changed,
                    }
                )
                if lease_check is not None:
                    lease_check()
            finally:
                os.close(fd)
    finally:
        os.close(data_fd)
    if lease_check is not None:
        lease_check()
    return results


def verify_live_completion_cohort(
    completion: Mapping[str, object], *, data_dir: Path = DATA_DIR
) -> None:
    """Re-hash the complete live cohort against completion ``after`` records.

    This is intentionally a separate full pass, not a side effect of the
    per-file APPLY loop.  It catches an early tile being replaced while later
    tiles are processed.  The split entrypoint calls the same verifier before
    and after generating its staged freeze so completion authority cannot go
    stale between APPLY publication and split publication.
    """
    files = completion.get("files")
    if (
        not isinstance(files, list)
        or len(files) != SOURCE_ACCESS_EXPECTED_CANDIDATES
    ):
        raise SourceAccessError(
            "live source-access cohort has an unexpected candidate count"
        )

    names: list[str] = []
    data_fd = _open_directory_tree(data_dir)
    try:
        for item in files:
            if not isinstance(item, dict):
                raise SourceAccessError(
                    "live source-access cohort has a malformed completion record"
                )
            tile_name = item.get("tile_name")
            after = item.get("after")
            if not _safe_name(tile_name) or not _valid_identity_record(after):
                raise SourceAccessError(
                    "live source-access cohort has malformed expected identity"
                )
            file_name = f"{tile_name}.npz"
            fd = _open_regular_at(data_fd, file_name)
            try:
                current = _identity(fd)
                _path_matches_fd(data_fd, file_name, os.fstat(fd))
            finally:
                os.close(fd)
            if current != after:
                raise SourceAccessError(
                    f"live source tile differs from completion after-state: {tile_name}"
                )
            names.append(str(tile_name))
    finally:
        os.close(data_fd)

    if names != sorted(set(names)):
        raise SourceAccessError(
            "live source-access cohort is not canonical and unique"
        )


def build_completion_record(
    *,
    identity: RuntimeIdentity,
    verified_runtime: Mapping[str, Any],
    plan_sha256: str,
    plan_pod_uid: str,
    source_index_identity: Mapping[str, object],
    results: list[dict[str, object]],
) -> dict[str, object]:
    counts = Counter(str(item["applied_action"]) for item in results)
    return {
        "schema": COMPLETION_SCHEMA,
        "pod_uid": identity.pod_uid,
        "status": "completed",
        "runtime": runtime_binding(identity, verified_runtime),
        "process_identity": {
            "effective_uid": os.geteuid(),
            "effective_gid": os.getegid(),
        },
        "plan": {"pod_uid": plan_pod_uid, "sha256": plan_sha256},
        "source_index": dict(source_index_identity),
        "summary": {
            "files": len(results),
            "repaired": counts["repaired"],
            "already_repaired": counts["already-repaired"],
            "no_op": counts["no-op"],
            "content_unchanged": all(
                bool(item["sha256_unchanged"])
                and bool(item["size_unchanged"])
                and bool(item["mtime_unchanged"])
                and bool(item["inode_unchanged"])
                for item in results
            ),
            "ctime_policy": "changed-on-repair; unchanged permitted on idempotent no-op",
        },
        "files": results,
    }


def publish_completion(
    record: Mapping[str, object],
    path: Path,
    *,
    data_dir: Path = DATA_DIR,
    final_check: Callable[[], object] | None = None,
) -> tuple[str, bytes]:
    """Publish only after a distinct complete live-cohort rescan."""
    verify_live_completion_cohort(record, data_dir=data_dir)
    if final_check is not None:
        final_check()
    payload = canonical_json_bytes(dict(record))
    write_once_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest(), payload


def verify_completion(
    path: Path,
    *,
    expected_sha256: str,
    expected_source_git_sha: str,
    expected_image_ref: str,
    expected_completion_pod_uid: str,
    expected_plan_sha256: str | None = None,
    expected_runtime_binding: Mapping[str, str] | None = None,
) -> dict[str, object]:
    payload, _ = _capture_path(path)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise SourceAccessError("source-access completion SHA256 mismatch")
    try:
        record = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SourceAccessError("source-access completion is invalid JSON") from exc
    if not isinstance(record, dict) or canonical_json_bytes(record) != payload:
        raise SourceAccessError("source-access completion is not canonical JSON")
    if set(record) != {
        "schema", "pod_uid", "status", "runtime", "process_identity",
        "plan", "source_index", "summary", "files",
    }:
        raise SourceAccessError("source-access completion schema is unexpected")
    if record.get("schema") != COMPLETION_SCHEMA or record.get("status") != "completed":
        raise SourceAccessError("source-access completion is not successful")
    if record.get("pod_uid") != expected_completion_pod_uid:
        raise SourceAccessError("source-access completion Pod UID mismatch")
    runtime = record.get("runtime")
    if not isinstance(runtime, dict) or (
        set(runtime) != {
            "source_git_sha", "image_ref", "runtime_manifest_sha256",
            "source_payload_sha256",
        }
        or
        runtime.get("source_git_sha") != expected_source_git_sha
        or runtime.get("image_ref") != expected_image_ref
        or re.fullmatch(
            r"[0-9a-f]{64}", str(runtime.get("runtime_manifest_sha256"))
        ) is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(runtime.get("source_payload_sha256"))
        ) is None
    ):
        raise SourceAccessError("source-access completion runtime mismatch")
    if (
        expected_runtime_binding is not None
        and runtime != dict(expected_runtime_binding)
    ):
        raise SourceAccessError("source-access completion runtime binding mismatch")
    process_identity = record.get("process_identity")
    if process_identity != {"effective_uid": 0, "effective_gid": 2000}:
        raise SourceAccessError("source-access completion process identity mismatch")
    plan = record.get("plan")
    if not isinstance(plan, dict) or set(plan) != {"pod_uid", "sha256"} or (
        expected_plan_sha256 is not None
        and plan.get("sha256") != expected_plan_sha256
    ):
        raise SourceAccessError("source-access completion plan mismatch")
    index = record.get("source_index")
    if not isinstance(index, dict) or (
        set(index) != _IDENTITY_FIELDS | {"path"}
        or
        index.get("path") != str(SOURCE_ACCESS_INDEX_INPUT)
        or index.get("sha256") != SOURCE_ACCESS_INDEX_SHA256
        or index.get("size") != SOURCE_ACCESS_INDEX_SIZE
        or index.get("nlink") != 1
    ):
        raise SourceAccessError("source-access completion index mismatch")
    if not _valid_identity_record({key: index[key] for key in _IDENTITY_FIELDS}):
        raise SourceAccessError("source-access completion index metadata is malformed")
    files = record.get("files")
    if not isinstance(files, list) or len(files) != SOURCE_ACCESS_EXPECTED_CANDIDATES:
        raise SourceAccessError("source-access completion candidate count mismatch")
    names: list[str] = []
    applied_counts: Counter[str] = Counter()
    for item in files:
        if not isinstance(item, dict) or set(item) != {
            "tile_name", "planned_action", "applied_action", "before", "after",
            "sha256_unchanged", "size_unchanged", "mtime_unchanged",
            "inode_unchanged", "ctime_changed",
        }:
            raise SourceAccessError("source-access completion file schema mismatch")
        name = item.get("tile_name")
        planned_action = item.get("planned_action")
        applied_action = item.get("applied_action")
        before = item.get("before")
        after = item.get("after")
        if (
            not _safe_name(name)
            or planned_action not in PLAN_ACTIONS
            or applied_action not in {"repaired", "already-repaired", "no-op"}
            or not _valid_identity_record(before)
            or not _valid_identity_record(after)
        ):
            raise SourceAccessError("source-access completion file is malformed")
        names.append(str(name))
        applied_counts[str(applied_action)] += 1
        unchanged = {
            "sha256_unchanged": before.get("sha256") == after.get("sha256"),
            "size_unchanged": before.get("size") == after.get("size"),
            "mtime_unchanged": before.get("mtime_ns") == after.get("mtime_ns"),
            "inode_unchanged": (
                before.get("dev"), before.get("inode")
            ) == (after.get("dev"), after.get("inode")),
            "ctime_changed": before.get("ctime_ns") != after.get("ctime_ns"),
        }
        if any(item.get(key) is not value for key, value in unchanged.items()):
            raise SourceAccessError("source-access completion proof is inconsistent")
        if not all(bool(unchanged[key]) for key in (
            "sha256_unchanged", "size_unchanged", "mtime_unchanged",
            "inode_unchanged",
        )):
            raise SourceAccessError("source-access completion content changed")
        if planned_action == ACTION_REPAIR:
            if applied_action not in {"repaired", "already-repaired"} or (
                after.get("uid"), after.get("gid"), after.get("mode")
            ) != (SOURCE_ACCESS_TARGET_UID, SOURCE_ACCESS_TARGET_GID, "0640"):
                raise SourceAccessError("source-access completion repair mismatch")
            if applied_action == "repaired" and _classify_apply(before) not in {
                ACTION_REPAIR,
                _ACTION_PARTIAL_REPAIR,
            }:
                raise SourceAccessError("source-access completion repair start mismatch")
            if applied_action == "repaired" and not unchanged["ctime_changed"]:
                raise SourceAccessError("source-access completion repair lacks ctime change")
            if applied_action == "already-repaired" and before != after:
                raise SourceAccessError("source-access idempotent repair changed metadata")
        elif applied_action != "no-op" or before != after or _classify(after) != planned_action:
            raise SourceAccessError("source-access completion no-op changed metadata")
    if names != sorted(set(names)):
        raise SourceAccessError("source-access completion files are not canonical")
    summary = record.get("summary")
    expected_summary = {
        "files": SOURCE_ACCESS_EXPECTED_CANDIDATES,
        "repaired": applied_counts["repaired"],
        "already_repaired": applied_counts["already-repaired"],
        "no_op": applied_counts["no-op"],
        "content_unchanged": True,
        "ctime_policy": "changed-on-repair; unchanged permitted on idempotent no-op",
    }
    if summary != expected_summary:
        raise SourceAccessError("source-access completion summary mismatch")
    if (
        expected_summary["repaired"] + expected_summary["already_repaired"]
        != SOURCE_ACCESS_EXPECTED_REPAIRS
        or expected_summary["no_op"] != SOURCE_ACCESS_EXPECTED_NOOPS
    ):
        raise SourceAccessError("source-access completion action counts mismatch")
    return record


def _expected_index_authority(environ: Mapping[str, str]) -> str:
    claimed = require_source_access_sha256(
        environ.get("CROP_SOURCE_ACCESS_INDEX_SHA256", ""),
        "CROP_SOURCE_ACCESS_INDEX_SHA256",
    )
    if claimed != SOURCE_ACCESS_INDEX_SHA256:
        raise SourceAccessError("manifest source-index SHA256 differs from runtime")
    return claimed


def _freeze_lease_path(environ: Mapping[str, str]) -> Path:
    return Path(environ.get("CROP_SOURCE_FREEZE_LEASE_PATH", str(FREEZE_LEASE_PATH)))


def run_plan(environ: Mapping[str, str]) -> dict[str, object]:
    identity = runtime_identity(environ)
    require_process_identity(0, role="crop source-access plan")
    expected_index_sha = _expected_index_authority(environ)
    verified = verify_runtime(
        RUNTIME_MANIFEST,
        source_git_sha=identity.source_git_sha,
        image_ref=identity.image_ref,
    )
    lease_path = _freeze_lease_path(environ)
    require_fresh_freeze_lease(lease_path, expected_phase="plan")
    with exclusive_dataset_lock(
        SOURCE_ACCESS_LOCK_FILE,
        create=False,
        expected_uid=0,
        expected_gid=SOURCE_ACCESS_TARGET_GID,
        expected_mode=SOURCE_ACCESS_LOCK_MODE,
    ):
        require_fresh_freeze_lease(lease_path, expected_phase="plan")
        record = build_plan_record(
            identity=identity,
            verified_runtime=verified,
            source_index=SOURCE_ACCESS_INDEX_INPUT,
            data_dir=DATA_DIR,
            expected_index_sha256=expected_index_sha,
            expected_index_size=SOURCE_ACCESS_INDEX_SIZE,
            enforce_production_counts=True,
        )
        require_fresh_freeze_lease(lease_path, expected_phase="plan")
        output = SOURCE_ACCESS_PLAN_RECORD_DIR / identity.pod_uid / "plan.json"
        digest, payload = publish_plan(record, output)
    print(_marker(PLAN_MARKER, payload), flush=True)
    return {"plan": str(output), "plan_sha256": digest, **record}


def run_apply(environ: Mapping[str, str]) -> dict[str, object]:
    identity = runtime_identity(environ)
    require_process_identity(0, role="crop source-access apply")
    expected_index_sha = _expected_index_authority(environ)
    plan_sha = require_source_access_sha256(
        environ.get("CROP_SOURCE_ACCESS_PLAN_SHA256", ""),
        "CROP_SOURCE_ACCESS_PLAN_SHA256",
    )
    plan_pod_uid = require_source_access_run_id(
        environ.get("CROP_SOURCE_ACCESS_PLAN_POD_UID", ""),
        "CROP_SOURCE_ACCESS_PLAN_POD_UID",
    )
    verified = verify_runtime(
        RUNTIME_MANIFEST,
        source_git_sha=identity.source_git_sha,
        image_ref=identity.image_ref,
    )
    lease_path = _freeze_lease_path(environ)

    def check_lease() -> dict[str, object]:
        return require_fresh_freeze_lease(lease_path, expected_phase="apply")

    check_lease()
    with exclusive_dataset_lock(
        SOURCE_ACCESS_LOCK_FILE,
        create=False,
        expected_uid=0,
        expected_gid=SOURCE_ACCESS_TARGET_GID,
        expected_mode=SOURCE_ACCESS_LOCK_MODE,
    ):
        check_lease()
        plan, index_plan_payload = load_plan(
            SOURCE_ACCESS_PLAN_INPUT,
            expected_sha256=plan_sha,
            identity=identity,
            expected_plan_pod_uid=plan_pod_uid,
            expected_index_sha256=expected_index_sha,
            expected_index_size=SOURCE_ACCESS_INDEX_SIZE,
            expected_index_path=SOURCE_ACCESS_INDEX_INPUT,
            expected_data_dir=DATA_DIR,
            expected_runtime_binding=runtime_binding(identity, verified),
        )
        index_payload, index_identity = _capture_path(SOURCE_ACCESS_INDEX_INPUT)
        if (
            index_identity["sha256"] != expected_index_sha
            or index_identity["size"] != SOURCE_ACCESS_INDEX_SIZE
        ):
            raise SourceAccessError("source index changed after plan")
        planned_index = plan["source_index"]
        assert isinstance(planned_index, dict)
        if any(
            index_identity.get(key) != planned_index.get(key)
            for key in ("dev", "inode", "size", "mtime_ns", "sha256", "nlink")
        ):
            raise SourceAccessError("source index identity differs from plan")
        names, crop_rows, crop_window = derive_crop_window_candidates(index_payload)
        plan_files = plan["files"]
        assert isinstance(plan_files, list)
        if names != [str(item["tile_name"]) for item in plan_files]:
            raise SourceAccessError("source index candidates changed after plan")
        if crop_rows != plan["crop_rows"] or crop_window != plan["crop_window"]:
            raise SourceAccessError("source index crop window changed after plan")
        if hashlib.sha256(index_plan_payload).hexdigest() != plan_sha:
            raise SourceAccessError("plan bytes changed during apply preflight")
        results = apply_plan_record(
            plan,
            data_dir=DATA_DIR,
            lease_check=check_lease,
        )
        completion = build_completion_record(
            identity=identity,
            verified_runtime=verified,
            plan_sha256=plan_sha,
            plan_pod_uid=plan_pod_uid,
            source_index_identity={
                "path": str(SOURCE_ACCESS_INDEX_INPUT), **index_identity,
            },
            results=results,
        )
        # A distinct full 2,074-file pass is the final publication gate.  Do
        # not rely on each file having matched when its turn in APPLY ended:
        # an early path can otherwise be replaced while later work proceeds.
        output = SOURCE_ACCESS_APPLY_RECORD_DIR / identity.pod_uid / "completion.json"
        digest, payload = publish_completion(
            completion,
            output,
            data_dir=DATA_DIR,
            final_check=check_lease,
        )
    print(_marker(COMPLETION_MARKER, payload), flush=True)
    return {"completion": str(output), "completion_sha256": digest, **completion}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "apply"))
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    environment = os.environ if environ is None else environ
    try:
        if args.phase == "plan":
            run_plan(environment)
        else:
            run_apply(environment)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
        print(f"crop source-access {args.phase} refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    # ``run_plan`` / ``run_apply`` already emitted the one canonical base64
    # marker used for restricted capture.  Never duplicate the full per-file
    # inventory as pretty JSON on stdout.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
