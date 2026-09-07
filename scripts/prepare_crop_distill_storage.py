#!/usr/bin/env python3
"""Prepare only the isolated backing directories used by crop-distill Jobs.

The existing CephFS ``distill`` and ``ops`` roots are 0755 root:root.  This
one-shot operator entrypoint prepares only the protocol-owned output roots; it
is separate from the later UID-0 source-access PLAN/APPLY jobs. It accepts no
paths or ownership arguments and changes only the baked targets in
``crop_distill_protocol.py``. Each model receives pre-owned head/evidence
directories before any unprivileged workload starts.
"""

from __future__ import annotations

import argparse
import base64
import errno
import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__:
    from .crop_distill_protocol import (
        FROZEN_SPLIT_MODE,
        RUNTIME_MANIFEST,
        STORAGE_GID,
        STORAGE_TARGETS,
        SOURCE_ACCESS_LOCK_BACKING_FILE,
        SOURCE_ACCESS_LOCK_MODE,
        StorageTarget,
        runtime_identity,
    )
    from .crop_distill_provenance import (
        canonical_json_bytes,
        verify_runtime,
        write_once_bytes,
    )
else:
    from crop_distill_protocol import (
        FROZEN_SPLIT_MODE,
        RUNTIME_MANIFEST,
        STORAGE_GID,
        STORAGE_TARGETS,
        SOURCE_ACCESS_LOCK_BACKING_FILE,
        SOURCE_ACCESS_LOCK_MODE,
        StorageTarget,
        runtime_identity,
    )
    from crop_distill_provenance import (
        canonical_json_bytes,
        verify_runtime,
        write_once_bytes,
    )

STORAGE_PREP_COMPLETION_SCHEMA = "imint-crop-distill-storage-prep-completion-v1"
STORAGE_PREP_COMPLETION_MARKER = "CROP_DISTILL_STORAGE_PREP_COMPLETION_V1"
STORAGE_PREP_RECORD_DIR = Path("/cephfs/ops/crop-distill/storage-prep")
DATASET_LOCK_UID = 0

class StoragePrepError(RuntimeError):
    """Raised when a writable root cannot be prepared without aliasing."""


def _open_real_directory(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise StoragePrepError(
            f"required parent is missing, aliased, or not a directory: {path}: {exc}"
        ) from exc


def _prepare_one(target: StorageTarget) -> dict[str, int | str]:
    path = target.path
    parent = path.parent
    parent_fd = _open_real_directory(parent)
    try:
        try:
            existing = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existed = False
            try:
                os.mkdir(path.name, mode=0o700, dir_fd=parent_fd)
            except OSError as exc:
                raise StoragePrepError(
                    f"cannot create storage root {path}: {exc}"
                ) from exc
        else:
            existed = True
            if not stat.S_ISDIR(existing.st_mode):
                raise StoragePrepError(f"storage root is not a real directory: {path}")

        flags = os.O_RDONLY | os.O_DIRECTORY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            child_fd = os.open(path.name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise StoragePrepError(f"cannot open storage root {path}: {exc}") from exc
        try:
            before = os.fstat(child_fd)
            if not stat.S_ISDIR(before.st_mode):
                raise StoragePrepError(
                    f"storage root changed type while opening: {path}"
                )
            if before.st_uid not in {0, target.uid}:
                raise StoragePrepError(
                    f"refusing unexpected owner {before.st_uid} on {path}"
                )
            before_mode = stat.S_IMODE(before.st_mode)
            preserve_frozen = (
                existed
                and target.preserve_mode is not None
                and before.st_uid == target.uid
                and before.st_gid == target.gid
                and before_mode == target.preserve_mode
            )
            if preserve_frozen:
                return {
                    "path": str(path),
                    "uid": before.st_uid,
                    "gid": before.st_gid,
                    "mode": format(before_mode, "04o"),
                    "device": before.st_dev,
                    "inode": before.st_ino,
                    "state": "preserved-frozen",
                }
            os.fchown(child_fd, target.uid, target.gid)
            os.fchmod(child_fd, target.mode)
            after = os.fstat(child_fd)
            if (
                after.st_dev != before.st_dev
                or after.st_ino != before.st_ino
                or after.st_uid != target.uid
                or after.st_gid != target.gid
                or stat.S_IMODE(after.st_mode) != target.mode
            ):
                raise StoragePrepError(
                    f"storage root identity or permissions did not settle: {path}"
                )
            return {
                "path": str(path),
                "uid": after.st_uid,
                "gid": after.st_gid,
                "mode": format(stat.S_IMODE(after.st_mode), "04o"),
                "device": after.st_dev,
                "inode": after.st_ino,
                "state": "writable",
            }
        finally:
            os.close(child_fd)
    finally:
        os.close(parent_fd)


def _prepare_dataset_lock() -> dict[str, int | str]:
    """Create the one shared lock inode without broadening its directory."""
    parent_fd = _open_real_directory(SOURCE_ACCESS_LOCK_BACKING_FILE.parent)
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    lock_fd: int | None = None
    try:
        try:
            lock_fd = os.open(
                SOURCE_ACCESS_LOCK_BACKING_FILE.name,
                flags,
                0o600,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise StoragePrepError(
                "cannot securely create/open dataset lock "
                f"{SOURCE_ACCESS_LOCK_BACKING_FILE}: "
                f"{exc}"
            ) from exc
        before = os.fstat(lock_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != 0
        ):
            raise StoragePrepError(
                "dataset lock must be one empty, unaliased regular file"
            )
        if before.st_uid != DATASET_LOCK_UID:
            raise StoragePrepError(
                f"dataset lock has unexpected owner {before.st_uid}"
            )
        os.fchown(lock_fd, DATASET_LOCK_UID, STORAGE_GID)
        os.fchmod(lock_fd, SOURCE_ACCESS_LOCK_MODE)
        os.fsync(lock_fd)
        after = os.fstat(lock_fd)
        current = os.stat(
            SOURCE_ACCESS_LOCK_BACKING_FILE.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or current.st_dev != after.st_dev
            or current.st_ino != after.st_ino
            or after.st_nlink != 1
            or current.st_nlink != 1
            or after.st_size != 0
            or after.st_uid != DATASET_LOCK_UID
            or after.st_gid != STORAGE_GID
            or stat.S_IMODE(after.st_mode) != SOURCE_ACCESS_LOCK_MODE
        ):
            raise StoragePrepError("dataset lock identity did not settle")
        try:
            os.fsync(parent_fd)
        except OSError as exc:
            # Some filesystems do not support syncing directories. The lock
            # inode itself has crossed its fsync boundary above.
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise StoragePrepError(
                    "cannot sync dataset lock directory"
                ) from exc
        return {
            "path": str(SOURCE_ACCESS_LOCK_BACKING_FILE),
            "uid": after.st_uid,
            "gid": after.st_gid,
            "mode": format(stat.S_IMODE(after.st_mode), "04o"),
            "device": after.st_dev,
            "inode": after.st_ino,
            "size_bytes": after.st_size,
            "nlink": after.st_nlink,
            "state": "ready",
        }
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(parent_fd)


def prepare_storage(
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Prepare the baked roots and return stdout-suitable audit evidence."""
    environment = os.environ if environ is None else environ
    identity = runtime_identity(environment)
    verified_runtime = verify_runtime(
        RUNTIME_MANIFEST,
        source_git_sha=identity.source_git_sha,
        image_ref=identity.image_ref,
    )
    if os.geteuid() != 0:
        raise StoragePrepError("storage preparation requires effective UID 0")
    if os.getegid() != STORAGE_GID:
        raise StoragePrepError(
            f"storage preparation requires effective GID {STORAGE_GID}"
        )
    targets = [_prepare_one(target) for target in STORAGE_TARGETS]
    dataset_lock = _prepare_dataset_lock()
    return {
        "schema": STORAGE_PREP_COMPLETION_SCHEMA,
        "pod_uid": identity.pod_uid,
        "status": "completed",
        "process_identity": {
            "effective_uid": 0,
            "effective_gid": STORAGE_GID,
        },
        "preserved_frozen_mode": format(FROZEN_SPLIT_MODE, "04o"),
        "runtime": {"verification": "verified", **verified_runtime},
        "targets": targets,
        "dataset_lock": dataset_lock,
    }


def publish_completion(record: Mapping[str, object]) -> tuple[Path, str, bytes]:
    """Publish the exact canonical storage-prep completion once per Pod."""
    pod_uid = record.get("pod_uid")
    if not isinstance(pod_uid, str) or not pod_uid:
        raise StoragePrepError("storage preparation lacks a Pod UID")
    payload = canonical_json_bytes(dict(record))
    digest = hashlib.sha256(payload).hexdigest()
    target = STORAGE_PREP_RECORD_DIR / pod_uid / "completion.json"
    write_once_bytes(target, payload)
    return target, digest, payload


def completion_marker(payload: bytes) -> str:
    """Encode exactly the bytes written to the immutable PVC record."""
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"{STORAGE_PREP_COMPLETION_MARKER} {digest} {encoded}"


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    build_parser().parse_args(argv)
    result = prepare_storage(environ)
    target, digest, payload = publish_completion(result)
    print(completion_marker(payload), flush=True)
    print(
        json.dumps(
            {"record": str(target), "record_sha256": digest, **result},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
