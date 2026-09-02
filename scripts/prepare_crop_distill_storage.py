#!/usr/bin/env python3
"""Prepare only the isolated backing directories used by crop-distill Jobs.

The existing CephFS ``distill`` and ``ops`` roots are 0755 root:root.  This
one-shot operator entrypoint is the sole UID-0 component in the protocol.  It
accepts no paths or ownership arguments and changes only the baked targets in
``crop_distill_protocol.py``. Each model receives pre-owned head/evidence
directories before any unprivileged workload starts.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path

from crop_distill_protocol import (
    FROZEN_SPLIT_MODE,
    RUNTIME_MANIFEST,
    STORAGE_GID,
    STORAGE_TARGETS,
    StorageTarget,
    runtime_identity,
)
from crop_distill_provenance import verify_runtime

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
    return {
        "schema": "imint-crop-distill-storage-prep-v2",
        "operator_uid": 0,
        "operator_gid": STORAGE_GID,
        "preserved_frozen_mode": format(FROZEN_SPLIT_MODE, "04o"),
        "runtime": verified_runtime,
        "targets": targets,
    }


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    build_parser().parse_args(argv)
    print(json.dumps(prepare_storage(environ), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
