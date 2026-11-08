"""Sealed-file I/O shared by inference, prediction caches and LCLU serving.

Regular files are opened without following links and checked again after
reading, so a file replaced or changed during the read is refused rather than
hashed or loaded. Files are published with one atomic replace.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re as _re
import stat
import tempfile
from pathlib import Path
from typing import BinaryIO

import numpy as np


class TileIdentityError(RuntimeError):
    """A frozen tile's path, size, or bytes differ from its inventory."""


class CheckpointIdentityError(RuntimeError):
    """A checkpoint path or its bytes violate the sealed-file contract."""


_READ_BLOCK_SIZE = 1024 * 1024


_SHA256_PATTERN = _re.compile(r"[0-9a-f]{64}")


def _file_snapshot(identity: os.stat_result) -> tuple[int, ...]:
    """Return metadata that changes when an opened file is replaced/mutated."""
    return (
        identity.st_dev,
        identity.st_ino,
        stat.S_IFMT(identity.st_mode),
        identity.st_nlink,
        identity.st_size,
        identity.st_mtime_ns,
        identity.st_ctime_ns,
    )


def _assert_path_names_open_file(
    path: Path,
    opened: os.stat_result,
    *,
    error_type: type[RuntimeError],
    label: str,
) -> None:
    """Reject a path swapped away from the descriptor after ``open(2)``."""
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise error_type(f"{label} path changed while open: {path}: {exc}") from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise error_type(f"{label} path changed while open: {path}")


def _open_regular_nofollow(
    path: Path,
    *,
    error_type: type[RuntimeError],
    label: str,
) -> tuple[int, os.stat_result]:
    """Open ``path`` descriptor-first and reject links/non-regular files."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise error_type("this platform cannot enforce O_NOFOLLOW")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise error_type(f"cannot open {label} without following links: {path}: {exc}") from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise error_type(f"{label} is not a regular file: {path}")
        if opened.st_nlink != 1:
            raise error_type(
                f"{label} must have exactly one hard link: {path} "
                f"(found {opened.st_nlink})"
            )
        _assert_path_names_open_file(
            path, opened, error_type=error_type, label=label
        )
    except BaseException:
        os.close(fd)
        raise
    return fd, opened


def _assert_open_file_unchanged(
    fd: int,
    path: Path,
    opened: os.stat_result,
    *,
    error_type: type[RuntimeError],
    label: str,
) -> None:
    """Detect in-place writes, link changes, and path swaps after opening."""
    try:
        current = os.fstat(fd)
    except OSError as exc:
        raise error_type(f"cannot re-inspect open {label} {path}: {exc}") from exc
    if _file_snapshot(current) != _file_snapshot(opened):
        raise error_type(f"{label} changed while being read: {path}")
    _assert_path_names_open_file(
        path, current, error_type=error_type, label=label
    )


def _validate_expected_identity(
    *,
    expected_sha256: str | None,
    expected_size: int | None,
    error_type: type[RuntimeError],
    label: str,
) -> bool:
    """Validate an optional exact size/SHA pair and return whether it exists."""
    if expected_sha256 is None and expected_size is None:
        return False
    if expected_sha256 is None or expected_size is None:
        raise error_type(f"expected {label} size and sha256 must be supplied together")
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        raise error_type(f"expected {label} sha256 must be 64 lowercase hex")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
    ):
        raise error_type(f"expected {label} size must be a positive integer")
    return True


def _copy_and_hash_open_file(source: BinaryIO, destination: BinaryIO) -> tuple[int, str]:
    """Copy one open file into private storage while hashing identical bytes."""
    digest = hashlib.sha256()
    total = 0
    for block in iter(lambda: source.read(_READ_BLOCK_SIZE), b""):
        digest.update(block)
        destination.write(block)
        total += len(block)
    return total, digest.hexdigest()


def _load_npz_for_inference(
    tile_path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
):
    """Open one tile, optionally authenticating the exact bytes in one read.

    Crop-distill columns run weeks apart, so their frozen tile inventory must
    bind the bytes consumed by every forward.  Reading the file into a
    ``BytesIO`` while hashing lets ``numpy.load`` consume that same buffer:
    the PVC is not swept once for verification and then read a second time
    for inference.  The returned buffer must remain alive until the NpzFile
    is closed.
    """
    authenticated = _validate_expected_identity(
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        error_type=TileIdentityError,
        label="tile",
    )
    if not authenticated:
        return np.load(tile_path, allow_pickle=False), None

    path = Path(tile_path)
    buffer = io.BytesIO()
    try:
        fd, opened = _open_regular_nofollow(
            path, error_type=TileIdentityError, label="frozen tile"
        )
        digest = hashlib.sha256()
        try:
            source = os.fdopen(fd, "rb", closefd=True)
        except BaseException:
            os.close(fd)
            raise
        with source:
            if opened.st_size != expected_size:
                raise TileIdentityError(
                    f"frozen tile size mismatch for {path}: expected "
                    f"{expected_size}, got {opened.st_size}"
                )
            for block in iter(lambda: source.read(_READ_BLOCK_SIZE), b""):
                digest.update(block)
                buffer.write(block)
            _assert_open_file_unchanged(
                source.fileno(),
                path,
                opened,
                error_type=TileIdentityError,
                label="frozen tile",
            )
    except TileIdentityError:
        buffer.close()
        raise
    except OSError as exc:
        buffer.close()
        raise TileIdentityError(
            f"cannot read frozen tile {path}: {exc}"
        ) from exc
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        buffer.close()
        raise TileIdentityError(
            f"frozen tile sha256 mismatch for {path}: expected "
            f"{expected_sha256}, got {actual_sha256}"
        )
    buffer.seek(0)
    try:
        return np.load(buffer, allow_pickle=False), buffer
    except Exception:
        buffer.close()
        raise


class CacheValidationError(ValueError):
    """The cache is stale, incomplete, or lacks authenticated provenance."""


def file_identity(path):
    """Hash a stable regular file using the inference loader's identity guards."""
    path = Path(path).absolute()
    fd, opened = _open_regular_nofollow(
        path, error_type=CacheValidationError, label="cache input")
    with os.fdopen(fd, "rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
        _assert_open_file_unchanged(
            stream.fileno(), path, opened,
            error_type=CacheValidationError, label="cache input")
    return {"path": str(path), "size": opened.st_size,
            "sha256": digest.hexdigest()}


def atomic_write(path, write):
    """Publish a complete file with one replace; clean up failed writes."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def contract_pin(inference):
    return hashlib.sha256(json.dumps(inference, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
