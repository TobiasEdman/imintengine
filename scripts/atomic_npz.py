#!/usr/bin/env python3
"""Durable, drift-detecting NPZ replacement for dataset producers.

The helpers in this module deliberately separate the initial descriptor read
from publication.  A producer captures the source bytes and their complete
filesystem identity before doing expensive work, then supplies that identity
to :func:`durable_atomic_savez`. Publication re-opens and re-hashes the live
path immediately before rename and refuses observed drift. POSIX replacement
has no portable inode-conditional rename; the shared lock closes the remaining
check-to-rename interval for every cooperating producer. The external freeze
is still required for arbitrary Kubernetes writers.

The advisory dataset lock serializes cooperating producers with the LUCAS
source-access PLAN/APPLY jobs.  It does not replace the external Kubernetes
controller/mount freeze used by that campaign.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import io
import os
import secrets
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_READ_SIZE = 1 << 20
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
DATASET_LOCK_MODE = 0o660


class AtomicReplaceError(RuntimeError):
    """A path cannot safely be captured or atomically replaced."""


@dataclass(frozen=True)
class FileIdentity:
    """Identity and content authority captured from one open file descriptor."""

    dev: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    uid: int
    gid: int
    mode: int
    nlink: int
    sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FileIdentity":
        """Build an identity from a record using octal-string or integer mode."""
        mode_value = value["mode"]
        mode = int(mode_value, 8) if isinstance(mode_value, str) else int(mode_value)
        return cls(
            dev=int(value["dev"]),
            inode=int(value["inode"]),
            size=int(value["size"]),
            mtime_ns=int(value["mtime_ns"]),
            ctime_ns=int(value["ctime_ns"]),
            uid=int(value["uid"]),
            gid=int(value["gid"]),
            mode=mode,
            nlink=int(value["nlink"]),
            sha256=str(value["sha256"]),
        )

    def as_record(self) -> dict[str, object]:
        return {
            "dev": self.dev,
            "inode": self.inode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "uid": self.uid,
            "gid": self.gid,
            "mode": format(self.mode, "04o"),
            "nlink": self.nlink,
            "sha256": self.sha256,
        }


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def open_directory_tree(path: Path, *, create: bool = False) -> int:
    """Open an absolute directory without following any path component."""
    absolute = _absolute(path)
    flags = os.O_RDONLY | _O_CLOEXEC | _O_DIRECTORY | _O_NOFOLLOW
    current_fd = os.open(os.sep, flags)
    try:
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise AtomicReplaceError(
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
                raise AtomicReplaceError(f"not a directory: {absolute}")
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _safe_name(name: str) -> bool:
    return (
        bool(name)
        and not name.isspace()
        and "\x00" not in name
        and name not in {".", ".."}
        and Path(name).name == name
    )


def _open_regular_at(directory_fd: int, name: str) -> int:
    if not _safe_name(name):
        raise AtomicReplaceError(f"unsafe destination name: {name!r}")
    try:
        fd = os.open(
            name,
            os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise AtomicReplaceError(f"refusing symlink destination: {name}") from exc
        raise
    identity = os.fstat(fd)
    if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
        os.close(fd)
        raise AtomicReplaceError(
            f"refusing aliased/non-regular destination: {name}"
        )
    if identity.st_size <= 0:
        os.close(fd)
        raise AtomicReplaceError(f"refusing empty source file: {name}")
    return fd


def _metadata_tuple(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_uid),
        int(value.st_gid),
        stat.S_IMODE(value.st_mode),
        int(value.st_nlink),
    )


def _read_identity(fd: int) -> tuple[bytes, FileIdentity]:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise AtomicReplaceError("source descriptor is not an unaliased regular file")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    while True:
        chunk = os.read(fd, _READ_SIZE)
        if not chunk:
            break
        digest.update(chunk)
        chunks.append(chunk)
    after = os.fstat(fd)
    if _metadata_tuple(before) != _metadata_tuple(after):
        raise AtomicReplaceError("file changed while its initial bytes were captured")
    payload = b"".join(chunks)
    if len(payload) != after.st_size:
        raise AtomicReplaceError("file size changed while its bytes were captured")
    return payload, FileIdentity(
        dev=int(after.st_dev),
        inode=int(after.st_ino),
        size=int(after.st_size),
        mtime_ns=int(after.st_mtime_ns),
        ctime_ns=int(after.st_ctime_ns),
        uid=int(after.st_uid),
        gid=int(after.st_gid),
        mode=stat.S_IMODE(after.st_mode),
        nlink=int(after.st_nlink),
        sha256=digest.hexdigest(),
    )


def _path_matches_identity(
    directory_fd: int, name: str, identity: FileIdentity
) -> None:
    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if _metadata_tuple(current) != (
        identity.dev,
        identity.inode,
        identity.size,
        identity.mtime_ns,
        identity.ctime_ns,
        identity.uid,
        identity.gid,
        identity.mode,
        identity.nlink,
    ):
        raise AtomicReplaceError(f"destination path identity changed: {name}")


def capture_npz(path: str | Path) -> tuple[dict[str, object], FileIdentity]:
    """Read an NPZ once through a no-follow fd and return data plus CAS identity."""
    source = _absolute(Path(path))
    directory_fd = open_directory_tree(source.parent)
    fd: int | None = None
    try:
        fd = _open_regular_at(directory_fd, source.name)
        payload, identity = _read_identity(fd)
        _path_matches_identity(directory_fd, source.name, identity)
        try:
            with np.load(io.BytesIO(payload), allow_pickle=True) as archive:
                data = {key: archive[key] for key in archive.files}
        except Exception as exc:
            raise AtomicReplaceError(
                f"cannot load initial NPZ snapshot {source}: {type(exc).__name__}"
            ) from exc
        _path_matches_identity(directory_fd, source.name, identity)
        return data, identity
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory_fd)


def capture_identity(path: str | Path) -> FileIdentity:
    """Capture a stable descriptor/path identity and SHA-256 for any regular file."""
    source = _absolute(Path(path))
    directory_fd = open_directory_tree(source.parent)
    fd: int | None = None
    try:
        fd = _open_regular_at(directory_fd, source.name)
        _payload, identity = _read_identity(fd)
        _path_matches_identity(directory_fd, source.name, identity)
        return identity
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory_fd)


def _create_temp_at(directory_fd: int, destination_name: str) -> tuple[int, str]:
    for _attempt in range(128):
        name = f".{destination_name}.{secrets.token_hex(12)}.tmp"
        try:
            fd = os.open(
                name,
                os.O_CREAT | os.O_EXCL | os.O_RDWR | _O_CLOEXEC | _O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            continue
        return fd, name
    raise AtomicReplaceError("could not allocate a unique atomic temporary file")


def durable_atomic_savez(
    destination: str | Path,
    data: Mapping[str, object],
    *,
    expected: FileIdentity | None = None,
) -> None:
    """Durably replace an NPZ, rejecting observed drift from ``expected``.

    When ``expected`` is omitted the helper captures the current destination at
    entry for backwards-compatible direct use.  Production in-place callers
    pass the identity returned by :func:`capture_npz`, closing the lost-update
    gap between their initial read/fetch and the final locked publication
    check. If the destination does not yet exist, hard-link no-replace
    publication atomically refuses a destination that appears concurrently.
    """
    destination_path = _absolute(Path(destination))
    directory_fd = open_directory_tree(destination_path.parent, create=True)
    live_fd: int | None = None
    temp_fd: int | None = None
    temp_name: str | None = None
    try:
        if expected is None:
            try:
                live_fd = _open_regular_at(directory_fd, destination_path.name)
            except FileNotFoundError:
                pass
            else:
                _payload, expected = _read_identity(live_fd)
                _path_matches_identity(directory_fd, destination_path.name, expected)
        else:
            try:
                live_fd = _open_regular_at(directory_fd, destination_path.name)
            except FileNotFoundError as exc:
                raise AtomicReplaceError(
                    f"destination disappeared before replacement: {destination_path}"
                ) from exc
            _payload, live_identity = _read_identity(live_fd)
            _path_matches_identity(directory_fd, destination_path.name, live_identity)
            if live_identity != expected:
                raise AtomicReplaceError(
                    f"destination changed since initial read: {destination_path}"
                )

        temp_fd, temp_name = _create_temp_at(
            directory_fd, destination_path.name
        )
        with os.fdopen(temp_fd, "wb", closefd=False) as stream:
            np.savez_compressed(stream, **dict(data))
            stream.flush()
            os.fsync(temp_fd)
            if expected is not None:
                # A rename publishes the temporary inode, so retain both
                # ownership fields from the captured destination.  If the
                # caller lacks authority to preserve either field, fail
                # before publication and leave the original untouched.
                os.fchown(temp_fd, expected.uid, expected.gid)
                os.fchmod(temp_fd, expected.mode)
                os.fsync(temp_fd)

        if expected is None:
            try:
                os.stat(
                    destination_path.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise AtomicReplaceError(
                    f"destination appeared during atomic replacement: {destination_path}"
                )
        else:
            assert live_fd is not None
            _payload, current = _read_identity(live_fd)
            _path_matches_identity(directory_fd, destination_path.name, current)
            if current != expected:
                raise AtomicReplaceError(
                    f"destination changed during atomic replacement: {destination_path}"
                )

        if expected is None:
            # A hard link is the portable create-if-absent primitive. Unlike
            # check-then-replace, it cannot overwrite a destination created in
            # the final publication interval.
            os.link(
                temp_name,
                destination_path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            os.unlink(temp_name, dir_fd=directory_fd)
        else:
            # Existing-path publication is safe for cooperating writers only
            # while the common dataset lock is held. POSIX has no portable
            # inode-conditional replacement primitive.
            os.replace(
                temp_name,
                destination_path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        temp_name = None
        os.fsync(directory_fd)
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        if live_fd is not None:
            os.close(live_fd)
        if temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def default_dataset_lock(data_dir: str | Path) -> Path:
    """Return the shared source-access lock next to a dataset directory."""
    data_path = _absolute(Path(data_dir))
    return (
        data_path.parent
        / "ops/crop-distill/source-access/locks/dataset.lock"
    )


def _lock_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_uid),
        int(value.st_gid),
        stat.S_IMODE(value.st_mode),
        int(value.st_nlink),
    )


def _validate_lock(
    directory_fd: int,
    name: str,
    fd: int,
    *,
    expected_uid: int | None,
    expected_gid: int | None,
    expected_mode: int | None,
) -> os.stat_result:
    identity = os.fstat(fd)
    if (
        not stat.S_ISREG(identity.st_mode)
        or identity.st_nlink != 1
        or identity.st_size != 0
    ):
        raise AtomicReplaceError(
            "dataset lock must be an empty, unaliased regular file"
        )
    if expected_uid is not None and identity.st_uid != expected_uid:
        raise AtomicReplaceError(
            f"dataset lock UID {identity.st_uid} != {expected_uid}"
        )
    if expected_gid is not None and identity.st_gid != expected_gid:
        raise AtomicReplaceError(
            f"dataset lock GID {identity.st_gid} != {expected_gid}"
        )
    mode = stat.S_IMODE(identity.st_mode)
    if expected_mode is not None and mode != expected_mode:
        raise AtomicReplaceError(
            f"dataset lock mode {mode:04o} != {expected_mode:04o}"
        )
    path_identity = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if _lock_identity(path_identity) != _lock_identity(identity):
        raise AtomicReplaceError("dataset lock path/fd identity changed")
    return identity


@contextmanager
def exclusive_dataset_lock(
    path: str | Path,
    *,
    create: bool = True,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    expected_mode: int | None = None,
) -> Iterator[None]:
    """Serialize cooperating producer writes with source PLAN/APPLY/split.

    Rollout workloads pass ``create=False`` and the exact storage-prep
    ownership/mode contract. General producer CLIs retain safe first-use
    creation for backwards compatibility; a newly created lock is durably
    settled to mode 0660 before use.
    """
    lock_path = _absolute(Path(path))
    parent_fd = open_directory_tree(lock_path.parent, create=create)
    fd: int | None = None
    created = False
    try:
        flags = os.O_RDWR | _O_CLOEXEC | _O_NOFOLLOW
        if create:
            try:
                fd = os.open(
                    lock_path.name,
                    flags | os.O_CREAT | os.O_EXCL,
                    DATASET_LOCK_MODE,
                    dir_fd=parent_fd,
                )
                created = True
            except FileExistsError:
                fd = os.open(lock_path.name, flags, dir_fd=parent_fd)
        else:
            fd = os.open(lock_path.name, flags, dir_fd=parent_fd)
        if created:
            # os.open's mode is filtered by umask. Set the exact cooperative
            # mode only on the inode this call created, then persist both the
            # inode metadata and its directory entry.
            os.fchmod(fd, DATASET_LOCK_MODE)
            os.fsync(fd)
            os.fsync(parent_fd)
        before = _validate_lock(
            parent_fd,
            lock_path.name,
            fd,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            expected_mode=expected_mode,
        )
        fcntl.flock(fd, fcntl.LOCK_EX)
        after_lock = _validate_lock(
            parent_fd,
            lock_path.name,
            fd,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            expected_mode=expected_mode,
        )
        if _lock_identity(after_lock) != _lock_identity(before):
            raise AtomicReplaceError(
                "dataset lock changed while acquisition was pending"
            )
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        os.close(parent_fd)
