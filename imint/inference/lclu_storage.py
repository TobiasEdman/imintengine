"""Single-process storage accounting and leases for the annual-map cache.

The API owns the output root's flock. Only rebuildable prepared inputs and
failure diagnostics are reclaimed; maps and staged source data are retained.
"""
from collections import OrderedDict
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import threading
import uuid


# Native inputs contain 64 MiB of embeddings and 11 MiB of auxiliaries.
# These caps include compression/atomic-write overhead and bounded diagnostics.
PREPARATION_BYTES = 256 * 2**20
PREPARATION_ARRAY_BYTES = 80 * 2**20
PREPARATION_METADATA_BYTES = 2**20
PREPARATION_LOG_BYTES = 2**20
MAP_BYTES = 64 * 2**20
STORAGE_MESSAGE = (
    "Demons lagringsutrymme räcker inte för en ny karta. "
    "Tidigare skapade kartor kan fortfarande visas."
)


class StorageLimitError(RuntimeError):
    """A new write would exceed the release budget or filesystem reserve."""


def _bytes(path: Path) -> int:
    """Count a single writer's files without following source-data symlinks."""
    if path.is_symlink():
        return path.lstat().st_size
    if path.is_file():
        return path.stat().st_size
    total = 0
    for directory, _, files in os.walk(path, followlinks=False):
        for name in files:
            total += (Path(directory) / name).lstat().st_size
    return total


class StorageBudget:
    def __init__(self, root: Path, *, max_bytes: int, min_free_bytes: int):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('Runtime storage budget must be positive')
        if type(min_free_bytes) is not int or min_free_bytes <= 0:
            raise ValueError('Runtime free-space reserve must be positive')
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self._reserved = 0
        self._free_estimate: int | None = None
        self._lock = threading.Lock()
        self._leases: dict[Path, int] = {}
        self._writing: set[Path] = set()
        # Full reconciliation happens once, before accepting requests. Later
        # transactions count only their own few files, including failure logs.
        for path in (self.root / 'lclu', self.root / 'lclu-inputs',
                     self.root / 'lclu-inputs' / 'failures'):
            if path.is_symlink():
                raise ValueError('Runtime cache directories must not be symlinks')
        self._used = _bytes(self.root)
        candidates = [p for p in (self.root / 'lclu-inputs').glob('*')
                      if re.fullmatch(r'[a-f0-9]{64}', p.name) and p.is_dir() and not p.is_symlink()]
        candidates += [p for p in (self.root / 'lclu-inputs' / 'failures').glob('*.log')
                       if p.is_file() and not p.is_symlink()]
        self._idle: OrderedDict[Path, int] = OrderedDict(
            (p, _bytes(p)) for p in sorted(candidates, key=lambda p: p.stat().st_mtime_ns))

    @staticmethod
    def cleanup_staging(root: Path) -> None:
        """Call only after acquiring the API flock, before creating any workers."""
        for parent, prefix in ((root / 'lclu-inputs', '.preparing-'),
                               (root / 'lclu', '.pending-')):
            if parent.is_symlink():
                raise ValueError('Runtime cache directories must not be symlinks')
            for path in parent.glob(prefix + '*'):
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)

    @classmethod
    def from_environment(cls, root: Path) -> 'StorageBudget':
        return cls(root,
                   max_bytes=int(os.environ.get('IMINT_LCLU_STORAGE_MAX_BYTES', str(8 * 2**30))),
                   min_free_bytes=int(os.environ.get('IMINT_LCLU_STORAGE_MIN_FREE_BYTES', str(2 * 2**30))))

    def used_bytes(self) -> int:
        with self._lock:
            return self._used

    def _path(self, path: Path) -> Path:
        path = Path(os.path.abspath(path))
        if self.root not in path.parents or path.resolve() != path:
            raise ValueError('Runtime writers must stay inside the owned cache without symlinks')
        return path

    def _rebuildable(self, path: Path) -> bool:
        inputs = self.root / 'lclu-inputs'
        return ((path.parent == inputs and bool(re.fullmatch(r'[a-f0-9]{64}', path.name)))
                or (path.parent == inputs / 'failures' and path.suffix == '.log'))

    def _available(self) -> list[Path]:
        return [p for p in self._idle if not self._leases.get(p) and p not in self._writing]

    def _available_free(self) -> int:
        observed = shutil.disk_usage(self.root).free
        if self._free_estimate is None:
            self._free_estimate = observed
        elif not self._reserved:
            # Keep local write debits even when statfs reports them late.
            # During writes a lower observation may include unaccounted files;
            # use it for this check only, avoiding a second debit on completion.
            self._free_estimate = min(self._free_estimate, observed)
        return min(self._free_estimate, observed)

    def _account(self, growth: int) -> None:
        self._used += growth
        if self._free_estimate is not None:
            self._free_estimate -= growth

    def _check(self, additional_bytes: int, reclaimable: int = 0, *, free_bytes: int | None = None) -> None:
        if type(additional_bytes) is not int or additional_bytes <= 0:
            raise ValueError('A positive write allowance is required')
        reserved = self._reserved + additional_bytes
        if self._used - reclaimable + reserved > self.max_bytes:
            raise StorageLimitError('Annual-map runtime byte budget exhausted')
        if free_bytes is None:
            free_bytes = self._available_free()
        if free_bytes + reclaimable < self.min_free_bytes + reserved:
            raise StorageLimitError('Annual-map filesystem reserve would be consumed')

    def check(self, additional_bytes: int) -> None:
        """Cheap admission hint, including idle reclaimable bytes; no disk walk."""
        with self._lock:
            self._check(additional_bytes, sum(self._idle[p] for p in self._available()))

    def _evict(self, path: Path) -> int:
        self._path(path)
        size = self._idle[path]
        if path.is_dir():
            # A partial deletion must never look like a reusable prepared key.
            # Startup's flock-protected sweep can finish an interrupted delete.
            tombstone = path.with_name('.preparing-evicted-' + uuid.uuid4().hex)
            os.rename(path, tombstone)
            self._idle.pop(path)
            try:
                shutil.rmtree(tombstone)
            finally:
                self._account(_bytes(tombstone) - size)
        else:
            path.unlink(missing_ok=True)
            self._idle.pop(path)
            self._account(-size)
        return size

    def _make_room(self, additional_bytes: int) -> None:
        available = self._available()
        # CephFS may report deletions late. Use one free-space snapshot and
        # credit our completed evictions, instead of repeatedly deleting to
        # chase a stale statfs value. The estimate also retains earlier write
        # debits, so delayed write reporting cannot oversubscribe later calls.
        free_bytes = self._available_free()
        self._check(additional_bytes, sum(self._idle[p] for p in available),
                    free_bytes=free_bytes)
        candidates = iter(available)
        while True:
            try:
                self._check(additional_bytes, free_bytes=free_bytes)
                return
            except StorageLimitError:
                path = next(candidates, None)
                if path is None:
                    raise
                # Only closed generated inputs/logs are candidates, not COGs.
                free_bytes += self._evict(path)

    @contextmanager
    def lease(self, path: Path):
        """Pin an input before lookup/preparation until its final read completes."""
        path = self._path(path)
        with self._lock:
            self._leases[path] = self._leases.get(path, 0) + 1
            if path in self._idle:
                self._idle.move_to_end(path)
        try:
            yield
        finally:
            with self._lock:
                self._leases[path] -= 1
                if not self._leases[path]:
                    del self._leases[path]
                if path in self._idle:
                    self._idle.move_to_end(path)
                    os.utime(path, None)  # preserve recent use across a restart

    @contextmanager
    def reserve(self, additional_bytes: int, *, paths: tuple[Path, ...] = ()):
        """Reserve worst-case growth; reconcile only the declared writer paths."""
        paths = tuple(self._path(p) for p in paths)
        with self._lock:
            if any(p == q or p in q.parents or q in p.parents
                   for p in paths for q in self._writing):
                raise RuntimeError('Overlapping runtime cache writers')
            self._writing.update(paths)
            try:
                self._make_room(additional_bytes)
                before = {p: _bytes(p) for p in paths}
            except BaseException:
                self._writing.difference_update(paths)
                raise
            self._reserved += additional_bytes
        try:
            yield
        finally:
            with self._lock:
                for path in paths:
                    size = _bytes(path)
                    self._account(size - before[path])
                    if self._rebuildable(path):
                        self._idle.pop(path, None)
                        if size:
                            self._idle[path] = size
                self._writing.difference_update(paths)
                self._reserved -= additional_bytes
