#!/usr/bin/env python3
"""Freeze the LUCAS crop distill/holdout split — ONCE, before any training.

LUCAS is the ladder's independent cross-validator ("never trained on by
any rung"). Distilling crop type from it burns that property unless the
data is split FIRST: a grouped-by-tile 70/30 freeze [user-approved
2026-08-31] keeps a holdout side that remains untouched-by-training, so
the cross-check survives as "LUCAS holdout never trained on".

Non-negotiables encoded here:
- **Grouped by tile** — points on one tile share context; a point-level
  split would leak tile context across sides (same isolation argument as
  the NFI head's grouped split).
- **The pre-existing index split is honoured**: the L1 index carries a
  'test' side (71 points) frozen by an earlier experiment; its tiles are
  FORCED into our holdout so no prior freeze leaks into distill-train.
- **Holdout must cover all 11 crop classes** (it is the future validator;
  a class absent there is a class we can never score). Seeded retry search,
  like the NFI grouped_split.
- Same physical pinning as the NFI set: tiles must carry s1_vv_vh (the
  SAR-cohort intersection) and points must sit inside every column's
  crop window (row/col in [off, off+min_img)).

Outputs (PVC):
- ``lucas_crop_distill_index.parquet`` — the 70% side, extract-ready
  (tile_name/tile_path/row/col/unified_class/point_id).
- ``lucas_crop_validator_holdout_index.parquet`` — the untouched 30% side
  with the same extract-ready schema. It is validator-only and must never be
  consumed by crop-distill training. Publishing it now means later validation
  never has to reconstruct identity from a mutable source.
- ``lucas_crop_split.json`` — provenance + the pinned plot list with
  ``key_cols`` so consumers verify the exact-match guard, plus exact
  holdout plot identity, and separate byte inventories/digests for distill
  and validator NPZ tiles.
- ``lucas_crop_split.MANIFEST.json`` — commit marker binding all root
  artifacts and protocol identities.
- ``crop_consumer/`` — byte-identical index/split/manifest projection only;
  the validator parquet is deliberately absent from the read-only crop mount.

The source LUCAS parquet's path and SHA-256 are historical build provenance.
Later verification is self-contained: the mutable source may move or change
without invalidating the already-frozen artifacts.

    python3 scripts/build_lucas_crop_split.py \
        --lucas-index /cephfs/lucas/lucas_tile_index.parquet \
        --data-dir /cephfs/unified_v2_512 \
        --out-dir /cephfs/distill
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
import math
import numbers
import os
import re
import secrets
import socket
import stat
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_pinned_plot_set import (
    REQUIRED_KEYS,
    VERSION_REQUIREMENTS,
    _crop_offset,
)
from crop_distill_protocol import CROP_MODELS

CROP_CLASSES = tuple(range(11, 22))  # vete..majs, unified schema v5
SEED = 42
HOLDOUT_FRAC = 0.30
MIN_HOLDOUT_PER_CLASS = 5
OOF_FOLDS = 5
PRIOR_TEST_POINT_COUNT = 71
PRIOR_TEST_TILE_COUNT = 53
PRIOR_TEST_SOURCE_REF = (
    "data/distill/distill_split.json@"
    "4378a6bea57fcabf4c8f78bb95a812ca474b61c2"
)
PRIOR_TEST_SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "distill"
    / "distill_split.json"
)
PRIOR_TEST_TILES_SHA256 = (
    "14b92d3f94eefcb5a46a237265876f0d3c01316f3281142a30201846d648885b"
)

INDEX_NAME = "lucas_crop_distill_index.parquet"
VALIDATOR_HOLDOUT_INDEX_NAME = "lucas_crop_validator_holdout_index.parquet"
SPLIT_NAME = "lucas_crop_split.json"
MANIFEST_NAME = "lucas_crop_split.MANIFEST.json"
LOCK_NAME = ".lucas_crop_split.lock"
CONSUMER_DIR_NAME = "crop_consumer"
ARTIFACT_NAMES = (INDEX_NAME, VALIDATOR_HOLDOUT_INDEX_NAME, SPLIT_NAME)
CONSUMER_ARTIFACT_NAMES = (INDEX_NAME, SPLIT_NAME, MANIFEST_NAME)
# What extract_plot_features actually reads from the index — the frozen
# parquet must carry every one of these to be consumable.
EXTRACT_COLUMNS = ("tile_name", "tile_path", "row", "col",
                   "unified_class", "point_id")
KEY_DIGEST_FORMAT = "sha256-jsonl-v1"
TILE_INVENTORY_FORMAT = "sha256-tile-bytes-jsonl-v1"
KEY_DIGEST_FIELDS = (
    "qualified_keys_sha256",
    "distill_keys_sha256",
    "holdout_keys_sha256",
    "partition_sha256",
)
PRIOR_TEST_BINDING_FIELDS = (
    "prior_test_tiles_sha256",
    "prior_test_keys_sha256",
)
TILE_BINDING_FIELDS = (
    "distill_input_data_sha256",
    "validator_holdout_input_data_sha256",
)
SOURCE_BINDING_FIELDS = (
    "source_index_path",
    "source_index_sha256",
    "forced_holdout_tiles_sha256",
    "forced_holdout_keys_sha256",
    "git_sha",
)
_CANONICAL_DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
_MAX_POINT_ID_DIGITS = 32
_READ_CHUNK_SIZE = 1 << 20
_OPEN_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_OPEN_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_OPEN_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


class UnsafeFileError(OSError):
    """A filesystem object violates the freeze's single-file identity rules."""


def _absolute_path(path: Path) -> Path:
    """Return a lexical absolute path without resolving through symlinks."""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _open_directory_tree(path: Path, *, create: bool = False) -> int:
    """Open a directory through descriptor-relative, no-symlink traversal.

    The returned descriptor pins the final directory even if an ancestor is
    renamed concurrently. When ``create`` is true, missing components are
    created relative to their already-open parent; a symlink can therefore
    never redirect an output write.
    """
    absolute = _absolute_path(path)
    flags = os.O_RDONLY | _OPEN_CLOEXEC | _OPEN_DIRECTORY | _OPEN_NOFOLLOW
    current_fd = os.open(os.sep, flags)
    try:
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise UnsafeFileError(
                    f"unsafe directory component {component!r} in {absolute}"
                )
            if create:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=current_fd)
                    os.fsync(current_fd)
                except FileExistsError:
                    pass
            child_fd = os.open(component, flags, dir_fd=current_fd)
            child_stat = os.fstat(child_fd)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child_fd)
                raise UnsafeFileError(
                    f"output path component is not a directory: {absolute}"
                )
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _entry_lstat(directory_fd: int, name: str) -> os.stat_result | None:
    """Stat a directory entry without following its final component."""
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _capture_regular_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    require_nonempty: bool = True,
) -> tuple[bytes, os.stat_result]:
    """Capture one unaliased regular file exactly once through a descriptor."""
    if Path(name).name != name or name in {"", ".", ".."}:
        raise UnsafeFileError(f"unsafe file name for {label}: {name!r}")
    flags = os.O_RDONLY | _OPEN_CLOEXEC | _OPEN_NOFOLLOW
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise UnsafeFileError(f"{label} must not be a symlink") from exc
        raise
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise UnsafeFileError(f"{label} is not a regular file")
        if before.st_nlink != 1:
            raise UnsafeFileError(
                f"{label} must have link count 1, found {before.st_nlink}"
            )
        if require_nonempty and before.st_size <= 0:
            raise UnsafeFileError(f"{label} is empty")

        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, _READ_CHUNK_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(fd)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or len(payload) != after.st_size
        ):
            raise UnsafeFileError(f"{label} changed while being captured")
        if after.st_nlink != 1:
            raise UnsafeFileError(
                f"{label} link count changed while being captured"
            )
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != after.st_dev
            or current.st_ino != after.st_ino
            or current.st_size != after.st_size
            or current.st_mtime_ns != after.st_mtime_ns
            or current.st_ctime_ns != after.st_ctime_ns
            or current.st_nlink != 1
        ):
            raise UnsafeFileError(f"{label} path changed while being captured")
        return payload, after
    finally:
        os.close(fd)


def _capture_regular_path(
    path: Path, *, label: str, require_nonempty: bool = True,
) -> tuple[bytes, os.stat_result]:
    parent_fd = _open_directory_tree(path.parent)
    try:
        return _capture_regular_at(
            parent_fd,
            path.name,
            label=label,
            require_nonempty=require_nonempty,
        )
    finally:
        os.close(parent_fd)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    """Hash one securely captured, unaliased regular file."""
    payload, _ = _capture_regular_path(path, label=f"hash input {path}")
    return _sha256_bytes(payload)


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def _jsonl_sha256(records: list[tuple[object, ...]]) -> str:
    """Hash sorted logical records, independent of JSON pretty-printing."""
    h = hashlib.sha256()
    for record in sorted(records):
        h.update(json.dumps(
            record, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _key_digest(keys: list[tuple[str, int]]) -> str:
    return _jsonl_sha256([(tile_name, point_id) for tile_name, point_id in keys])


def _partition_digest(
    distill_keys: list[tuple[str, int]],
    holdout_keys: list[tuple[str, int]],
) -> str:
    return _jsonl_sha256(
        [(tile_name, point_id, "distill")
         for tile_name, point_id in distill_keys]
        + [(tile_name, point_id, "holdout")
           for tile_name, point_id in holdout_keys]
    )


def _tile_digest(tile_names: list[str]) -> str:
    return _jsonl_sha256([(tile_name,) for tile_name in tile_names])


def _canonical_tile_names_sha256(tile_names: list[str]) -> str:
    payload = "\n".join(sorted(tile_names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tile_inventory_digest(records: list[dict[str, object]]) -> str:
    return _jsonl_sha256([
        (
            record["tile_name"],
            record["file_name"],
            record["tile_path"],
            record["size"],
            record["sha256"],
        )
        for record in records
    ])


def _derived_crop_window() -> tuple[int, int]:
    min_img = min(config.img_size for config in CROP_MODELS.values())
    off = _crop_offset(512, min_img)
    return off, off + min_img


def _valid_tile_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not value.isspace()
        and "\x00" not in value
        and value not in {".", ".."}
        and Path(value).name == value
    )


def _integral_values(
    values: list[object], field: str, label: str, *, minimum: int | None = None,
) -> tuple[list[int] | None, str | None]:
    result: list[int] = []
    for pos, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Real
        ):
            return None, (
                f"{label} has non-integral {field} at row {pos}: {value!r}"
            )
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            return None, (
                f"{label} has non-integral {field} at row {pos}: {value!r}"
            )
        integer = int(value)
        if minimum is not None and integer < minimum:
            return None, (
                f"{label} has {field} below {minimum} at row {pos}: {integer}"
            )
        result.append(integer)
    return result, None


def _point_id_values(
    values: list[object], label: str,
) -> tuple[list[int] | None, str | None]:
    """Normalize the real L1 point-id schema without accepting aliases.

    ``build_lucas_truth.py`` reads survey identifiers as strings and
    ``lucas_tile_coverage.py`` preserves that dtype. Frozen artifacts use
    integers, so accept either an integer or its one canonical decimal
    spelling. Whitespace, signs, decimal notation, and leading-zero aliases
    are rejected because they would make one logical identity hash in more
    than one way.
    """
    result: list[int] = []
    for pos, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)):
            return None, (
                f"{label} has non-canonical point_id at row {pos}: {value!r}"
            )
        if isinstance(value, numbers.Integral):
            integer = int(value)
            if integer < 0:
                return None, (
                    f"{label} has negative point_id at row {pos}: {integer}"
                )
        elif isinstance(value, str):
            if (
                len(value) > _MAX_POINT_ID_DIGITS
                or _CANONICAL_DECIMAL_RE.fullmatch(value) is None
            ):
                return None, (
                    f"{label} has non-canonical point_id at row {pos}: "
                    f"{value!r}"
                )
            integer = int(value)
        else:
            return None, (
                f"{label} has non-canonical point_id at row {pos}: {value!r}"
            )
        result.append(integer)
    return result, None


def _canonical_prior_test_tiles() -> tuple[list[str] | None, str | None]:
    """Load the exact historical tile set baked into the runtime image."""
    try:
        payload = json.loads(PRIOR_TEST_SOURCE_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, (
            f"cannot read canonical prior split {PRIOR_TEST_SOURCE_REF}: {exc}"
        )
    if not isinstance(payload, dict):
        return None, (
            f"canonical prior split {PRIOR_TEST_SOURCE_REF} is not an object"
        )
    raw_tiles = payload.get("test_tiles")
    if not isinstance(raw_tiles, list):
        return None, (
            f"canonical prior split {PRIOR_TEST_SOURCE_REF} lacks test_tiles"
        )
    if (
        any(not _valid_tile_name(tile) for tile in raw_tiles)
        or len(raw_tiles) != PRIOR_TEST_TILE_COUNT
        or len(set(raw_tiles)) != len(raw_tiles)
    ):
        return None, (
            f"canonical prior split {PRIOR_TEST_SOURCE_REF} has malformed "
            "test_tiles"
        )
    tiles = sorted(raw_tiles)
    digest = _canonical_tile_names_sha256(tiles)
    if digest != PRIOR_TEST_TILES_SHA256:
        return None, (
            f"canonical prior split {PRIOR_TEST_SOURCE_REF} has unexpected "
            f"test-tile digest {digest}"
        )
    return tiles, None


def _parse_key_records(
    records: object, label: str, *, allow_empty: bool = False,
) -> tuple[list[tuple[str, int]] | None, str | None]:
    if not isinstance(records, list) or (not records and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        return None, f"{label} must be {qualifier}"
    keys: list[tuple[str, int]] = []
    for pos, record in enumerate(records):
        if not isinstance(record, dict):
            return None, f"{label}[{pos}] is not an object"
        if set(record) != {"tile_name", "point_id"}:
            return None, (
                f"{label}[{pos}] must contain exactly tile_name and point_id"
            )
        tile_name = record["tile_name"]
        point_id = record["point_id"]
        if not _valid_tile_name(tile_name):
            return None, f"{label}[{pos}] has malformed tile_name {tile_name!r}"
        if (
            isinstance(point_id, bool)
            or not isinstance(point_id, int)
            or point_id < 0
        ):
            return None, f"{label}[{pos}] has malformed point_id {point_id!r}"
        keys.append((tile_name, point_id))
    if len(set(keys)) != len(keys):
        return None, f"duplicate plot keys in {label}"
    return keys, None


def _source_prior_test_identity(
    source: pd.DataFrame,
) -> tuple[tuple[list[str], list[tuple[str, int]]] | None, str | None]:
    """Return the canonical earlier holdout identity, or fail closed.

    The L1 ``split`` marker comes from the earlier *tile-level* distill split.
    The tracked split contains 53 held-out tiles; only a subset of them
    co-locates LUCAS rows (24 in the recovered Aug-11 L1 artifact), yielding
    71 rows. Therefore the historical anchor is the complete 53-tile set,
    not a fabricated requirement that all 53 tiles appear in the L1 index.
    The marker must equal the exact source intersection with those tiles,
    after which the observed 71 keys are frozen for all later checks.
    """
    required = ("split", "tile_name", "point_id")
    missing = [column for column in required if column not in source.columns]
    if missing:
        return None, (
            "source LUCAS index lacks required prior-test split marker "
            f"columns {missing}"
        )
    canonical_tiles, problem = _canonical_prior_test_tiles()
    if problem:
        return None, problem
    assert canonical_tiles is not None

    prior = source[source["split"] == "test"]
    intersection = source[source["tile_name"].isin(canonical_tiles)]
    if len(intersection) != PRIOR_TEST_POINT_COUNT:
        return None, (
            "source intersection with canonical prior-test tiles must contain "
            f"exactly {PRIOR_TEST_POINT_COUNT} rows, found {len(intersection)}"
        )
    if len(prior) != PRIOR_TEST_POINT_COUNT:
        return None, (
            "source prior-test marker must contain exactly "
            f"{PRIOR_TEST_POINT_COUNT} rows, found {len(prior)}"
        )

    def normalized_keys(
        frame: pd.DataFrame, label: str,
    ) -> tuple[list[tuple[str, int]] | None, str | None]:
        raw_tiles = frame["tile_name"].tolist()
        malformed = [tile for tile in raw_tiles if not _valid_tile_name(tile)]
        if malformed:
            return None, f"{label} has malformed tile names {malformed[:5]!r}"
        point_ids, point_problem = _point_id_values(
            frame["point_id"].tolist(), label
        )
        if point_problem:
            return None, point_problem
        assert point_ids is not None
        keys = sorted(zip(raw_tiles, point_ids, strict=True))
        if len(set(keys)) != len(keys):
            return None, f"{label} contains duplicate logical point keys"
        return keys, None

    prior_keys, problem = normalized_keys(prior, "source prior-test marker")
    if problem:
        return None, problem
    expected_keys, problem = normalized_keys(
        intersection, "source canonical prior-test intersection"
    )
    if problem:
        return None, problem
    assert prior_keys is not None and expected_keys is not None
    if prior_keys != expected_keys:
        difference = len(set(prior_keys) ^ set(expected_keys))
        return None, (
            "source prior-test marker does not equal the canonical 53-tile "
            f"intersection ({difference} differing keys)"
        )
    observed_tiles = {tile_name for tile_name, _ in prior_keys}
    if not observed_tiles or not observed_tiles <= set(canonical_tiles):
        return None, (
            "source prior-test rows must occupy a nonempty subset of the "
            "canonical 53-tile identity"
        )
    return (canonical_tiles, prior_keys), None


def _qualify_npz_bytes(
    payload: bytes, keys: tuple[str, ...],
) -> bool | None:
    """Inspect required keys and stamps from the bytes that were authenticated."""
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            names = set(archive.files)
            if any(key not in names for key in keys):
                return False
            for key in keys:
                requirement = VERSION_REQUIREMENTS.get(key)
                if requirement is None:
                    continue
                version_key, wanted = requirement
                if version_key not in names:
                    return False
                value = np.asarray(archive[version_key])
                if value.size != 1 or int(value.reshape(-1)[0]) != wanted:
                    return False
    except (EOFError, OSError, TypeError, ValueError):
        return None
    return True


def _capture_tile_at(
    directory_fd: int,
    *,
    tile_name: str,
    data_dir: Path,
    required_keys: tuple[str, ...],
) -> tuple[dict[str, object], bool | None]:
    """Read, hash, and qualify one tile from the same immutable byte capture."""
    file_name = f"{tile_name}.npz"
    path = data_dir / file_name
    payload, captured = _capture_regular_at(
        directory_fd, file_name, label=f"qualified tile {path}"
    )
    record: dict[str, object] = {
        "tile_name": tile_name,
        "file_name": file_name,
        "tile_path": str(path),
        "size": int(captured.st_size),
        "sha256": _sha256_bytes(payload),
    }
    return record, _qualify_npz_bytes(payload, required_keys)


def _build_tile_inventory(
    tile_names: set[str],
    captured_records: Mapping[str, dict[str, object]],
) -> tuple[list[dict[str, object]] | None, str | None]:
    """Select partition inventory from the build's single tile captures."""
    if not tile_names:
        return None, "cannot inventory an empty tile partition"
    records: list[dict[str, object]] = []
    for tile_name in sorted(tile_names):
        if not _valid_tile_name(tile_name):
            return None, f"cannot inventory malformed tile name {tile_name!r}"
        record = captured_records.get(tile_name)
        if record is None:
            return None, f"qualified tile lacks a captured identity: {tile_name}"
        records.append(dict(record))
    return records, None


def _validate_tile_inventory(
    records: object,
    *,
    label: str,
    expected_tiles: set[str],
    data_dir: Path,
    required_keys: tuple[str, ...],
    verify_bytes: bool,
) -> tuple[str | None, str | None]:
    """Validate one partition's inventory and its current tile bytes."""
    if not isinstance(records, list) or not records:
        return None, f"{label} must be a non-empty list"
    parsed: list[dict[str, object]] = []
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    previous_name: str | None = None
    for pos, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "tile_name", "file_name", "tile_path", "size", "sha256",
        }:
            return None, f"{label}[{pos}] has malformed schema"
        tile_name = record["tile_name"]
        if not _valid_tile_name(tile_name):
            return None, f"{label}[{pos}] has malformed tile_name {tile_name!r}"
        assert isinstance(tile_name, str)
        if previous_name is not None and tile_name <= previous_name:
            return None, f"{label} is not in unique canonical tile order"
        previous_name = tile_name
        file_name = record["file_name"]
        expected_file_name = f"{tile_name}.npz"
        if not isinstance(file_name, str) or file_name != expected_file_name:
            return None, (
                f"{label}[{pos}] file_name {file_name!r} != "
                f"{expected_file_name!r}"
            )
        expected_path = str(data_dir / expected_file_name)
        tile_path = record["tile_path"]
        if not isinstance(tile_path, str) or tile_path != expected_path:
            return None, (
                f"{label}[{pos}] tile_path {tile_path!r} != canonical "
                f"{expected_path!r}"
            )
        size = record["size"]
        if type(size) is not int or size <= 0:
            return None, f"{label}[{pos}] has malformed size {size!r}"
        digest = record["sha256"]
        if not _is_lower_hex(digest, 64):
            return None, f"{label}[{pos}] has malformed sha256 {digest!r}"
        if tile_name in seen_names or tile_path in seen_paths:
            return None, f"{label} contains a duplicate name or path"
        seen_names.add(tile_name)
        seen_paths.add(tile_path)
        parsed.append(record)
    if seen_names != expected_tiles:
        return None, (
            f"{label} tile identity differs from partition "
            f"({len(seen_names ^ expected_tiles)} differing tiles)"
        )

    inventory_digest = _tile_inventory_digest(parsed)
    if not verify_bytes:
        return inventory_digest, None

    try:
        data_fd = _open_directory_tree(data_dir)
    except OSError as exc:
        return None, f"cannot open inventoried tile directory {data_dir}: {exc}"
    try:
        for record in parsed:
            path = Path(str(record["tile_path"]))
            try:
                captured, qualifies = _capture_tile_at(
                    data_fd,
                    tile_name=str(record["tile_name"]),
                    data_dir=data_dir,
                    required_keys=required_keys,
                )
            except FileNotFoundError:
                return None, f"inventoried tile file missing: {path}"
            except OSError as exc:
                return None, f"cannot verify inventoried tile {path}: {exc}"
            if captured["size"] != record["size"]:
                return None, (
                    f"inventoried tile size changed for {path}: "
                    f"{captured['size']} != {record['size']}"
                )
            if captured["sha256"] != record["sha256"]:
                return None, f"inventoried tile sha256 changed for {path}"
            if qualifies is None:
                return None, f"inventoried tile is unreadable: {path}"
            if not qualifies:
                return None, (
                    "inventoried tile no longer satisfies required_keys: "
                    f"{path}"
                )
    finally:
        os.close(data_fd)
    return inventory_digest, None


def _validate_tile_files(
    tile_names: set[str], data_dir: Path, required_keys: tuple[str, ...],
) -> str | None:
    if not tile_names:
        return "qualified partition has no tiles"
    try:
        data_fd = _open_directory_tree(data_dir)
    except OSError as exc:
        return f"qualified tile directory cannot be opened: {data_dir}: {exc}"
    try:
        for tile_name in sorted(tile_names):
            tile = data_dir / f"{tile_name}.npz"
            try:
                _, qualifies = _capture_tile_at(
                    data_fd,
                    tile_name=tile_name,
                    data_dir=data_dir,
                    required_keys=required_keys,
                )
            except FileNotFoundError:
                return f"qualified tile file missing: {tile}"
            except OSError as exc:
                return f"qualified tile file cannot be inspected: {tile}: {exc}"
            if qualifies is None:
                return f"qualified tile file is unreadable: {tile}"
            if not qualifies:
                return (
                    "qualified tile file no longer satisfies required_keys: "
                    f"{tile}"
                )
    finally:
        os.close(data_fd)
    return None


def _validate_index_frame(
    frame: pd.DataFrame,
    *,
    data_dir: Path,
    required_keys: tuple[str, ...],
    crop_window: tuple[int, int],
    label: str = INDEX_NAME,
    exact_columns: bool = False,
    validate_tile_files: bool = True,
) -> tuple[list[tuple[str, int]] | None, str | None]:
    missing_cols = [column for column in EXTRACT_COLUMNS
                    if column not in frame.columns]
    if missing_cols:
        return None, f"{label} lacks extract columns {missing_cols}"
    if exact_columns and tuple(frame.columns) != EXTRACT_COLUMNS:
        return None, (
            f"{label} columns {list(frame.columns)!r} != exact extract schema "
            f"{list(EXTRACT_COLUMNS)!r}"
        )
    if frame.empty:
        return None, f"{label} has no rows"

    tile_names = frame["tile_name"].tolist()
    for pos, tile_name in enumerate(tile_names):
        if not _valid_tile_name(tile_name):
            return None, (
                f"{label} has malformed tile_name at row {pos}: {tile_name!r}"
            )

    point_ids, problem = _point_id_values(frame["point_id"].tolist(), label)
    if problem:
        return None, problem
    rows, problem = _integral_values(frame["row"].tolist(), "row", label)
    if problem:
        return None, problem
    cols, problem = _integral_values(frame["col"].tolist(), "col", label)
    if problem:
        return None, problem
    classes, problem = _integral_values(
        frame["unified_class"].tolist(), "unified_class", label
    )
    if problem:
        return None, problem
    assert point_ids is not None and rows is not None
    assert cols is not None and classes is not None

    keys = list(zip(tile_names, point_ids, strict=True))
    if len(set(keys)) != len(keys):
        return None, f"duplicate keys in {label}"

    invalid_classes = sorted(set(classes) - set(CROP_CLASSES))
    if invalid_classes:
        return None, (
            f"{label} has classes outside crop domain 11..21: "
            f"{invalid_classes[:5]}"
        )
    start, stop = crop_window
    for field, values in (("row", rows), ("col", cols)):
        invalid = [value for value in values if not start <= value < stop]
        if invalid:
            return None, (
                f"{label} has {field} outside crop window [{start}, {stop}): "
                f"{invalid[:5]}"
            )

    for pos, (tile_name, tile_path) in enumerate(zip(
        tile_names, frame["tile_path"].tolist(), strict=True
    )):
        expected = str(data_dir / f"{tile_name}.npz")
        if tile_path != expected:
            return None, (
                f"{label} tile_path at row {pos} is {tile_path!r}, "
                f"expected exact qualified file {expected!r}"
            )

    if validate_tile_files:
        problem = _validate_tile_files(set(tile_names), data_dir, required_keys)
        if problem:
            return None, problem
    return keys, None


def _prepare_source_crop_rows(
    source: pd.DataFrame, *, label: str = "source LUCAS index",
) -> tuple[pd.DataFrame | None, str | None]:
    missing_cols = [column for column in EXTRACT_COLUMNS
                    if column not in source.columns]
    if missing_cols:
        return None, f"{label} lacks required columns {missing_cols}"
    numeric_classes = pd.to_numeric(source["unified_class"], errors="coerce")
    crops = source[numeric_classes.isin(CROP_CLASSES)].copy()
    crops["unified_class"] = numeric_classes.loc[crops.index]
    if crops.empty:
        return None, f"{label} has no crop points in classes 11..21"
    point_ids, problem = _point_id_values(crops["point_id"].tolist(), label)
    if problem:
        return None, problem
    assert point_ids is not None
    crops["point_id"] = point_ids
    for field, minimum in (
        ("row", None), ("col", None), ("unified_class", None),
    ):
        values, problem = _integral_values(
            crops[field].tolist(), field, label, minimum=minimum
        )
        if problem:
            return None, problem
        assert values is not None
        crops[field] = values
    malformed_names = [
        name for name in crops["tile_name"].tolist()
        if not _valid_tile_name(name)
    ]
    if malformed_names:
        return None, (
            f"{label} has malformed tile_name values {malformed_names[:5]!r}"
        )
    return crops, None


def _validate_partition_against_source(
    source: pd.DataFrame,
    distill: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    crop_window: tuple[int, int],
) -> tuple[tuple[list[str], list[tuple[str, int]]] | None, str | None]:
    """Bind both published indices to exact rows in the source parquet.

    Only tiles in the frozen partition are compared. A previously absent
    source tile becoming available later must not invalidate an immutable
    freeze, while every published point on a frozen tile must still be an
    exact source row.
    """
    crops, problem = _prepare_source_crop_rows(source)
    if problem:
        return None, problem
    assert crops is not None
    start, stop = crop_window
    in_window = (
        (crops["row"] >= start) & (crops["row"] < stop)
        & (crops["col"] >= start) & (crops["col"] < stop)
    )
    crops = crops[in_window]
    partition = pd.concat([distill, holdout], ignore_index=True)
    partition_tiles = set(partition["tile_name"])
    source_partition = crops[crops["tile_name"].isin(partition_tiles)]

    source_keys = list(zip(
        source_partition["tile_name"].tolist(),
        source_partition["point_id"].tolist(),
        strict=True,
    ))
    if len(set(source_keys)) != len(source_keys):
        return None, "bound source index has duplicate qualified keys"
    published_keys = list(zip(
        partition["tile_name"].tolist(), partition["point_id"].tolist(),
        strict=True,
    ))
    source_key_set = set(source_keys)
    published_key_set = set(published_keys)
    if source_key_set != published_key_set:
        return None, (
            "published partition does not equal source rows on its frozen "
            f"tiles ({len(source_key_set ^ published_key_set)} differing keys)"
        )

    compare_columns = ("row", "col", "unified_class")
    source_values = {
        (row.tile_name, row.point_id): tuple(
            int(getattr(row, column)) for column in compare_columns
        )
        for row in source_partition.itertuples(index=False)
    }
    for row in partition.itertuples(index=False):
        key = (row.tile_name, row.point_id)
        published_values = tuple(
            int(getattr(row, column)) for column in compare_columns
        )
        if source_values[key] != published_values:
            return None, (
                f"published values for key {key!r} disagree with source index"
            )

    return _source_prior_test_identity(source)


def _publish(
    payload_writer: Callable[[BinaryIO], object], dest: Path,
) -> str:
    """Publish through an exclusive random temp fd, without overwriting.

    The writer receives the already-open file handle, so it cannot be
    redirected through a pre-created temp symlink. ``link(2)`` provides the
    atomic create-only commit: an existing destination is never replaced.
    """
    parent_fd = _open_directory_tree(dest.parent)
    temp_name: str | None = None
    published = False
    try:
        flags = (
            os.O_CREAT | os.O_EXCL | os.O_RDWR
            | _OPEN_CLOEXEC | _OPEN_NOFOLLOW
        )
        for _ in range(128):
            candidate = f".{dest.name}.tmp.{secrets.token_hex(16)}"
            try:
                temp_fd = os.open(
                    candidate, flags, mode=0o644, dir_fd=parent_fd
                )
            except FileExistsError:
                continue
            temp_name = candidate
            break
        else:
            raise FileExistsError(
                f"cannot reserve a unique temporary file for {dest}"
            )

        with os.fdopen(temp_fd, "w+b") as handle:
            payload_writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise UnsafeFileError(
                    f"temporary publication file for {dest} is aliased"
                )
            if before.st_size <= 0:
                raise UnsafeFileError(
                    f"temporary publication file for {dest} is empty"
                )
            handle.seek(0)
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(_READ_CHUNK_SIZE), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            if (
                before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or after.st_nlink != 1
            ):
                raise UnsafeFileError(
                    f"temporary publication file for {dest} changed"
                )

        try:
            os.link(
                temp_name,
                dest.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing publication {dest}"
            ) from exc
        published = True
        os.unlink(temp_name, dir_fd=parent_fd)
        temp_name = None
        published_stat = os.stat(
            dest.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(published_stat.st_mode)
            or published_stat.st_nlink != 1
            or published_stat.st_dev != after.st_dev
            or published_stat.st_ino != after.st_ino
        ):
            raise UnsafeFileError(f"published file identity is unsafe: {dest}")
        os.fsync(parent_fd)
        return digest.hexdigest()
    except BaseException:
        if published:
            try:
                current = os.stat(
                    dest.name, dir_fd=parent_fd, follow_symlinks=False
                )
                if current.st_dev == after.st_dev and current.st_ino == after.st_ino:
                    os.unlink(dest.name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except OSError:
                pass
        os.close(parent_fd)


def required_npz_keys() -> tuple[str, ...]:
    """The TRUE six-column input intersection, derived — never hand-listed.

    REQUIRED_KEYS is the SAR base, but a column may filter on more:
    tessera's dataset drops embedding-less tiles at init. A split frozen
    on a subset strands the stricter column — its extract drops the
    unqualified tiles' points and the pinned-OOF merge aborts the Job.
    """
    extra = {
        key
        for config in CROP_MODELS.values()
        for key in config.required_npz_keys
    }
    return tuple(sorted(set(REQUIRED_KEYS) | extra))


def tile_qualifies(path: Path, keys: tuple[str, ...]) -> bool | None:
    """Qualify one securely captured NPZ; ``None`` means unreadable/unsafe."""
    try:
        directory_fd = _open_directory_tree(path.parent)
    except OSError:
        return None
    try:
        try:
            _, qualifies = _capture_tile_at(
                directory_fd,
                tile_name=path.stem,
                data_dir=path.parent,
                required_keys=keys,
            )
        except OSError:
            return None
        return qualifies
    finally:
        os.close(directory_fd)


def _class_support_problem(
    frame: pd.DataFrame, *, label: str, require_groups: bool,
) -> str | None:
    support = frame["unified_class"].value_counts()
    missing_or_thin = {
        crop_class: int(support.get(crop_class, 0))
        for crop_class in CROP_CLASSES
        if support.get(crop_class, 0) < MIN_HOLDOUT_PER_CLASS
    }
    if missing_or_thin:
        return f"{label} lacks minimum class support: {missing_or_thin}"
    if require_groups:
        group_support = (
            frame.groupby("unified_class")["tile_name"].nunique().to_dict()
        )
        thin_groups = {
            crop_class: int(group_support.get(crop_class, 0))
            for crop_class in CROP_CLASSES
            if group_support.get(crop_class, 0) < OOF_FOLDS
        }
        if thin_groups:
            return (
                f"{label} lacks {OOF_FOLDS} distinct tile groups per class: "
                f"{thin_groups}"
            )
    return None


def _partition_support_problem(
    distill: pd.DataFrame, holdout: pd.DataFrame,
) -> str | None:
    """Require a scoreable holdout and a grouped-five-fold distill side."""
    return _class_support_problem(
        distill, label="distill", require_groups=True
    ) or _class_support_problem(
        holdout, label="holdout", require_groups=False
    )


def _select_holdout_tiles(
    crops: pd.DataFrame,
    *,
    seed: int,
    forced_tiles: set[str],
) -> tuple[tuple[int, set[str]] | None, str | None]:
    """Reproduce the protocol's first fully scoreable seeded partition."""
    all_tiles = set(crops["tile_name"])
    forced_present = forced_tiles & all_tiles
    candidates = np.array(sorted(all_tiles - forced_present))
    n_hold_total = round(len(all_tiles) * HOLDOUT_FRAC)
    n_random_hold = max(0, n_hold_total - len(forced_present))
    if n_random_hold > len(candidates):
        return None, (
            "holdout target exceeds the available non-forced tile pool"
        )
    best: tuple[tuple[int, int, int], int] | None = None
    for trial in range(50):
        rng = np.random.default_rng(seed + trial)
        selected = set(
            rng.choice(candidates, size=n_random_hold, replace=False).tolist()
        )
        hold_tiles = selected | forced_present
        holdout = crops[crops["tile_name"].isin(hold_tiles)]
        distill = crops[~crops["tile_name"].isin(hold_tiles)]
        hold_support = holdout["unified_class"].value_counts()
        dist_support = distill["unified_class"].value_counts()
        group_support = (
            distill.groupby("unified_class")["tile_name"].nunique().to_dict()
        )
        coverage = (
            sum(
                hold_support.get(crop_class, 0) >= MIN_HOLDOUT_PER_CLASS
                for crop_class in CROP_CLASSES
            ),
            sum(
                dist_support.get(crop_class, 0) >= MIN_HOLDOUT_PER_CLASS
                for crop_class in CROP_CLASSES
            ),
            sum(
                group_support.get(crop_class, 0) >= OOF_FOLDS
                for crop_class in CROP_CLASSES
            ),
        )
        if best is None or coverage > best[0]:
            best = (coverage, trial)
        if coverage == (len(CROP_CLASSES),) * 3:
            return (trial, hold_tiles), None
    assert best is not None
    return None, (
        "no seed in 50 trials produced a scoreable grouped partition "
        f"(best holdout/distill/group class coverage {best[0]}, trial "
        f"{best[1]})"
    )


def _capture_consumer_directory_fd(
    directory_fd: int, *, label: str,
) -> tuple[
    dict[str, bytes] | None,
    dict[str, os.stat_result] | None,
    str | None,
]:
    """Capture the exact consumer bundle without following any entry."""
    try:
        entries = sorted(os.listdir(directory_fd))
    except OSError as exc:
        return None, None, f"cannot inspect consumer directory {label}: {exc}"
    expected_entries = sorted(CONSUMER_ARTIFACT_NAMES)
    if entries != expected_entries:
        return None, None, (
            f"consumer directory must contain exactly {expected_entries}, "
            f"found {entries}"
        )
    captured: dict[str, bytes] = {}
    identities: dict[str, os.stat_result] = {}
    for name in CONSUMER_ARTIFACT_NAMES:
        try:
            payload, identity = _capture_regular_at(
                directory_fd, name, label=f"consumer file {label}/{name}"
            )
        except OSError as exc:
            return None, None, (
                f"consumer file missing or aliased: {label}/{name}: {exc}"
            )
        captured[name] = payload
        identities[name] = identity
    problem = _revalidate_regular_identities_at(
        directory_fd,
        identities,
        label=f"consumer file {label}",
    )
    if problem:
        return None, None, problem
    try:
        final_entries = sorted(os.listdir(directory_fd))
    except OSError as exc:
        return None, None, (
            f"cannot re-inspect consumer directory {label}: {exc}"
        )
    if final_entries != expected_entries:
        return None, None, (
            f"consumer directory changed while captured: expected "
            f"{expected_entries}, found {final_entries}"
        )
    return captured, identities, None


def _revalidate_regular_identities_at(
    directory_fd: int,
    identities: Mapping[str, os.stat_result],
    *,
    label: str,
) -> str | None:
    """Confirm every captured pathname still names its original safe inode."""
    for name, captured in identities.items():
        try:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            return f"{label} {name} changed after capture: {exc}"
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or current.st_dev != captured.st_dev
            or current.st_ino != captured.st_ino
            or current.st_mode != captured.st_mode
            or current.st_size != captured.st_size
            or current.st_mtime_ns != captured.st_mtime_ns
            or current.st_ctime_ns != captured.st_ctime_ns
        ):
            return f"{label} {name} changed or became aliased after capture"
    return None


def _validate_consumer_directory(bundle: Path) -> str | None:
    """Require exactly three unaliased regular consumer files."""
    try:
        directory_fd = _open_directory_tree(bundle)
    except OSError as exc:
        return f"consumer directory missing or aliased: {bundle}: {exc}"
    try:
        _, _, problem = _capture_consumer_directory_fd(
            directory_fd, label=str(bundle)
        )
        return problem
    finally:
        os.close(directory_fd)


def _validate_consumer_projection(
    out_dir_fd: int,
    canonical_artifacts: Mapping[str, bytes],
    *,
    require_manifest: bool = True,
) -> str | None:
    """Require an exact, byte-identical, independently stored projection."""
    flags = os.O_RDONLY | _OPEN_CLOEXEC | _OPEN_DIRECTORY | _OPEN_NOFOLLOW
    try:
        bundle_fd = os.open(CONSUMER_DIR_NAME, flags, dir_fd=out_dir_fd)
    except OSError as exc:
        return (
            "consumer projection invalid: directory missing or aliased: "
            f"{exc}"
        )
    try:
        if require_manifest:
            captured, _, problem = _capture_consumer_directory_fd(
                bundle_fd, label=CONSUMER_DIR_NAME
            )
        else:
            expected = (INDEX_NAME, SPLIT_NAME)
            entries = sorted(os.listdir(bundle_fd))
            if entries != sorted(expected):
                return (
                    "staged consumer projection must contain exactly "
                    f"{sorted(expected)}, found {entries}"
                )
            captured = {}
            staged_identities: dict[str, os.stat_result] = {}
            problem = None
            for name in expected:
                try:
                    payload, identity = _capture_regular_at(
                        bundle_fd,
                        name,
                        label=f"staged consumer file {CONSUMER_DIR_NAME}/{name}",
                    )
                except OSError as exc:
                    return (
                        "staged consumer file missing or aliased: "
                        f"{CONSUMER_DIR_NAME}/{name}: {exc}"
                    )
                captured[name] = payload
                staged_identities[name] = identity
            problem = _revalidate_regular_identities_at(
                bundle_fd,
                staged_identities,
                label="staged consumer file",
            )
            if problem:
                return problem
            final_entries = sorted(os.listdir(bundle_fd))
            if final_entries != sorted(expected):
                return (
                    "staged consumer projection changed while captured: "
                    f"found {final_entries}"
                )
    finally:
        os.close(bundle_fd)
    if problem:
        return f"consumer projection invalid: {problem}"
    assert captured is not None
    names = CONSUMER_ARTIFACT_NAMES if require_manifest else (INDEX_NAME, SPLIT_NAME)
    for name in names:
        if name not in canonical_artifacts:
            return f"consumer projection lacks canonical bytes for {name}"
        if captured[name] != canonical_artifacts[name]:
            return f"consumer projection bytes differ for {name}"
    return None


def freeze_state(
    out_dir: Path,
    *,
    expected_git_sha: str | None = None,
    include_validator_holdout: bool = True,
) -> tuple[str, str]:
    """('frozen'|'corrupt'|'partial'|'absent', detail). Split is AUTHORITATIVE.

    The k8s Job is deletable state (TTL 48 h) but the split must be frozen
    ONCE: a re-run after the PVC changed would move previously trained
    tiles into holdout and silently burn the holdout's never-trained-on
    property.

    The MANIFEST is the integrity-bearing marker that binds every root
    artifact by content hash. The consumer copy is published last, after full
    pre-commit validation, so its presence never authorizes a root-partial
    freeze. Existence checks alone accepted a mixed pair from two racing
    builders — and a truncated parquet — as 'frozen'.
    """
    if expected_git_sha is not None and not _is_lower_hex(expected_git_sha, 40):
        return "corrupt", f"malformed expected git SHA {expected_git_sha!r}"
    try:
        out_dir_fd = _open_directory_tree(out_dir)
    except FileNotFoundError:
        return "absent", ""
    except OSError as exc:
        return (
            "corrupt",
            f"output directory missing or aliased: {out_dir}: {exc}",
        )
    try:
        if _entry_lstat(out_dir_fd, MANIFEST_NAME) is None:
            visible_artifacts = (
                ARTIFACT_NAMES
                if include_validator_holdout
                else (INDEX_NAME, SPLIT_NAME)
            )
            projection_present = (
                include_validator_holdout
                and _entry_lstat(out_dir_fd, CONSUMER_DIR_NAME) is not None
            )
            if projection_present or any(
                _entry_lstat(out_dir_fd, name) is not None
                for name in visible_artifacts
            ):
                return (
                    "partial",
                    "artifact(s) present without a manifest — interrupted freeze",
                )
            return "absent", ""

        artifact_names = (
            ARTIFACT_NAMES
            if include_validator_holdout
            else (INDEX_NAME, SPLIT_NAME)
        )
        captured_artifacts: dict[str, bytes]
        captured_identities: dict[str, os.stat_result]
        if include_validator_holdout:
            captured_artifacts = {}
            captured_identities = {}
            for name in (MANIFEST_NAME, *artifact_names):
                try:
                    payload, identity = _capture_regular_at(
                        out_dir_fd, name, label=f"freeze artifact {name}"
                    )
                except FileNotFoundError:
                    return (
                        "corrupt",
                        f"{name} missing but {MANIFEST_NAME} claims it",
                    )
                except OSError as exc:
                    return "corrupt", f"unsafe or unreadable {name}: {exc}"
                captured_artifacts[name] = payload
                captured_identities[name] = identity
        else:
            captured, identities, problem = _capture_consumer_directory_fd(
                out_dir_fd, label=str(out_dir)
            )
            if problem:
                return "corrupt", problem
            assert captured is not None and identities is not None
            captured_artifacts = captured
            captured_identities = identities

        try:
            manifest = json.loads(captured_artifacts[MANIFEST_NAME])
            artifact_hashes = manifest["artifacts"]
        except (UnicodeError, ValueError, KeyError, TypeError) as exc:
            return "corrupt", f"unreadable {MANIFEST_NAME}: {exc}"
        if not isinstance(artifact_hashes, dict):
            return "corrupt", f"{MANIFEST_NAME} artifacts is not an object"
        for name in artifact_names:
            if name not in artifact_hashes:
                return "corrupt", f"{MANIFEST_NAME} lacks a hash for {name}"
            expected = artifact_hashes[name]
            if not _is_lower_hex(expected, 64):
                return "corrupt", (
                    f"{MANIFEST_NAME} has malformed sha256 for {name}"
                )
            got = _sha256_bytes(captured_artifacts[name])
            if got != expected:
                return "corrupt", (
                    f"{name} sha256 {got[:12]}… does not match "
                    f"{MANIFEST_NAME} ({expected[:12]}…)"
                )
        try:
            problem = _validate_frozen_semantics(
                manifest,
                artifact_bytes=captured_artifacts,
                out_dir_fd=out_dir_fd,
                expected_git_sha=expected_git_sha,
                include_validator_holdout=include_validator_holdout,
            )
        except Exception as exc:  # noqa: BLE001 - corrupt input is unbounded
            return "corrupt", f"semantic validation failed: {exc}"
        if problem:
            return "corrupt", problem
        problem = _revalidate_regular_identities_at(
            out_dir_fd,
            captured_identities,
            label="freeze artifact",
        )
        if problem:
            return "corrupt", problem
        if include_validator_holdout:
            # The semantic pass may spend minutes sweeping tile bytes. Capture
            # the consumer projection again at the return boundary and compare
            # it with the root bytes, then ensure root paths stayed pinned while
            # that final consumer capture ran.
            problem = _validate_consumer_projection(
                out_dir_fd, captured_artifacts
            )
            if problem:
                return "corrupt", problem
            problem = _revalidate_regular_identities_at(
                out_dir_fd,
                captured_identities,
                label="freeze artifact",
            )
            if problem:
                return "corrupt", problem
        else:
            final_artifacts, _, problem = _capture_consumer_directory_fd(
                out_dir_fd, label=str(out_dir)
            )
            if problem:
                return "corrupt", problem
            if final_artifacts != captured_artifacts:
                return "corrupt", "consumer artifacts changed during validation"
        return "frozen", f"validated against {MANIFEST_NAME}"
    finally:
        os.close(out_dir_fd)


def _validate_frozen_semantics(
    manifest: dict,
    *,
    artifact_bytes: Mapping[str, bytes],
    out_dir_fd: int,
    expected_git_sha: str | None = None,
    include_validator_holdout: bool = True,
    consumer_manifest_required: bool = True,
) -> str | None:
    """Cross-artifact consistency — hashes alone only bind BYTES to the
    marker. An attacker-free but corrupted freeze (tampered parquet with a
    refreshed marker hash, a schema drift, a duplicate key) passes the
    hash check while the JSON and parquet disagree on which plots exist.
    Returns a problem description, or None if consistent.
    """
    split = json.loads(artifact_bytes[SPLIT_NAME])
    for field in (
        "seed", "trial_offset", "holdout_frac", "min_holdout_per_class",
        "key_cols", "plots", "holdout_plots", "n_qualified", "n_distill",
        "n_holdout", "holdout_tiles", "required_keys", "truth_col",
        "crop_window", "key_digest_format", *KEY_DIGEST_FIELDS,
        *SOURCE_BINDING_FIELDS, *PRIOR_TEST_BINDING_FIELDS,
        *TILE_BINDING_FIELDS, "forced_holdout_plots_from_prior_split",
        "prior_test_point_count", "prior_test_tile_count",
        "prior_test_tiles", "prior_test_plots", "prior_test_source_ref",
        "tile_inventory_format",
        "distill_tile_inventory", "validator_holdout_tile_inventory",
    ):
        if field not in split:
            return f"{SPLIT_NAME} lacks required field '{field}'"
    for field in (
        *KEY_DIGEST_FIELDS, *SOURCE_BINDING_FIELDS,
        *PRIOR_TEST_BINDING_FIELDS, *TILE_BINDING_FIELDS,
    ):
        if field not in manifest:
            return f"{MANIFEST_NAME} lacks required field '{field}'"

    for field, minimum, maximum in (
        ("seed", 0, None),
        ("trial_offset", 0, 49),
        ("min_holdout_per_class", 1, None),
    ):
        value = split[field]
        if type(value) is not int:  # bool and integral floats are forbidden
            return f"{SPLIT_NAME} has non-integer {field}={value!r}"
        if value < minimum or (maximum is not None and value > maximum):
            return f"{SPLIT_NAME} has out-of-range {field}={value!r}"
    if split["min_holdout_per_class"] != MIN_HOLDOUT_PER_CLASS:
        return (
            f"unexpected min_holdout_per_class "
            f"{split['min_holdout_per_class']!r}"
        )
    if split["seed"] != SEED:
        return f"unexpected protocol seed {split['seed']!r}; expected {SEED}"
    holdout_frac = split["holdout_frac"]
    if (
        type(holdout_frac) is not float
        or not math.isfinite(holdout_frac)
        or holdout_frac != HOLDOUT_FRAC
    ):
        return f"unexpected holdout_frac {holdout_frac!r}"
    if split["key_cols"] != ["tile_name", "point_id"]:
        return f"unexpected key_cols {split['key_cols']}"

    expected_required_keys = list(required_npz_keys())
    if split["required_keys"] != expected_required_keys:
        return (
            f"unexpected required_keys {split['required_keys']!r}; "
            f"current intersection is {expected_required_keys!r}"
        )
    crop_window = split["crop_window"]
    if (
        not isinstance(crop_window, list)
        or len(crop_window) != 2
        or any(type(value) is not int for value in crop_window)
    ):
        return f"{SPLIT_NAME} has malformed crop_window {crop_window!r}"
    expected_window = list(_derived_crop_window())
    if crop_window != expected_window:
        return (
            f"unexpected crop_window {crop_window!r}; "
            f"current derived window is {expected_window!r}"
        )
    if split["truth_col"] != "unified_class":
        return f"unexpected truth_col {split['truth_col']!r}"
    if split["key_digest_format"] != KEY_DIGEST_FORMAT:
        return f"unexpected key_digest_format {split['key_digest_format']!r}"
    if split["tile_inventory_format"] != TILE_INVENTORY_FORMAT:
        return (
            "unexpected tile_inventory_format "
            f"{split['tile_inventory_format']!r}"
        )
    if manifest.get("tile_inventory_format") != TILE_INVENTORY_FORMAT:
        return f"{MANIFEST_NAME} has unexpected tile_inventory_format"

    split_git_sha = split["git_sha"]
    manifest_git_sha = manifest["git_sha"]
    if not _is_lower_hex(split_git_sha, 40):
        return f"{SPLIT_NAME} has malformed git_sha {split_git_sha!r}"
    if not _is_lower_hex(manifest_git_sha, 40):
        return f"{MANIFEST_NAME} has malformed git_sha {manifest_git_sha!r}"
    if manifest_git_sha != split_git_sha:
        return f"git_sha mismatch between {SPLIT_NAME} and {MANIFEST_NAME}"
    if expected_git_sha is not None and split_git_sha != expected_git_sha:
        return (
            f"frozen git_sha {split_git_sha} does not match expected "
            f"{expected_git_sha}"
        )

    distill_keys, problem = _parse_key_records(
        split["plots"], f"{SPLIT_NAME}.plots"
    )
    if problem:
        return problem
    holdout_keys, problem = _parse_key_records(
        split["holdout_plots"], f"{SPLIT_NAME}.holdout_plots"
    )
    if problem:
        return problem
    assert distill_keys is not None and holdout_keys is not None
    distill_key_set = set(distill_keys)
    holdout_key_set = set(holdout_keys)
    overlap = distill_key_set & holdout_key_set
    if overlap:
        return f"distill/holdout key overlap ({len(overlap)} keys)"
    distill_tiles = {tile_name for tile_name, _ in distill_keys}
    holdout_key_tiles = {tile_name for tile_name, _ in holdout_keys}
    tile_overlap = distill_tiles & holdout_key_tiles
    if tile_overlap:
        return (
            f"distill/holdout tile leak ({len(tile_overlap)} tiles; "
            f"first {sorted(tile_overlap)[:5]})"
        )

    for field in ("n_qualified", "n_distill", "n_holdout"):
        count = split[field]
        if type(count) is not int or count < 0:
            return f"{SPLIT_NAME} has malformed {field}={count!r}"
    expected_counts = {
        "n_distill": len(distill_keys),
        "n_holdout": len(holdout_keys),
        "n_qualified": len(distill_keys) + len(holdout_keys),
    }
    for field, expected in expected_counts.items():
        if split[field] != expected:
            return (
                f"{SPLIT_NAME} {field}={split[field]} != partition count "
                f"{expected}"
            )
        manifest_count = manifest.get(field)
        if type(manifest_count) is not int or manifest_count != expected:
            return (
                f"{MANIFEST_NAME} {field}={manifest_count!r} != "
                f"{SPLIT_NAME} {field}={expected}"
            )

    holdout_tiles = split["holdout_tiles"]
    if (
        not isinstance(holdout_tiles, list)
        or any(not _valid_tile_name(tile) for tile in holdout_tiles)
        or len(set(holdout_tiles)) != len(holdout_tiles)
        or holdout_tiles != sorted(holdout_tiles)
    ):
        return f"malformed holdout_tiles {holdout_tiles!r}"
    expected_holdout_tiles = sorted(holdout_key_tiles)
    if holdout_tiles != expected_holdout_tiles:
        return (
            f"holdout_tiles identity disagrees with holdout_plots: "
            f"expected {expected_holdout_tiles[:5]}"
        )

    for field, expected in (
        ("prior_test_point_count", PRIOR_TEST_POINT_COUNT),
        ("prior_test_tile_count", PRIOR_TEST_TILE_COUNT),
    ):
        value = split[field]
        if type(value) is not int or value != expected:
            return f"{SPLIT_NAME} has unexpected {field}={value!r}"
        if manifest.get(field) != expected:
            return f"{MANIFEST_NAME} has unexpected {field}"
    prior_tiles = split["prior_test_tiles"]
    if (
        not isinstance(prior_tiles, list)
        or any(not _valid_tile_name(tile) for tile in prior_tiles)
        or len(prior_tiles) != PRIOR_TEST_TILE_COUNT
        or len(set(prior_tiles)) != len(prior_tiles)
        or prior_tiles != sorted(prior_tiles)
    ):
        return f"malformed canonical prior_test_tiles {prior_tiles!r}"
    prior_keys, problem = _parse_key_records(
        split["prior_test_plots"], f"{SPLIT_NAME}.prior_test_plots"
    )
    if problem:
        return problem
    assert prior_keys is not None
    if len(prior_keys) != PRIOR_TEST_POINT_COUNT:
        return (
            f"canonical prior-test identity has {len(prior_keys)} keys, "
            f"expected {PRIOR_TEST_POINT_COUNT}"
        )
    if prior_keys != sorted(prior_keys):
        return "canonical prior-test keys are not in canonical order"
    observed_prior_tiles = {tile_name for tile_name, _ in prior_keys}
    if not observed_prior_tiles or not observed_prior_tiles <= set(prior_tiles):
        return (
            "canonical prior-test keys must occupy a nonempty subset of the "
            "53-tile identity"
        )
    if split["prior_test_source_ref"] != PRIOR_TEST_SOURCE_REF:
        return (
            f"unexpected prior_test_source_ref "
            f"{split['prior_test_source_ref']!r}"
        )
    if manifest.get("prior_test_source_ref") != PRIOR_TEST_SOURCE_REF:
        return f"{MANIFEST_NAME} has unexpected prior_test_source_ref"
    prior_tile_digest = _canonical_tile_names_sha256(prior_tiles)
    if prior_tile_digest != PRIOR_TEST_TILES_SHA256:
        return "canonical prior-test tile digest differs from protocol anchor"
    prior_key_digest = _key_digest(prior_keys)
    for field, expected in (
        ("prior_test_tiles_sha256", prior_tile_digest),
        ("prior_test_keys_sha256", prior_key_digest),
    ):
        if split[field] != expected:
            return f"{SPLIT_NAME} {field} does not match prior-test identity"
        if manifest.get(field) != expected:
            return f"{MANIFEST_NAME} {field} does not match {SPLIT_NAME}"

    forced_tiles = split.get("forced_holdout_tiles_from_prior_split")
    if (
        not isinstance(forced_tiles, list)
        or any(not _valid_tile_name(tile) for tile in forced_tiles)
        or len(set(forced_tiles)) != len(forced_tiles)
        or forced_tiles != sorted(forced_tiles)
    ):
        return (
            "malformed forced_holdout_tiles_from_prior_split "
            f"{forced_tiles!r}"
        )
    if not set(forced_tiles) <= holdout_key_tiles:
        return "forced holdout tiles are not a subset of the holdout"
    expected_forced_tiles = sorted(set(prior_tiles) & (
        distill_tiles | holdout_key_tiles
    ))
    if forced_tiles != expected_forced_tiles:
        return (
            "forced holdout tiles do not equal canonical prior-test tiles "
            "present in the qualified partition"
        )

    forced_keys, problem = _parse_key_records(
        split["forced_holdout_plots_from_prior_split"],
        f"{SPLIT_NAME}.forced_holdout_plots_from_prior_split",
        allow_empty=True,
    )
    if problem:
        return problem
    assert forced_keys is not None
    if forced_keys != sorted(forced_keys):
        return "forced holdout keys are not in canonical order"
    expected_forced_keys = sorted(
        key for key in prior_keys if key[0] in set(forced_tiles)
    )
    if forced_keys != expected_forced_keys:
        return "forced holdout keys disagree with canonical prior-test identity"
    if forced_tiles != sorted({tile_name for tile_name, _ in forced_keys}):
        return "forced holdout tile identity disagrees with forced keys"
    leaked_prior_keys = set(forced_keys) & distill_key_set
    if leaked_prior_keys:
        return "prior-test point identity leaked into the distill partition"

    forced_digest = _tile_digest(forced_tiles)
    if split["forced_holdout_tiles_sha256"] != forced_digest:
        return f"{SPLIT_NAME} forced_holdout_tiles_sha256 does not match tiles"
    if manifest["forced_holdout_tiles_sha256"] != forced_digest:
        return (
            f"{MANIFEST_NAME} forced_holdout_tiles_sha256 does not match "
            f"{SPLIT_NAME}"
        )
    forced_key_digest = _key_digest(forced_keys)
    if split["forced_holdout_keys_sha256"] != forced_key_digest:
        return f"{SPLIT_NAME} forced_holdout_keys_sha256 does not match keys"
    if manifest["forced_holdout_keys_sha256"] != forced_key_digest:
        return (
            f"{MANIFEST_NAME} forced_holdout_keys_sha256 does not match "
            f"{SPLIT_NAME}"
        )

    calculated_digests = {
        "distill_keys_sha256": _key_digest(distill_keys),
        "holdout_keys_sha256": _key_digest(holdout_keys),
        "qualified_keys_sha256": _key_digest(distill_keys + holdout_keys),
        "partition_sha256": _partition_digest(distill_keys, holdout_keys),
    }
    for field, expected in calculated_digests.items():
        if split[field] != expected:
            return f"{SPLIT_NAME} {field} does not match its partition"
        if manifest.get(field) != expected:
            return (
                f"{MANIFEST_NAME} {field} does not match {SPLIT_NAME}"
            )

    run_args = manifest.get("run_args")
    if not isinstance(run_args, dict):
        return f"{MANIFEST_NAME} has malformed run_args"
    raw_data_dir = run_args.get("data_dir")
    if not isinstance(raw_data_dir, str) or not raw_data_dir:
        return f"{MANIFEST_NAME} lacks a normalized data_dir"
    data_dir = Path(raw_data_dir)
    if not data_dir.is_absolute() or str(_absolute_path(data_dir)) != raw_data_dir:
        return f"{MANIFEST_NAME} data_dir is not an absolute normalized path"
    if type(run_args.get("seed")) is not int or run_args["seed"] != split["seed"]:
        return f"{MANIFEST_NAME} run_args.seed does not match {SPLIT_NAME}"

    source_path = split["source_index_path"]
    if (
        not isinstance(source_path, str)
        or not source_path
        or not Path(source_path).is_absolute()
        or str(_absolute_path(Path(source_path))) != source_path
    ):
        return f"{SPLIT_NAME} has malformed source_index_path {source_path!r}"
    if manifest["source_index_path"] != source_path:
        return "source_index_path mismatch between split and manifest"
    if run_args.get("lucas_index") != source_path:
        return f"{MANIFEST_NAME} run_args.lucas_index does not match source"
    source_sha = split["source_index_sha256"]
    if not _is_lower_hex(source_sha, 64):
        return f"{SPLIT_NAME} has malformed source_index_sha256 {source_sha!r}"
    if manifest["source_index_sha256"] != source_sha:
        return "source_index_sha256 mismatch between split and manifest"

    for field in TILE_BINDING_FIELDS:
        value = split[field]
        if not _is_lower_hex(value, 64):
            return f"{SPLIT_NAME} has malformed {field} {value!r}"
        if manifest.get(field) != value:
            return f"{field} mismatch between {SPLIT_NAME} and {MANIFEST_NAME}"

    distill_inventory_digest, problem = _validate_tile_inventory(
        split["distill_tile_inventory"],
        label=f"{SPLIT_NAME}.distill_tile_inventory",
        expected_tiles=distill_tiles,
        data_dir=data_dir,
        required_keys=tuple(expected_required_keys),
        verify_bytes=include_validator_holdout,
    )
    if problem:
        return problem
    if distill_inventory_digest != split["distill_input_data_sha256"]:
        return f"{SPLIT_NAME} distill_input_data_sha256 does not match inventory"

    distill = pd.read_parquet(io.BytesIO(artifact_bytes[INDEX_NAME]))
    parquet_keys, problem = _validate_index_frame(
        distill,
        data_dir=data_dir,
        required_keys=tuple(expected_required_keys),
        crop_window=tuple(expected_window),
        exact_columns=True,
        validate_tile_files=False,
    )
    if problem:
        return problem
    assert parquet_keys is not None
    parquet_key_set = set(parquet_keys)
    if parquet_key_set != distill_key_set:
        return (
            f"key sets disagree: {SPLIT_NAME} has {len(distill_key_set)}, "
            f"{INDEX_NAME} has {len(parquet_key_set)} "
            f"({len(distill_key_set ^ parquet_key_set)} differing)"
        )
    problem = _class_support_problem(
        distill, label="distill", require_groups=True
    )
    if problem:
        return problem
    if not include_validator_holdout:
        return None

    holdout_inventory_digest, problem = _validate_tile_inventory(
        split["validator_holdout_tile_inventory"],
        label=f"{SPLIT_NAME}.validator_holdout_tile_inventory",
        expected_tiles=holdout_key_tiles,
        data_dir=data_dir,
        required_keys=tuple(expected_required_keys),
        verify_bytes=True,
    )
    if problem:
        return problem
    if (
        holdout_inventory_digest
        != split["validator_holdout_input_data_sha256"]
    ):
        return (
            f"{SPLIT_NAME} validator_holdout_input_data_sha256 does not "
            "match inventory"
        )

    holdout = pd.read_parquet(
        io.BytesIO(artifact_bytes[VALIDATOR_HOLDOUT_INDEX_NAME])
    )
    holdout_parquet_keys, problem = _validate_index_frame(
        holdout,
        data_dir=data_dir,
        required_keys=tuple(expected_required_keys),
        crop_window=tuple(expected_window),
        label=VALIDATOR_HOLDOUT_INDEX_NAME,
        exact_columns=True,
        validate_tile_files=False,
    )
    if problem:
        return problem
    assert holdout_parquet_keys is not None
    holdout_parquet_key_set = set(holdout_parquet_keys)
    if holdout_parquet_key_set != holdout_key_set:
        return (
            f"key sets disagree: {SPLIT_NAME} holdout has "
            f"{len(holdout_key_set)}, {VALIDATOR_HOLDOUT_INDEX_NAME} has "
            f"{len(holdout_parquet_key_set)} "
            f"({len(holdout_key_set ^ holdout_parquet_key_set)} differing)"
        )
    problem = _partition_support_problem(distill, holdout)
    if problem:
        return problem

    union = pd.concat([distill, holdout], ignore_index=True)
    selection, problem = _select_holdout_tiles(
        union, seed=split["seed"], forced_tiles=set(prior_tiles)
    )
    if problem:
        return f"frozen partition cannot be deterministically reproduced: {problem}"
    assert selection is not None
    expected_trial, expected_selected_tiles = selection
    if split["trial_offset"] != expected_trial:
        return (
            f"trial_offset {split['trial_offset']} does not match first "
            f"valid deterministic trial {expected_trial}"
        )
    if holdout_key_tiles != expected_selected_tiles:
        return (
            "frozen holdout partition differs from deterministic seeded "
            "selection"
        )

    problem = _validate_consumer_projection(
        out_dir_fd,
        artifact_bytes,
        require_manifest=consumer_manifest_required,
    )
    if problem:
        return problem

    return None


def _validate_precommit_freeze(
    out_dir: Path,
    manifest_payload: bytes,
    *,
    expected_git_sha: str,
) -> str | None:
    """Fully validate staged bytes before exposing either commit marker.

    The consumer directory deliberately lacks its MANIFEST during this pass.
    Publishing the root marker and then the consumer marker is safe only after
    the same root artifacts, tile inventory, and projected bytes have passed
    the complete semantic validation here.
    """
    try:
        manifest = json.loads(manifest_payload)
        artifact_hashes = manifest["artifacts"]
    except (UnicodeError, ValueError, KeyError, TypeError) as exc:
        return f"cannot validate staged {MANIFEST_NAME}: {exc}"
    if not isinstance(artifact_hashes, dict):
        return f"staged {MANIFEST_NAME} artifacts is not an object"

    try:
        out_dir_fd = _open_directory_tree(out_dir)
    except OSError as exc:
        return f"cannot open staged output directory {out_dir}: {exc}"
    try:
        artifact_bytes: dict[str, bytes] = {MANIFEST_NAME: manifest_payload}
        identities: dict[str, os.stat_result] = {}
        for name in ARTIFACT_NAMES:
            try:
                payload, identity = _capture_regular_at(
                    out_dir_fd,
                    name,
                    label=f"staged freeze artifact {name}",
                )
            except OSError as exc:
                return f"unsafe or unreadable staged artifact {name}: {exc}"
            expected = artifact_hashes.get(name)
            if not _is_lower_hex(expected, 64):
                return f"staged {MANIFEST_NAME} has malformed sha256 for {name}"
            if _sha256_bytes(payload) != expected:
                return f"staged artifact {name} does not match {MANIFEST_NAME}"
            artifact_bytes[name] = payload
            identities[name] = identity

        try:
            problem = _validate_frozen_semantics(
                manifest,
                artifact_bytes=artifact_bytes,
                out_dir_fd=out_dir_fd,
                expected_git_sha=expected_git_sha,
                include_validator_holdout=True,
                consumer_manifest_required=False,
            )
        except Exception as exc:  # noqa: BLE001 - corrupt input is unbounded
            return f"staged semantic validation failed: {exc}"
        if problem:
            return problem
        problem = _revalidate_regular_identities_at(
            out_dir_fd,
            identities,
            label="staged freeze artifact",
        )
        if problem:
            return problem
        problem = _validate_consumer_projection(
            out_dir_fd,
            artifact_bytes,
            require_manifest=False,
        )
        if problem:
            return problem
        return _revalidate_regular_identities_at(
            out_dir_fd,
            identities,
            label="staged freeze artifact",
        )
    finally:
        os.close(out_dir_fd)


def acquire_lock(out_dir: Path) -> Path:
    """Exclusive cross-process lock (O_EXCL) — two racing builders must
    never both reach the publish step; the loser dies here, loudly."""
    lock = out_dir / LOCK_NAME
    directory_fd = _open_directory_tree(out_dir)
    try:
        try:
            fd = os.open(
                LOCK_NAME,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY
                | _OPEN_CLOEXEC | _OPEN_NOFOLLOW,
                mode=0o644,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            raise SystemExit(
                f"freeze lock {lock} exists — another builder is running, or a "
                f"crashed one left it behind. Recovery: verify no "
                f"ladder-lucas-crop-split pod is active, then delete the lock "
                f"and re-run.") from None
        with os.fdopen(fd, "w") as fh:
            lock_stat = os.fstat(fh.fileno())
            if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
                raise UnsafeFileError(f"freeze lock identity is unsafe: {lock}")
            fh.write(json.dumps({
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started": datetime.now(timezone.utc).isoformat(),
            }))
            fh.flush()
            os.fsync(fh.fileno())
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return lock


def release_lock(lock: Path) -> None:
    """Remove only the lock entry in its securely opened parent directory."""
    directory_fd = _open_directory_tree(lock.parent)
    try:
        os.unlink(lock.name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lucas-index", help="required to build")
    ap.add_argument("--data-dir", help="required to build")
    ap.add_argument("--out-dir", required=True)
    verify_mode = ap.add_mutually_exclusive_group()
    verify_mode.add_argument(
        "--verify",
        action="store_true",
        help="fully validate an existing freeze, including the validator-"
             "only holdout parquet, then exit — 0 iff frozen-valid",
    )
    verify_mode.add_argument(
        "--verify-consumer",
        action="store_true",
        help="validate the manifest, split metadata, and distill index "
             "without touching the validator-only holdout parquet; crop "
             "consumers run this before extraction",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--git-sha", default=None,
                    help="required lowercase 40-hex commit SHA for a new "
                         "build; recorded in every provenance artifact")
    ap.add_argument("--expected-git-sha", default=None,
                    help="optional lowercase 40-hex source anchor required "
                         "when verifying or re-entering a freeze")
    args = ap.parse_args()

    out_dir = _absolute_path(Path(args.out_dir))
    if (
        args.expected_git_sha is not None
        and not _is_lower_hex(args.expected_git_sha, 40)
    ):
        ap.error("--expected-git-sha must be exactly 40 lowercase hex characters")
    if args.seed != SEED:
        ap.error(f"--seed is protocol-pinned to {SEED}")
    if args.verify or args.verify_consumer:
        state, detail = freeze_state(
            out_dir,
            expected_git_sha=args.expected_git_sha,
            include_validator_holdout=not args.verify_consumer,
        )
        if state == "frozen":
            print(f"freeze VALID: {detail}")
            return
        raise SystemExit(f"freeze INVALID ({state}): {detail}")
    if not args.lucas_index or not args.data_dir:
        ap.error("--lucas-index and --data-dir are required to build")
    if not _is_lower_hex(args.git_sha, 40):
        ap.error(
            "--git-sha is required and must be exactly 40 lowercase hex "
            "characters"
        )
    if (
        args.expected_git_sha is not None
        and args.expected_git_sha != args.git_sha
    ):
        ap.error("--expected-git-sha must match --git-sha for a build")

    try:
        directory_fd = _open_directory_tree(out_dir, create=True)
    except OSError as exc:
        raise SystemExit(
            f"cannot create secure output directory {out_dir}: {exc}"
        ) from exc
    else:
        os.close(directory_fd)
    expected_git_sha = args.expected_git_sha or args.git_sha
    _guard(out_dir, expected_git_sha=expected_git_sha)
    lock = acquire_lock(out_dir)
    try:
        # Authoritative re-check UNDER the lock: the pre-lock guard is
        # advisory only — a racing builder may have published between the
        # check and the lock acquisition.
        _guard(out_dir, expected_git_sha=expected_git_sha)
        _build(args, out_dir)
    finally:
        release_lock(lock)


def _guard(out_dir: Path, *, expected_git_sha: str | None = None) -> None:
    state, detail = freeze_state(out_dir, expected_git_sha=expected_git_sha)
    if state == "frozen":
        # Idempotent no-op, exit 0: a re-applied Job must not fail, and it
        # must NOT re-freeze. Re-freezing is a deliberate manual act.
        print(f"split already frozen in {out_dir} ({detail}) — refusing to "
              "overwrite. To re-freeze deliberately, first authorize an "
              "explicitly reviewed recovery operation as UID 2000 that "
              "restores owner-write on the frozen 0550 root; routine "
              "storage-prep preserves the freeze and cannot unlock it. Then "
              f"verify nothing has trained on the current split, delete "
              f"{INDEX_NAME}, {VALIDATOR_HOLDOUT_INDEX_NAME}, {SPLIT_NAME} "
              f"and {MANIFEST_NAME}, remove {CONSUMER_DIR_NAME}/, and re-run.")
        raise SystemExit(0)
    if state == "corrupt":
        raise SystemExit(
            f"CORRUPT freeze in {out_dir}: {detail}. The artifacts do not "
            f"match their commit marker — do NOT train on them. Recovery: "
            "authorize an explicitly reviewed owner-write operation as UID "
            "2000 before deletion; routine storage-prep preserves a frozen "
            "0550 root and cannot unlock it. Confirm nothing has trained on "
            "this split, then delete "
            f"{INDEX_NAME}, {VALIDATOR_HOLDOUT_INDEX_NAME}, {SPLIT_NAME} and "
            f"{MANIFEST_NAME}, remove {CONSUMER_DIR_NAME}/, and re-run.")
    if state == "partial":
        raise SystemExit(
            f"PARTIAL freeze in {out_dir}: {detail}; neither side is "
            "trustworthy. Recovery: authorize an explicitly reviewed "
            "owner-write operation as UID 2000 before deletion; routine "
            "storage-prep cannot unlock a frozen 0550 root. Confirm nothing "
            "has trained on it, then delete the surviving file(s) and re-run.")


def _build(args, out_dir: Path) -> None:
    source_path = _absolute_path(Path(args.lucas_index))
    try:
        source_bytes, _ = _capture_regular_path(
            source_path, label="source LUCAS index"
        )
    except OSError as exc:
        raise SystemExit(f"cannot inspect source LUCAS index: {exc}") from exc
    source_sha = _sha256_bytes(source_bytes)
    df = pd.read_parquet(io.BytesIO(source_bytes))
    prior_identity, problem = _source_prior_test_identity(df)
    if problem:
        raise SystemExit(problem)
    assert prior_identity is not None
    prior_test_tiles, prior_test_keys = prior_identity
    crops, problem = _prepare_source_crop_rows(df)
    if problem:
        raise SystemExit(problem)
    assert crops is not None
    print(f"crop points: {len(crops)}/{len(df)} on "
          f"{crops['tile_name'].nunique()} tiles")

    # Crop-window intersection — identical arithmetic to the NFI pinned set.
    crop_window = _derived_crop_window()
    off, stop = crop_window
    in_win = ((crops["row"] >= off) & (crops["row"] < stop)
              & (crops["col"] >= off) & (crops["col"] < stop))
    print(f"crop window [{off}, {stop}): "
          f"{int((~in_win).sum())} border points excluded")
    crops = crops[in_win]

    # Tile qualification on the FULL six-column intersection (unreadable
    # aborts, as in the NFI set). REQUIRED_KEYS alone strands tessera.
    req_keys = required_npz_keys()
    print(f"qualifying on npz keys: {req_keys}")
    data_dir = _absolute_path(Path(args.data_dir))
    qual: dict[str, bool] = {}
    unreadable: list[str] = []
    captured_tiles: dict[str, dict[str, object]] = {}
    try:
        data_dir_fd = _open_directory_tree(data_dir)
    except OSError as exc:
        raise SystemExit(f"cannot open tile directory {data_dir}: {exc}") from exc
    try:
        for name in crops["tile_name"].unique():
            try:
                record, ok = _capture_tile_at(
                    data_dir_fd,
                    tile_name=str(name),
                    data_dir=data_dir,
                    required_keys=req_keys,
                )
            except FileNotFoundError:
                qual[name] = False
                continue
            except OSError as exc:
                unreadable.append(f"{name}: {exc}")
                continue
            if ok is None:
                unreadable.append(f"{name}: malformed NPZ")
            else:
                qual[name] = ok
                if ok:
                    captured_tiles[str(name)] = record
    finally:
        os.close(data_dir_fd)
    if unreadable:
        raise SystemExit(
            f"{len(unreadable)} unreadable tiles (first {unreadable[:5]}) — "
            f"no split is frozen on a degraded PVC.")
    crops = crops[crops["tile_name"].map(qual).fillna(False)]
    print(f"after key/window/existence pinning: {len(crops)} points on "
          f"{crops['tile_name'].nunique()} tiles")
    if crops.empty:
        raise SystemExit("no crop points have nonempty qualifying tile files")

    # The source index's tile_path is advisory and may point at a stale PVC
    # or a same-named file elsewhere. The frozen extract index always names
    # the exact file that passed qualification in this data-dir.
    crops = crops.copy()
    crops["tile_path"] = [
        str(data_dir / f"{tile_name}.npz")
        for tile_name in crops["tile_name"]
    ]
    qualified_keys, problem = _validate_index_frame(
        crops,
        data_dir=data_dir,
        required_keys=req_keys,
        crop_window=crop_window,
        label="qualified LUCAS source",
        validate_tile_files=False,
    )
    if problem:
        raise SystemExit(problem)
    assert qualified_keys is not None

    # The L1 index's own 'test' side is an earlier freeze — its tiles go
    # to OUR holdout unconditionally.
    forced_holdout = set(prior_test_tiles)
    selection, problem = _select_holdout_tiles(
        crops, seed=args.seed, forced_tiles=forced_holdout
    )
    if problem:
        raise SystemExit(problem)
    assert selection is not None
    trial, hold_tiles = selection
    if trial:
        print(f"  note: seed+{trial} used for full holdout class coverage")

    hold = crops[crops["tile_name"].isin(hold_tiles)]
    dist = crops[~crops["tile_name"].isin(hold_tiles)]
    assert not (set(dist.tile_name) & set(hold.tile_name)), "tile leak"
    if dist.empty or hold.empty:
        raise SystemExit(
            f"qualified partition must have nonempty distill and holdout sides "
            f"(distill={len(dist)}, holdout={len(hold)})"
        )
    problem = _partition_support_problem(dist, hold)
    if problem:
        raise SystemExit(problem)
    print(f"distill: {len(dist)} points / {dist.tile_name.nunique()} tiles; "
          f"holdout: {len(hold)} points / {hold.tile_name.nunique()} tiles")
    print("holdout class support:",
          hold["unified_class"].value_counts().sort_index().to_dict())

    dist = (
        dist.sort_values(["tile_name", "point_id"])
        .loc[:, EXTRACT_COLUMNS]
        .reset_index(drop=True)
    )
    hold = (
        hold.sort_values(["tile_name", "point_id"])
        .loc[:, EXTRACT_COLUMNS]
        .reset_index(drop=True)
    )
    distill_keys = [
        (str(tile_name), int(point_id))
        for tile_name, point_id
        in dist[["tile_name", "point_id"]].itertuples(index=False)
    ]
    holdout_keys = [
        (str(tile_name), int(point_id))
        for tile_name, point_id
        in hold[["tile_name", "point_id"]].itertuples(index=False)
    ]
    if set(qualified_keys) != set(distill_keys) | set(holdout_keys):
        raise SystemExit("internal error: split does not cover qualified keys")
    forced_tiles = sorted(
        str(tile) for tile in forced_holdout & set(crops["tile_name"])
    )
    validated_prior_identity, problem = _validate_partition_against_source(
        df, dist, hold, crop_window=crop_window
    )
    if problem:
        raise SystemExit(problem)
    if validated_prior_identity != prior_identity:
        raise SystemExit(
            "source prior-test identity changed during split construction"
        )
    forced_keys = sorted(
        key for key in prior_test_keys if key[0] in set(forced_tiles)
    )
    distill_inventory, problem = _build_tile_inventory(
        set(dist["tile_name"]), captured_tiles
    )
    if problem:
        raise SystemExit(problem)
    holdout_inventory, problem = _build_tile_inventory(
        set(hold["tile_name"]), captured_tiles
    )
    if problem:
        raise SystemExit(problem)
    assert distill_inventory is not None and holdout_inventory is not None
    key_digests = {
        "qualified_keys_sha256": _key_digest(distill_keys + holdout_keys),
        "distill_keys_sha256": _key_digest(distill_keys),
        "holdout_keys_sha256": _key_digest(holdout_keys),
        "partition_sha256": _partition_digest(distill_keys, holdout_keys),
    }
    source_bindings = {
        "source_index_path": str(source_path),
        "source_index_sha256": source_sha,
        "forced_holdout_tiles_sha256": _tile_digest(forced_tiles),
        "forced_holdout_keys_sha256": _key_digest(forced_keys),
        "git_sha": args.git_sha,
    }
    prior_test_bindings = {
        "prior_test_tiles_sha256": _canonical_tile_names_sha256(
            prior_test_tiles
        ),
        "prior_test_keys_sha256": _key_digest(prior_test_keys),
    }
    tile_bindings = {
        "distill_input_data_sha256": _tile_inventory_digest(
            distill_inventory
        ),
        "validator_holdout_input_data_sha256": _tile_inventory_digest(
            holdout_inventory
        ),
    }
    split_doc = {
        "seed": args.seed, "trial_offset": trial,
        "holdout_frac": HOLDOUT_FRAC,
        "min_holdout_per_class": MIN_HOLDOUT_PER_CLASS,
        "required_keys": list(req_keys),
        "crop_window": list(crop_window),
        "key_cols": ["tile_name", "point_id"],
        "truth_col": "unified_class",
        "key_digest_format": KEY_DIGEST_FORMAT,
        "tile_inventory_format": TILE_INVENTORY_FORMAT,
        "n_qualified": len(crops),
        "n_distill": len(dist),
        "n_holdout": len(hold),
        "holdout_tiles": sorted(str(tile) for tile in hold_tiles),
        "forced_holdout_tiles_from_prior_split": forced_tiles,
        "forced_holdout_plots_from_prior_split": [
            {"tile_name": tile_name, "point_id": point_id}
            for tile_name, point_id in forced_keys
        ],
        "prior_test_point_count": PRIOR_TEST_POINT_COUNT,
        "prior_test_tile_count": PRIOR_TEST_TILE_COUNT,
        "prior_test_source_ref": PRIOR_TEST_SOURCE_REF,
        "prior_test_tiles": prior_test_tiles,
        "prior_test_plots": [
            {"tile_name": tile_name, "point_id": point_id}
            for tile_name, point_id in prior_test_keys
        ],
        "distill_tile_inventory": distill_inventory,
        "validator_holdout_tile_inventory": holdout_inventory,
        "plots": [
            {"tile_name": str(t), "point_id": int(p)}
            for t, p in dist[["tile_name", "point_id"]].itertuples(index=False)
        ],
        "holdout_plots": [
            {"tile_name": str(t), "point_id": int(p)}
            for t, p in hold[["tile_name", "point_id"]].itertuples(index=False)
        ],
        **key_digests,
        **source_bindings,
        **prior_test_bindings,
        **tile_bindings,
    }

    # Publish order is load-bearing: data artifacts first (each atomically),
    # then full pre-commit validation, then root + consumer markers. A crash
    # before the final consumer marker leaves no consumer-valid freeze.
    try:
        final_source_bytes, _ = _capture_regular_path(
            source_path, label="source LUCAS index"
        )
    except OSError as exc:
        raise SystemExit(
            f"source LUCAS index became unsafe while building: {exc}"
        ) from exc
    if _sha256_bytes(final_source_bytes) != source_sha:
        raise SystemExit(
            "source LUCAS index changed while the split was being built"
        )
    hashes = {
        INDEX_NAME: _publish(
            lambda handle: dist.to_parquet(handle), out_dir / INDEX_NAME
        ),
        VALIDATOR_HOLDOUT_INDEX_NAME: _publish(
            lambda handle: hold.to_parquet(handle),
            out_dir / VALIDATOR_HOLDOUT_INDEX_NAME,
        ),
        SPLIT_NAME: _publish(
            lambda handle: handle.write(
                json.dumps(split_doc, indent=1).encode("utf-8")
            ),
            out_dir / SPLIT_NAME),
    }
    manifest_doc = {
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "run_args": {
            "lucas_index": str(source_path),
            "data_dir": str(data_dir),
            "seed": args.seed,
        },
        "artifacts": hashes,
        "n_qualified": len(crops),
        "n_distill": len(dist),
        "n_holdout": len(hold),
        "prior_test_point_count": PRIOR_TEST_POINT_COUNT,
        "prior_test_tile_count": PRIOR_TEST_TILE_COUNT,
        "prior_test_source_ref": PRIOR_TEST_SOURCE_REF,
        "tile_inventory_format": TILE_INVENTORY_FORMAT,
        **key_digests,
        **source_bindings,
        **prior_test_bindings,
        **tile_bindings,
    }
    manifest_payload = json.dumps(manifest_doc, indent=1).encode("utf-8")

    # Directory-level read-only projection for crop consumers.  Kubernetes
    # mounts this directory as one subPath, avoiding file-subPath inode
    # staleness while keeping validator rows out of the consumer mount.
    # Neither commit marker is exposed until the complete staged freeze has
    # passed full validation. The consumer MANIFEST is published last across
    # the whole operation, so a crop pod can never accept a root-partial freeze.
    consumer_dir = out_dir / CONSUMER_DIR_NAME
    parent_fd = _open_directory_tree(out_dir)
    try:
        os.mkdir(CONSUMER_DIR_NAME, mode=0o755, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        raise SystemExit(
            f"cannot create consumer projection {consumer_dir}: {exc}"
        ) from exc
    finally:
        os.close(parent_fd)
    for name in (INDEX_NAME, SPLIT_NAME):
        source = out_dir / name
        try:
            source_payload, _ = _capture_regular_path(
                source, label=f"canonical projection source {name}"
            )
        except OSError as exc:
            raise SystemExit(
                f"cannot capture projection source {source}: {exc}"
            ) from exc
        _publish(
            lambda handle, payload=source_payload: handle.write(payload),
            consumer_dir / name,
        )
    problem = _validate_precommit_freeze(
        out_dir,
        manifest_payload,
        expected_git_sha=args.git_sha,
    )
    if problem:
        raise SystemExit(
            f"staged split failed pre-commit validation: {problem}"
        )

    _publish(
        lambda handle: handle.write(manifest_payload),
        out_dir / MANIFEST_NAME,
    )
    _publish(
        lambda handle: handle.write(manifest_payload),
        consumer_dir / MANIFEST_NAME,
    )
    state, detail = freeze_state(out_dir, expected_git_sha=args.git_sha)
    if state != "frozen":
        raise SystemExit(
            f"published split failed self-validation ({state}): {detail}"
        )
    consumer_state, consumer_detail = freeze_state(
        consumer_dir,
        expected_git_sha=args.git_sha,
        include_validator_holdout=False,
    )
    if consumer_state != "frozen":
        raise SystemExit(
            "published consumer projection failed self-validation "
            f"({consumer_state}): {consumer_detail}"
        )
    print(
        f"wrote {out_dir}/{INDEX_NAME} + {VALIDATOR_HOLDOUT_INDEX_NAME} + "
        f"{SPLIT_NAME} + {MANIFEST_NAME} + {CONSUMER_DIR_NAME}/"
    )


if __name__ == "__main__":
    main()
