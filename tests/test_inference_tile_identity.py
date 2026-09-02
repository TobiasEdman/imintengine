"""Authenticate frozen NPZ bytes in the same read used by inference."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inference_comparison.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_tile_identity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inference = _load_module()


def _write_marker(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


def _identity(path: Path) -> tuple[int, str]:
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def test_verified_npz_uses_authenticated_memory_buffer(tmp_path: Path) -> None:
    tile = tmp_path / "tile.npz"
    np.savez(tile, spectral=np.arange(6, dtype=np.float32))
    size, sha256 = _identity(tile)

    data, buffer = inference._load_npz_for_inference(
        tile, expected_size=size, expected_sha256=sha256
    )
    try:
        assert buffer is not None
        assert np.array_equal(data["spectral"], np.arange(6, dtype=np.float32))
    finally:
        data.close()
        buffer.close()


def test_same_schema_byte_mutation_is_rejected(tmp_path: Path) -> None:
    tile = tmp_path / "tile.npz"
    np.savez(tile, spectral=np.zeros(6, dtype=np.float32))
    size, sha256 = _identity(tile)
    np.savez(tile, spectral=np.ones(6, dtype=np.float32))
    assert tile.stat().st_size == size

    with pytest.raises(inference.TileIdentityError, match="sha256 mismatch"):
        inference._load_npz_for_inference(
            tile, expected_size=size, expected_sha256=sha256
        )


def test_size_and_identity_format_fail_before_numpy_load(tmp_path: Path) -> None:
    tile = tmp_path / "tile.npz"
    np.savez(tile, spectral=np.zeros(2, dtype=np.float32))
    size, sha256 = _identity(tile)

    with pytest.raises(inference.TileIdentityError, match="size mismatch"):
        inference._load_npz_for_inference(
            tile, expected_size=size + 1, expected_sha256=sha256
        )
    with pytest.raises(inference.TileIdentityError, match="64 lowercase hex"):
        inference._load_npz_for_inference(
            tile, expected_size=size, expected_sha256=sha256.upper()
        )


def test_symlink_tile_is_rejected(tmp_path: Path) -> None:
    tile = tmp_path / "tile.npz"
    link = tmp_path / "link.npz"
    np.savez(tile, spectral=np.zeros(2, dtype=np.float32))
    link.symlink_to(tile)
    size, sha256 = _identity(tile)

    with pytest.raises(inference.TileIdentityError, match="without following links"):
        inference._load_npz_for_inference(
            link, expected_size=size, expected_sha256=sha256
        )


def test_hardlinked_tile_is_rejected(tmp_path: Path) -> None:
    tile = tmp_path / "tile.npz"
    alias = tmp_path / "alias.npz"
    np.savez(tile, spectral=np.zeros(2, dtype=np.float32))
    os.link(tile, alias)
    size, sha256 = _identity(tile)

    with pytest.raises(inference.TileIdentityError, match="exactly one hard link"):
        inference._load_npz_for_inference(
            tile, expected_size=size, expected_sha256=sha256
        )


def test_path_swap_after_descriptor_open_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tile = tmp_path / "tile.npz"
    original = tmp_path / "opened-original.npz"
    replacement = tmp_path / "replacement.npz"
    np.savez(tile, spectral=np.zeros(2, dtype=np.float32))
    np.savez(replacement, spectral=np.ones(2, dtype=np.float32))
    size, sha256 = _identity(tile)
    real_stat = inference.os.stat
    swapped = False

    def swap_then_stat(path, *, follow_symlinks=True):
        nonlocal swapped
        if Path(path) == tile and not swapped:
            tile.rename(original)
            tile.symlink_to(replacement)
            swapped = True
        return real_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(inference.os, "stat", swap_then_stat)
    with pytest.raises(inference.TileIdentityError, match="path changed while open"):
        inference._load_npz_for_inference(
            tile, expected_size=size, expected_sha256=sha256
        )


def test_post_fstat_detects_same_inode_mutation(tmp_path: Path) -> None:
    tile = tmp_path / "tile.bin"
    tile.write_bytes(b"before")
    fd, opened = inference._open_regular_nofollow(
        tile,
        error_type=inference.TileIdentityError,
        label="frozen tile",
    )
    try:
        tile.write_bytes(b"after!")
        with pytest.raises(inference.TileIdentityError, match="changed while being read"):
            inference._assert_open_file_unchanged(
                fd,
                tile,
                opened,
                error_type=inference.TileIdentityError,
                label="frozen tile",
            )
    finally:
        os.close(fd)


def test_object_array_cannot_trigger_pickle_in_authenticated_or_legacy_mode(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "pickle-executed"

    class Exploit:
        def __reduce__(self):
            return _write_marker, (str(marker),)

    tile = tmp_path / "object.npz"
    np.savez(tile, spectral=np.array([Exploit()], dtype=object))
    size, sha256 = _identity(tile)

    for kwargs in (
        {},
        {"expected_size": size, "expected_sha256": sha256},
    ):
        data, buffer = inference._load_npz_for_inference(tile, **kwargs)
        try:
            with pytest.raises(ValueError, match="allow_pickle=False"):
                data["spectral"]
        finally:
            data.close()
            if buffer is not None:
                buffer.close()
        assert not marker.exists()


def test_tile_identity_is_an_all_or_nothing_pair(tmp_path: Path) -> None:
    tile = tmp_path / "tile.npz"
    np.savez(tile, spectral=np.zeros(2, dtype=np.float32))

    with pytest.raises(inference.TileIdentityError, match="supplied together"):
        inference._load_npz_for_inference(
            tile, expected_sha256="0" * 64
        )
