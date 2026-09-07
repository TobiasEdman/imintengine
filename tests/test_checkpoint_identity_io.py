"""Checkpoint descriptor integrity tests that do not require a Torch install."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inference_comparison.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_checkpoint_identity_io", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inference = _load_module()


def _identity(path: Path) -> tuple[int, str]:
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _fake_torch(monkeypatch: pytest.MonkeyPatch, expected: bytes):
    module = types.ModuleType("torch")
    calls: list[dict] = []

    def load(source, **kwargs):
        assert source.tell() == 0
        assert os.fstat(source.fileno()).st_size == len(expected)
        calls.append(kwargs)
        return {"payload": source.read()}

    module.load = load
    monkeypatch.setitem(sys.modules, "torch", module)
    return calls


def test_checkpoint_hashes_shared_fd_and_loads_private_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    payload = b"safe-checkpoint-fixture"
    checkpoint.write_bytes(payload)
    size, sha256 = _identity(checkpoint)
    calls = _fake_torch(monkeypatch, payload)

    loaded = inference._load_checkpoint_for_inference(
        checkpoint,
        expected_size=size,
        expected_sha256=sha256,
    )

    assert loaded == {"payload": payload}
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_shared_checkpoint_mutation_before_load_cannot_change_private_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    safe = b"safe-checkpoint-fixture"
    malicious = b"evil-checkpoint-fixture"
    assert len(safe) == len(malicious)
    checkpoint.write_bytes(safe)
    size, sha256 = _identity(checkpoint)
    module = types.ModuleType("torch")

    def load(source, **_kwargs):
        checkpoint.write_bytes(malicious)
        source.seek(0)
        return {"payload": source.read()}

    module.load = load
    monkeypatch.setitem(sys.modules, "torch", module)

    loaded = inference._load_checkpoint_for_inference(
        checkpoint,
        expected_size=size,
        expected_sha256=sha256,
    )

    assert checkpoint.read_bytes() == malicious
    assert loaded == {"payload": safe}


def test_checkpoint_links_are_rejected_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    symlink = tmp_path / "symlink.pt"
    hardlink = tmp_path / "hardlink.pt"
    payload = b"safe-checkpoint-fixture"
    checkpoint.write_bytes(payload)
    size, sha256 = _identity(checkpoint)
    calls = _fake_torch(monkeypatch, payload)

    symlink.symlink_to(checkpoint)
    with pytest.raises(
        inference.CheckpointIdentityError,
        match="without following links",
    ):
        inference._load_checkpoint_for_inference(
            symlink,
            expected_size=size,
            expected_sha256=sha256,
        )

    os.link(checkpoint, hardlink)
    with pytest.raises(
        inference.CheckpointIdentityError,
        match="exactly one hard link",
    ):
        inference._load_checkpoint_for_inference(
            checkpoint,
            expected_size=size,
            expected_sha256=sha256,
        )
    assert calls == []


def test_checkpoint_loader_contains_no_pickle_fallback() -> None:
    source = inspect.getsource(inference._load_checkpoint_for_inference)
    assert "weights_only=True" in source
    assert "weights_only=False" not in source
    assert "tempfile.TemporaryFile" in source
