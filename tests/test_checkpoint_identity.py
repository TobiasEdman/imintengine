"""Authenticate checkpoints on the descriptor used by safe deserialization."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inference_comparison.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_checkpoint_identity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inference = _load_module()


def _identity(path: Path) -> tuple[int, str]:
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _write_marker(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


def test_checkpoint_is_hashed_and_loaded_from_one_safe_descriptor(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"config": {"num_classes": 23}, "epoch": 7}, checkpoint)
    size, sha256 = _identity(checkpoint)

    payload = inference._load_checkpoint_for_inference(
        checkpoint,
        expected_size=size,
        expected_sha256=sha256,
    )

    assert payload == {"config": {"num_classes": 23}, "epoch": 7}


def test_checkpoint_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    symlink = tmp_path / "symlink.pt"
    hardlink = tmp_path / "hardlink.pt"
    torch.save({"config": {}}, checkpoint)
    size, sha256 = _identity(checkpoint)

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


def test_checkpoint_rejects_size_digest_and_partial_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"config": {}}, checkpoint)
    size, sha256 = _identity(checkpoint)

    with pytest.raises(inference.CheckpointIdentityError, match="size mismatch"):
        inference._load_checkpoint_for_inference(
            checkpoint,
            expected_size=size + 1,
            expected_sha256=sha256,
        )
    with pytest.raises(inference.CheckpointIdentityError, match="sha256 mismatch"):
        inference._load_checkpoint_for_inference(
            checkpoint,
            expected_size=size,
            expected_sha256="0" * 64,
        )
    with pytest.raises(inference.CheckpointIdentityError, match="supplied together"):
        inference._load_checkpoint_for_inference(
            checkpoint,
            expected_size=size,
        )


def test_checkpoint_never_falls_back_to_unsafe_pickle(tmp_path: Path) -> None:
    marker = tmp_path / "pickle-executed"

    class Exploit:
        def __reduce__(self):
            return _write_marker, (str(marker),)

    checkpoint = tmp_path / "unsafe.pt"
    torch.save({"config": {}, "payload": Exploit()}, checkpoint)
    size, sha256 = _identity(checkpoint)

    with pytest.raises(Exception, match="Weights only load failed"):
        inference._load_checkpoint_for_inference(
            checkpoint,
            expected_size=size,
            expected_sha256=sha256,
        )
    assert not marker.exists()

    source = inspect.getsource(inference._load_checkpoint_for_inference)
    assert "weights_only=True" in source
    assert "weights_only=False" not in source
