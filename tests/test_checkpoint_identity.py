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


def test_checkpoint_with_numpy_scalar_metrics_loads(tmp_path: Path) -> None:
    """Training checkpoints pickle metrics as numpy scalars; the narrow
    allowlist must admit them without widening beyond numpy data types.
    Regression: all six crop-distill jobs failed at extract-features on
    ``numpy._core.multiarray.scalar`` (2026-09-07)."""
    np = pytest.importorskip("numpy")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "epoch": np.int64(30),
            "best_miou": np.float64(0.7123),
            "model_state_dict": {"w": torch.zeros(2)},
        },
        checkpoint,
    )
    size, sha256 = _identity(checkpoint)

    payload = inference._load_checkpoint_for_inference(
        checkpoint,
        expected_size=size,
        expected_sha256=sha256,
    )

    assert int(payload["epoch"]) == 30
    assert float(payload["best_miou"]) == pytest.approx(0.7123)


def test_numpy_safe_globals_via_module_import_not_attribute_chain() -> None:
    """The scalar reconstructor must come from a real module IMPORT.
    Reaching it as an attribute chain (np._core.multiarray) depends on the
    numpy version and on which submodules other libraries imported first —
    the 4f distill job died on exactly that under numpy 1.26.4 while
    numpy 2.x environments passed (2026-09-08)."""
    import importlib

    src = inspect.getsource(inference._numpy_metric_safe_globals)
    assert "import_module" in src
    assert 'getattr(np, "_core"' not in src, "attribute-chain access is back"

    try:
        expected = importlib.import_module("numpy._core.multiarray").scalar
    except ImportError:
        expected = importlib.import_module("numpy.core.multiarray").scalar
    first_obj, first_name = inference._numpy_metric_safe_globals()[0]
    assert first_obj is expected
    names = [n for _, n in inference._numpy_metric_safe_globals()[:2]]
    # BOTH pickle spellings must be present when importable — the fleet
    # carries checkpoints saved under numpy 1.x AND 2.x (2026-09-08).
    assert "numpy._core.multiarray.scalar" in names or \
        "numpy.core.multiarray.scalar" in names


def test_numpy_safe_globals_1x_fallback_sequence(monkeypatch) -> None:
    """Force the numpy-1.x path regardless of the installed numpy: when
    numpy._core.multiarray is unimportable the loader must fall back to
    numpy.core.multiarray — asserted by intercepting the imports, not by
    hoping CI happens to run 1.26.4 (PR #45 review, MEDIUM)."""
    import importlib

    calls: list[str] = []
    real = importlib.import_module

    def fake(name, *a, **k):
        calls.append(name)
        if name == "numpy._core.multiarray":
            raise ImportError("forced 1.x environment")
        if name == "numpy.core.multiarray":
            return real("numpy._core.multiarray") if _np2() else real(name)
        return real(name, *a, **k)

    def _np2() -> bool:
        import numpy as np
        return int(np.__version__.split(".")[0]) >= 2

    monkeypatch.setattr(importlib, "import_module", fake)
    result = inference._numpy_metric_safe_globals()
    assert calls[0] == "numpy._core.multiarray"
    assert "numpy.core.multiarray" in calls, "1.x spelling never attempted"
    obj, name = result[0]
    assert callable(obj) and name == "numpy.core.multiarray.scalar"


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
