"""The label builder must never unpickle shared-PVC tile data.

It runs as root in a pod mounting the full PVC: an object-array member in
any tile NPZ would execute arbitrary pickle at load. PR #45 review (HIGH).
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def _write_marker(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


def test_builder_source_forbids_pickle_loading() -> None:
    src = (REPO / "scripts" / "build_labels.py").read_text()
    assert "allow_pickle=True" not in src
    assert "allow_pickle=False" in src


def test_object_array_tile_fails_without_executing(tmp_path: Path) -> None:
    """A malicious tile must surface as a per-tile failure — visible in the
    run summary — with the payload never executed."""
    import build_labels

    marker = tmp_path / "pickle-executed"

    class Exploit:
        def __reduce__(self):
            return _write_marker, (str(marker),)

    tile = tmp_path / "holdoutval_9_9_2022.npz"
    np.savez(tile,
             spectral=np.zeros((24, 8, 8), dtype=np.float32),
             easting=np.float64(500000.0), northing=np.float64(6500000.0),
             payload=np.array([Exploit()], dtype=object))

    sig = inspect.signature(build_labels.build_tile_label)
    kwargs = dict(tile_path=str(tile),
                  nmd_raster=str(tmp_path / "missing.tif"),
                  lpis_dir=str(tmp_path), sks_dir=str(tmp_path))
    if "label_out_dir" in sig.parameters:
        kwargs["label_out_dir"] = str(tmp_path / "out")

    try:
        result = build_labels.build_tile_label(**kwargs)
    except ValueError:
        result = {"status": "failed"}
    assert result.get("status") == "failed"
    assert not marker.exists(), "pickle payload executed — unsafe load is back"
