"""The same-plots comparison must be able to score only held-out plots.

Without the restriction the producer scores the full NFI cohort, which for a
distilled model includes the plots whose labels trained its teacher — the
overlap that made an earlier dashboard call a full-cohort diagnostic a
held-out test. These tests pin the restriction, its provenance record, and
every way it is allowed to refuse.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_nmd2023_nfi import _restrict_to_holdout  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _dump() -> pd.DataFrame:
    """Six plots over four tiles; two of the tiles are the held-out ones."""
    return pd.DataFrame({
        "tile_name": ["tile_a", "tile_a", "tile_b", "tile_c", "tile_d", "tile_d"],
        "TractID": [1, 1, 2, 3, 4, 4],
        "PlotID": [1, 2, 1, 1, 1, 2],
        "nfi_forest": [1, 2, 0, 3, 4, -1],
        "model_pred": [1, 2, 0, 3, 4, 0],
    })


def _split(tmp_path: Path, test_tiles: list[str], key: str = "test_tiles") -> Path:
    p = tmp_path / "split.json"
    p.write_text(json.dumps({key: test_tiles, "train_tiles": ["tile_a", "tile_b"]}))
    return p


def test_keeps_only_test_tile_rows_and_aligns_truth(tmp_path):
    plots = _dump()
    truth = np.arange(len(plots))
    split = _split(tmp_path, ["tile_c", "tile_d"])

    kept, kept_truth, holdout = _restrict_to_holdout(plots, truth, str(split))

    assert list(kept["tile_name"]) == ["tile_c", "tile_d", "tile_d"]
    # Truth must travel with its rows, not be re-derived or re-indexed.
    assert list(kept_truth) == [3, 4, 5]
    assert holdout["n_plots_before"] == 6
    assert holdout["n_plots_after"] == 3
    assert holdout["n_test_tiles"] == 2
    assert holdout["n_test_tiles_present"] == 2
    assert kept.index.tolist() == [0, 1, 2], "index must be reset for downstream"


def test_records_split_identity(tmp_path):
    split = _split(tmp_path, ["tile_c"])
    _, _, holdout = _restrict_to_holdout(_dump(), np.zeros(6, int), str(split))

    assert holdout["split_json"] == str(split)
    assert holdout["split_sha256"] == hashlib.sha256(split.read_bytes()).hexdigest()


def test_test_tiles_absent_from_dump_raises(tmp_path):
    """The wrong split file for this cohort must not score an empty set."""
    split = _split(tmp_path, ["holdoutval_001", "holdoutval_002"])
    with pytest.raises(SystemExit, match="none of which appear"):
        _restrict_to_holdout(_dump(), np.zeros(6, int), str(split))


def test_partial_presence_is_allowed_and_counted(tmp_path):
    """Some test tiles missing is a coverage fact, not a fault — but recorded."""
    split = _split(tmp_path, ["tile_c", "tile_absent"])
    kept, _, holdout = _restrict_to_holdout(_dump(), np.zeros(6, int), str(split))

    assert len(kept) == 1
    assert holdout["n_test_tiles"] == 2
    assert holdout["n_test_tiles_present"] == 1


def test_missing_test_tiles_key_raises(tmp_path):
    split = _split(tmp_path, ["tile_c"], key="holdout_tiles")
    with pytest.raises(SystemExit, match="no 'test_tiles' key"):
        _restrict_to_holdout(_dump(), np.zeros(6, int), str(split))


def test_dump_without_tile_name_raises(tmp_path):
    plots = _dump().drop(columns=["tile_name"])
    split = _split(tmp_path, ["tile_c"])
    with pytest.raises(SystemExit, match="no tile_name column"):
        _restrict_to_holdout(plots, np.zeros(6, int), str(split))


def test_split_json_without_model_dump_is_refused(tmp_path):
    """nfi_plots.parquet has no tile_name, so the restriction cannot apply.

    Exercised through the CLI because that is where the two flags meet; a
    silent full-cohort run under a --split-json the user passed would be the
    exact mislabelling this change exists to prevent.
    """
    plots = tmp_path / "plots.parquet"
    pd.DataFrame({"Easting": [0.0], "Northing": [0.0]}).to_parquet(plots)
    split = _split(tmp_path, ["tile_c"])

    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/compare_nmd2023_nfi.py"),
         "--nmd2023", str(tmp_path / "absent.tif"),
         "--plots", str(plots),
         "--split-json", str(split),
         "--out", str(tmp_path / "out.json")],
        capture_output=True, text=True, timeout=300,
    )

    assert proc.returncode != 0
    assert "--split-json needs --model-per-plot" in proc.stderr
    assert not (tmp_path / "out.json").exists()
