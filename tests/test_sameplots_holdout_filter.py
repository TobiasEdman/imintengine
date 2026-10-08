"""Frozen plot-year support replaces the unsafe tile-only test filter."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_nmd2023_nfi import _restrict_to_holdout
from imint.eval.fieldtruth import sha256_file


def freeze_fixture(tmp_path):
    plots = pd.DataFrame({
        "TractID": [1, 1, 2], "PlotID": [1, 1, 1],
        "Year": [2022, 2022, 2024], "tile_name": ["train", "test", "campaign"],
        "nfi_forest": [1, 1, 2], "model_pred": [1, 3, 2],
    })
    held = plots.iloc[[2]].copy()
    train = plots.iloc[[0]][["TractID", "PlotID", "Year"]]
    held.to_parquet(tmp_path / "holdout.parquet", index=False)
    train.to_parquet(tmp_path / "training.parquet", index=False)
    manifest = {
        "schema": "nfi-fieldtruth-freeze-v1", "campaign": {"training_tiles": 0},
        **{k: {"file": f"{k}.parquet", "observations": 1,
               "sha256": sha256_file(tmp_path / f"{k}.parquet")}
           for k in ("holdout", "training")},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return plots, held, path


def test_keeps_frozen_observation_and_preserves_truth_alignment(tmp_path):
    plots, _, path = freeze_fixture(tmp_path)
    kept, truth, meta = _restrict_to_holdout(
        plots, plots["nfi_forest"].to_numpy(), str(path))
    assert kept["tile_name"].tolist() == ["campaign"]
    assert truth.tolist() == [2]
    assert meta["n_plots_before"] == 3 and meta["n_plots_after"] == 1
    assert meta["manifest_sha256"] == sha256_file(path)
    assert meta["identity"] == ["TractID", "PlotID", "Year"]


def test_changed_holdout_content_is_rejected(tmp_path):
    plots, held, path = freeze_fixture(tmp_path)
    held.assign(Year=2023).to_parquet(tmp_path / "holdout.parquet")
    with pytest.raises(ValueError, match="content changed"):
        _restrict_to_holdout(plots, np.ones(3), str(path))


def test_missing_selected_tile_is_not_replaced_by_other_tile(tmp_path):
    plots, _, path = freeze_fixture(tmp_path)
    plots.loc[2, "tile_name"] = "other"
    with pytest.raises(ValueError, match="lacks frozen"):
        _restrict_to_holdout(plots, np.array([1, 1, 2]), str(path))


def test_duplicate_prediction_rows_fail_instead_of_double_weighting(tmp_path):
    plots, _, path = freeze_fixture(tmp_path)
    plots = pd.concat([plots, plots.iloc[[2]]])
    with pytest.raises(ValueError, match="duplicate"):
        _restrict_to_holdout(plots, np.array([1, 1, 2, 2]), str(path))


def test_conflicting_frozen_truth_fails(tmp_path):
    plots, _, path = freeze_fixture(tmp_path)
    with pytest.raises(ValueError, match="truth differs"):
        _restrict_to_holdout(plots, np.array([1, 1, 4]), str(path))


def test_forged_holdout_still_cannot_include_training_identity(tmp_path):
    plots, held, path = freeze_fixture(tmp_path)
    train = held[["TractID", "PlotID", "Year"]]
    train.to_parquet(tmp_path / "training.parquet", index=False)
    manifest = json.loads(path.read_text())
    manifest["training"]["sha256"] = sha256_file(tmp_path / "training.parquet")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="training observation leaked"):
        _restrict_to_holdout(plots, np.array([1, 1, 2]), str(path))
