"""CPU preparation gates: no mutable inputs or teacher-trained plot-years."""
import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_nfi_holdout as prep
from imint.eval.fieldtruth import resolve_observation_year


def promotion():
    return ({"metadata": {"name": "tessera-promote-v2"},
             "status": {"conditions": [{"type": "Complete", "status": "True"}]}},
            {"source": "geotessera-0.10.2",
             "verify": {"has_v1": 7882, "stamped": 7882, "has_v2_left": 0}})


def test_terminal_clean_promotion_is_required():
    job, report = promotion()
    prep.check_promotion(job, report, 7882)
    job["status"]["active"] = 1
    with pytest.raises(ValueError, match="terminal"):
        prep.check_promotion(job, report, 7882)


@pytest.mark.parametrize("mutation", [
    {"FLAG_ON_EMPTY": 1}, {"has_v2_left": 1}, {"stamped": 7800}, {"has_v1": 0},
])
def test_dirty_or_partial_promotion_cannot_be_frozen(mutation):
    job, report = promotion()
    report["verify"].update(mutation)
    with pytest.raises(ValueError, match="clean v2"):
        prep.check_promotion(job, report, 7882)


def index():
    return pd.DataFrame({
        "TractID": [1, 1, 1, 2, 2], "PlotID": [1] * 5,
        "Year": [2022, 2022, 2024, 2024, 2024],
        "tile_name": ["train", "test", "campaign", "campaign", "overlap"],
        "row": [20] * 5, "col": [20] * 5,
    })


def test_cross_tile_training_leak_excluded_but_different_year_kept():
    source = index()
    trained = source.iloc[[0]][prep.NFI_KEY]
    meta = {n: {"height": 512, "width": 512, "year": y}
            for n, y in [("test", 2022), ("train", 2022),
                         ("campaign", 2024), ("overlap", 2024)]}
    held, report = prep.select_holdout(source, trained, meta)
    assert held[prep.NFI_KEY].values.tolist() == [[1, 1, 2024], [2, 1, 2024]]
    assert held["tile_name"].tolist() == ["campaign", "campaign"]
    assert report["training_overlap_rows"] == 2
    assert report["duplicate_rows_removed"] == 1
    assert report["training_observation_overlap"] == 0


def test_teacher_year_recovery_refuses_ambiguous_year():
    source = index()
    feature = source.iloc[[0]].drop(columns="Year")
    assert resolve_observation_year(feature, source)["Year"].tolist() == [2022]
    ambiguous = pd.concat([source, source.iloc[[0]].assign(Year=2024)])
    with pytest.raises(ValueError, match="ambiguous Year"):
        resolve_observation_year(feature, ambiguous)


def test_staging_cannot_join_training():
    source = index().iloc[[0, 2]]
    split = {"train_tiles": ["campaign"], "test_tiles": ["train"],
             "n_train_plots": 1, "n_test_plots": 1}
    with pytest.raises(ValueError, match="evaluation-only"):
        prep.teacher_training_set(source, split, source, {"campaign"})


def test_teacher_training_count_mismatch_fails():
    source = index().iloc[[0, 2]]
    split = {"train_tiles": ["train"], "test_tiles": ["campaign"],
             "n_train_plots": 2, "n_test_plots": 1}
    with pytest.raises(ValueError, match="row counts"):
        prep.teacher_training_set(source, split, source, {"campaign"})


def test_empty_tessera_positive_flag_fails_on_actual_npz(tmp_path):
    path = tmp_path / "tile.npz"
    np.savez(path, spectral=np.ones((6, 8, 8)), tessera=np.zeros((128, 8, 8)),
             has_tessera=1, tessera_source="geotessera-0.10.2",
             s1_vv_vh=np.ones((2, 8, 8)), s1_enrich_v=4, has_s1=1,
             b08=np.ones((8, 8)), rededge=np.ones((3, 8, 8)), year=2024)
    with pytest.raises(ValueError, match="positive has_tessera"):
        prep.tile_readiness(path)


def test_non_integral_teacher_year_is_rejected():
    source = index().iloc[[0]].assign(Year=2024.5)
    with pytest.raises(ValueError, match="non-integer"):
        resolve_observation_year(source.drop(columns="Year"), source)


def test_string_identities_cannot_bypass_numeric_training_identity():
    from imint.eval.fieldtruth import exclude_training_observations
    source = index().iloc[[0]]
    trained = source[prep.NFI_KEY].astype(str)
    with pytest.raises(ValueError, match="non-numeric"):
        exclude_training_observations(source, trained)
