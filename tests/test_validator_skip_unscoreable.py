"""The opt-in skip for tiles no backbone can score.

28 of 7,882 training tiles have no Sentinel-1 composite available at all
(``no_composite_ASCENDING/DESCENDING`` from Planetary Computer). That is a
data-coverage limit, not a fixable state. Without an opt-out they cost croma
and terramind every cell — 8 of 28 — rather than the 0.43% of LUCAS points
and 0.92% of NFI plots those tiles actually carry.

Raising stays the DEFAULT: the contract exists so a mis-composited SAR stack
is never scored silently. Declining to score a tile feeds nothing; it declares
a gap, and the gap is recorded in the output.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_{name}", str(ROOT / "scripts" / f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


lucas = _load("validate_against_lucas")
nfi = _load("validate_against_nfi")

BAD = "44883674"
from imint.training.unified_dataset import TilePrerequisiteError  # noqa: E402

S1_ERR = TilePrerequisiteError(
    "tile requires s1_enrich_v==4 but found s1_enrich_v=0")


def _lucas_index():
    return pd.DataFrame({
        "tile_name": ["good", "good", BAD],
        "tile_path": ["good.npz", "good.npz", f"{BAD}.npz"],
        "row": [0, 1, 0], "col": [0, 1, 0],
        "point_id": [1, 2, 3],
        "unified_class": [11, 12, 13],
        "unified_name": ["vete", "korn", "havre"],
        "split": ["test"] * 3, "Year": [2022] * 3,
        "source": ["lucas"] * 3, "forest_dominant": [None] * 3,
    })


def _predict(tile_path):
    if tile_path.startswith(BAD):
        raise S1_ERR
    cm = np.full((4, 4), 11, dtype=int)
    cm[1, 1] = 12
    return cm, np.zeros((28, 4, 4))


def test_lucas_raises_by_default():
    with pytest.raises(KeyError, match="s1_enrich_v"):
        lucas.score_against_lucas(_lucas_index(), _predict, min_support=1)


def test_lucas_skip_records_the_gap_and_scores_the_rest():
    res = lucas.score_against_lucas(_lucas_index(), _predict, min_support=1,
                                    skip_unscoreable=True)
    assert [s["tile"] for s in res["skipped_tiles"]] == [BAD]
    assert res["skipped_tiles"][0]["points"] == 1
    assert "s1_enrich_v" in res["skipped_tiles"][0]["reason"]


def _nfi_index():
    return pd.DataFrame({
        "tile_name": ["good", BAD],
        "tile_path": ["good.npz", f"{BAD}.npz"],
        "row": [0, 0], "col": [0, 0],
        "TractID": [1, 2], "PlotID": [1, 1], "Year": [2022, 2022],
        "Easting": [0.0, 1.0], "Northing": [0.0, 1.0],
        # derive_nfi_forest_class reads the per-species volumes
        "VolPine": [100.0, 100.0], "VolSpruce": [0.0, 0.0],
        "VolBirch": [0.0, 0.0], "VolContorta": [0.0, 0.0],
        "VolOtherDec": [0.0, 0.0],
    })


def test_nfi_raises_by_default():
    with pytest.raises(KeyError, match="s1_enrich_v"):
        nfi.score_against_nfi(_nfi_index(), _predict)


def test_nfi_skip_records_the_gap():
    res = nfi.score_against_nfi(_nfi_index(), _predict, skip_unscoreable=True)
    assert [s["tile"] for s in res["skipped_tiles"]] == [BAD]
    assert res["skipped_tiles"][0]["plots"] == 1


# --- the skip must not become a catch-all -------------------------------

def _raiser(exc):
    def f(tile_path):
        raise exc
    return f


@pytest.mark.parametrize("exc", [
    KeyError("enabled_aux_names"),                       # config fault
    ValueError("checkpoint config lists 1 aux but n_aux_channels=2"),
    RuntimeError("CUDA out of memory"),
])
def test_unrelated_failures_still_propagate_lucas(exc):
    """Only TilePrerequisiteError is eligible.

    Regression: catching bare KeyError/ValueError converted a checkpoint
    contract failure — which raises before any tile is read — into a
    "coverage gap" for every tile, and returned an empty result that looked
    like a successful run.
    """
    with pytest.raises(type(exc)):
        lucas.score_against_lucas(_lucas_index(), _raiser(exc),
                                  min_support=1, skip_unscoreable=True)


@pytest.mark.parametrize("exc", [
    KeyError("enabled_aux_names"),
    ValueError("checkpoint config lists 1 aux but n_aux_channels=2"),
])
def test_unrelated_failures_still_propagate_nfi(exc):
    with pytest.raises(type(exc)):
        nfi.score_against_nfi(_nfi_index(), _raiser(exc), skip_unscoreable=True)


def test_all_tiles_skipped_fails_loudly_lucas():
    """A run that measured nothing must not be written as one that measured 0."""
    with pytest.raises(SystemExit, match="no point could be scored"):
        lucas.score_against_lucas(_lucas_index(), _raiser(S1_ERR),
                                  min_support=1, skip_unscoreable=True)


def test_all_tiles_skipped_fails_loudly_nfi():
    with pytest.raises(SystemExit, match="no plot could be scored"):
        nfi.score_against_nfi(_nfi_index(), _raiser(S1_ERR), skip_unscoreable=True)
