"""Tests for the LUCAS validation scoring core (scripts/validate_against_lucas.py).

Real inference (sliding-window) is ICE-verified via the reused
``validate_against_nfi`` wiring; here we exercise the pure LUCAS scoring logic
with mock predict_fns:

  * crop year-match assertion fires on a wrong-year crop point (fail-loud),
  * year-stable (land-cover / forest) classes are NOT subject to the strict
    year filter,
  * L2b hard 28-class three-split breakdown + min-support gating,
  * L2a dominant-species argmax agreement + mixedness ROC-AUC on a tiny mock.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "validate_against_lucas",
    str(Path(__file__).resolve().parents[1] / "scripts" / "validate_against_lucas.py"),
)
val = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(val)


# ── helpers ───────────────────────────────────────────────────────────────────
def _pt(point_id, cls, row, col, *, tile="tileA", year=2022, split="train",
        dom=None, mixed=False, source="eu2022"):
    return {
        "point_id": point_id, "unified_class": cls,
        "unified_name": val.class_name(cls),
        "tile_name": tile, "tile_path": tile, "row": row, "col": col,
        "Year": year, "split": split, "source": source,
        "forest_dominant": dom, "is_mixed": mixed,
    }


# ── crop year-match assertion ─────────────────────────────────────────────────
def test_crop_year_match_fires_on_mismatch():
    df = pd.DataFrame([
        _pt("p1", 11, 0, 0, tile="tileA", year=2022),  # vete, obs 2022
    ])
    # tile spectral year 2021 ≠ point year 2022 → must raise
    with pytest.raises(ValueError, match="CROP YEAR-MATCH VIOLATION"):
        val.assert_crop_year_match(df, {"tileA": 2021})


def test_crop_year_match_fires_on_undeterminable_year():
    df = pd.DataFrame([_pt("p1", 12, 0, 0, tile="tileA", year=2022)])  # korn
    with pytest.raises(ValueError, match="CROP YEAR-MATCH VIOLATION"):
        val.assert_crop_year_match(df, {"tileA": None})


def test_crop_year_match_passes_on_exact_year():
    df = pd.DataFrame([_pt("p1", 11, 0, 0, tile="tileA", year=2022)])
    val.assert_crop_year_match(df, {"tileA": 2022})  # no raise


def test_year_stable_classes_not_year_filtered():
    """A forest point on a wrong-year tile must NOT raise — year-robust."""
    df = pd.DataFrame([
        _pt("f1", 1, 0, 0, tile="tileA", year=2022, dom="tall"),   # tallskog
        _pt("w1", 10, 1, 1, tile="tileA", year=2022),              # vatten
    ])
    # tile year 2018 ≠ 2022 but these are year-stable → assertion is a no-op
    val.assert_crop_year_match(df, {"tileA": 2018})


# ── L2b hard 28-class + splits + min-support ─────────────────────────────────
def _mock_hard_predict(_tile_path):
    """8×8 class map; place predictions at the (row,col) the mock index uses."""
    cm = np.zeros((8, 8), dtype=np.int64)
    # forest points predicted correctly, one water wrong
    cm[0, 0] = 1   # tallskog truth 1 → 1 ✓
    cm[1, 1] = 2   # granskog truth 2 → 2 ✓
    cm[2, 2] = 3   # lövskog  truth 3 → 3 ✓
    cm[3, 3] = 4   # blandskog truth 4 → 4 ✓
    cm[4, 4] = 8   # vatten truth 10 → 8 ✗
    probs = np.zeros((28, 8, 8), dtype=np.float32)
    return cm, probs


def _l2b_index():
    rows = []
    # 25 tallskog (≥ min_support 20) all correct → producer acc 1.0
    for i in range(25):
        rows.append(_pt(f"t{i}", 1, 0, 0, dom="tall",
                        split="test" if i < 5 else "train"))
    # 3 vatten (< min_support) predicted wrong (class 8) → insufficient support
    for i in range(3):
        rows.append(_pt(f"w{i}", 10, 4, 4))
    return pd.DataFrame(rows)


def test_l2b_min_support_gating_and_splits():
    df = _l2b_index()
    tile_years = {"tileA": 2022}
    res = val.score_against_lucas(
        df, _mock_hard_predict, num_classes=28, min_support=20,
        tile_years=tile_years)
    l2b = res["l2b_hard_28class"]
    # tallskog scored (support 25 ≥ 20), producer accuracy 1.0
    assert l2b["all"]["per_class"]["tallskog"]["producers_accuracy"] == pytest.approx(1.0)
    assert l2b["all"]["per_class"]["tallskog"]["support"] == 25
    # vatten under floor → insufficient support, NOT a score
    assert l2b["all"]["per_class"]["vatten"]["status"] == "insufficient support"
    assert l2b["all"]["per_class"]["vatten"]["support"] == 3
    # three splits present, each independently scored
    assert set(l2b.keys()) >= {"all", "test", "train"}
    assert l2b["test"]["n_points"] == 5     # the 5 test-split tallskog
    assert l2b["train"]["n_points"] == 23   # 20 tallskog + 3 vatten
    # year-match note present, crop-group empty (no crops here)
    assert "year-matched" in res["_year_match_note"]
    assert res["_class_groups"]["year_strict_crops"] == []


def test_l2b_crop_year_note_records_years():
    df = pd.DataFrame(
        [_pt(f"v{i}", 11, 0, 0, year=2022) for i in range(20)]
    )
    res = val.score_against_lucas(
        df, _mock_hard_predict, num_classes=28, min_support=20,
        tile_years={"tileA": 2022})
    assert res["_class_groups"]["year_strict_crops"] == [11]
    assert res["_crop_point_years"]["vete"] == [2022]


def test_l2b_assertion_blocks_scoring_on_bad_crop_year():
    df = pd.DataFrame([_pt("v0", 11, 0, 0, year=2022)])
    with pytest.raises(ValueError, match="CROP YEAR-MATCH VIOLATION"):
        val.score_against_lucas(
            df, _mock_hard_predict, num_classes=28, tile_years={"tileA": 2021})


# ── L2a fraction: dominant-argmax + mixedness AUC ────────────────────────────
def _mock_frac_predict(_tile_path):
    """Return (class_map, probs, fracs) with fracs designed so:
       (0,0) tall-dominant, (1,1) gran-dominant, (2,2) löv-dominant,
       (3,3) evenly mixed (high dispersion)."""
    cm = np.zeros((8, 8), dtype=np.int64)
    probs = np.zeros((28, 8, 8), dtype=np.float32)
    fracs = np.zeros((4, 8, 8), dtype=np.float32)  # tall,gran,trivial,adel
    # tall dominant
    fracs[:, 0, 0] = [0.8, 0.1, 0.05, 0.05]
    # gran dominant
    fracs[:, 1, 1] = [0.1, 0.8, 0.05, 0.05]
    # löv dominant (trivial+adel)
    fracs[:, 2, 2] = [0.1, 0.1, 0.4, 0.4]
    # evenly mixed → high dispersion
    fracs[:, 3, 3] = [0.3, 0.3, 0.2, 0.2]
    return cm, probs, fracs


def _l2a_index():
    return pd.DataFrame([
        _pt("d_tall", 1, 0, 0, dom="tall", mixed=False),
        _pt("d_gran", 2, 1, 1, dom="gran", mixed=False),
        _pt("d_lov", 3, 2, 2, dom="lov", mixed=False),
        _pt("mixed", 4, 3, 3, dom=None, mixed=True),
    ])


def test_l2a_dominant_argmax_and_mixedness_auc():
    df = _l2a_index()
    res = val.score_against_lucas(
        df, _mock_frac_predict, num_classes=28, min_support=1,
        is_fraction=True, tile_years={"tileA": 2022})
    l2a = res["l2a_forest_fraction"]
    # all 3 dominated points argmax-correct
    assert l2a["dominant_argmax"]["n_dominated"] == 3
    assert l2a["dominant_argmax"]["agreement"] == pytest.approx(1.0)
    assert l2a["dominant_argmax"]["per_species"]["tall"]["agreement"] == pytest.approx(1.0)
    # mixedness: the mixed point has the highest dispersion → perfect AUC
    assert l2a["mixedness_auc"]["n_mixed"] == 1
    assert l2a["mixedness_auc"]["roc_auc"] == pytest.approx(1.0)


def test_l2a_absent_forest_returns_status():
    df = pd.DataFrame([_pt("w", 10, 0, 0)])  # only water, no forest
    res = val.score_against_lucas(
        df, _mock_frac_predict, num_classes=28, min_support=1,
        is_fraction=True, tile_years={"tileA": 2022})
    assert res["l2a_forest_fraction"]["status"] == "no forest points"


# ── per-point sink ────────────────────────────────────────────────────────────
def test_per_point_sink_carries_fracs():
    df = _l2a_index()
    sink: list = []
    val.score_against_lucas(
        df, _mock_frac_predict, num_classes=28, min_support=1,
        is_fraction=True, tile_years={"tileA": 2022}, per_point_sink=sink)
    assert len(sink) == 4
    out = pd.DataFrame(sink)
    assert {"point_id", "unified_class", "pred_class", "split", "Year",
            "source", "frac_tall", "frac_gran", "frac_trivial",
            "frac_adel"} <= set(out.columns)
    # tall-dominant point carries its fraction
    assert out.loc[out.point_id == "d_tall", "frac_tall"].iloc[0] == pytest.approx(0.8)
