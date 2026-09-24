"""The two fairness rules in the field-truth standings, pinned.

Both rules exist because the per-cell validators cannot enforce them: each
sees only its own cell, so neither the shared point set nor the shared class
vocabulary is visible from inside one run.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "_lfs", str(ROOT / "scripts" / "ladder_fieldtruth_standings.py"))
lfs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lfs)


def _lucas(points, truth, pred):
    return pd.DataFrame({"point_id": points, "unified_class": truth,
                         "pred_class": pred})


def test_rung1_is_not_charged_for_classes_it_cannot_emit():
    """Classes 24/27 are 28-class only; a 23-class model can never emit them.

    Regression: validate_against_lucas divides by len(truth) while the matrix
    only counts in-vocabulary pairs, so those points are pure loss for rung 1
    — 7.1% of LUCAS. Folding 24/27 onto open land lets rung 1's answer of 8
    count as the agreement it is, so both cells score 1.0 and no point is
    thrown away to get there.
    """
    pts = [1, 2, 3, 4]
    dumps = {
        # rung 1 cannot emit 24; it guesses 8 there.
        "clay_r1": _lucas(pts, [1, 10, 24, 27], [1, 10, 8, 8]),
        "clay_r2": _lucas(pts, [1, 10, 24, 27], [1, 10, 24, 27]),
    }
    shared = {r["cell"]: r for r in lfs.score_lucas(dumps, shared_only=True)}
    assert shared["clay_r1"]["overall"] == 1.0
    assert shared["clay_r2"]["overall"] == 1.0
    assert shared["clay_r1"]["n"] == 4          # folded, not discarded

    # Without the rule, rung 1 looks worse purely from vocabulary.
    full = {r["cell"]: r for r in lfs.score_lucas(dumps, shared_only=False)}
    assert full["clay_r1"]["overall"] == 0.5
    assert full["clay_r2"]["overall"] == 1.0


def test_columns_are_scored_on_the_intersection_of_kept_points():
    """Different img_size keeps different plots; scoring must not reward that.

    Cell B dropped the hard point 3 (border-cropped at its img_size). Scoring
    each cell on its own subset would give B 1.0 against A's 0.667 — a win
    bought by seeing fewer points, not by predicting better.
    """
    dumps = {
        "a_r2": _lucas([1, 2, 3], [5, 6, 7], [5, 6, 9]),   # wrong on 3
        "b_r2": _lucas([1, 2], [5, 6], [5, 6]),            # never saw 3
    }
    rows = {r["cell"]: r for r in lfs.score_lucas(dumps, shared_only=True)}
    assert rows["a_r2"]["n"] == rows["b_r2"]["n"] == 2     # point 3 dropped
    assert rows["a_r2"]["overall"] == 1.0                  # both perfect on shared
    assert rows["b_r2"]["overall"] == 1.0


def test_full_28class_table_omits_rung_one_entirely():
    dumps = {
        "clay_r1": _lucas([1], [24], [8]),
        "clay_r3": _lucas([1], [24], [24]),
    }
    cells = [c for c in dumps if not c.endswith("_r1")]
    rows = lfs.score_lucas(dumps, shared_only=False, cells=cells)
    assert [r["cell"] for r in rows] == ["clay_r3"]


def test_nfi_needs_no_vocabulary_correction():
    """Forest classes 1-4 exist in both vocabularies, so rung 1 competes."""
    key = {"TractID": [1, 2], "PlotID": [1, 1], "Year": [2022, 2022]}
    dumps = {
        "clay_r1": pd.DataFrame({**key, "nfi_forest": [1, 2], "model_pred": [1, 2]}),
        "clay_r2": pd.DataFrame({**key, "nfi_forest": [1, 2], "model_pred": [1, 3]}),
    }
    rows = {r["cell"]: r for r in lfs.score_nfi(dumps)}
    assert rows["clay_r1"]["overall"] == 1.0
    assert rows["clay_r1"]["overall"] > rows["clay_r2"]["overall"]
    assert rows["clay_r1"]["n"] == 2


def test_nfi_intersection_keeps_treeless_drops_absent():
    """Absent plots are dropped; treeless ones (-1) are kept and scored as 0."""
    dumps = {
        "a_r2": pd.DataFrame({"TractID": [1, 2, 3], "PlotID": [1, 1, 1],
                              "Year": [2022] * 3, "nfi_forest": [1, 2, -1],
                              "model_pred": [1, 2, 4]}),
        "b_r2": pd.DataFrame({"TractID": [1, 3], "PlotID": [1, 1],
                              "Year": [2022] * 2, "nfi_forest": [1, -1],
                              "model_pred": [1, 4]}),
    }
    rows = {r["cell"]: r for r in lfs.score_nfi(dumps)}
    # Tract 2 is missing from b, so it leaves the intersection. Tract 3 is
    # treeless, not missing, so it stays and is scored against class 0.
    assert rows["a_r2"]["n"] == rows["b_r2"]["n"] == 2


def test_empty_dump_dir_yields_no_rows(tmp_path):
    assert lfs.load_dumps(tmp_path, "nfi-per-plot") == {}
    assert lfs.score_lucas({}, shared_only=True) == []


def test_cell_name_is_parsed_from_the_dump_filename(tmp_path):
    df = _lucas([1], [5], [5])
    for name in ("lucas-per-point-tessera_r3.parquet",
                 "lucas-per-point-prithvi300m4f_r1.parquet"):
        df.to_parquet(tmp_path / name)
    got = lfs.load_dumps(tmp_path, "lucas-per-point")
    assert set(got) == {"tessera_r3", "prithvi300m4f_r1"}


def test_fine_open_land_classes_fold_onto_their_parent():
    """A 28-class model answering 24 where truth is 8 is not wrong.

    Regression: the first shared-vocabulary rule filtered only the truth, so a
    28-class model was charged for finer-but-correct answers while a 23-class
    model structurally could not make that mistake. Here both cells are right
    on every point, and the table must say so.
    """
    pts = [1, 2, 3, 4]
    dumps = {
        "clay_r1": _lucas(pts, [8, 8, 24, 27], [8, 8, 8, 8]),
        "clay_r2": _lucas(pts, [8, 8, 24, 27], [8, 24, 24, 27]),
    }
    rows = {r["cell"]: r for r in lfs.score_lucas(dumps, shared_only=True)}
    assert rows["clay_r1"]["overall"] == 1.0
    assert rows["clay_r2"]["overall"] == 1.0
    assert rows["clay_r1"]["n"] == rows["clay_r2"]["n"] == 4   # nothing dropped


def test_folding_does_not_forgive_a_genuinely_wrong_class():
    """Only 23-27 fold; a wrong forest or crop answer stays wrong."""
    dumps = {"a_r2": _lucas([1, 2], [3, 11], [24, 12])}
    rows = lfs.score_lucas(dumps, shared_only=True)
    assert rows[0]["overall"] == 0.0


def test_full_28class_table_still_charges_the_finer_answer():
    """The unfolded table keeps the strict reading, for rungs 2-4."""
    dumps = {"a_r2": _lucas([1, 2], [8, 8], [24, 8])}
    rows = lfs.score_lucas(dumps, shared_only=False)
    assert rows[0]["overall"] == 0.5


def test_a_point_in_two_tiles_is_scored_once_per_cell():
    """LUCAS edge points appear in every tile that contains them.

    Regression: 10,329 index rows cover 7,143 distinct points, and a cell's
    img_size decides which duplicate tiles survive its crop — so keeping all
    rows both double-weighted those points and gave each cell a different n
    (9,892 vs 9,976 in the first real run).
    """
    a = pd.DataFrame({"point_id": [1, 1, 2], "unified_class": [5, 5, 6],
                      "pred_class": [5, 9, 6]})          # point 1 twice
    b = pd.DataFrame({"point_id": [1, 2], "unified_class": [5, 6],
                      "pred_class": [5, 6]})             # point 1 once
    rows = {r["cell"]: r for r in lfs.score_lucas({"a_r2": a, "b_r2": b},
                                                  shared_only=True)}
    assert rows["a_r2"]["n"] == rows["b_r2"]["n"] == 2
    assert rows["a_r2"]["overall"] == 1.0    # the kept row for point 1 is right
    assert rows["b_r2"]["overall"] == 1.0


def test_treeless_plots_are_scored_not_dropped():
    """nfi_forest == -1 is the producer's treeless plot, mapped to class 0.

    Regression: filtering them forgave every false forest positive. Here one
    forest plot is predicted correctly and one treeless plot is wrongly called
    forest — the honest score is 0.5 over two plots, not 1.0 over one.
    """
    key = {"TractID": [1, 2], "PlotID": [1, 1], "Year": [2022, 2022]}
    dumps = {"a_r2": pd.DataFrame({**key, "nfi_forest": [1, -1],
                                   "model_pred": [1, 1]})}
    row = lfs.score_nfi(dumps)[0]
    assert row["n"] == 2
    assert row["overall"] == 0.5


def test_nfi_observation_counted_once_per_cell():
    """A plot-year hit by two tiles in one cell must not outweigh the other.

    Regression: common_keys deduplicated the keys but _restrict kept every
    matching row, so two identical observations gave OA=0.667 over n=3 where
    the other column scored 0.5 over n=2.
    """
    a = pd.DataFrame({"TractID": [1, 1, 2], "PlotID": [1, 1, 1],
                      "Year": [2022] * 3, "tile_name": ["t1", "t2", "t1"],
                      "nfi_forest": [1, 1, 2], "model_pred": [1, 1, 3]})
    b = pd.DataFrame({"TractID": [1, 2], "PlotID": [1, 1],
                      "Year": [2022] * 2, "tile_name": ["t1", "t1"],
                      "nfi_forest": [1, 2], "model_pred": [1, 3]})
    rows = {r["cell"]: r for r in lfs.score_nfi({"a_r2": a, "b_r2": b})}
    assert rows["a_r2"]["n"] == rows["b_r2"]["n"] == 2
    assert rows["a_r2"]["overall"] == rows["b_r2"]["overall"] == 0.5


def test_lucas_same_point_in_two_years_is_two_observations():
    """point_id alone is not the identity; the observation year is part of it."""
    d = pd.DataFrame({"point_id": [1, 1], "Year": [2018, 2022],
                      "unified_class": [11, 12], "pred_class": [11, 9]})
    row = lfs.score_lucas({"a_r2": d}, shared_only=True)[0]
    assert row["n"] == 2          # not collapsed to one
    assert row["overall"] == 0.5
