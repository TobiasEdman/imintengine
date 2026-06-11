"""Tests for the NFI validation scoring core (scripts/validate_against_nfi.py).

The model path (sliding-window inference) is ICE-verified; here we exercise the
pure scoring logic with a mock predict_fn — NFI→forest-class derivation, the
forest-type confusion/accuracy, and the per-class AUROC wiring.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "validate_against_nfi",
    str(Path(__file__).resolve().parents[1] / "scripts" / "validate_against_nfi.py"),
)
van = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(van)


def _row(pine=0, contorta=0, spruce=0, birch=0, other=0, mat=None):
    return {
        "VolPine": pine, "VolContorta": contorta, "VolSpruce": spruce,
        "VolBirch": birch, "VolOtherDec": other, "Maturityclass": mat,
    }


def test_derive_forest_class():
    assert van.derive_nfi_forest_class(_row(pine=100)) == 1          # tallskog
    assert van.derive_nfi_forest_class(_row(spruce=100)) == 2        # granskog
    assert van.derive_nfi_forest_class(_row(birch=100)) == 3         # lövskog
    assert van.derive_nfi_forest_class(_row(pine=30, spruce=20, birch=50)) == 4  # blandskog
    assert van.derive_nfi_forest_class(_row(contorta=80, spruce=20)) == 1  # contorta = pine
    assert van.derive_nfi_forest_class(_row()) is None              # treeless


def test_nfi_is_mature():
    assert van.nfi_is_mature(_row(mat=41)) == 1
    assert van.nfi_is_mature(_row(mat=51)) == 1
    assert van.nfi_is_mature(_row(mat=31)) == 0
    assert van.nfi_is_mature(_row(mat=None)) == 0
    assert van.nfi_is_mature(_row(mat=float("nan"))) == 0


def _index():
    base = dict(tile_name="tileA", tile_path="tileA")
    return pd.DataFrame([
        {**base, "row": 0, "col": 0, **_row(pine=100, mat=41)},    # truth 1, pred 1 ✓
        {**base, "row": 1, "col": 1, **_row(spruce=100, mat=31)},  # truth 2, pred 1 ✗
        {**base, "row": 2, "col": 2, **_row(birch=100, mat=21)},   # truth 3, pred 3 ✓
        {**base, "row": 3, "col": 3, **_row(pine=30, spruce=20, birch=50)},  # truth 4, pred 4 ✓
        {**base, "row": 4, "col": 4, **_row()},                    # truth None (treeless)
    ])


def _mock_predict(_tile_path):
    H = W = 8
    class_map = np.zeros((H, W), dtype=int)
    class_map[0, 0], class_map[1, 1], class_map[2, 2], class_map[3, 3], class_map[4, 4] = 1, 1, 3, 4, 7
    probs = np.zeros((23, H, W))
    probs[1, 0, 0] = 0.9          # tallskog prob high only at the true tallskog plot
    for (r, c) in [(1, 1), (2, 2), (3, 3), (4, 4)]:
        probs[1, r, c] = 0.1
    return class_map, probs


def test_score_against_nfi_forest_type():
    res = van.score_against_nfi(_index(), _mock_predict, num_classes=23)
    assert res["n_plots"] == 5
    assert res["n_forest"] == 4          # treeless plot excluded
    assert res["n_mature"] == 1          # only the Maturityclass-41 plot
    assert res["forest_type_accuracy"] == pytest.approx(0.75)  # 3 of 4 forest plots
    # granskog plot was predicted tallskog:
    assert res["confusion_nfi_x_pred"]["granskog"] == {"tallskog": 1}
    assert res["confusion_nfi_x_pred"]["tallskog"] == {"tallskog": 1}
    # tallskog prob is highest at the one true tallskog plot → perfect ranking:
    assert res["per_class_auroc"]["tallskog"]["auroc"] == pytest.approx(1.0)
    assert "granskog" in res["per_class_auroc"]  # 1 positive, 4 negatives → defined
