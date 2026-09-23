"""The 28-class holdout eval's arithmetic, pinned against hand-built matrices."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "_lhe", str(ROOT / "scripts" / "ladder_holdout_eval.py"))
lhe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lhe)


def _cm(n=28):
    return np.zeros((n, n), dtype=np.int64)


def test_overall_accuracy_counts_predictions_that_fell_into_background():
    """Regression: excluding class 0 from both axes inflates the headline.

    Truth is class 5 for 100 pixels; the model calls 60 of them class 5 and
    40 of them background. The honest accuracy is 0.60. Scoring only the
    non-background submatrix would see 60 of 60 and report 1.0.
    """
    cm = _cm()
    cm[5, 5] = 60
    cm[5, 0] = 40
    out = lhe.derive(cm)
    assert out["overall_accuracy"] == 0.6
    assert out["pixels_scored"] == 100


def test_background_truth_is_excluded_from_every_aggregate():
    cm = _cm()
    cm[0, 0] = 10_000          # background predicted background
    cm[0, 7] = 5_000           # background predicted as class 7
    cm[7, 7] = 50
    out = lhe.derive(cm)
    # Only the 50 class-7 truth pixels are scored.
    assert out["pixels_scored"] == 50
    assert out["overall_accuracy"] == 1.0
    # ...but the 5,000 false positives still cost class 7 its precision.
    assert out["per_class"][7]["precision"] == pytest.approx(50 / 5050, abs=1e-4)
    assert out["per_class"][7]["recall"] == 1.0


def test_absent_classes_are_none_and_excluded_from_the_mean():
    cm = _cm()
    cm[3, 3] = 10
    cm[9, 9] = 30
    out = lhe.derive(cm)
    assert out["classes_present"] == 2
    assert out["per_class"][3]["iou"] == 1.0
    assert out["per_class"][11]["iou"] is None      # never appears
    assert out["mean_iou"] == 1.0                   # mean over present only


def test_iou_orientation_rows_are_truth_columns_are_prediction():
    """A one-sided confusion must not read the same as its transpose."""
    cm = _cm()
    cm[4, 4] = 40
    cm[4, 6] = 60      # truth 4 predicted 6 — hurts 4's recall, 6's precision
    out = lhe.derive(cm)
    assert out["per_class"][4]["recall"] == 0.4
    assert out["per_class"][4]["precision"] == 1.0
    assert out["per_class"][6]["precision"] == 0.0
    assert out["per_class"][6]["recall"] is None    # class 6 never true


def test_select_tiles_spreads_and_is_deterministic(tmp_path):
    hold = tmp_path / "hold"; lab = tmp_path / "lab"
    hold.mkdir(); lab.mkdir()
    for i in range(100):
        (hold / f"t{i:03d}.npz").touch()
        (lab / f"t{i:03d}.npz").touch()
    got = lhe.select_tiles(hold, lab, 10)
    assert got == lhe.select_tiles(hold, lab, 10)          # deterministic
    assert len(got) == 10
    assert got[0] == "t000.npz".replace(".npz", "")
    assert got[-1] != "t009"                                # strided, not a prefix
    assert lhe.select_tiles(hold, lab, None) == sorted(p.stem for p in hold.glob("*.npz"))


def test_tiles_without_28_class_truth_are_not_selected(tmp_path):
    hold = tmp_path / "hold"; lab = tmp_path / "lab"
    hold.mkdir(); lab.mkdir()
    for i in range(5):
        (hold / f"t{i}.npz").touch()
    (lab / "t2.npz").touch()
    assert lhe.select_tiles(hold, lab, None) == ["t2"]


def test_no_overlapping_truth_fails_loudly(tmp_path):
    hold = tmp_path / "hold"; lab = tmp_path / "lab"
    hold.mkdir(); lab.mkdir()
    (hold / "t0.npz").touch()
    with pytest.raises(SystemExit, match="sidecar"):
        lhe.select_tiles(hold, lab, None)


def test_truth_is_cropped_to_the_window_the_model_saw():
    """504-sized predictions score against truth[4:508, 4:508], not the tile.

    Regression: comparing a 504 prediction with the full 512 label skipped
    every tile (the probe scored 0 pixels). Resizing instead would be worse
    than skipping — interpolating categorical labels invents classes.
    """
    truth = np.zeros((512, 512), dtype=np.int64)
    truth[4:508, 4:508] = 7          # exactly the window the model sees
    out = lhe.centre_crop_truth(truth, 504)
    assert out.shape == (504, 504)
    assert (out == 7).all()          # the border is gone, nothing shifted


def test_crop_is_a_noop_when_the_model_sees_the_whole_tile():
    truth = np.arange(512 * 512, dtype=np.int64).reshape(512, 512)
    assert np.array_equal(lhe.centre_crop_truth(truth, 512), truth)
    # img_size larger than the tile cannot crop beyond it
    assert lhe.centre_crop_truth(truth, 1024).shape == (512, 512)


def test_crop_offset_matches_run_inference_arithmetic():
    """Pin the exact offset math copied from inference_comparison."""
    for h, w, img in ((512, 512, 504), (512, 512, 224), (500, 500, 504)):
        truth = np.zeros((h, w), dtype=np.int64)
        crop_sz = min(img, h, w)
        y0, x0 = (h - crop_sz) // 2, (w - crop_sz) // 2
        truth[y0:y0 + crop_sz, x0:x0 + crop_sz] = 3
        assert (lhe.centre_crop_truth(truth, img) == 3).all()
