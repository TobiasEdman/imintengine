"""Per-tile confusion files and the paired tile bootstrap over them."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from imint.eval.tile_bootstrap import (
    load_tile_confusions, paired_iou_difference, save_tile_confusions,
)

ROOT = Path(__file__).resolve().parents[1]


def _tile(n=4, *, right=100, wrong=0):
    """Truth: 100 px of class 1 and 100 px of class 2. ``wrong`` class-1 px called 2."""
    cm = np.zeros((n, n), dtype=np.int64)
    cm[1, 1], cm[1, 2] = right - wrong, wrong
    cm[2, 2] = 100
    return cm


def test_save_and_load_round_trip(tmp_path):
    matrices = {"b": _tile(wrong=3), "a": _tile()}
    path = tmp_path / "x.tiles.npz"
    sha = save_tile_confusions(path, matrices)
    assert len(sha) == 64 and not (tmp_path / "x.tiles.npz.tmp").exists()
    loaded = load_tile_confusions(path)
    assert list(loaded) == ["a", "b"]
    assert all(np.array_equal(loaded[k], matrices[k]) for k in matrices)
    assert loaded["a"].dtype == np.int64


@pytest.mark.parametrize("bad", [{}, {"a": np.zeros((3, 4), np.int64)},
                                 {"a": np.full((3, 3), -1)}, {"a": np.zeros((3, 3))}])
def test_save_refuses_what_is_not_a_count_matrix(tmp_path, bad):
    with pytest.raises(ValueError):
        save_tile_confusions(tmp_path / "x.npz", bad)


def test_known_difference_has_a_degenerate_interval():
    """Every tile identical → every resample gives the same IoUs."""
    a = {f"t{i}": _tile() for i in range(10)}
    b = {f"t{i}": _tile(wrong=50) for i in range(10)}
    out = paired_iou_difference(a, b, draws=200, seed=3)
    one, two = out["per_class"][1], out["per_class"][2]
    assert (one["a"], one["b"]) == (1.0, 0.5)
    assert one["difference"] == pytest.approx(0.5)
    assert one["interval"] == pytest.approx([0.5, 0.5])
    assert two["difference"] == pytest.approx(1 / 3)          # 100 / (100 + 50 fp)
    assert out["mean_iou"]["difference"] == pytest.approx((0.5 + 1 / 3) / 2)
    assert out["per_class"][3]["interval"] is None             # never true, never said
    assert out["per_class"][3]["defined_draws"] == 0
    assert 0 not in out["per_class"]
    json.dumps(out, allow_nan=False)


def test_bootstrap_is_seeded_and_the_interval_brackets_the_estimate():
    rng = np.random.default_rng(5)
    a, b = {}, {}
    for i in range(40):
        wrong = int(rng.integers(0, 60))
        a[f"t{i}"] = _tile(wrong=int(rng.integers(0, 20)))
        b[f"t{i}"] = _tile(wrong=wrong)
    first = paired_iou_difference(a, b, draws=300, seed=11)
    assert first == paired_iou_difference(a, b, draws=300, seed=11)
    other = paired_iou_difference(a, b, draws=300, seed=12)
    assert other["per_class"][1]["interval"] != first["per_class"][1]["interval"]
    lo, hi = first["per_class"][1]["interval"]
    assert lo < first["per_class"][1]["difference"] < hi and lo > 0
    assert first["draws"] == 300 and first["seed"] == 11 and first["tiles"] == 40


def test_unpaired_inputs_are_refused():
    a = {"t0": _tile(), "t1": _tile()}
    with pytest.raises(ValueError, match="different tiles"):
        paired_iou_difference(a, {"t0": _tile()})
    with pytest.raises(ValueError, match="label spaces"):
        paired_iou_difference(a, {k: _tile(n=5) for k in a})
    shifted = {"t0": _tile(), "t1": _tile(right=90)}           # other truth on t1
    with pytest.raises(ValueError, match="different truth.*t1"):
        paired_iou_difference(a, shifted)
    with pytest.raises(ValueError, match="draws"):
        paired_iou_difference(a, a, draws=0)


def test_cli_writes_the_difference_with_input_hashes(tmp_path):
    save_tile_confusions(tmp_path / "a.npz", {f"t{i}": _tile() for i in range(5)})
    save_tile_confusions(tmp_path / "b.npz", {f"t{i}": _tile(wrong=50) for i in range(5)})
    out = tmp_path / "diff.json"
    subprocess.run([sys.executable, "-m", "imint.eval.tile_bootstrap", str(tmp_path / "a.npz"),
                    str(tmp_path / "b.npz"), "--draws", "50", "--out", str(out)],
                   cwd=ROOT, check=True, timeout=120)
    report = json.loads(out.read_text())
    assert report["per_class"]["1"]["difference"] == pytest.approx(0.5)
    assert report["draws"] == 50 and report["seed"] == 42
    assert len(report["inputs"]["a"]["sha256"]) == 64
