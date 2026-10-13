"""Stored-confusion-matrix rescoring of the distillation ladder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imint.metrics import compute_miou, miou_from_confusion
from scripts import ladder_miou_v2_rescore as rescore


def _compute_miou_905c656_parent(pred, target, num_classes, ignore_index=0):
    """``compute_miou`` verbatim from 905c656^1 (pre-PR-#76), names dropped."""
    pred = pred.flatten()
    target = target.flatten()
    n = num_classes
    cm = np.zeros((n, n), dtype=np.int64)
    valid = (target >= 0) & (target < n) & (pred >= 0) & (pred < n)
    np.add.at(cm, (target[valid], pred[valid]), 1)
    per_class_iou = {}
    ious = []
    for c in range(n):
        if c == ignore_index:
            continue
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else float("nan")
        per_class_iou[c] = round(float(iou), 4)
        if not np.isnan(iou):
            ious.append(iou)
    miou = float(np.mean(ious)) if ious else 0.0
    return {"miou": round(miou, 4), "per_class_iou": per_class_iou,
            "confusion_matrix": cm}


def _full_cm(pred, target, n):
    """Matrix as the pre-v2 trainer stored it: target==0 rows included."""
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (target.ravel(), pred.ravel()), 1)
    return cm


def _cases():
    rng = np.random.default_rng(20261013)
    for n in (5, 23, 28):
        for _ in range(8):
            size = int(rng.integers(50, 4000))
            target = rng.integers(0, n, size)
            target[rng.random(size) < rng.uniform(0.1, 0.6)] = 0
            # A class predicted but absent from the target, sometimes only
            # on target-0 pixels (NaN in v2, 0.0 in the old metric).
            absent = int(rng.integers(1, n))
            target[target == absent] = 0
            pred = np.where(rng.random(size) < 0.6, target,
                            rng.integers(0, n, size))
            if rng.random() < 0.5:
                pred[(target == 0) & (rng.random(size) < 0.3)] = absent
            yield n, pred, target
    yield 2, np.array([1, 1, 1, 1]), np.array([1, 0, 0, 0])


@pytest.mark.parametrize("n,pred,target", list(_cases()))
def test_matrix_scores_equal_pixel_scores(n, pred, target):
    cm = _full_cm(pred, target, n)

    old, old_pc = rescore.legacy_miou_from_confusion(cm)
    legacy = _compute_miou_905c656_parent(pred, target, n)
    np.testing.assert_array_equal(legacy["confusion_matrix"], cm)
    assert round(old, 4) == legacy["miou"]
    assert {c: round(float(v), 4) for c, v in old_pc.items()} == pytest.approx(
        legacy["per_class_iou"], nan_ok=True)

    new, new_pc = miou_from_confusion(cm, ignore_index=0)
    shipped = compute_miou(pred, target, n)
    assert round(new, 4) == shipped["miou"]
    assert [round(float(v), 4) for v in new_pc.values()] == pytest.approx(
        list(shipped["per_class_iou"].values()), nan_ok=True)
    # compute_miou's own (already filtered) matrix scores identically.
    assert miou_from_confusion(shipped["confusion_matrix"])[0] == new


def test_documented_example_and_no_input_mutation():
    cm = _full_cm(np.array([1, 1, 1, 1]), np.array([1, 0, 0, 0]), 2)
    before = cm.copy()
    assert rescore.legacy_miou_from_confusion(cm)[0] == 0.25
    assert miou_from_confusion(cm)[0] == 1.0
    np.testing.assert_array_equal(cm, before)


def test_write_exclusive_refuses_overwrite(tmp_path):
    path = tmp_path / rescore.OUTPUT_NAME
    rescore.write_exclusive(path, {"a": 1})
    with pytest.raises(FileExistsError):
        rescore.write_exclusive(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def _make_cell(root: Path, name: str, *, stored_delta=0.0, ckpt_epoch=None):
    rng = np.random.default_rng(len(name))
    n = 5
    target = rng.integers(0, n, 3000)
    pred = np.where(rng.random(3000) < 0.7, target, rng.integers(0, n, 3000))
    cm = _full_cm(pred, target, n)
    old = round(rescore.legacy_miou_from_confusion(cm)[0], 4)
    cell = root / name
    cell.mkdir()
    log = {
        "best_epoch": 2, "best_metric": old + stored_delta,
        "epochs": [
            {"epoch": 1, "val_miou": old - 0.01, "is_best": True,
             "confusion_matrix": (cm // 2).tolist()},
            {"epoch": 2, "val_miou": old, "is_best": True,
             "confusion_matrix": cm.tolist(), "elapsed_s": 10.0},
            {"epoch": 3, "val_miou": old - 0.002, "is_best": False},
        ],
    }
    (cell / "training_log.json").write_text(json.dumps(log))
    (cell / "best_model.pt").write_bytes(b"fake")
    ckpt = {"epoch": ckpt_epoch or 2, "metrics": {"miou": old},
            "config": {"num_classes": n}}
    return cell, ckpt


def test_cell_record_and_reverse_check(tmp_path):
    cell, ckpt = _make_cell(tmp_path, "clay_r2")
    rec = rescore.rescore_cell(cell, load_checkpoint=lambda *_: ckpt)
    assert (rec["backbone"], rec["rung"], rec["best_epoch"]) == ("clay", 2, 2)
    assert round(rec["old_miou_recomputed"], 4) == rec["stored_best_miou"]
    assert rec["delta"] == pytest.approx(rec["new_miou"] - rec["old_miou_recomputed"], abs=2e-6)
    assert rec["log_context"]["runner_up_by_old_val_miou"]["epoch"] == 3
    assert rec["val_fingerprint"]["cm_total_pixels"] == 3000
    ctx = rec["log_context"]
    assert [e["epoch"] for e in ctx["recorded_epochs"]] == [1, 2]
    assert ctx["recorded_epochs"][1]["new_miou"] == rec["new_miou"]


def test_stored_mismatch_fails_cell(tmp_path):
    cell, ckpt = _make_cell(tmp_path, "clay_r2", stored_delta=0.001)
    with pytest.raises(rescore.CellError, match="does not reproduce"):
        rescore.rescore_cell(cell, load_checkpoint=lambda *_: ckpt)


def test_checkpoint_epoch_mismatch_fails_cell(tmp_path):
    cell, ckpt = _make_cell(tmp_path, "clay_r2", ckpt_epoch=1)
    with pytest.raises(rescore.CellError, match="best_model.pt epoch"):
        rescore.rescore_cell(cell, load_checkpoint=lambda *_: ckpt)


def test_run_is_all_or_nothing(tmp_path, monkeypatch):
    good, ckpt = _make_cell(tmp_path, "clay_r1")
    bad, _ = _make_cell(tmp_path, "croma_r1", stored_delta=0.01)
    monkeypatch.setattr(rescore, "load_checkpoint_cpu", lambda *_: ckpt)
    summary = tmp_path / "_summary.json"
    argv = ["--ladder-root", str(tmp_path), "--expected-cells", "2",
            "--tiles-root", str(tmp_path), "--summary-out", str(summary)]

    assert rescore.main(argv + ["--write"]) == 1           # one bad cell
    assert not (good / rescore.OUTPUT_NAME).exists() and not summary.exists()

    (bad / "training_log.json").unlink()                   # leaves one cell
    assert rescore.main(argv + ["--write"]) == 2           # count guard
    argv[3] = "1"
    assert rescore.main(argv) == 0                         # dry run writes nothing
    assert not (good / rescore.OUTPUT_NAME).exists()
    assert rescore.main(argv + ["--write"]) == 0
    assert (good / rescore.OUTPUT_NAME).exists() and summary.exists()
    assert rescore.main(argv + ["--write"]) == 2           # refuses a rerun


def test_checkpoint_loader_reads_numpy_scalar_metrics(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "best_model.pt"
    torch.save({"epoch": np.int64(7), "metrics": {"miou": np.float64(0.4321)},
                "config": {"num_classes": 28},
                "state_dict": {"w": torch.zeros(2)}}, path)
    ckpt = rescore.load_checkpoint_cpu(
        path, rescore.sha256_file(path), path.stat().st_size)
    assert int(ckpt["epoch"]) == 7 and float(ckpt["metrics"]["miou"]) == 0.4321
    with pytest.raises(Exception, match="sha256 mismatch"):
        rescore.load_checkpoint_cpu(path, "0" * 64, path.stat().st_size)
