"""The 28-class holdout eval's arithmetic, pinned against hand-built matrices."""
from __future__ import annotations

import importlib.util
import json
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




def _args(tmp_path, **kw):
    a = ["ladder_holdout_eval.py", "--holdout-dir", str(kw["hold"]),
         "--label-dir", str(kw["lab"]), "--ckpt-root", str(kw["ck"]),
         "--out", str(kw["out"]), "--device", "cpu"]
    return a + list(kw.get("extra", []))


def _ns(hold, lab, ck, **kw):
    """The fingerprint's view of a ladder-grid invocation."""
    import argparse
    return argparse.Namespace(**{"holdout_dir": hold, "label_dir": lab, "ckpt_root": ck,
                                 "num_classes": 28, "fold": None, "checkpoint": None,
                                 "cell_name": None, **kw})


def _fake_tree(tmp_path, cells):
    hold = tmp_path / "hold"; lab = tmp_path / "lab"; ck = tmp_path / "ck"
    for d in (hold, lab, ck):
        d.mkdir()
    (hold / "t0.npz").touch(); (lab / "t0.npz").touch()
    for c in cells:
        (ck / c).mkdir()
        (ck / c / "best_model.pt").touch()
    return hold, lab, ck


def _fake_cell(cell, column, ckpt, *a, tiles_out, **k):
    """What a scored cell looks like to main(), per-tile file included."""
    cm = _cm()
    cm[1, 1] = 10
    return {"cell": cell, "overall_accuracy": 0.5, "mean_iou": 0.3, "pixels_scored": 10,
            "checkpoint_sha256": lhe.sha256_file(ckpt),
            "tile_confusions": {"file": tiles_out.name,
                                "sha256": lhe.save_tile_confusions(tiles_out, {"t0": cm})}}


def test_one_unscoreable_cell_does_not_kill_the_sweep(monkeypatch, tmp_path, capsys):
    """Regression: a fail-loud dataset contract killed a 21-cell sweep.

    croma refuses a tile whose s1_enrich_v predates its training, which is a
    property of the holdout set — not a reason to lose the other twenty cells.
    """
    hold, lab, ck = _fake_tree(tmp_path, ["clay_r2", "croma_r2"])
    out = tmp_path / "out.json"

    def fake_score(cell, column, *a, **k):
        if column == "croma":
            raise KeyError("tile requires s1_enrich_v==4 but found 3")
        return _fake_cell(cell, column, *a, **k)

    monkeypatch.setattr(lhe, "score_cell", fake_score)
    monkeypatch.setattr("sys.argv", _args(
        tmp_path, hold=hold, lab=lab, ck=ck, out=out,
        extra=["--models", "clay,croma", "--rungs", "2"]))

    # Non-zero: the sweep did not do all of its work.
    assert lhe.main() == 1
    body = json.loads(out.read_text())
    assert [c["cell"] for c in body["cells"]] == ["clay_r2"]
    assert body["failed_cells"][0]["cell"] == "croma_r2"
    assert "s1_enrich_v" in body["failed_cells"][0]["error"]


def test_resume_skips_cells_already_scored(monkeypatch, tmp_path):
    hold, lab, ck = _fake_tree(tmp_path, ["clay_r2", "clay_r3"])
    out = tmp_path / "out.json"
    # A reusable report must carry the run fingerprint, each cell's checkpoint
    # hash and its intact per-tile matrices; resume refuses anything else.
    fp = lhe.run_fingerprint(["t0"], _ns(hold, lab, ck))
    prior = _fake_cell("clay_r2", "clay", ck / "clay_r2" / "best_model.pt",
                       tiles_out=lhe.tile_confusions_path(out, "clay_r2"))
    out.write_text(json.dumps({"fingerprint": fp, "cells": [prior]}))

    scored = []

    def fake_score(cell, *a, **k):
        scored.append(cell)
        return _fake_cell(cell, *a, **k)

    monkeypatch.setattr(lhe, "score_cell", fake_score)
    monkeypatch.setattr("sys.argv", _args(
        tmp_path, hold=hold, lab=lab, ck=ck, out=out,
        extra=["--models", "clay", "--rungs", "2,3", "--resume"]))

    assert lhe.main() == 0
    assert scored == ["clay_r3"]                     # r2 was not re-run
    cells = {c["cell"] for c in json.loads(out.read_text())["cells"]}
    assert cells == {"clay_r2", "clay_r3"}           # and r2 survived


def test_resume_rescores_a_cell_whose_tile_matrices_are_gone(monkeypatch, tmp_path):
    """Without its per-tile matrices a cell cannot be bootstrapped."""
    hold, lab, ck = _fake_tree(tmp_path, ["clay_r2"])
    out = tmp_path / "out.json"
    fp = lhe.run_fingerprint(["t0"], _ns(hold, lab, ck))
    prior = _fake_cell("clay_r2", "clay", ck / "clay_r2" / "best_model.pt",
                       tiles_out=lhe.tile_confusions_path(out, "clay_r2"))
    out.write_text(json.dumps({"fingerprint": fp, "cells": [prior]}))
    lhe.tile_confusions_path(out, "clay_r2").unlink()
    scored = []
    monkeypatch.setattr(lhe, "score_cell",
                        lambda cell, *a, **k: scored.append(cell) or _fake_cell(cell, *a, **k))
    monkeypatch.setattr("sys.argv", _args(
        tmp_path, hold=hold, lab=lab, ck=ck, out=out,
        extra=["--models", "clay", "--rungs", "2", "--resume"]))
    assert lhe.main() == 0
    assert scored == ["clay_r2"]


def test_resume_refuses_a_report_written_for_other_inputs(monkeypatch, tmp_path):
    """Cell name is not identity.

    Regression: a first run with --limit-tiles 1, then a changed checkpoint
    and an unlimited resume, skipped scoring and wrote the new tile list over
    a result whose tiles_scored was 1 and whose checkpoint hash was stale.
    """
    hold, lab, ck = _fake_tree(tmp_path, ["clay_r2"])
    out = tmp_path / "out.json"
    out.write_text(json.dumps({
        "fingerprint": {"n_tiles": 999, "tiles_sha256": "deadbeef"},
        "cells": [{"cell": "clay_r2", "overall_accuracy": 0.61,
                   "mean_iou": 0.3, "checkpoint_sha256": "x"}]}))
    monkeypatch.setattr(lhe, "score_cell", lambda *a, **k: pytest.fail("scored"))
    monkeypatch.setattr("sys.argv", _args(
        tmp_path, hold=hold, lab=lab, ck=ck, out=out,
        extra=["--models", "clay", "--rungs", "2", "--resume"]))
    with pytest.raises(SystemExit, match="different"):
        lhe.main()


def test_resume_rescores_a_cell_whose_checkpoint_changed(monkeypatch, tmp_path):
    hold, lab, ck = _fake_tree(tmp_path, ["clay_r2"])
    out = tmp_path / "out.json"
    fp = lhe.run_fingerprint(["t0"], _ns(hold, lab, ck))
    prior = _fake_cell("clay_r2", "clay", ck / "clay_r2" / "best_model.pt",
                       tiles_out=lhe.tile_confusions_path(out, "clay_r2"))
    prior["checkpoint_sha256"] = "sha-of-a-checkpoint-that-is-gone"
    out.write_text(json.dumps({"fingerprint": fp, "cells": [prior]}))
    scored = []
    monkeypatch.setattr(lhe, "score_cell",
                        lambda cell, *a, **k: scored.append(cell) or _fake_cell(cell, *a, **k))
    monkeypatch.setattr("sys.argv", _args(
        tmp_path, hold=hold, lab=lab, ck=ck, out=out,
        extra=["--models", "clay", "--rungs", "2", "--resume"]))
    assert lhe.main() == 0
    assert scored == ["clay_r2"]   # stale checkpoint → re-scored, not reused


def test_actual_write_resume_round_trip_and_changed_inputs(monkeypatch, tmp_path):
    hold, lab, ck = _fake_tree(tmp_path, ["clay_r2"])
    out = tmp_path / "out.json"
    scored = []

    def fake_score(cell, *args, **kwargs):
        scored.append(cell)
        return _fake_cell(cell, *args, **kwargs)

    argv = _args(tmp_path, hold=hold, lab=lab, ck=ck, out=out,
                 extra=["--models", "clay", "--rungs", "2"])
    monkeypatch.setattr(lhe, "score_cell", fake_score)
    monkeypatch.setattr(sys, "argv", argv)
    assert lhe.main() == 0
    assert "fingerprint" in json.loads(out.read_text())
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    assert lhe.main() == 0
    assert scored == ["clay_r2"]
    (hold / "t0.npz").write_bytes(b"changed input at the SAME pathname")
    with pytest.raises(SystemExit, match="different inputs"):
        lhe.main()
    assert scored == ["clay_r2"]


def test_fingerprint_binds_label_bytes_model_config_classes_and_fold(tmp_path, monkeypatch):
    import copy
    hold, lab, ck = _fake_tree(tmp_path, [])
    args = _ns(hold, lab, ck)
    original = lhe.run_fingerprint(["t0"], args)
    (lab / "t0.npz").write_bytes(b"new labels")
    assert lhe.run_fingerprint(["t0"], args) != original
    (lab / "t0.npz").write_bytes(b"")
    assert lhe.run_fingerprint(["t0"], args) == original
    direct = _ns(hold, lab, None, num_classes=29, checkpoint=tmp_path / "m.pt",
                 cell_name="main")
    unfolded = lhe.run_fingerprint(["t0"], direct)
    direct.fold = (28, 3)
    folded = lhe.run_fingerprint(["t0"], direct)
    assert folded != unfolded and folded["fold"] == {"prediction": 28, "scored_as": 3}
    assert unfolded["num_classes"] == 29 and unfolded["checkpoint"].endswith("m.pt")
    changed = copy.deepcopy(lhe.DISTILL)
    changed["clay"]["img_size"] = 496
    monkeypatch.setattr(lhe, "DISTILL", changed)
    assert lhe.run_fingerprint(["t0"], args) != original


# ── 29 classes, folds and the range contract ────────────────────────────────


def test_class_28_is_scored_as_its_own_class_with_29_classes():
    """Regression: any value >= 28 skipped the whole tile, so every
    barrblandskog tile silently left the score."""
    truth = np.array([[28, 28, 3, 0]], dtype=np.uint8)
    pred = np.array([[28, 3, 3, 28]])
    cm = lhe.tile_confusion(pred, truth, num_classes=29, fold=None)
    assert cm.shape == (29, 29)
    assert cm[28, 28] == 1 and cm[28, 3] == 1 and cm[3, 3] == 1
    assert cm.sum() == 3                    # background truth stays out (v2 contract)


def test_fold_28_into_3_scores_in_the_original_28_class_space():
    truth = np.array([[3, 3, 3, 1]], dtype=np.uint8)
    pred = np.array([[28, 3, 1, 28]])
    cm = lhe.tile_confusion(pred, truth, num_classes=29, fold=(28, 3))
    assert cm.shape == (28, 28)
    assert cm[3, 3] == 2 and cm[3, 1] == 1 and cm[1, 3] == 1


@pytest.mark.parametrize("pred,truth,num_classes,fold,what", [
    ([[29]], [[1]], 29, None, "prediction"),       # beyond the model's classes
    ([[28]], [[1]], 28, None, "prediction"),       # a 28-class model cannot say 28
    ([[1]], [[28]], 28, None, "truth"),            # b28 labels under R4
    ([[1]], [[28]], 29, (28, 3), "truth"),         # b28 labels under the fold
    ([[1]], [[255]], 29, None, "truth"),
    ([[-1]], [[1]], 29, None, "prediction"),
])
def test_values_outside_the_scored_space_raise_not_drop(pred, truth, num_classes, fold, what):
    with pytest.raises(ValueError, match=f"{what} pixel"):
        lhe.tile_confusion(np.array(pred), np.array(truth), num_classes=num_classes, fold=fold)


def test_float_labels_are_refused():
    with pytest.raises(ValueError, match="integer"):
        lhe.tile_confusion(np.array([[1]]), np.array([[1.0]]), num_classes=28, fold=None)


@pytest.mark.parametrize("num_classes,fold", [
    (29, (27, 3)), (29, (28, 28)), (29, (28, -1)), (28, (28, 3)), (28, (27, 3)), (30, None)])
def test_the_only_fold_is_28_onto_a_lower_class_with_29_classes(num_classes, fold):
    with pytest.raises(ValueError):
        lhe.scored_classes(num_classes, fold)


def test_scored_space_follows_the_fold():
    assert lhe.scored_classes(29, (28, 3)) == 28
    assert lhe.scored_classes(29, None) == 29
    assert lhe.scored_classes(28, None) == 28


def test_parse_fold():
    import argparse
    assert lhe.parse_fold("28:3") == (28, 3)
    for bad in ("28", "28:3:1", "a:3", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            lhe.parse_fold(bad)


# ── label directory bound to the class space ────────────────────────────────


def _b28_manifest(lab, *, rule=None, nmd=None):
    """What scripts/relabel_barrblandskog.py run writes into its NMD output."""
    (lab / lhe.RELABEL_MANIFEST).write_text(json.dumps({
        "rule": rule or lhe.RELABEL_RULE,
        "outputs": {"nmd": str(nmd or lab), "distilled": "/cephfs/distill/tessera_r2_b28"}}))


def test_relabel_constants_are_the_relabel_scripts():
    import relabel_barrblandskog as relabel
    assert (lhe.RELABEL_MANIFEST, lhe.RELABEL_RULE) == (relabel.MANIFEST, relabel.RULE)


def test_29_class_scoring_requires_barrblandskog_labels(tmp_path):
    with pytest.raises(ValueError, match="needs barrblandskog labels"):
        lhe.label_set(tmp_path, 29)
    assert lhe.label_set(tmp_path, 28) is None
    _b28_manifest(tmp_path)
    got = lhe.label_set(tmp_path, 29)
    assert got["rule"] == lhe.RELABEL_RULE and len(got["sha256"]) == 64


def test_28_class_scoring_refuses_barrblandskog_labels(tmp_path):
    """The fold scores in 28 classes, so it too needs the original labels."""
    _b28_manifest(tmp_path)
    with pytest.raises(ValueError, match="original labels"):
        lhe.label_set(tmp_path, lhe.scored_classes(29, (28, 3)))


@pytest.mark.parametrize("kw,match", [({"rule": "other-rule"}, "rule"),
                                      ({"nmd": "/cephfs/elsewhere"}, "NMD output")])
def test_a_manifest_for_other_labels_is_refused(tmp_path, kw, match):
    _b28_manifest(tmp_path, **kw)
    with pytest.raises(ValueError, match=match):
        lhe.label_set(tmp_path, 29)


# ── scoring one cell ─────────────────────────────────────────────────────────


def _backend(num_classes, predict, *, backbone="tessera_v1", img_size=504, calls=None):
    """inference_comparison stand-in: checkpoint config + a prediction rule."""
    import types

    def load_model(*args, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        return (object(), 7, 0.5, None, {"num_classes": num_classes,
                                         "backbone_name": backbone, "img_size": img_size})
    return types.SimpleNamespace(
        load_model=load_model,
        run_inference=lambda model, tile, *a, **k: predict(Path(tile).stem))


def _label_tiles(lab, labels):
    for name, label in labels.items():
        np.savez(lab / f"{name}.npz", label=np.asarray(label, dtype=np.uint8))


def _tessera_tiles(tmp_path, n=3):
    """Synthetic 512 tiles; the model sees the 504 centre (tessera img_size)."""
    hold, lab = tmp_path / "hold", tmp_path / "lab"
    hold.mkdir(); lab.mkdir()
    labels = {}
    for i in range(n):
        label = np.full((512, 512), 1, dtype=np.uint8)
        label[:, 256:] = 28 if i < 2 else 3        # two barrblandskog tiles
        labels[f"t{i}"] = label
        (hold / f"t{i}.npz").write_bytes(b"tile")
    _label_tiles(lab, labels)
    return hold, lab, labels


def _perfect(labels, img_size=504):
    return lambda name: lhe.centre_crop_truth(labels[name], img_size).astype(np.int64)


def _score(cell, column, ckpt, tiles, hold, lab, **kw):
    kw = {"fold": None, "check_column": False, **kw}
    return lhe.score_cell(cell, column, ckpt, tiles, hold, lab, "cpu", **kw)


def test_29_class_cell_scores_every_barrblandskog_tile_and_saves_per_tile(monkeypatch, tmp_path):
    from imint.eval.tile_bootstrap import load_tile_confusions
    hold, lab, labels = _tessera_tiles(tmp_path)
    ckpt = tmp_path / "main.pt"; ckpt.write_bytes(b"weights")
    monkeypatch.setattr(lhe, "_load_infcmp", lambda: _backend(29, _perfect(labels)))
    tiles_out = tmp_path / "r.main.tiles.npz"
    out = _score("main", "tessera", ckpt, ["t0", "t1", "t2"], hold, lab,
                 num_classes=29, tiles_out=tiles_out)
    assert out["tiles_scored"] == 3 and out["tiles_skipped"] == []
    assert out["per_class"][28]["iou"] == 1.0 and out["per_class"][28]["support"] == 2 * 504 * 252
    assert (out["num_classes"], out["scored_classes"], out["fold"]) == (29, 29, None)
    assert out["label_dir"] == str(lab)
    per_tile = load_tile_confusions(tiles_out)
    assert sorted(per_tile) == ["t0", "t1", "t2"]
    assert out["tile_confusions"] == {"file": tiles_out.name, "sha256": lhe.sha256_file(tiles_out)}
    assert np.array_equal(sum(per_tile.values()), np.array(out["confusion_matrix"]))
    assert per_tile["t2"][28].sum() == 0 and per_tile["t0"][28, 28] == 504 * 252


def test_checkpoint_is_hashed_before_loading_and_the_loader_is_bound_to_it(monkeypatch, tmp_path):
    hold, lab, labels = _tessera_tiles(tmp_path)
    ckpt = tmp_path / "main.pt"; ckpt.write_bytes(b"weights")
    calls = []
    monkeypatch.setattr(lhe, "_load_infcmp", lambda: _backend(29, _perfect(labels), calls=calls))
    out = _score("main", "tessera", ckpt, ["t0"], hold, lab, num_classes=29,
                 tiles_out=tmp_path / "x.npz")
    identity = lhe.file_identity(ckpt)
    assert calls[0]["expected_checkpoint_sha256"] == identity["sha256"] == out["checkpoint_sha256"]
    assert calls[0]["expected_checkpoint_size"] == identity["size"] == out["checkpoint_size"] == 7


def test_folded_cell_lands_in_28_classes_on_the_original_labels(monkeypatch, tmp_path):
    hold, lab, labels = _tessera_tiles(tmp_path)
    original = {k: np.where(v == 28, 3, v).astype(np.uint8) for k, v in labels.items()}
    _label_tiles(lab, original)                                  # NMD 113 as class 3
    ckpt = tmp_path / "main.pt"; ckpt.write_bytes(b"weights")
    # The model says 28 where b28 truth is 28 — exactly what the fold maps to 3.
    monkeypatch.setattr(lhe, "_load_infcmp", lambda: _backend(29, _perfect(labels)))
    out = _score("main_fold", "tessera", ckpt, ["t0", "t1", "t2"], hold, lab,
                 num_classes=29, fold=(28, 3), tiles_out=tmp_path / "f.npz")
    assert np.array(out["confusion_matrix"]).shape == (28, 28)
    assert out["per_class"][3]["iou"] == 1.0 and 28 not in out["per_class"]
    assert out["fold"] == {"prediction": 28, "scored_as": 3} and out["scored_classes"] == 28


def test_num_classes_mismatch_with_the_checkpoint_is_refused(monkeypatch, tmp_path):
    hold, lab, labels = _tessera_tiles(tmp_path)
    ckpt = tmp_path / "r4.pt"; ckpt.write_bytes(b"weights")
    monkeypatch.setattr(lhe, "_load_infcmp", lambda: _backend(28, _perfect(labels)))
    with pytest.raises(ValueError, match="num_classes=28.*--num-classes is 29"):
        _score("main", "tessera", ckpt, ["t0"], hold, lab, num_classes=29,
               tiles_out=tmp_path / "x.npz")
    assert not (tmp_path / "x.npz").exists()


@pytest.mark.parametrize("backbone,img_size", [("clay_v1_5", 504), ("tessera_v1", 512),
                                                (None, None)])
def test_a_single_checkpoint_must_match_its_column(monkeypatch, tmp_path, backbone, img_size):
    hold, lab, labels = _tessera_tiles(tmp_path)
    ckpt = tmp_path / "main.pt"; ckpt.write_bytes(b"weights")
    monkeypatch.setattr(lhe, "_load_infcmp", lambda: _backend(
        29, _perfect(labels), backbone=backbone, img_size=img_size))
    with pytest.raises(ValueError, match="backbone/img_size"):
        _score("main", "tessera", ckpt, ["t0"], hold, lab, num_classes=29,
               tiles_out=tmp_path / "x.npz", check_column=True)
    # The ladder grid does not ask (older checkpoints carry no img_size).
    _score("main", "tessera", ckpt, ["t0"], hold, lab, num_classes=29,
           tiles_out=tmp_path / "x.npz", check_column=False)


def test_out_of_range_truth_fails_the_cell_naming_the_tile(monkeypatch, tmp_path):
    hold, lab, labels = _tessera_tiles(tmp_path)
    ckpt = tmp_path / "r4.pt"; ckpt.write_bytes(b"weights")
    monkeypatch.setattr(lhe, "_load_infcmp",
                        lambda: _backend(28, lambda name: np.ones((504, 504), np.int64)))
    with pytest.raises(ValueError, match="tile t0: .*truth pixel.*28"):
        _score("tessera_r4", "tessera", ckpt, ["t0"], hold, lab, num_classes=28,
               tiles_out=tmp_path / "x.npz")


def test_background_only_cell_is_failed(monkeypatch, tmp_path):
    hold, lab, ck = _fake_tree(tmp_path, ["clay_r2"])
    np.savez(lab / "t0.npz", label=np.zeros((8, 8), dtype=np.int64))
    monkeypatch.setattr(lhe, "_load_infcmp",
                        lambda: _backend(28, lambda name: np.zeros((8, 8), np.int64)))
    with pytest.raises(RuntimeError, match="no pixel scored"):
        _score("clay_r2", "clay", ck / "clay_r2/best_model.pt", ["t0"], hold, lab,
               num_classes=28, tiles_out=tmp_path / "x.npz")


# ── the single-checkpoint CLI ────────────────────────────────────────────────


def _direct_argv(tmp_path, hold, lab, ckpt, out, *extra):
    return ["ladder_holdout_eval.py", "--holdout-dir", str(hold), "--label-dir", str(lab),
            "--checkpoint", str(ckpt), "--cell-name", "main", "--models", "tessera",
            "--out", str(out), "--device", "cpu", *extra]


def test_direct_checkpoint_mode_writes_report_and_per_tile_file(monkeypatch, tmp_path):
    from imint.eval.tile_bootstrap import load_tile_confusions
    hold, lab, labels = _tessera_tiles(tmp_path)
    _b28_manifest(lab)
    ckpt = tmp_path / "best_model.pt"; ckpt.write_bytes(b"weights")
    out = tmp_path / "eval" / "main.json"
    monkeypatch.setattr(lhe, "_load_infcmp", lambda: _backend(29, _perfect(labels)))
    monkeypatch.setattr(sys, "argv", _direct_argv(tmp_path, hold, lab, ckpt, out,
                                                  "--num-classes", "29"))
    assert lhe.main() == 0
    report = json.loads(out.read_text())
    assert report["schema"] == "ladder-holdout-v3"
    assert (report["num_classes"], report["scored_classes"], report["fold"]) == (29, 29, None)
    assert report["label_dir"] == str(lab)
    assert report["label_set"]["sha256"] == lhe.sha256_file(lab / lhe.RELABEL_MANIFEST)
    fp = report["fingerprint"]
    assert fp["checkpoint"] == str(ckpt) and fp["ckpt_root"] is None and fp["cell_name"] == "main"
    assert fp["label_manifest_sha256"] == report["label_set"]["sha256"]
    [cell] = report["cells"]
    assert cell["cell"] == "main" and cell["per_class"]["28"]["iou"] == 1.0
    tiles_file = out.parent / cell["tile_confusions"]["file"]
    assert tiles_file.name == "main.main.tiles.npz"
    assert sorted(load_tile_confusions(tiles_file)) == ["t0", "t1", "t2"]
    # A second run must not overwrite the report it would replace.
    with pytest.raises(SystemExit):
        lhe.main()
    assert json.loads(out.read_text()) == report


def test_direct_mode_records_a_mismatched_checkpoint_as_failed(monkeypatch, tmp_path):
    hold, lab, labels = _tessera_tiles(tmp_path)
    ckpt = tmp_path / "best_model.pt"; ckpt.write_bytes(b"weights")
    out = tmp_path / "main.json"
    monkeypatch.setattr(lhe, "_load_infcmp", lambda: _backend(28, _perfect(labels)))
    monkeypatch.setattr(sys, "argv", _direct_argv(tmp_path, hold, lab, ckpt, out,
                                                  "--num-classes", "29", "--fold", "28:3"))
    with pytest.raises(SystemExit, match="no cell scored"):
        lhe.main()
    report = json.loads(out.read_text())
    assert "num_classes=28" in report["failed_cells"][0]["error"]
    assert report["fold"] == {"prediction": 28, "scored_as": 3} and report["label_set"] is None


_DIRECT = ["--checkpoint", "m.pt", "--cell-name", "main", "--models", "tessera"]


@pytest.mark.parametrize("extra,b28,message", [
    (_DIRECT, False, "--num-classes"),
    (["--checkpoint", "m.pt", "--num-classes", "29", "--models", "tessera"], True, "--cell-name"),
    (_DIRECT[:4] + ["--num-classes", "29", "--models", "tessera,clay"], True, "exactly one"),
    (["--checkpoint", "m.pt", "--cell-name", "../main", "--num-classes", "29",
      "--models", "tessera"], True, "--cell-name"),
    (_DIRECT + ["--num-classes", "29", "--rungs", "4"], True, "ladder grid"),
    (_DIRECT + ["--num-classes", "29", "--fold", "27:3"], False, "only 28"),
    (_DIRECT + ["--num-classes", "28", "--fold", "28:3"], False, "only 28"),
    (_DIRECT + ["--num-classes", "29"], False, "needs barrblandskog labels"),
    (_DIRECT + ["--num-classes", "29", "--fold", "28:3"], True, "original labels"),
    (_DIRECT + ["--num-classes", "28"], True, "original labels"),
    (["--num-classes", "29"], False, "28-class"),
    (["--fold", "28:3"], False, "--fold"),
    (["--cell-name", "main"], False, "--checkpoint"),
    (_DIRECT + ["--num-classes", "30"], True, "invalid choice"),
    ([], True, "original labels"),                       # the ladder grid is 28-class
])
def test_cli_refuses_incomplete_mixed_or_mislabelled_runs(monkeypatch, tmp_path, capsys,
                                                          extra, b28, message):
    hold, lab, ck = _fake_tree(tmp_path, [])
    if b28:
        _b28_manifest(lab)
    monkeypatch.setattr(lhe, "score_cell", lambda *a, **k: pytest.fail("scored"))
    argv = ["ladder_holdout_eval.py", "--holdout-dir", str(hold), "--label-dir", str(lab),
            "--out", str(tmp_path / "o.json"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        lhe.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


def test_direct_mode_needs_an_explicit_label_dir_and_out(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["ladder_holdout_eval.py", *_DIRECT, "--num-classes", "29"])
    with pytest.raises(SystemExit):
        lhe.main()
    assert "--label-dir, --out" in capsys.readouterr().err


def test_an_existing_report_is_only_continued_with_resume(monkeypatch, tmp_path, capsys):
    hold, lab, ck = _fake_tree(tmp_path, ["clay_r2"])
    out = tmp_path / "out.json"
    out.write_text("{}")
    monkeypatch.setattr(lhe, "score_cell", lambda *a, **k: pytest.fail("scored"))
    monkeypatch.setattr(sys, "argv", _args(tmp_path, hold=hold, lab=lab, ck=ck, out=out,
                                           extra=["--models", "clay", "--rungs", "2"]))
    with pytest.raises(SystemExit) as exc:
        lhe.main()
    assert exc.value.code == 2 and "--resume" in capsys.readouterr().err
    assert out.read_text() == "{}"


def test_bootstrap_point_estimate_matches_derive():
    """The bootstrap and the report must agree on what IoU is."""
    from imint.eval.tile_bootstrap import paired_iou_difference
    rng = np.random.default_rng(0)
    tiles = {f"t{i}": rng.integers(0, 50, (28, 28)) for i in range(6)}
    for v in tiles.values():
        v[0] = 0                                 # background truth, as compute_miou leaves it
    other = {k: v.copy() for k, v in tiles.items()}
    for v in other.values():                     # same truth (row sums), other errors
        v[:, [1, 2]] = v[:, [2, 1]]
    result = paired_iou_difference(tiles, other, draws=20, seed=1)
    pooled = lhe.derive(sum(tiles.values()))
    pooled_b = lhe.derive(sum(other.values()))
    for c in (1, 2, 5):
        assert round(result["per_class"][c]["a"], 4) == pooled["per_class"][c]["iou"]
        assert round(result["per_class"][c]["b"], 4) == pooled_b["per_class"][c]["iou"]
    assert round(result["mean_iou"]["a"], 4) == pooled["mean_iou"]
