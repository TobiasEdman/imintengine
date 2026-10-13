"""Ignored labels must not influence checkpoint selection or evaluation."""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from imint.metrics import compute_miou
from imint.eval.metrics import per_class_iou
from imint.schema.unified_schema import UNIFIED_CLASSES


@pytest.mark.parametrize("ignore_index", [0, 2, -1, 255])
def test_predictions_on_ignored_targets_do_not_change_metrics(ignore_index):
    target = np.array([[1, ignore_index], [ignore_index, ignore_index]])
    for ignored_prediction in (0, 1, 2):
        pred = np.full(target.shape, ignored_prediction)
        pred[0, 0] = 1
        result = compute_miou(pred, target, 23, ignore_index)
        expected = np.zeros((23, 23), dtype=np.int64)
        expected[1, 1] = 1
        np.testing.assert_array_equal(result["confusion_matrix"], expected)
        assert result["miou"] == result["overall_accuracy"] == 1.0
        assert result["per_class_iou"][UNIFIED_CLASSES[1]] == 1.0
        assert all(np.isnan(iou) for name, iou in result["per_class_iou"].items()
                   if name != UNIFIED_CLASSES[1])


def test_background_predictions_remain_errors_on_valid_targets():
    result = compute_miou(np.array([1, 0, 2, 1]), np.array([1, 1, 2, 0]), 23)
    expected = np.zeros((23, 23), dtype=np.int64)
    expected[1, 1] = expected[1, 0] = expected[2, 2] = 1
    np.testing.assert_array_equal(result["confusion_matrix"], expected)
    assert result["miou"] == 0.75
    assert result["overall_accuracy"] == 0.6667
    assert result["per_class_iou"][UNIFIED_CLASSES[1]] == 0.5
    assert result["per_class_iou"][UNIFIED_CLASSES[2]] == 1.0


@pytest.mark.parametrize("bad_prediction", [-1, 23, 255, np.nan, 1.5])
def test_invalid_prediction_on_scored_target_is_rejected(bad_prediction):
    with pytest.raises(ValueError, match="predictions on valid targets"):
        compute_miou(np.array([1, bad_prediction]), np.array([1, 1]), 23)


def test_invalid_predictions_on_excluded_targets_are_ignored():
    result = compute_miou(np.array([1, 255, -1, 255]), np.array([1, 0, -1, 23]), 23)
    assert result["confusion_matrix"].sum() == 1
    assert result["miou"] == result["overall_accuracy"] == 1.0


@pytest.mark.parametrize("bad_prediction", [-1, 255])
def test_global_evaluation_rejects_invalid_prediction(tmp_path, bad_prediction):
    from imint.eval.eval_in_distribution import _eval_one_predictor

    np.savez(tmp_path / "tile.npz", label=np.array([[1, 1]]))
    with pytest.raises(ValueError, match="predictions on valid targets"):
        _eval_one_predictor(
            name="invalid", predict_fn=lambda path: (np.array([[1, bad_prediction]]), None),
            tile_names=["tile"], tiles_dir=str(tmp_path), num_classes=23,
            ignore_index=0, ece_pixel_sample=0, output_dir=tmp_path,
        )
    assert not (tmp_path / "confusion_phase_1_invalid.npy").exists()


@pytest.mark.parametrize("target,pred", [([], []), ([0, 0], [1, 2]), ([-1, 23], [1, 2])])
def test_no_valid_pairs_returns_zero_scores_and_absent_ious(target, pred):
    result = compute_miou(np.array(pred, dtype=int), np.array(target, dtype=int), 23)
    assert not result["confusion_matrix"].any()
    assert result["miou"] == result["overall_accuracy"] == 0.0
    assert all(np.isnan(iou) for iou in result["per_class_iou"].values())


@pytest.mark.parametrize("ignored_prediction", [0, 1, 2])
def test_evaluate_model_matches_global_and_per_tile_metrics(tmp_path, ignored_prediction):
    torch = pytest.importorskip("torch")
    from imint.training.evaluate import evaluate_model
    from imint.eval.eval_in_distribution import _eval_one_predictor

    # Unequal valid populations distinguish pooled IoU from mean tile IoU.
    targets = [np.array([[1, 0], [0, 0]]), np.array([[1, 1], [2, 0]])]
    predictions = [np.array([[1, ignored_prediction], [ignored_prediction, ignored_prediction]]),
                   np.array([[0, 1], [2, ignored_prediction]])]

    class FixedLogits(torch.nn.Module):
        fm_spec = SimpleNamespace(family="tessera")

        def forward(self, x, **kwargs):
            assert not self.training
            assert not torch.is_grad_enabled()
            return x[:, :23]

    samples = []
    names = ["a", "b"]
    by_name = {}
    for name, target, pred in zip(names, targets, predictions):
        logits = torch.nn.functional.one_hot(torch.as_tensor(pred), 23).permute(2, 0, 1).float()
        spectral = torch.zeros(128, 2, 2)
        spectral[:23] = logits
        samples.append({"spectral": spectral, "label": torch.as_tensor(target)})
        np.savez(tmp_path / f"{name}.npz", label=target)
        by_name[name] = pred

    config = SimpleNamespace(num_classes=23, ignore_index=0, enabled_aux_names=())
    training = evaluate_model(FixedLogits(), samples, config, torch.device("cpu"))
    tile_metrics = [per_class_iou(p, t, 23) for p, t in zip(predictions, targets)]
    pooled = compute_miou(np.stack(predictions), np.stack(targets), 23)
    global_result = _eval_one_predictor(
        name="fixed", predict_fn=lambda p: (by_name[Path(p).stem], None),
        tile_names=names, tiles_dir=str(tmp_path), num_classes=23,
        ignore_index=0, ece_pixel_sample=0, output_dir=tmp_path,
    )
    expected = np.zeros((23, 23), dtype=np.int64)
    expected[1, 0], expected[1, 1], expected[2, 2] = 1, 2, 1
    for result in (training, pooled):
        np.testing.assert_array_equal(result["confusion_matrix"], expected)
        assert result["miou"] == 0.8333
        assert result["overall_accuracy"] == 0.75
    np.testing.assert_array_equal(sum(m["confusion_matrix"] for m in tile_metrics), expected)
    assert [m["miou"] for m in tile_metrics] == [1.0, 0.75]
    np.testing.assert_array_equal(global_result.confusion_matrix, expected)
    np.testing.assert_array_equal(np.load(tmp_path / "confusion_phase_1_fixed.npy"), expected)
    assert global_result.metrics["mIoU"] == pytest.approx(5 / 6)
    assert global_result.metrics["accuracy"] == 0.75
    assert global_result.num_pixels == 4



def test_fractional_target_is_not_truncated_into_a_class():
    with pytest.raises(ValueError, match="valid targets must be integer"):
        compute_miou(np.array([1]), np.array([1.5]), 23)
