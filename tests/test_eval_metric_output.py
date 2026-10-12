"""Exercise phase-one metric persistence through the public evaluation entry point."""
import json
from pathlib import Path

import numpy as np
import pytest

from imint.eval.eval_in_distribution import run


def test_run_persists_numeric_tile_scores_and_percentiles(tmp_path):
    tiles = tmp_path / "tiles"
    split = tmp_path / "split"
    output = tmp_path / "output"
    tiles.mkdir()
    split.mkdir()
    # Unequal supervised populations distinguish global IoU from tile percentiles.
    targets = [np.array([[1, 0], [0, 0]]), np.array([[1, 1], [2, 0]])]
    predictions = [np.array([[1, 0], [0, 0]]), np.array([[0, 1], [2, 0]])]
    names = ["perfect", "partial"]
    for name, target, pred in zip(names, targets, predictions):
        target = np.repeat(np.repeat(target, 8, axis=0), 8, axis=1)
        pred = np.repeat(np.repeat(pred, 8, axis=0), 8, axis=1)
        # Synthetic class evidence, consumed by a deterministic CPU predictor.
        evidence = np.eye(23, dtype=np.float32)[pred].transpose(2, 0, 1)
        np.savez(tiles / f"{name}.npz", label=target, evidence=evidence)
    np.savez(tiles / "ignored.npz", label=np.zeros((16, 16), dtype=int),
             evidence=np.zeros((23, 16, 16), dtype=np.float32))
    (split / "test.txt").write_text("perfect\npartial\nignored\n")

    def factory(checkpoint):
        assert checkpoint == Path("synthetic")

        def predict(path):
            with np.load(path) as data:
                evidence = data["evidence"]
            return evidence.argmax(axis=0), evidence

        return predict

    result = run(Path("synthetic"), str(tiles), split, model_factory=factory,
                 output_dir=output, num_classes=23, ece_pixel_sample=64,
                 device="cpu")["ensemble"]
    rows = json.loads((output / "per_tile_phase_1_ensemble.json").read_text())
    summary_text = (output / "phase_1_in_distribution.json").read_text()
    summary = json.loads(summary_text)["ensemble"]
    assert [r["tile"] for r in rows] == names
    assert [r["n_pix"] for r in rows] == [64, 192]
    assert all(type(r["mIoU"]) is float for r in rows)
    assert [r["mIoU"] for r in rows] == [1.0, 0.75]
    for percentile, expected in [(10, 0.775), (50, 0.875), (90, 0.975)]:
        key = f"per_tile_mIoU_p{percentile}"
        assert summary["metrics"][key] == pytest.approx(expected)
        assert result.metrics[key] == pytest.approx(expected)
    assert summary["num_tiles"] == result.num_tiles == 2
    assert summary["num_pixels"] == result.num_pixels == 256
    assert summary["metrics"]["mIoU"] == pytest.approx(5 / 6)
    assert summary["metrics"]["accuracy"] == 0.75
    assert summary["metrics"]["ECE_n_pixels_sampled"] > 0
    expected_cm = np.zeros((23, 23), dtype=np.int64)
    expected_cm[1, 1], expected_cm[1, 0], expected_cm[2, 2] = 128, 64, 64
    np.testing.assert_array_equal(np.load(output / "confusion_phase_1_ensemble.npy"),
                                  expected_cm)
    np.testing.assert_array_equal(result.confusion_matrix, expected_cm)
    # Preserve the existing non-strict JSON contract for absent-class IoUs.
    assert "NaN" in summary_text
    assert np.isnan(summary["per_class"]["3"])
    assert np.isnan(result.per_class["3"])
