"""Every consumer of the height channel handles its NaN nodata (Trädhöjd, 2026-10-22):
the LULCDataset loader and inference_preview fill it like UnifiedDataset, so no NaN
reaches the model and a missing pixel normalizes to the channel mean (0)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from imint.training.config import TrainingConfig  # noqa: E402
from imint.training.dataset import LULCDataset  # noqa: E402
from imint.training.unified_dataset import AUX_CHANNEL_NAMES, normalize_aux_channel  # noqa: E402


def _tile(path: Path, *, height: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = {"spectral": np.full((6, 256, 256), 1000.0, np.float32), "label": np.zeros((256, 256), np.int64)}
    if height is not None:
        keys["height"] = height
    np.savez_compressed(path, **keys)


@pytest.mark.parametrize("with_height", [True, False])
def test_lulc_dataset_height_is_finite_and_nan_is_the_mean(tmp_path, with_height):
    height = np.full((256, 256), 12.0, np.float32)
    height[:, :128] = np.nan                                  # the west half has no data
    _tile(tmp_path / "tiles/t.npz", height=height if with_height else None)
    (tmp_path / "split_val.txt").write_text("t.npz\n")
    cfg = TrainingConfig(enable_height_channel=True)
    item = LULCDataset(str(tmp_path), "val", cfg)[0]
    h = item["height"]
    assert torch.isfinite(h).all()
    mean, std = cfg.aux_norm["height"]
    if with_height:
        assert (h[0, :, :50] == 0).all()                      # NaN → mean → 0
        assert h[0, :, -50:].allclose(torch.tensor((12.0 - mean) / std))
    else:
        assert (h == 0).all()                                 # absent → mean, not "height 0"


def test_inference_preview_fills_like_training():
    from scripts.inference_preview import preview_aux_stack

    height = np.full((40, 40), 12.0, np.float32)
    height[:5] = np.nan
    stack = preview_aux_stack({"height": height}, 48, 48)     # padded to 48 x 48, other channels absent
    assert np.isfinite(stack).all()
    hi = AUX_CHANNEL_NAMES.index("height")
    expected = np.full((48, 48), np.nan, np.float32)
    expected[:40, :40] = height
    np.testing.assert_array_equal(stack[hi], normalize_aux_channel("height", expected))
    assert (stack[hi][:5] == 0).all() and (stack[hi][40:] == 0).all()
