"""Real NPZ preprocessing must use training's year and frame conventions."""
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import inference_comparison as inference
from imint.training.unified_dataset import UnifiedDataset, PRITHVI_MEAN, PRITHVI_STD


def write_tile(tmp_path: Path, bands=24, **overrides):
    data = {
        "spectral": np.full((bands, 8, 8), .1, np.float32),
        "dates": np.array(["2023-10-07", "2024-05-29", "2024-06-28", "2024-07-28"]),
        "doy": np.array([280.9, 150.7, 180.2, 210.9]),
        "easting": np.array(500000.), "northing": np.array(6500000.),
    }
    data.update(overrides)
    path = tmp_path / "tile.npz"
    np.savez(path, **data)
    return path


def build(path, frames):
    return inference._build_inference_inputs(
        path, torch.device("cpu"), 8, [], family="prithvi", num_frames=frames)


def test_date_only_year_and_prior_autumn_match_training(tmp_path):
    path = write_tile(tmp_path)
    inputs = build(path, 4)
    try:
        actual = inputs["temporal_coords"]
        torch.testing.assert_close(
            actual, torch.tensor([[[2023.,280.], [2024.,150.],
                                   [2024.,180.], [2024.,210.]]]))
        with np.load(path, allow_pickle=False) as data:
            expected_time, expected_location = UnifiedDataset._build_coords(
                data, data["doy"].astype(np.int32), 4)
        torch.testing.assert_close(actual[0], expected_time)
        torch.testing.assert_close(inputs["location_coords"][0], expected_location)
    finally:
        inference._close_inference_inputs(inputs)


@pytest.mark.parametrize("bands", [6, 24])
def test_one_frame_uses_growing_year_and_zero_doy(tmp_path, bands):
    inputs = build(write_tile(tmp_path, bands=bands), 1)
    try:
        assert inputs["img5d"].shape == (1, 6, 1, 8, 8)
        torch.testing.assert_close(inputs["temporal_coords"], torch.tensor([[[2024.,0.]]]))
    finally:
        inference._close_inference_inputs(inputs)


def test_conflicting_years_never_use_default_or_explicit_override(tmp_path):
    with pytest.raises(ValueError, match="disagrees"):
        build(write_tile(tmp_path, year=2022), 4)


def test_unresolvable_year_never_defaults_to_2022(tmp_path):
    with pytest.raises(ValueError, match="resolved tile year"):
        build(write_tile(tmp_path, dates=np.array(["unknown"] * 4)), 4)


def test_unfrozen_missing_doy_retains_existing_no_coordinates_path(tmp_path):
    path = tmp_path / "tile.npz"
    np.savez(path, spectral=np.ones((24,8,8)), year=2024)
    inputs = build(path, 4)
    try:
        assert inputs["temporal_coords"] is None
        assert inputs["location_coords"] is None
    finally:
        inference._close_inference_inputs(inputs)


@pytest.mark.parametrize("frames", [None, 4])
@pytest.mark.parametrize("mask", [
    [0, 1, 1, 1],  # Missing autumn: the campaign's observed case.
    [1, 0, 1, 1],  # Equidistant: training selects the earlier valid frame.
    [1, 0, 0, 1],
    [1, 1, 1, 1],
])
def test_masked_spectral_frames_match_training_before_normalization(tmp_path, mask, frames):
    spectral = np.stack([
        np.full((8, 8), (channel + 1) / 100, dtype=np.float32)
        for channel in range(24)
    ])
    for frame, valid in enumerate(mask):
        if not valid:
            spectral[frame * 6:(frame + 1) * 6] = 0
    path = write_tile(tmp_path, spectral=spectral, temporal_mask=np.array(mask))
    inputs = build(path, frames)
    try:
        with np.load(path, allow_pickle=False) as tile:
            image, actual_mask, doy = UnifiedDataset._extract_all_frames(tile, "lulc", 4)
            expected_time, _ = UnifiedDataset._build_coords(tile, doy, 4)
        mean = np.tile(PRITHVI_MEAN.reshape(6, 1, 1), (4, 1, 1))
        std = np.tile(PRITHVI_STD.reshape(6, 1, 1), (4, 1, 1))
        expected = ((image * 10000.0 - mean) / std).reshape(4, 6, 8, 8).transpose(1, 0, 2, 3)
        np.testing.assert_array_equal(inputs["img5d"][0].numpy(), expected)
        np.testing.assert_array_equal(actual_mask, mask)
        torch.testing.assert_close(inputs["temporal_coords"][0], expected_time)
        # The original NPZ remains unchanged by replacement in memory.
        with np.load(path, allow_pickle=False) as tile:
            np.testing.assert_array_equal(tile["spectral"], spectral)
    finally:
        inference._close_inference_inputs(inputs)
