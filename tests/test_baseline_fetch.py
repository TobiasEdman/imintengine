"""Tests for cloud-free baseline fetching (STAC-based)."""
from __future__ import annotations

import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from imint.fetch import (
    fetch_cloud_free_baseline,
    ensure_baseline,
    FetchResult,
    FetchError,
)


class TestFetchCloudFreeBaseline:
    """Verify the STAC+SCL+COT baseline selection logic."""

    @patch("imint.fetch.fetch_des_data")
    @patch("imint.fetch._fetch_scl")
    @patch("imint.fetch._connect")
    @patch("imint.fetch._stac_available_dates")
    def test_selects_least_cloudy_date(self, mock_stac, mock_connect, mock_fetch_scl, mock_fetch_des):
        """Should select the date with lowest AOI cloud fraction."""
        mock_stac.return_value = [
            ("2024-05-20", 10.0),
            ("2024-05-15", 15.0),
            ("2024-05-10", 20.0),
        ]
        mock_connect.return_value = MagicMock()

        # Three STAC dates screened via SCL: 0.5, 0.05, 0.3 AOI cloud
        mock_fetch_scl.side_effect = [
            (np.zeros((64, 64), dtype=np.uint8), 0.5, "EPSG:3006", "mock"),
            (np.zeros((64, 64), dtype=np.uint8), 0.05, "EPSG:3006", "mock"),
            (np.zeros((64, 64), dtype=np.uint8), 0.3, "EPSG:3006", "mock"),
        ]

        mock_fetch_des.return_value = FetchResult(
            bands={}, scl=None, cloud_fraction=0.05,
            rgb=np.zeros((64, 64, 3), dtype=np.float32),
        )

        result = fetch_cloud_free_baseline(
            date="2024-07-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
        )

        assert isinstance(result, FetchResult)
        assert mock_fetch_des.called
        # Final fetch should accept any cloud (already verified)
        call_args = mock_fetch_des.call_args
        assert call_args[1]["cloud_threshold"] == 1.0

    @patch("imint.fetch.fetch_des_data")
    @patch("imint.fetch._fetch_scl")
    @patch("imint.fetch._connect")
    @patch("imint.fetch._stac_available_dates")
    def test_stac_discovery_called_with_correct_window(self, mock_stac, mock_connect, mock_fetch_scl, mock_fetch_des):
        """STAC should be queried with the correct date window."""
        mock_stac.return_value = [("2024-06-01", 5.0)]
        mock_connect.return_value = MagicMock()

        mock_fetch_scl.return_value = (
            np.zeros((64, 64), dtype=np.uint8), 0.0, "EPSG:3006", "mock"
        )
        mock_fetch_des.return_value = FetchResult(
            bands={}, scl=None, cloud_fraction=0.0,
            rgb=np.zeros((64, 64, 3), dtype=np.float32),
        )

        fetch_cloud_free_baseline(
            date="2024-07-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
            search_start_days=30,
            search_end_days=90,
        )

        # STAC should have been called with start = 90 days back, end = 30 days back
        mock_stac.assert_called_once()
        call_args = mock_stac.call_args
        assert call_args[0][1] == "2024-04-16"  # 90 days back from 2024-07-15
        assert call_args[0][2] == "2024-06-15"  # 30 days back from 2024-07-15

    @patch("imint.fetch._stac_available_dates")
    def test_raises_when_stac_returns_empty(self, mock_stac):
        """Should raise FetchError if STAC finds no dates."""
        mock_stac.return_value = []

        with pytest.raises(FetchError, match="No Sentinel-2 data available"):
            fetch_cloud_free_baseline(
                date="2024-07-15",
                coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
            )

    @patch("imint.fetch._fetch_scl")
    @patch("imint.fetch._connect")
    @patch("imint.fetch._stac_available_dates")
    def test_raises_when_all_cloudy(self, mock_stac, mock_connect, mock_fetch_scl):
        """Should raise FetchError if no candidate is below threshold."""
        mock_stac.return_value = [
            ("2024-05-20", 30.0),
            ("2024-05-15", 40.0),
        ]
        mock_connect.return_value = MagicMock()

        mock_fetch_scl.return_value = (
            np.full((64, 64), 9, dtype=np.uint8), 0.9, "EPSG:3006", "mock"
        )

        with pytest.raises(FetchError, match="No cloud-free baseline found"):
            fetch_cloud_free_baseline(
                date="2024-07-15",
                coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
                cloud_threshold=0.1,
            )

    @patch("imint.fetch.fetch_des_data")
    @patch("imint.fetch._fetch_scl")
    @patch("imint.fetch._connect")
    @patch("imint.fetch._stac_available_dates")
    def test_skips_dates_with_no_data(self, mock_stac, mock_connect, mock_fetch_scl, mock_fetch_des):
        """Should skip candidate dates where SCL fetch fails."""
        mock_stac.return_value = [
            ("2024-05-20", 10.0),
            ("2024-05-15", 15.0),
            ("2024-05-10", 20.0),
        ]
        mock_connect.return_value = MagicMock()

        # First two fail, third succeeds with clear sky
        mock_fetch_scl.side_effect = [
            Exception("no data"),
            Exception("no data"),
            (np.zeros((64, 64), dtype=np.uint8), 0.02, "EPSG:3006", "mock"),
        ]

        mock_fetch_des.return_value = FetchResult(
            bands={}, scl=None, cloud_fraction=0.02,
            rgb=np.zeros((64, 64, 3), dtype=np.float32),
        )

        result = fetch_cloud_free_baseline(
            date="2024-07-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
        )

        assert isinstance(result, FetchResult)
        assert mock_fetch_scl.call_count == 3

    @patch("imint.fetch._fetch_scl")
    @patch("imint.fetch._connect")
    @patch("imint.fetch._stac_available_dates")
    def test_raises_when_all_fetches_fail(self, mock_stac, mock_connect, mock_fetch_scl):
        """Should raise FetchError if all SCL fetches fail (no data)."""
        mock_stac.return_value = [
            ("2024-05-20", 10.0),
            ("2024-05-15", 15.0),
        ]
        mock_connect.return_value = MagicMock()

        mock_fetch_scl.side_effect = Exception("no data")

        with pytest.raises(FetchError, match="All SCL fetches failed"):
            fetch_cloud_free_baseline(
                date="2024-07-15",
                coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
            )


class TestEnsureBaseline:
    """Verify the ensure_baseline() orchestration function."""

    def test_returns_existing_baseline_path(self, tmp_path):
        """Should return path if baseline .npy already exists."""
        output_dir = str(tmp_path / "outputs" / "2024-07-15")
        os.makedirs(output_dir, exist_ok=True)

        baseline_dir = os.path.join(output_dir, "..", "baselines")
        os.makedirs(baseline_dir, exist_ok=True)

        baseline_path = os.path.join(baseline_dir, "14.5_56.0_15.5_57.0.npy")
        np.save(baseline_path, np.zeros((64, 64, 3)))

        result = ensure_baseline(
            date="2024-07-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
            output_dir=output_dir,
        )

        assert result == baseline_path

    @patch("imint.fetch.fetch_cloud_free_baseline")
    def test_fetches_and_saves_when_missing(self, mock_fetch, tmp_path):
        """Should fetch and save baseline when .npy doesn't exist."""
        output_dir = str(tmp_path / "outputs" / "2024-07-15")
        os.makedirs(output_dir, exist_ok=True)

        mock_rgb = np.random.rand(64, 64, 3).astype(np.float32)
        mock_fetch.return_value = FetchResult(
            bands={}, scl=None, cloud_fraction=0.05,
            rgb=mock_rgb,
        )

        result = ensure_baseline(
            date="2024-07-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
            output_dir=output_dir,
        )

        assert result is not None
        assert os.path.exists(result)
        loaded = np.load(result)
        np.testing.assert_array_equal(loaded, mock_rgb)

    @patch("imint.fetch.fetch_cloud_free_baseline")
    def test_returns_none_on_fetch_failure(self, mock_fetch, tmp_path):
        """Should return None (not raise) if baseline fetch fails."""
        output_dir = str(tmp_path / "outputs" / "2024-07-15")
        os.makedirs(output_dir, exist_ok=True)

        mock_fetch.side_effect = FetchError("all dates cloudy")

        result = ensure_baseline(
            date="2024-07-15",
            coords={"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0},
            output_dir=output_dir,
        )

        assert result is None

    @patch("imint.fetch.fetch_cloud_free_baseline")
    def test_does_not_refetch_after_save(self, mock_fetch, tmp_path):
        """Second call should use cached baseline, not re-fetch."""
        output_dir = str(tmp_path / "outputs" / "2024-07-15")
        os.makedirs(output_dir, exist_ok=True)

        mock_rgb = np.random.rand(64, 64, 3).astype(np.float32)
        mock_fetch.return_value = FetchResult(
            bands={}, scl=None, cloud_fraction=0.05,
            rgb=mock_rgb,
        )

        coords = {"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0}

        # First call: fetches and saves
        result1 = ensure_baseline(date="2024-07-15", coords=coords, output_dir=output_dir)
        assert mock_fetch.call_count == 1

        # Second call: should use existing file
        result2 = ensure_baseline(date="2024-07-15", coords=coords, output_dir=output_dir)
        assert mock_fetch.call_count == 1  # Not called again
        assert result1 == result2
