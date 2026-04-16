"""Tests for batch_fetch_openeo.py — two-stage SCL screening + spectral fetch."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys_path_set = False


def _ensure_path():
    global sys_path_set
    if not sys_path_set:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        sys_path_set = True


class TestScreenTileSCL:
    """Test the per-tile SCL screening logic."""

    def _make_mock_conn(self, execute_returns):
        """Create a mock openEO connection.

        execute_returns: dict (CDSE format) or list of dicts (per-frame).
        """
        mock_conn = MagicMock()
        mock_cube = MagicMock()
        mock_conn.load_collection.return_value = mock_cube
        mock_cube.band.return_value = mock_cube
        mock_cube.__eq__ = MagicMock(return_value=mock_cube)
        mock_cube.__or__ = MagicMock(return_value=mock_cube)
        mock_cube.aggregate_spatial.return_value = mock_cube
        if isinstance(execute_returns, list):
            mock_cube.execute.side_effect = execute_returns
        else:
            mock_cube.execute.return_value = execute_returns
        return mock_conn

    def test_returns_dict_with_frame_keys(self):
        _ensure_path()
        from scripts.batch_fetch_openeo import screen_tile_scl

        # Real CDSE format: {"date": [[cloud_frac]]}
        mock_conn = self._make_mock_conn({
            "2022-07-01T00:00:00Z": [[0.05]],
            "2022-07-06T00:00:00Z": [[0.15]],
            "2022-07-11T00:00:00Z": [[0.02]],
        })

        tile = {
            "name": "test_tile",
            "bbox_3006": {"west": 500000, "south": 6500000,
                          "east": 505120, "north": 6505120},
        }
        windows = [(150, 210)]

        with patch("scripts.batch_fetch_openeo.bbox_to_wgs84",
                   return_value={"west": 15.0, "south": 58.0,
                                  "east": 16.0, "north": 59.0}):
            result = screen_tile_scl(mock_conn, tile, windows, 2022)

        assert "0" in result
        assert result["0"]["date"] == "2022-07-11"
        assert result["0"]["cloud_frac"] == 0.02

    def test_picks_lowest_cloud_fraction(self):
        _ensure_path()
        from scripts.batch_fetch_openeo import screen_tile_scl

        mock_conn = self._make_mock_conn({
            "2022-06-01T00:00:00Z": [[0.40]],
            "2022-06-10T00:00:00Z": [[0.01]],
            "2022-06-20T00:00:00Z": [[0.30]],
        })

        tile = {
            "name": "tile_cloud_test",
            "bbox_3006": {"west": 500000, "south": 6500000,
                          "east": 505120, "north": 6505120},
        }

        with patch("scripts.batch_fetch_openeo.bbox_to_wgs84",
                   return_value={"west": 15.0, "south": 58.0,
                                  "east": 16.0, "north": 59.0}):
            result = screen_tile_scl(mock_conn, tile, [(150, 180)], 2022)

        assert result["0"]["date"] == "2022-06-10"
        assert result["0"]["cloud_frac"] == 0.01

    def test_multiple_frames(self):
        _ensure_path()
        from scripts.batch_fetch_openeo import screen_tile_scl

        # Single call returns all scenes across the full season
        mock_conn = self._make_mock_conn({
            "2022-05-01T00:00:00Z": [[0.10]],  # DOY 121 → frame 0 (120-150)
            "2022-05-15T00:00:00Z": [[0.05]],  # DOY 135 → frame 0
            "2022-07-01T00:00:00Z": [[0.03]],  # DOY 182 → frame 1 (170-200)
            "2022-07-10T00:00:00Z": [[0.20]],  # DOY 191 → frame 1
            "2022-08-15T00:00:00Z": [[0.08]],  # DOY 227 → frame 2 (220-250)
        })

        tile = {
            "name": "tile_multi",
            "bbox_3006": {"west": 500000, "south": 6500000,
                          "east": 505120, "north": 6505120},
        }
        windows = [(120, 150), (170, 200), (220, 250)]

        with patch("scripts.batch_fetch_openeo.bbox_to_wgs84",
                   return_value={"west": 15.0, "south": 58.0,
                                  "east": 16.0, "north": 59.0}):
            result = screen_tile_scl(mock_conn, tile, windows, 2022)

        assert result["0"]["date"] == "2022-05-15"
        assert result["1"]["date"] == "2022-07-01"
        assert result["2"]["date"] == "2022-08-15"

    def test_handles_empty_response(self):
        _ensure_path()
        from scripts.batch_fetch_openeo import screen_tile_scl

        mock_conn = self._make_mock_conn({})

        tile = {
            "name": "tile_empty",
            "bbox_3006": {"west": 500000, "south": 6500000,
                          "east": 505120, "north": 6505120},
        }

        with patch("scripts.batch_fetch_openeo.bbox_to_wgs84",
                   return_value={"west": 15.0, "south": 58.0,
                                  "east": 16.0, "north": 59.0}):
            result = screen_tile_scl(mock_conn, tile, [(150, 210)], 2022)

        assert result == {}

    def test_handles_exception(self):
        _ensure_path()
        from scripts.batch_fetch_openeo import screen_tile_scl

        mock_conn = MagicMock()
        mock_conn.load_collection.side_effect = Exception("API error")

        tile = {
            "name": "tile_error",
            "bbox_3006": {"west": 500000, "south": 6500000,
                          "east": 505120, "north": 6505120},
        }

        with patch("scripts.batch_fetch_openeo.bbox_to_wgs84",
                   return_value={"west": 15.0, "south": 58.0,
                                  "east": 16.0, "north": 59.0}):
            result = screen_tile_scl(mock_conn, tile, [(150, 210)], 2022)

        assert result == {}


class TestMergeToNpz:

    def test_merge_creates_npz(self):
        _ensure_path()
        from scripts.batch_fetch_openeo import _merge_to_npz

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # Create 2 frame files
            for fi in range(2):
                np.save(output_dir / f"_frame_test_tile_f{fi}.npy",
                        np.random.rand(6, 64, 64).astype(np.float32))

            tile_loc = {
                "bbox_3006": {"west": 500000, "south": 6500000,
                              "east": 500640, "north": 6500640},
                "source": "lulc",
            }
            ok = _merge_to_npz("test_tile", 2, output_dir, tile_loc, 64)
            assert ok

            # Verify .npz
            npz = np.load(output_dir / "test_tile.npz", allow_pickle=True)
            assert npz["spectral"].shape == (12, 64, 64)  # 2 frames × 6 bands
            assert npz["temporal_mask"].tolist() == [1, 1]
            assert int(npz["num_frames"]) == 2

            # Frame files cleaned up
            assert not (output_dir / "_frame_test_tile_f0.npy").exists()
            assert not (output_dir / "_frame_test_tile_f1.npy").exists()

    def test_merge_with_missing_frame(self):
        _ensure_path()
        from scripts.batch_fetch_openeo import _merge_to_npz

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # Only frame 0, frame 1 missing
            np.save(output_dir / "_frame_partial_f0.npy",
                    np.random.rand(6, 32, 32).astype(np.float32))

            tile_loc = {
                "bbox_3006": {"west": 0, "south": 0, "east": 320, "north": 320},
                "source": "lulc",
            }
            ok = _merge_to_npz("partial", 2, output_dir, tile_loc, 32)
            assert ok

            npz = np.load(output_dir / "partial.npz", allow_pickle=True)
            assert npz["temporal_mask"].tolist() == [1, 0]

    def test_merge_no_frames_returns_false(self):
        _ensure_path()
        from scripts.batch_fetch_openeo import _merge_to_npz

        with tempfile.TemporaryDirectory() as tmpdir:
            tile_loc = {
                "bbox_3006": {"west": 0, "south": 0, "east": 320, "north": 320},
                "source": "lulc",
            }
            ok = _merge_to_npz("missing", 2, Path(tmpdir), tile_loc, 32)
            assert not ok


class TestBestDatesResumability:

    def test_saves_and_loads_best_dates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best_dates.json"

            data = {
                "tile_001": {"0": {"date": "2022-07-01", "cloud_frac": 0.03}},
                "tile_002": {"0": {"date": "2022-07-05", "cloud_frac": 0.08}},
            }
            with open(path, "w") as f:
                json.dump(data, f)

            with open(path) as f:
                loaded = json.load(f)

            assert loaded["tile_001"]["0"]["date"] == "2022-07-01"
            assert loaded["tile_002"]["0"]["cloud_frac"] == 0.08
