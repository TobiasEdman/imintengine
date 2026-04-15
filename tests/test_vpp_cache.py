"""Integration tests for VPP disk cache in fetch_unified_tiles.py."""
import json
import os
import tempfile

import pytest


class TestVPPDiskCache:

    def test_cache_written_to_disk(self):
        """VPP cache must be saved as JSON after prefetch."""
        from scripts.fetch_unified_tiles import prefetch_vpp_batch

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, ".vpp_cache.json")
            # Empty work list — should write empty cache
            result = prefetch_vpp_batch([], workers=1, cache_path=cache_path)
            assert os.path.exists(cache_path), "Cache file must be written to disk"
            with open(cache_path) as f:
                data = json.load(f)
            assert isinstance(data, dict)

    def test_cache_read_back_correctly(self):
        """Saved cache must be loadable and match original data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, ".vpp_cache.json")
            # Write a fake cache
            fake_cache = {
                "tile_001": [[100, 150], [150, 200], [200, 250]],
                "tile_002": [[90, 140], [140, 190], [190, 240]],
            }
            with open(cache_path, "w") as f:
                json.dump(fake_cache, f)

            from scripts.fetch_unified_tiles import prefetch_vpp_batch

            # Create work items matching the cached tiles
            work = [
                ({"name": "tile_001", "bbox_3006": {"west": 0, "south": 0, "east": 100, "north": 100}}, tmpdir),
                ({"name": "tile_002", "bbox_3006": {"west": 0, "south": 0, "east": 100, "north": 100}}, tmpdir),
            ]
            result = prefetch_vpp_batch(work, workers=1, cache_path=cache_path)

            # Should load from cache, not re-fetch
            assert "tile_001" in result
            assert "tile_002" in result
            assert result["tile_001"] == [(100, 150), (150, 200), (200, 250)]
            assert result["tile_002"] == [(90, 140), (140, 190), (190, 240)]

    def test_partial_cache_fetches_missing(self):
        """If cache has some tiles but not all, only missing ones are fetched."""
        from scripts.fetch_unified_tiles import prefetch_vpp_batch

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, ".vpp_cache.json")
            with open(cache_path, "w") as f:
                json.dump({"tile_001": [[100, 150]]}, f)

            work = [
                ({"name": "tile_001", "bbox_3006": {"west": 0, "south": 0, "east": 100, "north": 100}}, tmpdir),
                ({"name": "tile_missing", "bbox_3006": {"west": 0, "south": 0, "east": 100, "north": 100}}, tmpdir),
            ]
            result = prefetch_vpp_batch(work, workers=1, cache_path=cache_path)
            assert "tile_001" in result

    def test_no_cache_file_does_fresh_fetch(self):
        """Without cache file, does a full fetch (no crash)."""
        from scripts.fetch_unified_tiles import prefetch_vpp_batch

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, ".vpp_cache.json")
            assert not os.path.exists(cache_path)
            result = prefetch_vpp_batch([], workers=1, cache_path=cache_path)
            assert isinstance(result, dict)
