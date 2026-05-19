"""Tests for the WEkEO HR-VPP fallback (imint.training.wekeo_vpp).

Endpoint-independent: no WEkEO network access. The HDA download path
(prefetch_vpp_cogs) is exercised by the k8s job verify-wekeo-vpp; here
we test the filename parser, the overlap test, and the local COG reader
(fetch_vpp_tiles_local) against synthetic GeoTIFFs.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from imint.training import wekeo_vpp


# ── filename parser ──────────────────────────────────────────────────────

class TestParseVppFilename:

    def test_parses_valid_name(self):
        meta = wekeo_vpp._parse_vpp_filename(
            "VPP_2021_S2_33VWJ-010m_V101_s1_SOSD.tif"
        )
        assert meta == {
            "metric": "SOSD", "tileId": "33VWJ", "year": 2021, "season": 1,
        }

    def test_parses_season_2_and_length(self):
        meta = wekeo_vpp._parse_vpp_filename(
            "VPP_2019_S2_32TPN-010m_V102_s2_LENGTH.tif"
        )
        assert meta["metric"] == "LENGTH"
        assert meta["season"] == 2
        assert meta["year"] == 2019

    def test_parses_id_form_without_extension(self):
        # hda result `id` carries the stem without the .tif extension.
        meta = wekeo_vpp._parse_vpp_filename(
            "VPP_2021_S2_T33VWJ-010m_V101_s1_SOSD"
        )
        assert meta is not None
        assert meta["metric"] == "SOSD"
        assert meta["year"] == 2021

    def test_rejects_garbage(self):
        assert wekeo_vpp._parse_vpp_filename("not_a_vpp_file.tif") is None
        assert wekeo_vpp._parse_vpp_filename("VPP_2021_S2_33VWJ.txt") is None


class _StubResult:
    """Stand-in for an iterated hda SearchResults item."""

    def __init__(self, results):
        self.results = results


class TestResultFilename:

    def test_extracts_basename_from_location(self):
        r = _StubResult([{
            "id": "VPP_2021_S2_T33VWJ-010m_V101_s1_SOSD",
            "properties": {
                "location": "s3://hr-vpp-products-vpp-v01-2021/CLMS/"
                            "Pan-European/Biophysical/VPP/v01/2021/s1/"
                            "VPP_2021_S2_T33VWJ-010m_V101_s1_SOSD.tif",
            },
        }])
        assert wekeo_vpp._result_filename(r) == (
            "VPP_2021_S2_T33VWJ-010m_V101_s1_SOSD.tif"
        )

    def test_falls_back_to_id_when_no_location(self):
        r = _StubResult([{
            "id": "VPP_2021_S2_T33VWJ-010m_V101_s1_SOSD",
            "properties": {},
        }])
        assert wekeo_vpp._result_filename(r) == (
            "VPP_2021_S2_T33VWJ-010m_V101_s1_SOSD"
        )


# ── bounds overlap ───────────────────────────────────────────────────────

class TestBoundsOverlap:

    def test_overlapping(self):
        assert wekeo_vpp._bounds_overlap([0, 0, 10, 10], (5, 5, 15, 15))

    def test_disjoint(self):
        assert not wekeo_vpp._bounds_overlap([0, 0, 10, 10], (20, 20, 30, 30))

    def test_edge_touch_is_not_overlap(self):
        assert not wekeo_vpp._bounds_overlap([0, 0, 10, 10], (10, 0, 20, 10))


# ── local COG reader ─────────────────────────────────────────────────────

def _write_cog(path: Path, value: float, *, utm_bounds, crs="EPSG:32633"):
    """Write a constant-valued single-band GeoTIFF (synthetic VPP COG)."""
    import rasterio
    from rasterio.transform import from_bounds

    h = w = 200
    w0, s0, e0, n0 = utm_bounds
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="float32", crs=crs, transform=from_bounds(w0, s0, e0, n0, w, h),
        nodata=0,
    ) as dst:
        dst.write(np.full((h, w), value, dtype=np.float32), 1)


class TestFetchVppTilesLocal:

    # Synthetic VPP product values — chosen so the scaling is observable.
    _VALUES = {
        "SOSD": 120.0, "EOSD": 280.0, "LENGTH": 160.0,
        "MAXV": 12000.0, "MINV": 3000.0,
    }

    def _build_cache(self, tmp: Path):
        """Populate tmp with 5 synthetic COGs + index.json; return bbox_3006."""
        import rasterio
        from rasterio.warp import transform_bounds

        # UTM 33N extent in Sweden (~lon 15, lat 59.5).
        utm_bounds = (500_000.0, 6_600_000.0, 520_000.0, 6_620_000.0)
        index = {}
        for metric, value in self._VALUES.items():
            fname = f"VPP_2021_S2_33VWJ-010m_V101_s1_{metric}.tif"
            _write_cog(tmp / fname, value, utm_bounds=utm_bounds)
            with rasterio.open(tmp / fname) as src:
                b4326 = list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))
            index[fname] = {
                "metric": metric, "tileId": "33VWJ", "year": 2021,
                "season": 1, "bounds_4326": b4326,
            }
        (tmp / "index.json").write_text(json.dumps(index))

        # A 3006 bbox well inside the COG footprint.
        bbox_3006 = transform_bounds(
            "EPSG:32633", "EPSG:3006",
            505_000, 6_605_000, 515_000, 6_615_000,
        )
        return bbox_3006

    def test_contract_and_scaling(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            w, s, e, n = self._build_cache(tmp)
            out = wekeo_vpp.fetch_vpp_tiles_local(
                w, s, e, n, size_px=64, vpp_cog_dir=tmp, year=2021,
            )
            assert set(out) == {"sosd", "eosd", "length", "maxv", "minv"}
            for band, arr in out.items():
                assert arr.shape == (64, 64), band
                assert arr.dtype == np.float32, band

            # DOY/day bands unscaled; PPI bands scaled by 0.0001.
            assert np.allclose(out["sosd"], 120.0)
            assert np.allclose(out["length"], 160.0)
            assert np.allclose(out["maxv"], 1.2)     # 12000 * 0.0001
            assert np.allclose(out["minv"], 0.3)     # 3000 * 0.0001

    def test_missing_index_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(FileNotFoundError):
                wekeo_vpp.fetch_vpp_tiles_local(
                    0, 0, 1, 1, size_px=32, vpp_cog_dir=d, year=2021,
                )

    def test_disjoint_bbox_returns_zeros(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._build_cache(tmp)
            # A 3006 bbox far from the COG footprint → no overlap → zeros.
            out = wekeo_vpp.fetch_vpp_tiles_local(
                0.0, 0.0, 1000.0, 1000.0,
                size_px=32, vpp_cog_dir=tmp, year=2021,
            )
            for arr in out.values():
                assert np.all(arr == 0.0)


# ── CDSE fallback wiring ─────────────────────────────────────────────────

class TestFallbackWiring:

    def test_fallback_reraises_without_cache(self, monkeypatch):
        """_fallback_to_wekeo must re-raise the CDSE error if no cache."""
        from imint.training import cdse_vpp

        with tempfile.TemporaryDirectory() as d:
            monkeypatch.setenv("VPP_WEKEO_DIR", d)  # empty — no index.json
            original = RuntimeError("CDSE quota exhausted")
            with pytest.raises(RuntimeError, match="quota exhausted"):
                cdse_vpp._fallback_to_wekeo(
                    0, 0, 1, 1, size_px=(32, 32), year=2021,
                    cdse_error=original,
                )

    def test_vpp_source_wekeo_skips_cdse(self, monkeypatch):
        """VPP_SOURCE=wekeo routes fetch_vpp_tiles straight to WEkEO.

        With no cache present the WEkEO path re-raises its own marker
        error — reaching it proves CDSE was skipped entirely (no CDSE
        credentials are set, so a CDSE attempt would fail differently).
        """
        from imint.training import cdse_vpp

        with tempfile.TemporaryDirectory() as d:
            monkeypatch.setenv("VPP_SOURCE", "wekeo")
            monkeypatch.setenv("VPP_WEKEO_DIR", d)  # empty — no index.json
            with pytest.raises(RuntimeError, match="CDSE skipped"):
                cdse_vpp.fetch_vpp_tiles(0.0, 0.0, 1.0, 1.0, size_px=32)
