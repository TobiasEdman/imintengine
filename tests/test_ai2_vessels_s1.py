"""Tests for the AI2 Sentinel-1 vessel port: fetch, analyzer logic, GeoJSON.

Pure-logic and offline fetch tests always run (catalogue searches are faked,
rasters are synthetic GCP-referenced GRDs on disk). The model test runs on a
192 px crop of AI2's own model input around one reference detection
(tests/fixtures/ai2_s1_vessel_crop.npz) and skips when the checkpoints in
imint/fm/ai2_vessels_s1/ are absent. Reference numbers come from
docs/marine/s1_vessels_reference.md.
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import timedelta
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.transform import Affine

from imint import fetch, geoid
from imint.analyzers import ai2_vessels_s1 as s1
from imint.analyzers.base import AnalysisResult
from imint.fetch import S1_HISTORICAL_BAND_KEYS, S1_TARGET_BAND_KEYS
from imint.job import GeoContext
from imint.data import mpc_s1
from tests.s1_synthetic import sea_surface_copy, synthetic_grd

FIXTURE = Path(__file__).parent / "fixtures" / "ai2_s1_vessel_crop.npz"
# Skagen target scene footprint (MPC item == CDSE OData GeoFootprint).
SKAGEN_FOOTPRINT = {"type": "Polygon", "coordinates": [[
    [13.518599999999992, 56.311493], [14.110050000000001, 57.798588],
    [9.702775000000003, 58.227348], [9.283924000000013, 56.734879],
    [13.518599999999992, 56.311493]]]}
SKAGEN_BBOX = {"west": 10.50, "south": 57.55, "east": 11.70, "north": 58.05}
ALL_BAND_KEYS = S1_TARGET_BAND_KEYS + S1_HISTORICAL_BAND_KEYS


def _geo(col0: int, row0: int, h: int, w: int) -> GeoContext:
    return GeoContext(crs="EPSG:32632", transform=Affine(10, 0, col0 * 10, 0, -10, row0 * -10),
                      bounds_projected=None, bounds_wgs84=None, shape=(h, w))


def _rect(west: float, south: float, east: float, north: float) -> dict:
    return {"type": "Polygon", "coordinates": [[
        [west, south], [east, south], [east, north], [west, north], [west, south]]]}


def _item(item_id: str, start: str, footprint: dict, *, end: str | None = None,
          direction: str = "descending", rel_orbit: int = 66, abs_orbit: int = 3432,
          pols: tuple[str, ...] = ("VV", "VH"), mode: str = "IW",
          platform: str = "sentinel-1d") -> SimpleNamespace:
    """An MPC ``sentinel-1-grd`` STAC item with the fields the fetch reads."""
    from shapely.geometry import shape

    return SimpleNamespace(
        id=item_id, geometry=footprint, bbox=list(shape(footprint).bounds),
        properties={"start_datetime": start, "end_datetime": end or start,
                    "sat:orbit_state": direction, "sat:relative_orbit": rel_orbit,
                    "sat:absolute_orbit": abs_orbit, "sar:instrument_mode": mode,
                    "sar:polarizations": list(pols), "platform": platform},
        assets={"vv": SimpleNamespace(
            href=f"https://example.blob/GRD/{item_id}_71E0.SAFE/measurement/iw-vv.tiff")})


class FakeCatalogue:
    """Stands in for ``mpc_s1.search_grd_items``: returns the items whose start lies in
    the requested interval (inclusive) and records every call."""

    def __init__(self, items: list) -> None:
        self.items = items
        self.calls: list[dict] = []

    def __call__(self, datetime_range: str, *, bbox_4326=None, intersects=None,
                 limit: int = 50) -> list:
        self.calls.append({"interval": datetime_range, "bbox_4326": bbox_4326,
                           "intersects": intersects, "limit": limit})
        lo, hi = (fetch._parse_utc(t) for t in datetime_range.split("/"))
        return [it for it in self.items
                if lo <= fetch._parse_utc(it.properties["start_datetime"]) <= hi]


def _no_network(*args, **kwargs):
    raise AssertionError("network access in an offline test")


# ── normalisation, headings, attributes ─────────────────────────────────────

def test_detector_normalisation_clips_to_0_4():
    dn = np.array([-3.0, 0.0, 250.0, 999.0, 1000.0, 28145.0])
    np.testing.assert_allclose(s1.normalize_detector(dn), [0, 0, 1, 3.996, 4, 4])
    assert s1.normalize_detector(dn).dtype == np.float32


def test_normalisation_of_uint16_equals_float32_input():
    dn = np.array([0, 1, 250, 999, 1000, 65535], dtype=np.uint16)
    for norm in (s1.normalize_detector, s1.normalize_attribute):
        assert norm(dn).dtype == np.float32
        np.testing.assert_array_equal(norm(dn), norm(dn.astype(np.float32)))


def test_attribute_normalisation_does_not_clip():
    np.testing.assert_allclose(s1.normalize_attribute(np.array([-250.0, 1500.0])), [-1, 6])


@pytest.mark.parametrize("raw, wrapped", [(-88.6, 271.4), (268.5, 268.5), (360.0, 0.0),
                                          (0.0, 0.0), (-360.0, 0.0), (725.0, 5.0),
                                          (-1e-14, 0.0), (360.0 - 1e-9, 360.0 - 1e-9)])
def test_heading_wraps_to_0_360(raw, wrapped):
    out = s1.wrap_heading(raw)
    assert 0.0 <= out < 360.0
    assert out == pytest.approx(wrapped, abs=1e-12)


def test_heading_wrap_never_returns_360_for_tiny_negative():
    assert np.mod(-1e-14, 360.0) == 360.0            # the numpy behaviour being guarded
    out = s1.wrap_heading(np.array([-1e-14, -1e-300, 0.0, 359.5]))
    np.testing.assert_array_equal(out, [0.0, 0.0, 0.0, 359.5])


def test_decode_attributes_inverts_training_encoding():
    # Training encoded angle = 90 - cog, x = cos, y = sin; length etc. as value * 0.01.
    cogs = np.array([0.0, 75.0, 180.0, 284.6, 359.0])
    angle = np.radians(90.0 - cogs)
    logits = np.zeros((len(cogs), 9))
    logits[np.arange(len(cogs)), [1, 0, 6, 1, 2]] = 3.0
    raw = {"length": np.full((5, 1), 2.414), "width": np.full((5, 1), 0.416),
           "speed": np.full((5, 1), 0.077), "heading_x": np.cos(angle)[:, None],
           "heading_y": np.sin(angle)[:, None], "ship_type": logits}
    out = s1.decode_attributes(raw)
    assert [o["heading_deg"] for o in out] == pytest.approx(cogs % 360, abs=1e-9)
    assert all(0.0 <= o["heading_deg"] < 360.0 for o in out)
    assert out[0]["length_m"] == pytest.approx(241.4)
    assert out[0]["width_m"] == pytest.approx(41.6)
    assert out[0]["speed_knots"] == pytest.approx(7.7)
    assert [o["vessel_type"] for o in out] == ["tanker", "cargo", "fishing", "tanker", "passenger"]
    expected_conf = math.exp(3) / (math.exp(3) + 8)
    assert all(o["type_confidence"] == pytest.approx(expected_conf) for o in out)


# ── tiling ──────────────────────────────────────────────────────────────────

def test_tiling_reproduces_reference_window():
    # Reference window [56781, -645167, 67589, -635319]: 24 x 22 = 528 patches.
    cols = s1.crop_starts(56781, 67589)
    rows = s1.crop_starts(-645167, -635319)
    assert (len(cols), len(rows)) == (24, 22)
    assert cols[:3] == [56781, 57242, 57703]
    assert cols[-1] == 67589 - 512 and rows[-1] == -635319 - 512


@pytest.mark.parametrize("length", [1, 300, 511, 512, 513, 973, 974, 2000, 10808])
def test_tiling_covers_every_pixel_and_stays_inside(length):
    starts = s1.crop_starts(0, length)
    assert starts[0] == 0
    covered = np.zeros(length, dtype=bool)
    for st in starts:
        covered[st:st + s1.PATCH_SIZE] = True
        if length >= s1.PATCH_SIZE:
            assert 0 <= st <= length - s1.PATCH_SIZE
    assert covered.all()
    assert all(b - a <= s1.PATCH_SIZE - s1.OVERLAP_PX for a, b in pairwise(starts))


def test_patch_origins_are_column_major():
    origins = s1.patch_origins(1000, 1500)
    assert origins[:3] == [(0, 0), (0, 461), (0, 488)]
    assert len(origins) == 4 * 3


# ── detector loop on a fake detector: patch cutting and global coordinates ──

# (row, col) markers on a 1000 x 1500 image whose patch grid is
# cols [0, 461, 922, 988] x rows [0, 461, 488]; value = covering patches.
_MARKERS = {(100, 100): 1, (480, 470): 4, (700, 1200): 4, (10, 1490): 1,
            (990, 5): 1, (995, 1495): 1}   # the last two only in the snapped last row/col
_HIST_MARKER_DN = 300


class _MarkerDetector:
    """Stands in for ``detector_forward``: a 9 px box centred on every pixel that is a
    target marker (normalised 4.0) AND a historical marker at the same patch position,
    plus one box below the score threshold. Records the patch tensors it sees."""

    def __init__(self) -> None:
        self.dtypes: set = set()

    def __call__(self, model, target, historical) -> list[dict]:
        import torch

        self.dtypes |= {target.dtype, historical.dtype}
        out = []
        for t, h in zip(target.numpy(), historical.numpy(), strict=True):
            ys, xs = np.nonzero((t[0] == 4.0)
                                & np.isclose(h[0], _HIST_MARKER_DN / 250.0)
                                & (t[1] == 4.0))
            boxes = [[x - 4.0, y - 4.0, x + 5.0, y + 5.0] for y, x in zip(ys, xs)]
            out.append({"boxes": torch.tensor(boxes + [[0.0, 0.0, 9.0, 9.0]]),
                        "scores": torch.tensor([0.9] * len(boxes) + [0.5]),
                        "labels": torch.ones(len(boxes) + 1, dtype=torch.int64)})
        return out


def _marker_bands(h: int = 1000, w: int = 1500, dtype=np.uint16) -> dict:
    bands = {}
    for key in ALL_BAND_KEYS:
        band = np.full((h, w), 100, dtype=dtype)
        for r, c in _MARKERS:
            band[r, c] = _HIST_MARKER_DN if key in S1_HISTORICAL_BAND_KEYS else 2000
        bands[key] = band
    return bands


def _fake_detector_analyzer(monkeypatch, config: dict | None = None):
    fake = _MarkerDetector()
    monkeypatch.setattr(s1, "detector_forward", fake)
    analyzer = s1.AI2S1VesselAnalyzer({"infra_filter": False, "predict_attributes": False,
                                       **(config or {})})
    analyzer._detector = object()          # never loaded: detector_forward is faked
    return analyzer, fake


def test_detect_places_patch_boxes_at_global_image_coordinates(monkeypatch):
    analyzer, fake = _fake_detector_analyzer(monkeypatch)
    bands = _marker_bands()
    target = tuple(bands[k] for k in S1_TARGET_BAND_KEYS)
    historical = tuple(bands[k] for k in S1_HISTORICAL_BAND_KEYS)
    boxes, scores, n_patches = analyzer._detect(target, historical, s1.SCORE_THRESHOLD, "cpu")
    assert n_patches == 12
    assert len(boxes) == sum(_MARKERS.values()) == 12       # the 0.5 boxes are dropped
    assert np.allclose(scores, 0.9)
    centres = {((b[1] + b[3]) / 2 - 0.5, (b[0] + b[2]) / 2 - 0.5) for b in boxes}
    assert centres == set(_MARKERS)
    # Each marker is reported once per covering patch, all at the same position.
    for (r, c), n in _MARKERS.items():
        hits = np.all(np.isclose(boxes, [c - 4, r - 4, c + 5, r + 5]), axis=1)
        assert hits.sum() == n, (r, c)
    import torch
    assert fake.dtypes == {torch.float32}


def test_analyzer_merges_patch_duplicates_and_keeps_uint16_input(monkeypatch):
    analyzer, _ = _fake_detector_analyzer(monkeypatch)
    bands = _marker_bands()
    col0, row0 = 60000, -640000
    res = analyzer.run(None, bands=bands, geo=_geo(col0, row0, 1000, 1500))
    assert res.success, res.error
    md = res.metadata
    assert (md["patches"], md["raw_detections"], md["after_nms"]) == (12, 12, 6)
    centres = {((r["box_px"][1] + r["box_px"][3]) / 2 - 0.5,
                (r["box_px"][0] + r["box_px"][2]) / 2 - 0.5) for r in res.outputs["regions"]}
    assert centres == set(_MARKERS)
    # AI2 point: int() of the absolute centre; the negative row truncates up by 1.
    assert {tuple(r["point_px"]) for r in res.outputs["regions"]} == \
        {(c, r + 1) for r, c in _MARKERS}
    assert all(b.dtype == np.uint16 for b in bands.values())     # read in place, not cast


def test_analyzer_float32_and_uint16_inputs_detect_identically(monkeypatch):
    analyzer, _ = _fake_detector_analyzer(monkeypatch)
    geo = _geo(60000, -640000, 1000, 1500)
    a = analyzer.run(None, bands=_marker_bands(), geo=geo).outputs["regions"]
    b = analyzer.run(None, bands=_marker_bands(dtype=np.float32), geo=geo).outputs["regions"]
    assert a == b


def test_analyzer_reports_and_warns_on_partial_historical_coverage(monkeypatch, caplog):
    analyzer, _ = _fake_detector_analyzer(monkeypatch)
    bands = _marker_bands()
    for key in S1_HISTORICAL_BAND_KEYS:
        bands[key][:, :750] = 0
    with caplog.at_level(logging.WARNING, logger=s1.__name__):
        res = analyzer.run(None, bands=bands, geo=_geo(60000, -640000, 1000, 1500))
    assert res.metadata["valid_frac_target"] == 1.0
    assert res.metadata["valid_frac_historical"] == 0.5
    assert any("historical scene covers 50.0 %" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=s1.__name__):
        analyzer.run(None, bands=_marker_bands(), geo=_geo(60000, -640000, 1000, 1500))
    assert not any("historical scene covers" in r.getMessage() for r in caplog.records)


def test_infra_file_is_not_loaded_without_detections(monkeypatch):
    analyzer, _ = _fake_detector_analyzer(monkeypatch, {"infra_filter": True})
    monkeypatch.setattr(s1, "load_infrastructure", _no_network)
    bands = {k: np.full((600, 600), 100, dtype=np.uint16) for k in ALL_BAND_KEYS}
    res = analyzer.run(None, bands=bands, geo=_geo(60000, -640000, 600, 600))
    assert res.success and res.outputs["vessel_count"] == 0
    assert res.metadata["infra_filter"] == {"applied": False, "reason": "no detections to filter"}


# ── distance NMS and the AI2 point ──────────────────────────────────────────

def test_distance_nms_keeps_strongest_within_10px_inclusive():
    centers = np.array([[0, 0], [10, 0], [30, 0], [40.01, 0]])
    scores = np.array([0.9, 0.95, 0.8, 0.85])
    # 0 and 1 are exactly 10 px apart (inclusive) -> 0 suppressed; 2 and 3 are > 10 px.
    assert sorted(s1.distance_nms(centers, scores)) == [1, 2, 3]


def test_distance_nms_equal_scores_lower_index_wins():
    assert s1.distance_nms(np.array([[0, 0], [5, 0]]), np.array([0.9, 0.9])) == [0]


def test_distance_nms_suppression_does_not_chain_back():
    # A > B > C with A-B and B-C within 10 px but A-C not: AI2 eliminates C (B is
    # stronger and not yet eliminated when C is visited), then B (by A).
    centers = np.array([[0, 0], [8, 0], [16, 0]])
    keep = s1.distance_nms(centers, np.array([0.99, 0.9, 0.8]))
    assert keep == [0]


def test_distance_nms_returns_ascending_score_order():
    centers = np.array([[0, 0], [100, 0], [200, 0]])
    assert s1.distance_nms(centers, np.array([0.8, 0.99, 0.9])) == [0, 2, 1]


def test_ai2_point_truncates_towards_zero():
    assert s1.ai2_point(59547.9, -639635.4) == (59547, -639635)
    assert s1.ai2_point(10.0, -10.0) == (10, -10)


def test_grid_origin_px_accepts_only_ai2_grid():
    assert s1.grid_origin_px(Affine(10, 0, 588530, 0, -10, 6437470)) == (58853, -643747)
    with pytest.raises(ValueError):
        s1.grid_origin_px(Affine(10, 0, 588535, 0, -10, 6437470))
    with pytest.raises(ValueError):
        s1.grid_origin_px(Affine(20, 0, 588520, 0, -20, 6437480))


# ── infrastructure filter ───────────────────────────────────────────────────

def _region_at(lon: float, lat: float) -> dict:
    return {"lon": lon, "lat": lat, "score": 0.9}


def _infra_file(tmp_path: Path, points: list[tuple[float, float]]) -> Path:
    path = tmp_path / "infra.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": list(p)},
         "properties": {"category": "offshore_platform"}} for p in points]}))
    return path


def test_infra_filter_drops_within_threshold(tmp_path):
    infra = _infra_file(tmp_path, [(11.0, 57.8)])
    near = _region_at(11.0, 57.8 + 150 / 111_195)       # ~150 m north
    far = _region_at(11.0, 57.8 + 400 / 111_195)        # ~400 m north
    analyzer = s1.AI2S1VesselAnalyzer({"infra_path": str(infra)})
    meta, kept = analyzer._infra_filter([dict(near), dict(far)])
    assert meta["applied"] and meta["removed"] == 1 and meta["distance_m"] == 200.0
    assert [round(r["infra_distance_m"]) for r in kept] == [400]
    meta, kept = s1.AI2S1VesselAnalyzer({"infra_path": str(infra), "infra_distance_m": 100}
                                        )._infra_filter([dict(near), dict(far)])
    assert meta["removed"] == 0 and len(kept) == 2


def test_infra_filter_failure_is_reported_not_silent(tmp_path):
    analyzer = s1.AI2S1VesselAnalyzer({"infra_path": str(tmp_path / "missing.geojson")})
    meta, kept = analyzer._infra_filter([_region_at(11.0, 57.8)])
    assert meta["applied"] is False
    assert "FileNotFoundError" in meta["reason"]
    assert len(kept) == 1


# ── analyzer contract ───────────────────────────────────────────────────────

def test_analyzer_skips_sentinel2_jobs():
    res = s1.AI2S1VesselAnalyzer().run(np.zeros((8, 8, 3)), bands={"B02": np.zeros((8, 8))})
    assert res.skipped and not res.success
    assert "VV_DN" in res.error


def test_analyzer_requires_geo():
    bands = {k: np.zeros((8, 8), dtype=np.uint16) for k in ALL_BAND_KEYS}
    res = s1.AI2S1VesselAnalyzer().run(None, bands=bands)
    assert not res.success and not res.skipped and "geo" in res.error


def test_analyzer_rejects_bands_on_different_grids():
    bands = {k: np.zeros((8, 8), dtype=np.uint16) for k in ALL_BAND_KEYS}
    bands["VH_DN_HIST"] = np.zeros((8, 9), dtype=np.uint16)
    res = s1.AI2S1VesselAnalyzer().run(None, bands=bands, geo=_geo(0, 0, 8, 8))
    assert not res.success and "one grid" in res.error


def test_registered_in_engine():
    from imint.engine import ANALYZER_REGISTRY

    assert ANALYZER_REGISTRY["ai2_vessels_s1"] is s1.AI2S1VesselAnalyzer
    assert s1.AI2S1VesselAnalyzer.name == "ai2_vessels_s1"


# ── GeoJSON for marine-awareness ────────────────────────────────────────────

REQUIRED_FEATURE_PROPS = {"detector", "score", "pixel_count", "pred_length_m", "pred_width_m",
                          "pred_speed_kn", "pred_heading_deg", "pred_type", "pred_type_conf"}
REQUIRED_FC_PROPS = {"date", "bbox", "crs", "grid_px", "source", "sensor", "scene_id",
                     "historical_scene_id", "scene_time_utc", "scene_start_utc",
                     "scene_end_utc", "valid_frac_target", "valid_frac_historical",
                     "n_outside_bbox", "detectors"}
_GEO = _geo(60409, -641154, 192, 192)
_BBOX = [10.5, 57.55, 11.7, 58.05]


def _fake_result(heading_deg: float = 284.6) -> AnalysisResult:
    """Three regions: two inside ``_BBOX``, one placed (by lon/lat) outside it."""
    from pyproj import Transformer

    to_wgs84 = Transformer.from_crs(_GEO.crs, 4326, always_xy=True).transform
    col0, row0 = s1.grid_origin_px(_GEO.transform)

    def region(box: list[float], pixel_count: int, score: float, attrs=None) -> dict:
        col, row = s1.ai2_point(col0 + (box[0] + box[2]) / 2, row0 + (box[1] + box[3]) / 2)
        lon, lat = to_wgs84(col * 10, row * -10)
        out = {"box_px": box, "pixel_count": pixel_count, "score": score, "lon": lon, "lat": lat}
        return out if attrs is None else {**out, "attributes": attrs}

    attrs = {"length_m": 241.43, "width_m": 41.6, "speed_knots": 7.7,
             "heading_deg": heading_deg, "vessel_type": "tanker", "type_confidence": 0.9312}
    regions = [region([80.5, 81.2, 110.4, 111.0], 890, 0.99831, attrs),
               region([10.0, 20.0, 27.0, 38.0], 306, 0.7123),
               {**region([150.0, 150.0, 170.0, 170.0], 400, 0.95), "lon": 11.75}]
    return AnalysisResult(analyzer="ai2_vessels_s1", success=True,
                          outputs={"regions": regions, "has_attributes": False},
                          metadata={"score_threshold": 0.7, "raw_detections": 4, "after_nms": 3,
                                    "infra_filter": {"applied": True, "removed": 0},
                                    "water_mask": "none", "valid_frac_target": 0.998,
                                    "valid_frac_historical": 0.91})


def _fc(result: AnalysisResult | None = None) -> dict:
    from imint.exporters.export import s1_vessel_feature_collection

    return s1_vessel_feature_collection(
        result or _fake_result(), _GEO, date="2026-06-28", bbox=_BBOX,
        scene_id="S1D_IW_GRDH_1SDV_X", historical_scene_id="S1D_IW_GRDH_1SDV_Y",
        scene_time_utc="2026-06-28T05:31:38.700000Z",
        scene_start_utc="2026-06-28T05:31:35.545314Z",
        scene_end_utc="2026-06-28T05:32:00.545314Z", source="test")


def test_geojson_schema_matches_marine_awareness():
    fc = _fc()
    assert fc["type"] == "FeatureCollection" and "crs" not in fc
    assert set(fc["properties"]) == REQUIRED_FC_PROPS
    p = fc["properties"]
    assert p["sensor"] == "sentinel-1" and p["crs"] == "EPSG:32632" and p["grid_px"] == [192, 192]
    assert all(p[k].endswith("Z") for k in ("scene_time_utc", "scene_start_utc", "scene_end_utc"))
    assert (p["valid_frac_target"], p["valid_frac_historical"]) == (0.998, 0.91)
    assert p["detectors"]["ai2_s1"]["count"] == len(fc["features"]) == 2
    for f in fc["features"]:
        assert set(f["properties"]) == REQUIRED_FEATURE_PROPS
        assert f["properties"]["detector"] == "ai2_s1"
        assert f["geometry"]["type"] == "Polygon" and "crs" not in f["geometry"]
        ring = f["geometry"]["coordinates"][0]
        assert len(ring) == 5 and ring[0] == ring[-1]
        assert all(10.5 < lon < 11.0 and 57.7 < lat < 57.9 for lon, lat in ring)
    first = fc["features"][0]["properties"]
    assert (first["pred_length_m"], first["pred_type"], first["pred_type_conf"]) == (241.4, "tanker", 0.931)
    assert (first["pixel_count"], first["pred_heading_deg"]) == (890, 284.6)
    assert fc["features"][1]["properties"]["pred_length_m"] is None
    json.dumps(fc)  # serialisable without a custom encoder


def test_geojson_records_the_pass_only_when_given():
    from imint.exporters.export import s1_vessel_feature_collection

    s1_pass = {"slice_ids": ["S1C_A", "S1C_B"], "selected_by": "orbit_state=descending"}
    fc = s1_vessel_feature_collection(
        _fake_result(), _GEO, date="2026-06-27", bbox=_BBOX, scene_id="S1C_A + S1C_B",
        historical_scene_id="S1C_H + S1D_H", scene_time_utc="2026-06-27T05:39:37.6Z",
        scene_start_utc="2026-06-27T05:39:11.6Z", scene_end_utc="2026-06-27T05:40:01.6Z",
        source="test", s1_pass=s1_pass)
    assert set(fc["properties"]) == REQUIRED_FC_PROPS | {"s1_pass"}
    assert fc["properties"]["s1_pass"] == s1_pass
    assert fc["properties"]["scene_id"] == "S1C_A + S1C_B"
    assert "s1_pass" not in _fc()["properties"]


def test_geojson_drops_detections_outside_the_requested_bbox():
    fc = _fc()
    assert fc["properties"]["n_outside_bbox"] == 1
    assert [f["properties"]["score"] for f in fc["features"]] == [0.9983, 0.7123]


def test_geojson_rings_are_counter_clockwise():
    # RFC 7946 3.1.6: exterior rings are counter-clockwise (positive shoelace area).
    for f in _fc()["features"]:
        ring = np.array(f["geometry"]["coordinates"][0])
        x, y = ring[:-1, 0], ring[:-1, 1]
        assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0


def test_geojson_heading_rounding_never_gives_360():
    fc = _fc(_fake_result(heading_deg=359.96))
    assert round(359.96, 1) == 360.0                    # the rounding being guarded
    assert fc["features"][0]["properties"]["pred_heading_deg"] == 0.0


def test_geojson_box_centre_is_what_load_detections_reads():
    # marine-awareness load_detections: drop the closing vertex, mean of the rest;
    # it reads "score" and "pixel_count".
    from pyproj import Transformer

    feat = _fc()["features"][0]
    ring = feat["geometry"]["coordinates"][0][:-1]
    lon, lat = np.mean(ring, axis=0)
    x, y = Transformer.from_crs(4326, 32632, always_xy=True).transform(lon, lat)
    cx, cy = (80.5 + 110.4) / 2, (81.2 + 111.0) / 2
    assert x == pytest.approx(604090 + cx * 10, abs=0.5)
    assert y == pytest.approx(6411540 - cy * 10, abs=0.5)
    assert int(feat["properties"].get("pixel_count", 0)) == 890
    assert float(feat["properties"].get("score", 0.0)) == 0.9983


@pytest.mark.parametrize("field", ["scene_time_utc", "scene_start_utc", "scene_end_utc"])
def test_geojson_rejects_non_utc_time(field):
    from imint.exporters.export import s1_vessel_feature_collection

    times = {"scene_time_utc": "2026-06-28T05:31:38Z", "scene_start_utc": "2026-06-28T05:31:35Z",
             "scene_end_utc": "2026-06-28T05:32:00Z", field: "2026-06-28T05:31:35"}
    with pytest.raises(ValueError, match=field):
        s1_vessel_feature_collection(_fake_result(), _geo(0, 0, 4, 4), date="d",
                                     bbox=[0, 0, 1, 1], scene_id="a", historical_scene_id="b",
                                     source="s", **times)


# ── pass and scene selection (catalogue faked) ──────────────────────────────

_T0 = "2026-06-28T05:31:35.545314Z"
_TARGET_ID = "S1D_IW_GRDH_1SDV_20260628T053135_20260628T053200_003432_0060CF"


def test_select_pass_takes_the_only_pass_iw_grdh_dual_pol_on_the_day(monkeypatch):
    from shapely.geometry import box

    aoi = box(10.5, 57.55, 11.7, 58.05)
    full = _rect(10.0, 57.0, 12.0, 59.0)
    items = [
        _item(_TARGET_ID, _T0, _rect(10.0, 57.0, 12.0, 58.04)),                 # cover 0.98
        _item("S1C_IW_GRDM_1SDV_20260628T053000_X", "2026-06-28T05:30:00Z", full,  # not GRDH
              platform="sentinel-1c", abs_orbit=1),
        _item("S1A_IW_GRDH_1SSV_20260628T053000_X", "2026-06-28T05:30:00Z", full,
              pols=("VV",), platform="sentinel-1a", abs_orbit=2),                  # single pol
        _item("S1A_EW_GRDH_1SDV_20260628T053000_X", "2026-06-28T05:30:00Z", full,
              mode="EW", platform="sentinel-1a", abs_orbit=3),                     # not IW
        _item("S1D_IW_GRDH_1SDV_20260627T053135_X", "2026-06-27T05:31:35Z", full,  # other day
              abs_orbit=3417),
    ]
    cat = FakeCatalogue(items)
    monkeypatch.setattr(mpc_s1, "search_grd_items", cat)
    chosen, passes, rule = fetch._select_s1_pass(aoi, "2026-06-28")
    assert [it.id for it in chosen.slices] == [_TARGET_ID] and passes == [chosen]
    assert chosen.covers == [pytest.approx(0.98)]
    assert rule == "the only pass on the date"
    assert cat.calls == [{"interval": "2026-06-28T00:00:00Z/2026-06-29T00:00:00Z",
                          "bbox_4326": aoi.bounds, "intersects": None, "limit": 50}]


def test_select_pass_ranks_slices_by_cover_then_earliest_start(monkeypatch):
    from shapely.geometry import box

    fp = _rect(10.0, 57.0, 12.0, 59.0)
    late = _item("S1D_IW_GRDH_1SDV_20260628T053200_X", "2026-06-28T05:32:00Z", fp)
    early = _item("S1D_IW_GRDH_1SDV_20260628T053135_X", _T0, fp)
    sliver = _item("S1D_IW_GRDH_1SDV_20260628T053110_X", "2026-06-28T05:31:10Z",
                   _rect(10.0, 57.5, 12.0, 57.6))
    monkeypatch.setattr(mpc_s1, "search_grd_items", FakeCatalogue([sliver, late, early]))
    chosen, _, _ = fetch._select_s1_pass(box(10.5, 57.55, 11.7, 58.05), "2026-06-28")
    assert chosen.slices == [early, late, sliver]          # one pass: same satellite + orbit
    assert chosen.describe()["slice_ids"] == [sliver.id, early.id, late.id]   # time order


def test_select_pass_raises_without_a_covering_scene(monkeypatch):
    from shapely.geometry import box

    aoi = box(10.5, 57.55, 11.7, 58.05)
    monkeypatch.setattr(mpc_s1, "search_grd_items", FakeCatalogue([]))
    with pytest.raises(fetch.FetchError, match="No Sentinel-1 IW GRDH"):
        fetch._select_s1_pass(aoi, "2026-06-28")
    monkeypatch.setattr(mpc_s1, "search_grd_items",
                        FakeCatalogue([_item(_TARGET_ID, _T0, _rect(20, 60, 21, 61))]))
    with pytest.raises(fetch.FetchError, match="intersects"):
        fetch._select_s1_pass(aoi, "2026-06-28")


def _poly(*lonlat: tuple[float, float]) -> dict:
    return {"type": "Polygon", "coordinates": [[list(p) for p in (*lonlat, lonlat[0])]]}


def _skagen_0627() -> dict[str, SimpleNamespace]:
    """MPC items of 2026-06-27 over the Skagen-Marstrand bbox (ids, times, footprints):
    descending S1C split over two slices of one datatake, ascending S1C in one."""
    def s1c(item_id, start, end, fp, **kw):
        return _item(item_id, start, fp, end=end, platform="sentinel-1c", **kw)

    return {
        "dsc_n": s1c("S1C_IW_GRDH_1SDV_20260627T053911_20260627T053936_008287_01063E",
                     "2026-06-27T05:39:11.688393Z", "2026-06-27T05:39:36.686143Z",
                     _poly((12.076245, 57.750389), (12.705921, 59.235237),
                           (8.065848, 59.675667), (7.627708, 58.184185)),
                     rel_orbit=139, abs_orbit=8287),
        "dsc_s": s1c("S1C_IW_GRDH_1SDV_20260627T053936_20260627T054001_008287_01063E",
                     "2026-06-27T05:39:36.687612Z", "2026-06-27T05:40:01.686858Z",
                     _poly((11.479633, 56.263756), (12.076209, 57.750301),
                           (7.629532, 58.183952), (7.206832, 56.691799)),
                     rel_orbit=139, abs_orbit=8287),
        "asc": s1c("S1C_IW_GRDH_1SDV_20260627T165257_20260627T165322_008294_010682",
                   "2026-06-27T16:52:57.190316Z", "2026-06-27T16:53:22.188725Z",
                   _poly((9.451989, 58.642635), (10.070417, 57.157291),
                         (14.449341, 57.589111), (14.015204, 59.080757)),
                   direction="ascending", rel_orbit=146, abs_orbit=8294),
    }


def test_two_passes_without_a_selector_are_refused_with_a_listing(monkeypatch):
    from shapely.geometry import box

    monkeypatch.setattr(mpc_s1, "search_grd_items",
                        FakeCatalogue(list(_skagen_0627().values())))
    with pytest.raises(fetch.FetchError) as err:
        fetch._select_s1_pass(box(10.5, 57.55, 11.7, 58.05), "2026-06-27")
    msg = str(err.value)
    assert msg.startswith("2 Sentinel-1 passes on 2026-06-27")
    assert "orbit_state" in msg and "pass_time_utc" in msg
    assert ("descending sentinel-1c abs. orbit 8287 (rel. 139) 05:39:11-05:40:01 UTC, "
            "2 slices, AOI cover 1.000") in msg
    assert ("ascending sentinel-1c abs. orbit 8294 (rel. 146) 16:52:57-16:53:22 UTC, "
            "1 slice, AOI cover 1.000") in msg


@pytest.mark.parametrize(("selector", "want"), [
    ({"orbit_state": "descending"}, "dsc"),
    ({"orbit_state": "ascending"}, "asc"),
    ({"pass_time": "16:53"}, "asc"),
    ({"pass_time": "2026-06-27T05:39:37.62Z"}, "dsc"),
    ({"pass_time": "05:41:30"}, "dsc"),                   # 89 s after the last slice ends
    ({"pass_time": "2026-06-27T07:39:00+02:00"}, "dsc"),
    ({"orbit_state": "descending", "pass_time": "05:39"}, "dsc"),
])
def test_selector_picks_the_pass_and_joins_its_slices(monkeypatch, selector, want):
    from shapely.geometry import box

    items = _skagen_0627()
    monkeypatch.setattr(mpc_s1, "search_grd_items", FakeCatalogue(list(items.values())))
    when = selector.get("pass_time")
    chosen, passes, rule = fetch._select_s1_pass(
        box(10.5, 57.55, 11.7, 58.05), "2026-06-27", orbit_state=selector.get("orbit_state"),
        pass_time=None if when is None else fetch._s1_pass_time(when, "2026-06-27"))
    assert len(passes) == 2 and chosen in passes
    if want == "asc":
        assert chosen.slices == [items["asc"]] and chosen.covers == [pytest.approx(1.0)]
    else:   # the southern slice covers more of the AOI (0.591 vs 0.409), so it ranks first
        assert chosen.slices == [items["dsc_s"], items["dsc_n"]]
        assert chosen.covers == [pytest.approx(0.591, abs=1e-3), pytest.approx(0.409, abs=1e-3)]
        assert chosen.union_cover > 0.999
    assert rule == ", ".join(
        f"orbit_state={v}" if k == "orbit_state"
        else f"pass_time_utc={fetch._s1_pass_time(v, '2026-06-27'):%H:%M:%S}"
        for k, v in selector.items())


def test_selector_matching_no_pass_is_refused(monkeypatch):
    from shapely.geometry import box

    monkeypatch.setattr(mpc_s1, "search_grd_items",
                        FakeCatalogue(list(_skagen_0627().values())))
    with pytest.raises(fetch.FetchError, match="matches pass_time_utc=05:43:00") as err:
        fetch._select_s1_pass(box(10.5, 57.55, 11.7, 58.05), "2026-06-27",
                              pass_time=fetch._s1_pass_time("05:43", "2026-06-27"))
    assert "05:39:11-05:40:01" in str(err.value) and "16:52:57-16:53:22" in str(err.value)
    with pytest.raises(fetch.FetchError, match="matches orbit_state=ascending, pass_time_utc"):
        fetch._select_s1_pass(box(10.5, 57.55, 11.7, 58.05), "2026-06-27",
                              orbit_state="ascending",
                              pass_time=fetch._s1_pass_time("05:39", "2026-06-27"))


def test_two_passes_in_one_direction_need_the_pass_time(monkeypatch):
    """2026-06-28 Skagen: an S1A pass (rel. 168) clips the bbox 7 min before the S1D one."""
    from shapely.geometry import box

    s1a = _item("S1A_IW_GRDH_1SDV_20260628T052351_20260628T052416_065165_0836F0",
                "2026-06-28T05:23:51Z", _rect(9.0, 57.0, 10.6, 58.5),
                end="2026-06-28T05:24:16Z", rel_orbit=168, abs_orbit=65165,
                platform="sentinel-1a")
    s1d_n = _item("S1D_IW_GRDH_1SDV_20260628T053110_20260628T053135_003432_0060CF",
                  "2026-06-28T05:31:10Z", _rect(10.0, 58.04, 12.0, 59.5),
                  end="2026-06-28T05:31:35Z")
    s1d = _item(_TARGET_ID, _T0, _rect(10.0, 57.0, 12.0, 58.04),
                end="2026-06-28T05:32:00.545314Z")
    monkeypatch.setattr(mpc_s1, "search_grd_items", FakeCatalogue([s1a, s1d_n, s1d]))
    aoi = box(10.5, 57.55, 11.7, 58.05)
    with pytest.raises(fetch.FetchError,
                       match="2 Sentinel-1 passes matching orbit_state=descending"):
        fetch._select_s1_pass(aoi, "2026-06-28", orbit_state="descending")
    chosen, passes, _ = fetch._select_s1_pass(
        aoi, "2026-06-28", pass_time=fetch._s1_pass_time("05:31", "2026-06-28"))
    assert chosen.slices == [s1d, s1d_n] and [p.slices[0] for p in passes] == [s1a, s1d]


def test_duplicate_catalogue_items_count_once(monkeypatch):
    """MPC lists 2025-07-19 Oresund twice (with and without the datatake id)."""
    from shapely.geometry import box

    fp = _rect(12.0, 55.5, 13.5, 56.5)
    kw = dict(end="2025-07-19T16:52:56.5Z", direction="ascending", rel_orbit=146,
              abs_orbit=3292, platform="sentinel-1c")
    short = _item("S1C_IW_GRDH_1SDV_20250719T165231_20250719T165256_003292",
                  "2025-07-19T16:52:31.5Z", fp, **kw)
    full = _item("S1C_IW_GRDH_1SDV_20250719T165231_20250719T165256_003292_006A41",
                 "2025-07-19T16:52:31.5Z", fp, **kw)
    for order in ([short, full], [full, short]):
        monkeypatch.setattr(mpc_s1, "search_grd_items", FakeCatalogue(order))
        chosen, passes, _ = fetch._select_s1_pass(box(12.54, 55.99, 12.72, 56.08), "2025-07-19")
        assert chosen.slices == [full] and len(passes) == 1


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"orbit_state": "north"}, "orbit_state must be one of"),
    ({"pass_time_utc": "25:00"}, "pass_time_utc must be"),
    ({"pass_time_utc": "2026-06-28T05:31:00Z"}, "is not on 2026-06-27"),
    ({"pass_time_utc": "23:30-02:00"}, "is not on 2026-06-27"),   # 01:30 UTC next day
])
def test_invalid_selectors_fail_before_any_search(tmp_path, monkeypatch, kwargs, match):
    monkeypatch.setattr(mpc_s1, "search_grd_items", _no_network)
    with pytest.raises(ValueError, match=match):
        fetch.fetch_s1_vessel_scenes(SKAGEN_BBOX, "2026-06-27", cache_dir=tmp_path, **kwargs)


def test_pass_time_forms():
    t = fetch._s1_pass_time
    assert t("05:39", "2026-06-27") == fetch._parse_utc("2026-06-27T05:39:00Z")
    assert t("05:39:37.5Z", "2026-06-27") == fetch._parse_utc("2026-06-27T05:39:37.5Z")
    assert t("2026-06-27T05:39:37", "2026-06-27") == fetch._parse_utc("2026-06-27T05:39:37Z")
    assert t(" 2026-06-27T07:39:00+02:00 ", "2026-06-27") == \
        fetch._parse_utc("2026-06-27T05:39:00Z")


def _historical_candidates() -> dict[str, SimpleNamespace]:
    """Around a target over [10, 14] x [56, 58] starting at ``_T0`` (descending, rel. 66)."""
    return {
        "in_window": _item("S1D_IW_GRDH_1SDV_20260523T053133_IN", "2026-05-23T05:31:33.1Z",
                           _rect(10.04, 56, 14.04, 58)),
        "other_rel_orbit": _item("S1D_IW_GRDH_1SDV_20260511T053100_REL", "2026-05-11T05:31:00Z",
                                 _rect(12, 56, 16, 58), rel_orbit=139),
        "ascending": _item("S1D_IW_GRDH_1SDV_20260520T170000_ASC", "2026-05-20T17:00:00Z",
                           _rect(10, 56, 14, 58), direction="ascending"),
        "61_days": _item("S1D_IW_GRDH_1SDV_20260428T053130_OLD", "2026-04-28T05:31:30Z",
                         _rect(10, 56, 14, 58)),
        "27_days": _item("S1D_IW_GRDH_1SDV_20260601T053130_NEW", "2026-06-01T05:31:30Z",
                         _rect(10, 56, 14, 58)),
    }


def test_select_historical_applies_ai2_window_direction_and_overlap(monkeypatch):
    target = _item(_TARGET_ID, _T0, _rect(10, 56, 14, 58))
    cands = _historical_candidates()
    cat = FakeCatalogue(list(cands.values()))
    monkeypatch.setattr(mpc_s1, "search_grd_items", cat)
    assert fetch._select_s1_historical(target) is cands["in_window"]
    t0 = fetch._parse_utc(_T0)
    (call,) = cat.calls
    assert call["interval"] == (f"{(t0 - timedelta(days=60)).isoformat()}/"
                                f"{(t0 - timedelta(days=30)).isoformat()}")
    assert call["intersects"] == target.geometry and call["limit"] == 100


def test_select_historical_tie_goes_to_the_earliest_start(monkeypatch):
    target = _item(_TARGET_ID, _T0, _rect(10, 56, 14, 58))
    cands = _historical_candidates()
    tie = _item("S1C_IW_GRDH_1SDV_20260522T053133_TIE", "2026-05-22T05:31:33Z",
                cands["in_window"].geometry)
    monkeypatch.setattr(mpc_s1, "search_grd_items", FakeCatalogue([*cands.values(), tie]))
    assert fetch._select_s1_historical(target) is tie


def test_select_historical_raises_without_same_direction_scene(monkeypatch):
    target = _item(_TARGET_ID, _T0, _rect(10, 56, 14, 58))
    cands = _historical_candidates()
    monkeypatch.setattr(mpc_s1, "search_grd_items", FakeCatalogue([cands["ascending"]]))
    with pytest.raises(fetch.FetchError, match="DESCENDING historical scene"):
        fetch._select_s1_historical(target)


def test_cli_hands_the_selectors_to_the_fetch_and_reports_a_refusal(tmp_path, monkeypatch,
                                                                     capsys):
    from scripts import detect_vessels_s1 as cli

    seen = {}

    def refuse(coords, date, *, cache_dir, orbit_state, pass_time_utc):
        seen.update(date=date, orbit_state=orbit_state, pass_time_utc=pass_time_utc)
        raise fetch.FetchError("2 Sentinel-1 passes on 2026-06-27 over (...); choose one ...")

    monkeypatch.setattr(fetch, "fetch_s1_vessel_scenes", refuse)
    args = ["--date", "2026-06-27", "--bbox", "10.5", "57.55", "11.7", "58.05",
            "--cache-dir", str(tmp_path), "--out", str(tmp_path / "out.geojson")]
    assert cli.main([*args, "--orbit-state", "descending", "--pass-time", "05:39"]) == 2
    assert seen == {"date": "2026-06-27", "orbit_state": "descending", "pass_time_utc": "05:39"}
    assert "fetch failed: 2 Sentinel-1 passes" in capsys.readouterr().err
    assert cli.main(args) == 2 and seen["orbit_state"] is seen["pass_time_utc"] is None
    with pytest.raises(SystemExit):                      # argparse error, no fetch
        cli.main([*args, "--pass-time", "29:00"])
    assert "pass_time_utc must be" in capsys.readouterr().err


# ── fetch geometry ──────────────────────────────────────────────────────────

def test_utm_zone_rule_matches_rslearn():
    assert fetch._ai2_utm_epsg(11.658, 57.27) == 32632      # footprint centroid
    assert fetch._ai2_utm_epsg(14.1108, 57.7985) == 32633   # first GCP of the full scene


def test_scene_window_matches_rslearn():
    # rslearn get_utm_ups_projection + STGeometry.to_projection + int() bounds,
    # computed in AI2's own environment for this footprint.
    assert fetch._ai2_scene_window(SKAGEN_FOOTPRINT) == (32632, (51737, -645423, 80357, -624992))


def test_aoi_lies_on_the_window_lattice_and_is_clipped():
    window = (51737, -645423, 80357, -624992)
    assert fetch._s1_aoi_pixel_bounds(SKAGEN_BBOX, 32632, window) == (58853, -643747, 66157, -637960)
    wide = {"west": 7.0, "south": 55.0, "east": 16.0, "north": 60.0}
    assert fetch._s1_aoi_pixel_bounds(wide, 32632, window) == window
    with pytest.raises(fetch.FetchError):
        fetch._s1_aoi_pixel_bounds({"west": 20.0, "south": 60.0, "east": 20.1, "north": 60.1},
                                   32632, window)


def test_cache_key_is_deterministic_and_pinned():
    key = fetch._s1_vessel_cache_key(SKAGEN_BBOX, "2026-06-28", [_TARGET_ID])
    # Pinned: a change of the key derivation (or of a version constant) must be deliberate.
    # (Format 2 keyed date + bbox only: "76b7613d2936b018" for this request; chain 1,
    # GCPs on the ellipsoid: "f531577c8b5af38c".)
    assert key == _PINNED_KEY[fetch.S1_GCP_SAMPLE_CENTRE]
    assert key == fetch._s1_vessel_cache_key(dict(reversed(list(SKAGEN_BBOX.items()))),
                                             "2026-06-28", [_TARGET_ID])
    assert key != fetch._s1_vessel_cache_key(SKAGEN_BBOX, "2026-06-29", [_TARGET_ID])
    assert key != fetch._s1_vessel_cache_key({**SKAGEN_BBOX, "west": 10.51}, "2026-06-28",
                                             [_TARGET_ID])


# Chain 2 by S1_GCP_SAMPLE_CENTRE: with the half pixel, and without it, where False
# must give exactly the key of the chain before the step.
_PINNED_KEY = {True: "a29762473a72f9cf", False: "1b0c720a3cc179a3"}


@pytest.mark.parametrize("flag", [True, False])
def test_sample_centre_flag_is_read_at_call_time(tmp_path, monkeypatch, flag):
    """The flag decides GCP offset, geocoding model, cache key and window names
    together, when they are computed; False is the chain without the step."""
    header = synthetic_grd(tmp_path / "grd.tif")
    monkeypatch.setattr(fetch, "S1_GCP_SAMPLE_CENTRE", flag)
    half = 0.5 if flag else 0.0
    assert fetch._s1_gcp_half_px() == half
    assert [(g[1], g[2]) for g in fetch._s1_sea_surface_gcps(header)] == [
        (g[1] + half, g[2] + half) for g in header["gcps"]]
    assert ("gcp_pixel_convention" in fetch._s1_geocoding_model()) is flag
    key = fetch._s1_vessel_cache_key(SKAGEN_BBOX, "2026-06-28", [_TARGET_ID])
    assert key == _PINNED_KEY[flag]
    stem = fetch._s1_window_stem("vv", {"col_off": 1, "row_off": 2, "width": 3, "height": 4})
    assert stem.endswith(f"_g{fetch._s1_geocoding_id()}")


def test_cache_key_and_window_files_follow_the_geocoding_model(monkeypatch):
    """Other tie points = other pixels: neither the model input nor a cached radar
    window written under another geocoding may be reused."""
    key = fetch._s1_vessel_cache_key(SKAGEN_BBOX, "2026-06-28", [_TARGET_ID])
    stem = fetch._s1_window_stem("vv", {"col_off": 1, "row_off": 2, "width": 3, "height": 4})
    model = fetch._s1_geocoding_model()
    assert model["grid_sha256"] == geoid.egm96_grid_sha256()
    monkeypatch.setattr(fetch, "_s1_geocoding_model", lambda: {**model, "grid_sha256": "x"})
    assert fetch._s1_vessel_cache_key(SKAGEN_BBOX, "2026-06-28", [_TARGET_ID]) != key
    assert fetch._s1_window_stem("vv", {"col_off": 1, "row_off": 2, "width": 3,
                                        "height": 4}) != stem


def test_cache_key_differs_per_pass_and_ignores_slice_order():
    s = _skagen_0627()
    dsc = [s["dsc_s"].id, s["dsc_n"].id]
    key = fetch._s1_vessel_cache_key(SKAGEN_BBOX, "2026-06-27", dsc)
    assert key == fetch._s1_vessel_cache_key(SKAGEN_BBOX, "2026-06-27", dsc[::-1])
    assert key != fetch._s1_vessel_cache_key(SKAGEN_BBOX, "2026-06-27", [s["asc"].id])
    assert key != fetch._s1_vessel_cache_key(SKAGEN_BBOX, "2026-06-27", dsc[:1])


def test_line_time_interpolates_within_the_slice():
    scene = SimpleNamespace(start_utc="2026-06-28T05:31:35.545314Z",
                            end_utc="2026-06-28T05:32:00.545314Z")
    assert fetch._s1_line_time_utc(scene, 0.0) == "2026-06-28T05:31:35.545314Z"
    assert fetch._s1_line_time_utc(scene, 0.5) == "2026-06-28T05:31:48.045314Z"
    assert fetch._s1_line_time_utc(scene, 1.0) == "2026-06-28T05:32:00.545314Z"
    with pytest.raises(ValueError):
        fetch._s1_line_time_utc(scene, 1.01)


def test_atomic_write_replaces_without_leaving_a_temp_file(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("old")
    fetch._atomic_write_text(path, '{"a": 1}')
    assert json.loads(path.read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


# ── synthetic GRD: sparse VRT, VRT XML, full offline fetch ──────────────────

def _synthetic_footprint(path: Path) -> dict:
    import rasterio
    from rasterio.transform import GCPTransformer

    with rasterio.open(path) as src:
        tps = GCPTransformer(src.gcps[0], tps=True)
        corners = [(0, 0), (0, src.width), (src.height, src.width), (src.height, 0), (0, 0)]
        return {"type": "Polygon",
                "coordinates": [[list(tps.xy(r, c, offset="ul")) for r, c in corners]]}


_SYNTH_AOI = {"west": 11.90, "south": 57.86, "east": 11.96, "north": 57.89}


def _cache_radar_window(full: Path, header: dict, pol: str, epsg: int, aoi_px: tuple,
                        scene_dir: Path) -> dict:
    """Pre-populate the fetch's radar-window cache from a local full GRD (no network)."""
    import rasterio
    from rasterio.windows import Window

    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / f"{pol}_header.json").write_text(json.dumps(header))
    rwin = fetch._s1_radar_window(header, epsg, aoi_px)
    stem = fetch._s1_window_stem(pol, rwin)
    with rasterio.open(full) as src:
        crop = src.read(1, window=Window(rwin["col_off"], rwin["row_off"],
                                         rwin["width"], rwin["height"]))
    with rasterio.open(scene_dir / f"{stem}.tif", "w", driver="GTiff", width=crop.shape[1],
                       height=crop.shape[0], count=1, dtype="uint16", nodata=0) as dst:
        dst.write(crop, 1)
    return rwin


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_sparse_scene_vrt_equals_full_scene_chain(tmp_path):
    """The fetch's core claim: a full-size VRT holding real DN only in the radar
    window gives, inside the AOI, exactly what the whole scene gives (the whole
    scene carrying the same sea-surface GCPs)."""
    full = tmp_path / "full.tif"
    header = synthetic_grd(full)
    full_sea = sea_surface_copy(full, header, tmp_path / "full_sea.tif")
    epsg, window = fetch._ai2_scene_window(_synthetic_footprint(full))
    aoi_px = fetch._s1_aoi_pixel_bounds(_SYNTH_AOI, epsg, window)
    scene_dir = tmp_path / "s1_grd" / "SYNTH"
    rwin = _cache_radar_window(full, header, "vv", epsg, aoi_px, scene_dir)
    assert rwin["width"] * rwin["height"] < 0.25 * header["width"] * header["height"]
    margin = fetch._S1_RADAR_MARGIN_PX            # the AOI centre lies inside the margin
    assert rwin["row_off"] + margin <= rwin["aoi_center_row"] <= \
        rwin["row_off"] + rwin["height"] - margin
    assert rwin["lines"] == header["height"]

    vrt, _, step1_epsg, signed = fetch._s1_sparse_scene_vrt(
        SimpleNamespace(id="SYNTH"), "vv", epsg, aoi_px, scene_dir, None)
    assert signed is None and (step1_epsg, epsg) == (32633, 32632)
    sparse = fetch._s1_grd_to_window_grid(vrt, step1_epsg, epsg, window, aoi_px)
    reference = fetch._s1_grd_to_window_grid(full_sea, step1_epsg, epsg, window, aoi_px)
    assert sparse.shape == (aoi_px[3] - aoi_px[1], aoi_px[2] - aoi_px[0])
    assert sparse.dtype == np.uint16
    assert (reference > 0).mean() > 0.99
    np.testing.assert_array_equal(sparse, reference)


def _synthetic_annotation(header: dict, pol: str) -> bytes:
    """A product annotation whose geolocation grid is the synthetic GRD's GCPs."""
    import xml.etree.ElementTree as ET

    root = ET.Element("product")
    ET.SubElement(ET.SubElement(root, "adsHeader"), "polarisation").text = pol.upper()
    grid = ET.SubElement(ET.SubElement(root, "geolocationGrid"), "geolocationGridPointList")
    for (_, c, r, x, y, z), inc in zip(header["gcps"], header["gcp_incidence_deg"], strict=True):
        point = ET.SubElement(grid, "geolocationGridPoint")
        for tag, value in (("line", r), ("pixel", c), ("latitude", y), ("longitude", x),
                           ("height", z), ("incidenceAngle", inc)):
            ET.SubElement(point, tag).text = repr(float(value))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_scene_header_and_radar_window_are_fetched_annotated_and_cached(tmp_path,
                                                                       monkeypatch):
    """The network/disk path with the network faked: measurement header + annotation
    download -> incidence per GCP -> header and XML cached -> read back offline; then
    the radar window cut from the measurement and written with the sea-surface GCPs."""
    import io
    import urllib.request

    import rasterio
    from rasterio.windows import Window

    safe = tmp_path / "remote" / "S1D_SYNTH_71E0.SAFE"
    meas = {pol: safe / "measurement" / f"iw-{pol}.tiff" for pol in ("vv", "vh")}
    truth = {pol: synthetic_grd(path) for pol, path in meas.items()}
    served = {str(safe / "annotation" / f"iw-{pol}.xml"): _synthetic_annotation(truth[pol], pol)
              for pol in meas}
    opened, signs = [], []

    def urlopen(url, timeout=None):
        opened.append(url)
        return io.BytesIO(served[url])

    signed = SimpleNamespace(id="SYNTH", assets={
        pol: SimpleNamespace(href=str(path), extra_fields={}) for pol, path in meas.items()})
    monkeypatch.setattr(mpc_s1, "sign_item", lambda item: signs.append(item.id) or signed)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    item = SimpleNamespace(id="SYNTH")
    scene_dir = tmp_path / "cache" / "s1_grd" / "SYNTH"

    header, got = fetch._s1_scene_header(item, "vv", scene_dir, None)
    ann = str(safe / "annotation" / "iw-vv.xml")
    assert got is signed and signs == ["SYNTH"] and opened == [ann]
    assert header["annotation_href"] == ann and header["href"] == str(meas["vv"])
    assert header["gcps"] == truth["vv"]["gcps"]
    assert header["gcp_incidence_deg"] == truth["vv"]["gcp_incidence_deg"]
    assert (header["width"], header["height"], header["dtype"]) == (2400, 1800, "uint16")
    assert (scene_dir / "vv_annotation.xml").read_bytes() == served[ann]
    assert json.loads((scene_dir / "vv_header.json").read_text()) == header
    assert not list(scene_dir.glob("*.part"))
    # Read back: neither signing nor any download.
    monkeypatch.setattr(mpc_s1, "sign_item", _no_network)
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    assert fetch._s1_scene_header(item, "vv", scene_dir, None) == (header, None)

    # The radar window: real DN cut from the measurement, sea-surface GCPs in crop space,
    # the sea-surface GCPs computed once for window, TIFF and VRT.
    monkeypatch.setattr(mpc_s1, "sign_item", lambda item: signed)
    real_sea, sea_calls = fetch._s1_sea_surface, []
    monkeypatch.setattr(fetch, "_s1_sea_surface",
                        lambda h: sea_calls.append(h["scene_id"]) or real_sea(h))
    epsg, window = fetch._ai2_scene_window(_synthetic_footprint(meas["vv"]))
    aoi_px = fetch._s1_aoi_pixel_bounds(_SYNTH_AOI, epsg, window)
    vrt, rwin, _, _ = fetch._s1_sparse_scene_vrt(item, "vv", epsg, aoi_px, scene_dir, None)
    assert sea_calls == ["SYNTH"]
    sea = real_sea(header)[0]
    win = scene_dir / f"{fetch._s1_window_stem('vv', rwin)}.tif"
    with rasterio.open(win) as src, rasterio.open(meas["vv"]) as full:
        gcps, crs = src.gcps
        np.testing.assert_array_equal(src.read(1), full.read(1, window=Window(
            rwin["col_off"], rwin["row_off"], rwin["width"], rwin["height"])))
        tags = src.tags()
    assert crs.to_epsg() == 4326
    assert (tags["SCENE_ID"], tags["COL_OFF"], tags["ROW_OFF"]) == (
        "SYNTH", str(rwin["col_off"]), str(rwin["row_off"]))
    assert [(g.col, g.row, g.x, g.y, g.z) for g in gcps] == [
        (c - rwin["col_off"], r - rwin["row_off"], x, y, z) for _, c, r, x, y, z in sea]
    with rasterio.open(vrt) as src:
        vgcps, _ = src.gcps
    assert [(g.col, g.row, g.x, g.y, g.z) for g in vgcps] == [tuple(g[1:]) for g in sea]


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_sparse_vrt_xml_escapes_wkt_and_rejects_unknown_dtype(tmp_path):
    import xml.etree.ElementTree as ET

    import rasterio
    from rasterio.crs import CRS

    wkt = CRS.from_epsg(4326).to_wkt()
    assert '"' in wkt                                   # would break an f-string attribute
    header = {"scene_id": "S", "width": 40, "height": 30, "dtype": "uint16", "nodata": 0.0,
              "gcp_crs": wkt,
              "gcps": [[str(i), float(c), float(r), 12.0 + c * 1e-4, 57.0 - r * 1e-4, 0.0]
                       for i, (c, r) in enumerate([(0, 0), (40, 0), (0, 30), (40, 30)])],
              "gcp_incidence_deg": [33.0, 37.0, 33.0, 37.0]}     # rises with pixel
    rwin = {"col_off": 10, "row_off": 5, "width": 20, "height": 10}
    with rasterio.open(tmp_path / "win.tif", "w", driver="GTiff", width=20, height=10, count=1,
                       dtype="uint16") as dst:
        dst.write(np.arange(1, 201, dtype=np.uint16).reshape(10, 20), 1)
    xml = fetch._s1_sparse_vrt_xml(header, rwin, "win.tif")
    assert ET.fromstring(xml).find("GCPList").get("Projection") == wkt
    (tmp_path / "s.vrt").write_text(xml)
    with rasterio.open(tmp_path / "s.vrt") as src:
        gcps, gcp_crs = src.gcps
        data = src.read(1)
    assert (len(gcps), gcp_crs.to_epsg(), data.shape) == (4, 4326, (30, 40))
    # The VRT carries the sea-surface GCPs: z = N at the raw position, moved in lon/lat.
    raw = header["gcps"]
    np.testing.assert_allclose([g.z for g in gcps],
                               geoid.egm96_undulation([g[3] for g in raw], [g[4] for g in raw]))
    assert all(abs(g.x - r[3]) > 1e-5 for g, r in zip(gcps, raw, strict=True))
    # ESA sample centres on GDAL's corner lattice (S1_GCP_SAMPLE_CENTRE): half on.
    half = fetch._s1_gcp_half_px()
    assert [(g.col, g.row) for g in gcps] == [(r[1] + half, r[2] + half) for r in raw]
    np.testing.assert_array_equal(data[5:15, 10:30], np.arange(1, 201).reshape(10, 20))
    assert data[:5].sum() == 0
    with pytest.raises(fetch.FetchError, match="'float32' is not supported"):
        fetch._s1_sparse_vrt_xml({**header, "dtype": "float32"}, rwin, "win.tif")


def _synthetic_scene_items(footprint: dict) -> tuple[SimpleNamespace, SimpleNamespace]:
    target = _item(_TARGET_ID, _T0, footprint, end="2026-06-28T05:32:00.545314Z")
    hist = _item("S1D_IW_GRDH_1SDV_20260523T053133_20260523T053158_002907_004F80",
                 "2026-05-23T05:31:33.100000Z", footprint,
                 end="2026-05-23T05:31:58.100000Z", abs_orbit=2907)
    return target, hist


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_fetch_offline_end_to_end_then_cache_round_trip(tmp_path, monkeypatch):
    """fetch_s1_vessel_scenes with the catalogue faked and the radar windows cached:
    uint16 output, AOI-centre scene time, cache written and answered offline."""
    import rasterio
    from rasterio.transform import GCPTransformer
    from pyproj import Transformer

    full = tmp_path / "full.tif"
    header = synthetic_grd(full)
    full_sea = sea_surface_copy(full, header, tmp_path / "full_sea.tif")
    footprint = _synthetic_footprint(full)
    target, hist = _synthetic_scene_items(footprint)
    epsg, window = fetch._ai2_scene_window(footprint)
    aoi_px = fetch._s1_aoi_pixel_bounds(_SYNTH_AOI, epsg, window)
    cache = tmp_path / "cache"
    for item in (target, hist):
        for pol in ("vv", "vh"):
            _cache_radar_window(full, {**header, "scene_id": item.id}, pol, epsg, aoi_px,
                                cache / "s1_grd" / item.id)
    monkeypatch.setattr(mpc_s1, "search_grd_items", FakeCatalogue([target, hist]))
    monkeypatch.setattr(mpc_s1, "sign_item", _no_network)

    scenes = fetch.fetch_s1_vessel_scenes(_SYNTH_AOI, "2026-06-28", cache_dir=cache)
    shape = (aoi_px[3] - aoi_px[1], aoi_px[2] - aoi_px[0])
    assert scenes.target.dtype == scenes.historical.dtype == np.uint16
    assert scenes.target.shape == scenes.historical.shape == (2, *shape)
    reference = fetch._s1_grd_to_window_grid(full_sea, 32633, epsg, window, aoi_px)
    np.testing.assert_array_equal(scenes.target[0], reference)
    np.testing.assert_array_equal(scenes.historical[1], reference)
    assert scenes.geo.transform == Affine(10, 0, aoi_px[0] * 10, 0, -10, aoi_px[1] * -10)
    assert scenes.target_scene.scene_id == target.id
    assert scenes.historical_scene.scene_id == hist.id
    assert scenes.provenance["valid_frac_target"] > 0.99
    # Chain 2: the tie-point model and each window's GCP shifts (z = 0 GCPs, all "sea":
    # N / tan(incidence), 34-63 m away from the satellite).
    assert scenes.provenance["chain_version"] == fetch.S1_VESSEL_CHAIN_VERSION == 2
    assert scenes.provenance["geocoding"] == fetch._s1_geocoding_model()
    n = geoid.egm96_undulation([g[3] for g in header["gcps"]], [g[4] for g in header["gcps"]])
    want = n / np.tan(np.radians(header["gcp_incidence_deg"]))
    for rw in scenes.provenance["radar_windows"].values():
        sea = rw["gcp_shift_m"]["sea"]
        assert sea["n"] == len(header["gcps"])
        assert (sea["min"], sea["max"]) == pytest.approx((want.min(), want.max()), abs=0.01)
        assert 30 < sea["min"] < sea["max"] < 65
    # One pass, one slice: the pass record says so and the slice API is the scene.
    assert (scenes.scene_id, scenes.historical_scene_id) == (target.id, hist.id)
    assert scenes.target_slices == [scenes.target_scene]
    assert scenes.historical_slices == [scenes.historical_scene]
    assert (scenes.start_utc, scenes.end_utc) == (scenes.target_scene.start_utc,
                                                  scenes.target_scene.end_utc)
    rec = scenes.provenance["pass"]
    assert rec["slice_ids"] == [target.id] and rec["other_passes"] == []
    assert rec["selected_by"] == "the only pass on the date" and rec["skipped_slices"] == []
    assert rec["scene_time_from"] == target.id
    assert [(s["scene_id"], s["historical_scene_id"], s["rank"]) for s in rec["slices"]] == \
        [(target.id, hist.id, 0)]
    assert rec["slices"][0]["aoi_px"] == scenes.provenance["aoi_px"] == list(aoi_px)
    assert rec["slices"][0]["ai2_window_px"] == scenes.provenance["ai2_window_px"] == list(window)
    assert rec["slices"][0]["mosaic_px"] == np.count_nonzero(scenes.target[0])
    bands = scenes.bands()
    assert set(bands) == set(ALL_BAND_KEYS)
    assert np.shares_memory(bands["VV_DN"], scenes.target)

    # Scene time: the AOI centre's radar line, found here independently of the fetch
    # (GDAL's TPS over the sea-surface GCPs, read back from the reference raster).
    with rasterio.open(full_sea) as src:
        tps = GCPTransformer(src.gcps[0], tps=True)
    lon_c, lat_c = Transformer.from_crs(epsg, 4326, always_xy=True).transform(
        (aoi_px[0] + aoi_px[2]) * 5, (aoi_px[1] + aoi_px[3]) * -5)
    row_c, _ = tps.rowcol(lon_c, lat_c, op=float)
    start = fetch._parse_utc(target.properties["start_datetime"])
    expected = start + (row_c / header["height"]) * timedelta(seconds=25)
    got = fetch._parse_utc(scenes.scene_time_utc)
    assert abs((got - expected).total_seconds()) < 1e-3
    assert 0 < (got - start).total_seconds() < 25

    # Cache: both files, no temp files. A second request runs the target-day search
    # (the pass's slices are in the key) but no historical search, read or warp.
    key = fetch._s1_vessel_cache_key(_SYNTH_AOI, "2026-06-28", [target.id])
    out_dir = cache / "s1_vessel_inputs"
    assert sorted(p.name for p in out_dir.iterdir()) == [f"2026-06-28_{key}.json",
                                                         f"2026-06-28_{key}.tif"]
    assert not list(cache.rglob("*.part"))
    cat = FakeCatalogue([target, hist])
    monkeypatch.setattr(mpc_s1, "search_grd_items", cat)
    monkeypatch.setattr(fetch, "_s1_grd_to_window_grid", _no_network)
    again = fetch.fetch_s1_vessel_scenes(_SYNTH_AOI, "2026-06-28", cache_dir=cache)
    assert [c["interval"] for c in cat.calls] == ["2026-06-28T00:00:00Z/2026-06-29T00:00:00Z"]
    np.testing.assert_array_equal(again.target, scenes.target)
    np.testing.assert_array_equal(again.historical, scenes.historical)
    assert again.target.dtype == np.uint16
    assert again.scene_time_utc == scenes.scene_time_utc
    assert again.provenance == scenes.provenance
    assert again.target_slices == scenes.target_slices
    assert again.historical_slices == scenes.historical_slices


def _iso(t) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _synthetic_pass(tmp_path: Path, cuts: list[tuple[int, int]]) -> list[dict]:
    """Slices of the synthetic scene (lines ``r0:r1`` each) as one S1D datatake,
    each with a historical twin 36 days earlier (same raster, so a correctly
    paired historical equals its target pixel for pixel)."""
    t0, line = fetch._parse_utc(_T0), timedelta(seconds=25) / 1800
    out = []
    for k, (r0, r1) in enumerate(cuts):
        path = tmp_path / f"slice{k}.tif"
        header = synthetic_grd(path, rows=(r0, r1))
        fp = _synthetic_footprint(path)
        start, end = t0 + r0 * line, t0 + r1 * line
        h_start, h_end = start - timedelta(days=36), end - timedelta(days=36)
        out.append({
            "path": path, "header": header, "rows": (r0, r1),
            "item": _item(f"S1D_IW_GRDH_1SDV_{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}_"
                          f"003432_0060CF_{k}", _iso(start), fp, end=_iso(end)),
            "hist": _item(f"S1D_IW_GRDH_1SDV_{h_start:%Y%m%dT%H%M%S}_{h_end:%Y%m%dT%H%M%S}_"
                          f"002907_004F80_{k}", _iso(h_start), fp, end=_iso(h_end),
                          abs_orbit=2907),
        })
    return out


def _cache_slice(cache: Path, sl: dict, epsg: int) -> tuple[tuple, tuple, np.ndarray]:
    """Cache one slice's radar windows (target + historical) as the fetch will request
    them, and return (window, AOI part, the slice's full-raster chain on that part)."""
    window = fetch._ai2_scene_window(sl["item"].geometry, epsg)[1]
    sub = fetch._s1_aoi_pixel_bounds(_SYNTH_AOI, epsg, window)
    for item in (sl["item"], sl["hist"]):
        for pol in ("vv", "vh"):
            _cache_radar_window(sl["path"], {**sl["header"], "scene_id": item.id}, pol, epsg,
                                sub, cache / "s1_grd" / item.id)
    sea = sea_surface_copy(sl["path"], sl["header"], sl["path"].with_suffix(".sea.tif"))
    first = fetch._s1_sea_surface_gcps(sl["header"])[0]
    ref = fetch._s1_grd_to_window_grid(sea, fetch._ai2_utm_epsg(first[3], first[4]),
                                       epsg, window, sub)
    return window, sub, ref


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
@pytest.mark.parametrize("split", [720, 500])     # AOI lines 392-822, its centre on line 607
def test_fetch_mosaics_the_slices_of_one_pass(tmp_path, monkeypatch, split):
    """An AOI across the cut between two slices of one datatake: both slices are used,
    each with its own historical scene, on one lattice; every pixel is its slice's own
    full-scene chain value; the AOI is dated by the slice holding its centre."""
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import GCPTransformer

    slices = _synthetic_pass(tmp_path, [(0, split), (split, 1800)])
    cache = tmp_path / "cache"
    epsg = 32632                      # both slice centroids lie in zone 32
    refs = [_cache_slice(cache, sl, epsg) for sl in slices]
    cat = FakeCatalogue([x for sl in slices for x in (sl["item"], sl["hist"])])
    monkeypatch.setattr(mpc_s1, "search_grd_items", cat)
    monkeypatch.setattr(mpc_s1, "sign_item", _no_network)

    scenes = fetch.fetch_s1_vessel_scenes(_SYNTH_AOI, "2026-06-28", cache_dir=cache)

    # Ranked by AOI cover: the slice holding more of lines 392-822 comes first.
    ranked = [0, 1] if split == 720 else [1, 0]
    assert scenes.target_scene.scene_id == slices[ranked[0]]["item"].id
    assert scenes.historical_scene.scene_id == slices[ranked[0]]["hist"].id
    # Grid: the AOI on the shared lattice (the union of the two windows).
    union = tuple(f(r[0][i] for r in refs) for i, f in enumerate((min, min, max, max)))
    aoi_px = fetch._s1_aoi_pixel_bounds(_SYNTH_AOI, epsg, union)
    c0, r0, c1, r1 = aoi_px
    assert scenes.target.shape == scenes.historical.shape == (2, r1 - r0, c1 - c0)
    assert scenes.geo.transform == Affine(10, 0, c0 * 10, 0, -10, r0 * -10)
    # Expected mosaic from the independent full-raster chains: the first-ranked slice
    # whole, then the other where it has data and the first has none.
    expected = np.zeros((r1 - r0, c1 - c0), dtype=np.uint16)
    owner = np.full(expected.shape, -1)
    for rank, k in enumerate(ranked):
        _, (a0, b0, a1, b1), ref = refs[k]
        rows, cols = slice(b0 - r0, b1 - r0), slice(a0 - c0, a1 - c0)
        part, own = expected[rows, cols], owner[rows, cols]
        take = (ref != 0) if rank == 0 else (own < 0) & (ref != 0)
        part[take] = ref[take]
        own[take] = rank
    np.testing.assert_array_equal(scenes.target[0], expected)
    # Each pixel's historical comes from the same slice as its target (twin rasters).
    np.testing.assert_array_equal(scenes.historical, scenes.target)
    second = np.count_nonzero(owner == 1)
    assert second > 0.1 * expected.size                 # the second slice really contributes
    assert scenes.provenance["valid_frac_target"] > 0.995
    assert np.count_nonzero(expected) > max(np.count_nonzero(r[2]) for r in refs)

    # Slices in acquisition order, joined ids, first start, last end.
    assert [s.scene_id for s in scenes.target_slices] == [sl["item"].id for sl in slices]
    assert [s.scene_id for s in scenes.historical_slices] == [sl["hist"].id for sl in slices]
    assert scenes.scene_id == " + ".join(sl["item"].id for sl in slices)
    assert scenes.historical_scene_id == " + ".join(sl["hist"].id for sl in slices)
    assert scenes.start_utc == slices[0]["item"].properties["start_datetime"]
    assert scenes.end_utc == slices[1]["item"].properties["end_datetime"]
    rec = scenes.provenance["pass"]
    assert rec["slice_ids"] == [sl["item"].id for sl in slices] and rec["skipped_slices"] == []
    assert [s["rank"] for s in rec["slices"]] == [ranked.index(0), ranked.index(1)]
    assert [s["aoi_px"] for s in rec["slices"]] == [list(r[1]) for r in refs]
    assert [s["mosaic_px"] for s in rec["slices"]] == \
        [int(np.count_nonzero(owner == ranked.index(k))) for k in (0, 1)]
    assert set(scenes.provenance["radar_windows"]) == {
        f"{x.id}/{pol}" for sl in slices for x in (sl["item"], sl["hist"]) for pol in ("vv", "vh")}

    # AOI time: line 607 of the datatake, wherever the cut is; found here with the
    # whole scene's TPS, independently of the slices.
    full = tmp_path / "full.tif"
    full_sea = sea_surface_copy(full, synthetic_grd(full), tmp_path / "full_sea.tif")
    with rasterio.open(full_sea) as src:
        tps = GCPTransformer(src.gcps[0], tps=True)
    lon_c, lat_c = Transformer.from_crs(epsg, 4326, always_xy=True).transform(
        (c0 + c1) * 5, (r0 + r1) * -5)
    row_c, _ = tps.rowcol(lon_c, lat_c, op=float)
    expected_t = fetch._parse_utc(_T0) + row_c * timedelta(seconds=25) / 1800
    assert abs((fetch._parse_utc(scenes.scene_time_utc) - expected_t).total_seconds()) < 2e-3
    holder = 0 if row_c < split else 1
    assert rec["scene_time_from"] == slices[holder]["item"].id

    # Key = the pass's slices; the cache answers a second request.
    key = fetch._s1_vessel_cache_key(_SYNTH_AOI, "2026-06-28",
                                     [sl["item"].id for sl in slices])
    assert (cache / "s1_vessel_inputs" / f"2026-06-28_{key}.json").exists()
    monkeypatch.setattr(fetch, "_s1_grd_to_window_grid", _no_network)
    again = fetch.fetch_s1_vessel_scenes(_SYNTH_AOI, "2026-06-28", cache_dir=cache)
    np.testing.assert_array_equal(again.target, scenes.target)
    assert again.target_slices == scenes.target_slices
    assert again.scene_id == scenes.scene_id and again.provenance == scenes.provenance


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_fetch_skips_a_slice_once_the_aoi_is_covered(tmp_path, monkeypatch):
    """A second slice of the pass that touches the AOI is neither searched for a
    historical scene nor downloaded when the first slice already covers every AOI
    pixel: the result is the one-slice fetch (2026-06-28 Skagen has such a sliver)."""
    slices = _synthetic_pass(tmp_path, [(0, 1800), (300, 420)])
    cache = tmp_path / "cache"
    window, sub, ref = _cache_slice(cache, slices[0], 32632)       # only the first is cached
    assert np.count_nonzero(ref) == ref.size
    cat = FakeCatalogue([x for sl in slices for x in (sl["item"], sl["hist"])])
    monkeypatch.setattr(mpc_s1, "search_grd_items", cat)
    monkeypatch.setattr(mpc_s1, "sign_item", _no_network)

    scenes = fetch.fetch_s1_vessel_scenes(_SYNTH_AOI, "2026-06-28", cache_dir=cache)
    np.testing.assert_array_equal(scenes.target[0], ref)
    assert scenes.provenance["aoi_px"] == list(sub)
    assert scenes.scene_id == slices[0]["item"].id
    assert scenes.historical_scene_id == slices[0]["hist"].id
    rec = scenes.provenance["pass"]
    assert rec["slice_ids"] == [sl["item"].id for sl in slices]
    assert rec["skipped_slices"] == [{"scene_id": slices[1]["item"].id,
                                      "reason": "AOI already covered"}]
    assert len(cat.calls) == 2                            # target day + one historical window
    assert not (cache / "s1_grd" / slices[1]["item"].id).exists()


def _synthetic_vessel_scenes():
    rng = np.random.default_rng(3)
    target = rng.integers(0, 65535, (2, 37, 53), dtype=np.uint16)
    historical = rng.integers(0, 65535, (2, 37, 53), dtype=np.uint16)
    transform = Affine(10, 0, 588530, 0, -10, 6437470)
    geo = GeoContext(crs="EPSG:32632", transform=transform,
                     bounds_projected={"west": 588530, "south": 6437100, "east": 589060,
                                       "north": 6437470},
                     bounds_wgs84=dict(SKAGEN_BBOX), shape=(37, 53))
    info = fetch.S1SceneInfo(scene_id="S1D_A", safe_name="S1D_A_71E0.SAFE",
                             start_utc="2026-06-28T05:31:35.545314Z",
                             end_utc="2026-06-28T05:32:00.545314Z", platform="sentinel-1d",
                             orbit_direction="DESCENDING", relative_orbit=66,
                             absolute_orbit=3432, footprint=SKAGEN_FOOTPRINT)
    hinfo = fetch.S1SceneInfo(**{**info.__dict__, "scene_id": "S1D_B", "absolute_orbit": 2907})
    scenes = fetch.S1VesselScenes(target=target, historical=historical, geo=geo,
                                  target_scene=info, historical_scene=hinfo,
                                  scene_time_utc="2026-06-28T05:31:38.700000Z",
                                  provenance={"chain_version": 1, "aoi_px": [1, 2, 3, 4]})
    return scenes, info, hinfo


def test_cache_write_read_round_trip(tmp_path):
    scenes, info, hinfo = _synthetic_vessel_scenes()
    target, historical, geo = scenes.target, scenes.historical, scenes.geo
    transform = geo.transform
    tif, meta = tmp_path / "c" / "x.tif", tmp_path / "c" / "x.json"
    fetch._write_s1_vessel_cache(tif, meta, scenes)
    back = fetch._load_s1_vessel_cache(tif, meta)
    assert back.target.dtype == back.historical.dtype == np.uint16
    np.testing.assert_array_equal(back.target, target)
    np.testing.assert_array_equal(back.historical, historical)
    assert back.geo.transform == transform and isinstance(back.geo.transform, Affine)
    assert (back.geo.crs, back.geo.shape) == ("EPSG:32632", (37, 53))
    assert back.geo.bounds_projected == geo.bounds_projected
    assert back.geo.bounds_wgs84 == geo.bounds_wgs84
    assert back.target_scene == info and back.historical_scene == hinfo
    assert back.scene_time_utc == scenes.scene_time_utc
    assert back.provenance == scenes.provenance
    assert sorted(p.name for p in tif.parent.iterdir()) == ["x.json", "x.tif"]
    with pytest.raises(TypeError, match="uint16"):
        fetch._write_s1_vessel_cache(tif, meta, fetch.S1VesselScenes(
            **{**scenes.__dict__, "target": target.astype(np.float32)}))


def _replace_after_barrier(monkeypatch, parties: int = 2):
    """Make every os.replace wait until all writers have written their temp file.

    That is the interleaving that broke the shared ``.part`` name: writer A
    publishes, then writer B's replace finds its (shared) temp file gone.
    """
    import threading

    barrier = threading.Barrier(parties, timeout=20)
    real_replace = os.replace

    def replace(src, dst):
        barrier.wait()
        return real_replace(src, dst)

    monkeypatch.setattr(fetch.os, "replace", replace)


def _run_concurrently(fn, n: int = 2) -> list:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(n) as pool:
        futures = [pool.submit(fn) for _ in range(n)]
        return [f.exception() for f in futures]


def test_concurrent_atomic_text_writers_do_not_collide(tmp_path, monkeypatch):
    target = tmp_path / "header.json"
    _replace_after_barrier(monkeypatch)
    errors = _run_concurrently(lambda: fetch._atomic_write_text(target, '{"a": 1}'))
    assert errors == [None, None]
    assert target.read_text() == '{"a": 1}'
    assert [p.name for p in tmp_path.iterdir()] == ["header.json"]   # no temp left behind


def test_concurrent_cache_writers_publish_a_consistent_pair(tmp_path, monkeypatch):
    scenes, _, _ = _synthetic_vessel_scenes()
    tif, meta = tmp_path / "c" / "x.tif", tmp_path / "c" / "x.json"
    _replace_after_barrier(monkeypatch)
    errors = _run_concurrently(lambda: fetch._write_s1_vessel_cache(tif, meta, scenes))
    assert errors == [None, None]
    back = fetch._load_s1_vessel_cache(tif, meta)
    np.testing.assert_array_equal(back.target, scenes.target)
    np.testing.assert_array_equal(back.historical, scenes.historical)
    assert sorted(p.name for p in tif.parent.iterdir()) == ["x.json", "x.tif"]


def test_failed_cache_write_leaves_no_temp_file(tmp_path, monkeypatch):
    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(fetch.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        fetch._atomic_write_text(tmp_path / "x.json", "{}")
    assert list(tmp_path.iterdir()) == []


# ── model (needs the checkpoints) ───────────────────────────────────────────

_HAVE_CKPTS = s1.DEFAULT_DETECT_CKPT.exists() and s1.DEFAULT_ATTR_CKPT.exists()


@pytest.mark.skipif(not _HAVE_CKPTS, reason="AI2 S1 checkpoints not in imint/fm/ai2_vessels_s1/")
def test_model_reproduces_reference_detection_on_crop():
    """192 px crop of AI2's own input around one reference tanker, as uint16 DN.

    The crop is zero-padded to the 512 px patch, so context differs from AI2's
    run: position may move by a pixel and attributes by a few percent.
    """
    z = np.load(FIXTURE)
    dn = z["dn"]
    assert dn.dtype == np.uint16
    col0, row0 = (int(v) for v in z["origin_px"])
    (expected,) = json.loads(str(z["ai2_reference"]))
    bands = dict(zip(ALL_BAND_KEYS, dn, strict=True))
    res = s1.AI2S1VesselAnalyzer({"infra_filter": False}).run(
        None, bands=bands, geo=_geo(col0, row0, *dn.shape[1:]))
    assert res.success, res.error
    (det,) = res.outputs["regions"]
    col, row = col0 + det["point_px"][0], row0 + det["point_px"][1]
    assert abs(col - expected["col"]) <= 1 and abs(row - expected["row"]) <= 1
    assert det["score"] > 0.99
    attrs = det["attributes"]
    assert attrs["vessel_type"] == expected["vessel_type"] == "tanker"
    assert attrs["length_m"] == pytest.approx(expected["length"], rel=0.03)
    assert attrs["width_m"] == pytest.approx(expected["width"], rel=0.05)
    assert attrs["speed_knots"] == pytest.approx(expected["speed"], abs=0.5)
    assert abs((attrs["heading_deg"] - expected["heading"] + 180) % 360 - 180) < 3
    assert res.metadata["infra_filter"]["applied"] is False


def _serve_stalled_response() -> tuple[str, "socket.socket"]:
    """Local server that sends headers for a large file, then no bytes."""
    import socket
    import threading

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)

    def handle():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            conn.recv(4096)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100000000\r\n"
                         b"Accept-Ranges: bytes\r\nContent-Type: image/tiff\r\n\r\n")
            # Hold the connection open without sending the body.

    threading.Thread(target=handle, daemon=True).start()
    return f"http://127.0.0.1:{srv.getsockname()[1]}/stalled.tif", srv


def test_s1_gdal_env_turns_a_stalled_read_into_an_error():
    """The env's timeout keys must be names GDAL honours; a stall must raise, not hang."""
    import time

    import rasterio
    from rasterio.errors import RasterioIOError

    assert fetch._S1_GDAL_ENV["GDAL_HTTP_LOW_SPEED_TIME"] == "60"
    assert fetch._S1_GDAL_ENV["GDAL_HTTP_CONNECTTIMEOUT"] == "30"
    url, srv = _serve_stalled_response()
    env = {**fetch._S1_GDAL_ENV, "GDAL_HTTP_LOW_SPEED_TIME": "2", "GDAL_HTTP_MAX_RETRY": "0"}
    t0 = time.monotonic()
    try:
        with pytest.raises(RasterioIOError), rasterio.Env(**env):
            with rasterio.open(f"/vsicurl/{url}") as src:
                src.read(1)
    finally:
        srv.close()
    assert time.monotonic() - t0 < 20
