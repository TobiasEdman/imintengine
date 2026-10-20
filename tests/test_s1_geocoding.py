"""S1 vessel geocoding against independent truth (chain 2: tie points on the sea surface).

Until chain 2 no test compared S1 positions with anything independent: the
checks were TPS round trips, AI2 parity on identical (equally wrong) pixels and
self-consistency, and ships sat 41-56 m toward the satellite. These tests close
that gap (docs/marine/s1_vessels_reference.md, "Geocoding"):

1. the pipeline's sea-surface GCPs vs an independent range-Doppler solver
   (orbit + slant range from the product annotation) at h = N;
2. frozen radar positions of anchored ships, geocoded through the pipeline's
   own TPS, vs their AIS hull centres;
3. (slow, opt-in) the whole fetch + detector on 2026-06-28 against the same
   ships, and AI2 parity in radar coordinates.

Fixtures: tests/fixtures/s1_geocoding/ (see build_fixtures.py there).
"""
from __future__ import annotations

import functools
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest

from imint import fetch
from imint.geoid import egm96_undulation
from tests.s1_rd_oracle import RangeDopplerOracle

FIX = Path(__file__).parent / "fixtures" / "s1_geocoding"
SCENES = {
    "dsc0628": "S1D_IW_GRDH_1SDV_20260628T053135_20260628T053200_003432_0060CF",
    "asc0627": "S1C_IW_GRDH_1SDV_20260627T165257_20260627T165322_008294_010682",
}
SKAGEN_BBOX = {"west": 10.50, "south": 57.55, "east": 11.70, "north": 58.05}
# Acceptance per layer (diagnosis: per-ship sd 11-19 m after the fix, so the
# median's standard error is 3-4 m; a correct layer fails +-10 m at n = 20 in
# 0.6% of draws).
MIN_SHIPS = 20
MAX_MEDIAN_M = 10.0
MAX_CROSS_PASS_M = 15.0


@functools.cache
def _header(scene_id: str) -> dict:
    with gzip.open(FIX / "headers" / f"{scene_id}_vv_header.json.gz", "rt") as f:
        return json.load(f)


@functools.cache
def _oracle(scene_id: str) -> RangeDopplerOracle:
    return RangeDopplerOracle(FIX / "annotation" / f"{scene_id}_vv_trimmed.xml.gz")


def _en(lon0, lat0, lon1, lat1) -> np.ndarray:
    """(dE, dN) metres of point 1 relative to point 0 (geodesic, WGS84); shape (2, n)."""
    from pyproj import Geod

    az, _, d = Geod(ellps="WGS84").inv(*np.broadcast_arrays(
        *(np.asarray(v, float) for v in (lon0, lat0, lon1, lat1))))
    az = np.radians(az)
    return np.array([d * np.sin(az), d * np.cos(az)])


# ── 1. sea-surface GCPs vs range-Doppler ────────────────────────────────────

@pytest.mark.parametrize("scene", sorted(SCENES))
def test_s1_sea_surface_gcps_match_range_doppler(scene):
    sid = SCENES[scene]
    header, oracle = _header(sid), _oracle(sid)
    raw = np.array([g[1:] for g in header["gcps"]], dtype=float)      # col row lon lat z
    sea_gcps = fetch._s1_sea_surface_gcps(header)
    new = np.array([g[1:] for g in sea_gcps], dtype=float)
    # The header's angles are the annotation's, through the pipeline's own parser
    # (which also proves the annotation grid is the TIFF's GCPs point for point).
    assert fetch._s1_gcp_incidence(oracle.xml, header, "vv") == header["gcp_incidence_deg"]
    np.testing.assert_array_equal(raw[:, :2], np.c_[oracle.grid["pixel"], oracle.grid["line"]])

    # Guard: ESA solves sea GCPs at z ~ 0. If that convention changes, so must the fix.
    sea = np.abs(raw[:, 4]) < 1.0
    assert sea.sum() >= 50, f"only {sea.sum()} GCPs at |z| < 1 m"

    n = egm96_undulation(raw[:, 2], raw[:, 3])
    np.testing.assert_allclose(new[:, 4], n)                          # z = N

    # Corrected sea GCPs = the exact zero-Doppler solution on the sea surface.
    lon_n, lat_n = oracle.geolocate_grid(n)
    err = np.hypot(*_en(new[:, 2], new[:, 3], lon_n, lat_n))
    assert err[sea].max() < 0.5, f"sea GCP vs range-Doppler(h=N): max {err[sea].max():.3f} m"
    assert err[~sea].max() < 5.0                    # first order on land (terrain z)

    # Each shift: (N - z) / tan(incidence), away from the satellite.
    d = _en(raw[:, 2], raw[:, 3], new[:, 2], new[:, 3])
    want = (n - raw[:, 4]) / np.tan(np.radians(header["gcp_incidence_deg"]))
    np.testing.assert_allclose(np.hypot(*d), np.abs(want), rtol=0, atol=0.5)
    # IW swath: incidence ~30-46 deg, N 35-39 m -> 34-71 m (41-56 m at the AOI's ships).
    assert 30.0 < want[sea].min() and want[sea].max() < 75.0
    east = d[0][sea]
    if oracle.pass_ == "Descending":                # looks west: away = west
        assert scene.startswith("dsc") and (east < 0).all()
    else:                                           # ascending looks east
        assert oracle.pass_ == "Ascending" and (east > 0).all()


def test_annotation_is_the_measurement_sibling_not_the_rfi_asset():
    href = ("https://x.blob.core.windows.net/s1-grd/GRD/2026/6/28/IW/DV/S1D_X_71E0/"
            "measurement/iw-vv.tiff?st=1&sig=abc")
    assert fetch._s1_annotation_url(href, "vv") == (
        "https://x.blob.core.windows.net/s1-grd/GRD/2026/6/28/IW/DV/S1D_X_71E0/"
        "annotation/iw-vv.xml?st=1&sig=abc")
    with pytest.raises(fetch.FetchError, match="cannot locate"):
        fetch._s1_annotation_url(href.replace("iw-vv.tiff", "iw-vh.tiff"), "vv")


def _swap_first_two(grid) -> None:
    second = grid[1]
    grid.remove(second)
    grid.insert(0, second)


def _edit_grid(xml: bytes, fn) -> bytes:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    fn(root, root.find("geolocationGrid/geolocationGridPointList"))
    return ET.tostring(root)


@pytest.mark.parametrize("edit, match", [
    (lambda r, g: setattr(r, "tag", "rfi"), "not a product annotation"),
    (lambda r, g: setattr(r.find("adsHeader/polarisation"), "text", "VH"), "polarisation 'vh'"),
    (lambda r, g: g.remove(g[-1]), "209 points"),
    (lambda r, g: _swap_first_two(g), "is not TIFF GCP"),                      # reordered
    (lambda r, g: setattr(g[7].find("latitude"), "text", "57.9"), "is not TIFF GCP"),
    (lambda r, g: setattr(g[3].find("incidenceAngle"), "text", "0.5"), "outside"),
])
def test_incidence_refuses_an_annotation_that_is_not_the_tiffs_grid(edit, match):
    sid = SCENES["dsc0628"]
    xml = _edit_grid(_oracle(sid).xml, edit)
    with pytest.raises(fetch.FetchError, match=match):
        fetch._s1_gcp_incidence(xml, _header(sid), "vv")


def test_chain_1_header_without_incidence_is_rebuilt_not_reused(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from imint.data import mpc_s1

    header = {k: v for k, v in _header(SCENES["dsc0628"]).items() if k != "gcp_incidence_deg"}
    (tmp_path / "vv_header.json").write_text(json.dumps(header))

    def no_network(item):
        raise RuntimeError("rebuild needs the network")

    monkeypatch.setattr(mpc_s1, "sign_item", no_network)
    with pytest.raises(RuntimeError, match="rebuild"):
        fetch._s1_scene_header(SimpleNamespace(id=header["scene_id"]), "vv", tmp_path, None)
    with pytest.raises(fetch.FetchError, match="no incidence angle"):
        fetch._s1_sea_surface_gcps(header)
    full = _header(SCENES["dsc0628"])                       # with the angles: reused
    (tmp_path / "vv_header.json").write_text(json.dumps(full))
    assert fetch._s1_scene_header(SimpleNamespace(id=full["scene_id"]), "vv", tmp_path,
                                  None)[0] == full


# ── 2. anchored ships: frozen radar positions vs AIS hull centres ──────────

@functools.cache
def _ships() -> dict:
    return json.loads((FIX / "anchored_ships.json").read_text())


def _geocode(ships: list[dict], gcps_of) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radar (row, col) -> lon/lat through GDAL's TPS over ``gcps_of(header)``.

    Returns (lon, lat, unit vectors) where the units are, per ship, toward the
    satellite (one pixel to near range) and along track (one line on), (2, 2, n).
    """
    lon, lat, units = [], [], []
    for s in ships:
        tps = _tps(s["scene_id"], gcps_of)
        r, c = s["row"], s["col"]
        (x, xs, xt), (y, ys, yt) = tps.xy([r, r, r + 1], [c, c - 1, c], offset="ul")
        u, v = _en(x, y, xs, ys), _en(x, y, xt, yt)
        lon.append(x)
        lat.append(y)
        units.append([u / np.hypot(*u), v / np.hypot(*v)])
    return np.array(lon), np.array(lat), np.moveaxis(np.array(units), 0, -1)


@functools.cache
def _tps(scene_id: str, gcps_of):
    return fetch._s1_gcp_tps(gcps_of(_header(scene_id)))


def _sea_gcps(header: dict) -> list:
    return fetch._s1_sea_surface_gcps(header)


def _raw_gcps(header: dict) -> list:
    return header["gcps"]


def _medians(layer: str, gcps_of) -> tuple[int, float, float]:
    """n, median residual toward the satellite, median along track (radar - hull, m)."""
    ships = _ships()[layer]["ships"]
    lon, lat, units = _geocode(ships, gcps_of)
    res = _en([s["hull_lon"] for s in ships], [s["hull_lat"] for s in ships], lon, lat)
    rng, trk = (np.sum(res * units[k], axis=0) for k in (0, 1))
    return len(ships), float(np.median(rng)), float(np.median(trk))


def test_s1_geocoding_vs_anchored_ships():
    fixed = {L: _medians(L, _sea_gcps) for L in _ships()}
    report = "; ".join(f"{L} n={n} range {r:+.1f} track {t:+.1f} m" for L, (n, r, t)
                       in fixed.items())
    for n, rng, trk in fixed.values():
        assert n >= MIN_SHIPS, report
        assert abs(rng) <= MAX_MEDIAN_M and abs(trk) <= MAX_MEDIAN_M, report
    # Same day, opposite look directions: the range medians agree.
    assert abs(fixed["asc0627"][1] - fixed["dsc0627"][1]) <= MAX_CROSS_PASS_M, report


def test_anchored_ships_fail_on_the_ellipsoid_gcps():
    """The check has teeth: chain 1's raw GCPs put every layer 45-57 m toward the satellite."""
    for layer in _ships():
        n, rng, _ = _medians(layer, _raw_gcps)
        assert rng > 40.0, f"{layer}: n={n} range {rng:+.1f} m"


def test_frozen_radar_positions_reproduce_chain_1():
    """The fixture's radar coordinates are the chain-1 detections (TPS round trip)."""
    for layer, rec in _ships().items():
        ships = rec["ships"]
        lon, lat, _ = _geocode(ships, _raw_gcps)
        d = np.hypot(*_en([s["chain1_lon"] for s in ships], [s["chain1_lat"] for s in ships],
                          lon, lat))
        assert d.max() < 0.01, layer


# ── 3. end to end on 2026-06-28 (slow, needs a local S1 cache) ──────────────

_E2E_CACHE = os.environ.get("IMINT_S1_E2E_CACHE")
_AI2_REFERENCE = os.environ.get("IMINT_S1_AI2_REFERENCE_DIR")


def _mutual_nearest(a: np.ndarray, b: np.ndarray, max_m: float) -> list[tuple[int, int]]:
    """Index pairs (i in a, j in b) that are each other's nearest within ``max_m``; a, b (n, 2) lon/lat."""
    d = np.array([np.hypot(*_en(a[i, 0], a[i, 1], b[:, 0], b[:, 1])) for i in range(len(a))])
    return [(i, int(np.argmin(d[i]))) for i in range(len(a))
            if d[i].min() <= max_m and int(np.argmin(d[:, int(np.argmin(d[i]))])) == i]


@pytest.mark.slow
@pytest.mark.skipif(not _E2E_CACHE, reason="set IMINT_S1_E2E_CACHE to a local S1 cache dir "
                                           "(the MPC catalogue search still needs network)")
def test_s1_pipeline_end_to_end_06_28():
    from rasterio.transform import Affine  # noqa: F401  (rasterio present)

    from imint.analyzers.ai2_vessels_s1 import (
        DEFAULT_ATTR_CKPT,
        DEFAULT_DETECT_CKPT,
        AI2S1VesselAnalyzer,
    )
    from imint.exporters.export import s1_vessel_feature_collection

    if not (DEFAULT_DETECT_CKPT.exists() and DEFAULT_ATTR_CKPT.exists()):
        pytest.skip("AI2 S1 checkpoints not in imint/fm/ai2_vessels_s1/")
    scenes = fetch.fetch_s1_vessel_scenes(SKAGEN_BBOX, "2026-06-28", cache_dir=_E2E_CACHE,
                                          pass_time_utc="05:31")
    assert scenes.provenance["chain_version"] == fetch.S1_VESSEL_CHAIN_VERSION
    assert scenes.scene_id == SCENES["dsc0628"]
    result = AI2S1VesselAnalyzer({}).run(None, bands=scenes.bands(), date="2026-06-28",
                                         coords=SKAGEN_BBOX, geo=scenes.geo)
    assert result.success, result.error
    fc = s1_vessel_feature_collection(
        result, scenes.geo, date="2026-06-28", bbox=list(SKAGEN_BBOX.values()),
        scene_id=scenes.scene_id, historical_scene_id=scenes.historical_scene_id,
        scene_time_utc=scenes.scene_time_utc, scene_start_utc=scenes.start_utc,
        scene_end_utc=scenes.end_utc, source=fetch.S1_VESSEL_SOURCE)
    # Box centre (what marine-awareness reads) vs AIS hull centre, slow anchored ships.
    centres = np.array([np.mean(f["geometry"]["coordinates"][0][:-1], axis=0)
                        for f in fc["features"]])
    ships = _ships()["dsc0628"]["ships"]
    hull = np.array([[s["hull_lon"], s["hull_lat"]] for s in ships])
    pairs = _mutual_nearest(centres, hull, 150.0)
    assert len(pairs) >= MIN_SHIPS, f"{len(pairs)} matched ships"
    tps = _tps(SCENES["dsc0628"], _sea_gcps)
    det = centres[[i for i, _ in pairs]]
    rows, cols = tps.rowcol(det[:, 0], det[:, 1], op=float)
    lon_s, lat_s = tps.xy(rows, np.asarray(cols) - 1, offset="ul")
    lon_t, lat_t = tps.xy(np.asarray(rows) + 1, cols, offset="ul")
    u, v = _en(det[:, 0], det[:, 1], lon_s, lat_s), _en(det[:, 0], det[:, 1], lon_t, lat_t)
    u, v = u / np.hypot(*u), v / np.hypot(*v)
    res = _en(hull[[j for _, j in pairs], 0], hull[[j for _, j in pairs], 1], det[:, 0], det[:, 1])
    rng, trk = float(np.median(np.sum(res * u, 0))), float(np.median(np.sum(res * v, 0)))
    assert abs(rng) <= MAX_MEDIAN_M and abs(trk) <= MAX_MEDIAN_M, (len(pairs), rng, trk)

    if not _AI2_REFERENCE:
        return
    # AI2 parity in radar coordinates: our AI2 points through the sea-surface TPS^-1,
    # AI2's own (reference.geojson) through the TPS^-1 of its 40-GCP crop.
    import rasterio
    from rasterio.control import GroundControlPoint
    from rasterio.transform import GCPTransformer

    ref_dir = Path(_AI2_REFERENCE)
    ref = json.loads((ref_dir / "reference.geojson").read_text())["features"]
    ref_ll = np.array([[f["properties"]["longitude"], f["properties"]["latitude"]] for f in ref])
    with rasterio.open(ref_dir / "inputs" / "target_vv.tif") as src:
        sub, _ = src.gcps
    off = json.loads((ref_dir / "inputs" / "crop_meta.json").read_text())["target"]["vv"]["window"]
    t40 = GCPTransformer([GroundControlPoint(row=g.row + off["row_off"], col=g.col + off["col_off"],
                                             x=g.x, y=g.y, z=g.z, id=g.id) for g in sub], tps=True)
    ours_ll = np.array([[r["lon"], r["lat"]] for r in result.outputs["regions"]])
    r1, c1 = tps.rowcol(ours_ll[:, 0], ours_ll[:, 1], op=float)
    r0, c0 = t40.rowcol(ref_ll[:, 0], ref_ll[:, 1], op=float)
    ours_rc, ref_rc = np.c_[r1, c1], np.c_[r0, c0]
    d_px = np.hypot(*(ours_rc[:, None, :] - ref_rc[None, :, :]).transpose(2, 0, 1))
    pairs = [(i, int(np.argmin(d_px[i]))) for i in range(len(ours_rc))
             if d_px[i].min() <= 5 and int(np.argmin(d_px[:, int(np.argmin(d_px[i]))])) == i]
    assert len(pairs) >= 50, f"{len(pairs)} radar-coordinate pairs with AI2"
    i, j = np.array(pairs).T
    assert np.median(d_px[i, j]) <= 1.0
    # Mean vector in metres, both radar positions through the same TPS.
    xa, ya = tps.xy(ours_rc[i, 0], ours_rc[i, 1], offset="ul")
    xb, yb = tps.xy(ref_rc[j, 0], ref_rc[j, 1], offset="ul")
    assert np.hypot(*_en(xb, yb, xa, ya).mean(axis=1)) <= 3.0
