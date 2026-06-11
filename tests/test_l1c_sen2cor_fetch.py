"""Mocked tests for the l1c_sen2cor fetch fallback (``_fetch_via_l1c_sen2cor``).

No network, no sen2cor: era5 / stac / SAFE-download / sen2cor / window-read are
patched at their SOURCE modules (the handler imports them lazily, so patching the
fs module namespace would not take). Proves: the ERA5-over-tile gate fires before
any download, the 12→6-band split with extras matches the openEO path, a missing
``L2A_Process`` marks the source dead (graceful no-op off the sen2cor image), and
the dispatcher short-circuits a dead source.
"""
from __future__ import annotations

import numpy as np

from imint.training import fetch_spectral as fs
from imint.training.openeo_tile_graph import ALL_BANDS

_BBOX = {"west": 500000, "south": 6500000, "east": 505120, "north": 6505120}
_COORDS = {"west": 14.0, "south": 60.0, "east": 14.1, "north": 60.1}
_DATE = "2016-07-15"
_SIZE = 8


def _l2a_cube() -> np.ndarray:
    """(12, _SIZE, _SIZE) ALL_BANDS cube; band i carries the value i+1 so the
    split can be checked positionally."""
    return np.stack(
        [np.full((_SIZE, _SIZE), i + 1, np.float32) for i in range(len(ALL_BANDS))],
        axis=0,
    )


def _patch_chain(monkeypatch, *, era5_clear=True, scenes=None, l2a="ok", cube="ok"):
    monkeypatch.setattr(
        "imint.training.optimal_fetch.era5_prefilter_dates",
        lambda bbox, d0, d1: [_DATE] if era5_clear else [],
    )
    if scenes is None:
        scenes = [{"scene_id": "S2A_MSIL1C_20160715", "datetime": f"{_DATE}T10:00:00Z",
                   "cloud_pct": 12.0, "mgrs_tile": "33VWE"}]
    monkeypatch.setattr("imint.training.sen2cor_l2a.stac_l1c_scenes",
                        lambda bbox, d0, d1: scenes)
    monkeypatch.setattr("imint.fetch.fetch_l1c_safe_by_name",
                        lambda name, dest_dir=None, **k: dest_dir)

    def _run(safe, work, **k):
        if l2a == "missing_binary":
            raise FileNotFoundError("[Errno 2] No such file or directory: 'L2A_Process'")
        return None if l2a is None else work
    monkeypatch.setattr("imint.training.sen2cor_l2a.run_sen2cor", _run)
    monkeypatch.setattr("imint.training.sen2cor_l2a.read_l2a_allband",
                        lambda l2a_dir, bbox, px: None if cube is None else _l2a_cube())


def test_success_returns_6band_plus_extras(monkeypatch):
    _patch_chain(monkeypatch)
    extra: dict = {}
    out = fs._fetch_via_l1c_sen2cor(_BBOX, _COORDS, _DATE, size_px=_SIZE, collect_extra=extra)
    assert out.shape == (6, _SIZE, _SIZE)
    # prithvi indices [0,1,2,7,8,9] → band values 1,2,3,8,9,10
    assert [int(out[i, 0, 0]) for i in range(6)] == [1, 2, 3, 8, 9, 10]
    assert extra["b08"].shape == (_SIZE, _SIZE)
    assert extra["rededge"].shape == (3, _SIZE, _SIZE)
    assert extra["b01"].shape == (_SIZE, _SIZE) and extra["b09"].shape == (_SIZE, _SIZE)


def test_era5_cloudy_skips_download(monkeypatch):
    calls = {"stac": 0}
    monkeypatch.setattr(
        "imint.training.optimal_fetch.era5_prefilter_dates", lambda *a, **k: [])
    monkeypatch.setattr(
        "imint.training.sen2cor_l2a.stac_l1c_scenes",
        lambda *a, **k: calls.__setitem__("stac", calls["stac"] + 1) or [])
    out = fs._fetch_via_l1c_sen2cor(_BBOX, _COORDS, _DATE, size_px=_SIZE)
    assert out is None
    assert calls["stac"] == 0   # gated before any STAC / SAFE download


def test_no_scene_returns_none(monkeypatch):
    _patch_chain(monkeypatch, scenes=[])
    assert fs._fetch_via_l1c_sen2cor(_BBOX, _COORDS, _DATE, size_px=_SIZE) is None


def test_sen2cor_crash_returns_none(monkeypatch):
    _patch_chain(monkeypatch, l2a=None)   # run_sen2cor returns None (scene-flaky)
    assert fs._fetch_via_l1c_sen2cor(_BBOX, _COORDS, _DATE, size_px=_SIZE) is None


def test_missing_binary_marks_dead(monkeypatch):
    _patch_chain(monkeypatch, l2a="missing_binary")
    dead: dict = {}
    monkeypatch.setattr(fs, "mark_source_dead", lambda s, why="": dead.__setitem__(s, why))
    out = fs._fetch_via_l1c_sen2cor(_BBOX, _COORDS, _DATE, size_px=_SIZE)
    assert out is None
    assert "l1c_sen2cor" in dead


def test_dispatch_skips_when_dead(monkeypatch):
    monkeypatch.setattr(fs, "is_source_dead", lambda s: s == "l1c_sen2cor")
    called = {"h": 0}
    monkeypatch.setattr(fs, "_fetch_via_l1c_sen2cor",
                        lambda *a, **k: called.__setitem__("h", 1))
    out = fs.fetch_spectral(_BBOX, _COORDS, _DATE, backend="l1c_sen2cor",
                            size_px=_SIZE, cloud_threshold=0.15)
    assert out is None and called["h"] == 0
