"""Step B: the scoped, logged l1c_sen2cor last-resort fallthrough in
``tile_fetch._fetch_single_scene``.

``fetch_spectral`` is patched at its source module (imported lazily inside the
function). Proves: des-primary fails a slot → fall through to l1c_sen2cor when
it's in ``sources``; NO fallthrough when l1c_sen2cor is absent (the old
single-backend rule still holds); a primary success never reaches the fallback.
"""
from __future__ import annotations

import numpy as np

from imint.training.tile_fetch import _fetch_single_scene
from imint.training.tile_config import TileConfig

_BBOX = {"west": 500000, "south": 6500000, "east": 505120, "north": 6505120}
_COORDS = {"west": 14.0, "south": 60.0, "east": 14.1, "north": 60.1}
_TILE = TileConfig(size_px=512)
_SCENE = np.ones((6, 8, 8), np.float32)


def _patch(monkeypatch, by_backend: dict, dead: set[str] | None = None):
    """Patch fetch_spectral to return ``by_backend[backend]``; record call order.
    ``dead`` marks backends as ``is_source_dead``."""
    dead = dead or set()
    calls: list[str] = []

    def _fake(bbox, coords, d, *, backend, size_px, cloud_threshold, collect_extra=None):
        calls.append(backend)
        return by_backend.get(backend)
    monkeypatch.setattr("imint.training.fetch_spectral.fetch_spectral", _fake)
    monkeypatch.setattr("imint.training.openeo_tile_graph.is_source_dead",
                        lambda s: s in dead)
    return calls


def test_fallthrough_to_l1c_sen2cor(monkeypatch):
    calls = _patch(monkeypatch, {"des": None, "l1c_sen2cor": _SCENE})
    scene, d = _fetch_single_scene(
        _BBOX, _COORDS, "2022-06-01", "2022-06-30", _TILE,
        sources=("des", "l1c_sen2cor"), prefetched_dates=["2022-06-15"],
    )
    assert scene is _SCENE and d == "2022-06-15"
    assert calls == ["des", "l1c_sen2cor"]     # primary first, THEN fallthrough


def test_no_fallthrough_when_not_in_sources(monkeypatch):
    calls = _patch(monkeypatch, {"des": None, "l1c_sen2cor": _SCENE})
    scene, d = _fetch_single_scene(
        _BBOX, _COORDS, "2022-06-01", "2022-06-30", _TILE,
        sources=("des",), prefetched_dates=["2022-06-15"],
    )
    assert scene is None and d == ""
    assert calls == ["des"]                     # l1c_sen2cor never tried


def test_primary_success_skips_fallthrough(monkeypatch):
    calls = _patch(monkeypatch, {"des": _SCENE, "l1c_sen2cor": _SCENE})
    scene, d = _fetch_single_scene(
        _BBOX, _COORDS, "2022-06-01", "2022-06-30", _TILE,
        sources=("des", "l1c_sen2cor"), prefetched_dates=["2022-06-15"],
    )
    assert scene is _SCENE
    assert calls == ["des"]                     # primary won; fallback untouched


def test_l1c_only_sources(monkeypatch):
    """l1c_sen2cor alone in sources → it's the only backend tried."""
    calls = _patch(monkeypatch, {"l1c_sen2cor": _SCENE})
    scene, d = _fetch_single_scene(
        _BBOX, _COORDS, "2022-06-01", "2022-06-30", _TILE,
        sources=("l1c_sen2cor",), prefetched_dates=["2022-06-15"],
    )
    assert scene is _SCENE and calls == ["l1c_sen2cor"]


def test_no_race_over_other_backends(monkeypatch):
    """A second live openEO backend (cdse) must NOT join the fallthrough — only
    l1c_sen2cor does. Pins the 'scoped, never a general race' guarantee (a
    general-race mutation would try cdse and fail this)."""
    calls = _patch(monkeypatch, {"des": None, "cdse": _SCENE, "l1c_sen2cor": _SCENE})
    scene, d = _fetch_single_scene(
        _BBOX, _COORDS, "2022-06-01", "2022-06-30", _TILE,
        sources=("des", "cdse", "l1c_sen2cor"), prefetched_dates=["2022-06-15"],
    )
    assert scene is _SCENE and d == "2022-06-15"
    assert calls == ["des", "l1c_sen2cor"]      # cdse skipped — scoped, not a race


def test_dead_l1c_skipped_clean(monkeypatch):
    """l1c_sen2cor marked dead (e.g. no L2A_Process in this pod) → skipped, never
    invoked; the slot fails cleanly. Pins the normal-pod no-op path."""
    calls = _patch(monkeypatch, {"des": None, "l1c_sen2cor": _SCENE},
                   dead={"l1c_sen2cor"})
    scene, d = _fetch_single_scene(
        _BBOX, _COORDS, "2022-06-01", "2022-06-30", _TILE,
        sources=("des", "l1c_sen2cor"), prefetched_dates=["2022-06-15"],
    )
    assert scene is None and d == ""
    assert "l1c_sen2cor" not in calls           # dead → fetch_spectral never called
