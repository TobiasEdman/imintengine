"""Unit tests for the per-slot download in
``imint.training.openeo_tile_graph.fetch_tile_all_slots_des_openeo``.

The des tile-graph now downloads ONE openEO job per slot (reusing one
connection) instead of a single merged N-slot cube — a merged download
hangs/times-out server-side on DES for large band counts (the 60-band → [408]
the through-entry fetch hit). Per-slot lets each failure mode skip its slot in
isolation (partial success); the caller's QC keeps tiles with >=3/4 frames.

No network: the connection, slot-cube builder, download, gtiff unpack, raster
read, reflectance, and grid-snap are monkeypatched. Each fake download encodes
its slot index into the bytes (``b"SLOT<n>"``) so the fake ``rasterio.open`` can
return a per-slot array — letting one test target one slot's failure mode.
"""
from __future__ import annotations

import numpy as np

from imint.training import openeo_tile_graph as otg

_BBOX = {"west": 400000, "south": 6400000, "east": 400040, "north": 6400040}
_WINDOWS = [(0, "2022-05-01", "2022-05-11"),
            (1, "2022-06-01", "2022-06-11"),
            (2, "2022-07-01", "2022-07-11")]


class _FakeCube:
    def __init__(self, slot_idx: int, *, fail: bool = False, empty: bool = False):
        self.slot_idx, self.fail, self.empty = slot_idx, fail, empty

    def download(self, format: str = "gtiff"):
        if self.fail:
            raise RuntimeError(f"[408] RequestTimeout: slot {self.slot_idx}")
        return b"" if self.empty else f"SLOT{self.slot_idx}".encode()


class _FakeSrc:
    def __init__(self, arr): self._arr, self.transform, self.crs = arr, "T", "CRS"
    def read(self): return self._arr
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _patch(monkeypatch, *, fail=frozenset(), empty=frozenset(),
           wrong_bands=frozenset(), zeros=frozenset()):
    """Wire the no-network seams; return (built-slots list, n_bands)."""
    _, _, _, des_bands = otg._bands_groups_for_source("des")
    n_bands = len(des_bands)
    built: list[int] = []

    def _fake_build(conn, bbox, slot_idx, ds, de, **k):
        built.append(slot_idx)
        return _FakeCube(slot_idx, fail=slot_idx in fail, empty=slot_idx in empty)

    def _fake_open(arg, *a, **k):
        slot = int(arg.getvalue().decode().removeprefix("SLOT"))
        if slot in wrong_bands:
            arr = np.ones((n_bands - 1, 4, 4), np.float32)   # band-count mismatch
        elif slot in zeros:
            arr = np.zeros((n_bands, 4, 4), np.float32)       # all-zero → skipped
        else:
            arr = np.ones((n_bands, 4, 4), np.float32)
        return _FakeSrc(arr)

    monkeypatch.setattr("imint.fetch._connect", lambda: object())
    monkeypatch.setattr(otg, "_build_slot_cube", _fake_build)
    monkeypatch.setattr("imint.fetch._unpack_openeo_gtiff_bytes", lambda b: b)
    monkeypatch.setattr("rasterio.open", _fake_open)
    monkeypatch.setattr(otg, "dn_to_reflectance", lambda arr, source: arr)
    monkeypatch.setattr("imint.fetch._snap_to_target_grid",
                        lambda arr, t, c, b, pixel_size: (arr, None))
    return built, n_bands


def test_all_slots_succeed(monkeypatch):
    built, n_bands = _patch(monkeypatch)
    result = otg.fetch_tile_all_slots_des_openeo(_BBOX, _WINDOWS)
    assert built == [0, 1, 2]                       # one openEO job per slot
    assert set(result) == {0, 1, 2}
    assert result[0][0].shape == (n_bands, 4, 4)
    assert result[0][1].startswith("2022-05")       # window-midpoint date str


def test_download_failure_slot_skipped(monkeypatch):
    # Slot 1's download raises a [408]; slots 0 and 2 still come back.
    built, _ = _patch(monkeypatch, fail={1})
    result = otg.fetch_tile_all_slots_des_openeo(_BBOX, _WINDOWS)
    assert built == [0, 1, 2]
    assert set(result) == {0, 2}


def test_empty_bytes_slot_skipped(monkeypatch):
    _patch(monkeypatch, empty={2})
    result = otg.fetch_tile_all_slots_des_openeo(_BBOX, _WINDOWS)
    assert set(result) == {0, 1}


def test_band_mismatch_slot_skipped(monkeypatch):
    # A wrong band count is a single-slot failure → skip, NOT abort the tile.
    _patch(monkeypatch, wrong_bands={0})
    result = otg.fetch_tile_all_slots_des_openeo(_BBOX, _WINDOWS)
    assert set(result) == {1, 2}


def test_all_zero_slot_skipped(monkeypatch):
    _patch(monkeypatch, zeros={1})
    result = otg.fetch_tile_all_slots_des_openeo(_BBOX, _WINDOWS)
    assert set(result) == {0, 2}


def test_empty_slot_windows_no_connect(monkeypatch):
    def _boom():
        raise AssertionError("must not connect for empty slot_windows")
    monkeypatch.setattr("imint.fetch._connect", _boom)
    assert otg.fetch_tile_all_slots_des_openeo(_BBOX, []) == {}
