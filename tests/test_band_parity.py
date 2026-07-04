"""Band-parity contract for the openEO tile-graph backends.

Every backend must request the FULL canonical 12-band set (``ALL_BANDS``)
per slot. The cdse-openeo path used to request only the 6 Prithvi bands
(and never loaded the 60 m group): downstream, the multi-slot caller's
``arr.shape != (len(ALL_BANDS), halo, halo)`` check then dropped every
slot — the orphan-512 campaign's "0 tiles via cdse-openeo on ~6 pod-hours"
bug (checkpoint 2026-07-03). This contract test pins the requested band
set at the call-site, no network needed: ``_build_slot_cube`` is patched
with a recorder that captures its kwargs and aborts the fetch.
"""
from __future__ import annotations

import numpy as np
import pytest

import imint.fetch
from imint.training import openeo_tile_graph as otg

_BBOX = {"west": 500000.0, "south": 6500000.0,
         "east": 505120.0, "north": 6505120.0}
_WINDOW = [(0, "2022-06-01", "2022-06-02")]


class _Stop(Exception):
    """Abort the fetch right after band selection is observable."""


@pytest.fixture()
def recorded(monkeypatch):
    """Patch _build_slot_cube to record kwargs and abort; stub connections."""
    captured: dict = {}

    def _recorder(conn, bbox, slot_idx, date_start, date_end, **kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(otg, "_build_slot_cube", _recorder)
    monkeypatch.setattr(otg, "is_source_dead", lambda s: False)
    monkeypatch.setattr(imint.fetch, "_connect_cdse", lambda: object())
    monkeypatch.setattr(imint.fetch, "_connect", lambda: object())
    return captured


def test_all_bands_is_canonical_12():
    assert len(otg.ALL_BANDS) == 12
    # Index groups must partition 0..11 exactly (no overlap, no gap).
    idx = [i for group in otg.ALL_BANDS_INDEX.values() for i in group]
    assert sorted(idx) == list(range(12))
    # Prithvi slice keeps the model's 6-band order exactly.
    prithvi = [otg.ALL_BANDS[i] for i in otg.ALL_BANDS_INDEX["prithvi"]]
    assert prithvi == list(otg.PRITHVI_BANDS) == \
        ["B02", "B03", "B04", "B8A", "B11", "B12"]


@pytest.mark.parametrize("source", ["cdse-openeo", "des"])
def test_backend_requests_all_12_bands(source, recorded):
    with pytest.raises(_Stop):
        otg.fetch_tile_all_slots(
            _BBOX, _WINDOW, source=source, cloud_max_pct=None)

    expected = tuple(otg.ALL_BANDS) if source == "cdse-openeo" \
        else tuple(b.lower() for b in otg.ALL_BANDS)
    assert tuple(recorded["output_bands"]) == expected, (
        f"{source} must request the canonical 12-band set, "
        f"got {recorded['output_bands']}")
    # The 60m atmospheric group (B01/B09) must actually be LOADED, not just
    # named in the output filter — the old cdse path skipped the 60m cube.
    assert recorded.get("bands_60m"), f"{source} did not load the 60m group"
    n10 = len(recorded["bands_10m"]); n20 = len(recorded["bands_20m"])
    n60 = len(recorded["bands_60m"])
    assert n10 + n20 + n60 == 12


def test_cdse_scl_rides_along(recorded):
    with pytest.raises(_Stop):
        otg.fetch_tile_all_slots(
            _BBOX, _WINDOW, source="cdse-openeo",
            cloud_max_pct=None, include_scl=True)
    assert recorded.get("scl_band") == "SCL"
