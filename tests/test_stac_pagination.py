"""Regression test for STAC pagination in _stac_available_dates.

A single page (limit=200) truncates wide windows — the catalogue orders
chronologically, so the dropped tail is the LATEST dates, silently
starving the late-summer slot. The function must follow links[rel=next].
"""
from __future__ import annotations

import json
from unittest import mock

from imint.fetch import _STAC_MAX_PAGES, _stac_available_dates

COORDS = {"west": 16.0, "south": 58.0, "east": 16.1, "north": 58.1}


class _FakeResp:
    def __init__(self, payload: dict):
        self._b = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


def _feat(date: str, cc: float) -> dict:
    return {"properties": {"datetime": f"{date}T10:00:00Z", "eo:cloud_cover": cc}}


def test_follows_next_link_across_pages():
    page1 = {
        "features": [_feat("2018-07-22", 10.0), _feat("2018-07-24", 20.0)],
        "links": [{"rel": "next", "href": "https://stac/next?token=p2"}],
    }
    page2 = {
        "features": [_feat("2018-08-28", 30.0), _feat("2018-08-31", 40.0)],
        "links": [],  # last page
    }
    with mock.patch(
        "urllib.request.urlopen",
        side_effect=[_FakeResp(page1), _FakeResp(page2)],
    ):
        out = _stac_available_dates(COORDS, "2018-04-20", "2018-09-01")
    dates = {d for d, _ in out}
    # The late-August dates (page 2) must be present — the whole point.
    assert dates == {"2018-07-22", "2018-07-24", "2018-08-28", "2018-08-31"}


def test_stops_at_max_pages_when_next_never_ends():
    # A catalogue that always advertises a next link must not loop forever.
    forever = {
        "features": [_feat("2018-07-22", 10.0)],
        "links": [{"rel": "next", "href": "https://stac/next"}],
    }
    with mock.patch(
        "urllib.request.urlopen",
        side_effect=[_FakeResp(forever) for _ in range(_STAC_MAX_PAGES + 5)],
    ) as m:
        _stac_available_dates(COORDS, "2018-04-20", "2018-09-01")
    assert m.call_count == _STAC_MAX_PAGES


def test_single_page_no_next_link():
    page = {
        "features": [_feat("2018-07-22", 10.0)],
        "links": [{"rel": "self", "href": "https://stac/self"}],
    }
    with mock.patch(
        "urllib.request.urlopen",
        side_effect=[_FakeResp(page)],
    ) as m:
        out = _stac_available_dates(COORDS, "2018-07-01", "2018-08-01")
    assert m.call_count == 1
    assert [d for d, _ in out] == ["2018-07-22"]


def test_cloud_filter_still_applies_across_pages():
    page1 = {
        "features": [_feat("2018-07-22", 10.0)],
        "links": [{"rel": "next", "href": "https://stac/next"}],
    }
    page2 = {
        "features": [_feat("2018-08-28", 95.0)],  # above threshold
        "links": [],
    }
    with mock.patch(
        "urllib.request.urlopen",
        side_effect=[_FakeResp(page1), _FakeResp(page2)],
    ):
        out = _stac_available_dates(
            COORDS, "2018-04-20", "2018-09-01", scene_cloud_max=80.0,
        )
    assert [d for d, _ in out] == ["2018-07-22"]
