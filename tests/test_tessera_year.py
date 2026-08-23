"""TESSERA year-0 derivation — the label year, never the autumn frame.

`dates` holds four frames: slot 0 = autumn of year-1, slots 1-3 = growing
season of year-0. Taking dates[0] yielded year-1, so every LULC tile (no
`lpis_year`) got an embedding describing the PREVIOUS season. TESSERA
embeddings are annual, so for a rotating crop that is a different crop —
which silently corrupts crop classes for the Tessera model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from enrich_tiles_tessera import (  # noqa: E402
    TESSERA_YEARS,
    _clamp_year_for_tessera,
    _infer_tile_year,
)


def _tile(dates, **extra):
    return {"dates": dates, **extra}


def test_dates_fallback_returns_growing_season_year_not_autumn():
    """The regression: autumn 2021 + growing 2022 must give 2022."""
    d = _tile(["2021-09-14", "2022-05-16", "2022-06-25", "2022-08-19"])
    assert _infer_tile_year(d) == 2022


def test_missing_autumn_slot_still_returns_year_zero():
    """~21% of holdout tiles lack slot 0; year-0 must be unchanged."""
    d = _tile(["", "2022-05-16", "2022-06-25", "2022-08-19"])
    assert _infer_tile_year(d) == 2022


def test_explicit_year_key_wins():
    d = _tile(["2021-09-14", "2022-06-25"], year=2022)
    assert _infer_tile_year(d) == 2022


def test_lpis_year_wins_over_dates():
    """Crop tiles carry the survey year; it must outrank the autumn frame."""
    d = _tile(["2021-09-14", "2022-06-25"], lpis_year=2022)
    assert _infer_tile_year(d) == 2022


def test_modal_year_wins_over_an_outlier_date():
    """Modal (not max): one stray later date must not hijack the year."""
    d = _tile(["2021-09-14", "2022-05-16", "2022-06-25", "2023-01-03"])
    assert _infer_tile_year(d) == 2022


def test_tie_breaks_to_most_recent():
    """2v2 tie resolves to the newer year — matches build_labels.py."""
    d = _tile(["2021-09-14", "2021-10-02", "2022-06-25", "2022-08-19"])
    assert _infer_tile_year(d) == 2022


def test_no_dates_returns_none():
    assert _infer_tile_year(_tile([])) is None
    assert _infer_tile_year(_tile(["", "  "])) is None


def test_malformed_dates_are_skipped_not_fatal():
    d = _tile(["not-a-date", "2022-06-25"])
    assert _infer_tile_year(d) == 2022


def test_clamp_is_identity_inside_covered_range():
    for y in TESSERA_YEARS:
        assert _clamp_year_for_tessera(y, TESSERA_YEARS) == y


def test_clamp_picks_nearest_outside_range():
    assert _clamp_year_for_tessera(2017, TESSERA_YEARS) == 2018
    assert _clamp_year_for_tessera(2026, TESSERA_YEARS) == 2024


def test_s1_year_matches_tessera_year_rule():
    """S1 composites over the growing season *of the returned year*, so it must
    resolve year-0 by the identical rule — not the autumn frame."""
    from enrich_tiles_s1 import _tile_year

    d = _tile(["2021-09-14", "2022-05-16", "2022-06-25", "2022-08-19"])
    assert _tile_year(d) == 2022 == _infer_tile_year(d)


def test_s1_ignores_clamped_tessera_year():
    """tessera_year is CLAMPED into TESSERA's covered range, so it is not a
    label year; S1 must not inherit it (or any TESSERA error) over the dates."""
    from enrich_tiles_s1 import _tile_year

    d = _tile(["2016-09-14", "2017-06-25", "2017-07-02"], tessera_year=2018)
    assert _tile_year(d) == 2017


def test_s1_prefers_explicit_keys():
    from enrich_tiles_s1 import _tile_year

    assert _tile_year(_tile(["2021-09-14", "2022-06-25"], year=2022)) == 2022
    assert _tile_year(_tile(["2021-09-14"], lpis_year=2022)) == 2022


def test_skip_comparison_is_clamp_stable():
    """A 2017 tile stores tessera_year=2018; the skip check must compare
    against the clamped value or it re-fetches that tile on every run."""
    want = _infer_tile_year(_tile(["2016-09-14", "2017-06-25"]))
    assert want == 2017
    assert _clamp_year_for_tessera(want, TESSERA_YEARS) == 2018
