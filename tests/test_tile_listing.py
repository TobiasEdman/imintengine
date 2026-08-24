"""Tile listing must never mistake an atomic-write temp for a tile.

Atomic writes land on a sibling path that also ends in ``.npz``
(``tile.npz.tmp.npz`` from ``savez``, or ``tile.tmp.npz`` from the
``[:-4]`` spelling), so a bare ``*.npz`` listing picks them up. When
``/data`` filled on 2026-08-24 the enrichers left thousands of zero-length
temps behind; the next run counted each one as a tile, failed to open it,
and exited non-zero on work that had actually succeeded.

``enrich_tiles_s1.py`` did clean the temps, but *after* globbing them into
its work list — so the deleted paths still had to fail one by one. Ordering
is the invariant under test here, not just filtering.
"""
from __future__ import annotations

import os

from imint.training.tile_fetch import (
    clean_stale_tile_tmps,
    is_tile_tmp,
    list_tile_paths,
)


def _touch(path, size=0):
    path.write_bytes(b"\0" * size)
    return path


def _populate(tmp_path):
    """Two real tiles plus one temp in each spelling seen in the repo."""
    real = [
        _touch(tmp_path / "holdoutval_99_1404_2024.npz", 32),
        _touch(tmp_path / "holdoutval_99_1405_2024.npz", 32),
    ]
    temps = [
        _touch(tmp_path / "holdoutval_99_1406_2024.npz.tmp.npz"),  # savez spelling
        _touch(tmp_path / "holdoutval_99_1407_2024.tmp.npz"),      # [:-4] spelling
        _touch(tmp_path / "holdoutval_99_1408_2024.vpp_tmp.npz"),  # mkstemp infix
    ]
    return real, temps


def test_temps_are_not_listed_as_tiles(tmp_path):
    real, _ = _populate(tmp_path)
    assert list_tile_paths(str(tmp_path)) == sorted(str(p) for p in real)


def test_clean_removes_only_temps(tmp_path):
    real, temps = _populate(tmp_path)
    removed = clean_stale_tile_tmps(str(tmp_path))

    assert removed == sorted(str(p) for p in temps)
    assert not any(p.exists() for p in temps)
    assert all(p.exists() for p in real), "cleanup must never touch a real tile"


def test_limit_applies_after_filtering(tmp_path):
    """The s1 bug: `--limit N` sliced the unfiltered list.

    With temps sorting ahead of real tiles, a smoke run asking for 2 tiles
    got 2 temps and did no work at all.
    """
    _touch(tmp_path / "aaa_tile.tmp.npz")
    _touch(tmp_path / "aab_tile.npz.tmp.npz")
    real = [
        _touch(tmp_path / "zzz_tile_a.npz", 32),
        _touch(tmp_path / "zzz_tile_b.npz", 32),
    ]

    listed = list_tile_paths(str(tmp_path), limit=2)
    assert listed == sorted(str(p) for p in real)


def test_clean_then_list_leaves_no_phantom_work(tmp_path):
    """The exact 2026-08-24 sequence, in the corrected order."""
    real, temps = _populate(tmp_path)

    clean_stale_tile_tmps(str(tmp_path))
    tiles = list_tile_paths(str(tmp_path))

    assert len(tiles) == len(real)
    assert all(os.path.exists(t) for t in tiles), (
        "every listed path must be openable — a stale entry here is exactly "
        "what inflated `failed` and produced the false-negative pod status"
    )


def test_clean_is_idempotent_and_survives_a_lost_race(tmp_path):
    _populate(tmp_path)
    first = clean_stale_tile_tmps(str(tmp_path))
    second = clean_stale_tile_tmps(str(tmp_path))

    assert first and second == [], "second pass has nothing left to remove"


def test_missing_dir_is_empty_not_fatal(tmp_path):
    missing = str(tmp_path / "nope")
    assert list_tile_paths(missing) == []
    assert clean_stale_tile_tmps(missing) == []


def test_subdirectories_are_not_listed(tmp_path):
    (tmp_path / "nested.npz").mkdir()
    _touch(tmp_path / "real.npz", 32)
    assert list_tile_paths(str(tmp_path)) == [str(tmp_path / "real.npz")]


def test_is_tile_tmp_predicate():
    assert is_tile_tmp("t.npz.tmp.npz")
    assert is_tile_tmp("t.tmp.npz")
    assert is_tile_tmp("t.vpp_tmp.npz")
    assert is_tile_tmp("/data/unified_v2_512/t.tmp.npz")
    assert not is_tile_tmp("t.npz")
    # ``FOO.npz.tmp`` does not match a ``*.npz`` listing, so it was never
    # part of this bug — but it must not be mistaken for a tile either.
    assert not is_tile_tmp("t.npz.tmp")
