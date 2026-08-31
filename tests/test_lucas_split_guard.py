"""A split file that cannot tag held-out points must fail loudly.

``lucas_tile_coverage.py`` guarded its split load with ``Path(...).exists()``,
so a missing file was not an error: ``test_tiles`` stayed empty, every
co-located point was labelled ``train``, and the index landed on disk looking
perfectly plausible. The held-out scorers then matched zero rows.

The case is not hypothetical. ``data/`` is gitignored, so
``data/distill/holdout_split.json`` exists only on the operator's laptop — a
pod that clones the repo has no split file at all, which is exactly the branch
the old check swallowed. The second half of the same failure is a split that
*does* exist but names a different tile namespace (the ``holdoutval_*`` tiles
scored against ``distill_split.json``): it matches nothing, and every point
again falls through to ``train``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lucas_tile_coverage import load_test_tiles  # noqa: E402


def _write(tmp_path: Path, payload: dict) -> str:
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_missing_split_is_fatal(tmp_path):
    """The gitignored-data case: the path simply is not there."""
    with pytest.raises(SystemExit) as exc:
        load_test_tiles(str(tmp_path / "nope.json"))
    assert "not found" in str(exc.value)


def test_absent_test_tiles_key_is_fatal(tmp_path):
    with pytest.raises(SystemExit) as exc:
        load_test_tiles(_write(tmp_path, {"train_tiles": ["a"]}))
    assert "test_tiles" in str(exc.value)


def test_empty_test_tiles_is_fatal(tmp_path):
    """An empty list disables the leakage guard just as thoroughly."""
    with pytest.raises(SystemExit) as exc:
        load_test_tiles(_write(tmp_path, {"test_tiles": []}))
    assert "no test_tiles" in str(exc.value)


def test_explicit_optout_returns_empty(tmp_path):
    """--split '' is the sanctioned way to run without the guard."""
    assert load_test_tiles("") == set()
    assert load_test_tiles(None) == set()


def test_valid_split_loads_as_strings(tmp_path):
    """Names are coerced to str — tile names arrive as ints in some manifests."""
    path = _write(tmp_path, {"train_tiles": [], "test_tiles": ["holdoutval_1_2_2022", 44123932]})
    assert load_test_tiles(path) == {"holdoutval_1_2_2022", "44123932"}
