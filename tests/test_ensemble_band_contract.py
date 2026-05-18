"""tests/test_ensemble_band_contract.py — band-order contract for the LULC ensemble.

Guards the rule documented in docs/training/ensemble_band_contract.md:
the 6-band ``spectral`` tensor carries B8A (narrow NIR) in slot 3, on
every frame including frame_2016. A spectral-writing script that
hardcodes its own band list can silently drift to B08 — which is what
broke the sen2cor frame_2016 pipeline once.

Run:
    pytest tests/test_ensemble_band_contract.py -v

Deterministic, no GPU or network.
"""
from __future__ import annotations

import re
from pathlib import Path

from imint.training.tile_fetch import N_BANDS, PRITHVI_BANDS

_REPO = Path(__file__).resolve().parents[1]
_RUNNER = _REPO / "scripts" / "sen2cor_pipeline" / "run_sen2cor_per_scene.py"
_SELECTOR = _REPO / "scripts" / "sen2cor_pipeline" / "select_scenes.py"


# ── 1. Canonical band order ──────────────────────────────────────────────

def test_prithvi_bands_canonical():
    assert PRITHVI_BANDS == ["B02", "B03", "B04", "B8A", "B11", "B12"]
    assert len(PRITHVI_BANDS) == N_BANDS == 6


def test_nir_slot_is_narrow_not_broad():
    """Slot 3 must be B8A (865 nm narrow NIR), never B08 (842 nm broad)."""
    assert PRITHVI_BANDS[3] == "B8A"
    assert "B08" not in PRITHVI_BANDS


# ── 2. The sen2cor runner uses the canonical constant ────────────────────

def test_runner_imports_canonical_constant():
    src = _RUNNER.read_text()
    assert "from imint.training.tile_fetch import PRITHVI_BANDS" in src


def test_runner_has_no_local_band_literal():
    """No hardcoded 6-band frame list — that is how B08 crept in before."""
    src = _RUNNER.read_text()
    assert "_FRAME_BANDS" not in src
    # No 6-band literal containing B08 in NIR position.
    assert not re.search(r'"B02".*"B03".*"B04".*"B08"', src)


def test_runner_crops_with_canonical_bands():
    src = _RUNNER.read_text()
    assert "for b in PRITHVI_BANDS" in src
    assert "_l2a_band_path(l2a_dir, b) for b in PRITHVI_BANDS" in src


# ── 3. frame_2016 records its band order (self-healing re-fetch) ──────────

def test_runner_writes_band_metadata():
    src = _RUNNER.read_text()
    assert 'data["frame_2016_bands"]' in src


def test_selector_validates_band_metadata():
    """select_scenes treats a wrong/absent frame_2016_bands as missing."""
    src = _SELECTOR.read_text()
    assert "frame_2016_bands" in src
    assert "PRITHVI_BANDS" in src
