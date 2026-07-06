"""Multi-backend claim coordination + postprocess_qc mask-aware/no-remove.

The claim dir is the work divider for DES + CDSE fetch jobs sharing one
staging dir (each tile fetched by whichever pod O_EXCL-claims it first; a
FAILED fetch releases the claim so the other backend may try). The
postprocess_qc tests pin the 2026-07-05 lesson: masked frames are contract,
not nodata, and --no-remove must delete nothing.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from fetch_unified_tiles import (  # noqa: E402
    _release_claim,
    _sweep_stale_claims,
    _try_claim,
)


def test_claim_exactly_one_winner(tmp_path):
    claim_dir = str(tmp_path / "claims")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda _: _try_claim(claim_dir, "tile_A"), range(8)))
    assert sum(results) == 1, f"expected exactly 1 winner, got {sum(results)}"


def test_scl_backend_threads_to_date_selection(monkeypatch):
    """--scl-backend must reach select_slot_dates (the date screen), so a DES
    openEO SCL outage can be bypassed by routing the screen to CDSE. The
    2026-07-05 [408] storm blocked BOTH legs because both screen dates via DES;
    this flag is the repo-sanctioned escape (scl_stack_screen backend=)."""
    import fetch_unified_tiles as fut

    captured = {}

    def _fake_select(coords, *, tile_year, vpp_windows, scl_backend="des", **kw):
        captured["scl_backend"] = scl_backend
        return {i: f"2021-0{i + 4}-01" for i in range(5)}

    monkeypatch.setattr(fut, "select_slot_dates", _fake_select)
    monkeypatch.setattr(fut, "fetch_tile_spectral", lambda *a, **k: None)
    monkeypatch.setattr(fut, "_valid_existing_tile", lambda p: False)
    monkeypatch.setattr(fut, "bbox_3006_to_wgs84",
                        lambda b: {"west": 0, "south": 0, "east": 0, "north": 0})

    from imint.training.tile_config import TileConfig
    loc = {"name": "t", "year": 2021,
           "bbox_3006": {"west": 600000, "south": 6600000,
                         "east": 605120, "north": 6605120}}
    fut.fetch_tile(loc, ["2021"], "/tmp", TileConfig(size_px=512),
                   vpp_cache={"t": [(120, 160)]}, scl_backend="cdse")
    assert captured.get("scl_backend") == "cdse", (
        "--scl-backend did not reach select_slot_dates")


def test_release_reopens_claim(tmp_path):
    claim_dir = str(tmp_path / "claims")
    assert _try_claim(claim_dir, "tile_B")
    assert not _try_claim(claim_dir, "tile_B")
    _release_claim(claim_dir, "tile_B")          # failed fetch → free the tile
    assert _try_claim(claim_dir, "tile_B"), "released claim must be claimable"
    _release_claim(claim_dir, "tile_missing")    # no-op, never raises


def test_sweep_removes_only_stale_unfetched(tmp_path):
    claim_dir = str(tmp_path / "claims")
    out_dir = str(tmp_path / "out"); os.makedirs(out_dir)
    for name in ("stale_unfetched", "stale_fetched", "fresh"):
        assert _try_claim(claim_dir, name)
    # A real npz behind the fetched claim.
    np.savez_compressed(os.path.join(out_dir, "stale_fetched.npz"),
                        spectral=np.ones((2, 4, 4), np.float32))
    old = time.time() - 7 * 3600
    for name in ("stale_unfetched", "stale_fetched"):
        os.utime(os.path.join(claim_dir, f"{name}.claim"), (old, old))

    swept = _sweep_stale_claims(claim_dir, out_dir, stale_h=6.0)
    assert swept == 1
    assert not os.path.exists(os.path.join(claim_dir, "stale_unfetched.claim"))
    assert os.path.exists(os.path.join(claim_dir, "stale_fetched.claim"))
    assert os.path.exists(os.path.join(claim_dir, "fresh.claim"))


# ── postprocess_qc: mask-aware nodata + --no-remove ──────────────────────

def _make_tile(d, name, *, masked_slot0):
    """4-frame 6-band tile; slot 0 all-zero + temporal_mask=0 when masked."""
    img = np.random.default_rng(1).uniform(
        0.05, 0.4, (24, 8, 8)).astype(np.float32)
    tmask = np.ones(4, np.uint8)
    if masked_slot0:
        img[:6] = 0.0
        tmask[0] = 0
    np.savez_compressed(os.path.join(d, f"{name}.npz"),
                        spectral=img, temporal_mask=tmask,
                        label=np.ones((8, 8), np.int8))


def _run_qc(data_dir, *extra):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "postprocess_qc.py"),
         "--data-dir", str(data_dir), *extra],
        capture_output=True, text=True)


def test_qc_masked_frame_is_not_nodata(tmp_path):
    """The 2026-07-05 regression: a by-contract-empty (masked) slot 0 must
    NOT count as nodata — the tile survives a destructive QC run."""
    _make_tile(tmp_path, "orphan_2018", masked_slot0=True)
    r = _run_qc(tmp_path)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(tmp_path / "orphan_2018.npz"), (
        "mask-aware nodata check must keep the masked-slot tile\n" + r.stdout)


def test_qc_no_remove_deletes_nothing(tmp_path):
    """--no-remove reports failures but never unlinks — even a genuinely
    nodata tile (unmasked all-zero frame) survives."""
    _make_tile(tmp_path, "bad_unmasked", masked_slot0=False)
    with np.load(tmp_path / "bad_unmasked.npz", allow_pickle=True) as z:
        data = dict(z)
    data["spectral"][:6] = 0.0           # all-zero frame, mask says present
    np.savez_compressed(tmp_path / "bad_unmasked.npz", **data)

    r = _run_qc(tmp_path, "--no-remove")
    assert r.returncode == 0, r.stderr
    assert os.path.exists(tmp_path / "bad_unmasked.npz"), (
        "--no-remove must never delete\n" + r.stdout)
    assert "NOTHING deleted" in r.stdout
    # Same tile WITHOUT --no-remove is removed (destructive default intact).
    r2 = _run_qc(tmp_path)
    assert not os.path.exists(tmp_path / "bad_unmasked.npz")


# ── Growing-season completeness: write gate + resume invalidation ─────────

from fetch_unified_tiles import (  # noqa: E402
    _growing_season_complete,
    _valid_existing_tile,
)


def test_growing_season_complete_semantics():
    # Slot 0 may be empty (2018 cohort awaits Phase-2) — still complete.
    assert _growing_season_complete([0, 1, 1, 1])
    assert _growing_season_complete([1, 1, 1, 1])
    # Any missing growing-season slot (1-3) = degraded.
    assert not _growing_season_complete([1, 0, 1, 1])
    assert not _growing_season_complete([1, 1, 1, 0])
    assert not _growing_season_complete([1, 0, 0, 0])
    # 5-slot masks (with 2016 background) judge only the first 4.
    assert _growing_season_complete([0, 1, 1, 1, 1])


def test_valid_existing_tile_rejects_storm_holes(tmp_path):
    """The 2026-07-05 storm wrote tiles with empty growing-season frames;
    resume must treat them as MISSING (re-fetch) — while a slot0-empty
    2018-cohort tile and a legacy tile without temporal_mask stay valid."""
    def _tile(name, tmask):
        p = str(tmp_path / f"{name}.npz")
        save = {"spectral": np.ones((24, 4, 4), np.float32)}
        if tmask is not None:
            save["temporal_mask"] = np.array(tmask, np.uint8)
        np.savez_compressed(p, **save)
        return p

    assert _valid_existing_tile(_tile("full", [1, 1, 1, 1]))
    assert _valid_existing_tile(_tile("cohort2018", [0, 1, 1, 1]))
    assert not _valid_existing_tile(_tile("storm_hole", [1, 0, 0, 0]))
    assert not _valid_existing_tile(_tile("one_missing", [1, 1, 0, 1]))
    assert _valid_existing_tile(_tile("legacy_no_mask", None))
