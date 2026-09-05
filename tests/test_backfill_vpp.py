"""Tests for scripts/backfill_vpp.py — VPP backfill into _recoreg tiles.

Builds a synthetic ``_recoreg`` directory (a tile with empty VPP, a tile
with present VPP, a no-year tile) and mocks the ``cdse_vpp.fetch_vpp_tiles``
fetch — keyed on the ``$VPP_SOURCE`` env switch the real router uses — so
every path is exercised without any network / cluster / CDSE PU spend:

  * enumeration finds ONLY the empty tile (present-VPP + no-year skipped);
  * the 5 ``vpp_*`` bands are written and read back, all other fields
    preserved byte-for-byte;
  * a second run is an idempotent no-op (tile now non-empty);
  * WEkEO-miss → CDSE fallback fills from the metered path;
  * a forced double-miss (WEkEO + CDSE) writes NOTHING and records the tile
    in the ``vpp_known_empty.json`` sidecar — no zero-fill.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

import scripts.backfill_vpp as bf
import scripts.strip_unstamped_vpp as strip_vpp

H = W = 8
_RAW = ("sosd", "eosd", "length", "maxv", "minv")


# ── synthetic fetch results (keys are the bare metric names) ──────────────

def _covered(value: float = 18122.0) -> dict[str, np.ndarray]:
    """A "covers the tile" VPP result — SOSD/EOSD well above the >5% floor."""
    return {
        "sosd": np.full((H, W), value, np.float32),
        "eosd": np.full((H, W), value + 155, np.float32),
        "length": np.full((H, W), 155.0, np.float32),
        "maxv": np.full((H, W), 0.8, np.float32),
        "minv": np.full((H, W), 0.05, np.float32),
    }


def _empty_result() -> dict[str, np.ndarray]:
    """An all-zero VPP result — what WEkEO returns for a coverage gap."""
    return {k: np.zeros((H, W), np.float32) for k in _RAW}


# ── synthetic tiles ───────────────────────────────────────────────────────

def _base_tile() -> dict:
    """Minimal valid _recoreg tile: spectral cube + bbox + tessera_year."""
    return {
        "spectral": np.ones((6, H, W), np.float32),
        "bbox_3006": np.array([600000.0, 6500000.0, 600080.0, 6500080.0]),
        "year": np.int32(2021),          # canonical year-0 — infer_tile_year
        "tessera_year": np.int32(2021),  # kept: real tiles carry both
        "label": np.full((H, W), 7, np.uint8),   # a field that must survive
    }


def _write(path: Path, data: dict) -> None:
    np.savez_compressed(path, **data)


# ── atomic replacement metadata ────────────────────────────────────────

def test_atomic_savez_preserves_existing_group_and_mode(tmp_path, monkeypatch):
    """A rewritten tile stays group-readable and remains a valid NPZ."""
    destination = tmp_path / "tile.npz"
    _write(destination, {"before": np.arange(3)})
    destination.chmod(0o640)
    original = destination.stat()

    calls: list[tuple[str, int, int] | tuple[str, int]] = []
    real_fchown = os.fchown
    real_fchmod = os.fchmod
    real_replace = os.replace

    def tracked_fchown(fd: int, uid: int, gid: int) -> None:
        calls.append(("fchown", uid, gid))
        real_fchown(fd, uid, gid)

    def tracked_fchmod(fd: int, mode: int) -> None:
        calls.append(("fchmod", mode))
        real_fchmod(fd, mode)

    def tracked_replace(source, target, **kwargs) -> None:
        calls.append(("replace", 0))
        real_replace(source, target, **kwargs)

    monkeypatch.setattr(bf.os, "fchown", tracked_fchown)
    monkeypatch.setattr(bf.os, "fchmod", tracked_fchmod)
    monkeypatch.setattr(bf.os, "replace", tracked_replace)

    bf._atomic_savez(str(destination), {"after": np.arange(4)})

    replaced = destination.stat()
    assert (replaced.st_uid, replaced.st_gid) == (original.st_uid, original.st_gid)
    assert stat.S_IMODE(replaced.st_mode) == 0o640
    assert calls == [
        ("fchown", original.st_uid, original.st_gid),
        ("fchmod", 0o640),
        ("replace", 0),
    ]
    with np.load(destination) as payload:
        np.testing.assert_array_equal(payload["after"], np.arange(4))


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() != 0,
    reason="numeric root:2000 ownership requires a root POSIX test runner",
)
def test_atomic_savez_preserves_root_group_2000(tmp_path):
    """Exercise the production root:2000 0640 metadata contract when possible."""
    destination = tmp_path / "tile.npz"
    _write(destination, {"before": np.arange(2)})
    try:
        os.chown(destination, 0, 2000)
    except OSError as exc:
        pytest.skip(f"filesystem cannot represent numeric gid 2000: {exc}")
    destination.chmod(0o640)

    bf._atomic_savez(str(destination), {"after": np.arange(5)})

    replaced = destination.stat()
    assert (replaced.st_uid, replaced.st_gid) == (0, 2000)
    assert stat.S_IMODE(replaced.st_mode) == 0o640
    with np.load(destination) as payload:
        np.testing.assert_array_equal(payload["after"], np.arange(5))


def test_atomic_savez_new_destination_uses_secure_mode(tmp_path):
    """The create path retains mkstemp's private 0600 default."""
    destination = tmp_path / "new-tile.npz"

    bf._atomic_savez(str(destination), {"value": np.arange(4)})

    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    with np.load(destination) as payload:
        np.testing.assert_array_equal(payload["value"], np.arange(4))


def test_atomic_savez_new_destination_never_overwrites_concurrent_create(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "new-tile.npz"
    real_link = os.link

    def create_intruder_before_link(source, target, **kwargs):
        _write(destination, {"intruder": np.arange(2)})
        real_link(source, target, **kwargs)

    monkeypatch.setattr(bf.os, "link", create_intruder_before_link)

    with pytest.raises(FileExistsError):
        bf._atomic_savez(str(destination), {"ours": np.arange(4)})

    with np.load(destination) as payload:
        np.testing.assert_array_equal(payload["intruder"], np.arange(2))
        assert "ours" not in payload.files
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("failed_operation", ["fchown", "fchmod"])
def test_atomic_savez_metadata_failure_leaves_original(
    tmp_path, monkeypatch, failed_operation,
):
    """Metadata failures happen before publish and leave no temporary file."""
    destination = tmp_path / "tile.npz"
    _write(destination, {"original": np.arange(3)})
    destination.chmod(0o640)
    original_bytes = destination.read_bytes()

    def fail_metadata(*_args) -> None:
        raise PermissionError(f"simulated {failed_operation} failure")

    def reject_replace(*_args) -> None:
        pytest.fail("os.replace must not run after a metadata failure")

    monkeypatch.setattr(bf.os, failed_operation, fail_metadata)
    monkeypatch.setattr(bf.os, "replace", reject_replace)

    with pytest.raises(PermissionError, match=failed_operation):
        bf._atomic_savez(str(destination), {"replacement": np.arange(6)})

    assert destination.read_bytes() == original_bytes
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_savez_fsyncs_payload_metadata_and_parent_directory(
    tmp_path, monkeypatch,
):
    destination = tmp_path / "tile.npz"
    _write(destination, {"original": np.arange(3)})
    destination.chmod(0o640)
    calls: list[str] = []
    real_fsync = os.fsync

    def tracked_fsync(fd: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        calls.append(kind)
        real_fsync(fd)

    monkeypatch.setattr(bf.os, "fsync", tracked_fsync)

    bf._atomic_savez(str(destination), {"replacement": np.arange(5)})

    assert calls == ["file", "file", "directory"]


def test_atomic_savez_replace_failure_keeps_original_and_cleans_temp(
    tmp_path, monkeypatch,
):
    destination = tmp_path / "tile.npz"
    _write(destination, {"original": np.arange(3)})
    original = destination.read_bytes()

    def fail_replace(*_args, **_kwargs) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(bf.os, "replace", fail_replace)

    with pytest.raises(OSError, match="rename failure"):
        bf._atomic_savez(str(destination), {"replacement": np.arange(5)})

    assert destination.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_savez_refuses_symlink_destination(tmp_path):
    target = tmp_path / "target.npz"
    destination = tmp_path / "tile.npz"
    _write(target, {"original": np.arange(3)})
    original = target.read_bytes()
    destination.symlink_to(target)

    with pytest.raises(RuntimeError, match="symlink"):
        bf._atomic_savez(str(destination), {"replacement": np.arange(6)})

    assert destination.is_symlink()
    assert target.read_bytes() == original


def test_atomic_savez_refuses_destination_swap_before_replace(
    tmp_path, monkeypatch,
):
    destination = tmp_path / "tile.npz"
    displaced = tmp_path / "displaced.npz"
    _write(destination, {"original": np.arange(3)})
    destination.chmod(0o640)
    original = destination.read_bytes()
    real_fchmod = os.fchmod

    def swap_after_metadata(fd: int, mode: int) -> None:
        real_fchmod(fd, mode)
        destination.rename(displaced)
        _write(destination, {"intruder": np.arange(2)})

    monkeypatch.setattr(bf.os, "fchmod", swap_after_metadata)

    with pytest.raises(RuntimeError, match="destination .*identity changed"):
        bf._atomic_savez(str(destination), {"replacement": np.arange(6)})

    assert displaced.read_bytes() == original
    with np.load(destination) as payload:
        np.testing.assert_array_equal(payload["intruder"], np.arange(2))
    assert list(tmp_path.glob("*.tmp")) == []


def test_backfill_refuses_lost_update_when_tile_is_replaced_during_fetch(
    tmp_path, monkeypatch,
):
    """Initial fd/hash authority, not a late re-open, controls publication."""
    data_dir = tmp_path / "unified_v2_512"
    data_dir.mkdir()
    destination = data_dir / "tile_600000_6500000.npz"
    _write(destination, _base_tile())

    def replace_during_fetch(*_args, **_kwargs):
        intruder = data_dir / "intruder.npz"
        concurrent = _base_tile()
        concurrent["concurrent_writer"] = np.int32(1)
        _write(intruder, concurrent)
        os.replace(intruder, destination)
        return _covered()

    monkeypatch.setattr(bf, "fetch_vpp_tiles", replace_during_fetch)

    result = bf.backfill_one_tile(str(destination), cache_dir=None)

    assert result["status"] == "failed"
    assert "destination changed since initial read" in result["reason"]
    with np.load(destination) as payload:
        assert int(payload["concurrent_writer"]) == 1
        assert "vpp_year" not in payload.files


def test_strip_passes_initial_identity_and_refuses_concurrent_replace(
    tmp_path, monkeypatch,
):
    data_dir = tmp_path / "unified_v2_512"
    data_dir.mkdir()
    destination = data_dir / "tile.npz"
    original = _base_tile()
    for key in _RAW:
        original[f"vpp_{key}"] = np.ones((H, W), np.float32)
    _write(destination, original)
    real_atomic_savez = strip_vpp._atomic_savez

    def replace_before_publish(path, data, *, expected):
        assert expected.sha256
        intruder = data_dir / "intruder.npz"
        _write(intruder, {"concurrent_writer": np.int32(1)})
        os.replace(intruder, destination)
        real_atomic_savez(path, data, expected=expected)

    monkeypatch.setattr(strip_vpp, "_atomic_savez", replace_before_publish)

    with pytest.raises(RuntimeError, match="changed since initial read"):
        strip_vpp.strip(str(destination))

    with np.load(destination) as payload:
        assert int(payload["concurrent_writer"]) == 1


def test_strip_revalidates_unstamped_semantics_on_captured_snapshot(tmp_path):
    data_dir = tmp_path / "unified_v2_512"
    data_dir.mkdir()
    destination = data_dir / "tile.npz"
    stamped = _base_tile()
    for key in _RAW:
        stamped[f"vpp_{key}"] = np.ones((H, W), np.float32)
    stamped["vpp_year"] = np.int32(2021)
    _write(destination, stamped)
    before = destination.read_bytes()

    assert strip_vpp.strip(str(destination)) is False
    assert destination.read_bytes() == before


@pytest.fixture
def recoreg_dir(tmp_path: Path) -> Path:
    """A _recoreg dir: one empty-VPP tile, one has-VPP tile, one no-year tile."""
    d = tmp_path / "unified_v2_512_recoreg"
    d.mkdir()

    # (1) empty-VPP: VPP channels entirely absent → must be filled.
    _write(d / "tile_600000_6500000.npz", _base_tile())

    # (2) has-VPP: all 5 channels present + non-zero → must be skipped.
    has = _base_tile()
    for k in _RAW:
        has[f"vpp_{k}"] = np.full((H, W), 123.0, np.float32)
    # The year-aware skip (backfill_one_tile) trusts present VPP only when
    # it is stamped for the right year; an unstamped tile predates the
    # year-0 fix and is deliberately re-fetched. Real filled tiles carry
    # the stamp — the fixture must too.
    has["vpp_year"] = np.int32(2021)
    _write(d / "tile_700000_6600000.npz", has)

    # (3) all-zero-VPP: channels present but identically zero → counts as empty.
    zero = _base_tile()
    for k in _RAW:
        zero[f"vpp_{k}"] = np.zeros((H, W), np.float32)
    _write(d / "tile_800000_6700000.npz", zero)

    return d


# ── emptiness / year / bbox unit checks ──────────────────────────────────

def test_vpp_is_empty_detects_absent_and_allzero():
    absent = {"spectral": np.ones((6, H, W), np.float32)}
    assert bf._vpp_is_empty(absent) is True

    allzero = {f"vpp_{k}": np.zeros((H, W), np.float32) for k in _RAW}
    assert bf._vpp_is_empty(allzero) is True

    present = {f"vpp_{k}": np.zeros((H, W), np.float32) for k in _RAW}
    present["vpp_sosd"] = np.full((H, W), 18122.0, np.float32)
    assert bf._vpp_is_empty(present) is False


def test_tile_year_precedence():
    """Canonical order (tile_fetch.infer_tile_year): year -> lpis_year ->
    modal dates. tessera_year is DELIBERATELY not consulted — it is a
    clamped value, not a label year (see infer_tile_year's docstring);
    this test once asserted the opposite against a long-gone local impl."""
    assert bf._tile_year({"year": np.int32(2021),
                          "lpis_year": np.int32(2019)}) == 2021
    assert bf._tile_year({"lpis_year": np.int32(2019)}) == 2019
    assert bf._tile_year({"tessera_year": np.int32(2021)}) is None
    assert bf._tile_year({"year": np.int32(2020)}) == 2020
    assert bf._tile_year({"dates": np.array(["2022-06-01", "2022-08-15"])}) == 2022
    assert bf._tile_year({"spectral": np.ones((6, H, W))}) is None


def test_tile_bbox_3006_roundtrip():
    """Bbox resolution goes through the ONE shared resolver (f822117);
    the private _tile_bbox_3006 helper this test once called was retired
    by that consolidation. The resolver deliberately does NOT round-trip
    raw corners: it preserves the CENTER and rebuilds the extent from the
    tile's own pixel grid — decoupled extent/pixels was the 256/512
    aux-misalignment the consolidation exists to prevent."""
    from imint.training.tile_bbox import resolve_fetch_bbox
    d = {"bbox_3006": np.array([0.0, 0.0, 100.0, 100.0]),
         "spectral": np.ones((6, H, W), np.float32)}
    bbox, size = resolve_fetch_bbox(name="tile_x", npz_data=d)
    assert size == H
    assert (bbox["west"] + bbox["east"]) / 2 == 50      # center preserved
    assert (bbox["south"] + bbox["north"]) / 2 == 50
    assert bbox["east"] - bbox["west"] == H * 10        # extent == grid @10m
    bbox2, _ = resolve_fetch_bbox(
        name="unresolvable", npz_data={"spectral": np.ones((6, H, W))})
    assert bbox2 is None


# ── enumeration: only the empty tiles get fetched ────────────────────────

def test_enumeration_targets_only_empty_tiles(recoreg_dir, monkeypatch):
    """The mocked fetch is invoked for tiles (1) + (3) only, not the has-VPP one."""
    fetched_bboxes: list[tuple] = []

    def _fake_fetch(west, south, east, north, *, size_px, year, cache_dir):
        fetched_bboxes.append((west, south, east, north))
        assert size_px == (H, W)        # size derived from the spectral cube
        assert year == 2021             # from canonical year field
        return _covered()

    monkeypatch.setattr(bf, "fetch_vpp_tiles", _fake_fetch)

    stats = bf.run(str(recoreg_dir), workers=1)

    # 2 empty tiles filled (absent + all-zero), 1 skipped (has VPP).
    assert stats["filled"] == 2
    assert stats["skipped"] == 1
    assert stats["empty"] == 0 and stats["failed"] == 0
    assert len(fetched_bboxes) == 2
    # The has-VPP tile's bbox was never fetched. Bboxes are CENTERED on
    # the tile since the shared resolve_fetch_bbox (f822117): centre
    # (700000, 6600000) ± half the 8-px/10-m extent.
    assert (699960, 6599960, 700040, 6600040) not in fetched_bboxes


# ── write + read-back, with field preservation ───────────────────────────

def test_vpp_written_and_read_back_preserving_other_fields(recoreg_dir, monkeypatch):
    monkeypatch.setattr(bf, "fetch_vpp_tiles", lambda *a, **k: _covered(19000.0))

    tile = recoreg_dir / "tile_600000_6500000.npz"
    before = dict(np.load(tile, allow_pickle=True))

    bf.run(str(recoreg_dir), workers=1)

    after = dict(np.load(tile, allow_pickle=True))
    # All 5 VPP channels present, float32, correct shape + value.
    for k in _RAW:
        arr = after[f"vpp_{k}"]
        assert arr.shape == (H, W) and arr.dtype == np.float32
    assert float(after["vpp_sosd"][0, 0]) == 19000.0
    assert bf._vpp_is_empty(after) is False

    # Every pre-existing field is preserved byte-for-byte.
    np.testing.assert_array_equal(after["spectral"], before["spectral"])
    np.testing.assert_array_equal(after["label"], before["label"])
    np.testing.assert_array_equal(after["bbox_3006"], before["bbox_3006"])
    assert int(after["tessera_year"]) == 2021


# ── idempotency: a second run does not re-fetch ──────────────────────────

def test_idempotent_second_run_skips(recoreg_dir, monkeypatch):
    calls = {"n": 0}

    def _counting_fetch(*a, **k):
        calls["n"] += 1
        return _covered()

    monkeypatch.setattr(bf, "fetch_vpp_tiles", _counting_fetch)

    bf.run(str(recoreg_dir), workers=1)
    first = calls["n"]
    assert first == 2  # the 2 empty tiles

    stats2 = bf.run(str(recoreg_dir), workers=1)
    assert calls["n"] == first          # no further fetches
    assert stats2["filled"] == 0
    assert stats2["skipped"] == 3       # all 3 now carry VPP


# ── WEkEO miss → CDSE fallback fills from the metered path ────────────────

def test_wekeo_miss_falls_through_to_cdse(recoreg_dir, monkeypatch):
    """WEkEO returns all-zero (gap); CDSE covers it. Filled, sourced cdse."""
    seen_sources: list[str] = []

    def _by_source(*a, **k):
        import os
        src = os.environ.get("VPP_SOURCE")
        seen_sources.append(src)
        if src == "wekeo":
            return _empty_result()      # coverage gap → miss
        return _covered()               # cdse covers it

    monkeypatch.setattr(bf, "fetch_vpp_tiles", _by_source)

    # Restrict to the single absent-VPP tile for a clean source assertion.
    stats = bf.run(str(recoreg_dir), workers=1, max_tiles=1)

    assert stats["filled"] == 1
    assert "wekeo" in seen_sources and "cdse" in seen_sources


def test_wekeo_runtimeerror_falls_through_to_cdse(recoreg_dir, monkeypatch):
    """No WEkEO cache (RuntimeError) is also a miss → CDSE fallback."""
    def _by_source(*a, **k):
        import os
        if os.environ.get("VPP_SOURCE") == "wekeo":
            raise RuntimeError("VPP_SOURCE=wekeo but no WEkEO cache at /data/vpp_wekeo")
        return _covered()

    monkeypatch.setattr(bf, "fetch_vpp_tiles", _by_source)
    stats = bf.run(str(recoreg_dir), workers=1, max_tiles=1)
    assert stats["filled"] == 1


# ── double-miss: NO zero-fill, recorded in the known-empty sidecar ───────

def test_double_miss_records_known_empty_and_writes_nothing(recoreg_dir, monkeypatch):
    """WEkEO + CDSE both miss → VPP left absent, tile catalogued, never faked."""
    monkeypatch.setattr(bf, "fetch_vpp_tiles", lambda *a, **k: _empty_result())

    tile = recoreg_dir / "tile_600000_6500000.npz"
    before = dict(np.load(tile, allow_pickle=True))

    stats = bf.run(str(recoreg_dir), workers=1)

    assert stats["empty"] == 2          # both empty tiles missed everywhere
    assert stats["filled"] == 0

    # The tile is UNCHANGED — no zero-filled VPP channels were written.
    after = dict(np.load(tile, allow_pickle=True))
    assert bf._vpp_is_empty(after) is True
    for k in _RAW:
        assert f"vpp_{k}" not in after  # nothing fabricated
    np.testing.assert_array_equal(after["spectral"], before["spectral"])

    # The known-empty sidecar lists the missed tiles with a reason.
    sidecar = recoreg_dir / "vpp_known_empty.json"
    assert sidecar.exists()
    mapping = json.loads(sidecar.read_text())
    assert "tile_600000_6500000" in mapping
    assert "tile_800000_6700000" in mapping
    assert mapping["tile_600000_6500000"]  # non-empty reason string


# ── dry-run: fetch + report, but write nothing ───────────────────────────

def test_dry_run_writes_nothing(recoreg_dir, monkeypatch):
    monkeypatch.setattr(bf, "fetch_vpp_tiles", lambda *a, **k: _covered())

    tile = recoreg_dir / "tile_600000_6500000.npz"
    before = dict(np.load(tile, allow_pickle=True))

    stats = bf.run(str(recoreg_dir), workers=1, dry_run=True)

    assert stats["filled"] == 2         # reported as would-fill
    after = dict(np.load(tile, allow_pickle=True))
    assert bf._vpp_is_empty(after) is True            # untouched on disk
    np.testing.assert_array_equal(after["spectral"], before["spectral"])
    assert not (recoreg_dir / "vpp_known_empty.json").exists()
