"""Repair of tiles whose S1 the SAR gates reject (task 0114).

``scripts/repair_s1_unstamped.py`` selects tiles that hold ``s1_vv_vh`` but
fail ``s1_usable`` (from member names + the two stamps, no array load),
backs each up byte-for-byte, and re-enriches it with ``enrich_one_tile``.
Synthetic tiles only; the enricher is monkeypatched (no network).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import scripts.enrich_tiles_s1 as enricher
import scripts.repair_s1_unstamped as repair
from imint.training.s1_enrichment import S1_ENRICH_VERSION, S1_SOURCE

_MISSING = object()

# name -> (has s1_vv_vh, s1_enrich_v, s1_source, expected reason or None)
_CASES = {
    "usable": (True, S1_ENRICH_VERSION, S1_SOURCE, None),
    "unstamped": (True, _MISSING, _MISSING, "s1_enrich_v=0"),
    "v3_rtc": (True, 3, S1_SOURCE, "s1_enrich_v=3"),
    "v4_no_source": (True, S1_ENRICH_VERSION, _MISSING, "s1_source=None"),
    "v4_retired": (True, S1_ENRICH_VERSION, "cdse-grd-sigma0",
                   "s1_source='cdse-grd-sigma0'"),
    "no_s1": (False, _MISSING, _MISSING, None),
    "no_s1_stamped": (False, 3, S1_SOURCE, None),
}
_SELECTED = sorted(n for n, c in _CASES.items() if c[3] is not None)


def _tile(path, *, s1=True, version=_MISSING, source=_MISSING):
    data = {
        "spectral": np.random.rand(6, 8, 8).astype(np.float32),
        "label": np.random.randint(0, 5, (8, 8)).astype(np.int64),
    }
    if s1:
        data["s1_vv_vh"] = np.random.rand(2, 8, 8).astype(np.float32)
    if version is not _MISSING:
        data["s1_enrich_v"] = np.int32(version)
    if source is not _MISSING:
        data["s1_source"] = np.bytes_(source)
    np.savez_compressed(path, **data)
    return str(path)


@pytest.fixture
def tile_dir(tmp_path):
    d = tmp_path / "tiles"
    d.mkdir()
    for name, (s1, version, source, _) in _CASES.items():
        _tile(d / f"{name}.npz", s1=s1, version=version, source=source)
    # An in-flight atomic-write temp is never a tile, even when it looks bad.
    # It belongs to an unselected tile, so --apply may proceed; a selected
    # tile's staging file refuses the run (real-producer tests below).
    _tile(d / "usable.npz.tmp.npz")
    return d


def _snapshot(root):
    return {
        os.path.relpath(os.path.join(dp, n), root):
            open(os.path.join(dp, n), "rb").read()
        for dp, _, names in os.walk(root) for n in names
    }


def _fake_enrich(calls, *, fix=True, fail=()):
    def fake(path, *, skip_existing, backend):
        calls.append((path, skip_existing, backend))
        name = os.path.splitext(os.path.basename(path))[0]
        if name in fail:
            return {"name": name, "status": "failed", "reason": "fetch_error: x"}
        if fix:
            data = dict(np.load(path, allow_pickle=True))
            data["s1_enrich_v"] = np.int32(S1_ENRICH_VERSION)
            data["s1_source"] = np.bytes_(S1_SOURCE)
            np.savez_compressed(path, **data)
        return {"name": name, "status": "ok", "orbit": "DESCENDING"}
    return fake


def _lines(out, prefix):
    return [json.loads(line[len(prefix):]) for line in out.splitlines()
            if line.startswith(prefix)]


def test_select_picks_s1_tiles_failing_the_rule(tile_dir):
    got = {os.path.basename(p)[:-4]: r for p, r in repair.select(str(tile_dir))}
    assert got == {n: _CASES[n][3] for n in _SELECTED}


def test_select_reads_only_member_names_and_stamps(tile_dir, monkeypatch):
    read = []
    real = np.lib.npyio.NpzFile.__getitem__

    def spy(self, key):
        read.append(key)
        return real(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", spy)
    repair.select(str(tile_dir))
    assert read and set(read) <= set(repair.STAMP_KEYS)


def test_unreadable_tile_is_reported_not_selected(tile_dir):
    (tile_dir / "broken.npz").write_bytes(b"not a zip")
    selected, unreadable = repair.scan(str(tile_dir))
    assert "broken.npz" not in {os.path.basename(p) for p, _ in selected}
    assert [os.path.basename(p) for p, _ in unreadable] == ["broken.npz"]


def test_dry_run_is_default_and_writes_nothing(tile_dir, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(enricher, "enrich_one_tile", _fake_enrich(calls))
    before = _snapshot(tile_dir.parent)
    assert repair.main(["--data-dir", str(tile_dir)]) == 0
    assert repair.main(["--data-dir", str(tile_dir), "--dry-run",
                        "--expect", str(len(_SELECTED))]) == 0
    assert _snapshot(tile_dir.parent) == before
    assert calls == []
    report = _lines(capsys.readouterr().out, "SELECTION ")[0]
    assert report["count"] == len(_SELECTED)
    assert sorted(t["name"] for t in report["tiles"]) == \
        [f"{n}.npz" for n in _SELECTED]


def test_apply_requires_backup_dir(tile_dir):
    with pytest.raises(SystemExit):
        repair.main(["--data-dir", str(tile_dir), "--apply"])


def test_expect_mismatch_refuses_before_writing(tile_dir, tmp_path,
                                                monkeypatch):
    calls = []
    monkeypatch.setattr(enricher, "enrich_one_tile", _fake_enrich(calls))
    backup = tmp_path / "backup"
    before = _snapshot(tile_dir)
    rc = repair.main(["--data-dir", str(tile_dir), "--apply",
                      "--backup-dir", str(backup), "--expect", "23"])
    assert rc == repair.EXIT_REFUSED
    assert calls == [] and not backup.exists()
    assert _snapshot(tile_dir) == before


def test_backup_refuses_an_existing_name(tile_dir, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(enricher, "enrich_one_tile", _fake_enrich(calls))
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "v3_rtc.npz").write_bytes(b"earlier backup")
    before = _snapshot(tile_dir)
    rc = repair.main(["--data-dir", str(tile_dir), "--apply",
                      "--backup-dir", str(backup)])
    assert rc == repair.EXIT_REFUSED
    assert calls == []
    assert os.listdir(backup) == ["v3_rtc.npz"]
    assert (backup / "v3_rtc.npz").read_bytes() == b"earlier backup"
    assert _snapshot(tile_dir) == before


def test_apply_backs_up_and_enriches_only_selected(tile_dir, tmp_path,
                                                   monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(enricher, "enrich_one_tile", _fake_enrich(calls))
    backup = tmp_path / "backup"
    originals = _snapshot(tile_dir)
    rc = repair.main(["--data-dir", str(tile_dir), "--apply", "--workers", "2",
                      "--backup-dir", str(backup),
                      "--expect", str(len(_SELECTED))])
    assert rc == 0
    assert sorted(os.path.basename(p)[:-4] for p, _, _ in calls) == _SELECTED
    assert {(s, b) for _, s, b in calls} == {(True, "pc-rtc")}
    # Byte-exact backups of exactly the selected tiles, taken before writing.
    assert _snapshot(backup) == {f"{n}.npz": originals[f"{n}.npz"]
                                 for n in _SELECTED}
    # Unselected tiles (and the temp) untouched; nothing deleted.
    for name in set(originals) - {f"{n}.npz" for n in _SELECTED}:
        assert (tile_dir / name).read_bytes() == originals[name]
    out = capsys.readouterr().out
    assert len(_lines(out, "  result ")) == len(_SELECTED)
    after = _lines(out, "AFTER ")[0]
    assert after["after_count"] == 0 and after["still_failing"] == []
    assert repair.select(str(tile_dir)) == []


@pytest.mark.parametrize("kwargs", [{"fail": ("v3_rtc",)}, {"fix": False}])
def test_apply_exits_nonzero_on_failure(tile_dir, tmp_path, monkeypatch,
                                        kwargs):
    calls = []
    monkeypatch.setattr(enricher, "enrich_one_tile",
                        _fake_enrich(calls, **kwargs))
    rc = repair.main(["--data-dir", str(tile_dir), "--apply",
                      "--backup-dir", str(tmp_path / "backup")])
    assert rc == repair.EXIT_FAILED
    assert len(calls) == len(_SELECTED)


def test_enrich_exception_is_a_failed_tile(tile_dir, tmp_path, monkeypatch):
    def boom(path, *, skip_existing, backend):
        raise RuntimeError("write failed")

    monkeypatch.setattr(enricher, "enrich_one_tile", boom)
    rc = repair.main(["--data-dir", str(tile_dir), "--apply",
                      "--backup-dir", str(tmp_path / "backup")])
    assert rc == repair.EXIT_FAILED


# --- Real producer: enrich_one_tile runs for real, only the RTC reads are
# stubbed. The fake enricher above writes the tile directly and so cannot
# see the producer's staging file (Codex review of PR 99).

_REAL_TILE = "tile_380000_6170000.npz"
_USABLE_TILE = "tile_380000_6170256.npz"
_RTC_SAR = np.full((2, 256, 256), 0.2, np.float32)


def _real_tile(path, **stamps):
    data = {
        "spectral": np.full((6, 256, 256), 0.3, np.float32),
        "label": np.full((256, 256), 7, np.int16),
        "lpis_year": np.int32(2023),
        "s1_vv_vh": np.full((2, 256, 256), -12, np.float32),
    }
    if "version" in stamps:
        data["s1_enrich_v"] = np.int32(stamps["version"])
    if "source" in stamps:
        data["s1_source"] = np.bytes_(stamps["source"])
    np.savez(path, **data)
    return str(path)


@pytest.fixture
def rtc_calls(monkeypatch):
    from imint.training import pc_s1_rtc as rtc

    calls = []

    def probe(*a, **k):
        calls.append("probe")
        return "ASCENDING", {0: []}

    def fetch(*a, **k):
        calls.append("fetch")
        return _RTC_SAR.copy(), ["2023-06-01"], "ASCENDING"

    monkeypatch.setattr(rtc, "_open_client", lambda: object())
    monkeypatch.setattr(rtc, "probe_orbits_with_items", probe)
    monkeypatch.setattr(rtc, "fetch_s1_season_composite", fetch)
    return calls


@pytest.fixture
def real_dir(tmp_path):
    d = tmp_path / "tiles"
    d.mkdir()
    _real_tile(d / _REAL_TILE)  # unstamped: selected
    _real_tile(d / _USABLE_TILE, version=S1_ENRICH_VERSION, source=S1_SOURCE)
    (d / "notes.txt").write_bytes(b"not a tile")
    return d


def test_staging_name_is_the_producers_rule(real_dir):
    path = str(real_dir / _REAL_TILE)
    assert enricher.tile_staging_path(path) == path + ".tmp.npz"


def test_real_enricher_refuses_an_existing_staging_file(
        real_dir, tmp_path, rtc_calls, capsys):
    staging = real_dir / f"{_REAL_TILE}.tmp.npz"
    staging.write_bytes(b"pre-existing stage that must remain untouched")
    backup = tmp_path / "backup"
    before = _snapshot(tmp_path)

    assert repair.main(["--data-dir", str(real_dir)]) == 0
    report = _lines(capsys.readouterr().out, "SELECTION ")[0]
    assert report["staging_conflicts"] == [staging.name]

    rc = repair.main(["--data-dir", str(real_dir), "--apply", "--expect", "1",
                      "--backup-dir", str(backup)])
    assert rc == repair.EXIT_REFUSED
    assert "REFUSED" in capsys.readouterr().out
    assert _snapshot(tmp_path) == before  # tiles, staging file, notes.txt
    assert not backup.exists()
    assert rtc_calls == []


def test_real_enricher_rewrites_only_the_s1(real_dir, tmp_path, rtc_calls):
    path = real_dir / _REAL_TILE
    original = path.read_bytes()
    with np.load(path) as z:
        before = {k: z[k] for k in z.files}
    others = _snapshot(real_dir)
    backup = tmp_path / "backup"

    rc = repair.main(["--data-dir", str(real_dir), "--apply", "--expect", "1",
                      "--backup-dir", str(backup)])
    assert rc == 0
    assert "fetch" in rtc_calls
    with np.load(path, allow_pickle=True) as z:
        after = {k: z[k] for k in z.files}
    assert int(after["s1_enrich_v"]) == S1_ENRICH_VERSION
    assert after["s1_source"].item().decode() == S1_SOURCE
    np.testing.assert_array_equal(after["s1_vv_vh"], _RTC_SAR)
    for key in ("spectral", "label", "lpis_year"):
        assert after[key].dtype == before[key].dtype
        assert after[key].tobytes() == before[key].tobytes(), key
    # No staging file left behind; the other files are byte-identical.
    assert sorted(os.listdir(real_dir)) == sorted(others)
    for name in set(others) - {_REAL_TILE}:
        assert (real_dir / name).read_bytes() == others[name]
    assert (backup / _REAL_TILE).read_bytes() == original
    assert repair.select(str(real_dir)) == []
