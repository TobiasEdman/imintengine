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


# --- --strip: tiles RTC cannot re-enrich lose their S1 (owner 2026-10-22).


def test_s1_keys_are_exactly_what_the_producer_owns(tmp_path, rtc_calls):
    """S1_KEYS == keys the real enrich_one_tile writes + the stale it drops."""
    path = tmp_path / _REAL_TILE
    np.savez(path, spectral=np.full((6, 256, 256), 0.3, np.float32),
             label=np.full((256, 256), 7, np.int16),
             lpis_year=np.int32(2023), has_frame_2016=np.int32(1),
             s1_temporal_mask=np.ones(4, np.uint8))
    with np.load(path) as z:
        before = set(z.files)
    r = enricher.enrich_one_tile(str(path), skip_existing=True,
                                 backend="pc-rtc")
    assert r["status"] == "ok" and r["has_2016"], r
    with np.load(path, allow_pickle=True) as z:
        after = set(z.files)
    written, dropped = after - before, before - after
    assert dropped == {"s1_temporal_mask"}
    assert written | dropped == set(repair.S1_KEYS)
    assert len(repair.S1_KEYS) == len(set(repair.S1_KEYS))


def _full_tile(path, *, compressed=True, extra=()):
    """A stale tile holding every S1 key plus non-S1 keys of mixed dtypes."""
    rng = np.random.default_rng(len(str(path)))
    data = {
        "spectral": rng.random((6, 16, 16), dtype=np.float32),
        "label": rng.integers(0, 9, (16, 16)).astype(np.int16),
        "lpis_year": np.int32(2021),
        "dates": np.array(["2021-05-01", "2021-06-11"]),
        "frame_2016": np.asfortranarray(rng.random((6, 16, 16))),
        "meta": np.array([{"src": "x"}, None], dtype=object),
        "has_frame_2016": np.int32(1),
        # every S1 key, as a v1 GCP-warp tile might hold them
        "s1_vv_vh": rng.random((2, 16, 16), dtype=np.float32),
        "s1_vv_vh_2016": rng.random((2, 16, 16), dtype=np.float32),
        "s1_dates": np.array(["2021-06-01"]),
        "s1_dates_2016": np.array(["2016-06-03"]),
        "s1_orbit": np.bytes_("DESCENDING"),
        "s1_temporal_mask": np.ones(3, np.uint8),
        "has_s1": np.int32(1),
    }
    data.update({k: np.int32(1) for k in extra})
    (np.savez_compressed if compressed else np.savez)(path, **data)
    return str(path)


_STRIP = ["tile_a.npz", "tile_b.npz"]  # stale: selected and stripped


@pytest.fixture
def strip_dir(tmp_path):
    d = tmp_path / "tiles"
    d.mkdir()
    _full_tile(d / _STRIP[0], compressed=True)
    _full_tile(d / _STRIP[1], compressed=False)
    _tile(d / "usable.npz", version=S1_ENRICH_VERSION, source=S1_SOURCE)
    _tile(d / "no_s1.npz", s1=False)
    (d / "notes.txt").write_bytes(b"not a tile")
    return d


def _members(path):
    import zipfile

    with zipfile.ZipFile(path) as zf:
        return {i.filename: (zf.read(i), i.compress_type)
                for i in zf.infolist()}


def _strip(d, backup, *extra):
    return repair.main(["--data-dir", str(d), "--strip",
                        "--backup-dir", str(backup), *extra])


def test_strip_removes_exactly_s1_keys_and_keeps_others(strip_dir, tmp_path,
                                                        capsys):
    backup = tmp_path / "backup"
    originals = _snapshot(strip_dir)
    members = {n: _members(strip_dir / n) for n in _STRIP}

    assert _strip(strip_dir, backup, "--expect", "2") == 0

    for name in _STRIP:
        before, after = members[name], _members(strip_dir / name)
        s1 = {m for m in before if m[:-4] in repair.S1_KEYS}
        assert {m[:-4] for m in s1} == set(repair.S1_KEYS) - {"s1_enrich_v",
                                                               "s1_source"}
        # Exactly the S1 members are gone; every other member keeps its
        # bytes and compression.
        assert after == {m: v for m, v in before.items() if m not in s1}
        with np.load(strip_dir / name, allow_pickle=True) as z:
            assert not set(z.files) & set(repair.S1_KEYS)
            assert z["frame_2016"].flags.f_contiguous
            assert z["meta"][0] == {"src": "x"}
        assert (backup / name).read_bytes() == originals[name]
    # Unselected files untouched, no file added or removed (no stage left).
    assert sorted(os.listdir(strip_dir)) == sorted(originals)
    for name in set(originals) - set(_STRIP):
        assert (strip_dir / name).read_bytes() == originals[name]
    assert sorted(os.listdir(backup)) == _STRIP
    assert repair.select(str(strip_dir)) == []

    out = capsys.readouterr().out
    results = _lines(out, "  result ")
    assert [r["status"] for r in results] == ["ok", "ok"]
    full = sorted(set(repair.S1_KEYS) - {"s1_enrich_v", "s1_source"})
    assert [r["removed"] for r in results] == [full, full]
    after = _lines(out, "AFTER ")[0]
    assert after["after_count"] == 0 and after["strip_failed"] == 0
    assert after["still_holding_s1"] == [] and after["backed_up"] == 2
    assert repair.main(["--data-dir", str(strip_dir), "--expect", "0"]) == 0


def test_strip_on_the_unstamped_fixture(tile_dir, tmp_path):
    """The 0114 shapes (unstamped / v3 / v4 without or retired source)."""
    originals = _snapshot(tile_dir)
    assert _strip(tile_dir, tmp_path / "backup",
                  "--expect", str(len(_SELECTED))) == 0
    assert repair.select(str(tile_dir)) == []
    assert sorted(os.listdir(tile_dir)) == sorted(originals)
    for n in _SELECTED:
        with np.load(tile_dir / f"{n}.npz") as z:
            assert sorted(z.files) == ["label", "spectral"]


def test_strip_reuses_an_identical_existing_backup(strip_dir, tmp_path,
                                                   capsys):
    import shutil

    backup = tmp_path / "backup"
    backup.mkdir()
    shutil.copy2(strip_dir / _STRIP[0], backup / _STRIP[0])
    kept = os.stat(backup / _STRIP[0])
    original = (strip_dir / _STRIP[0]).read_bytes()

    assert _strip(strip_dir, backup, "--expect", "2") == 0
    assert (backup / _STRIP[0]).read_bytes() == original
    assert os.stat(backup / _STRIP[0]).st_ino == kept.st_ino  # not rewritten
    out = capsys.readouterr().out
    assert f"backup {_STRIP[0]} = existing identical" in out
    after = _lines(out, "AFTER ")[0]
    assert (after["backed_up"], after["backup_reused"]) == (1, 1)


@pytest.mark.parametrize("kind", ["different", "directory", "symlink"])
def test_strip_refuses_a_backup_that_is_not_identical(strip_dir, tmp_path,
                                                      kind, capsys):
    backup = tmp_path / "backup"
    backup.mkdir()
    dst = backup / _STRIP[1]
    if kind == "different":
        dst.write_bytes(b"earlier backup")
    elif kind == "directory":
        dst.mkdir()
    else:  # identical bytes, but would follow the tile when it is replaced
        dst.symlink_to(strip_dir / _STRIP[1])
    before, before_backup = _snapshot(tmp_path), sorted(os.listdir(backup))

    assert _strip(strip_dir, backup) == repair.EXIT_REFUSED
    assert "REFUSED" in capsys.readouterr().out
    assert _snapshot(tmp_path) == before  # tiles and backup dir untouched
    assert sorted(os.listdir(backup)) == before_backup


def test_strip_refuses_a_staging_conflict(strip_dir, tmp_path, capsys):
    staging = strip_dir / f"{_STRIP[0]}.tmp.npz"
    assert enricher.tile_staging_path(str(strip_dir / _STRIP[0])) == \
        str(staging)
    staging.write_bytes(b"pre-existing stage that must remain untouched")
    backup = tmp_path / "backup"
    before = _snapshot(tmp_path)

    assert _strip(strip_dir, backup, "--expect", "2") == repair.EXIT_REFUSED
    assert "REFUSED" in capsys.readouterr().out
    assert _snapshot(tmp_path) == before
    assert not backup.exists()


def test_strip_tile_never_removes_a_stage_it_did_not_create(strip_dir,
                                                            tmp_path):
    """A stage that appears after the up-front check is left alone."""
    path = str(strip_dir / _STRIP[0])
    staging = strip_dir / f"{_STRIP[0]}.tmp.npz"
    staging.write_bytes(b"someone else's stage")
    original = (strip_dir / _STRIP[0]).read_bytes()
    with pytest.raises(FileExistsError):
        repair.strip_tile(path, path)
    assert staging.read_bytes() == b"someone else's stage"
    assert (strip_dir / _STRIP[0]).read_bytes() == original


def test_strip_failed_verification_leaves_the_tile(strip_dir, tmp_path,
                                                   monkeypatch, capsys):
    def bad(original, stripped):
        raise ValueError("member spectral.npy differs from the original")

    monkeypatch.setattr(repair, "verify_stripped", bad)
    originals = _snapshot(strip_dir)
    assert _strip(strip_dir, tmp_path / "backup") == repair.EXIT_FAILED
    assert _snapshot(strip_dir) == originals  # no stage left, tiles intact
    out = capsys.readouterr().out
    assert [r["status"] for r in _lines(out, "  result ")] == ["failed"] * 2
    assert len(_lines(out, "AFTER ")[0]["still_holding_s1"]) == 2


def test_strip_refuses_unknown_s1_keys(strip_dir, tmp_path, capsys):
    _full_tile(strip_dir / _STRIP[0], extra=("s1_mean",))
    backup = tmp_path / "backup"

    assert repair.main(["--data-dir", str(strip_dir)]) == 0
    report = _lines(capsys.readouterr().out, "SELECTION ")[0]
    assert report["unknown_s1_keys"] == {_STRIP[0]: ["s1_mean"]}

    before = _snapshot(tmp_path)
    assert _strip(strip_dir, backup) == repair.EXIT_REFUSED
    assert _snapshot(tmp_path) == before and not backup.exists()


def test_strip_dry_run_and_expect_write_nothing(strip_dir, tmp_path):
    backup = tmp_path / "backup"
    before = _snapshot(tmp_path)
    # Default is the dry run, even with a backup dir given.
    assert repair.main(["--data-dir", str(strip_dir),
                        "--backup-dir", str(backup)]) == 0
    with pytest.raises(SystemExit):  # modes are exclusive
        repair.main(["--data-dir", str(strip_dir), "--strip", "--dry-run",
                     "--backup-dir", str(backup)])
    with pytest.raises(SystemExit):  # --strip needs a backup dir
        repair.main(["--data-dir", str(strip_dir), "--strip"])
    assert _strip(strip_dir, backup, "--expect", "23") == repair.EXIT_REFUSED
    assert _snapshot(tmp_path) == before
    assert not backup.exists()


@pytest.mark.parametrize("damage, match", [
    ("bytes", "differs"),         # a kept member's content changed
    ("keep_s1", "members"),       # an S1 member survived
    ("drop_other", "members"),    # a non-S1 member went missing
])
def test_verify_stripped_rejects_a_wrong_strip(strip_dir, damage, match):
    import zipfile

    original = strip_dir / _STRIP[0]
    bad = strip_dir / "bad.npz"
    with zipfile.ZipFile(original) as zin, zipfile.ZipFile(bad, "w") as zout:
        for info in zin.infolist():
            key, body = info.filename[:-4], zin.read(info)
            if key in repair.S1_KEYS and not (damage == "keep_s1"
                                              and key == "has_s1"):
                continue
            if damage == "drop_other" and key == "label":
                continue
            if damage == "bytes" and key == "spectral":
                body = body[:-1] + bytes([body[-1] ^ 1])
            zout.writestr(info, body)
    with pytest.raises(ValueError, match=match):
        repair.verify_stripped(str(original), str(bad))


# --- The backup dir must not alias the data dir (Codex BLOCK, PR 100): a
# tile would be its own "identical backup" and its original bytes would
# have no backup at all.

_ALIASES = ["same", "dot", "symlink", "case", "backup_in_data",
            "backup_in_case_alias", "data_in_backup"]


def _alias(kind, tmp_path, d):
    if kind in ("case", "backup_in_case_alias"):
        alias = d.parent / d.name.upper()
        if not alias.exists():
            pytest.skip("case-sensitive file system")
        return str(alias if kind == "case" else alias / "backup")
    if kind == "symlink":
        link = tmp_path / "link"
        link.symlink_to(d, target_is_directory=True)
        return str(link)
    return {"same": str(d), "dot": os.path.join(str(d), "."),
            "backup_in_data": str(d / "backup"),
            "data_in_backup": str(tmp_path)}[kind]


def _tree(root):
    """Every path below ``root`` (dirs too) with file bytes; links as such."""
    from pathlib import Path

    out = {}
    for dp, dns, fns in os.walk(root):
        for n in dns + fns:
            p = Path(dp, n)
            out[str(p.relative_to(root))] = (
                ("link", os.readlink(p)) if p.is_symlink()
                else p.read_bytes() if p.is_file() else "dir")
    return out


@pytest.mark.parametrize("mode", ["--strip", "--apply"])
@pytest.mark.parametrize("kind", _ALIASES)
def test_backup_dir_aliasing_the_data_dir_is_refused(strip_dir, tmp_path,
                                                     monkeypatch, capsys,
                                                     mode, kind):
    calls = []
    monkeypatch.setattr(enricher, "enrich_one_tile", _fake_enrich(calls))
    backup = _alias(kind, tmp_path, strip_dir)
    before = _tree(tmp_path)

    rc = repair.main(["--data-dir", str(strip_dir), mode, "--expect", "2",
                      "--backup-dir", backup])
    assert rc == repair.EXIT_REFUSED
    assert "same directory or nested" in capsys.readouterr().out
    assert _tree(tmp_path) == before  # every byte, no dir created
    assert calls == []


@pytest.mark.parametrize("mode", ["--strip", "--apply"])
def test_backup_that_is_the_tile_itself_is_refused(strip_dir, tmp_path,
                                                   monkeypatch, capsys, mode):
    """A hard link in a separate dir is the same file as the tile."""
    calls = []
    monkeypatch.setattr(enricher, "enrich_one_tile", _fake_enrich(calls))
    backup = tmp_path / "backup"
    backup.mkdir()
    os.link(strip_dir / _STRIP[0], backup / _STRIP[0])
    before = _tree(tmp_path)

    rc = repair.main(["--data-dir", str(strip_dir), mode,
                      "--backup-dir", str(backup)])
    assert rc == repair.EXIT_REFUSED
    assert f"the tile itself: ['{_STRIP[0]}']" in capsys.readouterr().out
    assert _tree(tmp_path) == before
    assert calls == []


def test_separate_backup_dir_via_symlink_still_reuses(strip_dir, tmp_path,
                                                      capsys):
    """Not over-refused: a symlink to a separate dir is a separate dir."""
    import shutil

    real = tmp_path / "backup_real"
    real.mkdir()
    shutil.copy2(strip_dir / _STRIP[0], real / _STRIP[0])
    original = (strip_dir / _STRIP[0]).read_bytes()
    link = tmp_path / "backup_link"
    link.symlink_to(real, target_is_directory=True)

    assert _strip(strip_dir, link, "--expect", "2") == 0
    after = _lines(capsys.readouterr().out, "AFTER ")[0]
    assert (after["backed_up"], after["backup_reused"]) == (1, 1)
    assert (real / _STRIP[0]).read_bytes() == original
    assert sorted(os.listdir(real)) == _STRIP
    assert repair.select(str(strip_dir)) == []
