"""drop_skg_open_raw deletes only verified grunddata/markfukt raw zips and keeps every derived raster."""
from __future__ import annotations

import json
import os
import signal
import time

import pytest

from scripts import drop_skg_open_raw as dr
from scripts import stage_skg_open as st


def _mirror(tmp_path, monkeypatch):
    monkeypatch.setenv("IMINT_GIT_SHA", "abc")
    d = tmp_path / "m"
    for sub in ("raw", "grunddata/omdrev2", "markfukt", "tradhojd"):
        (d / sub).mkdir(parents=True)
    files, derived = {}, {}

    def add(name, product, der, body=b"zipzip"):
        (d / "raw" / name).write_bytes(body)
        (d / der).write_bytes(b"cog-" + name.encode())
        files[name] = {"product": product, "sha256": "a" * 64, "bytes": len(body),
                       "url": f"https://x/{name}", "downloaded_at": "2026-10-12T10:00:00Z"}
        derived[der] = {"from": name, "sha256": st.sha256_of(d / der), "bytes": 9}

    add("sks_Volym_omdrev2_01.zip", "grunddata", "grunddata/omdrev2/volym_01.tif")
    add("sksSLUMarkfuktighet01.zip", "markfukt", "markfukt/markfukt_01.tif")
    add("sksTradhojd01.zip", "tradhojd", "tradhojd/tradhojd_10m_01.tif")
    (d / "grunddata/volym_omdrev2.vrt").write_text("<VRT/>")
    st.save_manifest(d, {"files": files, "derived": derived, "vrt": {}, "history": []})
    return d


def _m(d):
    return json.loads((d / "MANIFEST.json").read_text())


def test_deletes_only_grunddata_and_markfukt_raw_and_keeps_every_raster(tmp_path, monkeypatch, capsys):
    d = _mirror(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in d.rglob("*") if p.is_file() and "raw" not in p.parts and "MANIFEST" not in p.name}
    assert dr.main(["--dest", str(d)]) == 0
    assert not (d / "raw/sks_Volym_omdrev2_01.zip").exists() and not (d / "raw/sksSLUMarkfuktighet01.zip").exists()
    assert (d / "raw/sksTradhojd01.zip").exists()                                     # Trädhöjd untouched
    assert {p: p.read_bytes() for p in before} == before                                # rasters + VRT unchanged
    m = _m(d)
    assert m["files"]["sks_Volym_omdrev2_01.zip"]["raw_deleted_at"]
    assert not m["files"]["sksTradhojd01.zip"].get("raw_deleted_at")
    assert "freed" in capsys.readouterr().out


def test_the_staging_script_then_treats_the_items_as_done(tmp_path, monkeypatch):
    d = _mirror(tmp_path, monkeypatch)
    dr.main(["--dest", str(d)])
    item = st.Item("sks_Volym_omdrev2_01.zip", "grunddata", "grunddata/omdrev2/volym_01.tif", 2, "volym", "01")
    assert st.is_done(item, _m(d), d)


def test_a_raster_that_does_not_match_its_record_keeps_its_zip(tmp_path, monkeypatch):
    d = _mirror(tmp_path, monkeypatch)
    (d / "markfukt/markfukt_01.tif").write_bytes(b"changed")
    assert dr.main(["--dest", str(d)]) == 2
    assert (d / "raw/sksSLUMarkfuktighet01.zip").exists()
    assert not (d / "raw/sks_Volym_omdrev2_01.zip").exists()


@pytest.mark.parametrize("field", ["sha256", "url", "downloaded_at", "bytes"])
def test_an_incomplete_record_keeps_its_zip(tmp_path, monkeypatch, field):
    d = _mirror(tmp_path, monkeypatch)
    m = _m(d); m["files"]["sks_Volym_omdrev2_01.zip"].pop(field); st.save_manifest(d, m)
    assert dr.main(["--dest", str(d)]) == 2
    assert (d / "raw/sks_Volym_omdrev2_01.zip").exists()


def test_a_size_mismatch_keeps_its_zip(tmp_path, monkeypatch):
    d = _mirror(tmp_path, monkeypatch)
    (d / "raw/sks_Volym_omdrev2_01.zip").write_bytes(b"truncated-or-longer-than-recorded")
    assert dr.main(["--dest", str(d)]) == 2
    assert (d / "raw/sks_Volym_omdrev2_01.zip").exists()


def test_dry_run_deletes_nothing(tmp_path, monkeypatch):
    d = _mirror(tmp_path, monkeypatch)
    assert dr.main(["--dest", str(d), "--dry-run"]) == 0
    assert len(list((d / "raw").iterdir())) == 3


def test_a_concurrent_manifest_writer_stops_the_run(tmp_path, monkeypatch):
    d = _mirror(tmp_path, monkeypatch)
    real = dr.sha256_of
    calls = {"n": 0}

    def other_writer(path):
        calls["n"] += 1
        if calls["n"] == 1:                       # another Job rewrites the manifest mid-run
            (d / "MANIFEST.json").write_text((d / "MANIFEST.json").read_text().replace("2026-10-12", "2026-10-13"))
        return real(path)

    monkeypatch.setattr(dr, "sha256_of", other_writer)
    with pytest.raises(dr.ConcurrentWriter):
        dr.main(["--dest", str(d)])
    assert len(list((d / "raw").iterdir())) == 3


def test_a_missing_manifest_refuses(tmp_path):
    with pytest.raises(SystemExit, match="REFUSING"):
        dr.main(["--dest", str(tmp_path)])


def test_a_crash_between_delete_and_save_is_marked_on_the_next_run(tmp_path, monkeypatch):
    d = _mirror(tmp_path, monkeypatch)
    real_save = dr.save_manifest

    def die(*a, **k):
        raise KeyboardInterrupt                   # the pod is killed after the first unlink
    monkeypatch.setattr(dr, "save_manifest", die)
    with pytest.raises(KeyboardInterrupt):
        dr.main(["--dest", str(d)])
    assert not (d / "raw/sksSLUMarkfuktighet01.zip").exists()           # first in sorted order
    assert not _m(d)["files"]["sksSLUMarkfuktighet01.zip"].get("raw_deleted_at")
    assert not (d / st.LOCK_NAME).exists()                             # lock released on the way out

    monkeypatch.setattr(dr, "save_manifest", real_save)
    assert dr.main(["--dest", str(d)]) == 0
    rec = _m(d)["files"]["sksSLUMarkfuktighet01.zip"]
    assert rec["raw_deleted_at"] and rec["raw_absent_at_drop"] is True
    assert _m(d)["files"]["sks_Volym_omdrev2_01.zip"]["raw_deleted_at"]
    assert "raw_absent_at_drop" not in _m(d)["files"]["sks_Volym_omdrev2_01.zip"]


def test_a_vanished_zip_whose_raster_does_not_verify_is_reported(tmp_path, monkeypatch, capsys):
    d = _mirror(tmp_path, monkeypatch)
    (d / "raw/sksSLUMarkfuktighet01.zip").unlink()
    (d / "markfukt/markfukt_01.tif").write_bytes(b"changed")
    assert dr.main(["--dest", str(d)]) == 2
    assert not _m(d)["files"]["sksSLUMarkfuktighet01.zip"].get("raw_deleted_at")
    assert "UNMARKED sksSLUMarkfuktighet01.zip" in capsys.readouterr().out


def test_a_held_lock_refuses_before_any_delete(tmp_path, monkeypatch):
    d = _mirror(tmp_path, monkeypatch)
    (d / st.LOCK_NAME).write_text('{"owner": "stage_skg_open"}')
    with pytest.raises(st.ManifestLocked, match="stage_skg_open"):
        dr.main(["--dest", str(d)])
    assert len(list((d / "raw").iterdir())) == 3
    assert (d / st.LOCK_NAME).exists()                                 # someone else's lock is left alone


def test_the_staging_script_refuses_while_the_drop_holds_the_lock(tmp_path, monkeypatch):
    d = _mirror(tmp_path, monkeypatch)
    with st.manifest_lock(d, "drop_skg_open_raw"):
        with pytest.raises(st.ManifestLocked, match="drop_skg_open_raw"):
            st.main(["--dest", str(d), "--products", "grunddata"])
    assert not (d / st.LOCK_NAME).exists()


def test_a_real_run_records_a_history_entry(tmp_path, monkeypatch):
    d = _mirror(tmp_path, monkeypatch)
    dr.main(["--dest", str(d)])
    (h,) = _m(d)["history"]
    assert h["argv"][0] == "drop_skg_open_raw" and h["git_sha"] == "abc" and h["at"]


@pytest.mark.parametrize("field", ["sha256", "url", "downloaded_at", "bytes"])
def test_a_vanished_zip_with_an_incomplete_record_is_refused_not_marked(tmp_path, monkeypatch, field):
    d = _mirror(tmp_path, monkeypatch)
    (d / "raw/sksSLUMarkfuktighet01.zip").unlink()
    m = _m(d); m["files"]["sksSLUMarkfuktighet01.zip"].pop(field); st.save_manifest(d, m)
    assert dr.main(["--dest", str(d)]) == 2
    assert not _m(d)["files"]["sksSLUMarkfuktighet01.zip"].get("raw_deleted_at")


def test_sigterm_during_the_drop_releases_the_lock(tmp_path, monkeypatch):
    d = _mirror(tmp_path, monkeypatch)
    monkeypatch.setattr(dr, "sha256_of", lambda p: os.kill(os.getpid(), signal.SIGTERM) or time.sleep(1))
    with pytest.raises(SystemExit) as exc:
        dr.main(["--dest", str(d)])
    assert exc.value.code == 143 and not (d / st.LOCK_NAME).exists()
    assert len(list((d / "raw").iterdir())) == 3
