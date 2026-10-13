"""RA-04: real temporary-directory moves with targeted failure injection."""
import errno
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import promote_recoreg as pr


@pytest.fixture(params=[False, True], ids=["promote", "rollback"])
def swap(request, tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "_utc_stamp", lambda: "20990101T000000Z")
    live = tmp_path / "live"
    candidate = tmp_path / ("live_pre_recoreg_20000101T000000Z"
                            if request.param else "candidate")
    saved = tmp_path / ("live_recoreg_rolledback_20990101T000000Z"
                        if request.param else "live_pre_recoreg_20990101T000000Z")
    for path, payload in ((live, b"original\x00live"), (candidate, b"candidate\xff")):
        path.mkdir()
        (path / "tile.npz").write_bytes(payload)
        (path / "metadata").mkdir()
        (path / "metadata/info.json").write_bytes(payload + b"metadata")

    def run(execute=True):
        if request.param:
            return pr._rollback(str(live), execute=execute)
        return pr._promote(str(live), str(candidate), execute=execute, min_frac=.9)

    return live, candidate, saved, run


def contents(path):
    return {str(p.relative_to(path)): p.read_bytes()
            for p in path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("code", [errno.EACCES, errno.EXDEV])
def test_regression_second_rename_restores_live(swap, monkeypatch, code):
    live, candidate, saved, run = swap
    old, new = contents(live), contents(candidate)
    rename = os.rename
    calls = []
    failure = OSError(code, "injected second rename failure")

    def injected(src, dst):
        calls.append((Path(src), Path(dst)))
        if Path(src) == candidate:
            assert not live.exists()
            assert contents(saved) == old  # first rename really happened
            raise failure
        rename(src, dst)

    monkeypatch.setattr(pr.os, "rename", injected)
    with pytest.raises((OSError, SystemExit)) as error:
        run()
    assert contents(live) == old
    assert contents(candidate) == new
    assert not saved.exists()
    assert calls == [(live, saved), (candidate, live), (saved, live)]
    assert error.value is failure or error.value.__cause__ is failure


@pytest.mark.parametrize("recovery_code", [errno.EACCES, errno.EXDEV])
def test_regression_recovery_failure_preserves_original(swap, monkeypatch, recovery_code):
    live, candidate, saved, run = swap
    old, new = contents(live), contents(candidate)
    rename = os.rename
    original = OSError(errno.EACCES, "original second rename error")
    recovery = OSError(recovery_code, "recovery rename error")

    def injected(src, dst):
        if Path(src) == candidate:
            raise original
        if Path(src) == saved:
            raise recovery
        rename(src, dst)

    monkeypatch.setattr(pr.os, "rename", injected)
    with pytest.raises(RuntimeError) as error:
        run()
    assert error.value.__cause__ is original
    for value in (str(live), str(candidate), str(saved), str(original)):
        assert value in str(error.value)
    assert "restoration failed" in str(error.value)
    assert not live.exists()
    assert contents(saved) == old
    assert contents(candidate) == new


def test_regression_reappeared_live_is_not_overwritten(swap, monkeypatch):
    live, candidate, saved, run = swap
    old, new = contents(live), contents(candidate)
    rename = os.rename
    calls = []

    def injected(src, dst):
        calls.append((Path(src), Path(dst)))
        if Path(src) == candidate:
            live.mkdir()  # even an empty directory must not be replaced
            raise OSError(errno.EACCES, "second rename error")
        rename(src, dst)

    monkeypatch.setattr(pr.os, "rename", injected)
    with pytest.raises(RuntimeError, match="live path reappeared"):
        run()
    assert live.is_dir() and not list(live.iterdir())
    assert contents(saved) == old and contents(candidate) == new
    assert len(calls) == 2


@pytest.mark.parametrize("code", [errno.EACCES, errno.EXDEV])
def test_first_rename_failure_changes_nothing(swap, monkeypatch, code):
    live, candidate, saved, run = swap
    old, new = contents(live), contents(candidate)
    calls = []

    def injected(src, dst):
        calls.append((Path(src), Path(dst)))
        raise OSError(code, "first rename error")

    monkeypatch.setattr(pr.os, "rename", injected)
    with pytest.raises((OSError, SystemExit)):
        run()
    assert calls == [(live, saved)]
    assert contents(live) == old and contents(candidate) == new
    assert not saved.exists()


@pytest.mark.parametrize("kind", ["saved-dir", "saved-dangling", "live-file",
                                   "live-symlink", "candidate-symlink", "candidate-file"])
def test_regression_preflight_rejects_unsafe_paths(swap, monkeypatch, kind):
    live, candidate, saved, run = swap
    if kind == "saved-dir":
        saved.mkdir()
    elif kind == "saved-dangling":
        saved.symlink_to(saved.parent / "missing")
    else:
        path = live if kind.startswith("live") else candidate
        target = path.with_name("displaced-target")
        path.rename(target)
        if kind.endswith("symlink"):
            path.symlink_to(target, target_is_directory=True)
        else:
            path.write_bytes(b"must not replace")
    before = contents(live.parent)

    def forbidden(*args):
        pytest.fail("unsafe paths reached os.rename")

    monkeypatch.setattr(pr.os, "rename", forbidden)
    with pytest.raises(SystemExit, match="not a directory" if kind.endswith("file") else None):
        run()
    assert contents(live.parent) == before


@pytest.mark.parametrize("relationship", ["same", "candidate-inside", "live-inside", "alias-parent"])
def test_regression_promote_rejects_overlapping_paths(tmp_path, monkeypatch, relationship):
    live = tmp_path / "live"
    live.mkdir()
    (live / "tile.npz").write_bytes(b"old")
    if relationship == "same":
        candidate = live
    elif relationship == "candidate-inside":
        candidate = live / "nested"
        candidate.mkdir()
    elif relationship == "live-inside":
        candidate = tmp_path
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        candidate = alias / "live"
    (candidate / "tile.npz").write_bytes(b"candidate")

    def forbidden(*args):
        pytest.fail("overlapping paths reached os.rename")

    monkeypatch.setattr(pr.os, "rename", forbidden)
    with pytest.raises(SystemExit, match="distinct.*non-nested"):
        pr._promote(str(live), str(candidate), execute=True, min_frac=0)
    assert live.is_dir() and candidate.is_dir()


def test_regression_preflight_cross_device_before_first_rename(swap, monkeypatch):
    live, candidate, saved, run = swap
    stat = os.stat

    def other_device(path, *args, **kwargs):
        result = stat(path, *args, **kwargs)
        if Path(path) == candidate:
            fields = list(result)
            fields[2] += 1  # st_dev; all other stat fields remain real
            return os.stat_result(fields)
        return result

    def forbidden(*args):
        pytest.fail("cross-device paths reached os.rename")

    monkeypatch.setattr(pr.os, "stat", other_device)
    monkeypatch.setattr(pr.os, "rename", forbidden)
    with pytest.raises(SystemExit, match="different filesystems"):
        run()
    assert live.is_dir() and candidate.is_dir() and not saved.exists()


def test_dry_run_and_success_preserve_bytes(swap):
    live, candidate, saved, run = swap
    old, new = contents(live), contents(candidate)
    assert run(execute=False) == 0
    assert contents(live) == old and contents(candidate) == new and not saved.exists()
    assert run() == 0
    assert contents(live) == new and contents(saved) == old and not candidate.exists()


def test_absent_live_success(swap):
    live, candidate, saved, run = swap
    live.rename(live.with_name("original"))
    new = contents(candidate)
    assert run() == 0
    assert contents(live) == new and not candidate.exists() and not saved.exists()


@pytest.mark.parametrize("code", [errno.EACCES, errno.EXDEV])
def test_regression_cli_failure_is_nonzero_after_restoration(swap, code):
    import subprocess

    live, candidate, saved, run = swap
    old, new = contents(live), contents(candidate)
    script = """
import errno, os, sys
import promote_recoreg as pr
live, candidate, code, mode = sys.argv[1:]
rename = os.rename
def injected(src, dst):
    if src == candidate:
        raise OSError(int(code), 'injected CLI second rename failure')
    rename(src, dst)
pr.os.rename = injected
sys.argv = ['promote_recoreg.py', '--live', live, '--execute']
if mode == 'rollback':
    sys.argv += ['--rollback']
else:
    sys.argv += ['--recoreg', candidate]
pr.main()
"""
    mode = "rollback" if "rolledback" in saved.name else "promote"
    result = subprocess.run(
        [sys.executable, "-c", script, str(live), str(candidate), str(code), mode],
        cwd=Path(pr.__file__).parent, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "promoted:" not in result.stdout and "rolled back:" not in result.stdout
    assert contents(live) == old and contents(candidate) == new
    assert sorted(p.name for p in live.parent.iterdir()) == sorted([live.name, candidate.name])
