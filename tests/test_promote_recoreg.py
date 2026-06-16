"""promote_recoreg — rename-with-backup swap, min-frac gate, rollback.

The script only globs ``*.npz`` (never loads them), so the fixtures are empty
touch-files keyed by name; promotion correctness is about which names live where.
"""
from __future__ import annotations

import errno
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import promote_recoreg as pr  # noqa: E402


def _touch_tiles(d: Path, names: list[str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / f"{n}.npz").touch()


def _names(d: Path) -> set[str]:
    return {p.stem for p in d.glob("*.npz")}


def test_dry_run_changes_nothing(tmp_path, capsys):
    live = tmp_path / "unified_v2_512"
    recoreg = tmp_path / "unified_v2_512_recoreg"
    _touch_tiles(live, ["a", "b", "c"])
    _touch_tiles(recoreg, ["a", "b", "c"])

    pr._promote(str(live), str(recoreg), execute=False, min_frac=0.9)

    assert _names(live) == {"a", "b", "c"}
    assert _names(recoreg) == {"a", "b", "c"}
    assert list(tmp_path.glob("*_pre_recoreg_*")) == []
    assert "DRY-RUN" in capsys.readouterr().out


def test_execute_swaps_and_backs_up(tmp_path):
    live = tmp_path / "unified_v2_512"
    recoreg = tmp_path / "unified_v2_512_recoreg"
    _touch_tiles(live, ["a", "b", "c"])             # old data
    _touch_tiles(recoreg, ["a", "b", "c", "d"])     # new data (+orphan d)

    pr._promote(str(live), str(recoreg), execute=True, min_frac=0.9)

    assert _names(live) == {"a", "b", "c", "d"}     # live is now the recoreg set
    assert not recoreg.exists()                     # recoreg consumed by the rename
    backups = list(tmp_path.glob("unified_v2_512_pre_recoreg_*"))
    assert len(backups) == 1
    assert _names(backups[0]) == {"a", "b", "c"}    # old data preserved


def test_min_frac_gate_blocks_shrunken_recoreg(tmp_path):
    live = tmp_path / "unified_v2_512"
    recoreg = tmp_path / "unified_v2_512_recoreg"
    _touch_tiles(live, [str(i) for i in range(10)])
    _touch_tiles(recoreg, ["0", "1", "2"])          # only 30% — well under 0.9

    with pytest.raises(SystemExit, match="min-frac"):
        pr._promote(str(live), str(recoreg), execute=True, min_frac=0.9)
    # nothing moved
    assert _names(live) == {str(i) for i in range(10)}
    assert list(tmp_path.glob("*_pre_recoreg_*")) == []


def test_empty_recoreg_aborts(tmp_path):
    live = tmp_path / "unified_v2_512"
    recoreg = tmp_path / "unified_v2_512_recoreg"
    _touch_tiles(live, ["a"])
    recoreg.mkdir()
    with pytest.raises(SystemExit, match="no .npz"):
        pr._promote(str(live), str(recoreg), execute=True, min_frac=0.9)


def test_rollback_restores_backup_without_destroying_promoted(tmp_path):
    live = tmp_path / "unified_v2_512"
    recoreg = tmp_path / "unified_v2_512_recoreg"
    _touch_tiles(live, ["old1", "old2"])
    _touch_tiles(recoreg, ["new1", "new2"])
    pr._promote(str(live), str(recoreg), execute=True, min_frac=0.0)
    assert _names(live) == {"new1", "new2"}

    pr._rollback(str(live), execute=True)

    assert _names(live) == {"old1", "old2"}                 # backup restored
    # the promoted data is preserved aside, never deleted.
    aside = list(tmp_path.glob("unified_v2_512_recoreg_rolledback_*"))
    assert len(aside) == 1 and _names(aside[0]) == {"new1", "new2"}


def test_rollback_without_backup_aborts(tmp_path):
    live = tmp_path / "unified_v2_512"
    _touch_tiles(live, ["a"])
    with pytest.raises(SystemExit, match="no .*backup"):
        pr._rollback(str(live), execute=True)


def test_promote_aborts_on_cross_filesystem(tmp_path, monkeypatch):
    """EXDEV guard: live and recoreg on different filesystems → os.rename raises
    EXDEV → abort with a clear message rather than a partial/failed move."""
    live = tmp_path / "unified_v2_512"
    recoreg = tmp_path / "unified_v2_512_recoreg"
    _touch_tiles(live, ["a", "b"])
    _touch_tiles(recoreg, ["a", "b"])

    def _raise_exdev(src, dst):
        raise OSError(errno.EXDEV, "Invalid cross-device link")
    monkeypatch.setattr(pr.os, "rename", _raise_exdev)

    with pytest.raises(SystemExit, match="different filesystems"):
        pr._promote(str(live), str(recoreg), execute=True, min_frac=0.0)
