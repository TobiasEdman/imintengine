"""Tests for scripts/demote_nodata_frames.py — the decision-2 frame demote.

Proves the demote does exactly what the frame-coverage gate's inverse should:
clear a no-data frame from ``temporal_mask`` so the tile passes, but ONLY when
the tile keeps ``>= min_good_frames`` good frames — otherwise leave it untouched
(residual, a separate accept/drop call). Covers the disk round-trip (every other
field preserved bit-identical), idempotency, dry-run, and the gate agreeing
afterwards. Mirrors test_frame_coverage_qc's synthetic-tile builder.
"""
import importlib.util
from pathlib import Path

import numpy as np

from imint.training.frame_coverage_qc import check_frame_coverage

# Load the script module by path (scripts/ is not a package).
_SPEC = importlib.util.spec_from_file_location(
    "demote_nodata_frames",
    Path(__file__).resolve().parents[1] / "scripts" / "demote_nodata_frames.py")
dnf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dnf)


def _spec(fracs, px: int = 10, nb: int = 6) -> np.ndarray:
    """(nf*nb, px, px) cube where frame f has ``fracs[f]`` of its rows non-zero."""
    nf = len(fracs)
    spec = np.zeros((nf * nb, px, px), np.float32)
    for f, fr in enumerate(fracs):
        rows = int(round(fr * px))
        spec[f * nb:(f + 1) * nb, :rows, :] = 0.2
    return spec


def _tile_dict(fracs, mask=None, **extra) -> dict:
    nf = len(fracs)
    d = {"spectral": _spec(fracs), "num_bands": np.int64(6),
         "temporal_mask": np.array(mask if mask is not None else [1] * nf, np.uint8)}
    d.update(extra)
    return d


def _write(path: Path, d: dict) -> str:
    np.savez_compressed(str(path), **d)
    return str(path)  # np.savez_compressed appends .npz


# ----------------------------- plan_demote (pure) ---------------------------

def test_plan_clean_when_all_frames_good():
    p = dnf.plan_demote(_tile_dict([1.0, 1.0, 1.0, 1.0]),
                        min_valid_frac=0.90, min_good_frames=3)
    assert p["action"] == "clean" and p["bad_slots"] == []


def test_plan_demote_one_bad_of_four():
    p = dnf.plan_demote(_tile_dict([1.0, 0.5, 1.0, 1.0]),
                        min_valid_frac=0.90, min_good_frames=3)
    assert p["action"] == "demote"
    assert p["bad_slots"] == [1] and p["n_present"] == 4 and p["n_good"] == 3


def test_plan_residual_below_floor():
    # 2 present (slots 0,1), slot 1 bad → n_good=1 < 3 → residual, untouched.
    p = dnf.plan_demote(_tile_dict([1.0, 0.4, 0.0, 0.0], mask=[1, 1, 0, 0]),
                        min_valid_frac=0.90, min_good_frames=3)
    assert p["action"] == "residual"
    assert p["bad_slots"] == [1] and p["n_present"] == 2 and p["n_good"] == 1


def test_plan_skipped_no_spectral():
    p = dnf.plan_demote({"label": np.zeros((4, 4))},
                        min_valid_frac=0.90, min_good_frames=3)
    assert p["action"] == "skipped"


# ----------------------------- _demoted_mask (pure) -------------------------

def test_demoted_mask_zeroes_bad_slot():
    m = dnf._demoted_mask(np.array([1, 1, 1, 1], np.uint8), [1], [0, 1, 2, 3])
    assert m.tolist() == [1, 0, 1, 1]


def test_demoted_mask_pads_short_mask():
    # bad slot 3 lies past a length-2 stored mask → mask padded, slot zeroed.
    m = dnf._demoted_mask(np.array([1, 1], np.uint8), [3], [0, 1, 2, 3])
    assert m.tolist() == [1, 1, 1, 0]


def test_demoted_mask_none_builds_ones():
    m = dnf._demoted_mask(None, [2], [0, 1, 2, 3])
    assert m.tolist() == [1, 1, 0, 1]


# --------------------------- demote_one (disk round-trip) -------------------

def test_demote_one_writes_mask_and_passes_gate(tmp_path):
    extra = {"label": np.full((10, 10), 7, np.uint8),
             "aux_height": np.full((10, 10), 1.5, np.float32),
             "bbox_3006": np.array([1.0, 2.0, 3.0, 4.0])}
    orig = _tile_dict([1.0, 0.5, 1.0, 1.0], **extra)
    p = _write(tmp_path / "t.npz", orig)

    r = dnf.demote_one(p, min_valid_frac=0.90, min_good_frames=3)
    assert r["status"] == "demote" and r["bad_slots"] == [1]

    with np.load(p, allow_pickle=True) as d:
        assert d["temporal_mask"].tolist() == [1, 0, 1, 1]
        # every other field preserved bit-identical
        assert np.array_equal(d["spectral"], orig["spectral"])
        assert np.array_equal(d["label"], orig["label"])
        assert np.array_equal(d["aux_height"], orig["aux_height"])
        assert np.array_equal(d["bbox_3006"], orig["bbox_3006"])
        # the gate now passes (the bad frame is no longer present)
        assert check_frame_coverage(d, min_valid_frac=0.90)["status"] == "pass"


def test_residual_tile_untouched(tmp_path):
    orig = _tile_dict([1.0, 0.4, 0.0, 0.0], mask=[1, 1, 0, 0])
    p = _write(tmp_path / "r.npz", orig)
    before = Path(p).read_bytes()

    r = dnf.demote_one(p, min_valid_frac=0.90, min_good_frames=3)
    assert r["status"] == "residual"
    assert Path(p).read_bytes() == before  # no write at all


def test_idempotent_rerun_is_clean(tmp_path):
    p = _write(tmp_path / "i.npz", _tile_dict([1.0, 0.5, 1.0, 1.0]))
    assert dnf.demote_one(p, min_valid_frac=0.90, min_good_frames=3)["status"] == "demote"
    after_first = Path(p).read_bytes()
    # second run: the demoted frame is no longer present → clean, no re-write
    assert dnf.demote_one(p, min_valid_frac=0.90, min_good_frames=3)["status"] == "clean"
    assert Path(p).read_bytes() == after_first


def test_dry_run_does_not_write(tmp_path):
    p = _write(tmp_path / "d.npz", _tile_dict([1.0, 0.5, 1.0, 1.0]))
    before = Path(p).read_bytes()
    r = dnf.demote_one(p, min_valid_frac=0.90, min_good_frames=3, dry_run=True)
    assert r["status"] == "demote" and r.get("dry_run") is True
    assert Path(p).read_bytes() == before


def test_run_partitions_directory(tmp_path):
    # one demote, one residual, one clean → run() reports the partition.
    _write(tmp_path / "demote.npz", _tile_dict([1.0, 0.5, 1.0, 1.0]))
    _write(tmp_path / "residual.npz", _tile_dict([1.0, 0.4, 0.0, 0.0], mask=[1, 1, 0, 0]))
    _write(tmp_path / "clean.npz", _tile_dict([1.0, 1.0, 1.0, 1.0]))
    res = dnf.run(str(tmp_path), workers=1, min_valid_frac=0.90, min_good_frames=3)
    assert res["ok"] and res["demote"] == 1 and res["residual"] == 1 and res["clean"] == 1
