"""restore_recoreg_labels — carry-forward of label/label_mask/label_year from the
original unified_v2_512 tiles into the _recoreg tiles.

Synthetic fixtures only (no /data, no cluster): a ``_recoreg`` tile with spectral +
aux but NO label, and a same-named ``orig`` tile carrying the three label fields.
Asserts the copy, field preservation, the skip/idempotent gates, and that the write
is atomic (verified by reading the npz back from disk).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import restore_recoreg_labels as rr  # noqa: E402


# ── fixtures ─────────────────────────────────────────────────────────────────


def _recoreg_tile(path: Path, *, with_label: bool = False) -> dict:
    """A post-refetch _recoreg tile: spectral + a couple of preserved non-spectral
    fields, NO label (unless ``with_label`` for the idempotency test)."""
    fields = {
        "spectral": np.arange(24 * 8 * 8, dtype=np.float32).reshape(24, 8, 8),
        "temporal_mask": np.array([0, 1, 1, 1], np.uint8),
        "dates": np.array(["", "2022-06-01", "2022-07-01", "2022-08-01"]),
        "coreg_ref_frame": np.int32(1),
        "dem": np.full((8, 8), 12.5, np.float32),          # an aux channel to preserve
        "source": np.str_("lulc"),
    }
    if with_label:
        fields["label"] = np.full((8, 8), 7, np.uint8)
    np.savez_compressed(path, **fields)
    return fields


def _orig_tile(path: Path, *, label: bool = True, mask: bool = True,
               year: bool = True) -> dict:
    """A same-named original unified_v2_512 tile carrying the label fields."""
    lab = np.zeros((8, 8), np.uint8)
    lab[:4] = 3      # granskog
    lab[4:] = 11     # vete
    fields: dict = {
        "spectral": np.zeros((24, 8, 8), np.float32),
    }
    if label:
        fields["label"] = lab
    if mask:
        fields["label_mask"] = (lab > 0).astype(np.uint8)
    if year:
        fields["label_year"] = np.int32(2022)
    np.savez_compressed(path, **fields)
    return fields


# ── happy path: all three copied, other fields preserved ─────────────────────


def test_restores_all_three_label_fields_and_preserves_others(tmp_path):
    rec_dir = tmp_path / "recoreg"; rec_dir.mkdir()
    orig_dir = tmp_path / "orig"; orig_dir.mkdir()
    rec_fields = _recoreg_tile(rec_dir / "t.npz")
    orig_fields = _orig_tile(orig_dir / "t.npz")

    res = rr.restore_one(str(rec_dir / "t.npz"), str(orig_dir))
    assert res["status"] == "restored"
    assert res["fields"] == ["label", "label_mask", "label_year"]

    # Read back from disk — proves the write landed (atomic os.replace), not just
    # an in-memory mutation.
    with np.load(rec_dir / "t.npz", allow_pickle=True) as d:
        keys = set(d.files)
        # all three label fields present and equal to the ORIGINAL's values
        for k in ("label", "label_mask", "label_year"):
            assert k in keys
            np.testing.assert_array_equal(d[k], orig_fields[k])
        # every original _recoreg field preserved byte-for-byte
        for k, v in rec_fields.items():
            assert k in keys
            np.testing.assert_array_equal(d[k], v)
        # the original's spectral did NOT bleed across (label-only carry)
        np.testing.assert_array_equal(d["spectral"], rec_fields["spectral"])


def test_restores_only_fields_present_in_original(tmp_path):
    """Original with label but no label_mask/label_year → only label carried."""
    rec_dir = tmp_path / "recoreg"; rec_dir.mkdir()
    orig_dir = tmp_path / "orig"; orig_dir.mkdir()
    _recoreg_tile(rec_dir / "t.npz")
    _orig_tile(orig_dir / "t.npz", label=True, mask=False, year=False)

    res = rr.restore_one(str(rec_dir / "t.npz"), str(orig_dir))
    assert res["status"] == "restored" and res["fields"] == ["label"]
    with np.load(rec_dir / "t.npz", allow_pickle=True) as d:
        assert "label" in d.files
        assert "label_mask" not in d.files and "label_year" not in d.files


# ── skip: original has no label ──────────────────────────────────────────────


def test_skips_when_original_lacks_label(tmp_path):
    rec_dir = tmp_path / "recoreg"; rec_dir.mkdir()
    orig_dir = tmp_path / "orig"; orig_dir.mkdir()
    _recoreg_tile(rec_dir / "t.npz")
    _orig_tile(orig_dir / "t.npz", label=False, mask=True, year=True)

    res = rr.restore_one(str(rec_dir / "t.npz"), str(orig_dir))
    assert res["status"] == "orig_missing" and res["reason"] == "orig_no_label"
    # _recoreg tile untouched — still no label, no stray label_mask copied.
    with np.load(rec_dir / "t.npz", allow_pickle=True) as d:
        assert "label" not in d.files and "label_mask" not in d.files


def test_orig_missing_file(tmp_path):
    rec_dir = tmp_path / "recoreg"; rec_dir.mkdir()
    orig_dir = tmp_path / "orig"; orig_dir.mkdir()
    _recoreg_tile(rec_dir / "t.npz")  # no same-named original at all

    res = rr.restore_one(str(rec_dir / "t.npz"), str(orig_dir))
    assert res["status"] == "orig_missing" and res["reason"] == "no_orig_file"


# ── idempotent: _recoreg already has a label ─────────────────────────────────


def test_idempotent_noop_when_recoreg_already_has_label(tmp_path):
    rec_dir = tmp_path / "recoreg"; rec_dir.mkdir()
    orig_dir = tmp_path / "orig"; orig_dir.mkdir()
    _recoreg_tile(rec_dir / "t.npz", with_label=True)   # already labelled (class 7)
    _orig_tile(orig_dir / "t.npz")                       # original is classes 3/11

    before = (rec_dir / "t.npz").stat().st_mtime_ns
    res = rr.restore_one(str(rec_dir / "t.npz"), str(orig_dir))
    assert res["status"] == "skipped"
    # File not rewritten (mtime unchanged) and the existing label NOT overwritten
    # by the original's differing label.
    assert (rec_dir / "t.npz").stat().st_mtime_ns == before
    with np.load(rec_dir / "t.npz", allow_pickle=True) as d:
        np.testing.assert_array_equal(d["label"], np.full((8, 8), 7, np.uint8))


def test_rerun_is_a_noop(tmp_path):
    """First pass restores; a second pass over the same tile skips (resumable)."""
    rec_dir = tmp_path / "recoreg"; rec_dir.mkdir()
    orig_dir = tmp_path / "orig"; orig_dir.mkdir()
    _recoreg_tile(rec_dir / "t.npz")
    _orig_tile(orig_dir / "t.npz")

    assert rr.restore_one(str(rec_dir / "t.npz"), str(orig_dir))["status"] == "restored"
    mtime_after_first = (rec_dir / "t.npz").stat().st_mtime_ns
    second = rr.restore_one(str(rec_dir / "t.npz"), str(orig_dir))
    assert second["status"] == "skipped"
    assert (rec_dir / "t.npz").stat().st_mtime_ns == mtime_after_first


# ── atomicity: no dangling tmp, write is complete & readable ──────────────────


def test_write_is_atomic_no_tmp_left_and_readable(tmp_path):
    rec_dir = tmp_path / "recoreg"; rec_dir.mkdir()
    orig_dir = tmp_path / "orig"; orig_dir.mkdir()
    _recoreg_tile(rec_dir / "t.npz")
    _orig_tile(orig_dir / "t.npz")

    rr.restore_one(str(rec_dir / "t.npz"), str(orig_dir))
    # No sibling ``.tmp.npz`` left behind, exactly one npz on disk.
    listing = sorted(os.listdir(rec_dir))
    assert listing == ["t.npz"]
    assert not (rec_dir / "t.tmp.npz").exists()
    # The file is a fully-valid npz (re-open + read a label pixel back).
    with np.load(rec_dir / "t.npz", allow_pickle=True) as d:
        assert int(d["label"][0, 0]) == 3


# ── dry-run: reports the restore but writes nothing ──────────────────────────


def test_dry_run_reports_but_does_not_write(tmp_path):
    rec_dir = tmp_path / "recoreg"; rec_dir.mkdir()
    orig_dir = tmp_path / "orig"; orig_dir.mkdir()
    _recoreg_tile(rec_dir / "t.npz")
    _orig_tile(orig_dir / "t.npz")
    before = (rec_dir / "t.npz").stat().st_mtime_ns

    res = rr.restore_one(str(rec_dir / "t.npz"), str(orig_dir), dry_run=True)
    assert res["status"] == "restored" and res["dry_run"] is True
    assert res["fields"] == ["label", "label_mask", "label_year"]
    # No write happened.
    assert (rec_dir / "t.npz").stat().st_mtime_ns == before
    with np.load(rec_dir / "t.npz", allow_pickle=True) as d:
        assert "label" not in d.files


# ── batch driver: summary counts across a mixed directory ────────────────────


def test_restore_all_summary_counts(tmp_path):
    rec_dir = tmp_path / "recoreg"; rec_dir.mkdir()
    orig_dir = tmp_path / "orig"; orig_dir.mkdir()

    # a: restorable. b: already labelled (skip). c: original lacks label.
    # d: no original file at all (orig_missing).
    _recoreg_tile(rec_dir / "a.npz")
    _orig_tile(orig_dir / "a.npz")
    _recoreg_tile(rec_dir / "b.npz", with_label=True)
    _orig_tile(orig_dir / "b.npz")
    _recoreg_tile(rec_dir / "c.npz")
    _orig_tile(orig_dir / "c.npz", label=False)
    _recoreg_tile(rec_dir / "d.npz")  # no orig/d.npz

    counts = rr.restore_all(str(rec_dir), str(orig_dir), workers=2)
    assert counts["restored"] == 1
    assert counts["skipped"] == 1
    assert counts["orig_missing"] == 2     # c (no label) + d (no file)
    assert counts["error"] == 0

    # Only tile a gained a label on disk.
    with np.load(rec_dir / "a.npz", allow_pickle=True) as d:
        assert "label" in d.files
    with np.load(rec_dir / "c.npz", allow_pickle=True) as d:
        assert "label" not in d.files
