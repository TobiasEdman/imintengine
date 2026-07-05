"""Unit tests for scripts/frame_cloud_qc.py — per-frame cloud QC + replace.

All network / heavy deps are mocked: ``era5_prefilter_dates``,
``optimal_fetch_dates`` and ``fetch_tile_spectral`` are patched at their source
modules (the QC script imports them lazily, so patching the definition site is
what the runtime lookup resolves to). Synthetic tiles are tiny in-memory
``(6, H, W)`` cubes written to a ``tmp_path`` npz; no /data, no cluster.

Coverage:
  * measure flags >threshold B02 frames correctly (and passes clean ones);
  * the B02 brightness metric counts only valid (non-zero) pixels;
  * replace re-selects a SAME-year + SAME-window date (asserts new year ==
    slot year for both a 2018-tile growing slot and slot 0 = year-1 autumn);
  * never-worse guard keeps the original when the candidate is cloudier;
  * idempotency — a second replace pass over an already-replaced frame no-ops.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.frame_cloud_qc as qc


# ── Synthetic tile helpers ───────────────────────────────────────────────────

H = W = 16  # tiny tiles keep the tests fast


def _frame(b02_value: float, *, fill_frac: float = 1.0) -> np.ndarray:
    """A ``(6, H, W)`` frame whose B02 band (idx 0) is ``b02_value``.

    ``fill_frac`` of the pixels carry ``b02_value``; the rest are 0 (nodata),
    to exercise the valid-pixel masking. Non-B02 bands are a mid value so the
    frame is non-empty (np.any → True).
    """
    f = np.full((6, H, W), 0.1, dtype=np.float32)
    b02 = np.zeros((H, W), dtype=np.float32)
    n_fill = int(round(fill_frac * H * W))
    flat = b02.reshape(-1)
    flat[:n_fill] = b02_value
    f[qc.B02_OFFSET] = flat.reshape(H, W)
    return f


def _make_tile(
    path: Path,
    *,
    tessera_year: int = 2022,
    slot_b02: dict[int, float] | None = None,
    frame_2016_b02: float | None = None,
    slot_dates: dict[int, str] | None = None,
    frame_2016_date: str = "2016-07-15",
    coreg_ref: int | None = None,
) -> Path:
    """Write a synthetic recoreg-style npz.

    ``slot_b02`` maps slot_idx → B02 value (defines its cloud fraction). Absent
    slots are left empty (mask 0). Geometry is a real EPSG:3006 bbox over Sweden
    so ``bbox_3006_to_wgs84`` resolves (it is NOT mocked — it is pure pyproj).
    ``coreg_ref`` sets ``coreg_ref_frame`` (and fills that slot clean if absent) so
    the replacement coreg-to-anchor has a stored anchor to align onto.
    """
    slot_b02 = dict(slot_b02 or {})
    slot_dates = slot_dates or {}
    if coreg_ref is not None:
        slot_b02.setdefault(coreg_ref, 0.05)   # clean anchor frame
    spec = np.zeros((qc.N_SLOTS * qc.N_BANDS, H, W), dtype=np.float32)
    tmask = np.zeros(qc.N_SLOTS, dtype=np.uint8)
    dates = [""] * qc.N_SLOTS
    for si in range(qc.N_SLOTS):
        if si in slot_b02:
            spec[si * qc.N_BANDS:(si + 1) * qc.N_BANDS] = _frame(slot_b02[si])
            tmask[si] = 1
            dates[si] = slot_dates.get(si, f"{tessera_year}-07-01")

    # A real ~5 km tile bbox in EPSG:3006 (central Sweden), 10 m grid-snapped.
    west, south = 600000, 6600000
    size_m = 512 * 10
    data = {
        "spectral": spec,
        "temporal_mask": tmask,
        "dates": np.array(dates),
        "tessera_year": np.int32(tessera_year),
        "bbox_3006": np.array(
            [west, south, west + size_m, south + size_m], dtype=np.int32),
        "easting": np.int32(west + size_m // 2),
        "northing": np.int32(south + size_m // 2),
        "tile_size_px": np.int32(512),
    }
    if frame_2016_b02 is not None:
        data["frame_2016"] = _frame(frame_2016_b02)
        data["has_frame_2016"] = np.int32(1)
        data["frame_2016_date"] = np.str_(frame_2016_date)
        data["frame_2016_year"] = np.int32(2016)
    if coreg_ref is not None:
        data["coreg_ref_frame"] = np.int32(coreg_ref)
    np.savez_compressed(str(path), **data)
    return path


# ── B02 metric ───────────────────────────────────────────────────────────────

def test_b02_cloud_fraction_all_bright():
    # Every valid pixel above the 0.20 cut → fraction 1.0.
    assert qc.b02_cloud_fraction(_frame(0.5)) == pytest.approx(1.0)


def test_b02_cloud_fraction_all_dark():
    # Dark blue band (vegetation/water) → 0.0 cloud.
    assert qc.b02_cloud_fraction(_frame(0.05)) == pytest.approx(0.0)


def test_b02_cloud_fraction_ignores_nodata_zeros():
    # Half the pixels are nodata (0), the other half are bright (0.5).
    # Valid-pixel masking → fraction over the VALID half = 1.0, not 0.5.
    f = _frame(0.5, fill_frac=0.5)
    assert qc.b02_cloud_fraction(f) == pytest.approx(1.0)


def test_b02_cloud_fraction_empty_is_max():
    # An all-zero frame has no valid pixels → treated as maximally cloudy.
    assert qc.b02_cloud_fraction(np.zeros((6, H, W), np.float32)) == 1.0


# ── slot → (year, window) temporal-matching ──────────────────────────────────

def test_slot_year_mapping():
    # slot 0 = autumn of year-1; slots 1-3 = tile year; frame_2016 = 2016.
    assert qc.slot_year(2022, "slot:0") == 2021
    assert qc.slot_year(2022, "slot:1") == 2022
    assert qc.slot_year(2022, "slot:3") == 2022
    assert qc.slot_year(2022, "frame_2016") == 2016


def test_frame_window_uses_slot_year():
    # slot 0 of a 2018 tile → autumn 2017 window (Aug-Oct 2017).
    start, end = qc.frame_window(2018, "slot:0")
    assert start == "2017-08-15" and end == "2017-10-31"
    # slot 2 of a 2022 tile → growing-season 2022 window.
    start2, end2 = qc.frame_window(2022, "slot:2")
    assert start2 == "2022-06-01" and end2 == "2022-07-31"


# ── Phase A: measure ─────────────────────────────────────────────────────────

def test_measure_flags_cloudy_frames(tmp_path, monkeypatch):
    # Slot 1 clean (B02=0.05), slot 2 cloudy (B02=0.5 → fraction 1.0 > 0.2).
    p = _make_tile(tmp_path / "tile_a.npz",
                   slot_b02={1: 0.05, 2: 0.5},
                   frame_2016_b02=0.05)
    # ERA5 patched to "all clear" so it never gates — we test the PIXEL gate.
    monkeypatch.setattr(
        "imint.training.optimal_fetch.era5_prefilter_dates",
        lambda bbox, d0, d1, **kw: [d0])

    res = qc.measure_tile(str(p), threshold=0.2, bright=0.2, with_era5=True)
    frames = res["frames"]
    assert frames["slot:1"]["present"] and frames["slot:1"]["pass"] is True
    assert frames["slot:2"]["present"] and frames["slot:2"]["pass"] is False
    assert frames["slot:2"]["cloud_frac"] == pytest.approx(1.0)
    # Empty slots reported absent, not failing.
    assert frames["slot:0"]["present"] is False
    assert frames["frame_2016"]["present"] is True
    assert frames["frame_2016"]["pass"] is True


def test_measure_report_aggregates_and_replace_list(tmp_path, monkeypatch):
    _make_tile(tmp_path / "t1.npz", slot_b02={1: 0.5, 2: 0.05})  # slot1 fails
    _make_tile(tmp_path / "t2.npz", slot_b02={1: 0.05, 2: 0.05})  # both clean
    monkeypatch.setattr(
        "imint.training.optimal_fetch.era5_prefilter_dates",
        lambda bbox, d0, d1, **kw: [d0])

    report_path = tmp_path / "report.json"
    args = qc.build_parser().parse_args([
        "--phase", "measure", "--data-dir", str(tmp_path),
        "--report", str(report_path), "--threshold", "0.2", "--workers", "1"])
    rc = qc.run_measure(args)
    assert rc == 0

    report = json.loads(report_path.read_text())
    # One failing frame total (t1 slot:1).
    assert report["summary"]["n_frames_to_replace"] == 1
    assert report["summary"]["per_slot"]["slot:1"]["fail"] == 1
    assert report["summary"]["per_slot"]["slot:1"]["present"] == 2
    rl = report["replace_list"]
    assert len(rl) == 1 and rl[0]["tile"] == "t1" and rl[0]["slot"] == "slot:1"


def test_measure_dry_run_writes_nothing(tmp_path, monkeypatch):
    _make_tile(tmp_path / "t1.npz", slot_b02={1: 0.5})
    monkeypatch.setattr(
        "imint.training.optimal_fetch.era5_prefilter_dates",
        lambda bbox, d0, d1, **kw: [d0])
    report_path = tmp_path / "report.json"
    args = qc.build_parser().parse_args([
        "--phase", "measure", "--data-dir", str(tmp_path),
        "--report", str(report_path), "--dry-run", "--workers", "1"])
    assert qc.run_measure(args) == 0
    assert not report_path.exists()


# ── Phase B: replace ─────────────────────────────────────────────────────────

def _patch_select(monkeypatch, returned_dates: list[str]):
    """Patch optimal_fetch_dates to return a FetchPlan-like object."""
    class _Plan:
        def __init__(self, dates):
            self.dates = dates

    def _fake(bbox, d0, d1, **kw):
        # Echo back only dates inside the requested [d0, d1] window, so the test
        # proves the SAME-WINDOW query is what bounds the candidate set.
        return _Plan([d for d in returned_dates if d0 <= d <= d1])

    monkeypatch.setattr(
        "imint.training.optimal_fetch.optimal_fetch_dates", _fake)


def _patch_fetch(monkeypatch, b02_value: float):
    """Patch fetch_tile_spectral to return a 1-frame cube with a given B02."""
    def _fake(center, *, tile, dates, n_frames, backend, halo_px, coregister,
              with_scl=False):
        # Mirror the real contract: spectral is (n_frames*6, H, W). QC replacement
        # calls with with_scl=False (spectral-only), which the entry accepts.
        return {"spectral": _frame(b02_value), "temporal_mask": np.array([1])}

    monkeypatch.setattr(
        "imint.training.fetch_spectral.fetch_tile_spectral", _fake)


def test_replace_reselects_same_year_growing_slot(tmp_path, monkeypatch):
    # 2022 tile, slot 2 cloudy. Candidate within the 2022 growing window, plus
    # a wrong-year 2021 date the guard MUST reject.
    p = _make_tile(tmp_path / "t.npz",
                   slot_b02={2: 0.5}, slot_dates={2: "2022-06-20"}, coreg_ref=1)
    _patch_select(monkeypatch, ["2021-06-25", "2022-06-28", "2022-07-05"])
    _patch_fetch(monkeypatch, 0.05)  # candidate is clean (frac 0.0)

    verdict = qc.replace_tile_frame(
        str(p), "slot:2", orig_cloud_frac=1.0,
        threshold=0.2, bright=0.2, halo_px=8, max_aoi_cloud=0.10,
        dry_run=False)

    assert verdict["action"] == "replaced"
    assert verdict["new_year"] == 2022
    # Same year as the tile; inside the growing window 2022-06-01..07-31.
    assert verdict["new_date"] == "2022-06-28"
    # Persisted: new date + the clean frame written into slot 2.
    with np.load(p, allow_pickle=True) as d:
        assert str(d["dates"][2]) == "2022-06-28"
        new_frame = d["spectral"][2 * qc.N_BANDS:(2 + 1) * qc.N_BANDS]
        assert qc.b02_cloud_fraction(new_frame) == pytest.approx(0.0)
        assert int(d["qc_replaced_slot_2"]) == 1


def test_replace_slot0_uses_year_minus_one(tmp_path, monkeypatch):
    # 2018 tile, slot 0 = autumn of 2017. Replacement MUST land in 2017.
    p = _make_tile(tmp_path / "t.npz", tessera_year=2018,
                   slot_b02={0: 0.5}, slot_dates={0: "2017-09-10"}, coreg_ref=1)
    # Offer a 2018 date (wrong year for slot 0) + a valid 2017-autumn date.
    _patch_select(monkeypatch, ["2018-09-15", "2017-09-20"])
    _patch_fetch(monkeypatch, 0.05)

    verdict = qc.replace_tile_frame(
        str(p), "slot:0", orig_cloud_frac=1.0,
        threshold=0.2, bright=0.2, halo_px=8, max_aoi_cloud=0.10,
        dry_run=False)

    assert verdict["action"] == "replaced"
    assert verdict["new_year"] == 2017
    assert verdict["new_date"] == "2017-09-20"


def _half_cloudy_frame() -> np.ndarray:
    """A frame where half the (all-valid, nonzero) pixels are bright → frac 0.5."""
    f = np.full((6, H, W), 0.1, dtype=np.float32)
    b02 = np.full((H, W), 0.05, dtype=np.float32)  # dark but VALID (nonzero)
    flat = b02.reshape(-1)
    flat[:flat.size // 2] = 0.5  # bright half
    f[qc.B02_OFFSET] = flat.reshape(H, W)
    return f


def test_replace_never_worse_keeps_original(tmp_path, monkeypatch):
    # Original slot 2 is moderately cloudy (frac 0.5); candidate is CLOUDIER
    # (frac 1.0) → never-worse guard keeps the original, writes nothing.
    p = _make_tile(tmp_path / "t.npz",
                   slot_b02={2: 0.5}, slot_dates={2: "2022-06-20"}, coreg_ref=1)
    # Original B02 fraction: half bright / half dark, all valid → 0.5.
    orig = _half_cloudy_frame()
    with np.load(p, allow_pickle=True) as d:
        data = {k: d[k] for k in d.files}
    spec = np.asarray(data["spectral"], np.float32).copy()   # keep the slot-1 anchor
    spec[2 * qc.N_BANDS:(2 + 1) * qc.N_BANDS] = orig
    data["spectral"] = spec
    np.savez_compressed(str(p), **data)
    orig_frac = qc.b02_cloud_fraction(orig)
    assert orig_frac == pytest.approx(0.5)

    _patch_select(monkeypatch, ["2022-07-01"])
    _patch_fetch(monkeypatch, 0.5)  # candidate all-bright → frac 1.0 (worse)

    verdict = qc.replace_tile_frame(
        str(p), "slot:2", orig_cloud_frac=orig_frac,
        threshold=0.2, bright=0.2, halo_px=8, max_aoi_cloud=0.10,
        dry_run=False)

    assert verdict["action"] == "keep-original-never-worse"
    # Disk unchanged: date still original, no qc_replaced marker.
    with np.load(p, allow_pickle=True) as d:
        assert str(d["dates"][2]) == "2022-06-20"
        assert "qc_replaced_slot_2" not in d.files


def test_replace_no_candidate_in_window(tmp_path, monkeypatch):
    # Selector returns only a wrong-year date → no same-year candidate → no-op.
    p = _make_tile(tmp_path / "t.npz",
                   slot_b02={2: 0.5}, slot_dates={2: "2022-06-20"})
    _patch_select(monkeypatch, ["2099-06-20"])  # impossible year
    verdict = qc.replace_tile_frame(
        str(p), "slot:2", orig_cloud_frac=1.0,
        threshold=0.2, bright=0.2, halo_px=8, max_aoi_cloud=0.10,
        dry_run=False)
    assert verdict["action"] == "no-candidate"


def test_replace_idempotent(tmp_path, monkeypatch):
    # First replace succeeds; a second pass over the same (now clean, marked)
    # frame must skip without re-fetching.
    p = _make_tile(tmp_path / "t.npz",
                   slot_b02={2: 0.5}, slot_dates={2: "2022-06-20"}, coreg_ref=1)
    _patch_select(monkeypatch, ["2022-06-28"])
    _patch_fetch(monkeypatch, 0.05)

    v1 = qc.replace_tile_frame(
        str(p), "slot:2", orig_cloud_frac=1.0,
        threshold=0.2, bright=0.2, halo_px=8, max_aoi_cloud=0.10,
        dry_run=False)
    assert v1["action"] == "replaced"

    # Make a second fetch FAIL loudly if it is wrongly called.
    def _boom(*a, **k):
        raise AssertionError("fetch_tile_spectral must not run on idempotent skip")
    monkeypatch.setattr(
        "imint.training.fetch_spectral.fetch_tile_spectral", _boom)

    v2 = qc.replace_tile_frame(
        str(p), "slot:2", orig_cloud_frac=0.0,
        threshold=0.2, bright=0.2, halo_px=8, max_aoi_cloud=0.10,
        dry_run=False)
    assert v2["action"] == "skip-already-replaced"


def test_replace_dry_run_plans_without_fetch(tmp_path, monkeypatch):
    # Dry-run resolves the candidate + asserts the year guard but never fetches
    # or writes.
    p = _make_tile(tmp_path / "t.npz",
                   slot_b02={2: 0.5}, slot_dates={2: "2022-06-20"})
    _patch_select(monkeypatch, ["2022-06-28"])

    def _boom(*a, **k):
        raise AssertionError("dry-run must not fetch")
    monkeypatch.setattr(
        "imint.training.fetch_spectral.fetch_tile_spectral", _boom)

    verdict = qc.replace_tile_frame(
        str(p), "slot:2", orig_cloud_frac=1.0,
        threshold=0.2, bright=0.2, halo_px=8, max_aoi_cloud=0.10,
        dry_run=True)
    assert verdict["action"] == "would-replace"
    assert verdict["new_date"] == "2022-06-28" and verdict["new_year"] == 2022
    with np.load(p, allow_pickle=True) as d:
        assert "qc_replaced_slot_2" not in d.files


def test_run_replace_reads_report_replace_list(tmp_path, monkeypatch):
    # End-to-end Phase-B driver: build a report with one failing frame, then
    # run_replace consumes it and replaces that frame.
    _make_tile(tmp_path / "t1.npz",
               slot_b02={2: 0.5}, slot_dates={2: "2022-06-20"}, coreg_ref=1)
    report = {
        "summary": {},
        "tiles": [],
        "replace_list": [
            {"tile": "t1", "slot": "slot:2", "date": "2022-06-20",
             "cloud_frac": 1.0}],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    _patch_select(monkeypatch, ["2022-06-28"])
    _patch_fetch(monkeypatch, 0.05)

    log_path = tmp_path / "verdicts.json"
    args = qc.build_parser().parse_args([
        "--phase", "replace", "--data-dir", str(tmp_path),
        "--report", str(report_path), "--replace-log", str(log_path),
        "--threshold", "0.2", "--workers", "1"])
    assert qc.run_replace(args) == 0

    log = json.loads(log_path.read_text())
    assert log["counts"].get("replaced") == 1


def test_coreg_to_anchor_reverse_fit_and_skips():
    """_coreg_to_stored_anchor drives a shifted fresh frame's structure back onto
    the stored anchor (dot/center-of-mass reverse-fit — CLAUDE.md mandates dot/COM
    for coreg, since a sign-flipped estimator would AMPLIFY the shift), and skips
    safely when there is no anchor or the frame IS the anchor slot."""
    import math
    from scipy.ndimage import center_of_mass, gaussian_filter
    from imint.coregistration import subpixel_shift

    sz = 160
    rng = np.random.default_rng(3)
    base = gaussian_filter(rng.standard_normal((sz, sz)).astype(np.float32), 1.6)
    base = (base - base.min()) / (base.max() - base.min())
    anchor6 = np.repeat(base[None], 6, axis=0).astype(np.float32)   # B04 = idx 2
    spec = np.zeros((qc.N_SLOTS * qc.N_BANDS, sz, sz), np.float32)
    spec[1 * qc.N_BANDS:2 * qc.N_BANDS] = anchor6                  # anchor at slot 1
    data = {"coreg_ref_frame": np.int32(1), "spectral": spec}

    # Sub-pixel shift — coregister_to_reference handles the post-M1 residual
    # (≤1 px budget; it auto-rejects larger as a no-op, same as the campaign's
    # frame_2016 coreg-to-anchor).
    shifted = subpixel_shift(base.astype(np.float64), 0.45, -0.30).astype(np.float32)
    fresh6 = np.repeat(shifted[None], 6, axis=0).astype(np.float32)

    aligned, reason = qc._coreg_to_stored_anchor(data, fresh6, "slot:2")
    assert reason == "ok" and aligned is not None

    def _com(b):
        return np.array(center_of_mass(np.clip(b - 0.3, 0.0, None)))
    a = _com(anchor6[2])
    pre = math.hypot(*(_com(fresh6[2]) - a))
    post = math.hypot(*(_com(aligned[2]) - a))
    assert post < pre                          # moved TOWARD the anchor (sign correct)

    # Skip guards (frame_cloud_qc-specific):
    assert qc._coreg_to_stored_anchor(
        {"spectral": spec}, fresh6, "slot:2")[1] == "no-anchor"
    assert qc._coreg_to_stored_anchor(
        data, fresh6, "slot:1")[1] == "is-anchor-frame"
