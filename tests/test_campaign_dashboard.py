"""campaign_dashboard — progress/rate/ETA math + DES-styled render.

Fixtures are empty ``*.npz`` touch-files with controlled mtimes (the dashboard
only counts files + reads mtimes; it never loads them).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import campaign_dashboard as cd  # noqa: E402


def _tiles(d: Path, n: int, *, mtimes: list[float] | None = None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        p = d / f"tile_{i}.npz"
        p.touch()
        if mtimes is not None:
            os.utime(p, (mtimes[i], mtimes[i]))


def test_counts_and_pct(tmp_path):
    d = tmp_path / "recoreg"
    _tiles(d, 25)
    s = cd.build_status(str(d), total=100)
    assert s["exists"] and s["done"] == 25 and s["total"] == 100
    assert s["pct"] == 25.0 and s["remaining"] == 75


def test_missing_dir(tmp_path):
    s = cd.build_status(str(tmp_path / "nope"), total=100)
    assert s["exists"] is False and s["done"] == 0
    assert s["eta_hours"] is None and s["rate_recent_per_h"] == 0.0


def test_recent_rate_drives_eta(tmp_path):
    d = tmp_path / "recoreg"
    now = time.time()
    # 60 tiles all within the last 30 min → recent rate = 60 / 0.5h = 120/h.
    _tiles(d, 60, mtimes=[now - 600 for _ in range(60)])
    s = cd.build_status(str(d), total=120, now=now)
    assert s["rate_recent_per_h"] == 120.0
    # remaining 60 at 120/h → 0.5 h ETA.
    assert s["eta_hours"] == 0.5


def test_overall_rate_fallback_when_no_recent(tmp_path):
    d = tmp_path / "recoreg"
    now = time.time()
    # All tiles older than the 30-min window → recent rate 0, overall drives ETA.
    _tiles(d, 10, mtimes=[now - 3600 * (10 - i) for i in range(10)])  # 1 tile/h-ish
    s = cd.build_status(str(d), total=20, now=now)
    assert s["rate_recent_per_h"] == 0.0
    assert s["rate_overall_per_h"] > 0
    assert s["eta_hours"] is not None        # falls back to overall rate


def test_just_started_does_not_explode_overall_rate(tmp_path):
    """All tiles written ~now (first tile seconds ago) → overall rate would
    divide by ~0; the ≥3-min-history guard pins it to 0 instead of a huge number."""
    d = tmp_path / "recoreg"
    now = time.time()
    _tiles(d, 137, mtimes=[now - 1 for _ in range(137)])   # all 1 s old
    s = cd.build_status(str(d), total=6921, now=now)
    assert s["rate_overall_per_h"] == 0.0                  # guarded, not ~493k/h
    assert s["rate_recent_per_h"] > 0                      # recent window still works


def test_complete_has_zero_eta(tmp_path):
    d = tmp_path / "recoreg"
    _tiles(d, 100)
    s = cd.build_status(str(d), total=100)
    assert s["remaining"] == 0 and s["eta_hours"] == 0.0


def test_render_html_is_des_styled_and_has_numbers(tmp_path):
    d = tmp_path / "recoreg"
    _tiles(d, 42)
    s = cd.build_status(str(d), total=6921)
    html = cd.render_html(s, title="Re-coreg campaign")
    assert "<!doctype html>" in html.lower()
    assert "#1A4338" in html and "Space+Grotesk" in html   # DES identity
    assert 'http-equiv="refresh"' in html                  # auto-refresh
    assert "42" in html and "6,921" in html                # done + total rendered


def test_write_emits_html_and_json(tmp_path):
    d = tmp_path / "recoreg"; _tiles(d, 5)
    out = tmp_path / "www"
    s = cd.build_status(str(d), total=10)
    cd._write(str(out), s, None, None, None, "T")
    assert (out / "index.html").exists() and (out / "campaign_status.json").exists()
    import json
    assert json.loads((out / "campaign_status.json").read_text())["done"] == 5


def test_build_label_cross_dir_from_original(tmp_path):
    import numpy as np
    rec = tmp_path / "recoreg"; rec.mkdir()
    orig = tmp_path / "orig"; orig.mkdir()
    # _recoreg tile has NO label; the same-named original carries the unified label.
    np.savez_compressed(rec / "t.npz", spectral=np.zeros((24, 16, 16), np.float32))
    lab = np.zeros((16, 16), np.uint8); lab[:8] = 3; lab[8:] = 11   # two classes
    np.savez_compressed(orig / "t.npz", label=lab)
    out = cd.build_label(str(rec), str(orig), max_px=16)
    assert out["tile"] == "t" and isinstance(out["b64"], str) and out["b64"]
    assert {e["idx"] for e in out["legend"]} == {3, 11}
    assert all(e["name"] for e in out["legend"])              # names from real schema
    assert abs(sum(e["pct"] for e in out["legend"]) - 100.0) < 1.0


def test_build_label_missing_original_is_empty(tmp_path):
    import numpy as np
    rec = tmp_path / "recoreg"; rec.mkdir()
    np.savez_compressed(rec / "t.npz", spectral=np.zeros((24, 8, 8), np.float32))
    assert cd.build_label(str(rec), str(tmp_path / "orig_absent")) == {}


def test_unified_palette_json_matches_schema():
    """scripts/unified_palette.json must stay in sync with the canonical schema —
    the dashboard reads the JSON (no imint/ on the slim pod), so a schema change
    has to regenerate it. This test fails loudly if they drift."""
    import json
    from imint.training.unified_schema import (
        NUM_UNIFIED_CLASSES, UNIFIED_CLASS_NAMES, UNIFIED_COLOR_LIST)
    p = Path(__file__).resolve().parents[1] / "scripts" / "unified_palette.json"
    pal = json.loads(p.read_text(encoding="utf-8"))
    assert pal["num_classes"] == NUM_UNIFIED_CLASSES
    assert pal["names"] == [str(n) for n in UNIFIED_CLASS_NAMES]
    assert [tuple(c) for c in pal["colors"]] == \
        [tuple(int(x) for x in rgb) for rgb in UNIFIED_COLOR_LIST]


def test_render_html_includes_label_section(tmp_path):
    import numpy as np
    rec = tmp_path / "recoreg"; rec.mkdir(); orig = tmp_path / "orig"; orig.mkdir()
    np.savez_compressed(rec / "t.npz", spectral=np.zeros((24, 8, 8), np.float32))
    np.savez_compressed(orig / "t.npz", label=np.full((8, 8), 3, np.uint8))
    s = cd.build_status(str(rec), total=10)
    lab = cd.build_label(str(rec), str(orig), max_px=8)
    html = cd.render_html(s, label=lab, title="X")
    assert "Training data" in html and "data:image/png;base64," in html
    assert "Training data" not in cd.render_html(s, title="X")   # back-compat


def test_build_aux_renders_present_channels_with_nodata(tmp_path):
    import numpy as np
    d = tmp_path / "recoreg"; d.mkdir()
    mark = np.random.rand(16, 16).astype(np.float32)
    mark[:, :4] = np.nan                                   # markfukt SLU gap
    np.savez_compressed(
        d / "t.npz",
        spectral=np.zeros((24, 16, 16), np.float32),
        dem=np.random.rand(16, 16).astype(np.float32) * 40,
        height=np.random.rand(16, 16).astype(np.float32) * 30,
        markfukt=mark)                                     # 'volume' etc. absent
    a = cd.build_aux(str(d), max_px=16)
    names = [c["name"] for c in a["channels"]]
    assert names == ["dem", "height", "markfukt"]          # only present, in panel order
    assert all(isinstance(c["b64"], str) and c["b64"] for c in a["channels"])


def test_build_aux_empty_when_no_aux(tmp_path):
    import numpy as np
    d = tmp_path / "recoreg"; d.mkdir()
    np.savez_compressed(d / "t.npz", spectral=np.zeros((24, 8, 8), np.float32))
    assert cd.build_aux(str(d)) == {}


def test_render_html_includes_aux_section(tmp_path):
    import numpy as np
    d = tmp_path / "recoreg"; d.mkdir()
    np.savez_compressed(d / "t.npz", spectral=np.zeros((24, 8, 8), np.float32),
                        dem=np.random.rand(8, 8).astype(np.float32))
    s = cd.build_status(str(d), total=10)
    aux = cd.build_aux(str(d), max_px=8)
    html = cd.render_html(s, aux=aux, title="X")
    assert "Aux channels" in html and "DEM" in html
    assert "Aux channels" not in cd.render_html(s, title="X")   # back-compat


def test_build_frames_renders_latest_tile(tmp_path):
    import numpy as np
    d = tmp_path / "recoreg"; d.mkdir()
    now = time.time()
    # older tile (should be ignored) + newer tile (the "latest fetched").
    np.savez_compressed(d / "tile_old.npz", spectral=np.zeros((24, 16, 16), np.float32))
    os.utime(d / "tile_old.npz", (now - 100, now - 100))
    spec = np.random.rand(24, 16, 16).astype(np.float32)
    spec[0:6] = 0.0                                   # slot 0 empty (2017 dropped)
    np.savez_compressed(
        d / "tile_new.npz", spectral=spec,
        temporal_mask=np.array([0, 1, 1, 1], np.uint8),
        dates=np.array(["", "2018-06-01", "2018-07-01", "2018-08-01"]))
    os.utime(d / "tile_new.npz", (now, now))

    fr = cd.build_frames(str(d), max_px=16)
    assert fr["tile"] == "tile_new"                   # newest mtime
    assert len(fr["frames"]) == 4
    assert fr["frames"][0]["filled"] is False and fr["frames"][0]["b64"] is None
    for fi in (1, 2, 3):
        assert fr["frames"][fi]["filled"] is True
        assert isinstance(fr["frames"][fi]["b64"], str) and fr["frames"][fi]["b64"]
    assert fr["frames"][1]["date"] == "2018-06-01"


def test_build_frames_empty_dir(tmp_path):
    d = tmp_path / "recoreg"; d.mkdir()
    assert cd.build_frames(str(d)) == {}


def test_render_html_includes_frames_section(tmp_path):
    import numpy as np
    d = tmp_path / "recoreg"; d.mkdir()
    np.savez_compressed(
        d / "t.npz", spectral=np.random.rand(24, 16, 16).astype(np.float32),
        temporal_mask=np.array([1, 1, 1, 1], np.uint8),
        dates=np.array(["2021-09-01", "2022-06-01", "2022-07-01", "2022-08-01"]))
    s = cd.build_status(str(d), total=10)
    fr = cd.build_frames(str(d), max_px=16)
    html = cd.render_html(s, frames=fr, title="X")
    assert "Latest tile · RGB frames" in html
    assert "t" in html and "data:image/png;base64," in html
    # no-frames render must omit the section (back-compat)
    assert "Latest tile" not in cd.render_html(s, title="X")


def test_label_progress_counts_labelled_tiles(tmp_path):
    import numpy as np
    d = tmp_path / "recoreg"; d.mkdir()
    for i in range(3):                                       # carry a label
        np.savez_compressed(d / f"lab_{i}.npz",
                            spectral=np.zeros((24, 8, 8), np.float32),
                            label=np.zeros((8, 8), np.uint8))
    for i in range(2):                                       # label dropped by refetch
        np.savez_compressed(d / f"nolab_{i}.npz",
                            spectral=np.zeros((24, 8, 8), np.float32))
    lp = cd.build_label_progress(str(d))
    assert lp["labelled"] == 3 and lp["labelled_total"] == 5
    assert lp["labelled_pct"] == 60.0


def test_label_progress_ignores_tmp_and_corrupt(tmp_path):
    import numpy as np
    d = tmp_path / "recoreg"; d.mkdir()
    np.savez_compressed(d / "ok.npz", spectral=np.zeros((24, 8, 8), np.float32),
                        label=np.zeros((8, 8), np.uint8))
    (d / "broken.npz").write_bytes(b"")            # 0-byte → BadZipFile, uncounted
    # prefetch_aux's REAL atomic-write temp is ``<stem>_tmp.npz`` (underscore) —
    # not ``.tmp.npz``. Must be excluded entirely.
    (d / "tile_0_tmp.npz").write_bytes(b"x")
    lp = cd.build_label_progress(str(d))
    assert lp["labelled"] == 1 and lp["labelled_total"] == 2   # ok+broken; tmp excluded


def test_label_progress_cache_reuses_then_refreshes_on_mtime(tmp_path):
    import numpy as np
    d = tmp_path / "recoreg"; d.mkdir()
    p = d / "t.npz"
    np.savez_compressed(p, spectral=np.zeros((24, 8, 8), np.float32))   # no label
    cache: dict = {}
    lp1 = cd.build_label_progress(str(d), cache=cache)
    assert lp1["labelled"] == 0 and cache[str(p)][1] is False
    mt0 = cache[str(p)][0]

    # Restore a label + bump mtime (what the atomic os.replace does) → recounted.
    np.savez_compressed(p, spectral=np.zeros((24, 8, 8), np.float32),
                        label=np.zeros((8, 8), np.uint8))
    os.utime(p, (mt0 + 10, mt0 + 10))
    lp2 = cd.build_label_progress(str(d), cache=cache)
    assert lp2["labelled"] == 1 and cache[str(p)][1] is True

    # Mutate content but keep mtime → cache HIT returns the stale verdict (proves the
    # cache short-circuits the zip-open rather than re-reading every cycle).
    mt1 = cache[str(p)][0]
    np.savez_compressed(p, spectral=np.zeros((24, 8, 8), np.float32))   # label dropped
    os.utime(p, (mt1, mt1))
    lp3 = cd.build_label_progress(str(d), cache=cache)
    assert lp3["labelled"] == 1                          # stale-by-design (mtime unchanged)


def test_render_html_includes_labelled_card(tmp_path):
    d = tmp_path / "recoreg"; _tiles(d, 10)
    s = cd.build_status(str(d), total=10)
    s.update({"labelled": 7, "labelled_total": 9, "labelled_pct": 77.8})
    html = cd.render_html(s, title="X")
    assert ">Labelled<" in html and "/ 9" in html            # card rendered (≠ Done's /10)
    # back-compat: a status without label keys renders no Labelled card.
    assert ">Labelled<" not in cd.render_html(cd.build_status(str(d), total=10), title="X")


# ── Phase-2 backfill (frame_2016 + slot:0 counts + sample-verify render) ────
def _p2_tile(p, *, f2016=False, slot0=False):
    """A recoreg npz optionally carrying the Phase-2 backfill members the
    dashboard counts: ``frame_2016.npy`` (Pass A) and/or ``slot_0_scene.npy``
    (Pass B — the per-scene provenance member ``_write_temporal_slot`` adds
    alongside folding the bands into ``spectral``, so slot:0 is namelist-
    detectable without decoding the cube)."""
    import numpy as np
    kw = {"spectral": np.zeros((24, 8, 8), np.float32)}
    if f2016:
        kw["frame_2016"] = np.zeros((6, 8, 8), np.float32)
    if slot0:
        kw["slot_0_scene"] = np.array("S2A_MSIL1C_20171003")
    np.savez_compressed(p, **kw)


def test_phase2_progress_counts_frame2016_and_slot0(tmp_path):
    d = tmp_path / "recoreg"; d.mkdir()
    _p2_tile(d / "a.npz", f2016=True)                  # frame_2016 only
    _p2_tile(d / "b.npz", f2016=True, slot0=True)      # both
    _p2_tile(d / "c.npz", slot0=True)                  # slot:0 only
    _p2_tile(d / "d.npz")                              # neither (pre-backfill)
    pp = cd.build_phase2_progress(str(d))
    assert pp["phase2_frame2016"] == 2 and pp["phase2_slot0"] == 2
    assert pp["phase2_total"] == 4
    assert pp["phase2_frame2016_pct"] == 50.0          # 2/4; slot:0 has no pct (subset)


def test_phase2_progress_ignores_tmp_and_corrupt(tmp_path):
    d = tmp_path / "recoreg"; d.mkdir()
    _p2_tile(d / "ok.npz", f2016=True)
    (d / "broken.npz").write_bytes(b"")                # 0-byte → BadZipFile, uncounted
    (d / "tile_0_tmp.npz").write_bytes(b"x")           # atomic-write temp → excluded
    pp = cd.build_phase2_progress(str(d))
    assert pp["phase2_frame2016"] == 1 and pp["phase2_total"] == 2   # ok+broken; tmp out


def test_phase2_progress_cache_reuses_then_refreshes_on_mtime(tmp_path):
    d = tmp_path / "recoreg"; d.mkdir(); p = d / "t.npz"
    _p2_tile(p)                                        # neither member yet
    cache: dict = {}
    pp1 = cd.build_phase2_progress(str(d), cache=cache)
    assert pp1["phase2_frame2016"] == 0 and cache[str(p)][1] == (False, False)
    mt0 = cache[str(p)][0]

    # Pass A re-saves the npz with frame_2016 + bumps mtime (atomic replace) → recounted.
    _p2_tile(p, f2016=True)
    os.utime(p, (mt0 + 10, mt0 + 10))
    pp2 = cd.build_phase2_progress(str(d), cache=cache)
    assert pp2["phase2_frame2016"] == 1 and cache[str(p)][1] == (True, False)

    # Add slot:0 but keep mtime → cache HIT returns the stale (True, False) tuple
    # (proves the cache short-circuits the zip-open rather than re-reading each cycle).
    mt1 = cache[str(p)][0]
    _p2_tile(p, f2016=True, slot0=True)
    os.utime(p, (mt1, mt1))
    pp3 = cd.build_phase2_progress(str(d), cache=cache)
    assert pp3["phase2_slot0"] == 0                    # stale-by-design (mtime unchanged)


def test_render_html_includes_phase2_section(tmp_path):
    d = tmp_path / "recoreg"; _tiles(d, 10)
    s = cd.build_status(str(d), total=10)
    s.update({"phase2_frame2016": 4500, "phase2_slot0": 120,
              "phase2_total": 6834, "phase2_frame2016_pct": 65.8})
    html = cd.render_html(s, title="X")
    # Section heading + live fetch sub-bar (the "follow the fetch" pattern).
    assert "sen2cor pre-2018 backfill" in html
    assert "65.8%" in html and "4,500 / 6,834 tiles" in html
    assert "120 filled" in html                        # slot:0 raw counter (no pct)
    # Compact verdict — mint badge + one-liner, not a full callout / cards grid.
    assert 'class="verdict-badge">VERIFIED' in html    # the mint pill itself
    assert "reverse-fit&nbsp;72/72" in html and "0.000&nbsp;px" in html
    assert "HEAD&nbsp;7014c95" in html                 # provenance kept
    # Negative: the previous full-callout + 3-card form must NOT appear.
    assert "Sample-verify PASS" not in html and ">Reverse-fit<" not in html
    # back-compat: a status without phase2 keys renders no Phase-2 section.
    assert "sen2cor pre-2018 backfill" not in cd.render_html(
        cd.build_status(str(d), total=10), title="X")


# ── aux-alignment φ (corr volume↔height) ──────────────────────────────────
def test_aux_corr_aligned_vs_random_vs_too_few(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    h = rng.random((16, 16)) * 30 + 1                      # 256 px, all > 0
    assert cd._aux_corr(h * 2.5, h) > 0.99                 # perfectly linear → φ≈1
    assert abs(cd._aux_corr(rng.random((16, 16)) + 1, h)) < 0.3   # independent → ~0
    # <200 joint-valid px (most masked to 0) → NaN, not a bogus φ.
    a = np.zeros((16, 16)); b = np.zeros((16, 16))
    a[:1] = h[:1] * 2.5; b[:1] = h[:1]                     # 16 valid px < 200
    assert cd._aux_corr(a, b) != cd._aux_corr(a, b)        # NaN


def _aux_tile(p, vol, hgt):
    import numpy as np
    np.savez_compressed(p, spectral=np.zeros((24, 16, 16), np.float32),
                        volume=vol.astype(np.float32), height=hgt.astype(np.float32))


def test_build_aux_alignment_high_for_corrected_grid(tmp_path):
    import numpy as np
    d = tmp_path / "recoreg"; d.mkdir()
    rng = np.random.default_rng(1)
    for i in range(4):
        h = rng.random((16, 16)) * 30 + 1
        _aux_tile(d / f"t{i}.npz", h * 2.5 + rng.standard_normal((16, 16)) * 0.4, h)
    a = cd.build_aux_alignment(str(d))
    assert a["align_n"] == 4 and a["align_phi_mean"] > 0.8
    assert a["align_frac_ok"] == 1.0                       # all ≥ 0.3


def test_build_aux_alignment_low_for_wrong_grid(tmp_path):
    import numpy as np
    d = tmp_path / "recoreg"; d.mkdir()
    rng = np.random.default_rng(2)
    for i in range(4):                                     # volume independent of height
        _aux_tile(d / f"t{i}.npz", rng.random((16, 16)) * 100 + 1,
                  rng.random((16, 16)) * 30 + 1)
    a = cd.build_aux_alignment(str(d))
    assert a["align_n"] == 4 and a["align_phi_mean"] < 0.3
    assert a["align_frac_ok"] == 0.0


def test_build_aux_alignment_empty_when_no_volume_or_height(tmp_path):
    import numpy as np
    d = tmp_path / "recoreg"; d.mkdir()
    np.savez_compressed(d / "t.npz", spectral=np.zeros((24, 8, 8), np.float32),
                        height=np.ones((8, 8), np.float32))   # volume absent
    a = cd.build_aux_alignment(str(d))
    assert a["align_n"] == 0 and a["align_phi_mean"] is None


def test_build_aux_alignment_cache_refreshes_on_mtime(tmp_path):
    import numpy as np
    d = tmp_path / "recoreg"; d.mkdir(); p = d / "t.npz"
    rng = np.random.default_rng(3)
    h = rng.random((16, 16)) * 30 + 1
    _aux_tile(p, h * 2.5, h)                               # aligned → φ high
    cache: dict = {}
    a1 = cd.build_aux_alignment(str(d), cache=cache)
    assert a1["align_phi_mean"] > 0.9 and cache[str(p)][1] > 0.9
    mt0 = cache[str(p)][0]

    # Refetch overwrites with wrong-grid data + bumps mtime (atomic rename) → recomputed.
    _aux_tile(p, rng.random((16, 16)) * 100 + 1, h)
    os.utime(p, (mt0 + 10, mt0 + 10))
    a2 = cd.build_aux_alignment(str(d), cache=cache)
    assert a2["align_phi_mean"] < 0.3

    # Mutate content but keep mtime → cache HIT returns the stale φ (proves short-circuit).
    mt1 = cache[str(p)][0]
    _aux_tile(p, h * 2.5, h)                               # aligned again, but...
    os.utime(p, (mt1, mt1))                                # ...mtime unchanged
    a3 = cd.build_aux_alignment(str(d), cache=cache)
    assert a3["align_phi_mean"] < 0.3                      # stale-by-design


def test_render_html_includes_align_card(tmp_path):
    d = tmp_path / "recoreg"; _tiles(d, 10)
    s = cd.build_status(str(d), total=10)
    s.update({"align_phi_mean": 0.74, "align_n": 24, "align_frac_ok": 0.96})
    html = cd.render_html(s, title="X")
    assert "Aux align" in html and "0.74" in html
    assert "Aux align" not in cd.render_html(cd.build_status(str(d), total=10), title="X")


# ── refetch progress bar ──────────────────────────────────────────────────
def test_build_refetch_progress_counts_by_mtime(tmp_path):
    d = tmp_path / "recoreg"
    now = time.time()
    # 4 tiles rewritten "now", 6 left untouched from before the refetch started.
    _tiles(d, 4, mtimes=[now, now, now, now])
    for i in range(6):
        p = d / f"old_{i}.npz"; p.touch(); os.utime(p, (now - 9999, now - 9999))
    r = cd.build_refetch_progress(str(d), since_epoch=now - 100, total=20)
    assert r["refetch_done"] == 4 and r["refetch_total"] == 20
    assert r["refetch_pct"] == 20.0 and r["refetch_since_utc"].endswith("+00:00")


def test_build_refetch_progress_total_falls_back_to_count(tmp_path):
    d = tmp_path / "recoreg"
    now = time.time()
    _tiles(d, 5, mtimes=[now] * 5)
    r = cd.build_refetch_progress(str(d), since_epoch=now - 100, total=0)
    assert r["refetch_total"] == 5 and r["refetch_pct"] == 100.0


def test_render_html_includes_refetch_bar(tmp_path):
    d = tmp_path / "recoreg"; _tiles(d, 10)
    s = cd.build_status(str(d), total=10)
    s.update({"refetch_done": 3, "refetch_total": 10, "refetch_pct": 30.0,
              "refetch_since_utc": "2026-06-25T09:47:30+00:00"})
    html = cd.render_html(s, title="X")
    assert "aux refetch" in html and "3 / 10 rewritten" in html and "09:47" in html
    assert "aux refetch" not in cd.render_html(cd.build_status(str(d), total=10), title="X")


def test_parse_since_accepts_epoch_and_iso():
    assert cd._parse_since("1000.0") == 1000.0
    from datetime import datetime, timezone
    want = datetime(2026, 6, 25, 9, 47, 30, tzinfo=timezone.utc).timestamp()
    assert abs(cd._parse_since("2026-06-25T09:47:30Z") - want) < 1e-6


# ── concurrent-writer race hardening (the live free-aux crash) ─────────────
# prefetch_aux writes ``<stem>_tmp.npz`` then os.replace → ``<stem>.npz``. The
# dashboard globs the same dir every 60 s while free-aux writes, so it must
# (a) never COUNT a temp as a done tile and (b) never CRASH when a globbed path
# vanishes (the os.replace) before getmtime/load. The prior ``'.tmp' not in
# name`` filter did neither for the real ``_tmp.npz`` name; _scan_mtimes had no
# guard at all and killed the whole regen cycle → /www/campaign_status.json was
# never written → dashboard served nothing.
def test_tile_npz_paths_excludes_real_atomic_tmp(tmp_path):
    d = tmp_path / "recoreg"; d.mkdir()
    (d / "tile_1.npz").touch()
    (d / "tile_2.npz").touch()
    (d / "tile_2_tmp.npz").touch()                 # in-flight atomic temp
    got = {os.path.basename(p) for p in cd._tile_npz_paths(str(d))}
    assert got == {"tile_1.npz", "tile_2.npz"}     # temp excluded


def test_scan_mtimes_excludes_tmp_from_count(tmp_path):
    d = tmp_path / "recoreg"; d.mkdir()
    for i in range(5):
        (d / f"tile_{i}.npz").touch()
    (d / "tile_5_tmp.npz").touch()                 # must not inflate the count
    assert len(cd._scan_mtimes(str(d))) == 5
    assert cd.build_status(str(d), total=10)["done"] == 5   # headline progress too


def test_scan_mtimes_survives_file_vanishing_mid_scan(tmp_path, monkeypatch):
    """The exact live crash: glob lists a path, the writer's os.replace removes
    it before getmtime → FileNotFoundError. Must skip it, not raise."""
    d = tmp_path / "recoreg"; d.mkdir()
    real = d / "tile_real.npz"; real.touch()
    ghost = str(d / "tile_ghost.npz")              # listed but never on disk
    monkeypatch.setattr(cd.glob, "glob", lambda *a, **k: [str(real), ghost])
    mt = cd._scan_mtimes(str(d))                   # must not raise
    assert len(mt) == 1                            # ghost skipped, real kept


def test_build_refetch_progress_ignores_tmp(tmp_path):
    """The refetch bar counts via _scan_mtimes, so temp-exclusion must hold there
    too — else it over-reports rewrites for every in-flight atomic temp."""
    d = tmp_path / "recoreg"
    now = time.time()
    _tiles(d, 3, mtimes=[now, now, now])
    p = d / "tile_3_tmp.npz"; p.touch(); os.utime(p, (now, now))
    r = cd.build_refetch_progress(str(d), since_epoch=now - 100, total=10)
    assert r["refetch_done"] == 3                  # temp not counted as rewritten


def test_latest_npz_excludes_tmp_and_survives_vanish(tmp_path, monkeypatch):
    d = tmp_path / "recoreg"; d.mkdir()
    now = time.time()
    old = d / "tile_old.npz"; old.touch(); os.utime(old, (now - 100, now - 100))
    new = d / "tile_new.npz"; new.touch(); os.utime(new, (now, now))
    tmp = d / "tile_new_tmp.npz"; tmp.touch(); os.utime(tmp, (now + 50, now + 50))
    # newest mtime is the TEMP — must be ignored, so latest = tile_new.
    assert os.path.basename(cd._latest_npz(str(d))) == "tile_new.npz"
    ghost = str(d / "tile_ghost.npz")
    monkeypatch.setattr(cd.glob, "glob", lambda *a, **k: [str(new), ghost])
    assert os.path.basename(cd._latest_npz(str(d))) == "tile_new.npz"   # vanish-safe


def test_latest_pinned_across_panels(tmp_path):
    """Cross-panel race fix: main() resolves the latest tile ONCE and passes it to
    build_frames/build_aux/build_label so all three render the SAME tile. tile_B is
    newest (what _latest_npz returns), but pinning ``latest`` to the OLDER tile_A
    must make every panel report tile_A — proving the pin overrides each builder's
    own _latest_npz call. With latest=None each still self-resolves to B (back-compat).
    """
    import numpy as np
    rec = tmp_path / "recoreg"; rec.mkdir()
    orig = tmp_path / "orig"; orig.mkdir()
    now = time.time()

    def _tile(stem: str, mt: float) -> None:
        np.savez_compressed(
            rec / f"{stem}.npz",
            spectral=np.random.rand(24, 16, 16).astype(np.float32),
            temporal_mask=np.array([1, 1, 1, 1], np.uint8),
            dates=np.array(["2021-09-01", "2022-06-01", "2022-07-01", "2022-08-01"]),
            dem=np.random.rand(16, 16).astype(np.float32) * 40)
        os.utime(rec / f"{stem}.npz", (mt, mt))
        np.savez_compressed(orig / f"{stem}.npz", label=np.full((16, 16), 3, np.uint8))

    _tile("tile_A", now - 100)        # older
    _tile("tile_B", now)              # newer → _latest_npz picks this
    assert os.path.basename(cd._latest_npz(str(rec))) == "tile_B.npz"

    # Pin to the OLDER tile; every panel must follow the pin, not re-resolve to B.
    pin = str(rec / "tile_A.npz")
    assert cd.build_frames(str(rec), latest=pin, max_px=16)["tile"] == "tile_A"
    assert cd.build_aux(str(rec), latest=pin, max_px=16)["tile"] == "tile_A"
    assert cd.build_label(str(rec), str(orig), latest=pin, max_px=16)["tile"] == "tile_A"

    # latest=None still self-resolves to the newest tile_B (back-compat).
    assert cd.build_frames(str(rec), max_px=16)["tile"] == "tile_B"
    assert cd.build_aux(str(rec), max_px=16)["tile"] == "tile_B"
    assert cd.build_label(str(rec), str(orig), max_px=16)["tile"] == "tile_B"
