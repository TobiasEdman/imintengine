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
