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
    cd._write(str(out), s, "T")
    assert (out / "index.html").exists() and (out / "campaign_status.json").exists()
    import json
    assert json.loads((out / "campaign_status.json").read_text())["done"] == 5
