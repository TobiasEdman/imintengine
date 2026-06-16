#!/usr/bin/env python3
"""Re-coreg campaign progress dashboard — Phase-1/2 tile completion.

Self-contained + RBAC-free: progress is derived purely from counting the
``*.npz`` files the campaign writes into ``unified_v2_512_recoreg`` on the PVC,
so the dashboard needs no kubectl/cluster access — just the data dir mounted.

``build_status`` counts done tiles vs the expected total and derives rate + ETA
from the ``.npz`` mtimes (a recent-window rate for a responsive ETA, plus the
since-first-tile average). ``render_html`` emits a single DES-styled page
(``docs/css/styles.css`` palette: white bg, forest green #1A4338/#245045, mint
#cff8e4 accent, Space Grotesk) that meta-refreshes.

Usage:
  # one-shot (local / test): write index.html + campaign_status.json once
  python scripts/campaign_dashboard.py \\
      --recoreg-dir /data/unified_v2_512_recoreg --total 6921 --out-dir /tmp/dash
  # served loop (in-pod): regenerate every 60 s
  python scripts/campaign_dashboard.py \\
      --recoreg-dir /data/unified_v2_512_recoreg --total 6921 \\
      --out-dir /www --watch 60
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime, timezone

_RECENT_WINDOW_MIN = 30          # rate window for the responsive ETA
_REFRESH_S = 60                  # page meta-refresh cadence


def _scan_mtimes(recoreg_dir: str) -> list[float]:
    """Epoch mtimes of every ``*.npz`` in the dir (the done tiles)."""
    return sorted(
        os.path.getmtime(p) for p in glob.glob(os.path.join(recoreg_dir, "*.npz"))
    )


def _fmt_eta(hours: float | None) -> str:
    if hours is None:
        return "—"
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} d"


def build_status(recoreg_dir: str, total: int, *, now: float | None = None) -> dict:
    """Progress snapshot from the on-disk tile count + mtimes.

    ``rate_recent`` (tiles/h over the last ``_RECENT_WINDOW_MIN``) drives the ETA
    when non-zero — it tracks the current throughput rather than being dragged
    down by a slow start; ``rate_overall`` (since the first tile) is the stable
    fallback. ETA is ``None`` until at least one tile lands and a rate exists.
    """
    now = time.time() if now is None else now
    mtimes = _scan_mtimes(recoreg_dir)
    done = len(mtimes)
    pct = round(100.0 * done / total, 2) if total else 0.0

    rate_overall = 0.0
    if done >= 1:
        elapsed_h = (now - mtimes[0]) / 3600.0
        # Require ≥3 min of history before reporting an average — otherwise a
        # just-started run (first tile seconds ago) divides by ~0 and the rate
        # explodes. The recent-window rate covers the early period instead.
        if elapsed_h >= 0.05:
            rate_overall = done / elapsed_h
    cutoff = now - _RECENT_WINDOW_MIN * 60
    n_recent = sum(1 for t in mtimes if t >= cutoff)
    rate_recent = n_recent / (_RECENT_WINDOW_MIN / 60.0)

    rate = rate_recent if rate_recent > 0 else rate_overall
    remaining = max(0, total - done)
    eta_h = (remaining / rate) if rate > 0 and remaining > 0 else (
        0.0 if remaining == 0 and done > 0 else None)

    return {
        "exists": os.path.isdir(recoreg_dir),
        "done": done,
        "total": total,
        "pct": pct,
        "remaining": remaining,
        "rate_recent_per_h": round(rate_recent, 1),
        "rate_overall_per_h": round(rate_overall, 1),
        "eta_hours": None if eta_h is None else round(eta_h, 2),
        "first_tile_utc": (datetime.fromtimestamp(mtimes[0], timezone.utc).isoformat()
                           if done else None),
        "last_tile_utc": (datetime.fromtimestamp(mtimes[-1], timezone.utc).isoformat()
                          if done else None),
        "updated_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
    }


def build_frames(recoreg_dir: str, *, max_px: int = 256, n_frames: int = 4) -> dict:
    """Render the latest-fetched tile's temporal frames as true-colour RGB.

    Picks the most-recently-written ``*.npz`` and renders each filled temporal
    slot (B04/B03/B02, 2-98% stretch via ``tile_rgb``) to a base64 PNG, downscaled
    to ``max_px``. Best-effort: returns ``{}`` if numpy/PIL/tile_rgb are missing or
    the npz can't be read — the progress view never depends on this.
    """
    try:
        import numpy as np
        from PIL import Image
        from tile_rgb import frame_rgb, png_b64
    except Exception:
        return {}
    npzs = [p for p in glob.glob(os.path.join(recoreg_dir, "*.npz"))
            if ".tmp" not in os.path.basename(p)]
    if not npzs:
        return {}
    latest = max(npzs, key=os.path.getmtime)
    try:
        with np.load(latest, allow_pickle=True) as d:
            spec = d.get("spectral")
            if spec is None:
                return {}
            spec = np.asarray(spec, np.float32)
            mask = list(d.get("temporal_mask")) if d.get("temporal_mask") is not None else []
            dates = [str(x)[:10] for x in (d.get("dates") if d.get("dates") is not None else [])]
    except Exception:
        return {}
    frames = []
    for fi in range(n_frames):
        filled = (fi < len(mask) and int(mask[fi]) == 1
                  and spec.shape[0] >= (fi + 1) * 6
                  and bool(np.any(spec[fi * 6:(fi + 1) * 6])))
        b64 = None
        if filled:
            try:
                rgb = frame_rgb(spec, fi)
                if max_px and rgb.shape[0] > max_px:
                    rgb = np.asarray(Image.fromarray(rgb).resize((max_px, max_px)))
                b64 = png_b64(rgb)
            except Exception:
                b64 = None
        frames.append({"fi": fi, "date": dates[fi] if fi < len(dates) else "",
                       "filled": bool(filled), "b64": b64})
    return {
        "tile": os.path.basename(latest)[:-4],
        "frames": frames,
        "mtime_utc": datetime.fromtimestamp(
            os.path.getmtime(latest), timezone.utc).isoformat(),
    }


def render_html(status: dict, *, frames: dict | None = None,
                title: str = "Re-coreg campaign") -> str:
    """A single DES-styled page (palette per docs/css/styles.css) with a
    progress bar, stat cards, and a meta-refresh. Self-contained — no JS deps."""
    pct = status["pct"]
    done, total = status["done"], status["total"]
    eta = _fmt_eta(status["eta_hours"])
    bar_pct = min(100.0, pct)
    stale = not status["exists"]
    banner = ("" if status["exists"] else
              '<div class="warn">Output dir not present yet — '
              'Phase&nbsp;1 may still be installing deps / cloning.</div>')

    frames_html = ""
    if frames and frames.get("frames"):
        cells = []
        for fr in frames["frames"]:
            cap = f"frame {fr['fi']}" + (f" · {fr['date']}" if fr["date"] else "")
            if fr["b64"]:
                cells.append(
                    f'<figure class="frame">'
                    f'<img src="data:image/png;base64,{fr["b64"]}" alt="{cap}">'
                    f'<figcaption class="cap">{cap}</figcaption></figure>')
            else:
                why = "empty&nbsp;·&nbsp;Phase&nbsp;2" if not fr["filled"] else "no preview"
                cells.append(f'<div class="frame empty">{cap}<br>{why}</div>')
        frames_html = (
            '<h2>Latest tile · RGB frames</h2>'
            f'<div class="tilename">{frames["tile"]} · written {frames["mtime_utc"]}</div>'
            f'<div class="frames">{"".join(cells)}</div>')

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{_REFRESH_S}">
<title>IMINT — {title}</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#fff;color:#171717;font-family:'Space Grotesk',system-ui,sans-serif;
    line-height:1.5;padding:0 0 48px}}
  .header{{background:#fff;border-bottom:1px solid #e5e7eb;padding:18px 32px;
    display:flex;align-items:center;gap:14px}}
  .header .logo{{font-weight:700;font-size:20px;letter-spacing:.04em;color:#1A4338}}
  .header .logo span{{color:#245045}}
  .header .sub{{color:#6b7280;font-size:13px;margin-left:auto}}
  .wrap{{max-width:880px;margin:34px auto;padding:0 24px}}
  h1{{font-size:24px;font-weight:700;color:#1A4338;margin-bottom:6px}}
  .byline{{color:#6b7280;font-size:14px;margin-bottom:28px}}
  .bar-wrap{{background:#e5e7eb;border-radius:10px;height:30px;overflow:hidden;
    margin:8px 0 6px}}
  .bar{{background:linear-gradient(90deg,#1A4338,#245045);height:100%;
    width:{bar_pct}%;border-radius:10px;transition:width .6s;min-width:4px}}
  .pct-row{{display:flex;justify-content:space-between;color:#6b7280;font-size:13px;
    margin-bottom:30px}}
  .pct-row strong{{color:#1A4338;font-weight:700}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:16px;margin-bottom:28px}}
  .card{{border:1px solid #e5e7eb;border-radius:12px;padding:18px 20px}}
  .card .label{{color:#6b7280;font-size:12px;text-transform:uppercase;
    letter-spacing:.05em;margin-bottom:6px}}
  .card .value{{font-size:26px;font-weight:700;color:#1A4338}}
  .card .unit{{font-size:13px;color:#6b7280;font-weight:400}}
  .warn{{background:#cff8e4;border:1px solid #245045;color:#1A4338;
    border-radius:10px;padding:12px 16px;margin-bottom:22px;font-size:14px}}
  h2{{font-size:16px;font-weight:700;color:#1A4338;margin:34px 0 4px}}
  .tilename{{color:#6b7280;font-size:13px;margin-bottom:12px;
    font-family:ui-monospace,Menlo,monospace}}
  .frames{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
  .frame{{border:1px solid #e5e7eb;border-radius:10px;overflow:hidden}}
  .frame img{{width:100%;display:block;aspect-ratio:1/1;object-fit:cover;
    background:#f3f4f6}}
  .frame .cap{{font-size:12px;color:#6b7280;padding:6px 9px;
    border-top:1px solid #e5e7eb}}
  .frame.empty{{display:flex;align-items:center;justify-content:center;
    min-height:130px;color:#6b7280;font-size:12px;background:#f9fafb;
    text-align:center;padding:10px;line-height:1.7}}
  @media(max-width:640px){{.frames{{grid-template-columns:repeat(2,1fr)}}}}
  .foot{{color:#6b7280;font-size:12px;margin-top:24px;border-top:1px solid #e5e7eb;
    padding-top:14px}}
</style>
</head>
<body>
  <div class="header">
    <span class="logo">IM<span>INT</span></span>
    <span class="sub">re-coreg campaign · refreshes every {_REFRESH_S}s</span>
  </div>
  <div class="wrap">
    <h1>{title}</h1>
    <div class="byline">Phase&nbsp;1 des-only re-coregistration → <code>unified_v2_512_recoreg</code></div>
    {banner}
    <div class="bar-wrap"><div class="bar"></div></div>
    <div class="pct-row"><span><strong>{pct:.1f}%</strong> · {done:,} / {total:,} tiles</span><span>{status['remaining']:,} remaining</span></div>
    <div class="cards">
      <div class="card"><div class="label">Done</div>
        <div class="value">{done:,}<span class="unit"> / {total:,}</span></div></div>
      <div class="card"><div class="label">Rate (30&nbsp;min)</div>
        <div class="value">{status['rate_recent_per_h']:.0f}<span class="unit"> tiles/h</span></div></div>
      <div class="card"><div class="label">Rate (avg)</div>
        <div class="value">{status['rate_overall_per_h']:.0f}<span class="unit"> tiles/h</span></div></div>
      <div class="card"><div class="label">ETA</div>
        <div class="value">{eta}</div></div>
    </div>
    {frames_html}
    <div class="foot">
      first tile: {status['first_tile_utc'] or '—'} ·
      last tile: {status['last_tile_utc'] or '—'} ·
      updated: {status['updated_utc']}
    </div>
  </div>
</body>
</html>"""


def _write(out_dir: str, status: dict, frames: dict | None = None,
           title: str = "Re-coreg campaign") -> None:
    os.makedirs(out_dir, exist_ok=True)
    # JSON mirrors the page minus the base64 frame blobs (keep it small + linkable).
    status_out = dict(status)
    if frames and frames.get("frames"):
        status_out["latest_tile"] = {
            "tile": frames["tile"],
            "mtime_utc": frames["mtime_utc"],
            "frames": [{k: v for k, v in fr.items() if k != "b64"}
                       for fr in frames["frames"]],
        }
    tmp_j = os.path.join(out_dir, "campaign_status.json.tmp")
    with open(tmp_j, "w") as f:
        json.dump(status_out, f, indent=2)
    os.replace(tmp_j, os.path.join(out_dir, "campaign_status.json"))
    tmp_h = os.path.join(out_dir, "index.html.tmp")
    with open(tmp_h, "w") as f:
        f.write(render_html(status, frames=frames, title=title))
    os.replace(tmp_h, os.path.join(out_dir, "index.html"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recoreg-dir", default="/data/unified_v2_512_recoreg")
    p.add_argument("--total", type=int, default=6921,
                   help="Expected tile count (default 6921 = unified_v2_512).")
    p.add_argument("--out-dir", default="/www")
    p.add_argument("--title", default="Re-coreg campaign · Phase 1")
    p.add_argument("--watch", type=int, default=0,
                   help="Regenerate every N seconds (0 = once and exit).")
    args = p.parse_args()

    while True:
        status = build_status(args.recoreg_dir, args.total)
        frames = build_frames(args.recoreg_dir)
        _write(args.out_dir, status, frames, args.title)
        latest = frames.get("tile", "—") if frames else "—"
        print(f"[campaign-dashboard] done={status['done']}/{status['total']} "
              f"({status['pct']}%) rate={status['rate_recent_per_h']}/h "
              f"eta={_fmt_eta(status['eta_hours'])} latest={latest}", flush=True)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
