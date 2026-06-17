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


def _unified_palette():
    """``(color_list, class_names, n_classes)`` from the canonical unified_schema,
    loaded by FILE PATH so we never import the ``imint`` package (its ``__init__``
    pulls torch). The schema module itself is numpy-only. Returns ``None`` if the
    file isn't checked out — the label panel then degrades to nothing."""
    try:
        import importlib.util
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo, "imint", "training", "unified_schema.py")
        spec = importlib.util.spec_from_file_location("_unified_schema", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return (list(mod.UNIFIED_COLOR_LIST), list(mod.UNIFIED_CLASS_NAMES),
                int(mod.NUM_UNIFIED_CLASSES))
    except Exception:
        return None


def _latest_npz(recoreg_dir: str) -> str | None:
    """Path of the most-recently-written ``*.npz`` (ignoring atomic-write tmps)."""
    npzs = [p for p in glob.glob(os.path.join(recoreg_dir, "*.npz"))
            if ".tmp" not in os.path.basename(p)]
    return max(npzs, key=os.path.getmtime) if npzs else None


# (npz key, tile_rgb colormap, label) — continuous aux rasters, mirrors
# render_tile_inspection_dashboard's AUX_PANELS.
AUX_PANELS = [
    ("dem", "terrain", "DEM"),
    ("height", "viridis", "tree height"),
    ("volume", "viridis", "volume"),
    ("basal_area", "viridis", "basal area"),
    ("diameter", "viridis", "diameter"),
    ("markfukt", "Blues", "soil moisture"),
    ("vpp_sosd", "RdYlGn_r", "VPP start"),
    ("vpp_eosd", "RdYlGn", "VPP end"),
    ("vpp_length", "magma", "VPP length"),
    ("vpp_maxv", "viridis", "VPP max"),
    ("vpp_minv", "viridis", "VPP min"),
]


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
    latest = _latest_npz(recoreg_dir)
    if latest is None:
        return {}
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


def build_aux(recoreg_dir: str, *, max_px: int = 200) -> dict:
    """Render the latest tile's continuous aux channels as colormapped PNGs.

    Each present ``AUX_PANELS`` raster is 2-98% normalised + colormapped (nodata →
    gray) via ``tile_rgb.aux_rgb``. Best-effort: ``{}`` on missing deps / unreadable
    npz / no aux present.
    """
    try:
        import numpy as np
        from PIL import Image
        from tile_rgb import aux_rgb, png_b64
    except Exception:
        return {}
    latest = _latest_npz(recoreg_dir)
    if latest is None:
        return {}
    try:
        d = np.load(latest, allow_pickle=True)
        files = set(d.files)
    except Exception:
        return {}
    channels = []
    for name, cmap, label in AUX_PANELS:
        if name not in files:
            continue
        try:
            arr = np.asarray(d[name], np.float32)
            if arr.ndim != 2 or not np.isfinite(arr).any():
                continue
            rgb = aux_rgb(arr, cmap)
            if max_px and rgb.shape[0] > max_px:
                rgb = np.asarray(Image.fromarray(rgb).resize((max_px, max_px)))
            channels.append({"name": name, "label": label, "b64": png_b64(rgb)})
        except Exception:
            continue
    if not channels:
        return {}
    return {"tile": os.path.basename(latest)[:-4], "channels": channels}


def build_label(recoreg_dir: str, orig_dir: str, *, max_px: int = 256) -> dict:
    """Render the latest tile's unified 23-class label, loaded CROSS-DIR from the
    original dataset.

    ``refetch`` drops ``label`` from the ``_recoreg`` tiles (re-coreg may shift the
    grid, so labels are re-derived later), but the tile bbox/centre is unchanged so
    the original ``unified_v2_512`` label still aligns. Colormapped via the schema's
    ``UNIFIED_COLORS`` (nearest-neighbour — categorical), with a per-class legend.
    Best-effort → ``{}``.
    """
    try:
        import numpy as np
        from PIL import Image
        from tile_rgb import label_rgb, png_b64
    except Exception:
        return {}
    pal = _unified_palette()
    if pal is None:
        return {}
    colors, names, nclasses = pal
    latest = _latest_npz(recoreg_dir)
    if latest is None:
        return {}
    fname = os.path.basename(latest)
    orig = os.path.join(orig_dir, fname)
    if not os.path.exists(orig):
        return {}
    try:
        with np.load(orig, allow_pickle=True) as d:
            if "label" not in d.files:
                return {}
            lab = np.asarray(d["label"]).astype(np.int64)
    except Exception:
        return {}
    if lab.ndim != 2:
        return {}
    rgb = label_rgb(lab, colors)
    if max_px and rgb.shape[0] > max_px:
        rgb = np.asarray(Image.fromarray(rgb).resize((max_px, max_px), Image.NEAREST))
    classes, counts = np.unique(lab, return_counts=True)
    total = float(lab.size)
    legend = sorted(
        ({"idx": int(c), "name": (names[int(c)] if int(c) < len(names) else str(int(c))),
          "rgb": list(colors[int(c)]), "pct": round(100.0 * int(n) / total, 1)}
         for c, n in zip(classes, counts) if 0 <= int(c) < nclasses),
        key=lambda e: -e["pct"])
    return {"tile": fname[:-4], "b64": png_b64(rgb), "legend": legend}


def render_html(status: dict, *, frames: dict | None = None,
                aux: dict | None = None, label: dict | None = None,
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

    aux_html = ""
    if aux and aux.get("channels"):
        cells = [
            f'<figure><img src="data:image/png;base64,{ch["b64"]}" alt="{ch["label"]}">'
            f'<figcaption class="cap">{ch["label"]}</figcaption></figure>'
            for ch in aux["channels"]]
        aux_html = ('<h2>Aux channels</h2>'
                    f'<div class="tilename">{aux["tile"]} · '
                    f'{len(aux["channels"])} channels</div>'
                    f'<div class="aux">{"".join(cells)}</div>')

    label_html = ""
    if label and label.get("b64"):
        rows = []
        for e in label["legend"]:
            r, g, b = e["rgb"]
            rows.append(
                f'<div><span class="sw" style="background:rgb({r},{g},{b})"></span>'
                f'{e["name"]}</div><div class="pct">{e["pct"]}%</div>')
        label_html = (
            '<h2>Training data · label (23-class)</h2>'
            f'<div class="tilename">{label["tile"]} · unified label from original '
            'unified_v2_512</div>'
            '<div class="label-wrap">'
            f'<img src="data:image/png;base64,{label["b64"]}" alt="unified label">'
            f'<div class="legend">{"".join(rows)}</div></div>')

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
  .aux{{display:grid;grid-template-columns:repeat(auto-fill,minmax(132px,1fr));
    gap:12px}}
  .aux figure{{border:1px solid #e5e7eb;border-radius:10px;overflow:hidden}}
  .aux img{{width:100%;display:block;aspect-ratio:1/1;image-rendering:pixelated;
    background:#f3f4f6}}
  .aux .cap{{font-size:11px;color:#6b7280;padding:5px 8px;
    border-top:1px solid #e5e7eb}}
  .label-wrap{{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}}
  .label-wrap img{{width:256px;max-width:100%;border:1px solid #e5e7eb;
    border-radius:10px;image-rendering:pixelated}}
  .legend{{display:grid;grid-template-columns:auto auto;gap:3px 14px;
    font-size:12px;color:#171717;align-content:start}}
  .legend .sw{{display:inline-block;width:12px;height:12px;border-radius:3px;
    margin-right:6px;vertical-align:-1px;border:1px solid #0000001f}}
  .legend .pct{{color:#6b7280;text-align:right}}
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
    {label_html}
    {aux_html}
    <div class="foot">
      first tile: {status['first_tile_utc'] or '—'} ·
      last tile: {status['last_tile_utc'] or '—'} ·
      updated: {status['updated_utc']}
    </div>
  </div>
</body>
</html>"""


def _write(out_dir: str, status: dict, frames: dict | None = None,
           aux: dict | None = None, label: dict | None = None,
           title: str = "Re-coreg campaign") -> None:
    os.makedirs(out_dir, exist_ok=True)
    # JSON mirrors the page minus the base64 blobs (keep it small + linkable).
    status_out = dict(status)
    if frames and frames.get("frames"):
        status_out["latest_tile"] = {
            "tile": frames["tile"],
            "mtime_utc": frames["mtime_utc"],
            "frames": [{k: v for k, v in fr.items() if k != "b64"}
                       for fr in frames["frames"]],
        }
    if aux and aux.get("channels"):
        status_out["latest_aux"] = [ch["name"] for ch in aux["channels"]]
    if label and label.get("legend"):
        status_out["latest_label"] = {
            "tile": label["tile"],
            "classes": [{"name": e["name"], "pct": e["pct"]} for e in label["legend"]],
        }
    tmp_j = os.path.join(out_dir, "campaign_status.json.tmp")
    with open(tmp_j, "w") as f:
        json.dump(status_out, f, indent=2)
    os.replace(tmp_j, os.path.join(out_dir, "campaign_status.json"))
    tmp_h = os.path.join(out_dir, "index.html.tmp")
    with open(tmp_h, "w") as f:
        f.write(render_html(status, frames=frames, aux=aux, label=label, title=title))
    os.replace(tmp_h, os.path.join(out_dir, "index.html"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recoreg-dir", default="/data/unified_v2_512_recoreg")
    p.add_argument("--orig-dir", default="/data/unified_v2_512",
                   help="Original dataset — source of the unified label (refetch "
                        "drops it from _recoreg; read cross-dir for the same tile).")
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
        aux = build_aux(args.recoreg_dir)
        label = build_label(args.recoreg_dir, args.orig_dir)
        _write(args.out_dir, status, frames, aux, label, args.title)
        latest = frames.get("tile", "—") if frames else "—"
        n_aux = len(aux.get("channels", [])) if aux else 0
        n_cls = len(label.get("legend", [])) if label else 0
        print(f"[campaign-dashboard] done={status['done']}/{status['total']} "
              f"({status['pct']}%) rate={status['rate_recent_per_h']}/h "
              f"eta={_fmt_eta(status['eta_hours'])} latest={latest} aux={n_aux} "
              f"label_classes={n_cls}", flush=True)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
