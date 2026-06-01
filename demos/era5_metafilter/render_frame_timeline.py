#!/usr/bin/env python3
"""Timeline figure: cloudiness vs the 4 frame windows + best date per frame.

Pulls REAL ERA5 overpass-time cloud cover from Open-Meteo for a sample
Swedish tile across the full Aug(y-1) → Sep(y) span, approximates the
Sentinel-2 pass cadence (~5 d), and plots:

  * each S2 pass as a dot, y = overpass cloud %, coloured clear/thin/thick
  * the four frame windows as shaded calendar bands (winter gap uncovered)
  * the lowest-cloud pass inside each window marked as that frame's pick

Real cloud values, approximated pass cadence (labelled as such). One clean
PNG, no compositing.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


C_INK    = "#1a4338"
C_CLEAR  = "#1a8f6a"
C_THIN   = "#e8a87c"
C_THICK  = "#ff826c"
C_GREY   = "#7b8a84"
C_ACCENT = "#0e6b8c"

# Sample tile centroid (West Sweden, the tile used in the e2e evidence).
LAT, LON = 60.85, 13.85

# Four frame windows. Slot 0 autumn is in year-1; slots 1-3 in year.
WINDOWS = [
    ("Ram 0 · Höst år-1",  date(2017, 8, 15), date(2017, 10, 31), "#c9b8e0"),
    ("Ram 1 · Vår",        date(2018, 4, 1),  date(2018, 5, 31),  "#a8d5e8"),
    ("Ram 2 · Försommar",  date(2018, 6, 1),  date(2018, 7, 19),  "#9ed3bd"),
    ("Ram 3 · Sensommar",  date(2018, 7, 20), date(2018, 9, 1),   "#f2d59b"),
]

THIN_T, THICK_T = 30.0, 60.0   # cloud-% thresholds for the dot colours
ERA5_GATE = 50.0                # overpass-cloud ceiling (DEFAULT_ATMOSPHERE_RULES)
S2_PASS_STRIDE_DAYS = 3         # ~S2A+S2B effective revisit at 60°N (overlapping swaths)


def _stac_s2_dates(d0: date, d1: date) -> dict[str, float]:
    """Real Sentinel-2-L2A acquisition dates over the AOI via earth-search STAC.

    Returns ``{date: granule_eo_cloud_cover_pct}`` for every acquisition in
    the window (no cloud filter) — these are the ACTUAL satellite pass days,
    not a synthetic cadence. Mirrors imint.training.optimal_fetch.
    stac_filter_dates but keeps all dates + their granule cloud.
    """
    import requests

    # ~0.1° AOI box around the tile centroid.
    half = 0.05
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [LON - half, LAT - half, LON + half, LAT + half],
        "datetime": f"{d0.isoformat()}T00:00:00Z/{d1.isoformat()}T23:59:59Z",
        "limit": 500,
    }
    r = requests.post(
        "https://earth-search.aws.element84.com/v1/search",
        json=body, timeout=90,
    )
    r.raise_for_status()
    by_date: dict[str, float] = {}
    for feat in r.json().get("features", []):
        props = feat.get("properties", {})
        d = props.get("datetime", "")[:10]
        cc = props.get("eo:cloud_cover")
        if not d or cc is None:
            continue
        prev = by_date.get(d)
        if prev is None or float(cc) < prev:
            by_date[d] = float(cc)
    return by_date


def _overpass_cloud_by_day(d0: date, d1: date) -> dict[str, float]:
    """Real Open-Meteo hourly cloud_cover → per-day 10-11h local mean."""
    import requests
    from collections import defaultdict

    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LAT, "longitude": LON,
            "start_date": d0.isoformat(), "end_date": d1.isoformat(),
            "hourly": "cloud_cover", "timezone": "Europe/Stockholm",
        },
        timeout=90,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    buckets: dict[str, list[float]] = defaultdict(list)
    for ts, c in zip(h["time"], h["cloud_cover"]):
        if c is None:
            continue
        if int(ts[11:13]) in (10, 11):
            buckets[ts[:10]].append(float(c))
    return {d: sum(v) / len(v) for d, v in buckets.items() if v}


def _dot_color(pct: float) -> str:
    if pct < THIN_T:
        return C_CLEAR
    if pct < THICK_T:
        return C_THIN
    return C_THICK


def render(out_path: Path) -> Path:
    cloud = _overpass_cloud_by_day(date(2017, 8, 1), date(2018, 9, 10))

    fig, ax = plt.subplots(figsize=(15.5, 6.6), dpi=150)

    # Shade the four frame windows.
    for label, ws, we, col in WINDOWS:
        ax.axvspan(ws, we, color=col, alpha=0.45, zorder=0)
        ax.text(ws + (we - ws) / 2, 104, label, ha="center", va="bottom",
                fontsize=9.2, weight="bold", color=C_INK)

    # ERA5 gate line.
    ax.axhline(50, color=C_THICK, lw=1.1, ls="--", zorder=1)
    ax.text(date(2017, 8, 3), 51.5, "ERA5 overpass-grind 50 %",
            fontsize=8, color=C_THICK, va="bottom")

    # REAL Sentinel-2 acquisition dates from STAC (earth-search), each with
    # its ERA5 overpass cloud value. No synthetic cadence — every dot is an
    # actual satellite pass over the AOI.
    stac_cc = _stac_s2_dates(date(2017, 8, 1), date(2018, 9, 10))
    passes: list[tuple[date, float]] = []
    for d_iso in sorted(stac_cc):
        v = cloud.get(d_iso)            # ERA5 overpass cloud at that real pass
        if v is None:
            continue
        passes.append((date.fromisoformat(d_iso), v))
    print(f"  {len(stac_cc)} STAC S2-L2A passes, "
          f"{len(passes)} with ERA5 overpass cloud")

    # Plot all passes (dim if outside any window).
    def _in_window(dd: date) -> bool:
        return any(ws <= dd <= we for _, ws, we, _ in WINDOWS)

    for dd, v in passes:
        inw = _in_window(dd)
        ax.scatter(dd, v, s=46 if inw else 26,
                   color=_dot_color(v), edgecolor=C_INK if inw else "none",
                   linewidth=0.8, alpha=1.0 if inw else 0.30, zorder=3)

    # Best (lowest-cloud) pass inside each window → frame pick. Gate-aware:
    # if even the cleanest pass is above the ERA5 gate, the frame is left
    # empty (better signal-absence than cloud) — shown as a warning star.
    for label, ws, we, col in WINDOWS:
        in_win = [(dd, v) for dd, v in passes if ws <= dd <= we]
        if not in_win:
            continue
        best_d, best_v = min(in_win, key=lambda x: x[1])
        passes_gate = best_v <= ERA5_GATE
        star_col = C_CLEAR if passes_gate else C_THICK
        ax.scatter(best_d, best_v, s=300, marker="*",
                   color=star_col, edgecolor=C_INK, linewidth=1.4, zorder=5)
        if passes_gate:
            note = f"{best_d.isoformat()}\n{best_v:.0f}% → vald"
        else:
            note = f"{best_d.isoformat()}\n{best_v:.0f}% > grind\nram lämnas tom"
        ax.annotate(
            note, (best_d, best_v), textcoords="offset points",
            xytext=(0, -40 if passes_gate else 30),
            ha="center", fontsize=8.0,
            color=C_INK if passes_gate else C_THICK, weight="bold",
            family="monospace",
            arrowprops=dict(arrowstyle="-",
                            color=C_INK if passes_gate else C_THICK, lw=0.8))

    ax.set_ylim(-8, 116)
    ax.set_xlim(date(2017, 8, 1), date(2018, 9, 12))
    ax.set_ylabel("Overpass-tid molntäcke (%)", fontsize=10.5, color=C_INK)
    ax.set_title(
        "Molnighet över tid · de fyra ramfönstren · bästa tillgängliga bild per ram",
        fontsize=14.5, weight="bold", color=C_INK, pad=26, loc="left")

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(labelsize=8.5, colors=C_GREY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GREY)

    # Legend (dot semantics) + winter-gap note.
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_CLEAR,
               markeredgecolor=C_INK, markersize=9, label=f"klar (<{THIN_T:.0f}%)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_THIN,
               markeredgecolor=C_INK, markersize=9, label=f"tunt ({THIN_T:.0f}-{THICK_T:.0f}%)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_THICK,
               markeredgecolor=C_INK, markersize=9, label=f"tjockt (>{THICK_T:.0f}%)"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=C_CLEAR,
               markeredgecolor=C_INK, markersize=15, label="vald bild per ram"),
    ]
    ax.legend(handles=leg, loc="upper right", fontsize=8.4, framealpha=0.9,
              ncol=4, columnspacing=1.2)

    ax.text(date(2018, 1, 5), -4,
            "vinter-glapp (Nov–Mar): inga ramfönster — S2-passager här "
            "ignoreras", ha="center", fontsize=8, color=C_GREY, style="italic")

    fig.text(0.012, 0.012,
             "Varje prick = faktisk S2-L2A-passage enligt STAC (earth-search) · "
             "y = ERA5 overpass-moln (Open-Meteo, 10-11 lokal) vid den passagen · "
             "AOI-SCL gör den exakta skärningen efter ERA5-grinden",
             fontsize=7.6, color=C_GREY, family="monospace")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("frame_timeline.png")
    print(f"wrote {render(out)}")
