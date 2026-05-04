"""
demos/era5_metafilter/replicate_metafilter.py

Showcase: replicates the metafilter pattern (github.com/erikkallman/metafilter)
against an ImintEngine-relevant AOI (Skåne 2022).

Question this answers:
    "How much do we save if we filter Sentinel-2 candidate days by ERA5
     weather BEFORE we hit STAC / asset endpoints?"

Pipeline:
  1. List all Sentinel-2-L2A scenes intersecting AOI for the growing season
     (Element84 earth-search STAC, anonymous).
  2. Fetch ERA5-equivalent daily weather from Open-Meteo Historical API
     (free, ERA5-based reanalysis — used here as a stand-in for the
     ECMWF Polytope/CDS API path that imint/training/era5_aux.py uses).
  3. Apply a meteorological pre-filter to identify "good observation days".
  4. Match S2 scenes to good days; compute query-reduction + cloud-cover
     metrics; render figures.

Run:
    python demos/era5_metafilter/replicate_metafilter.py

Outputs:
    demos/era5_metafilter/data/{stac_skane_2022.json, era5_skane_2022.json}
    demos/era5_metafilter/figures/{01,02,03}_*.png
    demos/era5_metafilter/metrics.json

Network: first run hits two public APIs (~1MB total). Subsequent runs are
fully offline thanks to JSON caches.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import requests

HERE = Path(__file__).parent
DATA = HERE / "data"
FIGS = HERE / "figures"
DATA.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# ---------- AOI + period ----------
# Small agricultural BBOX in Skåne, near Lund (mixed crops + open land).
# Chosen to match ImintEngine's training domain (LPIS-rich southern Sweden).
AOI_NAME = "Skåne (Lund-omgivning)"
BBOX_LL = (13.05, 55.65, 13.35, 55.80)   # (min_lon, min_lat, max_lon, max_lat)
CENTER_LL = ((BBOX_LL[0] + BBOX_LL[2]) / 2, (BBOX_LL[1] + BBOX_LL[3]) / 2)
YEAR = 2022
PERIOD_START = date(YEAR, 4, 1)
PERIOD_END = date(YEAR, 9, 30)

# ---------- Metafilter rules ----------
# Inspired by metafilter's "weather-driven candidate selection".
# A day passes the filter when:
#   - precipitation that day is ≤ 0.5 mm           (dry surface, less likely cloud)
#   - precipitation in preceding 2 days is ≤ 3 mm  (atmosphere had time to clear)
#   - mean 2m temperature ≥ 10 °C                  (growing season is active)
# These thresholds are intentionally simple and explainable for the showcase.
RULE = {
    "precip_today_max_mm": 0.5,
    "precip_prev2d_max_mm": 3.0,
    "t2m_mean_min_c": 10.0,
}


@dataclass
class Scene:
    date: str          # ISO date
    cloud_cover: float
    scene_id: str


@dataclass
class WeatherDay:
    date: str          # ISO date
    t2m_mean: float    # °C
    precip_mm: float   # mm


# ---------- Data fetchers ----------

def fetch_stac_scenes() -> list[Scene]:
    """Anonymous STAC search against Element84 earth-search for sentinel-2-l2a."""
    cache = DATA / "stac_skane_2022.json"
    if cache.exists():
        with open(cache) as f:
            payload = json.load(f)
    else:
        url = "https://earth-search.aws.element84.com/v1/search"
        body = {
            "collections": ["sentinel-2-l2a"],
            "bbox": list(BBOX_LL),
            "datetime": f"{PERIOD_START.isoformat()}T00:00:00Z/{PERIOD_END.isoformat()}T23:59:59Z",
            "limit": 500,
        }
        all_items: list[dict] = []
        next_body = body
        # follow pagination via 'next' link with token if present
        for _ in range(20):
            r = requests.post(url, json=next_body, timeout=60)
            r.raise_for_status()
            page = r.json()
            all_items.extend(page.get("features", []))
            nxt = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
            if not nxt:
                break
            # earth-search uses POST with a body merge for next page
            next_body = {**body, **(nxt.get("body") or {})}
        payload = {"features": all_items}
        with open(cache, "w") as f:
            json.dump(payload, f)

    scenes: list[Scene] = []
    for feat in payload["features"]:
        props = feat.get("properties", {})
        dt = props.get("datetime", "")[:10]
        cc = props.get("eo:cloud_cover")
        if not dt or cc is None:
            continue
        scenes.append(Scene(date=dt, cloud_cover=float(cc), scene_id=feat.get("id", "")))
    scenes.sort(key=lambda s: s.date)
    return scenes


def fetch_era5_daily() -> list[WeatherDay]:
    """Open-Meteo Historical API — ERA5-based daily reanalysis, no auth needed."""
    cache = DATA / "era5_skane_2022.json"
    if cache.exists():
        with open(cache) as f:
            payload = json.load(f)
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": f"{CENTER_LL[1]:.4f}",
            "longitude": f"{CENTER_LL[0]:.4f}",
            "start_date": PERIOD_START.isoformat(),
            "end_date": PERIOD_END.isoformat(),
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "Europe/Stockholm",
        }
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        with open(cache, "w") as f:
            json.dump(payload, f)

    daily = payload["daily"]
    out: list[WeatherDay] = []
    for d, t, p in zip(
        daily["time"], daily["temperature_2m_mean"], daily["precipitation_sum"]
    ):
        if t is None or p is None:
            continue
        out.append(WeatherDay(date=d, t2m_mean=float(t), precip_mm=float(p)))
    return out


# ---------- Filter logic ----------

def good_days(weather: list[WeatherDay]) -> set[str]:
    """Return ISO dates passing the metafilter rules."""
    by_date = {w.date: w for w in weather}
    keep: set[str] = set()
    for w in weather:
        d = date.fromisoformat(w.date)
        prev_sum = 0.0
        for offset in (1, 2):
            prev = by_date.get((d - timedelta(days=offset)).isoformat())
            if prev is not None:
                prev_sum += prev.precip_mm
        if (
            w.precip_mm <= RULE["precip_today_max_mm"]
            and prev_sum <= RULE["precip_prev2d_max_mm"]
            and w.t2m_mean >= RULE["t2m_mean_min_c"]
        ):
            keep.add(w.date)
    return keep


# ---------- Figures ----------

def plot_candidate_days(
    weather: list[WeatherDay],
    scenes: list[Scene],
    keep: set[str],
    path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    dates = [date.fromisoformat(w.date) for w in weather]
    t2m = [w.t2m_mean for w in weather]
    pr = [w.precip_mm for w in weather]

    ax1.plot(dates, t2m, color="#c0392b", lw=1.4, label="T2m mean (°C)")
    ax1.axhline(RULE["t2m_mean_min_c"], color="#c0392b", ls=":", lw=0.9, alpha=0.6)
    ax1.set_ylabel("T2m (°C)", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1b = ax1.twinx()
    ax1b.bar(dates, pr, color="#2980b9", alpha=0.55, label="Precip (mm)", width=1.0)
    ax1b.set_ylabel("Precip (mm)", color="#2980b9")
    ax1b.tick_params(axis="y", labelcolor="#2980b9")

    # Scene markers + filter shading
    s_dates_all = [date.fromisoformat(s.date) for s in scenes]
    s_keep = [date.fromisoformat(s.date) for s in scenes if s.date in keep]
    ax2.scatter(s_dates_all, [1] * len(s_dates_all),
                color="#7f8c8d", s=25, label=f"All S2 scenes (n={len(s_dates_all)})")
    ax2.scatter(s_keep, [1] * len(s_keep),
                color="#27ae60", s=55, label=f"Pass ERA5 filter (n={len(s_keep)})",
                zorder=3, edgecolor="white", linewidth=0.5)
    ax2.set_yticks([])
    ax2.set_xlabel(f"Datum {YEAR}")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    fig.suptitle(
        f"ERA5-driven kandidatdagar — {AOI_NAME}, {YEAR}",
        fontsize=12, y=0.99,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_cloud_distribution(scenes: list[Scene], keep: set[str], path: Path) -> None:
    cc_all = [s.cloud_cover for s in scenes]
    cc_keep = [s.cloud_cover for s in scenes if s.date in keep]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 100, 21)
    ax.hist(cc_all, bins=bins, color="#7f8c8d", alpha=0.6,
            label=f"Alla scener (median {np.median(cc_all):.0f}%)")
    if cc_keep:
        ax.hist(cc_keep, bins=bins, color="#27ae60", alpha=0.85,
                label=f"Efter ERA5-filter (median {np.median(cc_keep):.0f}%)")
    ax.set_xlabel("Cloud cover (%)")
    ax.set_ylabel("Antal scener")
    ax.set_title(f"Cloud cover-fördelning före/efter ERA5-filter — {AOI_NAME}, {YEAR}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_query_reduction(metrics: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Utan filter", "Efter ERA5-filter"]
    values = [metrics["queries_before"], metrics["queries_after"]]
    colors = ["#7f8c8d", "#27ae60"]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v),
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    reduction = metrics["reduction_pct"]
    ax.set_ylabel("Antal kandidat-scener (STAC-träffar)")
    ax.set_title(
        f"Query-reduktion: −{reduction:.0f}%  "
        f"({metrics['queries_before']} → {metrics['queries_after']})\n"
        f"{AOI_NAME}, {PERIOD_START.isoformat()} → {PERIOD_END.isoformat()}"
    )
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------- Main ----------

def main() -> int:
    print(f"AOI:    {AOI_NAME}  bbox={BBOX_LL}")
    print(f"Period: {PERIOD_START} → {PERIOD_END}")
    print(f"Rule:   {RULE}")

    print("\n[1/4] Listar Sentinel-2-L2A scener via STAC…")
    scenes = fetch_stac_scenes()
    print(f"       {len(scenes)} scener funna ({scenes[0].date} … {scenes[-1].date})")

    print("\n[2/4] Hämtar ERA5-baserat dagligt väder (Open-Meteo Archive)…")
    weather = fetch_era5_daily()
    print(f"       {len(weather)} dagliga observationer")

    print("\n[3/4] Tillämpar metafilter-regler…")
    keep = good_days(weather)
    print(f"       {len(keep)} av {len(weather)} dagar passerar filtret")

    scenes_keep = [s for s in scenes if s.date in keep]
    cc_all = [s.cloud_cover for s in scenes]
    cc_keep = [s.cloud_cover for s in scenes_keep]

    metrics = {
        "aoi": AOI_NAME,
        "bbox_ll": list(BBOX_LL),
        "period": [PERIOD_START.isoformat(), PERIOD_END.isoformat()],
        "rule": RULE,
        "days_total": len(weather),
        "days_pass_filter": len(keep),
        "queries_before": len(scenes),
        "queries_after": len(scenes_keep),
        "reduction_pct": round(100 * (1 - len(scenes_keep) / max(len(scenes), 1)), 1),
        "cloud_cover_median_before": float(np.median(cc_all)) if cc_all else None,
        "cloud_cover_median_after": float(np.median(cc_keep)) if cc_keep else None,
        "cloud_cover_mean_before": float(np.mean(cc_all)) if cc_all else None,
        "cloud_cover_mean_after": float(np.mean(cc_keep)) if cc_keep else None,
    }

    print("\n[4/4] Renderar figurer + metrics.json…")
    plot_candidate_days(weather, scenes, keep, FIGS / "01_candidate_days.png")
    plot_cloud_distribution(scenes, keep, FIGS / "02_cloud_cover_dist.png")
    plot_query_reduction(metrics, FIGS / "03_query_reduction.png")

    with open(HERE / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n=== Resultat ===")
    print(f"  Kandidat-scener före filter: {metrics['queries_before']}")
    print(f"  Kandidat-scener efter filter: {metrics['queries_after']}")
    print(f"  Reduktion: {metrics['reduction_pct']}%")
    print(f"  Median cloud cover före:  {metrics['cloud_cover_median_before']:.1f}%")
    print(f"  Median cloud cover efter: {metrics['cloud_cover_median_after']:.1f}%")
    print(f"\n  Artefakter:  {HERE.relative_to(Path.cwd())}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
