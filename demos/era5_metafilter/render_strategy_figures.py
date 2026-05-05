"""
demos/era5_metafilter/render_strategy_figures.py

Renders the four headline figures for the rebuilt Atmosfär showcase:

    F1 — Total wall-clock per strategy (stacked: selection vs spectral)
    F2 — Mean COT per strategy (lower = cleaner data)
    F3 — Pareto front: total time vs mean COT (the actual trade-off)
    F4 — API-call count per strategy (stacked: ERA5 / STAC / SCL / spectral)

Reads strategies_metrics.json (produced by benchmark_strategies.py),
writes PNGs to docs/showcase/era5_metafilter/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
DOCS = REPO_ROOT / "docs" / "showcase" / "era5_metafilter"

STRATEGY_ORDER = [
    "M0_stac_only", "M1_atmosphere", "M2_stac_then_scl",
    "M3_scl_only", "M5_era5_then_stac", "M4_era5_then_scl",
]
STRATEGY_COLORS = {
    "M0_stac_only":      "#d35400",
    "M1_atmosphere":     "#27ae60",
    "M2_stac_then_scl":  "#2980b9",
    "M3_scl_only":       "#8e44ad",
    "M4_era5_then_scl":  "#16a085",
    "M5_era5_then_stac": "#7f8c8d",
}
STAGE_COLORS = {
    "era5":      "#27ae60",
    "stac":      "#d35400",
    "scl_stack": "#8e44ad",
    "spectral":  "#34495e",
}


def short_label(s: dict) -> str:
    return f"{s['label']}\n(n={s['n_dates']})"


def f1_wall_clock(metrics: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    keys = STRATEGY_ORDER
    labels = [short_label(metrics["strategies"][k]) for k in keys]
    x = np.arange(len(keys))
    sel = [metrics["strategies"][k]["selection_total_s"] for k in keys]
    spec = [metrics["strategies"][k]["spectral_wall_s"]   for k in keys]

    ax.bar(x, sel, color="#9b59b6", label="Urvalsfas (ERA5/STAC/SCL)", width=0.6)
    ax.bar(x, spec, bottom=sel, color="#34495e",
           label="Spektral-fetch (parallell, 6 workers)", width=0.6)

    for i, (s, sp) in enumerate(zip(sel, spec)):
        total = s + sp
        ax.text(i, total + max(sel + spec) * 0.02, f"{total:.0f}s",
                ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Wall-clock (s)")
    ax.set_title("Total fetch-tid per strategi — DES openEO, 6 workers\n"
                 "Urvalsfas mäts live, spektral-fas estimeras från cachade per-dag-tider")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def f2_cot(metrics: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    keys = STRATEGY_ORDER
    labels = [short_label(metrics["strategies"][k]) for k in keys]
    cots = [metrics["strategies"][k]["cot"]["mean_cot"] or 0 for k in keys]
    medians = [metrics["strategies"][k]["cot"]["median_cot"] or 0 for k in keys]
    stds = [metrics["strategies"][k]["cot"]["std_cot"] or 0 for k in keys]
    colors = [STRATEGY_COLORS[k] for k in keys]
    x = np.arange(len(keys))

    bars = ax.bar(x, cots, yerr=stds, color=colors, width=0.6, capsize=8,
                  error_kw={"alpha": 0.5, "lw": 1.4})
    ax.scatter(x, medians, marker="D", color="#2c3e50", s=70, zorder=3,
               label="Median")

    ymax = max(cots) if cots else 1
    for i, c in enumerate(cots):
        ax.text(i, c + ymax * 0.04, f"{c:.4f}",
                ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Cloud Optical Thickness")
    ax.set_title("Datakvalitet per strategi — DES MLP5 COT-ensemble (Pirinen 2024)\n"
                 "Lägre = klarare. Felstaplar = ±1σ.")
    ax.axhline(0.025, color="#c0392b", ls=":", lw=1.0, alpha=0.6,
               label="Thick-cloud thresh (0.025)")
    ax.axhline(0.015, color="#e67e22", ls=":", lw=1.0, alpha=0.6,
               label="Thin-cloud thresh (0.015)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def f3_pareto(metrics: dict, out_path: Path) -> None:
    """Pareto-front: time vs data quality. Uses a numbered legend instead of
    inline labels to avoid the M2/M3/M4 cluster collision."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    keys = STRATEGY_ORDER
    short_labels = {
        "M0_stac_only":      "M0  Naiv STAC",
        "M1_atmosphere":     "M1  Atmosfär ensam",
        "M2_stac_then_scl":  "M2  Nuvarande pipeline",
        "M3_scl_only":       "M3  SCL-stack ensam",
        "M4_era5_then_scl":  "M4  ERA5 → SCL  ★",
        "M5_era5_then_stac": "M5  ERA5 → STAC  ★",
    }

    pts = []
    for k in keys:
        s = metrics["strategies"][k]
        x = s["total_wall_s"]
        y = s["cot"]["mean_cot"]
        if y is None:
            continue
        pts.append((x, y, k, s))
        marker_label = short_labels[k] + (
            f"  (n={s['n_dates']}, {x:.0f}s, COT {y:.4f})"
        )
        ax.scatter(x, y, s=260, color=STRATEGY_COLORS[k],
                   edgecolor="white", linewidth=2, zorder=3,
                   label=marker_label)
        ax.annotate(
            k.split("_")[0],   # "M0", "M1", …
            xy=(x, y), xytext=(0, 0), textcoords="offset points",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="white", zorder=4,
        )

    # Pareto frontier
    pts_sorted = sorted(pts, key=lambda p: p[0])
    frontier = []
    best_y = float("inf")
    for x, y, _, _ in pts_sorted:
        if y < best_y:
            frontier.append((x, y))
            best_y = y
    if len(frontier) >= 2:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.plot(fx, fy, ls="--", color="#27ae60", lw=2, alpha=0.6,
                zorder=2, label="— — — Pareto-front (icke-dominerade)")

    ax.set_xlabel("Total wall-clock (s) — lägre = snabbare", fontsize=11)
    ax.set_ylabel("Mean COT — lägre = renare data", fontsize=11)
    ax.set_title(
        "Tid vs datakvalitet — Pareto-front",
        fontsize=13, pad=10, fontweight="bold",
    )
    # Some breathing room around the cluster
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_pad = (max(xs) - min(xs)) * 0.12
    y_pad = (max(ys) - min(ys)) * 0.12
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92,
              title="Strategier (★ = Pareto-optimal)", title_fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def f4_api_calls(metrics: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    keys = STRATEGY_ORDER
    labels = [short_label(metrics["strategies"][k]) for k in keys]
    x = np.arange(len(keys))

    era5 = [metrics["strategies"][k]["api_calls"]["era5"] for k in keys]
    stac = [metrics["strategies"][k]["api_calls"]["stac"] for k in keys]
    scl  = [metrics["strategies"][k]["api_calls"]["scl_stack"] for k in keys]
    spec = [metrics["strategies"][k]["api_calls"]["spectral_fetch"] for k in keys]

    bottom = np.zeros(len(keys))
    for series, color, label in [
        (era5, STAGE_COLORS["era5"],      "ERA5 (Open-Meteo)"),
        (stac, STAGE_COLORS["stac"],      "STAC (earth-search)"),
        (scl,  STAGE_COLORS["scl_stack"], "openEO SCL-stack"),
        (spec, STAGE_COLORS["spectral"],  "openEO spektral-fetch"),
    ]:
        ax.bar(x, series, bottom=bottom, color=color, label=label, width=0.6)
        bottom += np.array(series)

    for i, total in enumerate(bottom):
        ax.text(i, total + 0.6, f"{int(total)}", ha="center", va="bottom",
                fontweight="bold", fontsize=11)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Antal API-anrop per tile per säsong")
    ax.set_title("API-trafik per strategi — vad pipelinen faktiskt skickar")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    metrics_path = HERE / "strategies_metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"Run benchmark_strategies.py first; missing {metrics_path}")
    with open(metrics_path) as f:
        metrics = json.load(f)
    DOCS.mkdir(parents=True, exist_ok=True)

    f1_wall_clock(metrics, DOCS / "S1_total_wall.png")
    f2_cot(metrics,        DOCS / "S2_mean_cot.png")
    f3_pareto(metrics,     DOCS / "S3_pareto.png")
    f4_api_calls(metrics,  DOCS / "S4_api_calls.png")
    print(f"Rendered S1-S4 to {DOCS.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
