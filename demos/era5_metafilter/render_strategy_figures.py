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
    fig, ax = plt.subplots(figsize=(11, 6.5))
    keys = STRATEGY_ORDER

    # Per-strategy label offset to avoid collisions on the M2/M3/M4 cluster
    offsets = {
        "M0_stac_only":      ( 8,   8),
        "M1_atmosphere":     ( 8,   8),
        "M2_stac_then_scl":  (-12,  18),
        "M3_scl_only":       ( 8,  -22),
        "M4_era5_then_scl":  ( 8,  18),
        "M5_era5_then_stac": ( 8,   8),
    }

    pts = []
    for k in keys:
        s = metrics["strategies"][k]
        x = s["total_wall_s"]
        y = s["cot"]["mean_cot"]
        if y is None:
            continue
        pts.append((x, y, k, s))
        ax.scatter(x, y, s=240, color=STRATEGY_COLORS[k],
                   edgecolor="white", linewidth=2, zorder=3)
        dx, dy = offsets.get(k, (8, 8))
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        ax.annotate(
            f"{s['label']}\nn={s['n_dates']}, COT={y:.4f}",
            xy=(x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=10, fontweight="bold", ha=ha, va=va,
            arrowprops={"arrowstyle": "-", "color": "#aaa", "lw": 0.7,
                        "alpha": 0.7},
        )

    # Trace the Pareto frontier (non-dominated points)
    pts_sorted = sorted(pts, key=lambda p: p[0])
    frontier = []
    best_y = float("inf")
    for x, y, k, s in pts_sorted:
        if y < best_y:
            frontier.append((x, y))
            best_y = y
    if len(frontier) >= 2:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.plot(fx, fy, ls="--", color="#27ae60", lw=2, alpha=0.5,
                label="Pareto-front (icke-dominerade)")
        ax.legend(loc="lower right")

    ax.set_xlabel("Total wall-clock (s) — lägre = snabbare", fontsize=11)
    ax.set_ylabel("Mean COT — lägre = renare data", fontsize=11)
    ax.set_title(
        "Tid vs datakvalitet — Pareto-front\n"
        "Strategier på den gröna linjen är icke-dominerade. "
        "Dominanta val: ERA5 → STAC (snabbast) eller ERA5 → SCL (renast).",
        fontsize=12,
    )
    ax.grid(alpha=0.3)
    ax.invert_yaxis()
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
