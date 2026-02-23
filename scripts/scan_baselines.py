"""
scripts/scan_baselines.py -- Scan and rank baseline candidates by visibility

Three-phase approach:
1. STAC API discovers all available Sentinel-2 dates
2. SCL cloud fraction computed within AOI for each date
3. COT (Cloud Optical Thickness) run on top 5 to pick best visibility

Usage:
    .venv/bin/python scripts/scan_baselines.py \
        --west 15.42 --south 61.92 --east 15.47 --north 61.97 \
        --date 2018-07-24 --search-start 30 --search-end 90
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import webbrowser
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from imint.fetch import scan_baseline_candidates, run_cot_on_candidates


def _rgb_to_base64(rgb: np.ndarray) -> str:
    """Convert float32 RGB array to base64-encoded PNG data URI."""
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _generate_html(top_candidates, all_candidates, date, coords,
                   search_start, search_end, n_stac_total):
    """Generate self-contained HTML report with COT-ranked candidates."""

    # --- Top candidates (COT-verified) ---
    top_cards = ""
    for i, c in enumerate(top_candidates):
        thumb_b64 = _rgb_to_base64(c.rgb_thumbnail)
        rank = i + 1
        badge = ""
        if i == 0:
            badge = '<span class="badge best">Vald</span>'

        if c.cloud_fraction < 0.10:
            bar_color = "#22c55e"
        elif c.cloud_fraction < 0.30:
            bar_color = "#eab308"
        else:
            bar_color = "#ef4444"
        bar_width = max(c.cloud_fraction * 100, 1)

        cot_html = ""
        if c.cot_stats:
            cf = c.cot_stats["clear_fraction"]
            cm = c.cot_stats["cot_mean"]
            cot_html = f"""
                <div class="cloud-row">
                    <span class="cloud-label">Klar himmel (COT):</span>
                    <span class="cloud-value">{cf*100:.1f}%</span>
                </div>
                <div class="cloud-row scene">
                    <span class="cloud-label">COT medel:</span>
                    <span class="scene-value">{cm:.6f}</span>
                </div>"""

        top_cards += f"""
        <div class="candidate-card{' best-card' if i == 0 else ''}">
            <div class="rank">#{rank}</div>
            <div class="thumbnail">
                <img src="{thumb_b64}" alt="RGB {c.date}">
            </div>
            <div class="info">
                <div class="date-row">
                    <span class="date">{c.date}</span>
                    {badge}
                </div>
                {cot_html}
                <div class="cloud-row">
                    <span class="cloud-label">Moln AOI (SCL):</span>
                    <span class="cloud-value">{c.cloud_fraction*100:.1f}%</span>
                </div>
                <div class="cloud-bar-bg">
                    <div class="cloud-bar" style="width:{bar_width:.1f}%;background:{bar_color}"></div>
                </div>
                <div class="cloud-row scene">
                    <span class="cloud-label">Moln scene:</span>
                    <span class="scene-value">{c.scene_cloud_fraction:.1f}%</span>
                </div>
                <div class="meta">{c.shape[1]}&times;{c.shape[0]} px</div>
            </div>
        </div>"""

    # --- All other candidates (SCL only) ---
    top_dates = {c.date for c in top_candidates}
    other = [c for c in all_candidates if c.date not in top_dates]
    other_cards = ""
    for c in other:
        thumb_b64 = _rgb_to_base64(c.rgb_thumbnail)
        if c.cloud_fraction < 0.10:
            bar_color = "#22c55e"
        elif c.cloud_fraction < 0.30:
            bar_color = "#eab308"
        else:
            bar_color = "#ef4444"
        bar_width = max(c.cloud_fraction * 100, 1)

        other_cards += f"""
        <div class="candidate-card other">
            <div class="thumbnail">
                <img src="{thumb_b64}" alt="RGB {c.date}">
            </div>
            <div class="info">
                <div class="date-row">
                    <span class="date">{c.date}</span>
                </div>
                <div class="cloud-row">
                    <span class="cloud-label">Moln AOI (SCL):</span>
                    <span class="cloud-value">{c.cloud_fraction*100:.1f}%</span>
                </div>
                <div class="cloud-bar-bg">
                    <div class="cloud-bar" style="width:{bar_width:.1f}%;background:{bar_color}"></div>
                </div>
                <div class="cloud-row scene">
                    <span class="cloud-label">Moln scene:</span>
                    <span class="scene-value">{c.scene_cloud_fraction:.1f}%</span>
                </div>
                <div class="meta">{c.shape[1]}&times;{c.shape[0]} px</div>
            </div>
        </div>"""

    area_str = (f"{coords['west']:.2f}-{coords['east']:.2f}E, "
                f"{coords['south']:.2f}-{coords['north']:.2f}N")

    # Best COT stats
    best = top_candidates[0] if top_candidates else None
    best_cot_val = ""
    best_cot_detail = ""
    if best and best.cot_stats:
        best_cot_val = f"{best.cot_stats['clear_fraction']*100:.1f}%"
        best_cot_detail = f"{best.date} (COT: {best.cot_stats['cot_mean']:.6f})"
    else:
        best_cot_val = "N/A"
        best_cot_detail = ""

    other_section = ""
    if other_cards:
        other_section = f"""
    <div class="section-header">
        <h2>\u00d6vriga kandidater (SCL)</h2>
    </div>
    <div class="grid other-grid">
        {other_cards}
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="sv">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Baseline-kandidater \u2014 {date}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0b0e17;
            color: #d8dae5;
            line-height: 1.5;
        }}
        .header {{
            background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
            padding: 24px 32px;
            border-bottom: 1px solid #334155;
        }}
        .header h1 {{
            font-size: 22px;
            font-weight: 700;
            color: #f8fafc;
        }}
        .header h1 span {{ color: #3b82f6; }}
        .header-meta {{
            font-size: 13px;
            color: #94a3b8;
            margin-top: 6px;
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .header-meta strong {{ color: #cbd5e1; }}
        .summary {{
            display: flex;
            gap: 10px;
            padding: 16px 20px;
            flex-wrap: wrap;
        }}
        .summary-card {{
            background: #111827;
            border-radius: 10px;
            padding: 14px 18px;
            min-width: 150px;
            flex: 1;
            border: 1px solid #1e293b;
            position: relative;
            overflow: hidden;
        }}
        .summary-card::before {{
            content: '';
            position: absolute;
            left: 0; top: 0; bottom: 0;
            width: 3px;
            background: #3b82f6;
        }}
        .summary-card h4 {{
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: #64748b;
            margin-bottom: 4px;
            font-weight: 600;
        }}
        .summary-card .value {{
            font-size: 22px;
            font-weight: 700;
            color: #f1f5f9;
        }}
        .summary-card .detail {{
            font-size: 11px;
            color: #64748b;
            margin-top: 2px;
        }}
        .section-header {{
            text-align: center;
            padding: 24px 20px 8px;
        }}
        .section-header h2 {{
            font-size: 17px;
            font-weight: 600;
            color: #e2e8f0;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
            gap: 10px;
            padding: 12px 20px 32px;
        }}
        .other-grid {{
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
        }}
        .candidate-card {{
            background: #111827;
            border: 1px solid #1e293b;
            border-radius: 10px;
            overflow: hidden;
            display: flex;
            position: relative;
        }}
        .candidate-card.best-card {{
            border-color: #22c55e;
            box-shadow: 0 0 12px rgba(34, 197, 94, 0.15);
        }}
        .candidate-card.other {{
            opacity: 0.7;
        }}
        .rank {{
            position: absolute;
            top: 8px;
            left: 8px;
            background: rgba(0,0,0,0.7);
            color: #94a3b8;
            font-size: 11px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 4px;
            z-index: 1;
        }}
        .best-card .rank {{
            background: rgba(34, 197, 94, 0.8);
            color: #fff;
        }}
        .thumbnail {{
            width: 200px;
            min-height: 140px;
            flex-shrink: 0;
            background: #0b0e17;
        }}
        .thumbnail img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }}
        .info {{
            padding: 14px 16px;
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 5px;
        }}
        .date-row {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .date {{
            font-size: 18px;
            font-weight: 700;
            color: #f1f5f9;
            font-variant-numeric: tabular-nums;
        }}
        .badge {{
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            padding: 2px 8px;
            border-radius: 4px;
        }}
        .badge.best {{
            background: rgba(34, 197, 94, 0.2);
            color: #22c55e;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }}
        .cloud-row {{
            display: flex;
            align-items: baseline;
            gap: 6px;
        }}
        .cloud-label {{
            font-size: 12px;
            color: #64748b;
        }}
        .cloud-value {{
            font-size: 15px;
            font-weight: 700;
            color: #e2e8f0;
            font-variant-numeric: tabular-nums;
        }}
        .scene-value {{
            font-size: 12px;
            color: #64748b;
            font-variant-numeric: tabular-nums;
        }}
        .cloud-bar-bg {{
            width: 100%;
            height: 6px;
            background: #1e293b;
            border-radius: 3px;
            overflow: hidden;
        }}
        .cloud-bar {{
            height: 100%;
            border-radius: 3px;
        }}
        .meta {{
            font-size: 11px;
            color: #475569;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            font-size: 11px;
            color: #475569;
            border-top: 1px solid #1e293b;
        }}
        @media (max-width: 600px) {{
            .grid {{ grid-template-columns: 1fr; }}
            .candidate-card {{ flex-direction: column; }}
            .thumbnail {{ width: 100%; height: 180px; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1><span>IMINT</span> Baseline-kandidater</h1>
        <div class="header-meta">
            <span><strong>Analysdatum:</strong> {date}</span>
            <span><strong>AOI:</strong> {area_str}</span>
            <span><strong>S\u00f6kf\u00f6nster:</strong> {search_start}\u2013{search_end} dagar bak\u00e5t</span>
        </div>
    </div>

    <div class="summary">
        <div class="summary-card">
            <h4>STAC-datum</h4>
            <div class="value">{n_stac_total}</div>
            <div class="detail">bilder i f\u00f6nstret</div>
        </div>
        <div class="summary-card">
            <h4>SCL-skannade</h4>
            <div class="value">{len(all_candidates)}</div>
            <div class="detail">med RGB + SCL</div>
        </div>
        <div class="summary-card">
            <h4>COT-analyserade</h4>
            <div class="value">{len(top_candidates)}</div>
            <div class="detail">topp-kandidater</div>
        </div>
        <div class="summary-card">
            <h4>B\u00e4st visibilitet</h4>
            <div class="value">{best_cot_val}</div>
            <div class="detail">{best_cot_detail}</div>
        </div>
    </div>

    <div class="section-header">
        <h2>Topp {len(top_candidates)} \u2014 COT-verifierade</h2>
    </div>
    <div class="grid">
        {top_cards}
    </div>

    {other_section}

    <div class="footer">
        IMINT Engine &middot; Baseline Scanner (STAC + SCL + COT) &middot; {date}
    </div>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Scan and rank baseline candidates by visibility (STAC + SCL + COT)",
    )
    parser.add_argument("--west", type=float, required=True)
    parser.add_argument("--south", type=float, required=True)
    parser.add_argument("--east", type=float, required=True)
    parser.add_argument("--north", type=float, required=True)
    parser.add_argument("--date", required=True, help="Analysis date (ISO format)")
    parser.add_argument("--search-start", type=int, default=30,
                        help="Start of search window (days before date)")
    parser.add_argument("--search-end", type=int, default=90,
                        help="End of search window (days before date)")
    parser.add_argument("--scene-cloud-max", type=float, default=80.0,
                        help="Max scene-level cloud %% to include (default: 80)")
    parser.add_argument("--cot-top", type=int, default=5,
                        help="Number of top candidates to run COT on (default: 5)")
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: outputs/baselines_<date>.html)")
    args = parser.parse_args()

    coords = {
        "west": args.west, "south": args.south,
        "east": args.east, "north": args.north,
    }

    print("=" * 60)
    print("  IMINT Baseline Scanner (STAC + SCL + COT)")
    print(f"  AOI:    {args.west:.2f}-{args.east:.2f}E, {args.south:.2f}-{args.north:.2f}N")
    print(f"  Datum:  {args.date}")
    print(f"  F\u00f6nster: {args.search_start}-{args.search_end} dagar bak\u00e5t")
    print(f"  COT topp: {args.cot_top}")
    print("=" * 60)

    # Phase 1+2: STAC discovery + SCL fetch
    all_candidates = scan_baseline_candidates(
        date=args.date,
        coords=coords,
        search_start_days=args.search_start,
        search_end_days=args.search_end,
        scene_cloud_max=args.scene_cloud_max,
    )

    if not all_candidates:
        print("\n  Inga kandidater hittades.")
        return

    # Phase 3: COT on top N
    top = run_cot_on_candidates(
        candidates=all_candidates,
        coords=coords,
        top_n=args.cot_top,
    )

    winner = top[0] if top else all_candidates[0]
    print(f"\n  Vald baseline: {winner.date}")
    if winner.cot_stats:
        print(f"  Klar himmel: {winner.cot_stats['clear_fraction']*100:.1f}%")
        print(f"  COT medel:   {winner.cot_stats['cot_mean']:.6f}")

    n_stac = len(all_candidates)
    print(f"\n  Genererar HTML ({n_stac} + {len(top)} COT)...")

    html = _generate_html(
        top_candidates=top,
        all_candidates=all_candidates,
        date=args.date,
        coords=coords,
        search_start=args.search_start,
        search_end=args.search_end,
        n_stac_total=n_stac,
    )

    if args.output:
        output_path = args.output
    else:
        output_dir = str(PROJECT_ROOT / "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"baselines_{args.date}.html")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Rapport sparad: {output_path}")
    webbrowser.open(f"file://{os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
