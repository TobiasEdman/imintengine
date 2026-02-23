"""
imint/exporters/html_report.py — Interactive HTML analysis report

Generates a self-contained HTML file with:
- Synchronized zoomable Leaflet.js map viewers (CRS.Simple + leaflet-sync)
- Opacity sliders per layer
- Chart.js charts for NMD cross-reference data
- All images base64-encoded inline for portability
"""
from __future__ import annotations

import os
import json
import base64
from pathlib import Path


def _img_to_base64(path: str) -> str:
    """Read a PNG file and return a base64-encoded data URI."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{data}"


# ── Map viewer definitions ────────────────────────────────────────────────────

MAP_VIEWERS = [
    {
        "id": "map-rgb",
        "title": "Sentinel-2 RGB",
        "key": "rgb",
        "legend": [],
    },
    {
        "id": "map-nmd",
        "title": "NMD Marktäcke (Nivå 2)",
        "key": "nmd",
        "legend": [
            {"color": "#006400", "label": "Tallskog"},
            {"color": "#228B22", "label": "Granskog"},
            {"color": "#32CD32", "label": "Lövskog"},
            {"color": "#3CB371", "label": "Blandskog"},
            {"color": "#90EE90", "label": "Temp. ej skog"},
            {"color": "#2E4F2E", "label": "Sumpsk. tall"},
            {"color": "#3A5F3A", "label": "Sumpsk. gran"},
            {"color": "#4A7F4A", "label": "Sumpsk. löv"},
            {"color": "#5A8F5A", "label": "Sumpsk. bland"},
            {"color": "#7AAF7A", "label": "Sumpsk. temp"},
            {"color": "#8B5A2B", "label": "Öpp. våtmark"},
            {"color": "#FFD700", "label": "Åkermark"},
            {"color": "#C8AD7F", "label": "Öpp. mark bar"},
            {"color": "#D2B48C", "label": "Öpp. mark veg."},
            {"color": "#FF0000", "label": "Byggnader"},
            {"color": "#FF4500", "label": "Infrastruktur"},
            {"color": "#FF6347", "label": "Vägar"},
            {"color": "#0000FF", "label": "Sjöar"},
            {"color": "#1E90FF", "label": "Hav"},
        ],
    },
    {
        "id": "map-change",
        "title": "Förändringsdetektering",
        "key": "change",
        "legend": [
            {"color": "#FF0000", "label": "Förändring"},
        ],
    },
    {
        "id": "map-ndvi",
        "title": "NDVI (Vegetationsindex)",
        "key": "ndvi",
        "legend": [
            {"color": "#a50026", "label": "-1.0"},
            {"color": "#f46d43", "label": "-0.5"},
            {"color": "#fee08b", "label": "0.0"},
            {"color": "#a6d96a", "label": "0.5"},
            {"color": "#006837", "label": "1.0"},
        ],
    },
    {
        "id": "map-prithvi",
        "title": "Prithvi Segmentering",
        "key": "prithvi_seg",
        "legend": [
            {"color": "#228B22", "label": "Ej bränt"},
            {"color": "#FF4500", "label": "Bränt"},
        ],
    },
    {
        "id": "map-cot",
        "title": "Molnoptisk tjocklek (COT)",
        "key": "cot",
        "legend": [
            {"color": "#FFFFB2", "label": "Låg COT"},
            {"color": "#FD8D3C", "label": "Medel COT"},
            {"color": "#BD0026", "label": "Hög COT"},
        ],
    },
]


# ── NMD palette for charts ────────────────────────────────────────────────────

NMD_L1_CHART = {
    "forest":    {"label": "Skog",        "color": "rgba(34,139,34,0.85)",  "border": "#228B22"},
    "wetland":   {"label": "Våtmark",     "color": "rgba(139,90,43,0.85)",  "border": "#8B5A2B"},
    "cropland":  {"label": "Åkermark",    "color": "rgba(255,215,0,0.85)",  "border": "#FFD700"},
    "open_land": {"label": "Öppen mark",  "color": "rgba(210,180,140,0.85)","border": "#D2B48C"},
    "developed": {"label": "Bebyggelse",  "color": "rgba(255,0,0,0.85)",    "border": "#FF0000"},
    "water":     {"label": "Vatten",      "color": "rgba(0,0,255,0.85)",    "border": "#0000FF"},
}

NMD_L2_CHART = {
    "forest_pine":              {"label": "Tallskog",           "color": "rgba(0,100,0,0.85)",    "border": "#006400"},
    "forest_spruce":            {"label": "Granskog",           "color": "rgba(34,139,34,0.85)",  "border": "#228B22"},
    "forest_deciduous":         {"label": "Lövskog",           "color": "rgba(50,205,50,0.85)",  "border": "#32CD32"},
    "forest_mixed":             {"label": "Blandskog",          "color": "rgba(60,179,113,0.85)", "border": "#3CB371"},
    "forest_temp_non_forest":   {"label": "Temporärt ej skog",  "color": "rgba(144,238,144,0.85)","border": "#90EE90"},
    "forest_wetland_pine":      {"label": "Sumpskog tall",      "color": "rgba(46,79,46,0.85)",   "border": "#2E4F2E"},
    "forest_wetland_spruce":    {"label": "Sumpskog gran",      "color": "rgba(58,95,58,0.85)",   "border": "#3A5F3A"},
    "forest_wetland_deciduous": {"label": "Sumpskog löv",       "color": "rgba(74,127,74,0.85)",  "border": "#4A7F4A"},
    "forest_wetland_mixed":     {"label": "Sumpskog bland",     "color": "rgba(90,143,90,0.85)",  "border": "#5A8F5A"},
    "forest_wetland_temp":      {"label": "Sumpskog temp",      "color": "rgba(122,175,122,0.85)","border": "#7AAF7A"},
    "open_wetland":             {"label": "Öppen våtmark",      "color": "rgba(139,90,43,0.85)",  "border": "#8B5A2B"},
    "cropland":                 {"label": "Åkermark",           "color": "rgba(255,215,0,0.85)",  "border": "#FFD700"},
    "open_land_bare":           {"label": "Öppen mark, bar",    "color": "rgba(200,173,127,0.85)","border": "#C8AD7F"},
    "open_land_vegetated":      {"label": "Öppen mark, veg.",   "color": "rgba(210,180,140,0.85)","border": "#D2B48C"},
    "developed_buildings":      {"label": "Byggnader",          "color": "rgba(255,0,0,0.85)",    "border": "#FF0000"},
    "developed_infrastructure": {"label": "Infrastruktur",      "color": "rgba(255,69,0,0.85)",   "border": "#FF4500"},
    "developed_roads":          {"label": "Vägar",              "color": "rgba(255,99,71,0.85)",  "border": "#FF6347"},
    "water_lakes":              {"label": "Sjöar",              "color": "rgba(0,0,255,0.85)",    "border": "#0000FF"},
    "water_sea":                {"label": "Hav",                "color": "rgba(30,144,255,0.85)", "border": "#1E90FF"},
}

L1_ORDER = ["forest", "wetland", "cropland", "open_land", "developed", "water"]

L2_ORDER = [
    "forest_pine", "forest_spruce", "forest_deciduous", "forest_mixed",
    "forest_temp_non_forest", "forest_wetland_pine", "forest_wetland_spruce",
    "forest_wetland_deciduous", "forest_wetland_mixed", "forest_wetland_temp",
    "open_wetland", "cropland", "open_land_bare", "open_land_vegetated",
    "developed_buildings", "developed_infrastructure", "developed_roads",
    "water_lakes", "water_sea",
]


def save_html_report(
    image_paths: dict[str, str],
    nmd_stats: dict,
    imint_summary: dict,
    image_shape: tuple[int, int],
    date: str,
    output_path: str,
) -> str:
    """Generate a self-contained interactive HTML analysis report.

    Creates a single HTML file with:
    - Synchronized zoomable Leaflet map viewers (one per analysis layer)
    - Opacity sliders per analysis layer
    - Chart.js bar charts for NMD cross-reference data
    - All images base64-encoded inline

    Args:
        image_paths: Dict mapping layer key to PNG file path.
            Keys: "rgb", "nmd", "change", "ndvi",
            "prithvi_seg", "cot", "cloud_class".
        nmd_stats: Parsed nmd_stats.json content with class_stats
            and cross_reference.
        imint_summary: Parsed imint_summary.json content.
        image_shape: (height, width) of analysis images.
        date: Analysis date string (e.g. "2018-07-24").
        output_path: Output HTML file path.

    Returns:
        The output file path.
    """
    img_h, img_w = image_shape

    # Base64-encode all available images
    images_b64 = {}
    for key, path in image_paths.items():
        if os.path.exists(path):
            images_b64[key] = _img_to_base64(path)

    # Filter map viewers to those with available images
    active_viewers = [v for v in MAP_VIEWERS if v["key"] in images_b64]

    # Build HTML
    html = _build_html(
        viewers=active_viewers,
        images_b64=images_b64,
        nmd_stats=nmd_stats,
        imint_summary=imint_summary,
        img_h=img_h,
        img_w=img_w,
        date=date,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"    saved: {output_path}")
    return output_path


def _build_html(
    viewers: list[dict],
    images_b64: dict[str, str],
    nmd_stats: dict,
    imint_summary: dict,
    img_h: int,
    img_w: int,
    date: str,
) -> str:
    """Build the complete HTML string."""

    # ── Map cells HTML ────────────────────────────────────────────────────
    map_cells_html = ""
    for v in viewers:
        legend_html = ""
        if v["legend"]:
            items = "".join(
                f'<span class="legend-item">'
                f'<span class="legend-swatch" style="background:{it["color"]}"></span>'
                f'{it["label"]}</span>'
                for it in v["legend"]
            )
            legend_html = f'<div class="legend-strip">{items}</div>'

        map_cells_html += f"""
        <div class="map-cell">
            <div class="map-cell-header">
                <h3>{v["title"]}</h3>
                <div class="opacity-control">
                    <label for="opacity-{v["id"]}">Opacitet</label>
                    <input type="range" id="opacity-{v["id"]}" min="0" max="100"
                           value="100" data-map-id="{v["id"]}">
                    <span class="opacity-value" id="opacity-val-{v["id"]}">100%</span>
                </div>
            </div>
            <div id="{v["id"]}" class="map-container"></div>
            {legend_html}
        </div>"""

    # ── Analyzer summary cards ────────────────────────────────────────────
    summary_cards_html = _build_summary_cards(imint_summary)

    # ── Chart data JSON ───────────────────────────────────────────────────
    chart_data = _build_chart_data(nmd_stats)
    chart_data_json = json.dumps(chart_data, ensure_ascii=False)

    # ── Image data JS object ──────────────────────────────────────────────
    images_js_items = []
    for key, b64 in images_b64.items():
        images_js_items.append(f'    "{key}": "{b64}"')
    images_js = "{\n" + ",\n".join(images_js_items) + "\n}"

    # ── Viewer IDs and keys for JS ────────────────────────────────────────
    viewer_configs = json.dumps(
        [{"id": v["id"], "key": v["key"], "title": v["title"]} for v in viewers],
        ensure_ascii=False,
    )

    # ── Active chart sections (only emit containers for charts with data) ─
    chart_sections_html = ""
    if chart_data.get("spectral"):
        chart_sections_html += """
        <div class="chart-card">
            <h3>Spektralindex per markklass (NDVI / NDWI)</h3>
            <canvas id="chart-spectral"></canvas>
        </div>"""
    if chart_data.get("change"):
        chart_sections_html += """
        <div class="chart-card">
            <h3>Förändringsdetektering per markklass</h3>
            <canvas id="chart-change"></canvas>
        </div>"""
    if chart_data.get("prithvi"):
        chart_sections_html += """
        <div class="chart-card">
            <h3>Brandsegmentering per markklass (Prithvi)</h3>
            <canvas id="chart-prithvi"></canvas>
        </div>"""
    if chart_data.get("l2"):
        chart_sections_html += """
        <div class="chart-card">
            <h3>Marktäcke — detaljerade klasser (Nivå 2)</h3>
            <canvas id="chart-l2"></canvas>
        </div>"""

    # Wrap in grid if there are any charts
    if chart_sections_html:
        chart_sections_html = f"""
    <div class="section-header" id="charts-section">
        <h2>Korsreferens mot NMD (Nationellt Marktäckedata)</h2>
    </div>
    <div class="charts-grid">
        {chart_sections_html}
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="sv">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IMINT Analysrapport \u2014 {date}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0b0e17;
            color: #d8dae5;
            line-height: 1.5;
        }}

        /* ── Header ──────────────────────────────────────────────────── */
        .header {{
            background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
            padding: 20px 32px;
            border-bottom: 1px solid #334155;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .header-left h1 {{
            font-size: 22px;
            font-weight: 700;
            color: #f8fafc;
            letter-spacing: -0.3px;
        }}
        .header-left h1 span {{
            color: #3b82f6;
        }}
        .header-left p {{
            font-size: 13px;
            color: #94a3b8;
            margin-top: 2px;
        }}
        .header-nav {{
            display: flex;
            gap: 6px;
        }}
        .header-nav a {{
            text-decoration: none;
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            color: #94a3b8;
            background: #1e293b;
            border: 1px solid #334155;
            transition: all 0.15s;
        }}
        .header-nav a:hover {{
            color: #e2e8f0;
            border-color: #3b82f6;
            background: #1e3a5f;
        }}

        /* ── Summary cards ───────────────────────────────────────────── */
        .summary-section {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            padding: 16px 20px 8px;
        }}
        .summary-card {{
            background: #111827;
            border-radius: 10px;
            padding: 14px 18px;
            min-width: 170px;
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
            margin-bottom: 6px;
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

        /* ── Section header ──────────────────────────────────────────── */
        .section-header {{
            text-align: center;
            padding: 28px 20px 10px;
        }}
        .section-header h2 {{
            font-size: 17px;
            font-weight: 600;
            color: #e2e8f0;
            letter-spacing: -0.2px;
        }}

        /* ── Map grid ────────────────────────────────────────────────── */
        .map-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 10px;
            padding: 12px 20px;
        }}
        .map-cell {{
            background: #111827;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #1e293b;
        }}
        .map-cell-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 14px;
            height: 40px;
            background: #1e293b;
        }}
        .map-cell-header h3 {{
            font-size: 13px;
            font-weight: 600;
            color: #cbd5e1;
        }}
        .opacity-control {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .opacity-control label {{
            font-size: 10px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .opacity-control input[type="range"] {{
            width: 70px;
            height: 4px;
            -webkit-appearance: none;
            appearance: none;
            background: #334155;
            border-radius: 2px;
            outline: none;
        }}
        .opacity-control input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #3b82f6;
            cursor: pointer;
        }}
        .opacity-control input[type="range"]::-moz-range-thumb {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #3b82f6;
            cursor: pointer;
            border: none;
        }}
        .opacity-value {{
            font-size: 10px;
            color: #94a3b8;
            width: 30px;
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        .map-container {{
            height: 500px;
            background: #0b0e17;
        }}
        .legend-strip {{
            display: flex;
            gap: 8px;
            padding: 6px 14px;
            flex-wrap: wrap;
            font-size: 10px;
            background: #1e293b;
            border-top: 1px solid #263244;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 4px;
            color: #94a3b8;
        }}
        .legend-swatch {{
            width: 10px;
            height: 10px;
            border-radius: 2px;
            flex-shrink: 0;
        }}

        /* ── Charts ──────────────────────────────────────────────────── */
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 10px;
            padding: 10px 20px 32px;
        }}
        .chart-card {{
            background: #111827;
            border-radius: 10px;
            padding: 18px 20px;
            border: 1px solid #1e293b;
        }}
        .chart-card h3 {{
            font-size: 13px;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 14px;
            text-align: center;
        }}
        .chart-card canvas {{
            max-height: 360px;
        }}

        /* ── Footer ──────────────────────────────────────────────────── */
        .footer {{
            text-align: center;
            padding: 20px;
            font-size: 11px;
            color: #475569;
            border-top: 1px solid #1e293b;
        }}

        /* ── Leaflet overrides ───────────────────────────────────────── */
        .leaflet-container {{
            background: #0b0e17 !important;
        }}

        /* ── Responsive ──────────────────────────────────────────────── */
        @media (max-width: 900px) {{
            .map-grid {{ grid-template-columns: 1fr; }}
            .charts-grid {{ grid-template-columns: 1fr; }}
            .header {{ padding: 16px; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-left">
            <h1><span>IMINT</span> Analysrapport</h1>
            <p>Datum: {date} &middot; Bildstorlek: {img_w}\u00d7{img_h} px &middot; Sentinel-2 L2A</p>
        </div>
        <div class="header-nav">
            <a href="#maps-section">Kartor</a>
            <a href="#charts-section">Diagram</a>
        </div>
    </div>

    {summary_cards_html}

    <div class="section-header" id="maps-section">
        <h2>Analyslager</h2>
    </div>

    <div class="map-grid">
        {map_cells_html}
    </div>

    {chart_sections_html}

    <div class="footer">
        IMINT Engine &middot; Genererad {date}
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/leaflet.sync@0.2.4/L.Map.Sync.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"></script>
    <script>
    (function() {{
        'use strict';

        const IMG_H = {img_h};
        const IMG_W = {img_w};
        const IMAGES = {images_js};
        const VIEWERS = {viewer_configs};
        const CHART_DATA = {chart_data_json};

        // ── Create Leaflet maps ──────────────────────────────────────────
        const bounds = [[0, 0], [IMG_H, IMG_W]];
        const maps = [];
        const overlays = {{}};

        VIEWERS.forEach(function(v) {{
            const container = document.getElementById(v.id);
            if (!container || !IMAGES[v.key]) return;

            const map = L.map(v.id, {{
                crs: L.CRS.Simple,
                minZoom: -2,
                maxZoom: 5,
                attributionControl: false,
                zoomSnap: 0.25,
            }});

            const overlay = L.imageOverlay(IMAGES[v.key], bounds).addTo(map);
            map.fitBounds(bounds);
            maps.push(map);
            overlays[v.id] = overlay;
        }});

        // ── Synchronize all maps ─────────────────────────────────────────
        for (let i = 0; i < maps.length; i++) {{
            for (let j = 0; j < maps.length; j++) {{
                if (i !== j) {{
                    maps[i].sync(maps[j]);
                }}
            }}
        }}

        // ── Opacity sliders ──────────────────────────────────────────────
        document.querySelectorAll('.opacity-control input[type="range"]').forEach(function(slider) {{
            slider.addEventListener('input', function() {{
                const mapId = this.dataset.mapId;
                const val = parseInt(this.value);
                document.getElementById('opacity-val-' + mapId).textContent = val + '%';
                if (overlays[mapId]) {{
                    overlays[mapId].setOpacity(val / 100);
                }}
            }});
        }});

        // ── Chart.js defaults for dark theme ─────────────────────────────
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
        Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";

        // ── Chart 1: Spectral cross-reference ────────────────────────────
        if (CHART_DATA.spectral && CHART_DATA.spectral.labels.length > 0) {{
            new Chart(document.getElementById('chart-spectral'), {{
                type: 'bar',
                data: {{
                    labels: CHART_DATA.spectral.labels,
                    datasets: [
                        {{
                            label: 'Medel-NDVI',
                            data: CHART_DATA.spectral.ndvi,
                            backgroundColor: 'rgba(34,197,94,0.8)',
                            borderColor: '#22c55e',
                            borderWidth: 1,
                            borderRadius: 3,
                        }},
                        {{
                            label: 'Medel-NDWI',
                            data: CHART_DATA.spectral.ndwi,
                            backgroundColor: 'rgba(59,130,246,0.8)',
                            borderColor: '#3b82f6',
                            borderWidth: 1,
                            borderRadius: 3,
                        }},
                    ],
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ position: 'top' }},
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: false,
                            title: {{ display: true, text: 'Indexvärde' }},
                            grid: {{ color: 'rgba(255,255,255,0.04)' }},
                        }},
                        x: {{
                            grid: {{ display: false }},
                        }},
                    }},
                }},
            }});
        }}

        // ── Chart 2: Change detection per NMD class ──────────────────────
        if (CHART_DATA.change && CHART_DATA.change.labels.length > 0) {{
            new Chart(document.getElementById('chart-change'), {{
                type: 'bar',
                data: {{
                    labels: CHART_DATA.change.labels,
                    datasets: [{{
                        label: 'Förändringsandel (%)',
                        data: CHART_DATA.change.fractions,
                        backgroundColor: CHART_DATA.change.colors,
                        borderColor: CHART_DATA.change.borders,
                        borderWidth: 1,
                        borderRadius: 3,
                    }}],
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ display: false }},
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100,
                            title: {{ display: true, text: 'Andel förändrad (%)' }},
                            grid: {{ color: 'rgba(255,255,255,0.04)' }},
                        }},
                        x: {{
                            grid: {{ display: false }},
                        }},
                    }},
                }},
            }});
        }}

        // ── Chart 3: Prithvi burn/flood per NMD class ────────────────────
        if (CHART_DATA.prithvi && CHART_DATA.prithvi.labels.length > 0) {{
            const datasets = [];
            CHART_DATA.prithvi.classes.forEach(function(cls) {{
                datasets.push({{
                    label: cls.label,
                    data: cls.data,
                    backgroundColor: cls.color,
                    borderColor: cls.border,
                    borderWidth: 1,
                    borderRadius: 3,
                }});
            }});
            new Chart(document.getElementById('chart-prithvi'), {{
                type: 'bar',
                data: {{
                    labels: CHART_DATA.prithvi.labels,
                    datasets: datasets,
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ position: 'top' }},
                    }},
                    scales: {{
                        x: {{ stacked: true, grid: {{ display: false }} }},
                        y: {{
                            stacked: true,
                            beginAtZero: true,
                            max: 100,
                            title: {{ display: true, text: 'Andel (%)' }},
                            grid: {{ color: 'rgba(255,255,255,0.04)' }},
                        }},
                    }},
                }},
            }});
        }}

        // ── Chart 4: Level 2 land cover distribution ─────────────────────
        if (CHART_DATA.l2 && CHART_DATA.l2.labels.length > 0) {{
            new Chart(document.getElementById('chart-l2'), {{
                type: 'bar',
                data: {{
                    labels: CHART_DATA.l2.labels,
                    datasets: [{{
                        label: 'Andel (%)',
                        data: CHART_DATA.l2.fractions,
                        backgroundColor: CHART_DATA.l2.colors,
                        borderColor: CHART_DATA.l2.colors,
                        borderWidth: 1,
                        borderRadius: 3,
                    }}],
                }},
                options: {{
                    indexAxis: 'y',
                    responsive: true,
                    plugins: {{
                        legend: {{ display: false }},
                    }},
                    scales: {{
                        x: {{
                            beginAtZero: true,
                            title: {{ display: true, text: 'Andel (%)' }},
                            grid: {{ color: 'rgba(255,255,255,0.04)' }},
                        }},
                        y: {{
                            grid: {{ display: false }},
                        }},
                    }},
                }},
            }});
        }}

    }})();
    </script>
</body>
</html>"""

    return html


def _build_summary_cards(imint_summary: dict) -> str:
    """Build HTML for analyzer summary info cards."""
    analyzers = imint_summary.get("analyzers", [])
    if not analyzers:
        return ""

    cards = []

    for a in analyzers:
        name = a.get("name", "")
        outputs = a.get("outputs", {})

        if name == "change_detection":
            frac = outputs.get("change_fraction", 0)
            cards.append(
                '<div class="summary-card">'
                '<h4>Förändringsdetektering</h4>'
                f'<div class="value">{frac*100:.1f}%</div>'
                f'<div class="detail">{outputs.get("n_regions", 0)} regioner</div>'
                '</div>'
            )
        elif name == "spectral":
            stats = outputs.get("stats", {})
            veg = stats.get("vegetation_fraction", 0)
            cards.append(
                '<div class="summary-card">'
                '<h4>Spektralanalys</h4>'
                f'<div class="value">{veg*100:.1f}% veg.</div>'
                f'<div class="detail">Vatten: {stats.get("water_fraction",0)*100:.1f}%</div>'
                '</div>'
            )
        elif name == "prithvi":
            cs = outputs.get("class_stats", {})
            mode = a.get("metadata", {}).get("task_head", "")
            # Find the "positive" class (burned, water, etc.)
            for cls_id, cls_info in cs.items():
                cname = cls_info.get("name", "")
                if cname in ("burned", "water", "flood"):
                    frac = cls_info.get("fraction", 0)
                    cards.append(
                        '<div class="summary-card">'
                        f'<h4>Prithvi ({mode})</h4>'
                        f'<div class="value">{frac*100:.1f}% {cname}</div>'
                        f'<div class="detail">{cls_info.get("pixel_count", 0)} px</div>'
                        '</div>'
                    )
                    break
        elif name == "cot":
            stats = outputs.get("stats", {})
            clear = stats.get("clear_fraction", 0)
            cards.append(
                '<div class="summary-card">'
                '<h4>Molnanalys (COT)</h4>'
                f'<div class="value">{clear*100:.1f}% klart</div>'
                f'<div class="detail">COT medel: {stats.get("cot_mean", 0):.4f}</div>'
                '</div>'
            )
        elif name == "nmd":
            cs = outputs.get("class_stats", {})
            l1 = cs.get("level1", {})
            dom_name = ""
            dom_frac = 0
            for cname, cinfo in l1.items():
                if cinfo.get("fraction", 0) > dom_frac:
                    dom_frac = cinfo["fraction"]
                    dom_name = cname
            if dom_name:
                label = NMD_L1_CHART.get(dom_name, {}).get("label", dom_name)
                cards.append(
                    '<div class="summary-card">'
                    '<h4>NMD Marktäcke</h4>'
                    f'<div class="value">{dom_frac*100:.1f}% {label}</div>'
                    f'<div class="detail">{len(l1)} klasser</div>'
                    '</div>'
                )

    if not cards:
        return ""

    return '<div class="summary-section">' + "".join(cards) + "</div>"


def _build_chart_data(nmd_stats: dict) -> dict:
    """Extract chart data from nmd_stats JSON structure."""
    cross_ref = nmd_stats.get("cross_reference", {})
    class_stats = nmd_stats.get("class_stats", {})

    chart_data = {}

    # ── Spectral chart (Level 2) ─────────────────────────────────────────
    spectral = cross_ref.get("spectral", {})
    if spectral:
        labels = []
        ndvi_vals = []
        ndwi_vals = []
        for key in L2_ORDER:
            if key in spectral:
                info = NMD_L2_CHART.get(key, {"label": key})
                labels.append(info["label"])
                ndvi_vals.append(round(spectral[key].get("mean_ndvi", 0), 4))
                ndwi_vals.append(round(spectral[key].get("mean_ndwi", 0), 4))
        chart_data["spectral"] = {
            "labels": labels,
            "ndvi": ndvi_vals,
            "ndwi": ndwi_vals,
        }

    # ── Change detection chart (Level 2) ──────────────────────────────────
    change = cross_ref.get("change_detection", {})
    if change:
        labels = []
        fractions = []
        colors = []
        borders = []
        for key in L2_ORDER:
            if key in change:
                info = NMD_L2_CHART.get(key, {"label": key, "color": "gray", "border": "gray"})
                labels.append(info["label"])
                fractions.append(round(change[key].get("change_fraction", 0) * 100, 1))
                colors.append(info["color"])
                borders.append(info["border"])
        chart_data["change"] = {
            "labels": labels,
            "fractions": fractions,
            "colors": colors,
            "borders": borders,
        }

    # ── Prithvi cross-reference chart (Level 2) ──────────────────────────
    prithvi = cross_ref.get("prithvi", {})
    if prithvi:
        labels = []
        # Discover class names from first entry
        first_entry = next(iter(prithvi.values()))
        class_keys = []
        for k in first_entry:
            if k.endswith("_fraction"):
                class_keys.append(k.replace("_fraction", ""))

        class_datasets = {cn: [] for cn in class_keys}
        for key in L2_ORDER:
            if key in prithvi:
                info = NMD_L2_CHART.get(key, {"label": key})
                labels.append(info["label"])
                for cn in class_keys:
                    frac = prithvi[key].get(f"{cn}_fraction", 0)
                    class_datasets[cn].append(round(frac * 100, 1))

        # Build dataset configs
        class_colors = {
            "burned":   {"color": "rgba(255,69,0,0.85)", "border": "#FF4500", "label": "Bränt"},
            "no_burn":  {"color": "rgba(34,139,34,0.85)", "border": "#228B22", "label": "Ej bränt"},
            "water":    {"color": "rgba(0,0,255,0.85)", "border": "#0000FF", "label": "Översvämmat"},
            "no_water": {"color": "rgba(139,90,43,0.85)", "border": "#8B5A2B", "label": "Ej översvämmat"},
            "flood":    {"color": "rgba(0,0,255,0.85)", "border": "#0000FF", "label": "Översvämmat"},
        }
        classes_list = []
        for cn in class_keys:
            cc = class_colors.get(cn, {"color": "rgba(128,128,128,0.85)", "border": "#808080", "label": cn})
            classes_list.append({
                "label": cc["label"],
                "data": class_datasets[cn],
                "color": cc["color"],
                "border": cc["border"],
            })

        chart_data["prithvi"] = {
            "labels": labels,
            "classes": classes_list,
        }

    # ── Level 2 distribution chart ────────────────────────────────────────
    l2_stats = class_stats.get("level2", {})
    if l2_stats:
        # Sort by fraction descending, skip tiny
        sorted_l2 = sorted(
            [(k, v) for k, v in l2_stats.items() if v.get("fraction", 0) > 0.001],
            key=lambda x: x[1]["fraction"],
            reverse=True,
        )
        labels = []
        fractions = []
        colors = []
        for key, stats in sorted_l2:
            info = NMD_L2_CHART.get(key, {"label": key, "color": "#808080"})
            labels.append(info["label"])
            fractions.append(round(stats["fraction"] * 100, 1))
            colors.append(info["color"])
        chart_data["l2"] = {
            "labels": labels,
            "fractions": fractions,
            "colors": colors,
        }

    return chart_data
