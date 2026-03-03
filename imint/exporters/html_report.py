"""
imint/exporters/html_report.py — Interactive HTML analysis report

Two report types:

  save_html_report()    Single-run report with base64-embedded images
                        (self-contained, used per pipeline run).

  save_tabbed_report()  Multi-tab showcase with EXTERNAL image files
                        (images in showcase/ subdirectory, used for demos).

Common features:
- Synchronized zoomable Leaflet.js map viewers (CRS.Simple + leaflet-sync)
- Opacity sliders per layer
- Chart.js charts for NMD cross-reference data
- All JS/CSS libraries inlined (no external CDN dependencies)
"""
from __future__ import annotations

import os
import re
import json
import base64
import shutil
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

# Project root for library cache
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LIB_CACHE_DIR = _PROJECT_ROOT / ".lib_cache"

# CDN URLs for external libraries
_CDN_LIBS = {
    "leaflet_css": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",
    "leaflet_js": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js",
    "leaflet_sync_js": "https://cdn.jsdelivr.net/npm/leaflet.sync@0.2.4/L.Map.Sync.js",
    "chart_js": "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js",
}


def _fetch_lib(url: str, cache_dir: Path | None = None) -> str:
    """Download a library file and cache it locally.

    Args:
        url: CDN URL to fetch.
        cache_dir: Local cache directory (default: .lib_cache/).

    Returns:
        File content as string.
    """
    if cache_dir is None:
        cache_dir = _LIB_CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rsplit("/", 1)[-1]
    cached = cache_dir / filename

    if cached.exists():
        return cached.read_text(encoding="utf-8")

    try:
        print(f"    [html_report] Downloading {filename} ...")
        with urlopen(url, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        cached.write_text(content, encoding="utf-8")
        print(f"    [html_report] Cached: {cached}")
        return content
    except (URLError, OSError) as e:
        print(f"    [html_report] WARNING: Could not download {url}: {e}")
        return f"/* Failed to fetch {url} */"


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
        "id": "map-dnbr",
        "title": "dNBR (Brandsvårighetsgrad)",
        "key": "dnbr",
        "legend": [
            {"color": "#1a9850", "label": "Hög återväxt (< -0.25)"},
            {"color": "#91cf60", "label": "Låg återväxt (-0.25 – -0.1)"},
            {"color": "#d9ef8b", "label": "Obränt (-0.1 – 0.1)"},
            {"color": "#fee08b", "label": "Låg svårighetsgrad (0.1 – 0.27)"},
            {"color": "#fdae61", "label": "Måttligt låg (0.27 – 0.44)"},
            {"color": "#f46d43", "label": "Måttligt hög (0.44 – 0.66)"},
            {"color": "#d73027", "label": "Hög svårighetsgrad (> 0.66)"},
        ],
    },
    {
        "id": "map-change-gradient",
        "title": "Förändring (gradient)",
        "key": "change_gradient",
        "legend": [
            {"color": "#FFFFB2", "label": "Liten förändring"},
            {"color": "#FD8D3C", "label": "Måttlig förändring"},
            {"color": "#BD0026", "label": "Stor förändring"},
        ],
    },
    {
        "id": "map-ndwi",
        "title": "NDWI (Vattenindex)",
        "key": "ndwi",
        "legend": [
            {"color": "#67001f", "label": "-1.0"},
            {"color": "#d6604d", "label": "-0.5"},
            {"color": "#f7f7f7", "label": "0.0"},
            {"color": "#4393c3", "label": "0.5"},
            {"color": "#053061", "label": "1.0 Vatten"},
        ],
    },
    {
        "id": "map-ndbi",
        "title": "NDBI (Bebyggelseindex)",
        "key": "ndbi",
        "legend": [
            {"color": "#313695", "label": "-1.0"},
            {"color": "#abd9e9", "label": "-0.5"},
            {"color": "#ffffbf", "label": "0.0"},
            {"color": "#fdae61", "label": "0.5"},
            {"color": "#a50026", "label": "1.0 Bebyggt"},
        ],
    },
    {
        "id": "map-evi",
        "title": "EVI (Enhanced Vegetation Index)",
        "key": "evi",
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
            {"color": "#FFFFB2", "label": "0 (Klart)"},
            {"color": "#FD8D3C", "label": "0.015 (Tunt moln)"},
            {"color": "#BD0026", "label": "0.05 (Tjockt moln)"},
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

    # ── Fetch and cache external libraries for inline embedding ───────────
    leaflet_css = _fetch_lib(_CDN_LIBS["leaflet_css"])
    leaflet_js = _fetch_lib(_CDN_LIBS["leaflet_js"])
    leaflet_sync_js = _fetch_lib(_CDN_LIBS["leaflet_sync_js"])
    chart_js = _fetch_lib(_CDN_LIBS["chart_js"])

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
    if chart_data.get("change"):
        chart_sections_html += """
        <div class="chart-card">
            <h3>Förändringsdetektering per markklass</h3>
            <canvas id="chart-change"></canvas>
        </div>
        <div class="chart-card">
            <h3>dNBR brandsvårighetsgrad per markklass</h3>
            <canvas id="chart-dnbr"></canvas>
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
    <style>{leaflet_css}</style>
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

    <script>{leaflet_js}</script>
    <script>{leaflet_sync_js}</script>
    <script>{chart_js}</script>
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

            // Add RGB as background layer (except for the RGB panel itself)
            if (v.key !== 'rgb' && IMAGES['rgb']) {{
                L.imageOverlay(IMAGES['rgb'], bounds, {{zIndex: 0}}).addTo(map);
            }}

            const overlay = L.imageOverlay(IMAGES[v.key], bounds, {{zIndex: 1}}).addTo(map);
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

        // ── Chart: Change detection per NMD class ──────────────────────
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

        // ── Chart 2b: dNBR severity per NMD class ──────────────────────
        if (CHART_DATA.change && CHART_DATA.change.dnbr && CHART_DATA.change.labels.length > 0) {{
            // Color bars by USGS severity class
            const dnbrColors = CHART_DATA.change.dnbr.map(function(v) {{
                if (v < -0.25) return 'rgba(26,152,80,0.85)';
                if (v < -0.1)  return 'rgba(145,207,96,0.85)';
                if (v < 0.1)   return 'rgba(217,239,139,0.85)';
                if (v < 0.27)  return 'rgba(254,224,139,0.85)';
                if (v < 0.44)  return 'rgba(253,174,97,0.85)';
                if (v < 0.66)  return 'rgba(244,109,67,0.85)';
                return 'rgba(215,48,39,0.85)';
            }});
            new Chart(document.getElementById('chart-dnbr'), {{
                type: 'bar',
                data: {{
                    labels: CHART_DATA.change.labels,
                    datasets: [{{
                        label: 'Medel-dNBR',
                        data: CHART_DATA.change.dnbr,
                        backgroundColor: dnbrColors,
                        borderColor: dnbrColors.map(c => c.replace('0.85', '1')),
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
                            title: {{ display: true, text: 'dNBR' }},
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
            meta = a.get("metadata", {})
            severity = meta.get("dnbr_severity", {})
            high_frac = severity.get("high_severity", 0)
            if high_frac:
                n_pixels = meta.get("valid_pixels", 0)
                # 10m × 10m Sentinel-2 pixels → km²
                area_km2 = high_frac * n_pixels * 0.0001
                cards.append(
                    '<div class="summary-card">'
                    '<h4>dNBR Hög svårighetsgrad</h4>'
                    f'<div class="value">{area_km2:.1f} km²</div>'
                    f'<div class="detail">{high_frac*100:.1f}% av området</div>'
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


def _build_marine_summary_cards(marine_dir: str, prefix: str,
                                 imint_summary: dict) -> str:
    """Build HTML summary cards for the marine analysis tab."""
    cards = []

    # ── Vessel heatmap summary (multi-date) ────────────────────────────
    heatmap_path = os.path.join(marine_dir, f"{prefix}vessel_heatmap_summary.json")
    if os.path.isfile(heatmap_path):
        with open(heatmap_path) as f:
            hs = json.load(f)
        total = hs.get("total_detections", 0)
        dates_used = len(hs.get("dates_used", []))
        dates_skip = len(hs.get("dates_skipped", []))
        cards.append(
            '<div class="summary-card">'
            '<h4>Båtdetektering</h4>'
            f'<div class="value">{total} båtar</div>'
            f'<div class="detail">{dates_used} datum ({dates_skip} skippade)</div>'
            '</div>'
        )
        # Best single-date count
        per_date = hs.get("per_date", [])
        if per_date:
            best = max(per_date, key=lambda d: d.get("vessels", 0))
            cards.append(
                '<div class="summary-card">'
                '<h4>Bästa datum</h4>'
                f'<div class="value">{best["vessels"]} båtar</div>'
                f'<div class="detail">{best["date"]}</div>'
                '</div>'
            )

    # ── NMD water coverage ─────────────────────────────────────────────
    analyzers = imint_summary.get("analyzers", [])
    for a in analyzers:
        name = a.get("name", "")
        outputs = a.get("outputs", {})

        if name == "nmd":
            cs = outputs.get("class_stats", {})
            l1 = cs.get("level1", {})
            water = l1.get("water", {})
            if water:
                cards.append(
                    '<div class="summary-card">'
                    '<h4>NMD Marktäcke</h4>'
                    f'<div class="value">{water.get("fraction",0)*100:.1f}% Vatten</div>'
                    f'<div class="detail">{len(l1)} klasser</div>'
                    '</div>'
                )

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

        elif name == "marine_vessels":
            meta = a.get("metadata", {})
            area = meta.get("area_km2", 0)
            if area:
                cards.append(
                    '<div class="summary-card">'
                    '<h4>Analysområde</h4>'
                    f'<div class="value">{area:.1f} km²</div>'
                    f'<div class="detail">Bohuslän kustzon</div>'
                    '</div>'
                )

    if not cards:
        return ""

    return '<div class="summary-section">' + "".join(cards) + "</div>"


def _build_grazing_summary_cards(grazing_dir: str) -> str:
    """Build HTML summary cards for the grazing/pasture tab.

    Reads ``grazing_meta.json`` from *grazing_dir* which should contain
    NMD statistics within LPIS polygons, NDVI means, area, grazing
    model predictions, etc.
    """
    meta_path = os.path.join(grazing_dir, "grazing_meta.json")
    if not os.path.isfile(meta_path):
        return ""

    with open(meta_path) as f:
        meta = json.load(f)

    cards = []

    # Card 1: Grazing model results (most important — show first)
    gp = meta.get("grazing_predictions", {})
    if gp:
        n_active = gp.get("active_grazing", 0)
        total = gp.get("total_polygons", 0)
        conf = gp.get("mean_confidence", 0)
        cards.append(
            '<div class="summary-card">'
            '<h4>Betesanalys (AI)</h4>'
            f'<div class="value">{n_active}/{total} aktiv</div>'
            f'<div class="detail">Konfidens: {conf*100:.0f}%</div>'
            '</div>'
        )
        n_inactive = gp.get("no_activity", 0)
        if n_inactive:
            cards.append(
                '<div class="summary-card">'
                '<h4>Ingen aktivitet</h4>'
                f'<div class="value">{n_inactive} block</div>'
                f'<div class="detail">{gp.get("num_dates", 0)} molnfria datum</div>'
                '</div>'
            )

    # Card 2: Number of betesblock + area
    lpis_count = meta.get("lpis_count", 0)
    area_ha = meta.get("lpis_total_area_ha", 0)
    if lpis_count:
        cards.append(
            '<div class="summary-card">'
            '<h4>LPIS Betesblock</h4>'
            f'<div class="value">{lpis_count} block</div>'
            f'<div class="detail">{area_ha:.0f} ha total areal</div>'
            '</div>'
        )

    # Card 3: NDVI mean inside blocks
    ndvi_mean = meta.get("ndvi_mean_inside", 0)
    ndvi_std = meta.get("ndvi_std_inside", 0)
    if ndvi_mean > 0:
        cards.append(
            '<div class="summary-card">'
            '<h4>NDVI i betesmark</h4>'
            f'<div class="value">{ndvi_mean:.2f}</div>'
            f'<div class="detail">\u00b1 {ndvi_std:.2f} standardavvikelse</div>'
            '</div>'
        )

    # Card 4: Dominant NMD class inside blocks
    nmd_within = meta.get("nmd_within_lpis", {})
    if nmd_within:
        top_name = next(iter(nmd_within.keys()))
        top = nmd_within[top_name]
        top_frac = top.get("fraction", 0)
        cards.append(
            '<div class="summary-card">'
            '<h4>NMD inom betesblock</h4>'
            f'<div class="value">{top_frac*100:.0f}% {top_name}</div>'
            f'<div class="detail">{len(nmd_within)} markklasser</div>'
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

    # ── Change detection chart (Level 2) ──────────────────────────────────
    change = cross_ref.get("change_detection", {})
    if change:
        labels = []
        fractions = []
        dnbr_vals = []
        colors = []
        borders = []
        for key in L2_ORDER:
            if key in change:
                info = NMD_L2_CHART.get(key, {"label": key, "color": "gray", "border": "gray"})
                labels.append(info["label"])
                fractions.append(round(change[key].get("change_fraction", 0) * 100, 1))
                dnbr_vals.append(round(change[key].get("mean_dnbr", 0), 4))
                colors.append(info["color"])
                borders.append(info["border"])
        chart_data["change"] = {
            "labels": labels,
            "fractions": fractions,
            "dnbr": dnbr_vals,
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


# ── Tabbed multi-dataset showcase report ──────────────────────────────────────

# Viewer definitions per tab (key must match filename pattern in output dir)
_FIRE_VIEWERS = [
    {"id": "f-rgb",  "title": "Sentinel-2 RGB", "key": "rgb", "legend": []},
    {"id": "f-nmd",  "title": "NMD Marktäcke",  "key": "nmd", "legend": [
        {"color": "#006400", "label": "Tallskog"}, {"color": "#228B22", "label": "Granskog"},
        {"color": "#32CD32", "label": "Lövskog"},  {"color": "#3CB371", "label": "Blandskog"},
        {"color": "#90EE90", "label": "Temp. ej skog"},
        {"color": "#2E4F2E", "label": "Sumpsk. tall"}, {"color": "#3A5F3A", "label": "Sumpsk. gran"},
        {"color": "#4A7F4A", "label": "Sumpsk. löv"},  {"color": "#5A8F5A", "label": "Sumpsk. bland"},
        {"color": "#7AAF7A", "label": "Sumpsk. temp"}, {"color": "#8B5A2B", "label": "Öpp. våtmark"},
        {"color": "#FFD700", "label": "Åkermark"},      {"color": "#C8AD7F", "label": "Öpp. mark bar"},
        {"color": "#D2B48C", "label": "Öpp. mark veg."},{"color": "#FF0000", "label": "Byggnader"},
        {"color": "#FF4500", "label": "Infrastruktur"}, {"color": "#FF6347", "label": "Vägar"},
        {"color": "#0000FF", "label": "Sjöar"},          {"color": "#1E90FF", "label": "Hav"},
    ]},
    {"id": "f-ndvi", "title": "NDVI (Vegetationsindex)", "key": "ndvi", "legend": [
        {"color": "#a50026", "label": "-1.0"}, {"color": "#f46d43", "label": "-0.5"},
        {"color": "#fee08b", "label": "0.0"},  {"color": "#a6d96a", "label": "0.5"},
        {"color": "#006837", "label": "1.0"},
    ]},
    {"id": "f-ndwi", "title": "NDWI (Vattenindex)", "key": "ndwi", "legend": [
        {"color": "#67001f", "label": "-1.0"}, {"color": "#d6604d", "label": "-0.5"},
        {"color": "#f7f7f7", "label": "0.0"},  {"color": "#4393c3", "label": "0.5"},
        {"color": "#053061", "label": "1.0 Vatten"},
    ]},
    {"id": "f-evi",  "title": "EVI (Enhanced Vegetation Index)", "key": "evi", "legend": [
        {"color": "#a50026", "label": "-1.0"}, {"color": "#f46d43", "label": "-0.5"},
        {"color": "#fee08b", "label": "0.0"},  {"color": "#a6d96a", "label": "0.5"},
        {"color": "#006837", "label": "1.0"},
    ]},
    {"id": "f-cot",  "title": "Molnoptisk tjocklek (COT)", "key": "cot", "legend": [
        {"color": "#FFFFB2", "label": "0 (Klart)"},
        {"color": "#FD8D3C", "label": "0.015 (Tunt moln)"},
        {"color": "#BD0026", "label": "0.05 (Tjockt moln)"},
    ]},
    {"id": "f-dnbr", "title": "dNBR (Brandsvårighetsgrad)", "key": "dnbr", "legend": [
        {"color": "#1a9850", "label": "Hög återväxt (< -0.25)"},
        {"color": "#91cf60", "label": "Låg återväxt (-0.25 – -0.1)"},
        {"color": "#d9ef8b", "label": "Obränt (-0.1 – 0.1)"},
        {"color": "#fee08b", "label": "Låg svårighetsgrad (0.1 – 0.27)"},
        {"color": "#fdae61", "label": "Måttligt låg (0.27 – 0.44)"},
        {"color": "#f46d43", "label": "Måttligt hög (0.44 – 0.66)"},
        {"color": "#d73027", "label": "Hög svårighetsgrad (> 0.66)"},
    ]},
    {"id": "f-gradient", "title": "Förändring (gradient)", "key": "change_gradient", "legend": [
        {"color": "#FFFFB2", "label": "Liten förändring"},
        {"color": "#FD8D3C", "label": "Måttlig förändring"},
        {"color": "#BD0026", "label": "Stor förändring"},
    ]},
    {"id": "f-prithvi", "title": "Prithvi Segmentering", "key": "prithvi_seg", "legend": [
        {"color": "#228B22", "label": "Ej bränt"},
        {"color": "#FF4500", "label": "Bränt"},
    ]},
]

_MARINE_VIEWERS = [
    {"id": "m-rgb",  "title": "Sentinel-2 RGB", "key": "rgb", "legend": []},
    {"id": "m-vessels", "title": "Båtdetektering (YOLO)", "key": "vessels", "vector": True, "legend": [
        {"color": "#00E5FF", "label": "Detekterad båt / anomali"},
    ]},
    {"id": "m-vessel-heatmap", "title": "Båtaktivitet (heatmap)", "key": "vessel_heatmap", "legend": [
        {"color": "#FFFFB2", "label": "Låg"},
        {"color": "#FD8D3C", "label": "Medel"},
        {"color": "#BD0026", "label": "Hög"},
    ]},
    {"id": "m-nmd",  "title": "NMD Marktäcke",  "key": "nmd", "legend": [
        {"color": "#006400", "label": "Tallskog"}, {"color": "#228B22", "label": "Granskog"},
        {"color": "#32CD32", "label": "Lövskog"},  {"color": "#3CB371", "label": "Blandskog"},
        {"color": "#90EE90", "label": "Temp. ej skog"},
        {"color": "#2E4F2E", "label": "Sumpsk. tall"}, {"color": "#3A5F3A", "label": "Sumpsk. gran"},
        {"color": "#4A7F4A", "label": "Sumpsk. löv"},  {"color": "#5A8F5A", "label": "Sumpsk. bland"},
        {"color": "#7AAF7A", "label": "Sumpsk. temp"}, {"color": "#8B5A2B", "label": "Öpp. våtmark"},
        {"color": "#FFD700", "label": "Åkermark"},      {"color": "#C8AD7F", "label": "Öpp. mark bar"},
        {"color": "#D2B48C", "label": "Öpp. mark veg."},{"color": "#FF0000", "label": "Byggnader"},
        {"color": "#FF4500", "label": "Infrastruktur"}, {"color": "#FF6347", "label": "Vägar"},
        {"color": "#0000FF", "label": "Sjöar"},          {"color": "#1E90FF", "label": "Hav"},
    ]},
    {"id": "m-ndvi", "title": "NDVI (Vegetationsindex)", "key": "ndvi", "legend": [
        {"color": "#a50026", "label": "-1.0"}, {"color": "#f46d43", "label": "-0.5"},
        {"color": "#fee08b", "label": "0.0"},  {"color": "#a6d96a", "label": "0.5"},
        {"color": "#006837", "label": "1.0"},
    ]},
    {"id": "m-ndwi", "title": "NDWI (Vattenindex)", "key": "ndwi", "legend": [
        {"color": "#67001f", "label": "-1.0"}, {"color": "#d6604d", "label": "-0.5"},
        {"color": "#f7f7f7", "label": "0.0"},  {"color": "#4393c3", "label": "0.5"},
        {"color": "#053061", "label": "1.0 Vatten"},
    ]},
    {"id": "m-cot",  "title": "Molnoptisk tjocklek (COT)", "key": "cot", "legend": [
        {"color": "#FFFFB2", "label": "0 (Klart)"},
        {"color": "#FD8D3C", "label": "0.015 (Tunt moln)"},
        {"color": "#BD0026", "label": "0.05 (Tjockt moln)"},
    ]},
]

_GRAZING_VIEWERS = [
    {"id": "g-rgb",  "title": "Sentinel-2<br>RGB", "key": "rgb", "legend": []},
    {"id": "g-nmd",  "title": "NMD<br>Marktäcke",  "key": "nmd", "legend": [
        {"color": "#FFD700", "label": "Åkermark"},
        {"color": "#D2B48C", "label": "Öpp. mark veg."},
        {"color": "#50B43C", "label": "Ädellövskog"},  {"color": "#32CD32", "label": "Triviallöv"},
        {"color": "#228B22", "label": "Granskog"},      {"color": "#46A064", "label": "Blandskog"},
        {"color": "#006400", "label": "Tallskog"},
        {"color": "#4A7F4A", "label": "Skog våtmark"},
        {"color": "#8B5A2B", "label": "Öpp. våtmark"},
        {"color": "#FF0000", "label": "Bebyggelse"},
        {"color": "#0000FF", "label": "Vatten"},
    ]},
    {"id": "g-lpis", "title": "LPIS<br>Betesblock", "key": "lpis", "vector": True, "legend": [
        {"color": "#00BFFF", "label": "Aktiv betesmark"},
        {"color": "#E6119D", "label": "Ingen aktivitet"},
        {"color": "#aaaaaa", "label": "Ej analyserad"},
    ]},
    {"id": "g-ndvi", "title": "NDVI<br>Vegetationsindex", "key": "ndvi", "legend": [
        {"color": "#a50026", "label": "-1.0"}, {"color": "#f46d43", "label": "-0.5"},
        {"color": "#fee08b", "label": "0.0"},  {"color": "#a6d96a", "label": "0.5"},
        {"color": "#006837", "label": "1.0"},
    ]},
    {"id": "g-evi",  "title": "EVI<br>Vegetationsindex", "key": "evi", "legend": [
        {"color": "#a50026", "label": "-0.5"}, {"color": "#fee08b", "label": "0.0"},
        {"color": "#a6d96a", "label": "0.5"},   {"color": "#006837", "label": "1.0"},
    ]},
    {"id": "g-ndwi", "title": "NDWI<br>Vattenindex", "key": "ndwi", "legend": [
        {"color": "#67001f", "label": "-1.0"}, {"color": "#d6604d", "label": "-0.5"},
        {"color": "#f7f7f7", "label": "0.0"},  {"color": "#4393c3", "label": "0.5"},
        {"color": "#053061", "label": "1.0 Vatten"},
    ]},
    {"id": "g-cot",  "title": "COT<br>Molnoptisk tjocklek", "key": "cot", "legend": [
        {"color": "#FFFFB2", "label": "0 (Klart)"},
        {"color": "#FD8D3C", "label": "0.015 (Tunt moln)"},
        {"color": "#BD0026", "label": "0.05 (Tjockt moln)"},
    ]},
]

# ── Analysis descriptions — single source of truth ───────────────────────────
# Each entry has a shared technical description ("body") + per-context notes.
# The helper _render_descriptions() builds HTML for a given tab.

_ANALYSIS_DESCRIPTIONS = {
    "sentinel2_rgb": {
        "title": "Sentinel-2 RGB — Satellitbilden",
        "body": (
            "Sentinel-2 är en konstellation av två satelliter (2A och 2B) som drivs av "
            "Europeiska rymdorganisationen (ESA) inom Copernicus-programmet. Satelliterna "
            "kretsar i en solsynkron bana på 786 km höjd och avbildar hela jorden "
            "var femte dag med en upplösning på 10 meter per pixel för de synliga "
            "banden. RGB-bilden visar området som det ser ut för ögat, med band 4 (rött), "
            "band 3 (grönt) och band 2 (blått)."
        ),
        "fire_note": (
            "De bruna och grå områdena är brandskadat landskap där vegetationen har förstörts."
        ),
        "marine_note": (
            "I kustmiljön syns land, öar, holmar och öppet vatten — och vid god sikt "
            "kan enskilda båtar och deras kölvatten urskiljas i bilden."
        ),
        "grazing_note": (
            "I betesmarksanalysen visar RGB-bilden jordbrukslandskapet i Skåne "
            "med blandning av åkermark, beteshagar och skogspartier. De gröna "
            "fälten i maj indikerar aktiv betessäsong."
        ),
        "ref": (
            '<em>Källa: <a href="https://sentinel.esa.int/web/sentinel/missions/sentinel-2" '
            'target="_blank">ESA Sentinel-2</a></em>'
        ),
    },
    "ndvi": {
        "title": "NDVI — Vegetationsindex",
        "body": (
            "NDVI (Normalized Difference Vegetation Index) är det mest använda vegetationsindexet "
            "inom fjärranalys. Det beräknas som (NIR − Röd) / (NIR + Röd) där NIR "
            "är det nära infraröda bandet (B08) och Röd är det synliga röda bandet (B04). "
            "Frisk vegetation reflekterar starkt i NIR och absorberar rött ljus, vilket ger "
            "höga NDVI-värden (0.3–0.9)."
        ),
        "fire_note": (
            "Brandskadat eller vegetationslöst område ger låga värden nära noll. "
            "På kartan visas högt NDVI i grönt (frisk skog) och lågt i rött/gult (skadad mark)."
        ),
        "marine_note": (
            "Vatten och bar mark ger värden nära eller under noll. I kustanalysen "
            "används NDVI för att skilja vegetationsklädda öar och strandremsor från "
            "kala klippor, vatten och bebyggelse — och för att bedöma "
            "kustvegetationens hälsotillstånd."
        ),
        "grazing_note": (
            "I betesmarksanalysen är NDVI centralt för att bedöma betesmarkens "
            "vitalitet. Aktivt betade marker visar typiskt NDVI 0.4–0.7, medan "
            "obetade gräsmarker ofta når 0.7–0.9. Genom att jämföra NDVI-värden "
            "inuti och utanför LPIS-polygonerna kan betestrycket uppskattas."
        ),
        "ref": (
            '<em>Källa: Rouse et al., 1974. &quot;Monitoring vegetation systems in the '
            'Great Plains with ERTS.&quot; Third Earth Resources Technology Satellite-1 '
            'Symposium, NASA SP-351.</em>'
        ),
    },
    "ndwi": {
        "title": "NDWI — Vattenindex",
        "body": (
            "NDWI (Normalized Difference Water Index) mäter förekomsten av vatten och "
            "fuktighet i landskapet. Det beräknas som (Grön − NIR) / (Grön + NIR) "
            "med band B03 och B08. Positiva värden (blått på kartan) indikerar "
            "öppet vatten, medan negativa värden visar torr mark."
        ),
        "fire_note": (
            "Sjöar och vattendrag syns tydligt som blå områden. "
            "Indexet är användbart för att identifiera hur en brand påverkat "
            "markfuktigheten och vattenbalansen i området."
        ),
        "marine_note": (
            "I den marina analysen ger NDWI en tydlig bild av gränsen mellan "
            "land och vatten samt hjälper till att identifiera grunda vattenområden, "
            "inlopp och vikar. Indexet kompletterar NMD-landmasken genom att visa "
            "vattenytor med högre detaljeringsgrad."
        ),
        "grazing_note": (
            "I beteslandskapet ger NDWI information om markfuktigheten, "
            "vilket är avgörande för beteskvalitet. Våta betesmarker kan indikera "
            "dränerings\u00adproblem eller naturliga våtmarker inom blockgränserna."
        ),
        "ref": (
            '<em>Källa: McFeeters, S.K., 1996. &quot;The use of the Normalized '
            'Difference Water Index (NDWI) in the delineation of open water '
            'features.&quot; Int. J. Remote Sensing, 17(7).</em>'
        ),
    },
    "evi": {
        "title": "EVI — Förbättrat vegetationsindex",
        "body": (
            "EVI (Enhanced Vegetation Index) är ett vidareutvecklat vegetationsindex som "
            "korrigerar för atmosfäriska störningar och markens bakgrundsreflektion. "
            "Till skillnad från NDVI mättas inte EVI lika lätt i områden med tät vegetation, "
            "vilket ger mer nyanserad information i frodiga skogsområden. EVI "
            "använder tre band — blått (B02), rött (B04) och NIR (B08) — för att "
            "skatta vegetationens tillstånd mer robust än NDVI ensamt."
        ),
        "grazing_note": (
            "EVI ger en bättre bild av betesmarkens biomassa än NDVI i frodiga "
            "områden. Eftersom EVI inte mättas vid hög vegetationstäthet kan det "
            "skilja på nyligen betade hagar (lägre EVI) och obetade fält (högre EVI)."
        ),
        "ref": (
            '<em>Källa: Huete et al., 2002. &quot;Overview of the radiometric and biophysical '
            'performance of the MODIS vegetation indices.&quot; Remote Sensing of '
            'Environment, 83(1-2).</em>'
        ),
    },
    "cot": {
        "title": "COT — Molnoptisk tjocklek",
        "body": (
            "COT (Cloud Optical Thickness) anger hur optiskt tjockt molntäcket är "
            "över analysområdet. Värden nära noll (ljusgult) innebär klar himmel "
            "och tillförlitliga satellitdata, medan högre värden (orange till rött) "
            "visar molniga områden där underliggande mark eller hav inte kan ses. "
            "Modellen är en ensemble av fem MLP5-nätverk (fem lager djupa neurala nät) "
            "som tränats på syntetiska molndata från SMHI. Varje nätverk tar emot "
            "11 Sentinel-2-band (B02–B12, exklusive B01 och B10) och predikterar ett "
            "kontinuerligt COT-värde per pixel. Genom att använda medelvärdet av alla "
            "fem modellers prediktioner fås ett robust och brusreducerat resultat."
        ),
        "fire_note": (
            "Molnanalysen används här för att säkerställa att det valda datumet har "
            "tillräckligt molnfria förhållanden för en pålitlig analys."
        ),
        "marine_note": (
            "Molnanalysen är särskilt viktig i den marina kedjan: om ett datum "
            "har för hög molntäckning exkluderas det automatiskt från "
            "heatmap-ackumuleringen för att undvika att moln feldetekteras som båtar."
        ),
        "grazing_note": (
            "I betesmarksanalysen används COT för att verifiera att de analyserade "
            "Sentinel-2-scenerna är molnfria. Pixlar med COT > 0.015 maskeras "
            "automatiskt före beräkning av vegetationsindex."
        ),
        "ref": (
            '<em>Källa: Pirinen, A. et al., 2024. &quot;Creating and Leveraging a Synthetic '
            'Dataset of Cloud Optical Thickness Measures for Cloud Detection in MSI.&quot; '
            'Remote Sensing. <a href="https://github.com/DigitalEarthSweden/ml-cloud-opt-thick" '
            'target="_blank">GitHub</a></em>'
        ),
    },
    "dnbr": {
        "title": "dNBR — Brandsvårighetsgrad",
        "body": (
            "dNBR (differentierat Normalized Burn Ratio) är standardmetoden för att "
            "kvantifiera hur allvarligt en brand har skadat vegetationen. Först beräknas "
            "NBR-indexet som (NIR − SWIR) / (NIR + SWIR) med band B08 och B12 både "
            "för ett datum före branden (baslinje) och för branddatumet. Sedan tas "
            "differensen: dNBR = NBR<sub>före</sub> − NBR<sub>efter</sub>. Höga positiva "
            "värden (röda på kartan) innebär hög brandsvårighetsgrad där träd och "
            "markvegetation har förstörts, medan värden nära noll (gult/grönt) visar "
            "obrända eller lätt påverkade områden. Klassificeringen följer USGS standard "
            "med sju klasser från hög återväxt till hög svårighetsgrad."
        ),
        "ref": (
            '<em>Källa: Key, C.H. &amp; Benson, N.C., 2006. &quot;Landscape Assessment: '
            'Ground measure of severity.&quot; USGS FIREMON, LA1-LA51.</em>'
        ),
    },
    "change_gradient": {
        "title": "Förändring (gradient) — Multispektral förändringsdetektering",
        "body": (
            "Förändringsgradientkartan visar hur mycket varje pixel har förändrats "
            "jämfört med en molnfri baslinjebild från före branden. Analysen använder "
            "sex Sentinel-2-band (B02, B03, B04, B08, B11, B12) — alltså synligt ljus, "
            "nära infrarött och kortvågigt infrarött — och beräknar den euklidiska "
            "normen av skillnaden mellan de två datumen. Baslinjejustering sker med "
            "IMINT Engines koregistreringsmodul som korrigerar både heltals- och "
            "subpixelförskjutningar mellan olika satellitöverfarter. Resultatet visas "
            "som en värmekarta där mörka områden är oförändrade och ljusa/heta "
            "områden visar störst förändring."
        ),
    },
    "prithvi": {
        "title": "Prithvi — AI-segmentering av brandområdet",
        "body": (
            "Prithvi är en geospatial foundation model utvecklad av NASA och IBM inom "
            "projektet NASA Earth Science. Modellen är förtränad på stora mängder "
            "HLS-satellitdata (Harmonized Landsat Sentinel-2) med en Masked Autoencoder-arkitektur "
            "(ViT-MAE) som lär sig representera markanvändning och landskap genom att "
            "rekonstruera maskerade delar av satellitbilder. För brandanalys har modellen "
            'finjusterats med uppgiftsspecifika segmenteringshuvuden (UPerNet) på datamängden '
            '"burn scars" som klassificerar varje pixel som antingen bränt (orange) eller '
            "obränt (grönt). Resultatet är en pixelnivåklassificering som "
            "kompletterar de spektrala indexen med en inlärd förståelse för hur "
            "brandskadat landskap ser ut."
        ),
        "ref": (
            '<em>Källa: Jakubik et al., 2023. &quot;Prithvi-100M: Foundation Model for '
            'Geospatial Applications.&quot; arXiv:2310.18660. '
            '<a href="https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M" '
            'target="_blank">HuggingFace</a></em>'
        ),
    },
    "nmd": {
        "title": "NMD — Nationellt Marktäckedata",
        "body": (
            "NMD (Nationellt Marktäckedata) är Sveriges rikstäckande kartläggning "
            "av marktäcke och markanvändning, producerad av Naturvårdsverket. Datat "
            "har 10 meters upplösning och klassificerar marken i över 25 kategorier "
            "— från tallskog och granskog till åkermark, bebyggelse och vatten."
        ),
        "fire_note": (
            "I brandanalysen används NMD för att korsreferera vilka naturtyper som "
            "drabbats hårdast: diagrammet visar hur stor andel av varje markklass som "
            "ligger inom det brandpåverkade området. Detta ger viktig kontext för "
            "ekologisk bedömning och återställningsplanering."
        ),
        "marine_note": (
            "I den marina analysen fyller NMD en dubbel funktion: dels som "
            "bakgrundsinformation som visar vilka landtyper som finns längs kusten, "
            "dels som landmask för båtdetekteringen. Genom att identifiera vilka "
            "pixlar som är land respektive vatten kan analyskedjan filtrera bort "
            "falsklarm på land och begränsa detektionerna till sjö- och havsområden."
        ),
        "grazing_note": (
            "I betesmarksanalysen korsrefereras NMD med LPIS-polygonerna för att "
            "verifiera vilka markklasser som faktiskt ligger inom de registrerade "
            "betesblockens gränser. Detta avslöjar t.ex. om ett betesblock innehåller "
            "skog, våtmark eller åkermark utöver den förväntade gräsmarken."
        ),
        "ref": (
            '<em>Källa: Naturvårdsverket, &quot;Nationellt Marktäckedata (NMD).&quot; '
            '<a href="https://www.naturvardsverket.se/verktyg-och-tjanster/kartor-och-karttjanster/'
            'nationella-marktackedata/" target="_blank">naturvardsverket.se</a></em>'
        ),
    },
    "yolo_vessels": {
        "title": "Båtdetektering (YOLO) — AI-objektdetektering",
        "body": (
            "Båtdetekteringen använder YOLO11s (You Only Look Once, version 11 small), "
            "en modern AI-modell för objektdetektering i realtid. YOLO-arkitekturen bygger på "
            "ett djupt konvolutionellt neuralt nätverk (CNN) som analyserar hela bilden i ett "
            "enda steg — till skillnad från äldre metoder som först föreslår kandidatområden "
            "och sedan klassificerar dem separat. Modellen har tränats på satellitbilder "
            "av båtar och delar in bilden i ett rutnät där varje cell förutsäger "
            "bounding boxes (rektanglar) och sannolikheter för att en båt finns. "
            "Överlappande detektioner filtreras med Non-Maximum Suppression (NMS). "
            "En NMD-baserad landmask säkerställer att enbart detektioner på vatten "
            "behålls, vilket eliminerar falsklarm på land. Varje detekterad båt "
            "markeras med en cyan-färgad ruta i bilden."
        ),
        "ref": (
            '<em>Källa: Jocher, G. et al., 2024. &quot;Ultralytics YOLO11.&quot; '
            '<a href="https://docs.ultralytics.com/" target="_blank">docs.ultralytics.com</a>; '
            'Redmon, J. et al., 2016. &quot;You Only Look Once: Unified, Real-Time Object '
            'Detection.&quot; CVPR 2016.</em>'
        ),
    },
    "vessel_heatmap": {
        "title": "Båtaktivitet (heatmap) — Multitemporal analys",
        "body": (
            "Heatmap-analysen aggregerar båtdetektioner från flera satellitöverfarter "
            "under en tidsperiod till en enda värmekarta som visar var båtar förekommer "
            "oftast. För varje molnfritt tillfälle körs YOLO-detektorn och resultaten "
            "ackumuleras i ett rutnät där varje cells intensitet ökar för varje "
            "detekterad båt. Områden med återkommande trafik — som farleder, "
            "hamninlopp och ankringsplatser — får höga värden (röda), medan enstaka "
            "passeringar ger lägre intensitet (gult). Bilder med för hög molntäckning "
            "filtreras automatiskt bort genom COT-analys för att undvika falsklarm. "
            "Resultatet ger en överblick av det maritima rörelsemönstret som enstaka "
            "ögonblicksbilder inte kan visa."
        ),
    },
    "lpis_betesmark": {
        "title": "LPIS Betesblock \u2014 Jordbruksverkets blockdatabas",
        "body": (
            "LPIS (Land Parcel Identification System) \u00e4r Jordbruksverkets "
            "databas \u00f6ver alla jordbruksblock i Sverige. Polygonerna h\u00e4mtas "
            "i realtid via Jordbruksverkets \u00f6ppna WFS-tj\u00e4nst och filtreras "
            "p\u00e5 \u00e4goslag = \u201cBete\u201d f\u00f6r att visa enbart betesmark. "
            "Datan uppdateras \u00e5rligen och inneh\u00e5ller ~252\u202f000 betesblock "
            "\u00f6ver hela Sverige. Varje block har ett unikt block-ID, areal, "
            "region och st\u00f6dkategori. Koordinatsystemet \u00e4r EPSG:3006 "
            "(SWEREF99 TM) \u2014 samma som v\u00e5r NMD-grid, vilket ger exakt "
            "pixeljustering utan omprojicering."
        ),
        "ref": (
            '<em>K\u00e4lla: Jordbruksverket, \u00d6ppna data, CC BY 4.0. '
            '<a href="https://jordbruksverket.se/e-tjanster-databaser-och-appar/'
            'ovriga-e-tjanster-och-databaser/oppna-data" target="_blank">'
            'jordbruksverket.se/oppna-data</a></em>'
        ),
    },
    "grazing_pipeline": {
        "title": "Betesmarkspipeline \u2014 Multitemporal analys",
        "body": (
            "IMINT Engines betesmarkspipeline h\u00e4mtar automatiskt alla "
            "molnfria Sentinel-2-scener under betess\u00e4songen (april\u2013oktober) "
            "f\u00f6r varje LPIS-polygon. F\u00f6r varje tidssnitt h\u00e4mtas "
            "alla 12 spektralband (B01\u2013B12 exkl. B10) och en SCL-baserad "
            "molnmask. Tidssnitt med \u22651% moln inom polygonen filtreras bort. "
            "Alla kvarst\u00e5ende datum co-registreras geometriskt mot ett "
            "referensdatum med sub-pixel Fourier-faskorrelation f\u00f6r att "
            "s\u00e4kerst\u00e4lla perfekt pixeljustering mellan tidssnitt. "
            "Resultatet \u00e4r en (T, 12, H, W)-tensor redo f\u00f6r "
            "CNN-LSTM-klassificering av betesaktivitet."
        ),
        "ref": (
            '<em>Pirinen, A. et al., 2024. \u201cDetecting Grazing Activity '
            'from Satellite Time Series Data.\u201d '
            '<a href="https://arxiv.org/abs/2510.14493" target="_blank">'
            'arXiv:2510.14493</a></em>'
        ),
    },
    "grazing_model": {
        "title": "CNN-biLSTM betesklassificerare (pib-ml-grazing)",
        "body": (
            "F\u00f6r varje LPIS-betespolygon klassificeras den multitemporala "
            "Sentinel-2-tidsserien som <em>aktiv betesmark</em> eller "
            "<em>ingen aktivitet</em>. Modellen \u00e4r en CNN-biLSTM fr\u00e5n "
            "RISE Research Institutes of Sweden: ett CNN-block (Conv2d \u2192 ReLU "
            "\u2192 MaxPool2d) extraherar rumsliga features per tidssnitt, sedan "
            "bearbetas tidsserien av en bidirektionell LSTM (hidden_dim=8). "
            "F\u00f6r slutprediktionen tas medianen av de sista 4 tidsstegen, "
            "vilket ger robust klassificering. Indata: 12 band \u00d7 46\u00d746 px "
            "(center crop) \u00d7 T tidssteg. Modellen \u00e4r f\u00f6rtr\u00e4nad p\u00e5 "
            "LPIS-polygoner i Sverige med data fr\u00e5n Jordbruksverket. "
            "MIT-licens."
        ),
        "ref": (
            '<em>Pirinen, A. et al., 2024. \u201cDetecting Grazing Activity '
            'from Satellite Time Series Data.\u201d '
            '<a href="https://github.com/aleksispi/pib-ml-grazing" target="_blank">'
            'github.com/aleksispi/pib-ml-grazing</a> (MIT)</em>'
        ),
    },
}

# Which descriptions to show in each tab (in order) and which context to use
_FIRE_DESCRIPTION_IDS = [
    "sentinel2_rgb", "ndvi", "ndwi", "evi", "cot", "dnbr",
    "change_gradient", "prithvi", "nmd",
]
_MARINE_DESCRIPTION_IDS = [
    "sentinel2_rgb", "yolo_vessels", "vessel_heatmap",
    "nmd", "ndvi", "ndwi", "cot",
]
_GRAZING_DESCRIPTION_IDS = [
    "sentinel2_rgb", "lpis_betesmark", "ndvi", "ndwi",
    "evi", "cot", "nmd", "grazing_pipeline", "grazing_model",
]


def _render_descriptions(desc_ids: list[str], context: str) -> str:
    """Render analysis description HTML from _ANALYSIS_DESCRIPTIONS.

    Args:
        desc_ids: List of keys into _ANALYSIS_DESCRIPTIONS.
        context: 'fire' or 'marine' — selects the context-specific note.
    """
    parts = []
    note_key = f"{context}_note"
    for did in desc_ids:
        d = _ANALYSIS_DESCRIPTIONS[did]
        body = d["body"]
        note = d.get(note_key, "")
        if note:
            body = f"{body} {note}"
        ref = d.get("ref", "")
        ref_html = f"\n                <br>{ref}" if ref else ""
        parts.append(
            f'            <h3>{d["title"]}</h3>\n'
            f'            <p>\n'
            f'                {body}{ref_html}\n'
            f'            </p>'
        )
    return "\n\n".join(parts)


def save_tabbed_report(
    fire_dir: str,
    marine_dir: str,
    output_path: str,
    fire_date: str = "",
    marine_date: str = "",
    grazing_dir: str | None = None,
    grazing_date: str = "",
) -> str:
    """Generate a tabbed HTML showcase with Fire, Marine, and Grazing tabs.

    Uses EXTERNAL image files (no base64 embedding).  Images are copied
    to ``showcase/fire/``, ``showcase/marine/``, and ``showcase/grazing/``
    subdirectories next to the output HTML.

    Args:
        fire_dir: Path to fire analysis output directory.
        marine_dir: Path to marine analysis output directory.
        output_path: Where to write the HTML file.
        fire_date: Date string for fire analysis (e.g. "2018-07-24").
        marine_date: Date string for marine analysis (e.g. "2025-07-10").
        grazing_dir: Path to grazing analysis output directory (optional).
        grazing_date: Date string for grazing analysis.

    Returns:
        The output file path.
    """
    # ── Resolve file-name patterns ────────────────────────────────────────
    fire_prefix = f"{fire_date}_" if fire_date else ""
    marine_prefix = f"{marine_date}_" if marine_date else ""

    file_map = {
        "rgb": "rgb.png",
        "nmd": "nmd_overlay.png",
        "ndvi": "ndvi_clean.png",
        "ndwi": "ndwi_clean.png",
        "evi": "evi_clean.png",
        "cot": "cot_clean.png",
        "dnbr": "dnbr_clean.png",
        "change_gradient": "change_gradient.png",
        "prithvi_seg": "prithvi_seg_clean.png",
        "vessels": "vessels_clean.png",
        "ai2_vessels": "ai2_vessels_clean.png",
        "vessel_heatmap": "vessel_heatmap_clean.png",
        "sjokort": "sjokort.png",
        "lpis": "lpis_overlay.png",
    }

    # ── Copy images to showcase subdirectories ──────────────────────────
    out_parent = os.path.dirname(output_path) or "."
    showcase_dir = os.path.join(out_parent, "showcase")

    def _copy_images(out_dir, prefix, viewers, tab_name):
        """Copy source PNGs to showcase/<tab_name>/ and return id→relative path."""
        dest_dir = os.path.join(showcase_dir, tab_name)
        os.makedirs(dest_dir, exist_ok=True)
        imgs = {}
        for v in viewers:
            key = v["key"]
            dst_name = file_map.get(key, f"{key}.png")
            dst = os.path.join(dest_dir, dst_name)
            # Try source with prefix first (original output dir)
            fname = prefix + dst_name
            src = os.path.join(out_dir, fname)
            if os.path.isfile(src):
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy2(src, dst)
                imgs[v["id"]] = f"showcase/{tab_name}/{dst_name}"
            elif os.path.isfile(dst):
                # File already exists in showcase dir (pre-generated)
                imgs[v["id"]] = f"showcase/{tab_name}/{dst_name}"
        return imgs

    fire_imgs = _copy_images(fire_dir, fire_prefix, _FIRE_VIEWERS, "fire")
    marine_imgs = _copy_images(marine_dir, marine_prefix, _MARINE_VIEWERS, "marine")

    # Grazing tab (optional) — grazing images have no date prefix
    grazing_imgs = {}
    _lpis_geojson_raw = ""
    if grazing_dir and os.path.isdir(grazing_dir):
        grazing_imgs = _copy_images(grazing_dir, "", _GRAZING_VIEWERS, "grazing")
        # Load LPIS vector GeoJSON (pixel coordinates) if available
        _lpis_geojson_path = os.path.join(grazing_dir, "lpis_polygons.json")
        if os.path.isfile(_lpis_geojson_path):
            with open(_lpis_geojson_path, encoding="utf-8") as _lf:
                _lpis_geojson_raw = _lf.read()
        else:
            _lpis_geojson_raw = ""
        # Read date from meta if not provided
        if not grazing_date:
            _gm_path = os.path.join(grazing_dir, "grazing_meta.json")
            if os.path.isfile(_gm_path):
                with open(_gm_path) as _gf:
                    grazing_date = json.load(_gf).get("date", "")

    # Also copy sjökort for the RGB panel toggle (not a separate viewer)
    sjokort_src = os.path.join(
        marine_dir, marine_prefix + file_map.get("sjokort", "sjokort.png")
    )
    if os.path.isfile(sjokort_src):
        sjokort_dest = os.path.join(showcase_dir, "marine", file_map["sjokort"])
        os.makedirs(os.path.join(showcase_dir, "marine"), exist_ok=True)
        if os.path.abspath(sjokort_src) != os.path.abspath(sjokort_dest):
            shutil.copy2(sjokort_src, sjokort_dest)
        marine_imgs["m-sjokort"] = f"showcase/marine/{file_map['sjokort']}"

    # Load marine vessel/detection GeoJSON (pixel coordinates) if available
    _vessel_geojson_raw = ""
    _vessel_geojson_path = os.path.join(marine_dir, "vessel_detections.json")
    # Also check in showcase/marine/ (may have been pre-generated)
    if not os.path.isfile(_vessel_geojson_path):
        _vessel_geojson_path = os.path.join(showcase_dir, "marine", "vessel_detections.json")
    if os.path.isfile(_vessel_geojson_path):
        with open(_vessel_geojson_path, encoding="utf-8") as _vf:
            _vessel_geojson_raw = _vf.read()

    # Generate baseline RGB PNG for fire tab (pre-fire reference image)
    # Match baseline by comparing coordinates from fire_dir name
    _baseline_dir = os.path.join(fire_dir, "..", "baselines")
    _area_candidates = [
        f.replace(".npy", "") for f in os.listdir(_baseline_dir)
        if f.endswith(".npy") and "_bands" not in f and "_scl" not in f and "_geo" not in f
    ] if os.path.isdir(_baseline_dir) else []
    _baseline_npy = None
    # Extract approximate lon/lat from fire_dir to match correct baseline
    _fire_bbox_m = re.search(r'([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)',
                             os.path.basename(fire_dir))
    _fire_lon = float(_fire_bbox_m.group(1)) if _fire_bbox_m else None
    # Collect area-matching candidates, prefer ones with _geo.json (= geo-aligned)
    _geo_candidates = []
    _plain_candidates = []
    for cand in _area_candidates:
        npy_path = os.path.join(_baseline_dir, cand + ".npy")
        if not os.path.isfile(npy_path):
            continue
        if _fire_lon is not None:
            _cand_m = re.match(r'([\d.]+)_', cand)
            if _cand_m and abs(float(_cand_m.group(1)) - _fire_lon) > 1.0:
                continue  # wrong area (e.g. marine baseline)
        if os.path.isfile(os.path.join(_baseline_dir, cand + "_geo.json")):
            _geo_candidates.append(npy_path)
        else:
            _plain_candidates.append(npy_path)
    # Prefer geo-aligned baselines over legacy ones
    if _geo_candidates:
        _baseline_npy = _geo_candidates[0]
    elif _plain_candidates:
        _baseline_npy = _plain_candidates[0]
    if _baseline_npy is not None:
        try:
            import numpy as np
            from PIL import Image as PILImage
            bl_arr = np.load(_baseline_npy)
            if bl_arr.dtype != np.uint8:
                if bl_arr.max() <= 1.0:
                    bl_arr = (bl_arr * 255).clip(0, 255).astype(np.uint8)
                else:
                    bl_arr = bl_arr.clip(0, 255).astype(np.uint8)
            bl_img = PILImage.fromarray(bl_arr)
            bl_dest = os.path.join(showcase_dir, "fire", "baseline_rgb.png")
            os.makedirs(os.path.join(showcase_dir, "fire"), exist_ok=True)
            bl_img.save(bl_dest)
            fire_imgs["f-baseline"] = "showcase/fire/baseline_rgb.png"
        except Exception:
            pass  # baseline not available, skip toggle

    fire_viewers = [v for v in _FIRE_VIEWERS if v["id"] in fire_imgs]
    marine_viewers = [
        v for v in _MARINE_VIEWERS
        if v["id"] in marine_imgs or (v.get("vector") and _vessel_geojson_raw)
    ]
    grazing_viewers = [
        v for v in _GRAZING_VIEWERS
        if v["id"] in grazing_imgs or (v.get("vector") and _lpis_geojson_raw)
    ]

    # Image dimensions (read from bands_meta or first image)
    def _read_shape(out_dir, prefix):
        meta_path = os.path.join(out_dir, "bands", f"{prefix}bands_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            s = meta.get("shape", [573, 324])
            return s[0], s[1]
        return 573, 324  # fallback

    fire_h, fire_w = _read_shape(fire_dir, fire_prefix)
    marine_h, marine_w = _read_shape(marine_dir, marine_prefix)

    # Grazing shape — read from meta or first image
    grazing_h, grazing_w = 344, 383  # default
    if grazing_dir:
        grazing_meta_path = os.path.join(grazing_dir, "grazing_meta.json")
        if os.path.isfile(grazing_meta_path):
            with open(grazing_meta_path) as f:
                _gm = json.load(f)
            _gs = _gm.get("shape", [344, 383])
            grazing_h, grazing_w = _gs[0], _gs[1]
        else:
            grazing_h, grazing_w = _read_shape(grazing_dir, grazing_prefix)

    # NMD stats for fire charts
    nmd_path = os.path.join(fire_dir, f"{fire_prefix}nmd_stats.json")
    nmd_stats = {}
    if os.path.isfile(nmd_path):
        with open(nmd_path) as f:
            nmd_stats = json.load(f)
    fire_chart_data = _build_chart_data(nmd_stats)
    fire_chart_json = json.dumps(fire_chart_data, ensure_ascii=False)

    # Summary cards for fire
    summary_path = os.path.join(fire_dir, f"{fire_prefix}imint_summary.json")
    imint_summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            imint_summary = json.load(f)
    fire_summary_html = _build_summary_cards(imint_summary)

    # Summary cards for marine
    marine_summary_path = os.path.join(marine_dir, f"{marine_prefix}imint_summary.json")
    marine_imint_summary = {}
    if os.path.isfile(marine_summary_path):
        with open(marine_summary_path) as f:
            marine_imint_summary = json.load(f)
    marine_summary_html = _build_marine_summary_cards(
        marine_dir, marine_prefix, marine_imint_summary
    )

    # Summary cards for grazing
    grazing_summary_html = ""
    if grazing_dir and os.path.isdir(grazing_dir):
        grazing_summary_html = _build_grazing_summary_cards(grazing_dir)

    # ── Fetch libraries ───────────────────────────────────────────────────
    leaflet_css = _fetch_lib(_CDN_LIBS["leaflet_css"])
    leaflet_js = _fetch_lib(_CDN_LIBS["leaflet_js"])
    leaflet_sync_js = _fetch_lib(_CDN_LIBS["leaflet_sync_js"])
    chart_js = _fetch_lib(_CDN_LIBS["chart_js"])

    # ── Build map cells HTML per tab ──────────────────────────────────────
    def _map_cells(viewers, tab_prefix, has_bg_toggle=False, hideable=False,
                   per_panel_toggle=None):
        """per_panel_toggle: dict mapping viewer key to list of (data-bg, label) tuples."""
        html = ""
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

            bg_toggle_html = ""
            panel_toggle = (per_panel_toggle or {}).get(v["key"])
            if panel_toggle:
                btns = ""
                for i, (bg_key, label) in enumerate(panel_toggle):
                    active = " active" if i == 0 else ""
                    btns += f'<button class="bg-btn{active}" data-bg="{bg_key}">{label}</button>'
                bg_toggle_html = f"""
                    <div class="bg-toggle" data-map-id="{v["id"]}">
                        <span class="bg-label">Visa:</span>
                        {btns}
                    </div>"""
            elif has_bg_toggle:
                bg_toggle_html = f"""
                    <div class="bg-toggle" data-map-id="{v["id"]}">
                        <span class="bg-label">Visa:</span>
                        <button class="bg-btn active" data-bg="rgb">RGB</button>
                        <button class="bg-btn" data-bg="sjokort">Sjökort</button>
                    </div>"""

            hide_btn_html = ""
            panel_attr = ""
            if hideable:
                hide_btn_html = (f'<button class="hide-panel-btn" data-panel-id="{v["id"]}"'
                                 f' title="Dölj panel">\u2715</button>')
                panel_attr = f' data-panel-id="{v["id"]}"'

            html += f"""
            <div class="map-cell"{panel_attr}>
                <div class="map-cell-header">
                    <h3>{v["title"]}</h3>
                    <div class="header-controls">
                        {bg_toggle_html}
                        <div class="opacity-control">
                            <label for="opacity-{v["id"]}">Opacitet</label>
                            <input type="range" id="opacity-{v["id"]}" min="0" max="100"
                                   value="100" data-map-id="{v["id"]}">
                            <span class="opacity-value" id="opacity-val-{v["id"]}">100%</span>
                        </div>
                        {hide_btn_html}
                    </div>
                </div>
                <div id="{v["id"]}" class="map-container"></div>
                {legend_html}
            </div>"""
        return html

    has_marine_bg = "m-sjokort" in marine_imgs and "m-rgb" in marine_imgs
    fire_bg_toggle = {}
    if "f-baseline" in fire_imgs:
        fire_bg_toggle = {
            "dnbr": [("rgb", "Efter"), ("baseline", "Före")],
            "change_gradient": [("rgb", "Efter"), ("baseline", "Före")],
        }
    fire_cells = _map_cells(fire_viewers, "f", hideable=True,
                            per_panel_toggle=fire_bg_toggle)
    marine_cells = _map_cells(marine_viewers, "m", has_bg_toggle=has_marine_bg, hideable=True)
    grazing_bg_toggle = {}
    if "g-lpis" in grazing_imgs and "g-nmd" in grazing_imgs:
        grazing_bg_toggle["lpis"] = [("rgb", "RGB"), ("nmd", "NMD")]
    grazing_cells = _map_cells(grazing_viewers, "g", hideable=True,
                               per_panel_toggle=grazing_bg_toggle)

    # ── Fire chart sections ───────────────────────────────────────────────
    fire_charts_html = ""
    if fire_chart_data.get("change"):
        fire_charts_html += """
        <div class="chart-card">
            <h3>Förändringsdetektering per markklass</h3>
            <canvas id="chart-change"></canvas>
        </div>
        <div class="chart-card">
            <h3>dNBR brandsvårighetsgrad per markklass</h3>
            <canvas id="chart-dnbr"></canvas>
        </div>"""
    if fire_chart_data.get("prithvi"):
        fire_charts_html += """
        <div class="chart-card">
            <h3>Brandsegmentering per markklass (Prithvi)</h3>
            <canvas id="chart-prithvi"></canvas>
        </div>"""
    if fire_chart_data.get("l2"):
        fire_charts_html += """
        <div class="chart-card">
            <h3>Marktäcke — detaljerade klasser (Nivå 2)</h3>
            <canvas id="chart-l2"></canvas>
        </div>"""

    if fire_charts_html:
        fire_charts_html = f"""
        <div class="section-header">
            <h2>Korsreferens mot NMD (Nationellt Marktäckedata)</h2>
        </div>
        <div class="charts-grid">
            {fire_charts_html}
        </div>"""

    # ── Viewer configs as JS ──────────────────────────────────────────────
    fire_viewer_js = json.dumps(
        [{"id": v["id"], "key": v["key"], "vector": v.get("vector", False),
          "legend": v.get("legend", [])}
         for v in fire_viewers],
        ensure_ascii=False,
    )
    marine_viewer_js = json.dumps(
        [{"id": v["id"], "key": v["key"], "vector": v.get("vector", False),
          "legend": v.get("legend", [])}
         for v in marine_viewers],
        ensure_ascii=False,
    )
    grazing_viewer_js = json.dumps(
        [{"id": v["id"], "key": v["key"], "vector": v.get("vector", False),
          "legend": v.get("legend", [])}
         for v in grazing_viewers],
        ensure_ascii=False,
    )

    # ── Panel visibility toolbar ──────────────────────────────────────────
    def _panel_toolbar(viewers):
        if len(viewers) <= 1:
            return ""
        chips = ""
        for v in viewers:
            chips += (f'<button class="panel-chip active" '
                      f'data-panel-id="{v["id"]}">{v["title"]}</button>')
        return (f'<div class="panel-toolbar">'
                f'<span class="panel-toolbar-label">Paneler:</span>'
                f'{chips}</div>')

    fire_toolbar = _panel_toolbar(fire_viewers)
    marine_toolbar = _panel_toolbar(marine_viewers)
    grazing_toolbar = _panel_toolbar(grazing_viewers)

    # ── Images as JS objects ──────────────────────────────────────────────
    def _imgs_js(imgs_dict):
        items = [f'    "{k}": "{v}"' for k, v in imgs_dict.items()]
        return "{\n" + ",\n".join(items) + "\n}"

    fire_imgs_js = _imgs_js(fire_imgs)
    marine_imgs_js = _imgs_js(marine_imgs)
    grazing_imgs_js = _imgs_js(grazing_imgs)

    # ── Render descriptions from shared objects ────────────────────────────
    fire_descriptions = _render_descriptions(_FIRE_DESCRIPTION_IDS, "fire")
    marine_descriptions = _render_descriptions(_MARINE_DESCRIPTION_IDS, "marine")
    grazing_descriptions = _render_descriptions(_GRAZING_DESCRIPTION_IDS, "grazing")

    # ── Pre-compute grazing HTML (avoids backslash in f-string on Py 3.9) ──
    grazing_tab_btn = (
        '<a href="#" class="theme-tab" data-tab="grazing">\U0001f404 Betesmark</a>'
        if grazing_viewers else ""
    )
    grazing_subtitle = (
        f" &middot; Betesmark ({grazing_date})" if grazing_viewers else ""
    )
    if grazing_viewers:
        grazing_tab_html = f"""<div class="tab-content" id="tab-grazing">
        <div class="section-header">
            <h2>Betesmarksanalys \u2014 {grazing_date}</h2>
        </div>
        {grazing_summary_html}
        <div class="tab-intro">
            <p>
                Analysomr\u00e5det \u00e4r bel\u00e4get nordost om Lund i Sk\u00e5ne \u2014 ett av
                Sveriges mest intensivt brukade jordbrukslandskap med en blandning
                av \u00e5kermark, beteshagar och sm\u00e5skaliga skogspartier. LPIS-polygoner
                fr\u00e5n Jordbruksverkets blockdatabas visar registrerade betesblock
                i omr\u00e5det. Sentinel-2-data fr\u00e5n {grazing_date} har analyserats
                med spektrala index (NDVI, NDWI, EVI), molnanalys (COT) och
                korsrefererats mot NMD markt\u00e4ckedata f\u00f6r att kartera
                vegetationens tillst\u00e5nd inom betesmarkerna.
            </p>
        </div>
        {grazing_toolbar}
        <div class="map-grid">
            {grazing_cells}
        </div>
        <div class="tab-description">

{grazing_descriptions}
        </div>
    </div>"""
        _geojson_js = f"const LPIS_GEOJSON = {_lpis_geojson_raw};" if _lpis_geojson_raw else "const LPIS_GEOJSON = null;"
        grazing_js_block = (
            f"const GRAZING_VIEWERS = {grazing_viewer_js};\n"
            f"        const GRAZING_IMAGES = {grazing_imgs_js};\n"
            f"        {_geojson_js}\n"
            f"        initMaps(GRAZING_VIEWERS, GRAZING_IMAGES, {grazing_h}, {grazing_w}, false, LPIS_GEOJSON);"
        )
    else:
        grazing_tab_html = ""
        grazing_js_block = ""

    # Marine vessel detection GeoJSON for Leaflet vector rendering
    if _vessel_geojson_raw:
        _marine_geojson_js = f"const VESSEL_GEOJSON = {_vessel_geojson_raw};"
    else:
        _marine_geojson_js = "const VESSEL_GEOJSON = null;"

    # ── Assemble HTML ─────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="sv">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IMINT Analysrapport — Showcase</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>{leaflet_css}</style>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Space Grotesk', sans-serif;
            background: #ffffff;
            color: #171717;
            font-size: 18px;
            line-height: 1.56;
        }}

        /* ── Header (white background, black text) ─────────────────── */
        .header {{
            background: #ffffff;
            padding: 20px 32px;
            border-bottom: 1px solid #e5e7eb;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .header-left {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        .header-left h1 {{
            font-size: 22px;
            font-weight: 700;
            color: #171717;
            letter-spacing: -0.3px;
        }}
        .header-left h1 span {{
            color: #171717;
        }}
        .header-left p {{
            font-size: 13px;
            color: #6b7280;
            margin-top: 2px;
        }}
        .header-nav {{
            display: flex;
            gap: 6px;
        }}
        .theme-tab {{
            text-decoration: none;
            padding: 6px 16px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 600;
            color: #6b7280;
            background: #f3f4f6;
            border: 1px solid #d1d5db;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .theme-tab:hover {{
            color: #171717;
            border-color: #9ca3af;
            background: #e5e7eb;
        }}
        .theme-tab.active {{
            color: #ffffff;
            background: #1a4338;
            border-color: #1a4338;
        }}

        /* ── Summary cards ───────────────────────────────────────────── */
        .summary-section {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            padding: 16px 20px 8px;
        }}
        .summary-card {{
            background: #1a4338;
            border-radius: 10px;
            padding: 14px 18px;
            min-width: 170px;
            flex: 1;
            border: 1px solid #245045;
            position: relative;
            overflow: hidden;
        }}
        .summary-card::before {{
            content: '';
            position: absolute;
            left: 0; top: 0; bottom: 0;
            width: 3px;
            background: #cff8e4;
        }}
        .summary-card h4 {{
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: rgba(207,248,228,0.5);
            margin-bottom: 6px;
            font-weight: 600;
        }}
        .summary-card .value {{
            font-size: 22px;
            font-weight: 700;
            color: #cff8e4;
        }}
        .summary-card .detail {{
            font-size: 11px;
            color: rgba(207,248,228,0.4);
            margin-top: 2px;
        }}

        /* ── Section header ──────────────────────────────────────────── */
        .section-header {{
            text-align: center;
            padding: 28px 20px 10px;
        }}
        .section-header h2 {{
            font-size: 20px;
            font-weight: 700;
            color: #171717;
        }}

        /* ── Tab intro / description ────────────────────────────────── */
        .tab-intro {{
            max-width: 820px;
            margin: 0 auto;
            padding: 24px 28px 0;
            text-align: center;
        }}
        .tab-intro p {{
            font-size: 18px;
            line-height: 1.56;
            color: #171717;
            margin: 0;
        }}
        .tab-description {{
            max-width: 820px;
            margin: 0 auto;
            padding: 10px 28px 20px;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 10px;
        }}
        .tab-description p {{
            font-size: 18px;
            line-height: 1.56;
            color: #171717;
            margin: 8px 0 0;
        }}
        .tab-description h3 {{
            color: #171717;
            margin-top: 28px;
        }}
        .tab-description a {{
            color: #171717;
            text-decoration: underline;
        }}
        .tab-description em {{
            color: #555555;
        }}

        /* ── Tab content ─────────────────────────────────────────────── */
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}

        /* ── Map grid ────────────────────────────────────────────────── */
        .map-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 10px;
            padding: 10px 20px;
        }}
        .map-cell {{
            background: #1a4338;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #245045;
        }}
        .map-cell-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 14px;
            border-bottom: 1px solid #245045;
            background: #163930;
        }}
        .map-cell-header h3 {{
            font-size: 13px;
            font-weight: 600;
            color: #cff8e4;
        }}
        .header-controls {{
            display: flex;
            align-items: center;
            gap: 14px;
        }}
        .opacity-control {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: rgba(207,248,228,0.5);
        }}
        .opacity-control input[type="range"] {{
            width: 70px;
            accent-color: #cff8e4;
        }}
        .opacity-value {{
            width: 32px;
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        .map-container {{
            height: 500px;
            background: #1a4338;
        }}

        /* ── Panel visibility toolbar ─────────────────────────────────── */
        .panel-toolbar {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }}
        .panel-toolbar-label {{
            font-size: 12px;
            color: #171717;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-right: 4px;
        }}
        .panel-chip {{
            font-size: 11px;
            padding: 4px 12px;
            border-radius: 9999px;
            border: 1px solid #245045;
            background: #1a4338;
            color: rgba(207,248,228,0.7);
            cursor: pointer;
            transition: all 0.15s;
            user-select: none;
        }}
        .panel-chip.active {{
            background: #245045;
            border-color: #cff8e4;
            color: #cff8e4;
            font-weight: 600;
        }}
        .panel-chip:hover {{
            border-color: #cff8e4;
            color: #cff8e4;
        }}

        /* ── Hide button in map-cell header ──────────────────────────── */
        .hide-panel-btn {{
            background: none;
            border: none;
            color: rgba(207,248,228,0.4);
            cursor: pointer;
            font-size: 16px;
            padding: 2px 6px;
            line-height: 1;
            border-radius: 4px;
            transition: all 0.15s;
        }}
        .hide-panel-btn:hover {{
            color: #ef4444;
            background: rgba(239, 68, 68, 0.15);
        }}

        .map-cell.hidden-panel {{
            display: none;
        }}

        /* ── Background toggle ───────────────────────────────────────── */
        .bg-toggle {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .bg-label {{
            font-size: 10px;
            color: rgba(207,248,228,0.5);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-right: 2px;
        }}
        .bg-btn {{
            font-size: 10px;
            padding: 2px 8px;
            border: 1px solid #245045;
            border-radius: 3px;
            background: transparent;
            color: rgba(207,248,228,0.6);
            cursor: pointer;
            transition: all 0.15s;
        }}
        .bg-btn:hover {{
            border-color: #cff8e4;
            color: #cff8e4;
        }}
        .bg-btn.active {{
            background: #cff8e4;
            border-color: #cff8e4;
            color: #1a4338;
        }}

        /* ── Legend ───────────────────────────────────────────────────── */
        .legend-strip {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px 12px;
            padding: 8px 14px;
            font-size: 10px;
            border-top: 1px solid #245045;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 4px;
            color: rgba(207,248,228,0.6);
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
            background: #1a4338;
            border-radius: 10px;
            padding: 18px 20px;
            border: 1px solid #245045;
        }}
        .chart-card h3 {{
            font-size: 13px;
            font-weight: 600;
            color: #cff8e4;
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
            color: #9ca3af;
            border-top: 1px solid #e5e7eb;
        }}
        .license-toggle {{
            background: none;
            border: 1px solid #d1d5db;
            color: #6b7280;
            padding: 6px 16px;
            border-radius: 9999px;
            cursor: pointer;
            font-size: 11px;
            font-family: 'Space Grotesk', sans-serif;
            margin-top: 10px;
            transition: all 0.15s;
        }}
        .license-toggle:hover {{
            border-color: #171717;
            color: #171717;
        }}
        .license-section {{
            display: none;
            text-align: left;
            max-width: 900px;
            margin: 16px auto 0;
            padding: 20px 24px;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            font-size: 11px;
            color: #4b5563;
            line-height: 1.6;
        }}
        .license-section.open {{ display: block; }}
        .license-section h4 {{
            font-size: 12px; color: #171717;
            margin: 14px 0 6px; font-weight: 600;
        }}
        .license-section h4:first-child {{ margin-top: 0; }}
        .license-table {{
            width: 100%; border-collapse: collapse; margin: 10px 0;
        }}
        .license-table th {{
            text-align: left; font-size: 10px;
            text-transform: uppercase; letter-spacing: 0.5px;
            color: #6b7280; padding: 6px 10px;
            border-bottom: 1px solid #e5e7eb;
        }}
        .license-table td {{
            padding: 6px 10px;
            border-bottom: 1px solid #f3f4f6;
            vertical-align: top;
        }}
        .license-table tr:last-child td {{ border-bottom: none; }}
        .license-table a {{ color: #171717; text-decoration: underline; }}
        .license-badge {{
            display: inline-block; padding: 1px 7px;
            border-radius: 4px; font-size: 10px; font-weight: 600;
        }}
        .badge-open {{ background: #dcfce7; color: #166534; }}
        .badge-restricted {{ background: #fef9c3; color: #854d0e; }}
        .badge-copyleft {{ background: #fee2e2; color: #991b1b; }}

        /* ── Leaflet overrides ───────────────────────────────────────── */
        .leaflet-container {{
            background: #1a4338 !important;
        }}

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
            <div>
                <h1><span>IMINT</span> Analysrapport</h1>
                <p>Showcase — Brand ({fire_date}) &middot; Marin ({marine_date}){grazing_subtitle}</p>
            </div>
        </div>
        <div class="header-nav">
            <a href="#" class="theme-tab active" data-tab="fire">\U0001f525 Brand</a>
            <a href="#" class="theme-tab" data-tab="marine">\u2693 Marin</a>
            {grazing_tab_btn}
        </div>
    </div>

    <!-- ── Fire tab ──────────────────────────────────────────────── -->
    <div class="tab-content active" id="tab-fire">
        <div class="section-header">
            <h2>Brandanalys — {fire_date}</h2>
        </div>
        {fire_summary_html}
        <div class="tab-intro">
            <p>
                Analysområdet är beläget i Ljusdals kommun, Gävleborgs län, och
                visar Kårbölebranden — en av de största skogsbränderna i Sveriges moderna historia
                sommaren 2018. Den 14 juli 2018 startade en skogsbrand som till slut
                bredde ut sig över cirka 9\u202f500 hektar skog, vilket gjorde den till
                den största skogsbranden i Sverige på över 50 år. Här har
                Sentinel-2-data från {fire_date} analyserats med flera kompletterande metoder
                för att kartlägga brandens utbredning och intensitet.
            </p>
        </div>
        {fire_toolbar}
        <div class="map-grid">
            {fire_cells}
        </div>
        {fire_charts_html}
        <div class="tab-description">
{fire_descriptions}
        </div>
    </div>

    <!-- ── Marine tab ────────────────────────────────────────────── -->
    <div class="tab-content" id="tab-marine">
        <div class="section-header">
            <h2>Marin analys — {marine_date}</h2>
        </div>
        {marine_summary_html}
        <div class="tab-intro">
            <p>
                Analysområdet visar skärgården utanför Hunnebostrand — ett område
                längs den norra bohuslänska kusten med intensiv maritim aktivitet
                från både kommersiell sjöfart, fiske och fritidsbåtar. Sentinel-2-data
                från {marine_date} har analyserats med flera kompletterande metoder
                för att kartlägga båtförekomst, vattenförhållanden och marktäcke
                i kust- och havsområdet.
            </p>
        </div>
        {marine_toolbar}
        <div class="map-grid">
            {marine_cells}
        </div>
        <div class="tab-description">

{marine_descriptions}
        </div>
    </div>

    <!-- ── Grazing tab ──────────────────────────────────────────── -->
    {grazing_tab_html}
    <div class="footer">
        IMINT Engine &middot; &copy; 2024&ndash;2025 RISE Research Institutes of Sweden AB
        &middot; CC0 1.0 Universal &middot; Genererad {fire_date} / {marine_date} / {grazing_date}
        <br>
        <button class="license-toggle" onclick="document.getElementById('license-info').classList.toggle('open'); this.textContent = this.textContent === 'Visa licenser och upphovsr\u00e4tt' ? 'D\u00f6lj licenser' : 'Visa licenser och upphovsr\u00e4tt';">Visa licenser och upphovsr\u00e4tt</button>
        <div id="license-info" class="license-section">
            <h4>Modeller</h4>
            <table class="license-table">
                <tr><th>Komponent</th><th>Licens</th><th>Upphovsr\u00e4tt</th><th>Anm\u00e4rkning</th></tr>
                <tr>
                    <td>YOLO11s (båtdetektering)</td>
                    <td><span class="license-badge badge-copyleft">AGPL-3.0</span></td>
                    <td>Ultralytics</td>
                    <td>Copyleft &mdash; kommersiellt bruk kr\u00e4ver Enterprise-licens</td>
                </tr>
                <tr>
                    <td>AI2 rslearn (detektering + attribut)</td>
                    <td><span class="license-badge badge-open">Apache 2.0</span></td>
                    <td>&copy; 2024 Allen Institute for AI</td>
                    <td>Modellvikter: Apache 2.0, annotationer: CC-BY 4.0</td>
                </tr>
                <tr>
                    <td>SatlasPretrain (Swin V2 B backbone)</td>
                    <td><span class="license-badge badge-open">Apache 2.0</span></td>
                    <td>&copy; 2024 Allen Institute for AI</td>
                    <td>F\u00f6rtr\u00e4nad p\u00e5 Sentinel-2 via Satlas</td>
                </tr>
                <tr>
                    <td>Prithvi-EO 2.0 (brandsegmentering)</td>
                    <td><span class="license-badge badge-open">Apache 2.0</span></td>
                    <td>&copy; IBM, NASA, J\u00fclich Supercomputing Centre</td>
                    <td>Geospatial foundation model (600M parametrar)</td>
                </tr>
                <tr>
                    <td>COT MLP5 ensemble (molndetektering)</td>
                    <td><span class="license-badge badge-restricted">Ej klargjord</span></td>
                    <td>Aleksis Pirinen / RISE</td>
                    <td>Pirinen et al., 2024. <a href="https://github.com/DigitalEarthSweden/ml-cloud-opt-thick" style="color:#171717;" target="_blank">GitHub</a> &mdash; kommersiell licens ej bekr\u00e4ftad</td>
                </tr>
                <tr>
                    <td>pib-ml-grazing CNN-biLSTM (betesanalys)</td>
                    <td><span class="license-badge badge-open">MIT</span></td>
                    <td>&copy; RISE / Jordbruksverket</td>
                    <td>Tidsserie\u00f6vervakad betesmarksklassificering. <a href="https://github.com/DigitalEarthSweden/pib-ml-grazing" style="color:#171717;" target="_blank">GitHub</a></td>
                </tr>
                <tr>
                    <td>PyTorch / Torchvision</td>
                    <td><span class="license-badge badge-open">BSD 3-Clause</span></td>
                    <td>PyTorch Foundation / Meta Platforms</td>
                    <td></td>
                </tr>
            </table>

            <h4>Data</h4>
            <table class="license-table">
                <tr><th>Datakälla</th><th>Licens</th><th>Upphovsr\u00e4tt</th><th>Anm\u00e4rkning</th></tr>
                <tr>
                    <td>Sentinel-2 L2A</td>
                    <td><span class="license-badge badge-open">Öppen &amp; fri</span></td>
                    <td>&copy; ESA / Copernicus</td>
                    <td>Fri anv\u00e4ndning, attribution rekommenderas</td>
                </tr>
                <tr>
                    <td>Digital Earth Sweden (openEO)</td>
                    <td><span class="license-badge badge-open">Apache 2.0 / CC0</span></td>
                    <td>Rymdstyrelsen / RISE</td>
                    <td>Kod: Apache 2.0, data: CC0</td>
                </tr>
                <tr>
                    <td>NMD (Nationellt Markt\u00e4ckedata)</td>
                    <td><span class="license-badge badge-open">CC0</span></td>
                    <td>Naturv\u00e5rdsverket</td>
                    <td>Public domain, attribution rekommenderas</td>
                </tr>
                <tr>
                    <td>LPIS Blockdatabas (Jordbruksverket)</td>
                    <td><span class="license-badge badge-open">CC BY 4.0</span></td>
                    <td>&copy; Jordbruksverket</td>
                    <td>Betesblock (ägoslag) via WFS. <a href="https://jordbruksverket.se/e-tjanster-databaser-och-appar/ovriga-e-tjanster-och-databaser/oppna-data" style="color:#171717;" target="_blank">Öppna data</a></td>
                </tr>
                <tr>
                    <td>Sj\u00f6kort (S-57) via SLU GET</td>
                    <td><span class="license-badge badge-restricted">Akademisk</span></td>
                    <td>&copy; Sj\u00f6fartsverket</td>
                    <td>Tillg\u00e4nglig via <a href="https://maps.slu.se/get/" style="color:#171717;" target="_blank">SLU GET</a>
                        f\u00f6r SLU-anst\u00e4llda/studenter. Publicering i vetenskapliga arbeten till\u00e5ten
                        med attribution: &ldquo;Sj\u00f6kortsdata &copy; Sj\u00f6fartsverket&rdquo;</td>
                </tr>
            </table>

            <h4>Bibliotek</h4>
            <table class="license-table">
                <tr><th>Bibliotek</th><th>Licens</th><th>Upphovsr\u00e4tt</th></tr>
                <tr>
                    <td>Leaflet.js</td>
                    <td><span class="license-badge badge-open">BSD 2-Clause</span></td>
                    <td>&copy; 2010&ndash;2026 Volodymyr Agafonkin</td>
                </tr>
                <tr>
                    <td>Chart.js</td>
                    <td><span class="license-badge badge-open">MIT</span></td>
                    <td>&copy; 2014&ndash;2024 Chart.js Contributors</td>
                </tr>
            </table>
            <p style="margin-top:14px; color:#6b7280;">
                <strong style="color:#374151;">OBS:</strong>
                YOLO11s distribueras under AGPL-3.0 vilket inneb\u00e4r att kommersiell anv\u00e4ndning
                utan \u00f6ppen k\u00e4llkod kr\u00e4ver en Enterprise-licens fr\u00e5n Ultralytics.
                Sj\u00f6kortsdata fr\u00e5n Sj\u00f6fartsverket tillhandah\u00e5lls via SLU GET f\u00f6r akademiskt bruk
                och kr\u00e4ver attribution vid publicering.
            </p>
        </div>
    </div>

    <script>{leaflet_js}</script>
    <script>{leaflet_sync_js}</script>
    <script>{chart_js}</script>
    <script>
    (function() {{
        'use strict';

        // ── Embed mode: hide DES logo when ?embed=1 or in iframe ────────
        (function() {{
            const params = new URLSearchParams(window.location.search);
            const isEmbed = params.get('embed') === '1' || window.self !== window.top;
            if (isEmbed) {{
                const logo = document.querySelector('.des-logo');
                if (logo) logo.style.display = 'none';
                const divider = document.querySelector('.header-divider');
                if (divider) divider.style.display = 'none';
                const footer = document.querySelector('.footer');
                if (footer) footer.style.display = 'none';
            }}
        }})();

        // ── Tab switching ────────────────────────────────────────────────
        const allMaps = {{}};  // id -> L.Map

        document.querySelectorAll('.theme-tab').forEach(function(tab) {{
            tab.addEventListener('click', function(e) {{
                e.preventDefault();
                document.querySelectorAll('.theme-tab').forEach(t => t.classList.remove('active'));
                this.classList.add('active');
                const target = this.dataset.tab;
                document.querySelectorAll('.tab-content').forEach(function(tc) {{
                    tc.classList.toggle('active', tc.id === 'tab-' + target);
                }});
                // Fix Leaflet size and re-fit bounds after showing hidden tab
                setTimeout(function() {{
                    Object.values(allMaps).forEach(function(m) {{
                        m.invalidateSize();
                        if (m._imgBounds) m.fitBounds(m._imgBounds);
                    }});
                }}, 50);
            }});
        }});

        // ── Create maps for a tab ────────────────────────────────────────
        function initMaps(viewers, images, imgH, imgW, hasBgToggle, geojsonData) {{
            const bounds = [[0, 0], [imgH, imgW]];
            const maps = [];
            const overlays = {{}};
            const bgLayers = {{}};

            viewers.forEach(function(v) {{
                const container = document.getElementById(v.id);
                // Vector layers don't need an image, but raster layers do
                if (!container || (!images[v.id] && !v.vector)) return;

                const map = L.map(v.id, {{
                    crs: L.CRS.Simple,
                    minZoom: -2,
                    maxZoom: 5,
                    attributionControl: false,
                    zoomSnap: 0.25,
                }});

                // Check if this specific panel has a bg-toggle element
                const cell = container.closest('.map-cell');
                const panelToggle = cell ? cell.querySelector('.bg-toggle') : null;
                const panelHasBg = hasBgToggle || !!panelToggle;

                // Background / overlay layer logic
                if (panelHasBg) {{
                    bgLayers[v.id] = {{}};
                    const prefix = v.id.split('-')[0];
                    const rgbId = prefix + '-rgb';

                    // Discover all bg keys from the toggle buttons
                    const bgKeys = [];
                    if (panelToggle) {{
                        panelToggle.querySelectorAll('.bg-btn').forEach(function(b) {{
                            bgKeys.push(b.dataset.bg);
                        }});
                    }}

                    // RGB / "efter" background layer
                    const rgbUrl = (v.key === 'rgb') ? images[v.id] : images[rgbId];
                    if (rgbUrl) {{
                        bgLayers[v.id].rgb = L.imageOverlay(
                            rgbUrl, bounds, {{zIndex: 0, opacity: 1}}
                        ).addTo(map);
                    }}

                    // Sjökort background layer (marine)
                    const sjokortId = prefix + '-sjokort';
                    if (images[sjokortId]) {{
                        bgLayers[v.id].sjokort = L.imageOverlay(
                            images[sjokortId], bounds, {{zIndex: 0, opacity: 0}}
                        ).addTo(map);
                    }}

                    // Baseline / "före" background layer (fire)
                    const baselineId = prefix + '-baseline';
                    if (images[baselineId]) {{
                        bgLayers[v.id].baseline = L.imageOverlay(
                            images[baselineId], bounds, {{zIndex: 0, opacity: 0}}
                        ).addTo(map);
                    }}

                    // NMD background layer (grazing LPIS toggle)
                    const nmdId = prefix + '-nmd';
                    if (images[nmdId] && v.key !== 'nmd') {{
                        bgLayers[v.id].nmd = L.imageOverlay(
                            images[nmdId], bounds, {{zIndex: 0, opacity: 0}}
                        ).addTo(map);
                    }}

                    if (v.key === 'rgb') {{
                        overlays[v.id] = bgLayers[v.id].rgb;
                    }} else if (v.vector && geojsonData) {{
                        // Vector overlay (GeoJSON polygons) — per-feature styling
                        const gjLayer = L.geoJSON(geojsonData, {{
                            style: function(feature) {{
                                const cls = feature.properties && feature.properties.predicted_class;
                                let color = '#aaaaaa';
                                if (cls === 1) color = '#00BFFF';
                                else if (cls === 0) color = '#E6119D';
                                return {{ color: color, weight: 2, fillOpacity: 0.15, opacity: 1 }};
                            }},
                            onEachFeature: function(feature, layer) {{
                                const p = feature.properties || {{}};
                                if (p.class_label) {{
                                    layer.bindPopup(
                                        '<b>Block ' + (p.blockid || '') + '</b><br>' +
                                        p.class_label + ' (' + Math.round((p.confidence||0)*100) + '%)'
                                    );
                                }}
                            }},
                            coordsToLatLng: function(coords) {{
                                return L.latLng(coords[1], coords[0]);
                            }},
                        }}).addTo(map);
                        overlays[v.id] = gjLayer;
                    }} else {{
                        const overlay = L.imageOverlay(images[v.id], bounds, {{zIndex: 1}}).addTo(map);
                        overlays[v.id] = overlay;
                    }}
                }} else {{
                    // No toggle available: static RGB background + overlay
                    if (v.vector && geojsonData) {{
                        // Vector overlay with RGB background
                        const prefix2 = v.id.split('-')[0];
                        const rgbId2 = prefix2 + '-rgb';
                        if (images[rgbId2]) {{
                            L.imageOverlay(images[rgbId2], bounds, {{zIndex: 0}}).addTo(map);
                        }}
                        const gjLayer = L.geoJSON(geojsonData, {{
                            style: function(feature) {{
                                const cls = feature.properties && feature.properties.predicted_class;
                                let color = '#aaaaaa';
                                if (cls === 1) color = '#00BFFF';
                                else if (cls === 0) color = '#E6119D';
                                return {{ color: color, weight: 2, fillOpacity: 0.15, opacity: 1 }};
                            }},
                            onEachFeature: function(feature, layer) {{
                                const p = feature.properties || {{}};
                                if (p.class_label) {{
                                    layer.bindPopup(
                                        '<b>Block ' + (p.blockid || '') + '</b><br>' +
                                        p.class_label + ' (' + Math.round((p.confidence||0)*100) + '%)'
                                    );
                                }}
                            }},
                            coordsToLatLng: function(coords) {{
                                return L.latLng(coords[1], coords[0]);
                            }},
                        }}).addTo(map);
                        overlays[v.id] = gjLayer;
                    }} else {{
                        if (v.key !== 'rgb') {{
                            const rgbId = v.id.split('-')[0] + '-rgb';
                            if (images[rgbId]) {{
                                L.imageOverlay(images[rgbId], bounds, {{zIndex: 0}}).addTo(map);
                            }}
                        }}
                        const overlay = L.imageOverlay(images[v.id], bounds, {{zIndex: 1}}).addTo(map);
                        overlays[v.id] = overlay;
                    }}
                }}

                map.fitBounds(bounds);
                map._imgBounds = bounds;
                maps.push(map);
                allMaps[v.id] = map;
            }});

            // Sync maps within tab
            for (let i = 0; i < maps.length; i++) {{
                for (let j = 0; j < maps.length; j++) {{
                    if (i !== j) maps[i].sync(maps[j]);
                }}
            }}

            // Opacity sliders
            document.querySelectorAll('.opacity-control input[type="range"]').forEach(function(slider) {{
                slider.addEventListener('input', function() {{
                    const mapId = this.dataset.mapId;
                    const val = parseInt(this.value);
                    const valEl = document.getElementById('opacity-val-' + mapId);
                    if (valEl) valEl.textContent = val + '%';
                    const ov = overlays[mapId];
                    if (ov) {{
                        if (ov.setOpacity) ov.setOpacity(val / 100);
                        else if (ov.setStyle) ov.setStyle({{ opacity: val / 100 }});
                    }}
                }});
            }});

            // Background toggle (generic: RGB ↔ Sjökort, Efter ↔ Före, etc.)
            document.querySelectorAll('.bg-toggle').forEach(function(toggle) {{
                toggle.querySelectorAll('.bg-btn').forEach(function(btn) {{
                    btn.addEventListener('click', function() {{
                        const mapId = toggle.dataset.mapId;
                        const activeBg = this.dataset.bg;
                        const layers = bgLayers[mapId];
                        if (!layers) return;
                        toggle.querySelectorAll('.bg-btn').forEach(b => b.classList.remove('active'));
                        this.classList.add('active');
                        // Hide all bg layers, then show the selected one
                        Object.keys(layers).forEach(function(key) {{
                            if (layers[key] && layers[key].setOpacity) {{
                                layers[key].setOpacity(key === activeBg ? 1 : 0);
                            }}
                        }});
                        // For RGB panel: update overlay ref (bg IS content)
                        // For analysis panels: keep analysis overlay as controlled layer
                        const prefix = mapId.split('-')[0];
                        if (mapId === prefix + '-rgb') {{
                            overlays[mapId] = (bg === 'rgb') ? layers.rgb : layers.sjokort;
                            const slider = document.getElementById('opacity-' + mapId);
                            if (slider) {{
                                slider.value = 100;
                                const valEl = document.getElementById('opacity-val-' + mapId);
                                if (valEl) valEl.textContent = '100%';
                            }}
                        }}
                    }});
                }});
            }});

            return {{ maps, overlays, bgLayers }};
        }}

        // ── Initialize both tabs ─────────────────────────────────────────
        const FIRE_VIEWERS = {fire_viewer_js};
        const FIRE_IMAGES = {fire_imgs_js};
        const MARINE_VIEWERS = {marine_viewer_js};
        const MARINE_IMAGES = {marine_imgs_js};

        initMaps(FIRE_VIEWERS, FIRE_IMAGES, {fire_h}, {fire_w}, false);
        {_marine_geojson_js}
        initMaps(MARINE_VIEWERS, MARINE_IMAGES, {marine_h}, {marine_w}, {str(has_marine_bg).lower()}, VESSEL_GEOJSON);

        {grazing_js_block}

        // ── Chart.js (Fire tab) ──────────────────────────────────────────
        const CHART_DATA = {fire_chart_json};

        Chart.defaults.color = 'rgba(207,248,228,0.6)';
        Chart.defaults.borderColor = 'rgba(207,248,228,0.1)';
        Chart.defaults.font.family = "'Space Grotesk', sans-serif";

        if (CHART_DATA.change && CHART_DATA.change.labels.length > 0) {{
            new Chart(document.getElementById('chart-change'), {{
                type: 'bar',
                data: {{
                    labels: CHART_DATA.change.labels,
                    datasets: [{{ label: 'Förändringsandel (%)', data: CHART_DATA.change.fractions,
                        backgroundColor: CHART_DATA.change.colors, borderColor: CHART_DATA.change.borders,
                        borderWidth: 1, borderRadius: 3 }}],
                }},
                options: {{
                    responsive: true,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        y: {{ beginAtZero: true, max: 100,
                              title: {{ display: true, text: 'Andel förändrad (%)' }},
                              grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
                        x: {{ grid: {{ display: false }} }},
                    }},
                }},
            }});

            if (CHART_DATA.change.dnbr) {{
                const dnbrColors = CHART_DATA.change.dnbr.map(function(v) {{
                    if (v < -0.25) return 'rgba(26,152,80,0.85)';
                    if (v < -0.1)  return 'rgba(145,207,96,0.85)';
                    if (v < 0.1)   return 'rgba(217,239,139,0.85)';
                    if (v < 0.27)  return 'rgba(254,224,139,0.85)';
                    if (v < 0.44)  return 'rgba(253,174,97,0.85)';
                    if (v < 0.66)  return 'rgba(244,109,67,0.85)';
                    return 'rgba(215,48,39,0.85)';
                }});
                new Chart(document.getElementById('chart-dnbr'), {{
                    type: 'bar',
                    data: {{
                        labels: CHART_DATA.change.labels,
                        datasets: [{{ label: 'Medel-dNBR', data: CHART_DATA.change.dnbr,
                            backgroundColor: dnbrColors,
                            borderColor: dnbrColors.map(c => c.replace('0.85', '1')),
                            borderWidth: 1, borderRadius: 3 }}],
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{ legend: {{ display: false }} }},
                        scales: {{
                            y: {{ title: {{ display: true, text: 'dNBR' }},
                                  grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
                            x: {{ grid: {{ display: false }} }},
                        }},
                    }},
                }});
            }}
        }}

        if (CHART_DATA.prithvi && CHART_DATA.prithvi.labels.length > 0) {{
            const datasets = [];
            CHART_DATA.prithvi.classes.forEach(function(cls) {{
                datasets.push({{ label: cls.label, data: cls.data,
                    backgroundColor: cls.color, borderColor: cls.border,
                    borderWidth: 1, borderRadius: 3 }});
            }});
            new Chart(document.getElementById('chart-prithvi'), {{
                type: 'bar',
                data: {{ labels: CHART_DATA.prithvi.labels, datasets: datasets }},
                options: {{
                    responsive: true,
                    plugins: {{ legend: {{ position: 'top' }} }},
                    scales: {{
                        x: {{ stacked: true, grid: {{ display: false }} }},
                        y: {{ stacked: true, beginAtZero: true, max: 100,
                              title: {{ display: true, text: 'Andel (%)' }},
                              grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
                    }},
                }},
            }});
        }}

        if (CHART_DATA.l2 && CHART_DATA.l2.labels.length > 0) {{
            new Chart(document.getElementById('chart-l2'), {{
                type: 'bar',
                data: {{
                    labels: CHART_DATA.l2.labels,
                    datasets: [{{ label: 'Andel (%)', data: CHART_DATA.l2.fractions,
                        backgroundColor: CHART_DATA.l2.colors, borderColor: CHART_DATA.l2.colors,
                        borderWidth: 1, borderRadius: 3 }}],
                }},
                options: {{
                    indexAxis: 'y',
                    responsive: true,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        x: {{ beginAtZero: true, title: {{ display: true, text: 'Andel (%)' }},
                              grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
                        y: {{ grid: {{ display: false }} }},
                    }},
                }},
            }});
        }}

        // ── Panel visibility toggle ──────────────────────────────────────
        function togglePanel(panelId, show) {{
            const cell = document.querySelector('.map-cell[data-panel-id="' + panelId + '"]');
            const chip = document.querySelector('.panel-chip[data-panel-id="' + panelId + '"]');
            if (!cell) return;
            if (show) {{
                cell.classList.remove('hidden-panel');
                if (chip) chip.classList.add('active');
                if (allMaps[panelId]) {{
                    setTimeout(function() {{ allMaps[panelId].invalidateSize(); }}, 50);
                }}
            }} else {{
                cell.classList.add('hidden-panel');
                if (chip) chip.classList.remove('active');
            }}
        }}

        document.querySelectorAll('.panel-chip').forEach(function(chip) {{
            chip.addEventListener('click', function() {{
                const pid = this.dataset.panelId;
                const isActive = this.classList.contains('active');
                if (!isActive) {{
                    togglePanel(pid, true);
                }}
                const cell = document.querySelector('.map-cell[data-panel-id="' + pid + '"]');
                if (cell) {{
                    cell.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }}
            }});
        }});

        document.querySelectorAll('.hide-panel-btn').forEach(function(btn) {{
            btn.addEventListener('click', function() {{
                togglePanel(this.dataset.panelId, false);
            }});
        }});

    }})();
    </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"    saved: {output_path}")
    return output_path
