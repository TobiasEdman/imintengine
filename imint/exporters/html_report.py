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
            '<h4>Fartygsdetektering</h4>'
            f'<div class="value">{total} fartyg</div>'
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
                f'<div class="value">{best["vessels"]} fartyg</div>'
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
    {"id": "m-vessels", "title": "Fartygsdetektering (YOLO)", "key": "vessels", "legend": [
        {"color": "#00E5FF", "label": "Detekterat fartyg"},
    ]},
    {"id": "m-vessel-heatmap", "title": "Fartygsaktivitet (heatmap)", "key": "vessel_heatmap", "legend": [
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


def save_tabbed_report(
    fire_dir: str,
    marine_dir: str,
    output_path: str,
    fire_date: str = "",
    marine_date: str = "",
) -> str:
    """Generate a tabbed HTML showcase with Fire and Marine analysis tabs.

    Uses EXTERNAL image files (no base64 embedding).  Images are copied
    to ``showcase/fire/`` and ``showcase/marine/`` subdirectories next
    to the output HTML.  The HTML references them via relative paths.

    Each tab contains its own set of Leaflet map viewers with opacity
    sliders, synced within the tab.  The Fire tab includes NMD charts.
    The Marine vessels panel has an RGB/Sjökort background toggle.

    Output structure::

        <output_dir>/
        ├── imint_showcase.html
        └── showcase/
            ├── fire/
            │   ├── rgb.png
            │   ├── nmd_overlay.png
            │   └── ...
            └── marine/
                ├── vessels_clean.png
                ├── sjokort.png
                └── ...

    Args:
        fire_dir: Path to fire analysis output directory.
        marine_dir: Path to marine analysis output directory.
        output_path: Where to write the HTML file.
        fire_date: Date string for fire analysis (e.g. "2018-07-24").
        marine_date: Date string for marine analysis (e.g. "2025-07-10").

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
            fname = prefix + file_map.get(key, f"{key}.png")
            src = os.path.join(out_dir, fname)
            if os.path.isfile(src):
                dst_name = file_map.get(key, f"{key}.png")
                dst = os.path.join(dest_dir, dst_name)
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy2(src, dst)
                # Relative path from HTML file to the image
                imgs[v["id"]] = f"showcase/{tab_name}/{dst_name}"
        return imgs

    fire_imgs = _copy_images(fire_dir, fire_prefix, _FIRE_VIEWERS, "fire")
    marine_imgs = _copy_images(marine_dir, marine_prefix, _MARINE_VIEWERS, "marine")

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

    fire_viewers = [v for v in _FIRE_VIEWERS if v["id"] in fire_imgs]
    marine_viewers = [v for v in _MARINE_VIEWERS if v["id"] in marine_imgs]

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

    # ── Fetch libraries ───────────────────────────────────────────────────
    leaflet_css = _fetch_lib(_CDN_LIBS["leaflet_css"])
    leaflet_js = _fetch_lib(_CDN_LIBS["leaflet_js"])
    leaflet_sync_js = _fetch_lib(_CDN_LIBS["leaflet_sync_js"])
    chart_js = _fetch_lib(_CDN_LIBS["chart_js"])

    # ── Build map cells HTML per tab ──────────────────────────────────────
    def _map_cells(viewers, tab_prefix, has_bg_toggle=False, hideable=False):
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
            if has_bg_toggle:
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
    fire_cells = _map_cells(fire_viewers, "f", hideable=True)
    marine_cells = _map_cells(marine_viewers, "m", has_bg_toggle=has_marine_bg, hideable=True)

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
        [{"id": v["id"], "key": v["key"]} for v in fire_viewers],
        ensure_ascii=False,
    )
    marine_viewer_js = json.dumps(
        [{"id": v["id"], "key": v["key"]} for v in marine_viewers],
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

    # ── Images as JS objects ──────────────────────────────────────────────
    def _imgs_js(imgs_dict):
        items = [f'    "{k}": "{v}"' for k, v in imgs_dict.items()]
        return "{\n" + ",\n".join(items) + "\n}"

    fire_imgs_js = _imgs_js(fire_imgs)
    marine_imgs_js = _imgs_js(marine_imgs)

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
            background: #1a4338;
            color: #cff8e4;
            line-height: 1.56;
        }}

        /* ── Header ──────────────────────────────────────────────────── */
        .header {{
            background: linear-gradient(135deg, #132e25 0%, #1a4338 100%);
            padding: 20px 32px;
            border-bottom: 1px solid #2d6354;
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
        .des-logo {{
            height: 42px;
            filter: invert(0.94) sepia(0.07) saturate(5.62) hue-rotate(97deg) brightness(0.98) contrast(1.01);
            flex-shrink: 0;
        }}
        .header-divider {{
            width: 1px;
            height: 36px;
            background: rgba(207,248,228,0.2);
            flex-shrink: 0;
        }}
        .header-left h1 {{
            font-size: 22px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.3px;
        }}
        .header-left h1 span {{
            color: #ffffff;
        }}
        .header-left p {{
            font-size: 13px;
            color: rgba(207,248,228,0.6);
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
            color: rgba(207,248,228,0.6);
            background: #245045;
            border: 1px solid #2d6354;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .theme-tab:hover {{
            color: #cff8e4;
            border-color: #cff8e4;
            background: rgba(207,248,228,0.1);
        }}
        .theme-tab.active {{
            color: #1a4338;
            background: #cff8e4;
            border-color: #cff8e4;
        }}

        /* ── Summary cards ───────────────────────────────────────────── */
        .summary-section {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            padding: 16px 20px 8px;
        }}
        .summary-card {{
            background: #163930;
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
            color: rgba(207,248,228,0.4);
            margin-bottom: 6px;
            font-weight: 600;
        }}
        .summary-card .value {{
            font-size: 22px;
            font-weight: 700;
            color: #ffffff;
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
            font-size: 16px;
            font-weight: 700;
            color: #cff8e4;
        }}

        /* ── Tab intro / description ────────────────────────────────── */
        .tab-intro {{
            max-width: 820px;
            margin: 0 auto;
            padding: 24px 28px 0;
            text-align: center;
        }}
        .tab-intro p {{
            font-size: 17px;
            line-height: 1.7;
            color: rgba(207,248,228,0.8);
            margin: 0;
        }}
        .tab-description {{
            max-width: 820px;
            margin: 0 auto;
            padding: 10px 28px 20px;
            border-bottom: 1px solid rgba(207,248,228,0.08);
            margin-bottom: 10px;
        }}
        .tab-description p {{
            font-size: 15px;
            line-height: 1.75;
            color: rgba(207,248,228,0.55);
            margin: 8px 0 0;
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
            background: #163930;
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
            color: rgba(207,248,228,0.4);
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
            font-size: 11px;
            color: rgba(207,248,228,0.4);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-right: 4px;
        }}
        .panel-chip {{
            font-size: 11px;
            padding: 4px 12px;
            border-radius: 9999px;
            border: 1px solid #2d6354;
            background: #245045;
            color: rgba(207,248,228,0.6);
            cursor: pointer;
            transition: all 0.15s;
            user-select: none;
        }}
        .panel-chip.active {{
            background: rgba(207,248,228,0.12);
            border-color: #cff8e4;
            color: #cff8e4;
        }}
        .panel-chip:hover {{
            border-color: #2d6354;
        }}

        /* ── Hide button in map-cell header ──────────────────────────── */
        .hide-panel-btn {{
            background: none;
            border: none;
            color: rgba(207,248,228,0.3);
            cursor: pointer;
            font-size: 16px;
            padding: 2px 6px;
            line-height: 1;
            border-radius: 4px;
            transition: all 0.15s;
        }}
        .hide-panel-btn:hover {{
            color: #ef4444;
            background: rgba(239, 68, 68, 0.1);
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
            color: rgba(207,248,228,0.4);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-right: 2px;
        }}
        .bg-btn {{
            font-size: 10px;
            padding: 2px 8px;
            border: 1px solid #2d6354;
            border-radius: 3px;
            background: transparent;
            color: rgba(207,248,228,0.4);
            cursor: pointer;
            transition: all 0.15s;
        }}
        .bg-btn:hover {{
            border-color: #2d6354;
            color: rgba(207,248,228,0.6);
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
            background: #163930;
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
            color: rgba(207,248,228,0.3);
            border-top: 1px solid #245045;
        }}
        .license-toggle {{
            background: none;
            border: 1px solid #2d6354;
            color: rgba(207,248,228,0.6);
            padding: 6px 16px;
            border-radius: 9999px;
            cursor: pointer;
            font-size: 11px;
            font-family: 'Space Grotesk', sans-serif;
            margin-top: 10px;
            transition: all 0.15s;
        }}
        .license-toggle:hover {{
            border-color: #cff8e4;
            color: #cff8e4;
        }}
        .license-section {{
            display: none;
            text-align: left;
            max-width: 900px;
            margin: 16px auto 0;
            padding: 20px 24px;
            background: #163930;
            border: 1px solid #245045;
            border-radius: 10px;
            font-size: 11px;
            color: rgba(207,248,228,0.6);
            line-height: 1.6;
        }}
        .license-section.open {{ display: block; }}
        .license-section h4 {{
            font-size: 12px; color: #cff8e4;
            margin: 14px 0 6px; font-weight: 600;
        }}
        .license-section h4:first-child {{ margin-top: 0; }}
        .license-table {{
            width: 100%; border-collapse: collapse; margin: 10px 0;
        }}
        .license-table th {{
            text-align: left; font-size: 10px;
            text-transform: uppercase; letter-spacing: 0.5px;
            color: rgba(207,248,228,0.4); padding: 6px 10px;
            border-bottom: 1px solid #245045;
        }}
        .license-table td {{
            padding: 6px 10px;
            border-bottom: 1px solid rgba(36,80,69,0.5);
            vertical-align: top;
        }}
        .license-table tr:last-child td {{ border-bottom: none; }}
        .license-badge {{
            display: inline-block; padding: 1px 7px;
            border-radius: 4px; font-size: 10px; font-weight: 600;
        }}
        .badge-open {{ background: rgba(34,197,94,0.15); color: #22c55e; }}
        .badge-restricted {{ background: rgba(234,179,8,0.15); color: #eab308; }}
        .badge-copyleft {{ background: rgba(239,68,68,0.15); color: #ef4444; }}

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
            <img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+IDxzdmcgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgeG1sbnM6c2VyaWY9Imh0dHA6Ly93d3cuc2VyaWYuY29tLyIgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgdmlld0JveD0iMCAwIDIzMCAxMTQiIHZlcnNpb249IjEuMSIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSIgc3R5bGU9ImZpbGwtcnVsZTpldmVub2RkO2NsaXAtcnVsZTpldmVub2RkO3N0cm9rZS1saW5lam9pbjpyb3VuZDtzdHJva2UtbWl0ZXJsaW1pdDoyOyI+IDxnIGlkPSJBcnRib2FyZDEiIHRyYW5zZm9ybT0ibWF0cml4KDAuOTQyOTIzLDAsMCwxLDAsMCkiPiA8Zz4gPGcgdHJhbnNmb3JtPSJtYXRyaXgoMC40NTc0MTksLTAuNDMxMzEsMC40NTc0MTksMC40MzEzMSwtMTA5LjUwNSw0Ni4yMzY1KSI+IDxyZWN0IHg9IjE3MyIgeT0iMTEzLjEiIHdpZHRoPSI0Ny4zIiBoZWlnaHQ9IjQ3LjMiPjwvcmVjdD4gPC9nPiA8ZyB0cmFuc2Zvcm09Im1hdHJpeCgwLjQ1NzQxOSwtMC40MzEzMSwwLjQ1NzQxOSwwLjQzMTMxLC0xMzMuNTY5LDQ1Ljg3MDUpIj4gPHJlY3QgeD0iMTU5LjUiIHk9IjE2My40IiB3aWR0aD0iMzUuOCIgaGVpZ2h0PSIzNS44Ij48L3JlY3Q+IDwvZz4gPGcgdHJhbnNmb3JtPSJtYXRyaXgoMC40NTc0MTksLTAuNDMxMzEsMC40NTc0MTksMC40MzEzMSwtMTU5LjEyMiw1NS44NzM5KSI+IDxyZWN0IHg9IjE1OS40IiB5PSIyMTkuMyIgd2lkdGg9IjM1LjgiIGhlaWdodD0iMzUuOCI+PC9yZWN0PiA8L2c+IDxnIHRyYW5zZm9ybT0ibWF0cml4KDAuNDU3NDE5LC0wLjQzMTMxLDAuNDU3NDE5LDAuNDMxMzEsLTE3NC45Nyw1My42NzgxKSI+IDxyZWN0IHg9IjE1MC45IiB5PSIyNTUuMiIgd2lkdGg9IjE5LjciIGhlaWdodD0iMTkuNyI+PC9yZWN0PiA8L2c+IDxnIHRyYW5zZm9ybT0ibWF0cml4KDAuNDU3NDE5LC0wLjQzMTMxLDAuNDU3NDE5LDAuNDMxMzEsLTE3MC41NzEsNDMuNjEzNikiPiA8cmVjdCB4PSIxMzQuMyIgeT0iMjM4LjYiIHdpZHRoPSIxOS43IiBoZWlnaHQ9IjE5LjciPjwvcmVjdD4gPC9nPiA8ZyB0cmFuc2Zvcm09Im1hdHJpeCgwLjQ1NzQxOSwtMC40MzEzMSwwLjQ1NzQxOSwwLjQzMTMxLC0xNDkuNDgzLDQzLjczNTYpIj4gPHJlY3QgeD0iMTUwLjkiIHk9IjE5OS40IiB3aWR0aD0iMTkuNyIgaGVpZ2h0PSIxOS43Ij48L3JlY3Q+IDwvZz4gPGcgdHJhbnNmb3JtPSJtYXRyaXgoMC40NTc0MTksLTAuNDMxMzEsMC40NTc0MTksMC40MzEzMSwtMTg0LjY3NCw1MS44NDgyKSI+IDxyZWN0IHg9IjE0My42IiB5PSIyNzUuNiIgd2lkdGg9IjEyLjEiIGhlaWdodD0iMTIuMSI+PC9yZWN0PiA8L2c+IDxnIHRyYW5zZm9ybT0ibWF0cml4KDAuNDU3NDE5LC0wLjQzMTMxLDAuNDU3NDE5LDAuNDMxMzEsLTE3NS4yOTQsNjIuMTU2NikiPiA8cmVjdCB4PSIxNzMuOSIgeT0iMjY5LjEiIHdpZHRoPSI2LjkiIGhlaWdodD0iNi45Ij48L3JlY3Q+IDwvZz4gPGcgdHJhbnNmb3JtPSJtYXRyaXgoMC40NTc0MTksLTAuNDMxMzEsMC40NTc0MTksMC40MzEzMSwtMTY5Ljc5NSw2NS41NzI0KSI+IDxyZWN0IHg9IjE4Mi40IiB5PSIyNTkuMiIgd2lkdGg9IjExLjciIGhlaWdodD0iMTEuNyI+PC9yZWN0PiA8L2c+IDxnIHRyYW5zZm9ybT0ibWF0cml4KDAuNjQ2ODg4LDAsMCwwLjYwOTk2NSwtODQuNjMxOCwtNjMuMDA4MykiPiA8cGF0aCBkPSJNMjI5LjgsMTk5LjVMMjI5LjgsMTYxLjdMMjQxLjUsMTYxLjdDMjQ2LjUsMTYxLjcgMjUwLjQsMTYzIDI1My4xLDE2NS41QzI1NS44LDE2OCAyNTcuMiwxNzEuOSAyNTcuMiwxNzcuMUwyNTcuMiwxODQuMUMyNTcuMiwxODkuNCAyNTUuOCwxOTMuMiAyNTMuMSwxOTUuN0MyNTAuNCwxOTguMiAyNDYuNSwxOTkuNSAyNDEuNSwxOTkuNUwyMjkuOCwxOTkuNVpNMjM0LjMsMTk1LjRMMjQxLjUsMTk1LjRDMjQ1LjEsMTk1LjQgMjQ3LjksMTk0LjUgMjQ5LjgsMTkyLjdDMjUxLjcsMTkwLjkgMjUyLjYsMTg4LjEgMjUyLjYsMTg0LjNMMjUyLjYsMTc2LjlDMjUyLjYsMTczIDI1MS43LDE3MC4yIDI0OS44LDE2OC41QzI0Ny45LDE2Ni43IDI0NS4yLDE2NS45IDI0MS41LDE2NS45TDIzNC4zLDE2NS45TDIzNC4zLDE5NS41TDIzNC4zLDE5NS40WiIgc3R5bGU9ImZpbGwtcnVsZTpub256ZXJvOyI+PC9wYXRoPiA8L2c+IDxnIHRyYW5zZm9ybT0ibWF0cml4KDAuNjQ2ODg4LDAsMCwwLjYwOTk2NSwtODQuNjMxOCwtNjMuMDA4MykiPiA8cGF0aCBkPSJNMjY0LjksMTY5QzI2NCwxNjkgMjYzLjIsMTY4LjcgMjYyLjUsMTY4LjFDMjYxLjgsMTY3LjUgMjYxLjUsMTY2LjcgMjYxLjUsMTY1LjhDMjYxLjUsMTY0LjkgMjYxLjgsMTY0IDI2Mi41LDE2My40QzI2My4xLDE2Mi44IDI2My45LDE2Mi41IDI2NC45LDE2Mi41QzI2NS45LDE2Mi41IDI2Ni42LDE2Mi44IDI2Ny4yLDE2My40QzI2Ny44LDE2NCAyNjguMSwxNjQuOCAyNjguMSwxNjUuOEMyNjguMSwxNjYuOCAyNjcuOCwxNjcuNSAyNjcuMiwxNjguMUMyNjYuNiwxNjguNyAyNjUuOCwxNjkgMjY0LjksMTY5Wk0yNjIuOCwxOTkuNUwyNjIuOCwxNzMuMUwyNjcuMSwxNzMuMUwyNjcuMSwxOTkuNUwyNjIuOCwxOTkuNVoiIHN0eWxlPSJmaWxsLXJ1bGU6bm9uemVybzsiPjwvcGF0aD4gPC9nPiA8ZyB0cmFuc2Zvcm09Im1hdHJpeCgwLjY0Njg4OCwwLDAsMC42MDk5NjUsLTg0LjYzMTgsLTYzLjAwODMpIj4gPHBhdGggZD0iTTI4MCwyMTAuNUMyNzguMiwyMTAuNSAyNzYuNywyMTAuMiAyNzUuNSwyMDkuN0MyNzQuMywyMDkuMSAyNzMuNCwyMDguNCAyNzIuOCwyMDcuNEMyNzIuMiwyMDYuNCAyNzEuOSwyMDUuMyAyNzEuOSwyMDQuMUMyNzEuOSwyMDIuOSAyNzIuMiwyMDEuNSAyNzIuOSwyMDAuN0MyNzMuNiwxOTkuOSAyNzQuMywxOTkuMyAyNzQuOSwxOTlMMjc0LjksMTk4LjRDMjc0LjMsMTk4IDI3My43LDE5Ny41IDI3My4yLDE5Ni44QzI3Mi43LDE5Ni4xIDI3Mi40LDE5NS4yIDI3Mi40LDE5NEMyNzIuNCwxOTIuOCAyNzIuNywxOTEuNyAyNzMuNCwxOTFDMjc0LjEsMTkwLjMgMjc0LjgsMTg5LjcgMjc1LjUsMTg5LjRMMjc1LjUsMTg4LjRMMjc4LjksMTkwLjdDMjc4LjIsMTkwLjggMjc3LjYsMTkxLjEgMjc3LDE5MS41QzI3Ni40LDE5MS45IDI3Ni4xLDE5Mi42IDI3Ni4xLDE5My41QzI3Ni4xLDE5NC40IDI3Ni40LDE5NS4xIDI3Ny4xLDE5NS41QzI3Ny43LDE5NS45IDI3OC42LDE5Ni4xIDI3OS42LDE5Ni4xTDI4OC43LDE5Ni4xQzI5MS40LDE5Ni4xIDI5My41LDE5Ni43IDI5NSwxOTcuOUMyOTYuNSwxOTkuMSAyOTcuMiwyMDAuOCAyOTcuMiwyMDMuMUwyOTcuMiwyMDMuNUMyOTcuMiwyMDUuNyAyOTYuNSwyMDcuNCAyOTUsMjA4LjdDMjkzLjUsMjEwIDI5MS40LDIxMC42IDI4OC43LDIxMC42TDI4MCwyMTAuNkwyODAsMjEwLjVaTTI4My40LDE5MS42QzI4MS40LDE5MS42IDI3OS42LDE5MS4yIDI3OCwxOTAuNUMyNzYuNCwxODkuOCAyNzUuMiwxODguNyAyNzQuMywxODcuM0MyNzMuNCwxODUuOSAyNzIuOSwxODQuMSAyNzIuOSwxODIuMUwyNzIuOSwxODEuOUMyNzIuOSwxNzkuOCAyNzMuNCwxNzguMSAyNzQuMywxNzYuN0MyNzUuMiwxNzUuMyAyNzYuNCwxNzQuMiAyNzgsMTczLjVDMjc5LjYsMTcyLjggMjgxLjMsMTcyLjQgMjgzLjMsMTcyLjRDMjg1LjMsMTcyLjQgMjg0LjksMTcyLjQgMjg1LjcsMTcyLjZDMjg2LjQsMTcyLjggMjg3LjEsMTcyLjkgMjg3LjcsMTczLjFMMjk2LjksMTczLjFMMjk2LjksMTc2LjZMMjkxLjksMTc2LjZMMjkxLjksMTc3LjJDMjkyLjQsMTc3LjggMjkyLjgsMTc4LjYgMjkzLjEsMTc5LjRDMjkzLjQsMTgwLjIgMjkzLjYsMTgxLjEgMjkzLjYsMTgyLjJDMjkzLjYsMTg1LjIgMjkyLjcsMTg3LjUgMjkwLjgsMTg5LjFDMjg4LjksMTkwLjcgMjg2LjUsMTkxLjUgMjgzLjUsMTkxLjVMMjgzLjQsMTkxLjZaTTI4MCwyMDdMMjg4LjksMjA3QzI5MC4zLDIwNyAyOTEuMywyMDYuNyAyOTIuMSwyMDYuMUMyOTIuOSwyMDUuNSAyOTMuMywyMDQuNiAyOTMuMywyMDMuNEMyOTMuMywyMDIuMiAyOTIuOSwyMDEuMyAyOTIuMSwyMDAuN0MyOTEuMywyMDAuMSAyOTAuMiwxOTkuOCAyODguOSwxOTkuOEwyODAsMTk5LjhDMjc4LjYsMTk5LjggMjc3LjYsMjAwLjEgMjc2LjgsMjAwLjdDMjc2LDIwMS4zIDI3NS42LDIwMi4yIDI3NS42LDIwMy40QzI3NS42LDIwNC42IDI3NiwyMDUuNSAyNzYuOCwyMDYuMUMyNzcuNiwyMDYuNyAyNzguNywyMDcgMjgwLDIwN1pNMjgzLjQsMTg4LjFDMjg1LjIsMTg4LjEgMjg2LjgsMTg3LjUgMjg4LDE4Ni40QzI4OS4yLDE4NS4zIDI4OS44LDE4My44IDI4OS44LDE4MS45QzI4OS44LDE4MCAyODkuMiwxNzguNSAyODgsMTc3LjRDMjg2LjgsMTc2LjMgMjg1LjMsMTc1LjcgMjgzLjQsMTc1LjdDMjgxLjUsMTc1LjcgMjgwLDE3Ni4zIDI3OC44LDE3Ny40QzI3Ny42LDE3OC41IDI3NywxODAgMjc3LDE4MS45QzI3NywxODMuOCAyNzcuNiwxODUuMyAyNzguOCwxODYuNEMyODAsMTg3LjUgMjgxLjYsMTg4LjEgMjgzLjQsMTg4LjFaIiBzdHlsZT0iZmlsbC1ydWxlOm5vbnplcm87Ij48L3BhdGg+IDwvZz4gPGcgdHJhbnNmb3JtPSJtYXRyaXgoMC42NDY4ODgsMCwwLDAuNjA5OTY1LC04NC42MzE4LC02My4wMDgzKSI+IDxwYXRoIGQ9Ik0zMDMuMSwxNjlDMzAyLjIsMTY5IDMwMS40LDE2OC43IDMwMC43LDE2OC4xQzMwMCwxNjcuNSAyOTkuNywxNjYuNyAyOTkuNywxNjUuOEMyOTkuNywxNjQuOSAzMDAsMTY0IDMwMC43LDE2My40QzMwMS4zLDE2Mi44IDMwMi4xLDE2Mi41IDMwMy4xLDE2Mi41QzMwNC4xLDE2Mi41IDMwNC44LDE2Mi44IDMwNS40LDE2My40QzMwNiwxNjQgMzA2LjMsMTY0LjggMzA2LjMsMTY1LjhDMzA2LjMsMTY2LjggMzA2LDE2Ny41IDMwNS40LDE2OC4xQzMwNC44LDE2OC43IDMwNCwxNjkgMzAzLjEsMTY5Wk0zMDAuOSwxOTkuNUwzMDAuOSwxNzMuMUwzMDUuMiwxNzMuMUwzMDUuMiwxOTkuNUwzMDAuOSwxOTkuNVoiIHN0eWxlPSJmaWxsLXJ1bGU6bm9uemVybzsiPjwvcGF0aD4gPC9nPiA8ZyB0cmFuc2Zvcm09Im1hdHJpeCgwLjY0Njg4OCwwLDAsMC42MDk5NjUsLTg0LjYzMTgsLTYzLjAwODMpIj4gPHBhdGggZD0iTTMyMS43LDE5OS41QzMyMC4yLDE5OS41IDMxOSwxOTkuMSAzMTguMiwxOTguM0MzMTcuNCwxOTcuNSAzMTcsMTk2LjMgMzE3LDE5NC45TDMxNywxNzYuOEwzMDksMTc2LjhMMzA5LDE3My4xTDMxNywxNzMuMUwzMTcsMTYzLjdMMzIxLjMsMTYzLjdMMzIxLjMsMTczLjFMMzI5LjksMTczLjFMMzI5LjksMTc2LjhMMzIxLjMsMTc2LjhMMzIxLjMsMTk0LjJDMzIxLjMsMTk1LjMgMzIxLjgsMTk1LjggMzIyLjksMTk1LjhMMzI4LjgsMTk1LjhMMzI4LjgsMTk5LjVMMzIxLjcsMTk5LjVaIiBzdHlsZT0iZmlsbC1ydWxlOm5vbnplcm87Ij48L3BhdGg+IDwvZz4gPGcgdHJhbnNmb3JtPSJtYXRyaXgoMC42NDY4ODgsMCwwLDAuNjA5OTY1LC04NC42MzE4LC02My4wMDgzKSI+IDxwYXRoIGQ9Ik0zNDIuNywyMDAuM0MzNDAuOSwyMDAuMyAzMzkuMiwyMDAgMzM3LjcsMTk5LjNDMzM2LjIsMTk4LjcgMzM1LDE5Ny43IDMzNC4xLDE5Ni41QzMzMy4yLDE5NS4zIDMzMi44LDE5My44IDMzMi44LDE5MkMzMzIuOCwxOTAuMiAzMzMuMiwxODguNyAzMzQuMSwxODcuNUMzMzUsMTg2LjMgMzM2LjEsMTg1LjQgMzM3LjcsMTg0LjlDMzM5LjIsMTg0LjMgMzQwLjksMTg0IDM0Mi44LDE4NEwzNTEuNywxODRMMzUxLjcsMTgyLjFDMzUxLjcsMTgwLjIgMzUxLjIsMTc4LjcgMzUwLjEsMTc3LjdDMzQ5LDE3Ni42IDM0Ny40LDE3Ni4xIDM0NS4yLDE3Ni4xQzM0MywxNzYuMSAzNDEuNCwxNzYuNiAzNDAuMiwxNzcuNkMzMzksMTc4LjYgMzM4LjIsMTgwIDMzNy44LDE4MS43TDMzMy44LDE4MC40QzMzNC4yLDE3OC45IDMzNC45LDE3Ny42IDMzNS44LDE3Ni40QzMzNi43LDE3NS4yIDMzOCwxNzQuMiAzMzkuNiwxNzMuNUMzNDEuMiwxNzIuOCAzNDMuMSwxNzIuNCAzNDUuMiwxNzIuNEMzNDguNiwxNzIuNCAzNTEuMiwxNzMuMyAzNTMuMSwxNzVDMzU1LDE3Ni44IDM1NS45LDE3OS4yIDM1NS45LDE4Mi40TDM1NS45LDE5NC4zQzM1NS45LDE5NS40IDM1Ni40LDE5NS45IDM1Ny40LDE5NS45TDM1OS44LDE5NS45TDM1OS44LDE5OS41TDM1Ni4xLDE5OS41QzM1NC45LDE5OS41IDM1NCwxOTkuMiAzNTMuMiwxOTguNUMzNTIuNSwxOTcuOCAzNTIuMSwxOTYuOSAzNTIuMSwxOTUuOEwzNTIuMSwxOTUuNUwzNTEuNSwxOTUuNUMzNTEuMSwxOTYuMiAzNTAuNiwxOTYuOSAzNDkuOSwxOTcuNkMzNDkuMiwxOTguMyAzNDguMywxOTkgMzQ3LjIsMTk5LjRDMzQ2LjEsMTk5LjkgMzQ0LjYsMjAwLjEgMzQyLjcsMjAwLjFMMzQyLjcsMjAwLjNaTTM0My4xLDE5Ni42QzM0NS42LDE5Ni42IDM0Ny43LDE5NS44IDM0OS4yLDE5NC40QzM1MC43LDE5MyAzNTEuNiwxOTAuOCAzNTEuNiwxODguMUwzNTEuNiwxODcuNUwzNDIuOCwxODcuNUMzNDEuMSwxODcuNSAzMzkuNywxODcuOSAzMzguNiwxODguNkMzMzcuNSwxODkuMyAzMzcsMTkwLjUgMzM3LDE5MS45QzMzNywxOTMuMyAzMzcuNiwxOTQuNSAzMzguNywxOTUuNEMzMzkuOCwxOTYuMiAzNDEuMywxOTYuNiAzNDMuMSwxOTYuNloiIHN0eWxlPSJmaWxsLXJ1bGU6bm9uemVybzsiPjwvcGF0aD4gPC9nPiA8ZyB0cmFuc2Zvcm09Im1hdHJpeCgwLjY0Njg4OCwwLDAsMC42MDk5NjUsLTg0LjYzMTgsLTYzLjAwODMpIj4gPHJlY3QgeD0iMzYzLjUiIHk9IjE2MS43IiB3aWR0aD0iNC4zIiBoZWlnaHQ9IjM3LjgiPjwvcmVjdD4gPC9nPiA8ZyB0cmFuc2Zvcm09Im1hdHJpeCgwLjY0Njg4OCwwLDAsMC42MDk5NjUsLTg0LjYzMTgsLTYzLjAwODMpIj4gPHBhdGggZD0iTTM4NCwxOTkuNUwzODQsMTYxLjdMNDA3LjMsMTYxLjdMNDA3LjMsMTY1LjhMMzg4LjUsMTY1LjhMMzg4LjUsMTc4LjRMNDA1LjcsMTc4LjRMNDA1LjcsMTgyLjVMMzg4LjUsMTgyLjVMMzg4LjUsMTk1LjRMNDA3LjUsMTk1LjRMNDA3LjUsMTk5LjVMMzgzLjksMTk5LjVMMzg0LDE5OS41WiIgc3R5bGU9ImZpbGwtcnVsZTpub256ZXJvOyI+PC9wYXRoPiA8L2c+IDxnIHRyYW5zZm9ybT0ibWF0cml4KDAuNjQ2ODg4LDAsMCwwLjYwOTk2NSwtODQuNjMxOCwtNjMuMDA4MykiPiA8cGF0aCBkPSJNNDIwLjUsMjAwLjNDNDE4LjcsMjAwLjMgNDE3LDIwMCA0MTUuNSwxOTkuM0M0MTQsMTk4LjcgNDEyLjgsMTk3LjcgNDExLjksMTk2LjVDNDExLDE5NS4zIDQxMC42LDE5My44IDQxMC42LDE5MkM0MTAuNiwxOTAuMiA0MTEsMTg4LjcgNDExLjksMTg3LjVDNDEyLjgsMTg2LjMgNDEzLjksMTg1LjQgNDE1LjUsMTg0LjlDNDE3LDE4NC4zIDQxOC43LDE4NCA0MjAuNiwxODRMNDI5LjUsMTg0TDQyOS41LDE4Mi4xQzQyOS41LDE4MC4yIDQyOSwxNzguNyA0MjcuOSwxNzcuN0M0MjYuOCwxNzYuNiA0MjUuMiwxNzYuMSA0MjMsMTc2LjFDNDIwLjgsMTc2LjEgNDE5LjIsMTc2LjYgNDE4LDE3Ny42QzQxNi44LDE3OC42IDQxNiwxODAgNDE1LjYsMTgxLjdMNDExLjYsMTgwLjRDNDEyLDE3OC45IDQxMi43LDE3Ny42IDQxMy42LDE3Ni40QzQxNC41LDE3NS4yIDQxNS44LDE3NC4yIDQxNy40LDE3My41QzQxOSwxNzIuOCA0MjAuOSwxNzIuNCA0MjMsMTcyLjRDNDI2LjQsMTcyLjQgNDI5LDE3My4zIDQzMC45LDE3NUM0MzIuOCwxNzYuOCA0MzMuNywxNzkuMiA0MzMuNywxODIuNEw0MzMuNywxOTQuM0M0MzMuNywxOTUuNCA0MzQuMiwxOTUuOSA0MzUuMiwxOTUuOUw0MzcuNiwxOTUuOUw0MzcuNiwxOTkuNUw0MzMuOSwxOTkuNUM0MzIuNywxOTkuNSA0MzEuOCwxOTkuMiA0MzEsMTk4LjVDNDMwLjMsMTk3LjggNDI5LjksMTk2LjkgNDI5LjksMTk1LjhMNDI5LjksMTk1LjVMNDI5LjMsMTk1LjVDNDI4LjksMTk2LjIgNDI4LjQsMTk2LjkgNDI3LjcsMTk3LjZDNDI3LDE5OC4zIDQyNi4xLDE5OSA0MjUsMTk5LjRDNDIzLjksMTk5LjkgNDIyLjQsMjAwLjEgNDIwLjUsMjAwLjFMNDIwLjUsMjAwLjNaTTQyMSwxOTYuNkM0MjMuNSwxOTYuNiA0MjUuNiwxOTUuOCA0MjcuMSwxOTQuNEM0MjguNiwxOTMgNDI5LjUsMTkwLjggNDI5LjUsMTg4LjFMNDI5LjUsMTg3LjVMNDIwLjcsMTg3LjVDNDE5LDE4Ny41IDQxNy42LDE4Ny45IDQxNi41LDE4OC42QzQxNS40LDE4OS4zIDQxNC45LDE5MC41IDQxNC45LDE5MS45QzQxNC45LDE5My4zIDQxNS41LDE5NC41IDQxNi42LDE5NS40QzQxNy43LDE5Ni4yIDQxOS4yLDE5Ni42IDQyMSwxOTYuNloiIHN0eWxlPSJmaWxsLXJ1bGU6bm9uemVybzsiPjwvcGF0aD4gPC9nPiA8ZyB0cmFuc2Zvcm09Im1hdHJpeCgwLjY0Njg4OCwwLDAsMC42MDk5NjUsLTg0LjYzMTgsLTYzLjAwODMpIj4gPHBhdGggZD0iTTQ0MS4zLDE5OS41TDQ0MS4zLDE3My4xTDQ0NS41LDE3My4xTDQ0NS41LDE3Ni4zTDQ0Ni4zLDE3Ni4zQzQ0Ni44LDE3NS4yIDQ0Ny41LDE3NC4zIDQ0OC41LDE3My44QzQ0OS41LDE3My4zIDQ1MC43LDE3MyA0NTIuMywxNzNMNDU1LjQsMTczTDQ1NS40LDE3Ni45TDQ1MiwxNzYuOUM0NTAuMSwxNzYuOSA0NDguNSwxNzcuNCA0NDcuNCwxNzguNUM0NDYuMiwxNzkuNiA0NDUuNiwxODEuMiA0NDUuNiwxODMuNEw0NDUuNiwxOTkuNUw0NDEuMywxOTkuNVoiIHN0eWxlPSJmaWxsLXJ1bGU6bm9uemVybzsiPjwvcGF0aD4gPC9nPiA8ZyB0cmFuc2Zvcm09Im1hdHJpeCgwLjY0Njg4OCwwLDAsMC42MDk5NjUsLTg0LjYzMTgsLTYzLjAwODMpIj4gPHBhdGggZD0iTTQ3MS40LDE5OS41QzQ2OS45LDE5OS41IDQ2OC43LDE5OS4xIDQ2Ny45LDE5OC4zQzQ2Ny4xLDE5Ny41IDQ2Ni43LDE5Ni4zIDQ2Ni43LDE5NC45TDQ2Ni43LDE3Ni44TDQ1OC43LDE3Ni44TDQ1OC43LDE3My4xTDQ2Ni43LDE3My4xTDQ2Ni43LDE2My43TDQ3MSwxNjMuN0w0NzEsMTczLjFMNDc5LjYsMTczLjFMNDc5LjYsMTc2LjhMNDcxLDE3Ni44TDQ3MSwxOTQuMkM0NzEsMTk1LjMgNDcxLjUsMTk1LjggNDcyLjYsMTk1LjhMNDc4LjUsMTk1LjhMNDc4LjUsMTk5LjVMNDcxLjQsMTk5LjVaIiBzdHlsZT0iZmlsbC1ydWxlOm5vbnplcm87Ij48L3BhdGg+IDwvZz4gPGcgdHJhbnNmb3JtPSJtYXRyaXgoMC42NDY4ODgsMCwwLDAuNjA5OTY1LC04NC42MzE4LC02My4wMDgzKSI+IDxwYXRoIGQ9Ik00ODMuMywxOTkuNUw0ODMuMywxNjEuN0w0ODcuNiwxNjEuN0w0ODcuNiwxNzcuM0w0ODguNCwxNzcuM0M0ODguOCwxNzYuNSA0ODkuMywxNzUuOCA0OTAsMTc1LjFDNDkwLjcsMTc0LjQgNDkxLjcsMTczLjggNDkyLjksMTczLjNDNDk0LjEsMTcyLjggNDk1LjYsMTcyLjYgNDk3LjQsMTcyLjZDNDk5LjIsMTcyLjYgNTAxLjIsMTczIDUwMi43LDE3My45QzUwNC4zLDE3NC44IDUwNS41LDE3NiA1MDYuNSwxNzcuNkM1MDcuNCwxNzkuMiA1MDcuOSwxODEuMiA1MDcuOSwxODMuNkw1MDcuOSwxOTkuNkw1MDMuNiwxOTkuNkw1MDMuNiwxODMuOUM1MDMuNiwxODEuMyA1MDIuOSwxNzkuNCA1MDEuNiwxNzguMkM1MDAuMywxNzcgNDk4LjUsMTc2LjQgNDk2LjMsMTc2LjRDNDk0LjEsMTc2LjQgNDkxLjcsMTc3LjIgNDkwLjEsMTc4LjlDNDg4LjUsMTgwLjYgNDg3LjcsMTgzIDQ4Ny43LDE4Ni4zTDQ4Ny43LDE5OS42TDQ4My40LDE5OS42TDQ4My4zLDE5OS41WiIgc3R5bGU9ImZpbGwtcnVsZTpub256ZXJvOyI+PC9wYXRoPiA8L2c+IDxnIHRyYW5zZm9ybT0ibWF0cml4KDAuNjQ2ODg4LDAsMCwwLjYwOTk2NSwtODQuNjMxOCwtNjMuMDA4MykiPiA8cGF0aCBkPSJNMjQyLjQsMjU1LjVDMjM5LjcsMjU1LjUgMjM3LjIsMjU1IDIzNSwyNTRDMjMyLjgsMjUzIDIzMS4xLDI1MS42IDIyOS45LDI0OS42QzIyOC43LDI0Ny42IDIyOCwyNDUuMiAyMjgsMjQyLjJMMjI4LDI0MS4xTDIzMi41LDI0MS4xTDIzMi41LDI0Mi4yQzIzMi41LDI0NS4zIDIzMy40LDI0Ny43IDIzNS4zLDI0OS4yQzIzNy4yLDI1MC43IDIzOS41LDI1MS41IDI0Mi40LDI1MS41QzI0NS4zLDI1MS41IDI0Ny42LDI1MC44IDI0OS4yLDI0OS41QzI1MC44LDI0OC4yIDI1MS41LDI0Ni41IDI1MS41LDI0NC41QzI1MS41LDI0Mi41IDI1MS4xLDI0MiAyNTAuNCwyNDEuMUMyNDkuNywyNDAuMiAyNDguNywyMzkuNiAyNDcuNSwyMzkuMUMyNDYuMiwyMzguNiAyNDQuOCwyMzguMSAyNDMuMSwyMzcuN0wyNDAuMywyMzdDMjM4LjEsMjM2LjQgMjM2LjIsMjM1LjcgMjM0LjUsMjM0LjlDMjMyLjgsMjM0LjEgMjMxLjUsMjMzIDIzMC41LDIzMS42QzIyOS41LDIzMC4yIDIyOS4xLDIyOC41IDIyOS4xLDIyNi40QzIyOS4xLDIyNC4zIDIyOS42LDIyMi40IDIzMC43LDIyMC45QzIzMS44LDIxOS4zIDIzMy4zLDIxOC4yIDIzNS4yLDIxNy4zQzIzNy4xLDIxNi41IDIzOS4zLDIxNi4xIDI0MS45LDIxNi4xQzI0NC41LDIxNi4xIDI0Ni44LDIxNi41IDI0OC44LDIxNy40QzI1MC44LDIxOC4zIDI1Mi40LDIxOS42IDI1My42LDIyMS4zQzI1NC44LDIyMyAyNTUuNCwyMjUuMiAyNTUuNCwyMjcuOEwyNTUuNCwyMzAuMUwyNTAuOSwyMzAuMUwyNTAuOSwyMjcuOEMyNTAuOSwyMjYgMjUwLjUsMjI0LjUgMjQ5LjcsMjIzLjRDMjQ4LjksMjIyLjMgMjQ3LjksMjIxLjUgMjQ2LjUsMjIwLjlDMjQ1LjEsMjIwLjQgMjQzLjYsMjIwLjEgMjQxLjksMjIwLjFDMjM5LjQsMjIwLjEgMjM3LjQsMjIwLjcgMjM1LjksMjIxLjdDMjM0LjQsMjIyLjggMjMzLjYsMjI0LjQgMjMzLjYsMjI2LjRDMjMzLjYsMjI4LjQgMjMzLjksMjI4LjggMjM0LjYsMjI5LjZDMjM1LjMsMjMwLjQgMjM2LjIsMjMxLjEgMjM3LjQsMjMxLjZDMjM4LjYsMjMyLjEgMjQwLDIzMi42IDI0MS43LDIzM0wyNDQuNSwyMzMuN0MyNDYuNywyMzQuMiAyNDguNywyMzQuOCAyNTAuNCwyMzUuNkMyNTIuMSwyMzYuNCAyNTMuNSwyMzcuNSAyNTQuNiwyMzguOUMyNTUuNiwyNDAuMyAyNTYuMiwyNDIuMSAyNTYuMiwyNDQuM0MyNTYuMiwyNDYuNSAyNTUuNiwyNDguNSAyNTQuNSwyNTAuMUMyNTMuNCwyNTEuNyAyNTEuOCwyNTMgMjQ5LjgsMjU0QzI0Ny44LDI1NC45IDI0NS40LDI1NS40IDI0Mi42LDI1NS40TDI0Mi40LDI1NS41WiIgc3R5bGU9ImZpbGwtcnVsZTpub256ZXJvOyI+PC9wYXRoPiA8L2c+IDxnIHRyYW5zZm9ybT0ibWF0cml4KDAuNjQ2ODg4LDAsMCwwLjYwOTk2NSwtODQuNjMxOCwtNjMuMDA4MykiPiA8cGF0aCBkPSJNMjY0LjYsMjU0LjdMMjU5LjMsMjI4LjNMMjYzLjYsMjI4LjNMMjY4LDI1MS45TDI2OC44LDI1MS45TDI3NC42LDIyOC4zTDI4MS43LDIyOC4zTDI4Ny41LDI1MS45TDI4OC4zLDI1MS45TDI5Mi43LDIyOC4zTDI5NywyMjguM0wyOTEuNywyNTQuN0wyODQuNCwyNTQuN0wyNzguNiwyMzEuMUwyNzcuOSwyMzEuMUwyNzIuMSwyNTQuN0wyNjQuNiwyNTQuN1oiIHN0eWxlPSJmaWxsLXJ1bGU6bm9uemVybzsiPjwvcGF0aD4gPC9nPiA8ZyB0cmFuc2Zvcm09Im1hdHJpeCgwLjY0Njg4OCwwLDAsMC42MDk5NjUsLTg0LjYzMTgsLTYzLjAwODMpIj4gPHBhdGggZD0iTTMxMi42LDI1NS41QzMwOS45LDI1NS41IDMwNy42LDI1NC45IDMwNS42LDI1My44QzMwMy42LDI1Mi43IDMwMiwyNTEuMSAzMDAuOSwyNDlDMjk5LjgsMjQ2LjkgMjk5LjIsMjQ0LjYgMjk5LjIsMjQxLjlMMjk5LjIsMjQxLjNDMjk5LjIsMjM4LjYgMjk5LjgsMjM2LjIgMzAwLjksMjM0LjFDMzAyLDIzMiAzMDMuNiwyMzAuNSAzMDUuNSwyMjkuM0MzMDcuNSwyMjguMiAzMDkuNywyMjcuNiAzMTIuMywyMjcuNkMzMTQuOSwyMjcuNiAzMTcsMjI4LjEgMzE4LjksMjI5LjJDMzIwLjgsMjMwLjMgMzIyLjMsMjMxLjggMzIzLjQsMjMzLjhDMzI0LjUsMjM1LjggMzI1LDIzOC4xIDMyNSwyNDAuOEwzMjUsMjQyLjdMMzAzLjQsMjQyLjdDMzAzLjUsMjQ1LjYgMzA0LjQsMjQ3LjggMzA2LjIsMjQ5LjRDMzA4LDI1MSAzMTAuMSwyNTEuOCAzMTIuNywyNTEuOEMzMTUuMywyNTEuOCAzMTYuNywyNTEuMyAzMTcuOSwyNTAuMkMzMTkuMSwyNDkuMSAzMjAuMSwyNDcuOSAzMjAuNywyNDYuNUwzMjQuNCwyNDguM0MzMjMuOSwyNDkuNCAzMjMuMSwyNTAuNSAzMjIuMiwyNTEuN0MzMjEuMiwyNTIuOCAzMjAsMjUzLjggMzE4LjUsMjU0LjVDMzE3LDI1NS4yIDMxNSwyNTUuNiAzMTIuNiwyNTUuNkwzMTIuNiwyNTUuNVpNMzAzLjQsMjM5LjFMMzIwLjcsMjM5LjFDMzIwLjYsMjM2LjcgMzE5LjcsMjM0LjcgMzE4LjIsMjMzLjRDMzE2LjcsMjMyLjEgMzE0LjcsMjMxLjMgMzEyLjMsMjMxLjNDMzA5LjksMjMxLjMgMzA3LjksMjMyIDMwNi4zLDIzMy40QzMwNC44LDIzNC44IDMwMy44LDIzNi43IDMwMy40LDIzOS4xWiIgc3R5bGU9ImZpbGwtcnVsZTpub256ZXJvOyI+PC9wYXRoPiA8L2c+IDxnIHRyYW5zZm9ybT0ibWF0cml4KDAuNjQ2ODg4LDAsMCwwLjYwOTk2NSwtODQuNjMxOCwtNjMuMDA4MykiPiA8cGF0aCBkPSJNMzQxLjUsMjU1LjVDMzM5LjIsMjU1LjUgMzM3LjEsMjU1IDMzNS4yLDI1My45QzMzMy4zLDI1Mi44IDMzMS44LDI1MS4zIDMzMC43LDI0OS4yQzMyOS42LDI0Ny4xIDMyOSwyNDQuNyAzMjksMjQxLjlMMzI5LDI0MS4yQzMyOSwyMzguNCAzMjkuNiwyMzYgMzMwLjcsMjMzLjlDMzMxLjgsMjMxLjkgMzMzLjMsMjMwLjMgMzM1LjIsMjI5LjJDMzM3LjEsMjI4LjEgMzM5LjIsMjI3LjYgMzQxLjUsMjI3LjZDMzQzLjgsMjI3LjYgMzQ0LjksMjI3LjggMzQ2LjIsMjI4LjNDMzQ3LjUsMjI4LjggMzQ4LjUsMjI5LjQgMzQ5LjMsMjMwLjFDMzUwLjEsMjMwLjggMzUwLjcsMjMxLjYgMzUxLjEsMjMyLjVMMzUxLjksMjMyLjVMMzUxLjksMjE2LjlMMzU2LjIsMjE2LjlMMzU2LjIsMjU0LjdMMzUyLDI1NC43TDM1MiwyNTAuM0wzNTEuMiwyNTAuM0MzNTAuNSwyNTEuNiAzNDkuNCwyNTIuOCAzNDcuOCwyNTMuOUMzNDYuMywyNTQuOSAzNDQuMiwyNTUuNSAzNDEuNCwyNTUuNUwzNDEuNSwyNTUuNVpNMzQyLjYsMjUxLjdDMzQ1LjMsMjUxLjcgMzQ3LjYsMjUwLjggMzQ5LjMsMjQ5LjFDMzUxLDI0Ny4zIDM1MS45LDI0NC45IDM1MS45LDI0MS44TDM1MS45LDI0MS4zQzM1MS45LDIzOC4yIDM1MSwyMzUuNyAzNDkuMywyMzRDMzQ3LjYsMjMyLjIgMzQ1LjQsMjMxLjQgMzQyLjYsMjMxLjRDMzM5LjgsMjMxLjQgMzM3LjcsMjMyLjMgMzM1LjksMjM0QzMzNC4xLDIzNS43IDMzMy4zLDIzOC4yIDMzMy4zLDI0MS4zTDMzMy4zLDI0MS44QzMzMy4zLDI0NC45IDMzNC4yLDI0Ny40IDMzNS45LDI0OS4xQzMzNy43LDI1MC45IDMzOS45LDI1MS43IDM0Mi42LDI1MS43WiIgc3R5bGU9ImZpbGwtcnVsZTpub256ZXJvOyI+PC9wYXRoPiA8L2c+IDxnIHRyYW5zZm9ybT0ibWF0cml4KDAuNjQ2ODg4LDAsMCwwLjYwOTk2NSwtODQuNjMxOCwtNjMuMDA4MykiPiA8cGF0aCBkPSJNMzc1LDI1NS41QzM3Mi4zLDI1NS41IDM3MCwyNTQuOSAzNjgsMjUzLjhDMzY2LDI1Mi43IDM2NC40LDI1MS4xIDM2My4zLDI0OUMzNjIuMiwyNDYuOSAzNjEuNiwyNDQuNiAzNjEuNiwyNDEuOUwzNjEuNiwyNDEuM0MzNjEuNiwyMzguNiAzNjIuMiwyMzYuMiAzNjMuMywyMzQuMUMzNjQuNCwyMzIgMzY2LDIzMC41IDM2Ny45LDIyOS4zQzM2OS45LDIyOC4yIDM3Mi4xLDIyNy42IDM3NC43LDIyNy42QzM3Ny4zLDIyNy42IDM3OS40LDIyOC4xIDM4MS4zLDIyOS4yQzM4My4yLDIzMC4zIDM4NC43LDIzMS44IDM4NS44LDIzMy44QzM4Ni45LDIzNS44IDM4Ny40LDIzOC4xIDM4Ny40LDI0MC44TDM4Ny40LDI0Mi43TDM2NS44LDI0Mi43QzM2NS45LDI0NS42IDM2Ni44LDI0Ny44IDM2OC42LDI0OS40QzM3MC40LDI1MSAzNzIuNSwyNTEuOCAzNzUuMSwyNTEuOEMzNzcuNywyNTEuOCAzNzkuMSwyNTEuMyAzODAuMywyNTAuMkMzODEuNSwyNDkuMSAzODIuNSwyNDcuOSAzODMuMSwyNDYuNUwzODYuOCwyNDguM0MzODYuMywyNDkuNCAzODUuNSwyNTAuNSAzODQuNiwyNTEuN0MzODMuNiwyNTIuOCAzODIuNCwyNTMuOCAzODAuOSwyNTQuNUMzNzkuNCwyNTUuMiAzNzcuNCwyNTUuNiAzNzUsMjU1LjZMMzc1LDI1NS41Wk0zNjUuOSwyMzkuMUwzODMuMiwyMzkuMUMzODMuMSwyMzYuNyAzODIuMiwyMzQuNyAzODAuNywyMzMuNEMzNzkuMiwyMzIuMSAzNzcuMiwyMzEuMyAzNzQuOCwyMzEuM0MzNzIuNCwyMzEuMyAzNzAuNCwyMzIgMzY4LjgsMjMzLjRDMzY3LjMsMjM0LjggMzY2LjMsMjM2LjcgMzY1LjksMjM5LjFaIiBzdHlsZT0iZmlsbC1ydWxlOm5vbnplcm87Ij48L3BhdGg+IDwvZz4gPGcgdHJhbnNmb3JtPSJtYXRyaXgoMC42NDY4ODgsMCwwLDAuNjA5OTY1LC04NC42MzE4LC02My4wMDgzKSI+IDxwYXRoIGQ9Ik0zOTEuNywyNTQuN0wzOTEuNywyMjguM0wzOTUuOSwyMjguM0wzOTUuOSwyMzIuOEwzOTYuNywyMzIuOEMzOTcuMywyMzEuNSAzOTguMywyMzAuNCAzOTkuNywyMjkuNEM0MDEuMSwyMjguNCA0MDMuMSwyMjcuOSA0MDUuOCwyMjcuOUM0MDguNSwyMjcuOSA0MDkuNiwyMjguMyA0MTEuMSwyMjkuMUM0MTIuNywyMjkuOSA0MTMuOSwyMzEuMiA0MTQuOSwyMzIuOEM0MTUuOCwyMzQuNCA0MTYuMywyMzYuNCA0MTYuMywyMzguOEw0MTYuMywyNTQuOEw0MTIsMjU0LjhMNDEyLDIzOS4xQzQxMiwyMzYuNSA0MTEuMywyMzQuNiA0MTAsMjMzLjRDNDA4LjcsMjMyLjIgNDA2LjksMjMxLjYgNDA0LjcsMjMxLjZDNDAyLjUsMjMxLjYgNDAwLjEsMjMyLjQgMzk4LjUsMjM0LjFDMzk2LjksMjM1LjggMzk2LjEsMjM4LjIgMzk2LjEsMjQxLjVMMzk2LjEsMjU0LjhMMzkxLjgsMjU0LjhMMzkxLjcsMjU0LjdaIiBzdHlsZT0iZmlsbC1ydWxlOm5vbnplcm87Ij48L3BhdGg+IDwvZz4gPC9nPiA8L2c+IDwvc3ZnPiA=" class="des-logo" alt="Digital Earth Sweden">
            <div class="header-divider"></div>
            <div>
                <h1><span>IMINT</span> Analysrapport</h1>
                <p>Showcase — Brand ({fire_date}) &middot; Marin ({marine_date})</p>
            </div>
        </div>
        <div class="header-nav">
            <a href="#" class="theme-tab active" data-tab="fire">\U0001f525 Brand</a>
            <a href="#" class="theme-tab" data-tab="marine">\u2693 Marin</a>
        </div>
    </div>

    <!-- ── Fire tab ──────────────────────────────────────────────── -->
    <div class="tab-content active" id="tab-fire">
        {fire_summary_html}
        <div class="tab-intro">
            <p>
                Multisensoranalys av skogsbrand med Sentinel-2 multispektraldata.
                Analysen kombinerar f\u00f6r\u00e4ndringsdetektering, brandsvårighetsindex (dNBR),
                AI-segmentering och markt\u00e4ckedata f\u00f6r att kartl\u00e4gga brandens utbredning och intensitet.
            </p>
        </div>
        <div class="section-header">
            <h2>Brandanalys — {fire_date}</h2>
        </div>
        {fire_toolbar}
        <div class="map-grid">
            {fire_cells}
        </div>
        {fire_charts_html}
        <div class="tab-description">
            <p>
                Analysomr\u00e5det \u00e4r beläget i Ljusdals kommun, Gävleborgs län, och
                visar K\u00e5rbölebranden — en av de st\u00f6rsta skogsbränderna i Sveriges moderna historia
                sommaren 2018. Sentinel-2-data fr\u00e5n {fire_date} har analyserats med flera
                kompletterande metoder: multispektral f\u00f6r\u00e4ndringsdetektering mot en f\u00f6rbrandsbaseline
                identifierar förändrade pixlar, dNBR (differentierat Normalized Burn Ratio) kvantifierar
                brandens svårighetsgrad fr\u00e5n l\u00e5g till h\u00f6g intensitet, och Prithvi — en
                NASA/IBM foundation model tr\u00e4nad p\u00e5 HLS-data — segmenterar brandomr\u00e5det
                med djupinl\u00e4rning.
            </p>
            <p>
                Resultaten korsrefereras mot Naturv\u00e5rdsverkets Nationella Markt\u00e4ckedata (NMD)
                f\u00f6r att visa hur branden p\u00e5verkat olika markklasser — fr\u00e5n tallskog
                och v\u00e5tmark till \u00f6ppen mark. Molnanalys via COT-modellen s\u00e4kerst\u00e4ller
                datakvaliteten f\u00f6r det valda datumet.
            </p>
        </div>
    </div>

    <!-- ── Marine tab ────────────────────────────────────────────── -->
    <div class="tab-content" id="tab-marine">
        {marine_summary_html}
        <div class="tab-intro">
            <p>
                Automatisk fartygsdetektering och havsövervakning l\u00e4ngs Bohusläns kust
                med Sentinel-2-satellitdata. Analysen kombinerar YOLO-objektdetektering
                och multitemporala heatmaps f\u00f6r att kartl\u00e4gga maritim aktivitet.
            </p>
        </div>
        <div class="section-header">
            <h2>Marin analys — {marine_date}</h2>
        </div>
        {marine_toolbar}
        <div class="map-grid">
            {marine_cells}
        </div>
        <div class="tab-description">
            <p>
                Analysomr\u00e5det t\u00e4cker Bohusläns sk\u00e4rg\u00e5rd — ett av Sveriges mest trafikerade
                kustomr\u00e5den med b\u00e5de kommersiell sjöfart, fiske och fritidsb\u00e5tar.
                YOLO11s-modellen detekterar fartyg i optiska Sentinel-2-bilder
                genom att skanna bilden i överlappande rutor och filtrera mot NMD-landmask.
            </p>
            <p>
                Multitemporala heatmaps aggregerar detektioner \u00f6ver flera
                satellit\u00f6verfarter f\u00f6r att visa återkommande trafikm\u00f6nster och
                ankringsplatser. Bilder med h\u00f6g molnt\u00e4ckning filtreras automatiskt
                via COT-analys. NMD-data anv\u00e4nds f\u00f6r landmaskering s\u00e5 att
                detektioner begr\u00e4nsas till vattenomr\u00e5den. Sjökort fr\u00e5n Sjöfartsverket
                (via SLU GET) kan visas som bakgrundslager f\u00f6r geografisk kontext.
            </p>
        </div>
    </div>

    <div class="footer">
        IMINT Engine &middot; Genererad {fire_date} / {marine_date}
        <br>
        <button class="license-toggle" onclick="document.getElementById('license-info').classList.toggle('open'); this.textContent = this.textContent === 'Visa licenser och upphovsr\u00e4tt' ? 'D\u00f6lj licenser' : 'Visa licenser och upphovsr\u00e4tt';">Visa licenser och upphovsr\u00e4tt</button>
        <div id="license-info" class="license-section">
            <h4>Modeller</h4>
            <table class="license-table">
                <tr><th>Komponent</th><th>Licens</th><th>Upphovsr\u00e4tt</th><th>Anm\u00e4rkning</th></tr>
                <tr>
                    <td>YOLO11s (fartygsdetektering)</td>
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
                    <td>Sj\u00f6kort (S-57) via SLU GET</td>
                    <td><span class="license-badge badge-restricted">Akademisk</span></td>
                    <td>&copy; Sj\u00f6fartsverket</td>
                    <td>Tillg\u00e4nglig via <a href="https://maps.slu.se/get/" style="color:#cff8e4;" target="_blank">SLU GET</a>
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
            <p style="margin-top:14px; color:rgba(207,248,228,0.4);">
                <strong style="color:rgba(207,248,228,0.6);">OBS:</strong>
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
        function initMaps(viewers, images, imgH, imgW, hasBgToggle) {{
            const bounds = [[0, 0], [imgH, imgW]];
            const maps = [];
            const overlays = {{}};
            const bgLayers = {{}};

            viewers.forEach(function(v) {{
                const container = document.getElementById(v.id);
                if (!container || !images[v.id]) return;

                const map = L.map(v.id, {{
                    crs: L.CRS.Simple,
                    minZoom: -2,
                    maxZoom: 5,
                    attributionControl: false,
                    zoomSnap: 0.25,
                }});

                // Background / overlay layer logic
                if (hasBgToggle) {{
                    bgLayers[v.id] = {{}};
                    const prefix = v.id.split('-')[0];
                    const rgbId = prefix + '-rgb';
                    const sjokortId = prefix + '-sjokort';

                    // RGB background layer
                    const rgbUrl = (v.key === 'rgb') ? images[v.id] : images[rgbId];
                    if (rgbUrl) {{
                        bgLayers[v.id].rgb = L.imageOverlay(
                            rgbUrl, bounds, {{zIndex: 0, opacity: 1}}
                        ).addTo(map);
                    }}
                    // Sjökort background layer
                    if (images[sjokortId]) {{
                        bgLayers[v.id].sjokort = L.imageOverlay(
                            images[sjokortId], bounds, {{zIndex: 0, opacity: 0}}
                        ).addTo(map);
                    }}

                    if (v.key === 'rgb') {{
                        // RGB panel: background IS content, opacity slider controls active bg
                        overlays[v.id] = bgLayers[v.id].rgb;
                    }} else {{
                        // Analysis panels: overlay on top, opacity slider controls it
                        const overlay = L.imageOverlay(images[v.id], bounds, {{zIndex: 1}}).addTo(map);
                        overlays[v.id] = overlay;
                    }}
                }} else {{
                    // No toggle available: static RGB background + overlay
                    if (v.key !== 'rgb') {{
                        const rgbId = v.id.split('-')[0] + '-rgb';
                        if (images[rgbId]) {{
                            L.imageOverlay(images[rgbId], bounds, {{zIndex: 0}}).addTo(map);
                        }}
                    }}
                    const overlay = L.imageOverlay(images[v.id], bounds, {{zIndex: 1}}).addTo(map);
                    overlays[v.id] = overlay;
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
                    if (overlays[mapId]) overlays[mapId].setOpacity(val / 100);
                }});
            }});

            // Background toggle (RGB ↔ Sjökort)
            document.querySelectorAll('.bg-toggle').forEach(function(toggle) {{
                toggle.querySelectorAll('.bg-btn').forEach(function(btn) {{
                    btn.addEventListener('click', function() {{
                        const mapId = toggle.dataset.mapId;
                        const bg = this.dataset.bg;
                        const layers = bgLayers[mapId];
                        if (!layers) return;
                        toggle.querySelectorAll('.bg-btn').forEach(b => b.classList.remove('active'));
                        this.classList.add('active');
                        if (bg === 'rgb') {{
                            if (layers.rgb) layers.rgb.setOpacity(1);
                            if (layers.sjokort) layers.sjokort.setOpacity(0);
                        }} else {{
                            if (layers.rgb) layers.rgb.setOpacity(0);
                            if (layers.sjokort) layers.sjokort.setOpacity(1);
                        }}
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
        initMaps(MARINE_VIEWERS, MARINE_IMAGES, {marine_h}, {marine_w}, {str(has_marine_bg).lower()});

        // ── Chart.js (Fire tab) ──────────────────────────────────────────
        const CHART_DATA = {fire_chart_json};

        Chart.defaults.color = 'rgba(207,248,228,0.6)';
        Chart.defaults.borderColor = 'rgba(207,248,228,0.06)';
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
                togglePanel(pid, !isActive);
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
