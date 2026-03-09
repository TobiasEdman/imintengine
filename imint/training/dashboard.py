"""
imint/training/dashboard.py — Training progress dashboard

Generates a self-contained HTML dashboard for monitoring the full
LULC pipeline (data preparation + training) in real-time.  Uses
Chart.js for visualisation.  A lightweight HTTP server (stdlib
``http.server``) serves the data directory so the browser can poll
JSON metrics via ``fetch()``.
"""
from __future__ import annotations

import threading
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Re-use the library cache from html_report
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LIB_CACHE_DIR = _PROJECT_ROOT / ".lib_cache"
_CHART_JS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"


def _fetch_chart_js() -> str:
    """Download and cache Chart.js, returning the JS source."""
    from urllib.request import urlopen
    from urllib.error import URLError

    _LIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _LIB_CACHE_DIR / "chart.umd.js"

    if cached.exists():
        return cached.read_text(encoding="utf-8")

    try:
        print(f"    [dashboard] Downloading chart.js ...")
        with urlopen(_CHART_JS_CDN, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        cached.write_text(content, encoding="utf-8")
        return content
    except (URLError, OSError) as e:
        print(f"    [dashboard] WARNING: Could not download chart.js: {e}")
        return "/* chart.js unavailable */"


# ── NMD L2 colours (hex) matching html_report.py ──────────────────────────

_CLASS_COLORS = {
    "BG":              "#333333",
    "Pine":            "#006400",
    "Spruce":          "#228B22",
    "Deciduous":       "#32CD32",
    "Mixed":           "#3CB371",
    "Temp non-forest": "#90EE90",
    "Wetl. pine":      "#2E4F2E",
    "Wetl. spruce":    "#3A5F3A",
    "Wetl. decid.":    "#4A7F4A",
    "Wetl. mixed":     "#5A8F5A",
    "Wetl. temp":      "#7AAF7A",
    "Open wetland":    "#8B5A2B",
    "Cropland":        "#FFD700",
    "Bare land":       "#C8AD7F",
    "Vegetated":       "#D2B48C",
    "Buildings":       "#FF0000",
    "Infra":           "#FF4500",
    "Roads":           "#FF6347",
    "Lakes":           "#0000FF",
    "Sea":             "#1E90FF",
}


# ── Section builders ─────────────────────────────────────────────────────


def _css_styles() -> str:
    """Return the full <style>...</style> block."""
    return f"""<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
  background: #0b0e17;
  color: #d8dae5;
  min-height: 100vh;
}}
.header {{
  background: linear-gradient(135deg, #111827, #1e293b);
  padding: 20px 32px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #1e293b;
}}
.header h1 {{
  font-size: 18px;
  font-weight: 600;
  color: #f1f5f9;
}}
.header-right {{
  display: flex;
  align-items: center;
  gap: 12px;
}}
.status-badge {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  padding: 6px 14px;
  border-radius: 20px;
  background: #111827;
  border: 1px solid #374151;
}}
.status-dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
}}
.status-preparing .status-dot {{
  background: #60a5fa;
  animation: pulse 2s ease-in-out infinite;
}}
.status-training .status-dot {{
  background: #22c55e;
  animation: pulse 2s ease-in-out infinite;
}}
.status-completed .status-dot {{ background: #22c55e; }}
.status-stopped .status-dot {{ background: #eab308; }}
.status-waiting .status-dot {{ background: #6b7280; animation: pulse 2s ease-in-out infinite; }}
@keyframes pulse {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0.3; }}
}}
.container {{ padding: 24px 32px; max-width: 1500px; margin: 0 auto; display: flex; gap: 24px; }}
.main-content {{ flex: 1; min-width: 0; }}
.sidebar {{ width: 200px; flex-shrink: 0; position: sticky; top: 24px; align-self: flex-start; }}
.sidebar .section-title {{ font-size: 11px; margin-bottom: 10px; }}
.sidebar .gauge-card {{ margin-bottom: 10px; padding: 8px 10px; min-width: unset; }}
.sidebar .gauge-svg {{ width: 80px; height: 52px; }}
.sidebar .gauge-value {{ font-size: 16px; margin-top: -6px; }}
.sidebar .gauge-label {{ font-size: 9px; }}
.sidebar .net-card {{ padding: 10px 8px; }}
.preview-thumb {{ width: 100%; border-radius: 6px; border: 1px solid #1e293b; margin-bottom: 4px; }}
.preview-label {{ font-size: 9px; color: #6b7280; margin-bottom: 10px; line-height: 1.3; }}
.section {{
  margin-bottom: 28px;
}}
.section-header {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
}}
.section-title {{
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: #6b7280;
}}
.section-badge {{
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 10px;
  font-weight: 600;
  text-transform: uppercase;
}}
.section-badge.active {{
  background: #22c55e20;
  color: #22c55e;
  border: 1px solid #22c55e40;
}}
.section-badge.done {{
  background: #6b728020;
  color: #6b7280;
  border: 1px solid #6b728040;
}}
.section-badge.pending {{
  background: #6b728010;
  color: #4b5563;
  border: 1px solid #4b556340;
}}
.cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}}
.card {{
  background: #111827;
  border-radius: 10px;
  padding: 16px 18px;
  border: 1px solid #1e293b;
}}
.card-label {{
  font-size: 11px;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 4px;
}}
.card-value {{
  font-size: 22px;
  font-weight: 700;
  color: #f1f5f9;
}}
.card-value.sm {{
  font-size: 16px;
}}
.card-sub {{
  font-size: 11px;
  color: #6b7280;
  margin-top: 2px;
}}
.chart-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 18px;
}}
.chart-box {{
  background: #111827;
  border-radius: 10px;
  padding: 14px;
  border: 1px solid #1e293b;
  min-height: 200px;
  max-height: 280px;
}}
.chart-box h3 {{
  font-size: 12px;
  color: #94a3b8;
  margin-bottom: 8px;
  font-weight: 500;
}}
.chart-box canvas {{ width: 100% !important; max-height: 230px; }}
.chart-box.pie-chart {{ min-height: auto; max-height: 280px; }}
.chart-box.pie-chart canvas {{ max-height: 230px; }}
.progress-bar-outer {{
  background: #1e293b;
  border-radius: 6px;
  height: 10px;
  overflow: hidden;
  margin-top: 8px;
}}
.progress-bar-inner {{
  height: 100%;
  border-radius: 6px;
  background: linear-gradient(90deg, #3b82f6, #60a5fa);
  transition: width 0.5s ease;
}}
.patience-row {{
  display: flex;
  gap: 5px;
  margin-top: 6px;
}}
.patience-dot {{
  width: 12px;
  height: 12px;
  border-radius: 50%;
  border: 2px solid #374151;
  transition: background 0.3s;
}}
.patience-dot.used {{
  background: #eab308;
  border-color: #eab308;
}}
.full-width {{ grid-column: 1 / -1; }}
.gauge-card {{
  background: #111827;
  border-radius: 10px;
  padding: 10px 16px;
  border: 1px solid #1e293b;
  min-width: 120px;
  text-align: center;
}}
.gauge-svg {{ width: 100px; height: 65px; display: block; margin: 0 auto; }}
.gauge-value {{
  font-size: 20px;
  font-weight: 700;
  color: #f1f5f9;
  margin-top: -8px;
}}
.gauge-label {{
  font-size: 10px;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-top: 2px;
}}
/* ── LULC inference tile gallery ─────────────────────────── */
.lulc-gallery-header {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr;
  gap: 4px;
  padding: 6px 0;
  margin-bottom: 4px;
}}
.lulc-gallery-header span {{
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: #6b7280;
  text-align: center;
}}
.lulc-gallery-row {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr;
  gap: 0;
  background: #111827;
  border: 1px solid #1e293b;
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 4px;
  transition: border-color 0.15s;
}}
.lulc-gallery-row:hover {{
  border-color: #60a5fa;
}}
.lulc-gallery-cell {{
  position: relative;
  aspect-ratio: 1;
  overflow: hidden;
}}
.lulc-gallery-cell img {{
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  image-rendering: pixelated;
}}
.lulc-gallery-cell .cell-label {{
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  font-size: 9px;
  font-weight: 600;
  color: #e2e8f0;
  background: rgba(17,24,39,0.85);
  padding: 2px 6px;
  text-align: center;
}}
.lulc-legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 4px 12px;
  margin: 8px 0;
}}
.lulc-legend-item {{
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 10px;
  color: #94a3b8;
}}
.lulc-legend-swatch {{
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex-shrink: 0;
}}
@media (max-width: 900px) {{
  .container {{ flex-direction: column; }}
  .sidebar {{ width: 100%; position: static; display: flex; flex-wrap: wrap; gap: 10px; }}
  .sidebar .gauge-card {{ margin-bottom: 0; }}
  .chart-grid {{ grid-template-columns: 1fr; }}
  .lulc-gallery-header, .lulc-gallery-row {{ grid-template-columns: 1fr 1fr; }}
  .cards {{ grid-template-columns: repeat(2, 1fr); }}
}}
.sf-source-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-top: 8px;
}}
.sf-source-box {{
  background: #0d1117;
  border-radius: 8px;
  padding: 12px;
  border: 1px solid #1e293b;
}}
.sf-source-title {{
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 1px;
}}
.sf-source-stat {{
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  padding: 3px 0;
  border-bottom: 1px solid #1e293b20;
}}
.sf-stat-label {{
  color: #6b7280;
}}
.sf-source-stat span:last-child {{
  color: #e2e8f0;
  font-weight: 600;
}}
</style>"""


def _html_header() -> str:
    """Return the page header with status badge."""
    return f"""<div class="header">
  <h1>IMINT Training Dashboard</h1>
  <div class="header-right">
    <div class="status-badge status-waiting" id="status-badge">
      <span class="status-dot"></span>
      <span id="status-text">Waiting...</span>
    </div>
  </div>
</div>"""


def _html_nmd_section() -> str:
    """Return the NMD pre-filter section."""
    return f"""  <!-- NMD Pre-filter Section -->
  <div class="section" id="section-nmd">
    <div class="section-header">
      <div class="section-title">NMD Pre-filter</div>
      <span class="section-badge pending" id="nmd-badge">pending</span>
    </div>
    <div class="cards">
      <div class="card">
        <div class="card-label">Progress</div>
        <div class="card-value" id="nmd-progress-text">-</div>
        <div class="progress-bar-outer">
          <div class="progress-bar-inner" id="nmd-progress" style="width: 0%"></div>
        </div>
      </div>
      <div class="card">
        <div class="card-label">Land Cells</div>
        <div class="card-value" id="nmd-land" style="color: #22c55e">-</div>
        <div class="card-sub">passed filter</div>
      </div>
      <div class="card">
        <div class="card-label">Water / Empty</div>
        <div class="card-value" id="nmd-water" style="color: #3b82f6">-</div>
        <div class="card-sub">skipped</div>
      </div>
      <div class="card">
        <div class="card-label">Failed</div>
        <div class="card-value" id="nmd-failed" style="color: #ef4444">-</div>
        <div class="card-sub">NMD fetch error</div>
      </div>
    </div>
  </div>"""


def _html_seasonal_fetch_section() -> str:
    """Return the seasonal fetch (ColonyOS) section."""
    return f"""  <!-- Seasonal Fetch (ColonyOS) Section -->
  <div class="section" id="section-seasonal">
    <div class="section-header">
      <div class="section-title">Seasonal Fetch (ColonyOS)</div>
      <span class="section-badge pending" id="sf-badge">pending</span>
    </div>
    <div class="cards">
      <div class="card">
        <div class="card-label">Progress</div>
        <div class="card-value" id="sf-progress-text">-</div>
        <div class="progress-bar-outer">
          <div class="progress-bar-inner" id="sf-progress" style="width: 0%"></div>
        </div>
      </div>
      <div class="card">
        <div class="card-label">Completed</div>
        <div class="card-value" id="sf-completed" style="color: #22c55e">-</div>
        <div class="card-sub">tiles saved</div>
      </div>
      <div class="card">
        <div class="card-label">Running</div>
        <div class="card-value" id="sf-running" style="color: #60a5fa">-</div>
        <div class="card-sub">active jobs</div>
      </div>
      <div class="card">
        <div class="card-label">Failed</div>
        <div class="card-value" id="sf-failed" style="color: #ef4444">-</div>
        <div class="card-sub" id="sf-failed-sub"></div>
      </div>
      <div class="card">
        <div class="card-label">Rate</div>
        <div class="card-value" id="sf-rate">-</div>
        <div class="card-sub">tiles / hour</div>
      </div>
      <div class="card">
        <div class="card-label">ETA</div>
        <div class="card-value" id="sf-eta">-</div>
        <div class="card-sub" id="sf-elapsed"></div>
      </div>
    </div>
    <div class="chart-grid">
      <div class="chart-box">
        <h3>CDSE vs DES &mdash; Completed &amp; Failed</h3>
        <canvas id="chart-sf-comparison"></canvas>
      </div>
      <div class="chart-box">
        <h3>Source Performance</h3>
        <div class="sf-source-grid">
          <div class="sf-source-box">
            <div class="sf-source-title" style="color:#3b82f6">CDSE (Copernicus)</div>
            <div class="sf-source-stat"><span class="sf-stat-label">Completed</span> <span id="sf-cdse-done">-</span></div>
            <div class="sf-source-stat"><span class="sf-stat-label">Failed</span> <span id="sf-cdse-fail">-</span></div>
            <div class="sf-source-stat"><span class="sf-stat-label">Avg time</span> <span id="sf-cdse-avg">-</span></div>
            <div class="sf-source-stat"><span class="sf-stat-label">Success rate</span> <span id="sf-cdse-rate">-</span></div>
          </div>
          <div class="sf-source-box">
            <div class="sf-source-title" style="color:#f59e0b">DES</div>
            <div class="sf-source-stat"><span class="sf-stat-label">Completed</span> <span id="sf-des-done">-</span></div>
            <div class="sf-source-stat"><span class="sf-stat-label">Failed</span> <span id="sf-des-fail">-</span></div>
            <div class="sf-source-stat"><span class="sf-stat-label">Avg time</span> <span id="sf-des-avg">-</span></div>
            <div class="sf-source-stat"><span class="sf-stat-label">Success rate</span> <span id="sf-des-rate">-</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>"""


def _html_dataprep_section() -> str:
    """Return the data preparation section."""
    return f"""  <!-- Data Preparation Section -->
  <div class="section" id="section-dataprep">
    <div class="section-header">
      <div class="section-title">Spectral Data Fetch</div>
      <span class="section-badge pending" id="dp-badge">pending</span>
    </div>
    <div class="cards">
      <div class="card">
        <div class="card-label">Progress</div>
        <div class="card-value" id="dp-progress-text">-</div>
        <div class="progress-bar-outer">
          <div class="progress-bar-inner" id="dp-progress" style="width: 0%"></div>
        </div>
      </div>
      <div class="card">
        <div class="card-label">Tiles Saved</div>
        <div class="card-value" id="dp-tiles">-</div>
        <div class="card-sub" id="dp-failed-sub"></div>
      </div>
      <div class="card">
        <div class="card-label">Current Split</div>
        <div class="card-value sm" id="dp-split">-</div>
      </div>
      <div class="card">
        <div class="card-label">Latest Tile</div>
        <div class="card-value sm" id="dp-latest">-</div>
        <div class="card-sub" id="dp-latest-sub"></div>
      </div>
      <div class="card">
        <div class="card-label">ETA</div>
        <div class="card-value" id="dp-eta">-</div>
        <div class="card-sub" id="dp-elapsed"></div>
      </div>
      <div class="card">
        <div class="card-label">Rare Classes</div>
        <div class="card-value" id="dp-rare">-</div>
      </div>
    </div>
    <div class="chart-grid">
      <div class="chart-box full-width pie-chart">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <h3 style="margin:0">Class Distribution</h3>
          <button id="toggle-classdist" onclick="toggleClassView()"
            style="background:#334155; color:#e2e8f0; border:1px solid #475569;
                   padding:4px 12px; border-radius:6px; cursor:pointer; font-size:11px;">
            Show details
          </button>
        </div>
        <canvas id="chart-classdist"></canvas>
        <canvas id="chart-classdist-detail" style="display:none"></canvas>
      </div>
    </div>
  </div>"""


def _html_eval_section() -> str:
    """Return the evaluation section."""
    return f"""  <!-- Evaluation Section -->
  <div class="section" id="section-eval">
    <div class="section-header">
      <div class="section-title">Evaluation</div>
      <span class="section-badge pending" id="eval-badge">pending</span>
    </div>
    <div class="cards">
      <div class="card">
        <div class="card-label">mIoU</div>
        <div class="card-value" id="eval-miou">-</div>
        <div class="card-sub" id="eval-checkpoint"></div>
      </div>
      <div class="card">
        <div class="card-label">Overall Accuracy</div>
        <div class="card-value" id="eval-oa">-</div>
      </div>
      <div class="card">
        <div class="card-label">Test Tiles</div>
        <div class="card-value" id="eval-tiles">-</div>
        <div class="card-sub" id="eval-time"></div>
      </div>
      <div class="card">
        <div class="card-label">Aux Channels</div>
        <div class="card-value sm" id="eval-aux">-</div>
      </div>
    </div>
    <div class="chart-grid">
      <div class="chart-box full-width">
        <h3>Per-Class IoU (test)</h3>
        <canvas id="chart-eval-perclass"></canvas>
      </div>
    </div>
  </div>"""


def _html_training_section() -> str:
    """Return the training section."""
    return f"""  <!-- Training Section -->
  <div class="section" id="section-training">
    <div class="section-header">
      <div class="section-title">Training</div>
      <span class="section-badge pending" id="tr-badge">pending</span>
    </div>
    <div class="cards">
      <div class="card">
        <div class="card-label">Epoch</div>
        <div class="card-value" id="tr-epoch">-</div>
      </div>
      <div class="card">
        <div class="card-label">Best mIoU</div>
        <div class="card-value" id="tr-best-miou">-</div>
        <div class="card-sub" id="tr-best-epoch"></div>
      </div>
      <div class="card">
        <div class="card-label">Learning Rate</div>
        <div class="card-value" id="tr-lr">-</div>
      </div>
      <div class="card">
        <div class="card-label">ETA</div>
        <div class="card-value" id="tr-eta">-</div>
      </div>
      <div class="card">
        <div class="card-label">Loss</div>
        <div class="card-value" id="tr-loss">-</div>
        <div class="card-sub" id="tr-loss-type"></div>
      </div>
      <div class="card">
        <div class="card-label">Patience</div>
        <div class="card-value" id="tr-patience-text">-</div>
        <div class="patience-row" id="tr-patience-dots"></div>
      </div>
    </div>
    <div class="chart-grid">
      <div class="chart-box">
        <h3>Training Loss</h3>
        <canvas id="chart-loss"></canvas>
      </div>
      <div class="chart-box">
        <h3>Validation mIoU</h3>
        <canvas id="chart-miou"></canvas>
      </div>
      <div class="chart-box">
        <h3>Per-Class IoU (latest epoch)</h3>
        <canvas id="chart-perclass"></canvas>
      </div>
      <div class="chart-box">
        <h3>Worst-Class IoU Trend</h3>
        <canvas id="chart-worst"></canvas>
      </div>
    </div>
  </div>"""


def _html_lulc_section() -> str:
    """Return the LULC inference section."""
    return f"""  <!-- LULC Inference Section -->
  <div class="section" id="section-lulc">
    <div class="section-header">
      <div class="section-title">LULC Inference</div>
      <span class="section-badge pending" id="lulc-badge">pending</span>
    </div>
    <div class="cards" id="lulc-cards">
      <div class="card">
        <div class="card-label">Overall Accuracy</div>
        <div class="card-value" id="lulc-accuracy">-</div>
        <div class="card-sub" id="lulc-pixels-sub"></div>
      </div>
      <div class="card">
        <div class="card-label">High-Conf Wrong</div>
        <div class="card-value" id="lulc-hcw" style="color: #f472b6">-</div>
        <div class="card-sub">model &gt;80% sure, NMD disagrees</div>
      </div>
      <div class="card">
        <div class="card-label">Tiles</div>
        <div class="card-value" id="lulc-tiles">-</div>
        <div class="card-sub">val split</div>
      </div>
      <div class="card">
        <div class="card-label">Disagree</div>
        <div class="card-value" id="lulc-disagree" style="color: #ef4444">-</div>
        <div class="card-sub" id="lulc-disagree-sub"></div>
      </div>
    </div>
    <div class="lulc-legend" id="lulc-class-legend"></div>
    <div class="lulc-legend">
      <span class="lulc-legend-item"><span class="lulc-legend-swatch" style="background:#2ecc40"></span>Korrekt</span>
      <span class="lulc-legend-item"><span class="lulc-legend-swatch" style="background:#ff4136"></span>Fel</span>
      <span class="lulc-legend-item"><span class="lulc-legend-swatch" style="background:#ff00ff"></span>Hög konf. fel</span>
    </div>
    <div class="lulc-gallery-header">
      <span>S2 pseudofärg (B8/B3/B4)</span>
      <span>NMD grundsanning</span>
      <span>Modellprediktion</span>
      <span>Kvalitet</span>
    </div>
    <div id="lulc-gallery">
      <div style="font-size:11px; color:#4b5563; padding:12px 0;">
        Kör <code style="background:#1e293b; padding:2px 6px; border-radius:4px;">make predict-aux &amp;&amp; make lulc-showcase</code> för att generera
      </div>
    </div>
    <div class="chart-grid">
      <div class="chart-box full-width">
        <h3>Per-Class Accuracy (val)</h3>
        <canvas id="chart-lulc-perclass"></canvas>
      </div>
    </div>
  </div>"""


def _html_sidebar() -> str:
    """Return the system metrics sidebar."""
    return f"""  <!-- Sidebar: System Metrics -->
  <div class="sidebar" id="section-system">
    <div class="section-title">System</div>

    <div class="gauge-card">
      <svg viewBox="0 0 120 80" class="gauge-svg">
        <path d="M 10 70 A 50 50 0 0 1 110 70" fill="none" stroke="#1e293b" stroke-width="10" stroke-linecap="round"/>
        <path id="gauge-cpu-arc" d="M 10 70 A 50 50 0 0 1 110 70" fill="none" stroke="#3b82f6" stroke-width="10" stroke-linecap="round" stroke-dasharray="0 157"/>
      </svg>
      <div class="gauge-value" id="gauge-cpu-val">-</div>
      <div class="gauge-label">CPU</div>
    </div>

    <div class="gauge-card">
      <svg viewBox="0 0 120 80" class="gauge-svg">
        <path d="M 10 70 A 50 50 0 0 1 110 70" fill="none" stroke="#1e293b" stroke-width="10" stroke-linecap="round"/>
        <path id="gauge-mem-arc" d="M 10 70 A 50 50 0 0 1 110 70" fill="none" stroke="#8b5cf6" stroke-width="10" stroke-linecap="round" stroke-dasharray="0 157"/>
      </svg>
      <div class="gauge-value" id="gauge-mem-val">-</div>
      <div class="gauge-label">RAM</div>
    </div>

    <div class="gauge-card">
      <svg viewBox="0 0 120 80" class="gauge-svg">
        <path d="M 10 70 A 50 50 0 0 1 110 70" fill="none" stroke="#1e293b" stroke-width="10" stroke-linecap="round"/>
        <path id="gauge-gpu-arc" d="M 10 70 A 50 50 0 0 1 110 70" fill="none" stroke="#22c55e" stroke-width="10" stroke-linecap="round" stroke-dasharray="0 157"/>
      </svg>
      <div class="gauge-value" id="gauge-gpu-val">-</div>
      <div class="gauge-label">GPU</div>
    </div>

    <div class="gauge-card net-card">
      <div style="font-size:10px; color:#6b7280; margin-bottom:4px;">NETWORK</div>
      <div style="font-size:12px; color:#60a5fa;">↓ <span id="net-recv">-</span> MB</div>
      <div style="font-size:12px; color:#f59e0b; margin-top:3px;">↑ <span id="net-sent">-</span> MB</div>
      <div style="font-size:9px; color:#4b5563; margin-top:4px;" id="net-device">-</div>
    </div>

    <div class="section-title" style="margin-top:14px;">Recent Tiles</div>
    <div id="recent-previews">
      <div style="font-size:10px; color:#4b5563;">No tiles yet</div>
    </div>
  </div><!-- /sidebar -->"""


def _js_constants(colors_json: str) -> str:
    """Return JS constant declarations (CLASS_COLORS, CLASS_NAMES, CLASS_GROUPS, chart variables)."""
    return f""""use strict";

const REFRESH_MS = 5000;
const CLASS_COLORS = {colors_json};

const CLASS_NAMES = {{
  0: "BG", 1: "Pine", 2: "Spruce",
  3: "Deciduous", 4: "Mixed", 5: "Temp non-forest",
  6: "Wetl. pine", 7: "Wetl. spruce",
  8: "Wetl. decid.", 9: "Wetl. mixed",
  10: "Wetl. temp", 11: "Open wetland", 12: "Cropland",
  13: "Bare land", 14: "Vegetated",
  15: "Buildings", 16: "Infra",
  17: "Roads", 18: "Lakes", 19: "Sea"
}};

const CLASS_GROUPS = {{
  "Forest": [1,2,3,4,5],
  "Wetland forest": [6,7,8,9,10],
  "Open wetland": [11],
  "Cropland": [12],
  "Open land": [13,14],
  "Developed": [15,16,17],
  "Water": [18,19],
}};

Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#1e293b';
Chart.defaults.font.family = "'SF Mono', 'Cascadia Code', monospace";
Chart.defaults.font.size = 11;

let lossChart, miouChart, perClassChart, worstChart, classDistChart, classDistDetailChart, evalPerClassChart, sfCompChart;
let _classViewDetailed = false;"""


def _js_utils() -> str:
    """Return utility functions (toggleClassView, fmtNum, fmtTime, computeETA, computePatience, makeLineChart)."""
    return f"""function toggleClassView() {{
  _classViewDetailed = !_classViewDetailed;
  const btn = document.getElementById('toggle-classdist');
  const grouped = document.getElementById('chart-classdist');
  const detail = document.getElementById('chart-classdist-detail');
  if (_classViewDetailed) {{
    btn.textContent = 'Show grouped';
    grouped.style.display = 'none';
    detail.style.display = 'block';
  }} else {{
    btn.textContent = 'Show details';
    grouped.style.display = 'block';
    detail.style.display = 'none';
  }}
}}

function fmtNum(v, d=4) {{
  if (v == null || isNaN(v)) return '-';
  return Number(v).toFixed(d);
}}

function fmtTime(secs) {{
  if (!secs || secs <= 0) return '-';
  if (secs < 60) return Math.round(secs) + 's';
  if (secs < 3600) return Math.round(secs / 60) + ' min';
  return (secs / 3600).toFixed(1) + ' h';
}}

function computeETA(epochs, totalEpochs) {{
  if (!epochs || epochs.length < 1) return '-';
  const rem = totalEpochs - epochs.length;
  if (rem <= 0) return 'Done';
  const avg = epochs.reduce((s,e) => s + e.elapsed_s, 0) / epochs.length;
  return fmtTime(rem * avg);
}}

function computePatience(epochs, maxP) {{
  let c = 0;
  for (let i = epochs.length - 1; i >= 0; i--) {{
    if (epochs[i].is_best) break;
    c++;
  }}
  return Math.min(c, maxP);
}}

function makeLineChart(ctx, label, color) {{
  return new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: [],
      datasets: [{{
        label: label,
        data: [],
        borderColor: color,
        backgroundColor: color + '20',
        borderWidth: 2,
        pointRadius: 2,
        fill: true,
        tension: 0.3,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Epoch' }} }},
        y: {{ beginAtZero: false }}
      }}
    }}
  }});
}}"""


def _js_init_charts() -> str:
    """Return the initCharts function."""
    return f"""function initCharts() {{
  lossChart = makeLineChart(
    document.getElementById('chart-loss'), 'Loss', '#ef4444');
  miouChart = new Chart(document.getElementById('chart-miou'), {{
    type: 'line',
    data: {{
      labels: [],
      datasets: [{{
        label: 'mIoU',
        data: [],
        borderColor: '#3b82f6',
        backgroundColor: '#3b82f620',
        borderWidth: 2,
        pointRadius: [],
        pointBackgroundColor: [],
        fill: true,
        tension: 0.3,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Epoch' }} }},
        y: {{ beginAtZero: true, max: 1.0 }}
      }}
    }}
  }});
  worstChart = makeLineChart(
    document.getElementById('chart-worst'), 'Worst-Class IoU', '#f97316');
  worstChart.options.scales.y = {{ beginAtZero: true, max: 1.0 }};

  perClassChart = new Chart(document.getElementById('chart-perclass'), {{
    type: 'bar',
    data: {{ labels: [], datasets: [{{ data: [], backgroundColor: [], borderWidth: 0 }}] }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ beginAtZero: true, max: 1.0, title: {{ display: true, text: 'IoU' }} }}
      }}
    }}
  }});

  classDistChart = new Chart(document.getElementById('chart-classdist'), {{
    type: 'doughnut',
    data: {{ labels: [], datasets: [{{ data: [], backgroundColor: [], borderColor: '#0f172a', borderWidth: 2 }}] }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      cutout: '55%',
      plugins: {{
        legend: {{
          display: true,
          position: 'right',
          labels: {{ color: '#e2e8f0', padding: 12, font: {{ size: 12 }} }}
        }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              const v = ctx.parsed;
              const total = ctx.dataset.data.reduce((a,b) => a+b, 0);
              return ' ' + ctx.label + ': ' + v.toLocaleString() + ' px (' + (v/total*100).toFixed(1) + '%)';
            }}
          }}
        }}
      }}
    }}
  }});

  classDistDetailChart = new Chart(document.getElementById('chart-classdist-detail'), {{
    type: 'doughnut',
    data: {{ labels: [], datasets: [{{ data: [], backgroundColor: [], borderColor: '#0f172a', borderWidth: 1 }}] }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      cutout: '55%',
      plugins: {{
        legend: {{
          display: true,
          position: 'right',
          labels: {{ color: '#e2e8f0', padding: 6, font: {{ size: 10 }} }}
        }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              const v = ctx.parsed;
              const total = ctx.dataset.data.reduce((a,b) => a+b, 0);
              return ' ' + ctx.label + ': ' + v.toLocaleString() + ' px (' + (v/total*100).toFixed(1) + '%)';
            }}
          }}
        }}
      }}
    }}
  }});

  evalPerClassChart = new Chart(document.getElementById('chart-eval-perclass'), {{
    type: 'bar',
    data: {{ labels: [], datasets: [{{ data: [], backgroundColor: [], borderWidth: 0 }}] }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ beginAtZero: true, max: 1.0, title: {{ display: true, text: 'IoU' }} }}
      }}
    }}
  }});

  sfCompChart = new Chart(document.getElementById('chart-sf-comparison'), {{
    type: 'bar',
    data: {{
      labels: ['Completed', 'Failed', 'Running'],
      datasets: [
        {{ label: 'CDSE', data: [0, 0, 0], backgroundColor: '#3b82f6', borderRadius: 4 }},
        {{ label: 'DES', data: [0, 0, 0], backgroundColor: '#f59e0b', borderRadius: 4 }},
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: true, position: 'top', labels: {{ color: '#e2e8f0', padding: 12 }} }}
      }},
      scales: {{
        y: {{ beginAtZero: true, ticks: {{ precision: 0 }} }}
      }}
    }}
  }});
}}"""


def _js_update_sections() -> str:
    """Return all update functions (updateNmdPrefilter, updateSeasonalFetch, updateGlobalStatus, updateDataPrep, updateClassDistChart, updateEvaluation, updateTraining)."""
    return f"""// ── NMD Pre-filter ──────────────────────────────────────────────
function updateNmdPrefilter(nmdLog) {{
  const nmdBadge = document.getElementById('nmd-badge');
  if (!nmdLog) return;

  const total = nmdLog.total_cells || 0;
  const processed = nmdLog.processed || 0;
  const land = nmdLog.land_kept || 0;
  const water = nmdLog.water_skipped || 0;
  const fail = nmdLog.failed || 0;

  document.getElementById('nmd-progress-text').textContent =
    total > 0 ? processed + '/' + total : '-';
  document.getElementById('nmd-land').textContent = land;
  document.getElementById('nmd-water').textContent = water;
  document.getElementById('nmd-failed').textContent = fail;

  if (total > 0) {{
    const pct = Math.round(processed / total * 100);
    document.getElementById('nmd-progress').style.width = pct + '%';
  }}

  if (nmdLog.status === 'completed') {{
    nmdBadge.className = 'section-badge done';
    nmdBadge.textContent = 'done';
  }} else if (nmdLog.status === 'running') {{
    nmdBadge.className = 'section-badge active';
    nmdBadge.textContent = 'running';
  }}
}}

// ── Seasonal Fetch (ColonyOS) ────────────────────────────────────
function updateSeasonalFetch(sf) {{
  const badge = document.getElementById('sf-badge');
  if (!sf) return;

  // Badge
  if (sf.status === 'completed') {{
    badge.className = 'section-badge done';
    badge.textContent = 'done';
  }} else if (sf.status === 'running') {{
    badge.className = 'section-badge active';
    badge.textContent = 'running';
  }} else {{
    badge.className = 'section-badge pending';
    badge.textContent = sf.status || 'waiting';
  }}

  // Cards
  const total = sf.total_tiles || 0;
  const done = sf.completed || 0;
  const fail = sf.failed || 0;
  const run = sf.running || 0;
  const processed = done + fail + run;

  document.getElementById('sf-progress-text').textContent =
    total > 0 ? done + '/' + total : '-';
  document.getElementById('sf-completed').textContent = done;
  document.getElementById('sf-running').textContent = run;
  document.getElementById('sf-failed').textContent = fail;
  document.getElementById('sf-rate').textContent =
    sf.rate_tiles_per_hour ? sf.rate_tiles_per_hour.toFixed(0) : '-';
  document.getElementById('sf-eta').textContent = fmtTime(sf.eta_s);
  document.getElementById('sf-elapsed').textContent =
    sf.elapsed_s ? fmtTime(sf.elapsed_s) + ' elapsed' : '';

  if (total > 0) {{
    const pct = Math.round(done / total * 100);
    document.getElementById('sf-progress').style.width = pct + '%';
  }}

  if (fail > 0) {{
    document.getElementById('sf-failed-sub').textContent = 'fetch errors';
  }}

  // Per-source stats
  const cdse = (sf.sources || {{}}).copernicus || {{}};
  const des = (sf.sources || {{}}).des || {{}};

  document.getElementById('sf-cdse-done').textContent = cdse.completed || 0;
  document.getElementById('sf-cdse-fail').textContent = cdse.failed || 0;
  document.getElementById('sf-cdse-avg').textContent =
    cdse.avg_time_s ? cdse.avg_time_s.toFixed(0) + 's' : '-';
  document.getElementById('sf-cdse-rate').textContent =
    cdse.success_rate != null ? (cdse.success_rate * 100).toFixed(1) + '%' : '-';

  document.getElementById('sf-des-done').textContent = des.completed || 0;
  document.getElementById('sf-des-fail').textContent = des.failed || 0;
  document.getElementById('sf-des-avg').textContent =
    des.avg_time_s ? des.avg_time_s.toFixed(0) + 's' : '-';
  document.getElementById('sf-des-rate').textContent =
    des.success_rate != null ? (des.success_rate * 100).toFixed(1) + '%' : '-';

  // Comparison bar chart
  if (sfCompChart) {{
    sfCompChart.data.datasets[0].data = [
      cdse.completed || 0, cdse.failed || 0, cdse.running || 0
    ];
    sfCompChart.data.datasets[1].data = [
      des.completed || 0, des.failed || 0, des.running || 0
    ];
    sfCompChart.update('none');
  }}
}}

// ── Global status logic ─────────────────────────────────────────
function updateGlobalStatus(nmdLog, prepLog, trainLog) {{
  const badge = document.getElementById('status-badge');
  const text = document.getElementById('status-text');
  const dpBadge = document.getElementById('dp-badge');
  const trBadge = document.getElementById('tr-badge');
  const nmdBadge = document.getElementById('nmd-badge');

  let status = 'waiting';
  let label = 'Waiting for data...';

  // NMD pre-filter running
  if (nmdLog && nmdLog.status === 'running') {{
    status = 'preparing';
    label = 'NMD pre-filtering...';
    nmdBadge.className = 'section-badge active';
    nmdBadge.textContent = 'running';
    dpBadge.className = 'section-badge pending';
    dpBadge.textContent = 'pending';
    trBadge.className = 'section-badge pending';
    trBadge.textContent = 'pending';
  }} else if (prepLog && prepLog.status === 'running') {{
    status = 'preparing';
    label = 'Fetching spectral data...';
    if (nmdLog) {{
      nmdBadge.className = 'section-badge done';
      nmdBadge.textContent = 'done';
    }}
    dpBadge.className = 'section-badge active';
    dpBadge.textContent = 'running';
    trBadge.className = 'section-badge pending';
    trBadge.textContent = 'pending';
  }} else if (trainLog && trainLog.status === 'running') {{
    status = 'training';
    label = 'Training...';
    if (nmdLog) {{ nmdBadge.className = 'section-badge done'; nmdBadge.textContent = 'done'; }}
    dpBadge.className = 'section-badge done';
    dpBadge.textContent = 'done';
    trBadge.className = 'section-badge active';
    trBadge.textContent = 'running';
  }} else if (trainLog && trainLog.status === 'completed') {{
    status = 'completed';
    label = 'Completed';
    if (nmdLog) {{ nmdBadge.className = 'section-badge done'; nmdBadge.textContent = 'done'; }}
    dpBadge.className = 'section-badge done';
    dpBadge.textContent = 'done';
    trBadge.className = 'section-badge done';
    trBadge.textContent = 'done';
  }} else if (trainLog && trainLog.status === 'stopped') {{
    status = 'stopped';
    label = 'Stopped (early)';
    if (nmdLog) {{ nmdBadge.className = 'section-badge done'; nmdBadge.textContent = 'done'; }}
    dpBadge.className = 'section-badge done';
    dpBadge.textContent = 'done';
    trBadge.className = 'section-badge done';
    trBadge.textContent = 'stopped';
  }} else if (prepLog && prepLog.status === 'completed' && !trainLog) {{
    // --skip-train: data prep done, no training expected
    status = 'completed';
    label = 'Completed (data only)';
    if (nmdLog) {{ nmdBadge.className = 'section-badge done'; nmdBadge.textContent = 'done'; }}
    dpBadge.className = 'section-badge done';
    dpBadge.textContent = 'done';
    trBadge.className = 'section-badge pending';
    trBadge.textContent = 'skipped';
  }} else if (prepLog && prepLog.status === 'completed') {{
    status = 'preparing';
    label = 'Data ready, waiting for training...';
    if (nmdLog) {{ nmdBadge.className = 'section-badge done'; nmdBadge.textContent = 'done'; }}
    dpBadge.className = 'section-badge done';
    dpBadge.textContent = 'done';
    trBadge.className = 'section-badge pending';
    trBadge.textContent = 'pending';
  }} else if (trainLog && trainLog.status === 'initialized') {{
    status = 'training';
    label = 'Initializing model...';
    if (nmdLog) {{ nmdBadge.className = 'section-badge done'; nmdBadge.textContent = 'done'; }}
    dpBadge.className = 'section-badge done';
    dpBadge.textContent = 'done';
    trBadge.className = 'section-badge active';
    trBadge.textContent = 'starting';
  }}

  badge.className = 'status-badge status-' + status;
  text.textContent = label;
}}

// ── Data Preparation ────────────────────────────────────────────
function updateDataPrep(prepLog, stats) {{
  if (prepLog) {{
    const ok = prepLog.completed || 0;
    const fail = prepLog.failed || 0;
    const total = prepLog.grid_cells || 0;
    const processed = ok + fail;

    document.getElementById('dp-progress-text').textContent =
      total > 0 ? processed + '/' + total : '-';
    document.getElementById('dp-tiles').textContent = prepLog.tiles_saved || 0;
    document.getElementById('dp-failed-sub').textContent =
      fail > 0 ? fail + ' failed' : '';

    if (total > 0) {{
      const pct = Math.round(processed / total * 100);
      document.getElementById('dp-progress').style.width = pct + '%';
    }}

    document.getElementById('dp-split').textContent =
      prepLog.current_split || '-';

    // Latest tile
    if (prepLog.latest_tile) {{
      const name = prepLog.latest_tile.replace('tile_', '').replace('.npz', '');
      document.getElementById('dp-latest').textContent = name;
      const parts = [];
      if (prepLog.latest_date) parts.push(prepLog.latest_date);
      if (prepLog.latest_cloud) parts.push('cloud ' + (prepLog.latest_cloud * 100).toFixed(1) + '%');
      document.getElementById('dp-latest-sub').textContent = parts.join(', ');
    }}

    // ETA
    const elapsed = prepLog.elapsed_s || 0;
    document.getElementById('dp-elapsed').textContent = fmtTime(elapsed) + ' elapsed';
    if (processed > 0 && total > processed) {{
      const avg = elapsed / processed;
      const remaining = (total - processed) * avg;
      document.getElementById('dp-eta').textContent = fmtTime(remaining);
    }} else if (prepLog.status === 'completed') {{
      document.getElementById('dp-eta').textContent = 'Done';
    }}

    // Live class distribution from prepare_log
    const counts = prepLog.class_counts || {{}};
    if (Object.keys(counts).length > 0) {{
      updateClassDistChart(counts, []);
    }}
  }}

  // Final stats (from class_stats.json) override prepare_log counts
  if (stats) {{
    const rareClasses = stats.rare_classes || [];
    document.getElementById('dp-rare').textContent = rareClasses.length;
    const counts = stats.class_counts || {{}};
    if (Object.keys(counts).length > 0) {{
      updateClassDistChart(counts, rareClasses);
    }}
  }}
}}

function updateClassDistChart(counts, rareClasses) {{
  const total = Object.entries(counts)
    .filter(([k]) => parseInt(k) > 0)
    .reduce((s,[,v]) => s + v, 0);

  // ── Grouped doughnut (default) ──
  const groupColors = {{
    "Forest": "#22c55e", "Wetland forest": "#14b8a6",
    "Open wetland": "#06b6d4", "Cropland": "#eab308",
    "Open land": "#f59e0b", "Developed": "#ef4444", "Water": "#3b82f6"
  }};

  const grouped = Object.entries(CLASS_GROUPS).map(([name, ids]) => {{
    const sum = ids.reduce((s, id) => s + (counts[id] || 0), 0);
    const pct = total > 0 ? (sum/total*100).toFixed(1) : '0';
    return {{ name, count: sum, pct }};
  }}).sort((a,b) => b.count - a.count);

  const gds = classDistChart.data.datasets[0];
  classDistChart.data.labels = grouped.map(e => e.name + ' ' + e.pct + '%');
  gds.data = grouped.map(e => e.count);
  gds.backgroundColor = grouped.map(e => groupColors[e.name] || '#6b7280');
  classDistChart.update('none');

  // ── Detail pie ──
  const entries = Object.entries(counts)
    .map(([k,v]) => ({{ idx: parseInt(k), count: v, name: CLASS_NAMES[parseInt(k)] || k }}))
    .filter(e => e.idx > 0)
    .sort((a,b) => b.count - a.count);

  const dds = classDistDetailChart.data.datasets[0];
  classDistDetailChart.data.labels = entries.map(e =>
    e.name + ' ' + (total > 0 ? (e.count/total*100).toFixed(1) : '0') + '%');
  dds.data = entries.map(e => e.count);
  dds.backgroundColor = entries.map(e => CLASS_COLORS[e.name] || '#6b7280');
  classDistDetailChart.update('none');
}}

// ── Evaluation ──────────────────────────────────────────────────
function updateEvaluation(evalData) {{
  const badge = document.getElementById('eval-badge');
  if (!evalData) return;

  badge.className = 'section-badge done';
  badge.textContent = 'done';

  document.getElementById('eval-miou').textContent = fmtNum(evalData.miou);
  document.getElementById('eval-oa').textContent =
    evalData.overall_accuracy != null ? (evalData.overall_accuracy * 100).toFixed(1) + '%' : '-';
  document.getElementById('eval-tiles').textContent = evalData.n_tiles || '-';
  document.getElementById('eval-time').textContent =
    evalData.elapsed_s ? fmtTime(evalData.elapsed_s) : '';
  document.getElementById('eval-checkpoint').textContent =
    evalData.checkpoint ? evalData.checkpoint.split('/').pop() : '';

  const aux = evalData.aux_channels || [];
  document.getElementById('eval-aux').textContent =
    aux.length > 0 ? aux.join(', ') : 'none (baseline)';

  // Per-class IoU bar chart
  if (evalData.per_class_iou) {{
    const items = Object.entries(evalData.per_class_iou)
      .sort((a,b) => a[1] - b[1]);
    evalPerClassChart.data.labels = items.map(e => e[0]);
    const ds = evalPerClassChart.data.datasets[0];
    ds.data = items.map(e => e[1]);
    ds.backgroundColor = items.map(e => {{
      const v = e[1];
      if (v >= 0.6) return '#22c55e';
      if (v >= 0.4) return '#eab308';
      if (v >= 0.2) return '#f97316';
      return '#ef4444';
    }});
    evalPerClassChart.update('none');
  }}
}}

// ── Training ────────────────────────────────────────────────────
function updateTraining(log) {{
  const epochs = log.epochs || [];
  const cfg = log.config || {{}};
  const totalEpochs = cfg.epochs || 0;
  const last = epochs.length > 0 ? epochs[epochs.length - 1] : null;

  document.getElementById('tr-epoch').textContent =
    last ? last.epoch + '/' + totalEpochs : '-';
  document.getElementById('tr-best-miou').textContent =
    log.best_metric != null ? fmtNum(log.best_metric) : '-';
  document.getElementById('tr-best-epoch').textContent =
    log.best_epoch ? 'epoch ' + log.best_epoch : '';
  document.getElementById('tr-lr').textContent =
    last ? Number(last.lr).toExponential(1) : '-';
  document.getElementById('tr-eta').textContent =
    computeETA(epochs, totalEpochs);
  document.getElementById('tr-loss').textContent =
    last ? fmtNum(last.train_loss) : '-';
  document.getElementById('tr-loss-type').textContent =
    cfg.loss_type || '';

  // Patience
  const maxP = cfg.early_stopping_patience || 5;
  const usedP = epochs.length > 0 ? computePatience(epochs, maxP) : 0;
  document.getElementById('tr-patience-text').textContent = usedP + '/' + maxP;
  const dotsEl = document.getElementById('tr-patience-dots');
  dotsEl.innerHTML = '';
  for (let i = 0; i < maxP; i++) {{
    const d = document.createElement('span');
    d.className = 'patience-dot' + (i < usedP ? ' used' : '');
    dotsEl.appendChild(d);
  }}

  // Loss chart
  const labels = epochs.map(e => e.epoch);
  lossChart.data.labels = labels;
  lossChart.data.datasets[0].data = epochs.map(e => e.train_loss);
  lossChart.update('none');

  // mIoU chart with best marked
  const bestEp = log.best_epoch;
  miouChart.data.labels = labels;
  const ds = miouChart.data.datasets[0];
  ds.data = epochs.map(e => e.val_miou);
  ds.pointRadius = epochs.map(e => e.epoch === bestEp ? 6 : 2);
  ds.pointBackgroundColor = epochs.map(e =>
    e.epoch === bestEp ? '#eab308' : '#3b82f6');
  miouChart.update('none');

  // Worst-class chart
  worstChart.data.labels = labels;
  worstChart.data.datasets[0].data = epochs.map(e => e.worst_class_iou);
  worstChart.update('none');

  // Per-class IoU (latest epoch)
  if (last && last.per_class_iou) {{
    const items = Object.entries(last.per_class_iou)
      .sort((a,b) => a[1] - b[1]);
    perClassChart.data.labels = items.map(e => e[0]);
    const pds = perClassChart.data.datasets[0];
    pds.data = items.map(e => e[1]);
    pds.backgroundColor = items.map(e => CLASS_COLORS[e[0]] || '#6b7280');
    perClassChart.update('none');
  }}
}}"""


def _js_refresh_loop() -> str:
    """Return fetchJSON, updateGauge, updateSystemMetrics, LULC code, refresh function, and boot code."""
    return f"""// ── Refresh loop ────────────────────────────────────────────────
let _fetchFails = 0;
let _lastGoodData = {{}};

async function fetchJSON(url) {{
  try {{
    const r = await fetch(url + '?t=' + Date.now());
    if (!r.ok) return null;
    _fetchFails = 0;
    return await r.json();
  }} catch(e) {{
    _fetchFails++;
    return null;
  }}
}}

function updateGauge(arcId, valId, pct, suffix) {{
  const arc = document.getElementById(arcId);
  const val = document.getElementById(valId);
  if (!arc || !val) return;
  const p = Math.max(0, Math.min(100, pct || 0));
  // Arc length: 157 is the full semicircle path length
  arc.setAttribute('stroke-dasharray', (p / 100 * 157) + ' 157');
  // Color: blue→yellow→red
  if (p > 80) arc.setAttribute('stroke', '#ef4444');
  else if (p > 60) arc.setAttribute('stroke', '#eab308');
  val.textContent = Math.round(p) + (suffix || '%');
}}

function updateSystemMetrics(m) {{
  if (!m) return;
  updateGauge('gauge-cpu-arc', 'gauge-cpu-val', m.cpu_percent, '%');
  updateGauge('gauge-mem-arc', 'gauge-mem-val', m.memory_percent, '%');
  if (m.gpu_percent != null) {{
    updateGauge('gauge-gpu-arc', 'gauge-gpu-val', m.gpu_percent, '%');
  }} else if (m.gpu_memory_used_gb != null) {{
    document.getElementById('gauge-gpu-val').textContent = m.gpu_memory_used_gb + ' GB';
    document.getElementById('gauge-gpu-arc').setAttribute('stroke-dasharray', '0 157');
  }} else {{
    document.getElementById('gauge-gpu-val').textContent = 'N/A';
  }}
  if (m.net_recv_mb != null) document.getElementById('net-recv').textContent = m.net_recv_mb;
  if (m.net_sent_mb != null) document.getElementById('net-sent').textContent = m.net_sent_mb;
  document.getElementById('net-device').textContent = (m.device || '').toUpperCase();
}}

// ── LULC Inference ────────────────────────────────────────────
const LULC_CLASS_LEGEND = [
  {{color:'#006400',label:'Tallskog'}},{{color:'#228B22',label:'Granskog'}},
  {{color:'#32CD32',label:'Lövskog'}},{{color:'#3CB371',label:'Blandskog'}},
  {{color:'#2E4F2E',label:'Sumpskog'}},{{color:'#8B5A2B',label:'Öpp. våtmark'}},
  {{color:'#FFD700',label:'Åkermark'}},{{color:'#D2B48C',label:'Öpp. mark'}},
  {{color:'#FF0000',label:'Bebyggelse'}},{{color:'#0000FF',label:'Vatten'}}
];
let _lulcChartCreated = false;
let _lulcGalleryRendered = false;

// Init class legend
(function() {{
  const el = document.getElementById('lulc-class-legend');
  if (!el) return;
  LULC_CLASS_LEGEND.forEach(function(item) {{
    el.innerHTML += '<span class="lulc-legend-item">' +
      '<span class="lulc-legend-swatch" style="background:' + item.color + '"></span>' +
      item.label + '</span>';
  }});
}})();

function updateLulcInference(summary, gallery) {{
  if (!summary) return;
  const badge = document.getElementById('lulc-badge');
  if (summary.tiles > 0) {{
    badge.textContent = 'done';
    badge.className = 'section-badge done';
  }}
  const acc = summary.overall_agreement_pct || summary.overall_accuracy || 0;
  document.getElementById('lulc-accuracy').textContent = acc.toFixed(1) + '%';
  document.getElementById('lulc-pixels-sub').textContent =
    (summary.total_pixels || 0).toLocaleString() + ' pixlar';
  document.getElementById('lulc-hcw').textContent =
    (summary.high_confidence_wrong || 0).toLocaleString();
  document.getElementById('lulc-tiles').textContent =
    String(summary.tiles || 0);
  const disagree_pct = summary.total_pixels > 0
    ? (100 * summary.disagree_pixels / summary.total_pixels).toFixed(1) + '%'
    : '-';
  document.getElementById('lulc-disagree').textContent = disagree_pct;
  document.getElementById('lulc-disagree-sub').textContent =
    (summary.disagree_pixels || 0).toLocaleString() + ' pixlar';

  // Per-class chart
  if (summary.per_class && !_lulcChartCreated) {{
    _lulcChartCreated = true;
    const pc = summary.per_class;
    const labels = Object.keys(pc);
    const values = labels.map(k => pc[k].accuracy_pct || 0);
    const colors = LULC_CLASS_LEGEND.map(l => l.color);
    const ctx = document.getElementById('chart-lulc-perclass');
    if (ctx) {{
      new Chart(ctx, {{
        type: 'bar',
        data: {{
          labels: labels,
          datasets: [{{ label: 'Accuracy (%)', data: values,
            backgroundColor: colors.slice(0, labels.length),
            borderWidth: 1 }}]
        }},
        options: {{
          indexAxis: 'y', responsive: true,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ beginAtZero: true, max: 100, title: {{ display: true, text: '%' }} }},
            y: {{ grid: {{ display: false }} }}
          }}
        }}
      }});
    }}
  }}

  // Render gallery
  if (gallery && gallery.length > 0 && !_lulcGalleryRendered) {{
    _lulcGalleryRendered = true;
    const container = document.getElementById('lulc-gallery');
    container.innerHTML = '';
    gallery.forEach(function(tile) {{
      const accColor = tile.accuracy_pct >= 70 ? '#2ecc40' :
                       tile.accuracy_pct >= 50 ? '#eab308' : '#ef4444';
      container.innerHTML +=
        '<div class="lulc-gallery-row">' +
        '<div class="lulc-gallery-cell"><img src="' + tile.s2 + '" loading="lazy">' +
        '<span class="cell-label">' + (tile.dominant_class || '') + '</span></div>' +
        '<div class="lulc-gallery-cell"><img src="' + tile.nmd + '" loading="lazy"></div>' +
        '<div class="lulc-gallery-cell"><img src="' + tile.pred + '" loading="lazy"></div>' +
        '<div class="lulc-gallery-cell"><img src="' + tile.quality + '" loading="lazy">' +
        '<span class="cell-label" style="color:' + accColor + '">' +
        tile.accuracy_pct + '% · ' + tile.unique_classes + ' klasser</span></div>' +
        '</div>';
    }});
  }}
}}

async function refresh() {{
  const [nmdLog, prepLog, trainLog, stats, sysMetrics, evalTest, sfLog, lulcSummary, lulcGallery] = await Promise.all([
    fetchJSON('nmd_prefilter_log.json'),
    fetchJSON('prepare_log.json'),
    fetchJSON('training_log.json'),
    fetchJSON('class_stats.json'),
    fetchJSON('system_metrics.json'),
    fetchJSON('eval_test.json'),
    fetchJSON('seasonal_fetch_log.json'),
    fetchJSON('predictions/val/prediction_summary.json'),
    fetchJSON('predictions/val/gallery.json'),
  ]);

  // Keep last good data so UI stays populated after disconnect
  if (nmdLog) _lastGoodData.nmdLog = nmdLog;
  if (prepLog) _lastGoodData.prepLog = prepLog;
  if (trainLog) _lastGoodData.trainLog = trainLog;
  if (stats) _lastGoodData.stats = stats;
  if (evalTest) _lastGoodData.evalTest = evalTest;
  if (sfLog) _lastGoodData.sfLog = sfLog;
  if (lulcSummary) _lastGoodData.lulcSummary = lulcSummary;
  if (lulcGallery) _lastGoodData.lulcGallery = lulcGallery;

  const n = nmdLog || _lastGoodData.nmdLog;
  const p = prepLog || _lastGoodData.prepLog;
  const t = trainLog || _lastGoodData.trainLog;
  const s = stats || _lastGoodData.stats;
  const ev = evalTest || _lastGoodData.evalTest;
  const sf = sfLog || _lastGoodData.sfLog;
  const ls = lulcSummary || _lastGoodData.lulcSummary;
  const lg = lulcGallery || _lastGoodData.lulcGallery;

  updateGlobalStatus(n, p, t);
  if (n) updateNmdPrefilter(n);
  if (sf) updateSeasonalFetch(sf);
  if (p || s) updateDataPrep(p, s);
  if (ev) updateEvaluation(ev);
  if (t) updateTraining(t);
  if (sysMetrics) updateSystemMetrics(sysMetrics);
  if (ls) updateLulcInference(ls, lg);

  // Update preview thumbnails
  if (p && p.recent_previews && p.recent_previews.length > 0) {{
    const container = document.getElementById('recent-previews');
    const previews = p.recent_previews;
    let html = '';
    for (let i = previews.length - 1; i >= 0; i--) {{
      const fname = previews[i];
      const base = fname.replace('preview_', '').replace('.png', '');
      const dateStr = base.slice(-10);
      const cellKey = base.slice(0, base.length - 11);
      html += '<img class="preview-thumb" src="tiles/' + fname + '?t=' + Date.now() + '" '
           + 'onerror="this.hidden=true">';
      html += '<div class="preview-label">' + cellKey + '<br>' + dateStr + '</div>';
    }}
    container.innerHTML = html;
  }}

  // Show disconnect warning after 3 consecutive failures
  if (_fetchFails >= 3) {{
    const text = document.getElementById('status-text');
    const badge = document.getElementById('status-badge');
    if (!text.textContent.includes('Completed')) {{
      text.textContent += ' (server disconnected)';
      badge.className = 'status-badge status-stopped';
    }}
  }}
}}

initCharts();
refresh();
setInterval(refresh, REFRESH_MS);"""


def _build_html(chart_js_src: str) -> str:
    """Build the complete dashboard HTML string."""
    import json
    colors_json = json.dumps(_CLASS_COLORS)

    return f"""<!DOCTYPE html>
<html lang="sv">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IMINT Training Dashboard</title>
{_css_styles()}
</head>
<body>

{_html_header()}

<div class="container">

  <!-- Main content -->
  <div class="main-content">

{_html_nmd_section()}

{_html_seasonal_fetch_section()}

{_html_dataprep_section()}

{_html_eval_section()}

{_html_training_section()}

{_html_lulc_section()}

  </div><!-- /main-content -->

{_html_sidebar()}

</div><!-- /container -->

<script>
{chart_js_src}
</script>
<script>
{_js_constants(colors_json)}

{_js_utils()}

{_js_init_charts()}

{_js_update_sections()}

{_js_refresh_loop()}
</script>
</body>
</html>"""


def generate_dashboard(data_dir: str) -> str:
    """Generate the training dashboard HTML file in data_dir.

    Returns:
        Path to the generated HTML file.
    """
    chart_js = _fetch_chart_js()
    html = _build_html(chart_js)

    out_path = Path(data_dir) / "training_dashboard.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  Dashboard: {out_path}")
    return str(out_path)


def start_dashboard_server(
    data_dir: str,
    port: int = 8000,
    open_browser: bool = True,
) -> tuple:
    """Start a background HTTP server for the dashboard.

    The server runs as a daemon thread and stops automatically when
    the main process exits.

    Returns:
        Tuple of (HTTPServer, Thread) so callers can keep the process
        alive or shut down the server when done.
    """
    data_path = Path(data_dir).resolve()
    generate_dashboard(str(data_path))

    data_dir_str = str(data_path)

    class QuietHandler(SimpleHTTPRequestHandler):
        """Serve from data_dir without logging each request."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=data_dir_str, **kwargs)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("localhost", port), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://localhost:{port}/training_dashboard.html"
    print(f"  Dashboard URL: {url}")

    if open_browser:
        webbrowser.open(url)

    return server, thread
