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
<style>
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
.sidebar {{ width: 160px; flex-shrink: 0; position: sticky; top: 24px; align-self: flex-start; }}
.sidebar .section-title {{ font-size: 11px; margin-bottom: 10px; }}
.sidebar .gauge-card {{ margin-bottom: 10px; padding: 8px 10px; min-width: unset; }}
.sidebar .gauge-svg {{ width: 80px; height: 52px; }}
.sidebar .gauge-value {{ font-size: 16px; margin-top: -6px; }}
.sidebar .gauge-label {{ font-size: 9px; }}
.sidebar .net-card {{ padding: 10px 8px; }}
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
@media (max-width: 900px) {{
  .container {{ flex-direction: column; }}
  .sidebar {{ width: 100%; position: static; display: flex; flex-wrap: wrap; gap: 10px; }}
  .sidebar .gauge-card {{ margin-bottom: 0; }}
  .chart-grid {{ grid-template-columns: 1fr; }}
  .cards {{ grid-template-columns: repeat(2, 1fr); }}
}}
</style>
</head>
<body>

<div class="header">
  <h1>IMINT Training Dashboard</h1>
  <div class="header-right">
    <div class="status-badge status-waiting" id="status-badge">
      <span class="status-dot"></span>
      <span id="status-text">Waiting...</span>
    </div>
  </div>
</div>

<div class="container">

  <!-- Main content -->
  <div class="main-content">

  <!-- NMD Pre-filter Section -->
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
  </div>

  <!-- Data Preparation Section -->
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
  </div>

  <!-- Training Section -->
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
  </div>

  </div><!-- /main-content -->

  <!-- Sidebar: System Metrics -->
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
  </div><!-- /sidebar -->

</div><!-- /container -->

<script>
{chart_js_src}
</script>
<script>
"use strict";

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

let lossChart, miouChart, perClassChart, worstChart, classDistChart, classDistDetailChart;
let _classViewDetailed = false;

function toggleClassView() {{
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
}}

function initCharts() {{
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
    type: 'pie',
    data: {{ labels: [], datasets: [{{ data: [], backgroundColor: [], borderColor: '#0f172a', borderWidth: 1 }}] }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
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
}}

// ── NMD Pre-filter ──────────────────────────────────────────────
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
}}

// ── Refresh loop ────────────────────────────────────────────────
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

async function refresh() {{
  const [nmdLog, prepLog, trainLog, stats, sysMetrics] = await Promise.all([
    fetchJSON('nmd_prefilter_log.json'),
    fetchJSON('prepare_log.json'),
    fetchJSON('training_log.json'),
    fetchJSON('class_stats.json'),
    fetchJSON('system_metrics.json'),
  ]);

  // Keep last good data so UI stays populated after disconnect
  if (nmdLog) _lastGoodData.nmdLog = nmdLog;
  if (prepLog) _lastGoodData.prepLog = prepLog;
  if (trainLog) _lastGoodData.trainLog = trainLog;
  if (stats) _lastGoodData.stats = stats;

  const n = nmdLog || _lastGoodData.nmdLog;
  const p = prepLog || _lastGoodData.prepLog;
  const t = trainLog || _lastGoodData.trainLog;
  const s = stats || _lastGoodData.stats;

  updateGlobalStatus(n, p, t);
  if (n) updateNmdPrefilter(n);
  if (p || s) updateDataPrep(p, s);
  if (t) updateTraining(t);
  if (sysMetrics) updateSystemMetrics(sysMetrics);

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
setInterval(refresh, REFRESH_MS);
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
