"""Generate an interactive HTML vessel annotation tool.

Embeds the Bohuslän RGB image + existing model detections into a
self-contained HTML page where the user can click to mark missed vessels.
"""
import base64, json, os

IMG_PATH = "outputs/vessel_test/bohuslan_rgb.png"
OUT_PATH = "outputs/vessel_test/vessel_annotator.html"

# Model detections from conf=0.1 run (low-confidence to show everything the model saw)
MODEL_DETECTIONS = [
    {"score": 0.171, "bbox": {"y_min": 114, "y_max": 124, "x_min": 205, "x_max": 218}},
    {"score": 0.119, "bbox": {"y_min": 148, "y_max": 153, "x_min": 96, "x_max": 101}},
]

# Image dimensions (from fetch result)
IMG_H, IMG_W = 573, 324


def main():
    with open(IMG_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    detections_json = json.dumps(MODEL_DETECTIONS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vessel Annotation Tool — Bohuslän 2025-07-10</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  background: #0b0e17; color: #d8dae5; font-family: system-ui, -apple-system, sans-serif;
  display: flex; height: 100vh; overflow: hidden;
}}

/* ── Sidebar ───────────────────────────────────────────────────── */
#sidebar {{
  width: 340px; min-width: 340px; background: #12162a;
  border-right: 1px solid #2a2f45; display: flex; flex-direction: column;
  overflow: hidden;
}}
#sidebar h2 {{
  padding: 16px; font-size: 15px; border-bottom: 1px solid #2a2f45;
  background: #181d33;
}}
#controls {{
  padding: 12px 16px; border-bottom: 1px solid #2a2f45;
  display: flex; gap: 8px; flex-wrap: wrap;
}}
#controls button {{
  padding: 6px 14px; border: 1px solid #3a4060; border-radius: 6px;
  background: #1e2340; color: #d8dae5; cursor: pointer; font-size: 13px;
  transition: background .15s;
}}
#controls button:hover {{ background: #2a3050; }}
#controls button.primary {{
  background: #1a6b3a; border-color: #28a050;
}}
#controls button.primary:hover {{ background: #228b48; }}
#controls button.danger {{ border-color: #a03030; }}
#controls button.danger:hover {{ background: #401818; }}

#mode-indicator {{
  padding: 8px 16px; font-size: 12px; color: #8090b0;
  border-bottom: 1px solid #2a2f45;
}}
#mode-indicator .mode {{ color: #50d080; font-weight: 600; }}

#annotation-list {{
  flex: 1; overflow-y: auto; padding: 8px;
}}
.anno-item {{
  background: #1a1f35; border: 1px solid #2a2f45; border-radius: 8px;
  padding: 10px; margin-bottom: 8px; cursor: pointer;
  transition: border-color .15s;
}}
.anno-item:hover, .anno-item.selected {{ border-color: #ff4060; }}
.anno-item .header {{
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 6px;
}}
.anno-item .header .id {{ color: #ff6080; font-weight: 700; font-size: 14px; }}
.anno-item .coords {{ font-size: 12px; color: #8090b0; }}
.anno-item .crop-row {{
  display: flex; align-items: center; gap: 10px; margin-top: 6px;
}}
.anno-item .crop-canvas {{
  border: 1px solid #3a4060; border-radius: 4px; image-rendering: pixelated;
}}
.anno-item .rgb-vals {{ font-size: 11px; color: #8090b0; font-family: monospace; }}
.anno-item .delete-btn {{
  background: none; border: none; color: #804050; cursor: pointer;
  font-size: 16px; padding: 2px 6px;
}}
.anno-item .delete-btn:hover {{ color: #ff4060; }}

/* Detection items (model) */
.det-item {{
  background: #0f1a2a; border: 1px solid #1a3050; border-radius: 8px;
  padding: 10px; margin-bottom: 6px;
}}
.det-item .header {{ display: flex; justify-content: space-between; font-size: 13px; }}
.det-item .score {{ color: #40d0e0; font-weight: 600; }}
.det-item .coords {{ font-size: 12px; color: #607090; }}

#det-section, #anno-section {{
  padding: 8px 16px 4px; font-size: 13px; color: #6080a0;
  border-bottom: 1px solid #2a2f45; font-weight: 600;
}}

#stats {{
  padding: 12px 16px; border-top: 1px solid #2a2f45;
  font-size: 12px; color: #6080a0; background: #0e1224;
}}

/* ── Canvas area ───────────────────────────────────────────────── */
#canvas-wrap {{
  flex: 1; position: relative; overflow: hidden; cursor: crosshair;
}}
#main-canvas {{
  position: absolute; top: 0; left: 0;
  image-rendering: pixelated;
}}

/* ── Zoom lens (follows mouse) ─────────────────────────────────── */
#zoom-lens {{
  position: fixed; width: 180px; height: 180px;
  border: 2px solid #40d0e0; border-radius: 8px;
  pointer-events: none; display: none; overflow: hidden;
  box-shadow: 0 4px 20px rgba(0,0,0,.6);
  z-index: 100;
}}
#zoom-canvas {{
  width: 180px; height: 180px; image-rendering: pixelated;
}}

/* ── Help overlay ──────────────────────────────────────────────── */
#help {{
  position: fixed; bottom: 16px; right: 16px;
  background: rgba(18,22,42,.92); border: 1px solid #2a2f45;
  border-radius: 8px; padding: 12px 16px; font-size: 12px;
  color: #8090b0; line-height: 1.6; z-index: 50;
}}
#help kbd {{
  background: #1e2340; border: 1px solid #3a4060; border-radius: 3px;
  padding: 1px 5px; font-family: monospace; color: #d8dae5;
}}
</style>
</head>
<body>

<div id="sidebar">
  <h2>🚢 Vessel Annotation — Bohuslän 2025-07-10</h2>
  <div id="controls">
    <button class="primary" onclick="setMode('annotate')">✚ Annotate</button>
    <button onclick="setMode('pan')">✋ Pan</button>
    <button onclick="undoLast()">↩ Undo</button>
    <button class="primary" onclick="exportJSON()">💾 Export JSON</button>
    <button class="danger" onclick="clearAll()">🗑 Clear all</button>
  </div>
  <div id="mode-indicator">Mode: <span class="mode" id="mode-label">annotate</span> · Scroll to zoom</div>

  <div id="det-section">Model detections (conf ≥ 0.1)</div>
  <div id="det-list"></div>

  <div id="anno-section">Your annotations (<span id="anno-count">0</span>)</div>
  <div id="annotation-list"></div>

  <div id="stats">
    Image: {IMG_W}×{IMG_H}px · 10m/px<br>
    Scene: Bohuslän coast · 2025-07-10<br>
    Model: yolo11s_tci · conf threshold: 0.286
  </div>
</div>

<div id="canvas-wrap">
  <canvas id="main-canvas"></canvas>
</div>

<div id="zoom-lens">
  <canvas id="zoom-canvas" width="180" height="180"></canvas>
</div>

<div id="help">
  <kbd>Click</kbd> place marker · <kbd>Z</kbd> undo · <kbd>Space</kbd> toggle pan ·
  <kbd>Scroll</kbd> zoom · <kbd>Esc</kbd> clear selection
</div>

<script>
// ── Data ──────────────────────────────────────────────────────────
const IMG_W = {IMG_W}, IMG_H = {IMG_H};
const IMG_SRC = "data:image/png;base64,{b64}";
const MODEL_DETS = {detections_json};

// ── State ─────────────────────────────────────────────────────────
let annotations = [];
let nextId = 1;
let mode = "annotate";          // "annotate" | "pan"
let selectedAnno = null;

// View transform
let scale = 1, offsetX = 0, offsetY = 0;
let dragging = false, dragStartX = 0, dragStartY = 0, dragOffX = 0, dragOffY = 0;

// ── Elements ──────────────────────────────────────────────────────
const wrap      = document.getElementById("canvas-wrap");
const canvas    = document.getElementById("main-canvas");
const ctx       = canvas.getContext("2d");
const zoomLens  = document.getElementById("zoom-lens");
const zoomCvs   = document.getElementById("zoom-canvas");
const zoomCtx   = zoomCvs.getContext("2d");
const annoList  = document.getElementById("annotation-list");
const detList   = document.getElementById("det-list");
const annoCount = document.getElementById("anno-count");
const modeLabel = document.getElementById("mode-label");

// ── Load image ────────────────────────────────────────────────────
const img = new Image();
img.onload = () => {{
  fitView();
  render();
  renderDetList();
}};
img.src = IMG_SRC;

// ── Hidden canvas for pixel readback ──────────────────────────────
const pixCvs = document.createElement("canvas");
pixCvs.width = IMG_W; pixCvs.height = IMG_H;
const pixCtx = pixCvs.getContext("2d");
img.addEventListener("load", () => {{
  pixCtx.drawImage(img, 0, 0);
}});

function getPixelRGB(x, y) {{
  const d = pixCtx.getImageData(x, y, 1, 1).data;
  return [d[0], d[1], d[2]];
}}

function getCropData(cx, cy, r) {{
  // r = half-size in pixels
  const x0 = Math.max(0, cx - r), y0 = Math.max(0, cy - r);
  const x1 = Math.min(IMG_W, cx + r), y1 = Math.min(IMG_H, cy + r);
  return pixCtx.getImageData(x0, y0, x1 - x0, y1 - y0);
}}

// ── View helpers ──────────────────────────────────────────────────
function fitView() {{
  const ww = wrap.clientWidth, wh = wrap.clientHeight;
  scale = Math.min(ww / IMG_W, wh / IMG_H) * 0.95;
  offsetX = (ww - IMG_W * scale) / 2;
  offsetY = (wh - IMG_H * scale) / 2;
}}

function screenToImg(sx, sy) {{
  return [(sx - offsetX) / scale, (sy - offsetY) / scale];
}}

function imgToScreen(ix, iy) {{
  return [ix * scale + offsetX, iy * scale + offsetY];
}}

// ── Render ────────────────────────────────────────────────────────
function render() {{
  const ww = wrap.clientWidth, wh = wrap.clientHeight;
  canvas.width = ww; canvas.height = wh;
  ctx.clearRect(0, 0, ww, wh);

  // Draw image
  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);
  ctx.imageSmoothingEnabled = scale < 3;
  ctx.drawImage(img, 0, 0);

  // Draw model detections (cyan)
  ctx.strokeStyle = "#40d0e0"; ctx.lineWidth = 2 / scale;
  for (const det of MODEL_DETS) {{
    const b = det.bbox;
    ctx.strokeRect(b.x_min, b.y_min, b.x_max - b.x_min, b.y_max - b.y_min);
    ctx.fillStyle = "rgba(64,208,224,0.15)";
    ctx.fillRect(b.x_min, b.y_min, b.x_max - b.x_min, b.y_max - b.y_min);
  }}

  // Draw annotations (red markers)
  for (const a of annotations) {{
    const sel = a.id === selectedAnno;
    const r = Math.max(6, 8 / scale);

    // Circle
    ctx.beginPath();
    ctx.arc(a.x, a.y, r, 0, Math.PI * 2);
    ctx.fillStyle = sel ? "rgba(255,64,96,0.5)" : "rgba(255,64,96,0.3)";
    ctx.fill();
    ctx.strokeStyle = sel ? "#ff2040" : "#ff4060";
    ctx.lineWidth = (sel ? 2.5 : 1.5) / scale;
    ctx.stroke();

    // Crosshair
    const cr = r * 1.6;
    ctx.beginPath();
    ctx.moveTo(a.x - cr, a.y); ctx.lineTo(a.x + cr, a.y);
    ctx.moveTo(a.x, a.y - cr); ctx.lineTo(a.x, a.y + cr);
    ctx.strokeStyle = "rgba(255,64,96,0.6)";
    ctx.lineWidth = 1 / scale;
    ctx.stroke();

    // Label
    ctx.fillStyle = "#ff4060";
    ctx.font = `bold ${{Math.max(11, 13 / scale)}}px system-ui`;
    ctx.fillText(`#${{a.id}}`, a.x + r + 3 / scale, a.y - r);
  }}

  ctx.restore();
}}

// ── Annotation list rendering ─────────────────────────────────────
function renderAnnoList() {{
  annoCount.textContent = annotations.length;
  annoList.innerHTML = "";
  for (const a of annotations) {{
    const item = document.createElement("div");
    item.className = "anno-item" + (a.id === selectedAnno ? " selected" : "");
    item.onclick = () => {{ selectedAnno = a.id; render(); renderAnnoList(); }};

    // Crop canvas
    const cropSize = 16;
    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = 64; cropCanvas.height = 64;
    cropCanvas.className = "crop-canvas";
    const cctx = cropCanvas.getContext("2d");
    cctx.imageSmoothingEnabled = false;
    const cropData = getCropData(Math.round(a.x), Math.round(a.y), cropSize);
    // Draw crop scaled up to 64x64
    const tmpCvs = document.createElement("canvas");
    tmpCvs.width = cropData.width; tmpCvs.height = cropData.height;
    tmpCvs.getContext("2d").putImageData(cropData, 0, 0);
    cctx.drawImage(tmpCvs, 0, 0, 64, 64);

    // Draw center crosshair on crop
    cctx.strokeStyle = "rgba(255,64,96,0.7)";
    cctx.lineWidth = 1;
    cctx.beginPath();
    cctx.moveTo(32, 24); cctx.lineTo(32, 40);
    cctx.moveTo(24, 32); cctx.lineTo(40, 32);
    cctx.stroke();

    const rgb = a.rgb;
    item.innerHTML = `
      <div class="header">
        <span class="id">#${{a.id}}</span>
        <button class="delete-btn" onclick="event.stopPropagation(); deleteAnno(${{a.id}})">✕</button>
      </div>
      <div class="coords">Pixel: (${{Math.round(a.x)}}, ${{Math.round(a.y)}})</div>
      <div class="crop-row">
        <span class="rgb-vals">R:${{rgb[0]}} G:${{rgb[1]}} B:${{rgb[2]}}</span>
      </div>
    `;
    // Insert crop canvas into crop-row
    const cropRow = item.querySelector(".crop-row");
    cropRow.insertBefore(cropCanvas, cropRow.firstChild);

    annoList.appendChild(item);
  }}
}}

function renderDetList() {{
  detList.innerHTML = "";
  for (let i = 0; i < MODEL_DETS.length; i++) {{
    const d = MODEL_DETS[i];
    const b = d.bbox;
    const item = document.createElement("div");
    item.className = "det-item";
    item.innerHTML = `
      <div class="header">
        <span>Detection ${{i + 1}}</span>
        <span class="score">score: ${{d.score.toFixed(3)}}</span>
      </div>
      <div class="coords">Bbox: (${{b.x_min}},${{b.y_min}}) → (${{b.x_max}},${{b.y_max}}) · ${{b.x_max-b.x_min}}×${{b.y_max-b.y_min}}px</div>
    `;
    item.style.cursor = "pointer";
    item.onclick = () => {{
      const cx = (b.x_min + b.x_max) / 2;
      const cy = (b.y_min + b.y_max) / 2;
      const [sx, sy] = imgToScreen(cx, cy);
      const ww = wrap.clientWidth, wh = wrap.clientHeight;
      offsetX += ww / 2 - sx;
      offsetY += wh / 2 - sy;
      render();
    }};
    detList.appendChild(item);
  }}
}}

// ── Annotation CRUD ───────────────────────────────────────────────
function addAnnotation(imgX, imgY) {{
  const x = Math.round(imgX), y = Math.round(imgY);
  if (x < 0 || x >= IMG_W || y < 0 || y >= IMG_H) return;
  const rgb = getPixelRGB(x, y);
  const a = {{ id: nextId++, x: imgX, y: imgY, rgb, label: "missed_vessel" }};
  annotations.push(a);
  selectedAnno = a.id;
  render();
  renderAnnoList();
}}

function deleteAnno(id) {{
  annotations = annotations.filter(a => a.id !== id);
  if (selectedAnno === id) selectedAnno = null;
  render();
  renderAnnoList();
}}

function undoLast() {{
  if (annotations.length === 0) return;
  annotations.pop();
  selectedAnno = annotations.length ? annotations[annotations.length - 1].id : null;
  render();
  renderAnnoList();
}}

function clearAll() {{
  if (annotations.length && !confirm("Clear all annotations?")) return;
  annotations = [];
  selectedAnno = null;
  nextId = 1;
  render();
  renderAnnoList();
}}

// ── Mode ──────────────────────────────────────────────────────────
function setMode(m) {{
  mode = m;
  modeLabel.textContent = m;
  wrap.style.cursor = m === "annotate" ? "crosshair" : "grab";
}}

// ── Mouse events ──────────────────────────────────────────────────
wrap.addEventListener("mousedown", (e) => {{
  if (e.button === 1 || (e.button === 0 && mode === "pan")) {{
    dragging = true;
    dragStartX = e.clientX; dragStartY = e.clientY;
    dragOffX = offsetX; dragOffY = offsetY;
    wrap.style.cursor = "grabbing";
    e.preventDefault();
    return;
  }}
  if (e.button === 0 && mode === "annotate") {{
    const rect = wrap.getBoundingClientRect();
    const [ix, iy] = screenToImg(e.clientX - rect.left, e.clientY - rect.top);
    addAnnotation(ix, iy);
  }}
}});

wrap.addEventListener("mousemove", (e) => {{
  if (dragging) {{
    offsetX = dragOffX + (e.clientX - dragStartX);
    offsetY = dragOffY + (e.clientY - dragStartY);
    render();
    return;
  }}

  // Update zoom lens
  const rect = wrap.getBoundingClientRect();
  const [ix, iy] = screenToImg(e.clientX - rect.left, e.clientY - rect.top);
  if (ix >= 0 && ix < IMG_W && iy >= 0 && iy < IMG_H) {{
    zoomLens.style.display = "block";
    zoomLens.style.left = (e.clientX + 20) + "px";
    zoomLens.style.top  = (e.clientY - 200) + "px";

    // Draw zoomed view (8x)
    const zf = 8;
    const srcR = 180 / (2 * zf);
    zoomCtx.imageSmoothingEnabled = false;
    zoomCtx.clearRect(0, 0, 180, 180);
    zoomCtx.drawImage(
      pixCvs,
      Math.max(0, ix - srcR), Math.max(0, iy - srcR), srcR * 2, srcR * 2,
      0, 0, 180, 180
    );
    // Center crosshair
    zoomCtx.strokeStyle = "rgba(255,64,96,0.6)";
    zoomCtx.lineWidth = 1;
    zoomCtx.beginPath();
    zoomCtx.moveTo(90, 75); zoomCtx.lineTo(90, 105);
    zoomCtx.moveTo(75, 90); zoomCtx.lineTo(105, 90);
    zoomCtx.stroke();
  }} else {{
    zoomLens.style.display = "none";
  }}
}});

wrap.addEventListener("mouseup", () => {{
  if (dragging) {{
    dragging = false;
    wrap.style.cursor = mode === "annotate" ? "crosshair" : "grab";
  }}
}});

wrap.addEventListener("mouseleave", () => {{
  zoomLens.style.display = "none";
}});

// Scroll to zoom
wrap.addEventListener("wheel", (e) => {{
  e.preventDefault();
  const rect = wrap.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  const zoom = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  const newScale = Math.max(0.2, Math.min(80, scale * zoom));

  // Zoom towards cursor
  offsetX = mx - (mx - offsetX) * (newScale / scale);
  offsetY = my - (my - offsetY) * (newScale / scale);
  scale = newScale;

  render();
}}, {{ passive: false }});

// ── Keyboard ──────────────────────────────────────────────────────
document.addEventListener("keydown", (e) => {{
  if (e.key === "z" || e.key === "Z") {{ undoLast(); }}
  if (e.key === "Escape") {{ selectedAnno = null; render(); renderAnnoList(); }}
  if (e.key === " ") {{ e.preventDefault(); setMode(mode === "pan" ? "annotate" : "pan"); }}
}});

// ── Resize ────────────────────────────────────────────────────────
window.addEventListener("resize", render);

// ── Export ─────────────────────────────────────────────────────────
function exportJSON() {{
  const data = {{
    image: "bohuslan_rgb.png",
    image_shape: [IMG_H, IMG_W],
    pixel_size_m: 10,
    scene_date: "2025-07-10",
    location: "Bohuslän coast, Sweden",
    coords: {{ west: 11.25049, south: 58.42763, east: 11.30049, north: 58.47763 }},
    model: "yolo11s_tci",
    model_confidence_threshold: 0.286,
    model_detections: MODEL_DETS,
    annotations: annotations.map(a => ({{
      id: a.id,
      x: Math.round(a.x),
      y: Math.round(a.y),
      rgb: a.rgb,
      label: a.label,
    }})),
  }};
  const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: "application/json" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "vessel_annotations.json";
  a.click();
  URL.revokeObjectURL(url);
}}
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"Generated: {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
