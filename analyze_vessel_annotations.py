"""Diagnostic analysis of vessel detection failures: L2A vs TCI-scaled.

Fetches L1C data from DES, creates TCI product, runs YOLO detection on
both L2A and TCI-scaled, computes per-vessel diagnostics, and generates a
self-contained HTML report.

Usage:
    DES_USER=testuser DES_PASSWORD=secretpassword \
    python analyze_vessel_annotations.py --annotations vessel_annotations.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import tarfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from imint.fetch import _connect, _to_nmd_grid, OPENEO_URL
from imint.analyzers.marine_vessels import MarineVesselAnalyzer, _resolve_model_path

# ── Constants ──────────────────────────────────────────────────────────────
L1C_COLLECTION = "s2_msi_l1c"
L2A_COLLECTION = "s2_msi_l2a"
OUT_DIR = "outputs/vessel_test"
SCL_NAMES = {
    0: "no_data", 1: "saturated", 2: "dark_area",
    3: "cloud_shadow", 4: "vegetation", 5: "bare_soil",
    6: "water", 7: "unclassified", 8: "cloud_med",
    9: "cloud_high", 10: "cirrus", 11: "snow",
}


def fetch_l2a_tci(conn, projected_coords, temporal):
    """Fetch L2A B02/B03/B04 from DES and create TCI-style uint8 RGB.

    Instead of L1C (which DES doesn't serve via openEO), we fetch L2A
    raw reflectance and apply the TCI formula: clip(refl * 2.5 * 255, 0, 255).
    This produces a different scaling than the percentile stretch used in our
    standard RGB pipeline, and is closer to what the YOLO model expects.
    """
    import rasterio

    print("  Fetching L2A B02/B03/B04 (raw reflectance)...")
    cube = conn.load_collection(
        collection_id=L2A_COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=["b02", "b03", "b04"],
    )
    cube = cube.reduce_dimension(dimension="t", reducer="first")
    data = cube.download(format="gtiff")

    if not data:
        raise RuntimeError("DES returned empty data")

    # Parse tar.gz or raw GeoTIFF
    if isinstance(data, bytes) and data[:2] == b"\x1f\x8b":
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        tif_data = None
        for member in tf.getmembers():
            if member.name.lower().endswith((".tif", ".tiff")):
                f = tf.extractfile(member)
                if f:
                    tif_data = f.read()
                    break
        tf.close()
        if tif_data is None:
            raise RuntimeError("No GeoTIFF in archive")
    else:
        tif_data = data

    with rasterio.open(io.BytesIO(tif_data)) as src:
        bands = src.read()  # (3, H, W)
        nodata = src.nodata

    print(f"  Raw shape: {bands.shape}, dtype: {bands.dtype}")
    print(f"  DN range: [{bands[bands != nodata].min():.0f}, {bands[bands != nodata].max():.0f}]" if nodata else f"  DN range: [{bands.min()}, {bands.max()}]")

    # DES offset: reflectance = (DN - 1000) / 10000
    refl = (bands.astype(np.float32) - 1000.0) / 10000.0
    if nodata is not None:
        for b in range(refl.shape[0]):
            refl[b][bands[b] == nodata] = 0.0
    refl = np.clip(refl, 0, 1)

    print(f"  Reflectance range: [{refl.min():.4f}, {refl.max():.4f}], mean={refl.mean():.4f}")

    # TCI formula: clip(reflectance * 2.5 * 255, 0, 255)
    # Band order from DES: [b02, b03, b04] → RGB = [b04, b03, b02]
    refl_rgb = np.stack([refl[2], refl[1], refl[0]], axis=-1)  # (H, W, 3) as R,G,B
    tci = np.clip(refl_rgb * 2.5 * 255, 0, 255).astype(np.uint8)
    print(f"  TCI shape: {tci.shape}, range: [{tci.min()}, {tci.max()}], mean={tci.mean():.0f}")

    return tci, refl_rgb


def fetch_scl(conn, projected_coords, temporal):
    """Fetch SCL band from L2A."""
    import rasterio

    print("  Fetching SCL...")
    cube_ref = conn.load_collection(
        collection_id=L2A_COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=["b02"],
    )
    cube_scl = conn.load_collection(
        collection_id=L2A_COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=["scl"],
    )
    cube_scl = cube_scl.resample_cube_spatial(target=cube_ref, method="near")
    cube_scl = cube_scl.reduce_dimension(dimension="t", reducer="first")
    data = cube_scl.download(format="gtiff")

    if isinstance(data, bytes) and data[:2] == b"\x1f\x8b":
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        tif_data = None
        for member in tf.getmembers():
            if member.name.lower().endswith((".tif", ".tiff")):
                f = tf.extractfile(member)
                if f:
                    tif_data = f.read()
                    break
        tf.close()
    else:
        tif_data = data

    with rasterio.open(io.BytesIO(tif_data)) as src:
        scl = src.read()[0].astype(np.uint8)

    print(f"  SCL shape: {scl.shape}")
    return scl


def compute_local_contrast(img_uint8, x, y, radius=8):
    """Weber contrast: vessel vs surrounding background."""
    h, w = img_uint8.shape[:2]
    x, y = int(round(x)), int(round(y))

    # Vessel: 3x3 patch mean brightness
    vy0, vx0 = max(0, y - 1), max(0, x - 1)
    vy1, vx1 = min(h, y + 2), min(w, x + 2)
    vessel_brightness = float(np.mean(img_uint8[vy0:vy1, vx0:vx1]))

    # Background: ring around vessel
    ry0, rx0 = max(0, y - radius), max(0, x - radius)
    ry1, rx1 = min(h, y + radius), min(w, x + radius)
    bg_patch = img_uint8[ry0:ry1, rx0:rx1].astype(np.float32)

    mask = np.ones(bg_patch.shape[:2], dtype=bool)
    cy_local, cx_local = y - ry0, x - rx0
    mask[max(0, cy_local - 1):cy_local + 2,
         max(0, cx_local - 1):cx_local + 2] = False

    bg_mean = float(np.mean(bg_patch[mask])) if mask.any() else 128.0
    weber = (vessel_brightness - bg_mean) / max(bg_mean, 1.0)

    return {
        "vessel_brightness": round(vessel_brightness, 1),
        "bg_mean": round(bg_mean, 1),
        "weber_contrast": round(weber, 3),
    }


def yolo_centered_score(img_uint8, x, y, model, chip_size=320):
    """Run YOLO on a centered crop, return best score near center."""
    h, w = img_uint8.shape[:2]
    x, y = int(round(x)), int(round(y))
    half = chip_size // 2

    chip = np.zeros((chip_size, chip_size, 3), dtype=np.uint8)
    y0, x0 = y - half, x - half

    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(h, y0 + chip_size), min(w, x0 + chip_size)
    dy0, dx0 = sy0 - y0, sx0 - x0
    dy1, dx1 = dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)

    chip[dy0:dy1, dx0:dx1] = img_uint8[sy0:sy1, sx0:sx1]

    results = model.predict(chip, conf=0.001, verbose=False)

    best_score = 0.0
    cx, cy_c = chip_size // 2, chip_size // 2
    for result in results:
        for box in result.boxes:
            bx0, by0, bx1, by1 = box.xyxy[0].cpu().numpy()
            bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
            dist = np.sqrt((bcx - cx) ** 2 + (bcy - cy_c) ** 2)
            if dist < 40 and float(box.conf) > best_score:
                best_score = float(box.conf)

    return round(best_score, 4)


def compute_sahi_chips(img_h, img_w, chip_size=320, overlap=0.2):
    """Replicate SAHI chip grid."""
    stride = int(chip_size * (1 - overlap))
    chips = []
    y = 0
    while True:
        if y + chip_size > img_h:
            y = max(0, img_h - chip_size)
        x = 0
        while True:
            if x + chip_size > img_w:
                x = max(0, img_w - chip_size)
            chips.append((x, y, min(x + chip_size, img_w), min(y + chip_size, img_h)))
            if x + chip_size >= img_w:
                break
            x += stride
        if y + chip_size >= img_h:
            break
        y += stride
    return chips


def chip_edge_distance(vx, vy, chips, chip_size=320):
    """Min distance from vessel to nearest chip edge (across all containing chips)."""
    min_dist = chip_size
    for cx0, cy0, cx1, cy1 in chips:
        if cx0 <= vx < cx1 and cy0 <= vy < cy1:
            dist = min(vx - cx0, cx1 - vx, vy - cy0, cy1 - vy)
            min_dist = min(min_dist, dist)
    return int(min_dist)


def run_detection(img_uint8, scl, confidence):
    """Run MarineVesselAnalyzer and return regions."""
    analyzer = MarineVesselAnalyzer(config={
        "confidence": confidence,
        "chip_size": 320,
        "overlap_ratio": 0.2,
        "water_filter": True,
    })
    result = analyzer.analyze(img_uint8, scl=scl)
    if result.success:
        return result.outputs.get("regions", []), result.metadata
    return [], {}


def img_to_b64(img_uint8):
    """Convert uint8 array to base64 PNG string."""
    buf = io.BytesIO()
    Image.fromarray(img_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def crop_b64(img_uint8, x, y, radius=16, scale=4):
    """Extract a crop and return base64 PNG."""
    h, w = img_uint8.shape[:2]
    x, y = int(round(x)), int(round(y))
    y0, x0 = max(0, y - radius), max(0, x - radius)
    y1, x1 = min(h, y + radius), min(w, x + radius)
    crop = img_uint8[y0:y1, x0:x1]
    pil = Image.fromarray(crop).resize(
        (crop.shape[1] * scale, crop.shape[0] * scale), Image.NEAREST
    )
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate_html_report(
    l2a_img, l1c_tci, scl, annotations, model_dets,
    l2a_regions_286, l1c_regions_286, l2a_regions_01, l1c_regions_01,
    vessel_diags, scene_meta, output_path,
):
    """Generate the self-contained HTML diagnostic report."""

    l2a_b64 = img_to_b64(l2a_img)
    l1c_b64 = img_to_b64(l1c_tci)
    img_h, img_w = l2a_img.shape[:2]

    # Summary stats
    n_annot = len(annotations)
    n_l2a_286 = len(l2a_regions_286)
    n_l1c_286 = len(l1c_regions_286)
    n_l2a_01 = len(l2a_regions_01)
    n_l1c_01 = len(l1c_regions_01)

    # Per-vessel data for charts
    vessel_ids = [d["id"] for d in vessel_diags]
    l2a_brightness = [d["l2a_contrast"]["vessel_brightness"] for d in vessel_diags]
    l1c_brightness = [d["l1c_contrast"]["vessel_brightness"] for d in vessel_diags]
    l2a_weber = [d["l2a_contrast"]["weber_contrast"] for d in vessel_diags]
    l1c_weber = [d["l1c_contrast"]["weber_contrast"] for d in vessel_diags]
    l2a_yolo = [d["l2a_yolo_score"] for d in vessel_diags]
    l1c_yolo = [d["l1c_yolo_score"] for d in vessel_diags]
    scl_classes = [d["scl_class"] for d in vessel_diags]
    edge_dists = [d["chip_edge_dist"] for d in vessel_diags]

    # Build vessel cards HTML
    cards_html = ""
    for d in vessel_diags:
        scl_name = SCL_NAMES.get(d["scl_class"], "unknown")
        scl_color = "#40d0e0" if d["scl_class"] == 6 else "#ff4060"
        l2a_score_color = "#40d0e0" if d["l2a_yolo_score"] >= 0.286 else ("#e0a020" if d["l2a_yolo_score"] >= 0.1 else "#ff4060")
        l1c_score_color = "#40d0e0" if d["l1c_yolo_score"] >= 0.286 else ("#e0a020" if d["l1c_yolo_score"] >= 0.1 else "#ff4060")

        cards_html += f"""
        <div class="vessel-card">
          <div class="card-header">
            <span class="vessel-id">#{d['id']}</span>
            <span class="vessel-coords">({d['x']}, {d['y']})</span>
            <span class="scl-badge" style="color:{scl_color}">SCL: {d['scl_class']} ({scl_name})</span>
          </div>
          <div class="card-crops">
            <div class="crop-col">
              <div class="crop-label">L2A</div>
              <img src="data:image/png;base64,{d['l2a_crop_b64']}" class="crop-img">
              <div class="crop-stats">
                Bright: {d['l2a_contrast']['vessel_brightness']:.0f}<br>
                Weber: {d['l2a_contrast']['weber_contrast']:.3f}<br>
                <span style="color:{l2a_score_color}">YOLO: {d['l2a_yolo_score']:.4f}</span>
              </div>
            </div>
            <div class="crop-col">
              <div class="crop-label">TCI-scaled</div>
              <img src="data:image/png;base64,{d['l1c_crop_b64']}" class="crop-img">
              <div class="crop-stats">
                Bright: {d['l1c_contrast']['vessel_brightness']:.0f}<br>
                Weber: {d['l1c_contrast']['weber_contrast']:.3f}<br>
                <span style="color:{l1c_score_color}">YOLO: {d['l1c_yolo_score']:.4f}</span>
              </div>
            </div>
          </div>
          <div class="card-meta">Edge dist: {d['chip_edge_dist']}px · RGB(L2A): {d['rgb_l2a']} · RGB(L1C): {d['rgb_l1c']}</div>
        </div>"""

    # Detection overlay for side-by-side
    def draw_overlay_b64(img, regions_286, regions_01, annots):
        """Draw detections and annotations on image."""
        from PIL import ImageDraw
        pil = Image.fromarray(img.copy())
        draw = ImageDraw.Draw(pil)
        # conf>=0.286 detections in cyan
        for r in regions_286:
            b = r["bbox"]
            draw.rectangle([b["x_min"], b["y_min"], b["x_max"], b["y_max"]],
                           outline="#40d0e0", width=2)
        # conf>=0.1 (not in 0.286) in yellow
        bboxes_286 = {(r["bbox"]["x_min"], r["bbox"]["y_min"]) for r in regions_286}
        for r in regions_01:
            b = r["bbox"]
            if (b["x_min"], b["y_min"]) not in bboxes_286:
                draw.rectangle([b["x_min"], b["y_min"], b["x_max"], b["y_max"]],
                               outline="#e0a020", width=1)
        # annotations as red circles
        for a in annots:
            x, y = int(a["x"]), int(a["y"])
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline="#ff4060", width=2)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    l2a_overlay_b64 = draw_overlay_b64(l2a_img, l2a_regions_286, l2a_regions_01, annotations)
    l1c_overlay_b64 = draw_overlay_b64(l1c_tci, l1c_regions_286, l1c_regions_01, annotations)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Vessel Detection Diagnostic — Percentile-stretch vs TCI-formula</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0b0e17; color: #d8dae5; font-family: system-ui, sans-serif; padding: 20px; }}
h1 {{ font-size: 22px; margin-bottom: 8px; }}
h2 {{ font-size: 17px; margin: 24px 0 12px; padding-bottom: 6px; border-bottom: 1px solid #2a2f45; }}
h3 {{ font-size: 14px; margin: 16px 0 8px; color: #8090b0; }}

.summary-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px; margin: 16px 0;
}}
.stat-card {{
  background: #12162a; border: 1px solid #2a2f45; border-radius: 8px;
  padding: 16px; text-align: center;
}}
.stat-card .value {{ font-size: 28px; font-weight: 700; }}
.stat-card .label {{ font-size: 12px; color: #6080a0; margin-top: 4px; }}
.stat-card.green .value {{ color: #40d080; }}
.stat-card.red .value {{ color: #ff4060; }}
.stat-card.cyan .value {{ color: #40d0e0; }}
.stat-card.yellow .value {{ color: #e0a020; }}

.comparison {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0;
}}
.comparison img {{
  width: 100%; border-radius: 6px; border: 1px solid #2a2f45;
  image-rendering: pixelated;
}}
.comparison .img-label {{
  text-align: center; font-size: 13px; color: #8090b0; margin-top: 4px;
}}

.chart-container {{ background: #12162a; border-radius: 8px; padding: 16px; margin: 16px 0; }}
.chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
canvas {{ max-height: 300px; }}

.vessel-cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }}
.vessel-card {{
  background: #12162a; border: 1px solid #2a2f45; border-radius: 8px; padding: 12px;
}}
.card-header {{ display: flex; gap: 10px; align-items: center; margin-bottom: 8px; }}
.vessel-id {{ color: #ff6080; font-weight: 700; font-size: 16px; }}
.vessel-coords {{ color: #6080a0; font-size: 12px; }}
.scl-badge {{ font-size: 11px; font-weight: 600; margin-left: auto; }}
.card-crops {{ display: flex; gap: 12px; }}
.crop-col {{ text-align: center; }}
.crop-label {{ font-size: 11px; color: #8090b0; margin-bottom: 4px; font-weight: 600; }}
.crop-img {{ width: 128px; height: 128px; image-rendering: pixelated; border-radius: 4px; border: 1px solid #2a2f45; }}
.crop-stats {{ font-size: 11px; font-family: monospace; color: #8090b0; margin-top: 4px; line-height: 1.5; }}
.card-meta {{ font-size: 10px; color: #505868; margin-top: 6px; }}

.conclusions {{
  background: #12162a; border: 1px solid #2a2f45; border-radius: 8px;
  padding: 20px; margin: 16px 0;
}}
.conclusions li {{ margin: 6px 0; line-height: 1.5; }}

.legend {{ display: flex; gap: 20px; font-size: 12px; color: #8090b0; margin: 8px 0; }}
.legend span {{ display: inline-flex; align-items: center; gap: 4px; }}
.legend .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
</style>
</head>
<body>

<h1>Vessel Detection Diagnostic Report</h1>
<p style="color:#6080a0; font-size:13px; margin-bottom:4px;">
  Scene: {scene_meta.get('location', 'Bohuslan')} · {scene_meta.get('date', '2025-07-10')} ·
  {img_w}×{img_h}px · 10m/px · Model: yolo11s_tci
</p>

<h2>1. Detection Summary</h2>
<div class="summary-grid">
  <div class="stat-card red"><div class="value">{n_annot}</div><div class="label">Annotated vessels (missed)</div></div>
  <div class="stat-card red"><div class="value">{n_l2a_286}</div><div class="label">L2A detections (conf≥0.286)</div></div>
  <div class="stat-card cyan"><div class="value">{n_l1c_286}</div><div class="label">TCI-scaled detections (conf≥0.286)</div></div>
  <div class="stat-card yellow"><div class="value">{n_l2a_01}</div><div class="label">L2A detections (conf≥0.1)</div></div>
  <div class="stat-card green"><div class="value">{n_l1c_01}</div><div class="label">TCI-scaled detections (conf≥0.1)</div></div>
</div>

<h2>2. L2A vs TCI-scaled Comparison</h2>
<p style="font-size:13px; color:#8090b0;">
  Red circles = annotated vessels · Cyan boxes = detections (conf≥0.286) · Yellow boxes = detections (0.1≤conf&lt;0.286)
</p>
<div class="comparison">
  <div>
    <img src="data:image/png;base64,{l2a_overlay_b64}">
    <div class="img-label">L2A percentile-stretch — {n_l2a_286} detections</div>
  </div>
  <div>
    <img src="data:image/png;base64,{l1c_overlay_b64}">
    <div class="img-label">L2A TCI-formula (refl×2.5×255) — {n_l1c_286} detections</div>
  </div>
</div>

<h2>3. Brightness & Contrast Analysis</h2>
<div class="chart-row">
  <div class="chart-container">
    <h3>Vessel brightness (L2A vs TCI-scaled)</h3>
    <canvas id="brightnessChart"></canvas>
  </div>
  <div class="chart-container">
    <h3>Weber contrast (L2A vs TCI-scaled)</h3>
    <canvas id="contrastChart"></canvas>
  </div>
</div>

<h2>4. YOLO Centered Score per Vessel</h2>
<div class="chart-container">
  <canvas id="yoloChart"></canvas>
  <div class="legend">
    <span><span class="dot" style="background:#ff4060"></span> Below 0.1 (invisible)</span>
    <span><span class="dot" style="background:#e0a020"></span> 0.1–0.286 (marginal)</span>
    <span><span class="dot" style="background:#40d0e0"></span> ≥0.286 (detected)</span>
  </div>
</div>

<h2>5. SCL & Chip Edge Analysis</h2>
<div class="chart-row">
  <div class="chart-container">
    <h3>SCL class at vessel locations</h3>
    <canvas id="sclChart"></canvas>
  </div>
  <div class="chart-container">
    <h3>Distance to nearest SAHI chip edge</h3>
    <canvas id="edgeChart"></canvas>
  </div>
</div>

<h2>6. Per-Vessel Diagnostic Cards</h2>
<div class="vessel-cards">
  {cards_html}
</div>

<h2>7. Conclusions</h2>
<div class="conclusions">
  <ul>
    <li><strong>TCI-scaled vs L2A:</strong> TCI-scaled yields <strong>{n_l1c_286}</strong> detections at conf≥0.286
        vs <strong>{n_l2a_286}</strong> for L2A — {"confirming the radiometry mismatch as a major factor" if n_l1c_286 > n_l2a_286 else "suggesting radiometry is not the only factor"}.</li>
    <li><strong>Brightness:</strong> Mean vessel brightness is {np.mean(l2a_brightness):.0f} (L2A) vs {np.mean(l1c_brightness):.0f} (TCI-scaled).
        TCI-scaled is {"brighter" if np.mean(l1c_brightness) > np.mean(l2a_brightness) else "similar"}, giving the model more signal.</li>
    <li><strong>SCL filtering:</strong> {sum(1 for s in scl_classes if s != 6)}/{n_annot} vessels are on non-water SCL pixels
        — these would be filtered out even if detected.</li>
    <li><strong>Sub-pixel vessels:</strong> At 10m resolution, small leisure boats (5-8m) are sub-pixel.
        {sum(1 for s in l1c_yolo if s < 0.01)}/{n_annot} vessels have zero YOLO score even on TCI-scaled centered crops,
        meaning they are fundamentally invisible to this model.</li>
    <li><strong>SAHI chip edges:</strong> Mean edge distance is {np.mean(edge_dists):.0f}px.
        {sum(1 for d in edge_dists if d < 10)}/{n_annot} vessels are within 10px of a chip edge.</li>
  </ul>
</div>

<script>
const IDS = {json.dumps(vessel_ids)};
const L2A_BRIGHT = {json.dumps(l2a_brightness)};
const TCI_BRIGHT = {json.dumps(l1c_brightness)};
const L2A_WEBER = {json.dumps(l2a_weber)};
const TCI_WEBER = {json.dumps(l1c_weber)};
const L2A_YOLO = {json.dumps(l2a_yolo)};
const TCI_YOLO = {json.dumps(l1c_yolo)};
const SCL = {json.dumps(scl_classes)};
const EDGE = {json.dumps(edge_dists)};

Chart.defaults.color = '#8090b0';
Chart.defaults.borderColor = '#2a2f45';

// Brightness chart
new Chart(document.getElementById('brightnessChart'), {{
  type: 'bar',
  data: {{
    labels: IDS.map(i => '#' + i),
    datasets: [
      {{ label: 'L2A', data: L2A_BRIGHT, backgroundColor: '#ff406080', borderColor: '#ff4060', borderWidth: 1 }},
      {{ label: 'TCI-scaled', data: TCI_BRIGHT, backgroundColor: '#40d0e080', borderColor: '#40d0e0', borderWidth: 1 }},
    ]
  }},
  options: {{ scales: {{ y: {{ title: {{ display: true, text: 'Mean brightness (0-255)' }} }} }}, plugins: {{ legend: {{ position: 'top' }} }} }}
}});

// Contrast chart
new Chart(document.getElementById('contrastChart'), {{
  type: 'bar',
  data: {{
    labels: IDS.map(i => '#' + i),
    datasets: [
      {{ label: 'L2A', data: L2A_WEBER, backgroundColor: '#ff406080', borderColor: '#ff4060', borderWidth: 1 }},
      {{ label: 'TCI-scaled', data: TCI_WEBER, backgroundColor: '#40d0e080', borderColor: '#40d0e0', borderWidth: 1 }},
    ]
  }},
  options: {{ scales: {{ y: {{ title: {{ display: true, text: 'Weber contrast' }} }} }}, plugins: {{ legend: {{ position: 'top' }} }} }}
}});

// YOLO scores chart
new Chart(document.getElementById('yoloChart'), {{
  type: 'bar',
  data: {{
    labels: IDS.map(i => '#' + i),
    datasets: [
      {{ label: 'L2A', data: L2A_YOLO, backgroundColor: '#ff406080', borderColor: '#ff4060', borderWidth: 1 }},
      {{ label: 'TCI-scaled', data: TCI_YOLO, backgroundColor: '#40d0e080', borderColor: '#40d0e0', borderWidth: 1 }},
    ]
  }},
  options: {{
    scales: {{ y: {{ title: {{ display: true, text: 'YOLO confidence score' }}, max: 1 }} }},
    plugins: {{ legend: {{ position: 'top' }},
      annotation: {{ annotations: {{
        line1: {{ type: 'line', yMin: 0.286, yMax: 0.286, borderColor: '#40d0e060', borderDash: [5,5], label: {{ content: 'threshold', display: true }} }}
      }} }}
    }}
  }}
}});

// SCL chart
const sclCounts = {{}};
SCL.forEach(s => {{ sclCounts[s] = (sclCounts[s] || 0) + 1; }});
const sclLabels = Object.keys(sclCounts).sort((a,b) => a-b);
const SCL_NAMES = {json.dumps(SCL_NAMES)};
new Chart(document.getElementById('sclChart'), {{
  type: 'bar',
  data: {{
    labels: sclLabels.map(s => s + ' (' + (SCL_NAMES[s] || '?') + ')'),
    datasets: [{{ data: sclLabels.map(s => sclCounts[s]),
      backgroundColor: sclLabels.map(s => s === '6' ? '#40d0e080' : '#ff406080'),
      borderColor: sclLabels.map(s => s === '6' ? '#40d0e0' : '#ff4060'),
      borderWidth: 1 }}]
  }},
  options: {{ plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ title: {{ display: true, text: 'Vessel count' }} }} }} }}
}});

// Edge distance chart
const edgeBins = [0, 10, 20, 40, 80, 160, 320];
const edgeCounts = new Array(edgeBins.length - 1).fill(0);
EDGE.forEach(d => {{
  for (let i = 0; i < edgeBins.length - 1; i++) {{
    if (d >= edgeBins[i] && d < edgeBins[i+1]) {{ edgeCounts[i]++; break; }}
  }}
}});
new Chart(document.getElementById('edgeChart'), {{
  type: 'bar',
  data: {{
    labels: edgeBins.slice(0, -1).map((b, i) => b + '-' + edgeBins[i+1] + 'px'),
    datasets: [{{ data: edgeCounts, backgroundColor: '#6080a080', borderColor: '#6080a0', borderWidth: 1 }}]
  }},
  options: {{ plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ title: {{ display: true, text: 'Vessel count' }} }} }} }}
}});
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"  Report: {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Vessel detection diagnostic analysis")
    parser.add_argument("--annotations", required=True, help="Path to vessel_annotations.json")
    parser.add_argument("--output-dir", default=OUT_DIR)
    args = parser.parse_args()

    with open(args.annotations) as f:
        annot_data = json.load(f)

    annotations = annot_data["annotations"]
    model_dets = annot_data.get("model_detections", [])
    coords = annot_data["coords"]
    scene_date = annot_data.get("scene_date", "2025-07-10")

    print(f"Loaded {len(annotations)} annotations for {scene_date}")

    # Load L2A image
    l2a_path = os.path.join(args.output_dir, "bohuslan_rgb.png")
    l2a_img = np.array(Image.open(l2a_path))
    print(f"L2A image: {l2a_img.shape}")

    # Connect to DES
    from datetime import datetime, timedelta
    dt = datetime.strptime(scene_date, "%Y-%m-%d")
    temporal = [scene_date, (dt + timedelta(days=1)).strftime("%Y-%m-%d")]
    projected = _to_nmd_grid(coords)

    conn = _connect()

    # Fetch L2A raw reflectance and create TCI-style image
    print("\n[1/5] Fetching L2A raw reflectance for TCI creation...")
    l1c_tci, toa_rgb = fetch_l2a_tci(conn, projected, temporal)

    # Check size match — L1C might have slightly different dimensions
    if l1c_tci.shape[:2] != l2a_img.shape[:2]:
        print(f"  Size mismatch: L2A={l2a_img.shape[:2]}, L1C={l1c_tci.shape[:2]}")
        # Resize L1C to match L2A
        l1c_pil = Image.fromarray(l1c_tci).resize(
            (l2a_img.shape[1], l2a_img.shape[0]), Image.NEAREST
        )
        l1c_tci = np.array(l1c_pil)
        print(f"  Resized L1C to {l1c_tci.shape}")

    # Save L1C TCI
    l1c_path = os.path.join(args.output_dir, "bohuslan_tci_scaled.png")
    Image.fromarray(l1c_tci).save(l1c_path)
    print(f"  Saved: {l1c_path}")

    # Fetch SCL
    print("\n[2/5] Fetching SCL...")
    scl_cache = os.path.join(args.output_dir, "bohuslan_scl.npy")
    if os.path.exists(scl_cache):
        scl = np.load(scl_cache)
        print(f"  Loaded cached SCL: {scl.shape}")
    else:
        scl = fetch_scl(conn, projected, temporal)
        np.save(scl_cache, scl)
        print(f"  Cached SCL to {scl_cache}")

    # Resize SCL if needed
    if scl.shape != l2a_img.shape[:2]:
        scl_pil = Image.fromarray(scl).resize(
            (l2a_img.shape[1], l2a_img.shape[0]), Image.NEAREST
        )
        scl = np.array(scl_pil)

    # Run detections
    print("\n[3/5] Running YOLO detection on both images...")
    print("  L2A conf=0.286...")
    l2a_regions_286, l2a_meta_286 = run_detection(l2a_img, scl, 0.286)
    print(f"  → {len(l2a_regions_286)} detections")

    print("  L2A conf=0.1...")
    l2a_regions_01, _ = run_detection(l2a_img, scl, 0.1)
    print(f"  → {len(l2a_regions_01)} detections")

    print("  TCI-scaled conf=0.286...")
    l1c_regions_286, _ = run_detection(l1c_tci, scl, 0.286)
    print(f"  → {len(l1c_regions_286)} detections")

    print("  TCI-scaled conf=0.1...")
    l1c_regions_01, _ = run_detection(l1c_tci, scl, 0.1)
    print(f"  → {len(l1c_regions_01)} detections")

    # Per-vessel diagnostics
    print("\n[4/5] Computing per-vessel diagnostics...")
    from ultralytics import YOLO
    model_path = _resolve_model_path(None)
    yolo_model = YOLO(model_path)

    chips = compute_sahi_chips(l2a_img.shape[0], l2a_img.shape[1])
    print(f"  SAHI grid: {len(chips)} chips for {l2a_img.shape[1]}x{l2a_img.shape[0]}")

    vessel_diags = []
    for a in annotations:
        x, y = a["x"], a["y"]
        aid = a["id"]

        l2a_contrast = compute_local_contrast(l2a_img, x, y)
        l1c_contrast = compute_local_contrast(l1c_tci, x, y)

        scl_val = int(scl[int(round(y)), int(round(x))]) if (
            0 <= int(round(y)) < scl.shape[0] and 0 <= int(round(x)) < scl.shape[1]
        ) else -1

        edge_dist = chip_edge_distance(int(round(x)), int(round(y)), chips)

        l2a_score = yolo_centered_score(l2a_img, x, y, yolo_model)
        l1c_score = yolo_centered_score(l1c_tci, x, y, yolo_model)

        l2a_crop = crop_b64(l2a_img, x, y)
        l1c_crop = crop_b64(l1c_tci, x, y)

        ix, iy = int(round(x)), int(round(y))
        rgb_l2a = [int(v) for v in l2a_img[iy, ix]] if 0 <= iy < l2a_img.shape[0] and 0 <= ix < l2a_img.shape[1] else [0, 0, 0]
        rgb_l1c = [int(v) for v in l1c_tci[iy, ix]] if 0 <= iy < l1c_tci.shape[0] and 0 <= ix < l1c_tci.shape[1] else [0, 0, 0]

        diag = {
            "id": aid, "x": ix, "y": iy,
            "l2a_contrast": l2a_contrast,
            "l1c_contrast": l1c_contrast,
            "scl_class": scl_val,
            "chip_edge_dist": edge_dist,
            "l2a_yolo_score": l2a_score,
            "l1c_yolo_score": l1c_score,
            "l2a_crop_b64": l2a_crop,
            "l1c_crop_b64": l1c_crop,
            "rgb_l2a": rgb_l2a,
            "rgb_l1c": rgb_l1c,
        }
        vessel_diags.append(diag)
        print(f"  #{aid}: L2A={l2a_score:.4f} L1C={l1c_score:.4f} SCL={scl_val} edge={edge_dist}px")

    # Save raw diagnostics
    diag_json_path = os.path.join(args.output_dir, "vessel_diagnostics.json")
    diag_export = [{k: v for k, v in d.items() if not k.endswith("_b64")} for d in vessel_diags]
    with open(diag_json_path, "w") as f:
        json.dump(diag_export, f, indent=2)
    print(f"\n  Raw diagnostics: {diag_json_path}")

    # Generate HTML report
    print("\n[5/5] Generating HTML report...")
    scene_meta = {
        "location": annot_data.get("location", "Bohuslan coast"),
        "date": scene_date,
    }
    report_path = os.path.join(args.output_dir, "vessel_diagnostic_report.html")
    generate_html_report(
        l2a_img, l1c_tci, scl, annotations, model_dets,
        l2a_regions_286, l1c_regions_286, l2a_regions_01, l1c_regions_01,
        vessel_diags, scene_meta, report_path,
    )

    print("\nDone!")
    print(f"  L2A detections:     {len(l2a_regions_286)} (conf≥0.286), {len(l2a_regions_01)} (conf≥0.1)")
    print(f"  TCI-scaled detections: {len(l1c_regions_286)} (conf≥0.286), {len(l1c_regions_01)} (conf≥0.1)")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
