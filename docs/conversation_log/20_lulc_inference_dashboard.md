# LULC Inference Dashboard & Tile Gallery

> Per-tile LULC inference pipeline, training dashboard LULC section, NIR false-color pseudocolor, modular dashboard refactoring.

---

## LULC Inference Pipeline

### predict_lulc.py

Runs inference on all tiles in a split using the trained Prithvi-EO-2.0 + UPerNet + AuxEncoder model. For each tile, saves a `.npz` file with:

| Array | Shape | Description |
|-------|-------|-------------|
| `prediction` | (H, W) | Argmax class index (0-10) |
| `confidence` | (H, W) | Softmax max confidence (0-1) |
| `label` | (H, W) | NMD ground truth label |
| `disagree` | (H, W) | Boolean: prediction != label |
| `entropy` | (H, W) | Shannon entropy of softmax distribution |
| `s2_rgb` | (H, W, 3) | NIR false-color uint8 (B8/B3/B4) |

Also writes `prediction_summary.json` with aggregate metrics (overall accuracy, per-class accuracy, high-confidence-wrong count).

### NIR False-Color Rendering

S2 pseudocolor uses NIR false-color (not true-color RGB):
- R = B8A (NIR, band index 3)
- G = B03 (Green, band index 1)
- B = B04 (Red, band index 2)

Denormalization: `dn = normalized * std + mean`
- Mean: [1087, 1342, 1433, 2734, 1958, 1363]
- Std: [2248, 2179, 2178, 1850, 1242, 1049]

Stretch: `min=400, max=[4000, 1500, 1500]` — highlights vegetation structure for forest class differentiation.

### Results (val split, epoch 42)

| Metric | Value |
|--------|-------|
| Tiles | 949 |
| Overall accuracy | 57.3% |
| High-confidence wrong | 221,375 pixels |
| Total pixels | 47.5M |
| Best class | Water (96.8%) |
| Worst class | Pine (36.8%) |

Per-class accuracy:

| Class | Accuracy | Pixel share |
|-------|----------|-------------|
| forest_pine | 36.8% | 17.1% |
| forest_spruce | 55.7% | 10.5% |
| forest_deciduous | 41.1% | 3.3% |
| forest_mixed | 39.5% | 29.3% |
| forest_wetland | 75.9% | 4.6% |
| open_wetland | 80.9% | 11.8% |
| cropland | 93.0% | 1.5% |
| open_land | 62.2% | 9.5% |
| developed | 86.3% | 1.8% |
| water | 96.8% | 10.7% |

---

## Tile Gallery Generation

### generate_lulc_showcase.py

Picks diverse tiles from the prediction set and renders 4 images per tile:

1. **S2 pseudocolor** (NIR B8/B3/B4) — from `s2_rgb` array
2. **NMD ground truth** — 10-class palette
3. **Model prediction** — same palette
4. **Quality overlay** — green=correct, red=wrong, magenta=high-confidence wrong

Tile selection: scores tiles by `(class_diversity/10)*0.5 + disagree_ratio*0.3 + (valid_pixels/50176)*0.2`, then picks N tiles with unique dominant classes.

Output: `predictions/val/showcase/*.png` + `predictions/val/gallery.json`

### Gallery JSON Format

```json
[
  {
    "index": 0,
    "name": "tile_731280_7131280",
    "s2": "predictions/val/showcase/tile_00_s2.png",
    "nmd": "predictions/val/showcase/tile_00_nmd.png",
    "pred": "predictions/val/showcase/tile_00_pred.png",
    "quality": "predictions/val/showcase/tile_00_quality.png",
    "accuracy_pct": 26.2,
    "disagree_pct": 73.8,
    "unique_classes": 10,
    "high_conf_wrong": 746,
    "dominant_class": "Tallskog"
  }
]
```

---

## Training Dashboard — LULC Section

### HTML Section

Added to `_html_lulc_section()` in `dashboard.py`:
- Summary cards: Overall accuracy, High-conf wrong, Tiles, Disagree
- Class legend (10 LULC classes with colored swatches)
- Quality legend (correct/wrong/high-conf-wrong)
- Gallery header: S2 pseudofarg | NMD grundsanning | Modellprediktion | Kvalitet
- Per-row tile gallery with lazy-loaded images
- Per-class accuracy horizontal bar chart

### Data Fetching

The `refresh()` function polls two new endpoints:
- `predictions/val/prediction_summary.json` — summary metrics
- `predictions/val/gallery.json` — tile gallery metadata

Gallery images are served directly from the data directory via the HTTP server.

---

## Dashboard Modular Refactoring

Split the monolithic `_build_html()` function (1560-line f-string) into 14 section-builder functions:

| Function | Content |
|----------|---------|
| `_css_styles()` | Full `<style>` block |
| `_html_header()` | Page header with status badge |
| `_html_nmd_section()` | NMD pre-filter section |
| `_html_seasonal_fetch_section()` | ColonyOS seasonal fetch |
| `_html_dataprep_section()` | Spectral data fetch / preparation |
| `_html_eval_section()` | Model evaluation |
| `_html_training_section()` | Training progress |
| `_html_lulc_section()` | LULC inference gallery |
| `_html_sidebar()` | System metrics (CPU/RAM/GPU gauges) |
| `_js_constants()` | Chart.js config, class mappings |
| `_js_utils()` | Helper functions |
| `_js_init_charts()` | Chart initialization |
| `_js_update_sections()` | All update functions |
| `_js_refresh_loop()` | Fetch loop, LULC update, boot code |

`_build_html()` now just concatenates the sections.

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `predict-aux` | Run inference with AUX model on all splits |
| `predict-summary` | Quick summary only (no saved predictions) |
| `lulc-gallery` | Generate gallery images + JSON for training dashboard |
| `lulc-showcase` | Generate gallery + docs chart data for showcase |
| `lulc-showcase-placeholder` | Placeholder chart data only |

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/predict_lulc.py` | Inference script (per-tile .npz + summary) |
| `scripts/generate_lulc_showcase.py` | Gallery image generation + tile selection |
| `imint/training/dashboard.py` | Training dashboard (14 modular sections) |
| `Makefile` | `predict-aux`, `lulc-gallery`, `lulc-showcase` targets |

---

## Key Commits

| Commit | Description |
|--------|-------------|
| `fd43ed8` | feat: NIR false-color pseudocolor + multi-tile gallery for LULC showcase |
| `3a247af` | refactor: split dashboard.py into modular sections, add LULC gallery to training dashboard |
