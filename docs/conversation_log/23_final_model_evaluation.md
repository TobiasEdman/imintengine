# Final Model Evaluation Results

> Best model results: 44.14% mIoU, mean AUC-ROC 0.9334. Per-class user accuracy (precision) and producer accuracy (recall) reveal precision/recall gaps in forest classes. Inference runs 949 val tiles in ~60 seconds on M1 Max MPS. Dashboard deployed with grouped accuracy bars and ROC curves.

---

## Best Model Summary

| Metric | Value |
|--------|-------|
| **Model** | Prithvi-EO-2.0 + UPerNet + AuxEncoder (5 aux channels) |
| **Best mIoU** | 44.14% (epoch 42) |
| **Mean AUC-ROC** | 0.9334 |
| **Overall agreement** | 57.3% |
| **Checkpoint** | `checkpoints/lulc_aux/best_model.pt` |
| **Validation tiles** | 949 (from 5,801 total, lat-based split) |
| **Inference time** | ~60 seconds (MPS, M1 Max) |

---

## Per-Class Detailed Results

| Class | IoU | Producer Acc (Recall) | User Acc (Precision) | AUC-ROC | Notes |
|-------|-----|----------------------|---------------------|---------|-------|
| water | 93.4% | 96.8% | 97.1% | 0.999 | Near-perfect both ways |
| cropland | 71.3% | -- | -- | 0.998 | Strong phenology signal |
| open_wetland | 62.4% | -- | -- | 0.977 | DEM + height helps |
| open_land | 47.2% | -- | -- | -- | Diverse category |
| forest_spruce | 36.3% | -- | -- | 0.83-0.92 | Conifer discrimination |
| forest_mixed | 33.5% | 39.5% | 89.2% | 0.83-0.92 | Conservative: high precision, low recall |
| developed | 31.5% | -- | -- | 0.981 | Urban areas |
| forest_pine | 26.8% | -- | -- | 0.83-0.92 | Similar to spruce |
| forest_wetland | 20.3% | 75.9% | 23.1% | 0.83-0.92 | Massive over-prediction |
| forest_deciduous | 9.9% | 41.1% | 11.8% | 0.83-0.92 | Over-predicts deciduous (many false positives) |

### Key Precision/Recall Insights

- **forest_mixed**: 89.2% user acc but only 39.5% producer acc -- the model is **conservative**. When it predicts mixed, it's usually right, but it misses many mixed pixels.
- **forest_deciduous**: 41.1% producer but only 11.8% user -- the model **over-predicts** deciduous (many false positives).
- **forest_wetland**: 75.9% producer but only 23.1% user -- **massive over-prediction**.
- **water**: 96.8% producer, 97.1% user -- near-perfect both ways.

### AUC-ROC Analysis

Mean AUC of 0.9334 is quite good -- the model's probability rankings are strong even where hard accuracy thresholds don't look great. This means the model has learned meaningful class discriminability; the threshold-based metrics are pulled down by NMD label noise.

Top AUC scores: water (0.999), cropland (0.998), developed (0.981), open_wetland (0.977).
Forest classes: 0.83-0.92 range (still good, just harder to separate).

---

## Inference Pipeline

```bash
# Run inference (saves .npz per tile with prediction, confidence, label, disagree, entropy, s2_rgb)
make predict-aux ARGS="--splits val"

# Generate tile gallery for dashboard
make lulc-gallery
```

Each `.npz` output contains:
- `prediction` -- argmax class per pixel
- `confidence` -- max softmax probability per pixel
- `label` -- NMD ground truth
- `disagree` -- boolean mask where prediction != label
- `entropy` -- prediction uncertainty (Shannon entropy of softmax)
- `s2_rgb` -- NIR false-color (B8A/B3/B4) for visualization

The `prediction_summary.json` includes:
- Per-class `producer_accuracy_pct`, `user_accuracy_pct`, `auc`
- `mean_auc` (top-level)
- `auc_roc` with FPR/TPR curve data for 100 threshold points per class

---

## Dashboard Implementation

The training dashboard (`imint/training/dashboard.py`) was refactored from a ~1560-line f-string into 14 modular section-builder functions:

- `_css_styles()`, `_html_header()`, `_html_nmd_section()`, `_html_seasonal_fetch_section()`
- `_html_dataprep_section()`, `_html_eval_section()`, `_html_training_section()`
- `_html_lulc_section()`, `_html_sidebar()`
- `_js_constants()`, `_js_utils()`, `_js_init_charts()`, `_js_update_sections()`, `_js_refresh_loop()`

The LULC Inference section shows:
- Summary cards (accuracy, tiles, disagree count, mean AUC)
- Grouped horizontal bar chart: Producer accuracy (recall) vs User accuracy (precision) per class
- AUC-ROC curves: 10 per-class curves + diagonal reference line, AUC values in legend
- Per-tile gallery: S2 pseudocolor (B8/B3/B4) | NMD | LULC prediction | Quality overlay

Access: `http://<m1max-ip>:8000/training_dashboard.html`

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/predict_lulc.py` | Per-tile inference with AUC-ROC computation |
| `scripts/generate_lulc_showcase.py` | Gallery generation (16 diverse tiles) |
| `imint/training/dashboard.py` | Modular dashboard with LULC section |
| `checkpoints/lulc_aux/best_model.pt` | Best model checkpoint |
