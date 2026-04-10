# Training Results

## Baselines (M1 Max, 10-class LULC schema)

| Run | Epochs | Best val mIoU | Checkpoint | Notes |
|-----|--------|--------------|------------|-------|
| `lulc_full_10class_baseline` | 50 | **44.14%** (ep 42) | `checkpoints/lulc_seasonal_v2/best_model.pt` (1.5 GB) | Dense SegFormer, full dataset, 10 classes. Reference baseline. |
| `lulc_seasonal_single_temporal` | 50 | 33.58% (ep 50) | — | Single temporal frame only, seasonal dataset |

### lulc_full baseline — per-class IoU at best epoch (42)
| Class | IoU |
|-------|-----|
| water | 93.6% |
| cropland | 68.3% |
| open_wetland | 62.5% |
| open_land | 47.3% |
| forest_mixed | 37.7% |
| forest_spruce | 36.7% |
| developed | 34.0% |
| forest_pine | 29.5% |
| forest_wetland | 21.5% |
| forest_deciduous | 10.1% ← hardest class |

**Architecture**: SegFormer-style dense segmentation, 10-class LULC schema, 4 seasonal frames (no 2016 background).

---

## Flawed / Invalid Runs

See `flawed/` — all runs between the 44% baseline and the current train-pixel-v1 are quarantined
there. Do not use those results. See `flawed/README.md` for details, including a list of M1 Max
checkpoints to mark when back on the same network.

---

## Current / In-Progress (ICE H100)

| Run | Config | Status |
|-----|--------|--------|
| `train-pixel-v1` | PrithviPixelClassifier, 23-class, T=5, n_aux=11, 35 epochs, batch 512 | **Running** (started 2026-04-10) |

Expected to beat baseline by adding: 2016 background frame (T=5), auxiliary channels (height/volume/DEM/VPP), 23-class unified schema, and center-pixel context classification approach.
