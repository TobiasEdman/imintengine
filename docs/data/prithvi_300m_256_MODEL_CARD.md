# Prithvi-EO-2.0 300M — ImintEngine unified v5 (256 px) · baseline

> ⚠️ **Baseline checkpoint.** Trained for **10 epochs** to **val mIoU
> 0.4716** on the 23-class schema. Useful as a fine-tuning seed or
> reproduction baseline — **not** a production model.

## Files

| file | size | notes |
|---|---|---|
| `prithvi_300m_256_best.pt` | 1.3 GB | best checkpoint (`best_model.pt`, epoch with highest val mIoU) |
| `training_log.json` | — | full per-epoch log (loss, val mIoU, per-class IoU, confusion matrix) |

## Architecture

- **Backbone:** `prithvi_eo_v2_300m_tl` (Prithvi-EO-2.0, 300M)
- **Temporal:** `num_temporal_frames = 4`
- **Input:** `(4×6 spectral + 11 aux, 256, 256)` = 35 channels
  - 6 bands per frame: **B02, B03, B04, B8A, B11, B12** (raw reflectance,
    ~[0,0.4] — do **not** percentile-stretch model input)
  - 11 aux: height, volume, basal_area, diameter, dem, vpp_sosd, vpp_eosd,
    vpp_length, vpp_maxv, vpp_minv, harvest_probability
- **Heads (dual):**
  1. LULC — 23-class softmax segmentation (focal loss)
  2. Harvest-readiness — binary sigmoid map (BCE loss)

## Training

| | |
|---|---|
| Dataset | unified_v2 (256 px), this release |
| Epochs | 10 |
| Batch size | 4 |
| LR | 1e-4 |
| Image size | 256 |
| Classes | 23 |
| Best val mIoU | **0.4716** |
| Status | completed |

## Loading

The checkpoint is a PyTorch `state_dict` saved with `torch.save`. Rebuild
the model with the training architecture, then load weights:

```python
import torch
ckpt = torch.load("prithvi_300m_256_best.pt", map_location="cpu")
# `ckpt` is a state_dict (or a dict containing one under "model"/"state_dict").
# Instantiate the dual-head Prithvi-EO-2.0 300M model as in
# scripts/train_unified.py (backbone "prithvi_eo_v2_300m_tl",
# num_temporal_frames=4, in_channels=35, num_classes=23) and call
# model.load_state_dict(...).
```

See `scripts/train_unified.py` and `imint/analyzers/prithvi.py` in the
ImintEngine repo for the exact model construction and band handling.

## License & provenance

Trained on data derived from open sources (ESA Copernicus Sentinel-2,
Naturvårdsverket NMD, Jordbruksverket LPIS/SJV, Skogsstyrelsen SKS, SLU,
Copernicus DEM). The Prithvi-EO-2.0 backbone is released by IBM/NASA under
its own license — check upstream terms before redistribution.
