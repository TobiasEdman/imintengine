# Training Runs & Model Comparison

> Summary of all LULC training runs, architecture variants (base vs AuxEncoder), results, and checkpoint locations.

---

## Training Run History

### Run 1: Baseline Test (19-class, CPU)

| Parameter | Value |
|-----------|-------|
| **Data** | `data/lulc_training_test` (89 tiles) |
| **Classes** | 19 (full NMD L2 schema) |
| **Loss** | Cross-entropy |
| **Device** | CPU |
| **Epochs** | 11/15 (early stopped) |
| **Best mIoU** | 6.63% (epoch 6) |
| **Aux channels** | None |
| **Checkpoint** | `checkpoints/lulc/` (overwritten) |
| **Date** | Feb 24, 2026 |

Very low accuracy due to 19-class fragmentation and tiny dataset.

---

### Run 2: Scaled 10-Class (no aux)

| Parameter | Value |
|-----------|-------|
| **Data** | `data/lulc_full` (5,801 tiles) |
| **Classes** | 10 (grouped schema) |
| **Loss** | Focal (gamma=2.0) |
| **Device** | MPS (M1 Max) |
| **Backbone** | Partial unfreeze (last 6 blocks, lr_factor=0.1) |
| **Epochs** | 44/50 (still running) |
| **Best mIoU** | **40.89%** (epoch 44) |
| **Aux channels** | None |
| **Checkpoint** | `checkpoints/lulc/` |
| **Date** | Mar 1-6, 2026 |
| **Status** | Running (was accidentally restarted) |

Improvements over Run 1: grouped classes, focal loss, backbone unfreezing, full dataset.

---

### Run 3: AuxEncoder Late Fusion (5 aux channels) -- BEST

| Parameter | Value |
|-----------|-------|
| **Data** | `data/lulc_full` (5,801 tiles) |
| **Classes** | 10 (grouped schema) + background = 11 |
| **Loss** | Focal (gamma=2.0) |
| **Device** | MPS (M1 Max) |
| **Backbone** | Partial unfreeze (last 6 blocks) |
| **Epochs** | 25 (early stopped, patience 2/15) |
| **Best mIoU** | **43.27%** (epoch 23) |
| **Aux channels** | 5 (height, volume, basal_area, diameter, DEM) |
| **Checkpoint** | `checkpoints/lulc_aux/` |
| **Date** | Mar 3, 2026 |
| **Status** | Stopped (early stopping) |

This is the best performing model. The AuxEncoder provides +2.4% mIoU improvement.

---

## Run 3 Detailed Results (AuxEncoder)

### Per-Class IoU at Best Epoch (23)

| Class | IoU | Notes |
|-------|-----|-------|
| water | 93.38% | Easy spectral separation |
| cropland | 71.34% | Strong phenology signal |
| open_wetland | 62.40% | DEM + height helps |
| open_land | 47.15% | Diverse category |
| forest_spruce | 36.29% | Conifer discrimination |
| forest_mixed | 33.54% | Hardest forest type |
| developed | 31.54% | Urban areas |
| forest_pine | 26.81% | Similar to spruce |
| forest_wetland | 20.33% | Low sample count |
| forest_deciduous | 9.90% | Worst class |

**Overall accuracy:** 55.47%

### Training Progression

| Epoch | mIoU | Delta | Notes |
|-------|------|-------|-------|
| 5 | 38.70% | -- | Already strong start |
| 10 | 39.82% | +1.12% | Steady improvement |
| 15 | 40.65% | +0.83% | |
| 20 | 41.60% | +0.95% | |
| 23 | **43.27%** | +1.67% | **Best** |
| 25 | 42.29% | -0.98% | Declining (patience 2) |

### Loss Convergence

Train loss decreased monotonically: 1.085 (ep 1) -> 0.398 (ep 25). No overfitting observed -- the model was still learning when early stopping triggered on validation plateau.

---

## Architecture: AuxEncoder Late Fusion

### Diagram

```
Input (B, 6, 1, H, W) -> PrithviViT backbone (partial unfreeze)
    -> 4 feature levels [256, 512, 1024, 1024]
    -> UPerNet decoder -> (B, 256, H/16, W/16)
    -> Upsample to (B, 256, H, W)

Aux (B, 5, H, W) -> AuxEncoder (2x Conv3x3+BN+ReLU) -> (B, 64, H, W)

Cat [decoded, aux_feat] -> (B, 320, H, W)
    -> Fusion Conv 1x1+BN+ReLU -> (B, 256, H, W)
    -> SegmentationHead -> (B, 11, H, W) logits
```

### Why Late Fusion Works

1. **Full resolution** -- aux data at 10m (224x224 px), backbone operates at 14x14 tokens. Late fusion preserves spatial detail.
2. **Different data types** -- optical reflectance vs laser-scanned forestry variables. Separate encoders avoid distribution mismatch.
3. **Zero risk to backbone** -- Prithvi's 300M pretrained weights are unaffected.
4. **Regional flexibility** -- AuxEncoder can be swapped for regions with different data availability.

### AuxEncoder Implementation

```python
class AuxEncoder(nn.Module):
    """Lightweight CNN for auxiliary raster channels.
    Two 3x3 conv layers: (B, N, H, W) -> (B, 64, H, W)"""
    def __init__(self, in_channels: int, out_channels: int = 64):
        self.net = nn.Sequential(
            ConvBnRelu(in_channels, out_channels, kernel=3, padding=1),
            ConvBnRelu(out_channels, out_channels, kernel=3, padding=1),
        )
```

File: `imint/fm/upernet.py` (lines 184-204)

---

## Auxiliary Data Sources

| Channel | Source | Resolution | Mean | Std |
|---------|--------|-----------|------|-----|
| height | Skogsstyrelsen (laser) | 12.5m | 7.31 m | 6.49 |
| volume | Skogliga grunddata | 12.5m | 109.69 m3/ha | 106.77 |
| basal_area | Skogliga grunddata | 12.5m | 15.07 m2/ha | 9.95 |
| diameter | Skogliga grunddata | 12.5m | 15.28 cm | 7.88 |
| dem | Copernicus GLO-30 | 30m | 261.56 m | 214.59 |

All resampled to 10m and z-score normalized. Coverage: 5,766/5,801 tiles (99.4%) have all 5 channels.

---

## 10-Class Grouped Schema

Remapped from 19 NMD L2 classes:

| # | Grouped Class | Original NMD Classes |
|---|--------------|---------------------|
| 1 | forest_pine | Tallskog (pine-dominant) |
| 2 | forest_spruce | Granskog (spruce-dominant) |
| 3 | forest_deciduous | Lovskog (deciduous) |
| 4 | forest_mixed | Blandskog, temp non-forest |
| 5 | forest_wetland | All sumpskog variants |
| 6 | open_wetland | Oppen vatmark |
| 7 | cropland | Akerjord |
| 8 | open_land | Oppen mark (bare + vegetated) |
| 9 | developed | Bebyggelse (buildings + infra + roads) |
| 10 | water | Sjoar + hav |

---

## How to Train

### Base Model (no aux)

```bash
python scripts/train_lulc.py \
  --data-dir data/lulc_full \
  --epochs 50 \
  --batch-size 8 \
  --loss-type focal \
  --unfreeze-layers 6 \
  --checkpoint-dir checkpoints/lulc \
  --dashboard --background
```

### AuxEncoder Model (with all 5 aux channels)

```bash
python scripts/train_lulc.py \
  --data-dir data/lulc_full \
  --epochs 50 \
  --batch-size 8 \
  --loss-type focal \
  --unfreeze-layers 6 \
  --enable-all-aux \
  --checkpoint-dir checkpoints/lulc_aux \
  --dashboard --background
```

### Evaluate Best Checkpoint

```bash
python scripts/train_lulc.py \
  --evaluate-only \
  --checkpoint checkpoints/lulc_aux/best_model.pt \
  --data-dir data/lulc_full \
  --enable-all-aux
```

---

## Checkpoint Inventory (M1 Max)

### `checkpoints/lulc_aux/` (best model)

| File | Epoch | mIoU | Size | Date |
|------|-------|------|------|------|
| best_model.pt | 23 | **43.27%** | 1.28 GB | Mar 3 20:20 |
| epoch_005.pt | 5 | 38.70% | 1.28 GB | Mar 3 06:18 |
| epoch_010.pt | 10 | 39.82% | 1.28 GB | Mar 3 07:26 |
| epoch_015.pt | 15 | 40.65% | 1.28 GB | Mar 3 15:21 |
| epoch_020.pt | 20 | 41.60% | 1.28 GB | Mar 3 18:46 |
| epoch_025.pt | 25 | 42.29% | 1.28 GB | Mar 3 20:41 |
| last_checkpoint.pt | 25 | -- | 2.00 GB | Mar 3 20:41 |

### `checkpoints/lulc/` (non-aux, running)

| File | Epoch | mIoU | Size | Date |
|------|-------|------|------|------|
| best_model.pt | 40/44 | 40.89% | 1.28 GB | Mar 6 |
| epoch_005-040.pt | 5-40 | -- | 1.28 GB each | Mar 1-6 |
| last_checkpoint.pt | 44 | -- | 2.00 GB | Mar 6 |

---

## Next Steps

1. **Resume AuxEncoder training** -- the model was still improving (43.27% at epoch 23, train loss still decreasing). Consider increasing patience or resuming from `last_checkpoint.pt`
2. **Add VPP channels** -- HR-VPP phenology (5 bands: SOSD, EOSD, length, maxV, minV) already supported in config but not yet trained
3. **Evaluate on test set** -- run `--evaluate-only` on the aux best checkpoint with the test split
4. **Forest deciduous** -- worst class at 9.9% IoU, needs investigation (class imbalance? spectral confusion with mixed forest?)

---

## Key Files

| File | Purpose |
|------|---------|
| `imint/fm/upernet.py` | AuxEncoder + PrithviSegmentationModel |
| `imint/training/trainer.py` | Training loop with aux support |
| `imint/training/config.py` | TrainingConfig with all aux flags |
| `imint/training/dataset.py` | LULCDataset with aux channel stacking |
| `scripts/train_lulc.py` | CLI entry point for training |
| `scripts/prefetch_aux.py` | Batch aux data prefetch |
| `scripts/compute_aux_stats.py` | Normalization statistics |
