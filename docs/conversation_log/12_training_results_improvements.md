# Training Results & Improvements

> 19-class baseline results (27% mIoU), 10-class schema design, focal loss, backbone unfreezing, and training configuration improvements.

---

## First Run Results (19-class, CPU, 5,801 tiles)

Best mIoU: **27.16%** (epoch 14, early stopped at epoch 19)

| Class | IoU | Assessment |
|-------|-----|------------|
| water_lakes | 74.0% | Excellent |
| cropland | 63.5% | Good |
| open_wetland | 51.9% | Good |
| water_sea | 51.2% | Good |
| open_land_vegetated | 41.8% | Medium |
| forest_temp_non_forest | 36.7% | Medium |
| forest_spruce | 34.3% | Medium |
| infrastructure | 31.4% | Medium |
| forest_pine | 28.9% | Weak |
| developed_roads | 22.2% | Weak |
| forest_mixed | 18.2% | Poor |
| open_land_bare | 16.4% | Poor |
| developed_buildings | 11.5% | Poor |
| forest_wetland_pine | 10.6% | Poor |
| forest_deciduous | 9.5% | Very poor |
| forest_wetland_temp | 8.2% | Very poor |
| forest_wetland_mixed | 3.6% | Very poor |
| forest_wetland_spruce | 2.0% | Very poor |
| forest_wetland_deciduous | 0.2% | Zero |

Training time: ~9 hours on M1 Max (28 feb -> 1 mar)

---

## Why 27% mIoU?

### 1. 19 classes too granular (biggest problem)
Wetland forest classes (6-10) all <10% IoU and differ minimally in Sentinel-2. These 5 classes = 4.3% of data but drag down mIoU heavily.

### 2. Backbone completely frozen
All 300M Prithvi parameters frozen — only decoder (~15M) trained. Swedish forest types need finer features than global pretraining provides.

### 3. Cross-entropy dominated by common classes
Forest and water dominate loss. Focal loss already implemented but not used.

### 4. Early stopping too aggressive
patience=5 stopped at epoch 19, but train loss still decreasing. Model had more to learn.

---

## Improvement Plan (3 steps)

### Step 1: 10-Class Schema (biggest impact)

Custom 10-class schema keeping pine/spruce separate (user requirement):

```python
LULC_CLASS_NAMES_10 = {
    0: "background",
    1: "forest_pine",
    2: "forest_spruce",
    3: "forest_deciduous",
    4: "forest_mixed",
    5: "forest_wetland",      # merged 5 wetland forest classes
    6: "open_wetland",
    7: "cropland",
    8: "open_land",           # merged bare + vegetated
    9: "developed",           # merged buildings + infrastructure + roads
    10: "water",              # merged lakes + sea
}
```

Runtime remapping via numpy LUT in `dataset.__getitem__()`:
```python
if self.config.use_grouped_classes:
    label = _LUT_19_TO_10[np.clip(label, 0, 19)].astype(np.int64)
```

Tiles stored with 19-class labels, remapped at training time. Class weights aggregated via `_MAP_19_TO_10` before computing.

### Step 2: Focal Loss + Longer Patience

- `loss_type: "focal"` (gamma=2.0)
- `FocalLoss(gamma=0)` = CrossEntropy (verified: both = 2.053)
- `FocalLoss(gamma=2)` = 1.589 (focuses on hard examples)
- `early_stopping_patience: 15` (was 5)

### Step 3: Unfreeze Last Backbone Layers

Fine-tune last 6 transformer blocks + layernorm with differential LR:

```python
# trainer.py
n_unfreeze = self.config.unfreeze_backbone_layers  # default: 6
total_blocks = len(self.model.encoder.blocks)       # 24 for Prithvi
start_idx = total_blocks - n_unfreeze               # blocks 18-23

# Differential LR optimizer
param_groups = [
    {"params": decoder_params, "lr": cfg.lr},            # 1e-4
    {"params": backbone_params, "lr": cfg.lr * 0.1},     # 1e-5
]
```

---

## Config Changes

```python
# config.py defaults updated:
num_classes: int = 10                    # was 19
use_grouped_classes: bool = True         # was False
loss_type: str = "focal"                 # was "cross_entropy"
epochs: int = 50                         # was 30
early_stopping_patience: int = 15        # was 5
unfreeze_backbone_layers: int = 6
backbone_lr_factor: float = 0.1
train_loss_min_delta: float = 0.005      # convergence detection
train_loss_patience: int = 5
```

### Train Loss Convergence Early Stopping

In addition to val-mIoU patience, training stops if train loss changes < 0.005 over 5 consecutive epochs:

```python
if len(train_loss_history) >= cfg.train_loss_patience:
    recent = train_loss_history[-cfg.train_loss_patience:]
    if max(recent) - min(recent) < cfg.train_loss_min_delta:
        train_stop = True
```

---

## Argparse Fix

**Critical bug:** `train_lulc.py` had hardcoded argparse defaults (num_classes=19, epochs=30, loss_type="cross_entropy") that overrode `config.py`. Fixed by reading defaults from `TrainingConfig()`:

```python
defaults = TrainingConfig()
parser.add_argument("--num-classes", type=int, default=defaults.num_classes)
```

---

## Improved Training Results (10-class baseline, no aux)

Best mIoU: **40.76%** (epoch 40/50, early stopped at 42)

| Class | IoU |
|-------|-----|
| water | 92.7% |
| cropland | 63.5% |
| open_wetland | 55.3% |
| open_land | 44.2% |
| forest_spruce | 36.6% |
| forest_mixed | 32.1% |
| developed | 28.2% |
| forest_pine | 26.8% |
| forest_wetland | 18.4% |
| forest_deciduous | 9.7% |

Test evaluation: mIoU=38.23%, OA=57.0% on 1,012 test tiles.

Improvement: 27% -> 41% mIoU (10-class schema + focal loss + backbone unfreezing).

---

## Training Estimates

| Machine | Time/epoch | Total (50 ep) | With early stopping |
|---------|------------|---------------|---------------------|
| M1 16GB (MPS) | ~2-3 min | ~2.5h | ~1.5-2h |
| M1 Max 32GB | ~5-6 min | ~4-5h | ~3-4h |
| A100 | ~2-3 min | ~1.5h | ~1h |

Epoch 1 with unfreezing: mIoU=0.2957, loss=1.0732, 225s/epoch.
75.6M trainable encoder params (blocks 18-23).

---

## Checkpoint Resumability

```python
# trainer.py
_save_resume_checkpoint()  # saves model, optimizer, scheduler, epoch, best_metric, patience
_load_resume_checkpoint()  # restores all state
# Saves last_checkpoint.pt after every epoch, before early stopping check
```

Config:
```python
checkpoint_dir: str = "checkpoints/lulc"
save_every_n_epochs: int = 5
resume_from_checkpoint: str | None = None
```

---

## Key Files Modified

| File | Changes |
|------|---------|
| `imint/training/class_schema.py` | 10-class schema, `_MAP_19_TO_10`, `_LUT_19_TO_10` |
| `imint/training/dataset.py` | Runtime label remapping via LUT |
| `imint/training/config.py` | New defaults, unfreezing params, convergence detection |
| `imint/training/trainer.py` | Selective unfreezing, differential LR, train loss convergence |
| `scripts/train_lulc.py` | Fixed argparse defaults, new CLI args |
| `scripts/run_evaluation.py` | **New** — standalone evaluation script |
| `imint/training/dashboard.py` | Evaluation section added |
