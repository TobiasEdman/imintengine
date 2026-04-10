# Flawed / Invalid Training Results

**All runs listed here are considered crap. Do not use any of these checkpoints or metrics.**
The only valid baseline is `../lulc_full_10class_baseline.json` (44.14% mIoU).
The next valid result will be `train-pixel-v1` (currently running).

---

## Local JSON logs (this directory)

| File | Date | Best mIoU | Why flawed |
|------|------|-----------|------------|
| `lulc_training_test.json` | 2026-02-24 | 6.6% | Early sanity-check, 11/50 epochs, toy dataset |
| `lulc_seasonal_single_temporal.json` | 2026-03-11 | 33.7% | Single temporal frame ablation |
| `abandoned_mar30.json` | 2026-03-30 | 0 epochs | Abandoned before first epoch |
| `ice_unified_v2_interrupted.json` | 2026-04 | 39.7% | ICE PVC, interrupted mid-train, old Swedish schema, UperNet dense decoder |

---

## ICE cluster — training-checkpoints PVC (`training-checkpoints`, 100Gi)

All checkpoint directories on this PVC are flawed. Checkpoints are stored at `/ckpt/` on the PVC.

| Directory | Checkpoints present | Notes |
|-----------|-------------------|-------|
| `unified_v1/` | ep 5–50 + best + last (15 GB) | No training log found — metrics unknown |
| `unified_v3/` | empty (4 KB) | Aborted before any checkpoint saved |
| `unified_v4/` | ep 5–65 + best + last (24 GB) | No training log — likely the ~55% run |
| `unified_v5/` | ep 5–35 + best + last (12 GB) | No training log |
| `unified_v6/` | ep 5 + best + last (5 GB) | Stopped very early |
| `unified_v7/` | ep 5–45 + best + last (15 GB) | No training log |
| `pixel_v1/` | empty (4 KB) | Current run — in progress, **not flawed** |
| `crop_sweden/` | best + final (18 MB) | Crop classification side-experiment |
| `crop_v2/` | best + final (9 MB) | Crop classification side-experiment |
| *(root)* | `best_model.pt`, `epoch_005–020.pt`, `last_checkpoint.pt` | Orphaned — no run directory, origin unknown |

No training logs were found inside the checkpoints PVC. The `unified_v2` log is on the training-data PVC.

---

## ICE cluster — training-data PVC (`training-data`, 50Gi)

| Path | Epochs | mIoU | Notes |
|------|--------|------|-------|
| `/data/lulc_seasonal/tiles/training_log.json` | 0 | — | Config written, never trained |
| `/data/unified_v2/training_log.json` | 48 | 39.7% | Interrupted, old schema → `ice_unified_v2_interrupted.json` |

---

## M1 Max checkpoints — ✓ DONE (2026-04-10)

All flawed checkpoints and stale logs deleted. Only `checkpoints/lulc_seasonal_v2/` (44% baseline, 1.5 GB) preserved.

Deleted (~61 GB freed):
- `checkpoints/lulc/` (15 GB)
- `checkpoints/lulc_aux/` (15 GB)
- `checkpoints/lulc_seasonal/` (15 GB)
- `checkpoints/lulc_seasonal_10class/` (10 GB)
- `checkpoints/lulc_seasonal_single_temporal/` (1.2 GB)
- `checkpoints/lulc_seasonal_stage1/` (1.5 GB)
- `checkpoints/lulc_seasonal_twostage/` (1.5 GB)
- `checkpoints/lulc_seasonal_twostage_novpp/` (1.5 GB)
- `data/lulc_seasonal/training_log.json` (0 B, empty)
- `data/lulc_seasonal/training_log_single_temporal.json` (33 KB)

---

## Valid baseline

`../lulc_full_10class_baseline.json` — 44.14% mIoU, 10-class dense segmentation, epoch 42/50, M1 Max.
