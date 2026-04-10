# Flawed / Invalid Training Results

These runs are preserved for reference only. **Do not use these checkpoints or metrics as baselines.**
They were produced between the 44% mIoU lulc_full benchmark and the current train-pixel-v1 run,
and are considered flawed due to incomplete training, wrong schema, or abandoned experiments.

## Files

| File | Date | Best mIoU | Why flawed |
|------|------|-----------|------------|
| `lulc_training_test.json` | 2026-02-24 | 6.6% | Early sanity-check run, stopped at 11/50 epochs, `lulc_training_test` dataset (toy subset) |
| `lulc_seasonal_single_temporal.json` | 2026-03-11 | 33.7% | Single temporal frame only — deliberately crippled ablation, not a valid architecture |
| `abandoned_mar30.json` | 2026-03-30 | 0 epochs | Abandoned before any epoch completed |

## M1 Max checkpoints (to be marked when back on same network)

The following checkpoint directories on M1 Max (192.168.50.100) also fall in the flawed window
and need to be marked or moved when network access is restored:

- `checkpoints/lulc_seasonal_10class/` (2026-03-27)
- `checkpoints/lulc_seasonal_single_temporal/` (2026-03-27)
- `checkpoints/lulc_seasonal_stage1/` (2026-03-27)
- `checkpoints/lulc_seasonal_twostage/` (2026-03-28)
- `checkpoints/lulc_seasonal_twostage_novpp/` (2026-03-27)
- `data/lulc_seasonal/training_log.json` (2026-03-20, empty/corrupt)

## Valid baselines

See `../lulc_full_10class_baseline.json` — 44.14% mIoU, 10-class dense segmentation, epoch 42/50.
