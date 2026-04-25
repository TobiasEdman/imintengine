# Training hyperparameter reference

Canonical values and rationale for the unified 23-class LULC training
loop (`imint.training.trainer.LULCTrainer`). Everything here is the
**baseline** — values are overridable per-run via flags in
`scripts/train_unified.py` or env vars in `k8s/unified-train-template.yaml`.

Last verified: 2026-04-24 (commit with H100 optimizations).

## Model

| Param | Default | Why |
|---|---|---|
| `backbone` | Prithvi-EO-2.0 300M TL (`prithvi_300m`) | Temporal-Location aware ViT-L. See `imint.fm.registry.MODEL_CONFIGS`. |
| `decoder_type` | `upernet` | UPerNet aggregates ViT features at 4 depths; strong for 10 m GSD segmentation. |
| `feature_indices` | From `ModelSpec` (e.g. 5,11,17,23 for Prithvi 300M) | Evenly spaced taps through the ViT depth. |
| `num_temporal_frames` | 4 | 1 autumn (year-1) + 3 VPP-phenology frames — see CLAUDE.md. |
| `img_size`, `patch_size` | 256 (Prithvi 300M, patch=16) | 16×16 tokens. Prithvi 600M uses patch=14 → must use multiples of 14. |

## Optimization

| Param | Default | Why |
|---|---|---|
| `epochs` | 10 | v6a converged without triggering early-stopping at 10; longer schedules saturate. |
| `batch_size` | **32** | H100 80 GB. BF16 + bs=32 at 256² uses ~45–55 GB, leaves headroom. |
| `lr` | **3e-4** | sqrt-scaled from the original 1e-4 @ bs=4: `1e-4 × sqrt(32/4) = 2.83e-4 ≈ 3e-4`. |
| `backbone_lr_factor` | 0.1 | Backbone gets `lr × 0.1` to preserve pretrained features. |
| `weight_decay` | **0.35** | Prithvi multi-crop standard (see *Weight-decay history* below). |
| `warmup_fraction` | 0.05 | Linear ramp preserving per-group differential (see commit 884fefc). |
| `unfreeze_backbone_layers` | 6 | Last 6 of 24 transformer blocks unfrozen. |
| `label_smoothing` | 0.05 | Slight regularization; combats overconfidence on rare classes. |
| `loss_type` | `focal_dice` | Focal (γ=2) for class imbalance + Dice for mask quality. |
| `lovasz_weight` | 0.3 | Adds Lovasz-softmax auxiliary loss; improves per-class IoU. |
| `weighting_method` | `sqrt` | Class weights ∝ 1/√freq, capped by `max_class_weight`. |
| `early_stopping_patience` | 20 | Epochs without val improvement before stop. |

## H100 performance knobs

Set in `LULCTrainer._setup_cuda_perf_knobs` + the training loop's
autocast context. Enabled automatically when `device.type == "cuda"`.

| Knob | What it does |
|---|---|
| TF32 (`torch.set_float32_matmul_precision("high")`) | ~1.5× matmul throughput on H100/A100 at ~3 bits mantissa cost. No measurable effect on segmentation IoU. |
| `cudnn.benchmark = True` | Picks fastest conv kernel per shape. Safe because `img_size` is fixed across batches. |
| BF16 autocast (`torch.autocast` in the train loop) | ~2–3× forward+backward on H100. BF16 has full FP32 exponent range → no GradScaler needed. |
| Optimizer stays FP32 | Standard AMP pattern. Weights/optimizer state never leave FP32. |

What's explicitly NOT enabled:

- **`channels_last`** — applies to 4D `(B, C, H, W)`. Our Prithvi input
  is 5D `(B, C, T, H, W)`; the memory-format win doesn't apply to 5D
  conv3d. Decoder is 4D but the marginal win isn't worth the wiring.
- **`torch.compile`** — often finicky with hybrid custom models
  (Prithvi + UPerNet + AuxEncoder), and the 15–30 % win doesn't justify
  potential correctness debugging. Revisit after a single-model
  baseline is locked.
- **FP8 / Transformer Engine** — only marginal gain for training at
  our scale; adds significant complexity.

## Data

| Param | Default | Why |
|---|---|---|
| `num_workers` | 16 | On 48-cpu pod, 16 workers saturates I/O without CPU thrashing. 32 was the old default — caused context-switch overhead. |
| `patch_pixels` | 256 | Matches `img_size`; no random crop in current setup. |
| `enable_area_weighting` | True | Per-pixel loss weighting gives small LPIS parcels higher weight. |
| `max_tile_weight` | 5.0 | Max oversampling for rare-class tiles. |

## Weight-decay history

Originally `0.01`. Changed to `0.35` in commit `884fefc` ("Fix training
mode collapse + enable Prithvi TL encoding") after the model collapsed
to predicting only water. Root cause was a triple: too-low weight
decay, a warmup bug giving the backbone 10× its intended LR, and a
scheduler bug. The weight-decay bump follows the **Prithvi multi-crop
segmentation standard** from the original IBM/NASA recipe — higher
decay is needed because the large backbone + focal loss combination
otherwise drifts fast from pretrained features.

**Do not lower** without retesting for mode collapse on a rare class
(potatis, bebyggelse are the canonical canaries).

## v6a → v7 → v7b optimizer changes

The v6a baseline (`unified-train-v6a`, 62 min, mIoU 0.3663) ran in FP32
at bs=4, lr=1e-4, num_workers=32.

**v7** was the first H100-optimized attempt (bs=32, lr=3e-4). It
reproducibly caused **rare-class collapse**: `bete`, `trindsäd`, `råg`,
and `majs` all dropped to exactly 0.0 IoU, and v7-prithvi300 peaked at
mIoU 0.3082 at epoch 5 before degrading to 0.2675 by epoch 10.
Root cause: at bs=32, the sqrt class weighting can't hold rare-class
signal against the majority-class gradient magnitudes, and lr=3e-4
accelerates drift away from rare-class-friendly regions.

**v7b** is the corrected setting: bs=16, lr=2e-4. Still meaningfully
H100-optimized vs v6a (4× batch, BF16, TF32), but sqrt-scaled from v6a
with only a 2× effective-LR multiplier — preserving the per-update
rare-class gradient signal. This is the baseline going forward.

| | v6a | v7 (bad) | **v7b (current)** | Why |
|---|---|---|---|---|
| precision | FP32 | BF16 | **BF16** | H100 Tensor Cores |
| TF32 | off | on | **on** | Free matmul speedup |
| cudnn.benchmark | off | on | **on** | Free conv speedup |
| batch_size | 4 | 32 | **16** | Compromise — keeps rare-class gradient above noise floor |
| lr | 1e-4 | 3e-4 | **2e-4** | sqrt(16/4) × 1e-4 |
| num_workers | 32 | 16 | **16** | Less CPU contention |

v7b-prithvi300 becomes the new reference number for the ensemble;
per-class IoU comparisons happen within the v7b family.

### v7 post-mortem (what it taught us)

- Rare-class collapse at 0.0 IoU is a distinct signature from general
  training instability — it's specifically the sqrt-weighted loss
  failing when batch variance washes out rare-class gradients.
- The sqrt class weighting + focal loss works at bs=4 because each
  batch is dominated by 1–2 tiles and rare-class pixels dominate within
  those tiles. At bs=32 the per-step gradient averages too much.
- Linear LR scaling `(lr ∝ bs)` from bs=4 to bs=32 would give 8× lr =
  8e-4, which would be even worse. Sqrt scaling is the right shape,
  but bs=32 is past the knee for this class-imbalanced task regardless.
- Takeaway: when scaling batch for H100, test on canary classes
  (potatis, bebyggelse, bete) before committing to the hyperparams.

## v7d — Collapse rewind

Empirically, all v7/v7b/v7c runs share the same failure mode: rare
classes (vete, korn, bete, trindsäd, råg, majs, …) drop from healthy
IoU to *exactly 0.0* between consecutive epochs and never recover. The
common factor across the runs is **the cosine LR schedule decays past
the threshold needed to recover rare-class predictions** before the
schedule ends. v7c-2080 T2 (prithvi_600m, FP32, bs=4 — i.e. v6a
hyperparams exactly, on different hardware) showed the cleanest
example: epoch 1 collapsed (mIoU 0.31), epoch 2 fully recovered
(0.36, all classes alive), epoch 3 re-collapsed (0.24).

`enable_collapse_rewind` adds a recovery loop:

1. After each val epoch, compute `now_collapsed` = set of classes with
   IoU == 0.0.
2. If `|now_collapsed - best_collapsed| ≥ collapse_threshold` (default
   2 new classes), declare collapse.
3. Reload weights from `best_model.pt`, multiply each optimizer param
   group's LR by `collapse_lr_factor` (default 0.5), re-anneal cosine
   over the **remaining** epochs.
4. Reset `patience_counter` so we don't trip early-stop on the rewind.
5. Allow up to `collapse_max_rewinds` (default 3) recoveries; after
   that, training continues without further intervention.

The flag is **opt-in** — `enable_collapse_rewind: bool = False` — so
existing runs are unaffected. Enable via `--collapse-rewind` flag or
`COLLAPSE_REWIND=1` env in the k8s template.

Why per-group LR multiplication: backbone-vs-decoder LR ratio
(`backbone_lr_factor=0.1`) was deliberately tuned in v6a; halving in
place preserves it.

Why reload weights, not just lower LR: once a class collapses, its
output logits have drifted such that any positive update via gradient
flow is too small to push them above the decision boundary. Restoring
the pre-collapse weights is necessary; lowering LR alone doesn't help.

## See also

- `imint/training/config.py` — canonical defaults
- `imint/training/trainer.py` — the actual loop (collapse-rewind block
  is right after the per-class IoU print)
- `k8s/unified-train-template.yaml` — how to launch a run
- Commit `884fefc` — weight-decay rationale
- Commit `9119fb6` — schema/pipeline showcase
