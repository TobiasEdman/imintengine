"""
imint/training/config.py — Training configuration

Centralised dataclass for all hyperparameters, paths, and options
used by the LULC training pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Configuration for Prithvi LULC training pipeline."""

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir: str = "data/lulc_training"
    years: list[str] = field(default_factory=lambda: ["2017", "2018"])
    growing_season: tuple[int, int] = (5, 9)          # May – September
    grid_spacing_m: int = 10_000                       # 10 km grid
    patch_pixels: int = 224                            # Training patch size
    fetch_pixels: int = 256                            # Fetch larger for crop
    cloud_threshold: float = 0.10

    # ── Class schema ─────────────────────────────────────────────────────
    num_classes: int = 19                              # Full NMD L2
    use_grouped_classes: bool = False                  # Set True for 10-class
    ignore_index: int = 0                              # Background

    # ── Model ─────────────────────────────────────────────────────────────
    backbone: str = "prithvi_eo_v2_300m_tl"
    decoder_type: str = "upernet"
    decoder_channels: int = 256
    feature_indices: list[int] = field(
        default_factory=lambda: [5, 11, 17, 23]
    )
    dropout: float = 0.1

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    early_stopping_patience: int = 5
    max_class_weight: float = 10.0
    device: str | None = None                          # Auto-detect

    # ── Loss ───────────────────────────────────────────────────────────
    loss_type: str = "cross_entropy"                   # "cross_entropy" or "focal"
    focal_gamma: float = 2.0                           # Focal loss focusing parameter

    # ── Rare class handling ────────────────────────────────────────────
    rare_class_threshold: float = 0.02                 # Classes < 2% are "rare"
    max_tile_weight: float = 5.0                       # Max oversampling for rare tiles
    early_stop_metric: str = "miou"                    # "miou", "worst_class_iou", "combined"
    worst_class_weight: float = 0.3                    # Weight in combined metric

    # ── Grid densification ─────────────────────────────────────────────
    enable_grid_densification: bool = False             # Opt-in for rare-class areas
    densify_spacing_m: int = 5_000                     # 5 km sub-grid in dense areas

    # ── Validation split (latitude-based) ─────────────────────────────────
    val_latitude_min: float = 64.0
    val_latitude_max: float = 66.0
    test_latitude_min: float = 66.0

    # ── Augmentation ──────────────────────────────────────────────────────
    random_flip: bool = True
    random_rotation: bool = True
    random_crop: bool = True

    # ── Checkpoint ────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints/lulc"
    save_every_n_epochs: int = 5

    # ── Prithvi normalization (from config.json, DN-scale) ────────────────
    prithvi_mean: list[float] = field(
        default_factory=lambda: [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
    )
    prithvi_std: list[float] = field(
        default_factory=lambda: [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]
    )
    prithvi_bands: list[str] = field(
        default_factory=lambda: ["B02", "B03", "B04", "B8A", "B11", "B12"]
    )
