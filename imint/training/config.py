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
    years: list[str] = field(default_factory=lambda: ["2019", "2018"])
    growing_season: tuple[int, int] = (6, 8)          # June – August
    grid_spacing_m: int = 10_000                       # 10 km grid
    patch_pixels: int = 224                            # Training patch size
    fetch_pixels: int = 256                            # Fetch larger for crop
    cloud_threshold: float = 0.05
    b02_haze_threshold: float = 0.06                  # Max mean B02 reflectance
                                                       # for clear-sky quality gate

    # ── Multitemporal / seasonal ─────────────────────────────────────────
    # When enabled, each tile stores T temporal frames as (T*6, H, W).
    # Prithvi-EO-2.0 supports up to num_frames=4 temporal steps.
    enable_multitemporal: bool = False                 # Single-date by default
    num_temporal_frames: int = 4                       # Prithvi max = 4
    seasonal_windows: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (4, 5),    # spring:  April – May
            (6, 7),    # summer:  June – July
            (8, 9),    # autumn:  August – September
            (1, 2),    # winter:  January – February
        ]
    )
    seasonal_cloud_threshold: float = 0.10            # Slightly relaxed for winter
    seasonal_require_all: bool = False                 # If False, pad missing seasons
                                                       # with zeros (masked in model)

    # ── Class schema ─────────────────────────────────────────────────────
    num_classes: int = 10                              # Grouped schema (pine/spruce separate)
    use_grouped_classes: bool = True                   # Remap 19→10 at load time
    ignore_index: int = 0                              # Background

    # ── Model ─────────────────────────────────────────────────────────────
    backbone: str = "prithvi_eo_v2_300m_tl"
    decoder_type: str = "upernet"
    decoder_channels: int = 256
    feature_indices: list[int] = field(
        default_factory=lambda: [5, 11, 17, 23]
    )
    dropout: float = 0.1
    unfreeze_backbone_layers: int = 6                  # Unfreeze last N transformer blocks
    backbone_lr_factor: float = 0.1                    # Backbone LR = lr * this factor

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    early_stopping_patience: int = 15                  # Epochs without val improvement
    train_loss_min_delta: float = 0.005                # Stop if train loss change < this
    train_loss_patience: int = 5                       # over this many epochs
    max_class_weight: float = 10.0
    device: str | None = None                          # Auto-detect

    # ── Loss ───────────────────────────────────────────────────────────
    loss_type: str = "focal"                           # "cross_entropy" or "focal"
    focal_gamma: float = 2.0                           # Focal loss focusing parameter

    # ── Rare class handling ────────────────────────────────────────────
    rare_class_threshold: float = 0.02                 # Classes < 2% are "rare"
    max_tile_weight: float = 5.0                       # Max oversampling for rare tiles
    early_stop_metric: str = "miou"                    # "miou", "worst_class_iou", "combined"
    worst_class_weight: float = 0.3                    # Weight in combined metric

    # ── Grid densification ─────────────────────────────────────────────
    enable_grid_densification: bool = False             # Opt-in for rare-class areas
    densify_spacing_m: int = 5_000                     # 5 km sub-grid in dense areas
    enable_scb_densification: bool = False              # SCB tätort urban oversampling
    scb_min_population: int = 2_000                    # Min population for inclusion
    scb_densify_spacing_m: int = 2_500                 # Tighter spacing in urban areas
    enable_sea_densification: bool = False              # Coastal water oversampling
    max_sea_distance_m: int = 5_000                    # Max distance from land (meters)
    enable_sumpskog_densification: bool = False         # Skogsstyrelsen sumpskog
    sumpskog_min_density_pct: float = 5.0              # Min sumpskog % per scan cell
    sumpskog_densify_spacing_m: int = 10_000           # Sub-grid spacing in wetland areas

    # ── Auxiliary raster channels ──────────────────────────────────────────
    enable_height_channel: bool = False                  # Skogsstyrelsen tree height
    enable_volume_channel: bool = False                  # Skogliga grunddata volume
    enable_basal_area_channel: bool = False              # Grundyta (Gy), m²/ha
    enable_diameter_channel: bool = False                # Medeldiameter (Dgv), cm
    enable_dem_channel: bool = False                     # Copernicus DEM (terrain elev.)
    aux_cache_enabled: bool = True                       # Cache aux tiles as .npy

    # Z-score normalization for aux channels: {name: (mean, std)}
    # Computed from non-zero pixels across all training tiles.
    # Placeholder values — recalculate with scripts/compute_aux_stats.py
    aux_norm: dict = field(default_factory=lambda: {
        "height":     (5.0, 6.0),      # meters (Skogsstyrelsen trädhöjd)
        "volume":     (50.0, 70.0),    # m³sk/ha (Skogliga grunddata)
        "basal_area": (7.0, 8.0),     # m²/ha (grundyta)
        "diameter":   (6.0, 7.0),     # cm (medeldiameter)
        "dem":        (200.0, 200.0), # meters a.s.l. (Copernicus DEM)
    })

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
    resume_from_checkpoint: str | None = None     # Path to last_checkpoint.pt

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
