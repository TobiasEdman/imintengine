"""
imint/training/config.py — Training configuration

Centralised dataclass for all hyperparameters, paths, and options
used by the LULC training pipeline.

Environment-aware: if ``IMINT_*`` environment variables are set
(e.g. via ``config/environments/dev.env``), they override the
dataclass defaults.  CLI arguments still take highest priority.
"""
from __future__ import annotations

import os
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
    fetch_sources: list[str] = field(                  # Backends for data fetching
        default_factory=lambda: ["copernicus", "des"]  # "copernicus" (CDSE), "des"
    )                                                  # Multiple = parallel load balance

    # ── Multitemporal / seasonal ─────────────────────────────────────────
    # When enabled, each tile stores T temporal frames as (T*6, H, W).
    # Prithvi-EO-2.0 supports up to num_frames=4 temporal steps.
    enable_multitemporal: bool = False                 # Single-date by default
    num_temporal_frames: int = 4                       # Prithvi max = 4
    seasonal_windows: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (4, 4),    # early spring:  April (leaf-out)
            (5, 6),    # early summer:  May – June (green-up)
            (7, 7),    # peak summer:   July (peak NDVI)
            (8, 9),    # late summer:   August – September (senescence)
        ]
    )
    enable_vpp_guided_windows: bool = True             # Use VPP phenology per tile
    seasonal_cloud_threshold: float = 0.10             # Slightly relaxed
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
    enable_vpp_channels: bool = False                    # HR-VPP phenology (5 bands)
    enable_harvest_probability: bool = False              # SKS harvest probability (1 band)
    aux_cache_enabled: bool = True                       # Cache aux tiles as .npy

    # Z-score normalization for aux channels: {name: (mean, std)}
    # Computed from non-zero pixels across 5796 seasonal tiles (lulc_seasonal).
    # Recompute with: python scripts/compute_aux_stats.py --data-dir data/lulc_seasonal
    aux_norm: dict = field(default_factory=lambda: {
        "height":     (7.36, 6.55),        # meters (Skogsstyrelsen trädhöjd)
        "volume":     (118.53, 112.67),    # m³sk/ha (Skogliga grunddata)
        "basal_area": (15.98, 10.20),      # m²/ha (grundyta)
        "diameter":   (16.33, 7.84),       # cm (medeldiameter)
        "dem":        (264.03, 215.37),    # meters a.s.l. (Copernicus DEM)
        "vpp_sosd":   (21130.90, 49.13),   # CNES Julian days (days since 1960-01-01)
        "vpp_eosd":   (21280.29, 78.28),   # CNES Julian days (days since 1960-01-01)
        "vpp_length": (141.61, 41.39),     # days, season length
        "vpp_maxv":   (0.88, 0.57),        # PPI unitless, max vegetation index
        "vpp_minv":   (0.04, 0.05),        # PPI unitless, min vegetation index
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
    freeze_spectral: bool = False                  # Stage 2: freeze backbone+decoder, train only aux

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

    @property
    def enabled_aux_names(self) -> tuple[str, ...]:
        """Canonical ordered list of enabled auxiliary channel names.

        Use this everywhere instead of hardcoding channel name lists.
        Order must match how channels are stacked in the dataset.
        """
        names: list[str] = []
        if self.enable_height_channel:
            names.append("height")
        if self.enable_volume_channel:
            names.append("volume")
        if self.enable_basal_area_channel:
            names.append("basal_area")
        if self.enable_diameter_channel:
            names.append("diameter")
        if self.enable_dem_channel:
            names.append("dem")
        if self.enable_vpp_channels:
            names.extend([
                "vpp_sosd", "vpp_eosd", "vpp_length",
                "vpp_maxv", "vpp_minv",
            ])
        if self.enable_harvest_probability:
            names.append("harvest_probability")
        return tuple(names)

    def __post_init__(self) -> None:
        """Override defaults from ``IMINT_*`` environment variables.

        Only applies when the env var is set *and* the field still has
        its default value (so CLI argparse overrides still win).
        """
        _env_overrides: dict[str, tuple[str, type]] = {
            "IMINT_BATCH_SIZE":   ("batch_size",   int),
            "IMINT_NUM_WORKERS":  ("num_workers",  int),
            "IMINT_EPOCHS":       ("epochs",       int),
            "IMINT_DEVICE":       ("device",       str),
            "IMINT_DATA_DIR":     ("data_dir",     str),
        }
        for env_key, (attr, typ) in _env_overrides.items():
            val = os.environ.get(env_key)
            if val is not None and val != "":
                try:
                    setattr(self, attr, typ(val))
                except (ValueError, TypeError):
                    pass  # ignore malformed env values
