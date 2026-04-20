"""
imint/fm/normalize.py — Per-model input normalizers

Each foundation model has its own normalization scheme:
- Prithvi: DN-scale (reflectance * 10000 - mean) / std
- Clay: reflectance-scale (x - mean) / std on reflectance [0, 1]
- TerraMind: per-modality (S2, S1, DEM handled separately)
- CROMA: 12-band S2 z-score + S1 log-dB

Normalization is applied inside the model's forward() pass, not in the
dataset, so the dataset stays raw and reusable across experiments.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """Base class for model-specific input normalizers.

    Subclasses should register mean/std as buffers so they move with .to(device).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PrithviNormalizer(Normalizer):
    """DN-scale normalization for Prithvi-EO-2.0 (300M and 600M).

    Expects input in reflectance [0, 1] range. Multiplies by 10000 to
    DN scale, then (x - mean) / std.

    Input:  (B, 6, T, H, W) or (B, 6, H, W) reflectance
    Output: same shape, normalized
    """

    MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
    STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 6, 1, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 6, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both (B, C, T, H, W) and (B, C, H, W)
        if x.dim() == 4:
            mean = self.mean.squeeze(2)  # (1, 6, 1, 1)
            std = self.std.squeeze(2)
        else:
            mean = self.mean
            std = self.std
        return (x * 10000.0 - mean) / std


class ClayNormalizer(Normalizer):
    """Reflectance-scale normalization for Clay v1.5 (7 S2 bands).

    Bands: B02, B03, B04, B08, B8A, B11, B12 (note: B08 broad NIR)
    Expects reflectance [0, 1], applies (x - mean) / std directly.

    Input:  (B, 7, H, W) reflectance
    Output: same shape, normalized
    """

    # Clay constants from clay-foundation/model
    MEAN = [0.1369, 0.1597, 0.1741, 0.2858, 0.2916, 0.2104, 0.1594]
    STD = [0.2026, 0.2011, 0.2146, 0.2138, 0.2003, 0.1500, 0.1204]

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 7, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 7, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class TerraMindNormalizer(Normalizer):
    """Per-modality normalization for TerraMind.

    Accepts a dict {"s2": tensor, "s1": tensor, "dem": tensor} and
    normalizes each modality independently. Returns dict with same keys.

    S2: (B, 6, H, W) DN * 10000, Prithvi-compatible
    S1: (B, 2, H, W) linear σ⁰ → log-dB → z-score
    DEM: (B, 1, H, W) meters → z-score (Sweden-adapted)
    """

    S2_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
    S2_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

    # S1 typical σ⁰ in linear scale → convert to dB → center
    # dB median ~-15 for land; spread ~±10
    S1_DB_MEAN = [-15.0, -22.0]  # VV, VH typical
    S1_DB_STD = [5.0, 5.0]

    DEM_MEAN = 264.0  # Sweden mean elevation
    DEM_STD = 215.0

    def __init__(self):
        super().__init__()
        self.register_buffer("s2_mean", torch.tensor(self.S2_MEAN).view(1, 6, 1, 1))
        self.register_buffer("s2_std", torch.tensor(self.S2_STD).view(1, 6, 1, 1))
        self.register_buffer("s1_mean", torch.tensor(self.S1_DB_MEAN).view(1, 2, 1, 1))
        self.register_buffer("s1_std", torch.tensor(self.S1_DB_STD).view(1, 2, 1, 1))

    def forward(self, inputs: dict) -> dict:
        out = {}
        if "s2" in inputs:
            x = inputs["s2"]
            # Handle (B, C, T, H, W) or (B, C, H, W)
            if x.dim() == 5:
                mean = self.s2_mean.unsqueeze(2)
                std = self.s2_std.unsqueeze(2)
            else:
                mean = self.s2_mean
                std = self.s2_std
            out["s2"] = (x * 10000.0 - mean) / std
        if "s1" in inputs:
            x = inputs["s1"]
            # Linear σ⁰ → dB (clip to avoid log(0)), then z-score
            x_db = 10.0 * torch.log10(x.clamp(min=1e-5))
            out["s1"] = (x_db - self.s1_mean) / self.s1_std
        if "dem" in inputs:
            out["dem"] = (inputs["dem"] - self.DEM_MEAN) / self.DEM_STD
        return out


class CromaNormalizer(Normalizer):
    """CROMA normalization for 12-band S2 + 2-band S1.

    S2: 12 bands DN, z-score per band (padding 0s for missing bands)
    S1: 2 bands linear σ⁰ → dB → z-score
    """

    # CROMA S2 means/stds (12 bands: B01-B12 minus B09/B10 plus B08A)
    # Approximated from standard S2 statistics over land
    S2_MEAN = [
        1354.0, 1087.0, 1342.0, 1433.0, 1590.0, 2109.0,
        2387.0, 2480.0, 2734.0, 1930.0, 1958.0, 1363.0,
    ]
    S2_STD = [
        245.0, 2248.0, 2179.0, 2178.0, 1960.0, 1766.0,
        1775.0, 1804.0, 1850.0, 1340.0, 1242.0, 1049.0,
    ]

    S1_DB_MEAN = [-15.0, -22.0]
    S1_DB_STD = [5.0, 5.0]

    def __init__(self):
        super().__init__()
        self.register_buffer("s2_mean", torch.tensor(self.S2_MEAN).view(1, 12, 1, 1))
        self.register_buffer("s2_std", torch.tensor(self.S2_STD).view(1, 12, 1, 1))
        self.register_buffer("s1_mean", torch.tensor(self.S1_DB_MEAN).view(1, 2, 1, 1))
        self.register_buffer("s1_std", torch.tensor(self.S1_DB_STD).view(1, 2, 1, 1))

    def forward(self, inputs: dict) -> dict:
        out = {}
        if "s2_full" in inputs:
            x = inputs["s2_full"]  # (B, 12, H, W)
            out["s2_full"] = (x * 10000.0 - self.s2_mean) / self.s2_std
        if "s1" in inputs:
            x = inputs["s1"]
            x_db = 10.0 * torch.log10(x.clamp(min=1e-5))
            out["s1"] = (x_db - self.s1_mean) / self.s1_std
        return out


class TesseraNormalizer(Normalizer):
    """Passthrough normalizer for TESSERA.

    TESSERA embeddings are already normalized by the upstream encoder.
    Our enrichment dequantizes int8 × scales → float; no further
    standardization is needed (and applying any would corrupt the
    learned embedding space).
    """

    def forward(self, x):
        return x


class ThorNormalizer(Normalizer):
    """Passthrough normalizer for THOR.

    THOR (FlexiViT, terratorch-native) applies its own per-band normalization
    internally when built through BACKBONE_REGISTRY, using the statistics
    embedded in thor_terratorch_ext. We forward inputs unchanged.

    If you inspect the loaded THOR model and find it does NOT apply input
    normalization, replace this with an explicit per-band z-score using
    THOR's pretraining statistics.
    """

    def forward(self, x):
        # Accept either a tensor or a dict (multi-modal) — passthrough both.
        return x


NORMALIZERS = {
    "prithvi": PrithviNormalizer,
    "clay": ClayNormalizer,
    "terramind": TerraMindNormalizer,
    "croma": CromaNormalizer,
    "thor": ThorNormalizer,
    "tessera": TesseraNormalizer,
}
