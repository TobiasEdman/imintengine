"""
imint/analyzers/cot.py — Cloud Optical Thickness analyzer

Estimates per-pixel Cloud Optical Thickness (COT) using an ensemble of
MLP5 models from the DES ml-cloud-opt-thick project.

Reference:
    Pirinen et al. "Creating and Leveraging a Synthetic Dataset of Cloud
    Optical Thickness Measures for Cloud Detection in MSI", Remote Sensing 2024.
    https://github.com/DigitalEarthSweden/ml-cloud-opt-thick

The model expects 11 Sentinel-2 bands (B01 and B10 excluded):
    B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12

Input values should be BOA reflectance (already the case in our pipeline).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .base import BaseAnalyzer, AnalysisResult

# Band order expected by the COT MLP (skip B01 and B10)
COT_BAND_ORDER = ["B02", "B03", "B04", "B05", "B06", "B07",
                   "B08", "B8A", "B09", "B11", "B12"]

# Normalization statistics from SMHI synthetic training data
# (from swe_forest_agency_cls.py, with B01 and B10 removed)
COT_MEANS = torch.tensor([
    0.4967399, 0.47297233, 0.56489476, 0.52922534, 0.65842892,
    0.93619591, 0.90525398, 0.99455938, 0.45607598, 0.53310641,
    0.43227456,
])
COT_STDS = torch.tensor([
    0.28320853, 0.27819884, 0.28527526, 0.31613214, 0.28244289,
    0.32065759, 0.33095272, 0.36282185, 0.29398295, 0.41964159,
    0.33375454,
])

# Default paths to ensemble weights (relative to this file)
_FM_DIR = Path(__file__).resolve().parent.parent / "fm" / "cot_models"
DEFAULT_MODEL_PATHS = sorted(_FM_DIR.glob("cot_mlp5_ensemble_*.pt"))

# COT thresholds (from the paper)
DEFAULT_THICK_CLOUD_THRESH = 0.025
DEFAULT_THIN_CLOUD_THRESH = 0.015


class MLP5(nn.Module):
    """5-layer MLP for COT regression (from ml-cloud-opt-thick)."""

    def __init__(self, input_dim: int = 11, output_dim: int = 1,
                 hidden_dim: int = 64, apply_relu: bool = True):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim)
        self.lin4 = nn.Linear(hidden_dim, hidden_dim)
        self.lin5 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.apply_relu = apply_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.relu(self.lin4(x))
        x = self.lin5(x)
        if self.apply_relu:
            x[:, 0] = self.relu(x[:, 0])
        return x


def _load_ensemble(model_paths: list[Path], device: str = "cpu") -> list[MLP5]:
    """Load an ensemble of MLP5 models."""
    models = []
    for path in model_paths:
        model = MLP5(input_dim=11, output_dim=1, apply_relu=True)
        model.load_state_dict(torch.load(str(path), map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    return models


def cot_inference(
    bands: dict[str, np.ndarray],
    models: list[MLP5],
    device: str = "cpu",
    batch_size: int = 65536,
) -> np.ndarray:
    """Run COT inference on a multi-band image.

    Args:
        bands: Dict with keys from COT_BAND_ORDER, each (H, W) float32
               in reflectance [0, 1].
        models: List of MLP5 models for ensemble averaging.
        device: PyTorch device.
        batch_size: Pixels per batch.

    Returns:
        COT map (H, W) float32 — continuous cloud optical thickness.
    """
    # Stack bands in expected order
    h, w = bands[COT_BAND_ORDER[0]].shape
    img = np.stack([bands[b] for b in COT_BAND_ORDER], axis=-1)  # (H, W, 11)

    # Flatten to (N, 11)
    pixels = img.reshape(-1, len(COT_BAND_ORDER))
    pixels_t = torch.tensor(pixels, dtype=torch.float32, device=device)

    # Normalize
    means = COT_MEANS.to(device)
    stds = COT_STDS.to(device)
    pixels_t = (pixels_t - means) / stds

    # Ensemble inference
    pred_sum = np.zeros(h * w, dtype=np.float32)
    with torch.no_grad():
        for model in models:
            pred = np.zeros(h * w, dtype=np.float32)
            for i in range(0, h * w, batch_size):
                out = model(pixels_t[i:i + batch_size])
                pred[i:i + batch_size] = out[:, 0].cpu().numpy()
            pred_sum += pred / len(models)

    return pred_sum.reshape(h, w)


class COTAnalyzer(BaseAnalyzer):
    """Cloud Optical Thickness analyzer using DES MLP5 ensemble."""

    name = "cot"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._models = None
        self._device = self.config.get("device", "cpu")

    def _ensure_models(self):
        if self._models is None:
            model_paths = self.config.get("model_paths", DEFAULT_MODEL_PATHS)
            if not model_paths:
                raise RuntimeError(
                    "No COT model weights found. Expected .pt files in "
                    f"{_FM_DIR}"
                )
            self._models = _load_ensemble(model_paths, self._device)

    def analyze(self, rgb, bands=None, date=None, coords=None,
                output_dir="outputs"):
        if bands is None:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error="COT requires multi-band input (no bands provided)",
            )

        # Check all required bands are present
        missing = [b for b in COT_BAND_ORDER if b not in bands]
        if missing:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error=f"COT missing bands: {missing}",
            )

        self._ensure_models()

        # Run inference
        cot_map = cot_inference(bands, self._models, self._device)

        # Classify
        thick_thresh = self.config.get("thick_cloud_threshold",
                                       DEFAULT_THICK_CLOUD_THRESH)
        thin_thresh = self.config.get("thin_cloud_threshold",
                                      DEFAULT_THIN_CLOUD_THRESH)

        thick_cloud = cot_map >= thick_thresh
        thin_cloud = (cot_map >= thin_thresh) & (cot_map < thick_thresh)
        clear = cot_map < thin_thresh

        h, w = cot_map.shape
        total = h * w
        stats = {
            "clear_fraction": float(clear.sum()) / total,
            "thin_cloud_fraction": float(thin_cloud.sum()) / total,
            "thick_cloud_fraction": float(thick_cloud.sum()) / total,
            "cot_mean": float(np.nanmean(cot_map)),
            "cot_max": float(np.nanmax(cot_map)),
            "cot_min": float(np.nanmin(cot_map)),
        }

        # 3-class map: 0=clear, 1=thin cloud, 2=thick cloud
        cloud_class = np.zeros((h, w), dtype=np.uint8)
        cloud_class[thin_cloud] = 1
        cloud_class[thick_cloud] = 2

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "cot_map": cot_map,
                "cloud_class": cloud_class,
                "stats": stats,
            },
            metadata={
                "date": date,
                "n_models": len(self._models),
                "thick_threshold": thick_thresh,
                "thin_threshold": thin_thresh,
                "device": self._device,
            },
        )
