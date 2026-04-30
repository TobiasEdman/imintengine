"""Bicubic interpolation baseline — the floor every learned SR must beat.

Uses scipy.ndimage.zoom (order=3) directly on float reflectance so the
output preserves the input's absolute scale. PIL.Image.BICUBIC would
require uint8 quantisation which destroys the reflectance values needed
for downstream Spectral Angle Mapper analysis.
"""
from __future__ import annotations

import numpy as np

from .base import BaseSRModel


class BicubicSR(BaseSRModel):
    name = "bicubic"
    scale = 4

    def _load(self) -> None:
        # No model to load.
        pass

    def _predict(self, rgb_lr: np.ndarray) -> np.ndarray:
        from scipy.ndimage import zoom
        # order=3 is true bicubic. mode='reflect' avoids border darkening.
        sr = zoom(rgb_lr, (self.scale, self.scale, 1), order=3, mode="reflect")
        # Bicubic can overshoot slightly; clip back to physical reflectance.
        return np.clip(sr, 0.0, 1.0).astype(np.float32)
