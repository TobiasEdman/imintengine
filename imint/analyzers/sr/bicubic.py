"""Bicubic interpolation baseline — the floor every learned SR must beat."""
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
        from PIL import Image
        H, W, _ = rgb_lr.shape
        img = Image.fromarray((rgb_lr * 255.0).astype(np.uint8))
        up = img.resize((W * self.scale, H * self.scale), Image.BICUBIC)
        return np.asarray(up).astype(np.float32) / 255.0
