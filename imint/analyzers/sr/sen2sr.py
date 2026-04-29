"""SEN2SR wrapper — ESA OpenSR's radiometrically-consistent 4× SR.

API verified against https://github.com/ESAOpenSR/SEN2SR README:

    pip install sen2sr mlstac
    import mlstac
    model = mlstac.load("model/SEN2SRLite").compiled_model(device=device)
    superX = model(X[None]).squeeze(0)   # X: (4, H, W) float32 [0,1]

SEN2SRLite is the lightweight 4× variant; the full model has additional
20m bands. We use Lite for the RGB-only showcase.

Reference: Aybar et al. 2025, "A Radiometrically and Spatially
Consistent Super-Resolution Framework for Sentinel-2", RSE.
"""
from __future__ import annotations

import numpy as np

from .base import BaseSRModel


class SEN2SR(BaseSRModel):
    name = "sen2sr"
    scale = 4

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._model = None
        self._device = self.config.get("device", "cuda")
        self._mlstac_id = self.config.get("mlstac_id", "model/SEN2SRLite")

    def _load(self) -> None:
        import torch
        import mlstac  # type: ignore[import-not-found]

        self._torch = torch
        self._model = mlstac.load(self._mlstac_id).compiled_model(
            device=self._device
        )

    def _predict(self, rgb_lr: np.ndarray) -> np.ndarray:
        torch = self._torch
        # SEN2SRLite expects 4 channels (R, G, B, NIR). For an RGB-only
        # showcase we duplicate the green band as a NIR proxy — not
        # physically correct but keeps the model's input signature happy
        # without requiring a separate B08 fetch. The output's NIR
        # channel is discarded; only RGB is returned.
        h, w, _ = rgb_lr.shape
        x4 = np.empty((4, h, w), dtype=np.float32)
        x4[0] = rgb_lr[..., 0]                 # R
        x4[1] = rgb_lr[..., 1]                 # G
        x4[2] = rgb_lr[..., 2]                 # B
        x4[3] = rgb_lr[..., 1]                 # NIR proxy (G)
        x = torch.from_numpy(x4).to(self._device)
        with torch.no_grad():
            y = self._model(x[None]).squeeze(0)  # (4, H*4, W*4)
        rgb_sr = y[:3].permute(1, 2, 0).cpu().numpy()
        return rgb_sr
