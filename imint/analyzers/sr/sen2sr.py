"""SEN2SR wrapper — ESA OpenSR's radiometrically-consistent 4× SR.

Uses the pretrained ``tacofoundation/SEN2SR`` weights via the
``opensr-utils`` production wrapper. The model's hard low-frequency
constraint guarantees ``bicubic_downsample(SR(x)) ≡ x``, so spectral
fidelity is preserved by construction.

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

    def _load(self) -> None:
        # Imports deferred so the rest of the showcase still runs even
        # if opensr-utils isn't installed in the local venv.
        import torch
        from opensr_utils import SEN2SR_Model  # type: ignore[import-not-found]

        ckpt = self.config.get("checkpoint", "tacofoundation/SEN2SR")
        self._model = SEN2SR_Model.from_pretrained(ckpt)
        self._model.to(self._device).eval()
        self._torch = torch

    def _predict(self, rgb_lr: np.ndarray) -> np.ndarray:
        torch = self._torch
        # SEN2SR expects (B, C, H, W) reflectance in [0, 1] for B02/B03/B04
        # in BGR order (Sentinel band order). The wrapper accepts RGB
        # (R,G,B) but we feed in the order the showcase pipeline holds
        # them: (R, G, B) → reorder to (B, G, R) for the model.
        bgr = rgb_lr[..., ::-1].copy()
        x = torch.from_numpy(bgr).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.no_grad():
            y = self._model(x)
        y = y.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Back to RGB display order.
        return y[..., ::-1].copy()
