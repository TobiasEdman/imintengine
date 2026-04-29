"""SR4RS wrapper — Cresson 2022 GAN baseline.

Included as a contrast point: a fast GAN method without radiometric
guarantees, so the showcase can illustrate the difference between
faithful (SEN2SR) and hallucinatory (SR4RS) output. The original
implementation is TF/Orfeo-based; we wrap a PyTorch port if available
and fall back to a clear error otherwise.

Repo: https://github.com/remicres/sr4rs
"""
from __future__ import annotations

import numpy as np

from .base import BaseSRModel


class SR4RS(BaseSRModel):
    name = "sr4rs"
    scale = 4

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._model = None
        self._device = self.config.get("device", "cuda")

    def _load(self) -> None:
        import torch
        # Two paths: native TF (heavy, separate stack) or community
        # PyTorch port. Prefer the port if present.
        try:
            from sr4rs_pt import SR4RS_RGB  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "SR4RS PyTorch port not installed. The original "
                "TensorFlow implementation lives at "
                "https://github.com/remicres/sr4rs and needs a separate "
                "container image. Run that as a side-job and skip this "
                "wrapper, or install a PyTorch port."
            ) from e

        ckpt = self.config.get("checkpoint", "sr4rs/rgb-4x")
        self._model = SR4RS_RGB.from_pretrained(ckpt)
        self._model.to(self._device).eval()
        self._torch = torch

    def _predict(self, rgb_lr: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.from_numpy(rgb_lr).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.no_grad():
            y = self._model(x)
        return y.squeeze(0).permute(1, 2, 0).cpu().numpy()
