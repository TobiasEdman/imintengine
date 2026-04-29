"""LDSR-S2 wrapper — ESA OpenSR latent diffusion 4× SR.

API verified against https://github.com/ESAOpenSR/opensr-model README:

    pip install opensr-model
    from omegaconf import OmegaConf
    import opensr_model, requests
    config = OmegaConf.load(...)  # config_10m.yaml from repo
    model = opensr_model.SRLatentDiffusion(config, device=device)
    model.load_pretrained(config.ckpt_version)
    out = model.forward(X, sampling_steps=100)   # X: (B, 4, H, W) [0,1]

The default config and weights come from
``simon-donike/RS-SR-LTDF`` on HuggingFace.

A 936×654 LR tile fed in one shot OOMs the 2080ti (1.13 GB checkpoint
+ activation latents need ~3.3 GB contiguous VRAM). The wrapper
processes the tile in **128×128 LR chunks** with an overlap and a Hann
window blend so seams stay invisible — the model was trained on this
input size, and chunked inference keeps peak VRAM under ~1.5 GB.

Reference: Donike et al. 2025, "Trustworthy SR of Multispectral
Sentinel-2 Imagery With Latent Diffusion", IEEE JSTARS.
"""
from __future__ import annotations

import numpy as np

from .base import BaseSRModel


class LDSR(BaseSRModel):
    name = "ldsr"
    scale = 4

    DEFAULT_CONFIG_URL = (
        "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/"
        "refs/heads/main/opensr_model/configs/config_10m.yaml"
    )

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._model = None
        self._device = self.config.get("device", "cuda")
        self._sampling_steps = self.config.get("sampling_steps", 100)
        self._config_url = self.config.get("config_url", self.DEFAULT_CONFIG_URL)
        # Chunked inference parameters. 128 matches the model's training
        # resolution; overlap=16 gives ~12% redundancy which is enough
        # for the Hann blend to hide the seams without doubling compute.
        self._chunk_lr = int(self.config.get("chunk_lr", 128))
        self._overlap_lr = int(self.config.get("overlap_lr", 16))

    def _load(self) -> None:
        from io import StringIO
        import requests
        import torch
        from omegaconf import OmegaConf  # type: ignore[import-not-found]
        import opensr_model  # type: ignore[import-not-found]

        resp = requests.get(self._config_url, timeout=30)
        resp.raise_for_status()
        cfg = OmegaConf.load(StringIO(resp.text))

        self._torch = torch
        self._model = opensr_model.SRLatentDiffusion(cfg, device=self._device)
        self._model.load_pretrained(cfg.ckpt_version)

    def _hann2d(self, n: int) -> np.ndarray:
        """2D separable Hann window — falls smoothly to 0 at borders so
        chunk edges blend without seams."""
        w = np.hanning(n).astype(np.float32)
        # Avoid zero weights at the very edge — clamp so the cumulative
        # weight in normalization never divides by exactly zero.
        w = np.clip(w, 1e-3, 1.0)
        return np.outer(w, w)

    def _forward_chunk(self, rgb_chunk: np.ndarray) -> np.ndarray:
        """Run the diffusion model on a single (chunk_lr, chunk_lr, 3) RGB
        patch. Returns (chunk_lr*4, chunk_lr*4, 3). Same RGB→4-channel
        adaptation as SEN2SR: G duplicated as NIR proxy; output NIR
        discarded."""
        torch = self._torch
        h, w, _ = rgb_chunk.shape
        x4 = np.empty((1, 4, h, w), dtype=np.float32)
        x4[0, 0] = rgb_chunk[..., 0]
        x4[0, 1] = rgb_chunk[..., 1]
        x4[0, 2] = rgb_chunk[..., 2]
        x4[0, 3] = rgb_chunk[..., 1]
        x = torch.from_numpy(x4).to(self._device)
        with torch.no_grad():
            y = self._model.forward(x, sampling_steps=self._sampling_steps)
        return y[0, :3].permute(1, 2, 0).cpu().numpy()

    def _predict(self, rgb_lr: np.ndarray) -> np.ndarray:
        H, W, _ = rgb_lr.shape
        c = self._chunk_lr
        ov = self._overlap_lr
        stride = c - ov
        s = self.scale

        # Reflect-pad so chunks tile evenly. Padding is symmetric and
        # the trim happens at the end.
        pad_h = (-H) % stride or 0
        pad_w = (-W) % stride or 0
        if pad_h or pad_w:
            rgb = np.pad(rgb_lr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        else:
            rgb = rgb_lr
        Hp, Wp, _ = rgb.shape

        sr_canvas = np.zeros((Hp * s, Wp * s, 3), dtype=np.float32)
        wt_canvas = np.zeros((Hp * s, Wp * s), dtype=np.float32)
        win = self._hann2d(c * s)

        # Iterate chunk top-left corners. Pin to [0, dim - c] so the last
        # row/col always covers the right/bottom edge.
        ys = list(range(0, max(1, Hp - c) + 1, stride))
        xs = list(range(0, max(1, Wp - c) + 1, stride))
        if ys[-1] + c < Hp: ys.append(Hp - c)
        if xs[-1] + c < Wp: xs.append(Wp - c)

        for y0 in ys:
            for x0 in xs:
                chunk = rgb[y0:y0 + c, x0:x0 + c]
                sr_chunk = self._forward_chunk(chunk)
                sy0, sx0 = y0 * s, x0 * s
                sr_canvas[sy0:sy0 + c * s, sx0:sx0 + c * s] += sr_chunk * win[..., None]
                wt_canvas[sy0:sy0 + c * s, sx0:sx0 + c * s] += win

        sr_canvas /= wt_canvas[..., None]
        # Trim back to the un-padded SR size.
        return sr_canvas[: H * s, : W * s]
