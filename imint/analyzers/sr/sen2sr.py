"""SEN2SR wrapper — ESA OpenSR's radiometrically-consistent 4× SR.

API verified against https://github.com/ESAOpenSR/SEN2SR README and
local CPU traceback debugging (see scripts/_sr_local_smoke.py):

    pip install sen2sr mlstac
    import mlstac

    # Two-step load: download to a local cache, then load from path.
    mlstac.download(file=HF_URL, output_dir=local_dir)
    model = mlstac.load(local_dir).compiled_model(device=device)

    # Input: (10, H, W) float32 reflectance [0,1]
    # Bands: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
    # The model internally indexes [0, 1, 2, 6] → (B02, B03, B04, B08)
    # for the RGBN reference branch.
    superX = model(X[None]).squeeze(0)   # (10, H*4, W*4)

The model returns all 10 bands super-resolved; we slice [B04, B03, B02]
for the RGB display panel.

Inference is chunked (128×128 LR patches with Hann blend) so a full
~900×600 LR tile fits in 11 GB VRAM and avoids the size-dependent
asserts in the model's reference branch.

Reference: Aybar et al. 2025, "A Radiometrically and Spatially
Consistent Super-Resolution Framework for Sentinel-2", RSE.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .base import BaseSRModel


class SEN2SR(BaseSRModel):
    name = "sen2sr"
    scale = 4

    DEFAULT_HF_URL = (
        "https://huggingface.co/tacofoundation/sen2sr/"
        "resolve/main/SEN2SRLite/main/mlm.json"
    )
    DEFAULT_CACHE_DIR = "outputs/sr_cache/SEN2SRLite"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._model = None
        self._device = self.config.get("device", "cuda")
        self._hf_url = self.config.get("hf_url", self.DEFAULT_HF_URL)
        self._cache_dir = Path(self.config.get("cache_dir", self.DEFAULT_CACHE_DIR))
        self._chunk_lr = int(self.config.get("chunk_lr", 128))
        self._overlap_lr = int(self.config.get("overlap_lr", 16))

    def _load(self) -> None:
        import torch
        import mlstac  # type: ignore[import-not-found]

        self._torch = torch
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        if not (self._cache_dir / "mlm.json").exists():
            mlstac.download(file=self._hf_url, output_dir=str(self._cache_dir))
        self._model = mlstac.load(str(self._cache_dir)).compiled_model(
            device=self._device
        )

    def _hann2d(self, n: int) -> np.ndarray:
        w = np.hanning(n).astype(np.float32)
        w = np.clip(w, 1e-3, 1.0)
        return np.outer(w, w)

    def _forward_chunk(self, x10: np.ndarray) -> np.ndarray:
        """Run SEN2SRLite on a single (10, 128, 128) chunk → (3, 512, 512)
        RGB (B04, B03, B02 sliced from the 10-band model output)."""
        torch = self._torch
        x = torch.from_numpy(x10).to(self._device)
        with torch.no_grad():
            y = self._model(x[None]).squeeze(0)  # (10, H*4, W*4)
        # Slice [R=B04, G=B03, B=B02] → indices [2, 1, 0] in the band
        # order [B02, B03, B04, ...]
        rgb = y[[2, 1, 0]].permute(1, 2, 0).cpu().numpy()
        return rgb

    def _predict(self, x10: np.ndarray) -> np.ndarray:
        """Input: (10, H, W). Output: (H*4, W*4, 3) RGB."""
        if x10.ndim != 3 or x10.shape[0] != 10:
            raise ValueError(
                f"sen2sr expects (10, H, W) band stack; got {x10.shape}"
            )

        _, H, W = x10.shape
        c = self._chunk_lr
        ov = self._overlap_lr
        stride = c - ov
        s = self.scale

        pad_h = (-H) % stride or 0
        pad_w = (-W) % stride or 0
        if pad_h or pad_w:
            x10p = np.pad(
                x10, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
            )
        else:
            x10p = x10
        _, Hp, Wp = x10p.shape

        sr_canvas = np.zeros((Hp * s, Wp * s, 3), dtype=np.float32)
        wt_canvas = np.zeros((Hp * s, Wp * s), dtype=np.float32)
        win = self._hann2d(c * s)

        ys = list(range(0, max(1, Hp - c) + 1, stride))
        xs = list(range(0, max(1, Wp - c) + 1, stride))
        if ys[-1] + c < Hp: ys.append(Hp - c)
        if xs[-1] + c < Wp: xs.append(Wp - c)

        for y0 in ys:
            for x0 in xs:
                chunk = x10p[:, y0:y0 + c, x0:x0 + c]
                rgb_chunk = self._forward_chunk(chunk)
                sy0, sx0 = y0 * s, x0 * s
                sr_canvas[sy0:sy0 + c * s, sx0:sx0 + c * s] += rgb_chunk * win[..., None]
                wt_canvas[sy0:sy0 + c * s, sx0:sx0 + c * s] += win

        sr_canvas /= wt_canvas[..., None]
        return sr_canvas[: H * s, : W * s]
