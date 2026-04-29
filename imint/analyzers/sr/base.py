"""Abstract base for SR model wrappers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SRResult:
    """Output from a single SR model run."""
    model: str
    success: bool
    sr: np.ndarray | None = None  # (H*scale, W*scale, 3), float32, [0,1]
    scale: int = 4
    metadata: dict = field(default_factory=dict)
    error: str | None = None


class BaseSRModel(ABC):
    """Uniform interface across SR backbones.

    Subclasses load weights lazily on first ``predict()`` to avoid
    paying the import/download cost when the model isn't selected.
    """

    name: str = "base"
    scale: int = 4

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._loaded = False

    @abstractmethod
    def _load(self) -> None:
        """Load weights / construct underlying model. Called lazily."""

    @abstractmethod
    def _predict(self, rgb_lr: np.ndarray) -> np.ndarray:
        """Run SR on a single LR RGB tile.

        Args:
            rgb_lr: (H, W, 3) float32 in [0, 1].

        Returns:
            (H*scale, W*scale, 3) float32 in [0, 1].
        """

    def predict(self, x: np.ndarray) -> SRResult:
        """Run SR with error capture — never raises.

        Input shape varies per model:
          - 3-channel: (H, W, 3) RGB for bicubic / ldsr
          - N-channel: (N, H, W) band stack for sen2sr (10 bands)
        Wrappers validate their own input shape in ``_predict``. The
        public output contract is always (H*scale, W*scale, 3) RGB
        float32 in [0, 1].
        """
        try:
            if not self._loaded:
                self._load()
                self._loaded = True
            sr = self._predict(x.astype(np.float32))
            sr = np.clip(sr, 0.0, 1.0).astype(np.float32)
            return SRResult(
                model=self.name, success=True, sr=sr, scale=self.scale,
                metadata={"input_shape": x.shape, "output_shape": sr.shape},
            )
        except Exception as e:
            return SRResult(
                model=self.name, success=False,
                error=f"{type(e).__name__}: {e}",
            )
