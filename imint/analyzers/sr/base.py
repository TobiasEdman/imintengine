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

    def predict(self, rgb_lr: np.ndarray) -> SRResult:
        """Run SR with error capture — never raises.

        Each wrapper handles its own dtype/range conversion in
        ``_predict``. The public contract here is float32 [0,1] in,
        float32 [0,1] out.
        """
        if rgb_lr.ndim != 3 or rgb_lr.shape[-1] != 3:
            return SRResult(
                model=self.name, success=False,
                error=f"expected (H,W,3) got shape {rgb_lr.shape}",
            )
        try:
            if not self._loaded:
                self._load()
                self._loaded = True
            sr = self._predict(rgb_lr.astype(np.float32))
            sr = np.clip(sr, 0.0, 1.0).astype(np.float32)
            return SRResult(
                model=self.name, success=True, sr=sr, scale=self.scale,
                metadata={"input_shape": rgb_lr.shape, "output_shape": sr.shape},
            )
        except Exception as e:
            return SRResult(
                model=self.name, success=False,
                error=f"{type(e).__name__}: {e}",
            )
