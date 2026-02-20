"""
imint/analyzers/base.py — Abstract base analyzer and result dataclass

All analyzers subclass BaseAnalyzer and implement analyze().
BaseAnalyzer.run() wraps analyze() with error handling so analyzers
never raise exceptions to the engine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class AnalysisResult:
    """Output from a single analyzer run."""
    analyzer: str
    success: bool
    outputs: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    error: str | None = None

    def summary(self) -> str:
        if self.success:
            keys = ", ".join(self.outputs.keys()) if self.outputs else "none"
            return f"[{self.analyzer}] OK — outputs: {keys}"
        return f"[{self.analyzer}] FAILED — {self.error}"


class BaseAnalyzer(ABC):
    """
    Abstract base class for all analyzers.

    Subclasses implement analyze(). The engine calls run(), which
    wraps analyze() with exception handling so a failing analyzer
    never crashes the pipeline.
    """

    name: str = "base"

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abstractmethod
    def analyze(
        self,
        rgb: np.ndarray,
        bands: dict[str, np.ndarray] | None = None,
        date: str | None = None,
        coords: dict | None = None,
        output_dir: str = "outputs",
    ) -> AnalysisResult:
        ...

    def run(
        self,
        rgb: np.ndarray,
        bands: dict[str, np.ndarray] | None = None,
        date: str | None = None,
        coords: dict | None = None,
        output_dir: str = "outputs",
    ) -> AnalysisResult:
        """Run analyze() with error handling."""
        try:
            return self.analyze(
                rgb, bands=bands, date=date,
                coords=coords, output_dir=output_dir,
            )
        except Exception as e:
            return AnalysisResult(
                analyzer=self.name,
                success=False,
                error=f"{type(e).__name__}: {e}",
            )
