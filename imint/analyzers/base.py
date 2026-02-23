"""
imint/analyzers/base.py — Abstract base analyzer and result dataclass

All analyzers subclass BaseAnalyzer and implement analyze().
BaseAnalyzer.run() wraps analyze() with error handling so analyzers
never raise exceptions to the engine.

The ``previous_results`` parameter (Alt A) enables cross-analyzer
dependencies — analyzers that need results from earlier stages can
declare it in their analyze() signature. Analyzers that don't need it
simply omit the parameter; run() detects this via inspect and skips it.

Migration to Alt B (shared JobContext) is a mechanical change: replace
``previous_results: list[AnalysisResult]`` with ``context: JobContext``.
"""
from __future__ import annotations

import inspect
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
        previous_results: list[AnalysisResult] | None = None,
        scl: np.ndarray | None = None,
    ) -> AnalysisResult:
        ...

    def run(
        self,
        rgb: np.ndarray,
        bands: dict[str, np.ndarray] | None = None,
        date: str | None = None,
        coords: dict | None = None,
        output_dir: str = "outputs",
        previous_results: list[AnalysisResult] | None = None,
        scl: np.ndarray | None = None,
    ) -> AnalysisResult:
        """Run analyze() with error handling.

        Passes ``previous_results`` and ``scl`` only if the concrete
        analyze() declares them in its signature — existing analyzers
        that omit them are called without, requiring zero changes.
        """
        try:
            sig = inspect.signature(self.analyze)
            kwargs = dict(bands=bands, date=date, coords=coords, output_dir=output_dir)
            if "previous_results" in sig.parameters:
                kwargs["previous_results"] = previous_results
            if "scl" in sig.parameters:
                kwargs["scl"] = scl
            return self.analyze(rgb, **kwargs)
        except Exception as e:
            return AnalysisResult(
                analyzer=self.name,
                success=False,
                error=f"{type(e).__name__}: {e}",
            )
