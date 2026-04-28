"""WaterQualityAnalyzer package.

Sentinel-2 water quality retrieval combining two AI methods (Pahlevan MDN,
ACOLITE C2RCC) with two classical indices (NDCI, MCI). Designed for
Bohuslän coastal Case-2 water (Stigfjorden + offshore Skagerrak).

Public entry point: ``WaterQualityAnalyzer``. Registered as ``"water_quality"``
in ``imint.engine.ANALYZER_REGISTRY``.
"""
from __future__ import annotations

from .analyzer import WaterQualityAnalyzer

__all__ = ["WaterQualityAnalyzer"]
