"""
imint/job.py — Core data model

An IMINTJob is the unit of work passed between the executor and the engine.
It is executor-agnostic: ColonyOS, local runner, Airflow, cron — all produce
the same IMINTJob and hand it to run_job().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class IMINTJob:
    """
    Everything the analysis engine needs to process one satellite acquisition.

    Populated by the executor (ColonyOS, local, etc.) and consumed by run_job().
    """
    # Where and when
    date: str                          # ISO date, e.g. "2022-06-15"
    coords: dict                       # {"west": ..., "south": ..., "east": ..., "north": ...}

    # Image data — set by the data fetcher after cloud verification
    rgb: np.ndarray | None = None      # (H, W, 3) float32 [0, 1]
    bands: dict[str, np.ndarray] = field(default_factory=dict)  # {"B04": arr, ...}

    # Routing
    output_dir: str = "outputs"
    config_path: str = "config/analyzers.yaml"

    # Optional metadata passed through from the executor
    job_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class IMINTResult:
    """
    Output from run_job(). Returned to the executor for logging/storage.
    """
    job_id: str | None
    date: str
    success: bool
    analyzer_results: list = field(default_factory=list)
    summary_path: str | None = None
    error: str | None = None
