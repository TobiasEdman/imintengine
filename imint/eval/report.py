"""Phase 5 — report generation.

Reads every ``EvalResult`` produced by phases 1-4 and emits:

    REPORT.md           — top-level human-readable summary with all
                          tables + links to plots/artefacts. Pinned to
                          ``outputs/eval/<run_id>/REPORT.md``.
    results.json        — every metric in machine-readable form,
                          including per-class arrays.
    confusion_*.npy     — full confusion matrices (kept out of the
                          markdown to keep it scrollable).
    plots/*.png         — reliability diagrams, per-class IoU bars,
                          per-tile mIoU CDF, geographic heatmap.

The report's structure mirrors the plan doc: one section per phase, a
robustness card at the top, and a go/no-go checklist at the end with
the thresholds from phase 0.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .metrics import EvalResult


def generate(
    all_results: Iterable[EvalResult],
    out_dir: Path,
    *,
    run_id: str,
    go_no_go_thresholds: dict | None = None,
    git_sha: str | None = None,
) -> Path:
    """Render the full report.

    Args:
        all_results: Every EvalResult from every phase (a flat list).
        out_dir: Created if missing. Existing files NOT touched — caller
            should pass a fresh directory per run.
        run_id: Used as title; e.g. ``"ensemble_v3_2026-06-15"``.
        go_no_go_thresholds: Phase-0 pass criteria; the bottom checklist
            cross-references each.
        git_sha: Commit hash of the codebase that produced the results;
            stamped into the report so we know which version was eval'd.

    Returns:
        Path to the generated REPORT.md.

    TODO: emit jinja2 / f-string templated markdown. Plot generation
    delegated to a small ``_plots.py`` helper (matplotlib only —
    keep dependencies minimal so a CPU-only reporting pod is feasible).
    """
    raise NotImplementedError


def _build_robustness_card(results: Iterable[EvalResult]) -> str:
    """Produce the top-of-report robustness card.

    Format::

        | Axis         | In-dist mIoU | Out-of-dist mIoU | Δ (pp) |
        |---           |---           |---               |---     |
        | Temporal     | …            | …                | …      |
        | Geographic   | …            | …                | …      |
        | Sensor       | …            | …                | …      |
        | Phenology    | …            | …                | …      |

    Each row averages over the variants tested for that axis. The
    delta is the headline robustness number.
    """
    raise NotImplementedError


def _check_go_no_go(
    results: Iterable[EvalResult],
    thresholds: dict,
) -> list[tuple[str, bool, str]]:
    """Map phase-0 criteria onto observed numbers.

    Returns list of ``(criterion, passed, observed_vs_threshold)`` rows
    the report renders as a checklist. The orchestrator uses the same
    output to decide whether to advance to deployment.
    """
    raise NotImplementedError
