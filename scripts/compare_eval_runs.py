#!/usr/bin/env python3
"""Diff two eval-run output directories.

When iterating on the ensemble we want to know: did v3 actually beat
v2, and on which axes? This script reads ``REPORT.md`` JSON fences (or
``results.json`` siblings) from both runs and produces a delta table
plus a verdict.

Usage::

    python scripts/compare_eval_runs.py \\
        --baseline /outputs/eval/ensemble_v2_2026-05-30 \\
        --candidate /outputs/eval/ensemble_v3_2026-06-15 \\
        --output    /outputs/eval/ensemble_v2_vs_v3.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path,
                   help="Older run's output directory.")
    p.add_argument("--candidate", required=True, type=Path,
                   help="Newer run's output directory.")
    p.add_argument("--output", required=True, type=Path,
                   help="Where to write the comparison markdown.")
    p.add_argument("--significance-threshold", type=float, default=1.0,
                   help="Minimum |Δ pp| considered meaningful in the verdict.")
    args = p.parse_args()

    # TODO:
    #   baseline_results = json.load((args.baseline / "results.json").open())
    #   candidate_results = json.load((args.candidate / "results.json").open())
    #   for each phase + per-class:
    #       delta = candidate - baseline
    #       mark improvement / regression / noise based on threshold
    #   render markdown table + verdict
    #
    # Verdict shape:
    #   "v3 is better overall on N axes (mIoU Δ +X pp), regression on M
    #    axes (mIoU Δ -Y pp). NOT a Pareto improvement — investigate
    #    regressions before deploying."
    #
    # No bias toward "candidate is better" — if v3 lost on too many
    # rare classes the verdict says so.
    raise NotImplementedError(
        "Skeleton — fill when at least two eval runs exist to compare."
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
