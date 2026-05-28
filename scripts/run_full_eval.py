#!/usr/bin/env python3
"""Orchestrate the full 5-phase evaluation suite.

Composes :mod:`imint.eval` phase entry points into a single run and
writes the report at ``outputs/eval/<run_id>/REPORT.md``.

Designed to be safely Ctrl-C-able mid-run: each phase persists its own
``EvalResult`` to disk before the next phase starts, and the report
generator can be re-run against a partial result set.

Usage::

    python scripts/run_full_eval.py \\
        --ensemble-checkpoint /checkpoints/ensemble_v3.pt \\
        --tiles-dir /cephfs/unified_v2_512 \\
        --splits-root /checkpoints/eval_splits \\
        --output-dir /outputs/eval/ensemble_v3_2026-06-15 \\
        --phases 1,2a,2b,2c,2d,3,4 \\
        --baselines trivial_majority,random_forest,single_frame
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PHASE_NAMES = {
    "1":   "in_distribution",
    "2a":  "temporal_shift",
    "2b":  "geographic_shift",
    "2c":  "sensor_shift",
    "2d":  "phenology_shift",
    "3":   "failure_modes",
    "4":   "operational",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ensemble-checkpoint", required=True, type=Path)
    p.add_argument("--tiles-dir", required=True)
    p.add_argument("--splits-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--phases", default=",".join(PHASE_NAMES),
                   help="Comma-separated phase keys; subset of "
                        + ", ".join(PHASE_NAMES))
    p.add_argument("--baselines", default="trivial_majority,random_forest",
                   help="Subset of imint.eval.baselines.BASELINES keys.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-id", default=None,
                   help="Defaults to output-dir basename.")
    p.add_argument("--skip-existing-phases", action="store_true",
                   help="Resume an interrupted run.")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    # from imint.eval import (
    #     evaluate_in_distribution, evaluate_temporal_shift,
    #     evaluate_geographic_shift, evaluate_sensor_shift,
    #     evaluate_phenology_shift, evaluate_failure_modes,
    #     evaluate_operational, generate_report,
    # )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    phases = [pk.strip() for pk in args.phases.split(",") if pk.strip()]
    unknown = set(phases) - set(PHASE_NAMES)
    if unknown:
        print(f"Unknown phase keys: {sorted(unknown)}", file=sys.stderr)
        return 2

    # TODO: for each phase key, dispatch to its imint.eval module, save
    # EvalResult bundle to <output-dir>/phase_<key>.json, advance.
    # Then call generate_report(all_results, output_dir, run_id=...).
    raise NotImplementedError("Skeleton — fill phase-by-phase as they land.")


if __name__ == "__main__":
    sys.exit(main() or 0)
