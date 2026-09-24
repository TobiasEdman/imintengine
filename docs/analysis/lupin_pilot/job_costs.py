#!/usr/bin/env python3
"""Report CDSE credit cost per batch job, straight from the backend.

The CDSE credit dashboard lags by hours, which makes it useless for
deciding whether a run in progress is affordable. Each openEO batch job
carries its own ``costs`` and ``usage`` in job metadata the moment it
finishes, so the running total is available immediately and per group —
strictly better than a delayed account-level number.

Note the unit: CDSE openEO bills in platform CREDITS, which are not the
same currency as Sentinel Hub Processing Units. A run costed here cannot
be compared directly against the PU figures in CLAUDE.md.

    python job_costs.py              # summary + projection to all groups
    python job_costs.py --per-job    # one line per job
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(HERE / "s2_jobs.json"))
    ap.add_argument("--total-groups", type=int, default=490,
                    help="denominator for the projection")
    ap.add_argument("--per-job", action="store_true")
    args = ap.parse_args()

    from imint.fetch import _connect_cdse

    state = json.loads(Path(args.state).read_text())
    jobs = [(t, v["job_id"]) for t, v in state.items() if v.get("job_id")]
    if not jobs:
        print("no jobs in state file")
        return

    conn = _connect_cdse()
    costs, pixels, seconds, unknown = [], [], [], 0
    for tag, jid in jobs:
        try:
            d = conn.job(jid).describe_job()
        except Exception as e:
            print(f"  {tag}: metadata unavailable ({type(e).__name__})")
            unknown += 1
            continue
        c = d.get("costs")
        if c is None:            # still running: no cost booked yet
            unknown += 1
            continue
        u = d.get("usage", {})
        mp = u.get("input_pixel", {}).get("value")
        sec = u.get("duration", {}).get("value")
        costs.append(c)
        if mp is not None:
            pixels.append(mp)
        if sec is not None:
            seconds.append(sec)
        if args.per_job:
            print(f"  {tag}: {c} credits, {mp} Mpx, {sec} s")

    if not costs:
        print(f"no completed jobs yet ({unknown} pending)")
        return

    n, tot = len(costs), sum(costs)
    mean = tot / n
    print(f"\ncompleted jobs: {n}  (pending/unknown: {unknown})")
    print(f"credits so far: {tot}   min {min(costs)}  mean {mean:.2f}  "
          f"max {max(costs)}")
    if pixels:
        print(f"input pixels:   mean {sum(pixels)/len(pixels):.1f} Mpx")
    if seconds:
        print(f"server time:    mean {sum(seconds)/len(seconds):.0f} s/job")
    print(f"\nprojection to {args.total_groups} groups: "
          f"~{mean * args.total_groups:.0f} credits "
          f"(range {min(costs) * args.total_groups}-"
          f"{max(costs) * args.total_groups})")


if __name__ == "__main__":
    main()
