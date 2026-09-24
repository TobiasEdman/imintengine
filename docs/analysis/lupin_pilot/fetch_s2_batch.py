#!/usr/bin/env python3
"""Batch-job runner for the June feature cubes.

Synchronous fetching was measured at ~20 min per (block, year) group —
490 groups would be ~165 h serial, which is not a run, it is a week.
A June composite is inherently slow server-side (it processes a month of
scenes), so the fix is not to make each request faster but to stop
waiting on them one at a time.

openEO batch jobs are built for exactly this: the job is queued on the
backend, runs asynchronously, and results are collected whenever they are
ready. Wall-clock becomes (total work / concurrency) instead of a sum,
and — the part that matters more here — the run becomes RESUMABLE. Job
ids are persisted after every state change, so a killed session, a laptop
lid, or a 502 costs nothing: rerun and it picks up the same jobs.

Usage:
    python fetch_s2_batch.py --max-groups 10 --concurrency 4   # calibrate
    python fetch_s2_batch.py                                   # all groups
    python fetch_s2_batch.py --collect-only                    # just gather

State lives in ``s2_jobs.json``; delete it to start over (this abandons,
but does not cancel, any running jobs — use --cancel-stale for that).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

PAD_M = 200.0
TERMINAL_OK = {"finished"}
TERMINAL_BAD = {"error", "canceled"}


def group_bbox(pts, to4326) -> dict:
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    lon0, lat0 = to4326.transform(min(xs) - PAD_M, min(ys) - PAD_M)
    lon1, lat1 = to4326.transform(max(xs) + PAD_M, max(ys) + PAD_M)
    return {"west": lon0, "south": lat0, "east": lon1, "north": lat1}


class JobState:
    """Persisted {tag: {job_id, status}}, flushed on every change."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def set(self, tag: str, **kw) -> None:
        self.data.setdefault(tag, {}).update(kw)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.rename(self.path)          # atomic: never a half-written state

    def get(self, tag: str) -> dict:
        return self.data.get(tag, {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=str(HERE / "chosen_points.json"))
    ap.add_argument("--band-set", default="10m", choices=["all", "10m"],
                    help="10m (default) = B02/B03/B04/B08 only, ~4/11 of "
                         "the input pixels and therefore roughly that "
                         "share of the credits; all = every L2A band. "
                         "State and results are kept per band-set so the "
                         "two schemas can never be mixed.")
    ap.add_argument("--state", default=None)
    ap.add_argument("--result-dir", default=None)
    ap.add_argument("--max-groups", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="simultaneous RUNNING jobs. CDSE caps this per "
                         "account; too high just queues server-side, and "
                         "risks tripping a rate limit for no throughput.")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--collect-only", action="store_true",
                    help="download finished jobs, start nothing new")
    args = ap.parse_args()

    from pyproj import Transformer
    from imint.fetch import _connect_cdse
    from s2_june_composite import build_feature_cube, feature_names

    suffix = "" if args.band_set == "all" else f"_{args.band_set}"
    args.state = args.state or str(HERE / f"s2_jobs{suffix}.json")
    args.result_dir = args.result_dir or str(HERE / f"s2_jobs{suffix}")
    print(f"band set {args.band_set}: "
          f"{len(feature_names(args.band_set))} features -> "
          f"{Path(args.result_dir).name}", flush=True)

    outdir = Path(args.result_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    state = JobState(Path(args.state))
    to4326 = Transformer.from_crs(3006, 4326, always_xy=True)

    pts = json.loads(Path(args.points).read_text())
    groups = defaultdict(list)
    for p in pts:
        groups[(p["block"], p["year"])].append(p)
    keys = sorted(groups)
    if args.max_groups:
        keys = keys[: args.max_groups]

    conn = _connect_cdse()
    done = {t for t, v in state.data.items() if v.get("collected")}
    print(f"{len(keys)} groups requested, {len(done)} already collected, "
          f"concurrency {args.concurrency}", flush=True)

    pending = [f"{b}_{y}" for b, y in keys if f"{b}_{y}" not in done]
    tag_to_key = {f"{b}_{y}": (b, y) for b, y in keys}
    n_err = 0

    while pending:
        # --- reap: collect finished, count failures, free slots ---
        running = []
        for tag in list(pending):
            info = state.get(tag)
            jid = info.get("job_id")
            if not jid:
                continue
            try:
                job = conn.job(jid)
                st = job.status()
            except Exception as e:
                print(f"  {tag}: status check failed ({type(e).__name__}); "
                      f"will retry next poll", flush=True)
                running.append(tag)
                continue
            if st in TERMINAL_OK:
                try:
                    job.get_results().download_files(str(outdir / tag))
                    state.set(tag, status=st, collected=True)
                    pending.remove(tag)
                    print(f"  COLLECTED {tag} ({len(pending)} left)",
                          flush=True)
                except Exception as e:
                    print(f"  {tag}: download failed "
                          f"({type(e).__name__}: {str(e)[:70]})", flush=True)
                    running.append(tag)
            elif st in TERMINAL_BAD:
                # Do not retry blindly: a failed job usually means the
                # graph or the area is the problem, and a retry loop would
                # burn credits rediscovering that.
                state.set(tag, status=st, collected=False, failed=True)
                pending.remove(tag)
                n_err += 1
                print(f"  FAILED {tag}: job status {st}", flush=True)
            else:
                running.append(tag)

        if args.collect_only:
            break

        # --- launch: fill free slots ---
        free = args.concurrency - len(running)
        for tag in pending:
            if free <= 0:
                break
            if state.get(tag).get("job_id"):
                continue
            block, year = tag_to_key[tag]
            bbox = group_bbox(groups[(block, year)], to4326)
            try:
                cube = build_feature_cube(conn, bbox, year,
                                          band_set=args.band_set)
                job = cube.create_job(title=f"lupin-{tag}",
                                      out_format="GTiff")
                job.start()
                state.set(tag, job_id=job.job_id, status="queued")
                print(f"  started {tag} -> {job.job_id}", flush=True)
                free -= 1
            except Exception as e:
                print(f"  {tag}: could not start "
                      f"({type(e).__name__}: {str(e)[:70]})", flush=True)

        if pending:
            time.sleep(args.poll_seconds)

    print(f"batch done: {len(done) + len(keys) - len(pending) - n_err} "
          f"collected, {n_err} failed")
    if n_err:
        print(f"WARNING: {n_err} group(s) failed — coverage is partial",
              file=sys.stderr)


if __name__ == "__main__":
    main()
