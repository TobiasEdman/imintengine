#!/usr/bin/env python3
"""
Monitor ColonyOS seasonal fetch jobs and write dashboard-compatible JSON.

Polls ColonyOS process state every N seconds, aggregates per-source
(CDSE/DES) statistics, and writes seasonal_fetch_log.json for the
training dashboard to display.

Usage:
    python scripts/monitor_seasonal_jobs.py \\
        --data-dir ~/training_data \\
        --total-tiles 4381 \\
        --interval 15
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────


def _write_log(path: Path, log: dict) -> None:
    """Atomic write (tmp + rename), same pattern as _write_prepare_log."""
    try:
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(log, f, indent=2)
        tmp.rename(path)
    except Exception as exc:
        print(f"  Warning: failed to write {path}: {exc}")


def _colonies_cmd(subcmd: list[str], timeout: int = 60) -> list[dict] | None:
    """Run a colonies CLI command with --json and parse output."""
    cmd = ["colonies"] + subcmd + ["--json", "--insecure"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return None
        out = result.stdout.strip()
        if not out or out == "null":
            return []
        return json.loads(out)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as exc:
        print(f"  Warning: {' '.join(cmd[:3])} failed: {exc}")
        return None


def _parse_iso(ts: str) -> datetime | None:
    """Parse an ISO 8601 timestamp from ColonyOS."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _query_all_processes() -> dict[str, list[dict]]:
    """Query waiting, running, successful, and failed processes."""
    states = {}
    for state, cmd in [
        ("waiting", ["process", "psw"]),
        ("running", ["process", "ps"]),
        ("successful", ["process", "pss", "--count", "10000"]),
        ("failed", ["process", "psf", "--count", "10000"]),
    ]:
        procs = _colonies_cmd(cmd)
        states[state] = procs if procs is not None else []
    return states


def _is_seasonal(proc: dict) -> bool:
    """Check if a process is a seasonal-tile-fetch job."""
    spec = proc.get("spec", proc)
    funcname = spec.get("funcname", "")
    return funcname == "seasonal-tile-fetch"


def _get_env(proc: dict) -> dict:
    """Extract env vars from a process spec."""
    spec = proc.get("spec", proc)
    return spec.get("env", {})


def _get_source(proc: dict) -> str:
    """Get the fetch source (copernicus or des) from process env."""
    env = _get_env(proc)
    raw = env.get("FETCH_SOURCE", "copernicus")
    return "copernicus" if raw == "copernicus" else "des"


def _get_cell_key(proc: dict) -> str:
    """Get tile cell key from process env."""
    env = _get_env(proc)
    return f"{env.get('EASTING', '?')}_{env.get('NORTHING', '?')}"


def _get_elapsed(proc: dict) -> float | None:
    """Compute wall-clock elapsed time from process timestamps."""
    start = _parse_iso(proc.get("starttime", ""))
    end = _parse_iso(proc.get("endtime", ""))
    if start and end:
        return max(0, (end - start).total_seconds())
    return None


# ── Main aggregation ─────────────────────────────────────────────────────


def build_stats(
    all_procs: dict[str, list[dict]],
    total_tiles: int,
    started_at: str,
) -> dict:
    """Aggregate per-source stats from ColonyOS process lists."""

    sources: dict[str, dict] = {}

    def _ensure_source(name: str) -> dict:
        if name not in sources:
            sources[name] = {
                "submitted": 0,
                "completed": 0,
                "failed": 0,
                "running": 0,
                "total_time_s": 0.0,
                "avg_time_s": 0.0,
                "success_rate": 0.0,
                "recent_failures": [],
            }
        return sources[name]

    recent_tiles: list[dict] = []

    for state_name, procs in all_procs.items():
        for proc in procs:
            if not _is_seasonal(proc):
                continue
            src = _get_source(proc)
            s = _ensure_source(src)
            s["submitted"] += 1

            if state_name == "successful":
                s["completed"] += 1
                elapsed = _get_elapsed(proc)
                if elapsed is not None:
                    s["total_time_s"] += elapsed
                    recent_tiles.append({
                        "key": _get_cell_key(proc),
                        "source": src,
                        "elapsed_s": round(elapsed, 1),
                    })

            elif state_name == "failed":
                s["failed"] += 1
                key = _get_cell_key(proc)
                s["recent_failures"].append(key)
                s["recent_failures"] = s["recent_failures"][-5:]

            elif state_name == "running":
                s["running"] += 1

    # Compute derived stats
    for s in sources.values():
        done = s["completed"]
        s["avg_time_s"] = round(s["total_time_s"] / max(done, 1), 1)
        total_decided = done + s["failed"]
        s["success_rate"] = round(done / max(total_decided, 1), 3)

    total_completed = sum(s["completed"] for s in sources.values())
    total_failed = sum(s["failed"] for s in sources.values())
    total_running = sum(s["running"] for s in sources.values())
    total_pending = total_tiles - total_completed - total_failed - total_running

    # ETA and rate
    now = datetime.now(timezone.utc)
    start_dt = _parse_iso(started_at) or now
    elapsed_s = max(1, (now - start_dt).total_seconds())
    rate = total_completed / (elapsed_s / 3600) if total_completed > 0 else 0
    remaining = max(0, total_tiles - total_completed - total_failed)
    eta_s = (remaining / rate * 3600) if rate > 0 else 0

    status = "completed" if (total_completed + total_failed) >= total_tiles else (
        "running" if (total_running > 0 or total_completed > 0) else "waiting"
    )

    return {
        "status": status,
        "total_tiles": total_tiles,
        "completed": total_completed,
        "failed": total_failed,
        "running": total_running,
        "pending": max(0, total_pending),
        "started_at": started_at,
        "updated_at": now.isoformat(),
        "elapsed_s": round(elapsed_s, 1),
        "sources": sources,
        "rate_tiles_per_hour": round(rate, 1),
        "eta_s": round(eta_s),
        "recent_tiles": recent_tiles[-15:],
    }


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Monitor ColonyOS seasonal fetch jobs",
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory where seasonal_fetch_log.json is written (dashboard reads from here)",
    )
    parser.add_argument(
        "--total-tiles", type=int, default=4381,
        help="Total number of tiles to fetch (default: 4381 = full Sweden grid)",
    )
    parser.add_argument(
        "--interval", type=int, default=15,
        help="Poll interval in seconds (default: 15)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run once and exit (for testing)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "seasonal_fetch_log.json"
    started_at = datetime.now(timezone.utc).isoformat()

    # Restore started_at from existing log if present
    if log_path.exists():
        try:
            existing = json.loads(log_path.read_text())
            started_at = existing.get("started_at", started_at)
        except (json.JSONDecodeError, OSError):
            pass

    print(f"Seasonal Fetch Monitor")
    print(f"  Data dir: {data_dir}")
    print(f"  Total tiles: {args.total_tiles}")
    print(f"  Poll interval: {args.interval}s")
    print(f"  Log file: {log_path}")
    print()

    iteration = 0
    while True:
        iteration += 1
        t0 = time.monotonic()

        all_procs = _query_all_processes()
        stats = build_stats(all_procs, args.total_tiles, started_at)
        _write_log(log_path, stats)

        dt = time.monotonic() - t0
        c = stats["completed"]
        f = stats["failed"]
        r = stats["running"]
        rate = stats["rate_tiles_per_hour"]
        print(
            f"  [{iteration:>4}] "
            f"done={c} running={r} failed={f} "
            f"rate={rate:.0f}/h "
            f"({dt:.1f}s)"
        )

        # Per-source one-liner
        for name, src in stats.get("sources", {}).items():
            label = "CDSE" if name == "copernicus" else "DES"
            print(
                f"         {label}: "
                f"{src['completed']}/{src['submitted']} "
                f"avg={src['avg_time_s']:.0f}s "
                f"ok={src['success_rate']:.1%}"
            )

        if args.once or stats["status"] == "completed":
            if stats["status"] == "completed":
                print(f"\n  All tiles processed! ({c} done, {f} failed)")
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
