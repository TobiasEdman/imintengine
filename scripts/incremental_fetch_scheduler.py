#!/usr/bin/env python3
"""Incremental tile-fetch scheduler.

Per agentic_workflow rollout plan W4.2.

Compares LPIS / NMD reference data timestamps against the local tile
manifest. Determines which tiles need re-fetch (LPIS updated since the
tile's last fetch) and emits the list. The K8s CronJob feeds this list
to scripts/fetch_unified_tiles.py.

Pure-stdlib + pathlib. No torch / no rasterio at scheduling time.

Outputs:
  --json   list of tile IDs as JSON to stdout (machine-readable)
  --txt    one tile ID per line (human-readable; default for piping)

Exit codes:
  0  list emitted (may be empty)
  2  reference data missing or unreadable

Optional Pushgateway emission (for Prometheus integration):
  --pushgateway URL    push fetch_planned_total + last_run_timestamp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _scan_manifest(manifest_path: Path) -> dict[str, float]:
    """Return {tile_id: last_fetched_unix_ts}. Missing manifest → empty."""
    if not manifest_path.exists():
        return {}
    try:
        return {
            entry["tile_id"]: float(entry["last_fetched"])
            for entry in json.loads(manifest_path.read_text())
            if "tile_id" in entry and "last_fetched" in entry
        }
    except (OSError, ValueError, KeyError) as e:
        print(f"WARN: tile manifest at {manifest_path} unreadable: {e}", file=sys.stderr)
        return {}


def _scan_lpis_mtimes(lpis_dir: Path) -> dict[str, float]:
    """Return {tile_id: lpis_file_mtime} for every tile that has an LPIS parcel file.

    LPIS files are named `<tile_id>.parquet` in the layout the fetch pipeline
    produces. mtime is sufficient — Jordbruksverket releases come as full files.
    """
    if not lpis_dir.exists():
        return {}
    out: dict[str, float] = {}
    for f in lpis_dir.glob("*.parquet"):
        out[f.stem] = f.stat().st_mtime
    return out


def needs_refetch(
    last_fetched_ts: dict[str, float],
    reference_mtimes: dict[str, float],
    lookback_days: int = 7,
) -> list[str]:
    """Return tile IDs whose reference data has been updated since last fetch,
    OR which were updated within the lookback window even if previously fetched.

    The lookback window catches the case where reference data was updated
    after the previous CronJob run but before the new tile manifest was
    written, by re-checking anything touched in the last N days.
    """
    cutoff = time.time() - lookback_days * 86_400
    targets: list[str] = []
    for tile_id, ref_mtime in reference_mtimes.items():
        last = last_fetched_ts.get(tile_id, 0.0)
        if ref_mtime > last or ref_mtime > cutoff and last < ref_mtime:
            targets.append(tile_id)
    targets.sort()
    return targets


def _push_metrics(
    url: str,
    *,
    job: str,
    planned: int,
    elapsed_ms: float,
) -> None:
    """Best-effort Prometheus Pushgateway emission. Never raises."""
    try:
        import urllib.request

        body = "\n".join(
            [
                "# TYPE imint_fetch_planned_total counter",
                f"imint_fetch_planned_total {planned}",
                "# TYPE imint_fetch_scheduler_last_run_timestamp gauge",
                f"imint_fetch_scheduler_last_run_timestamp {int(time.time())}",
                "# TYPE imint_fetch_scheduler_duration_ms gauge",
                f"imint_fetch_scheduler_duration_ms {elapsed_ms:.1f}",
                "",
            ]
        )
        req = urllib.request.Request(
            f"{url.rstrip('/')}/metrics/job/{job}",
            data=body.encode("utf-8"),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status >= 400:
                print(
                    f"WARN: Pushgateway returned {resp.status}",
                    file=sys.stderr,
                )
    except Exception as e:  # pragma: no cover
        print(f"WARN: Pushgateway push failed: {e}", file=sys.stderr)


def _emit(targets: Iterable[str], fmt: str) -> None:
    if fmt == "json":
        json.dump(list(targets), sys.stdout)
        sys.stdout.write("\n")
    else:
        for t in targets:
            print(t)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lpis-dir", type=Path, required=True,
                   help="Directory of LPIS parcel parquets, named <tile_id>.parquet")
    p.add_argument("--manifest", type=Path, required=True,
                   help="Tile manifest JSON: list of {tile_id, last_fetched} entries")
    p.add_argument("--lookback-days", type=int, default=7,
                   help="Re-check tiles whose reference data was updated in the last N days")
    p.add_argument("--format", choices=["json", "txt"], default="txt")
    p.add_argument("--pushgateway", type=str, default=None,
                   help="Prometheus Pushgateway URL, e.g. http://pushgateway:9091")
    p.add_argument("--job-name", type=str, default="imint_fetch_scheduler",
                   help="Pushgateway job name")
    args = p.parse_args()

    start = time.monotonic()
    last_fetched = _scan_manifest(args.manifest)
    refs = _scan_lpis_mtimes(args.lpis_dir)
    if not refs:
        print(f"ERROR: no LPIS files found under {args.lpis_dir}", file=sys.stderr)
        return 2

    targets = needs_refetch(last_fetched, refs, lookback_days=args.lookback_days)

    elapsed_ms = (time.monotonic() - start) * 1000
    print(
        f"# scanned {len(refs)} LPIS tiles, "
        f"{len(last_fetched)} in manifest, "
        f"{len(targets)} need refetch "
        f"(lookback={args.lookback_days}d, took {elapsed_ms:.1f}ms)",
        file=sys.stderr,
    )

    if args.pushgateway:
        _push_metrics(
            args.pushgateway,
            job=args.job_name,
            planned=len(targets),
            elapsed_ms=elapsed_ms,
        )

    _emit(targets, args.format)
    return 0


if __name__ == "__main__":
    sys.exit(main())
