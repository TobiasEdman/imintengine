"""Minimum reproducible example — DES openEO `aggregate_spatial` NaN
serialization 500.

Companion to ``2026-07-07_aggregate_spatial_fix_proposal.md`` (section
"LIVE STATUS"). The aggregation itself completes on the current worker;
the 500 comes from the driver's response layer: the result dict's
``attrs`` carry ``nodata: NaN`` and starlette's ``JSONResponse`` encodes
with ``allow_nan=False``.

Two parts:

  Part 1 (default)   — pure python, no openEO, no credentials, <1 s.
                       Reproduces the exact ValueError in starlette's
                       own render path, then shows the proposed
                       ``_json_safe`` guard making the same payload pass.

  Part 2 (--live)    — the real minimal graph against
                       openeo.digitalearth.se (needs OPENEO/DES creds in
                       env: DES_USER / DES_PASSWORD). Shows the
                       production HTTP 500 with the same traceback tail.

Usage:
    python mre_aggregate_spatial_nan.py            # part 1 only
    python mre_aggregate_spatial_nan.py --live     # part 1 + 2

Requires: starlette (part 1); openeo + shapely (part 2).
"""
from __future__ import annotations

import math
import sys


def _json_safe(o):
    """Proposed fix for openeo_des_driver/apis/data_processing_api.py:
    map non-finite floats to null before JSONResponse."""
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def part1_starlette() -> None:
    from starlette.responses import JSONResponse

    # Shape mirrors the failing production payload: an xarray-style dict
    # whose attrs carry the cube's nodata as NaN.
    payload = {
        "2018-06-13": [[0.041]],
        "attrs": {"nodata": float("nan"), "crs": "EPSG:4326"},
    }

    print("Part 1 — starlette render of a NaN-carrying result dict:")
    try:
        JSONResponse(payload).body
        print("  UNEXPECTED: no error — starlette version may allow NaN")
        sys.exit(2)
    except ValueError as e:
        print(f"  REPRODUCED - ValueError: {e}")

    body = JSONResponse(_json_safe(payload)).body
    print(f"  WITH _json_safe GUARD: renders fine -> {body.decode()[:80]}...")


def part2_live() -> None:
    import os

    import openeo
    from shapely.geometry import box, mapping

    user = os.environ.get("DES_USER") or os.environ.get("OPENEO_USERNAME")
    pw = os.environ.get("DES_PASSWORD") or os.environ.get("OPENEO_PASSWORD")
    if not user or not pw:
        print("Part 2 — skipped: set DES_USER / DES_PASSWORD")
        return

    print("Part 2 — live minimal graph vs openeo.digitalearth.se:")
    conn = openeo.connect("https://openeo.digitalearth.se/")
    conn.authenticate_basic(username=user, password=pw)

    west, south, east, north = 13.79, 60.83, 13.89, 60.87
    scl = conn.load_collection(
        "s2_msi_l2a",
        spatial_extent={"west": west, "south": south,
                        "east": east, "north": north, "crs": "EPSG:4326"},
        temporal_extent=["2018-06-01", "2018-06-30"],
        bands=["scl"],
    )
    b = scl.band("scl")
    cloud = (b == 3) | (b == 8) | (b == 9) | (b == 10)
    try:
        out = cloud.aggregate_spatial(
            geometries=mapping(box(west, south, east, north)),
            reducer="mean",
        ).execute()
        print(f"  UNEXPECTED SUCCESS (fixed?): {str(out)[:200]}")
    except Exception as e:
        msg = str(e)
        hit = "not JSON compliant: nan" in msg
        print(f"  {'REPRODUCED' if hit else 'DIFFERENT ERROR'} - "
              f"{type(e).__name__}: {msg[-400:] if hit else msg[:400]}")


if __name__ == "__main__":
    part1_starlette()
    if "--live" in sys.argv:
        part2_live()
