"""Sentinel-1 GRD fetch façade — CDSE → MPC → AWS cascade.

Single entry point that picks the cheapest available backend per call.
Order is fixed by the operational reality of mid-2026:

    1. ``cdse_s1_stac``  — primary. CDSE STAC + COG, no PU billing.
    2. ``mpc_s1``         — fallback. Microsoft Planetary Computer signed
                            URLs, anonymous, free, fastest second choice.
    3. ``aws_s1``         — last resort. Element84 STAC + s3://sentinel-s1-l1c.
                            Bucket is requester-pays so this costs us
                            $0.09/GB egress. Skipped automatically if AWS
                            credentials are not configured.

Each backend is tried in order; the first one that returns a non-``None``
result wins. A failure (exception or ``None``) is logged and the next
backend is attempted. The final return matches each backend's contract:
``(sar, orbit_direction)`` on success, ``None`` if every backend declined.

Backend selection can be overridden via ``S1_BACKENDS`` env var as a comma-
separated list (e.g. ``"mpc,cdse"`` or ``"cdse"``). Useful for stress-testing
a single backend or working around a known outage.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from . import aws_s1, cdse_s1_stac, mpc_s1


_DEFAULT_ORDER = ("cdse", "mpc", "aws")

_BACKENDS: dict[str, Callable[..., Any]] = {
    "cdse": cdse_s1_stac.fetch_s1_scene,
    "mpc": mpc_s1.fetch_s1_scene,
    "aws": aws_s1.fetch_s1_scene,
}


def _resolve_order() -> tuple[str, ...]:
    env = os.environ.get("S1_BACKENDS", "").strip()
    if not env:
        return _DEFAULT_ORDER
    chosen = tuple(name.strip().lower() for name in env.split(",") if name.strip())
    unknown = [n for n in chosen if n not in _BACKENDS]
    if unknown:
        raise ValueError(
            f"S1_BACKENDS contains unknown backend(s): {unknown}. "
            f"Known: {sorted(_BACKENDS)}"
        )
    return chosen


def fetch_s1_scene(
    west: float,
    south: float,
    east: float,
    north: float,
    date: str,
    *,
    crs: str = "http://www.opengis.net/def/crs/EPSG/0/3006",
    size_px: int | tuple[int, int] = 256,
    orbit_direction: str | None = None,
    output_db: bool = False,
    nodata_threshold: float | None = 0.10,
) -> tuple[np.ndarray, str] | None:
    """Try each backend in cascade order; return the first hit.

    Args / return contract identical to the underlying backends — see
    :func:`cdse_s1_stac.fetch_s1_scene` for the canonical definition.
    """
    last_errors: list[str] = []
    for backend_name in _resolve_order():
        fetch = _BACKENDS[backend_name]
        try:
            result = fetch(
                west, south, east, north, date,
                crs=crs,
                size_px=size_px,
                orbit_direction=orbit_direction,
                output_db=output_db,
                nodata_threshold=nodata_threshold,
            )
        except RuntimeError as e:
            # Misconfiguration (e.g. missing AWS creds) — skip backend
            # without aborting the cascade.
            last_errors.append(f"{backend_name}: {e}")
            continue
        except Exception as e:
            last_errors.append(f"{backend_name}: {type(e).__name__}: {e}")
            continue

        if result is not None:
            if backend_name != _DEFAULT_ORDER[0]:
                # Surface fallback usage so operators see the cascade in action.
                print(f"    [s1_fetch] backend={backend_name} (fallback used)")
            return result
        last_errors.append(f"{backend_name}: returned None")

    if last_errors:
        print("    [s1_fetch] all backends declined: " + " | ".join(last_errors))
    return None
