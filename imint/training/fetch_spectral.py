"""Unified per-slot spectral fetch dispatcher.

ONE entry point — :func:`fetch_spectral` — for every supported backend.
The orchestrator (``scripts/fetch_unified_tiles.py::repair_to_canonical_layout``)
hands over ``(bbox, coords, date, backend, size_px, cloud_threshold)`` and
gets back a ``(6, H, W)`` float32 reflectance array or ``None``. Backend
selection is the orchestrator's responsibility; this module only executes
the per-slot fetch for the backend it is told to use.

Design contract — read this before adding code here:

* ``None`` means **"this date is not usable for this slot via this
  backend"** — either cloud-rejected, no scene available, the backend
  is dead, or it errored. The orchestrator treats every ``None`` the
  same: advance to the next ranked candidate. **Exceptions are logged
  visibly** (no silent ``except Exception: pass`` — that was the
  pre-refactor failure mode that lost an entire workday).

* **Verify + fetch lives here, NOT in the orchestrator.** Per-backend
  verify nuance is the adapter's job:
  - ``cdse`` (SH Process Process API) — ``fetch_s2_scene`` two-stages
    SCL prescreen + spectral fetch in one HTTP cycle.
  - ``cdse-openeo`` / ``des`` (openEO) — explicit
    ``verify_aoi_scl`` first (one cheap SCL call), then
    ``fetch_tile_at_specific_dates`` only if AOI-clean (saves the
    expensive openEO spectral call on cloudy rejects).

* **No backend race, no cross-backend fallback.** Single backend per
  call. If a slot exhausts its candidate list on one backend, the slot
  fails. Backend health is surfaced cleanly via ``is_source_dead``
  (402 PaymentRequired marks ``cdse-openeo`` dead for the process).

The unified shape is what makes "no more silent fall-throughs to a
different code path" the architectural property of the system, not a
discipline the next maintainer has to remember.
"""
from __future__ import annotations

import numpy as np

from imint.training.cdse_s2 import fetch_s2_scene
from imint.training.openeo_tile_graph import (
    _is_payment_required_error,
    fetch_tile_at_specific_dates,
    is_source_dead,
    mark_source_dead,
)
from imint.training.optimal_fetch import verify_aoi_scl
from imint.training.tile_fetch import (
    _CDSE_OPENEO_SEMAPHORE,
    _CDSE_SEMAPHORE,
    _DES_SEMAPHORE,
)

SUPPORTED_BACKENDS = ("cdse", "cdse-openeo", "des")


def fetch_spectral(
    bbox_3006: dict,
    coords_wgs84: dict,
    date_str: str,
    *,
    backend: str,
    size_px: int,
    cloud_threshold: float,
) -> np.ndarray | None:
    """Fetch one slot's spectral via ``backend`` on ``date_str``.

    Args:
        bbox_3006: SWEREF99 TM bbox dict with keys ``west/south/east/north``.
        coords_wgs84: WGS84 coords dict (required by ``verify_aoi_scl``
            on openEO backends).
        date_str: ISO date ``YYYY-MM-DD`` of the S2 acquisition to fetch.
        backend: One of :data:`SUPPORTED_BACKENDS`.
        size_px: Tile edge in pixels; bbox must be ``size_px × 10 m`` aligned.
        cloud_threshold: AOI cloud-fraction ceiling (0–1). Dates with
            higher cloud cover (per the backend's own SCL check) return
            ``None``.

    Returns:
        ``(6, size_px, size_px)`` float32 reflectance array, or ``None``
        if the date is cloud-rejected / no scene available / backend dead
        / an error occurred (errors are logged but not raised).
    """
    if backend == "cdse":
        if is_source_dead("cdse"):
            return None  # SH Process PUs exhausted this session — fail fast.
        return _fetch_via_cdse_sh_process(
            bbox_3006, date_str,
            size_px=size_px, cloud_threshold=cloud_threshold,
        )
    if backend == "cdse-openeo":
        if is_source_dead("cdse-openeo"):
            return None
        return _fetch_via_openeo(
            bbox_3006, coords_wgs84, date_str,
            backend="cdse-openeo",
            semaphore=_CDSE_OPENEO_SEMAPHORE,
            cloud_threshold=cloud_threshold,
        )
    if backend == "des":
        return _fetch_via_openeo(
            bbox_3006, coords_wgs84, date_str,
            backend="des",
            semaphore=_DES_SEMAPHORE,
            cloud_threshold=cloud_threshold,
        )
    raise ValueError(
        f"fetch_spectral: unknown backend {backend!r}. "
        f"Supported: {SUPPORTED_BACKENDS}."
    )


# ── Backend adapters ────────────────────────────────────────────────────────


def _fetch_via_cdse_sh_process(
    bbox_3006: dict,
    date_str: str,
    *,
    size_px: int,
    cloud_threshold: float,
) -> np.ndarray | None:
    """SH Process Process API — fused SCL prescreen + spectral fetch.

    ``fetch_s2_scene`` runs two-stage internally for tiles ≥ 384 px:
    one HTTP for SCL prescreen, then one HTTP for spectral only if the
    AOI cloud fraction passes ``cloud_threshold``. Saves ~6× bandwidth on
    cloudy rejects. For tiles < 384 px the threshold is not applied —
    that is a property of ``fetch_s2_scene`` and out of scope for this
    dispatcher.
    """
    _CDSE_SEMAPHORE.acquire()
    try:
        result = fetch_s2_scene(
            bbox_3006["west"], bbox_3006["south"],
            bbox_3006["east"], bbox_3006["north"],
            date=date_str,
            size_px=size_px,
            cloud_threshold=cloud_threshold,
        )
        _CDSE_SEMAPHORE.report_success()
        if result is None:
            return None
        spectral = result[0]
        return spectral if np.any(spectral) else None
    except Exception as e:
        _CDSE_SEMAPHORE.report_failure()
        print(
            f"    [fetch_spectral:cdse] {date_str}: "
            f"{type(e).__name__}: {str(e)[:200]}",
            flush=True,
        )
        return None
    finally:
        _CDSE_SEMAPHORE.release()


def _fetch_via_openeo(
    bbox_3006: dict,
    coords_wgs84: dict,
    date_str: str,
    *,
    backend: str,
    semaphore,
    cloud_threshold: float,
) -> np.ndarray | None:
    """CDSE openEO or DES openEO — explicit verify-then-fetch.

    openEO calls are expensive (credit / connection slot per process
    graph), so we always SCL-verify first and only submit the spectral
    process graph if the AOI is clean. The two semaphores serialise
    concurrent access to the backend's hard limits.
    """
    # 1) Cheap SCL verify (one openEO call). Bail on cloudy dates so we
    #    never submit a spectral graph we know will be wasted.
    try:
        cloud = verify_aoi_scl(coords_wgs84, date_str, backend=backend)
    except Exception as e:
        print(
            f"    [fetch_spectral:{backend}] verify {date_str}: "
            f"{type(e).__name__}: {str(e)[:200]}",
            flush=True,
        )
        return None
    if cloud is None or cloud > cloud_threshold:
        return None

    # 2) Spectral fetch via tile-graph (single-slot dict). Sub-second
    #    serialisation on cdse-openeo (1 perm), 6-fan-out on DES.
    semaphore.acquire()
    try:
        result = fetch_tile_at_specific_dates(
            bbox_3006, {0: date_str}, source=backend,
        )
        semaphore.report_success()
        entry = result.get(0)
        if entry is None:
            return None
        spectral = entry[0]
        return spectral if np.any(spectral) else None
    except Exception as e:
        semaphore.report_failure()
        if backend == "cdse-openeo" and _is_payment_required_error(e):
            mark_source_dead(
                "cdse-openeo",
                f"402 in fetch_spectral: {str(e)[:160]}",
            )
        print(
            f"    [fetch_spectral:{backend}] fetch {date_str}: "
            f"{type(e).__name__}: {str(e)[:200]}",
            flush=True,
        )
        return None
    finally:
        semaphore.release()
