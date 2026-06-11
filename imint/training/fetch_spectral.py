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
    ALL_BANDS,
    ALL_BANDS_INDEX,
    _is_payment_required_error,
    fetch_tile_at_specific_dates,
    is_source_dead,
    mark_source_dead,
)


def _split_all_bands(arr: np.ndarray, collect_extra: dict | None) -> np.ndarray:
    """Split a full-band slot array into the 6-band Prithvi cube + extras.

    The openEO fetch now returns all 12 S2 bands in ``ALL_BANDS`` order.
    Return the 6-band Prithvi stack (model input, unchanged) and, when
    ``collect_extra`` is provided, populate it with the per-band extras for
    this slot: ``b08`` (H,W), ``rededge`` (3,H,W), ``b01`` (H,W),
    ``b09`` (H,W). A 6-band array (CDSE-SH path, not yet all-bands) is
    returned as-is with no extras.
    """
    if arr.shape[0] != len(ALL_BANDS):
        return arr  # already 6-band (legacy / CDSE-SH) — no extras
    spectral = arr[list(ALL_BANDS_INDEX["prithvi"])]
    if collect_extra is not None:
        collect_extra["b08"] = arr[ALL_BANDS_INDEX["b08"][0]]
        collect_extra["rededge"] = arr[list(ALL_BANDS_INDEX["rededge"])]
        collect_extra["b01"] = arr[ALL_BANDS_INDEX["b01"][0]]
        collect_extra["b09"] = arr[ALL_BANDS_INDEX["b09"][0]]
    return spectral
from imint.training.optimal_fetch import verify_aoi_scl
from imint.training.tile_fetch import (
    _CDSE_OPENEO_SEMAPHORE,
    _CDSE_SEMAPHORE,
    _DES_SEMAPHORE,
)
from imint.coregistration import _COREG_BAND, clearest_frame_idx, coregister_interframe
from imint.training.tile_assemble import assemble_fresh, crop_halo, date_to_doy
from imint.training.tile_config import TileConfig

SUPPORTED_BACKENDS = ("cdse", "cdse-openeo", "des")

# Backends whose openEO tile-graph returns all-band frames on a snapped halo grid
# — the precondition for M2 (inter-frame coreg needs 4 frames on one shared grid).
# SH-Process is 6-band and renders to the request grid (no halo) → cannot M2.
_M2_CAPABLE_BACKENDS = ("des", "cdse-openeo")


def fetch_spectral(
    bbox_3006: dict,
    coords_wgs84: dict,
    date_str: str,
    *,
    backend: str,
    size_px: int,
    cloud_threshold: float,
    collect_extra: dict | None = None,
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
            collect_extra=collect_extra,
        )
    if backend == "des":
        return _fetch_via_openeo(
            bbox_3006, coords_wgs84, date_str,
            backend="des",
            semaphore=_DES_SEMAPHORE,
            cloud_threshold=cloud_threshold,
            collect_extra=collect_extra,
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
            # fetch_s2_scene returns None for both cloud-rejection AND
            # no-scene-available (the two-stage prescreen path). Can't
            # distinguish from here without instrumenting fetch_s2_scene.
            print(
                f"    [fetch_spectral:cdse] result-none {date_str}: "
                f"two-stage returned None (cloud reject or no scene)",
                flush=True,
            )
            return None
        spectral = result[0]
        if not np.any(spectral):
            print(
                f"    [fetch_spectral:cdse] result-zero {date_str}: "
                f"spectral all-zero (degenerate response)",
                flush=True,
            )
            return None
        return spectral
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
    collect_extra: dict | None = None,
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
    if cloud is None:
        # SCL returned None — the date isn't in the screening dict.
        # Distinct from exception (logged above): the call succeeded
        # but the backend had no data for this date+1 window.
        print(
            f"    [fetch_spectral:{backend}] verify-none {date_str}: "
            f"SCL dict missing date (likely no-scene / STAC↔openEO mismatch)",
            flush=True,
        )
        return None
    if cloud > cloud_threshold:
        # Normal cloud rejection — log compact so the per-tile fan-out
        # stays readable.
        print(
            f"    [fetch_spectral:{backend}] verify-cloud {date_str}: "
            f"{cloud:.2f} > {cloud_threshold:.2f}",
            flush=True,
        )
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
            print(
                f"    [fetch_spectral:{backend}] fetch-none {date_str}: "
                f"tile-graph returned no entry for slot 0",
                flush=True,
            )
            return None
        # Split full-band (12) → 6-band Prithvi cube + per-band extras.
        spectral = _split_all_bands(entry[0], collect_extra)
        if not np.any(spectral):
            print(
                f"    [fetch_spectral:{backend}] fetch-zero {date_str}: "
                f"spectral all-zero (degenerate response)",
                flush=True,
            )
            return None
        return spectral
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


# ── Per-tile canonical entry (M1 + M2) ──────────────────────────────────────


def fetch_tile_spectral(
    center_3006: tuple[int, int],
    *,
    tile: TileConfig,
    dates: dict[int, str],
    n_frames: int,
    backend: str = "des",
    halo_px: int = 8,
    coregister: bool = True,
) -> dict | None:
    """Canonical per-tile spectral fetch — M1 (grid-snap) + M2 (inter-frame MI).

    Promotes ``scripts/regrid_national_512.py::regrid_one_tile``'s proven
    M1→M2→crop→assemble composition into the library so it is the ONE production
    fetch path. All temporal slots are fetched on a halo grid in a single
    tile-graph download (M1 snaps each scene's transform to the halo bbox); the
    frames are coregistered to the clearest one by mutual information (M2), the
    halo is cropped away, and the ``(T*6, H, W)`` cube + all-band extras are
    assembled.

    Args:
        center_3006: ``(easting, northing)`` in EPSG:3006. Snapped to the 10 m
            lattice; the canonical and halo bboxes share the snapped centre so the
            inner crop is co-centred with the canonical bbox.
        tile: ``TileConfig`` defining the canonical edge ``size_px`` (e.g. 512).
        dates: ``{slot: ISO-date}`` for the temporal frames to fetch.
        n_frames: number of temporal slots in the assembled cube.
        backend: ``"des"`` or ``"cdse-openeo"`` — an M2-capable all-band openEO
            tile-graph backend (SH-Process is 6-band/no-halo and cannot M2).
        halo_px: total halo (``2*crop``) added to the fetch extent and cropped
            away after M2; absorbs the sinc wrap. Must be even.
        coregister: apply M2 inter-frame coregistration (default ``True``).

    Returns:
        A result dict — ``spectral``/``temporal_mask``/``doy``/``dates`` + the
        ``b08``/``rededge``/``b01``/``b09`` extras + geometry/provenance
        (``bbox_3006``/``easting``/``northing``/``tile_size_px``/``source``/
        ``coreg_ref_frame``/``coreg_m2``) — ready for a caller to persist, or
        ``None`` if no slot fetched.
    """
    if backend not in _M2_CAPABLE_BACKENDS:
        raise ValueError(
            f"fetch_tile_spectral: backend {backend!r} cannot do M1+M2 — use one "
            f"of {_M2_CAPABLE_BACKENDS} (SH-Process is 6-band/no-halo)."
        )
    if halo_px % 2 != 0:
        raise ValueError(f"fetch_tile_spectral: halo_px must be even, got {halo_px}.")

    canon = tile.size_px
    crop = halo_px // 2
    halo = canon + halo_px
    cx0, cy0 = center_3006

    # M1 geometry: canonical and halo bboxes from the SAME snapped centre, so the
    # inner crop is co-centred with the canonical bbox.
    bbox_canon = TileConfig(size_px=canon, gsd_m=tile.gsd_m).bbox_from_center(cx0, cy0)
    bbox_halo = TileConfig(size_px=halo, gsd_m=tile.gsd_m).bbox_from_center(cx0, cy0)
    cx_new = (bbox_canon["west"] + bbox_canon["east"]) // 2
    cy_new = (bbox_canon["south"] + bbox_canon["north"]) // 2

    slot_dates = {fi: d for fi, d in dates.items() if d}
    if not slot_dates:
        return None

    # Fetch all slots on the HALO grid in ONE tile-graph download. M1 (the
    # per-scene transform snap to the halo bbox) happens inside the tile-graph.
    res = fetch_tile_at_specific_dates(bbox_halo, slot_dates, source=backend)

    fresh: dict[int, np.ndarray] = {}
    for fi, entry in res.items():
        if entry is None or entry[0] is None:
            continue
        arr = np.asarray(entry[0], np.float32)
        if arr.shape != (len(ALL_BANDS), halo, halo):
            continue
        fresh[fi] = arr
    if not fresh:
        return None

    # M2 — inter-frame MI coreg on the shared halo grid, before the crop. The
    # search budget is the halo width (the non-central reference can see up to
    # ~the halo of pairwise drift); applied shifts are halo-bounded by the crop.
    did_m2 = coregister and len(fresh) >= 2
    if did_m2:
        ref_idx = clearest_frame_idx(fresh)
        fresh, shifts = coregister_interframe(fresh, ref_idx, search_px=float(crop))
    else:
        ref_idx = next(iter(fresh))
        shifts = {ref_idx: (0.0, 0.0)}

    # Coreg-quality signals — flag low-confidence coreg, don't block (the dataset
    # loader / audit decides). anchor_valid_frac on B04 (clouds flatten/zero it);
    # n_aligned = movers given a real MI shift; per-slot applied shifts persisted.
    anchor_valid_frac = float((fresh[ref_idx][_COREG_BAND] > 1e-6).mean())
    coreg_n_aligned = int(sum(
        1 for fi, (dy, dx) in shifts.items()
        if fi != ref_idx and abs(dy) + abs(dx) > 0.0))
    coreg_max_shift = float(max(
        (float(np.hypot(dy, dx)) for dy, dx in shifts.values()), default=0.0))

    cropped = {fi: crop_halo(a, crop=crop, canon=canon) for fi, a in fresh.items()}

    dates_list = [slot_dates.get(fi, "") for fi in range(n_frames)]
    spectral, extras = assemble_fresh(cropped, dates_list, n_frames, canon=canon)
    temporal_mask = np.array(
        [1 if fi in cropped else 0 for fi in range(n_frames)], np.uint8)
    doy = np.array([date_to_doy(dates_list[fi]) for fi in range(n_frames)], np.int32)
    out_dates = np.array(
        [dates_list[fi] if fi in cropped else "" for fi in range(n_frames)])

    return {
        "spectral": spectral,
        "temporal_mask": temporal_mask,
        "doy": doy,
        "dates": out_dates,
        "multitemporal": np.int32(1),
        "num_frames": np.int32(n_frames),
        "num_bands": np.int32(6),
        "bbox_3006": np.array(
            [bbox_canon["west"], bbox_canon["south"],
             bbox_canon["east"], bbox_canon["north"]], dtype=np.int32),
        "easting": np.int32(cx_new),
        "northing": np.int32(cy_new),
        "tile_size_px": np.int32(canon),
        "source": backend,
        "coreg_ref_frame": np.int32(ref_idx),
        "coreg_m2": np.int32(1 if did_m2 else 0),
        "coreg_n_aligned": np.int32(coreg_n_aligned),
        "coreg_max_shift": np.float32(coreg_max_shift),
        "coreg_anchor_valid_frac": np.float32(anchor_valid_frac),
        "coreg_shifts": np.array(
            [shifts.get(fi, (0.0, 0.0)) for fi in range(n_frames)], np.float32),
        **extras,
    }
