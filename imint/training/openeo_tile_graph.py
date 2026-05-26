"""One openEO process graph per tile — fetch all 4 slots in a single call.

This is the Nivå-3 throughput optimisation. The classical per-slot path
(``_fetch_single_scene`` in ``tile_fetch.py``) submits up to ~28 separate
openEO process-graph evaluations per tile (3 candidate dates × 3 sources
× 4 slots, with race-pool fanout). Each one is a full HTTP round-trip
through the backend's process-graph queue.

The tile-graph path collapses all 4 slots into ONE process graph::

    slot_0_cube = load_collection(window_0).filter_cloud(<30%).reduce(t,"first")
                  .rename_labels(bands → ["s0_B02", "s0_B03", ...])
    slot_1_cube = same with window_1, prefix s1_
    slot_2_cube = same with window_2, prefix s2_
    slot_3_cube = same with window_3, prefix s3_

    merged = slot_0_cube.merge_cubes(slot_1).merge_cubes(s2).merge_cubes(s3)
    download(format="gtiff")      # 24-band geotiff, 6 bands × 4 slots

Throughput impact (per tile, with CDSE openEO single-flight):

    Old: ~7-28 openEO calls per tile (race-pool, candidate ×source ×slot)
    New: 1 openEO call per tile

Combined with the existing tile-level worker fanout (6 workers), this
gives us:

    Old throughput ≈ 1 tile / N×60 s   (serial through CDSE 1-conn cap)
    New throughput ≈ 6 tiles / 60 s    (6 workers × 1 call each)

— so ~6× speedup in the CDSE openEO single-flight regime.

Compatibility:
    * CDSE openEO 1.2 (``openeo.dataspace.copernicus.eu``): primary.
    * DES openEO 1.1 (``openeo.digitalearth.se``): supported in principle
      (``merge_cubes`` / ``reduce_dimension`` / ``rename_labels`` all in
      openEO 1.1). Not yet exercised end-to-end — leave as Nivå-2 follow-up.

Date metadata caveat:
    ``reduce_dimension(t, "first")`` does NOT return which timestamp was
    selected; the date is consumed by the reducer. We approximate per-slot
    ``date_str`` with the window mid-point (see :func:`_window_midpoint`).
    Downstream training only uses the date for slot ordering, so this is
    acceptable; pixel-level fidelity is unaffected.

Cloud handling:
    Scene-level cloud filter via ``properties={"eo:cloud_cover": lambda
    v: v < cloud_max_pct}`` — drops whole acquisitions over the threshold.
    Combined with ``reduce(t, "first")`` the result is the first below-
    threshold scene in the window. No pixel-level SCL mask is applied;
    the upstream ERA5+SCL pre-filter on date selection is therefore
    *skipped* in this path (the server does its own filtering).
"""
from __future__ import annotations

import io
import threading
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np


# ── Session-scoped credit guard ─────────────────────────────────────────────
#
# A source (currently only ``cdse-openeo``) gets marked DEAD for the rest
# of the Python process whenever the backend returns 402 PaymentRequired
# / "insufficient credits". The mark prevents subsequent fetch attempts
# from spending HTTP roundtrips just to get the same 402 back, and lets
# the per-tile dispatch fall straight through to the next source in
# ``--sources``.
#
# Why session-scoped, not durable: a pod restart (eviction, redeploy,
# new pod after backoff) clears the mark, so when CDSE credits refill
# (monthly reset, new package purchased) we automatically retry. No
# scheduled cron-job needed — first new pod after reset succeeds and
# stays on CDSE for the rest of its lifetime.
#
# Why per-source, not per-call-type: the credit pool is shared between
# all openEO calls on the account (synchronous fetches, batch jobs,
# aggregate_spatial). Once 402 surfaces, the whole source is unusable.
_CREDIT_LOCK = threading.Lock()
_DEAD_SOURCES: set[str] = set()


def is_source_dead(source: str) -> bool:
    """True if ``source`` has been marked dead this process."""
    with _CREDIT_LOCK:
        return source in _DEAD_SOURCES


def mark_source_dead(source: str, reason: str) -> None:
    """Idempotent: log once on first mark, ignore re-marks.

    Workers across threads may race to mark the same source dead in
    rapid succession; the lock + set membership check ensures we only
    print the warning once.
    """
    with _CREDIT_LOCK:
        if source in _DEAD_SOURCES:
            return
        _DEAD_SOURCES.add(source)
    # Print outside the lock to avoid serialising on stdout flushes.
    print(f"    [credit-guard] {source} marked DEAD for this session: {reason}",
          flush=True)


def _is_payment_required_error(exc: BaseException) -> bool:
    """Detect 402 PaymentRequired / insufficient-credits from openEO.

    The openeo-python-client surfaces backend errors as ``OpenEoApiError``
    with the HTTP status code embedded in the message. We match by string
    rather than a direct exception type to stay loose against client
    library version churn.
    """
    msg = f"{type(exc).__name__}: {exc}"
    return (
        "[402]" in msg
        or "PaymentRequired" in msg
        or "insufficient credit" in msg.lower()
        or "insufficient credits" in msg.lower()
    )


# Band layout matches PRITHVI_BANDS in tile_fetch.py and CDSE_BANDS_*
# constants in imint/fetch.py. Kept here as a local constant so this
# module is self-contained.
PRITHVI_BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")

# CDSE band groups (mirrors imint.fetch.CDSE_BANDS_*). Defined separately
# so the multi-cube merge in :func:`_build_slot_cube` can pick the right
# resolution group per band.
_CDSE_BANDS_10M = ("B02", "B03", "B04", "B08")
_CDSE_BANDS_20M = ("B05", "B06", "B07", "B8A", "B11", "B12")

# DES band groups (mirrors imint.fetch.BANDS_*). DES uses lowercase band
# names with a different collection id (``s2_msi_l2a`` vs CDSE's
# ``SENTINEL2_L2A``). Same 10m / 20m grouping; same Prithvi output
# subset, just lowercase in the load_collection call.
_DES_BANDS_10M = ("b02", "b03", "b04", "b08")
_DES_BANDS_20M = ("b05", "b06", "b07", "b8a", "b11", "b12")


def _prithvi_bands_for_source(source: str, prithvi_bands: tuple[str, ...]) -> tuple[str, ...]:
    """Map canonical (uppercase) PRITHVI_BANDS to the backend's casing.

    DES uses lowercase band ids; CDSE openEO uses uppercase. Filter_bands
    requires the exact label as advertised by ``load_collection`` —
    case-sensitive.
    """
    if source == "des":
        return tuple(b.lower() for b in prithvi_bands)
    return tuple(prithvi_bands)


def _window_midpoint(start: str, end: str) -> str:
    """Return ``YYYY-MM-DD`` halfway between ``start`` and ``end``.

    Used as a stand-in for the actual scene date when ``reduce("t",
    "first")`` consumed the temporal dimension. Downstream training only
    uses the date for slot ordering, so an approximate label is fine.
    """
    d0 = datetime.fromisoformat(start)
    d1 = datetime.fromisoformat(end)
    mid = d0 + (d1 - d0) / 2
    return mid.strftime("%Y-%m-%d")


def _build_slot_cube(
    conn,
    bbox_3006: dict,
    slot_idx: int,
    date_start: str,
    date_end: str,
    *,
    collection_id: str,
    cloud_max_pct: float | None,
    bands_10m: Sequence[str],
    bands_20m: Sequence[str],
    output_bands: Sequence[str],
):
    """Build ONE slot's sub-cube ready to be merged into the tile graph.

    Loads 10m + 20m groups separately (different native resolutions),
    resamples 20m to the 10m grid with bilinear, reduces over time with
    ``first``, filters down to ``output_bands``, then renames band labels
    with the per-slot ``"s{slot_idx}_"`` prefix so the eventual
    multi-slot merge does not collide on identical band names.

    Cloud handling: scene-level filter via ``properties=...`` when
    ``cloud_max_pct`` is supplied. DES openEO 1.1 may not honour the
    ``eo:cloud_cover`` lambda filter (silently no-op or 400); pass
    ``cloud_max_pct=None`` to skip the filter and rely on ``reduce(t,
    "first")`` selecting the chronologically first scene. Pixel-level
    SCL masking is intentionally NOT applied here — keeps the process
    graph portable between CDSE 1.2 and DES 1.1.
    """
    # Scene-level cloud filter — drops acquisitions over threshold before
    # the temporal reduce. Documented as `properties` in the openEO
    # spec; the lambda body becomes a server-side comparison expression.
    load_kwargs: dict = dict(
        collection_id=collection_id,
        spatial_extent=bbox_3006,
        temporal_extent=[date_start, date_end],
    )
    if cloud_max_pct is not None:
        load_kwargs["properties"] = {
            "eo:cloud_cover": lambda v: v < cloud_max_pct,
        }

    cube_10m = conn.load_collection(bands=list(bands_10m), **load_kwargs)
    cube_20m = conn.load_collection(bands=list(bands_20m), **load_kwargs)
    cube_20m = cube_20m.resample_cube_spatial(target=cube_10m, method="bilinear")

    # Merge 10m + 20m groups into one cube (still temporal).
    cube = cube_10m.merge_cubes(cube_20m)

    # Reduce time → first available below-cloud-threshold scene in window.
    cube = cube.reduce_dimension(dimension="t", reducer="first")

    # Filter down to the output bands we actually want (drops B05/B06/B07
    # if they're not in PRITHVI_BANDS, etc.).
    cube = cube.filter_bands(bands=list(output_bands))

    # Tag with per-slot prefix so the multi-slot merge below has unique
    # band labels. Order matches `output_bands` (filter_bands preserves
    # the order given).
    cube = cube.rename_labels(
        dimension="bands",
        target=[f"s{slot_idx}_{b}" for b in output_bands],
    )
    return cube


def fetch_tile_all_slots_cdse_openeo(
    bbox_3006: dict,
    slot_windows: Sequence[tuple[int, str, str]],
    *,
    prithvi_bands: Sequence[str] = PRITHVI_BANDS,
    cloud_max_pct: float = 30.0,
) -> dict[int, tuple[np.ndarray, str]]:
    """Fetch all requested slots for a tile in ONE openEO call.

    Args:
        bbox_3006: EPSG:3006 bbox dict ``{"west", "south", "east", "north",
            "crs"}``.
        slot_windows: List of ``(slot_idx, date_start, date_end)``. May
            cover any subset of slots (1-4). Order in the returned dict
            is the same as input.
        prithvi_bands: Bands to fetch per slot. Default matches the
            project-wide ``PRITHVI_BANDS`` constant.
        cloud_max_pct: Scene-level cloud cover ceiling, in percent.
            Acquisitions over this are excluded before the temporal
            reduce.

    Returns:
        ``{slot_idx: (array, date_str)}`` — one entry per input slot.
        ``array`` is ``(len(prithvi_bands), H, W)`` ``float32`` in
        reflectance units (DN / 10000). ``date_str`` is the window
        midpoint (see module docstring; the true scene date is consumed
        by the temporal reduce).

    Raises:
        FetchError: If the openEO download returns empty bytes or the
            parsed band count does not match the expected
            ``n_slots × len(prithvi_bands)``.
    """
    # Local import — keep openeo dependency optional for the rest of the
    # repo and avoid importing it during ``imint.training`` package
    # initialisation.
    from imint.fetch import (
        CDSE_COLLECTION,
        FetchError,
        _connect_cdse,
        _unpack_openeo_gtiff_bytes,
    )
    import rasterio

    if not slot_windows:
        return {}

    if is_source_dead("cdse-openeo"):
        raise FetchError(
            "CDSE openEO marked dead this session (prior 402). "
            "Use a different source or restart the pod after credit reset."
        )

    conn = _connect_cdse()

    # Build one sub-cube per slot, then merge into a single download.
    sub_cubes = []
    for slot_idx, date_start, date_end in slot_windows:
        sub_cubes.append(_build_slot_cube(
            conn,
            bbox_3006,
            slot_idx,
            date_start,
            date_end,
            collection_id=CDSE_COLLECTION,
            cloud_max_pct=cloud_max_pct,
            bands_10m=_CDSE_BANDS_10M,
            bands_20m=_CDSE_BANDS_20M,
            output_bands=prithvi_bands,
        ))

    merged = sub_cubes[0]
    for c in sub_cubes[1:]:
        merged = merged.merge_cubes(c)

    print(f"    [CDSE-tile-graph] downloading {len(slot_windows)} slots × "
          f"{len(prithvi_bands)} bands = {len(slot_windows)*len(prithvi_bands)} bands",
          flush=True)
    try:
        raw_bytes = merged.download(format="gtiff")
    except Exception as exc:
        if _is_payment_required_error(exc):
            mark_source_dead("cdse-openeo", f"402 during tile-graph: {str(exc)[:160]}")
        raise
    if not raw_bytes:
        raise FetchError("CDSE openEO returned empty bytes from tile-graph download")
    raw_bytes = _unpack_openeo_gtiff_bytes(raw_bytes)

    # Parse the multi-band geotiff. The merge_cubes + rename_labels chain
    # above produces bands in the order:
    #   s{slot_windows[0][0]}_B02, s..._B03, ..., s{slot_windows[1][0]}_B02, ...
    # i.e. all bands of slot 0 first, then all bands of slot 1, etc.
    with rasterio.open(io.BytesIO(raw_bytes)) as src:
        full = src.read()  # (n_slots * n_bands, H, W)

    n_bands = len(prithvi_bands)
    expected_bands = len(slot_windows) * n_bands
    if full.shape[0] != expected_bands:
        raise FetchError(
            f"CDSE tile-graph: expected {expected_bands} bands "
            f"({len(slot_windows)} slots × {n_bands} bands), got {full.shape[0]}. "
            f"Likely a band-ordering mismatch in merge_cubes; investigate."
        )

    result: dict[int, tuple[np.ndarray, str]] = {}
    for i, (slot_idx, date_start, date_end) in enumerate(slot_windows):
        slot_arr = full[i * n_bands:(i + 1) * n_bands].astype(np.float32) / 10000.0
        # All-zeros guard — degenerate server response. Caller already
        # has the same guard in repair_to_canonical_layout but flagging
        # here lets us skip writing the slot at all.
        if not np.any(slot_arr):
            continue
        result[slot_idx] = (slot_arr, _window_midpoint(date_start, date_end))

    return result


def fetch_tile_all_slots_des_openeo(
    bbox_3006: dict,
    slot_windows: Sequence[tuple[int, str, str]],
    *,
    prithvi_bands: Sequence[str] = PRITHVI_BANDS,
    cloud_max_pct: float | None = None,
) -> dict[int, tuple[np.ndarray, str]]:
    """DES openEO 1.1 variant of :func:`fetch_tile_all_slots_cdse_openeo`.

    Same merge_cubes + rename_labels + single download pattern, but
    against DES (``openeo.digitalearth.se``, API 1.1) using the
    lowercase band convention and ``s2_msi_l2a`` collection id.

    Args:
        cloud_max_pct: Scene-level cloud cover ceiling. Default ``None``
            skips the ``properties={"eo:cloud_cover"}`` filter — DES 1.1
            handling of property-lambda filters is not documented as
            reliable as CDSE 1.2's. With ``None`` the temporal reduce
            picks chronologically first; pair with caller-side ERA5+SCL
            pre-filtered date windows for cloud-clear selection.

    Returns:
        ``{slot_idx: (array, date_str)}`` — same shape as the CDSE
        variant. ``date_str`` is the window midpoint.
    """
    # Local imports to avoid forcing openeo on the rest of the package.
    from imint.fetch import (
        COLLECTION as DES_COLLECTION,
        FetchError,
        _connect as _connect_des,
        _unpack_openeo_gtiff_bytes,
    )
    import rasterio

    if not slot_windows:
        return {}

    conn = _connect_des()
    des_bands = _prithvi_bands_for_source("des", tuple(prithvi_bands))

    sub_cubes = []
    for slot_idx, date_start, date_end in slot_windows:
        sub_cubes.append(_build_slot_cube(
            conn,
            bbox_3006,
            slot_idx,
            date_start,
            date_end,
            collection_id=DES_COLLECTION,
            cloud_max_pct=cloud_max_pct,
            bands_10m=_DES_BANDS_10M,
            bands_20m=_DES_BANDS_20M,
            output_bands=des_bands,
        ))

    merged = sub_cubes[0]
    for c in sub_cubes[1:]:
        merged = merged.merge_cubes(c)

    print(f"    [DES-tile-graph] downloading {len(slot_windows)} slots × "
          f"{len(des_bands)} bands = {len(slot_windows)*len(des_bands)} bands",
          flush=True)
    raw_bytes = merged.download(format="gtiff")
    if not raw_bytes:
        raise FetchError("DES openEO returned empty bytes from tile-graph download")
    raw_bytes = _unpack_openeo_gtiff_bytes(raw_bytes)

    with rasterio.open(io.BytesIO(raw_bytes)) as src:
        full = src.read()  # (n_slots * n_bands, H, W)

    n_bands = len(des_bands)
    expected_bands = len(slot_windows) * n_bands
    if full.shape[0] != expected_bands:
        raise FetchError(
            f"DES tile-graph: expected {expected_bands} bands "
            f"({len(slot_windows)} slots × {n_bands} bands), got {full.shape[0]}."
        )

    result: dict[int, tuple[np.ndarray, str]] = {}
    for i, (slot_idx, date_start, date_end) in enumerate(slot_windows):
        slot_arr = full[i * n_bands:(i + 1) * n_bands].astype(np.float32) / 10000.0
        if not np.any(slot_arr):
            continue
        result[slot_idx] = (slot_arr, _window_midpoint(date_start, date_end))

    return result


def fetch_tile_all_slots(
    bbox_3006: dict,
    slot_windows: Sequence[tuple[int, str, str]],
    *,
    source: str = "cdse-openeo",
    prithvi_bands: Sequence[str] = PRITHVI_BANDS,
    cloud_max_pct: float | None = 30.0,
) -> dict[int, tuple[np.ndarray, str]]:
    """Dispatch :func:`fetch_tile_all_slots_*` by source.

    Args:
        source: ``"cdse-openeo"`` or ``"des"``. Other values raise
            ``ValueError``.
        cloud_max_pct: Forwarded to the backend-specific function.
            CDSE 1.2 supports the property-lambda filter; DES 1.1 may
            not, so callers using DES should pass ``None``.
    """
    if source == "cdse-openeo":
        return fetch_tile_all_slots_cdse_openeo(
            bbox_3006, slot_windows,
            prithvi_bands=prithvi_bands,
            cloud_max_pct=cloud_max_pct or 30.0,
        )
    if source == "des":
        return fetch_tile_all_slots_des_openeo(
            bbox_3006, slot_windows,
            prithvi_bands=prithvi_bands,
            cloud_max_pct=cloud_max_pct,
        )
    raise ValueError(f"fetch_tile_all_slots: unknown source {source!r}")


def score_dates_aoi_cloud(
    bbox_3006: dict,
    date_start: str,
    date_end: str,
    *,
    source: str = "des",
    cloud_classes: tuple[int, ...] = (3, 8, 9, 10),
) -> dict[str, float]:
    """Per-AOI cloud fraction per date, computed server-side via openEO.

    Single openEO call that:

      1. ``load_collection`` SCL band for the full window (one timestep
         per S2 acquisition; SCL is part of L2A so this implicitly
         filters to actual pass dates — no ERA5 prefilter needed).
      2. ``cloud_flag = SCL ∈ cloud_classes`` (boolean per pixel per t).
      3. ``aggregate_spatial(geometries=bbox_polygon, reducer="mean")``
         — server computes the AOI cloud fraction per timestep.
      4. ``execute()`` returns a small JSON ``{date: cloud_frac}``;
         **no pixel-level data crosses the wire.**

    Compared to :func:`imint.training.optimal_fetch.scl_stack_screen`
    (which downloads the SCL stack and computes the mean client-side):

      * No 19-day chunking needed (DES NetCDF cap doesn't apply to
        aggregate_spatial output — the response is tiny).
      * One openEO call instead of 3-6 chunks.
      * Network payload is ~N×8 bytes instead of ``N × W × H bytes``.

    The pattern mirrors :func:`scripts.batch_fetch_openeo.screen_tile_scl`
    which has been used in production for CDSE for a year. DES openEO
    historically had ``aggregate_spatial`` issues (geopandas dtype
    error noted in ``scl_stack_screen``); this function exercises the
    same path on DES so we can verify whether the bug is still present.

    Args:
        bbox_3006: EPSG:3006 bbox dict. Converted internally to WGS84
            for the aggregate_spatial geometry.
        date_start, date_end: ISO ``YYYY-MM-DD``. Inclusive of both.
        source: ``"des"`` or ``"cdse-openeo"``. CDSE uses uppercase
            ``"SCL"`` band; DES uses lowercase ``"scl"``.
        cloud_classes: SCL class codes that count as "cloud":
            3 = shadow, 8 = cloud_medium, 9 = cloud_high, 10 = cirrus.
            Caller may extend (e.g. include 11 for snow) per use case.

    Returns:
        ``{date_str: aoi_cloud_fraction}``. Empty dict if no S2 passes
        in the window or the server errored.

    Raises:
        Whatever the openeo client raises — caller decides whether to
        fall back to ``optimal_fetch_dates``.
    """
    from shapely.geometry import box, mapping
    from pyproj import Transformer

    if source == "des":
        from imint.fetch import _connect as _connect_backend, COLLECTION as collection_id
        scl_band_name = "scl"
    elif source == "cdse-openeo":
        if is_source_dead("cdse-openeo"):
            # Fast-exit so the caller's exception handler routes
            # immediately to the fallback path.
            raise RuntimeError(
                "CDSE openEO marked dead this session (prior 402)"
            )
        from imint.fetch import _connect_cdse as _connect_backend, CDSE_COLLECTION as collection_id
        scl_band_name = "SCL"
    else:
        raise ValueError(f"score_dates_aoi_cloud: unknown source {source!r}")

    # Convert EPSG:3006 bbox to WGS84 — aggregate_spatial geometries are
    # safest in 4326 (CDSE expects it; DES tolerates it).
    transformer = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)
    west, south = transformer.transform(bbox_3006["west"], bbox_3006["south"])
    east, north = transformer.transform(bbox_3006["east"], bbox_3006["north"])

    conn = _connect_backend()
    scl = conn.load_collection(
        collection_id=collection_id,
        spatial_extent={
            "west":  west, "south": south,
            "east":  east, "north": north,
            "crs":   "EPSG:4326",
        },
        temporal_extent=[date_start, date_end],
        bands=[scl_band_name],
    )

    # Build cloud_flag = OR over (SCL == c) for c in cloud_classes.
    # ``cube.band(name)`` returns a band-extracted single-band cube where
    # comparison operators work pixel-wise.
    scl_b = scl.band(scl_band_name)
    cloud_flag = (scl_b == cloud_classes[0])
    for c in cloud_classes[1:]:
        cloud_flag = cloud_flag | (scl_b == c)

    poly = box(west, south, east, north)
    cloud_frac_ts = cloud_flag.aggregate_spatial(
        geometries=mapping(poly), reducer="mean",
    )

    print(f"    [{source}:aoi-cloud-aggregate] window={date_start}→{date_end}",
          flush=True)
    try:
        ts_json = cloud_frac_ts.execute()
    except Exception as exc:
        if source == "cdse-openeo" and _is_payment_required_error(exc):
            mark_source_dead("cdse-openeo",
                             f"402 during aoi-cloud-aggregate: {str(exc)[:160]}")
        raise

    # Parse: same schema as scripts/batch_fetch_openeo.py:screen_tile_scl.
    # openEO returns {date_str: [[value]]} where the outer list is per
    # geometry (we have 1) and the inner is per band (also 1).
    result: dict[str, float] = {}
    if isinstance(ts_json, dict):
        for date_key, val in ts_json.items():
            if date_key == "data":
                continue
            date_str = str(date_key)[:10]
            frac = val
            while isinstance(frac, list) and frac:
                frac = frac[0]
            if isinstance(frac, (int, float)):
                result[date_str] = float(frac)
    return result


def fetch_tile_at_specific_dates(
    bbox_3006: dict,
    slot_dates: dict[int, str],
    *,
    source: str = "des",
    prithvi_bands: Sequence[str] = PRITHVI_BANDS,
) -> dict[int, tuple[np.ndarray, str]]:
    """Tile-graph variant where the caller has already chosen one date per slot.

    This is the **architecturally correct** entry point for the refetch
    pipeline. The caller is expected to have run an SCL-based cloud
    scorer (e.g. :func:`imint.training.optimal_fetch.optimal_fetch_dates`
    with ``mode="era5_then_scl"``) and picked the lowest-AOI-cloud-count
    date per slot.

    Compared to :func:`fetch_tile_all_slots` with broad windows + server-
    side reducer:

      * **Spatially coherent scenes.** Each slot's pixels come from
        exactly one acquisition date — same sun-angle, same atmosphere,
        same BRDF. The window-+-reduce(``first``) approach would compose
        pixels from different dates, which is wrong for training data.
      * **No reliance on server-side cloud-cover metadata.** Scene-level
        ``eo:cloud_cover`` is too coarse — a 25 % cloud scene can have
        the whole 5×5 km AOI under that cloud. Our client-side AOI
        scoring is finer.
      * **No DES ``NoDataAvailable`` failures.** Single-date windows on
        pre-vetted dates always have a scene; the failure mode where
        DES strict-fails an empty load_collection inside a multi-cube
        graph cannot trigger.

    Per-tile cost: 1 openEO call for all (≤ 4) slots — the multi-band
    geotiff is parsed back to ``{slot_idx: (array, date_str)}``.

    Args:
        bbox_3006: EPSG:3006 bbox dict.
        slot_dates: ``{slot_idx: "YYYY-MM-DD"}``. May cover any subset
            of slots (1-4). Slots not in the dict are not fetched.
        source: ``"cdse-openeo"`` or ``"des"``.
        prithvi_bands: Bands to fetch per slot.

    Returns:
        ``{slot_idx: (array, date_str)}`` — one entry per input slot.
        ``date_str`` is the input date verbatim (no midpoint approx).
    """
    if not slot_dates:
        return {}

    from datetime import datetime, timedelta

    # Build narrow [date, date+1] windows so reduce_dimension(t, "first")
    # picks exactly the requested scene. Some backends interpret the
    # temporal_extent as a half-open interval [start, end); padding with
    # one day is safe across both DES 1.1 and CDSE 1.2.
    slot_windows: list[tuple[int, str, str]] = []
    for slot_idx, date_str in slot_dates.items():
        d = datetime.fromisoformat(date_str)
        end = (d + timedelta(days=1)).strftime("%Y-%m-%d")
        slot_windows.append((slot_idx, date_str, end))

    # cloud_max_pct=None — caller already filtered dates by AOI cloud
    # count, so the server-side ``eo:cloud_cover`` lambda would only
    # confuse the picture (it could exclude a date the caller deemed
    # acceptable).
    result = fetch_tile_all_slots(
        bbox_3006, slot_windows,
        source=source,
        prithvi_bands=prithvi_bands,
        cloud_max_pct=None,
    )
    # Replace the per-slot date approximation with the caller's exact
    # date — we trust the input over the window midpoint.
    return {
        slot_idx: (arr, slot_dates[slot_idx])
        for slot_idx, (arr, _) in result.items()
    }
