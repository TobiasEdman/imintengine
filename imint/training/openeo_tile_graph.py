"""openEO process-graph implementations for per-tile S2 spectral fetch.

This module is the **openEO backend adapter** sitting behind the unified
:func:`imint.training.fetch_spectral.fetch_spectral` dispatcher. It is
how ``backend="cdse-openeo"`` and ``backend="des"`` calls reach the
respective openEO servers. The dispatcher passes a single-slot mapping
``{0: date_str}`` for per-candidate fetches; the multi-slot form of
:func:`fetch_tile_at_specific_dates` (one process graph covering up to
four slots at once) is kept available for callers that want the batched
form but is *not* used by the unified flow today.

Mechanics
---------
``fetch_tile_at_specific_dates(bbox_3006, {sidx: date_str, ...}, source)``
builds one process graph::

    sub_cube_i = load_collection(date_i .. date_i+1d, bbox, bands=...)
                 .reduce_dimension("t", "first")
                 .rename_labels(bands → ["s{i}_B02", ...])
    merged     = sub_cube_0.merge_cubes(sub_cube_1)...
    download(format="gtiff")

For N=1 slot it reduces to a single ``load_collection → download``.
For N=4 it produces a (4×6, H, W) cube parsed back per slot.

Compatibility
-------------
* CDSE openEO 1.2 (``openeo.dataspace.copernicus.eu``) — primary; HARD
  1-concurrent-connection-per-account limit, serialised by
  ``_CDSE_OPENEO_SEMAPHORE``.
* DES openEO 1.1 (``openeo.digitalearth.se``) — same graph shape;
  ``aggregate_spatial`` has a known geopandas-dtype server bug so
  AOI cloud verification uses the pixel ``scl_stack_screen`` path
  (handled upstream in :func:`imint.training.optimal_fetch.verify_aoi_scl`).

Date metadata caveat
--------------------
``reduce_dimension(t, "first")`` does not return which timestamp was
selected — the date dimension is consumed. The returned ``date_str``
approximates with the window mid-point (see :func:`_window_midpoint`);
downstream training only uses dates for slot ordering, so this is fine.
Pixel-level fidelity is unaffected.

Cloud handling
--------------
Scene-level filter via ``properties={"eo:cloud_cover": lambda v: v <=
cloud_max_pct}`` drops whole acquisitions over the threshold; combined
with ``reduce(t, "first")`` the result is the first below-threshold
scene in the window. Pixel-level SCL gating is the unified dispatcher's
responsibility (``verify_aoi_scl`` for openEO backends, or
``fetch_s2_scene``'s two-stage SCL prescreen for the SH Process backend).
This module relies on that upstream gate and does not re-apply one.
"""
from __future__ import annotations

import io
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from imint.utils import dn_to_reflectance


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
    low = msg.lower()
    return (
        "[402]" in msg
        or "PaymentRequired" in msg
        or "insufficient credit" in low
        or "insufficient credits" in low
        # SH Process API PU exhaustion surfaces as HTTP 403
        # ACCESS_INSUFFICIENT_PROCESSING_UNITS — same shared PU pool as the
        # openEO 402, so it must trip the same credit guard.
        or "insufficient_processing_units" in low
        or "processing units or requests available" in low
    )


@contextmanager
def _cdse_single_flight(source: str):
    """Serialise CDSE openEO calls through the shared single-flight semaphore.

    CDSE openEO enforces a 1-concurrent-connection-per-account limit. With
    multiple refetch workers each firing a season-SCL aggregate AND a
    tile-graph spectral download, concurrent calls collide and the backend
    returns ``[429] Too Many Requests`` (observed 2026-06-01). Routing every
    CDSE call through ``_CDSE_OPENEO_SEMAPHORE`` (max_permits=1) turns that
    contention into an orderly queue — one CDSE call at a time, no 429 — so
    the CDSE arm of the source race contributes clean tiles steadily while
    DES handles the other workers in parallel.

    No-op for non-CDSE sources (DES has its own semaphore in the per-slot
    fetcher). The semaphore is imported lazily because
    ``imint.training.tile_fetch`` imports the credit-guard helpers from
    THIS module — a top-level import here would be circular.
    """
    if source != "cdse-openeo":
        yield
        return
    from imint.training.tile_fetch import _CDSE_OPENEO_SEMAPHORE
    _CDSE_OPENEO_SEMAPHORE.acquire()
    try:
        yield
    finally:
        _CDSE_OPENEO_SEMAPHORE.release()


# Band layout matches PRITHVI_BANDS in tile_fetch.py and CDSE_BANDS_*
# constants in imint/fetch.py. Kept here as a local constant so this
# module is self-contained.
PRITHVI_BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")

# CDSE band groups (mirrors imint.fetch.CDSE_BANDS_*). Defined separately
# so the multi-cube merge in :func:`_build_slot_cube` can pick the right
# resolution group per band.
_CDSE_BANDS_10M = ("B02", "B03", "B04", "B08")
_CDSE_BANDS_20M = ("B05", "B06", "B07", "B8A", "B11", "B12")

# 60m atmospheric group — B01 (coastal aerosol), B09 (water vapour).
# Native 60m, resampled to the 10m grid like the 20m bands. Included so a
# single fetch captures the FULL S2 L2A spectral (12 bands) as training +
# sample material, not just the model's 6.
_CDSE_BANDS_60M = ("B01", "B09")

# DES band groups (mirrors imint.fetch.BANDS_*). DES uses lowercase band
# names with a different collection id (``s2_msi_l2a`` vs CDSE's
# ``SENTINEL2_L2A``). Same resolution grouping; lowercase in load_collection.
_DES_BANDS_10M = ("b02", "b03", "b04", "b08")
_DES_BANDS_20M = ("b05", "b06", "b07", "b8a", "b11", "b12")
_DES_BANDS_60M = ("b01", "b09")

# Canonical ALL-bands output order (grouped by native resolution, as
# loaded/merged): 10m, then 20m, then 60m. Every fetch returns bands in
# THIS order; downstream splits by fixed index (see ALL_BANDS_INDEX).
ALL_BANDS = (
    "B02", "B03", "B04", "B08",          # 10m  (idx 0-3)
    "B05", "B06", "B07", "B8A", "B11", "B12",  # 20m (idx 4-9)
    "B01", "B09",                        # 60m  (idx 10-11)
)
# Index of each logical group within an ALL_BANDS-ordered (N_bands, H, W)
# array. PRITHVI keeps the model's 6-band order exactly.
ALL_BANDS_INDEX = {
    "prithvi": (0, 1, 2, 7, 8, 9),   # B02,B03,B04,B8A,B11,B12
    "b08": (3,),
    "rededge": (4, 5, 6),            # B05,B06,B07
    "b01": (10,),
    "b09": (11,),
}


def _bands_groups_for_source(source: str):
    """(bands_10m, bands_20m, bands_60m, all_output_bands) in source casing."""
    if source == "des":
        return (_DES_BANDS_10M, _DES_BANDS_20M, _DES_BANDS_60M,
                tuple(b.lower() for b in ALL_BANDS))
    return (_CDSE_BANDS_10M, _CDSE_BANDS_20M, _CDSE_BANDS_60M, ALL_BANDS)


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
    bands_60m: Sequence[str] | None = None,
    scl_band: str | None = None,
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
    # Build spatial_extent explicitly with an EPSG:3006 crs. The caller's
    # bbox dict (from tile.bbox_from_center) often lacks a `crs` key;
    # without it openEO assumes EPSG:4326 and rejects our SWEREF metre
    # coordinates as out-of-lat/lon-range ([400] ProcessParameterInvalid,
    # observed 2026-06-01 in production). Setting crs here makes the
    # spectral fetch robust regardless of what the caller passed.
    spatial_extent = {
        "west":  float(bbox_3006["west"]),
        "south": float(bbox_3006["south"]),
        "east":  float(bbox_3006["east"]),
        "north": float(bbox_3006["north"]),
        "crs":   3006,
    }

    # Scene-level cloud filter — drops acquisitions over threshold before
    # the temporal reduce. Documented as `properties` in the openEO
    # spec; the lambda body becomes a server-side comparison expression.
    load_kwargs: dict = dict(
        collection_id=collection_id,
        spatial_extent=spatial_extent,
        temporal_extent=[date_start, date_end],
    )
    if cloud_max_pct is not None:
        # CDSE openEO 1.2 property filtering only supports
        # eq / lte / gte / array_contains — NOT 'lt' ([400]
        # PropertyConditionInvalid observed 2026-06-01). Use <= so the
        # generated condition is 'lte'. The 1-percentage-point inclusivity
        # difference vs '<' is immaterial for a cloud-cover ceiling.
        load_kwargs["properties"] = {
            "eo:cloud_cover": lambda v: v <= cloud_max_pct,
        }

    cube_10m = conn.load_collection(bands=list(bands_10m), **load_kwargs)
    cube_20m = conn.load_collection(bands=list(bands_20m), **load_kwargs)
    cube_20m = cube_20m.resample_cube_spatial(target=cube_10m, method="bilinear")

    # Merge 10m + 20m groups into one cube (still temporal).
    cube = cube_10m.merge_cubes(cube_20m)

    # Optional 60m atmospheric group (B01/B09), resampled to the 10m grid.
    # Continuous reflectance → bilinear, /10000 like the rest.
    if bands_60m:
        cube_60m = conn.load_collection(bands=list(bands_60m), **load_kwargs)
        cube_60m = cube_60m.resample_cube_spatial(target=cube_10m, method="bilinear")
        cube = cube.merge_cubes(cube_60m)

    # Final output band order. When scl_band is set, the SCL band rides
    # along in the SAME download so the caller can AOI-cloud-gate the
    # chosen scene without a separate SCL call. SCL is categorical, so it
    # is resampled NEAREST (not bilinear) and must NOT be divided by 10000
    # at parse time — the caller handles that split.
    out_bands = list(output_bands)
    if scl_band is not None:
        cube_scl = conn.load_collection(bands=[scl_band], **load_kwargs)
        cube_scl = cube_scl.resample_cube_spatial(target=cube_10m, method="near")
        cube = cube.merge_cubes(cube_scl)
        out_bands = out_bands + [scl_band]

    # Reduce time → first available below-cloud-threshold scene in window.
    cube = cube.reduce_dimension(dimension="t", reducer="first")

    # Filter down to the output bands we actually want (drops B05/B06/B07
    # if they're not in PRITHVI_BANDS, etc.; keeps SCL when requested).
    cube = cube.filter_bands(bands=out_bands)

    # Tag with per-slot prefix so the multi-slot merge below has unique
    # band labels. Order matches `out_bands` (filter_bands preserves the
    # order given).
    cube = cube.rename_labels(
        dimension="bands",
        target=[f"s{slot_idx}_{b}" for b in out_bands],
    )
    return cube


_SCL_CLOUD_CLASSES = (3, 8, 9, 10)   # shadow, cloud_medium, cloud_high, cirrus


def fetch_tile_all_slots_cdse_openeo(
    bbox_3006: dict,
    slot_windows: Sequence[tuple[int, str, str]],
    *,
    prithvi_bands: Sequence[str] = PRITHVI_BANDS,
    cloud_max_pct: float = 30.0,
    include_scl: bool = False,
) -> dict:
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
        When ``include_scl=False`` (default): ``{slot_idx: (array,
        date_str)}`` — ``array`` is ``(len(prithvi_bands), H, W)``
        ``float32`` reflectance (DN / 10000).

        When ``include_scl=True``: ``{slot_idx: (array, aoi_cloud_frac,
        date_str)}`` — the SCL band rides along in the SAME download and
        ``aoi_cloud_frac`` is the fraction of AOI pixels in cloud/shadow
        classes (3/8/9/10). Lets the caller AOI-gate the chosen scene
        without a separate SCL call.

    Raises:
        FetchError: If the openEO download returns empty bytes or the
            parsed band count does not match the expected per-slot total.
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
    scl_band = "SCL" if include_scl else None

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
            scl_band=scl_band,
        ))

    merged = sub_cubes[0]
    for c in sub_cubes[1:]:
        merged = merged.merge_cubes(c)

    print(f"    [CDSE-tile-graph] downloading {len(slot_windows)} slots × "
          f"{len(prithvi_bands)} bands = {len(slot_windows)*len(prithvi_bands)} bands",
          flush=True)
    try:
        with _cdse_single_flight("cdse-openeo"):
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

    n_spec = len(prithvi_bands)
    per_slot = n_spec + (1 if include_scl else 0)   # +1 for the SCL band
    expected_bands = len(slot_windows) * per_slot
    if full.shape[0] != expected_bands:
        raise FetchError(
            f"CDSE tile-graph: expected {expected_bands} bands "
            f"({len(slot_windows)} slots × {per_slot} bands"
            f"{' incl SCL' if include_scl else ''}), got {full.shape[0]}. "
            f"Likely a band-ordering mismatch in merge_cubes; investigate."
        )

    result: dict = {}
    for i, (slot_idx, date_start, date_end) in enumerate(slot_windows):
        base = i * per_slot
        # Spectral bands → reflectance (DN / 10000). NO -1000 offset here:
        # CDSE openEO applies RADIO_ADD_OFFSET server-side (unlike DES, which
        # bakes it into COGs — see the DES path). SCL (if present) is the LAST
        # band of the slot and is categorical — NOT scaled.
        slot_arr = full[base:base + n_spec].astype(np.float32) / 10000.0
        if not np.any(slot_arr):
            continue
        date_str = _window_midpoint(date_start, date_end)
        if include_scl:
            scl = full[base + n_spec]            # raw SCL class codes
            scl_int = np.rint(scl).astype(np.int16)
            valid = scl_int > 0                  # 0 = no_data, exclude from frac
            n_valid = int(valid.sum())
            if n_valid == 0:
                aoi_cloud_frac = 1.0             # all nodata → treat as unusable
            else:
                cloud = np.isin(scl_int, _SCL_CLOUD_CLASSES) & valid
                aoi_cloud_frac = float(cloud.sum()) / float(n_valid)
            result[slot_idx] = (slot_arr, aoi_cloud_frac, date_str)
        else:
            result[slot_idx] = (slot_arr, date_str)

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
    # Fetch the FULL 12-band S2 spectral (10m+20m+60m), not just Prithvi-6,
    # so one download yields all training/sample material. Downstream
    # fetch_spectral splits to the 6-band model cube + per-band extras.
    b10, b20, b60, des_bands = _bands_groups_for_source("des")

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
            bands_10m=b10,
            bands_20m=b20,
            bands_60m=b60,
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
        # DES bakes the PB04.00 -1000 BOA offset into COGs (CDSE openEO applies
        # it server-side); subtract it so output matches the rest of the dataset.
        slot_arr = dn_to_reflectance(full[i * n_bands:(i + 1) * n_bands], source="des")
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
    include_scl: bool = False,
) -> dict:
    """Dispatch :func:`fetch_tile_all_slots_*` by source.

    Args:
        source: ``"cdse-openeo"`` or ``"des"``. Other values raise
            ``ValueError``.
        cloud_max_pct: Forwarded to the backend-specific function.
            CDSE 1.2 supports the property-lambda filter; DES 1.1 may
            not, so callers using DES should pass ``None``.
    """
    if source == "cdse-openeo":
        # Preserve None — the specific-dates path passes cloud_max_pct=None
        # because the date was already vetted by the season-SCL gate, so
        # the per-scene properties filter is redundant. `or 30.0` would
        # wrongly coerce None → 30.0 and re-apply the (now lte) filter.
        return fetch_tile_all_slots_cdse_openeo(
            bbox_3006, slot_windows,
            prithvi_bands=prithvi_bands,
            cloud_max_pct=cloud_max_pct,
            include_scl=include_scl,
        )
    if source == "des":
        if include_scl:
            raise ValueError(
                "include_scl is only supported for source='cdse-openeo'"
            )
        return fetch_tile_all_slots_des_openeo(
            bbox_3006, slot_windows,
            prithvi_bands=prithvi_bands,
            cloud_max_pct=cloud_max_pct,
        )
    raise ValueError(f"fetch_tile_all_slots: unknown source {source!r}")


def fetch_tile_at_specific_dates(
    bbox_3006: dict,
    slot_dates: dict[int, str],
    *,
    source: str = "des",
    prithvi_bands: Sequence[str] = PRITHVI_BANDS,
    with_scl: bool = False,
) -> dict:
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
    #
    # with_scl=True (cdse-openeo only): the SCL band rides along in the
    # same download so the caller can AOI-gate the fetched scene without
    # a separate SCL call. Returns {slot: (spectral, aoi_cloud_frac, date)}.
    result = fetch_tile_all_slots(
        bbox_3006, slot_windows,
        source=source,
        prithvi_bands=prithvi_bands,
        cloud_max_pct=None,
        include_scl=with_scl,
    )
    # Replace the per-slot date approximation with the caller's exact
    # date — we trust the input over the window midpoint.
    if with_scl:
        return {
            slot_idx: (arr, frac, slot_dates[slot_idx])
            for slot_idx, (arr, frac, _) in result.items()
        }
    return {
        slot_idx: (arr, slot_dates[slot_idx])
        for slot_idx, (arr, _) in result.items()
    }
