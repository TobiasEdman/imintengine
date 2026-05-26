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
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np


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
    raw_bytes = merged.download(format="gtiff")
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
