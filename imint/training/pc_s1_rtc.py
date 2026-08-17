"""Sentinel-1 RTC γ⁰ season composites via Microsoft Planetary Computer.

Why this module exists (user decision 2026-08-17)
-------------------------------------------------
The CDSE-GRD path (``cdse_s1_stac``) downloads whole ~2 GB IW-GRDH products,
computes σ⁰ **on the WGS84 ellipsoid** with a local calibration LUT (no
terrain correction), and caches the products on the PVC. That run managed
~45 tiles/h and filled the 1.6 T volume mid-pass (the LRU cap in
``cdse_s1_stac`` was the band-aid).

Planetary Computer's ``sentinel-1-rtc`` collection is a strictly better
product for this project:

    * **RTC γ⁰** — Radiometrically Terrain Corrected gamma-naught. The
      terrain-geometry effects (foreshortening / layover / slope-driven
      brightness) that σ⁰-on-ellipsoid leaves in are removed. Swedish terrain
      makes this material.
    * **Analysis-ready COGs** — float32 γ⁰ already, no DN², no calibration
      XML, no SNAP. We read the tile's window straight off the COG.
    * **Streaming, windowed** — sign the item, open ``/vsicurl/<signed>``,
      read ONLY the tile's window (~MB) instead of the whole product (~2 GB).
      No product cache, no PVC fill.

Since CROMA/TerraMind have never trained on S1 in this repo, adopting the
better product now costs nothing downstream.

Spatial / radiometric contract (verified against live PC data 2026-08-17)
-------------------------------------------------------------------------
    * asset ``vv`` / ``vh``: float32, native **UTM** COG per scene (e.g.
      EPSG:32633), 10 m pixel spacing, north-up (a real ``ds.crs`` — NOT the
      GCP-referenced rotated swath the GRD COGs carry, so no WarpedVRT).
    * **nodata = -32768** (NOT 0 — the GRD path's DN==0 nodata convention
      does not apply here; do not reuse it).
    * **units = linear γ⁰ power** (median VV ≈ 0.30 ≈ -5 dB over Sweden;
      dB range roughly [-25, +5] for land, higher on double-bounce).

Units stored: **LINEAR γ⁰** (NOT dB)
------------------------------------
``imint.fm.normalize.CromaNormalizer`` / ``TerraMindNormalizer`` both do
``x_db = 10*log10(x.clamp(1e-5)); (x_db - mean)/std`` with mean ≈ [-15,-22]
dB — i.e. they take **linear** σ⁰/γ⁰ and convert to dB themselves. Storing
linear γ⁰ is therefore the consistent choice; storing dB (as the CDSE-v2
path did, ``OUTPUT_DB=True``) would double-log inside the normalizer. v3
fixes this by storing linear. (The CDSE-v2 tiles are re-enriched to v3, so
no dB leftover survives — see scripts/enrich_tiles_s1.py.)

Return contract matches ``cdse_s1_stac.fetch_s1_season_composite`` exactly so
``scripts/enrich_tiles_s1.py`` selects the backend with one flag.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from . import s1_shared

# ── Module constants ─────────────────────────────────────────────────────

_STAC_ROOT = "https://planetarycomputer.microsoft.com/api/stac/v1"
_STAC_COLLECTION = "sentinel-1-rtc"

# RTC COGs carry this sentinel in the float32 raster (verified: raster:bands
# nodata == -32768). Anything ≤ this (or non-finite) is nodata, scrubbed to
# NaN before compositing so a swath-edge pixel never drags the median.
_RTC_NODATA = -32768.0

# ── Public API ───────────────────────────────────────────────────────────


def _open_client() -> Any:
    """Return an opened PC STAC client with the sign-in-place modifier.

    ``planetary_computer.sign_inplace`` as the client ``modifier`` signs every
    item the search yields, so asset hrefs are ready for ``/vsicurl/`` without
    a per-item ``pc.sign`` round-trip. Anonymous works; a
    ``PC_SDK_SUBSCRIPTION_KEY`` in the environment (if a secret exists) is
    picked up automatically by the planetary-computer SDK and lifts rate
    limits — no code change needed either way.
    """
    try:
        from pystac_client import Client
    except ImportError as e:  # pragma: no cover - env guard
        raise ImportError(
            "pc_s1_rtc requires pystac-client. Install: pip install pystac-client"
        ) from e
    try:
        import planetary_computer as pc
    except ImportError as e:  # pragma: no cover - env guard
        raise ImportError(
            "pc_s1_rtc requires planetary-computer. "
            "Install: pip install planetary-computer"
        ) from e
    return Client.open(_STAC_ROOT, modifier=pc.sign_inplace)


def _search_rtc(
    client: Any,
    bbox_4326: tuple[float, float, float, float],
    dt_from: str,
    dt_to: str,
    label: str,
) -> list[Any] | None:
    """RTC STAC search over the window; returns (signed) items or ``None``.

    PC's STAC frontend does not WAF-throttle the way CDSE's does, so no
    process-global search spacing is needed here (the CDSE path's
    ``_stac_rate_limit`` stays where it is, untouched). Returns ``None`` only
    on a hard search failure so the caller can distinguish "no scene" (``[]``)
    from "search broke".
    """
    try:
        search = client.search(
            collections=[_STAC_COLLECTION],
            bbox=list(bbox_4326),
            datetime=f"{dt_from}/{dt_to}",
            limit=100,
        )
        return list(search.items())
    except Exception as e:  # noqa: BLE001 — surfaced to the caller as None
        print(f"    [PC RTC] {label}: search failed: {e}")
        return None


def probe_orbits_with_items(
    west: float, south: float, east: float, north: float,
    *,
    windows: list[tuple[tuple[int, int], int]],
    crs: str = "http://www.opengis.net/def/crs/EPSG/0/3006",
    client: Any | None = None,
) -> tuple[str | None, dict[int, list[Any]]]:
    """Dominant orbit across ``windows`` + the (signed) items per window index.

    One STAC search per window. Returns the orbit direction with the most
    valid IW passes across all windows (chosen ONCE, reused for both
    composites so their backscatter is comparable — RTC is terrain-corrected
    but ASC vs DESC still differ in shadow/layover masking, so never mix), or
    ``None`` if no scene was found in any window; and ``{window_index:
    [items]}`` so the composites reuse the fetched items without re-searching.
    """
    from rasterio.warp import transform_bounds

    from .vpp_windows import doy_to_date_str

    epsg = s1_shared.crs_uri_to_epsg(crs)
    bbox_4326 = transform_bounds(
        f"EPSG:{epsg}", "EPSG:4326", west, south, east, north, densify_pts=21,
    )
    if client is None:
        client = _open_client()

    counts: dict[str, int] = {"ASCENDING": 0, "DESCENDING": 0}
    items_by_window: dict[int, list[Any]] = {}
    for wi, ((doy_start, doy_end), year) in enumerate(windows):
        date_start = doy_to_date_str(year, max(1, doy_start))
        date_end = doy_to_date_str(year, min(365, doy_end))
        items = _search_rtc(
            client, bbox_4326,
            f"{date_start}T00:00:00Z", f"{date_end}T23:59:59Z",
            f"{date_start}..{date_end}",
        )
        items = list(items) if items else []
        items_by_window[wi] = items
        for it in s1_shared.filter_iw_grdh(items, None):
            orbit = s1_shared.orbit_from_item(it)
            if orbit in counts:
                counts[orbit] += 1

    if counts["ASCENDING"] == 0 and counts["DESCENDING"] == 0:
        return None, items_by_window
    return max(counts, key=counts.get), items_by_window


def fetch_s1_season_composite(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    doy_window: tuple[int, int],
    year: int,
    orbit_direction: str,
    crs: str = "http://www.opengis.net/def/crs/EPSG/0/3006",
    size_px: int | tuple[int, int] = 256,
    max_scenes: int = 3,
    output_db: bool = False,
    nodata_threshold: float = 0.10,
    items: list[Any] | None = None,
    client: Any | None = None,
) -> tuple[np.ndarray, list[str], str] | None:
    """Per-orbit **median** VV/VH RTC γ⁰ season composite over ≤``max_scenes``.

    Same semantics as ``cdse_s1_stac.fetch_s1_season_composite`` — one orbit
    direction, ≤``max_scenes`` scenes spread across the window, per-pixel
    nodata-aware median, >``nodata_threshold`` scene rejection — but reads RTC
    γ⁰ COGs by windowed ``/vsicurl/`` streaming (no product download) and
    returns **linear** γ⁰ (see module docstring on the units choice).

    Args:
        west, south, east, north: Tile bbox in ``crs`` (EPSG:3006 grid edges).
        doy_window: ``(doy_start, doy_end)`` inclusive growing-season DOYs.
        year: Calendar year the window belongs to (label year or 2016).
        orbit_direction: ``"ASCENDING"`` / ``"DESCENDING"`` — the tile's
            dominant orbit, used for BOTH composites so they stay comparable.
        crs: OGC CRS URI for the bbox. Default EPSG:3006.
        size_px: Output H×W (int square or ``(H, W)``).
        max_scenes: Cap on scenes contributing to the median (speckle sweet
            spot ≈ 3).
        output_db: Kept for signature-parity with the CDSE backend; RTC v3
            stores **linear** γ⁰ so the default is ``False``. Setting it True
            would break consistency with the normalizer (which log-transforms
            internally) — left as a knob, not used by the enrich script.
        nodata_threshold: Reject a scene whose VV nodata (NaN) fraction over
            the window exceeds this (swath-edge / partial coverage).
        items: Pre-fetched, PC-signed items for this window (from
            :func:`probe_orbits_with_items`) — skips the redundant search.
        client: Opened PC STAC client (reused across windows). ``None`` opens
            one — but items must then already be signed, or a fresh client is
            opened for signing parity.

    Returns:
        ``(sar, contributing_dates, orbit)`` on success:

            * ``sar``: ``(2, H, W)`` float32 median composite, [VV, VH],
              **linear** γ⁰. Genuine all-scene-nodata pixels are 0.
            * ``contributing_dates``: ISO ``YYYY-MM-DD`` per scene used.
            * ``orbit``: the resolved orbit direction (== ``orbit_direction``).

        ``None`` when no same-orbit scene survived the window + nodata gate.
    """
    from rasterio.warp import transform_bounds

    from .vpp_windows import doy_to_date_str

    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px
    s1_shared.assert_bbox_size_match(west, south, east, north, h_px, w_px)
    epsg = s1_shared.crs_uri_to_epsg(crs)

    bbox_4326 = transform_bounds(
        f"EPSG:{epsg}", "EPSG:4326", west, south, east, north, densify_pts=21,
    )

    if items is None:
        if client is None:
            client = _open_client()
        doy_start, doy_end = doy_window
        date_start = doy_to_date_str(year, max(1, doy_start))
        date_end = doy_to_date_str(year, min(365, doy_end))
        items = _search_rtc(
            client, bbox_4326,
            f"{date_start}T00:00:00Z", f"{date_end}T23:59:59Z",
            f"{date_start}..{date_end}",
        )
    if not items:
        return None

    # ONE orbit direction only — RTC removes terrain geometry, but ASC/DESC
    # still differ in shadow/layover masking, so a mixed median corrupts the
    # composite. filter_iw_grdh drops the other orbit; the loop re-checks.
    items = s1_shared.filter_iw_grdh(items, orbit_direction)
    if not items:
        return None

    ordered = _select_spread_scenes(items, bbox_4326, max_scenes)

    vv_stack: list[np.ndarray] = []
    vh_stack: list[np.ndarray] = []
    used_dates: list[str] = []
    for item in ordered:
        obs_orbit = s1_shared.orbit_from_item(item)
        if obs_orbit and obs_orbit.upper() != orbit_direction.upper():
            continue  # defensive; filter_iw_grdh should have removed these
        sar = _read_rtc_scene(item, west, south, east, north, epsg,
                              h_px, w_px, output_db=output_db)
        if sar is None:
            continue
        # nodata gate on the VV NaN fraction (RTC nodata → NaN in _read).
        if float(np.isnan(sar[0]).mean()) > nodata_threshold:
            continue
        vv_stack.append(sar[0])
        vh_stack.append(sar[1])
        dt = s1_shared.item_datetime(item)
        used_dates.append(dt.strftime("%Y-%m-%d") if dt else "")
        if len(vv_stack) >= max_scenes:
            break

    if not vv_stack:
        return None

    vv = _nan_median(np.stack(vv_stack, axis=0))
    vh = _nan_median(np.stack(vh_stack, axis=0))
    composite = np.stack([vv, vh], axis=0).astype(np.float32)
    return composite, used_dates, orbit_direction.upper()


# ── Internals ─────────────────────────────────────────────────────────────


def _select_spread_scenes(
    items: list[Any],
    bbox_4326: tuple[float, float, float, float],
    max_scenes: int,
) -> list[Any]:
    """Order candidate scenes so a ≤``max_scenes`` prefix is spread in time.

    Identical policy to the CDSE backend (delegates to it to keep the
    spread/tie-break rule in one place): sort by date, tie-break by bbox
    overlap, take a ``2*max_scenes`` evenly-spaced subset so nodata rejects
    leave slack and the surviving median samples the whole season, not a
    temporal cluster.
    """
    from .cdse_s1_stac import _select_spread_scenes as _cdse_spread
    return _cdse_spread(items, bbox_4326, max_scenes)


def _read_rtc_scene(
    item: Any,
    west: float, south: float, east: float, north: float,
    epsg: int, h_px: int, w_px: int,
    *,
    output_db: bool,
) -> np.ndarray | None:
    """Windowed read of one RTC scene's VV+VH → ``(2, H, W)`` linear γ⁰.

    RTC is analysis-ready: no calibration, no DN². The COG is a north-up UTM
    float32 raster, so ``s1_shared.read_window`` reprojects the tile's
    EPSG:3006 bbox into the COG's CRS and reads the window at ``size_px`` —
    the same reproject-onto-tile-grid path the GRD backend uses, minus the
    WarpedVRT branch (RTC has a real ``ds.crs``).

    RTC nodata (-32768) is mapped to **NaN** so the composite median and the
    nodata gate treat swath edges correctly. Returns ``None`` on any read
    failure so the composite loop skips a bad scene without aborting the tile.
    """
    try:
        vv_url, vh_url = _rtc_measurement_urls(item)
    except RuntimeError as e:
        print(f"    [PC RTC] {item.id}: asset selection failed: {e}")
        return None
    try:
        vv, _ = s1_shared.read_window(
            f"/vsicurl/{vv_url}", west, south, east, north, epsg, h_px, w_px)
        vh, _ = s1_shared.read_window(
            f"/vsicurl/{vh_url}", west, south, east, north, epsg, h_px, w_px)
    except Exception as e:  # noqa: BLE001
        print(f"    [PC RTC] {item.id}: window read failed: {e}")
        return None

    vv = _mask_nodata(vv)
    vh = _mask_nodata(vh)
    if output_db:
        # Optional dB — not used by the v3 enrich script; the normalizer
        # log-transforms linear input, so linear is the stored default.
        vv = _to_db_nan(vv)
        vh = _to_db_nan(vh)
    return np.stack([vv, vh], axis=0).astype(np.float32)


def _rtc_measurement_urls(item: Any) -> tuple[str, str]:
    """Return the (signed) VV and VH RTC asset hrefs.

    RTC items name the measurement assets exactly ``vv`` / ``vh`` (verified
    against the collection's item_assets), so this is a direct lookup — no
    calibration/noise/annotation disambiguation the GRD path needs. Items are
    pre-signed by the ``sign_inplace`` client modifier.
    """
    assets = item.assets or {}
    vv = assets.get("vv")
    vh = assets.get("vh")
    if vv is None or vh is None:
        raise RuntimeError(
            f"Item {item.id} missing vv/vh RTC asset (assets: {list(assets)})"
        )
    return vv.href, vh.href


def _mask_nodata(arr: np.ndarray) -> np.ndarray:
    """RTC nodata (-32768) and non-finite → NaN; keep valid γ⁰ as-is."""
    out = arr.astype(np.float32, copy=True)
    out[out <= _RTC_NODATA] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def _to_db_nan(x: np.ndarray) -> np.ndarray:
    """Linear γ⁰ → dB, preserving NaN nodata (unused by the v3 default path)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(x)
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


def _nan_median(stack: np.ndarray) -> np.ndarray:
    """Per-pixel median over ``(N, H, W)``, NaN = nodata.

    A pixel valid in ≥1 scene → median of its finite values. A pixel NaN in
    EVERY scene → 0 (genuine gap, matching the composite's "0 = nodata"
    output contract consumed by the dataset's NaN-scrub).
    """
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(stack, axis=0)
    return np.nan_to_num(med, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
