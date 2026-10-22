"""Sentinel-1 GRD STAC helpers for Microsoft Planetary Computer.

Anonymous search of the ``sentinel-1-grd`` collection plus SAS signing of
the asset hrefs (``planetary_computer.sign()``, ~1 h). Used by the vessel
chain (``imint.fetch.fetch_s1_vessel_scenes``), which geocodes the GRD on
the sea surface itself.

The σ⁰ fetcher that lived here (``fetch_s1_scene``) is retired (task 0113):
it warped the GRD over its ellipsoid-height tie points, so the sea landed
41–56 m toward the satellite. Use ``imint.training.pc_s1_rtc`` (RTC) for
backscatter.
"""
from __future__ import annotations

from typing import Any

# ── Module constants ─────────────────────────────────────────────────────

_STAC_ROOT = "https://planetarycomputer.microsoft.com/api/stac/v1"
_STAC_COLLECTION = "sentinel-1-grd"


# ── STAC helpers (shared with imint.fetch.fetch_s1_vessel_scenes) ────────

def _stac_client() -> Any:
    """Anonymous MPC STAC client. Items come back unsigned; see :func:`sign_item`."""
    try:
        from pystac_client import Client
    except ImportError as e:
        raise ImportError(
            "mpc_s1 requires pystac-client. Install: pip install pystac-client"
        ) from e
    return Client.open(_STAC_ROOT)


def sign_item(item: Any) -> Any:
    """Return a copy of ``item`` whose asset hrefs carry an MPC SAS token (~1 h)."""
    try:
        import planetary_computer as pc
    except ImportError as e:
        raise ImportError(
            "mpc_s1 requires planetary-computer. "
            "Install: pip install planetary-computer"
        ) from e
    return pc.sign(item)


def search_grd_items(
    datetime_range: str,
    *,
    bbox_4326: tuple[float, float, float, float] | None = None,
    intersects: dict | None = None,
    limit: int = 50,
) -> list[Any]:
    """Search the MPC ``sentinel-1-grd`` collection.

    Args:
        datetime_range: STAC interval, ``"<start>/<end>"`` (ISO 8601).
        bbox_4326: (west, south, east, north) in degrees, or
        intersects: a GeoJSON geometry; exactly one of the two must be given.
        limit: page size.

    Returns:
        Unsigned items, unfiltered (use ``s1_shared.filter_iw_grdh``).
    """
    if (bbox_4326 is None) == (intersects is None):
        raise ValueError("give exactly one of bbox_4326 / intersects")
    spatial = {"bbox": list(bbox_4326)} if bbox_4326 is not None else {"intersects": intersects}
    search = _stac_client().search(
        collections=[_STAC_COLLECTION], datetime=datetime_range, limit=limit, **spatial,
    )
    return list(search.items())
