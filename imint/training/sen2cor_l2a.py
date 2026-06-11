"""L1C scene-selection + sen2cor-L2A reflectance-window primitives.

The shared plumbing for the inline ``l1c_sen2cor`` fetch fallback: CDSE-STAC L1C
existence lookup, sen2cor ``L2A_Process`` invocation, and L2A all-band window
reads. Lifted from the (retiring) ``scripts/sen2cor_pipeline/*`` scene-batch so
the inline path reuses that proven logic rather than reimplementing it.

Runtime: ``run_sen2cor`` shells out to ``L2A_Process``, which exists only inside
the ``ghcr.io/tobiasedman/imint-sen2cor`` image. Outside it, ``subprocess.run``
raises ``FileNotFoundError`` — the inline backend treats that as "source dead"
and degrades to a no-op (it never reaches here without the SAFE already on disk).
"""
from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np

# The inline fallback returns the full 12-band ALL_BANDS cube — the SAME
# contract the openEO backends return — so the caller's _split_all_bands derives
# the 6-band model input PLUS the b08/rededge/b01/b09 extras. Import the
# canonical order so the sen2cor path and the openEO path can never drift.
from imint.training.openeo_tile_graph import ALL_BANDS


# ── L1C scene selection (CDSE STAC, existence-only) ──────────────────────────

_CDSE_STAC_ROOT = "https://stac.dataspace.copernicus.eu/v1"
_L1C_COLLECTION = "sentinel-2-l1c"


def stac_l1c_scenes(bbox_wgs84: dict, date_start: str, date_end: str) -> list[dict]:
    """All L1C scenes intersecting bbox in the window, via the CDSE STAC.

    Existence-only — the **full archive incl. pre-2018**, which is the whole
    point of the fallback. NO cloud gate here: the granule-average
    ``eo:cloud_cover`` (~110×110 km) is captured as a tie-breaker only; the real
    cloud decision is the tile-level ERA5 screen in the caller. (This is why the
    l1c_sen2cor fallback does NOT use ``fetch._stac_best_l1c_scene``, whose
    ranking is per-scene cloud over the pre-2018-blind DES STAC.)

    Each dict: ``scene_id`` (== the SAFE name ``fetch_l1c_safe_by_name`` takes),
    ``datetime``, ``cloud_pct``, ``mgrs_tile``.
    """
    from pystac_client import Client

    from imint.training.optimal_fetch import retry_on_rate_limit

    def _query() -> list:
        client = Client.open(_CDSE_STAC_ROOT)
        search = client.search(
            collections=[_L1C_COLLECTION],
            bbox=[bbox_wgs84["west"], bbox_wgs84["south"],
                  bbox_wgs84["east"], bbox_wgs84["north"]],
            datetime=f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
            limit=300,
        )
        return list(search.items())

    out: list[dict] = []
    for item in retry_on_rate_limit(_query):
        props = item.properties or {}
        mgrs = props.get("s2:mgrs_tile") or props.get("mgrs_tile") or ""
        if not mgrs:
            seg = item.id.split("_T")
            mgrs = seg[-1].split("_")[0] if len(seg) > 1 else ""
        out.append({
            "scene_id": item.id,
            "datetime": props.get("datetime") or props.get("start_datetime"),
            "cloud_pct": float(props.get("eo:cloud_cover", 100.0)),
            "mgrs_tile": mgrs,
        })
    return out


# ── L1C / L2A band-file location + window reads ──────────────────────────────


def find_band_jp2(safe_dir: Path, band: str) -> Path | None:
    """Locate the JP2 for a band inside an L1C SAFE GRANULE/.../IMG_DATA."""
    matches = list(safe_dir.glob(f"GRANULE/*/IMG_DATA/*_{band}.jp2"))
    return matches[0] if matches else None


def l2a_band_path(l2a_dir: Path, band: str, res: int = 10) -> Path | None:
    """Find an L2A band JP2 at the given resolution (falls back to R20m)."""
    matches = list(l2a_dir.glob(f"GRANULE/*/IMG_DATA/R{res}m/*_{band}_*.jp2"))
    if not matches:
        # B11/B12 are native 20 m; sen2cor also writes them under R10m
        matches = list(l2a_dir.glob(f"GRANULE/*/IMG_DATA/R20m/*_{band}_*.jp2"))
    return matches[0] if matches else None


def read_window(jp2_path: Path, bbox_3006: dict, out_px: int) -> np.ndarray | None:
    """Read an ``out_px × out_px`` window at the EPSG:3006 bbox, bilinear.

    Returns float32 reflectance (DN / 10000) or ``None`` on failure.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds

    try:
        with rasterio.open(jp2_path) as ds:
            dst_bounds = transform_bounds(
                "EPSG:3006", ds.crs,
                bbox_3006["west"], bbox_3006["south"],
                bbox_3006["east"], bbox_3006["north"],
                densify_pts=21,
            )
            win = window_from_bounds(*dst_bounds, transform=ds.transform)
            dn = ds.read(
                1, window=win, out_shape=(out_px, out_px),
                resampling=Resampling.bilinear, boundless=True, fill_value=0,
            ).astype(np.float32)
    except Exception:
        return None
    return dn / 10000.0


# L2A native resolution per band — sen2cor writes each band in its native-res
# folder (R10m: B02/B03/B04/B08; R20m: B05/B06/B07/B8A/B11/B12; R60m: B01/B09).
# Reading each at native res avoids upsampling artefacts from a wrong-res folder.
_L2A_BAND_RES = {
    "B02": 10, "B03": 10, "B04": 10, "B08": 10,
    "B05": 20, "B06": 20, "B07": 20, "B8A": 20, "B11": 20, "B12": 20,
    "B01": 60, "B09": 60,
}


def read_l2a_allband(l2a_dir: Path, bbox_3006: dict, out_px: int) -> np.ndarray | None:
    """L2A SAFE → ``(len(ALL_BANDS), out_px, out_px)`` cube in ALL_BANDS order.

    The full 12-band contract the openEO backends return — so the caller's
    ``_split_all_bands`` yields the 6-band model input + the b08/rededge/b01/b09
    extras (a 6-band-only read would silently drop the extras and make this a
    second-class backend). Each band is read from its native-resolution folder
    (10/20/60 m) and resampled to ``out_px``. Returns ``None`` if any band JP2 is
    missing or any window read fails, so a partial scene never half-fills a frame.
    """
    chans = []
    for b in ALL_BANDS:
        jp2 = l2a_band_path(l2a_dir, b, res=_L2A_BAND_RES[b])
        if jp2 is None:
            return None
        arr = read_window(jp2, bbox_3006, out_px)
        if arr is None:
            return None
        chans.append(arr)
    return np.stack(chans, axis=0)


# ── sen2cor invocation ───────────────────────────────────────────────────────


def ensure_l1c_datastrip_qi_data(safe_dir: Path, *, log: Callable[[str], None] = print) -> None:
    """Create an empty ``DATASTRIP/<DS_dir>/QI_DATA`` if it's missing.

    Sen2Cor 2.12.04's ``L2A_ProcessDataStrip.generate()`` (TOOLBOX mode)
    renames L1C's ``DATASTRIP/DS_<...>/`` to the L2A name and then does
    ``os.listdir(newdir/QI_DATA)`` to scrub stale ``.xml`` files. If
    ``QI_DATA/`` doesn't exist inside the renamed dir, ``os.listdir`` raises
    ``OSError [Errno 2]`` and the whole scene is lost.

    Some Collection-1 reprocessed L1C SAFEs ship ``DATASTRIP/<DS>/MTD_DS.xml``
    but no ``QI_DATA`` subdirectory at all (per-SAFE, not baseline-dependent).
    The Sen2Cor iteration body removes ``*.xml`` from the dir — an empty
    ``QI_DATA`` is a valid no-op, so creating an empty directory satisfies the
    listdir call without affecting L2A correctness. A robustness workaround for
    a Sen2Cor gap, not a data fix.
    """
    datastrip = safe_dir / "DATASTRIP"
    if not datastrip.is_dir():
        return
    for ds_subdir in datastrip.iterdir():
        if not ds_subdir.is_dir():
            continue
        qi = ds_subdir / "QI_DATA"
        if not qi.exists():
            try:
                qi.mkdir()
                log(f"    [qi-data-shim] created empty QI_DATA in "
                    f"{ds_subdir.name} (was missing from L1C SAFE)")
            except Exception as exc:
                log(f"    [qi-data-shim] WARN: mkdir {qi} failed: "
                    f"{type(exc).__name__}: {exc}")


def run_sen2cor(
    safe_dir: Path, work_dir: Path, *, log: Callable[[str], None] = print,
) -> Path | None:
    """Run ``L2A_Process`` on an L1C SAFE; return the produced L2A SAFE dir.

    Returns ``None`` if sen2cor crashes or times out on this scene. Raises
    ``FileNotFoundError`` if ``L2A_Process`` is not installed (caller's cue that
    this backend is unavailable in the current image).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    # Workaround for Sen2Cor 2.12.04 TOOLBOX-mode crash on L1C SAFEs that lack
    # DATASTRIP/<DS>/QI_DATA. See ensure_l1c_datastrip_qi_data for the diagnosis.
    ensure_l1c_datastrip_qi_data(safe_dir, log=log)
    cmd = [
        "L2A_Process",
        "--resolution", "10",
        "--output_dir", str(work_dir),
        str(safe_dir),
    ]
    try:
        # 1 h: under multi-worker core contention a single L2A_Process is far
        # slower than standalone (30 min timed out under 6-way contention).
        subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        # Surface why L2A_Process died — a scene-flaky crash is otherwise
        # indistinguishable from a transient one.
        tail = ""
        for stream in (getattr(e, "stderr", None), getattr(e, "stdout", None)):
            if stream:
                text = stream.decode("utf-8", "replace") if isinstance(stream, bytes) else str(stream)
                lines = [ln for ln in text.splitlines() if ln.strip()]
                if lines:
                    tail = "\n      ".join(lines[-15:])
                    break
        log(f"    sen2cor failed: {type(e).__name__}"
            + (f"\n      {tail}" if tail else " (no output captured)"))
        return None
    l2a = sorted(work_dir.glob("*MSIL2A*.SAFE"))
    return l2a[0] if l2a else None
