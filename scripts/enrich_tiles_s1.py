#!/usr/bin/env python3
"""Add Sentinel-1 SAR (VV/VH) season composites to existing tiles (v3).

For each tile this builds TWO per-orbit **median season composites** and
overwrites any older per-frame / v2 S1 stack:

    1. Label-year growing season  → ``s1_vv_vh``      (2, H, W)
       — what CROMA / TerraMind consume.
    2. 2016 season (same DOY window) → ``s1_vv_vh_2016`` (2, H, W)
       — SAR analogue of the ``frame_2016`` clearcut change anchor; written
       only where ``has_frame_2016 == 1``.

Rationale (docs/plans/s1_monthly_enrichment.md): land cover is stable over
weeks, and the model glue consumes exactly ONE S1 frame. A per-orbit median
over ≤3 same-orbit scenes across the growing season gives full coverage and
suppresses speckle, replacing the sparse ±3-day co-dating that left
1,871/7,882 tiles SAR-blind.

Growing-season window
    Preferred: the tile's own persisted VPP phenology (``vpp_sosd`` /
    ``vpp_eosd``, YYDDD-encoded) → SOSD→EOSD DOY span, offline, no CDSE call.
    Fallback: a latitude-scaled May–Sep window (documented in
    ``_growing_season_window``) when VPP is absent/degenerate.

Orbit consistency (the correctness risk)
    A median across mixed ASC/DESC orbits corrupts backscatter (different
    look geometry). The tile's dominant orbit — most valid passes across
    both windows — is chosen ONCE and used for BOTH composites.

Backends (``--s1-backend``)
    ``pc-rtc`` (default) — Planetary Computer ``sentinel-1-rtc``: RTC γ⁰,
        terrain-corrected, analysis-ready COGs, **windowed** ``/vsicurl/``
        reads (no product download → ~MB/tile, no PVC cache). Stores
        **linear** γ⁰ (the SAR normalizers log-transform internally). Free,
        anonymous. ``imint.training.pc_s1_rtc``.
    ``cdse-stac`` — the legacy CDSE STAC + full-GRD-download + local-σ⁰ path
        (``imint.training.cdse_s1_stac``), kept as a fallback. σ⁰ on the
        ellipsoid (no terrain correction), ~2 GB/product cached on the PVC,
        stores dB. Bills OData bandwidth + COG requests, NOT the PU pool.

Both need pystac-client + rasterio; pc-rtc also needs planetary-computer;
cdse-stac also needs scipy (calibration LUT) + CDSE creds.

Atomicity
    Writes go to ``<tile>.npz.tmp.npz`` and are atomically renamed on
    success (unchanged). Stale .tmp files are cleaned on start.

Keys written (v3)
    s1_vv_vh        (2, H, W) float32 — label-year median composite
                      (pc-rtc: LINEAR γ⁰; cdse-stac: dB σ⁰)
    s1_vv_vh_2016   (2, H, W) float32 — 2016 median composite (if has_frame_2016)
    s1_dates        (K,) str  — contributing scene dates, label year
    s1_dates_2016   (K,) str  — contributing scene dates, 2016
    s1_orbit        str        — chosen orbit ("ASCENDING"/"DESCENDING")
    s1_source       str        — "pc-rtc-gamma0" or "cdse-grd-sigma0"
    has_s1          int32      — 1 if the label-year composite was written
    s1_enrich_v     int32      — 3 (version marker; --skip-existing keys on ==3)

Re-enrichment
    Older tiles (v1 ±3-day, v2 CDSE-dB) are RE-enriched: ``has_s1`` no longer
    short-circuits; ``--skip-existing`` skips only tiles already at
    ``s1_enrich_v == 3``. The 5 CDSE-v2 tiles are re-enriched to v3 so the
    whole set is linear-γ⁰ / RTC-consistent.

Usage:
    python scripts/enrich_tiles_s1.py \\
        --data-dir /data/unified_v2_512 \\
        --s1-backend pc-rtc \\
        --workers 4 \\
        --skip-existing \\
        --limit 5            # smoke run over the first 5 tiles
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The scope of a "growing season" median composite. Both composites share the
# tile's dominant orbit; ≤3 scenes spread across the window is the speckle
# sweet spot (plan §Budget).
MAX_SCENES = 3
NODATA_THRESHOLD = 0.10
S1_ENRICH_VERSION = 3

# Per-backend module + stored-units + provenance. pc-rtc stores LINEAR γ⁰
# (the CROMA/TerraMind normalizers apply 10*log10 internally — storing dB
# would double-log); cdse-stac keeps its historical dB σ⁰ output. Default is
# pc-rtc: RTC gamma0, windowed, no product download.
_BACKENDS = {
    "pc-rtc": {
        "module": "imint.training.pc_s1_rtc",
        "output_db": False,
        "source": "pc-rtc-gamma0",
    },
    "cdse-stac": {
        "module": "imint.training.cdse_s1_stac",
        "output_db": True,
        "source": "cdse-grd-sigma0",
    },
}
_DEFAULT_BACKEND = "pc-rtc"

# Latitude-scaled May–Sep fallback window when VPP phenology is unavailable.
# Sweden spans ~55.3°N (Smygehuk) to ~69.1°N (Treriksröset). Green-up is later
# and senescence earlier further north, so the window shifts + shortens with
# latitude. DOY 121 = May 1, 273 = Sep 30 (non-leap; ±1 day is immaterial for
# a season composite). Anchored to the southern extreme, then nudged.
_FALLBACK_SOUTH_LAT = 55.0
_FALLBACK_NORTH_LAT = 69.0
_FALLBACK_START_SOUTH_DOY = 121  # May 1 in the south
_FALLBACK_START_NORTH_DOY = 152  # Jun 1 in the far north
_FALLBACK_END_SOUTH_DOY = 273    # Sep 30 in the south
_FALLBACK_END_NORTH_DOY = 244    # Sep 1 in the far north


def _growing_season_window(data: dict) -> tuple[tuple[int, int], str]:
    """Return the tile's ``(doy_start, doy_end)`` growing season + its source.

    Preferred source is the persisted VPP phenology (``vpp_sosd``/``vpp_eosd``,
    YYDDD-encoded HR-VPP bands) decoded to a SOSD→EOSD DOY span — fully
    offline, no CDSE call. Falls back to a latitude-scaled May–Sep window
    when VPP is absent or decodes to a degenerate span.

    Returns ``((doy_start, doy_end), source)`` where source is ``"vpp"`` or
    ``"latitude_fallback"``.
    """
    sosd = data.get("vpp_sosd")
    eosd = data.get("vpp_eosd")
    if sosd is not None and eosd is not None:
        try:
            from imint.training.vpp_windows import compute_growing_season_doy

            gs = compute_growing_season_doy(
                np.asarray(sosd, dtype=np.float64),
                np.asarray(eosd, dtype=np.float64),
            )
            if gs is not None:
                return gs, "vpp"
        except Exception:
            pass
    return _latitude_fallback_window(data), "latitude_fallback"


def _latitude_fallback_window(data: dict) -> tuple[int, int]:
    """Latitude-scaled May–Sep DOY window from the tile's EPSG:3006 northing."""
    lat = _tile_latitude(data)
    if lat is None:
        # No geometry to scale on — southern-Sweden default (widest window).
        return (_FALLBACK_START_SOUTH_DOY, _FALLBACK_END_SOUTH_DOY)
    frac = (lat - _FALLBACK_SOUTH_LAT) / (_FALLBACK_NORTH_LAT - _FALLBACK_SOUTH_LAT)
    frac = float(np.clip(frac, 0.0, 1.0))
    start = round(_FALLBACK_START_SOUTH_DOY
                  + frac * (_FALLBACK_START_NORTH_DOY - _FALLBACK_START_SOUTH_DOY))
    end = round(_FALLBACK_END_SOUTH_DOY
                + frac * (_FALLBACK_END_NORTH_DOY - _FALLBACK_END_SOUTH_DOY))
    return (int(start), int(end))


def _tile_latitude(data: dict) -> float | None:
    """Approximate tile-centre latitude (deg) from persisted EPSG:3006 geom."""
    northing = data.get("northing")
    easting = data.get("easting")
    if northing is None or easting is None:
        return None
    try:
        from pyproj import Transformer

        tf = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)
        _, lat = tf.transform(float(easting), float(northing))
        return float(lat)
    except Exception:
        return None


# Year derivation is centralised in imint.training.tile_fetch so the
# TESSERA enricher, the label builder and this module cannot drift apart.
from imint.training.tile_fetch import (  # noqa: E402
    infer_tile_year as _tile_year,
)


def enrich_one_tile(
    tile_path: str,
    *,
    skip_existing: bool = True,
    backend: str = _DEFAULT_BACKEND,
) -> dict:
    """Build both S1 season composites for one tile .npz (v3).

    Tile geometry + label year come from persisted keys (no module-level
    grid constants). Writes atomically via ``<tile>.npz.tmp.npz`` →
    ``os.replace`` so a killed job leaves the original tile intact.

    ``backend`` selects the source module (``pc-rtc`` default → RTC γ⁰
    windowed reads; ``cdse-stac`` → legacy full-GRD σ⁰). Both expose the same
    ``probe_orbits_with_items`` / ``fetch_s1_season_composite`` contract.
    """
    import importlib

    from imint.training.tile_config import TileConfig
    from imint.training.tile_bbox import resolve_tile_bbox

    cfg = _BACKENDS[backend]
    mod = importlib.import_module(cfg["module"])
    fetch_s1_season_composite = mod.fetch_s1_season_composite
    probe_orbits_with_items = mod.probe_orbits_with_items
    output_db = cfg["output_db"]
    s1_source = cfg["source"]

    name = Path(tile_path).stem
    try:
        data = dict(np.load(tile_path, allow_pickle=True))
    except Exception as e:
        return {"name": name, "status": "failed", "reason": str(e)}

    if skip_existing and int(data.get("s1_enrich_v", 0)) == S1_ENRICH_VERSION:
        return {"name": name, "status": "skipped"}

    spectral = data.get("spectral", data.get("image"))
    if spectral is None:
        return {"name": name, "status": "failed", "reason": "no_spectral"}
    h, w = spectral.shape[1], spectral.shape[2]

    size_px = int(data.get("tile_size_px", h))
    tile_cfg = TileConfig(size_px=size_px)
    bbox = resolve_tile_bbox(name=name, tile=tile_cfg, npz_data=data)
    if bbox is None:
        return {"name": name, "status": "failed", "reason": "no_bbox"}
    tile_cfg.assert_bbox_matches(bbox)

    year = _tile_year(data)
    if year is None:
        return {"name": name, "status": "failed", "reason": "no_year"}

    (doy_start, doy_end), gs_source = _growing_season_window(data)
    has_2016 = int(data.get("has_frame_2016", 0)) == 1

    # Windows the orbit probe / composites cover: label year always; 2016 only
    # where the optical clearcut anchor exists (so the SAR anchor pairs it).
    windows = [((doy_start, doy_end), year)]
    if has_2016:
        windows.append(((doy_start, doy_end), 2016))

    w0, s0, e0, n0 = bbox["west"], bbox["south"], bbox["east"], bbox["north"]

    # pc-rtc reuses ONE signed STAC client across the probe + both composites
    # (connection reuse + consistent SAS signing). cdse-stac takes no client
    # kwarg, so pass it only when the backend exposes an _open_client.
    client_kw: dict = {}
    if hasattr(mod, "_open_client"):
        try:
            client_kw["client"] = mod._open_client()
        except Exception as exc:  # noqa: BLE001
            return {"name": name, "status": "failed",
                    "reason": f"client_open_error: {type(exc).__name__}: {exc}"}

    # --- Pick ONE orbit for the tile (dominant across both windows) ---------
    # probe_orbits_with_items runs ONE search per window and hands the items
    # back so the composites don't re-search.
    try:
        orbit, items_by_window = probe_orbits_with_items(
            w0, s0, e0, n0, windows=windows, **client_kw,
        )
    except Exception as exc:  # noqa: BLE001 — infra vs data distinction below
        return {"name": name, "status": "failed",
                "reason": f"orbit_probe_error: {type(exc).__name__}: {exc}"}
    if orbit is None:
        return {"name": name, "status": "failed",
                "reason": "no_s1_scene_in_windows"}

    # --- Label-year composite (required) ------------------------------------
    try:
        primary = fetch_s1_season_composite(
            w0, s0, e0, n0,
            doy_window=(doy_start, doy_end), year=year,
            orbit_direction=orbit, size_px=size_px,
            max_scenes=MAX_SCENES, output_db=output_db,
            nodata_threshold=NODATA_THRESHOLD,
            items=items_by_window.get(0), **client_kw,
        )
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "status": "failed",
                "reason": f"fetch_error: {type(exc).__name__}: {exc}"}
    if primary is None:
        return {"name": name, "status": "failed",
                "reason": f"no_composite_{orbit}_{year}"}

    sar, dates, resolved_orbit = primary

    # --- 2016 composite (optional; same orbit, same DOY window) -------------
    sar_2016 = None
    dates_2016: list[str] = []
    if has_2016:
        try:
            comp16 = fetch_s1_season_composite(
                w0, s0, e0, n0,
                doy_window=(doy_start, doy_end), year=2016,
                orbit_direction=orbit, size_px=size_px,
                max_scenes=MAX_SCENES, output_db=output_db,
                nodata_threshold=NODATA_THRESHOLD,
                items=items_by_window.get(1), **client_kw,
            )
        except Exception as exc:  # noqa: BLE001 — 2016 gap must not fail the tile
            comp16 = None
            print(f"    [{name}] 2016 composite error "
                  f"({type(exc).__name__}: {exc}) — writing label-year only")
        if comp16 is not None:
            sar_2016, dates_2016, _ = comp16

    # --- Write v3 keys; drop any v1/v2 leftovers ----------------------------
    for stale in ("s1_temporal_mask",):
        data.pop(stale, None)

    data["s1_vv_vh"] = sar.astype(np.float32)              # (2, H, W)
    data["s1_dates"] = np.array(dates)
    data["s1_orbit"] = np.bytes_(resolved_orbit)
    data["s1_source"] = np.bytes_(s1_source)
    data["has_s1"] = np.int32(1)
    data["s1_enrich_v"] = np.int32(S1_ENRICH_VERSION)
    if sar_2016 is not None:
        data["s1_vv_vh_2016"] = sar_2016.astype(np.float32)  # (2, H, W)
        data["s1_dates_2016"] = np.array(dates_2016)
    else:
        # No 2016 composite this pass — don't leave a stale one from a prior run.
        data.pop("s1_vv_vh_2016", None)
        data.pop("s1_dates_2016", None)

    tmp_path = tile_path + ".tmp.npz"
    try:
        np.savez_compressed(tmp_path[:-4], **data)  # strips .npz, re-adds
        os.replace(tmp_path, tile_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise

    return {
        "name": name, "status": "ok",
        "orbit": resolved_orbit, "gs_source": gs_source,
        "n_primary": len(dates), "n_2016": len(dates_2016),
        "has_2016": sar_2016 is not None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Enrich tiles with S1 SAR season composites (v3)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--s1-backend", choices=sorted(_BACKENDS), default=_DEFAULT_BACKEND,
        help="Source backend. 'pc-rtc' (default): Planetary Computer RTC γ⁰, "
             "windowed reads, linear γ⁰, no download. 'cdse-stac': legacy "
             "CDSE full-GRD σ⁰ (dB), cached on PVC.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing",
                        action="store_false")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Alias for --max-tiles; caps the number of tiles processed "
             "(use for cluster smoke runs, e.g. --limit 5).",
    )
    args = parser.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    cap = args.limit if args.limit is not None else args.max_tiles
    if cap:
        tiles = tiles[:cap]

    # Clean stale .tmp.npz from prior aborted runs.
    stale = glob.glob(os.path.join(args.data_dir, "*.npz.tmp.npz"))
    stale += glob.glob(os.path.join(args.data_dir, "*.tmp.npz"))
    for s in stale:
        try:
            os.unlink(s)
        except FileNotFoundError:
            pass
    if stale:
        print(f"  Cleaned {len(stale)} stale .tmp.npz file(s) from prior runs")

    print("=== S1 SAR season-composite enrichment (v3) ===")
    print(f"  Tiles:   {len(tiles)}")
    print(f"  Workers: {args.workers}")
    print(f"  Backend: {args.s1_backend} "
          f"({_BACKENDS[args.s1_backend]['module']}, "
          f"source={_BACKENDS[args.s1_backend]['source']})")

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()
    completed = 0
    consecutive_fetch_error_fails = 0
    abort = threading.Event()
    t0 = time.time()

    def _run(path):
        nonlocal completed, consecutive_fetch_error_fails
        if abort.is_set():
            return
        r = enrich_one_tile(path, skip_existing=args.skip_existing,
                            backend=args.s1_backend)
        with lock:
            completed += 1
            reason = r.get("reason", "")
            # Systematic infra failure (missing dep, bad creds) aborts loudly
            # rather than marking thousands of tiles failed at 1000s/h.
            if "_error" in reason and r.get("status") == "failed":
                consecutive_fetch_error_fails += 1
                if consecutive_fetch_error_fails >= 10 and not abort.is_set():
                    abort.set()
                    print(f"\nABORT: {consecutive_fetch_error_fails} consecutive "
                          f"tiles failed with fetch/infra errors — environment "
                          f"problem, not scene absence. Last: {reason}",
                          flush=True)
            elif r.get("status") != "skipped":
                consecutive_fetch_error_fails = 0
            stats[r.get("status", "failed")] = \
                stats.get(r.get("status", "failed"), 0) + 1
            elapsed = time.time() - t0
            rate = completed / elapsed * 3600 if elapsed > 0 else 0
            extra = ""
            if r.get("status") == "ok":
                extra = (f" [{r['orbit']} gs={r['gs_source']} "
                         f"n={r['n_primary']}"
                         + (f"+2016:{r['n_2016']}" if r["has_2016"] else "")
                         + "]")
            elif reason:
                extra = f" [{reason}]"
            print(f"  [{completed}/{len(tiles)}] {r['name']}: {r['status']}"
                  f"{extra} | {rate:.0f}/h", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_run, t): t for t in tiles}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"  Error: {e}")

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed/60:.1f} min ===")
    print(f"  OK={stats['ok']}  Skipped={stats['skipped']}  "
          f"Failed={stats['failed']}")
    if abort.is_set():
        sys.exit(2)
    if stats["failed"] > stats["ok"] + stats["skipped"]:
        print("FAILED: more failed than ok+skipped — review before re-run")
        sys.exit(1)


if __name__ == "__main__":
    main()
