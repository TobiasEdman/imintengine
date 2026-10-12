#!/usr/bin/env python3
"""Crop MPC Sentinel-1 GRD measurement COGs in radar geometry for AI2's local-file run.

AI2's predict pipeline (local-file mode) needs radar-geometry rasters with
GCPs: it reads ``raster.gcps`` from the VV file to define the window, and
rslearn's DefaultTileStore warps GCP rasters itself (TPS, cubic). The crop is
snapped to the ESA geolocation grid (GCP rows/cols) so the kept GCP subset
spans exactly the cropped data. DN values are copied bit for bit. Writes
``<out_dir>/{target,historical1}_{vv,vh}.tif``, ``target_vv_allgcps.tif`` and
``crop_meta.json`` (the record's ``inputs/``).

    python scripts/marine/s1_port/crop_grd.py <out_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _record as rec

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
)


def gcp_grid(gcps):
    cols = sorted({round(g.col) for g in gcps})
    rows = sorted({round(g.row) for g in gcps})
    assert len(cols) * len(rows) == len(gcps), (len(cols), len(rows), len(gcps))
    return cols, rows, {(round(g.col), round(g.row)): g for g in gcps}


def select_window(gcps, region, extra_ring=0):
    """GCP-cell-snapped window (col0..col1, row0..row1 indices) covering ``region``."""
    from shapely.geometry import Polygon

    cols, rows, grid = gcp_grid(gcps)
    hit_c, hit_r = [], []
    for i in range(len(cols) - 1):
        for j in range(len(rows) - 1):
            corners = [grid[(cols[i], rows[j])], grid[(cols[i + 1], rows[j])],
                       grid[(cols[i + 1], rows[j + 1])], grid[(cols[i], rows[j + 1])]]
            if Polygon([(g.x, g.y) for g in corners]).intersects(region):
                hit_c += [i, i + 1]
                hit_r += [j, j + 1]
    c0 = max(min(hit_c) - extra_ring, 0)
    c1 = min(max(hit_c) + extra_ring, len(cols) - 1)
    r0 = max(min(hit_r) - extra_ring, 0)
    r1 = min(max(hit_r) + extra_ring, len(rows) - 1)
    return cols, rows, grid, (c0, c1, r0, r1)


def footprint(grid, cols, rows, c0, c1, r0, r1):
    from shapely.geometry import Polygon

    ring = ([grid[(cols[i], rows[r0])] for i in range(c0, c1 + 1)]
            + [grid[(cols[c1], rows[j])] for j in range(r0, r1 + 1)]
            + [grid[(cols[i], rows[r1])] for i in range(c1, c0 - 1, -1)]
            + [grid[(cols[c0], rows[j])] for j in range(r1, r0 - 1, -1)])
    return Polygon([(g.x, g.y) for g in ring])


def crop_item(item, region, out_dir: Path, tag: str, extra_ring=0, write_full_gcp_variant=False):
    import rasterio
    from rasterio.control import GroundControlPoint
    from rasterio.windows import Window
    from shapely.geometry import mapping

    meta = {"item": item.id, "tag": tag}
    for band in rec.POLS:
        with rasterio.Env(**GDAL_ENV), rasterio.open(item.assets[band].href) as src:
            gcps, gcp_crs = src.gcps
            cols, rows, grid, (c0, c1, r0, r1) = select_window(gcps, region, extra_ring)
            x0, x1, y0, y1 = cols[c0], cols[c1], rows[r0], rows[r1]
            win = Window(x0, y0, x1 - x0 + 1, y1 - y0 + 1)
            arr = src.read(1, window=win)
            sub = [GroundControlPoint(row=g.row - y0, col=g.col - x0, x=g.x, y=g.y, z=g.z,
                                      id=g.id, info=g.info)
                   for (c, r), g in grid.items() if x0 <= c <= x1 and y0 <= r <= y1]
            profile = dict(driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
                           dtype=arr.dtype, nodata=src.nodata, compress="zstd", tiled=True,
                           blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
            out = out_dir / f"{tag}_{band}.tif"
            with rasterio.open(out, "w", **profile) as dst:
                dst.write(arr, 1)
                dst.gcps = (sub, gcp_crs)
                dst.update_tags(**src.tags(), CROP_SOURCE_ITEM=item.id,
                                CROP_WINDOW_COLOFF=str(x0), CROP_WINDOW_ROWOFF=str(y0))
            if write_full_gcp_variant and band == "vv":
                full = [GroundControlPoint(row=g.row - y0, col=g.col - x0, x=g.x, y=g.y, z=g.z,
                                           id=g.id, info=g.info) for g in gcps]
                with rasterio.open(out_dir / f"{tag}_{band}_allgcps.tif", "w", **profile) as dst:
                    dst.write(arr, 1)
                    dst.gcps = (full, gcp_crs)
            meta[band] = dict(
                file=out.name, src_shape=[src.height, src.width],
                window=dict(col_off=x0, row_off=y0, width=int(win.width), height=int(win.height)),
                n_gcps_kept=len(sub), n_gcps_total=len(gcps), gcp_grid=[len(cols), len(rows)],
                gcp_crs=str(gcp_crs), nodata=src.nodata, dtype=str(arr.dtype),
                dn_min=int(arr.min()), dn_max=int(arr.max()), dn_mean=float(arr.mean()),
                zero_frac=float((arr == 0).mean()),
                footprint_wgs84=mapping(footprint(grid, cols, rows, c0, c1, r0, r1)),
                first_gcp_lonlat=[sub[0].x, sub[0].y])
            print(tag, band, meta[band]["window"], arr.shape, "gcps", len(sub))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--bbox", nargs=4, type=float, default=rec.SKAGEN_BBOX,
                    metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    ap.add_argument("--target-id", default=rec.TARGET_ID)
    ap.add_argument("--historical-id", default=rec.HISTORICAL_ID)
    a = ap.parse_args()

    import planetary_computer as pc
    from pystac_client import Client
    from shapely.geometry import Polygon, box

    a.out_dir.mkdir(parents=True, exist_ok=True)
    col = Client.open(rec.MPC_STAC, modifier=pc.sign_inplace).get_collection("sentinel-1-grd")
    m_t = crop_item(col.get_item(a.target_id), box(*a.bbox), a.out_dir, "target",
                    write_full_gcp_variant=True)
    # The historical crop must cover the whole target window: the target crop's
    # footprint plus one extra GCP ring of margin.
    tgt_fp = Polygon(m_t["vv"]["footprint_wgs84"]["coordinates"][0])
    m_h = crop_item(col.get_item(a.historical_id), tgt_fp, a.out_dir, "historical1", extra_ring=1)
    assert Polygon(m_h["vv"]["footprint_wgs84"]["coordinates"][0]).contains(tgt_fp)
    (a.out_dir / "crop_meta.json").write_text(json.dumps(
        {"bbox": list(a.bbox), "target": m_t, "historical1": m_h}, indent=1))


if __name__ == "__main__":
    main()
