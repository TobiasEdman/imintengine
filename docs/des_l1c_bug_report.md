# DES openEO `s2_msi_l1c` returns all-nodata pixels for valid scenes

**Reported:** 2026-04-28
**Endpoint:** `https://openeo.digitalearth.se/` (openEO API 1.1.0)
**Affected collection:** `s2_msi_l1c`
**Reference scene:** `S2A_MSIL1C_20260408T104041_N0512_R008_T33VUE_20260408T155548` (cloud cover 0.077% per STAC)
**Reporter setup:** Python 3.9, `openeo` client, `imint.fetch._connect()` (basic auth via `DES_USER`/`DES_PASSWORD`)

## TL;DR

For the **identical** `load_collection` request shape — same bbox, same date, same band, same auth — the `s2_msi_l2a` collection returns valid pixels and `s2_msi_l1c` returns a constant `-9999.0` (uniform nodata) raster. Both jobs report `status: finished` with no server-side error in `py_proc_std_out.txt`. Reproducible across the smallest possible request (single date, single band, WGS84 bbox).

## Reproducer

```python
from openeo import connect
conn = connect("https://openeo.digitalearth.se/")
conn.authenticate_basic(username=DES_USER, password=DES_PASSWORD)

# Identical request shape — only the collection_id differs
COORDS_WGS = {"west": 11.55, "south": 58.10, "east": 11.75, "north": 58.20}
TEMPORAL = ["2026-04-08", "2026-04-09"]

def fetch_one(collection_id):
    cube = conn.load_collection(
        collection_id=collection_id,
        spatial_extent=COORDS_WGS,
        temporal_extent=TEMPORAL,
        bands=["b04"],
    )
    blob = cube.download(format="gtiff")
    # blob is gzipped tar of GeoTIFF(s); unwrap and parse with rasterio.
    return blob

l2a_blob = fetch_one("s2_msi_l2a")    # ← returns real data
l1c_blob = fetch_one("s2_msi_l1c")    # ← returns -9999 for every pixel
```

## Numerical evidence (calculation vectors)

After unwrapping the response (gzip → tar → first GeoTIFF member) and reading with `rasterio`:

| Collection         | shape (1, H, W) | dtype   | min     | max     | n_unique | sample of unique values        |
| :----------------- | :-------------- | :------ | :------ | :------ | :------- | :----------------------------- |
| **`s2_msi_l2a`**   | (1, 1172, 1233) | float32 | 996.0   | 7421.0  | **3317** | `[996, 997, 1001, 1003, 1005]` |
| **`s2_msi_l1c`**   | (1, 1172, 1233) | float32 | -9999.0 | -9999.0 | **1**    | `[-9999.0]`                    |

L2A has 3,317 distinct DN values across the bbox — a normal Sentinel-2 reflectance histogram. L1C has exactly **one** value — the openEO worker's nodata sentinel.

Worker `stac.json` for both jobs reports `data_sources: [{tile: "32VPK"}, {tile: "33VUE"}, {tile: "32VPK"}, {tile: "33VUE"}]` and the L1C output's geometry/transform is computed correctly:

```
{"bbox": [296690.0, 6444450.0, 309020.0, 6456170.0]}    # EPSG:3006-snapped extent
"image-to-model-transform": (10.0, 0.0, 296690.0, 0.0, -10.0, 6456170.0)
```

So scene discovery and projection both work — only the pixel-reading layer is broken.

`py_proc_std_out.txt` for the L1C job:
```
Executing process graph:
Saving result of type <class 'xarray.core.dataarray.DataArray'>.
Process graph executed
```

No error logged. Job status `"finished"`.

## What we ruled out as the cause

- **Auth.** Same DES_USER/DES_PASSWORD work for L2A.
- **Format.** Both responses are wrapped as gzip+tar; both unwrap with the same code path. The L1C tar contains valid TIF members with valid headers — they're just filled with -9999.
- **Bbox CRS.** Same `spatial_extent` dict (WGS84) succeeds for L2A and fails for L1C. Tested also with EPSG:3006 and EPSG:32633 (the L1C native CRS) for `spatial_extent` — all three return identical -9999 output for L1C.
- **Band naming.** `b04` (lowercase) succeeds for L2A. `b04` for L1C downloads but is empty.
- **Process-graph.** Minimal graph: `load_collection → save_result(format="gtiff")`. Same shape for both. `process.json` for the failing job:
  ```json
  {"process_graph": {
    "loadcollection1": {"process_id": "load_collection", "arguments": {
      "bands": ["b04"], "id": "s2_msi_l1c",
      "spatial_extent": {"west": 11.55, "south": 58.1, "east": 11.75, "north": 58.2},
      "temporal_extent": ["2026-04-08", "2026-04-09"]}},
    "saveresult1": {"process_id": "save_result", "result": true,
      "arguments": {"data": {"from_node": "loadcollection1"}, "format": "gtiff"}}
  }}
  ```
- **Properties filter.** Tried `properties={"tile_id": lambda v: v == "T33VUE"}` to pin a single MGRS tile — backend returns HTTP 500 with `AttributeError` (separate bug in DES's properties handling).

## Workaround used

`imint.fetch.fetch_l1c_safe_from_gcp()` — bypass DES openEO entirely and fetch the SAFE archive from Google's public mirror `gs://gcp-public-data-sentinel-2/`. STAC search still uses DES (which works for catalogue lookups), but the bulk transfer is anonymous HTTPS. The reference scene downloaded in 19 s (589 MB).

## Suggested investigation paths for DES

1. Compare load-collection worker logs for `s2_msi_l2a` vs `s2_msi_l1c` for the same bbox+date — the L2A path that succeeds and the L1C path that returns -9999 likely diverge at JP2 / pixel-reading time, not at scene discovery.
2. Check whether the `eodata-sentinel2-s2msi1c-2026/4/8/...SAFE/...` S3 backend is reachable from the openEO worker container with the credentials it uses.
3. Possible PSD-15 / processing-baseline-05.12 mismatch on the L1C read path — DES's L1C reader may not have been updated for the 2024+ SAFE structure (the L2A reader on the same data is fine).

## Reproducer artefacts

The full request/response triples (process.json, stac.json, py_proc_std_out.txt, the all-nodata GeoTIFF) for both the failing L1C job and the succeeding L2A job were captured during this investigation. They sit in `outputs/c2rcc_runs/` (and the workspace tarball saved by the worker's openEO download).
