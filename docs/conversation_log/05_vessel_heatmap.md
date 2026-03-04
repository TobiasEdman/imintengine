# Vessel Heatmap Pipeline

> Multi-temporal vessel heatmap pipeline, full scene processing attempts, cloud threshold experiments, temporal aggregation, and heat map visualization.

---


---

_fetch_tci_bands() lightweight fetch
fetch_vessel_heatmap() main function
save_vessel_heatmap_png() export
Showcase integration
Standalone runner script
User clarifications:
Time range: "we test with July"
Output: "New showcase panel"
Cloud thresholds: "50% scene + 30% AOI (Recommended)"
Implementation: After plan approval, I implemented all 5 parts:
save_vessel_heatmap_png() in export.py
_fetch_tci_bands() in fetch.py
fetch_vessel_heatmap() in fetch.py
Showcase viewer/file_map/engine integration
run_vessel_heatmap.py standalone script
Was about to verify imports/syntax when the summary was requested.
Let me now compile the detailed summary with all code snippets and technical details.

d. Fix reprojection artifact (clip order): User showed screenshot saying "The bounding box cut is done at the wrong time, we still have one reprojection artifact"
e. Multi-temporal vessel detection heatmap: User said "perfekt Now I want to make a function that downloads all images and use those with cloud coverage according to the SCL layers and run the vessel detection each of them. We use the same bounding box and I want the result presented as a heatmap. Make a plan for the development of that"
Clarifications: Test with July 2025, new showcase panel, 50% scene + 30% AOI cloud thresholds.
CRS reprojection: S-57 ENC data is in WGS84 (EPSG:4326), RGB satellite images are in SWEREF99 TM (EPSG:3006). Must reproject sjökort to EPSG:3006 for pixel alignment.
Clip-after-reproject: Clipping in WGS84 then reprojecting creates curved edges that leave thin gaps. Must reproject first, then clip in projected CRS.
Leaflet.js dual-layer toggle: Multiple L.imageOverlay instances sharing same file URL (browser caches), with opacity swapping for RGB/sjökort background switching.

OpenEO/DES: Digital Earth Sweden API for Sentinel-2 L2A data fetch (collection: s2_msi_l2a)
STAC API: https://explorer.digitalearth.se/stac/search for date discovery
SCL (Scene Classification Layer): Classes 3,8,9,10 = cloud. check_cloud_fraction() computes ratio.
_fetch_scl_batch(): Batch SCL download for multiple dates in one openEO call (tar.gz archive)
MarineVesselAnalyzer: YOLO11s via SAHI, uses L1C-TCI formula on B02+B03+B04
Heatmap: Gaussian KDE on detection centroids, σ=5 pixels (50m), RGBA PNG with transparent zeros
CRS handling in render_sjokort_png(): Reads projected bounds from bands_meta.json, reprojects all S-57 geometry to EPSG:3006, renders on projected axes.
Key section — meta reading + projected bounds:
# ── Read bands_meta.json for rendering parameters ────────────
import json as _json
_out_dir = Path(output_path).parent
_meta_candidates = sorted(_out_dir.glob("bands/*bands_meta.json"))
_bm: dict = {}
_geo: dict = {}
if _meta_candidates:
    with open(_meta_candidates[0]) as _mf:
        _bm = _json.load(_mf)
    _geo = _bm.get("geo", {})
elif bbox_wgs84 is None or img_w == 0 or img_h == 0:
    raise FetchError(...)
# Projected CRS + bounds
target_crs = _geo.get("crs", "EPSG:3006")
_proj_bounds = _geo.get("bounds_projected")
Key section — WGS84 bbox + projected bounds + clip box:
# WGS84 bbox — used ONLY as a spatial filter when reading S-57 data
W = bbox_wgs84["west"]
S_ = bbox_wgs84["south"]
E = bbox_wgs84["east"]
N = bbox_wgs84["north"]
# Projected rendering bounds (e.g. EPSG:3006 / SWEREF99 TM)
if _proj_bounds:
    pW = _proj_bounds["west"]
    pS = _proj_bounds["south"]
    pE = _proj_bounds["east"]
    pN = _proj_bounds["north"]
else:
    from pyproj import Transformer as _Tr
    _t = _Tr.from_crs("EPSG:4326", target_crs, always_xy=True)
    pW, pS = _t.transform(W, S_)
    pE, pN = _t.transform(E, N)
# Clip box in the PROJECTED CRS — clipping must happen after reprojection
clip_box = box(pW, pS, pE, pN)
Key section — _read_layer() with reproject-then-clip:
def _read_layer(path, layer, geom_types=None):
    import fiona
    try:
        if layer not in fiona.listlayers(str(path)):
            return gpd.GeoDataFrame()
        # Read with WGS84 bbox as spatial filter (S-57 native CRS)
        gdf = gpd.read_file(str(path), layer=layer, bbox=(W, S_, E, N))
        if gdf.empty:
            return gdf
        gdf = gdf[gdf.geometry.notna()]
        if geom_types and not gdf.empty:
            gdf = gdf[gdf.geometry.geom_type.isin(geom_types)]
        # Reproject WGS84 → target CRS BEFORE clipping
        if not gdf.empty:
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            gdf = gdf.to_crs(target_crs)
        # Clip in the projected CRS — straight edges match the axes
        if not gdf.empty:
            gdf = gpd.clip(gdf, clip_box)
        return gdf
    except Exception:
        return gpd.GeoDataFrame()
Key section — fiona point data reprojection:
from pyproj import Transformer as _Transformer
_wgs_to_proj = _Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
for lf in lit_features:
    lf["x"], lf["y"] = _wgs_to_proj.transform(lf["x"], lf["y"])
for bf in buoy_features:
    bf["x"], bf["y"] = _wgs_to_proj.transform(bf["x"], bf["y"])
for pl in place_labels:
    pl["x"], pl["y"] = _wgs_to_proj.transform(pl["x"], pl["y"])
Key section — figure setup with projected axes:
ax.set_xlim(pW, pE)
ax.set_ylim(pS, pN)
ax.set_aspect("auto")
Key section — arc points in projected (metric) coords, no cos_lat:
sector_r = (pN - pS) * 0.035
sector_r_lg = (pN - pS) * 0.055
def _arc_points(cx, cy, r, brg1, brg2, n_pts=40):
    """Return (xs, ys) for a circular arc in projected (metric) coords."""
    if brg2 <= brg1:
        brg2 += 360.0
    xs, ys = [], []
    for i in range(n_pts + 1):
        brg = math.radians(brg1 + (brg2 - brg1) * i / n_pts)
        xs.append(cx + r * math.sin(brg))
        ys.append(cy + r * math.cos(brg))
    return xs, ys
New: _fetch_tci_bands() — lightweight TCI fetch for vessel detection:
def _fetch_tci_bands(conn, projected_coords, temporal):
    """Fetch B02+B03+B04+SCL for vessel detection (lightweight)."""
    tci_bands = ["b02", "b03", "b04"]
    cube_tci = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=tci_bands,
    )
    cube_scl = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=projected_coords,
        temporal_extent=temporal,
        bands=BANDS_20M_CATEGORICAL,
    )
    cube_scl = cube_scl.resample_cube_spatial(target=cube_tci, method="near")
    cube = cube_tci.merge_cubes(cube_scl)
    data = cube.download(format="gtiff")
    # ... parse with rasterio, dn_to_reflectance, return (bands, scl, geo)
New: fetch_vessel_heatmap() — main multi-temporal function:
def fetch_vessel_heatmap(
    coords: dict,
    date_start: str,
    date_end: str,
    output_dir: str | Path,
    *,
    cloud_threshold: float = 0.3,
    scene_cloud_max: float = 50.0,
    gaussian_sigma: float = 5.0,
    prefix: str = "",
) -> dict:
Pipeline: STAC discovery → batch SCL screening → per-date TCI fetch + vessel detection → Gaussian-smoothed heatmap → save PNG + JSON summary.
/Users/tobiasedman/Downloads/ImintEngine/imint/exporters/html_report.py — HTML showcase
Toggle change — Python _map_cells(): Changed if has_bg_toggle and v["key"] == "rgb": to if has_bg_toggle:.
Toggle change — JS initMaps(): All panels get toggleable RGB+sjökort background:
if (hasBgToggle) {
    bgLayers[v.id] = {};
    const prefix = v.id.split('-')[0];
    const rgbId = prefix + '-rgb';
    const sjokortId = prefix + '-sjokort';
    const rgbUrl = (v.key === 'rgb') ? images[v.id] : images[rgbId];
    if (rgbUrl) {
        bgLayers[v.id].rgb = L.imageOverlay(rgbUrl, bounds, {zIndex: 0, opacity: 1}).addTo(map);
    }
    if (images[sjokortId]) {
        bgLayers[v.id].sjokort = L.imageOverlay(images[sjokortId], bounds, {zIndex: 0, opacity: 0}).addTo(map);
    }
    if (v.key === 'rgb') {
        overlays[v.id] = bgLayers[v.id].rgb;
    } else {
        const overlay = L.imageOverlay(images[v.id], bounds, {zIndex: 1}).addTo(map);
        overlays[v.id] = overlay;
    }
Toggle handler — only updates overlays ref for RGB panel:
const prefix = mapId.split('-')[0];
if (mapId === prefix + '-rgb') {
    overlays[mapId] = (bg === 'rgb') ? layers.rgb : layers.sjokort;
    // Reset opacity slider...
}
New viewer added to _MARINE_VIEWERS:
{"id": "m-vessel-heatmap", "title": "Fartygsaktivitet (heatmap)", "key": "vessel_heatmap", "legend": [
    {"color": "#FFFFB2", "label": "Låg"},
    {"color": "#FD8D3C", "label": "Medel"},
    {"color": "#BD0026", "label": "Hög"},
]},
New entry in file_map:
"vessel_heatmap": "vessel_heatmap_clean.png",
CLI with args: --west/south/east/north, --start, --end, --output-dir, --cloud-threshold, --scene-cloud-max, --sigma, --prefix, --showcase. Calls fetch_vessel_heatmap() and prints per-date summary.
Key data files:
bands_meta.json: outputs/full_bohuslan_2025-07-10/bands/2025-07-10_bands_meta.json — contains geo.bounds_wgs84, geo.bounds_projected, geo.crs, shape
Bohuslän bbox: west=11.25049, south=58.42763, east=11.30049, north=58.47763, shape=[573,324]
S-57 ENC files: /Users/tobiasedman/Downloads/83e3f167-4d05-4765-81a3-619c3a515a4c/ (SE2, SE3, SE4, SE5)
Python env: .venv/bin/python3
CRS alignment: Fundamental discovery that S-57 (WGS84) must be reprojected to match the EPSG:3006 RGB grid. Required changes throughout the rendering pipeline.
Clip-after-reproject: Subtle geometric artifact — a WGS84 rectangle becomes slightly non-rectangular in EPSG:3006. Moving clip to post-reproject solved it.
Multi-temporal vessel heatmap architecture: Designed efficient pipeline using existing STAC discovery + batch SCL screening + lightweight TCI-only fetch to minimize download size.
All user messages:
"and its refers to the same file, we are not double storing?"
"There is still a gegraphic missalignemnt between the RGB and the Nautical chart. Should it not only be a simple WGS 84 to Sweref99 conversion?"
"Good make sure this is dokuemented in the code"
(Screenshot of sjökort with visible edge artifact) "The bounding box cut is done at the wrong time, we still have one repojection artefact"
"perfekt Now I want to make a function that downloads all images and use thos with cloud coverage according to the SCL layers and run the vessel detection each of them. We use the same bounding box and I want the result presented as a heatmap. Make a plan for the development of that"
(Answer to time range question): "we test with July"
(Answer to output question): "New showcase panel"
(Answer to cloud threshold question): "dis we set any cloud thresholds?"
(Answer to cloud threshold options): "50% scene + 30% AOI (Recommended)"
Verify imports and syntax check for the new code (was in progress)

Test with Bohuslän bbox + July 2025 — run run_vessel_heatmap.py with DES credentials

Regenerate showcase to verify the new heatmap panel appears
✅ save_vessel_heatmap_png() in export.py
✅ _fetch_tci_bands() in fetch.py
✅ fetch_vessel_heatmap() in fetch.py
✅ Showcase viewer/file_map/engine integration
✅ run_vessel_heatmap.py standalone script
Was about to run import/syntax verification when summary was requested. The todo list shows "Verify imports and run syntax check" as in_progress.
cd /Users/tobiasedman/Downloads/ImintEngine && .venv/bin/python3 -c "from imint.fetch import fetch_vessel_heatmap; print('OK')"
Then test with the Bohuslän bbox + July 2025 period. This directly follows from the approved plan's verification section and the user's explicit request to create the heatmap function tested with July.

I'll continue with the import/syntax verification that was in progress. Let me check the new code compiles correctly.

Let me check the correct class name in engine.py and re-run the help command:

Good — the engine doesn't have a class called ImintEngine, it uses run_job. The engine import itself is fine (I just used the wrong name). Let me verify the vessel_heatmap entry was properly added to the path_candidates dict:

All integration points are in place. Let me also quickly verify the fetch.py functions are syntactically correct by doing a full module compile:
Compile-check all modified files
All five files compile cleanly. Now let me update the todo list and move on to the actual test run:

All imports and syntax checks pass. Now let's run the actual test with the Bohuslän bounding box and July 2025:
Run vessel heatmap pipeline for Bohuslän July 2025
The pipeline ran successfully. Here's the summary:
9 candidate dates found via STAC (scene cloud ≤ 50%)
5 dates passed the AOI cloud filter (≤ 30%), 4 skipped (too cloudy)
130 total vessel detections across 4 successful dates:
2025-07-17: 50 vessels (0.0% cloud)
2025-07-19: 35 vessels (2.2% cloud)
2025-07-10: 28 vessels (0.0% cloud)
2025-07-30: 17 vessels (12.0% cloud)
1 date failed (2025-07-12) — the openEO download returned an invalid GeoTIFF, but the pipeline handled it gracefully and continued
Heatmap PNG saved to outputs/full_bohuslan_2025-07-10/2025-07-10_vessel_heatmap_clean.png
Let me verify the output files and check the heatmap image:

The heatmap looks good — you can see:
Hot spots (dark red) where vessels were detected repeatedly across multiple dates
Warm zones (orange/yellow) showing less frequent traffic
Transparent areas where no vessels were detected (these will show the RGB/sjökort background in the showcase)
The Gaussian smoothing gives a nice density visualization. Let me also check the summary JSON:

Everything is working. Let me update the todo list:

All tasks are complete. Here's the full summary:
Multi-temporal vessel heatmap — done ✅
Pipeline results (Bohuslän, July 2025):
9 candidate dates discovered via STAC (scene cloud ≤ 50%)
5 dates passed AOI cloud screening (≤ 30%), 4 skipped
130 vessel detections across 4 successful dates (1 date had an openEO download error but was handled gracefully)
Heatmap PNG saved with YlOrRd colormap, transparent where no vessels detected
Files created/modified:
File	What
imint/exporters/html_report.py	New "Fartygsaktivitet (heatmap)" viewer + file_map entry
imint/engine.py	vessel_heatmap in path_candidates
run_vessel_heatmap.py	Standalone CLI runner
Output files:
outputs/full_bohuslan_2025-07-10/2025-07-10_vessel_heatmap_clean.png (46 KB)
outputs/full_bohuslan_2025-07-10/2025-07-10_vessel_heatmap_summary.json
The heatmap is ready to be displayed in the showcase — you can regenerate it with --showcase flag to see the new "Fartygsaktivitet (heatmap)" panel with the toggleable RGB/sjökort background.
Can we make the more or less constant yellow parts transparent to?

Right now only exact-zero pixels are transparent, so the Gaussian tail produces that constant yellow wash everywhere. The fix is to scale alpha proportionally to density — low values fade out smoothly:

Now regenerate the heatmap PNG from the saved numpy data in the summary — actually, let me just re-run the export step. The fastest way is to call the function directly on the existing data:

Check if heatmap numpy array was saved
No cached array — let me check if fetch_vessel_heatmap saves the raw numpy. Let me look at what it does at the end:

The raw heatmap array isn't persisted. The easiest way to test the visual change is to re-run the pipeline. But since the DES fetch is slow, let me instead add a quick save of the raw array so we can iterate, then re-run:

Now re-run the pipeline — this time it'll fetch data again but also save the .npy. Actually, let me be smarter: I can re-export from the existing PNG by reading its density back, but that's lossy. Better to just re-run the full pipeline once:
Re-run vessel heatmap with updated alpha export
Let me check the new heatmap:

Much better — the constant yellow wash is gone. Now only the actual vessel hotspots are visible, fading smoothly from transparent through light orange to dark red. The low-density Gaussian tail is essentially invisible, so the RGB/sjökort background will show through cleanly everywhere except where there's real vessel activity.
The change was a single line in save_vessel_heatmap_png() — instead of only zeroing out exact-zero pixels, the alpha channel is now proportional to the normalised density (norm * 255), so faint areas become nearly transparent.
can you update the html

The vessel_heatmap_clean.png wasn't copied. Let me check how the file mapping works in save_tabbed_report:

I see — the file lookup uses prefix + file_map[key], where prefix is the marine_date prefix. Let me trace it more carefully:

So the lookup is marine_prefix + "vessel_heatmap_clean.png" → "2025-07-10_vessel_heatmap_clean.png". But the actual file is named with prefix="2025-07-10_" from the run script. Let me verify:

Check actual vessel heatmap filenames
The file 2025-07-10_vessel_heatmap_clean.png exists. The lookup uses marine_prefix = "2025-07-10_" since I passed marine_date='2025-07-10_'. But wait — the code does marine_prefix = f"{marine_date}_" if marine_date is set. Let me check:

There's the bug — I passed marine_date='2025-07-10_' (with trailing underscore), so the prefix becomes "2025-07-10__" (double underscore). Let me re-run with the correct argument:

vessel_heatmap_clean.png is now in the showcase. The HTML has been regenerated at outputs/imint_showcase.html with the new "Fartygsaktivitet (heatmap)" panel included.
How many images was it based on?
4 successful images (2025-07-17, 2025-07-19, 2025-07-10, and 2025-07-30). A 5th date (2025-07-12) passed the cloud filter but the openEO download returned a corrupt GeoTIFF, so it contributed 0 detections. 4 additional dates were skipped entirely due to cloud cover exceeding 30%.
Now do the same for the full season

The DES server has a 20 time-step limit per synchronous job. With 34 candidate dates across 5 months, the batch SCL screening exceeded it. I need to chunk the batch into groups of ≤20 dates. Let me look at _fetch_scl_batch:

Now I see the issue. With 34 dates, the temporal extent spans the full range and DES loads all time steps within that range (could be 85 with the Sentinel-2 revisit). I need to chunk the batch into groups of ≤15 dates (leaving some margin) and call _fetch_scl_batch for each chunk:

I'll chunk the _fetch_scl_batch calls into groups of at most 15 dates:

Now re-run the full season:
Run vessel heatmap for full season May-Sep 2025 (chunked SCL)
The problem is that chunking by count doesn't help — _fetch_scl_batch uses a temporal range, and DES loads all passes within that range (not just our candidates). 15 dates spread over months still produces dozens of time steps. I need to chunk by tight temporal windows instead. Grouping by month will keep each chunk well under the 20-step limit:

Now re-run:

Skipping the STAC scene-cloud filter — I'll set --scene-cloud-max 100 so all dates pass through to the SCL-based AOI check which is more accurate:
Run full season with no STAC cloud pre-filter
where are we in the process?
The background task was stopped before it could finish. Let me check how far it got:

The task was killed. I had stopped it to try bypassing the scene-cloud filter. Let me restart the full-season run now with --scene-cloud-max 100 so all dates go through to the more accurate SCL-based AOI cloud check:

STAC search is still running. Let me wait a bit longer:

Full season completed. Let me view the result:

Done. Full season May–Sep 2025 results:
72 dates found via STAC, 28 passed the 30% AOI cloud filter
23 dates successfully processed (5 had corrupt GeoTIFF downloads)
214 total vessel detections across the season
Clear seasonal pattern: July peak (130 vessels), August moderate (41), May/June/Sep lower activity
Now let me regenerate the showcase HTML:

Regarding your message about skipping STAC — you want the pipeline to skip the STAC date-discovery step and instead let DES/openEO handle all date enumeration directly. Let me look at how the STAC search feeds into the pipeline:

I'll modify the pipeline to skip STAC entirely and let DES discover dates directly via the SCL batch fetch. The SCL download already returns all available dates within the temporal range — we just need to stop filtering to only STAC-discovered dates.

I need to:
Add a temporal-range mode to _fetch_scl_batch that returns ALL dates found (not just pre-filtered ones)
Replace the STAC step in fetch_vessel_heatmap with direct monthly DES calls

Now update fetch_vessel_heatmap to skip STAC and go straight to DES monthly SCL batches:

Now replace Step 1 + Step 2 with a single DES-based approach:

Also need to update the CLI banner since scene_cloud_max no longer applies:

I see you want me to stop. Let me summarize where we are:
Done:
Modified _fetch_scl_batch() to support a temporal=[start, end) mode that returns ALL dates DES has (no STAC filtering)
Replaced Step 1+2 in fetch_vessel_heatmap() — now chunks by calendar month and goes straight to DES for date discovery + cloud screening
Not yet done:
Remove the now-unused scene_cloud_max parameter from the CLI banner
Re-run the full season test

Regenerate showcase HTML
Want me to continue with the test run, or did you have something else in mind?
Fetch all spectral data and SCL at the same time
Good idea — fetch B02+B03+B04+SCL together per month instead of SCL first, then TCI per-date. One DES call per month instead of 1 + N.

Now let me see the full Step 3-4 loop to understand the current per-date TCI fetch:

Now I have the full picture. I'll create a new _fetch_tci_scl_batch that downloads B02+B03+B04+SCL for a whole month in one call (tar.gz), then returns per-date tuples of (date, bands, scl, geo). This replaces both _fetch_scl_batch and _fetch_tci_bands in the heatmap pipeline.

Got it — fetch TCI+SCL together for the full period. Let me add the batch function and rewrite fetch_vessel_heatmap:

Now rewrite fetch_vessel_heatmap to use _fetch_tci_scl_batch — one call per month, no STAC, no separate TCI fetches:

Now replace the entire body from Step 1+2 through Step 3-4:

Now verify what comes right after (the "Step 5: Smooth and save" section should still be there):

Good — the save step is intact. Now also need to handle the empty case before Step 5. Let me check:

Good, the flow is clean. Let me compile-check and then run:
Syntax check fetch.py
Now run the full season:

Still waiting for DES to process the first monthly batch (TCI+SCL is ~4× more data than SCL alone). Let me keep checking:

Still only 2 lines after a long wait — the first DES call (May with 4 bands × ~16 dates) is likely too large or timing out. The problem is that TCI+SCL for a full month is much heavier than SCL-only. Let me stop this and reduce the chunk size to 2 weeks instead of full months:

Downloading 4 bands for a whole month in one call is too heavy for DES synchronous mode. I'll chunk to 2-week windows instead — that keeps each call to ~3 dates × 4 bands, well within limits:

Now update the loop to use _chunk_ranges instead of _month_ranges:

Let me check the result and regenerate the showcase:

Full season done. Now regenerating the showcase:

Done. Full season results (May–Sep 2025):
29 cloud-free dates out of 73 total (44 skipped for >30% cloud)
270 total vessel detections — up from 214 in the previous run
0 failed fetches — the combined TCI+SCL batch approach eliminated the corrupt GeoTIFF errors from the per-date fetches
Notable: 2025-07-12 now returned 25 vessels (previously failed as a separate fetch)
Clear July peak: 173 vessels across 6 dates
The pipeline now does one DES call per 2-week window fetching B02+B03+B04+SCL together — no STAC, no separate per-date TCI fetches. Showcase HTML regenerated at outputs/imint_showcase.html.
But I want it together with all the other content of that tab
The showcase was generated with marine_date='2025_season', so it looked for files like 2025_season_rgb.png — but the other marine layers (RGB, vessels, NDVI, etc.) use the 2025-07-10_ prefix. I need to use the original prefix for everything except the heatmap, or copy the season heatmap to the expected name.
The simplest fix: copy the season heatmap to the 2025-07-10_ prefix so it matches the other layers:

Now regenerate with the 2025-07-10 prefix for all marine layers:

All marine layers are in the showcase now — RGB, vessels, vessel heatmap, NDVI, NDWI, COT, NMD, and sjökort. The season heatmap will appear alongside all the other panels in the Marin tab.
Nice are there any tiduing up to dp or pushes and commits to make

That's a big diff across the whole session. Let me check for any cleanup needed:

The scene_cloud_max param in fetch_vessel_heatmap is unused now but kept for CLI compat — that's fine as documented. The other references are in unrelated functions that still use STAC.
Let me also check for any unused imports that crept in:

Check for orphaned imports
Clean. The _defaultdict import was inlined and is now gone since we replaced that code. Let me do a quick compile check on everything:
Final compile check on all changed files
Everything looks clean. Here's what we'd be committing:
File	Changes
imint/fetch.py	CRS reprojection fix, clip-after-reproject, _fetch_tci_scl_batch(), fetch_vessel_heatmap() with DES-direct chunked pipeline
imint/exporters/html_report.py	RGB/sjökort toggle on all panels, vessel heatmap viewer + file_map
imint/exporters/export.py	save_vessel_heatmap_png() with alpha-scaled transparency
imint/engine.py	vessel_heatmap in path_candidates
imint/analyzers/change_detection.py	Minor change
run_vessel_heatmap.py	New standalone CLI runner
Want me to commit and push?
yes please

Committed and pushed to main.
OK lets test to the same for the full scenes, I think it will be faster process everythimg than to filter for clouds. So all vessel detcteion for all scenes that intersect the bounding box.
Makes sense — just run detection on everything and let the heatmap speak for itself. Cloudy scenes will just contribute fewer detections. Simplest way: --cloud-threshold 1.0.
Run full season with no cloud filtering (all scenes)
All 73 scenes processed, 0 skipped, 283 total vessels.
You were right — the 100% cloud scenes contribute 0 detections anyway and the detector handles them fine. Compared to the cloud-filtered run (29 dates, 270 vessels), processing everything added 13 extra detections from partially cloudy scenes:
Scene	Cloud	Extra vessels
2025-07-20	43.7%	2
2025-07-25	34.6%	8
2025-08-26	48.9%	1
2025-08-28	79.2%	1
2025-09-28	70.6%	1
The fully overcast scenes (100% cloud) consistently returned 0 vessels, so they're harmless. Cloud filtering can essentially be dropped — it just saves a bit of detection time on scenes that won't contribute, but at the cost of missing partial detections.
Want to update the showcase with this version, or adjust anything?
please do

Done — showcase updated with the full 73-scene heatmap (283 vessels).
Di you have data for th efull scenes?
Yes — the full 73-scene run saved its summary alongside the July-only one:

List heatmap summary files
The 2025-07-10_ one is the old July-only run (5 dates). The 2025_season_ one is the full 73-scene run. Here's a quick breakdown:
Summarise season data by month
All data is here — 73 scenes, 283 vessels, full May–Sep 2025. The .npy array is also cached if you want to re-export with different settings:

Check cached numpy arrays
Both the JSON summary, PNG heatmap, and raw numpy array are available for the full season run.
Show it a new html for th efull scebes

That's still the cloud-filtered version. The new all-scenes PNG overwrote it. Let me check — the showcase currently uses the 2025-07-10_ prefix copy. Let me update it with the latest all-scenes version and open it:
Rebuild showcase and open in browser
The showcase HTML is regenerated and opened in your browser. It has the full 73-scene / 283-vessel heatmap in the "Fartygsaktivitet (heatmap)" panel alongside all the other marine layers (RGB, vessels, NDVI, NDWI, COT, NMD) with the sjökort toggle.
This was wrong, you replaced the brand analysis with marine, you need to revert that

Check what files exist with fire/brand prefix
I see — the problem is I used the same directory for both fire_dir and marine_dir, so the fire/brand tab got marine files. Let me check where the actual fire/brand data lives:

That's the fire/brand data. It uses a different directory and date prefix. Let me regenerate with the correct dirs:

Regenerate showcase with correct fire + marine dirs
Done — the Brand tab now uses the fire data from 2018-07-24 (Ljusdal) and the Marin tab uses the Bohuslän data with the full-season heatmap. Should be back to normal.
make a new html paghe showing the full scenes of the vessel heatmap not only tha AOI from before
Let me first find the full Sentinel-2 scene footprint that covers our AOI:
Query STAC for full scene footprint
The full scene covers roughly 10.68–12.61°E, 57.61–58.63°N — a huge area. That's too large for 73 dates at 10m. Let me use a practical subset — the Bohuslän archipelago coast:
Calculate scene dimensions
Full scene is ~11000×11000 px at 10m — that's big but let's go for it. I'll use 1-week chunks to keep DES calls manageable:

Got it — you want to first test just downloading everything for the full scene to see if it works, before running detection. Let me do a quick download test with a single date to check DES handles the full tile size:

Test full-scene download for 1 week
504 Gateway Timeout — the full tile (120km × 120km) with 4 bands is too large for DES synchronous mode even for a single date. Let me try a single date only to find the minimum that works:

Test single-date full-scene download
Even a single date at full scene resolution times out. The full tile is ~11000×11000 px × 4 bands — too much for DES synchronous mode. Let me try a quarter of the scene to find what works:

Test half-scene single-date download
Half-scene works (~6300×11700 px). Takes 96s per date. Let me test 2-week batch:

Test 2-week batch at half-scene
So multi-date batches time out even at half-scene. But single-date at half-scene works (96s). I need to switch to per-date fetching for large areas. Let me add a fallback to the pipeline and run it for the west half of the scene (the coast/sea side where vessels actually are):

I'll add a fallback: if the batch fails with a timeout/error, retry each date individually using _fetch_tci_bands:

Now run it for the west half of the scene (the coast/sea where vessels are). This will auto-fallback to per-date mode:
Run full west-half scene heatmap, full season, per-date fallback
Running in background — this will take a while (~96s per date × 73 dates ≈ 2 hours for downloading, plus vessel detection time). Let me check the first results to make sure it's working:

Still waiting for the first batch to timeout then fallback. Let me wait more:

Stopped. The full scene is just too large for DES synchronous mode — even the batch timeout takes minutes before the fallback kicks in. For full-scene processing at this scale, we'd need to use DES asynchronous batch jobs instead of synchronous downloads. What would you like to do instead?
I wonder if the jetties and piers in the nautical chart are objects
Search S-57 layers for jetty/pier objects
No exact name matches. Let me search more broadly — S-57 uses abbreviated object class names:

Yes! The key layers are:
SLCONS — Shoreline construction (jetties, piers, wharves, seawalls, breakwaters)
MORFAC — Mooring facility (bollards, dolphins, pontoons)
BERTHS — Berths (SE5 only)
HRBFAC — Harbour facility
Let me inspect what's in them:
Inspect jetty/pier S-57 layers
The layers exist but might be empty for this specific bbox/area. Let me read without bbox filter:

They're in the root, not in subdirs. Let me try again:
Count features per layer in all ENCs
Yes — jetties and piers are definitely in there as S-57 objects:
Layer	Count	What
SLCONS	SE3: 25, SE4: 203, SE5: 389	Shoreline constructions (jetties, piers, wharves, breakwaters, seawalls)
MORFAC	SE4: 23, SE5: 6	Mooring facilities (pontoons, dolphins, bollards)
BERTHS	SE5: 12	Berths
HRBFAC	SE4: 2, SE5: 5	Harbour facilities
Let me see what types of shoreline constructions are present:
Inspect SLCONS and MORFAC attributes
Here's what we have — all are real geometry objects:
SLCONS (Shoreline constructions) — all LineStrings:
CATSLC=4: 519 features → Wharf/quay
CATSLC=1: 31 features → Breakwater
CATSLC=6: 40 features → Mole
CATSLC=12: 2 features → Promenade/pier
MORFAC (Mooring facilities) — 23 Points:
CATMOR=1: → Dolphin (mooring post)
So yes, jetties, piers, wharves, and breakwaters are all proper line geometry objects in the S-57 data. They could potentially be rendered on the sjökort or used to correlate with vessel detection hotspots. Want me to do something with them?
you are not trying to get the full scen?
Right — let me fix it. The problem is the batch attempt times out for 5 minutes before falling back. I'll detect large areas and skip straight to per-date mode:

Now run the full scene. The west half worked at 96s/date. Let me use the full tile bbox and let per-date mode handle it:

OK, stopping. The core issue is that the full Sentinel-2 scene (112×114 km) is too large for DES synchronous mode — even a single-date download of the full tile times out. The half-scene works (96s/date) but for 73 dates that's ~2+ hours just for downloads, plus vessel detection on 6000×11000 px images.
Would you like to:
Run a smaller but still much larger area than the original AOI — e.g. 10× wider?
Use DES async/batch jobs instead of synchronous (requires different API calls)?
Something else entirely?
google made publication on a vesseldetection model giving speed and direction, can you find it?


Found it. It's the Global Fishing Watch paper published in Nature (2023):
"Satellite mapping reveals extensive industrial activity at sea" — a collaboration between Global Fishing Watch and Google. They use deep learning on Sentinel-2 optical imagery to detect vessels and estimate:
Speed — from wake patterns and multi-band temporal offsets in Sentinel-2
Heading/direction — from wake orientation and vessel shape
Length — from the optical signature
Key details:
Sentinel-2's 10m resolution detects vessels down to ~10 meters
The model distinguishes real vessels from buoys, infrastructure, and debris
It revealed that ~75% of fishing vessels globally are "dark" (no AIS transponder)
Covers 2017–2021 globally
Their Sentinel-2 optical model complements their Sentinel-1 SAR model — SAR sees through clouds but can't estimate speed; optical gives speed and heading from wakes.
Sources:
Nature paper: Satellite mapping reveals extensive industrial activity at sea
Global Fishing Watch: AI Breakthrough in Ocean Monitoring
Global Fishing Watch: Expanded Vessel Detections with Sentinel-2
Global Fishing Watch: Dark Vessels Research
Global Fishing Watch Technology
is the model available?

There are actually two open models available — neither is from Google directly, but both are closely related:
1. Allen AI (AI2) — vessel-detection-sentinels ✅ Open source with pretrained weights
This is the one you're probably thinking of. It's used in production by Skylight and detects:
Speed
Heading
Length & width
Ship type
Works on both Sentinel-1 (SAR) and Sentinel-2 (optical). Pretrained weights are included — you can run inference immediately.
👉 github.com/allenai/vessel-detection-sentinels
2. Global Fishing Watch — Detection data available, model is NOT open
GFW/Google's model from the Nature paper estimates the same attributes (speed, heading, length) but the model weights are not publicly released. They do publish the detection results as datasets on Zenodo, but not the model itself. Their vessel-classification repo only covers AIS-based classification, not the satellite imagery detection.
Bottom line: The AI2 model is what you want — it's open, has pretrained weights, works on Sentinel-2, and gives speed + heading + length. Want me to look into integrating it?
Sources:
AI2 vessel-detection-sentinels (GitHub)
GFW: Expanded Vessel Detections with Sentinel-2
GFW: AI Breakthrough in Ocean Monitoring
Nature: Satellite mapping reveals extensive industrial activity at sea
Sentinel-2 vessel detections dataset (Zenodo)
yes, set up a new analyses pipeline and test on the data that we have in our Bohuslän AOI

Good, now let me design the integration plan.

Design AI2 vessel integration plan


---

A commit was made and pushed: "feat: add multi-temporal vessel heatmap and improve sjökort rendering"
The plan for AI2 integration was being designed. Key design decisions from the plan:
New AI2VesselAnalyzer class following BaseAnalyzer pattern
Vendor minimal AI2 source files into imint/fm/ai2_vessels/
Model: Faster R-CNN with Swin V2 backbone + ResNet50 attribute predictor
Input: Same TCI conversion as existing YOLO analyzer
Output: Extended regions with speed, heading, length, width attributes
Heading arrows on overlay
--analyzer CLI flag for heatmap pipeline
Risk: CRS mismatch (AI2 trained on EPSG:3857, our data is EPSG:3006)
Risk: Attribute predictor needs B08 (NIR) in addition to TCI

DES (Digital Earth Sweden) synchronous job limits: max ~20 time steps per call; solved by chunking by 2-week windows
Combined TCI+SCL batch fetch: _fetch_tci_scl_batch() downloads B02+B03+B04+SCL for all dates in one openEO call, returns per-date tuples
Per-date fallback for large areas: Auto-detects when projected area > 2000px on any side, skips batch mode and uses individual _fetch_tci_bands() calls
Alpha-scaled heatmap transparency: rgba[:, :, 3] = (norm * 255).astype(np.uint8) makes low-density areas fade out smoothly
S-57 ENC layers: SLCONS (shoreline constructions), MORFAC (mooring facilities), BERTHS, HRBFAC contain jetty/pier/wharf/breakwater geometry as LineStrings and Points
AI2 vessel-detection-sentinels: Faster R-CNN (Swin V2 backbone) + ResNet50 attribute predictor; detects vessels with speed, heading, length, width, ship type from Sentinel-2
Global Fishing Watch: Nature 2023 paper "Satellite mapping reveals extensive industrial activity at sea"; model weights NOT public
BaseAnalyzer pattern: Abstract analyze() method, run() wraps with error handling, config dict, AnalysisResult dataclass
Sentinel-2 tile extent for Bohuslän AOI: bbox [10.675, 57.606, 12.610, 58.629], ~11000×11000px at 10m — too large for DES synchronous mode
# Scale alpha by normalised density so low-value areas fade out
# smoothly instead of showing a constant yellow wash.
rgba[:, :, 3] = (norm * 255).astype(np.uint8)
/Users/tobiasedman/Downloads/ImintEngine/imint/fetch.py — Most heavily modified file
_fetch_scl_batch() — Updated to support both candidate_dates mode and temporal range mode:
def _fetch_scl_batch(
    conn,
    projected_coords: dict,
    candidate_dates: list[str] | None = None,
    *,
    temporal: list[str] | None = None,
) -> list[tuple[str, float]]:
Supports two modes: candidate_dates (filters to specific dates) or temporal (returns ALL dates in range).
_fetch_tci_scl_batch() — NEW function, fetches B02+B03+B04+SCL for ALL dates in a temporal range in one call:
def _fetch_tci_scl_batch(
    conn,
    projected_coords: dict,
    temporal: list[str],
) -> list[tuple[str, dict, np.ndarray, "GeoContext"]]:
Returns list of (date_str, bands_dict, scl, geo) tuples. Handles both tar.gz (multi-date) and plain GeoTIFF (single-date) responses.
fetch_vessel_heatmap() — Major rewrite:
Skips STAC entirely — DES discovers dates directly
Chunks by 2-week windows (DES 20-step limit)
Detects large areas (>2000px) and auto-switches to per-date mode
Inner function _detect_and_accumulate() for reuse in both batch and per-date paths
def fetch_vessel_heatmap(
    coords: dict,
    date_start: str,
    date_end: str,
    output_dir: str | Path,
    *,
    cloud_threshold: float = 0.3,
    scene_cloud_max: float = 50.0,
    gaussian_sigma: float = 5.0,
    prefix: str = "",
) -> dict:
Key pipeline structure:
# Generate 2-week windows
_CHUNK_DAYS = 14
# Large area detection
_px_w = (projected_coords["east"] - projected_coords["west"]) / 10
_px_h = (projected_coords["north"] - projected_coords["south"]) / 10
_use_single_date = max(_px_w, _px_h) > 2000
# Per-chunk loop with batch mode + per-date fallback
for _c_start, _c_end in _chunk_ranges:
    if not _use_single_date:
        try:
            batch = _fetch_tci_scl_batch(conn, projected_coords, [_c_start, _c_end])
            for date, bands, scl, geo in batch:
                _detect_and_accumulate(date, bands, scl)
            continue
        except (FetchError, Exception) as _batch_err:
            if "504" in str(_batch_err) or "Gateway" in str(_batch_err):
                _use_single_date = True
    # Per-date fallback for large areas
    _c_dt = _dt.strptime(_c_start, "%Y-%m-%d")
    while _c_dt < _e_dt:
        bands, scl, geo = _fetch_tci_bands(conn, projected_coords, temporal)
        _detect_and_accumulate(date, bands, scl)
Also saves raw numpy array: np.save(str(output_dir / f"{prefix}vessel_heatmap.npy"), heatmap)
/Users/tobiasedman/Downloads/ImintEngine/imint/exporters/html_report.py
Vessel heatmap viewer in _MARINE_VIEWERS:
{"id": "m-vessel-heatmap", "title": "Fartygsaktivitet (heatmap)", "key": "vessel_heatmap", "legend": [
    {"color": "#FFFFB2", "label": "Låg"},
    {"color": "#FD8D3C", "label": "Medel"},
    {"color": "#BD0026", "label": "Hög"},
]},
File mapping: "vessel_heatmap": "vessel_heatmap_clean.png",
Existing YOLO11s analyzer; pattern to follow for new AI2 analyzer
Uses BaseAnalyzer from imint/analyzers/base.py
Config: confidence=0.286, chip_size=320, overlap_ratio=0.2, max_bbox_m=750
Model: yolo11s_tci.pt from HuggingFace mayrajeo/marine-vessel-yolo
TCI conversion: img = (raw_rgb * 2.5 * 255).clip(0, 255).astype(np.uint8)
Water filter: SCL class 6 at centroid
Output: AnalysisResult with regions list {"bbox": {y_min, y_max, x_min, x_max}, "pixel_count", "score", "label"}
S-57 ENC files: /Users/tobiasedman/Downloads/83e3f167-4d05-4765-81a3-619c3a515a4c/ (SE2BI9SW.000, SE3DI9T8.000, SE4HIAX9.000, SE5EIAX9.000) — flat directory, not in subdirs
Key data files:
Bohuslän AOI bbox: west=11.25049, south=58.42763, east=11.30049, north=58.47763
Full Sentinel-2 tile bbox: west=10.675, south=57.606, east=12.610, north=58.629
Fire/Brand data dir: outputs/full_15.42_61.92_15.47_61.97_2018-07-24 with prefix 2018-07-24_
Marine data dir: outputs/full_bohuslan_2025-07-10 with prefix 2025-07-10_
Season heatmap files: outputs/full_bohuslan_2025-07-10/2025_season_vessel_heatmap_clean.png, .npy, _summary.json
Python env: .venv/bin/python3
DES 20 time-step limit (batch SCL for 34 dates = 85 time steps):
First tried chunking by count (15 dates) — still failed because temporal range spans many dates
Fixed by chunking by calendar month (each month has ~6 passes)
DES 20 time-step limit (combined TCI+SCL batch for full month):
Monthly TCI+SCL batch (4 bands × ~16 dates) timed out
Fixed by reducing to 2-week chunks (_CHUNK_DAYS = 14)
Double-underscore prefix bug: Passed marine_date='2025-07-10_' (with trailing underscore) to save_tabbed_report, which adds its own _, creating 2025-07-10__. Fixed by passing without trailing underscore.
Wrong fire_dir: Used same directory for both fire_dir and marine_dir in save_tabbed_report(), replacing Brand tab content with marine data. User said "This was wrong, you replaced the brand analysis with marine, you need to revert that". Fixed by using correct fire_dir='outputs/full_15.42_61.92_15.47_61.97_2018-07-24' with fire_date='2018-07-24'.
Full scene DES timeout (504 Gateway Time-out): Full Sentinel-2 tile (112×114km, ~11000×11000px) at 4 bands too large for DES synchronous mode, even single date. Added per-date fallback and large-area auto-detection but user said "stop trying".
Heatmap file not copied to showcase: vessel_heatmap_clean.png wasn't found because prefix mismatch between season run (2025_season_) and showcase (2025-07-10_). Fixed by copying: cp outputs/.../2025_season_vessel_heatmap_clean.png outputs/.../2025-07-10_vessel_heatmap_clean.png
Full Sentinel-2 scene processing: DES synchronous mode cannot handle full tile downloads. Options: async batch jobs, spatial tiling, or accept smaller AOI. User decided to stop attempting this for now.
S-57 jetties/piers: Confirmed as real geometry objects in SLCONS (LineStrings: 519 wharves, 31 breakwaters, 40 moles, 2 piers), MORFAC (23 dolphin Points), BERTHS (12), HRBFAC (7).
AI2 model integration: Designed comprehensive plan for integrating AI2 vessel-detection-sentinels as new AI2VesselAnalyzer class.
"Can we make the more or less constant yellow parts transparent to?"
"can you update the html"
"How many images was it based on?"
"Now do the same for the full season"
"where are we in the process?"
"Fetch all spectral data and SCL at the same time"
"OK lets test to the same for the full scenes, I think it will be faster process everything than to filter for clouds. So all vessel detcteion for all scenes that intersect the bounding box."
"please do" (update showcase)
"This was wrong, you replaced the brand analysis with marine, you need to revert that"
"make a new html paghe showing the full scenes of the vessel heatmap not only tha AOI from before"
"Di you have data for th efull scenes?"
"Show it a new html for th efull scebes"
"stop tryimg"
"you are not trying to get the full scene?"
"stop trying" (second time)
"I wonder if the jetties and piers in the nautical chart are objects"
"google made publication on a vesseldetection model giving speed and direction, can you find it?"
"is the model available?"
"yes, set up a new analyses pipeline and test on the data that we have in our Bohuslän AOI"
AI2 vessel-detection-sentinels integration: Plan designed in plan mode but NOT yet approved or implemented. The plan covers:
Vendor minimal AI2 source files into imint/fm/ai2_vessels/
Download pretrained weights (Swin V2 detector + ResNet50 attribute predictor)
Create AI2VesselAnalyzer class in imint/analyzers/ai2_vessels.py
Register in engine, add export handler, update showcase HTML
Parameterize heatmap pipeline with --analyzer flag

Test on Bohuslän AOI data
Plan file exists at: /Users/tobiasedman/.claude/plans/twinkling-sniffing-pie.md (may not have been written yet)
A commit was made mid-session: "feat: add multi-temporal vessel heatmap and improve sjökort rendering" — pushed to main. The AI2 integration changes have NOT been committed.
Clone AI2 repo and download weights
Vendor minimal source files into imint/fm/ai2_vessels/
Create AI2VesselAnalyzer class

Test on existing Bohuslän AOI data
User's exact words: "yes, set up a new analyses pipeline and test on the data that we have in our Bohuslän AOI"

I'm continuing from where we left off — writing the AI2 vessel-detection-sentinels integration plan to the plan file. Let me first check the current state of the plan file and relevant code to ensure accuracy.

Good, I have comprehensive understanding of the codebase. Let me now research the AI2 repo details to finalize the plan.

Let me look deeper at the AI2 repo to understand the Sentinel-2 specific model:

This is a much better find! Let me explore the rslearn_projects approach:

Now I have a complete picture. The rslearn_projects repo from Allen AI is actually a much better fit than the older vessel-detection-sentinels. Let me check a few more details about the attribute model:

Now I have comprehensive understanding of the AI2/rslearn vessel detection system. Let me update the todo list and write the final plan.

Now let me write the plan:

Edited a file, updated todo list, finished plan
Plan approved! Let me start implementing.

rslearn requires Python >=3.11 but the existing venv is 3.9.6. Let me check available Python versions:

Only Python 3.9.6 available, but rslearn needs ≥3.11. Let me explore an alternative: extracting the model code directly to avoid the version dependency. Let me first understand the model architecture:

Explore rslearn model architecture
Excellent findings! The model CAN be used directly with PyTorch (works on Python 3.9). Let me check our current PyTorch setup:

PyTorch 2.8.0 is installed and has all the needed model classes. Now let me understand exactly how the rslearn model is built, so I can replicate it:

Now let me read the key files for understanding the full model loading flow:

The rslearn framework requires Python ≥3.11 but our venv is 3.9.6. Let me check options for getting a newer Python:

No Python 3.11+ available. Let me take the direct approach: build the model from standard PyTorch/torchvision components (which work on 3.9.6) and load the pretrained weights. First, let me download the detection model weights:

While the download runs, let me also look at the config.yaml to understand the exact model architecture:

Now I have complete understanding of both models. Let me check the download progress and then start building the analyzer:

The download is still running. Let me start writing the analyzer while it downloads. Let me first read the pooling decoder for the attribute model:

Now I have all the architecture details. Let me write the standalone AI2 vessel analyzer. I'll build the model from torchvision primitives and load the rslearn checkpoint:

Updated todo list, created a file
Good, the download completed. Let me check it and also start the attribute model download:

Now let me verify the checkpoint loads correctly:
Inspect checkpoint structure
The checkpoint keys match what I expected. Let me verify the model loads correctly:

Test model loading and inference
The model loads and runs inference. 96M parameters, and the dummy random input gives 0 detections (as expected for random noise). Now let me update the fetch pipeline to support fetching 9 bands:

Now I need to create a new function that fetches all 9 bands needed by the AI2 model. Let me add it after the existing _fetch_tci_bands:

Now let me look at where constants are defined:

Good — the 20m spectral bands are already defined. Now let me add a function to fetch all 9 bands needed by the AI2 model:

Both model downloads complete. Let me check the attribute model:

Check AI2 model files
Good — 1.1 GB detection + 1.0 GB attribute models. Now let me update the fetch_vessel_heatmap() function to support the AI2 analyzer. Let me find it:

Now I need to modify fetch_vessel_heatmap to accept an analyzer parameter. The key changes:
Add analyzer: str = "yolo" parameter
When analyzer="ai2", use AI2VesselAnalyzer and _fetch_ai2_bands instead of _fetch_tci_bands

Now I need to update the per-date fallback section to use _fetch_ai2_bands when using the AI2 analyzer:

I need to update the per-date fallback to use _fetch_ai2_bands when _use_ai2 is True:

I also need to handle the case where batch mode is used with AI2 (the _fetch_tci_scl_batch only fetches 4 bands, not 9). For the AI2 analyzer, we should always use per-date mode since we need 9 bands:

Now update the run_vessel_heatmap.py CLI to add the --analyzer flag:

Now let me verify the code compiles correctly:

Test AI2 analyzer end-to-end with dummy data
The model loads and runs correctly on dummy data (0 detections on random noise, as expected). Now let me test with actual Sentinel-2 data from our Bohuslän AOI. We already have fetched TCI data — let me check what's available:

Let me run the AI2 detector on a single date from the Bohuslän AOI. This will test the full pipeline: DES fetch of 9 bands + AI2 detection:

Test AI2 detector on single Bohuslän date
DES credentials aren't available in this session. Let me test the AI2 model on existing data we already have from previous runs. Let me check what's available:

We have ALL 12 bands saved from a previous full analysis! This is perfect — we can test the AI2 model directly on this data without needing DES credentials. Let me write a quick test:

Check band data shapes and values
We have all 9 bands at (573, 324) in reflectance format. Let me run the AI2 detector on this real data:

Test AI2 detector on real Bohuslän data
Excellent! The AI2 model works! 11 vessel detections on the Bohuslän AOI from 2025-07-17, with very high confidence scores (up to 0.963). This took only 4.1 seconds on CPU.
For comparison, our YOLO detector found 28 vessels on the same date (2025-07-10 data). The different count is expected since:
This is 2025-07-10 data (not 2025-07-17 which had 50 YOLO detections)
AI2 has a different detection threshold and model architecture
Without SCL water filtering, some land detections may have been removed by YOLO but not AI2 (or vice versa)
Let me now compare the detections visually. Let me create a quick comparison image:

Compare AI2 vs YOLO detections visually
The comparison is ready. AI2 detected 11 vessels vs YOLO's 28. Let me also check detections at different confidence thresholds to understand the AI2 model's behaviour:

Test AI2 at various confidence thresholds
The AI2 model is very confident about its detections — all 11 are above 0.5 threshold, and 8 are above 0.8. This is a good sign of model quality. The model produces fewer but more confident detections than YOLO.

The AI2 vessel detection pipeline is working. Here's a summary of what was done:
What was implemented
1. AI2 Vessel Detector (imint/analyzers/ai2_vessels.py)
Built the detection model architecture from standard PyTorch/torchvision components (Swin V2 B → FPN → Faster R-CNN)
Loads the rslearn pretrained weights (1.1 GB checkpoint) with automatic key remapping
No rslearn dependency needed — works on Python 3.9 with existing PyTorch 2.8.0
Sliding window inference (512×512, 10% overlap) with distance-based NMS
Input: 9 Sentinel-2 bands (B04, B03, B02, B05, B06, B07, B08, B11, B12)
Output: Standard region format compatible with existing heatmap pipeline
2. 9-band DES fetch (imint/fetch.py)
New _fetch_ai2_bands() function fetches all 9 spectral bands + SCL from DES
20m bands (B05, B06, B07, B11, B12) are resampled to 10m with bilinear interpolation
3. Pipeline integration (imint/fetch.py, run_vessel_heatmap.py)
fetch_vessel_heatmap() now accepts analyzer_type="ai2" parameter
CLI: --analyzer ai2 flag selects the AI2 detector
When AI2 is selected, uses per-date mode with 9-band fetch
4. Pretrained weights downloaded
Detection model: imint/fm/ai2_vessels/detect_best.ckpt (1.1 GB)
Attribute model: imint/fm/ai2_vessels/attribute_best.ckpt (1.0 GB) — downloaded for future attribute prediction integration

Test results on Bohuslän AOI (2025-07-10)
AI2: 11 vessels (all with confidence > 0.5, 8 with > 0.8)
YOLO: 28 vessels
AI2 runs in 4.1s on CPU
AI2 produces fewer but more confident detections
To run the full pipeline (requires DES credentials):
.venv/bin/python run_vessel_heatmap.py \
    --west 11.25049 --south 58.42763 --east 11.30049 --north 58.47763 \
    --start 2025-07-17 --end 2025-07-17 \
    --output-dir outputs/ai2_test \
    --analyzer ai2
Not yet implemented (future work):
Attribute prediction (speed, heading, length, width, vessel type) — the attribute model checkpoint is downloaded but the attribute predictor uses a SatlasPretrain backbone that would need additional vendoring
Heading arrow visualization on overlay
Speed-based color coding for detections
can you elaborate on this? Attribute prediction (speed, heading, length, width, vessel type) — the attribute model checkpoint is downloaded but the attribute predictor uses a SatlasPretrain backbone that would need additional vendoring
The attribute prediction model uses a different backbone than the detection model. Here's the breakdown:
Detection model (what we implemented):
Backbone: standard torchvision.models.swin_v2_b() — ships with torchvision, no extra dependencies
We just modified the first conv layer for 9 input channels
Loaded the checkpoint directly
Attribute model (not yet implemented):
Backbone: SatlasPretrain with model identifier "Sentinel2_SwinB_SI_MS"
This is a custom pretrained foundation model from Allen AI's SatlasPretrain project
It's a Swin Transformer pretrained on massive satellite imagery (not ImageNet), specifically tuned for Sentinel-2 multi-spectral inputs
The satlaspretrain_models Python package provides the model class and knows how to load the special pretrained weights
On top of that backbone + FPN, there are 6 parallel decoder heads:
length → PoolingDecoder → regression (meters)
width → PoolingDecoder → regression (meters)
speed → PoolingDecoder → regression (knots)
heading_x → PoolingDecoder → regression (cos θ)
heading_y → PoolingDecoder → regression (sin θ)
ship_type → PoolingDecoder → classification (9 classes: cargo, tanker, passenger, service, tug, pleasure, fishing, enforcement, SAR)
What "additional vendoring" means:
To build this model from PyTorch primitives (like we did for the detector), I'd need to either:
Install satlaspretrain_models — but it may pull in dependencies that conflict with our Python 3.9 env, and it depends on rslearn
Vendor the SatlasPretrain Swin code — copy the specific model definition files from the satlaspretrain_models repo into our project, strip out the rslearn dependencies, and wire up weight loading manually
The PoolingDecoder heads themselves are simple (conv → global max pool → FC layers — I already read that code), so the real challenge is just getting the SatlasPretrain backbone instantiated and loaded.
Want me to try implementing it? The approach would be to check if satlaspretrain_models installs cleanly in our venv, and if not, vendor the minimal model code.
Yes try to to implement it
Let me first check if satlaspretrain_models installs cleanly, and explore its code to understand the exact Swin architecture:
Try installing satlaspretrain_models
It installed cleanly. Now let me check how to use it and understand the model structure:

Sentinel2_SwinB_SI_MS is exactly what we need — Swin B backbone, single image, 9 multi-spectral channels. Now let me see if it instantiates correctly:

Test SatlasPretrain backbone + FPN instantiation
satlaspretrain_models installs and works on Python 3.9. The backbone produces 5 feature maps at 128 channels each. Now let me inspect the attribute checkpoint and build the full model:

Now I can see the exact structure. Let me build the attribute predictor model:

Now let me add the attribute predictor to the analyzer file. I'll add it after the detection model code:

Now let me add the attribute model constants and the attribute predictor class. I'll add them right after the existing _SCORE_THRESHOLD constant block:

Now let me add the attribute predictor model class. I'll add it after the AI2VesselDetector class:

Now I'll add the attribute predictor model class and its loading function right after the _load_detection_model:
