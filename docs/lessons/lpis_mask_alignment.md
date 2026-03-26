# Lessons Learned: LPIS Parcel Mask Alignment with Sentinel-2 Imagery

**Date:** 2026-03-26
**Pipeline:** ImintEngine Crop Classification
**Author:** Tobias Edman (with Claude session assist)
**Status:** Resolved and verified

---

## 1. Problem Statement

When rasterizing LPIS (Land Parcel Identification System) polygon masks from Jordbruksverket
and overlaying them on Sentinel-2 imagery tiles, the masks were spatially misaligned. Parcels
appeared in the wrong locations, rotated, or mirrored relative to the underlying satellite
pixels. This made the crop classification labels unusable until the root cause was identified
and a correct geometric transform was applied.

---

## 2. Root Cause: CRS Axis Order Mismatch Between LPIS and Sentinel-2

The core issue is a coordinate axis order conflict between two data sources that both
nominally use EPSG:3006 (SWEREF99 TM).

### 2.1 LPIS GeoParquet from Jordbruksverket

- **Source:** epub.sjv.se (Jordbruksverket open data)
- **Format:** GeoParquet
- **CRS:** SWEREF99 TM (EPSG:3006)
- **Axis order:** N, E (Northing as x-coordinate, Easting as y-coordinate)

This is the "natural" axis order for SWEREF99 TM as defined by the EPSG registry. The EPSG
definition specifies that the first axis is Northing and the second is Easting. GeoParquet
files from Jordbruksverket follow this convention faithfully, meaning that when you read a
geometry's x-coordinate, you are actually getting Northing, and the y-coordinate is Easting.

### 2.2 Sentinel-2 Tiles via CDSE Sentinel Hub Process API

- **Source:** Copernicus Data Space Ecosystem (CDSE) Sentinel Hub Process API
- **CRS request parameter:** EPSG:3006
- **Axis order:** E, N (Easting as x, Northing as y)

The Sentinel Hub Process API follows the standard GIS software convention (matching GDAL,
rasterio, QGIS, and most geospatial libraries) where x = Easting and y = Northing,
regardless of what the EPSG registry formally specifies. This is the de facto standard in
nearly all GIS tooling.

### 2.3 The Bbox Stored in .npz Tile Files

The `bbox_3006` array stored in the pipeline's `.npz` tile files uses:

```
[E_west, N_south, E_east, N_north]
```

This is standard E,N (GIS convention) order, matching how the Sentinel-2 data was fetched.

### 2.4 Summary Table

| Data Source              | Axis Order  | x-coordinate meaning | y-coordinate meaning |
|--------------------------|-------------|----------------------|----------------------|
| LPIS GeoParquet (SJV)    | N, E        | Northing             | Easting              |
| Sentinel-2 (CDSE API)    | E, N        | Easting              | Northing             |
| bbox_3006 in .npz files  | E, N        | Easting              | Northing             |

---

## 3. The Spatial Query: Swapping Axes for LPIS

When querying LPIS polygons spatially using the bbox from a Sentinel-2 tile, the axis order
must be swapped to match the LPIS N,E convention:

```python
# bbox_3006 from .npz file: [E_west, N_south, E_east, N_north]
E_west, N_south, E_east, N_north = tile["bbox_3006"]

# LPIS spatial query needs N,E order:
from shapely.geometry import box
query_box = box(N_south, E_west, N_north, E_east)
# box(minx, miny, maxx, maxy) -> box(N_south, E_west, N_north, E_east)
# Because in LPIS space: x=Northing, y=Easting
```

If you pass `box(E_west, N_south, E_east, N_north)` directly (without swapping), the query
will select parcels from a completely wrong geographic region or return no results.

---

## 4. Rasterization: Operating in LPIS N,E Space

When rasterizing the LPIS polygons using `rasterio.features.rasterize()`, the `from_bounds`
transform must also use N,E order to match the LPIS geometry coordinates:

```python
from rasterio.transform import from_bounds
from rasterio.features import rasterize

# Build transform in LPIS N,E space
transform = from_bounds(
    N_south, E_west,  # west, south in rasterio terms (but N,E in LPIS)
    N_north, E_east,  # east, north in rasterio terms (but N,E in LPIS)
    width=tile_width,
    height=tile_height
)

mask = rasterize(
    [(geom, crop_code) for geom, crop_code in parcels],
    out_shape=(tile_height, tile_width),
    transform=transform,
    fill=0,
    dtype="uint16"
)
```

At this point, `mask` is a valid rasterization but it lives in LPIS N,E pixel space, not
Sentinel-2 E,N pixel space. The pixels are correct relative to the LPIS coordinate system,
but they will not overlay correctly on the S2 image array.

---

## 5. The Correct Transform: rot90(mask, 2).T

To convert from LPIS N,E raster space to Sentinel-2 E,N raster space:

```python
import numpy as np
aligned_mask = np.rot90(mask, 2).T
```

This is equivalent to rotating the array 180 degrees and then transposing it. Breaking it
down:

1. `np.rot90(mask, 2)` -- rotate 180 degrees (equivalent to `flipud(fliplr(mask))`)
2. `.T` -- transpose (swap rows and columns)

The combined effect correctly maps the N,E raster grid onto the E,N raster grid used by
the Sentinel-2 pixel array.

**No other single operation works.** See Section 7 for the full enumeration of attempts.

---

## 6. Verification

### 6.1 Verification Method

The transform was verified by overlaying the colored LPIS crop mask (with transparency) on
top of the Sentinel-2 RGB composite. Parcel boundaries in the mask should align with visible
field boundaries in the satellite image.

### 6.2 Verified Tiles

| Tile ID    | Status   | Notes                                           |
|------------|----------|-------------------------------------------------|
| 44703784   | Verified | Parcels align with visible field edges in RGB   |
| 44983876   | Verified | Second independent tile confirms the transform  |

Always verify on at least two geographically distinct tiles to rule out coincidental alignment.

---

## 7. Debugging History: What Did NOT Work

During the debugging process, many incorrect transforms were tried. Documenting these to
prevent future repetition:

| Attempt                          | Result                                      |
|----------------------------------|---------------------------------------------|
| Identity (no transform)          | Mask rotated/mirrored relative to S2        |
| `np.flipud(mask)`                | Partial fix, still misaligned               |
| `np.fliplr(mask)`                | Partial fix, still misaligned               |
| `mask.T` (transpose only)        | Partial fix, axis swap correct but flipped  |
| `np.rot90(mask, 1)`             | Wrong rotation angle                         |
| `np.rot90(mask, 3)`             | Wrong rotation angle                         |
| `np.flipud(mask.T)`             | Close but not quite                          |
| `np.fliplr(mask.T)`             | Close but not quite                          |
| **`np.rot90(mask, 2).T`**       | **Correct -- verified on two tiles**         |

### 7.1 Common Mistakes Made During Debugging

1. **Trying individual operations in isolation.** `flipud`, `fliplr`, and `transpose` were
   each tried alone. None of them worked because the axis order mismatch requires a compound
   transformation (both a rotation and an axis swap).

2. **Applying multiple transforms without verifying each step.** When one transform did not
   work, another was applied on top of it without first reverting to the untransformed state.
   This led to cascading errors where the mask was doubly transformed.

3. **Re-running enrichment without checking prior state.** The enrichment pipeline was re-run
   multiple times. Some runs already had a partial transform baked into the tile data from a
   previous attempt. This made it impossible to know whether the current transform was being
   applied to raw or pre-transformed data.

4. **Losing track of "ground truth" orientation.** After several enrichment cycles, it became
   unclear which orientation was the original untransformed state. Each re-enrichment
   potentially changed the baseline, making it harder to reason about what transform was
   actually needed.

---

## 8. Correct Approach for Future CRS Alignment Issues

If you encounter a similar axis order or spatial alignment problem in the future, follow this
protocol:

### Step 1: Establish Ground Truth

Open the satellite image (or a known reference layer) in a tool where you can verify
geographic location. Google Maps, Google Earth, or QGIS with a basemap are all suitable.
Confirm that the satellite tile covers the area you expect.

### Step 2: Enumerate All Possible Orientations

There are exactly 8 possible orientations of a 2D array (the dihedral group D4):

```python
orientations = {
    "identity":       lambda m: m,
    "rot90":          lambda m: np.rot90(m, 1),
    "rot180":         lambda m: np.rot90(m, 2),
    "rot270":         lambda m: np.rot90(m, 3),
    "transpose":      lambda m: m.T,
    "rot90_T":        lambda m: np.rot90(m, 1).T,
    "rot180_T":       lambda m: np.rot90(m, 2).T,   # <-- the correct one for LPIS
    "rot270_T":       lambda m: np.rot90(m, 3).T,
}
```

### Step 3: Test All 8 on a Single Tile

Generate a visualization of each orientation overlaid on the S2 RGB for one tile. Use a
simple matplotlib grid:

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ax, (name, fn) in zip(axes.flat, orientations.items()):
    oriented_mask = fn(raw_mask)
    ax.imshow(s2_rgb)
    ax.imshow(oriented_mask, alpha=0.4, cmap="tab20")
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.savefig("orientation_test.png", dpi=150)
```

### Step 4: Confirm with Visual Inspection

Have a human (the user, a colleague, yourself) confirm which of the 8 orientations produces
correct alignment. Parcel boundaries should match visible field edges in the satellite image.

### Step 5: Apply Exactly That Transform

Apply the confirmed transform and nothing else. Do not add extra flips or rotations "just in
case." Do not combine it with transforms from previous attempts.

### Step 6: Verify on a Second Tile

Before applying the transform to the full dataset, verify it on a second, geographically
distinct tile. If it works on two independent tiles, you can be confident it is correct.

---

## 9. Technical Context: Why EPSG:3006 Has This Ambiguity

The EPSG registry defines SWEREF99 TM with axis order (Northing, Easting). This is the
mathematically "correct" order per the EPSG standard. However, virtually all GIS software
(GDAL, QGIS, rasterio, Shapely, PostGIS, Sentinel Hub, Google Earth Engine) uses (Easting,
Northing) as the default axis order, following the lon/lat and x/y convention that has been
the de facto standard in software for decades.

This creates a persistent trap: data producers who follow the EPSG standard strictly (like
Jordbruksverket's GeoParquet export) will have swapped axes relative to data consumed
through standard GIS APIs. The OGC tried to address this with the `AXIS` element in WKT and
the `urn:ogc:def:crs:EPSG::3006` vs `EPSG:3006` distinction, but in practice the ambiguity
persists across the ecosystem.

**Rule of thumb:** Always check the actual coordinate values. If the x-coordinates are in
the range of 6,000,000-7,700,000 and y-coordinates are in the range of 200,000-1,000,000,
then x = Northing and the data uses the EPSG-native N,E order. If x is in the smaller range
(200,000-1,000,000), then x = Easting and the data uses the GIS-convention E,N order.

---

## 10. Session Context: Other Work Completed

This lesson was learned during a broader working session that covered multiple ImintEngine
and related projects. For context, the other topics addressed in the same session are
summarized below.

### 10.1 One-Pager Editorial Review

Reviewed and refined the Imint AB one-pager document for clarity, messaging, and positioning.
Ensured the value proposition and technical differentiators were communicated effectively for
the intended audience.

### 10.2 ImintEngine New Analyzers

Designed and scoped several new analyzer modules for the ImintEngine platform:

- **SAMGeo:** Segment Anything Model adapted for geospatial imagery segmentation.
- **InSAR:** Interferometric SAR analysis for surface deformation and change detection.
- **Clay:** Clay Foundation Model integration for general-purpose Earth observation embeddings.
- **TerraMind:** TerraMind foundation model for multi-modal geospatial understanding.
- **S1 CDSE:** Sentinel-1 SAR data ingestion via the Copernicus Data Space Ecosystem API.

### 10.3 Crop Classification Pipeline

Built out the core crop classification pipeline with the following components:

- **Label schema:** Dual schema support for SJV (Jordbruksverket) crop codes and SCB
  (Statistics Sweden) classification. Mapping between the two systems.
- **Training data:** Combined LUCAS (Land Use/Cover Area frame Survey) field samples with
  LPIS parcel boundaries for supervised classification labels.
- **Sentinel-2 fetch strategy:** Vegetation Phenology and Productivity (VPP) guided temporal
  selection of Sentinel-2 scenes. Instead of fixed date ranges, the pipeline selects
  acquisitions aligned with phenological stages (green-up, peak, senescence) to maximize
  crop type separability.
- **Auxiliary data:** ERA5 climate reanalysis variables (temperature, precipitation,
  growing degree days) as supplementary features for the classifier.
- **LPIS mask alignment:** The subject of this lessons learned document.

### 10.4 DES Chatbot

Developed a domain-specific chatbot for DES (likely internal or client project):

- **Model:** Mistral 7B as the base language model.
- **Architecture:** Retrieval-Augmented Generation (RAG) with curated source documents.
- **Deployment:** WordPress frontend integration, Kubernetes (K8s) backend orchestration.
- **Knowledge sources:** Curated document corpus with quality-controlled ingestion.
- **Knowledge Graph (KG) integration:** Structured knowledge graph used alongside vector
  retrieval for improved factual grounding and relationship-aware responses.

### 10.5 ICE Connect Deployment

Deployment work on the ICE Connect system:

- **Qdrant:** Vector database instance running successfully for embedding storage and
  similarity search.
- **vLLM:** Encountered CUDA compatibility issues with vLLM (the high-throughput LLM serving
  engine). The GPU driver version, CUDA toolkit version, and vLLM build need to be mutually
  compatible. This is a common deployment friction point. Resolution requires matching the
  vLLM wheel to the specific CUDA version installed on the host, or building from source
  against the correct CUDA headers.

---

## 11. Key Takeaways

1. **EPSG axis order is a trap.** Never assume two datasets in the same CRS have the same
   axis order. Always inspect actual coordinate values.

2. **There are exactly 8 possible 2D orientations.** Test all of them systematically instead
   of guessing with individual flips and rotations.

3. **Verify visually on real data.** No amount of reasoning about coordinate systems
   substitutes for overlaying the mask on the image and checking with your eyes.

4. **Work from a clean state.** When debugging alignment, always start from the raw
   untransformed data. Never stack transforms from previous failed attempts.

5. **Verify on two tiles.** One tile could align by coincidence (e.g., a symmetric region).
   Two geographically distinct tiles give confidence.

6. **The specific fix for LPIS-on-S2 in EPSG:3006 is `np.rot90(mask, 2).T`.**

---

*End of document.*
