# Plan for Swedish DTM enrichment at 10 m

Status: ready for implementation planning, 29 October 2026.

Add Lantmäteriet's bare-earth elevation to the existing tile collection as a
separate, versioned layer at exactly 10 m. The forest channels already describe
vegetation; this layer supplies the terrain beneath it for later retraining.
The initial R4 service and the R5 crop experiment retain their existing inputs.

## Data contract

| Property | Planned contract |
|---|---|
| Product | Lantmäteriet Markhöjdmodell Nedladdning, native 1 m DTM |
| Horizontal grid | EPSG:3006, exactly the source tile's existing 10 m lattice |
| Vertical reference | Metres in RH 2000, retained without a vertical shift |
| Reduction | Area-weighted mean of valid native 1 m cells over each 10 m cell |
| Stored extent | Full 512 × 512 source tile, covering 5,120 × 5,120 m |
| Evaluation/inference crop | Centre [4:508, 4:508] crop: 504 × 504 and 5,040 × 5,040 m |
| Training crop | Full DTM and masks enter the shared random crop, flip and rotation path |
| Arrays | dtm: float32; dtm_coverage: float32 fraction; dtm_valid: uint8 |
| Missing values | NaN in dtm where coverage is incomplete; zero and negative heights remain valid |
| Release | Immutable source inventory and processing version, bound by hashes |

Lantmäteriet documents nationwide 1 m coverage, RH 2000 heights and authenticated
STAC access after a free access order. Its 5 October 2026 notice says all DTM
files were regenerated to correct missing-height areas. Start from the current
corrected assets and freeze the actual asset identities, rather than assuming
a product name identifies fixed bytes.
[Product documentation](https://geotorget.lantmateriet.se/dokument/projects/markhoejdmodell-nedladdning/released/1/),
[correction notice](https://www.lantmateriet.se/sv/geodata/vara-produkter/Produktnyheter/Geografisk-information/rattning-av-markhojdmodell-nedladdning/).
Sources checked 29 October 2026.

DTM is a dated ancillary release, not an annual observation matched to each
tile's label year. Record acquisition ranges and source revision separately
from the satellite year. A later historical experiment must state this temporal
limitation or select an independently justified historical terrain release.

## 1. Inventory and establish access

Create a private inventory of active training and evaluation tile roots, their
cohort manifests, tile hashes, dimensions, years and resolved grids. Include
the frozen evaluation copies through read-only access. Separate unsupported
legacy sizes and conflicting geometry for review; do not silently resize them.

Use [resolve_fetch_bbox](../../imint/training/tile_bbox.py) and
[TileConfig](../../imint/training/tile_config.py). The shared resolver rebuilds
bounds from the centre, so also check any stored affine and authoritative
campaign geometry for disagreement. A reconstructed bbox alone does not prove
that the original spectral array was correctly located.

Count tile records and distinct spatial grids separately. Different years
on the same grid may share the same DTM artifact, but retain separate manifest
bindings. Use source-root identity plus tile SHA and grid hash; basenames alone
can collide between years, campaigns and frozen copies.

Obtain or reuse Geotorget permission for the height API; keep credentials in
the existing secret mechanism. Query and paginate the current DTM collection,
check access to a small source window, and record its current terms and
attribution. An access order is a prerequisite, not an assumption that a free
product permits anonymous requests.

Deliverable: inventory, duplicate-grid count, unavailable/conflicting records,
source coverage index, and a bounded pilot manifest. Actual cohort IDs,
coordinates and paths remain in private storage.

## 2. Implement one reusable DTM provider

Add a provider following the existing auxiliary fetcher signature:
west, south, east, north, size_px and cache_dir. Reuse repository geometry,
tile discovery and caching conventions. There is currently no Lantmäteriet
height client in imint or scripts.

For each grid:

1. Discover every intersecting source asset. Assemble a source mosaic before
   aggregation so a target pixel crossing two files is treated as one cell.
2. Read the original 1 m band and its validity mask. Respect the GeoTIFF affine
   and pixel interpretation; never invent a half-pixel offset. Disable automatic
   overview selection for the reference reduction. Provider-generated overviews
   are usable later only after their averaging semantics are verified.
3. Resolve overlapping revisions deterministically using the frozen source
   inventory, with one selected asset per overlapping area. Preserve every
   contributing source identity; iteration order must not decide the elevation.
4. Aggregate to the exact destination affine
   (10, 0, west, 0, -10, north), with explicit bounds and dimensions.
   Do not snap the destination again or enlarge it to fit a source tile.
5. Compute coverage as valid source area divided by the full 100 square metre
   destination footprint. Outside-source areas and holes contribute zero
   coverage. Averaging only the available source extent would overstate coverage.
6. Accept a pixel only when coverage is at least 1 minus 1e-6 and its height is
   finite. Otherwise store NaN and dtm_valid=0. Record the incomplete fraction;
   do not substitute GLO-30 or infer validity from height being nonzero.

An aligned complete 10 m cell is the mean of 100 native cells. For fractional
source-grid intersections, use area weights. GDAL average excludes nodata,
so coverage must be computed independently over a complete target footprint.
Use native-resolution input, equivalent to GDAL's overview selection NONE.
[GDAL warp documentation](https://gdal.org/en/stable/programs/gdalwarp.html).

Keep RH 2000 as the new training contract. A diagnostic subtraction from GLO-30
needs an explicit vertical-reference transformation first: GLO-30 uses EGM2008
and includes surface structures. A raw difference is neither a DTM error score
nor canopy height.
[Copernicus DEM specification](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).

## 3. Publish immutable enrichment artifacts

Store a new sidecar per unique grid under a versioned DTM release directory;
do not rewrite source tile NPZ files, existing dem arrays, labels or frozen
manifests. The release manifest maps each source tile hash to its DTM sidecar.
This preserves current checkpoint inputs and avoids rewriting large TESSERA
embeddings merely to add one small channel.

Each sidecar contains the three arrays above plus affine, horizontal CRS and
vertical reference. Its manifest entry records source item/asset IDs, source
acquisition range and update revision, immutable asset/checksum identifiers,
the exact native window's data-and-mask hash, processing configuration,
code SHA, runtime digest and output hash. Exclude secrets and signed URLs.
If an upstream asset is mutable, use conditional reads against a strong
version identifier or mirror and hash it before processing.

Write temporary files outside the tile-discovery glob, reopen them, validate
arrays and geometry, then atomically rename them. Publish a release manifest
last, once all expected entries have terminal status. Resume only when the
source, grid, processing and output hashes match; a presence flag is insufficient.
Use a lock per output identity if concurrent workers can reach the same grid.

A successful complete sidecar has every pixel valid. Partial sidecars may be
retained for diagnosis but remain explicitly partial and are not silently
counted as training-ready. Missing sea/border coverage remains visible.
The later model experiment must declare its missing-input/cohort policy before
training or evaluation; it must not silently drop difficult evaluation tiles.

## 4. Verify before a collection-wide pass

Start with 16 non-held-out tiles selected across forest, agriculture,
settlement, steep terrain, northern Sweden, coast/border, source-file seams
and incomplete coverage. Export georeferenced pilot GeoTIFFs for inspection.

Acceptance checks:

- Exact 10 m affine, lattice origin, 512 × 512 dimensions and source-tile bounds.
  For evaluation/inference centre cropping, verify the 40 m east/south origin
  shift and 504 × 504 shape. Training uses the shared augmentation path below.
- Independent native-window means agree with sampled output cells within
  1 mm, allowing float32 rounding. Test seams using separately assembled
  reference windows, and check that overlapping target grids give identical cells.
- Zero and negative valid elevations survive; holes and outside coverage cannot
  become valid through averaging. A source error cannot become a successful tile.
- Inspect source versus reduced DTM and terrain features on a geographic map.
  Do not use forest-class correlation to force an offset into a terrain raster.
- Reopen every pilot sidecar and inspect its manifest. Verify source tile and
  frozen-manifest hashes are unchanged.
- Interrupt and resume the pilot; completed artifacts remain intact, incomplete
  files are ignored, duplicate workers publish once, and altered inputs invalidate
  cached results.

Measure cold and cached reads, download bytes, CPU seconds, peak memory and
disk usage. Begin with one download worker; increase concurrency only within
the provider's observed limits. This is a CPU/raster campaign, not a GPU job.

Then run a 100-tile batch from the inventoried cohort. Continue to the full
collection only after the pilot and batch pass and their measured download,
storage and runtime budget is accepted. Pin the producing code and runtime
before the campaign; validate any Kubernetes manifest server-side before launch.

## 5. Reconcile and prepare the later training change

For every inventoried tile, report exactly one terminal outcome: complete,
partial, failed or unsupported. Reconcile those counts against the inventory
and report both tile records and unique DTM grids. Publish the successful
subset explicitly; a campaign with unresolved gaps is not a complete enrichment.

The earlier [terrain-channel decision](independent_holdout_validation.md#8b-future-work--lantmäteriet-1-m-dem-not-for-this-run),
recorded on 18 August 2026, proposes pairing DTM with slope, aspect and TWI in a
later retrain. This enrichment delivers the 10 m elevation sidecars needed as
input; it does not replace that proposal with a raw-elevation-only experiment.
The later experiment design must specify derivative methods, source scale and
context: local slope/aspect need a halo, while TWI needs hydrologically adequate
upstream terrain beyond an isolated 512-pixel tile. These derived products are
outside this enrichment pass.

Later, add an explicit DTM sidecar loader and preprocessing version while
retaining the existing forest channels. Load the full 512 × 512 DTM, validity
and coverage arrays before the [shared spatial augmentation](../../imint/training/unified_dataset.py).
Training chooses crop offsets from 0 through 8 for 512→504, then applies flips
and 90-degree rotations jointly to imagery, labels and auxiliaries. DTM and its
masks must follow those exact same transformations; never crop them independently.
Only the non-augmented evaluation/inference path uses the fixed centre crop.
Add registration tests across training crop offsets, flips and rotations before
using the new loader. Compute normalization from the training partition only;
the existing Copernicus DEM constants are not a Swedish DTM estimate. Use the
same source and preprocessing contract in training and serving.

Design matched comparisons with the same cohorts, forest inputs, labels and
seeds. A raw-DTM arm can isolate the elevation-source effect alongside the
planned terrain-feature arm; its exact scope belongs to the later retraining
design. Keep these comparisons separate from the R5 crop-label experiment.
Do not select preprocessing or missing-data policy on the protected holdout.

## Proposed implementation pieces

| Piece | Work |
|---|---|
| imint/training/lantmateriet_dtm.py | Authenticated source discovery, immutable cache and exact-grid 10 m reduction |
| scripts/enrich_tiles_dtm.py | Inventory, pilot, resumable sidecars and reconciliation |
| scripts/audit_enrichment_integrity.py | Extend audit to validate DTM sidecars and their manifest; no repair of source tiles |
| tests/test_lantmateriet_dtm.py | Mean, coverage, datum metadata, source seam and grid tests |
| tests/test_dtm_enrichment.py | Write/readback, interruption, deduplication and immutable-source integration tests |
| Later loader/service change | Explicit DTM release selection and trained preprocessing identity |

Follow [skg_aux_pass](../../scripts/skg_aux_pass.py) for provenance, explicit
gaps and atomic publication, but write separate DTM sidecars. The existing
[Copernicus reader](../../imint/training/copernicus_dem.py) remains the R4
provider. Its zero-filling and negative-height clipping are unsuitable for
the new DTM contract.

## Storage estimate and prerequisites

Three raw arrays use 9 bytes per pixel: 2.25 MiB per distinct 512 × 512 grid,
before compression and metadata. Budget is therefore N_unique × 2.25 MiB plus
source-window cache, pilot GeoTIFFs and temporary output space. Determine
N_unique from the inventory; no current tile count or download-time estimate
is assumed. A full native 5,120 × 5,120 float32 window alone is 100 MiB, so
process source blocks/windows rather than retaining many native tiles at once.

Before implementation: confirm API access and selected tile-root inventory.
Before bulk execution: reviewed code, a pinned runtime, passed pilot/batch,
sufficient disk space and a measured campaign budget. Retraining is a later
experiment. No tiles have been enriched by this planning change.

## Executable geometry and aggregation check

This synthetic check verifies the plan's numeric contract with the repository
grid helper and rasterio. It does not establish source availability or validate
a production DTM provider.

~~~python
import numpy as np
from affine import Affine
from rasterio.warp import reproject, Resampling
from imint.training.tile_config import TileConfig

tile = TileConfig(size_px=512)
bbox = tile.bbox_from_center(east=500000, north=6500000)
tile.assert_bbox_matches(bbox)
transform = Affine(10, 0, bbox["west"], 0, -10, bbox["north"])
assert transform.a == 10 and transform.e == -10
assert bbox["east"] - bbox["west"] == 5120
assert bbox["north"] - bbox["south"] == 5120
assert all(value % 10 == 0 for value in bbox.values())
# Fixed centre crop applies to non-augmented evaluation/inference.
crop_transform = transform * Affine.translation(4, 4)
assert crop_transform.c == transform.c + 40
assert crop_transform.f == transform.f - 40
assert np.zeros((512, 512))[4:508, 4:508].shape == (504, 504)
assert 504 * transform.a == 5040

native = np.arange(400, dtype=np.float32).reshape(20, 20)
native[:10, 10:] = 0
native[10:, :10] = -3
native[:2, :2] = np.nan
native_mask = np.isfinite(native).astype(np.float32)
mean = np.full((2, 2), np.nan, dtype=np.float32)
coverage = np.zeros((2, 2), dtype=np.float32)
args = dict(src_transform=Affine(1, 0, 0, 0, -1, 20),
            dst_transform=Affine(10, 0, 0, 0, -10, 20),
            src_crs="EPSG:3006", dst_crs="EPSG:3006",
            resampling=Resampling.average)
reproject(native, mean, src_nodata=np.nan, dst_nodata=np.nan, **args)
reproject(native_mask, coverage, **args)
assert abs(float(mean[0, 0]) - float(np.nanmean(native[:10, :10]))) < 0.001
assert abs(float(coverage[0, 0]) - 0.96) < 1e-6
assert mean[0, 1] == 0 and coverage[0, 1] == 1
assert mean[1, 0] == -3 and coverage[1, 0] == 1
valid = (coverage >= 1 - 1e-6) & np.isfinite(mean)
dtm = np.where(valid, mean, np.nan).astype(np.float32)
assert not valid[0, 0] and np.isnan(dtm[0, 0])
assert valid[0, 1] and valid[1, 0]
assert 512 * 512 * (4 + 4 + 1) / 1024**2 == 2.25
assert 5120 * 5120 * 4 / 1024**2 == 100
print("PASS: 10 m grid, 504 crop, weighted mean, nodata and storage arithmetic")
~~~
