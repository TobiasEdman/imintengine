# Annual LCLU prototype verification — 2026-10-29

Implementation verified locally and real 2024 R4 cases verified on ICE for
Uppsala and Stockholm after explicit approval of bounded CPU jobs. No private
field results, weights, training tiles or held-out locations were published.
Stockholm staged 28 public TESSERA blocks on the approved private cache volume.
No training or public deployment ran. Initial synthetic checks are retained
below; subsequent real-case and Stockholm evidence follow them.

## Automated verification

```sh
/Users/tobiasedman/Developer/ImintEngine/.venv/bin/python -m pytest \
  tests/lclu tests/api_services tests/test_inference_aux_parity.py \
  tests/test_tessera_wiring.py tests/test_tessera_year.py -q
```

**188 passed in 28.01s**. [Captured output](pytest.txt).
`git diff --check` also passed.

Coverage includes canonical 512→504 crop/input normalization parity; strict
checkpoint loader and real architecture forward with random weights; exact-year
rejection; CRS/affine/shape/nodata TIFF reopening; nearest-neighbor categorical
display; absent and corrupt sources; cache metadata, source identity receipts,
DEM true zero versus missing coverage; independent county-mask pins; download
budgets, subprocess timeout/log retention; duplicate requests, queue bounds,
failure retry, restart cache reuse, atomic publication and shutdown drain.
Legacy API/inference tests are included. Annual integration asserts that DES
and the exact-day engine were never called.

## Browser and delivered-file checks

The Codex in-app browser exercised a loopback server running
`python -m tests.lclu.browser_fixture`. Its visible banner said **SYNTHETIC
TEST · RANDOM WEIGHTS · NOT AN R4 CLASSIFICATION**. The test uses deterministic
random TESSERA weights and synthetic canonical-shape input at a public Uppsala
coordinate. Production startup has no synthetic fallback.

Verified in the rendered interface:

- An unconfigured production API disables classification and offers no years.
- Clicking an unqualified map position shows no years and disables submission.
- Coordinate form and prepared geographic marker select the exact fixture;
  2024 is the sole available year. Repeated marker selection works.
- Submission displays queued progress, then a completed overlay and download
  links. Changing the position clears the previous output.
- Overlay image loads as 523×524 Web Mercator display pixels; its opacity
  control changes both displayed percentage and actual image opacity.
- After restarting the API, clicking the marker and submitting again displays
  the previously verified result, with the same cache key and artifact hashes.

The HTTP GeoTIFF download was reopened, independently of the export function:
EPSG:3006, 504×504, 10m, bounds 645330/6636250/650370/6641290, exactly 5,040m
square, and nodata 255. Exactly 3,600 synthetic unavailable pixels survived;
250,416 were valid, summing to 254,016. Downloaded provenance names 2024 and
contains no private filesystem prefix. These counts describe the test fixture.

## Synthetic timing only

Single local CPU run, torch limited to two threads, prebuilt synthetic bundle,
no upstream download. This is a smoke measurement, not a performance benchmark
or production latency estimate. [Machine-readable receipt](synthetic-browser.json).

| Stage | Milliseconds |
|---|---:|
| Strict model load + catalogue/service construction | 539 |
| Cold input verification/preparation | 980 |
| Canonical forward + argmax | 3,828 |
| Export (including first-use imports) | 1,701 |
| Complete cold worker duration | 6,533 |
| Warm worker after restart and integrity checks | 429 |

## Initial access status

At the end of the initial build, the [public source checks](source-checks.md)
verified registry coverage only. No local real checkpoint or complete auxiliary
mirror was available, and no existing active pod mounted the data volume.
The first build therefore stopped at synthetic verification. The explicitly
approved real test below supersedes that access blocker for one case.

[Independent review](2026-10-29-review.md) found no remaining implementation
blocker. That initial review preceded the later real-data qualification and
user approval to commit the completed demo.

## Interactive prototype handoff

Follow-up request: “Build the prototype”. Added `scripts/serve_lclu.py` and
`make lclu-demo` / `make lclu-serve` so the prototype starts with one command.
`--demo` and `--manifest` are required, mutually exclusive choices. The demo
uses a separate default output directory and never substitutes for real input.
Its synthetic banner stays visible while scrolling; downloadable provenance
explicitly records `synthetic: true` and `weights: random`.

Reverified: 149 LCLU/source/API tests passed in 78.44s; launcher help and Makefile
dry run succeeded; independent launcher/demo review returned LGTM. In the
browser, both 2023 and 2024 completed with visible overlays/download links.
Both years intentionally use the same synthetic pattern; no change is inferred.

The follow-up preview is running locally at http://127.0.0.1:8784/lclu for user
inspection. No real checkpoint/mirror was added, no deployment or implementation
commit was made. The earlier 188-test report remains the broader core regression
result; changes in this follow-up are launcher, demo fixture and documentation.

## Real R4 end-to-end test — Uppsala 2024

The user requested a real test and explicitly approved one ICE job with 2 CPU,
4 GiB, no GPU, a read-only data PVC and a 30-minute deadline. The actual R4
checkpoint was opened, hash-verified and strictly loaded on ICE. Its private
weights stayed there. Only the public-area map and verification metadata were
retrieved locally. [Machine-readable evidence](real-browser.json).

- Public centre: 17.64 E, 59.86 N; exact requested year 2024.
- GeoTessera 0.10.2, dataset v1/vultr; two actual embedding blocks downloaded.
  Ten exact-year season-1 WEkEO COGs, selected SKG sources/masks and a pinned
  public GLO-30 tile were verified by the canonical preparation code.
- The global registry exceeded 4 GiB. A streaming regional extract retained
  exactly two rows in each registry, using the client's ±0.05° footprint
  predicate and exact year. Full parent and extracted file hashes are recorded.
  The retry succeeded within the original resource limit.
- Browser form → real API → completed overlay and downloadable GeoTIFF passed.
  The downloaded files matched the API's SHA256/size records. Reopened TIFF:
  EPSG:3006, 504×504, 10m, bounds 645330/6636250/650370/6641290, nodata 255.
- All 254,016 pixels have valid embeddings. Supporting rasters retain their
  own missing-value conventions and counts in provenance.
- A separate invocation of canonical `run_inference` matched the delivered
  class map exactly: 254,016 / 254,016 pixels. No model replacement was used.
- CPU inference: 5,064ms; full API worker with prepared inputs: 8,847ms.
  Canonical source preparation took 102.52s, excluding environment setup,
  initial source pinning, regional-registry creation and DEM staging.
- The job completed and its TTL cleanup removed the job and pod; zero test
  pods remained. The private checkpoint was not exported.
- A saved, read-only result viewer is available at http://127.0.0.1:8786/.
  Its overlay is verified loaded; the original demo remains at port 8784.

This is execution and parity verification for one public area/year, not an
accuracy assessment. Compatibility of current open height/markfukt products
with training-era inputs, additional areas/years and operational performance
still require separate qualification. Global registry loading is unsuitable
for the tested 4 GiB runtime without a regional extract.

## TESSERA GeoTIFF storage estimate — Sweden

The public v1/vultr manifest was downloaded and hash-pinned, then streamed
against the repository's coarse Lantmäteriet Swedish land polygon. Full
intersecting blocks, including coastal/border pixels, total 8,335–8,349 complete
blocks per year for 2018–2024. One advertised 2024 row lacks a scale-file size
and is excluded explicitly. This is not a precise national coverage audit.
[Inventory, hashes, measurements and calculations](tessera-storage.json).

One public Uppsala 2024 block (17.65 E, 59.85 N), 1,135×605 pixels and 128 bands,
was converted locally using Rasterio 1.4.4 / GDAL 3.10.3. Every base-resolution
band was reopened and compared bit-exact, with matching CRS, transform and
shape. Its 90.64 MB of native int8 embeddings plus float32 scales became
314.89 MB with tiled LZW, 296.83 MB with ZSTD COG, or 393.23 MB with ZSTD COG
and average overviews at factors 2/4/8. Two separate COGs preserving int8 and
float32 scales totaled 73.53 MB; reconstruction matched all float32 values
bit-exact. That compact representation needs a reader that applies the scale.

| Format | 2024 estimate, TB | 2018–2024 estimate, TB |
|---|---:|---:|
| Float32, uncompressed payload | 2.57 | 17.98 |
| Float32, tiled LZW | 2.30 | 16.10 |
| Float32, ZSTD COG, no overviews | 2.17 | 15.18 |
| Float32, ZSTD COG with overviews | 2.88 | 20.11 |
| Int8 COG + float32 scale COG, no overviews | 0.54 | 3.76 |
| Original NPY + scales, registry sizes | 0.66 | 4.63 |

TB is decimal (10^12 bytes). Compressed national figures extrapolate this
single measured block; compression will vary geographically. These estimates
cover one stored copy, excluding backups/replication, supporting rasters,
prepared inference caches and conversion scratch. Keeping original NPY files
adds the last row. Average embedding overviews were measured for storage only,
not qualified as model input. The selected full-block pixel counts and
seven-year totals were independently asserted in the calculation script.

Raw data, scripts and detailed receipts remain in the local ignored directory
`outputs/tessera-storage-20261029/`. The completed pyramid-only retry used a
1 GiB GDAL cache; the initial 128 MiB-cache attempt was terminated after it
became excessively slow. No cluster job or national raster download was used.

## Stockholm preparation — 2026-10-29 (before demo approval)

The user requested a real Stockholm municipality demo. A separately approved
read-only ICE preflight completed successfully in 14 seconds (1 CPU, 1 GiB,
10-minute hard deadline). It found all five HR-VPP metrics in 20 existing 2024
files whose metadata footprints cover the whole buffered Stockholm rectangle.
The selected files total 1,897,956,920 bytes and remain on the existing PVC.
No 2025 HR-VPP files were present for this area; the demo explicitly uses 2024.
The R4 checkpoint and forestry manifest exist. The full demo has not yet run.

The new compact COG path preserves the original int8 vectors and float32 scale
factors. On the saved real Uppsala block, all 33,554,432 reprojected float16
values and the validity mask matched the original NPY path exactly. Reopened
base-resolution COG values were also identical before reprojection. An
independent reviewer additionally reported zero differences in 67,108,864
values for two EPSG:32634 cases near Stockholm. The real Stockholm UTM33/34
seam test is prepared but awaits the full ICE demo.

Verification completed locally:

- 32 focused existing service/API and new COG/municipality tests passed.
- 3 staging/recipe tests passed, including exact-year/full-area VPP admission,
  interrupted/corrupted download handling, resource bounds and authenticated
  bootstrap extraction.
- Runtime import/head-construction/export smoke passed in the existing local
  Python environment; GeoTIFF CRS, transform, nodata and PNG reopening checked.
- Server-side `kubectl create --dry-run=server` accepted the final immutable
  ConfigMap, 20 GiB PVC and 24-hour 4-CPU/8-GiB Job specification.
- Independent review found a missing OpenCV dependency and a bootstrap/archive
  provenance mismatch. Both were fixed; the hash lock was regenerated and the
  bootstrap mismatch regression passed.

The clean local Docker build installed the original lock successfully but failed
committing its layer with a Docker storage I/O error. Host free space was under
0.5 GiB. It did not produce a runnable application image. Recipe mode instead
reconstructs the same hash-locked environment from the pinned official Python
base on the isolated demo PVC; its smoke check must pass before fetching sources.
The final recipe also includes the reviewed OpenCV fix. It has not yet executed.

Evidence: `outputs/lclu-stockholm-20261029/` holds the preflight Job/result,
coverage check, real COG parity, build log, source manifest, final rendered Job
and resumable run-state. See `docker/lclu/README.md` for the bounded run recipe.
No real Stockholm map, cache latency or browser validation is claimed yet.


## Stockholm live demo — completed 2026-10-29

The user approved 4 CPU, 8 GiB RAM and a persistent 20 GiB CephFS cache for
24 hours. The measured private Job was `lclu-stockholm-0124-r2`; its hard deadline
is 2026-10-30 12:45:33 UTC (13:45:33 Stockholm), within the original window.
The local operator tunnel serves `http://127.0.0.1:8788/lclu`.

All 28 original 2024 blocks were downloaded once (2,589,836,344 bytes) and
converted to separately scaled, lossless COG pairs (1,958,791,490 bytes).
Initial staging, including source identity checks and a real prepared input,
took 460.900 seconds. Recovery reused the sources and completed in 26.445
seconds. About 4.4 GiB holds sources and qualification, 1.5 GiB the Python
environment; result/input caches are additional and remain within the 20 GiB PVC.

Real verification passed:

- At 18.0 E, 59.33 N, all 33,554,432 float16 values and the validity mask
  match original-NPY preparation across native EPSG:32633/32634. All 254,016
  valid output pixels match canonical R4 inference exactly.
- That service request took 18.097 s: 12.858 s preparation, 4.634 s model,
  0.583 s export. Reopening its verified disk cache took 1.036 s.
- A fresh central Stockholm HTTP request took 18.253 s; repeating the same
  completed request took 0.154 s (in-memory job reuse). The saved GeoTIFF has
  EPSG:3006, a 10 m affine grid, 504×504 pixels and 252,916 valid pixels.
  Reopened TIFF/PNG and all downloaded artifact hashes passed.
- The browser independently created a western Stockholm map (17.81 E,
  59.37 N) in 19.1 s, with 254,016 valid pixels. Repeating it displayed in
  0.5 s. Municipality outline, overlay alignment, opacity control, status,
  year selection and export links were visually checked.
- HTTP requests outside Stockholm and for unqualified 2025 were rejected.
  These timings are observed individual requests, not a load benchmark.

Runtime integration found two defects before delivery: the pinned Python base
lacked libexpat, and resolving the virtual-environment executable's symlink
launched system Python in the preparation subprocess. The first was fixed with
an exact Debian package hash; the second preserves the invoked executable path.
Twelve focused source/staging tests pass, including a real symlinked-venv
regression, and independent review found no remaining issues in both fixes.
The clean runtime smoke and full real-data verification passed on ICE. No
private checkpoint was exported; these tests verify numerical equivalence,
not classification accuracy against reference labels.

Immutable deployed archive SHA256:
`2730f76fd6be469510382cfac7d66558671aaf97a3b81f761e222435070840d7`.

Evidence under `outputs/lclu-stockholm-20261029/`: `verification.json`,
`api-verification.json`, `export-verification.json`, `browser-verification.json`,
`staging-original.json`, `staging-resume.json`, `active-job.json`,
`observed-pod-r2.json`, `recipe-r2/`, and downloaded public maps in `result/`.


## Commit verification — 2026-10-29

`python -m pytest tests/lclu tests/api_services -q`: **156 passed in 35.89 s**.
[Captured output](pytest-commit.txt). `git diff --check` passed. This final suite
includes the COG path, bounded staging recipe and real-venv regression added
after the earlier 188-test run. The two counts cover different test selections.

The [compact Stockholm receipt](stockholm.json) records the observed source
sizes, runtime identity, exact pixel comparisons, API/browser timings and export
checks without private source paths or model contents. Final packaging review
corrected stale historical descriptions and made initial-start and active-r2
access commands explicit.


## DES legend and detail zoom — 2026-10-29

User feedback requested DES colours and closer map zoom. Shared CSS had applied
translucent mint text, intended for dark panels, to the white legend. The annual
page now explicitly uses DES forest green `#1a4338` on white, at 14px. Browser
computed styles confirmed all 29 entries; contrast is 11.03:1. A rendered
screenshot verified legibility.

Location selection previously fitted the whole 5.04 km footprint on every click,
resetting the map to at most zoom 12. It now fits only from a broader view and
otherwise pans while retaining the selected zoom. Browser checks reached zoom 15
and confirmed level 15 remained after a new map click and availability response.
The explicit maximum is 18. Independent review found no issues; JS syntax passed.

The immutable r3 archive changes only `docs/css/lclu.css` and `docs/js/lclu.js`
relative to the numerically verified r2 source. The existing cache is reused
and the original resource/time limits apply. [UI verification](ui-r3.json).

The updated live r3 runtime passed its import/export smoke and served the
corrected CSS. Its deadline is 2026-10-30 13:45:02 Stockholm time.


## Opaque, sharp result pixels — 2026-10-29

The result now defaults to 100% opacity; the existing slider remains active.
Only classification overlays use pixelated image rendering. Zoom and fade
animations are disabled to avoid a blurred transition. The original 10 m
classification and nearest-neighbour exported overlay are unchanged.

The browser preview verified default opacity 1, slider adjustment to 0.01 and
back to 1, a 526×527 native overlay, and sharp display at zoom 15. Live r4
verification confirmed the same default, rendering and zoom; screenshots were
visually inspected. The downloaded GeoTIFF is byte-identical to the earlier
verified result. JS syntax and independent review passed.
[Machine-readable receipt](ui-r4.json). The same 4 CPU, 8 GiB and 20 GiB cache
are retained; the deadline remains within the original approval.


## Kiruna tätort — 2026-10-29

The user selected Kiruna tätort instead of Abisko or the full municipality.
The demo uses SCB's exact 2023 tätort polygon (2584TC103), with a 3,700 m
source buffer. All 116,264 possible snapped centre candidates, including a
rounding margin, were checked: every input needs at most four native blocks.
The five exact-year HR-VPP metrics cover the entire buffered preparation extent.

Nine 2024 TESSERA blocks took 211.55 s to stage and verify. Originals occupy
577,614,636 bytes and compact COGs 452,870,047 bytes. Every base-resolution COG
pixel was reopened and compared to its original during conversion.

A real four-block window at 20.197573827965233 E, 67.87611539512876 N matched
all 33,554,432 original-pipeline embedding values and their validity mask.
All 254,016 classified output pixels matched canonical R4 inference, with
no nodata in this test window. The first run took 18.91 s and verified cache
reuse 1.07 s. This establishes pipeline parity, not classification accuracy
against independent reference labels. [Numeric receipt](kiruna.json).

Both regions retain separate inventories/caches and use one resident model.
The UI's area selector switches between Stockholm and Kiruna tätort.
[158 focused tests passed](pytest-kiruna.txt), including exact-year routing,
outside-area rejection, ambiguous-overlap rejection and combined startup in
a fresh directory. Independent review found no remaining actionable issues.

Live r5 API checks passed for both Stockholm and Kiruna. Both GeoTIFF hashes
match the previously verified outputs. Requests outside the tätort and for
2025 are rejected (422). Browser area switching Stockholm → Kiruna → Stockholm
→ Kiruna passed; stale results clear on switching. The representative Kiruna
map took 18.5 s, then 0.2 s on repeat; all 254,016 pixels were valid. The
GeoTIFF/PNG were downloaded, hash-checked and reopened. The browser was left
on Kiruna at zoom 15, 100% opacity, with pixelated rendering. Live r5 ends
2026-10-30 13:42:51 Stockholm time within the original approval.


## Native pixel inspection and NMD legend (2026-10-29, r6)

Task 0126 follow-up scope includes the user-requested click inspector and
NMD legend check. The official QML and aggregation limitations are recorded
in [nmd-legend.md](nmd-legend.md). Tests: 161 passed; independent review
found no actionable issues. Full receipts: [pixel-nmd.json](pixel-nmd.json).

Live r6 compared both Kiruna and Stockholm with the saved r5 GeoTIFFs:
254,016 native class values per map were exactly equal, with unchanged CRS,
transform and nodata. The palette metadata and resulting PNG colours changed
together. HTTP pixel queries matched all 15 classes present in each map;
outside queries returned inside=false. Synthetic tests additionally cover
nodata, class 0, all four edges, invalid coordinates, corrupt artifacts and
absence of additional inference.

Browser checks: Kiruna result retained after click; popup displayed
Buskdominerad mark and its NMD aggregation note. Zoom 15, opacity 100%,
pixelated rendering, all 29 legend entries and expandable notes verified.
The live Job is lclu-stockholm-0124-r6, source archive
d36c27894b1461767e958c8dfe427d2901f5e9a58116c347894ddd0275dda6d4.
It uses the same 4CPU/8Gi/20Gi budget, preserves caches and expires
2026-10-30 12:40:48 UTC. Implementation commit remains pending user request.
